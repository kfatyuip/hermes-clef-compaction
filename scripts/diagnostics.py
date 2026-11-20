"""Read-only structural health checks for an already-loaded message set.

This module is deliberately side-effect free: it never calls the Clef API,
never opens state.db for writing, never creates cache directories and never
mutates the messages it inspects. It produces a JSON-serializable report even
when the input contains SQLite BLOB/bytes columns that JSON cannot represent
(the report names the offending fields; it never embeds their values).

Checks:
  * session mixing (including partially missing session ids)
  * orphan tool calls / tool results
  * oversized messages
  * bytes / BLOB fields (top-level and nested)
  * duplicate message ids
  * chronological order inversions
"""

from __future__ import annotations

import json
import math
from typing import Any

from conversation_grouper import is_tool_call, is_tool_result
from validator import _tool_call_ids

DEFAULT_MAX_MESSAGE_CHARS = 8000

_ID_KEYS = ("id", "_rowid", "message_id")


def _float_timestamp(value: Any) -> float:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return 0.0
    return ts if math.isfinite(ts) else 0.0


def _message_id(msg: dict[str, Any]) -> str | None:
    """Short, JSON-safe locator; never returns a bytes value."""
    for key in _ID_KEYS:
        if key in msg and msg[key] is not None:
            return f"{key}:{msg[key]!s}"
    return None


def _contains_binary(value: Any, depth: int = 0) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if depth >= 6:
        return False
    if isinstance(value, dict):
        return any(_contains_binary(v, depth + 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_binary(v, depth + 1) for v in value)
    return False


def _check_sessions(messages: list[Any]) -> dict[str, Any]:
    sessions: set[str] = set()
    missing: list[int] = []
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        sid = msg.get("session_id")
        if sid is None or sid == "":
            missing.append(index)
        else:
            sessions.add(str(sid))
    mixed = len(sessions) > 1
    partially_missing = bool(sessions and missing)
    return {
        "ok": not mixed and not partially_missing,
        "sessions": sorted(sessions),
        "missing_indices": missing,
        "mixed": mixed,
        "partially_missing": partially_missing,
    }


def _check_tool_links(messages: list[Any]) -> dict[str, Any]:
    calls: dict[str, list[int]] = {}
    results: dict[str, list[int]] = {}
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if is_tool_call(msg):
            for call_id in _tool_call_ids(msg):
                calls.setdefault(call_id, []).append(index)
        if is_tool_result(msg) and msg.get("tool_call_id") is not None:
            results.setdefault(str(msg["tool_call_id"]), []).append(index)

    orphan_results = [
        {"tool_call_id": cid, "indices": indices}
        for cid, indices in sorted(results.items())
        if cid not in calls
    ]
    orphan_calls = [
        {"tool_call_id": cid, "indices": indices}
        for cid, indices in sorted(calls.items())
        if cid not in results
    ]
    return {
        "ok": not orphan_results and not orphan_calls,
        "call_count": len(calls),
        "result_count": len(results),
        "orphan_results": orphan_results,
        "orphan_calls": orphan_calls,
    }


def _check_oversized(
    messages: list[Any], max_message_chars: int
) -> dict[str, Any]:
    oversized: list[dict[str, Any]] = []
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        try:
            chars = len(str(msg.get("content", "")))
        except Exception:  # pragma: no cover - defensive for exotic objects
            chars = 0
        if chars > max_message_chars:
            oversized.append(
                {"index": index, "id": _message_id(msg), "chars": chars}
            )
    return {
        "ok": not oversized,
        "threshold_chars": max_message_chars,
        "count": len(oversized),
        "messages": oversized,
    }


def _check_binary_fields(messages: list[Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        fields = sorted(
            key for key, value in msg.items() if _contains_binary(value)
        )
        if fields:
            findings.append(
                {"index": index, "id": _message_id(msg), "fields": fields}
            )
    return {
        "ok": not findings,
        "count": len(findings),
        "messages": findings,
    }


def _check_duplicate_ids(messages: list[Any]) -> dict[str, Any]:
    seen: dict[tuple[str, str], list[int]] = {}
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        for key in _ID_KEYS:
            if key not in msg or msg[key] is None:
                continue
            token = (key, str(msg[key]))
            seen.setdefault(token, []).append(index)
    duplicates = [
        {"field": key, "value": value, "indices": indices}
        for (key, value), indices in sorted(seen.items())
        if len(indices) > 1
    ]
    return {
        "ok": not duplicates,
        "count": len(duplicates),
        "duplicates": duplicates,
    }


def _check_order(messages: list[Any]) -> dict[str, Any]:
    inversions: list[dict[str, Any]] = []
    previous: tuple[int, float] | None = None
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        ts = _float_timestamp(msg.get("timestamp"))
        if previous is not None and ts < previous[1]:
            inversions.append(
                {
                    "index": index,
                    "previous_index": previous[0],
                    "timestamp": ts,
                    "previous_timestamp": previous[1],
                }
            )
        previous = (index, ts)
    return {
        "ok": not inversions,
        "count": len(inversions),
        "inversions": inversions,
    }


def diagnose_messages(
    messages: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
) -> dict[str, Any]:
    """Return a JSON-serializable structural health report for *messages*.

    ``messages`` is treated as an opaque sequence and is never modified. The
    report has one entry per check under ``checks``, a flat ``issues`` list of
    failing check names, and ``healthy`` (True iff every check passed).
    """
    items = list(messages) if not isinstance(messages, list) else messages
    checks: dict[str, Any] = {
        "session_mixing": _check_sessions(items),
        "tool_orphans": _check_tool_links(items),
        "oversized_messages": _check_oversized(items, int(max_message_chars)),
        "bytes_fields": _check_binary_fields(items),
        "duplicate_ids": _check_duplicate_ids(items),
        "order_inversions": _check_order(items),
    }
    issues = [name for name, check in checks.items() if not check["ok"]]
    report: dict[str, Any] = {
        "message_count": len(items),
        "non_dict_message_count": sum(
            1 for item in items if not isinstance(item, dict)
        ),
        "healthy": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "checks": checks,
    }
    # Guarantee the diagnosis itself is always serializable (defensive: a
    # hostile message object must not make the read-only report unusable).
    json.dumps(report, ensure_ascii=False)
    return report
