"""Two-level retention and mandatory structural verification.

Level A -- hard preservation by *transparent, configurable, testable rules*:
  * the newest N messages (conversation continuity),
  * messages explicitly marked by the user (configurable markers),
  * tool units whose internal messages were partially retained (dependency).

Level B -- Clef model relevance with the configured threshold.

Final set = HardPreserved ∪ ClefRelevant ∪ FailSafeRetained.

Structural verification is mandatory; semantic verification is an optional
second Agent pass and never a substitute for the structural checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import DEFAULT_HARD_PRESERVE_MARKERS
from conversation_grouper import ConversationUnit, flatten_units, is_tool_result


@dataclass
class ValidationIssue:
    code: str
    message: str
    affected: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    ok: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, code: str, message: str, affected: list[str] | None = None) -> None:
        self.issues.append(ValidationIssue(code, message, affected or []))
        self.ok = False


@dataclass
class RetentionOutcome:
    units: list[ConversationUnit]
    hard_preserved: set[str] = field(default_factory=set)
    clef_retained: set[str] = field(default_factory=set)
    fail_safe_retained: set[str] = field(default_factory=set)
    dropped: set[str] = field(default_factory=set)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return flatten_units(self.units)


def _content_key(msg: dict[str, Any]) -> tuple[Any, ...]:
    return (
        msg.get("role", ""),
        msg.get("content", ""),
        msg.get("timestamp"),
        msg.get("_rowid"),
        msg.get("id"),
    )


def _tool_call_ids(msg: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for call in msg.get("tool_calls") or []:
        if isinstance(call, dict):
            call_id = call.get("id") or call.get("tool_call_id")
            if call_id:
                ids.add(str(call_id))
    if msg.get("function_call"):
        ids.add("function_call")
    return ids


def has_hard_preserve_marker(msg: dict[str, Any], markers: tuple[str, ...]) -> bool:
    content = str(msg.get("content", ""))
    return any(marker in content for marker in markers)


def apply_retention_policy(
    units: list[ConversationUnit],
    decisions: dict[str, Any],
    config: dict[str, Any],
) -> RetentionOutcome:
    """Combine hard-preservation rules with Clef/fail-safe decisions."""
    retained: set[str] = {
        qid for qid, decision in decisions.items() if getattr(decision, "retain", False)
    }
    hard: set[str] = set()
    markers = tuple(config.get("hard_preserve_markers", DEFAULT_HARD_PRESERVE_MARKERS))

    # Rule 1: newest N messages (whole unit preserved to keep exchanges intact).
    recent = int(config.get("hard_preserve_recent", 2))
    if recent > 0:
        count = 0
        for unit in reversed(units):
            if count >= recent:
                break
            hard.add(unit.unit_id)
            count += len(unit.messages)

    # Rule 2: explicit user markers.
    for unit in units:
        if any(has_hard_preserve_marker(m, markers) for m in unit.messages):
            hard.add(unit.unit_id)

    # Rule 3: tool units are atomic. If any member was retained by Clef, a
    # fail-safe rule, or hard preservation, keep the whole unit so a tool
    # result can never survive without its call (and vice versa).
    if config.get("hard_preserve_tool_links", True):
        for unit in units:
            if "tool" in unit.kind and (
                unit.unit_id in retained or unit.unit_id in hard
            ):
                hard.add(unit.unit_id)

    final = retained | hard
    clef = {
        qid for qid, d in decisions.items()
        if getattr(d, "source", "") == "clef" and getattr(d, "retain", False)
    }
    fail_safe = {
        qid for qid, d in decisions.items()
        if getattr(d, "source", "") not in ("clef", "forced")
        and getattr(d, "retain", False)
    }

    return RetentionOutcome(
        units=[u for u in units if u.unit_id in final],
        hard_preserved=hard,
        clef_retained=clef,
        fail_safe_retained=fail_safe,
        dropped={u.unit_id for u in units if u.unit_id not in final},
    )


def structural_validate(
    original: list[dict[str, Any]],
    retained: list[dict[str, Any]],
) -> ValidationReport:
    """Deterministic checks that must all pass before a database write."""
    report = ValidationReport()

    # 1. Chronological order preserved and retained is a subsequence.
    oi = 0
    for msg in retained:
        key = _content_key(msg)
        found = False
        while oi < len(original):
            if _content_key(original[oi]) == key:
                oi += 1
                found = True
                break
            oi += 1
        if not found:
            report.add(
                "retained_not_subsequence",
                "A retained message is not present in the original in order",
                [str(key)],
            )
            break

    # 2. Session identity consistent.
    sids = {str(m.get("session_id")) for m in retained if m.get("session_id")}
    if len(sids) > 1:
        report.add("mixed_sessions", "Retained messages span multiple sessions", sorted(sids))

    # 3. Tool-result dependencies not broken by compaction.
    original_calls: set[str] = set()
    original_dependents: set[str] = set()
    for msg in original:
        original_calls |= _tool_call_ids(msg)
        if is_tool_result(msg) and msg.get("tool_call_id"):
            original_dependents.add(str(msg["tool_call_id"]))

    seen_calls: set[str] = set()
    for msg in retained:
        seen_calls |= _tool_call_ids(msg)
        if is_tool_result(msg) and msg.get("tool_call_id"):
            call_id = str(msg["tool_call_id"])
            if call_id not in seen_calls and call_id in original_dependents:
                report.add(
                    "broken_tool_dependency",
                    f"Tool result {call_id} retained without its tool call",
                    [call_id],
                )

    # 4. No duplicate rows.
    keys = [_content_key(m) for m in retained]
    if len(keys) != len(set(keys)):
        report.add("duplicate_messages", "Retained set contains duplicate messages")

    # 5. Empty set is never a successful compaction.
    if not retained:
        report.add("empty_retention", "Retained message set is empty")

    return report
