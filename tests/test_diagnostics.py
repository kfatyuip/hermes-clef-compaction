"""Tests for read-only structural diagnostics and the --diagnose CLI mode.

Every test here is offline: no Clef call, no state.db mutation, and no writes
outside pytest's tmp_path (the F1 regression writes only to a tmp cache dir).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import compaction_engine
from compaction_engine import HermesCompaction, build_parser, main
from context_loader import ContextError
from diagnostics import diagnose_messages


def _clean_messages():
    """A healthy, matched call/result conversation in chronological order."""
    return [
        {
            "session_id": "s1",
            "role": "user",
            "content": "hi",
            "timestamp": 1.0,
            "id": 1,
            "_rowid": 1,
        },
        {
            "session_id": "s1",
            "role": "assistant",
            "content": "calling a tool",
            "timestamp": 2.0,
            "id": 2,
            "_rowid": 2,
            "tool_calls": json.dumps([{"id": "call-1", "name": "lookup"}]),
        },
        {
            "session_id": "s1",
            "role": "tool",
            "content": "the result",
            "timestamp": 3.0,
            "id": 3,
            "_rowid": 3,
            "tool_call_id": "call-1",
        },
    ]


# ---------------------------------------------------------------------------
# diagnose_messages(): individual checks
# ---------------------------------------------------------------------------

def test_clean_messages_are_healthy():
    report = diagnose_messages(_clean_messages())
    assert report["healthy"] is True
    assert report["issue_count"] == 0
    assert report["issues"] == []
    assert report["message_count"] == 3
    assert set(report["checks"]) == {
        "session_mixing",
        "tool_orphans",
        "oversized_messages",
        "bytes_fields",
        "duplicate_ids",
        "order_inversions",
    }
    assert all(check["ok"] for check in report["checks"].values())
    json.dumps(report)  # report is always JSON-safe


def test_diagnose_detects_session_mixing_and_partial_ids():
    messages = _clean_messages()
    messages[1]["session_id"] = "s2"
    messages[2].pop("session_id")

    check = diagnose_messages(messages)["checks"]["session_mixing"]
    assert check["ok"] is False
    assert check["mixed"] is True
    assert check["partially_missing"] is True
    assert check["sessions"] == ["s1", "s2"]
    assert check["missing_indices"] == [2]
    assert "session_mixing" in diagnose_messages(messages)["issues"]


def test_diagnose_detects_orphan_tool_result_and_orphan_tool_call():
    no_call = _clean_messages()
    no_call[1].pop("tool_calls")
    check = diagnose_messages(no_call)["checks"]["tool_orphans"]
    assert check["ok"] is False
    assert [item["tool_call_id"] for item in check["orphan_results"]] == ["call-1"]
    assert check["orphan_calls"] == []

    no_result = _clean_messages()
    no_result[2].pop("tool_call_id")
    check = diagnose_messages(no_result)["checks"]["tool_orphans"]
    assert check["ok"] is False
    assert [item["tool_call_id"] for item in check["orphan_calls"]] == ["call-1"]
    assert check["orphan_results"] == []


def test_diagnose_detects_oversized_messages():
    messages = _clean_messages()
    messages[0]["content"] = "x" * 9000
    check = diagnose_messages(messages)["checks"]["oversized_messages"]
    assert check["ok"] is False
    assert check["count"] == 1
    assert check["messages"][0]["index"] == 0
    assert check["messages"][0]["chars"] == 9000
    assert check["threshold_chars"] == 8000

    lowered = diagnose_messages(_clean_messages(), max_message_chars=5)
    assert lowered["checks"]["oversized_messages"]["count"] == 2


def test_diagnose_reports_bytes_fields_without_embedding_values():
    messages = _clean_messages()
    messages[0]["display_identity"] = b"\x00binary"
    messages[1]["meta"] = {"nested": bytearray(b"\xff")}

    report = diagnose_messages(messages)
    check = report["checks"]["bytes_fields"]
    assert check["ok"] is False
    assert check["count"] == 2
    assert check["messages"][0]["fields"] == ["display_identity"]
    assert check["messages"][1]["fields"] == ["meta"]
    assert "bytes_fields" in report["issues"]

    serialized = json.dumps(report)
    assert "binary" not in serialized  # no raw BLOB value leaked into the report
    assert isinstance(messages[0]["display_identity"], bytes)  # input untouched


def test_diagnose_detects_duplicate_ids():
    messages = _clean_messages()
    messages[1]["id"] = 1
    check = diagnose_messages(messages)["checks"]["duplicate_ids"]
    assert check["ok"] is False
    assert check["count"] == 1
    assert check["duplicates"][0]["field"] == "id"
    assert check["duplicates"][0]["value"] == "1"
    assert check["duplicates"][0]["indices"] == [0, 1]


def test_diagnose_detects_order_inversions_and_nonfinite_timestamps():
    messages = _clean_messages()
    messages[2]["timestamp"] = 1.5
    check = diagnose_messages(messages)["checks"]["order_inversions"]
    assert check["ok"] is False
    assert check["count"] == 1
    assert check["inversions"][0]["index"] == 2
    assert check["inversions"][0]["previous_index"] == 1

    messages[1]["timestamp"] = float("nan")
    report = diagnose_messages(messages)  # non-finite coerced, still serializable
    assert report["checks"]["order_inversions"]["count"] >= 1
    json.dumps(report)


def test_diagnose_handles_non_dict_entries_without_crashing():
    report = diagnose_messages(["not-a-dict", None, _clean_messages()[0]])
    assert report["non_dict_message_count"] == 2
    assert report["healthy"] is True  # no dicts -> no structural findings
    json.dumps(report)


# ---------------------------------------------------------------------------
# CLI --diagnose wiring
# ---------------------------------------------------------------------------

def test_help_lists_diagnose():
    help_text = build_parser().format_help()
    assert "--diagnose" in help_text
    assert build_parser().parse_args([]).diagnose is False
    assert build_parser().parse_args(["--diagnose"]).diagnose is True


def test_cli_diagnose_prints_report_and_bypasses_pipeline(monkeypatch, capsys):
    loaded = {}

    def fake_load(config):
        loaded["config"] = config
        return _clean_messages()

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnose must not instantiate the compactor")

    def forbidden_cache(*args, **kwargs):
        raise AssertionError("diagnose must not touch the cache")

    monkeypatch.setattr(compaction_engine, "load_messages", fake_load)
    monkeypatch.setattr(compaction_engine, "HermesCompaction", forbidden)
    monkeypatch.setattr(compaction_engine, "CacheManager", forbidden_cache)
    monkeypatch.setattr("relevance_filter.requests.post", forbidden_cache)

    assert main(["--diagnose", "--max-messages", "7"]) == 0
    assert loaded["config"]["max_messages"] == 7

    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is True
    assert report["message_count"] == 3


def test_cli_diagnose_load_failure_returns_2(monkeypatch, capsys):
    def fail(config):
        raise ContextError("no context available")

    monkeypatch.setattr(compaction_engine, "load_messages", fail)
    assert main(["--diagnose"]) == 2
    assert "no context available" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# F1 regression: bytes/BLOB must not crash the cache-write path
# ---------------------------------------------------------------------------

def test_cache_write_path_sanitizes_blob_messages(monkeypatch, base_config):
    """A real state.db bytes column must survive the cache.put path."""
    messages = [{
        "session_id": "s1",
        "role": "user",
        "content": "hi",
        "timestamp": 1.0,
        "id": 1,
        "display_identity": b"\x00blob-from-state-db",
    }]
    monkeypatch.setattr(
        compaction_engine, "load_messages", lambda config: [dict(m) for m in messages]
    )

    compactor = HermesCompaction(dict(base_config))
    result = compactor.run()  # previously raised TypeError: Object of type bytes...

    assert "error" not in result.get("metadata", {})
    assert result["messages"][0]["content"] == "hi"

    cache_files = sorted((Path(base_config["cache_dir"]) / "v2").glob("*/*/compacted.json"))
    assert cache_files, "no cache entry was written"
    cached = json.loads(cache_files[0].read_text(encoding="utf-8"))
    json.dumps(cached)  # the persisted cache payload is JSON-safe
    assert cached["messages"][0]["content"] == "hi"
    assert "display_identity" not in cached["messages"][0]
    assert cached["messages"][0]["_sanitized_fields"] == ["display_identity"]
    assert isinstance(messages[0]["display_identity"], bytes)  # source row untouched
