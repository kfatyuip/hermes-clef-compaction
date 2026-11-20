"""Regressions for serialization/CLI fixes in the working tree.

Covers:
  1. HermesCompaction._sanitize_message + _result output (BLOB/bytes stripping)
  2. RelevanceFilter._sanitize_message + API payload sanitization
  3. validator._tool_call_ids with JSON-string tool_calls (state.db shape)
  4. --max-messages CLI override of config["max_messages"]

No real Clef API call and no state.db write occurs: requests.post and
HermesCompaction are mocked/monkeypatched, and all file writes stay in tmp_path.
"""

from __future__ import annotations

import json

from compaction_engine import HermesCompaction, build_parser, main
from config import build_config
from conversation_grouper import ConversationUnit
from relevance_filter import RelevanceFilter
from validator import _tool_call_ids, structural_validate


# ---------------------------------------------------------------------------
# 1. HermesCompaction: bytes/BLOB stripping and JSON-safe output
# ---------------------------------------------------------------------------

def test_compaction_sanitize_message_strips_bytes_and_records_sidecar():
    msg = {
        "role": "user",
        "content": "hello",
        "display_identity": b"\x00binary",
        "avatar": bytearray(b"\xff\xfe"),
        "timestamp": 1.0,
    }
    clean = HermesCompaction._sanitize_message(msg)

    assert clean["role"] == "user"
    assert clean["content"] == "hello"
    assert clean["timestamp"] == 1.0
    assert "display_identity" not in clean
    assert "avatar" not in clean
    assert set(clean["_sanitized_fields"]) == {"display_identity", "avatar"}
    json.dumps(clean)  # no TypeError: Object of type bytes is not JSON serializable
    assert isinstance(msg["display_identity"], bytes)  # original dict untouched


def test_compaction_sanitize_message_drops_other_unserializable_values():
    clean = HermesCompaction._sanitize_message(
        {"content": "ok", "tags": {"a", "b"}, "n": 3}
    )
    assert clean["content"] == "ok" and clean["n"] == 3
    assert clean["_sanitized_fields"] == ["tags"]


def test_compaction_result_sanitizes_and_writes_valid_json(tmp_path, base_config):
    out = tmp_path / "result.json"
    cfg = dict(base_config)
    cfg["output"] = str(out)
    compactor = HermesCompaction(cfg)

    result = compactor._result(
        cached=False,
        elapsed=0.01,
        result_data={"messages": [
            {"role": "user", "content": "kept", "display_identity": b"blob"},
            {"role": "assistant", "content": "also kept"},
        ]},
    )

    first = result["messages"][0]
    assert first["content"] == "kept"
    assert "display_identity" not in first
    assert first["_sanitized_fields"] == ["display_identity"]
    assert result["messages"][1] == {"role": "assistant", "content": "also kept"}
    assert result["metadata"]["cached"] is False
    json.dumps(result)  # whole result must be JSON-serializable

    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert "display_identity" not in parsed["messages"][0]
    assert parsed["messages"][0]["_sanitized_fields"] == ["display_identity"]
    assert parsed["messages"][0]["content"] == "kept"


# ---------------------------------------------------------------------------
# 2. RelevanceFilter: bytes/BLOB filtering of the API payload
# ---------------------------------------------------------------------------

def test_filter_sanitize_message_strips_bytes_and_records_sidecar():
    clean = RelevanceFilter._sanitize_message({
        "role": "tool",
        "content": "result",
        "display_identity": b"blob",
        "raw": bytearray(b"\x01"),
    })
    assert clean["role"] == "tool"
    assert clean["content"] == "result"
    assert "display_identity" not in clean and "raw" not in clean
    assert set(clean["_sanitized_fields"]) == {"display_identity", "raw"}
    assert RelevanceFilter._sanitize_message({"content": "clean"}) == {"content": "clean"}


class _FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {"success": True, "result": {"answers": {"u0": {"noul": 0.9}}}}


def test_default_call_api_sanitizes_payload_messages(monkeypatch, base_config):
    cfg = {
        **base_config,
        "cloudflare_account_id": "acct",
        "cloudflare_auth_token": "token",
    }
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr("relevance_filter.requests.post", fake_post)
    engine = RelevanceFilter(cfg)
    payload = {
        "state": {
            "context_summary": "summary",
            "messages": [
                {"role": "user", "content": "hi", "display_identity": b"blob"},
                "not-a-dict",
            ],
        },
    }

    engine._default_call_api(payload)

    sent = captured["payload"]
    msg = sent["state"]["messages"][0]
    assert "display_identity" not in msg
    assert msg["_sanitized_fields"] == ["display_identity"]
    assert sent["state"]["messages"][1] == "not-a-dict"  # non-dicts left alone
    json.dumps(sent)  # the original TypeError would happen here


def test_filter_chunk_payload_is_json_safe_before_api_call(monkeypatch, base_config):
    cfg = {
        **base_config,
        "cloudflare_account_id": "acct",
        "cloudflare_auth_token": "token",
        "api_max_retries": 0,
    }
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr("relevance_filter.requests.post", fake_post)
    units = [ConversationUnit("u0", "solo", [
        {"role": "user", "content": "hi", "display_identity": b"blob"},
    ])]
    engine = RelevanceFilter(cfg)  # no injected call_api -> real _default_call_api
    decisions, _ = engine.filter_chunk(units, "summary")

    sent = captured["payload"]
    msg = sent["state"]["messages"][0]
    assert "display_identity" not in msg
    assert msg["_sanitized_fields"] == ["display_identity"]
    json.dumps(sent)
    assert decisions[0].source == "clef"


# ---------------------------------------------------------------------------
# 3. validator._tool_call_ids: JSON-string tool_calls from state.db
# ---------------------------------------------------------------------------

def test_tool_call_ids_parses_json_string():
    msg = {
        "tool_calls": json.dumps([
            {"id": "call-1"},
            {"tool_call_id": "call-2"},
        ])
    }
    assert _tool_call_ids(msg) == {"call-1", "call-2"}


def test_tool_call_ids_still_accepts_preparsed_list():
    msg = {"tool_calls": [{"id": "call-1"}, {"tool_call_id": "call-2"}]}
    assert _tool_call_ids(msg) == {"call-1", "call-2"}


def test_tool_call_ids_handles_malformed_or_non_list_json():
    assert _tool_call_ids({"tool_calls": "not json"}) == set()
    assert _tool_call_ids({"tool_calls": json.dumps({"id": "x"})}) == set()
    assert _tool_call_ids({"tool_calls": "17"}) == set()
    assert _tool_call_ids({"tool_calls": None}) == set()


def test_tool_call_ids_keeps_function_call_marker():
    assert _tool_call_ids({"function_call": {"name": "f"}, "tool_calls": "[]"}) == {
        "function_call"
    }


def test_structural_validation_detects_broken_dependency_with_json_tool_calls():
    original = [
        {
            "session_id": "s1",
            "role": "assistant",
            "content": "call",
            "timestamp": 1.0,
            "tool_calls": json.dumps([{"id": "call-1"}]),
        },
        {
            "session_id": "s1",
            "role": "tool",
            "content": "result",
            "timestamp": 2.0,
            "tool_call_id": "call-1",
        },
    ]
    report = structural_validate(original, [original[1]])
    assert not report.ok
    codes = {issue.code for issue in report.issues}
    assert "broken_tool_dependency" in codes

    complete = structural_validate(original, original)
    assert complete.ok, [i.code for i in complete.issues]


# ---------------------------------------------------------------------------
# 4. CLI --max-messages overrides config["max_messages"]
# ---------------------------------------------------------------------------

def test_parser_max_messages_defaults_to_none():
    parser = build_parser()
    assert parser.parse_args([]).max_messages is None
    assert parser.parse_args(["--max-messages", "123"]).max_messages == 123


def test_cli_max_messages_overrides_config(monkeypatch):
    captured = {}

    class FakeCompactor:
        def __init__(self, config):
            captured["config"] = build_config(config)

        def run(self):
            return {"messages": [], "metadata": {}}

    monkeypatch.setattr("compaction_engine.HermesCompaction", FakeCompactor)

    assert main(["--max-messages", "123"]) == 0
    assert captured["config"]["max_messages"] == 123


def test_cli_without_max_messages_uses_default_48(monkeypatch):
    captured = {}

    class FakeCompactor:
        def __init__(self, config):
            captured["config"] = build_config(config)

        def run(self):
            return {"messages": [], "metadata": {}}

    monkeypatch.setattr("compaction_engine.HermesCompaction", FakeCompactor)

    assert main([]) == 0
    assert captured["config"]["max_messages"] == 48


def test_cli_invalid_max_messages_reaches_config_validation(monkeypatch):
    class StrictCompactor:
        def __init__(self, config):
            build_config(config)  # 0 fails validate_config -> ValueError

        def run(self):
            raise AssertionError("run() must not execute on invalid config")

    monkeypatch.setattr("compaction_engine.HermesCompaction", StrictCompactor)
    assert main(["--max-messages", "0"]) == 2
