"""Regression tests for the deep-audit findings D1-D5.

Finding ids match ``compaction-stress-deep/audit.md``. Each test targets one
concrete defect and is written so it FAILS on the pre-fix code.
"""

from __future__ import annotations

import json

import pytest

from chunk_manager import _split_oversized_unit, build_chunks
from config import build_config
from context_loader import ContextError, load_from_file, normalize_order, ordered_fingerprint
from conversation_grouper import ConversationUnit
from hashing import compute_messages_hash
from relevance_filter import Decision, RelevanceFilter
from validator import apply_retention_policy, structural_validate

from compaction_engine import HermesCompaction


# ---------------------------------------------------------------------------
# D1 -- SC4-ORDER-STALE: equal/missing timestamps + unordered fingerprint made
# a cache hit return the first run's ordering for a permuted input.
# ---------------------------------------------------------------------------
def test_d1_normalize_order_is_permutation_invariant_without_timestamps():
    messages = [{"role": "user", "content": f"m{i}"} for i in range(4)]
    forward = [m["content"] for m in normalize_order(messages)]
    reverse = [m["content"] for m in normalize_order(list(reversed(messages)))]
    assert forward == reverse
    assert sorted(forward) == ["m0", "m1", "m2", "m3"]


def test_d1_ordered_fingerprint_is_order_sensitive_raw_but_stable_normalized():
    messages = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    # Raw hash must be order-sensitive: a residual ordering difference can no
    # longer silently share an identity with another ordering.
    assert compute_messages_hash(messages) != compute_messages_hash(list(reversed(messages)))
    # After canonicalization the SAME set always yields the SAME fingerprint.
    permuted = [{"role": "user", "content": "b"}, {"role": "user", "content": "a"}]
    assert ordered_fingerprint(messages) == ordered_fingerprint(permuted)


# ---------------------------------------------------------------------------
# D3 -- SC2-F1 / SC1: individual mode marks every unit "solo", so a kind-based
# rule never sees tool content and hard_preserve_tool_links is inert.
# ---------------------------------------------------------------------------
def _tool_units():
    call = {"role": "assistant", "content": "c", "tool_calls": [{"id": "t1"}]}
    result = {"role": "tool", "content": "r", "tool_call_id": "t1"}
    other = {"role": "user", "content": "x"}
    units = [
        ConversationUnit("msg_0", "solo", [call]),
        ConversationUnit("msg_1", "solo", [result]),
        ConversationUnit("msg_2", "solo", [other]),
    ]
    return units, [call, result, other]


def _decisions(retain_call, retain_result):
    return {
        "msg_0": Decision("msg_0", retain_call, 0.9 if retain_call else 0.1, "clef"),
        "msg_1": Decision("msg_1", retain_result, 0.9 if retain_result else 0.1, "clef"),
        "msg_2": Decision("msg_2", False, 0.1, "clef"),
    }


def test_d3_flag_true_links_retained_result_back_to_its_call(base_config):
    units, original = _tool_units()
    cfg = build_config({**base_config, "hard_preserve_tool_links": True})
    out = apply_retention_policy(units, _decisions(False, True), cfg)

    assert "msg_1" in out.hard_preserved  # result is a tool unit by content
    assert "msg_0" in out.hard_preserved  # cross-unit link back to its call
    assert {u.unit_id for u in out.units} == {"msg_0", "msg_1"}
    assert structural_validate(original, out.messages).ok


def test_d3_flag_true_links_retained_call_forward_to_its_result(base_config):
    units, original = _tool_units()
    cfg = build_config({**base_config, "hard_preserve_tool_links": True})
    out = apply_retention_policy(units, _decisions(True, False), cfg)

    assert "msg_1" in out.hard_preserved  # result kept so no orphan is possible
    assert structural_validate(original, out.messages).ok


def test_d3_flag_false_is_observable_and_does_not_link(base_config):
    units, original = _tool_units()
    cfg = build_config({**base_config, "hard_preserve_tool_links": False})
    out = apply_retention_policy(units, _decisions(False, True), cfg)

    assert out.hard_preserved == set()
    assert {u.unit_id for u in out.units} == {"msg_1"}
    # Dropping the call while retaining the result is now an orphan: the
    # structural validator must reject it (and the engine falls back).
    assert not structural_validate(original, out.messages).ok


# ---------------------------------------------------------------------------
# D4 -- SC3: the split path ignored chunk_size when token budget was loose;
# 15 short messages became one oversized request.
# ---------------------------------------------------------------------------
def test_d4_split_honours_max_messages():
    messages = [{"role": "user", "content": f"short {i}"} for i in range(15)]
    unit = ConversationUnit("u0", "exchange", messages)
    groups = _split_oversized_unit(unit, budget=10**9, summary="s", max_messages=5)

    assert len(groups) == 3
    assert all(len(g[0].messages) <= 5 for g in groups)
    assert sum(len(g[0].messages) for g in groups) == 15
    assert [g[0].unit_id for g in groups] == ["u0_s0", "u0_s1", "u0_s2"]


def test_d4_three_argument_call_still_works():
    """Backward compatibility: max_messages defaults to None (unbounded)."""
    messages = [{"role": "user", "content": f"short {i}"} for i in range(15)]
    unit = ConversationUnit("u0", "exchange", messages)
    groups = _split_oversized_unit(unit, budget=10**9, summary="s")
    assert len(groups) == 1
    assert len(groups[0][0].messages) == 15


def test_d4_build_chunks_split_path_enforces_chunk_size(base_config):
    cfg = build_config({**base_config, "chunk_size": 5})
    messages = [{"role": "user", "content": f"m{i}"} for i in range(15)]
    unit = ConversationUnit("u0", "exchange", messages)
    chunks = build_chunks([unit], "summary", cfg)

    assert all(c.message_count <= 5 for c in chunks)
    assert sum(c.message_count for c in chunks) == 15


# ---------------------------------------------------------------------------
# D5 -- SC5 B3/B3b: non-dict messages crashed with AttributeError instead of
# producing a structured error / fail-safe verdict.
# ---------------------------------------------------------------------------
def test_d5_loader_rejects_non_dict_message(tmp_path):
    path = tmp_path / "ctx.json"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "ok"}, "bad"]}),
        encoding="utf-8",
    )
    with pytest.raises(ContextError, match="must be a dict"):
        load_from_file(path)


def test_d5_filter_non_dict_is_failsafe_not_a_crash(base_config):
    units = [
        ConversationUnit("u0", "solo", [{"role": "user", "content": "ok"}]),
        ConversationUnit("u1", "solo", ["not a dict"]),
    ]
    engine = RelevanceFilter(
        base_config,
        call_api=lambda _: pytest.fail("API must not be called for malformed input"),
        sleep=lambda _: None,
    )
    decisions, outcome = engine.filter_chunk(units, "summary")

    assert len(decisions) == 2 == len(outcome.decisions)
    assert all(d.retain for d in decisions)
    assert {d.source for d in decisions} == {"malformed_input"}
    assert any("not a dict" in d.reason for d in decisions)


def test_d5_engine_ctx_json_non_dict_returns_structured_error(tmp_path):
    ctx_dir = tmp_path / "ctx"
    ctx_dir.mkdir()
    (ctx_dir / "ctx.json").write_text(
        json.dumps([{"role": "user", "content": "ok"}, 42]), encoding="utf-8"
    )
    cfg = build_config({
        "cache_dir": str(tmp_path / "cache"),
        "context_dir": str(ctx_dir),
        "throttle_interval": 0,
    })
    result = HermesCompaction(cfg).run()

    assert "error" in result["metadata"]
    assert "must be a dict" in result["metadata"]["error"]
    assert result["messages"] == []
