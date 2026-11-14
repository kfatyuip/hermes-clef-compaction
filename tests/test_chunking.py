"""Token estimation and adaptive chunk construction."""

from __future__ import annotations

from chunk_manager import build_chunks, estimate_tokens, payload_tokens
from config import build_config
from conversation_grouper import ConversationUnit


def unit(i, content="x"):
    return ConversationUnit(
        unit_id=f"u{i}", kind="solo", messages=[{"role": "user", "content": content}]
    )


def test_token_estimation_is_conservative_for_cjk():
    assert estimate_tokens("中文" * 100) > estimate_tokens("a" * 200)
    assert estimate_tokens("") == 0


def test_payload_tokens_include_summary_and_questions():
    units = [unit(0)]
    assert payload_tokens(units, "summary") > payload_tokens(units, "")


def test_chunk_size_is_a_real_upper_bound(base_config):
    cfg = build_config({**base_config, "chunk_size": 3})
    units = [unit(i) for i in range(10)]
    chunks = build_chunks(units, "summary", cfg)
    assert all(c.message_count <= 3 for c in chunks)
    assert sum(c.message_count for c in chunks) == 10


def test_conversation_unit_not_split_when_it_fits(base_config):
    cfg = build_config({**base_config, "chunk_size": 10})
    exchange = ConversationUnit("u0", "exchange", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ])
    chunks = build_chunks([exchange], "summary", cfg)
    assert len(chunks) == 1
    assert len(chunks[0].units) == 1
    assert chunks[0].message_count == 2


def test_oversized_single_message_is_forced_retain(base_config):
    cfg = build_config({**base_config, "chunk_size": 1})
    huge = unit(0, "x" * 200_000)
    chunks = build_chunks([huge], "summary", cfg)
    assert len(chunks) == 1
    assert chunks[0].forced_retain
    assert "budget" in chunks[0].reason


def test_oversized_multi_message_unit_is_split(base_config):
    cfg = build_config({**base_config, "chunk_size": 20})
    big = ConversationUnit("u0", "exchange", [
        {"role": "user", "content": "y" * 60_000},
        {"role": "assistant", "content": "z" * 60_000},
    ])
    chunks = build_chunks([big], "summary", cfg)
    assert len(chunks) >= 2
    assert all(not c.forced_retain for c in chunks)
    assert sum(c.message_count for c in chunks) == 2


def test_chunks_preserve_order(base_config):
    cfg = build_config({**base_config, "chunk_size": 2})
    units = [unit(i) for i in range(6)]
    chunks = build_chunks(units, "s", cfg)
    ordered = [u.unit_id for c in chunks for u in c.units]
    assert ordered == [f"u{i}" for i in range(6)]
