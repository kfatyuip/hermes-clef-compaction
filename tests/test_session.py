"""Session isolation, canonical ordering, context source selection."""

from __future__ import annotations

import json

import pytest

from config import build_config
from context_loader import (
    ContextError,
    assert_single_session,
    load_from_db,
    load_messages,
    normalize_order,
)

from conftest import insert_rows


def test_db_messages_are_returned_chronological(tmp_state_db, base_config):
    insert_rows(tmp_state_db, "sess-1", [
        {"role": "user", "content": "oldest", "timestamp": 100.0},
        {"role": "assistant", "content": "middle", "timestamp": 200.0},
        {"role": "user", "content": "newest", "timestamp": 300.0},
    ])
    msgs = load_from_db(tmp_state_db, "sess-1", 48)
    assert [m["content"] for m in msgs] == ["oldest", "middle", "newest"]


def test_db_messages_limited_to_max_and_still_chronological(tmp_state_db):
    insert_rows(tmp_state_db, "sess-1", [
        {"role": "user", "content": str(i), "timestamp": float(i)} for i in range(6)
    ])
    msgs = load_from_db(tmp_state_db, "sess-1", 3)
    assert [m["content"] for m in msgs] == ["3", "4", "5"]


def test_mixed_session_ids_are_rejected():
    msgs = [
        {"session_id": "a", "role": "user", "content": "x"},
        {"session_id": "b", "role": "user", "content": "y"},
    ]
    with pytest.raises(ContextError):
        assert_single_session(msgs)


def test_missing_session_id_allowed_only_in_batch_mode():
    msgs = [{"role": "user", "content": "x"}]
    assert assert_single_session(msgs, allow_missing=True) == msgs
    with pytest.raises(ContextError):
        assert_single_session(msgs, allow_missing=False)


def test_expected_session_mismatch_is_rejected():
    msgs = [{"session_id": "a", "role": "user", "content": "x"}]
    with pytest.raises(ContextError):
        assert_single_session(msgs, expected="b")


def test_equal_timestamps_canonical_order_is_permutation_invariant():
    """D1: equal timestamps must not let input order decide canonical order.

    The old contract broke ties by list position, so the same message set in
    two different file orders produced two different canonical orders and a
    cache hit could return the stale one. The tie-break is now the message's
    content hash, which is invariant to input permutation.
    """
    msgs = [
        {"role": "user", "content": "first", "timestamp": 1.0},
        {"role": "assistant", "content": "second", "timestamp": 1.0},
        {"role": "user", "content": "third", "timestamp": 1.0},
    ]
    forward = [m["content"] for m in normalize_order(msgs)]
    reversed_order = [m["content"] for m in normalize_order(list(reversed(msgs)))]
    assert forward == reversed_order
    assert sorted(forward) == ["first", "second", "third"]


def test_batch_mode_reads_ctx_json(tmp_path, base_config):
    ctx = tmp_path / "cache" / "ctx.json"
    ctx.parent.mkdir(parents=True, exist_ok=True)
    ctx.write_text(json.dumps({
        "messages": [
            {"role": "user", "content": "a", "timestamp": 1.0},
            {"role": "assistant", "content": "b", "timestamp": 2.0},
        ]
    }), encoding="utf-8")
    cfg = build_config({**base_config, "context_dir": str(tmp_path / "cache")})
    msgs = load_messages(cfg)
    assert [m["content"] for m in msgs] == ["a", "b"]


def test_explicit_session_id_switches_source_to_db(tmp_state_db, base_config):
    cfg = build_config({
        **base_config,
        "state_db": tmp_state_db,
        "session_id": "sess-2",
    })
    insert_rows(tmp_state_db, "sess-2", [
        {"role": "user", "content": "only-sess-2", "timestamp": 1.0}
    ])
    msgs = load_messages(cfg)
    assert len(msgs) == 1
    assert msgs[0]["session_id"] == "sess-2"
