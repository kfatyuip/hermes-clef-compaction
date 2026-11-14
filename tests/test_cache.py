"""Cache identity correctness and integrity validation."""

from __future__ import annotations

import json

from cache_manager import (
    CACHE_HIT,
    CACHE_INVALID,
    CACHE_MISS,
    CACHE_VERSION_MISMATCH,
    CacheManager,
)
from config import ALGORITHM_VERSION, build_config, result_config
from hashing import (
    compute_cache_signature,
    compute_config_hash,
    compute_messages_hash,
    compute_summary_hash,
)


def identity_for(cfg, messages, summary, session_id="sess-1", version=ALGORITHM_VERSION):
    mh = compute_messages_hash(messages)
    sh = compute_summary_hash(summary)
    ch = compute_config_hash(result_config(cfg))
    sig = compute_cache_signature(
        session_id=session_id,
        messages_hash=mh,
        summary_hash=sh,
        threshold=cfg["compress_threshold"],
        config_hash=ch,
        algorithm_version=version,
    )
    return sig, {
        "signature": sig,
        "algorithm_version": version,
        "session_id": session_id,
        "messages_hash": mh,
        "summary_hash": sh,
        "threshold": cfg["compress_threshold"],
        "config_hash": ch,
    }


def test_same_messages_same_summary_is_hit(base_config, sample_messages):
    sig, ident = identity_for(base_config, sample_messages, "summary A")
    data = {"messages": sample_messages[:2], "original_count": 6, "retained_count": 2}
    with CacheManager(base_config["cache_dir"]) as cm:
        cm.put(sig, data, ident)
        lookup = cm.get(sig, ident)
    assert lookup.status == CACHE_HIT
    assert lookup.usable


def test_different_summary_changes_signature(base_config, sample_messages):
    _, ident_a = identity_for(base_config, sample_messages, "summary A")
    _, ident_b = identity_for(base_config, sample_messages, "summary B")
    assert ident_a["signature"] != ident_b["signature"]
    assert ident_a["summary_hash"] != ident_b["summary_hash"]


def test_different_summary_never_hits_same_entry(base_config, sample_messages):
    sig, ident_a = identity_for(base_config, sample_messages, "summary A")
    data = {"messages": sample_messages[:1], "original_count": 6, "retained_count": 1}
    with CacheManager(base_config["cache_dir"]) as cm:
        cm.put(sig, data, ident_a)
        _, ident_b = identity_for(base_config, sample_messages, "summary B")
        lookup = cm.get(sig, ident_b)
    assert lookup.status == CACHE_INVALID
    assert "summary_hash" in lookup.reason


def test_different_messages_change_signature(base_config, sample_messages):
    _, a = identity_for(base_config, sample_messages, "s")
    _, b = identity_for(base_config, sample_messages[:-1], "s")
    assert a["signature"] != b["signature"]


def test_different_threshold_changes_signature(base_config, sample_messages):
    other = build_config({**base_config, "compress_threshold": 0.8})
    _, a = identity_for(base_config, sample_messages, "s")
    _, b = identity_for(other, sample_messages, "s")
    assert a["signature"] != b["signature"]


def test_different_config_changes_signature(base_config, sample_messages):
    other = build_config({**base_config, "hard_preserve_recent": 5})
    _, a = identity_for(base_config, sample_messages, "s")
    _, b = identity_for(other, sample_messages, "s")
    assert a["signature"] != b["signature"]


def test_version_mismatch_is_distinguished(base_config, sample_messages):
    sig, ident = identity_for(base_config, sample_messages, "s")
    data = {"messages": [], "original_count": 6, "retained_count": 0}
    with CacheManager(base_config["cache_dir"]) as cm:
        cm.put(sig, data, ident)
        _, old_ident = identity_for(base_config, sample_messages, "s", version="1.0.0")
        lookup = cm.get(sig, old_ident)
    assert lookup.status == CACHE_VERSION_MISMATCH


def test_missing_cache_is_miss(base_config, sample_messages):
    sig, ident = identity_for(base_config, sample_messages, "s")
    with CacheManager(base_config["cache_dir"]) as cm:
        assert cm.get(sig, ident).status == CACHE_MISS


def test_corrupted_cache_is_invalid(base_config, sample_messages):
    sig, ident = identity_for(base_config, sample_messages, "s")
    data = {"messages": [], "original_count": 6, "retained_count": 0}
    with CacheManager(base_config["cache_dir"]) as cm:
        cm.put(sig, data, ident)
        target = cm._dir_for(sig) / "compacted.json"
        target.write_text("{broken json", encoding="utf-8")
        assert cm.get(sig, ident).status == CACHE_INVALID


def test_incomplete_cache_is_invalid(base_config, sample_messages):
    sig, ident = identity_for(base_config, sample_messages, "s")
    with CacheManager(base_config["cache_dir"]) as cm:
        directory = cm._dir_for(sig)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text("{}", encoding="utf-8")
        assert cm.get(sig, ident).status == CACHE_INVALID


def test_put_preserves_previous_result_until_replacement(base_config, sample_messages):
    sig, ident = identity_for(base_config, sample_messages, "s")
    with CacheManager(base_config["cache_dir"]) as cm:
        cm.put(sig, {"messages": [sample_messages[0]], "original_count": 6,
                     "retained_count": 1}, ident)
        cm.put(sig, {"messages": sample_messages[:2], "original_count": 6,
                     "retained_count": 2}, ident)
        lookup = cm.get(sig, ident)
        assert lookup.status == CACHE_HIT
        assert lookup.data["retained_count"] == 2
        assert list(cm._dir_for(sig).parent.glob(".previous_*"))
