"""End-to-end lifecycle tests with mocked Clef and a temporary Hermes-like DB."""

from __future__ import annotations

import json
import sqlite3

import pytest
import requests

from compaction_engine import HermesCompaction
from config import build_config

from conftest import insert_rows


class FakeResponse:
    def __init__(self, payload, status=200):
        self.status_code = status
        self.text = ""
        self._payload = payload

    def json(self):
        return self._payload


def install_fake_clef(monkeypatch, calls, *, fail_on=None, on_call=None, scorer=None):
    """Patch requests.post. ``fail_on`` is a set of 1-based call indices."""
    import relevance_filter

    def fake_post(url, json=None, headers=None, timeout=None):
        idx = len(calls) + 1
        calls.append({"url": url, "payload": json})
        if on_call:
            on_call(idx)
        if fail_on and idx in fail_on:
            raise requests.ConnectionError("simulated network failure")
        questions = (json or {}).get("questions", {})
        answers = {}
        for qid in questions:
            if scorer:
                score = scorer(idx, qid, questions[qid])
            else:
                score = 0.9
            answers[qid] = {"noul": score}
        return FakeResponse({"success": True, "result": {"answers": answers}})

    monkeypatch.setattr(relevance_filter.requests, "post", fake_post)


def make_cfg(tmp_path, tmp_state_db, **overrides):
    return build_config({
        "cache_dir": str(tmp_path / "cache"),
        "context_dir": str(tmp_path / "cache"),
        "state_db": tmp_state_db,
        "throttle_interval": 0,
        "cloudflare_account_id": "test-account",
        "cloudflare_auth_token": "test-token",
        "api_max_retries": 1,
        "retry_base_delay": 0,
        "fallback": True,
        "session_id": "sess-1",
        "chunk_size": 10,
        **overrides,
    })


def seed(db, n=6):
    rows = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"msg-{i}-drop" if i == 1 else f"msg-{i}"
        rows.append({"role": role, "content": content, "timestamp": 1000.0 + i})
    insert_rows(db, "sess-1", rows)


def test_full_pipeline_dry_run(monkeypatch, tmp_path, tmp_state_db):
    seed(tmp_state_db)
    calls = []
    install_fake_clef(monkeypatch, calls)
    cfg = make_cfg(tmp_path, tmp_state_db, dry_run=True,
                   external_summary="Semantic summary of the session")

    result = HermesCompaction(cfg).run()
    assert "error" not in result["metadata"]
    assert result["metadata"]["database_status"] == "dry_run"
    assert result["retained_count"] == 6
    assert calls  # Clef actually evaluated original messages
    conn = sqlite3.connect(tmp_state_db)
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE active=1").fetchone()[0] == 6
    conn.close()


def test_full_pipeline_with_recreate(monkeypatch, tmp_path, tmp_state_db):
    seed(tmp_state_db)
    calls = []
    install_fake_clef(
        monkeypatch,
        calls,
        scorer=lambda i, q, question: (
            0.1 if "msg-1-drop" in question["instructions"] else 0.9
        ),
    )
    cfg = make_cfg(tmp_path, tmp_state_db, recreate=True,
                   external_summary="Semantic summary")

    result = HermesCompaction(cfg).run()
    assert result["metadata"]["database_status"] == "reconstructed"
    assert result["metadata"]["success"] is True
    assert result["metadata"]["backup_path"]
    conn = sqlite3.connect(tmp_state_db)
    rows = conn.execute(
        "SELECT content, active, extra_required FROM messages ORDER BY id"
    ).fetchall()
    conn.close()
    # msg-0 and msg-1-drop form one conversation unit; the low score drops
    # the WHOLE unit (conversation-aware filtering), leaving the other four.
    assert [r[0] for r in rows if r[1] == 1] == [
        "msg-2", "msg-3", "msg-4", "msg-5"
    ]
    assert all(r[2] == "keep-me" for r in rows)


def test_api_failure_on_second_chunk_retains_that_chunk(
    monkeypatch, tmp_path, tmp_state_db
):
    seed(tmp_state_db)
    calls = []
    # chunk_size=2 with 6 messages grouped into 3 pairs -> 3 Clef calls.
    install_fake_clef(monkeypatch, calls, fail_on={2},
                      scorer=lambda i, q, question: (
                          0.1 if "msg-0" in question["instructions"] else 0.9
                      ))
    cfg = make_cfg(tmp_path, tmp_state_db, chunk_size=2,
                   external_summary="summary")
    result = HermesCompaction(cfg).run()
    # 3 conversation units; unit-1 (msg-0/msg-1) is dropped by a valid low
    # score, unit-2's API call failed so both its messages are kept fail-safe,
    # unit-3 is retained normally -> 4 messages.
    assert result["retained_count"] == 4
    contents = [m["content"] for m in result["messages"]]
    assert "msg-2" in contents and "msg-3" in contents
    report = result["metadata"]
    assert report["validation_ok"] is True


def test_changed_summary_produces_different_cache_identity(
    monkeypatch, tmp_path, tmp_state_db
):
    seed(tmp_state_db)
    calls = []
    install_fake_clef(monkeypatch, calls)
    cfg_a = make_cfg(tmp_path, tmp_state_db, external_summary="summary A")
    result_a = HermesCompaction(cfg_a).run()
    summary_hash_a = result_a["metadata"]["summary_fingerprint"]
    calls_after_a = len(calls)

    cfg_b = make_cfg(tmp_path, tmp_state_db, external_summary="summary B")
    result_b = HermesCompaction(cfg_b).run()
    assert result_b["metadata"]["summary_fingerprint"] != summary_hash_a
    assert result_b["metadata"]["cached"] is False
    assert len(calls) > calls_after_a


def test_cache_hit_skips_api(monkeypatch, tmp_path, tmp_state_db):
    seed(tmp_state_db)
    calls = []
    install_fake_clef(monkeypatch, calls)
    cfg = make_cfg(tmp_path, tmp_state_db, external_summary="stable summary")

    first = HermesCompaction(cfg).run()
    after_first = len(calls)
    second = HermesCompaction(cfg).run()
    assert second["metadata"]["cached"] is True
    assert second["metadata"]["cache_status"] == "hit"
    assert len(calls) == after_first
    assert second["messages"] == first["messages"]


def test_new_messages_during_compression_remain_active(
    monkeypatch, tmp_path, tmp_state_db
):
    seed(tmp_state_db, n=3)
    calls = []

    def on_call(idx):
        if idx == 1:
            insert_rows(tmp_state_db, "sess-1", [
                {"role": "user", "content": "arrived-during-clef", "timestamp": 9999.0}
            ])

    install_fake_clef(monkeypatch, calls, on_call=on_call)
    cfg = make_cfg(tmp_path, tmp_state_db, recreate=True,
                   external_summary="summary")
    result = HermesCompaction(cfg).run()
    assert result["metadata"]["database_status"] == "reconstructed"
    conn = sqlite3.connect(tmp_state_db)
    active = conn.execute(
        "SELECT content FROM messages WHERE active=1 ORDER BY id"
    ).fetchall()
    conn.close()
    assert "arrived-during-clef" in [r[0] for r in active]


def test_failed_reconstruction_does_not_poison_cache(
    monkeypatch, tmp_path, tmp_state_db
):
    seed(tmp_state_db)
    calls = []

    def mutate_source(idx):
        if idx == 1:
            conn = sqlite3.connect(tmp_state_db)
            conn.execute("UPDATE messages SET content='mutated' WHERE id=1")
            conn.commit()
            conn.close()

    install_fake_clef(monkeypatch, calls, on_call=mutate_source)
    cfg = make_cfg(tmp_path, tmp_state_db, recreate=True,
                   external_summary="summary")
    result = HermesCompaction(cfg).run()
    assert result["metadata"]["database_status"] == "failed"
    assert result["metadata"]["success"] is False
    assert result["metadata"]["backup_path"]

    cache_files = list((tmp_path / "cache" / "v2").rglob("compacted.json"))
    assert cache_files == []


def test_no_semantic_summary_retains_all_without_api(
    monkeypatch, tmp_path, tmp_state_db
):
    seed(tmp_state_db)
    calls = []
    install_fake_clef(monkeypatch, calls)
    cfg = make_cfg(
        tmp_path, tmp_state_db,
        external_summary=None,
        use_external_summary=True,
        fallback=False,
        analyze_only=True,
    )
    # With no summary and not in fallback mode, the engine must not call Clef.
    result = HermesCompaction(cfg).run()
    assert result["retained_count"] == result["original_count"]
    assert calls == []
