"""IO/validation defect fixes (dag:compaction-audit-fix/fix-io).

Finding ids reference the audit handoff ``audit-io.md``. Only in-scope
P0/P1 findings are covered here. V1 (validator.py, P0) is out of this
node's runtime write scope and is reported in fix-io.md instead.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import requests

from cache_manager import CacheManager
from config import build_config
from database_manager import DatabaseManager
from relevance_filter import RelevanceFilter

from conftest import insert_rows


# ---------------------------------------------------------------------------
# C1 (P1) -- cache_manager.py:224-229. _prune_previous globbed .previous_*
# across the whole shard directory, deleting a sibling signature's rollback
# backup whenever a different signature was rewritten.
# ---------------------------------------------------------------------------
def _identity(signature: str) -> dict:
    return {
        "signature": signature,
        "algorithm_version": "v1",
        "session_id": "sess-1",
        "messages_hash": "mh",
        "summary_hash": "sh",
        "threshold": 0.5,
        "config_hash": "ch",
    }


def _put(cm: CacheManager, signature: str, marker: str) -> None:
    cm.put(
        signature,
        {
            "messages": [{"role": "user", "content": marker}],
            "original_count": 1,
            "retained_count": 1,
        },
        _identity(signature),
    )


def test_c1_prune_is_scoped_to_signature(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    sig1 = "aa" + "1" * 10
    sig2 = "aa" + "2" * 10  # same shard directory (root/aa/) as sig1

    with cm:
        _put(cm, sig1, "v1")
        _put(cm, sig1, "v2")
        _put(cm, sig1, "v3")  # two backups created, prune must keep one
        _put(cm, sig2, "v1")
        _put(cm, sig2, "v2")  # used to delete sig1's backup

    shard = cm.root / "aa"
    backups1 = sorted(shard.glob(f".previous_{sig1}_*"))
    backups2 = sorted(shard.glob(f".previous_{sig2}_*"))

    assert len(backups1) == 1, "sig1 rollback backup must survive sibling writes"
    assert len(backups2) == 1, "sig2 must retain its own latest backup"
    assert (backups1[0] / "compacted.json").is_file()
    assert (backups2[0] / "compacted.json").is_file()


# ---------------------------------------------------------------------------
# D1 (P1) -- database_manager.py:115-119. Restore replaced only the main DB
# file, leaving <db>-wal/-shm sidecars that can replay stale frames onto the
# restored database (corruption / resurrected compacted rows).
# ---------------------------------------------------------------------------
def test_d1_restore_removes_stale_wal_sidecars(tmp_state_db):
    insert_rows(
        tmp_state_db,
        "sess-1",
        [{"role": "user", "content": "original", "timestamp": 1.0}],
    )
    dm = DatabaseManager(tmp_state_db)
    backup = dm.backup("test")

    wal = Path(str(tmp_state_db) + "-wal")
    shm = Path(str(tmp_state_db) + "-shm")
    wal.write_bytes(b"stale-wal-frames")
    shm.write_bytes(b"stale-shm")

    dm.restore(backup)

    assert not wal.exists(), "stale WAL sidecar must not survive restore"
    assert not shm.exists(), "stale SHM sidecar must not survive restore"
    conn = sqlite3.connect(tmp_state_db)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT content FROM messages").fetchone()[0] == "original"
    finally:
        conn.close()


def test_d1_neutralize_helper_clears_live_wal(tmp_state_db):
    conn = sqlite3.connect(tmp_state_db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) "
        "VALUES (?, ?, ?, ?)",
        ("sess-1", "user", "wal-frame", 1.0),
    )
    conn.commit()
    wal = Path(str(tmp_state_db) + "-wal")
    shm = Path(str(tmp_state_db) + "-shm")
    assert wal.exists()
    try:
        DatabaseManager._neutralize_wal_sidecars(Path(tmp_state_db))
        assert not wal.exists()
        assert not shm.exists()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# R1 (P1) -- relevance_filter.py:262,278-284,329-330. Duplicate/None unit ids
# collapsed in the questions/decisions dicts, leaving units without a verdict
# (silent drop). Every submitted unit must now receive a unique verdict.
# ---------------------------------------------------------------------------
def _unit(unit_id):
    return SimpleNamespace(
        unit_id=unit_id, messages=[{"role": "user", "content": "unit body"}]
    )


def _retain_all_api(payload):
    qids = list(payload["questions"].keys())
    return {
        "success": True,
        "result": {"answers": {q: {"noul": 0.9} for q in qids}},
    }


def _engine(cfg, call_api=_retain_all_api):
    return RelevanceFilter(cfg, call_api=call_api, sleep=lambda _: None)


def test_r1_none_ids_get_distinct_verdicts(base_config):
    cfg = build_config({**base_config, "compress_threshold": 0.5})
    units = [_unit(None), _unit(None)]
    decisions, outcome = _engine(cfg).filter_chunk(units, "summary")

    assert len(decisions) == 2
    assert len(outcome.decisions) == 2
    assert {d.question_id for d in decisions} == {"unit_0", "unit_1"}
    assert outcome.retained_ids == {"unit_0", "unit_1"}
    assert all(d.retain for d in decisions)


def test_r1_duplicate_ids_get_distinct_verdicts(base_config):
    cfg = build_config({**base_config, "compress_threshold": 0.5})
    units = [_unit("dup"), _unit("dup"), _unit("unique")]
    decisions, outcome = _engine(cfg).filter_chunk(units, "summary")

    assert len(decisions) == 3
    assert len(outcome.decisions) == 3
    assert {d.question_id for d in decisions} == {"dup", "unit_1", "unique"}


def test_r1_forced_retain_path_is_complete(base_config):
    cfg = build_config({**base_config, "compress_threshold": 0.5})
    units = [_unit(None), _unit(None)]
    decisions, outcome = _engine(cfg).filter_chunk(
        units, "summary", chunk_forced_retain=True, chunk_reason="oversized"
    )
    assert len(decisions) == 2
    assert all(d.source == "forced" for d in decisions)


def test_r1_api_error_path_is_complete(base_config):
    cfg = build_config({**base_config, "api_max_retries": 0})

    def boom(_payload):
        raise requests.ConnectionError("offline")

    units = [_unit(None), _unit(None)]
    decisions, outcome = _engine(cfg, call_api=boom).filter_chunk(units, "summary")
    assert len(decisions) == 2
    assert all(d.source == "api_error" and d.retain for d in decisions)

