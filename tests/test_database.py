"""Database safety: backups, in-place reconstruction, concurrency, rollback."""

from __future__ import annotations

import sqlite3

from config import build_config
from context_loader import input_fingerprint, load_from_db
from database_manager import DatabaseError, DatabaseManager

from conftest import insert_rows


def active_rows(db):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT id, content, active, compacted, extra_required FROM messages "
            "WHERE session_id = 'sess-1' ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def test_reconstruction_preserves_all_original_columns(tmp_state_db, base_config):
    insert_rows(tmp_state_db, "sess-1", [
        {"role": "user", "content": "a", "timestamp": 1.0},
        {"role": "assistant", "content": "b", "timestamp": 2.0},
        {"role": "user", "content": "c", "timestamp": 3.0},
    ])
    cfg = build_config({**base_config, "state_db": tmp_state_db})
    msgs = load_from_db(tmp_state_db, "sess-1", 48)
    retained = [m for m in msgs if m["content"] in ("a", "c")]
    dm = DatabaseManager(tmp_state_db)
    result = dm.reconstruct(
        "sess-1", retained, input_fingerprint(msgs), source_messages=msgs
    )
    assert result.ok, result.errors
    rows = active_rows(tmp_state_db)
    assert [r[1] for r in rows if r[2] == 1] == ["a", "c"]
    assert all(r[4] == "keep-me" for r in rows)  # extra column survived
    assert result.backup_path


def test_backup_is_verified_and_complete(tmp_state_db):
    insert_rows(tmp_state_db, "sess-1", [
        {"role": "user", "content": "a", "timestamp": 1.0}
    ])
    dm = DatabaseManager(tmp_state_db)
    backup = dm.backup()
    conn = sqlite3.connect(str(backup))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    finally:
        conn.close()


def test_backup_works_in_wal_mode(tmp_state_db):
    conn = sqlite3.connect(tmp_state_db)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.close()
    insert_rows(tmp_state_db, "sess-1", [
        {"role": "user", "content": "wal", "timestamp": 1.0}
    ])
    backup = DatabaseManager(tmp_state_db).backup()
    conn = sqlite3.connect(str(backup))
    try:
        assert conn.execute("SELECT content FROM messages").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_messages_added_after_read_are_not_deactivated(tmp_state_db, base_config):
    insert_rows(tmp_state_db, "sess-1", [
        {"role": "user", "content": "a", "timestamp": 1.0},
        {"role": "assistant", "content": "b", "timestamp": 2.0},
    ])
    msgs = load_from_db(tmp_state_db, "sess-1", 48)
    fingerprint = input_fingerprint(msgs)

    # Simulate Hermes receiving a new message while Clef is being called.
    insert_rows(tmp_state_db, "sess-1", [
        {"role": "user", "content": "arrived-late", "timestamp": 3.0}
    ])
    dm = DatabaseManager(tmp_state_db)
    result = dm.reconstruct(
        "sess-1", [msgs[0]], fingerprint, source_messages=msgs
    )
    assert result.ok, result.errors
    rows = active_rows(tmp_state_db)
    assert [r[1] for r in rows if r[2] == 1] == ["a", "arrived-late"]


def test_changed_source_aborts_and_rolls_back(tmp_state_db, base_config):
    insert_rows(tmp_state_db, "sess-1", [
        {"role": "user", "content": "a", "timestamp": 1.0},
        {"role": "assistant", "content": "b", "timestamp": 2.0},
    ])
    msgs = load_from_db(tmp_state_db, "sess-1", 48)
    fingerprint = input_fingerprint(msgs)

    conn = sqlite3.connect(tmp_state_db)
    conn.execute("UPDATE messages SET content = 'concurrent-edit' WHERE id = 1")
    conn.commit()
    conn.close()

    dm = DatabaseManager(tmp_state_db)
    result = dm.reconstruct(
        "sess-1", [msgs[0]], fingerprint, source_messages=msgs
    )
    assert not result.ok
    assert any("concurrent" in e for e in result.errors)
    assert [r[1] for r in active_rows(tmp_state_db) if r[2] == 1] == [
        "concurrent-edit", "b"
    ]


def test_mixed_session_retained_set_is_rejected(tmp_state_db, base_config):
    insert_rows(tmp_state_db, "sess-1", [
        {"role": "user", "content": "a", "timestamp": 1.0}
    ])
    msgs = load_from_db(tmp_state_db, "sess-1", 48)
    bad = [
        {"session_id": "sess-1", "role": "user", "content": "a", "timestamp": 1.0},
        {"session_id": "sess-2", "role": "user", "content": "b", "timestamp": 2.0},
    ]
    dm = DatabaseManager(tmp_state_db)
    result = dm.reconstruct("sess-1", bad, "fp", source_messages=msgs)
    assert not result.ok
    assert any("mix" in e for e in result.errors)


def test_empty_retained_set_is_rejected(tmp_state_db):
    dm = DatabaseManager(tmp_state_db)
    result = dm.reconstruct("sess-1", [], "fp")
    assert not result.ok
    assert any("empty" in e for e in result.errors)


def test_backup_failure_aborts_before_destructive_work(tmp_path, base_config):
    dm = DatabaseManager(str(tmp_path / "missing.db"))
    result = dm.reconstruct(
        "sess-1",
        [{"session_id": "sess-1", "_rowid": 1, "role": "user", "content": "a"}],
        "fp",
    )
    assert not result.ok
    assert any("backup" in e.lower() for e in result.errors)


def test_restore_from_backup(tmp_state_db):
    insert_rows(tmp_state_db, "sess-1", [
        {"role": "user", "content": "original", "timestamp": 1.0}
    ])
    dm = DatabaseManager(tmp_state_db)
    backup = dm.backup("test")
    conn = sqlite3.connect(tmp_state_db)
    conn.execute("UPDATE messages SET content = 'damaged'")
    conn.commit()
    conn.close()

    dm.restore(backup)
    assert active_rows(tmp_state_db)[0][1] == "original"


def test_corrupt_backup_refused(tmp_state_db, tmp_path):
    bad = tmp_path / "bad.bak"
    bad.write_text("not a sqlite database", encoding="utf-8")
    try:
        DatabaseManager(tmp_state_db).restore(bad)
    except DatabaseError as exc:
        assert "corrupt" in str(exc).lower() or "integrity" in str(exc).lower()
    else:
        raise AssertionError("corrupt backup must be refused")
