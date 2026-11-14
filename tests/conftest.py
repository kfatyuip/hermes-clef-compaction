"""Shared fixtures. Imports are wired to the flat scripts/ modules."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from config import build_config  # noqa: E402


@pytest.fixture
def base_config(tmp_path):
    return build_config(
        {
            "cache_dir": str(tmp_path / "cache"),
            "state_db": str(tmp_path / "state.db"),
            "context_dir": str(tmp_path / "cache"),
            "throttle_interval": 0,
            "api_max_retries": 2,
            "retry_base_delay": 0,
            "hard_preserve_recent": 0,
        }
    )


@pytest.fixture
def sample_messages():
    return [
        {
            "session_id": "sess-1",
            "role": "user",
            "content": f"message {i}",
            "timestamp": 1000.0 + i,
            "_rowid": i + 1,
            "id": i + 1,
        }
        for i in range(6)
    ]


@pytest.fixture
def tmp_state_db(tmp_path):
    """A realistic Hermes-like schema with EXTRA required columns."""
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            ended_at TEXT,
            last_activity_at REAL,
            started_at REAL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT,
            content TEXT,
            timestamp REAL,
            active INTEGER DEFAULT 1,
            compacted INTEGER DEFAULT 0,
            metadata TEXT,
            extra_required TEXT NOT NULL DEFAULT 'keep-me'
        );
        INSERT INTO sessions VALUES ('sess-1','local',NULL,2000.0,1000.0);
        INSERT INTO sessions VALUES ('sess-2','local',NULL,1500.0,1200.0);
        """
    )
    conn.commit()
    conn.close()
    return str(db)


def insert_rows(db: str, session_id: str, rows: list[dict]) -> None:
    conn = sqlite3.connect(db)
    try:
        for r in rows:
            conn.execute(
                "INSERT INTO messages "
                "(session_id, role, content, timestamp, active, compacted, metadata, extra_required) "
                "VALUES (?, ?, ?, ?, 1, 0, ?, 'keep-me')",
                (
                    session_id,
                    r["role"],
                    r["content"],
                    r["timestamp"],
                    r.get("metadata", "{}"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def clef_response(ids, scores):
    return {
        "success": True,
        "result": {
            "answers": {
                qid: {"noul": float(scores[qid])} for qid in ids
            }
        },
    }
