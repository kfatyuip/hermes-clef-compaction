"""Context loading with canonical ordering and strict session isolation.

One canonical internal ordering is used everywhere: chronological,
oldest -> newest. Messages retrieved newest-first for LIMIT efficiency are
explicitly reversed. Ties on timestamp are broken with a stable secondary key
(``id`` / ``rowid``) so repeated runs produce byte-identical input hashes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from hashing import compute_messages_hash_unordered

logger = logging.getLogger("hermes_compaction.context")


class ContextError(ValueError):
    """Raised when the context source is ambiguous or unsafe."""


def _columns(cursor: sqlite3.Cursor, table: str) -> list[str]:
    cursor.execute(f"PRAGMA table_info({table})")  # table name is internal
    return [row[1] for row in cursor.fetchall()]


def _rows_to_messages(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    names = [d[0] for d in (cursor.description or [])]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def get_current_session_id(state_db: str) -> str | None:
    """Return the most recent active local session id, or ``None``."""
    db = Path(state_db)
    if not db.is_file():
        logger.warning("state.db not found at %s", state_db)
        return None
    try:
        conn = sqlite3.connect(str(db))
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id FROM sessions
                WHERE source = 'local' AND ended_at IS NULL
                ORDER BY last_activity_at DESC, started_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return str(row[0]) if row else None
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.error("Error fetching current session id: %s", exc)
        return None


def load_from_db(
    state_db: str,
    session_id: str | None,
    max_messages: int,
) -> list[dict[str, Any]]:
    """Load active messages for exactly one session, oldest -> newest.

    All table columns are preserved so no Hermes-required metadata is lost on
    reconstruction. Mixed-session input is rejected by ``assert_single_session``.
    """
    db = Path(state_db)
    if not db.is_file():
        raise ContextError(f"state.db not found: {state_db}")

    sid = session_id or get_current_session_id(state_db)
    if not sid:
        raise ContextError("No session id available (auto-detect failed and none supplied)")

    try:
        conn = sqlite3.connect(str(db))
        try:
            cur = conn.cursor()
            cols = _columns(cur, "messages")
            secondary = "id" if "id" in cols else "rowid"
            order = "timestamp DESC" + (f", {secondary} DESC" if secondary else "")
            # Capture rowid as a stable locator for safe in-place reconstruction.
            cur.execute(
                f"""
                SELECT rowid AS _rowid, * FROM messages
                WHERE active = 1 AND session_id = ?
                ORDER BY {order}
                LIMIT ?
                """,
                (sid, max_messages),
            )
            newest_first = _rows_to_messages(cur)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise ContextError(f"Failed to load messages from state.db: {exc}") from exc

    messages = list(reversed(newest_first))  # canonical: oldest -> newest
    logger.info(
        "Loaded %d messages for session %s from state.db", len(messages), sid
    )
    return assert_single_session(messages, expected=sid, allow_missing=False)


def load_from_file(path: str | Path) -> list[dict[str, Any]]:
    """Load messages from a ctx.json-style file."""
    p = Path(path)
    if not p.is_file():
        raise ContextError(f"Context file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextError(f"Cannot read context file {p}: {exc}") from exc

    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict):
        messages = data.get("messages", data.get("recent_48_messages", []))
    else:
        raise ContextError("Context file must contain a list or an object with 'messages'")

    if not isinstance(messages, list):
        raise ContextError("'messages' in context file is not a list")
    messages = normalize_order(messages)
    logger.info("Loaded %d messages from context file %s", len(messages), p)
    return messages


def load_messages(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Select the explicit, consistent context source.

    ``ctx.json`` batch input and the live Hermes database are distinct sources.
    The chosen source is unambiguous: fallback / explicit session id / database
    for Agent mode; ctx.json only for batch mode.
    """
    if config.get("fallback") or config.get("session_id"):
        return load_from_db(
            config["state_db"],
            config.get("session_id"),
            int(config["max_messages"]),
        )

    context_dir = Path(config.get("context_dir", config["cache_dir"]))
    ctx_path = context_dir / "ctx.json"
    if ctx_path.is_file():
        return load_from_file(ctx_path)

    explicit = config.get("context_file")
    if explicit:
        return load_from_file(explicit)
    return []


def normalize_order(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a chronologically ordered copy with stable tie-breaking."""
    def key(item: tuple[int, dict[str, Any]]) -> tuple[float, int]:
        idx, msg = item
        ts = msg.get("timestamp", 0)
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            ts = 0.0
        return (ts, idx)

    indexed = sorted(enumerate(messages), key=key)
    return [msg for _, msg in indexed]


def assert_single_session(
    messages: list[dict[str, Any]],
    expected: str | None = None,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    """Reject inputs that mix sessions; optionally enforce an expected id.

    A missing session id can never silently become a synthetic id that later
    gets written into the real database. When ``allow_missing`` is true (batch
    mode only) all messages must consistently omit it.
    """
    ids = {str(m["session_id"]) for m in messages if m.get("session_id")}
    if len(ids) > 1:
        raise ContextError(f"Input mixes {len(ids)} session ids: {sorted(ids)}")
    if not ids:
        if allow_missing:
            return messages
        raise ContextError("Messages have no session_id; refusing to guess one")
    sid = next(iter(ids))
    if expected is not None and sid != expected:
        raise ContextError(f"Session mismatch: expected {expected}, got {sid}")
    if any(not m.get("session_id") for m in messages):
        raise ContextError("Some messages are missing session_id while others have it")
    return messages


def input_fingerprint(messages: list[dict[str, Any]]) -> str:
    """Stable fingerprint of the exact input message set (order-independent)."""
    return compute_messages_hash_unordered(messages)
