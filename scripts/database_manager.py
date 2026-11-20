"""Safe, transactional session reconstruction with verified backups.

Design decisions (see final report):

* **In-place activation, not delete-and-reinsert.** Retained messages are a
  subset of the original rows. We flip ``active``/``compacted`` flags on the
  *existing* rows, so every Hermes-required column and metadata field survives.
  This also avoids guessing the INSERT column list of an unknown schema and
  makes recovery a single flag flip.
* **Only rows we actually read are touched.** Messages appended to the session
  after the initial read keep their ``active`` state and stay visible to
  Hermes. This removes the read/API/write race condition of the original code.
* **SQLite backup API**, not ``shutil.copy2``: correct under WAL mode.
* **Optimistic concurrency check**: the source fingerprint is captured at read
  time and re-verified inside the write transaction. If our source rows changed,
  the write aborts rather than overwriting concurrent edits.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hashing import compute_messages_hash_unordered

logger = logging.getLogger("hermes_compaction.database")


class DatabaseError(RuntimeError):
    pass


@dataclass
class ReconstructResult:
    ok: bool
    session_id: str
    retained: int = 0
    backup_path: str | None = None
    errors: list[str] = field(default_factory=list)
    skipped_reason: str = ""


def _rows_to_messages(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    names = [d[0] for d in (cursor.description or [])]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _locator(msg: dict[str, Any]) -> tuple[str, Any] | None:
    """Return a stable primary locator for a loaded row, if available."""
    if msg.get("_rowid") is not None:
        return ("_rowid", msg["_rowid"])
    if msg.get("id") is not None:
        return ("id", msg["id"])
    return None


class DatabaseManager:
    def __init__(self, state_db: str | Path) -> None:
        self.path = Path(state_db)

    def _connect(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise DatabaseError(f"state.db not found: {self.path}")
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    # -- backup -------------------------------------------------------------
    def backup(self, label: str = "precompaction") -> Path:
        """Create a consistent backup using SQLite's online backup API."""
        if not self.path.is_file():
            raise DatabaseError(f"Cannot back up missing database: {self.path}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = self.path.with_name(f"{self.path.name}.{label}.{stamp}.bak")
        source: sqlite3.Connection | None = None
        dest: sqlite3.Connection | None = None
        try:
            source = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
            dest = sqlite3.connect(str(backup_path))
            source.backup(dest)  # WAL-safe, consistent snapshot
            result = dest.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                raise DatabaseError(f"Backup integrity check failed: {result}")
            logger.info("Verified database backup at %s", backup_path)
            return backup_path
        except sqlite3.Error as exc:
            if backup_path.exists():
                backup_path.unlink(missing_ok=True)
            raise DatabaseError(f"Backup failed: {exc}") from exc
        finally:
            if dest is not None:
                dest.close()
            if source is not None:
                source.close()

    def restore(self, backup_path: str | Path) -> None:
        """Restore the database from a verified backup (recovery procedure)."""
        backup = Path(backup_path)
        if not backup.is_file():
            raise DatabaseError(f"Backup not found: {backup}")
        try:
            conn = sqlite3.connect(str(backup))
            try:
                result = conn.execute("PRAGMA integrity_check").fetchone()
            finally:
                conn.close()
        except sqlite3.DatabaseError as exc:
            raise DatabaseError(f"Refusing to restore a corrupt backup: {exc}") from exc
        if not result or result[0] != "ok":
            raise DatabaseError("Refusing to restore a corrupt backup")

        # D1: a WAL-mode database keeps uncheckpointed frames in <db>-wal and
        # <db>-shm beside the main file. Replacing only the main file leaves
        # those sidecars behind, and SQLite may replay stale frames onto the
        # restored database -- corrupting it or resurrecting compacted rows.
        # Checkpoint/truncate and remove them before swapping the main file.
        self._neutralize_wal_sidecars(self.path)

        target = self.path.with_name(f"{self.path.name}.before_restore")
        if self.path.exists():
            self.path.replace(target)
        backup.replace(self.path)

        # Re-verify the restored main file before handing control back.
        try:
            check = sqlite3.connect(str(self.path))
            try:
                restored = check.execute("PRAGMA integrity_check").fetchone()
            finally:
                check.close()
        except sqlite3.DatabaseError as exc:
            raise DatabaseError(f"Restored database failed integrity check: {exc}") from exc
        if not restored or restored[0] != "ok":
            raise DatabaseError("Restored database failed integrity check")
        logger.warning("Restored state.db from %s (previous copy: %s)", backup, target)

    @staticmethod
    def _neutralize_wal_sidecars(db_path: Path) -> None:
        """Drop WAL/SHM sidecars so no stale frames survive a main-file swap."""
        wal = Path(str(db_path) + "-wal")
        shm = Path(str(db_path) + "-shm")
        if wal.exists() or shm.exists():
            try:
                conn = sqlite3.connect(str(db_path), timeout=30)
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                finally:
                    conn.close()
            except sqlite3.Error:
                # Best effort: the sidecars are removed below regardless of
                # whether a checkpoint was possible (e.g. another reader holds
                # the DB). Stale frames must never survive the restore.
                pass
        wal.unlink(missing_ok=True)
        shm.unlink(missing_ok=True)

    # -- snapshot -----------------------------------------------------------
    def session_fingerprint(self, session_id: str) -> str:
        """Fingerprint the current active message set of a session."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT rowid AS _rowid, * FROM messages "
                "WHERE active = 1 AND session_id = ? ORDER BY timestamp, rowid",
                (session_id,),
            )
            return compute_messages_hash_unordered(_rows_to_messages(cur))
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to snapshot session: {exc}") from exc
        finally:
            conn.close()

    def session_exists(self, session_id: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    # -- reconstruction -----------------------------------------------------
    def reconstruct(
        self,
        session_id: str,
        retained: list[dict[str, Any]],
        expected_fingerprint: str,
        source_messages: list[dict[str, Any]] | None = None,
        create_backup: bool = True,
    ) -> ReconstructResult:
        """Atomically reconstruct one session, aborting on source changes.

        Only rows present in ``source_messages`` are deactivated. Rows appended
        after the read are untouched, so concurrent Hermes activity cannot be
        silently lost. ``expected_fingerprint`` must be the fingerprint of that
        exact source set, computed at read time by the orchestrator.
        """
        result = ReconstructResult(ok=False, session_id=session_id)
        if not retained:
            result.errors.append("empty retained set is not a valid compression result")
            return result

        sids = {str(m.get("session_id")) for m in retained if m.get("session_id")}
        if sids and sids != {session_id}:
            result.errors.append(f"retained messages mix sessions: {sorted(sids)}")
            return result

        if create_backup:
            try:
                result.backup_path = str(self.backup())
            except DatabaseError as exc:
                result.errors.append(str(exc))
                return result

        source_rowids = [
            m["_rowid"] for m in (source_messages or [])
            if m.get("_rowid") is not None
        ]
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE")

            if source_rowids:
                placeholders = ",".join("?" for _ in source_rowids)
                cur.execute(
                    f"SELECT rowid AS _rowid, * FROM messages "
                    f"WHERE rowid IN ({placeholders}) AND session_id = ?",
                    (*source_rowids, session_id),
                )
                current = _rows_to_messages(cur)
                if compute_messages_hash_unordered(current) != expected_fingerprint:
                    raise DatabaseError(
                        "concurrent modification detected: source messages changed "
                        "since they were read"
                    )
                cur.execute(
                    f"UPDATE messages SET active = 0, compacted = 1 "
                    f"WHERE rowid IN ({placeholders}) AND session_id = ?",
                    (*source_rowids, session_id),
                )
            else:
                # Legacy path: callers without rowid locators (tests / explicit
                # session reconstruction). Still guarded by a fingerprint check.
                if self._fingerprint_on(conn, session_id) != expected_fingerprint:
                    raise DatabaseError("concurrent modification detected")
                cur.execute(
                    "UPDATE messages SET active = 0, compacted = 1 "
                    "WHERE session_id = ? AND active = 1",
                    (session_id,),
                )

            for msg in retained:
                locator = _locator(msg)
                if locator is None:
                    raise DatabaseError(
                        "retained message has no stable row locator; refusing to guess"
                    )
                column, value = locator
                sql_column = "rowid" if column == "_rowid" else column
                cur.execute(
                    f"UPDATE messages SET active = 1, compacted = 0 "
                    f"WHERE session_id = ? AND {sql_column} = ?",
                    (session_id, value),
                )

            # Verify inside the same transaction before committing.
            if source_rowids:
                placeholders = ",".join("?" for _ in source_rowids)
                cur.execute(
                    f"SELECT rowid AS _rowid, * FROM messages "
                    f"WHERE rowid IN ({placeholders}) AND session_id = ? AND active = 1",
                    (*source_rowids, session_id),
                )
            else:
                cur.execute(
                    "SELECT rowid AS _rowid, * FROM messages "
                    "WHERE active = 1 AND session_id = ?",
                    (session_id,),
                )
            final_rows = _rows_to_messages(cur)
            if len(final_rows) != len(retained):
                raise DatabaseError(
                    f"verification failed: expected {len(retained)} active rows, "
                    f"found {len(final_rows)}"
                )
            if (compute_messages_hash_unordered(final_rows)
                    != compute_messages_hash_unordered(retained)):
                raise DatabaseError(
                    "verification failed: retained content differs from reconstruction"
                )

            conn.commit()
            result.ok = True
            result.retained = len(final_rows)
            logger.info(
                "Reconstructed session %s with %d active messages",
                session_id,
                len(final_rows),
            )
        except (DatabaseError, sqlite3.Error) as exc:
            if conn is not None:
                conn.rollback()
            result.errors.append(str(exc))
            logger.error("Reconstruction failed for %s: %s", session_id, exc)
        finally:
            if conn is not None:
                conn.close()
        return result

    @staticmethod
    def _fingerprint_on(conn: sqlite3.Connection, session_id: str) -> str:
        cur = conn.cursor()
        cur.execute(
            "SELECT rowid AS _rowid, * FROM messages "
            "WHERE active = 1 AND session_id = ? ORDER BY timestamp, rowid",
            (session_id,),
        )
        return compute_messages_hash_unordered(_rows_to_messages(cur))
