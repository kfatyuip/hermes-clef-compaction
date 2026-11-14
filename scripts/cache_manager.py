"""Deterministic, content-addressed, safe cache.

Quality levels are explicit:

* ``hit``  -- every identity component and integrity check matched.
* ``miss`` -- no entry for this identity.
* ``invalid`` -- entry exists but is corrupted, tampered, expired or has
  different input/summary/threshold/config metadata. Never used.
* ``version_mismatch`` -- entry was produced by another algorithm version.

Writes are atomic and keep the previous valid result until the replacement is
fully written and verified. A directory lock serializes concurrent writers.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

CACHE_HIT = "hit"
CACHE_MISS = "miss"
CACHE_INVALID = "invalid"
CACHE_VERSION_MISMATCH = "version_mismatch"

_METADATA_KEYS = (
    "signature",
    "algorithm_version",
    "session_id",
    "messages_hash",
    "summary_hash",
    "threshold",
    "config_hash",
)


@dataclass
class CacheLookup:
    status: str
    data: dict[str, Any] | None = None
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.status == CACHE_HIT and self.data is not None


class CacheManager:
    def __init__(self, cache_dir: str | Path, retention_days: int = 30) -> None:
        self.root = Path(cache_dir) / "v2"
        self.retention_days = int(retention_days)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root / ".lock"
        self._lock_handle = None

    # -- lock ---------------------------------------------------------------
    def __enter__(self) -> "CacheManager":
        try:
            import fcntl  # POSIX; sandbox is Linux

            self._lock_handle = open(self._lock_path, "a+")
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            self._lock_handle = None
        return self

    def __exit__(self, *_: Any) -> None:
        if self._lock_handle is not None:
            try:
                import fcntl

                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                self._lock_handle.close()
            except (ImportError, OSError):
                pass
            self._lock_handle = None

    # -- paths --------------------------------------------------------------
    def _dir_for(self, signature: str) -> Path:
        return self.root / signature[:2] / signature

    @staticmethod
    def _read_json(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -- lookup -------------------------------------------------------------
    def get(
        self, signature: str, identity: dict[str, Any]
    ) -> CacheLookup:
        """Validate a cached result against the complete identity."""
        cache_dir = self._dir_for(signature)
        meta_path = cache_dir / "metadata.json"
        data_path = cache_dir / "compacted.json"

        if not cache_dir.is_dir():
            return CacheLookup(CACHE_MISS, reason="no_cache_directory")
        if not meta_path.is_file() or not data_path.is_file():
            return CacheLookup(
                CACHE_INVALID, reason="incomplete_cache_missing_files"
            )

        try:
            metadata = self._read_json(meta_path)
            data = self._read_json(data_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return CacheLookup(CACHE_INVALID, reason=f"corrupted_cache: {exc}")

        if not isinstance(metadata, dict) or not isinstance(data, dict):
            return CacheLookup(CACHE_INVALID, reason="cache_wrong_json_type")

        if metadata.get("signature") != signature:
            return CacheLookup(CACHE_INVALID, reason="signature_mismatch")
        if metadata.get("algorithm_version") != identity.get("algorithm_version"):
            return CacheLookup(
                CACHE_VERSION_MISMATCH,
                reason=(
                    f"version {metadata.get('algorithm_version')} != "
                    f"{identity.get('algorithm_version')}"
                ),
            )
        for key in ("session_id", "messages_hash", "summary_hash", "threshold", "config_hash"):
            if metadata.get(key) != identity.get(key):
                return CacheLookup(
                    CACHE_INVALID,
                    reason=f"identity_mismatch:{key}",
                )

        if self._is_expired(metadata):
            return CacheLookup(CACHE_INVALID, reason="expired")

        # Structural sanity of the cached result itself.
        if not isinstance(data.get("messages"), list):
            return CacheLookup(CACHE_INVALID, reason="cached_messages_not_list")

        return CacheLookup(CACHE_HIT, data=data, reason="validated")

    def _is_expired(self, metadata: dict[str, Any]) -> bool:
        created = metadata.get("created_at")
        if not created:
            return False
        try:
            moment = datetime.fromisoformat(str(created))
        except ValueError:
            return False
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - moment > timedelta(days=self.retention_days)

    # -- write --------------------------------------------------------------
    def put(
        self,
        signature: str,
        data: dict[str, Any],
        identity: dict[str, Any],
    ) -> Path:
        """Atomically write a validated result, preserving any previous entry."""
        target = self._dir_for(signature)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".tmp_", dir=str(target.parent)))
        try:
            metadata = {
                "signature": signature,
                "algorithm_version": identity.get("algorithm_version"),
                **{k: identity.get(k) for k in (
                    "session_id", "messages_hash", "summary_hash",
                    "threshold", "config_hash",
                )},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "original_count": data.get("original_count"),
                "retained_count": data.get("retained_count"),
                "compression_ratio": data.get("compression_ratio"),
            }
            self._write_json(tmp / "metadata.json", metadata)
            self._write_json(tmp / "compacted.json", data)

            # Verify what we just wrote before making it visible.
            check = self.get_from_dir(tmp, signature, identity)
            if check.status != CACHE_HIT:
                raise OSError(f"written cache failed verification: {check.reason}")

            if target.exists():
                previous = target.parent / f".previous_{signature}_{uuid.uuid4().hex[:8]}"
                os.replace(str(target), str(previous))
                self._prune_previous(target.parent, keep=1)
            os.replace(str(tmp), str(target))
            return target
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def get_from_dir(
        self, directory: Path, signature: str, identity: dict[str, Any]
    ) -> CacheLookup:
        """Validate a specific directory (used for atomic-write verification)."""
        meta_path = directory / "metadata.json"
        data_path = directory / "compacted.json"
        try:
            metadata = self._read_json(meta_path)
            data = self._read_json(data_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return CacheLookup(CACHE_INVALID, reason=f"corrupted_cache: {exc}")
        if metadata.get("signature") != signature:
            return CacheLookup(CACHE_INVALID, reason="signature_mismatch")
        for key in ("session_id", "messages_hash", "summary_hash", "threshold", "config_hash"):
            if metadata.get(key) != identity.get(key):
                return CacheLookup(CACHE_INVALID, reason=f"identity_mismatch:{key}")
        if not isinstance(data.get("messages"), list):
            return CacheLookup(CACHE_INVALID, reason="cached_messages_not_list")
        return CacheLookup(CACHE_HIT, data=data)

    @staticmethod
    def _prune_previous(parent: Path, keep: int = 1) -> None:
        previous = sorted(
            parent.glob(".previous_*"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for stale in previous[keep:]:
            shutil.rmtree(stale, ignore_errors=True)

    def report_dir(self, signature: str) -> Path:
        report_dir = self._dir_for(signature)
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir
