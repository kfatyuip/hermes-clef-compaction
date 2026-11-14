"""Stable hashing and content-addressed cache identity.

A cache hit means: the exact same relevant inputs AND the exact same algorithm
configuration have already been processed. Merely sharing a session id is not
a cache hit -- that was the central bug of the original implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization (sorted keys, no incidental spaces)."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_messages_hash(messages: list[dict[str, Any]]) -> str:
    """Hash the complete input message set, including every preserved field.

    We deliberately hash ALL fields (role, content, timestamp, tool-call ids,
    message ids, ...) instead of only content+timestamp: two runs that differ
    in metadata are different inputs and must never share a cache entry.
    """
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        normalized.append(_normalize_message(msg))
    return sha256_text(canonical_json(normalized))


def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Copy a message with deterministic value normalization.

    Floats that are NaN/inf are converted to stable strings so hashing never
    depends on platform-specific float repr.
    """
    out: dict[str, Any] = {}
    for key in sorted(msg):
        value = msg[key]
        if isinstance(value, float) and not math.isfinite(value):
            out[key] = f"<nonfinite:{value!r}>"
        elif isinstance(value, dict):
            out[key] = {k: value[k] for k in sorted(value)}
        else:
            out[key] = value
    return out


def compute_messages_hash_unordered(messages: list[dict[str, Any]]) -> str:
    """Order-insensitive fingerprint of a message SET.

    Used for database state comparison, where the retrieval tie-breaker may
    differ from the one used at load time. Ordering correctness is verified
    separately by the structural validator.
    """
    normalized = [_normalize_message(m) for m in messages]
    ordered = sorted(normalized, key=lambda m: canonical_json(m))
    return sha256_text(canonical_json(ordered))


def compute_summary_hash(summary: str | None) -> str:
    """Hash the external summary. ``None`` and empty string are distinct."""
    if summary is None:
        return sha256_text("<no-summary>")
    return sha256_text(summary)


def compute_config_hash(result_config: dict[str, Any]) -> str:
    return sha256_text(canonical_json(result_config))


def compute_cache_signature(
    *,
    session_id: str,
    messages_hash: str,
    summary_hash: str,
    threshold: float,
    config_hash: str,
    algorithm_version: str,
) -> str:
    """Build the single content-addressed cache identity.

    Conceptual formula (see spec):
        SHA256(session_id + messages_hash + summary_hash + threshold
               + config_hash + algorithm_version)

    Each component is length-delimited inside the canonical JSON payload to
    remove any ambiguity about component boundaries.
    """
    payload = {
        "v": 1,
        "session_id": session_id,
        "messages_hash": messages_hash,
        "summary_hash": summary_hash,
        "threshold": float(threshold),
        "config_hash": config_hash,
        "algorithm_version": algorithm_version,
    }
    return sha256_text(canonical_json(payload))
