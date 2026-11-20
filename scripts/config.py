"""Configuration defaults, validation, and normalization.

Design notes
------------
* All defaults live here so that documentation and code cannot silently drift.
* ``result_relevant_keys()`` defines exactly which configuration values form
  part of the cache identity. Values that do not affect the *result* (e.g.
  throttling, logging, output paths) are deliberately excluded so that purely
  operational changes do not invalidate a valid cache entry.
* The compaction algorithm version is bumped whenever filtering / grouping /
  reconstruction semantics change. It is part of every cache identity so an
  old result can never be mistaken for a new one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Bump when filtering, grouping, retention or reconstruction semantics change.
# 2.1.0: canonical order is now content-intrinsic (permutation-invariant),
# cache identity uses an order-sensitive fingerprint, tool call/result pairs
# are linked across individual units, split chunks honour the message ceiling,
# and malformed (non-dict) input fails safe instead of crashing. All of these
# change cache identity / retention semantics, so every 2.0.0 cache entry is
# intentionally invalidated by this bump.
ALGORITHM_VERSION = "2.1.0"

DEFAULT_HARD_PRESERVE_MARKERS = ("[IMPORTANT]", "[MUST KEEP]", "[PRESERVE]", "[CRITICAL]")

# Cloudflare Clef-flash published limits (documented, not verified against a
# live account in this repository -- see final report's limitations section).
MAX_INPUT_TOKENS = 24_576
MAX_QUESTIONS_PER_REQUEST = 64

# Workers AI uses two names for the same model:
#   * the URL path uses the full model id, e.g. @cf/cloudflare/clef-flash
#   * the JSON body's "model" field uses the bare name, e.g. clef-flash
# Keep both here and derive the short name from the configured id.
CLEF_MODEL_ID = "@cf/cloudflare/clef-flash"
CLEF_MODEL_NAME = "clef-flash"


def clef_model_name(model_id: str | None) -> str:
    """Derive the JSON-body model name from a Workers AI model id.

    Accepts both the full id (``@cf/cloudflare/clef-flash``) and an already-short
    name (``clef-flash``), so a custom ``config["model"]`` keeps working.
    """
    if not model_id or not model_id.strip():
        return CLEF_MODEL_NAME
    name = model_id.strip().rsplit("/", 1)[-1]
    return name or CLEF_MODEL_NAME


DEFAULT_THRESHOLD = 0.3
DEFAULT_MAX_MESSAGES = 48
DEFAULT_SUMMARY_TARGET_CHARS = 2000
DEFAULT_SUMMARY_MAX_CHARS = 8000  # configurable ceiling, still token-budget bound

DEFAULT_CONFIG: dict[str, Any] = {
    "cloudflare_account_id": os.getenv("CLOUDFLARE_ACCOUNT_ID"),
    "cloudflare_auth_token": os.getenv("CLOUDFLARE_AUTH_TOKEN"),
    "model": CLEF_MODEL_ID,
    "chunk_size": 20,            # hard upper bound on messages per request
    "throttle_interval": 2,      # operational only; NOT part of cache identity
    "retention_days": 30,
    "compress_threshold": DEFAULT_THRESHOLD,
    "max_messages": DEFAULT_MAX_MESSAGES,
    "summary_target_chars": DEFAULT_SUMMARY_TARGET_CHARS,
    "summary_max_chars": DEFAULT_SUMMARY_MAX_CHARS,
    "cache_dir": str(Path.home() / ".hermes" / "cache" / "hcompaction"),
    "state_db": str(Path.home() / ".hermes" / "state.db"),
    "context_file": None,
    "output": None,
    # Safe defaults: destructive operations are OFF unless explicitly enabled.
    "recreate": False,
    "use_external_summary": True,
    "external_summary": None,
    "external_summary_file": None,
    "fallback": False,
    "session_id": None,
    "dry_run": False,
    # Relevance grouping & retention policy.
    "grouping_mode": "group",       # "group" (conversation units) | "individual"
    "hard_preserve_recent": 2,      # newest N messages always kept
    "hard_preserve_tool_links": True,
    "hard_preserve_markers": list(DEFAULT_HARD_PRESERVE_MARKERS),
    # API resilience.
    "api_max_retries": 3,
    "retry_base_delay": 0.5,
    "api_timeout": 30,
    # Verification.
    "semantic_verification": False,  # optional second Agent pass
}


def build_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge overrides over defaults and validate. Returns a new dict."""
    cfg: dict[str, Any] = {**DEFAULT_CONFIG, **(overrides or {})}
    cfg["cache_dir"] = str(cfg["cache_dir"])
    cfg["state_db"] = str(cfg["state_db"])
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Raise ValueError for invalid configuration. Fail fast, never guess."""
    threshold = cfg.get("compress_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError("compress_threshold must be a number")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("compress_threshold must be within [0, 1]")

    for key in ("chunk_size", "max_messages", "throttle_interval", "retention_days"):
        value = cfg.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")

    if cfg.get("chunk_size", 0) < 1:
        raise ValueError("chunk_size must be at least 1")
    if cfg.get("max_messages", 0) < 1:
        raise ValueError("max_messages must be at least 1")

    for key in ("summary_target_chars", "summary_max_chars"):
        value = cfg.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["summary_target_chars"] > cfg["summary_max_chars"]:
        raise ValueError("summary_target_chars cannot exceed summary_max_chars")

    recent = cfg.get("hard_preserve_recent")
    if not isinstance(recent, int) or isinstance(recent, bool) or recent < 0:
        raise ValueError("hard_preserve_recent must be a non-negative integer")

    if cfg.get("grouping_mode") not in ("group", "individual"):
        raise ValueError("grouping_mode must be 'group' or 'individual'")


def result_relevant_keys() -> tuple[str, ...]:
    """Config keys that must be part of the cache identity."""
    return (
        "compress_threshold",
        "max_messages",
        "chunk_size",
        "summary_max_chars",
        "model",
        "grouping_mode",
        "hard_preserve_recent",
        "hard_preserve_tool_links",
        "hard_preserve_markers",
        "algorithm_version",
    )


def result_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Project the full config down to result-relevant values."""
    projected: dict[str, Any] = {}
    for key in result_relevant_keys():
        if key == "algorithm_version":
            projected[key] = ALGORITHM_VERSION
        else:
            projected[key] = cfg.get(key)
    return projected
