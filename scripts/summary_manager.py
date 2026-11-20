"""External summary loading, validation and conservative fallback.

The external summary is the single most important interface between the Agent
and Clef. It is a *relevance reference*, never a replacement for the original
messages. This module keeps text and file-path handling strictly separated and
never interprets arbitrary long text as a filesystem path without guarding
the filesystem call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("hermes_compaction.summary")

# The five dimensions an Agent-generated semantic summary should cover. Used
# by SKILL.md and available to callers that want to validate a summary.
SUMMARY_DIMENSIONS = (
    "active_context",
    "important_historical_context",
    "persistent_constraints",
    "unresolved_issues",
    "preservation_priorities",
)

# Filesystem paths longer than this are rejected before any stat() call; this
# prevents "filename too long" errors on explicit ``external_summary_file``
# values. It is never applied to raw ``external_summary`` text.
_MAX_PATH_CANDIDATE_CHARS = 4096


class SummaryError(ValueError):
    """Raised when a summary cannot be loaded or is invalid."""


def _looks_like_readable_file(value: str) -> Path | None:
    """Return a Path only when *value* safely resolves to an existing file.

    A string is never treated as a path when it contains newlines or is longer
    than ``_MAX_PATH_CANDIDATE_CHARS``. All filesystem errors are swallowed
    here -- this function is a detector, not a loader.
    """
    if not value or "\n" in value or "\r" in value:
        return None
    if len(value) > _MAX_PATH_CANDIDATE_CHARS:
        return None
    try:
        path = Path(value).expanduser()
        return path if path.is_file() else None
    except (OSError, ValueError):
        return None


def _read_summary_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SummaryError(f"Cannot read summary file {path}: {exc}") from exc


def load_external_summary(config: dict[str, Any]) -> str | None:
    """Load the external summary from explicit text or file.

    Resolution order:
      1. ``external_summary_file`` (explicit path; errors are fatal)
      2. ``external_summary`` (raw text; ALWAYS treated as text, never as a
         path -- filesystem access happens only via ``external_summary_file``)
      3. ``None`` when external summaries are disabled or nothing is set

    A bare-string summary such as ``"README"`` is text even when a file with
    that name happens to exist in the working directory. Auto-detecting paths
    from raw text silently substituted unrelated file contents for the
    relevance reference (finding F14) and could drop important messages.

    Returns ``None`` when no summary is available so the caller can decide
    between a conservative fallback and aborting (batch vs Agent mode).
    """
    if not config.get("use_external_summary", False):
        return None

    file_value = config.get("external_summary_file")
    if file_value:
        path = _looks_like_readable_file(str(file_value))
        if path is None:
            # Explicitly requested a file that does not exist -> fail loudly.
            raise SummaryError(f"external_summary_file does not exist: {file_value}")
        return validate_summary(_read_summary_file(path), config)

    raw_value = config.get("external_summary")
    if raw_value is None or raw_value is False:
        return None
    text = str(raw_value)
    if not text:
        return None

    # F14 fix: raw external_summary is ALWAYS text. Never consult the
    # filesystem here -- a bare "README"/"TODO" that happens to name an
    # existing file must not silently replace the relevance reference.
    # Explicit file reads go through external_summary_file above.
    return validate_summary(text, config)


def validate_summary(summary: str, config: dict[str, Any]) -> str:
    """Strip and length-check a summary, applying the configured ceiling.

    The 2000-char recommendation is a *target*, not a hard limit. The ceiling
    is configurable and remains subject to the Clef token budget downstream.
    """
    text = (summary or "").strip()
    if not text:
        raise SummaryError("External summary is empty")

    max_chars = int(config.get("summary_max_chars", 8000))
    if len(text) > max_chars:
        logger.warning(
            "Summary truncated from %d to %d chars (configured ceiling)",
            len(text),
            max_chars,
        )
        text = text[:max_chars].rstrip()
    return text


def generate_fallback_summary(
    messages: list[dict[str, Any]], target_chars: int
) -> str:
    """Conservative, explicitly-labelled truncation summary.

    This is NOT semantic summarization. It exists only for batch mode when no
    Agent-generated summary is available. Because the summary is weak, the
    engine must pair it with a more conservative retention policy; the label
    makes that visible in reports and logs.
    """
    combined = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}" for m in messages
    )
    body = combined[: max(1, int(target_chars))]
    return (
        "[FALLBACK TRUNCATION SUMMARY - not semantically generated; "
        "conservative retention advised]\n" + body
    )


def is_fallback_summary(summary: str) -> bool:
    return summary.startswith("[FALLBACK TRUNCATION SUMMARY")
