"""Core-chain defect fixes (dag:compaction-audit-fix/fix-core).

Finding ids reference the audit handoff ``audit-core.md``. Only in-scope
P0/P1 findings are covered here; P2 findings remain open by design.
"""

from __future__ import annotations

import pytest

from config import build_config
from summary_manager import load_external_summary


# ---------------------------------------------------------------------------
# F14 (P1) -- summary_manager.py: raw external_summary must never be read as a
# file just because its text coincides with an existing filename.
# ---------------------------------------------------------------------------

def test_f14_bare_filename_matching_existing_file_is_text(tmp_path, monkeypatch):
    """A summary equal to a real filename in cwd stays that literal text."""
    marker = "CONTENT MUST NEVER BE USED AS RELEVANCE REFERENCE"
    (tmp_path / "README").write_text(marker, encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # make the bare name resolvable via cwd

    cfg = build_config({"external_summary": "README"})
    result = load_external_summary(cfg)

    assert result == "README"
    assert marker not in result


def test_f14_relative_looking_path_is_text_even_when_it_exists(tmp_path, monkeypatch):
    """Raw text with path separators is still text, not a filesystem read."""
    sub = tmp_path / "docs"
    sub.mkdir()
    (sub / "summary.txt").write_text("WRONG FILE CONTENT", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    raw = "docs/summary.txt"
    cfg = build_config({"external_summary": raw})
    assert load_external_summary(cfg) == raw


def test_f14_explicit_summary_file_still_reads_file(tmp_path, monkeypatch):
    """Guard: explicit external_summary_file remains the only file-read path."""
    path = tmp_path / "summary.txt"
    path.write_text("语义摘要：保持架构决策", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    cfg = build_config({"external_summary_file": "summary.txt"})
    assert load_external_summary(cfg) == "语义摘要：保持架构决策"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
