"""Summary loading: text vs file, robustness, Unicode, limits."""

from __future__ import annotations

import pytest

from config import build_config
from summary_manager import (
    SummaryError,
    generate_fallback_summary,
    is_fallback_summary,
    load_external_summary,
)


def test_valid_raw_text_summary():
    cfg = build_config({"external_summary": "Current task: refactor the compactor."})
    assert load_external_summary(cfg) == "Current task: refactor the compactor."


def test_summary_file_explicit(tmp_path):
    path = tmp_path / "summary.txt"
    path.write_text("语义摘要：保持架构决策\n约束：中文优先", encoding="utf-8")
    cfg = build_config({"external_summary_file": str(path)})
    assert "架构决策" in load_external_summary(cfg)


def test_empty_summary_raises():
    cfg = build_config({"external_summary": "   \n  "})
    with pytest.raises(SummaryError):
        load_external_summary(cfg)


def test_missing_summary_returns_none():
    cfg = build_config({"external_summary": None})
    assert load_external_summary(cfg) is None


def test_disabled_external_summary_returns_none():
    cfg = build_config(
        {"external_summary": "ignored", "use_external_summary": False}
    )
    assert load_external_summary(cfg) is None


def test_very_long_summary_truncated_to_configured_ceiling():
    text = "长" * 5000
    cfg = build_config({
        "external_summary": text,
        "summary_target_chars": 100,
        "summary_max_chars": 120,
    })
    result = load_external_summary(cfg)
    assert len(result) == 120


def test_unicode_and_chinese_preserved():
    text = "任务：修复缓存 bug。约束：不要丢失工具调用上下文。"
    cfg = build_config({"external_summary": text})
    assert load_external_summary(cfg) == text


def test_summary_file_does_not_exist_raises():
    cfg = build_config({"external_summary_file": "/definitely/not/here.txt"})
    with pytest.raises(SummaryError):
        load_external_summary(cfg)


def test_summary_file_invalid_encoding_raises(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe\x00invalid-utf8")
    cfg = build_config({"external_summary_file": str(path)})
    with pytest.raises(SummaryError):
        load_external_summary(cfg)


def test_fallback_summary_is_labelled_and_bounded(sample_messages):
    summary = generate_fallback_summary(sample_messages, 50)
    assert is_fallback_summary(summary)
    assert len(summary) < 200
