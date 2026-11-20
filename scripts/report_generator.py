"""Comprehensive, deterministic retention reports.

Reports explain what actually happened: identities, configuration, API calls,
failures, per-source retention decisions, cache status, database status and
validation results. No authentication tokens or full message bodies are logged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _truncate(text: str, limit: int = 100) -> str:
    text = str(text)
    return text[:limit] + ("..." if len(text) > limit else "")


def build_report(
    *,
    session_id: str,
    messages_hash: str,
    summary_hash: str,
    algorithm_version: str,
    model: str,
    threshold: float,
    original_count: int,
    retained_count: int,
    chunk_count: int,
    filter_summary: dict[str, Any],
    retention_breakdown: dict[str, int],
    cache_status: str,
    cache_reason: str,
    database_status: str,
    backup_path: str | None,
    validation: dict[str, Any],
    elapsed: float,
    retained_messages: list[dict[str, Any]],
    fallback_summary: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    removed = max(0, original_count - retained_count)
    return {
        "session_id": session_id,
        "input_fingerprint": messages_hash,
        "summary_fingerprint": summary_hash,
        "algorithm_version": algorithm_version,
        "model": model,
        "threshold": threshold,
        "original_message_count": original_count,
        "retained_message_count": retained_count,
        "removed_message_count": removed,
        "retention_ratio": (retained_count / original_count) if original_count else 0.0,
        "chunk_count": chunk_count,
        "api_calls": filter_summary.get("api_calls", 0),
        "api_failures": filter_summary.get("failures", []),
        "missing_answers": filter_summary.get("missing_answers", 0),
        "malformed_answers": filter_summary.get("malformed_answers", 0),
        # Persist the full nested summary so a later cache hit can reconstruct
        # the §5 gate fields from this report instead of returning {}.
        "filter_summary": filter_summary,
        "retention_breakdown": retention_breakdown,
        "cache_status": cache_status,
        "cache_reason": cache_reason,
        "database_status": database_status,
        "backup_path": backup_path,
        "validation": validation,
        "elapsed_seconds": elapsed,
        "fallback_summary": fallback_summary,
        "dry_run": dry_run,
        "retained_preview": [
            {
                "role": m.get("role", ""),
                "content": _truncate(m.get("content", "")),
            }
            for m in retained_messages[:50]
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hermes Context Compression Report",
        "",
        "## Executive Summary",
        f"- **Session ID**: `{report['session_id']}`",
        f"- **Original messages**: {report['original_message_count']}",
        f"- **Retained messages**: {report['retained_message_count']}",
        f"- **Removed messages**: {report['removed_message_count']}",
        f"- **Retention ratio**: {report['retention_ratio'] * 100:.1f}%",
        f"- **Threshold**: {report['threshold']}",
        f"- **Algorithm version**: {report['algorithm_version']}",
        f"- **Model**: {report['model']}",
        f"- **Elapsed**: {report['elapsed_seconds']:.2f}s",
        "",
        "## Identity & Configuration",
        f"- Input fingerprint: `{report['input_fingerprint']}`",
        f"- Summary fingerprint: `{report['summary_fingerprint']}`",
        f"- Fallback (non-semantic) summary: {report['fallback_summary']}",
        f"- Dry run: {report['dry_run']}",
        "",
        "## Execution",
        f"- Chunks: {report['chunk_count']}",
        f"- API calls: {report['api_calls']}",
        f"- Missing answers: {report['missing_answers']}",
        f"- Malformed answers: {report['malformed_answers']}",
        f"- Cache: {report['cache_status']} ({report['cache_reason']})",
        f"- Database: {report['database_status']}",
        f"- Backup: `{report['backup_path']}`",
        f"- API failures: {report['api_failures'] or 'none'}",
        "",
        "## Retention Decisions",
    ]
    for key, value in report["retention_breakdown"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Validation"]
    validation = report["validation"]
    lines.append(f"- Passed: {validation.get('ok')}")
    for issue in validation.get("issues", []):
        lines.append(f"- ISSUE `{issue['code']}`: {issue['message']}")

    lines += ["", "## Retained Messages (preview)"]
    for i, msg in enumerate(report["retained_preview"], 1):
        lines.append(f"{i}. **{msg['role']}**: {msg['content']}")
    lines.append("")
    return "\n".join(lines)


def write_report(directory: str | Path, report: dict[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "retention_report.json"
    md_path = directory / "compaction_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return md_path
