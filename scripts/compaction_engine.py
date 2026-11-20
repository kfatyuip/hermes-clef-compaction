#!/usr/bin/env python3
"""Hermes context compression -- orchestration layer.

This module intentionally contains no filtering, grouping, hashing or database
logic. Those responsibilities live in independently testable modules; this file
only executes the lifecycle:

    load -> summarize -> identify -> cache-check -> group -> chunk -> filter
         -> retain -> validate -> (backup) -> reconstruct -> cache/report

Agent mode (semantic summary supplied) and batch/fallback mode (conservative
truncation summary) are explicitly separated. Destructive database writes are
OFF by default and require ``--recreate``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from cache_manager import CACHE_MISS, CacheManager
from chunk_manager import build_chunks
from config import ALGORITHM_VERSION, CLEF_MODEL_ID, build_config, result_config
from context_loader import ContextError, assert_single_session, input_fingerprint, load_messages
from conversation_grouper import ConversationUnit, group_messages
from database_manager import DatabaseManager
from diagnostics import diagnose_messages
from hashing import (
    compute_cache_signature,
    compute_config_hash,
    compute_summary_hash,
)
from relevance_filter import FilterOutcome, RelevanceFilter
from report_generator import build_report, write_report
from summary_manager import (
    SummaryError,
    generate_fallback_summary,
    is_fallback_summary,
    load_external_summary,
)
from validator import RetentionOutcome, apply_retention_policy, structural_validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("hermes_compaction")


class HermesCompaction:
    """Orchestrates the compaction pipeline. ``run()`` returns a result dict."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = build_config(config)
        self.cache_dir = Path(self.config["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = "batch-no-session"
        self.summary_error: str | None = None

    # -- helpers ------------------------------------------------------------
    def _error_result(self, message: str) -> dict[str, Any]:
        return {
            "session_id": None,
            "original_count": 0,
            "retained_count": 0,
            "compression_ratio": 0,
            "messages": [],
            "metadata": {
                "error": message,
                "cached": False,
                "algorithm_version": ALGORITHM_VERSION,
                "timestamp": datetime.now().isoformat(),
            },
        }

    def _identity(
        self, session_id: str, messages: list[dict[str, Any]], summary: str | None
    ) -> tuple[str, dict[str, Any]]:
        messages_hash = input_fingerprint(messages)
        summary_hash = compute_summary_hash(summary)
        config_hash = compute_config_hash(result_config(self.config))
        threshold = float(self.config["compress_threshold"])
        signature = compute_cache_signature(
            session_id=session_id,
            messages_hash=messages_hash,
            summary_hash=summary_hash,
            threshold=threshold,
            config_hash=config_hash,
            algorithm_version=ALGORITHM_VERSION,
        )
        identity = {
            "signature": signature,
            "algorithm_version": ALGORITHM_VERSION,
            "session_id": session_id,
            "messages_hash": messages_hash,
            "summary_hash": summary_hash,
            "threshold": threshold,
            "config_hash": config_hash,
        }
        return signature, identity

    def _process(
        self, messages: list[dict[str, Any]], summary: str | None
    ) -> tuple[list[dict[str, Any]], FilterOutcome, RetentionOutcome, Any, int]:
        """Group, chunk, filter, retain and structurally validate."""
        mode = self.config.get("grouping_mode", "group")

        # No semantic summary in Agent mode: do NOT call the API and do NOT
        # pretend truncation is a summary. Retain everything conservatively.
        if summary is None:
            units = [
                ConversationUnit(f"msg_{i}", "solo", [msg])
                for i, msg in enumerate(messages)
            ]
            retention = RetentionOutcome(
                units=units,
                hard_preserved={u.unit_id for u in units},
            )
            validation = structural_validate(messages, messages)
            return messages[:], FilterOutcome(), retention, validation, 0

        if mode == "individual":
            units = [
                ConversationUnit(f"msg_{i}", "solo", [msg])
                for i, msg in enumerate(messages)
            ]
        else:
            units = group_messages(messages)

        chunks = build_chunks(units, summary, self.config)
        logger.info("Built %d chunks from %d units", len(chunks), len(units))

        engine = RelevanceFilter(self.config)
        combined = FilterOutcome()
        all_decisions: dict[str, Any] = {}
        for index, chunk in enumerate(chunks):
            if index > 0 and self.config.get("throttle_interval", 0):
                time.sleep(float(self.config["throttle_interval"]))
            decisions, outcome = engine.filter_chunk(
                chunk.units,
                summary,
                chunk_forced_retain=chunk.forced_retain,
                chunk_reason=chunk.reason,
            )
            all_decisions.update({d.question_id: d for d in decisions})
            combined.api_calls += outcome.api_calls
            combined.failures.extend(outcome.failures)
            combined.missing_count += outcome.missing_count
            combined.malformed_count += outcome.malformed_count
            for qid, decision in outcome.decisions.items():
                combined.decisions[qid] = decision

        retention = apply_retention_policy(units, all_decisions, self.config)
        validation = structural_validate(messages, retention.messages)
        return retention.messages, combined, retention, validation, len(chunks)

    def _result(
        self,
        *,
        cached: bool,
        elapsed: float,
        result_data: dict[str, Any],
        metadata_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Single, consistent result structure for cache-hit and miss paths."""
        threshold = float(self.config["compress_threshold"])
        has_external = bool(
            (self.config.get("use_external_summary") and self.config.get("external_summary"))
            or self.config.get("external_summary_file")
        )
        if has_external:
            summary_kind = "external semantic summary"
        elif self.config.get("fallback"):
            summary_kind = "fallback truncation summary"
        else:
            summary_kind = "no summary (conservative retain-all)"
        metadata: dict[str, Any] = {
            "filter_method": f"clef-flash noul threshold {threshold} with {summary_kind}",
            "cached": cached,
            "algorithm_version": ALGORITHM_VERSION,
            "threshold": threshold,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
        metadata.update(metadata_extra or {})
        result_data = dict(result_data)
        result_data["metadata"] = {**metadata, **result_data.get("metadata", {})}

        # Sanitize message fields before JSON serialization: strip BLOB/bytes
        # columns (e.g. display_identity) that SQLite returns but JSON cannot
        # represent. The original rows are untouched in state.db.
        if "messages" in result_data and isinstance(result_data["messages"], list):
            result_data["messages"] = [
                self._sanitize_message(m) if isinstance(m, dict) else m
                for m in result_data["messages"]
            ]

        if self.config.get("output"):
            Path(self.config["output"]).write_text(
                json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return result_data

    @staticmethod
    def _sanitize_message(msg: dict[str, Any]) -> dict[str, Any]:
        """Strip non-JSON-serializable fields (BLOBs, bytes) from a message dict.

        Hermes stores binary columns like ``display_identity`` as BLOBs; the
        Clef API payload and the output JSON must be pure JSON. We keep all
        text fields and drop anything that cannot be serialized, recording
        the dropped field names in ``_sanitized_fields`` so the original row
        remains locatable in state.db post-compaction.
        """
        clean: dict[str, Any] = {}
        dropped: list[str] = []
        for key, value in msg.items():
            if isinstance(value, (bytes, bytearray)):
                dropped.append(key)
                continue
            try:
                json.dumps(value)
                clean[key] = value
            except (TypeError, ValueError):
                dropped.append(key)
        if dropped:
            clean["_sanitized_fields"] = dropped
        return clean

    # -- main lifecycle -----------------------------------------------------
    def run(self) -> dict[str, Any]:
        started = time.time()
        logger.info("Starting Hermes context compression (v%s)", ALGORITHM_VERSION)

        try:
            messages = load_messages(self.config)
        except ContextError as exc:
            logger.error("Context load failed: %s", exc)
            return self._error_result(str(exc))
        if not messages:
            logger.warning("No session context found")
            return self._error_result("No session context found")

        from_db = bool(self.config.get("fallback") or self.config.get("session_id"))
        try:
            messages = assert_single_session(
                messages,
                expected=self.config.get("session_id"),
                allow_missing=not from_db,
            )
        except ContextError as exc:
            logger.error("Session isolation check failed: %s", exc)
            return self._error_result(str(exc))

        session_id = str(messages[0].get("session_id") or "batch-no-session")
        self.session_id = session_id

        # -- summary (must be resolved BEFORE cache identity) ---------------
        summary: str | None = None
        try:
            summary = load_external_summary(self.config)
        except SummaryError as exc:
            self.summary_error = str(exc)
            logger.error("External summary error: %s", exc)
        if summary is None and self.config.get("fallback"):
            summary = generate_fallback_summary(
                messages, int(self.config["summary_target_chars"])
            )
        fallback_summary = bool(summary and is_fallback_summary(summary))

        signature, identity = self._identity(session_id, messages, summary)

        cache_status = CACHE_MISS
        cache_reason = ""
        cache_data: dict[str, Any] | None = None
        with CacheManager(self.cache_dir, int(self.config["retention_days"])) as cache:
            lookup = cache.get(signature, identity)
            cache_status, cache_reason = lookup.status, lookup.reason
            if lookup.usable:
                cache_data = lookup.data

            if cache_data is not None:
                elapsed = time.time() - started
                logger.info("Using validated cached result (%s)", signature[:12])
                retained = cache_data.get("messages", [])
                prior_report = cache_data.get("report", {})
                report = dict(prior_report)
                report.update({
                    "cache_status": cache_status,
                    "cache_reason": cache_reason,
                    "elapsed_seconds": elapsed,
                })
                if not self.config.get("dry_run"):
                    write_report(cache.report_dir(signature), report)
                return self._result(
                    cached=True,
                    elapsed=elapsed,
                    result_data={
                        "session_id": session_id,
                        "original_count": cache_data.get("original_count", 0),
                        "retained_count": len(retained),
                        "compression_ratio": cache_data.get("compression_ratio", 0.0),
                        "messages": retained,
                    },
                    metadata_extra={
                        "cache_status": cache_status,
                        "summary_fingerprint": identity["summary_hash"],
                        "input_fingerprint": identity["messages_hash"],
                        "dry_run": bool(self.config.get("dry_run")),
                    },
                )

            # -- process -----------------------------------------------------
            analyze_only = bool(self.config.get("analyze_only"))
            if analyze_only:
                retained_candidates = messages[:]
                filter_outcome = FilterOutcome()
                retention = RetentionOutcome(
                    units=[ConversationUnit(f"msg_{i}", "solo", [m])
                           for i, m in enumerate(messages)]
                )
                validation = structural_validate(messages, messages)
                chunk_count = 0
            else:
                (
                    retained_candidates,
                    filter_outcome,
                    retention,
                    validation,
                    chunk_count,
                ) = self._process(messages, summary)

            conservative_fallback = False
            if not validation.ok:
                logger.warning(
                    "Structural validation failed (%d issues); using conservative "
                    "retain-all strategy",
                    len(validation.issues),
                )
                retained_candidates = messages[:]
                conservative_fallback = True

            # -- database reconstruction ------------------------------------
            db_status = "skipped"
            backup_path: str | None = None
            reconstruct_errors: list[str] = []
            can_write_db = (
                self.config.get("recreate")
                and not self.config.get("dry_run")
                and not analyze_only
                and from_db
                and session_id != "batch-no-session"
            )
            if can_write_db:
                dm = DatabaseManager(self.config["state_db"])
                rr = dm.reconstruct(
                    session_id,
                    retained_candidates,
                    identity["messages_hash"],
                    source_messages=messages,
                    create_backup=True,
                )
                db_status = "reconstructed" if rr.ok else "failed"
                backup_path = rr.backup_path
                reconstruct_errors = rr.errors
                if not rr.ok:
                    logger.error("Database reconstruction failed: %s", rr.errors)
            elif self.config.get("dry_run") or analyze_only:
                db_status = "dry_run"
            elif not self.config.get("recreate"):
                db_status = "disabled"

            # -- report ------------------------------------------------------
            elapsed = time.time() - started
            breakdown = {
                "hard_preserved": len(retention.hard_preserved),
                "clef_retained": len(retention.clef_retained),
                "fail_safe_retained": len(retention.fail_safe_retained),
                "dropped": len(retention.dropped),
                "conservative_retain_all": int(conservative_fallback),
            }
            report = build_report(
                session_id=session_id,
                messages_hash=identity["messages_hash"],
                summary_hash=identity["summary_hash"],
                algorithm_version=ALGORITHM_VERSION,
                model=str(self.config.get("model", CLEF_MODEL_ID)),
                threshold=float(self.config["compress_threshold"]),
                original_count=len(messages),
                retained_count=len(retained_candidates),
                chunk_count=chunk_count,
                filter_summary=filter_outcome.summary(),
                retention_breakdown=breakdown,
                cache_status=cache_status,
                cache_reason=cache_reason,
                database_status=db_status,
                backup_path=backup_path,
                validation={
                    "ok": validation.ok,
                    "issues": [i.__dict__ for i in validation.issues],
                },
                elapsed=elapsed,
                retained_messages=retained_candidates,
                fallback_summary=fallback_summary,
                dry_run=bool(self.config.get("dry_run") or analyze_only),
            )
            if self.summary_error:
                report["summary_error"] = self.summary_error
            if reconstruct_errors:
                report["database_errors"] = reconstruct_errors

            # -- cache write (only after successful validation) ---------------
            db_write_failed = can_write_db and db_status == "failed"
            if (
                not self.config.get("dry_run")
                and not analyze_only
                and not db_write_failed
            ):
                cache_data = {
                    "session_id": session_id,
                    "original_count": len(messages),
                    "retained_count": len(retained_candidates),
                    "compression_ratio": (
                        len(retained_candidates) / len(messages) if messages else 0.0
                    ),
                    # F1: state.db rows can carry bytes/BLOB columns (e.g.
                    # display_identity). CacheManager.put json.dumps()es this
                    # payload, so sanitize exactly as _result() does -- otherwise
                    # a real BLOB raises TypeError and aborts the whole run.
                    "messages": [
                        self._sanitize_message(m) if isinstance(m, dict) else m
                        for m in retained_candidates
                    ],
                    "metadata": {"algorithm_version": ALGORITHM_VERSION},
                    "report": report,
                }
                cache.put(signature, cache_data, identity)
            write_report(cache.report_dir(signature), report)

            return self._result(
                cached=False,
                elapsed=elapsed,
                result_data={
                    "session_id": session_id,
                    "original_count": len(messages),
                    "retained_count": len(retained_candidates),
                    "compression_ratio": (
                        len(retained_candidates) / len(messages) if messages else 0.0
                    ),
                    "messages": retained_candidates,
                },
                metadata_extra={
                    "success": not db_write_failed and validation.ok,
                    "cache_status": cache_status,
                    "database_status": db_status,
                    "backup_path": backup_path,
                    "database_errors": reconstruct_errors,
                    "validation_ok": validation.ok,
                    "input_fingerprint": identity["messages_hash"],
                    "summary_fingerprint": identity["summary_hash"],
                    "dry_run": bool(self.config.get("dry_run") or analyze_only),
                    "fallback_summary": fallback_summary,
                },
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic, conservative, reversible Hermes context compression."
    )
    parser.add_argument(
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write retained messages back to state.db (default: false). "
             "This is a destructive operation; a verified backup is created first.",
    )
    parser.add_argument(
        "--use-external-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the Agent-provided semantic summary (default: true).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline including Clef calls, but do NOT write the "
             "database or cache (default: false).",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only load, group and report. Do NOT call Clef, write the database "
             "or cache (default: false).",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Read-only structural health check of the loaded message set. "
             "Prints a JSON report; never calls Clef and never writes state.db "
             "or the cache (default: false).",
    )
    parser.add_argument("--throttle", type=int, default=2,
                        help="Seconds between chunks (default: 2).")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Retention threshold in [0,1] (default: 0.3).")
    parser.add_argument("--chunk-size", type=int, default=20,
                        help="Hard upper bound on messages per request (default: 20).")
    parser.add_argument("--output", type=str, default=None,
                        help="Write the result JSON to this path instead of stdout.")
    parser.add_argument("--external-summary", type=str, default=None,
                        help="Raw semantic summary text supplied by the Agent.")
    parser.add_argument("--external-summary-file", type=str, default=None,
                        help="Path to a file containing the semantic summary.")
    parser.add_argument("--context-dir", type=str, default=None,
                        help="Directory containing ctx.json (batch mode).")
    parser.add_argument("--fallback", action="store_true",
                        help="Load messages from state.db and use a conservative "
                             "truncation summary when no semantic summary is given.")
    parser.add_argument("--session-id", type=str, default=None,
                        help="Explicit session id (Agent/fallback mode).")
    parser.add_argument("--max-messages", type=int, default=None,
                        help="Max messages to load from session (default: 48).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config: dict[str, Any] = {
        "throttle_interval": args.throttle,
        "compress_threshold": args.threshold,
        "chunk_size": args.chunk_size,
        "output": args.output,
        "recreate": args.recreate,
        "use_external_summary": args.use_external_summary,
        "external_summary": args.external_summary,
        "external_summary_file": args.external_summary_file,
        "context_dir": args.context_dir or str(
            Path.home() / ".hermes" / "cache" / "hcompaction"
        ),
        "fallback": args.fallback,
        "session_id": args.session_id,
        "dry_run": args.dry_run,
        "analyze_only": args.analyze_only,
    }
    if args.max_messages is not None:
        config["max_messages"] = args.max_messages

    # --diagnose is a strictly read-only mode: load the message set and run
    # structural checks. It bypasses the compactor entirely, so no Clef call,
    # no cache/report directory creation and no state.db write can happen.
    if args.diagnose:
        try:
            messages = load_messages(build_config(config))
        except (ContextError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps(diagnose_messages(messages), ensure_ascii=False, indent=2))
        return 0

    try:
        compactor = HermesCompaction(config)
        result = compactor.run()
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2

    # Single output owner: only main prints. ``run()`` never prints JSON.
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if "error" not in result.get("metadata", {}) else 1


if __name__ == "__main__":
    raise SystemExit(main())
