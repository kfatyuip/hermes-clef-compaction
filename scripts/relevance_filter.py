"""Strict, fail-safe Clef relevance filtering.

Golden rule: a missing or malformed answer is NEVER a negative relevance
decision. Every submitted unit must receive a verdict, and any answer we cannot
trust results in retention of the affected unit plus a recorded reason.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

logger = logging.getLogger("hermes_compaction.filter")

RETAIN = "retain"
DROP = "drop"


class ClefAPIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class AnswerParse:
    score: float | None = None
    status: str = "missing"  # ok | missing | malformed | ambiguous
    reason: str = ""


@dataclass
class Decision:
    question_id: str
    retain: bool
    score: float | None
    source: str  # clef | missing | malformed | ambiguous | api_error | forced
    reason: str = ""


@dataclass
class FilterOutcome:
    decisions: dict[str, Decision] = field(default_factory=dict)
    api_calls: int = 0
    failures: list[str] = field(default_factory=list)
    missing_count: int = 0
    malformed_count: int = 0

    @property
    def retained_ids(self) -> set[str]:
        return {qid for qid, d in self.decisions.items() if d.retain}

    def summary(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "failures": list(self.failures),
            "missing_answers": self.missing_count,
            "malformed_answers": self.malformed_count,
            "retained_by_clef": sum(
                1 for d in self.decisions.values() if d.retain and d.source == "clef"
            ),
            "retained_by_fail_safe": sum(
                1 for d in self.decisions.values()
                if d.retain and d.source != "clef"
            ),
        }


def _locate_answers(result: Any) -> Any:
    """Locate the answers container across documented/observed variants.

    Supported shapes (explicit compatibility, no guessing by accident):
      A. {"success": true, "result": {"answers": {qid: {"noul": x}}}}
      B. {"answers": {qid: {"noul": x}}}
      C. {"result": {"answers": {"keep": {"noul": x}}}}   (README variant)
      D. {"result": {"answers": {qid: x}}}                (bare number)
    """
    if not isinstance(result, dict):
        return None
    if result.get("success") is False:
        return None
    root: Any = result.get("result", result)
    if not isinstance(root, dict):
        return None
    answers = root.get("answers", None)
    if answers is None:
        answers = root.get("keep", None)
    return answers


def parse_answers(result: Any, expected_ids: list[str]) -> dict[str, AnswerParse]:
    """Parse and validate one answer per expected question id."""
    expected = set(expected_ids)
    parsed = {qid: AnswerParse(status="missing", reason="answer_absent") for qid in expected}
    answers = _locate_answers(result)
    if not isinstance(answers, dict):
        for qid in expected:
            parsed[qid] = AnswerParse(status="missing", reason="no_answers_container")
        return parsed

    # README-style aggregate "keep" object: it cannot express per-message
    # relevance, so applying it to every message would be an unsafe guess.
    if set(answers.keys()) == {"keep"} and isinstance(answers["keep"], dict):
        has_noul = "noul" in answers["keep"]
        for qid in expected:
            parsed[qid] = AnswerParse(
                status="ambiguous",
                reason="aggregate_keep_object_without_question_ids" if has_noul
                else "unexpected_keep_object",
            )
        return parsed

    for qid in expected:
        if qid not in answers:
            continue
        entry = answers[qid]
        if isinstance(entry, dict):
            if "noul" not in entry:
                parsed[qid] = AnswerParse(status="malformed", reason="missing_noul_field")
                continue
            raw = entry["noul"]
        else:
            raw = entry

        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            parsed[qid] = AnswerParse(status="malformed", reason="noul_not_numeric")
            continue
        score = float(raw)
        if not math.isfinite(score):
            parsed[qid] = AnswerParse(status="malformed", reason="noul_not_finite")
            continue
        if not 0.0 <= score <= 1.0:
            parsed[qid] = AnswerParse(status="malformed", reason="noul_out_of_range")
            continue
        parsed[qid] = AnswerParse(score=score, status="ok", reason="valid")
    return parsed


class RelevanceFilter:
    """Calls Clef with bounded retries and applies fail-safe decisions."""

    def __init__(
        self,
        config: dict[str, Any],
        call_api: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.threshold = float(config["compress_threshold"])
        self._call_api = call_api or self._default_call_api
        self._sleep = sleep

    @staticmethod
    def _sanitize_message(msg: dict[str, Any]) -> dict[str, Any]:
        """Strip non-JSON-serializable fields (BLOBs, bytes) from a message dict.

        Hermes stores binary columns like ``display_identity`` as BLOBs; the
        Clef API payload must be pure JSON. We keep all text fields and drop
        anything that cannot be serialized, recording the drop in a sidecar
        field so the original row is still locatable post-compaction.
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

    def _default_call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        account = self.config.get("cloudflare_account_id")
        token = self.config.get("cloudflare_auth_token")
        if not account or not token:
            raise ClefAPIError("Cloudflare credentials are not configured")
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account}"
            f"/ai/run/{self.config.get('model', '@cf/cloudflare/clef-flash')}"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        # Sanitize messages in the payload before sending: strip BLOB/bytes
        # columns (e.g. display_identity) that SQLite returns but the API
        # cannot serialize.
        if "state" in payload and isinstance(payload["state"], dict):
            msgs = payload["state"].get("messages")
            if isinstance(msgs, list):
                payload["state"]["messages"] = [
                    self._sanitize_message(m) if isinstance(m, dict) else m
                    for m in msgs
                ]
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            raise ClefAPIError(
                f"Clef API returned HTTP {response.status_code}: {response.text[:200]}",
                status=response.status_code,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ClefAPIError(f"Invalid JSON from Clef API: {exc}") from exc

    def _call_with_retry(
        self, payload: dict[str, Any], expected_ids: list[str]
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Bounded retries with exponential backoff and jitter.

        Retries 429 / 5xx / timeouts / invalid JSON / structurally empty
        answers. Partial answers are NOT retried indefinitely: per-question
        fail-safe retention handles them deterministically.
        """
        max_retries = int(self.config.get("api_max_retries", 3))
        base_delay = float(self.config.get("retry_base_delay", 0.5))
        failures: list[str] = []

        for attempt in range(max_retries + 1):
            try:
                result = self._call_api(payload)
                # Retry only when there is no usable answers container at all
                # (API/structure failure). Individual missing/malformed answers
                # are handled per-question by fail-safe retention below, and
                # must NOT be collapsed into a generic "api_error".
                answers = _locate_answers(result)
                if not isinstance(answers, dict) or not answers:
                    raise ClefAPIError("no answers container in response")
                return result, failures
            except (ClefAPIError, requests.RequestException) as exc:
                status = getattr(exc, "status", None)
                retryable = status is None or status == 429 or (status is not None and 500 <= status < 600)
                failures.append(f"attempt {attempt + 1}: {exc}")
                if retryable and attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                    self._sleep(delay)
                    continue
                break
        return None, failures

    @staticmethod
    def _assign_question_ids(units: list[Any]) -> list[str]:
        """Give every submitted unit a unique question id.

        A duplicate or missing ``unit_id`` would collapse entries in the
        ``questions``/``decisions`` dicts, leaving some units with no verdict
        (which violates the fail-safe contract and can silently drop a unit).
        Fall back to a positional id for any id that is absent or already used.
        """
        seen: set[str] = set()
        qids: list[str] = []
        for i, unit in enumerate(units):
            raw = getattr(unit, "unit_id", None)
            qid = str(raw) if raw is not None else ""
            if not qid or qid in seen:
                qid = f"unit_{i}"
                suffix = 1
                while qid in seen:
                    qid = f"unit_{i}_{suffix}"
                    suffix += 1
            seen.add(qid)
            qids.append(qid)
        return qids

    @staticmethod
    def _ensure_complete(decisions: dict[str, Decision], expected: int) -> None:
        if len(decisions) != expected:
            raise AssertionError(
                f"filter produced {len(decisions)} verdicts for {expected} units; "
                "every submitted unit must receive a verdict"
            )

    def filter_chunk(
        self,
        units: list[Any],
        summary: str,
        chunk_forced_retain: bool = False,
        chunk_reason: str = "",
    ) -> tuple[list[Decision], FilterOutcome]:
        """Evaluate one chunk and return one decision per unit."""
        outcome = FilterOutcome()
        qids = self._assign_question_ids(units)

        if chunk_forced_retain:
            for qid in qids:
                outcome.decisions[qid] = Decision(
                    question_id=qid, retain=True, score=None,
                    source="forced", reason=chunk_reason or "oversized_chunk",
                )
            self._ensure_complete(outcome.decisions, len(units))
            return list(outcome.decisions.values()), outcome

        questions: dict[str, Any] = {}
        for i, (unit, qid) in enumerate(zip(units, qids)):
            text = "\n".join(
                f"{m.get('role', '')}: {str(m.get('content', ''))[:240]}"
                for m in getattr(unit, "messages", [])
            )
            questions[qid] = {
                "type": "noul",
                "instructions": (
                    "Based on the provided context summary, should this "
                    "conversation unit be retained? " + text
                ),
            }
        payload = {
            "model": self.config.get("model", "@cf/cloudflare/clef-flash"),
            "state": {
                "context_summary": summary,
                "messages": [
                    dict(m)
                    for unit in units
                    for m in getattr(unit, "messages", [])
                ],
            },
            "questions": questions,
        }

        result, failures = self._call_with_retry(payload, qids)
        outcome.api_calls += 1
        outcome.failures.extend(failures)

        if result is None:
            for qid in qids:
                outcome.decisions[qid] = Decision(
                    question_id=qid, retain=True, score=None,
                    source="api_error", reason="; ".join(failures[-1:]) or "api_failure",
                )
            self._ensure_complete(outcome.decisions, len(units))
            return list(outcome.decisions.values()), outcome

        parsed = parse_answers(result, qids)
        decisions: list[Decision] = []
        for qid in qids:
            answer = parsed[qid]
            if answer.status == "ok":
                retain = bool(answer.score is not None and answer.score >= self.threshold)
                decisions.append(
                    Decision(qid, retain, answer.score, "clef", "valid_answer")
                )
            else:
                # Fail-safe: anything we cannot verify -> retain.
                retain = True
                if answer.status == "missing":
                    outcome.missing_count += 1
                else:
                    outcome.malformed_count += 1
                decisions.append(
                    Decision(qid, retain, None, answer.status, answer.reason)
                )
        for decision in decisions:
            outcome.decisions[decision.question_id] = decision
        self._ensure_complete(outcome.decisions, len(units))
        return decisions, outcome
