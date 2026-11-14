"""Token estimation and adaptive, conversation-safe chunking.

Chunking is driven by the *actual estimated payload size* (messages + summary
+ questions + serialization overhead), not by message count alone. Chunk
boundaries prefer conversation units. A single oversized message is never
silently dropped: it is flagged so the engine can hard-preserve it and record
the reason in the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from config import MAX_INPUT_TOKENS, MAX_QUESTIONS_PER_REQUEST
from conversation_grouper import ConversationUnit, units_to_text

# Conservative estimation constants (documented limitations in README):
#   * CJK characters ~= 1 token each (pessimistic but safe)
#   * other characters ~= 3 per token (more conservative than the old 4)
QUESTION_TOKENS_PER_UNIT = 32
REQUEST_OVERHEAD_TOKENS = 400
SAFETY_FACTOR = 0.9
QUESTION_CONTENT_CHARS = 240


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF00 <= code <= 0xFFEF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def estimate_tokens(text: str | None) -> int:
    """Conservatively estimate tokens for mixed-language / code / JSON text.

    This is a bounded heuristic, not a tokenizer. It intentionally errs on the
    high side so a chunk we consider safe is genuinely within budget.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return max(1, int(cjk + other / 3.0 + 1))


@dataclass
class Chunk:
    """A request-safe bundle of conversation units."""

    units: list[ConversationUnit] = field(default_factory=list)
    estimated_tokens: int = 0
    forced_retain: bool = False
    reason: str = ""

    @property
    def message_count(self) -> int:
        return sum(len(u.messages) for u in self.units)


def payload_tokens(
    units: list[ConversationUnit], summary: str, num_questions: int | None = None
) -> int:
    """Estimate the whole request payload, including overhead and questions."""
    questions = num_questions if num_questions is not None else len(units)
    # Estimate on FULL content: the state payload sent to Clef contains the
    # complete messages, not the truncated question text. Estimating on
    # truncated text would let oversized messages through unsafely.
    body = "\n".join(units_to_text(u) for u in units)
    summary_tokens = estimate_tokens(summary)
    body_tokens = estimate_tokens(body)
    questions_tokens = questions * QUESTION_TOKENS_PER_UNIT
    serialization_overhead = int((body_tokens + summary_tokens) * 0.10)
    return (
        summary_tokens
        + body_tokens
        + questions_tokens
        + serialization_overhead
        + REQUEST_OVERHEAD_TOKENS
    )


def _split_oversized_unit(
    unit: ConversationUnit, budget: int, summary: str
) -> list[list[ConversationUnit]]:
    """Split a unit whose full payload exceeds budget into smaller sub-units."""
    groups: list[list[ConversationUnit]] = []
    current: list[dict[str, Any]] = []
    counter = 0

    def flush() -> None:
        nonlocal current, counter
        if current:
            sub = ConversationUnit(
                unit_id=f"{unit.unit_id}_s{counter}", kind=unit.kind, messages=list(current)
            )
            groups.append([sub])
            counter += 1
            current = []

    for msg in unit.messages:
        candidate = [ConversationUnit(unit_id=unit.unit_id, kind=unit.kind, messages=current + [msg])]
        if current and payload_tokens(candidate, summary) > budget:
            flush()
            candidate = [ConversationUnit(unit_id=unit.unit_id, kind=unit.kind, messages=[msg])]
        current.append(msg)
    flush()
    return groups


def build_chunks(
    units: list[ConversationUnit],
    summary: str,
    config: dict[str, Any],
) -> list[Chunk]:
    """Pack units into request-safe chunks preserving chronological order.

    ``chunk_size`` is a real upper bound on messages per request. The token
    budget is the binding constraint; whichever is tighter wins. Oversized
    single messages are returned as forced-retain chunks (never filtered,
    never silently skipped) with an explicit reason.
    """
    budget = int(MAX_INPUT_TOKENS * SAFETY_FACTOR)
    max_messages = int(config["chunk_size"])
    max_questions = min(MAX_QUESTIONS_PER_REQUEST, max_messages)

    chunks: list[Chunk] = []
    pending: list[ConversationUnit] = []

    def flush_pending() -> None:
        if not pending:
            return
        chunks.append(
            Chunk(units=list(pending), estimated_tokens=payload_tokens(pending, summary))
        )
        pending.clear()

    for unit in units:
        single = [unit]
        unit_tokens = payload_tokens(single, summary)
        unit_msg_count = len(unit.messages)

        if unit_tokens > budget or unit_msg_count > max_messages:
            if unit_msg_count == 1:
                flush_pending()
                chunks.append(
                    Chunk(
                        units=single,
                        estimated_tokens=unit_tokens,
                        forced_retain=True,
                        reason="single_message_exceeds_token_or_message_budget",
                    )
                )
                continue
            flush_pending()
            for subgroup in _split_oversized_unit(unit, budget, summary):
                chunks.append(
                    Chunk(
                        units=subgroup,
                        estimated_tokens=payload_tokens(subgroup, summary),
                    )
                )
            continue

        prospective = pending + single
        prospective_msgs = sum(len(u.messages) for u in prospective)
        if (
            payload_tokens(prospective, summary) > budget
            or len(prospective) > max_questions
            or prospective_msgs > max_messages
        ):
            flush_pending()
        pending.append(unit)

    flush_pending()
    return chunks


def chunks_payload_preview(chunks: list[Chunk]) -> list[dict[str, Any]]:
    """Small helper for tests / reports: summarize chunk composition."""
    return [
        {
            "messages": c.message_count,
            "units": len(c.units),
            "estimated_tokens": c.estimated_tokens,
            "forced_retain": c.forced_retain,
        }
        for c in chunks
    ]
