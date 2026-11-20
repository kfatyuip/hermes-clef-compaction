"""Regression for P0 V1 (validator.py): oversized-unit split child ids.

In group mode, ``chunk_manager._split_oversized_unit`` evaluates a single
oversized unit ``unit_0`` as sub-units ``unit_0_s0``, ``unit_0_s1``, ... The
relevance filter returns one verdict per sub-unit id, but retention is applied
against the original, unsplit unit list. Before the fix, the child-id verdicts
never matched the parent unit id, so every message was silently dropped and
``reconstruct`` marked the rows ``active=0, compacted=1``.
"""

from __future__ import annotations

from chunk_manager import _split_oversized_unit, payload_tokens
from config import build_config
from conversation_grouper import ConversationUnit
from relevance_filter import Decision
from validator import apply_retention_policy, structural_validate


def _messages(n: int = 3) -> list[dict]:
    """Messages large enough that the split fixture genuinely splits."""
    return [
        {
            "session_id": "sess-1",
            "role": "user",
            "content": ("alpha " * 400) + f" marker-{i}",
            "timestamp": 1000.0 + i,
            "_rowid": i + 1,
            "id": i + 1,
        }
        for i in range(n)
    ]


def _split_parent(parent: ConversationUnit, summary: str) -> list[ConversationUnit]:
    """Force ``_split_oversized_unit`` to emit one child per message."""
    one = ConversationUnit(parent.unit_id, parent.kind, [parent.messages[0]])
    budget = payload_tokens([one], summary)
    groups = _split_oversized_unit(parent, budget, summary)
    return [g[0] for g in groups]


def _verdicts(
    units: list[ConversationUnit], *, retain: bool = True, source: str = "clef"
) -> dict[str, Decision]:
    return {
        u.unit_id: Decision(u.unit_id, retain, 0.9, source, "valid_answer")
        for u in units
    }


def _cfg() -> dict:
    # hard_preserve_recent=0 so no unrelated rule can mask the parent-id bug.
    return build_config({"hard_preserve_recent": 0, "throttle_interval": 0})


def test_v1_oversized_child_verdicts_retain_all_messages():
    msgs = _messages()
    parent = ConversationUnit("unit_0", "exchange", msgs)
    children = _split_parent(parent, "short summary")

    child_ids = [c.unit_id for c in children]
    assert len(children) >= 2, "fixture must exercise the real split path"
    assert all(cid.startswith("unit_0_s") for cid in child_ids), child_ids

    outcome = apply_retention_policy([parent], _verdicts(children), _cfg())

    # All four required properties.
    assert [u.unit_id for u in outcome.units], "retained units must be non-empty"
    assert "unit_0" not in outcome.dropped, "parent unit must not be dropped"
    assert outcome.messages == msgs, "every message of the oversized unit is kept"
    report = structural_validate(msgs, outcome.messages)
    assert report.ok, [issue.code for issue in report.issues]


def test_v1_failsafe_child_verdicts_also_retain_parent():
    """The normalization applies to fail-safe verdicts, not only clef ones."""
    msgs = _messages(2)
    parent = ConversationUnit("unit_0", "exchange", msgs)
    children = _split_parent(parent, "short summary")
    decisions = _verdicts(children, source="missing")

    outcome = apply_retention_policy([parent], decisions, _cfg())

    assert [u.unit_id for u in outcome.units] == ["unit_0"]
    assert outcome.fail_safe_retained == {"unit_0"}
    assert outcome.messages == msgs


def test_v1_plain_parent_id_still_matches():
    """Guard: units that were never split keep matching on their own id."""
    msgs = _messages(2)
    unit = ConversationUnit("unit_0", "exchange", msgs)
    decisions = {"unit_0": Decision("unit_0", True, 0.9, "clef")}

    outcome = apply_retention_policy([unit], decisions, _cfg())

    assert [u.unit_id for u in outcome.units] == ["unit_0"]
    assert outcome.messages == msgs


def test_v1_child_suffix_without_a_parent_is_not_collapsed():
    """A genuine id ending in ``_sN`` is kept, never misattributed to a parent."""
    msgs = _messages(2)
    unit = ConversationUnit("unit_0_s0", "exchange", msgs)
    decisions = {"unit_0_s0": Decision("unit_0_s0", True, 0.9, "clef")}

    outcome = apply_retention_policy([unit], decisions, _cfg())

    assert [u.unit_id for u in outcome.units] == ["unit_0_s0"]
    assert "unit_0_s0" not in outcome.dropped
    assert outcome.messages == msgs
