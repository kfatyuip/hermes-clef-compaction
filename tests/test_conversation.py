"""Conversation-aware grouping and metadata preservation."""

from __future__ import annotations

from conversation_grouper import (
    flatten_units,
    group_messages,
    is_tool_call,
    is_tool_result,
    units_to_text,
)


def test_user_assistant_pair_is_one_unit():
    msgs = [
        {"role": "user", "content": "propose a design"},
        {"role": "assistant", "content": "here is the design"},
    ]
    units = group_messages(msgs)
    assert len(units) == 1
    assert units[0].kind == "exchange"
    assert len(units[0].messages) == 2


def test_multiple_assistant_responses_stay_with_question():
    msgs = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "assistant", "content": "elaboration"},
    ]
    units = group_messages(msgs)
    assert len(units) == 1
    assert len(units[0].messages) == 3


def test_tool_call_and_result_are_one_unit():
    msgs = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "content": "result", "tool_call_id": "t1"},
        {"role": "assistant", "content": "interpretation"},
    ]
    units = group_messages(msgs)
    assert len(units) == 2
    assert units[1].kind == "tool"
    assert len(units[1].messages) == 3


def test_multiple_tool_calls_grouped():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "content": "r1", "tool_call_id": "a"},
        {"role": "tool", "content": "r2", "tool_call_id": "b"},
    ]
    units = group_messages(msgs)
    assert len(units) == 1
    assert units[0].kind == "tool"


def test_follow_up_questions_are_separate_units():
    msgs = [
        {"role": "user", "content": "what is X?"},
        {"role": "assistant", "content": "X is ..."},
        {"role": "user", "content": "and Y?"},
        {"role": "assistant", "content": "Y is ..."},
    ]
    units = group_messages(msgs)
    assert [u.kind for u in units] == ["exchange", "continuation"]


def test_interrupted_conversation_never_loses_messages():
    msgs = [
        {"role": "user", "content": "long task"},
        {"role": "assistant", "content": "partial"},
        {"role": "user", "content": "wait, change of plan"},
        {"role": "assistant", "content": "ok"},
        {"role": "assistant", "content": "new plan"},
    ]
    units = group_messages(msgs)
    assert flatten_units(units) == msgs
    assert len(units) == 2


def test_metadata_is_preserved_not_discarded():
    msgs = [
        {
            "role": "tool",
            "content": "r",
            "tool_call_id": "tc-9",
            "message_id": "m-9",
            "tool_calls": [],
        }
    ]
    units = group_messages(msgs)
    assert flatten_units(units) == msgs
    rendered = units_to_text(units[0])
    assert "tool_call_id=tc-9" in rendered
    assert "message_id=m-9" in rendered


def test_tool_result_detected_without_standard_role():
    msg = {"role": "user", "content": "result", "tool_call_id": "tc-1"}
    assert is_tool_result(msg)
    assert not is_tool_call(msg)


def test_empty_input_produces_no_units():
    assert group_messages([]) == []
