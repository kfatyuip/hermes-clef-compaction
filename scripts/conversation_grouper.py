"""Conversation-aware message grouping.

Conversation messages are not independent documents. A retained answer without
its question, or a tool result without its tool call, loses its meaning. This
module groups logically related messages into coherent units *before* relevance
evaluation so that filtering can preserve the minimum coherent context.

It inspects real message metadata (``tool_calls``, ``tool_call_id``,
``message_id``) rather than assuming ``role`` + ``content`` are the only fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationUnit:
    """A logically coherent group of messages evaluated as one unit."""

    unit_id: str
    kind: str  # "exchange" | "tool" | "solo" | "continuation" | "tool_result"
    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(str(m.get("role", "")) for m in self.messages)

    def indices(self, offset: int = 0) -> list[int]:
        return list(range(offset, offset + len(self.messages)))


def _role(msg: dict[str, Any]) -> str:
    return str(msg.get("role", "")).lower()


def is_tool_call(msg: dict[str, Any]) -> bool:
    """Detect a tool invocation without assuming a single message shape."""
    if msg.get("tool_calls"):
        return True
    if msg.get("function_call"):
        return True
    return _role(msg) in ("assistant",) and bool(msg.get("tool_use"))


def is_tool_result(msg: dict[str, Any]) -> bool:
    """Detect a tool result by role or by an explicit tool-call reference."""
    if _role(msg) in ("tool", "function"):
        return True
    return msg.get("tool_call_id") is not None and not is_tool_call(msg)


def _is_pure_assistant(msg: dict[str, Any]) -> bool:
    return _role(msg) == "assistant" and not is_tool_call(msg)


def group_messages(messages: list[dict[str, Any]]) -> list[ConversationUnit]:
    """Group messages into conversation units, preserving input order.

    Rules (applied greedily, left to right):

    * A tool invocation starts a ``tool`` unit that absorbs its immediately
      following tool results and the assistant's interpretation of them.
    * A user message starts an ``exchange`` unit that absorbs one or more
      subsequent (non-tool-call) assistant responses -- this handles multiple
      assistant messages and follow-up answers.
    * Consecutive user messages each start a new unit but are labelled
      ``continuation`` so follow-up questions remain individually addressable.
    * Leading assistant/tool messages form ``solo``/``tool_result`` units.

    Unusual sequences never raise: every message ends up in exactly one unit.
    """
    units: list[ConversationUnit] = []
    n = len(messages)
    i = 0
    seq = 0

    def new_unit(kind: str, members: list[dict[str, Any]]) -> None:
        nonlocal seq
        units.append(
            ConversationUnit(unit_id=f"unit_{seq}", kind=kind, messages=members)
        )
        seq += 1

    while i < n:
        msg = messages[i]

        if is_tool_call(msg):
            members = [msg]
            i += 1
            while i < n and is_tool_result(messages[i]):
                members.append(messages[i])
                i += 1
            # Assistant interpretation following the tool result stays with it.
            while i < n and _is_pure_assistant(messages[i]):
                members.append(messages[i])
                i += 1
            new_unit("tool", members)
            continue

        if _role(msg) == "user":
            kind = "continuation" if units and units[-1].kind == "exchange" else "exchange"
            members = [msg]
            i += 1
            while i < n and _is_pure_assistant(messages[i]):
                members.append(messages[i])
                i += 1
            # Some Hermes variants attach tool results directly to a user turn.
            while i < n and is_tool_result(messages[i]):
                members.append(messages[i])
                i += 1
            new_unit(kind, members)
            continue

        if is_tool_result(msg):
            members = [msg]
            i += 1
            while i < n and (is_tool_result(messages[i]) or _is_pure_assistant(messages[i])):
                members.append(messages[i])
                i += 1
            new_unit("tool_result", members)
            continue

        # Leading or otherwise unmatched assistant message.
        members = [msg]
        i += 1
        new_unit("solo", members)

    return units


def flatten_units(units: list[ConversationUnit]) -> list[dict[str, Any]]:
    """Rebuild the flat message list in original chronological order."""
    out: list[dict[str, Any]] = []
    for unit in units:
        out.extend(unit.messages)
    return out


def units_to_text(unit: ConversationUnit, content_chars: int | None = None) -> str:
    """Render a unit for a Clef question. Metadata is included by name."""
    parts: list[str] = []
    for m in unit.messages:
        role = m.get("role", "")
        content = str(m.get("content", ""))
        if content_chars is not None and len(content) > content_chars:
            content = content[:content_chars] + "..."
        extra: list[str] = []
        if m.get("tool_calls"):
            extra.append("tool_calls")
        if m.get("tool_call_id"):
            extra.append(f"tool_call_id={m['tool_call_id']}")
        if m.get("message_id"):
            extra.append(f"message_id={m['message_id']}")
        suffix = f" [{', '.join(extra)}]" if extra else ""
        parts.append(f"{role}{suffix}: {content}")
    return "\n".join(parts)
