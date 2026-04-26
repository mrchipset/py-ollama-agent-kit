from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUMMARY_PREFIX = "Conversation summary of earlier context"


@dataclass(slots=True)
class ConversationTurnSummary:
    user_messages: list[str]
    assistant_reply: str | None = None
    tool_descriptions: list[str] = field(default_factory=list)


def compact_messages(messages: list[dict[str, Any]], *, max_messages: int) -> list[dict[str, Any]]:
    if max_messages <= 0:
        return list(messages)

    if len(messages) <= max_messages + 1:
        return list(messages)

    if messages and messages[0].get("role") == "system":
        system_messages = [messages[0]]
        conversation_messages = messages[1:]
    else:
        system_messages = []
        conversation_messages = messages

    trimmed_messages = conversation_messages[-max_messages:]
    return system_messages + trimmed_messages


def compress_messages(
    messages: list[dict[str, Any]],
    *,
    max_messages: int,
    previous_summary: str | None = None,
    summary_max_chars: int = 1200,
) -> tuple[list[dict[str, Any]], str | None]:
    if max_messages <= 0:
        return list(messages), previous_summary

    if messages and messages[0].get("role") == "system":
        system_messages = [messages[0]]
        conversation_messages = messages[1:]
    else:
        system_messages = []
        conversation_messages = messages

    if len(conversation_messages) <= max_messages:
        return list(messages), previous_summary

    overflow_messages = conversation_messages[:-max_messages]
    retained_messages = conversation_messages[-max_messages:]
    new_summary = build_history_summary(
        overflow_messages,
        previous_summary=previous_summary,
        max_chars=summary_max_chars,
    )
    return system_messages + retained_messages, new_summary


def build_history_summary(
    messages: list[dict[str, Any]],
    *,
    previous_summary: str | None = None,
    max_chars: int = 1200,
) -> str | None:
    turns = _group_turn_summaries(messages)
    summary_lines: list[str] = []

    if previous_summary:
        summary_lines.append("Earlier summary:")
        summary_lines.append(previous_summary.strip())
        summary_lines.append("")

    if not turns:
        if summary_lines:
            return _truncate_summary("\n".join(summary_lines).strip(), max_chars)
        return None

    summary_lines.append(f"Compressed {len(messages)} older messages across {len(turns)} turns.")

    for index, turn in enumerate(turns[-4:], start=1):
        summary_lines.append(f"- Turn {index}")
        if turn.user_messages:
            summary_lines.append(f"  User: {_truncate_text(' | '.join(turn.user_messages), 180)}")
        if turn.tool_descriptions:
            summary_lines.append(f"  Tools: {_truncate_text('; '.join(turn.tool_descriptions), 220)}")
        if turn.assistant_reply:
            summary_lines.append(f"  Assistant: {_truncate_text(turn.assistant_reply, 180)}")

    summary = "\n".join(summary_lines).strip()
    if not summary:
        return None

    return _truncate_summary(summary, max_chars)


def _group_turn_summaries(messages: list[dict[str, Any]]) -> list[ConversationTurnSummary]:
    turns: list[ConversationTurnSummary] = []
    current_turn: ConversationTurnSummary | None = None

    for message in messages:
        role = message.get("role")
        if role == "system":
            continue

        content = message.get("content")
        text = content.strip() if isinstance(content, str) else ""

        if role == "user":
            if current_turn is None:
                current_turn = ConversationTurnSummary(user_messages=[])
                turns.append(current_turn)
            current_turn.user_messages.append(text or "[empty user message]")
        elif role == "tool":
            if current_turn is None:
                current_turn = ConversationTurnSummary(user_messages=["[tool result without user prompt]"])
                turns.append(current_turn)
            name = message.get("name") or "unknown_tool"
            current_turn.tool_descriptions.append(f"{name}: {_truncate_text(text or '[empty tool result]', 120)}")
        elif role == "assistant":
            if current_turn is None:
                current_turn = ConversationTurnSummary(user_messages=["[assistant message without user prompt]"])
                turns.append(current_turn)
            current_turn.assistant_reply = text or "[empty assistant reply]"

    return turns


def _truncate_summary(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."