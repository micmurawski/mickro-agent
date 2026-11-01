"""Validation shared by session continuation and context compaction."""

from collections.abc import Sequence

from .types import Message


def validate_history(history: Sequence[Message]) -> None:
    pending: set[str] = set()
    for message in history:
        # Revalidate even model_copy(update=...) values at this persistence boundary.
        message = Message.model_validate(message.model_dump())
        if message.role == "system":
            raise ValueError("Set system_prompt on Agent; history must not include system messages")
        if message.role == "tool":
            if message.tool_call_id not in pending:
                raise ValueError("History contains an unmatched tool result")
            pending.remove(message.tool_call_id)
        else:
            if pending:
                raise ValueError("History contains unresolved tool calls")
            ids = [call.id for call in message.tool_calls]
            if len(ids) != len(set(ids)):
                raise ValueError("History contains duplicate tool call IDs")
            pending.update(ids)
    if pending:
        raise ValueError("History contains unresolved tool calls")


def validate_conversation(messages: Sequence[Message]) -> None:
    """The system message, when present, must be first; all tool groups must close."""
    offset = int(bool(messages) and messages[0].role == "system")
    validate_history(messages[offset:])
