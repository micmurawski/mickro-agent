"""Explicit context selection and compaction strategies; Runtime owns the commit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .providers.base import Provider
from .types import Message, ModelResponse

SUMMARY_PROMPT = """\
Your task is to create a detailed summary of the conversation so far, paying close attention to the user's
     explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions
     that would be essential for continuing with the conversation and supporting any continuing tasks.

Your summary should be structured as follows:
Context: The context to continue the conversation with. If applicable based on the current task, this should
     include:
  1. Previous Conversation: High level details about what was discussed throughout the entire conversation
     with the user. This should be written to allow someone to be able to follow the general overarching
     conversation flow.
  2. Current Work: Describe in detail what was being worked on prior to this request to summarize the
     conversation. Pay special attention to the more recent messages in the conversation.
  3. Key Technical Concepts: List all important technical concepts, technologies, coding conventions, and
     frameworks discussed, which might be relevant for continuing with this work.
  4. Relevant Files and Code: If applicable, enumerate specific files and code sections examined, modified, or
     created for the task continuation. Pay special attention to the most recent messages and changes.
  5. Problem Solving: Document problems solved thus far and any ongoing troubleshooting efforts.
  6. Pending Tasks and Next Steps: Outline all pending tasks that you have explicitly been asked to work on,
     as well as list the next steps you will take for all outstanding work, if applicable. Include code
     snippets where they add clarity. For any next steps, include direct quotes from the most recent
     conversation showing exactly what task you were working on and where you left off. This should be
     verbatim to ensure there's no information loss in context between tasks.

Example summary structure:
1. Previous Conversation:
  [Detailed description]
2. Current Work:
  [Detailed description]
3. Key Technical Concepts:
  - [Concept 1]
  - [Concept 2]
  - [...]
4. Relevant Files and Code:
  - [File Name 1]
    - [Summary of why this file is important]
    - [Summary of the changes made to this file, if any]
    - [Important Code Snippet]
  - [File Name 2]
    - [Important Code Snippet]
  - [...]
5. Problem Solving:
  [Detailed description]
6. Pending Tasks and Next Steps:
  - [Task 1 details & next steps]
  - [Task 2 details & next steps]
  - [...]

Output only the summary of the conversation so far, without any additional commentary or explanation.
"""


@dataclass(frozen=True)
class MessageMarker:
    """Opaque, invocation-local boundary returned by Runtime.mark_messages()."""

    _owner: object = field(repr=False)
    _version: int = field(repr=False)
    _prefix: tuple[Message, ...] = field(repr=False)


@dataclass(frozen=True)
class MessageRange:
    """Messages between markers; omit stop to include everything after start."""

    start: MessageMarker
    stop: MessageMarker | None = None


@dataclass(frozen=True)
class OlderHistory:
    """Preserve the first user message and the most recent user-initiated turns.

    A turn starts at a user message and includes all following assistant/tool
    exchanges up to the next user message. Runtime.say() also starts such a turn.
    """

    keep_recent_turns: int = 3

    def __post_init__(self) -> None:
        if (
            not isinstance(self.keep_recent_turns, int)
            or isinstance(self.keep_recent_turns, bool)
            or self.keep_recent_turns < 1
        ):
            raise ValueError("keep_recent_turns must be a positive integer")


@dataclass(frozen=True)
class SummaryHistory:
    """Legacy summary window: preserve the first message and at least N recent messages.

    The system prompt is also preserved. Expand the recent tail to keep tool
    calls with all their results. Skip if that tail contains a summary or there
    is too little older history to summarize.
    """

    keep_recent_messages: int = 3

    def __post_init__(self) -> None:
        if (
            not isinstance(self.keep_recent_messages, int)
            or isinstance(self.keep_recent_messages, bool)
            or self.keep_recent_messages < 1
        ):
            raise ValueError("keep_recent_messages must be a positive integer")


def is_summary(message: Message) -> bool:
    compaction = message.metadata.get("compaction")
    return (
        isinstance(compaction, dict) and compaction.get("kind") == "summary"
    ) or message.metadata.get("isSummary") is True


class SummaryGenerator(Protocol):
    async def __call__(
        self, prompt: str, *, provider: Provider | None = None, start_index: int = 0,
    ) -> ModelResponse: ...


class TokenCounter(Protocol):
    async def __call__(self, messages: Sequence[Message]) -> int: ...


@dataclass(frozen=True)
class CompactionRequest:
    """Detached source messages and a budgeted, tool-free summary request function.

    Await generate_summary inside compact(); it is valid only for this operation.
    The callback summarizes the original selected snapshot, not mutations to messages.
    start_index optionally limits its input to a suffix of that snapshot.
    """

    messages: tuple[Message, ...]
    generate_summary: SummaryGenerator = field(repr=False)


@dataclass(frozen=True)
class CompactionProposal:
    messages: tuple[Message, ...] = ()


class CompactionStrategy(Protocol):
    async def compact(self, request: CompactionRequest) -> CompactionProposal: ...


@dataclass(frozen=True)
class Remove:
    """Discard the selected region without a model call."""

    async def compact(self, request: CompactionRequest) -> CompactionProposal:
        return CompactionProposal()


@dataclass(frozen=True)
class Replace:
    """Retain an already generated handoff without another model call."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Replacement text must be nonempty; use Remove() to discard messages")

    async def compact(self, request: CompactionRequest) -> CompactionProposal:
        return CompactionProposal(
            (Message(role="assistant", content=self.text, metadata={"compaction": {"kind": "replacement"}}),)
        )


@dataclass(frozen=True)
class Summarize:
    """Summarize from the latest existing summary, using a configurable prompt.

    Older messages before that summary are replaced too, but are not sent again
    to the summarizer. The provider defaults to the run's provider.
    """

    prompt: str = SUMMARY_PROMPT
    provider: Provider | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Summarization prompt must be nonempty")

    async def compact(self, request: CompactionRequest) -> CompactionProposal:
        start = next(
            (i for i in range(len(request.messages) - 1, -1, -1) if is_summary(request.messages[i])), 0,
        )
        response = await request.generate_summary(self.prompt, provider=self.provider, start_index=start)
        text = response.text.strip()
        if not text:
            raise ValueError("Failed to generate a nonempty summary")
        # Provider-specific signatures/thinking blocks belong to the summary
        # request, not the main conversation, and must not be replayed there.
        return CompactionProposal(
            (Message(role="assistant", content=text, metadata={"compaction": {"kind": "summary"}}),)
        )


@dataclass(frozen=True)
class CompactionResult:
    applied: bool
    reason: str
    strategy: str
    start: int
    stop: int
    replacement: tuple[Message, ...] = ()
    before_tokens: int | None = None
    after_tokens: int | None = None
