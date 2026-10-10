"""Provider-independent messages and serializable experiment records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NotRequired, Required, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .session_log import SessionLogError, event_line


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCall(Record):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Message(Record):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    is_error: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_role(self) -> Message:
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Only assistant messages may contain tool calls")
        if (self.role == "tool") != bool(self.tool_call_id):
            raise ValueError("A tool message requires tool_call_id; other roles must omit it")
        return self


class ToolSpec(Record):
    name: str
    description: str
    parameters: dict[str, Any]


# Canonical message input, independent of any provider SDK.
MessageParam = Message


class ApiHandlerCreateMessageMetadata(TypedDict, total=False):
    run_id: str
    task_id: str
    turn: int
    purpose: str


class TextChunk(TypedDict):
    type: Literal["text"]
    text: str


class ReasoningChunk(TypedDict):
    type: Literal["reasoning"]
    text: str


class ToolUseChunk(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, JsonValue]
    metadata: NotRequired[dict[str, JsonValue]]


class UsageChunk(TypedDict, total=False):
    type: Required[Literal["usage"]]
    input_tokens: float
    output_tokens: float
    cache_read_tokens: float
    cache_write_tokens: float
    reasoning_tokens: float
    total_cost: float


class MetadataChunk(TypedDict):
    type: Literal["metadata"]
    metadata: dict[str, JsonValue]


class GroundingChunk(TypedDict):
    type: Literal["grounding"]
    sources: list[JsonValue]


StreamChunk = TextChunk | ReasoningChunk | ToolUseChunk | UsageChunk | MetadataChunk | GroundingChunk


class ModelRequest(Record):
    run_id: str = ""
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    turn: int = Field(ge=1)
    purpose: str = "act"


class ModelResponse(Record):
    text: str = ""
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Limits(Record):
    max_model_calls: int = Field(default=20, gt=0)
    max_tool_calls: int = Field(default=100, ge=0)
    max_steps: int = Field(default=100, gt=0)
    timeout: float = Field(default=300, gt=0)
    tool_timeout: float = Field(default=60, gt=0)


class Event(Record):
    run_id: str
    session_id: str | None = None
    sequence: int
    kind: str
    data: dict[str, JsonValue] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Run:
    """Mutable state owned by exactly one execution; context is never logged."""

    id: str = field(default_factory=lambda: uuid4().hex)
    session_id: str = field(default_factory=lambda: uuid4().hex)
    parent_session_id: str | None = None
    parent_run_id: str | None = None
    job_id: str | None = None
    messages: list[Message] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict, repr=False)
    state: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    usage: dict[str, float] = field(default_factory=dict)
    model_calls: int = 0
    tool_calls: int = 0
    steps: int = 0
    status: Literal["running", "completed", "limit", "timeout", "failed", "cancelled"] = "running"
    output: str | None = None
    error: str | None = None
    log_path: Path | None = field(default=None, init=False)
    _event_sink: Callable[[Event], None] | None = field(default=None, init=False, repr=False)

    def emit(self, kind: str, **data: JsonValue) -> None:
        event = Event(run_id=self.id, session_id=self.session_id, sequence=len(self.events), kind=kind, data=data)
        snapshot = event.model_copy(deep=True)
        self.events.append(snapshot)
        if self._event_sink is not None:
            try:
                self._event_sink(snapshot)
            except SessionLogError:
                self._event_sink = None
                raise

    def write_events(self, path: str | Path) -> None:
        """Write a completed run's events as JSONL (overwrites the given file)."""
        path = Path(path)
        if path.expanduser().resolve() == self.log_path:
            raise ValueError("Cannot overwrite the session log with a single-run export")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(event_line(event) for event in self.events), encoding="utf-8")
