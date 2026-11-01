"""The common execution boundary used by every loop variation."""

from __future__ import annotations

import asyncio
import json
from typing import Protocol

from .context_ops import (
    CompactionProposal,
    CompactionRequest,
    CompactionResult,
    CompactionStrategy,
    MessageMarker,
    MessageRange,
    OlderHistory,
    SummaryHistory,
    TokenCounter,
    is_summary,
)
from .history import validate_conversation
from .providers.base import Provider
from .providers.streaming import collect_response
from .session_log import SessionLogError
from .tools import Tools
from .types import Limits, Message, ModelRequest, ModelResponse, Run, StreamChunk, ToolCall


class Loop(Protocol):
    async def run(self, runtime: Runtime) -> None: ...


class LimitReached(Exception):
    pass


class Runtime:
    def __init__(self, run: Run, provider: Provider, tools: Tools, limits: Limits):
        self.run = run
        self.provider = provider
        self.tools = tools
        self.limits = limits
        self._pending: tuple[ToolCall, ...] = ()
        self._busy: str | None = None
        self._model_inflight = False
        self._marker_owner = object()
        self._context_version = 0
        self._compaction_token: object | None = None

    def _active(self) -> None:
        if self.run.status != "running":
            raise RuntimeError("Run has already finished")

    def _idle(self) -> None:
        self._active()
        if self._busy is not None:
            raise RuntimeError(f"Conversation is busy with {self._busy}")

    def _context_ready(self) -> None:
        self._idle()
        if self._pending:
            raise ValueError("Resolve pending tool calls before changing conversation context")

    def step(self, name: str, *, metadata: dict | None = None) -> None:
        self._active()
        if self.run.steps >= self.limits.max_steps:
            raise LimitReached("Graph step limit reached")
        self.run.steps += 1
        self.run.emit("step", node=name, step=self.run.steps, metadata=metadata or {})

    def say(self, content: str) -> None:
        self._idle()
        if self._pending:
            raise ValueError("Resolve pending tool calls before adding an instruction")
        self.run.messages.append(Message(role="user", content=content))
        self.run.emit("instruction", content=content)

    async def complete(self, *, allow_tools: bool = True, purpose: str = "act") -> ModelResponse:
        self._context_ready()
        self._busy = "a model turn"
        try:
            response = await self._request(
                tuple(self.run.messages), allow_tools=allow_tools, purpose=purpose, conversation=True
            )
            self._pending = tuple(call.model_copy(deep=True) for call in response.tool_calls)
            self.run.messages.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                    metadata={**response.metadata, **({"reasoning": response.reasoning} if response.reasoning else {})},
                ).model_copy(deep=True)
            )
            return response
        finally:
            self._busy = None

    async def _request(
        self,
        messages: tuple[Message, ...],
        *,
        allow_tools: bool,
        purpose: str,
        conversation: bool,
        provider: Provider | None = None,
    ) -> ModelResponse:
        """Account for a model call without implicitly editing the conversation."""
        self._active()
        if self._model_inflight:
            raise RuntimeError("Concurrent model requests on one runtime are not supported")
        if self.run.model_calls >= self.limits.max_model_calls:
            raise LimitReached("Model call limit reached")
        self.run.model_calls += 1
        request = ModelRequest(
            run_id=self.run.id,
            messages=tuple(message.model_copy(deep=True) for message in messages),
            tools=self.tools.specs if allow_tools else (),
            turn=self.run.model_calls,
            purpose=purpose,
        )
        request_provider = provider if provider is not None else self.provider
        model = getattr(request_provider, "model", None)
        self.run.emit(
            "model_request", request=request.model_dump(mode="json"), conversation=conversation,
            provider=type(request_provider).__name__, model=model if isinstance(model, str) else None,
        )

        def on_chunk(chunk: StreamChunk) -> None:
            # Usage already received remains charged if a later chunk fails or
            # the call is cancelled. collect_response validates the counters first.
            if chunk["type"] == "usage":
                for key, value in chunk.items():
                    if key != "type":
                        self.run.usage[key] = self.run.usage.get(key, 0) + value
            self.run.emit("model_chunk", turn=request.turn, chunk=chunk)

        self._model_inflight = True
        try:
            response = await collect_response(request_provider, request, on_chunk=on_chunk)
        finally:
            self._model_inflight = False
        if response.tool_calls and not allow_tools:
            raise ValueError("Provider returned tool calls when tools were disabled")
        ids = [call.id for call in response.tool_calls]
        if len(ids) != len(set(ids)):
            raise ValueError("Provider returned duplicate tool call IDs")
        self.run.emit(
            "model_response", response=response.model_dump(mode="json"),
            turn=request.turn, purpose=purpose, conversation=conversation,
        )
        return response

    async def execute_tools(self, *, concurrency: int = 1) -> None:
        self._idle()
        self._busy = "tool execution"
        try:
            await self._execute_tools(concurrency=concurrency)
        finally:
            self._busy = None

    async def _execute_tools(self, *, concurrency: int) -> None:
        self._active()
        if concurrency < 1:
            raise ValueError("Tool concurrency must be positive")
        calls = self._pending
        if not calls:
            raise ValueError("No pending tool calls to execute")
        if self.run.tool_calls + len(calls) > self.limits.max_tool_calls:
            raise LimitReached("Tool call limit reached; batch was not executed")
        self.run.tool_calls += len(calls)
        semaphore = asyncio.Semaphore(concurrency)

        async def execute(call: ToolCall) -> Message:
            async with semaphore:
                self.run.emit("tool_start", call=call.model_dump(mode="json"))
                # A timed-out effect has an unknown outcome: stop the run, do not retry it.
                async with asyncio.timeout(self.limits.tool_timeout):
                    result = await self.tools.invoke(call, {**self.run.context, "runtime": self})
                message = Message(
                    role="tool", tool_call_id=call.id, content=result.content(), is_error=result.error is not None
                )
                self.run.emit("tool_result", message=message.model_dump(mode="json"))
                return message

        if concurrency == 1:
            for call in calls:
                self.run.messages.append(await execute(call))
        else:
            # TaskGroup cancels siblings on failure; results retain the provider's order.
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(execute(call)) for call in calls]
            self.run.messages.extend(task.result() for task in tasks)
        self._pending = ()

    def finish(self, output: str) -> None:
        self._idle()
        if self._pending:
            raise ValueError("Resolve pending tool calls before finishing")
        self.run.output = output
        self.run.status = "completed"

    def mark_messages(self) -> MessageMarker:
        """Mark a complete conversation boundary for a later compaction.

        Markers belong to this invocation. Any applied compaction invalidates
        existing markers; appending messages does not invalidate them.
        """
        self._context_ready()
        validate_conversation(self.run.messages)
        return MessageMarker(
            self._marker_owner, self._context_version,
            tuple(message.model_copy(deep=True) for message in self.run.messages),
        )

    def _marker_index(self, marker: MessageMarker) -> int:
        if not isinstance(marker, MessageMarker) or marker._owner is not self._marker_owner:
            raise ValueError("Message marker belongs to another runtime")
        if marker._version != self._context_version:
            raise ValueError("Stale message marker: conversation was compacted")
        index = len(marker._prefix)
        if tuple(self.run.messages[:index]) != marker._prefix:
            raise ValueError("Stale message marker: earlier messages changed")
        return index

    def _select_messages(self, selection: MessageRange | OlderHistory | SummaryHistory) -> tuple[int, int]:
        if isinstance(selection, MessageRange):
            start = self._marker_index(selection.start)
            stop = len(self.run.messages) if selection.stop is None else self._marker_index(selection.stop)
        elif isinstance(selection, OlderHistory):
            users = [i for i, message in enumerate(self.run.messages) if message.role == "user"]
            if len(users) <= selection.keep_recent_turns:
                return len(self.run.messages), len(self.run.messages)
            start, stop = users[0] + 1, users[-selection.keep_recent_turns]
        elif isinstance(selection, SummaryHistory):
            messages = self.run.messages
            first = int(bool(messages) and messages[0].role == "system")
            start = min(first + 1, len(messages))
            stop = max(start, len(messages) - selection.keep_recent_messages)
            # A tool-result tail must retain the assistant and every sibling
            # result, including parallel calls returned in a different order.
            while stop > start and messages[stop].role == "tool":
                stop -= 1
        else:
            raise TypeError("selection must be a MessageRange, OlderHistory or SummaryHistory")
        if start > stop:
            raise ValueError("Compaction range starts after its stop marker")
        if any(message.role == "system" for message in self.run.messages[start:stop]):
            raise ValueError("Compaction cannot replace system messages")
        # Both boundaries must close every tool call, including parallel calls.
        validate_conversation(self.run.messages[:start])
        validate_conversation(self.run.messages[:stop])
        return start, stop

    async def compact_since(
        self,
        marker: MessageMarker,
        *,
        strategy: CompactionStrategy,
        token_counter: TokenCounter | None = None,
    ) -> CompactionResult:
        return await self.compact(
            selection=MessageRange(marker), strategy=strategy, token_counter=token_counter
        )

    async def compact(
        self,
        *,
        selection: MessageRange | OlderHistory | SummaryHistory,
        strategy: CompactionStrategy,
        token_counter: TokenCounter | None = None,
    ) -> CompactionResult:
        """Prepare, validate and atomically replace a conversation region.

        With an explicit token_counter, apply only a strict reduction in the
        full conversation's token count. Failures propagate without a fallback.
        Events and model usage are retained even when a proposal is not applied.
        """
        self._context_ready()
        validate_conversation(self.run.messages)
        start, stop = self._select_messages(selection)
        name = type(strategy).__name__
        snapshot = tuple(message.model_copy(deep=True) for message in self.run.messages)
        source = snapshot[start:stop]
        skip = "empty_selection" if not source else None
        if isinstance(selection, SummaryHistory):
            if any(is_summary(message) for message in snapshot[stop:]):
                skip = "recent_summary"
            elif len(source) < 2 and not any(is_summary(message) for message in source):
                skip = "not_enough_messages"
        if skip:
            self.run.emit("context_compaction_skipped", strategy=name, start=start, stop=stop, reason=skip)
            return CompactionResult(False, skip, name, start, stop)

        self._busy = "compaction"
        token = self._compaction_token = object()

        async def generate_summary(
            prompt: str, *, provider: Provider | None = None, start_index: int = 0,
        ) -> ModelResponse:
            if self._compaction_token is not token:
                raise RuntimeError("Summary generator is only valid during its compaction")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Summarization prompt must be nonempty")
            if (
                not isinstance(start_index, int) or isinstance(start_index, bool)
                or not 0 <= start_index < len(source)
            ):
                raise ValueError("Summary start_index must select a nonempty source suffix")
            # Encode tool calls/results as transcript data rather than native
            # protocol messages, so a different summarizer provider can read them.
            transcript = json.dumps(
                [message.model_dump(mode="json") for message in source[start_index:]], ensure_ascii=False,
            )
            return await self._request(
                (
                    Message(role="system", content=prompt),
                    Message(role="user", content="Conversation region:\n" + transcript),
                ),
                allow_tools=False, purpose="compact", conversation=False, provider=provider,
            )

        async def count(messages: tuple[Message, ...]) -> int | None:
            if token_counter is None:
                return None
            value = await token_counter(tuple(message.model_copy(deep=True) for message in messages))
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Token counter must return a nonnegative integer")
            return value

        try:
            before_tokens = await count(snapshot)
            proposal = await strategy.compact(
                CompactionRequest(tuple(message.model_copy(deep=True) for message in source), generate_summary)
            )
            if not isinstance(proposal, CompactionProposal):
                raise TypeError("Compaction strategy must return a CompactionProposal")
            if not all(isinstance(message, Message) for message in proposal.messages):
                raise TypeError("Compaction replacements must be Message objects")
            replacement = tuple(Message.model_validate(message.model_dump()) for message in proposal.messages)
            if any(message.role == "system" for message in replacement):
                raise ValueError("Compaction cannot insert system messages")
            candidate = snapshot[:start] + replacement + snapshot[stop:]
            validate_conversation(candidate)
            after_tokens = await count(candidate)
            self._active()
            if self._pending or self._model_inflight or tuple(self.run.messages) != snapshot:
                raise RuntimeError("Conversation changed during compaction; proposal was not applied")
            reason = "applied"
            if replacement == source:
                reason = "unchanged"
            elif before_tokens is not None and after_tokens >= before_tokens:
                reason = "not_smaller"
            result = CompactionResult(
                reason == "applied", reason, name, start, stop,
                tuple(message.model_copy(deep=True) for message in replacement), before_tokens, after_tokens,
            )
            if not result.applied:
                self.run.emit(
                    "context_compaction_skipped", strategy=name, start=start, stop=stop, reason=reason,
                    before_tokens=before_tokens, after_tokens=after_tokens,
                )
                return result
            # A requested log failure must not mutate the live conversation.
            self.run.emit(
                "context_compaction", strategy=name, start=start, stop=stop,
                removed_messages=[message.model_dump(mode="json") for message in source],
                replacement_messages=[message.model_dump(mode="json") for message in replacement],
                before_tokens=before_tokens, after_tokens=after_tokens,
            )
            self.run.messages[:] = candidate
            self._context_version += 1
            return result
        except SessionLogError:
            raise
        except (Exception, asyncio.CancelledError) as exc:
            self.run.emit(
                "context_compaction_failed", strategy=name, start=start, stop=stop,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            self._busy = None
            self._compaction_token = None
