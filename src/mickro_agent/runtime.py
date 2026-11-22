"""The common execution boundary used by every loop variation."""

from __future__ import annotations

import asyncio
import json
import random as _random
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

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
from .journal import (
    INTERRUPTED_CONTENT,
    JournalEntry,
    ReplayExhausted,
    RunJournal,
    VersionUnsupported,
    command_key,
)
from .providers.base import Provider
from .providers.streaming import collect_response
from .session_log import SessionLogError
from .tools import Tools
from .types import Limits, Message, ModelRequest, ModelResponse, Run, StreamChunk, ToolCall


class Loop(Protocol):
    async def run(self, runtime: Runtime) -> None: ...


class LimitReached(Exception):
    pass


def _model_key(request: ModelRequest) -> str:
    # run_id is excluded so that a fork can match its parent's recorded prefix.
    return command_key("model", request.model_dump(mode="json", exclude={"run_id"}))


def _tool_key(call: ToolCall) -> str:
    return command_key("tool", {"id": call.id, "name": call.name, "arguments": call.arguments})


class Runtime:
    def __init__(
        self,
        run: Run,
        provider: Provider,
        tools: Tools,
        limits: Limits,
        *,
        journal: RunJournal | None = None,
        log_replay: bool = False,
        live_event: tuple[str, dict] = ("run_recovered", {}),
    ):
        self.run = run
        self.provider = provider
        self.tools = tools
        self.limits = limits
        self.journal = journal
        self._seq = 0
        self._applied = 0  # Last command whose result is reflected in the run's state.
        # Replaying: upcoming commands were recorded earlier and are not repeated.
        self.replaying = journal is not None and journal.last_recorded_seq > 0
        # A recovered run's replayed events are already in its log; a fork's are not.
        run._suppress_sink = self.replaying and not log_replay
        self._live_event = live_event
        self._pending: tuple[ToolCall, ...] = ()
        self._busy: str | None = None
        self._model_inflight = False
        self._marker_owner = object()
        self._context_version = 0
        self._compaction_token: object | None = None

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _go_live(self) -> None:
        """Leave replay before the first command that was not recorded."""
        if self.journal is not None and self.journal.readonly:
            raise ReplayExhausted(self._seq, "end of the recorded journal")
        if not self.replaying:
            return
        self.replaying = False
        if self.run._suppress_sink:
            self.run._suppress_sink = False
            self.run._sequence = max(self.run._sequence, self.run._log_sequence)
        kind, data = self._live_event
        self.run.emit(kind, **data, epoch=self.journal.epoch, recorded_commands=self.journal.last_recorded_seq)

    @property
    def seq(self) -> int:
        """Number of journaled commands whose results are reflected in the run's state."""
        return self._applied

    def _side_effect(self, name: str, produce) -> Any:
        self._active()
        seq = self._next_seq()
        if self.journal is None:
            self._applied = seq
            return produce()
        key = command_key("side_effect", name)
        _, completed = self.journal.check(seq, "side_effect", key)
        if completed is not None:
            self._applied = seq
            return completed.payload["value"]
        self._go_live()
        value = produce()
        # Queued without waiting: ordered writes make it durable before any later effect.
        self.journal.enqueue(JournalEntry(seq=seq, phase="completed", kind="side_effect", key=key,
                                          payload={"value": value}))
        self._applied = seq
        return value

    def now(self) -> datetime:
        """Current UTC time, recorded: a replay returns the time of the original run."""
        return datetime.fromisoformat(self._side_effect("now", lambda: datetime.now(UTC).isoformat()))

    def uuid(self) -> str:
        """A random hex UUID, recorded so that replays return the same one."""
        return self._side_effect("uuid", lambda: uuid4().hex)

    def random(self) -> float:
        """A float in [0, 1), recorded so that replays return the same one."""
        return self._side_effect("random", _random.random)

    def version(self, change_id: str, min_supported: int = 0, max_version: int = 1) -> int:
        """Choose a code path for a change so that old journals still replay.

        A new execution records and returns ``max_version``. A replay returns the
        recorded version. A replay of history from before the change existed
        returns 0, the code as it was before any version() call. Raise
        ``min_supported`` once journals with older versions no longer need to
        replay; they then fail with VersionUnsupported and stay recoverable.
        Markers use no command number, so adding a version() call never shifts
        later commands.
        """
        self._active()
        if not isinstance(change_id, str) or not change_id:
            raise ValueError("change_id must be a nonempty string")
        if not 0 <= min_supported <= max_version:
            raise ValueError("Versions need 0 <= min_supported <= max_version")
        if self.journal is None:
            return max_version
        recorded = self.journal.markers.get(change_id)
        if recorded is None:
            # Recorded commands still ahead: this point was executed by code without the change.
            recorded = 0 if self._seq < self.journal.last_recorded_seq else max_version
            if not min_supported <= recorded <= max_version:
                raise VersionUnsupported(change_id, recorded, min_supported, max_version)
            self.journal.marker(change_id, recorded, at_seq=self._seq)
        elif not min_supported <= recorded <= max_version:
            raise VersionUnsupported(change_id, recorded, min_supported, max_version)
        return recorded

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
        seq = self._next_seq()
        request_provider = provider if provider is not None else self.provider
        model = getattr(request_provider, "model", None)
        self.run.emit(
            "model_request", request=request.model_dump(mode="json"), conversation=conversation,
            provider=type(request_provider).__name__, model=model if isinstance(model, str) else None,
        )
        journal_key = scheduled = None
        if self.journal is not None:
            journal_key = _model_key(request)
            scheduled, completed = self.journal.check(seq, "model", journal_key)
            if completed is not None:
                response = ModelResponse.model_validate(completed.payload["response"])
                for name, value in response.usage.items():
                    self.run.usage[name] = self.run.usage.get(name, 0) + value
                self.run.emit(
                    "model_response", response=response.model_dump(mode="json"),
                    turn=request.turn, purpose=purpose, conversation=conversation,
                )
                self._applied = seq
                return response
            self._go_live()
            if scheduled is None:
                await self.journal.append(JournalEntry(seq=seq, phase="scheduled", kind="model", key=journal_key,
                                                       payload={"purpose": purpose, "turn": request.turn}))

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
        if self.journal is not None:
            await self.journal.append(JournalEntry(seq=seq, phase="completed", kind="model", key=journal_key,
                                                   payload={"response": response.model_dump(mode="json")}))
        self.run.emit(
            "model_response", response=response.model_dump(mode="json"),
            turn=request.turn, purpose=purpose, conversation=conversation,
        )
        self._applied = seq
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
        # Sequence numbers follow the provider's order, never completion order.
        seqs = [self._next_seq() for _ in calls]
        keys = [_tool_key(call) for call in calls]
        known: dict[int, Message] = {}
        if self.journal is not None:
            new_rows, interrupted = [], []
            for index, (call, seq, key) in enumerate(zip(calls, seqs, keys, strict=True)):
                scheduled, completed = self.journal.check(seq, "tool", key)
                if completed is not None:
                    known[index] = Message.model_validate(completed.payload["message"])
                elif scheduled is None:
                    new_rows.append(JournalEntry(seq=seq, phase="scheduled", kind="tool", key=key,
                                                 payload={"call_id": call.id, "name": call.name,
                                                          "effect": self.tools.effect(call.name)}))
                elif self.tools.effect(call.name) == "unsafe":
                    # Started before a crash with no recorded result: do not repeat the effect.
                    interrupted.append(index)
            for index, message in sorted(known.items()):
                self.run.emit("tool_start", call=calls[index].model_dump(mode="json"))
                self.run.emit("tool_result", message=message.model_dump(mode="json"))
            if len(known) < len(calls):
                self._go_live()
            if new_rows:
                await self.journal.append(*new_rows)
            for index in interrupted:
                message = Message(role="tool", tool_call_id=calls[index].id, is_error=True, content=INTERRUPTED_CONTENT)
                await self.journal.append(JournalEntry(seq=seqs[index], phase="completed", kind="tool",
                                                       key=keys[index],
                                                       payload={"message": message.model_dump(mode="json"),
                                                                "synthetic": True}))
                self.run.emit("tool_result", message=message.model_dump(mode="json"), synthetic=True)
                known[index] = message
        semaphore = asyncio.Semaphore(concurrency)

        async def execute(index: int) -> Message:
            call, seq = calls[index], seqs[index]
            async with semaphore:
                self.run.emit("tool_start", call=call.model_dump(mode="json"))
                context = {
                    **self.run.context,
                    "runtime": self,
                    "idempotency_key": f"{self.run.session_id}:{self.run.id}:{seq}:{call.id}",
                }
                # A timed-out effect has an unknown outcome: stop the run, do not retry it.
                async with asyncio.timeout(self.limits.tool_timeout):
                    result = await self.tools.invoke(call, context)
                message = Message(
                    role="tool", tool_call_id=call.id, content=result.content(), is_error=result.error is not None
                )
                if self.journal is not None:
                    await self.journal.append(JournalEntry(seq=seq, phase="completed", kind="tool", key=keys[index],
                                                           payload={"message": message.model_dump(mode="json")}))
                self.run.emit("tool_result", message=message.model_dump(mode="json"))
                return message

        if concurrency == 1:
            for index in range(len(calls)):
                self.run.messages.append(known[index] if index in known else await execute(index))
        else:
            # TaskGroup cancels siblings on failure; results retain the provider's order.
            async with asyncio.TaskGroup() as group:
                tasks = {
                    index: group.create_task(execute(index)) for index in range(len(calls)) if index not in known
                }
            self.run.messages.extend(
                known[index] if index in known else tasks[index].result() for index in range(len(calls))
            )
        self._pending = ()
        self._applied = seqs[-1]

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
