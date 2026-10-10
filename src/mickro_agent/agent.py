from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .loops import ReAct
from .runtime import LimitReached, Loop, Provider, Runtime
from .session_log import SessionLog, SessionLogError, validate_session_id
from .sessions import MemorySessionStore, Session, SessionStore
from .tools import Tools
from .types import Limits, Message, Run


def _validate_history(history: Sequence[Message]) -> None:
    pending: set[str] = set()
    for message in history:
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


def _errors(exc: Exception) -> list[Exception]:
    if isinstance(exc, ExceptionGroup):
        return [leaf for child in exc.exceptions for leaf in _errors(child)]
    return [exc]


def _checkpoint_messages(run: Run) -> tuple[Message, ...]:
    """Close interrupted calls without repeating effects whose outcome is unknown."""
    messages = list(run.messages[1:])
    pending = {}
    for message in messages:
        if message.role == "assistant":
            pending.update((call.id, call) for call in message.tool_calls)
        elif message.role == "tool":
            pending.pop(message.tool_call_id, None)
    # A parallel batch may have returned some results before a sibling failed.
    observed = {}
    for event in run.events:
        if event.kind == "model_response":
            observed.clear()
        elif event.kind == "tool_result":
            observed[event.data["message"]["tool_call_id"]] = event.data["message"]
    for call_id in pending:
        messages.append(
            Message.model_validate(observed[call_id])
            if call_id in observed
            else Message(
                role="tool",
                tool_call_id=call_id,
                is_error=True,
                content="Execution interrupted. Outcome unknown; inspect external state before retrying.",
            )
        )
    _validate_history(messages)
    return tuple(messages)


@dataclass(frozen=True)
class Agent:
    provider: Provider
    tools: Tools = field(default_factory=Tools)
    loop: Loop = field(default_factory=ReAct)
    system_prompt: str = "You are a helpful assistant."
    limits: Limits = field(default_factory=Limits)
    name: str = "agent"
    log_dir: str | Path | None = None
    sessions: SessionStore = field(default_factory=MemorySessionStore, repr=False, compare=False)

    async def run(
        self,
        prompt: str,
        *,
        context: Mapping[str, Any] | None = None,
        history: Sequence[Message] = (),
        log_dir: str | Path | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        parent_run_id: str | None = None,
        job_id: str | None = None,
    ) -> Run:
        """Start a conversation, or continue a known session with a fresh run ID.

        A supplied, unknown ID creates a named session. Context and loop state are
        invocation-local; only conversation (including provider metadata) resumes.
        """
        _validate_history(history)
        session_id = uuid4().hex if session_id is None else session_id
        validate_session_id(session_id)
        if session_id == parent_session_id:
            raise ValueError("A child cannot run in its active parent session")
        async with self.sessions.lock(session_id):
            previous = self.sessions.load(session_id)
            if previous is not None:
                if history:
                    raise ValueError("Cannot combine history with an existing session")
                if previous.agent != self.name or previous.system_prompt != self.system_prompt:
                    raise ValueError("Session belongs to a different agent configuration")
                if previous.status == "running":
                    raise ValueError("Session was interrupted before checkpointing; inspect it before recovery")
                history = previous.messages
                _validate_history(history)
            return await self._run_session(
                prompt,
                context=context,
                history=history,
                log_dir=log_dir,
                session_id=session_id,
                previous=previous,
                parent_session_id=parent_session_id,
                parent_run_id=parent_run_id,
                job_id=job_id,
            )

    async def _run_session(
        self, prompt, *, context, history, log_dir, session_id, previous, parent_session_id, parent_run_id, job_id
    ) -> Run:
        run = Run(
            session_id=session_id,
            parent_session_id=parent_session_id,
            parent_run_id=parent_run_id,
            job_id=job_id,
            context=dict(context or {}),
            messages=[
                Message(role="system", content=self.system_prompt),
                *(message.model_copy(deep=True) for message in history),
                Message(role="user", content=prompt),
            ],
        )
        runtime = Runtime(run, self.provider, self.tools, self.limits)
        directory = self.log_dir if log_dir is None else log_dir
        if previous is not None and previous.log_path is not None:
            previous_path = Path(previous.log_path)
            if directory is not None and Path(directory).expanduser().resolve() != previous_path.parent:
                raise ValueError("A resumed session must keep its original log directory")
            directory = previous_path.parent

        def checkpoint(status: str) -> None:
            self.sessions.save(
                Session(
                    session_id=session_id,
                    agent=self.name,
                    system_prompt=self.system_prompt,
                    messages=_checkpoint_messages(run),
                    status=status,
                    log_path=str(run.log_path) if run.log_path else None,
                )
            )

        with ExitStack() as stack:
            if directory is not None:
                session_log = stack.enter_context(
                    SessionLog(directory, session_id, resume=previous is not None and previous.log_path is not None)
                )
                run.log_path = session_log.path
                run._event_sink = session_log.write
            try:
                checkpoint("running")
                model = getattr(self.provider, "model", None)
                run.emit(
                    "run_start",
                    agent=self.name,
                    loop=type(self.loop).__name__,
                    cwd=str(Path.cwd()),
                    provider=type(self.provider).__name__,
                    model=model if isinstance(model, str) else None,
                    limits=self.limits.model_dump(mode="json"),
                    messages=[message.model_dump(mode="json") for message in run.messages],
                    resumed=previous is not None,
                    parent_session_id=parent_session_id,
                    parent_run_id=parent_run_id,
                    job_id=job_id,
                )
                async with asyncio.timeout(self.limits.timeout):
                    await self.loop.run(runtime)
                if run.status == "running":
                    raise RuntimeError("Loop ended without runtime.finish(output)")
            except LimitReached as exc:
                run.status, run.error = "limit", str(exc)
            except TimeoutError:
                run.status, run.error = (
                    "timeout",
                    "Run or tool deadline exceeded; an in-flight effect may be incomplete",
                )
            except asyncio.CancelledError:
                run.status, run.error = "cancelled", "Run cancelled by caller"
                raise
            except SessionLogError as exc:
                run.status, run.error = "failed", str(exc)
                raise  # Logging was requested: never silently run without it.
            except Exception as exc:  # noqa: BLE001 - Record arbitrary provider/node failures.
                errors = _errors(exc)
                run.status = "timeout" if all(isinstance(error, TimeoutError) for error in errors) else "failed"
                run.error = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            finally:
                if run.status != "completed":
                    run.output = None
                try:
                    run.emit("run_end", status=run.status, output=run.output, error=run.error, usage=run.usage)
                finally:
                    run._event_sink = None
                    checkpoint(run.status)
        return run
