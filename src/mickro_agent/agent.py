from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .history import validate_history as _validate_history
from .journal import DurableStore, JournalEntry, JournalError, NonDeterminismError, RunJournal, RunRecord
from .loops import ReAct
from .runtime import LimitReached, Loop, Provider, Runtime
from .session_log import SessionLog, SessionLogError, validate_session_id
from .sessions import MemorySessionStore, Session, SessionStore
from .tools import Tools
from .types import Limits, Message, Run

# Failures after which the journal must be left as it is: another owner may hold
# the run, the store is unavailable, or this code cannot replay the run.
_STOP_ERRORS = (JournalError, NonDeterminismError)


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
        if event.kind == "model_response" and event.data.get("conversation", True):
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


def _fingerprint(agent: Agent) -> str:
    """Hash of the configuration a journal was recorded with (checked in a later phase)."""
    flow = None
    to_flow = getattr(agent.loop, "to_flow", None)
    if callable(to_flow):
        try:
            compiled = to_flow()
            flow = {"start": compiled.start, "edges": sorted([*key, value] for key, value in compiled.edges.items())}
        except Exception:  # noqa: BLE001 - A fingerprint must never fail the run.
            flow = "unavailable"
    body = {
        "agent": agent.name,
        "system_prompt": agent.system_prompt,
        "loop": f"{type(agent.loop).__module__}.{type(agent.loop).__qualname__}",
        "flow": flow,
        "tools": [spec.model_dump(mode="json") for spec in agent.tools.specs],
        "provider": type(agent.provider).__name__,
        "model": model if isinstance(model := getattr(agent.provider, "model", None), str) else None,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class _Recovery:
    epoch: int
    entries: tuple[JournalEntry, ...]
    active_seconds: float


@dataclass(frozen=True)
class Agent:
    """An agent configuration. Runs are durable when ``store`` is a DurableStore.

    With a store, sessions are checkpointed in it too unless ``sessions`` is given.
    """

    provider: Provider
    tools: Tools = field(default_factory=Tools)
    loop: Loop = field(default_factory=ReAct)
    system_prompt: str = "You are a helpful assistant."
    limits: Limits = field(default_factory=Limits)
    name: str = "agent"
    log_dir: str | Path | None = None
    sessions: SessionStore | None = field(default=None, repr=False, compare=False)
    store: DurableStore | None = field(default=None, repr=False, compare=False)
    lease_ttl: float = 30.0

    def __post_init__(self) -> None:
        if self.sessions is None:
            object.__setattr__(self, "sessions", self.store if self.store is not None else MemorySessionStore())
        if not self.lease_ttl > 0:
            raise ValueError("lease_ttl must be positive")

    def get_flow_graph(self, *, direction: str = "LR") -> str:
        """Return Mermaid for a Flow or a loop exposing to_flow()."""
        from .viz import build_mermaid

        return build_mermaid(self.loop, direction=direction)

    def get_flow_graph_png(self, filename: str | Path, *, direction: str = "LR") -> Path:
        from .viz import to_png

        return to_png(self.loop, filename, direction=direction)

    def render_graph(self, filename: str | Path, *, direction: str = "LR") -> Path:
        """Save a PNG/SVG locally using Graphviz."""
        from .viz import render_graph

        return render_graph(self.loop, filename, direction=direction)

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
            if self.store is not None:
                interrupted = self.store.runs(session_id=session_id, status="running")
                if interrupted:
                    raise ValueError(
                        f"Session has an interrupted run {interrupted[0].run_id}; call Agent.recover(session_id) first"
                    )
            previous = self.sessions.load(session_id)
            if previous is not None:
                if history:
                    raise ValueError("Cannot combine history with an existing session")
                if previous.agent != self.name or previous.system_prompt != self.system_prompt:
                    raise ValueError("Session belongs to a different agent configuration")
                # With a store, interrupted runs are found in the journal, not the checkpoint.
                if previous.status == "running" and self.store is None:
                    raise ValueError("Session was interrupted before checkpointing; inspect it before recovery")
                history = previous.messages
                _validate_history(history)
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
            directory = self.log_dir if log_dir is None else log_dir
            if previous is not None and previous.log_path is not None:
                previous_path = Path(previous.log_path)
                if directory is not None and Path(directory).expanduser().resolve() != previous_path.parent:
                    raise ValueError("A resumed session must keep its original log directory")
                directory = previous_path.parent
            return await self._execute(
                run, directory=directory, resume_log=previous is not None and previous.log_path is not None
            )

    async def recover(
        self, session_id: str, *, context: Mapping[str, Any] | None = None, force: bool = False
    ) -> Run:
        """Resume the session's interrupted run from its journal.

        Recorded model calls and tool results are replayed without repeating them.
        Supply ``context`` again: live resources are never journaled. ``force``
        takes over a run whose previous owner still holds an unexpired lease;
        use it only when that owner is known to be gone.
        """
        if self.store is None:
            raise ValueError("recover() requires an Agent with a durable store")
        validate_session_id(session_id)
        async with self.sessions.lock(session_id):
            interrupted = self.store.runs(session_id=session_id, status="running")
            if not interrupted:
                raise ValueError(f"Session {session_id} has no interrupted run")
            record = interrupted[0]
            start = record.start
            if record.agent != self.name or start.get("system_prompt") != self.system_prompt:
                raise ValueError("Run belongs to a different agent configuration")
            epoch = await self.store.claim_run(session_id, record.run_id, ttl=self.lease_ttl, force=force)
            run = Run(
                id=record.run_id,
                session_id=session_id,
                parent_session_id=start.get("parent_session_id"),
                parent_run_id=start.get("parent_run_id"),
                job_id=start.get("job_id"),
                context=dict(context or {}),
                messages=[Message.model_validate(message) for message in start["messages"]],
            )
            log_path = start.get("log_path")
            return await self._execute(
                run,
                directory=Path(log_path).parent if log_path else None,
                resume_log=True,
                recovery=_Recovery(epoch, tuple(self.store.journal(session_id, record.run_id)), record.active_seconds),
            )

    def _session(self, run: Run, status: str) -> Session:
        return Session(
            session_id=run.session_id,
            agent=self.name,
            system_prompt=self.system_prompt,
            messages=_checkpoint_messages(run),
            status=status,
            log_path=str(run.log_path) if run.log_path else None,
        )

    async def _checkpoint(self, run: Run, status: str) -> None:
        session = self._session(run, status)
        asave = getattr(self.sessions, "asave", None)
        if asave is not None:
            await asave(session)
        else:
            self.sessions.save(session)

    async def _heartbeat(self, journal: RunJournal, active: Callable[[], float]) -> None:
        while True:
            await asyncio.sleep(self.lease_ttl / 3)
            try:
                await self.store.heartbeat(
                    journal.session_id, journal.run_id, journal.epoch, ttl=self.lease_ttl, active_seconds=active()
                )
            except asyncio.CancelledError:
                raise
            except BaseException:  # noqa: BLE001 - The run's own next append reports the failure.
                return

    async def _execute(
        self, run: Run, *, directory: str | Path | None, resume_log: bool, recovery: _Recovery | None = None
    ) -> Run:
        journal: RunJournal | None = None
        heartbeat: asyncio.Task | None = None
        runtime: Runtime | None = None
        stopped = False  # The journal must not be finished or checkpointed by this owner.
        active_before = recovery.active_seconds if recovery is not None else 0.0
        started = time.monotonic()

        def active() -> float:
            return active_before + time.monotonic() - started

        with ExitStack() as stack:
            if directory is not None:
                session_log = stack.enter_context(
                    SessionLog(directory, run.session_id, resume=resume_log, repair=recovery is not None)
                )
                run.log_path = session_log.path
                run._event_sink = session_log.write
                run._log_sequence = session_log.next_sequence.get(run.id, 0)
            try:
                if self.store is not None:
                    if recovery is None:
                        await self.store.create_run(
                            RunRecord(
                                session_id=run.session_id,
                                run_id=run.id,
                                agent=self.name,
                                owner=self.store.owner_id,
                                epoch=1,
                                lease_expires_at=time.time() + self.lease_ttl,
                                start={
                                    "system_prompt": self.system_prompt,
                                    "messages": [message.model_dump(mode="json") for message in run.messages],
                                    "limits": self.limits.model_dump(mode="json"),
                                    "loop": type(self.loop).__name__,
                                    "fingerprint": _fingerprint(self),
                                    "log_path": str(run.log_path) if run.log_path else None,
                                    "parent_session_id": run.parent_session_id,
                                    "parent_run_id": run.parent_run_id,
                                    "job_id": run.job_id,
                                },
                            )
                        )
                        journal = RunJournal(self.store, run.session_id, run.id, 1)
                    else:
                        journal = RunJournal(self.store, run.session_id, run.id, recovery.epoch, recovery.entries)
                    heartbeat = asyncio.create_task(self._heartbeat(journal, active), name=f"heartbeat-{run.id}")
                else:
                    # Without a journal, a "running" checkpoint is how a crash is detected later.
                    await self._checkpoint(run, "running")
                runtime = Runtime(run, self.provider, self.tools, self.limits, journal=journal)
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
                    resumed=resume_log,
                    parent_session_id=run.parent_session_id,
                    parent_run_id=run.parent_run_id,
                    job_id=run.job_id,
                )
                # The timeout budget counts active execution only, not downtime between crash and recovery.
                async with asyncio.timeout(max(self.limits.timeout - active_before, 0)):
                    await self.loop.run(runtime)
                if run.status == "running":
                    raise RuntimeError("Loop ended without runtime.finish(output)")
            except _STOP_ERRORS as exc:
                stopped = True
                run.status, run.error = "failed", f"{type(exc).__name__}: {exc}"
                raise
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
            except Exception as exc:  # Record arbitrary provider/node failures.
                errors = _errors(exc)
                stop = next((error for error in errors if isinstance(error, _STOP_ERRORS)), None)
                if stop is not None:
                    stopped = True
                    run.status, run.error = "failed", f"{type(stop).__name__}: {stop}"
                    raise stop from exc
                run.status = "timeout" if all(isinstance(error, TimeoutError) for error in errors) else "failed"
                run.error = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            finally:
                if heartbeat is not None:
                    heartbeat.cancel()
                if run.status == "running":
                    # An unhandled BaseException (process exit): leave the run recoverable.
                    stopped = True
                if run.status != "completed":
                    run.output = None
                if stopped:
                    run._event_sink = None
                else:
                    if runtime is not None:
                        runtime._go_live()
                    try:
                        run.emit("run_end", status=run.status, output=run.output, error=run.error, usage=run.usage)
                    finally:
                        run._event_sink = None
                        if journal is None:
                            await self._checkpoint(run, run.status)
                        elif self.sessions is self.store:
                            await self.store.finish_run(
                                run.session_id,
                                run.id,
                                journal.epoch,
                                status=run.status,
                                result={"output": run.output, "error": run.error, "usage": run.usage},
                                active_seconds=active(),
                                session=self._session(run, run.status),
                            )
                        else:
                            await self.store.finish_run(
                                run.session_id,
                                run.id,
                                journal.epoch,
                                status=run.status,
                                result={"output": run.output, "error": run.error, "usage": run.usage},
                                active_seconds=active(),
                            )
                            await self._checkpoint(run, run.status)
        return run


class Worker:
    """Recovers every interrupted run in a store, for example on process start.

    ``context`` gives each recovered run its live resources: a mapping shared by
    all runs, or a callable receiving the run's RunRecord.
    """

    def __init__(
        self,
        store: DurableStore,
        agents: Iterable[Agent] | Mapping[str, Agent],
        *,
        context: Mapping[str, Any] | Callable[[RunRecord], Mapping[str, Any]] | None = None,
        concurrency: int = 50,
    ):
        agents = list(agents.values()) if isinstance(agents, Mapping) else list(agents)
        self.agents = {agent.name: agent for agent in agents}
        if len(self.agents) != len(agents):
            raise ValueError("Agent names must be unique")
        if any(agent.store is not store for agent in agents):
            raise ValueError("Every agent must use this worker's store")
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.store = store
        self.context = context
        self.concurrency = concurrency

    async def recover_all(self, *, force: bool = False) -> list[Run | BaseException]:
        """Recover all running runs; results are in store order, failures as exceptions.

        On a single host, ``force=True`` at startup is safe: no other process can
        own these runs, so there is no need to wait for their leases to expire.
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def recover(record: RunRecord) -> Run:
            agent = self.agents.get(record.agent)
            if agent is None:
                raise ValueError(f"No agent named {record.agent!r} for run {record.run_id}")
            context = self.context(record) if callable(self.context) else self.context
            async with semaphore:
                return await agent.recover(record.session_id, context=context, force=force)

        records = self.store.runs(status="running")
        return await asyncio.gather(*(recover(record) for record in records), return_exceptions=True)
