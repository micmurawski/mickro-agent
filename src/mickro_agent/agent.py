from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from .history import validate_history as _validate_history
from .journal import (
    ConfigurationChanged,
    DurableStore,
    JournalEntry,
    JournalError,
    NonDeterminismError,
    ReplayExhausted,
    RunJournal,
    RunRecord,
)
from .loops import ReAct
from .runtime import LimitReached, Loop, Provider, Runtime
from .session_log import SessionLog, SessionLogError, validate_session_id
from .sessions import MemorySessionStore, Session, SessionStore
from .shared import SharedWorld
from .tools import Tools
from .types import Limits, Message, Run, ToolCall

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


def _fingerprint(agent: Agent) -> dict[str, str]:
    """Hash per configuration part a journal was recorded with, so a mismatch can name the part."""
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
    return {
        name: hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]
        for name, value in body.items()
    }


def _changed_parts(recorded: Mapping[str, Any], current: Mapping[str, str]) -> list[str]:
    parts = recorded.get("fingerprint")
    if not isinstance(parts, Mapping):
        return []  # Recorded before fingerprints were checked.
    return sorted(name for name in set(parts) | set(current) if parts.get(name) != current.get(name))


@dataclass(frozen=True)
class _Recovery:
    epoch: int
    entries: tuple[JournalEntry, ...]
    active_seconds: float
    lenient_until: int = 0
    log_replay: bool = False
    live_event: tuple[str, dict] = ("run_recovered", {})


@dataclass(frozen=True)
class Replay:
    """A run rebuilt from its journal without performing any effect.

    ``run`` holds the messages, state, usage and events as of ``commands`` replayed
    commands. ``reason`` says why replay stopped: ``finished`` (the loop ended),
    ``until_seq`` (the requested step), or ``end_of_journal`` (the next command was
    never recorded, e.g. the run was interrupted there).
    """

    run: Run
    commands: int
    reason: Literal["finished", "until_seq", "end_of_journal"]
    pending_tool_calls: tuple[ToolCall, ...]


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
    shared: SharedWorld | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.sessions is None:
            object.__setattr__(self, "sessions", self.store if self.store is not None else MemorySessionStore())
        if not self.lease_ttl > 0:
            raise ValueError("lease_ttl must be positive")
        if self.shared is not None and self.store is not None and self.shared.store is not self.store:
            # One store: a shared write and the run's journal row must commit together.
            raise ValueError("Agent.shared must be a world in the agent's own store")

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
        self,
        session_id: str,
        *,
        context: Mapping[str, Any] | None = None,
        force: bool = False,
        allow_fingerprint_change: bool = False,
    ) -> Run:
        """Resume the session's interrupted run from its journal.

        Recorded model calls and tool results are replayed without repeating them.
        Supply ``context`` again: live resources are never journaled. ``force``
        takes over a run whose previous owner still holds an unexpired lease;
        use it only when that owner is known to be gone.

        A configuration that differs from the recorded one (loop, flow, tools,
        provider, model) raises ConfigurationChanged, naming the parts. With
        ``allow_fingerprint_change`` recovery proceeds; commands that no longer
        match the journal still raise NonDeterminismError.
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
            changed = _changed_parts(start, _fingerprint(self))
            if changed and not allow_fingerprint_change:
                raise ConfigurationChanged(changed)
            epoch = await self.store.claim_run(session_id, record.run_id, ttl=self.lease_ttl, force=force)
            run = self._run_from_record(record, context)
            log_path = start.get("log_path")
            return await self._execute(
                run,
                directory=Path(log_path).parent if log_path else None,
                resume_log=True,
                recovery=_Recovery(
                    epoch,
                    tuple(self.store.journal(session_id, record.run_id)),
                    record.active_seconds,
                    lenient_until=_lenient_until(start),
                ),
            )

    async def replay(
        self,
        session_id: str,
        run_id: str,
        *,
        until_seq: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Replay:
        """Rebuild a run's state from its journal, optionally only up to command ``until_seq``.

        Nothing is written and no provider or tool is called. Use it to inspect a
        finished or interrupted run at any step. Code that diverges from the
        journal raises NonDeterminismError. A parallel tool batch is replayed
        whole: ``until_seq`` inside a batch stops before the batch.
        """
        if self.store is None:
            raise ValueError("replay() requires an Agent with a durable store")
        if until_seq is not None and until_seq < 0:
            raise ValueError("until_seq must be nonnegative")
        record = self.store.get_run(session_id, run_id)
        if record is None:
            raise ValueError(f"Run {run_id} not found in session {session_id}")
        run = self._run_from_record(record, context)
        journal = RunJournal(
            None, session_id, run_id, record.epoch, self.store.journal(session_id, run_id),
            readonly=True, until_seq=until_seq, lenient_until=_lenient_until(record.start),
        )
        runtime = Runtime(run, self.provider, self.tools, self.limits, journal=journal, shared=self.shared)
        reason = "finished"
        try:
            await self.loop.run(runtime)
        except ReplayExhausted as exc:
            reason = "until_seq" if exc.reason.startswith("until_seq") else "end_of_journal"
        except LimitReached as exc:
            run.status, run.error = "limit", str(exc)
        except NonDeterminismError:
            raise
        except Exception as exc:  # noqa: BLE001 - A replayed failure is part of the inspected state.
            run.status, run.error = "failed", f"{type(exc).__name__}: {exc}"
        return Replay(run, runtime.seq, reason, tuple(runtime._pending))

    async def fork(
        self,
        session_id: str,
        run_id: str,
        *,
        at_seq: int,
        context: Mapping[str, Any] | None = None,
        new_session_id: str | None = None,
    ) -> Run:
        """Start a new run that reuses a recorded run's first ``at_seq`` commands, then runs live.

        Call it on an agent with the configuration to try, e.g.
        ``replace(agent, provider=other).fork(...)``. Recorded results of the
        prefix are reused by position, so a changed system prompt or tool set does
        not invalidate them. The fork gets its own session (``new_session_id`` or a
        fresh ID) and its own journal, so it can itself be recovered or forked.
        """
        if self.store is None:
            raise ValueError("fork() requires an Agent with a durable store")
        if at_seq < 0:
            raise ValueError("at_seq must be nonnegative")
        source = self.store.get_run(session_id, run_id)
        if source is None:
            raise ValueError(f"Run {run_id} not found in session {session_id}")
        new_session_id = uuid4().hex if new_session_id is None else new_session_id
        validate_session_id(new_session_id)
        entries = []
        for entry in self.store.journal(session_id, run_id):
            if entry.phase == "marker":
                if entry.payload.get("at_seq", 0) <= at_seq:
                    entries.append(entry)
            elif entry.seq <= at_seq:
                entries.append(entry)
        forked_from = {"session_id": session_id, "run_id": run_id, "at_seq": at_seq}
        async with self.sessions.lock(new_session_id):
            if self.sessions.load(new_session_id) is not None or self.store.runs(session_id=new_session_id):
                raise ValueError(f"Session {new_session_id} already exists")
            recorded = [Message.model_validate(message) for message in source.start["messages"]]
            if recorded and recorded[0].role == "system":
                recorded[0] = Message(role="system", content=self.system_prompt)
            run = Run(session_id=new_session_id, context=dict(context or {}), messages=recorded)
            await self.store.create_run(
                RunRecord(
                    session_id=run.session_id,
                    run_id=run.id,
                    agent=self.name,
                    owner=self.store.owner_id,
                    epoch=1,
                    lease_expires_at=time.time() + self.lease_ttl,
                    start={**self._start(run), "forked_from": forked_from},
                ),
                entries,
            )
            return await self._execute(
                run,
                directory=self.log_dir,
                resume_log=False,
                recovery=_Recovery(
                    1, tuple(entries), 0.0, lenient_until=at_seq, log_replay=True,
                    live_event=("fork_live", {"forked_from": forked_from}),
                ),
                forked_from=forked_from,
            )

    def _start(self, run: Run) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "messages": [message.model_dump(mode="json") for message in run.messages],
            "limits": self.limits.model_dump(mode="json"),
            "loop": type(self.loop).__name__,
            "fingerprint": _fingerprint(self),
            "log_path": str(run.log_path) if run.log_path else None,
            "parent_session_id": run.parent_session_id,
            "parent_run_id": run.parent_run_id,
            "job_id": run.job_id,
        }

    @staticmethod
    def _run_from_record(record: RunRecord, context: Mapping[str, Any] | None) -> Run:
        start = record.start
        return Run(
            id=record.run_id,
            session_id=record.session_id,
            parent_session_id=start.get("parent_session_id"),
            parent_run_id=start.get("parent_run_id"),
            job_id=start.get("job_id"),
            context=dict(context or {}),
            messages=[Message.model_validate(message) for message in start["messages"]],
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
        self,
        run: Run,
        *,
        directory: str | Path | None,
        resume_log: bool,
        recovery: _Recovery | None = None,
        forked_from: dict | None = None,
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
                    SessionLog(directory, run.session_id, resume=resume_log, repair=recovery is not None and resume_log)
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
                                start=self._start(run),
                            )
                        )
                        journal = RunJournal(self.store, run.session_id, run.id, 1)
                    else:
                        journal = RunJournal(
                            self.store, run.session_id, run.id, recovery.epoch, recovery.entries,
                            lenient_until=recovery.lenient_until,
                        )
                    heartbeat = asyncio.create_task(self._heartbeat(journal, active), name=f"heartbeat-{run.id}")
                else:
                    # Without a journal, a "running" checkpoint is how a crash is detected later.
                    await self._checkpoint(run, "running")
                runtime = Runtime(
                    run, self.provider, self.tools, self.limits, journal=journal, shared=self.shared,
                    **({"log_replay": recovery.log_replay, "live_event": recovery.live_event} if recovery else {}),
                )
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
                    **({"forked_from": forked_from} if forked_from else {}),
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
                        if journal is not None:
                            await journal.flush()  # Queued side effects precede the final status.
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


def _lenient_until(start: Mapping[str, Any]) -> int:
    forked_from = start.get("forked_from")
    return forked_from.get("at_seq", 0) if isinstance(forked_from, Mapping) else 0


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

    async def recover_all(
        self, *, force: bool = False, allow_fingerprint_change: bool = False
    ) -> list[Run | BaseException]:
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
                return await agent.recover(
                    record.session_id, context=context, force=force, allow_fingerprint_change=allow_fingerprint_change
                )

        records = self.store.runs(status="running")
        return await asyncio.gather(*(recover(record) for record in records), return_exceptions=True)
