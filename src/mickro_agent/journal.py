"""Durable run journal: run ownership, recorded commands and a group committer.

A run's model and tool commands are journaled at the Runtime boundary. Recovery
re-executes the loop and returns recorded results instead of repeating effects.
See docs/durable-state.md, sections 4, 5 and 11.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, JsonValue

from .session_log import validate_session_id
from .sessions import Session
from .types import Record

SCHEMA_VERSION = 1
INTERRUPTED_CONTENT = "Execution interrupted. Outcome unknown; inspect external state before retrying."


class JournalError(RuntimeError):
    """The journal could not be written. The run must stop without further effects."""


class RunOwnershipLost(JournalError):
    """Another owner claimed this run (a newer epoch), or the run is no longer running."""


class RunLeaseHeld(RuntimeError):
    """The run's current owner still holds a valid lease; pass force=True to take over."""


class NonDeterminismError(RuntimeError):
    """Replay produced a different command than the journal recorded at this position."""

    def __init__(self, seq: int, expected: str, actual: str):
        super().__init__(f"Command {seq}: journal recorded {expected}, code issued {actual}")
        self.seq, self.expected, self.actual = seq, expected, actual


class JournalEntry(Record):
    seq: int = Field(ge=1)
    phase: Literal["scheduled", "completed"]
    kind: Literal["model", "tool"]
    key: str
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class RunRecord(Record):
    session_id: str
    run_id: str
    agent: str
    status: str = "running"
    owner: str | None = None
    epoch: int = Field(default=1, ge=0)
    lease_expires_at: float = 0.0
    active_seconds: float = 0.0
    start: dict[str, JsonValue] = Field(default_factory=dict)
    result: dict[str, JsonValue] | None = None


def command_key(kind: str, value: Any) -> str:
    """Stable hash identifying a command, used to detect divergent replays."""
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return f"{kind}:{hashlib.sha256(body.encode()).hexdigest()[:32]}"


# Operations submitted to the committer. Each is applied atomically on its own.
@dataclass(frozen=True)
class _CreateRun:
    record: RunRecord


@dataclass(frozen=True)
class _ClaimRun:
    session_id: str
    run_id: str
    owner: str
    lease_expires_at: float
    now: float
    force: bool


@dataclass(frozen=True)
class _Append:
    session_id: str
    run_id: str
    epoch: int
    entries: tuple[JournalEntry, ...]


@dataclass(frozen=True)
class _Heartbeat:
    session_id: str
    run_id: str
    epoch: int
    lease_expires_at: float
    active_seconds: float


@dataclass(frozen=True)
class _Finish:
    session_id: str
    run_id: str
    epoch: int
    status: str
    result: dict
    active_seconds: float
    session: Session | None = None


@dataclass(frozen=True)
class _SaveSession:
    session: Session


class _Committer:
    """Batches concurrent submissions into one write; each op gets its own result.

    The worker drains the queue and exits, so no task outlives an event loop.
    Submissions that arrive while a batch is being written join the next batch.
    """

    def __init__(self, write: Callable[[list], list], *, threaded: bool, max_batch: int = 512):
        self._write = write
        self._threaded = threaded
        self._max_batch = max_batch
        self._queue: list[tuple[Any, asyncio.Future]] = []
        self._worker: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def submit(self, op: Any) -> Any:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop, self._queue, self._worker = loop, [], None
        future = loop.create_future()
        self._queue.append((op, future))
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._drain(), name="journal-committer")
        return await future

    async def _drain(self) -> None:
        while self._queue:
            await asyncio.sleep(0)  # Let concurrently scheduled submitters join this batch.
            batch, self._queue = self._queue[: self._max_batch], self._queue[self._max_batch :]
            ops = [op for op, _ in batch]
            try:
                results = await asyncio.to_thread(self._write, ops) if self._threaded else self._write(ops)
            except asyncio.CancelledError:
                for _, future in batch:
                    if not future.done():
                        future.cancel()
                raise
            except BaseException as exc:  # noqa: BLE001 - A failed batch fails each waiting submitter.
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
                continue
            for (_, future), result in zip(batch, results, strict=True):
                if future.done():
                    continue
                if isinstance(result, BaseException):
                    future.set_exception(result)
                else:
                    future.set_result(result)


class _Tx:
    """Primitive reads/writes one backend offers inside a write batch."""

    def get_run(self, session_id: str, run_id: str) -> RunRecord | None: ...
    def put_run(self, record: RunRecord) -> None: ...
    def insert_entries(self, session_id: str, run_id: str, epoch: int, entries: Sequence[JournalEntry]) -> None: ...
    def put_session(self, session: Session) -> None: ...


def _owned(tx: _Tx, session_id: str, run_id: str, epoch: int) -> RunRecord:
    record = tx.get_run(session_id, run_id)
    if record is None:
        raise RunOwnershipLost(f"Run {run_id} does not exist")
    if record.status != "running" or record.epoch != epoch:
        raise RunOwnershipLost(
            f"Run {run_id} is {record.status} at epoch {record.epoch}; this owner holds epoch {epoch}"
        )
    return record


def _apply(tx: _Tx, op: Any) -> Any:
    if isinstance(op, _CreateRun):
        if tx.get_run(op.record.session_id, op.record.run_id) is not None:
            raise JournalError(f"Run {op.record.run_id} already exists")
        tx.put_run(op.record)
        return op.record.epoch
    if isinstance(op, _ClaimRun):
        record = tx.get_run(op.session_id, op.run_id)
        if record is None:
            raise ValueError(f"Run {op.run_id} does not exist")
        if record.status != "running":
            raise ValueError(f"Run {op.run_id} is {record.status}, not running")
        if not op.force and record.owner != op.owner and record.lease_expires_at > op.now:
            raise RunLeaseHeld(
                f"Run {op.run_id} is leased by {record.owner} for {record.lease_expires_at - op.now:.1f}s more"
            )
        claimed = record.model_copy(
            update={"owner": op.owner, "epoch": record.epoch + 1, "lease_expires_at": op.lease_expires_at}
        )
        tx.put_run(claimed)
        return claimed.epoch
    if isinstance(op, _Append):
        _owned(tx, op.session_id, op.run_id, op.epoch)
        tx.insert_entries(op.session_id, op.run_id, op.epoch, op.entries)
        return None
    if isinstance(op, _Heartbeat):
        record = _owned(tx, op.session_id, op.run_id, op.epoch)
        tx.put_run(
            record.model_copy(update={"lease_expires_at": op.lease_expires_at, "active_seconds": op.active_seconds})
        )
        return None
    if isinstance(op, _Finish):
        record = _owned(tx, op.session_id, op.run_id, op.epoch)
        tx.put_run(
            record.model_copy(
                update={
                    "status": op.status,
                    "result": op.result,
                    "active_seconds": op.active_seconds,
                    "owner": None,
                    "lease_expires_at": 0.0,
                }
            )
        )
        if op.session is not None:
            tx.put_session(op.session)  # Same transaction: the checkpoint cannot lag the run.
        return None
    if isinstance(op, _SaveSession):
        tx.put_session(op.session)
        return None
    raise TypeError(f"Unknown journal operation {type(op).__name__}")


class DurableStore:
    """Journal, run ownership and session checkpoints behind one group committer.

    Also implements SessionStore, so one store holds everything an Agent persists.
    Session locks are in-process: run every agent of a store in one process.
    """

    def __init__(self, *, threaded: bool):
        self.owner_id = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
        self._committer = _Committer(self._write_batch, threaded=threaded)
        self._locks: dict[str, asyncio.Lock] = {}

    # Backend hooks --------------------------------------------------------
    def _write_batch(self, ops: list) -> list:
        """Apply ops in order, each atomically; return a result or exception per op."""
        raise NotImplementedError

    def get_run(self, session_id: str, run_id: str) -> RunRecord | None:
        raise NotImplementedError

    def runs(self, *, status: str | None = None, session_id: str | None = None) -> list[RunRecord]:
        raise NotImplementedError

    def journal(self, session_id: str, run_id: str) -> list[JournalEntry]:
        raise NotImplementedError

    def load(self, session_id: str) -> Session | None:
        raise NotImplementedError

    # SessionStore ---------------------------------------------------------
    @asynccontextmanager
    async def lock(self, session_id: str):
        validate_session_id(session_id)
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            yield

    def save(self, session: Session) -> None:
        (result,) = self._write_batch([_SaveSession(session)])
        if isinstance(result, BaseException):
            raise result

    async def asave(self, session: Session) -> None:
        await self._committer.submit(_SaveSession(session))

    # Runs and journal -----------------------------------------------------
    async def create_run(self, record: RunRecord) -> None:
        await self._committer.submit(_CreateRun(record))

    async def claim_run(self, session_id: str, run_id: str, *, ttl: float, force: bool = False) -> int:
        """Take ownership of a running run; returns the new epoch."""
        now = time.time()
        return await self._committer.submit(_ClaimRun(session_id, run_id, self.owner_id, now + ttl, now, force))

    async def append(self, session_id: str, run_id: str, epoch: int, entries: Iterable[JournalEntry]) -> None:
        await self._committer.submit(_Append(session_id, run_id, epoch, tuple(entries)))

    async def heartbeat(self, session_id: str, run_id: str, epoch: int, *, ttl: float, active_seconds: float) -> None:
        await self._committer.submit(_Heartbeat(session_id, run_id, epoch, time.time() + ttl, active_seconds))

    async def finish_run(
        self,
        session_id: str,
        run_id: str,
        epoch: int,
        *,
        status: str,
        result: dict,
        active_seconds: float,
        session: Session | None = None,
    ) -> None:
        """Record the run's outcome and, atomically with it, the session checkpoint."""
        await self._committer.submit(_Finish(session_id, run_id, epoch, status, result, active_seconds, session))


# Memory ------------------------------------------------------------------
@dataclass
class MemoryStoreData:
    """Backing data of a MemoryStore. Share it between stores to simulate a restart."""

    runs: dict[tuple[str, str], RunRecord] = field(default_factory=dict)
    entries: dict[tuple[str, str], dict[tuple[int, str], JournalEntry]] = field(default_factory=dict)
    sessions: dict[str, Session] = field(default_factory=dict)


class _MemoryTx(_Tx):
    """Stages one op's changes; commit() applies them, so a failed op changes nothing."""

    def __init__(self, data: MemoryStoreData):
        self.data = data
        self.runs: dict[tuple[str, str], RunRecord] = {}
        self.entries: list[tuple[tuple[str, str], JournalEntry]] = []
        self.sessions: dict[str, Session] = {}

    def get_run(self, session_id, run_id):
        key = (session_id, run_id)
        return self.runs.get(key, self.data.runs.get(key))

    def put_run(self, record):
        self.runs[record.session_id, record.run_id] = record

    def insert_entries(self, session_id, run_id, epoch, entries):
        existing = self.data.entries.get((session_id, run_id), {})
        seen = set()
        for entry in entries:
            position = (entry.seq, entry.phase)
            if position in existing or position in seen:
                raise JournalError(f"Journal entry {position} already exists for run {run_id}")
            seen.add(position)
            self.entries.append(((session_id, run_id), entry))

    def put_session(self, session):
        self.sessions[session.session_id] = session

    def commit(self) -> None:
        self.data.runs.update(self.runs)
        for key, entry in self.entries:
            self.data.entries.setdefault(key, {})[entry.seq, entry.phase] = entry
        self.data.sessions.update(self.sessions)


class MemoryStore(DurableStore):
    """In-process durable store for tests and experiments. Survives nothing but itself."""

    def __init__(self, data: MemoryStoreData | None = None):
        super().__init__(threaded=False)
        self.data = data if data is not None else MemoryStoreData()

    def _write_batch(self, ops):
        results = []
        for op in ops:
            tx = _MemoryTx(self.data)
            try:
                results.append(_apply(tx, op))
                tx.commit()
            except Exception as exc:  # noqa: BLE001 - Returned to the op's submitter.
                results.append(exc)
        return results

    def get_run(self, session_id, run_id):
        return self.data.runs.get((session_id, run_id))

    def runs(self, *, status=None, session_id=None):
        return [
            record
            for record in self.data.runs.values()
            if (status is None or record.status == status) and (session_id is None or record.session_id == session_id)
        ]

    def journal(self, session_id, run_id):
        entries = self.data.entries.get((session_id, run_id), {})
        return [entries[key] for key in sorted(entries, key=lambda key: (key[0], key[1] != "scheduled"))]

    def load(self, session_id):
        validate_session_id(session_id)
        session = self.data.sessions.get(session_id)
        return session.model_copy(deep=True) if session is not None else None


# SQLite ------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  session_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  status TEXT NOT NULL,
  owner TEXT,
  epoch INTEGER NOT NULL,
  lease_expires_at REAL NOT NULL,
  active_seconds REAL NOT NULL,
  start TEXT NOT NULL,
  result TEXT,
  updated_at REAL NOT NULL,
  PRIMARY KEY (session_id, run_id)
);
CREATE INDEX IF NOT EXISTS runs_by_status ON runs (status, session_id);
CREATE TABLE IF NOT EXISTS journal (
  session_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  phase TEXT NOT NULL,
  kind TEXT NOT NULL,
  key TEXT NOT NULL,
  payload TEXT NOT NULL,
  epoch INTEGER NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (session_id, run_id, seq, phase)
);
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  body TEXT NOT NULL
);
"""


def _record_from_row(row) -> RunRecord:
    session_id, run_id, agent, status, owner, epoch, lease, active, start, result = row
    return RunRecord(
        session_id=session_id,
        run_id=run_id,
        agent=agent,
        status=status,
        owner=owner,
        epoch=epoch,
        lease_expires_at=lease,
        active_seconds=active,
        start=json.loads(start),
        result=json.loads(result) if result is not None else None,
    )


_RUN_COLUMNS = "session_id, run_id, agent, status, owner, epoch, lease_expires_at, active_seconds, start, result"


class _SqliteTx(_Tx):
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def get_run(self, session_id, run_id):
        row = self.connection.execute(
            f"SELECT {_RUN_COLUMNS} FROM runs WHERE session_id = ? AND run_id = ?", (session_id, run_id)
        ).fetchone()
        return _record_from_row(row) if row is not None else None

    def put_run(self, record):
        self.connection.execute(
            f"INSERT OR REPLACE INTO runs ({_RUN_COLUMNS}, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.session_id,
                record.run_id,
                record.agent,
                record.status,
                record.owner,
                record.epoch,
                record.lease_expires_at,
                record.active_seconds,
                json.dumps(record.start, ensure_ascii=False),
                json.dumps(record.result, ensure_ascii=False) if record.result is not None else None,
                time.time(),
            ),
        )

    def insert_entries(self, session_id, run_id, epoch, entries):
        now = time.time()
        try:
            self.connection.executemany(
                "INSERT INTO journal (session_id, run_id, seq, phase, kind, key, payload, epoch, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        session_id,
                        run_id,
                        entry.seq,
                        entry.phase,
                        entry.kind,
                        entry.key,
                        json.dumps(entry.payload, ensure_ascii=False),
                        epoch,
                        now,
                    )
                    for entry in entries
                ],
            )
        except sqlite3.IntegrityError as exc:
            raise JournalError(f"Journal entry already exists for run {run_id}: {exc}") from exc

    def put_session(self, session):
        self.connection.execute(
            "INSERT OR REPLACE INTO sessions (session_id, body) VALUES (?, ?)",
            (session.session_id, session.model_dump_json()),
        )


class SqliteStore(DurableStore):
    """Single-file durable store (WAL, synchronous=FULL) for one process on one host.

    Writes run in a worker thread, batched by the group committer: one
    transaction per batch, one savepoint per operation.
    """

    def __init__(self, path: str | Path):
        super().__init__(threaded=True)
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        with self._db_lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA busy_timeout=5000")
            version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise JournalError(f"Unsupported journal schema version {version} in {self.path}")
            self._connection.executescript(_SCHEMA)
            self._connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        with self._db_lock:
            self._connection.close()

    def _write_batch(self, ops):
        with self._db_lock:
            connection = self._connection
            tx = _SqliteTx(connection)
            results = []
            try:
                connection.execute("BEGIN IMMEDIATE")
                for op in ops:
                    connection.execute("SAVEPOINT op")
                    try:
                        results.append(_apply(tx, op))
                        connection.execute("RELEASE op")
                    except Exception as exc:  # noqa: BLE001 - Returned to the op's submitter.
                        connection.execute("ROLLBACK TO op")
                        connection.execute("RELEASE op")
                        results.append(exc)
                connection.execute("COMMIT")
            except sqlite3.Error as exc:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise JournalError(f"Cannot write journal {self.path}: {exc}") from exc
            return results

    def _read(self, sql: str, parameters: tuple = ()) -> list:
        with self._db_lock:
            return self._connection.execute(sql, parameters).fetchall()

    def get_run(self, session_id, run_id):
        rows = self._read(
            f"SELECT {_RUN_COLUMNS} FROM runs WHERE session_id = ? AND run_id = ?", (session_id, run_id)
        )
        return _record_from_row(rows[0]) if rows else None

    def runs(self, *, status=None, session_id=None):
        clauses, parameters = [], []
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        if session_id is not None:
            clauses.append("session_id = ?")
            parameters.append(session_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return [_record_from_row(row) for row in self._read(f"SELECT {_RUN_COLUMNS} FROM runs{where}", tuple(parameters))]

    def journal(self, session_id, run_id):
        rows = self._read(
            "SELECT seq, phase, kind, key, payload FROM journal WHERE session_id = ? AND run_id = ?"
            " ORDER BY seq, phase = 'completed'",
            (session_id, run_id),
        )
        return [
            JournalEntry(seq=seq, phase=phase, kind=kind, key=key, payload=json.loads(payload))
            for seq, phase, kind, key, payload in rows
        ]

    def load(self, session_id):
        validate_session_id(session_id)
        rows = self._read("SELECT body FROM sessions WHERE session_id = ?", (session_id,))
        if not rows:
            return None
        session = Session.model_validate_json(rows[0][0])
        if session.session_id != session_id:
            raise ValueError("Session checkpoint ID does not match its key")
        return session


# Per-run view used by Runtime ------------------------------------------------
class RunJournal:
    """One owner's view of a run's journal: recorded entries plus appends at its epoch."""

    def __init__(
        self, store: DurableStore, session_id: str, run_id: str, epoch: int, entries: Iterable[JournalEntry] = ()
    ):
        self.store, self.session_id, self.run_id, self.epoch = store, session_id, run_id, epoch
        self._recorded = {(entry.seq, entry.phase): entry for entry in entries}
        self.last_recorded_seq = max((seq for seq, _ in self._recorded), default=0)

    def recorded(self, seq: int) -> tuple[JournalEntry | None, JournalEntry | None]:
        return self._recorded.get((seq, "scheduled")), self._recorded.get((seq, "completed"))

    def check(self, seq: int, kind: str, key: str) -> tuple[JournalEntry | None, JournalEntry | None]:
        scheduled, completed = self.recorded(seq)
        for entry in (scheduled, completed):
            if entry is not None and (entry.kind != kind or entry.key != key):
                raise NonDeterminismError(seq, f"{entry.kind} {entry.key}", f"{kind} {key}")
        return scheduled, completed

    async def append(self, *entries: JournalEntry) -> None:
        await self.store.append(self.session_id, self.run_id, self.epoch, entries)
