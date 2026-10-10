"""Conversation checkpoints and exclusive access for resumable sessions."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from .session_log import validate_session_id
from .types import Message, Record


class Session(Record):
    session_id: str
    agent: str
    system_prompt: str
    messages: tuple[Message, ...] = ()
    log_path: str | None = None
    status: str = "completed"


class SessionStore(Protocol):
    """load/save are called while holding lock for the entire agent invocation."""

    def lock(self, session_id: str): ...
    def load(self, session_id: str) -> Session | None: ...
    def save(self, session: Session) -> None: ...


class MemorySessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def lock(self, session_id: str):
        validate_session_id(session_id)
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            yield

    def load(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        return session.model_copy(deep=True) if session is not None else None

    def save(self, session: Session) -> None:
        self._sessions[session.session_id] = session.model_copy(deep=True)


class FileSessionStore:
    """Atomic JSON checkpoints with OS locks (macOS/Linux, local filesystem).

    Locks are released by the OS on process exit. Waiting never blocks the event
    loop. Checkpoints contain conversation only, not live Python resources.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()

    def _path(self, session_id: str) -> Path:
        validate_session_id(session_id)
        return self.directory / f"{session_id}.json"

    @asynccontextmanager
    async def lock(self, session_id: str):
        import fcntl

        path = self._path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".lock").open("a") as handle:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.02)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def load(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        try:
            session = Session.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if session.session_id != session_id:
            raise ValueError("Session checkpoint ID does not match its filename")
        return session

    def save(self, session: Session) -> None:
        path = self._path(session.session_id)
        temporary = path.with_suffix(".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(session.model_dump_json())
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
