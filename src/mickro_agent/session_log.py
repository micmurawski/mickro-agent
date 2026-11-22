"""Append-only, per-session JSONL logging. No SDK or harness dependency."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Self, TextIO

if TYPE_CHECKING:
    from .types import Event


class SessionLogError(OSError):
    """The requested session log could not be opened or written."""


def validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", session_id) is None:
        raise ValueError(
            "session_id must be 1–128 letters, digits, underscores or hyphens, starting with a letter/digit"
        )


def event_line(event: Event) -> str:
    return (
        json.dumps(
            {"schema_version": 2, **event.model_dump(mode="json"), "session_id": event.session_id or event.run_id},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


class SessionLog:
    """Per-session JSONL log. ``repair`` (crash recovery) drops a partial final line."""

    def __init__(self, directory: str | Path, session_id: str, *, resume: bool = False, repair: bool = False):
        validate_session_id(session_id)
        self.path = Path(directory).expanduser().resolve() / f"mikro-agent-{session_id}.jsonl"
        self._file: TextIO | None = None
        self.resume = resume
        self.repair = repair
        self.session_id = session_id
        # Next free event sequence per run already in the log, filled when resuming.
        self.next_sequence: dict[str, int] = {}

    def __enter__(self) -> Self:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Appending is authorized only by a matching, locked checkpoint.
            if self.resume:
                if self.repair:
                    data = self.path.read_bytes()
                    if data and not data.endswith(b"\n"):
                        with self.path.open("r+b") as handle:
                            handle.truncate(data.rfind(b"\n") + 1)
                with self.path.open(encoding="utf-8") as existing:
                    for line in existing:
                        record = json.loads(line) if line.endswith("\n") else None
                        if record is None or record.get("session_id") != self.session_id:
                            raise ValueError("Incomplete or mismatched session log")
                        run_id, sequence = record.get("run_id"), record.get("sequence", -1)
                        self.next_sequence[run_id] = max(self.next_sequence.get(run_id, 0), sequence + 1)
            self._file = self.path.open("a" if self.resume else "x", encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise SessionLogError(f"Cannot create session log {self.path}: {exc}") from exc
        return self

    def write(self, event: Event) -> None:
        if self._file is None:
            raise SessionLogError("Session log is not open")
        try:
            self._file.write(event_line(event))
            self._file.flush()
        except OSError as exc:
            raise SessionLogError(f"Cannot write session log {self.path}: {exc}") from exc

    def __exit__(self, *args) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError as exc:
                raise SessionLogError(f"Cannot close session log {self.path}: {exc}") from exc
            finally:
                self._file = None
