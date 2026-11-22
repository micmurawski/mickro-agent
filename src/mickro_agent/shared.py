"""Shared state: one JSON tree per world, many agents, an append-only log.

Paths are JSON Pointers (RFC 6901); writes are JSON Patch operations (RFC 6902).
Every committed entry gets a gap-free offset, so the world can be rebuilt at any
offset. Each path has a version (the offset of the last write affecting it), so
a write can require that what it read has not changed (``based_on``).
See docs/durable-state.md, sections 6.1, 6.2 and 7.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import Field, JsonValue

from .journal import DurableStore, JournalEntry, JournalError, _owned
from .types import Record

if TYPE_CHECKING:
    from .runtime import Runtime

MAX_OPS = 100
MAX_ENTRY_BYTES = 256 * 1024
_MISSING = object()


class SharedError(Exception):
    """Base class for shared-state errors."""


class InvalidPatch(SharedError, ValueError):
    """The patch is malformed. Raised before anything is journaled or committed."""


class Rejected(SharedError):
    """The store rejected a well-formed write; the rejection is journaled and replays."""

    error = "rejected"

    def __init__(self, message: str, paths: Sequence[str] = ()):
        super().__init__(message)
        self.paths = tuple(paths)

    def to_result(self) -> dict:
        return {"status": "rejected", "error": self.error, "message": str(self), "paths": list(self.paths)}


class PatchConflict(Rejected):
    """A path did not exist where the patch needed it, or a ``test`` op failed."""

    error = "conflict"


class StaleRead(Rejected):
    """A ``based_on`` read is no longer current: something it covered was written since."""

    error = "stale"


_REJECTIONS = {cls.error: cls for cls in (PatchConflict, StaleRead)}


def raise_for(result: Mapping) -> Committed:
    if result["status"] == "committed":
        return Committed(result["offset"])
    raise _REJECTIONS[result["error"]](result["message"], result.get("paths", ()))


# Paths ------------------------------------------------------------------------
def parse_path(path: str) -> tuple[str, ...]:
    """JSON Pointer ("/a/b", "" for the root) or dotted shorthand ("a.b") to segments."""
    if not isinstance(path, str):
        raise InvalidPatch("A path must be a string")
    if path == "":
        return ()
    if path.startswith("/"):
        return tuple(part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/"))
    parts = path.split(".")
    if any(not part or "/" in part or "~" in part for part in parts):
        raise InvalidPatch(f"Invalid path {path!r}: use a JSON Pointer such as '/a/b' for keys with '.', '/' or '~'")
    return tuple(parts)


def format_path(segments: Iterable[str]) -> str:
    return "".join("/" + segment.replace("~", "~0").replace("/", "~1") for segment in segments)


def canonical_path(path: str) -> str:
    return format_path(parse_path(path))


def _index(segment: str, length: int, *, append: bool) -> int:
    if append and segment == "-":
        return length
    if not segment.isdigit() or (segment != "0" and segment.startswith("0")):
        raise PatchConflict(f"Invalid array index {segment!r}", [])
    index = int(segment)
    if index > length or (index == length and not append):
        raise PatchConflict(f"Array index {index} out of range", [])
    return index


def _child(value: Any, segment: str) -> Any:
    if isinstance(value, dict):
        return value.get(segment, _MISSING)
    if isinstance(value, list):
        try:
            return value[_index(segment, len(value), append=False)]
        except PatchConflict:
            return _MISSING
    return _MISSING


def _lookup(value: Any, segments: Sequence[str]) -> Any:
    for segment in segments:
        value = _child(value, segment)
        if value is _MISSING:
            return _MISSING
    return value


def _json(value: Any) -> Any:
    """A private, validated JSON copy of a value."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise InvalidPatch(f"Value is not JSON: {exc}") from exc


def _same(left: Any, right: Any) -> bool:
    # JSON equality: 1 and true differ, key order does not matter.
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


# Patch ops ------------------------------------------------------------------------
_OPS = {"add", "remove", "replace", "test"}


def normalize_ops(ops: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Validate a patch and return it with canonical paths and private JSON values."""
    if isinstance(ops, Mapping):
        raise InvalidPatch("A patch is a list of operations")
    normalized = []
    for op in ops:
        if not isinstance(op, Mapping) or op.get("op") not in _OPS:
            raise InvalidPatch(f"Each operation needs 'op' in {sorted(_OPS)}: {op!r}")
        extra = set(op) - {"op", "path", "value"}
        if extra:
            raise InvalidPatch(f"Unsupported operation fields {sorted(extra)}")
        if "path" not in op:
            raise InvalidPatch(f"Operation {op['op']!r} needs a path")
        item = {"op": op["op"], "path": canonical_path(op["path"])}
        if op["op"] == "remove":
            if "value" in op:
                raise InvalidPatch("'remove' takes no value")
        elif "value" not in op:
            raise InvalidPatch(f"'{op['op']}' needs a value")
        else:
            item["value"] = _json(op["value"])
        normalized.append(item)
    if not normalized:
        raise InvalidPatch("A patch needs at least one operation")
    if len(normalized) > MAX_OPS:
        raise InvalidPatch(f"A patch has at most {MAX_OPS} operations")
    if len(json.dumps(normalized)) > MAX_ENTRY_BYTES:
        raise InvalidPatch(f"A patch is at most {MAX_ENTRY_BYTES} bytes; store large values elsewhere by reference")
    return normalized


def _set_parent(value: Any, segments: Sequence[str], change, path: str) -> Any:
    """Copy the containers along ``segments`` and apply ``change`` to the last one."""
    if not segments:
        return change(value)
    head, rest = segments[0], segments[1:]
    if isinstance(value, dict):
        if head not in value:
            raise PatchConflict(f"Path {path} does not exist", [path])
        copy = dict(value)
        copy[head] = _set_parent(value[head], rest, change, path)
        return copy
    if isinstance(value, list):
        index = _index(head, len(value), append=False)
        copy = list(value)
        copy[index] = _set_parent(value[index], rest, change, path)
        return copy
    raise PatchConflict(f"Path {path} does not exist", [path])


def _apply_op(value: Any, segments: tuple[str, ...], op: dict) -> Any:
    path, kind = op["path"], op["op"]
    if not segments:
        if kind == "remove":
            raise PatchConflict("The root cannot be removed", [path])
        return op["value"]
    parent, last = segments[:-1], segments[-1]

    def change(container):
        if isinstance(container, dict):
            if kind == "add":
                return {**container, last: op["value"]}
            if last not in container:
                raise PatchConflict(f"Path {path} does not exist", [path])
            copy = dict(container)
            if kind == "remove":
                del copy[last]
            else:
                copy[last] = op["value"]
            return copy
        if isinstance(container, list):
            index = _index(last, len(container), append=kind == "add")
            copy = list(container)
            if kind == "add":
                copy.insert(index, op["value"])
            elif kind == "remove":
                del copy[index]
            else:
                copy[index] = op["value"]
            return copy
        raise PatchConflict(f"Parent of {path} is not an object or array", [path])

    return _set_parent(value, parent, change, path)


# Versions ------------------------------------------------------------------------
@dataclass(frozen=True)
class _VNode:
    own: int = 0  # Last write at exactly this path.
    sub: int = 0  # Last write at or below this path.
    kids: Mapping[str, _VNode] = field(default_factory=dict)

    def to_json(self) -> dict:
        body = {"o": self.own, "s": self.sub}
        if self.kids:
            body["k"] = {key: node.to_json() for key, node in self.kids.items()}
        return body

    @staticmethod
    def from_json(body: Mapping) -> _VNode:
        return _VNode(body["o"], body["s"], {key: _VNode.from_json(kid) for key, kid in body.get("k", {}).items()})


def _version_target(value: Any, segments: tuple[str, ...]) -> tuple[str, ...]:
    """Versions stop at arrays: an index is not a stable identity, so a whole array is one unit."""
    for depth, segment in enumerate(segments):
        if isinstance(value, list):
            return segments[:depth]
        value = _child(value, segment)
    return segments


def _bump(node: _VNode, target: tuple[str, ...], offset: int) -> _VNode:
    if not target:
        return _VNode(offset, offset, {})  # The value was replaced: older versions below no longer apply.
    kids = dict(node.kids)
    kids[target[0]] = _bump(node.kids.get(target[0], _VNode()), target[1:], offset)
    return _VNode(node.own, offset, kids)


# State ------------------------------------------------------------------------
@dataclass(frozen=True)
class Versioned:
    """A read: the value at ``path`` and the version a later write can be ``based_on``."""

    path: str
    value: Any
    version: int
    offset: int
    exists: bool = True

    def to_json(self) -> dict:
        return {"path": self.path, "value": self.value, "version": self.version, "offset": self.offset,
                "exists": self.exists}


@dataclass(frozen=True)
class Committed:
    offset: int


class LogEntry(Record):
    offset: int = Field(ge=1)
    op_id: str
    ops: list[dict[str, JsonValue]]
    based_on: list[dict[str, JsonValue]] = Field(default_factory=list)
    author: dict[str, JsonValue] | None = None
    committed_at: float


@dataclass(frozen=True)
class SharedState:
    """An immutable world state at ``offset``. Writes return a new state (path copying)."""

    value: Any = field(default_factory=dict)
    versions: _VNode = field(default_factory=_VNode)
    offset: int = 0
    committed_at: float = 0.0

    def version(self, path: str) -> int:
        segments = parse_path(path)
        value, node, latest = self.value, self.versions, self.versions.own
        for segment in segments:
            if isinstance(value, list):
                break
            child = node.kids.get(segment)
            if child is None:
                return latest  # Never written below here: only ancestor replacements matter.
            node, value = child, _child(value, segment)
            latest = max(latest, node.own)
        return max(latest, node.sub)

    def read(self, path: str) -> Versioned:
        canonical = canonical_path(path)
        value = _lookup(self.value, parse_path(canonical))
        exists = value is not _MISSING
        return Versioned(canonical, _json(value) if exists else None, self.version(canonical), self.offset, exists)

    def stale(self, based_on: Iterable[Mapping]) -> list[str]:
        return [item["path"] for item in based_on if self.version(item["path"]) != item["version"]]

    def apply(self, ops: Sequence[dict], *, offset: int, committed_at: float) -> SharedState:
        """Apply a normalized patch atomically; raises PatchConflict and changes nothing."""
        value, versions = self.value, self.versions
        for op in ops:
            segments = parse_path(op["path"])
            if op["op"] == "test":
                current = _lookup(value, segments)
                if current is _MISSING or not _same(current, op["value"]):
                    raise PatchConflict(f"Test failed at {op['path']}", [op["path"]])
                continue
            target = _version_target(value, segments)
            value = _apply_op(value, segments, op)
            versions = _bump(versions, target, offset)
        return SharedState(value, versions, offset, committed_at)

    def apply_entry(self, entry: LogEntry) -> SharedState:
        if entry.offset != self.offset + 1:
            raise JournalError(f"Shared log gap: state at {self.offset}, entry {entry.offset}")
        return self.apply(entry.ops, offset=entry.offset, committed_at=entry.committed_at)

    def to_json(self) -> dict:
        return {"value": self.value, "versions": self.versions.to_json(), "offset": self.offset,
                "committed_at": self.committed_at}

    @staticmethod
    def from_json(body: Mapping) -> SharedState:
        return SharedState(body["value"], _VNode.from_json(body["versions"]), body["offset"], body["committed_at"])


# Commit operation (runs inside the store's write batch) -------------------------------
@dataclass(frozen=True)
class _JournalTarget:
    session_id: str
    run_id: str
    epoch: int
    seq: int
    key: str


@dataclass(frozen=True)
class _SharedCommit:
    world: SharedWorld
    op_id: str
    ops: tuple[dict, ...]
    based_on: tuple[dict, ...]
    author: dict | None
    journal: _JournalTarget | None

    def apply_in(self, tx) -> dict:
        target = self.journal
        if target is not None:
            _owned(tx, target.session_id, target.run_id, target.epoch)  # A stale owner cannot write shared state.
        world = self.world
        state = world._working(tx)
        existing = tx.find_op(world.name, self.op_id)
        new_state = None
        if existing is not None:
            result = {"status": "committed", "offset": existing.offset}  # Deduplicated retry.
        else:
            try:
                stale = state.stale(self.based_on)
                if stale:
                    raise StaleRead(f"Changed since read: {', '.join(stale)}", stale)
                committed_at = max(time.time(), state.committed_at)
                new_state = state.apply(self.ops, offset=state.offset + 1, committed_at=committed_at)
            except Rejected as rejection:
                result = rejection.to_result()
            else:
                entry = LogEntry(offset=new_state.offset, op_id=self.op_id, ops=list(self.ops),
                                 based_on=list(self.based_on), author=self.author, committed_at=committed_at)
                tx.insert_log(world.name, entry)
                tx.set_head(world.name, entry.offset)
                if entry.offset % world.snapshot_every == 0:
                    tx.put_snapshot(world.name, entry.offset, new_state.to_json())
                result = {"status": "committed", "offset": entry.offset}
        if target is not None:
            # The run's journal records the outcome in the same transaction as the log entry.
            tx.insert_entries(target.session_id, target.run_id, target.epoch, [
                JournalEntry(seq=target.seq, phase="completed", kind="shared_write", key=target.key,
                             payload={"result": result}),
            ])
        if new_state is not None:
            world._stage(new_state)  # Last: nothing after this can fail and roll the op back.
        return result


# World ------------------------------------------------------------------------
class SharedWorld:
    """A named JSON tree in a DurableStore, cached in memory for reads.

    Writes go through the store's committer and are validated in commit order
    against the cached state. Run one writer process per store; other processes
    may ``refresh()`` to follow the log.
    """

    def __init__(self, store: DurableStore, name: str = "default", *, snapshot_every: int = 1000):
        if not name or not isinstance(name, str):
            raise ValueError("A world needs a nonempty name")
        if snapshot_every < 1:
            raise ValueError("snapshot_every must be positive")
        if name in store._worlds:
            raise ValueError(f"World {name!r} is already open on this store")
        self.store, self.name, self.snapshot_every = store, name, snapshot_every
        self._published = self._load(None)
        store._worlds[name] = self

    # Reading ---------------------------------------------------------------
    @property
    def offset(self) -> int:
        return self._published.offset

    @property
    def state(self) -> SharedState:
        return self._published

    def read(self, path: str = "") -> Versioned:
        return self._published.read(path)

    async def get(self, path: str = "") -> Versioned:
        return self.read(path)

    def at(self, offset: int) -> SharedState:
        """The world as it was right after the entry at ``offset`` (time travel)."""
        if not 0 <= offset <= self.store._shared_head(self.name):
            raise ValueError(f"Offset {offset} is outside 0..{self.store._shared_head(self.name)}")
        return self._load(offset)

    def log(self, *, after: int = 0, until: int | None = None) -> list[LogEntry]:
        return self.store._shared_log(self.name, after, until)

    def refresh(self) -> None:
        """Catch up with entries committed by another process."""
        state = self._published
        for entry in self.store._shared_log(self.name, state.offset, None):
            state = state.apply_entry(entry)
        self._published = state

    # Writing ---------------------------------------------------------------
    async def patch(
        self, ops: Iterable[Mapping], *, based_on: Iterable[Versioned | Mapping] = (), op_id: str | None = None
    ) -> Committed:
        """Commit a patch outside any run. ``op_id`` makes a retry idempotent."""
        result = await self._commit(normalize_ops(ops), normalize_basis(based_on), op_id or uuid4().hex, None, None)
        return raise_for(result)

    async def _commit(self, ops, based_on, op_id, author, journal) -> dict:
        return await self.store._committer.submit(
            _SharedCommit(self, op_id, tuple(ops), tuple(based_on), author, journal)
        )

    # Store hooks -------------------------------------------------------------
    def _load(self, offset: int | None) -> SharedState:
        snapshot = self.store._shared_snapshot(self.name, offset)
        state = SharedState.from_json(snapshot) if snapshot is not None else SharedState()
        for entry in self.store._shared_log(self.name, state.offset, offset):
            state = state.apply_entry(entry)
        return state

    def _working(self, tx) -> SharedState:
        """State at the store's head, including earlier ops of the current batch."""
        state = self.store._staged.get(self.name, self._published)
        head = tx.get_head(self.name)
        if state.offset < head:  # Another process wrote; catch up inside this transaction.
            for entry in tx.read_log(self.name, state.offset):
                state = state.apply_entry(entry)
        if state.offset != head:
            raise JournalError(f"World {self.name!r} cache at {state.offset} but store head is {head}")
        return state

    def _stage(self, state: SharedState) -> None:
        self.store._staged[self.name] = state

    def _publish(self, state: SharedState) -> None:
        self._published = state


def normalize_basis(based_on: Iterable[Versioned | Mapping]) -> list[dict]:
    basis = []
    for item in based_on:
        if isinstance(item, Versioned):
            path, version = item.path, item.version
        elif isinstance(item, Mapping) and set(item) == {"path", "version"}:
            path, version = item["path"], item["version"]
        else:
            raise InvalidPatch("based_on takes reads (Versioned) or {'path', 'version'} mappings")
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise InvalidPatch("A based_on version is a nonnegative integer")
        basis.append({"path": canonical_path(path), "version": version})
    return basis


# Run-bound access ------------------------------------------------------------------
class SharedView:
    """``runtime.shared``: reads and writes journaled as commands of the run."""

    def __init__(self, runtime: Runtime, world: SharedWorld):
        self._runtime, self.world = runtime, world

    async def get(self, path: str = "") -> Versioned:
        from .journal import command_key

        runtime = self._runtime
        runtime._active()
        canonical = canonical_path(path)
        seq = runtime._next_seq()
        journal = runtime.journal
        if journal is not None:
            key = command_key("shared_read", {"world": self.world.name, "path": canonical})
            _, completed = journal.check(seq, "shared_read", key)
            if completed is not None:
                runtime._applied = seq
                return Versioned(**completed.payload["read"])
            runtime._go_live()
        read = self.world.read(canonical)
        if journal is not None:
            # Queued without waiting: ordered writes make it durable before any later effect.
            journal.enqueue(JournalEntry(seq=seq, phase="completed", kind="shared_read", key=key,
                                         payload={"read": read.to_json()}))
        runtime._applied = seq
        return read

    async def patch(self, ops: Iterable[Mapping], *, based_on: Iterable[Versioned | Mapping] = ()) -> Committed:
        """Commit a patch; raises StaleRead or PatchConflict (recorded, so a replay raises them too)."""
        from .journal import command_key

        runtime = self._runtime
        runtime._active()
        ops, basis = normalize_ops(ops), normalize_basis(based_on)
        seq = runtime._next_seq()
        journal = runtime.journal
        key = command_key("shared_write", {"world": self.world.name, "ops": ops, "based_on": basis})
        target = None
        if journal is not None:
            _, completed = journal.check(seq, "shared_write", key)
            if completed is not None:
                runtime._applied = seq
                return raise_for(completed.payload["result"])
            runtime._go_live()
            await journal.flush()
            target = _JournalTarget(journal.session_id, journal.run_id, journal.epoch, seq, key)
        run = runtime.run
        result = await self.world._commit(
            ops, basis, f"{run.session_id}:{run.id}:{seq}", {"session": run.session_id, "run": run.id, "seq": seq},
            target,
        )
        runtime._applied = seq
        return raise_for(result)


class ToolShared:
    """Shared access injected into a tool call (``shared: Hidden[ToolShared]``).

    The tool call itself is journaled, so its reads are not. Writes get op IDs
    derived from the call's idempotency key, so re-executing an idempotent tool
    after a crash does not apply its writes twice, as long as it issues them in
    the same order.
    """

    def __init__(self, world: SharedWorld, op_prefix: str, author: dict):
        self.world, self._prefix, self._author, self._count = world, op_prefix, author, 0

    async def get(self, path: str = "") -> Versioned:
        return self.world.read(path)

    async def patch(self, ops: Iterable[Mapping], *, based_on: Iterable[Versioned | Mapping] = ()) -> Committed:
        self._count += 1
        result = await self.world._commit(
            normalize_ops(ops), normalize_basis(based_on), f"{self._prefix}:{self._count}", self._author, None
        )
        return raise_for(result)

