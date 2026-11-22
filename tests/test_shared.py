"""Shared state: conformance suite for every backend, and journaled use from runs."""

import asyncio
import sqlite3
from dataclasses import dataclass, replace

import pytest
from test_durable import CountingMemory, CrashingMemory, World, crashes, make_tools

from mickro_agent import (
    Agent,
    Hidden,
    InvalidPatch,
    MemoryStore,
    ModelResponse,
    PatchConflict,
    RunOwnershipLost,
    Runtime,
    SharedWorld,
    SqliteStore,
    StaleRead,
    ToolCall,
    Tools,
    ToolShared,
    tool,
)
from mickro_agent.journal import MemoryStoreData
from mickro_agent.providers import ScriptedProvider
from mickro_agent.shared import format_path, parse_path


@pytest.fixture(params=["memory", "sqlite"])
def reopen(request, tmp_path):
    """Returns open(): a new store over the same data, like a new process."""
    data, path, stores = MemoryStoreData(), tmp_path / "durable.db", []

    def open_store():
        store = MemoryStore(data) if request.param == "memory" else SqliteStore(path)
        stores.append(store)
        return store

    yield open_store
    for store in stores:
        if isinstance(store, SqliteStore):
            store.close()


def run(coroutine):
    return asyncio.run(coroutine)


# Paths -------------------------------------------------------------------------
def test_paths_are_json_pointers_with_dotted_shorthand():
    assert parse_path("") == ()
    assert parse_path("/a~1b/c~0d") == ("a/b", "c~d")
    assert format_path(("a/b", "c~d")) == "/a~1b/c~0d"
    assert parse_path("tasks.t42.status") == ("tasks", "t42", "status")
    with pytest.raises(InvalidPatch):
        parse_path("a..b")
    with pytest.raises(InvalidPatch):
        parse_path("a.b~c")


# Patch semantics ------------------------------------------------------------------
def test_patch_operations_on_objects_and_arrays(reopen):
    async def scenario():
        world = SharedWorld(reopen())
        await world.patch([{"op": "add", "path": "/tasks", "value": {"t1": {"status": "open"}}},
                           {"op": "add", "path": "/log", "value": ["a"]}])
        await world.patch([{"op": "add", "path": "/log/-", "value": "c"},
                           {"op": "add", "path": "/log/1", "value": "b"},
                           {"op": "replace", "path": "tasks.t1.status", "value": "done"},
                           {"op": "test", "path": "/tasks/t1/status", "value": "done"}])
        assert world.read("").value == {"tasks": {"t1": {"status": "done"}}, "log": ["a", "b", "c"]}
        await world.patch([{"op": "remove", "path": "/log/0"}, {"op": "remove", "path": "/tasks/t1"}])
        assert world.read("").value == {"tasks": {}, "log": ["b", "c"]}
        missing = world.read("/tasks/t9")
        assert not missing.exists and missing.value is None

    run(scenario())


def test_rejected_patches_change_nothing_and_take_no_offset(reopen):
    async def scenario():
        world = SharedWorld(reopen())
        await world.patch([{"op": "add", "path": "/a", "value": 1}])
        for ops in (
            [{"op": "replace", "path": "/a", "value": 2}, {"op": "test", "path": "/a", "value": 1}],  # test fails
            [{"op": "add", "path": "/b", "value": 1}, {"op": "replace", "path": "/missing", "value": 1}],
            [{"op": "add", "path": "/x/y", "value": 1}],  # parent missing
            [{"op": "test", "path": "/a", "value": True}],  # JSON equality: 1 is not true
            [{"op": "remove", "path": ""}],
        ):
            with pytest.raises(PatchConflict):
                await world.patch(ops)
        for ops in ([], [{"op": "move", "path": "/a"}], [{"op": "add", "path": "/a"}],
                    [{"op": "add", "path": "/a", "value": {1, 2}}], [{"op": "add", "path": "/a", "value": 1}] * 101):
            with pytest.raises(InvalidPatch):
                await world.patch(ops)
        assert world.offset == 1 and world.read("").value == {"a": 1}
        assert [entry.offset for entry in world.log()] == [1]

    run(scenario())


# Versions ------------------------------------------------------------------------
def test_versions_detect_exactly_the_writes_that_affect_a_read(reopen):
    async def scenario():
        world = SharedWorld(reopen())
        await world.patch([{"op": "add", "path": "/a", "value": {"b": {"x": 1}, "c": 1}},
                           {"op": "add", "path": "/list", "value": [1, 2]}])

        async def stale_after(read_path, ops):
            read = world.read(read_path)
            await world.patch(ops)
            try:
                await world.patch([{"op": "test", "path": "", "value": world.read("").value}], based_on=[read])
            except StaleRead as error:
                assert error.paths == (read_path,)
                return True
            return False

        assert not await stale_after("/a/b", [{"op": "replace", "path": "/a/c", "value": 2}])  # sibling
        assert await stale_after("/a/b", [{"op": "replace", "path": "/a/b/x", "value": 2}])  # child
        assert await stale_after("/a", [{"op": "add", "path": "/a/b/y", "value": 1}])  # descendant
        assert await stale_after("/a/b/x", [{"op": "replace", "path": "/a", "value": {"b": {"x": 2}}}])  # ancestor
        assert await stale_after("/a/z", [{"op": "add", "path": "/a/z", "value": 1}])  # created after read
        assert await stale_after("/new/x", [{"op": "add", "path": "/new", "value": {"x": 1}}])  # created by ancestor
        assert await stale_after("/list/0", [{"op": "add", "path": "/list/-", "value": 3}])  # arrays are one unit
        assert not await stale_after("/a", [{"op": "test", "path": "/a/b/x", "value": 2}])  # test writes nothing

    run(scenario())


def test_op_ids_make_retries_idempotent(reopen):
    async def scenario():
        world = SharedWorld(reopen())
        first = await world.patch([{"op": "add", "path": "/n", "value": 1}], op_id="once")
        again = await world.patch([{"op": "add", "path": "/n", "value": 99}], op_id="once")
        assert first == again and world.read("/n").value == 1 and world.offset == 1

    run(scenario())


# Log, snapshots and time travel ------------------------------------------------------
def test_time_travel_and_restart_match_every_offset(reopen):
    async def scenario():
        world = SharedWorld(reopen(), snapshot_every=3)
        states = [world.state]
        await world.patch([{"op": "add", "path": "/items", "value": []}])
        states.append(world.state)
        for index in range(10):
            await world.patch([{"op": "add", "path": "/items/-", "value": index},
                               {"op": "add", "path": f"/k{index}", "value": {"v": index}}])
            states.append(world.state)
        for offset, state in enumerate(states):
            travelled = world.at(offset)
            assert travelled.offset == offset and travelled.value == state.value
            assert all(travelled.version(f"/k{i}/v") == state.version(f"/k{i}/v") for i in range(10))
        restarted = SharedWorld(reopen(), snapshot_every=3)  # loads the latest snapshot plus the log tail
        assert restarted.state == world.state
        with pytest.raises(ValueError):
            world.at(len(states))

    run(scenario())


def test_concurrent_compare_and_swap_through_group_commit(reopen):
    async def scenario():
        world = SharedWorld(reopen())
        await world.patch([{"op": "add", "path": "/n", "value": 0}])
        retries = 0

        async def increment():
            nonlocal retries
            while True:
                read = world.read("/n")
                try:
                    return await world.patch([{"op": "replace", "path": "/n", "value": read.value + 1}],
                                             based_on=[read])
                except StaleRead:
                    retries += 1

        await asyncio.gather(*(increment() for _ in range(50)))
        assert world.read("/n").value == 50
        assert [entry.offset for entry in world.log()] == list(range(1, 52))
        assert retries > 0  # writers really raced

    run(scenario())


def test_sqlite_version_one_files_are_upgraded(tmp_path):
    path = tmp_path / "old.db"
    SqliteStore(path).close()
    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE shared_log")
    connection.execute("PRAGMA user_version=1")
    connection.commit()
    connection.close()
    store = SqliteStore(path)

    async def scenario():
        world = SharedWorld(store)
        await world.patch([{"op": "add", "path": "/ok", "value": True}])
        assert world.read("/ok").value is True

    run(scenario())
    assert store._read("PRAGMA user_version")[0][0] == 2
    store.close()


def test_world_names_and_agent_stores_are_checked():
    store = MemoryStore()
    world = SharedWorld(store)
    with pytest.raises(ValueError, match="already open"):
        SharedWorld(store)
    with pytest.raises(ValueError, match="own store"):
        Agent(provider=ScriptedProvider([]), store=MemoryStore(), shared=world)
    runtime = Runtime.__new__(Runtime)
    runtime._world = None
    with pytest.raises(RuntimeError, match="no shared world"):
        runtime.shared  # noqa: B018


# Runs ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Counter:
    """Read a counter, increment it based on that read, then ask the model."""

    async def run(self, runtime: Runtime) -> None:
        read = await runtime.shared.get("/counter")
        try:
            await runtime.shared.patch([{"op": "replace", "path": "/counter", "value": read.value + 1}],
                                       based_on=[read])
            outcome = f"set {read.value + 1}"
        except StaleRead:
            outcome = "stale"
        runtime.say(outcome)
        await runtime.complete()
        runtime.finish(outcome)


def counter_agent(store, world):
    return Agent(provider=ScriptedProvider([ModelResponse(text="ok")]), loop=Counter(), name="counter",
                 store=store, shared=world, lease_ttl=60)


async def seeded(store):
    world = SharedWorld(store)
    if not world.read("/counter").exists:
        await world.patch([{"op": "add", "path": "/counter", "value": 0}])
    return world


def test_shared_writes_from_a_run_happen_exactly_once_across_crashes():
    async def reference():
        store = CountingMemory()
        world = await seeded(store)
        before = store.written
        await counter_agent(store, world).run("Go", session_id="s1")
        return store.written - before + 1

    writes = run(reference())
    for crash_at in range(2, writes + 1):
        for after in (False, True):
            async def scenario(crash_at=crash_at, after=after):
                data = MemoryStoreData()
                crashing = CrashingMemory(data, crash_at=crash_at, after=after)
                world = await seeded(crashing)
                await crashes(counter_agent(crashing, world).run("Go", session_id="s1"),
                              may_finish=after and crash_at == writes)
                store = MemoryStore(data)
                world = SharedWorld(store)
                if not store.runs():
                    assert world.read("/counter").value == 0  # The run never started.
                    return
                if store.runs(status="running"):
                    recovered = await counter_agent(store, world).recover("s1", force=True)
                    assert recovered.status == "completed" and recovered.output == "set 1"
                # Exactly one increment, by this run, however the crash fell.
                assert store.runs()[0].status == "completed" and world.read("/counter").value == 1
                authors = [entry.author for entry in world.log() if entry.author]
                assert len(authors) == 1 and authors[0]["session"] == "s1"

            run(scenario())


def test_a_stale_read_is_journaled_and_replays():
    async def scenario():
        data = MemoryStoreData()
        crashing = CrashingMemory(data, crash_at=3, after=True)  # seed, create_run, read: then crash
        world = await seeded(crashing)
        await crashes(counter_agent(crashing, world).run("Go", session_id="s1"))
        store = MemoryStore(data)
        world = SharedWorld(store)
        assert world.read("/counter").value == 0
        await world.patch([{"op": "replace", "path": "/counter", "value": 10}])  # another writer, meanwhile
        recovered = await counter_agent(store, world).recover("s1", force=True)
        assert recovered.output == "stale" and world.read("/counter").value == 10
        replayed = await counter_agent(store, world).replay("s1", recovered.id)
        assert replayed.run.output == "stale"

    run(scenario())


def test_replay_returns_what_the_run_read_not_the_current_value():
    async def scenario():
        store = MemoryStore()
        world = await seeded(store)
        source = await counter_agent(store, world).run("Go", session_id="s1")
        await world.patch([{"op": "replace", "path": "/counter", "value": 500}])
        replayed = await counter_agent(store, world).replay("s1", source.id)
        assert replayed.run.output == "set 1" and world.read("/counter").value == 500
        read_offset = next(e for e in store.journal("s1", source.id) if e.kind == "shared_read").payload["read"]
        assert world.at(read_offset["offset"]).read("/counter").value == 0  # the world as the run saw it

    run(scenario())


def test_a_stale_owner_cannot_write_shared_state():
    async def scenario():
        store = MemoryStore()
        world = await seeded(store)
        entered, release = asyncio.Event(), asyncio.Event()

        class Paused(Counter):
            async def run(self, runtime):
                read = await runtime.shared.get("/counter")
                await runtime.journal.flush()  # Nothing queued: the commit itself must refuse.
                entered.set()
                await release.wait()
                await runtime.shared.patch([{"op": "replace", "path": "/counter", "value": 1}], based_on=[read])

        task = asyncio.create_task(replace(counter_agent(store, world), loop=Paused()).run("Go", session_id="s1"))
        await entered.wait()
        record = store.runs(status="running")[0]
        await store.claim_run("s1", record.run_id, ttl=60, force=True)  # another process took over
        release.set()
        with pytest.raises(RunOwnershipLost):
            await task
        assert world.read("/counter").value == 0 and world.offset == 1

    run(scenario())


def test_idempotent_tools_write_once_when_re_executed():
    async def scenario():
        world_effects = World()

        @tool(effect="idempotent")
        async def record(note: str, shared: Hidden[ToolShared]) -> str:
            """Append a note to the shared log."""
            await shared.patch([{"op": "add", "path": "/notes/-", "value": note}])
            world_effects.reads += 1
            return "noted"

        def agent(store, world):
            script = [ModelResponse(tool_calls=(ToolCall(id="n", name="record", arguments={"note": "hi"}),)),
                      ModelResponse(text="done")]
            return Agent(provider=ScriptedProvider(script), tools=Tools([record]) | make_tools(world_effects),
                         name="noter", store=store, shared=world, lease_ttl=60)

        data = MemoryStoreData()
        # seed, create_run, model(s), model(c), tool(s), shared write -> crash before the tool's result
        crashing = CrashingMemory(data, crash_at=6, after=True)
        world = SharedWorld(crashing)
        await world.patch([{"op": "add", "path": "/notes", "value": []}])
        await crashes(agent(crashing, world).run("Note it", session_id="s1"))
        assert world.read("/notes").value == ["hi"]
        store = MemoryStore(data)
        world = SharedWorld(store)
        recovered = await agent(store, world).recover("s1", force=True)
        assert recovered.status == "completed"
        assert world_effects.reads == 2  # the tool ran again ...
        assert world.read("/notes").value == ["hi"]  # ... but its write was deduplicated

    run(scenario())
