"""Determinism APIs, configuration checks, inspection replay and forks."""

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime

import pytest
from test_durable import (
    REACT_SCRIPT,
    CountingMemory,
    CrashingMemory,
    World,
    build,
    call,
    crashes,
    make_tools,
)

from mickro_agent import (
    Agent,
    ConfigurationChanged,
    Limits,
    MemoryStore,
    ModelResponse,
    NonDeterminismError,
    PlanThenAct,
    Run,
    Runtime,
    Tools,
    VersionUnsupported,
)
from mickro_agent.journal import MemoryStoreData
from mickro_agent.providers import ScriptedProvider


class Watching(ScriptedProvider):
    """Records the last user message of every request it actually receives."""

    def __init__(self, responses, seen):
        super().__init__(responses)
        self.seen = seen

    async def create_message(self, system_prompt, messages, metadata=None, **kwargs):
        self.seen.append(next(m.content for m in reversed(messages) if m.role == "user"))
        async for chunk in super().create_message(system_prompt, messages, metadata, **kwargs):
            yield chunk


class Exploding(ScriptedProvider):
    def __init__(self):
        super().__init__([])

    async def create_message(self, *args, **kwargs):
        raise AssertionError("replay must not call the provider")
        yield  # pragma: no cover


@dataclass(frozen=True)
class Stamped:
    """A loop whose behaviour depends on time and randomness, through the runtime."""

    async def run(self, runtime: Runtime) -> None:
        stamp = f"{runtime.uuid()} {runtime.random()} {runtime.now().isoformat()}"
        runtime.say(stamp)
        await runtime.complete()
        await runtime.execute_tools()
        await runtime.complete()
        runtime.finish(stamp)


STAMP_SCRIPT = [ModelResponse(tool_calls=(call("c", "send", text="hello"),)), ModelResponse(text="done")]


def stamped_agent(store, world, seen):
    return Agent(provider=Watching(STAMP_SCRIPT, seen), tools=make_tools(world), loop=Stamped(),
                 name="worker", store=store, lease_ttl=60)


def test_side_effects_return_recorded_values_after_a_crash():
    async def reference():
        store = CountingMemory()
        await stamped_agent(store, World(), []).run("Task", session_id="s1")
        return store.written

    writes = asyncio.run(reference())
    for crash_at in range(1, writes + 1):
        for after in (False, True):
            async def scenario(crash_at=crash_at, after=after):
                world, seen, data = World(), [], MemoryStoreData()
                await crashes(stamped_agent(CrashingMemory(data, crash_at=crash_at, after=after), world, seen)
                              .run("Task", session_id="s1"), may_finish=after and crash_at == writes)
                store = MemoryStore(data)
                if store.runs(status="running"):
                    run = await stamped_agent(store, world, seen).recover("s1", force=True)
                    output = run.output
                elif store.runs():
                    output = store.runs()[0].result["output"]
                else:
                    return
                # Every model request in every attempt saw the one stamp the run finished with.
                assert seen and set(seen) == {output}
                uuid, number, moment = output.split(" ")
                assert len(uuid) == 32 and 0 <= float(number) < 1 and datetime.fromisoformat(moment).tzinfo

            asyncio.run(scenario())


def test_helpers_work_without_a_journal():
    runtime = Runtime(Run(), ScriptedProvider([]), Tools(), Limits())
    assert len(runtime.uuid()) == 32 and 0 <= runtime.random() < 1 and runtime.now().tzinfo
    assert runtime.version("change", 0, 3) == 3
    with pytest.raises(ValueError):
        runtime.version("change", 2, 1)


@dataclass(frozen=True)
class Versioned:
    """The same loop class before (gate=False) and after (gate=True) a code change."""

    gate: bool = False
    min_supported: int = 0

    async def run(self, runtime: Runtime) -> None:
        if self.gate and runtime.version("double-check", self.min_supported, 1) == 1:
            runtime.say("Double-check your work.")
        await runtime.complete()
        await runtime.execute_tools()
        await runtime.complete()
        runtime.finish(runtime.run.messages[-1].content)


def versioned_agent(store, world, seen, loop):
    return Agent(provider=Watching(STAMP_SCRIPT, seen), tools=make_tools(world), loop=loop,
                 name="worker", store=store, lease_ttl=60)


def test_version_keeps_old_journals_replayable():
    async def scenario():
        world, seen = World(), []
        # Old code crashes after its first model call is recorded.
        data = MemoryStoreData()
        old = versioned_agent(CrashingMemory(data, crash_at=3, after=True), world, seen, Versioned())
        await crashes(old.run("Task", session_id="old"))
        store = MemoryStore(data)
        # New code still recovers the old run: version() returns 0, so no extra instruction.
        unsupported = versioned_agent(store, world, seen, Versioned(gate=True, min_supported=1))
        with pytest.raises(VersionUnsupported):
            await unsupported.recover("old", force=True)
        assert store.runs(session_id="old")[0].status == "running"
        run = await versioned_agent(store, world, seen, Versioned(gate=True)).recover("old", force=True)
        assert run.status == "completed"
        assert "Double-check your work." not in [m.content for m in run.messages]
        # A new run of the new code takes the new path, and its own recovery replays it.
        data = MemoryStoreData()
        new = versioned_agent(CrashingMemory(data, crash_at=4, after=True), world, seen, Versioned(gate=True))
        await crashes(new.run("Task", session_id="new"))
        store = MemoryStore(data)
        run = await versioned_agent(store, world, seen, Versioned(gate=True, min_supported=1)).recover(
            "new", force=True
        )
        assert run.status == "completed"
        assert "Double-check your work." in [m.content for m in run.messages]

    asyncio.run(scenario())


def test_changed_configuration_is_named_and_can_be_allowed():
    async def scenario():
        world, data = World(), MemoryStoreData()
        await crashes(build("react", CrashingMemory(data, crash_at=6, after=True), world).run("T", session_id="s1"))
        store = MemoryStore(data)

        class OtherProvider(ScriptedProvider):
            pass

        changed = replace(build("react", store, world), provider=OtherProvider(REACT_SCRIPT))
        with pytest.raises(ConfigurationChanged) as error:
            await changed.recover("s1", force=True)
        assert error.value.changed == ("provider",)
        run = await changed.recover("s1", force=True, allow_fingerprint_change=True)
        assert run.status == "completed" and run.output == "All done"

    asyncio.run(scenario())


def test_replay_inspects_any_step_without_effects():
    async def scenario():
        world = World()
        store = CountingMemory()
        source = await build("react", store, world).run("Do the task", session_id="s1")
        written, effects = store.written, (world.reads, len(world.idempotent_keys), len(world.unsafe))
        inspector = replace(build("react", store, world), provider=Exploding())

        start = await inspector.replay("s1", source.id, until_seq=0)
        assert start.commands == 0 and start.reason == "until_seq"
        assert [m.role for m in start.run.messages] == ["system", "user"]

        first = await inspector.replay("s1", source.id, until_seq=1)
        assert first.commands == 1 and [c.id for c in first.pending_tool_calls] == ["a", "b"]
        # A parallel batch (commands 2 and 3) is replayed whole.
        split = await inspector.replay("s1", source.id, until_seq=2)
        assert split.commands == 1 and len(split.run.messages) == len(first.run.messages)
        batch = await inspector.replay("s1", source.id, until_seq=3)
        assert batch.commands == 3 and [m.tool_call_id for m in batch.run.messages if m.role == "tool"] == ["a", "b"]

        full = await inspector.replay("s1", source.id)
        assert full.reason == "finished" and full.run.status == "completed" and full.run.output == "All done"
        assert full.run.messages == source.messages and full.run.usage == source.usage
        assert store.written == written
        assert (world.reads, len(world.idempotent_keys), len(world.unsafe)) == effects

        changed = replace(inspector, loop=PlanThenAct())
        with pytest.raises(NonDeterminismError):
            await changed.replay("s1", source.id)

    asyncio.run(scenario())


def test_replay_of_an_interrupted_run_stops_at_the_end_of_the_journal():
    async def scenario():
        world, data = World(), MemoryStoreData()
        await crashes(build("react", CrashingMemory(data, crash_at=6, after=True), world).run("T", session_id="s1"))
        store = MemoryStore(data)
        record = store.runs()[0]
        replayed = await replace(build("react", store, world), provider=Exploding()).replay("s1", record.run_id)
        assert replayed.reason == "end_of_journal" and replayed.run.status == "running"
        assert replayed.commands == 3

    asyncio.run(scenario())


def test_fork_reuses_the_prefix_and_diverges_after_it():
    async def scenario():
        world = World()
        store = MemoryStore()
        source = await build("react", store, world).run("Do the task", session_id="s1")
        effects = (world.reads, len(world.idempotent_keys), len(world.unsafe))
        # After the first tool batch (commands 1-3), try a different model and system prompt.
        other = replace(
            build("react", store, world),
            provider=ScriptedProvider([ModelResponse(text="unused"), ModelResponse(text="Forked answer")]),
            system_prompt="Be brief.",
        )
        fork = await other.fork("s1", source.id, at_seq=3, new_session_id="fork-1")
        assert fork.status == "completed" and fork.output == "Forked answer"
        assert fork.session_id == "fork-1" and fork.messages[0].content == "Be brief."
        assert (world.reads, len(world.idempotent_keys), len(world.unsafe)) == effects  # prefix not re-executed
        assert [m.tool_call_id for m in fork.messages if m.role == "tool"] == ["a", "b"]
        assert store.runs(session_id="s1")[0].result["output"] == "All done"
        forked = store.runs(session_id="fork-1")[0]
        assert forked.start["forked_from"] == {"session_id": "s1", "run_id": source.id, "at_seq": 3}
        assert any(event.kind == "fork_live" for event in fork.events)
        with pytest.raises(ValueError, match="already exists"):
            await other.fork("s1", source.id, at_seq=3, new_session_id="fork-1")
        # A fork can be inspected like any run.
        replayed = await replace(other, provider=Exploding()).replay("fork-1", fork.id)
        assert replayed.run.output == "Forked answer"

    asyncio.run(scenario())


def test_a_fork_is_itself_recoverable():
    async def scenario():
        world = World()
        data = MemoryStoreData()
        source = await build("react", MemoryStore(data), world).run("Do the task", session_id="s1")
        script = [ModelResponse(text="unused"), ModelResponse(tool_calls=(call("d", "send", text="forked"),)),
                  ModelResponse(text="Forked answer")]

        def fork_agent(store):
            return replace(build("react", store, world), provider=ScriptedProvider(script), system_prompt="Brief.")

        # Writes: create_run (with prefix), model(s), model(c) -> crash after the second model call starts.
        await crashes(fork_agent(CrashingMemory(data, crash_at=2, after=True)).fork(
            "s1", source.id, at_seq=3, new_session_id="fork-1"))
        run = await fork_agent(MemoryStore(data)).recover("fork-1", force=True)
        assert run.status == "completed" and run.output == "Forked answer"
        assert world.unsafe == ["hello", "forked"]

    asyncio.run(scenario())


def test_fork_at_zero_reruns_from_the_input():
    async def scenario():
        world = World()
        store = MemoryStore()
        source = await build("react", store, world).run("Do the task", session_id="s1")
        fork = await build("react", store, world).fork("s1", source.id, at_seq=0)
        assert fork.output == "All done" and world.unsafe == ["hello", "hello"]

    asyncio.run(scenario())
