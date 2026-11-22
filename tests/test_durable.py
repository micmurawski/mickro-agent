"""Durable runs: crash at every journal write, then recover and compare."""

import asyncio
import json
from dataclasses import replace

import pytest

from mickro_agent import (
    Agent,
    Hidden,
    Limits,
    MemoryStore,
    ModelResponse,
    NonDeterminismError,
    PlanThenAct,
    ReAct,
    ReActWithSummary,
    RunLeaseHeld,
    RunOwnershipLost,
    SqliteStore,
    ToolCall,
    Tools,
    Worker,
    tool,
)
from mickro_agent.journal import INTERRUPTED_CONTENT, MemoryStoreData
from mickro_agent.providers import ScriptedProvider


class Crash(BaseException):
    """Simulated process death: nothing after it reaches the store."""


async def crashes(awaitable, *, may_finish=False):
    """Await until the simulated crash; a TaskGroup may wrap it in a group."""
    try:
        await awaitable
    except Crash:
        return
    except BaseExceptionGroup as group:
        assert group.subgroup(Crash) is not None, group
        return
    assert may_finish, "expected a simulated crash"


class _Crashing:
    """Writes the first ``crash_at - 1`` operations (or ``crash_at`` with after=True), then dies."""

    def __init__(self, *args, crash_at: int, after: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = crash_at if after else crash_at - 1
        self.written = 0
        self.dead = False

    def _write_batch(self, ops):
        if self.dead:
            raise Crash()
        allowed = self.limit - self.written
        if len(ops) <= allowed:
            self.written += len(ops)
            return super()._write_batch(ops)
        super()._write_batch(ops[:allowed])
        self.written += allowed
        self.dead = True
        raise Crash()


class CrashingMemory(_Crashing, MemoryStore):
    pass


class CrashingSqlite(_Crashing, SqliteStore):
    pass


class Counting:
    """Counts committed operations of an uninterrupted reference run."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.written = 0

    def _write_batch(self, ops):
        self.written += len(ops)
        return super()._write_batch(ops)


class CountingMemory(Counting, MemoryStore):
    pass


class World:
    """External state that survives crashes: what tools really did."""

    def __init__(self):
        self.unsafe = []
        self.idempotent_keys = []
        self.reads = 0


def make_tools(world: World) -> Tools:
    @tool(effect="read")
    def lookup(key: str) -> str:
        """Read a record."""
        world.reads += 1
        return f"record {key}"

    @tool(effect="idempotent")
    def upsert(key: str, idempotency_key: Hidden[str]) -> str:
        """Write a record with an idempotency key."""
        world.idempotent_keys.append(idempotency_key)
        return f"stored {key}"

    @tool
    def send(text: str) -> str:
        """Send a message (unsafe: never repeated)."""
        world.unsafe.append(text)
        return "sent"

    return Tools([lookup, upsert, send])


def call(id, name, **arguments):
    return ToolCall(id=id, name=name, arguments=arguments)


REACT_SCRIPT = [
    ModelResponse(tool_calls=(call("a", "lookup", key="x"), call("b", "upsert", key="y")), usage={"input_tokens": 3}),
    ModelResponse(tool_calls=(call("c", "send", text="hello"),), usage={"input_tokens": 5}),
    ModelResponse(text="All done", usage={"input_tokens": 7}),
]


async def count_tokens(messages):
    return sum(len(message.content) + 1 for message in messages)


def scenarios():
    return {
        "react": lambda: {"loop": ReAct(), "script": REACT_SCRIPT},
        "react_parallel": lambda: {
            "loop": ReAct(tool_concurrency=3),
            "script": [
                ModelResponse(tool_calls=(call("a", "lookup", key="x"), call("b", "upsert", key="y"),
                                          call("c", "send", text="hello"))),
                ModelResponse(text="All done"),
            ],
        },
        "plan_then_act": lambda: {
            "loop": PlanThenAct(),
            "script": [ModelResponse(text="Plan: lookup, send"), *REACT_SCRIPT],
        },
        "react_with_summary": lambda: {
            "loop": ReActWithSummary(context_window=400, token_counter=count_tokens, max_output_tokens=10,
                                     threshold_percent=40, keep_recent_messages=2),
            "script": [
                ModelResponse(text="Looking " * 20, tool_calls=(call("a", "lookup", key="x"),)),
                ModelResponse(text="Writing " * 20, tool_calls=(call("b", "upsert", key="y"),)),
                ModelResponse(text="Summary of earlier work"),
                ModelResponse(tool_calls=(call("c", "send", text="hello"),)),
                ModelResponse(text="Second summary"),
                ModelResponse(text="All done"),
            ],
        },
    }


def build(name, store, world):
    spec = scenarios()[name]()
    return Agent(provider=ScriptedProvider(spec["script"]), tools=make_tools(world), loop=spec["loop"],
                 name="worker", store=store, lease_ttl=60)


def reference(name):
    async def scenario():
        world = World()
        store = CountingMemory()
        run = await build(name, store, world).run("Do the task", session_id="s1")
        assert run.status == "completed", run.error
        return run, store.written, world

    return asyncio.run(scenario())


def without_results(messages, call_ids):
    return [m for m in messages if not (m.role == "tool" and m.tool_call_id in call_ids)]


@pytest.mark.parametrize("name", sorted(scenarios()))
def test_crash_at_every_write_then_recover(name):
    expected, writes, _ = reference(name)
    if name == "react_with_summary":
        assert any(event.kind == "context_compaction" for event in expected.events)
    for crash_at in range(1, writes + 1):
        for after in (False, True):
            async def scenario(crash_at=crash_at, after=after):
                world = World()
                data = MemoryStoreData()
                crashing = CrashingMemory(data, crash_at=crash_at, after=after)
                # Crashing after the very last write is indistinguishable from finishing.
                await crashes(build(name, crashing, world).run("Do the task", session_id="s1"),
                              may_finish=after and crash_at == writes)
                store = MemoryStore(data)  # A new process over the surviving data.
                record = store.runs(session_id="s1")
                if not record:
                    assert world.unsafe == [] and world.idempotent_keys == []
                    return
                if record[0].status == "running":
                    with pytest.raises(RunLeaseHeld):
                        await build(name, store, world).recover("s1")
                    run = await build(name, store, world).recover("s1", force=True)
                    assert run.id == record[0].run_id
                    messages, output, status = run.messages[1:], run.output, run.status
                    assert run.usage == expected.usage
                else:
                    session = store.load("s1")
                    messages, output, status = list(session.messages), store.runs()[0].result["output"], "completed"
                assert status == "completed" and output == expected.output
                synthetic = [m for m in messages if m.role == "tool" and m.content == INTERRUPTED_CONTENT]
                ids = {m.tool_call_id for m in synthetic}
                assert without_results(messages, ids) == without_results(expected.messages[1:], ids)
                # The unsafe effect happened at most once; without a recorded result it was not repeated.
                assert len(world.unsafe) <= 1
                assert world.unsafe == [] or synthetic or len(world.unsafe) == 1
                if synthetic:
                    assert [m.tool_call_id for m in synthetic] == ["c"]
                # Idempotent retries reuse one key.
                assert len(set(world.idempotent_keys)) <= 1
                assert store.runs(status="running") == []
                assert store.load("s1").status == "completed"

            asyncio.run(scenario())


def test_recovery_replays_instead_of_repeating_effects(tmp_path):
    """SQLite: crash after the unsafe tool's result was recorded; nothing runs twice."""

    async def scenario():
        world = World()
        path = tmp_path / "durable.db"
        # Writes: create_run, model(s, c), tools a,b (s), a(c), b(c), model(s, c), send(s), send(c) -> crash after 10.
        crashing = CrashingSqlite(path, crash_at=10, after=True)
        await crashes(build("react", crashing, world).run("Do the task", session_id="s1"))
        crashing.close()
        assert world.unsafe == ["hello"] and world.reads == 1 and len(world.idempotent_keys) == 1
        store = SqliteStore(path)
        run = await build("react", store, world).recover("s1", force=True)
        assert run.status == "completed" and run.output == "All done"
        assert world.unsafe == ["hello"] and world.reads == 1 and len(world.idempotent_keys) == 1
        assert store.runs()[0].status == "completed" and store.runs()[0].epoch == 2
        # The completed session continues as a normal conversation.
        follow = replace(build("react", store, world), provider=ScriptedProvider([ModelResponse(text="Again")]))
        again = await follow.run("And now?", session_id="s1")
        assert again.output == "Again" and [m.content for m in again.messages if m.role == "user"][-1] == "And now?"
        store.close()

    asyncio.run(scenario())


def test_inflight_tools_follow_their_effect_kind():
    """Crash while all three tools of a batch are scheduled but none has a result."""

    async def scenario():
        world = World()
        data = MemoryStoreData()
        # create_run, model(s), model(c), batch scheduled (one append) -> crash before the first result.
        crashing = CrashingMemory(data, crash_at=5, after=False)
        agent = build("react_parallel", crashing, world)
        await crashes(agent.run("Do the task", session_id="s1"))
        executed = (world.reads, len(world.idempotent_keys), len(world.unsafe))
        store = MemoryStore(data)
        run = await build("react_parallel", store, world).recover("s1", force=True)
        tool_messages = {m.tool_call_id: m for m in run.messages if m.role == "tool"}
        assert run.status == "completed"
        assert world.reads == executed[0] + 1  # read: executed again
        assert len(world.idempotent_keys) == executed[1] + 1 and len(set(world.idempotent_keys)) == 1
        assert len(world.unsafe) == executed[2]  # unsafe: never repeated
        assert tool_messages["c"].is_error and tool_messages["c"].content == INTERRUPTED_CONTENT
        assert not tool_messages["a"].is_error and not tool_messages["b"].is_error

    asyncio.run(scenario())


def test_stale_owner_stops_without_finishing():
    async def scenario():
        world = World()
        store = MemoryStore()
        entered, release = asyncio.Event(), asyncio.Event()

        class Slow(ScriptedProvider):
            async def create_message(self, *args, **kwargs):
                entered.set()
                await release.wait()
                async for chunk in super().create_message(*args, **kwargs):
                    yield chunk

        first = replace(build("react", store, world), provider=Slow(REACT_SCRIPT))
        task = asyncio.create_task(first.run("Do the task", session_id="s1"))
        await entered.wait()
        # A second owner (another process) takes over while the first is mid-call.
        other = MemoryStore(store.data)
        record = other.runs(status="running")[0]
        with pytest.raises(RunLeaseHeld):
            await other.claim_run("s1", record.run_id, ttl=60)
        assert await other.claim_run("s1", record.run_id, ttl=60, force=True) == 2
        release.set()
        with pytest.raises(RunOwnershipLost):
            await task
        assert store.runs()[0].status == "running" and store.runs()[0].epoch == 2
        assert world.reads == 0

    asyncio.run(scenario())


def test_divergent_code_raises_and_keeps_run_recoverable():
    async def scenario():
        world = World()
        data = MemoryStoreData()
        crashing = CrashingMemory(data, crash_at=4, after=True)
        await crashes(build("plan_then_act", crashing, world).run("Do the task", session_id="s1"))
        store = MemoryStore(data)
        changed = replace(build("plan_then_act", store, world), loop=PlanThenAct(planning_prompt="Plan differently"))
        with pytest.raises(NonDeterminismError):
            await changed.recover("s1", force=True)
        assert store.runs()[0].status == "running"
        run = await build("plan_then_act", store, world).recover("s1", force=True)
        assert run.status == "completed" and run.output == "All done"

    asyncio.run(scenario())


def test_interrupted_session_must_be_recovered_before_continuing():
    async def scenario():
        world = World()
        data = MemoryStoreData()
        await crashes(build("react", CrashingMemory(data, crash_at=3, after=True), world).run("Task", session_id="s1"))
        store = MemoryStore(data)
        with pytest.raises(ValueError, match="recover"):
            await build("react", store, world).run("Next", session_id="s1")
        with pytest.raises(ValueError, match="configuration"):
            await replace(build("react", store, world), name="other").recover("s1", force=True)
        with pytest.raises(ValueError, match="no interrupted run"):
            await build("react", store, world).recover("missing")
        with pytest.raises(ValueError, match="durable store"):
            await Agent(provider=ScriptedProvider([])).recover("s1")

    asyncio.run(scenario())


def test_recovery_continues_the_session_log(tmp_path):
    async def scenario():
        world = World()
        data = MemoryStoreData()
        agent = replace(build("react", CrashingMemory(data, crash_at=6, after=True), world), log_dir=tmp_path)
        await crashes(agent.run("Do the task", session_id="s1"))
        log = tmp_path / "mikro-agent-s1.jsonl"
        with log.open("a") as handle:
            handle.write('{"partial":')  # The process died mid-line.
        run = await replace(build("react", MemoryStore(data), world), log_dir=tmp_path).recover("s1", force=True)
        assert run.status == "completed"
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        sequences = [row["sequence"] for row in rows]
        assert len(sequences) == len(set(sequences)), "event sequences must not repeat across recovery"
        kinds = [row["kind"] for row in rows]
        assert kinds.count("run_start") == 1 and kinds.count("run_recovered") == 1 and kinds[-1] == "run_end"
        # Replayed commands are not logged twice.
        assert kinds.count("model_response") == len(REACT_SCRIPT)

    asyncio.run(scenario())


def test_two_hundred_concurrent_runs_recover_after_a_crash(tmp_path):
    async def scenario():
        world = World()
        path = tmp_path / "durable.db"
        crashing = CrashingSqlite(path, crash_at=1500, after=False)
        agent = build("react", crashing, world)
        results = await asyncio.gather(
            *(agent.run("Do the task", session_id=f"s{index}") for index in range(200)), return_exceptions=True
        )
        assert all(isinstance(result, Crash) for result in results)
        crashing.close()
        store = SqliteStore(path)
        interrupted = store.runs(status="running")
        assert interrupted, "the crash should interrupt runs"
        worker = Worker(store, [build("react", store, world)])
        recovered = await worker.recover_all(force=True)
        assert all(run.status == "completed" and run.output == "All done" for run in recovered)
        assert len(recovered) == len(interrupted)
        assert store.runs(status="running") == []
        # Each run's unsafe tool ran at most once in total.
        assert len(world.unsafe) <= len(store.runs())
        store.close()

    asyncio.run(scenario())


def test_timeout_budget_counts_active_time_only():
    async def scenario():
        world = World()
        data = MemoryStoreData()
        await crashes(build("react", CrashingMemory(data, crash_at=3, after=True), world).run("Task", session_id="s1"))
        store = MemoryStore(data)
        record = store.runs()[0]
        store.data.runs["s1", record.run_id] = record.model_copy(update={"active_seconds": 1000.0})
        run = await replace(build("react", store, world), limits=Limits(timeout=999)).recover("s1", force=True)
        assert run.status == "timeout"
        assert store.runs()[0].status == "timeout"

    asyncio.run(scenario())


def test_non_durable_agents_are_unchanged():
    async def scenario():
        world = World()
        agent = Agent(provider=ScriptedProvider(REACT_SCRIPT), tools=make_tools(world))
        run = await agent.run("Do the task")
        assert run.status == "completed" and agent.store is None
        assert agent.sessions.load(run.session_id).status == "completed"

    asyncio.run(scenario())
