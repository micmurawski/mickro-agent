import asyncio
import gc
import weakref

import pytest

from mickro_agent import END, Agent, Flow, ModelResponse, Runtime, node
from mickro_agent.providers import ScriptedProvider


def run(flow, prompt="Task", **kwargs):
    return asyncio.run(Agent(provider=ScriptedProvider([]), loop=flow, **kwargs).run(prompt))


def test_default_chain_supports_sync_async_and_explicit_end():
    @node
    def first(runtime):
        runtime.run.state["visited"] = ["first"]

    @node
    async def second(runtime):
        await asyncio.sleep(0)
        runtime.run.state["visited"].append("second")
        return "default"

    @node
    def finish(runtime):
        runtime.finish(",".join(runtime.run.state["visited"]))

    first >> second >> finish >> END
    result = run(Flow(start=first))
    assert result.status == "completed" and result.output == "first,second"


def test_named_branches_and_cycle_use_run_state():
    @node
    def route(runtime):
        count = runtime.run.state.get("count", 0)
        return "again" if count < 2 else "done"

    @node
    def increment(runtime):
        runtime.run.state["count"] = runtime.run.state.get("count", 0) + 1

    @node
    def finish(runtime):
        runtime.finish(str(runtime.run.state["count"]))

    route - "again" >> increment
    increment >> route
    route - "done" >> finish
    # A leaf ends on its default action without an explicit >> END.
    flow = Flow(route)
    assert run(flow).output == "2"
    assert run(flow).output == "2"


def test_default_is_not_a_fallback_for_unrecognized_actions():
    @node
    def start(runtime):
        return "typo"

    @node
    def finish(runtime):
        runtime.finish("Must not be reached")

    start >> finish
    result = run(Flow(start))
    assert result.status == "failed" and "No transition" in result.error


def test_flow_snapshots_edges_and_metadata():
    @node(metadata={"phase": {"name": "original"}})
    def start(runtime):
        return "later"

    @node
    def finish(runtime):
        runtime.finish("Changed")

    original = Flow(start)
    start - "later" >> finish
    start.metadata["phase"]["name"] = "modified"
    changed = Flow(start)
    before, after = run(original), run(changed)
    assert before.status == "failed" and after.output == "Changed"
    step = next(event for event in before.events if event.kind == "step")
    assert step.data["metadata"] == {"phase": {"name": "original"}}


def test_duplicate_edges_names_and_bad_targets_fail_during_definition():
    start = node(lambda runtime: None, name="start")
    finish = node(lambda runtime: runtime.finish("Done"), name="finish")
    start >> finish
    with pytest.raises(ValueError, match="Duplicate transition"):
        start >> END
    with pytest.raises(TypeError, match="target"):
        finish >> "typo"
    with pytest.raises(TypeError, match="action"):
        finish - 123
    with pytest.raises(ValueError, match="reserved"):
        node(lambda runtime: None, name=END)

    duplicate = node(lambda runtime: None, name="start")
    finish >> duplicate
    with pytest.raises(ValueError, match="Duplicate node name"):
        Flow(start)


def test_bound_method_dsl_is_per_instance_and_does_not_leak_owner():
    class LoopDefinition:
        def __init__(self, label):
            self.label = label

        @node
        async def call_llm(self, runtime: Runtime):
            response = await runtime.complete()
            runtime.run.state["answer"] = response.text
            return "answer"

        @node
        def finish(self, runtime: Runtime):
            runtime.finish(self.label + runtime.run.state["answer"])

    first, second = LoopDefinition("first:"), LoopDefinition("second:")
    assert first.call_llm is first.call_llm
    assert first.call_llm is not second.call_llm
    first.call_llm - "answer" >> first.finish
    second.call_llm - "answer" >> second.finish
    first_flow, second_flow = Flow(first.call_llm), Flow(second.call_llm)
    provider = ScriptedProvider([ModelResponse(text="result")])

    async def both(flow_a, flow_b):
        return await asyncio.gather(
            Agent(provider=provider, loop=flow_a).run("A"),
            Agent(provider=provider, loop=flow_b).run("B"),
        )

    results = asyncio.run(both(first_flow, second_flow))
    assert [result.output for result in results] == ["first:result", "second:result"]
    reference = weakref.ref(first)
    del first, first_flow
    gc.collect()
    assert reference() is None


def test_one_compiled_flow_can_run_concurrently():
    @node
    async def remember(runtime):
        runtime.run.state["task"] = runtime.run.messages[-1].content
        await asyncio.sleep(0)

    @node
    def finish(runtime):
        runtime.finish(runtime.run.state["task"])

    remember >> finish
    agent = Agent(provider=ScriptedProvider([]), loop=Flow(remember))

    async def both():
        return await asyncio.gather(agent.run("A"), agent.run("B"))

    assert [result.output for result in asyncio.run(both())] == ["A", "B"]


def test_non_action_return_fails_with_clear_error():
    @node
    def invalid(runtime):
        return {"answer": 42}

    result = run(Flow(invalid))
    assert result.status == "failed" and "string action or None" in result.error
