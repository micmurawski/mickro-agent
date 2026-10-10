import asyncio
import json

import pytest

from mickro_agent import (
    END,
    Agent,
    Flow,
    Hidden,
    Limits,
    Message,
    ModelResponse,
    PlanThenAct,
    ReAct,
    ToolCall,
    Tools,
    tool,
)
from mickro_agent.providers import ScriptedProvider


def call(name="add", *, id="c1", **arguments):
    return ToolCall(id=id, name=name, arguments=arguments)


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_react_records_tool_turn_usage_and_events(tmp_path):
    agent = Agent(
        provider=ScriptedProvider(
            [
                ModelResponse(tool_calls=(call(a=2, b=3),), usage={"input_tokens": 10}),
                ModelResponse(text="5", usage={"input_tokens": 15, "output_tokens": 2}),
            ]
        ),
        tools=Tools([add]),
    )
    run = asyncio.run(agent.run("Add 2 and 3"))
    assert run.status == "completed"
    assert run.output == "5"
    assert run.model_calls == 2 and run.tool_calls == 1
    assert run.usage == {"input_tokens": 25, "output_tokens": 2}
    assert [message.role for message in run.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert run.messages[3].tool_call_id == "c1" and run.messages[3].content == "5"
    run.write_events(tmp_path / "events.jsonl")
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert events[-1]["data"]["status"] == "completed"
    assert {event["run_id"] for event in events} == {run.id}


def test_plan_then_act_disables_tools_for_plan():
    class Provider:
        async def create_message(self, system_prompt, messages, metadata=None, **kwargs):
            if metadata["turn"] == 1:
                assert metadata["purpose"] == "plan" and kwargs["tools"] == []
                yield {"type": "text", "text": "Look up the answer."}
            else:
                assert kwargs["tools"][0].name == "add"
                assert any(message.content == "Look up the answer." for message in messages)
                yield {"type": "text", "text": "Done"}

    run = asyncio.run(Agent(provider=Provider(), tools=Tools([add]), loop=PlanThenAct()).run("Task"))
    assert run.status == "completed" and run.state["plan"] == "Look up the answer."
    assert run.model_calls == 2


def test_injection_override_is_rejected_and_nullable_value_is_accepted():
    observed = []

    @tool
    def lookup(value: str | None, cwd: Hidden[str]):
        observed.append((value, cwd))
        return value

    assert "cwd" not in lookup.spec.parameters["properties"]
    agent = Agent(
        tools=Tools([lookup]),
        provider=ScriptedProvider(
            [
                ModelResponse(tool_calls=(call("lookup", value=None, cwd="untrusted"),)),
                ModelResponse(tool_calls=(call("lookup", id="c2", value=None),)),
                ModelResponse(text="Done"),
            ]
        ),
    )
    run = asyncio.run(agent.run("Task", context={"cwd": "trusted", "secret": "do-not-log"}))
    assert run.status == "completed"
    assert observed == [(None, "trusted")]
    assert run.messages[3].is_error
    assert run.messages[5].content == "null"
    assert "do-not-log" not in "".join(event.model_dump_json() for event in run.events)


@pytest.mark.parametrize("arguments", [{"a": "2", "b": 3}, {"a": 1}, {"a": 1, "b": 2, "extra": 3}])
def test_invalid_arguments_never_execute(arguments):
    result = asyncio.run(add.invoke(arguments, {}))
    assert result.error and "Invalid arguments" in result.error


def test_missing_injected_dependency_fails_run():
    @tool
    def lookup(cwd: Hidden[str]):
        return cwd

    run = asyncio.run(
        Agent(
            provider=ScriptedProvider(
                [
                    ModelResponse(tool_calls=(call("lookup"),)),
                ]
            ),
            tools=Tools([lookup]),
        ).run("Task")
    )
    assert run.status == "failed" and "injected context 'cwd'" in run.error


def test_tool_exception_is_observation_without_retry():
    attempts = []

    @tool
    def broken():
        attempts.append(1)
        raise RuntimeError("Something broke")

    run = asyncio.run(
        Agent(
            provider=ScriptedProvider(
                [
                    ModelResponse(tool_calls=(call("broken"),)),
                    ModelResponse(text="Reported the failure"),
                ]
            ),
            tools=Tools([broken]),
        ).run("Task")
    )
    assert run.status == "completed" and len(attempts) == 1
    assert run.messages[3].is_error and "Something broke" in run.messages[3].content


def test_unknown_tool_is_observation():
    run = asyncio.run(
        Agent(
            provider=ScriptedProvider(
                [
                    ModelResponse(tool_calls=(call("missing"),)),
                    ModelResponse(text="No tools available"),
                ]
            )
        ).run("Task")
    )
    assert run.status == "completed"
    assert run.messages[3].is_error and "Unknown tool" in run.messages[3].content


def test_concurrent_runs_have_independent_state_and_script_positions():
    @tool
    async def echo(value: Hidden[str]):
        await asyncio.sleep(0)
        return value

    agent = Agent(
        provider=ScriptedProvider(
            [
                ModelResponse(tool_calls=(call("echo"),)),
                ModelResponse(text="Done"),
            ]
        ),
        tools=Tools([echo]),
    )

    async def run_both():
        return await asyncio.gather(agent.run("A", context={"value": "a"}), agent.run("B", context={"value": "b"}))

    first, second = asyncio.run(run_both())
    assert first.status == second.status == "completed"
    assert first.id != second.id
    assert first.messages[3].content == "a" and second.messages[3].content == "b"
    first.state["changed"] = True
    assert second.state == {}


def test_parallel_tools_are_bounded_and_results_keep_call_order():
    active, peak = 0, 0

    @tool
    async def echo(value: int):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.005 * (4 - value))
        active -= 1
        return value

    run = asyncio.run(
        Agent(
            provider=ScriptedProvider(
                [
                    ModelResponse(tool_calls=tuple(call("echo", id=str(i), value=i) for i in range(4))),
                    ModelResponse(text="Done"),
                ]
            ),
            tools=Tools([echo]),
            loop=ReAct(tool_concurrency=2),
        ).run("Task")
    )
    assert run.status == "completed"
    assert peak == 2
    assert [message.content for message in run.messages if message.role == "tool"] == ["0", "1", "2", "3"]


def test_model_limit_also_applies_without_legacy_task():
    run = asyncio.run(
        Agent(
            provider=ScriptedProvider(
                [
                    ModelResponse(tool_calls=(call(a=1, b=2),)),
                    ModelResponse(text="Must not be reached"),
                ]
            ),
            tools=Tools([add]),
            limits=Limits(max_model_calls=1),
        ).run("Task")
    )
    assert run.status == "limit" and run.model_calls == 1


def test_tool_budget_refuses_entire_batch():
    run = asyncio.run(
        Agent(
            provider=ScriptedProvider(
                [
                    ModelResponse(tool_calls=(call(a=1, b=2), call(id="c2", a=2, b=3))),
                ]
            ),
            tools=Tools([add]),
            limits=Limits(max_tool_calls=1),
        ).run("Task")
    )
    assert run.status == "limit" and run.tool_calls == 0
    assert not any(event.kind == "tool_start" for event in run.events)


def test_bad_transition_fails_explicitly():
    flow = Flow(nodes={"route": lambda runtime: "typo"}, edges={("route", "done"): END}, start="route")
    run = asyncio.run(Agent(provider=ScriptedProvider([]), loop=flow).run("Task"))
    assert run.status == "failed" and "No transition" in run.error


def test_graph_cycle_hits_step_limit():
    flow = Flow(nodes={"repeat": lambda runtime: "next"}, edges={("repeat", "next"): "repeat"}, start="repeat")
    run = asyncio.run(Agent(provider=ScriptedProvider([]), loop=flow, limits=Limits(max_steps=3)).run("Task"))
    assert run.status == "limit" and run.steps == 3


@pytest.mark.parametrize("concurrency", [1, 2])
def test_tool_timeout_cancels_and_never_retries(concurrency):
    attempts, cancelled = [], []

    @tool
    async def wait_forever():
        attempts.append(1)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(1)

    run = asyncio.run(
        Agent(
            provider=ScriptedProvider(
                [
                    ModelResponse(tool_calls=(call("wait_forever"),)),
                ]
            ),
            tools=Tools([wait_forever]),
            loop=ReAct(tool_concurrency=concurrency),
            limits=Limits(tool_timeout=0.01),
        ).run("Task")
    )
    assert run.status == "timeout"
    assert attempts == cancelled == [1]


def test_run_deadline_and_caller_cancellation():
    class Provider:
        async def create_message(self, **kwargs):
            yield {"type": "text", "text": "Starting"}
            await asyncio.Event().wait()

    run = asyncio.run(Agent(provider=Provider(), limits=Limits(timeout=0.01)).run("Task"))
    assert run.status == "timeout"

    async def cancel():
        pending = asyncio.create_task(Agent(provider=Provider()).run("Task"))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(cancel())


def test_provider_failure_is_not_success():
    class Provider:
        async def create_message(self, **kwargs):
            yield {"type": "text", "text": "Starting"}
            raise ConnectionError("offline")

    run = asyncio.run(Agent(provider=Provider()).run("Task"))
    assert run.status == "failed" and run.output is None
    assert "offline" in run.error


def test_duplicate_ids_are_rejected_before_tools():
    run = asyncio.run(
        Agent(
            provider=ScriptedProvider(
                [
                    ModelResponse(tool_calls=(call(a=1, b=2), call(a=3, b=4))),
                ]
            ),
            tools=Tools([add]),
        ).run("Task")
    )
    assert run.status == "failed" and "duplicate" in run.error
    assert run.tool_calls == 0


def test_incomplete_loop_does_not_report_success():
    flow = Flow(nodes={"empty": lambda runtime: "done"}, edges={("empty", "done"): END}, start="empty")
    run = asyncio.run(Agent(provider=ScriptedProvider([]), loop=flow).run("Task"))
    assert run.status == "failed" and "without runtime.finish" in run.error


@pytest.mark.parametrize("operation", ["complete", "finish", "say"])
def test_custom_loop_cannot_skip_tool_results(operation):
    class BadLoop:
        async def run(self, runtime):
            await runtime.complete()
            if operation == "complete":
                await runtime.complete()
            elif operation == "finish":
                runtime.finish("Skipped")
            else:
                runtime.say("Skipped")

    run = asyncio.run(
        Agent(
            provider=ScriptedProvider(
                [
                    ModelResponse(tool_calls=(call(a=1, b=2),)),
                ]
            ),
            tools=Tools([add]),
            loop=BadLoop(),
        ).run("Task")
    )
    assert run.status == "failed" and "pending tool calls" in run.error


def test_history_is_copied_and_incomplete_history_is_rejected():
    history = [Message(role="assistant", content="Before", metadata={"nested": [1]})]
    run = asyncio.run(Agent(provider=ScriptedProvider([ModelResponse(text="After")])).run("Task", history=history))
    run.messages[1].metadata["nested"].append(2)
    assert history[0].metadata == {"nested": [1]}
    with pytest.raises(ValueError, match="unresolved"):
        asyncio.run(
            Agent(provider=ScriptedProvider([])).run(
                "Task",
                history=[
                    Message(role="assistant", tool_calls=(call(a=1, b=2),)),
                ],
            )
        )


def test_tool_composition_is_ordered_and_checks_duplicates():
    @tool(tags=["read"])
    def lookup():
        return "ok"

    combined = add | lookup | add
    assert [spec.name for spec in combined.specs] == ["add", "lookup"]
    assert [spec.name for spec in combined.select("read").specs] == ["lookup"]
    with pytest.raises(ValueError, match="Duplicate"):
        combined | tool(lambda: 1, name="add")


def test_bound_methods_are_not_retained_by_a_global_descriptor():
    import gc
    import weakref

    class Service:
        def read(self, value: str | None):
            return value

    service = Service()
    reference = weakref.ref(service)
    wrapped = tool(service.read)
    assert asyncio.run(wrapped.invoke({"value": None}, {})).error is None
    del wrapped, service
    gc.collect()
    assert reference() is None
