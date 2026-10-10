import asyncio
import json

import pytest

from mickro_agent import (
    AGENT_JOB_TOOLS,
    Agent,
    AgentJobs,
    Hidden,
    Limits,
    ModelResponse,
    Run,
    Runtime,
    ToolCall,
    Tools,
    agent_call_tool_factory,
    agent_tool,
    tool,
)
from mickro_agent.providers import ScriptedProvider


class Remember:
    async def create_message(self, messages, **kwargs):
        yield {"type": "text", "text": " | ".join(m.content for m in messages if m.role == "user")}


def runtime(context=None):
    return Runtime(Run(context=context or {}), Remember(), Tools(), Limits())


def test_factory_has_only_model_arguments_and_preserves_child_history(tmp_path):
    async def scenario():
        child = Agent(provider=Remember(), name="researcher", log_dir=tmp_path)
        ask = agent_call_tool_factory(child, name="ask_researcher", description="Research a topic.")
        assert set(ask.spec.parameters["properties"]) == {"prompt", "session_id"}
        assert ask.spec.parameters["required"] == ["prompt"]
        parent = runtime()
        first = await ask.invoke({"prompt": "First"}, {"runtime": parent})
        second = await ask.invoke({"prompt": "Second", "session_id": first.result["session_id"]}, {"runtime": parent})
        assert second.result["output"] == "First | Second"
        assert first.result["session_id"] == second.result["session_id"]
        assert first.result["run_id"] != second.result["run_id"]
        assert "ask_researcher(" in second.result["continuation"]
        records = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
        headers = [row["data"] for row in records if row["kind"] == "run_start"]
        assert len(headers) == 2
        assert all(row["parent_session_id"] == parent.run.session_id for row in headers)
        assert all(row["parent_run_id"] == parent.run.id for row in headers)
        bad = await ask.invoke({"prompt": "Bad", "runtime": {}}, {"runtime": parent})
        assert bad.error and "Invalid arguments" in bad.error

    asyncio.run(scenario())


def test_parent_react_can_call_child_and_child_context_is_explicit():
    @tool
    def read_secret(secret: Hidden[str]) -> str:
        return secret

    async def scenario():
        child = Agent(
            provider=ScriptedProvider(
                [ModelResponse(tool_calls=(ToolCall(id="read", name="read_secret"),)), ModelResponse(text="Done")]
            ),
            tools=Tools([read_secret]),
        )
        ask = agent_tool(child, context={"secret": "child-only"})
        parent = Agent(
            provider=ScriptedProvider(
                [
                    ModelResponse(tool_calls=(ToolCall(id="delegate", name=ask.name, arguments={"prompt": "Work"}),)),
                    ModelResponse(text="Finished delegating"),
                ]
            ),
            tools=Tools([ask]),
        )
        run = await parent.run("Start", context={"secret": "parent-only", "runtime": "spoofed"})
        assert run.status == "completed"
        result = json.loads(next(m.content for m in run.messages if m.role == "tool"))
        checkpoint = child.sessions.load(result["session_id"])
        assert next(m.content for m in checkpoint.messages if m.role == "tool") == "child-only"

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["completed", "failed", "timeout", "cancelled", "early_cancel"])
def test_background_returns_immediately_and_every_outcome_wakes_parent(outcome):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class Slow:
            async def create_message(self, **kwargs):
                entered.set()
                await release.wait()
                if outcome == "failed":
                    raise RuntimeError("provider failed")
                yield {"type": "text", "text": "Done"}

        mailbox = asyncio.Queue()
        async with AgentJobs(mailbox) as jobs:
            child = Agent(provider=Slow(), limits=Limits(timeout=0.03 if outcome == "timeout" else 2))
            ask = agent_tool(child, execution="background")
            parent = runtime()
            context = {"runtime": parent, "agent_jobs": jobs}
            handle = (await ask.invoke({"prompt": "Work"}, context)).result
            assert handle["status"] == "running"
            assert not entered.is_set()  # The tool did not wait for the child.
            if outcome != "early_cancel":
                await asyncio.wait_for(entered.wait(), 1)
            if outcome in {"cancelled", "early_cancel"}:
                result = await AGENT_JOB_TOOLS.invoke(
                    ToolCall(id="cancel", name="cancel_agent_job", arguments={"job_id": handle["job_id"]}), context
                )
                assert result.result["status"] == "stopping"
            elif outcome != "timeout":
                release.set()
            event = await asyncio.wait_for(mailbox.get(), 2)
            assert event["kind"] == "AgentFinished"
            assert event["status"] == ("cancelled" if outcome == "early_cancel" else outcome)
            assert event["job_id"] == handle["job_id"]
            assert event["session_id"] == handle["session_id"]
            await jobs.wait_all()
            jobs.cancel(handle["job_id"], parent_session_id=parent.run.session_id)
            assert mailbox.empty()
            with pytest.raises(ValueError, match="another parent"):
                jobs.inspect(handle["job_id"], parent_session_id="stranger")
            if outcome == "completed":
                assert event["output"] == "Done" and event["run_id"]

    asyncio.run(scenario())


def test_background_continuation_and_supervisor_cleanup():
    async def scenario():
        mailbox = asyncio.Queue()
        jobs = AgentJobs(mailbox)
        parent = runtime()
        ask = agent_tool(Agent(provider=Remember()), execution="background")
        ctx = {"agent_jobs": jobs, "runtime": parent}
        async with jobs:
            first = (await ask.invoke({"prompt": "A"}, ctx)).result
            await jobs.wait_all()
            mailbox.get_nowait()
            await ask.invoke({"prompt": "B", "session_id": first["session_id"]}, ctx)
            second = await asyncio.wait_for(mailbox.get(), 1)
            assert second["output"] == "A | B"
            await jobs.wait_all()
        closed = await ask.invoke({"prompt": "C"}, ctx)
        assert "closed" in closed.error

    asyncio.run(scenario())


def test_supervisor_exit_cancels_children_and_queues_outcome():
    async def scenario():
        entered = asyncio.Event()

        class Slow:
            async def create_message(self, **kwargs):
                entered.set()
                await asyncio.Event().wait()
                yield {"type": "text", "text": "unreachable"}

        mailbox = asyncio.Queue()
        async with AgentJobs(mailbox) as jobs:
            ask = agent_tool(Agent(provider=Slow()), execution="background")
            await ask.invoke({"prompt": "A"}, {"runtime": runtime(), "agent_jobs": jobs})
            await entered.wait()
        assert mailbox.get_nowait()["status"] == "cancelled"
        assert mailbox.empty()

    asyncio.run(scenario())


def test_same_parent_session_is_rejected_and_mode_is_validated():
    async def scenario():
        child = Agent(provider=Remember())
        ask = agent_tool(child)
        parent = runtime()
        result = await ask.invoke({"prompt": "Work", "session_id": parent.run.session_id}, {"runtime": parent})
        assert "active parent" in result.error
        with pytest.raises(ValueError, match="execution"):
            agent_tool(child, execution="typo")
        with pytest.raises(ValueError, match="unbounded"):
            AgentJobs(asyncio.Queue(maxsize=1))

    asyncio.run(scenario())


def test_delegating_chef_wakes_and_continues_same_child_session(tmp_path):
    from delegating_chef import demo

    run = asyncio.run(demo(report=lambda _: None, log_dir=tmp_path))
    assert run.status == "completed", run.error
    assert "gluten-free" in run.output
    events = [e.data["event"] for e in run.events if e.kind == "mailbox_event"]
    assert len(events) == 2
    assert events[0]["session_id"] == events[1]["session_id"]
    assert events[0]["run_id"] != events[1]["run_id"]
    assert events[0]["job_id"] != events[1]["job_id"]
    assert len(list(tmp_path.glob("*.jsonl"))) == 2  # Parent and child, despite three invocations.
    for index, event in enumerate(run.events):
        if event.kind == "mailbox_event":
            assert any(later.kind == "model_request" for later in run.events[index + 1 :])


def test_delegation_logging_failure_does_not_launch_child():
    from mickro_agent import SessionLogError

    async def scenario():
        def fail(_):
            raise SessionLogError("disk unavailable")

        mailbox = asyncio.Queue()
        async with AgentJobs(mailbox) as jobs:
            parent = runtime()
            parent.run._event_sink = fail
            ask = agent_tool(Agent(provider=Remember()), execution="background")
            with pytest.raises(SessionLogError, match="disk unavailable"):
                await ask.invoke({"prompt": "Work"}, {"runtime": parent, "agent_jobs": jobs})
            assert jobs.list_jobs(parent_session_id=parent.run.session_id) == []
        assert mailbox.empty()

    asyncio.run(scenario())
