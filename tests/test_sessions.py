import asyncio
import json
from dataclasses import replace

import pytest

from mickro_agent import Agent, FileSessionStore, Limits, Message, ModelResponse, ToolCall, Tools, tool
from mickro_agent.providers import ScriptedProvider
from mickro_agent.session_log import SessionLogError


class Remember:
    async def create_message(self, messages, **kwargs):
        yield {"type": "text", "text": " | ".join(m.content for m in messages if m.role == "user")}
        yield {"type": "metadata", "metadata": {"minimax_content": [{"type": "thinking", "signature": "opaque"}]}}


def test_resume_from_disk_retains_metadata_and_appends_to_log(tmp_path):
    async def scenario():
        agent = Agent(provider=Remember(), sessions=FileSessionStore(tmp_path / "sessions"), log_dir=tmp_path / "logs")
        first = await agent.run("First", session_id="conversation")
        original = first.log_path.read_text()
        first.messages[2].metadata.clear()  # Returned Run cannot mutate the saved checkpoint.
        recreated = Agent(provider=Remember(), sessions=FileSessionStore(tmp_path / "sessions"))
        second = await recreated.run("Second", session_id=first.session_id)
        assert second.status == "completed" and second.output == "First | Second"
        assert second.session_id == first.session_id and second.id != first.id
        assert second.messages[2].metadata["minimax_content"][0]["signature"] == "opaque"
        assert first.log_path == second.log_path
        assert second.log_path.read_text().startswith(original)
        rows = [json.loads(line) for line in second.log_path.read_text().splitlines()]
        assert {row["run_id"] for row in rows} == {first.id, second.id}
        assert {row["session_id"] for row in rows} == {first.session_id}
        for run in (first, second):
            assert [r["sequence"] for r in rows if r["run_id"] == run.id] == list(range(len(run.events)))
        with pytest.raises(ValueError, match="overwrite"):
            second.write_events(second.log_path)

    asyncio.run(scenario())


@pytest.mark.parametrize("disk", [False, True])
def test_simultaneous_continuations_are_serialized(tmp_path, disk):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class BlockOnce(Remember):
            async def create_message(self, messages, **kwargs):
                if messages[-1].content == "Second":
                    entered.set()
                    await release.wait()
                async for chunk in super().create_message(messages=messages):
                    yield chunk

        agent = Agent(provider=BlockOnce())
        if disk:
            agent = replace(agent, sessions=FileSessionStore(tmp_path))
        first = await agent.run("First")
        other = replace(agent, sessions=FileSessionStore(tmp_path)) if disk else agent
        second = asyncio.create_task(agent.run("Second", session_id=first.session_id))
        await entered.wait()
        third = asyncio.create_task(other.run("Third", session_id=first.session_id))
        await asyncio.sleep(0.03)
        assert not third.done()
        release.set()
        result2, result3 = await asyncio.gather(second, third)
        assert result2.output == "First | Second"
        assert result3.output == "First | Second | Third"

    asyncio.run(scenario())


def test_cancelled_tool_is_checkpointed_without_reexecuting():
    async def scenario():
        entered = asyncio.Event()
        effects = []

        @tool
        async def effect() -> str:
            effects.append("started")
            entered.set()
            await asyncio.Event().wait()
            return "never"

        agent = Agent(
            provider=ScriptedProvider([ModelResponse(tool_calls=(ToolCall(id="effect", name="effect"),))]),
            tools=Tools([effect]),
        )
        task = asyncio.create_task(agent.run("First", session_id="interrupted"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        resumed = await replace(agent, provider=Remember()).run("Inspect status", session_id="interrupted")
        assert resumed.status == "completed"
        result = next(m for m in resumed.messages if m.role == "tool")
        assert result.is_error and "Outcome unknown" in result.content
        assert effects == ["started"]

    asyncio.run(scenario())


def test_resume_checks_configuration_history_and_log_integrity(tmp_path):
    async def scenario():
        agent = Agent(provider=Remember(), log_dir=tmp_path)
        run = await agent.run("First")
        with pytest.raises(ValueError, match="history"):
            await agent.run("Second", session_id=run.session_id, history=[Message(role="user", content="Other")])
        with pytest.raises(ValueError, match="configuration"):
            await replace(agent, name="other").run("Second", session_id=run.session_id)
        with pytest.raises(ValueError, match="directory"):
            await agent.run("Second", session_id=run.session_id, log_dir=tmp_path / "other")
        with run.log_path.open("a") as handle:
            handle.write('{"partial":')
        with pytest.raises(SessionLogError):
            await agent.run("Second", session_id=run.session_id)

    asyncio.run(scenario())


def test_hard_crash_checkpoint_requires_explicit_recovery():
    async def scenario():
        agent = Agent(provider=Remember())
        run = await agent.run("First")
        saved = agent.sessions.load(run.session_id)
        agent.sessions.save(saved.model_copy(update={"status": "running"}))
        with pytest.raises(ValueError, match="before checkpointing"):
            await agent.run("Continue", session_id=run.session_id)

    asyncio.run(scenario())


def test_limits_reset_each_invocation_and_sessions_are_independent():
    async def scenario():
        agent = Agent(provider=Remember(), limits=Limits(max_model_calls=1))
        first = await agent.run("First")
        first.messages[2].metadata.clear()
        second = await agent.run("Second", session_id=first.session_id)
        separate = await agent.run("Separate")
        assert first.model_calls == second.model_calls == 1
        assert second.messages[2].metadata
        assert separate.output == "Separate"
        assert second.state == {} and second.context == {}

    asyncio.run(scenario())


def test_interrupted_parallel_batch_preserves_known_results():
    from mickro_agent import ReAct

    async def scenario():
        completed = asyncio.Event()

        @tool
        async def done() -> str:
            completed.set()
            return "effect succeeded"

        @tool
        async def slow() -> str:
            await completed.wait()
            await asyncio.Event().wait()
            return "never"

        agent = Agent(
            provider=ScriptedProvider(
                [ModelResponse(tool_calls=(ToolCall(id="known", name="done"), ToolCall(id="unknown", name="slow")))]
            ),
            tools=Tools([done, slow]),
            loop=ReAct(tool_concurrency=2),
            limits=Limits(tool_timeout=0.02),
        )
        run = await agent.run("Perform effects")
        assert run.status == "timeout"
        resumed = await replace(agent, provider=Remember()).run("Inspect", session_id=run.session_id)
        results = {m.tool_call_id: m for m in resumed.messages if m.role == "tool"}
        assert results["known"].content == "effect succeeded" and not results["known"].is_error
        assert results["unknown"].is_error and "Outcome unknown" in results["unknown"].content

    asyncio.run(scenario())


def test_live_cli_prints_working_continuation_options(tmp_path, monkeypatch, capsys):
    import shlex
    from types import SimpleNamespace

    import live_chat

    class Offline(Remember):
        def __init__(self, **kwargs):
            pass

        async def aclose(self):
            pass

    monkeypatch.setattr(live_chat, "OpenAIChat", Offline)
    args = SimpleNamespace(
        provider="openai",
        model="offline",
        base_url=None,
        log_dir=None,
        events=None,
        session_dir=str(tmp_path / "sessions with spaces"),
        session_id=None,
        prompt="First",
    )
    asyncio.run(live_chat.main(args))
    output = capsys.readouterr().out
    command = shlex.split(output.split("To continue: ", 1)[1].strip())
    assert command[command.index("--session-dir") + 1] == args.session_dir
    args.session_id = command[command.index("--session-id") + 1]
    args.prompt = "Second"
    asyncio.run(live_chat.main(args))
    assert "First | Second" in capsys.readouterr().out
