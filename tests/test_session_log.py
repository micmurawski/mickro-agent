import asyncio
import json

import pytest

from mickro_agent import Agent, Limits, Message, ModelResponse, SessionLogError, ToolCall, Tools, tool
from mickro_agent.providers import ScriptedProvider
from mickro_agent.session_log import SessionLog


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_session_log_contains_metadata_conversation_tool_events_and_outcome(tmp_path):
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=(ToolCall(id="call-1", name="add", arguments={"a": 2, "b": 3}),)),
            ModelResponse(text="5", usage={"output_tokens": 2}),
        ]
    )
    run = asyncio.run(
        Agent(provider=provider, tools=Tools([add]), log_dir=tmp_path / "nested", name="chef").run(
            "Add 2 and 3\nPlease",
            history=[Message(role="user", content="Earlier")],
            context={"credential": "not-automatically-logged"},
            session_id="dinner-01",
        )
    )
    assert run.status == "completed"
    assert run.session_id == "dinner-01" and run.id != run.session_id
    assert run.log_path == tmp_path / "nested" / "mikro-agent-dinner-01.jsonl"
    data = records(run.log_path)
    assert len(data) == len(run.events)
    assert [row["sequence"] for row in data] == list(range(len(data)))
    assert all(
        row["session_id"] == run.session_id and row["run_id"] == run.id and row["schema_version"] == 2 for row in data
    )
    assert all(row["timestamp"] for row in data)
    header = data[0]
    assert header["kind"] == "run_start" and header["data"]["agent"] == "chef"
    assert header["data"]["cwd"] and header["data"]["provider"] == "ScriptedProvider"
    assert header["data"]["messages"][-1]["content"] == "Add 2 and 3\nPlease"
    assert header["data"]["messages"][1]["content"] == "Earlier"
    assert {"model_request", "model_chunk", "model_response", "tool_start", "tool_result"} <= {
        row["kind"] for row in data
    }
    assert data[-1]["kind"] == "run_end" and data[-1]["data"]["status"] == "completed"
    assert data[-1]["data"]["usage"] == {"output_tokens": 2}
    assert "not-automatically-logged" not in run.log_path.read_text()
    exported = tmp_path / "snapshot.jsonl"
    run.write_events(exported)
    assert records(exported) == data


def test_chunks_are_flushed_before_turn_finishes_and_cancellation_is_logged(tmp_path):
    async def scenario():
        ready = asyncio.Event()

        class SlowProvider:
            async def create_message(self, **kwargs):
                yield {"type": "text", "text": "First token"}
                ready.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(Agent(provider=SlowProvider(), log_dir=tmp_path).run("Task", session_id="live"))
        await asyncio.wait_for(ready.wait(), timeout=2)
        path = tmp_path / "mikro-agent-live.jsonl"
        before = records(path)
        assert before[-1]["kind"] == "model_chunk"
        assert before[-1]["data"]["chunk"]["text"] == "First token"
        assert not any(row["kind"] == "run_end" for row in before)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert records(path)[-1]["data"]["status"] == "cancelled"

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["failed", "timeout", "limit"])
def test_failure_timeout_and_limit_have_final_records(tmp_path, outcome):
    class BrokenProvider:
        async def create_message(self, **kwargs):
            yield {"type": "text", "text": "Partial"}
            if outcome == "failed":
                raise RuntimeError("offline")
            await asyncio.Event().wait()

    if outcome == "limit":
        provider = ScriptedProvider([ModelResponse(tool_calls=(ToolCall(id="a", name="unknown"),))])
        limits = Limits(max_tool_calls=0)
    else:
        provider = BrokenProvider()
        limits = Limits(timeout=0.01)
    run = asyncio.run(Agent(provider=provider, limits=limits, log_dir=tmp_path).run("Task"))
    assert run.status == outcome
    assert records(run.log_path)[-1]["data"]["status"] == outcome


def test_concurrent_sessions_have_separate_files_and_run_override(tmp_path):
    agent = Agent(provider=ScriptedProvider([ModelResponse(text="Done")]), log_dir=tmp_path / "default")

    async def both():
        return await asyncio.gather(agent.run("A"), agent.run("B", log_dir=tmp_path / "override"))

    first, second = asyncio.run(both())
    assert first.id != second.id and first.log_path.parent != second.log_path.parent
    for run in (first, second):
        assert {row["session_id"] for row in records(run.log_path)} == {run.session_id}
        assert run.log_path.name == f"mikro-agent-{run.session_id}.jsonl"


def test_logging_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = asyncio.run(Agent(provider=ScriptedProvider([ModelResponse(text="Done")])).run("Task"))
    assert run.log_path is None and list(tmp_path.iterdir()) == []


def test_existing_session_is_not_overwritten(tmp_path):
    path = tmp_path / "mikro-agent-existing.jsonl"
    path.write_text("original\n")
    agent = Agent(provider=ScriptedProvider([]), log_dir=tmp_path)
    with pytest.raises(SessionLogError, match="Cannot create"):
        asyncio.run(agent.run("Task", session_id="existing"))
    assert path.read_text() == "original\n"


@pytest.mark.parametrize("session_id", ["../escape", "a/b", "a\\b", "", ".", "a" * 129])
def test_invalid_session_id_does_not_create_a_file(tmp_path, session_id):
    with pytest.raises(ValueError, match="session_id"):
        asyncio.run(Agent(provider=ScriptedProvider([]), log_dir=tmp_path).run("Task", session_id=session_id))
    assert list(tmp_path.iterdir()) == []


def test_log_creation_failure_happens_before_provider_call(tmp_path):
    invalid_directory = tmp_path / "file"
    invalid_directory.write_text("not a directory")
    with pytest.raises(SessionLogError, match="Cannot create"):
        asyncio.run(Agent(provider=ScriptedProvider([]), log_dir=invalid_directory).run("Task"))


def test_write_failure_is_visible_and_file_is_closed(tmp_path, monkeypatch):
    original_write = SessionLog.write
    handles = []

    def fail(self, event):
        if event.kind == "model_chunk":
            handles.append(self._file)
            raise SessionLogError("disk unavailable")
        original_write(self, event)

    monkeypatch.setattr(SessionLog, "write", fail)
    with pytest.raises(SessionLogError, match="disk unavailable"):
        asyncio.run(Agent(provider=ScriptedProvider([ModelResponse(text="Done")]), log_dir=tmp_path).run("Task"))
    assert len(handles) == 1 and handles[0].closed
