import asyncio
import json
from dataclasses import replace

import pytest

from mickro_agent import (
    Agent,
    CompactionProposal,
    FileSessionStore,
    Limits,
    Message,
    MessageRange,
    ModelResponse,
    OlderHistory,
    Remove,
    Replace,
    Run,
    Runtime,
    Summarize,
    ToolCall,
    Tools,
)
from mickro_agent.history import validate_conversation
from mickro_agent.providers import ScriptedProvider
from mickro_agent.runtime import LimitReached
from mickro_agent.session_log import SessionLogError


def runtime(provider=None, **limits):
    return Runtime(
        Run(messages=[Message(role="system", content="System"), Message(role="user", content="Original task")]),
        provider or ScriptedProvider([]), Tools(), Limits(**limits),
    )


def snapshot(rt):
    return [message.model_copy(deep=True) for message in rt.run.messages]


def tool_turn():
    return [
        Message(role="assistant", content="Checking", tool_calls=(ToolCall(id="a", name="check"),
                                                                   ToolCall(id="b", name="check"))),
        Message(role="tool", tool_call_id="b", content="B verified"),
        Message(role="tool", tool_call_id="a", content="A failed", is_error=True),
        Message(role="assistant", content="Use B; A failed"),
    ]


@pytest.mark.parametrize("strategy", [Remove(), Replace("Final plan")])
def test_replace_and_remove_only_selected_region(strategy):
    async def scenario():
        rt = runtime()
        before = snapshot(rt)
        marker = rt.mark_messages()
        rt.say("Plan")
        rt.run.messages.extend(tool_turn())
        stop = rt.mark_messages()
        rt.say("Latest user request")
        messages = rt.run.messages
        result = await rt.compact(selection=MessageRange(marker, stop), strategy=strategy)
        assert result.applied and result.start == 2 and result.stop == 7
        assert rt.run.messages is messages
        assert rt.run.messages[:2] == before
        assert rt.run.messages[-1].content == "Latest user request"
        assert rt.run.model_calls == 0 and rt.run.tool_calls == 0
        assert len(rt.run.messages) == (3 if isinstance(strategy, Remove) else 4)
        assert rt.run.events[-1].kind == "context_compaction"
        assert len(rt.run.events[-1].data["removed_messages"]) == 5
        if result.replacement:
            result.replacement[0].metadata.clear()
            assert rt.run.messages[2].metadata  # Result is detached from live messages.
        validate_conversation(rt.run.messages)
    asyncio.run(scenario())


def test_summary_isolated_accounted_and_provider_metadata_not_replayed():
    class SummaryProvider:
        model = "summary-model"

        async def create_message(self, system_prompt, messages, metadata, **kwargs):
            assert system_prompt == "Summarize verified facts."
            assert metadata["purpose"] == "compact" and metadata["turn"] == 1
            assert kwargs["tools"] == []
            assert len(messages) == 1 and messages[0].role == "user"
            transcript = json.loads(messages[0].content.split("\n", 1)[1])
            assert [m["role"] for m in transcript] == ["assistant", "tool", "tool", "assistant"]
            assert transcript[2]["is_error"]
            yield {"type": "usage", "input_tokens": 10}
            yield {"type": "text", "text": " B verified; A failed. "}
            yield {"type": "reasoning", "text": "internal"}
            yield {"type": "usage", "output_tokens": 4}
            yield {"type": "metadata", "metadata": {"minimax_content": [{"type": "thinking", "signature": "opaque"}]}}

    async def scenario():
        rt = runtime()
        marker = rt.mark_messages()
        rt.run.messages.extend(tool_turn())
        result = await rt.compact_since(marker, strategy=Summarize(
            prompt="Summarize verified facts.", provider=SummaryProvider(),
        ))
        assert result.applied and rt.run.messages[-1].content == "B verified; A failed."
        assert len(rt.run.messages) == 3 and not rt._pending
        assert rt.run.messages[-1].metadata == {"compaction": {"kind": "summary"}}
        assert rt.run.model_calls == 1 and rt.run.usage == {"input_tokens": 10, "output_tokens": 4}
        event = next(event for event in rt.run.events if event.kind == "model_response")
        assert event.data["conversation"] is False and event.data["purpose"] == "compact"
        assert event.data["response"]["metadata"]["minimax_content"][0]["signature"] == "opaque"
        request_event = next(event for event in rt.run.events if event.kind == "model_request")
        assert request_event.data["provider"] == "SummaryProvider" and request_event.data["model"] == "summary-model"
        assert not any(m.content == "Summarize verified facts." for m in rt.run.messages)
    asyncio.run(scenario())


def test_summarization_uses_shared_budget_and_default_provider():
    async def scenario():
        rt = runtime(ScriptedProvider([ModelResponse(text="summary")]), max_model_calls=1)
        marker = rt.mark_messages()
        rt.say("Long draft")
        assert (await rt.compact_since(marker, strategy=Summarize())).applied
        with pytest.raises(LimitReached):
            await rt.complete()
        marker = rt.mark_messages()
        rt.say("More draft")
        original = snapshot(rt)
        with pytest.raises(LimitReached):
            await rt.compact_since(marker, strategy=Summarize())
        assert rt.run.messages == original and rt.run.model_calls == 1
        assert (await rt.compact_since(marker, strategy=Remove())).applied
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["empty", "tool", "error"])
def test_failed_summary_never_removes_history_and_retains_received_usage(kind):
    class Broken:
        async def create_message(self, **kwargs):
            yield {"type": "usage", "input_tokens": 3}
            if kind == "tool":
                yield {"type": "tool_use", "id": "c", "name": "effect", "input": {}}
            elif kind == "error":
                yield {"type": "text", "text": "partial"}
                raise OSError("stream failed")
            else:
                yield {"type": "text", "text": " "}

    async def scenario():
        rt = runtime(Broken())
        marker = rt.mark_messages()
        rt.say("Keep this")
        original = snapshot(rt)
        with pytest.raises((ValueError, OSError)):
            await rt.compact_since(marker, strategy=Summarize())
        assert rt.run.messages == original and not rt._pending
        assert rt.run.model_calls == 1 and rt.run.usage == {"input_tokens": 3}
        assert rt.run.events[-1].kind == "context_compaction_failed"
        rt.say("Runtime is unlocked")
    asyncio.run(scenario())


def test_cancellation_closes_stream_preserves_history_and_usage():
    async def scenario():
        entered, closed = asyncio.Event(), asyncio.Event()

        class Blocked:
            async def create_message(self, **kwargs):
                try:
                    yield {"type": "usage", "input_tokens": 7}
                    entered.set()
                    await asyncio.Event().wait()
                finally:
                    closed.set()

        rt = runtime(Blocked())
        marker = rt.mark_messages()
        rt.say("Keep original")
        original = snapshot(rt)
        task = asyncio.create_task(rt.compact_since(marker, strategy=Summarize()))
        await entered.wait()
        with pytest.raises(RuntimeError, match="busy"):
            rt.say("Concurrent instruction")
        with pytest.raises(RuntimeError, match="busy"):
            await rt.complete()
        with pytest.raises(RuntimeError, match="busy"):
            await rt.compact_since(marker, strategy=Remove())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set() and rt.run.messages == original
        assert rt.run.usage == {"input_tokens": 7}
        assert (await rt.compact_since(marker, strategy=Remove())).applied
    asyncio.run(scenario())


def test_markers_reject_foreign_stale_and_changed_prefixes():
    async def scenario():
        rt = runtime()
        marker = rt.mark_messages()
        rt.say("Draft")
        with pytest.raises(ValueError, match="another runtime"):
            await runtime().compact_since(marker, strategy=Remove())
        rt.run.messages[0].metadata["changed"] = True
        with pytest.raises(ValueError, match="earlier messages"):
            await rt.compact_since(marker, strategy=Remove())
        rt.run.messages[0].metadata.clear()
        stop = rt.mark_messages()
        with pytest.raises(ValueError, match="starts after"):
            await rt.compact(selection=MessageRange(stop, marker), strategy=Remove())
        await rt.compact_since(marker, strategy=Replace("Summary"))
        with pytest.raises(ValueError, match="Stale"):
            await rt.compact_since(marker, strategy=Remove())
        with pytest.raises(ValueError, match="Stale"):
            await rt.compact_since(stop, strategy=Remove())
    asyncio.run(scenario())


def test_pending_calls_and_inflight_requests_block_marking_and_compaction():
    async def scenario():
        rt = runtime(ScriptedProvider([ModelResponse(tool_calls=(ToolCall(id="c", name="check"),))]))
        marker = rt.mark_messages()
        await rt.complete()
        with pytest.raises(ValueError, match="pending"):
            rt.mark_messages()
        with pytest.raises(ValueError, match="pending"):
            await rt.compact_since(marker, strategy=Remove())
        await rt.execute_tools()
        assert (await rt.compact_since(marker, strategy=Remove())).applied

        entered = asyncio.Event()

        class Slow:
            async def create_message(self, **kwargs):
                entered.set()
                await asyncio.Event().wait()
                yield {"type": "text", "text": "unreachable"}

        rt = runtime(Slow())
        marker = rt.mark_messages()
        task = asyncio.create_task(rt.complete())
        await entered.wait()
        with pytest.raises(RuntimeError, match="busy"):
            rt.mark_messages()
        with pytest.raises(RuntimeError, match="busy"):
            await rt.compact_since(marker, strategy=Remove())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())


def test_older_history_keeps_first_user_and_recent_complete_turns():
    async def scenario():
        rt = runtime()
        rt.run.messages.extend(tool_turn())
        rt.say("Second task")
        recent = tool_turn()
        rt.run.messages.extend(recent)
        rt.say("Third task")
        original = snapshot(rt)
        result = await rt.compact(selection=OlderHistory(keep_recent_turns=2), strategy=Replace("Earlier results"))
        assert result.applied and result.start == 2 and result.stop == 6
        assert rt.run.messages == original[:2] + list(result.replacement) + original[6:]
        validate_conversation(rt.run.messages)
        assert not (await rt.compact(selection=OlderHistory(keep_recent_turns=10), strategy=Remove())).applied
    asyncio.run(scenario())


def test_custom_proposals_cannot_break_tool_history_or_change_system_prompt():
    class Custom:
        def __init__(self, messages):
            self.messages = messages

        async def compact(self, request):
            request.messages[0].metadata["mutated"] = True
            return CompactionProposal(tuple(self.messages))

    async def scenario():
        rt = runtime()
        marker = rt.mark_messages()
        rt.say("Draft")
        original = snapshot(rt)
        for messages in [tool_turn()[:1], tool_turn()[1:], [Message(role="system", content="Changed")]]:
            with pytest.raises(ValueError):
                await rt.compact_since(marker, strategy=Custom(messages))
            assert rt.run.messages == original
        result = await rt.compact_since(marker, strategy=Custom(tool_turn()))
        assert result.applied
        validate_conversation(rt.run.messages)
    asyncio.run(scenario())


def test_token_counter_rejects_growth_without_refunding_summary_call():
    async def scenario():
        rt = runtime(ScriptedProvider([ModelResponse(text="much longer summary", usage={"output_tokens": 4})]))
        marker = rt.mark_messages()
        rt.say("tiny")
        original = snapshot(rt)
        counted = []

        async def character_counter(messages):
            counted.append([message.content for message in messages])
            return sum(len(message.content) for message in messages)

        result = await rt.compact_since(marker, strategy=Summarize(), token_counter=character_counter)
        assert not result.applied and result.reason == "not_smaller"
        assert result.after_tokens > result.before_tokens
        assert rt.run.messages == original and rt.run.model_calls == 1
        assert rt.run.usage == {"output_tokens": 4}
        assert all(messages[:2] == ["System", "Original task"] for messages in counted)
        result = await rt.compact_since(marker, strategy=Remove(), token_counter=character_counter)
        assert result.applied and result.after_tokens < result.before_tokens
    asyncio.run(scenario())


def test_empty_and_unchanged_regions_do_not_invalidate_markers_or_call_models():
    class Identity:
        async def compact(self, request):
            return CompactionProposal(request.messages)

    async def scenario():
        rt = runtime()
        marker = rt.mark_messages()
        result = await rt.compact_since(marker, strategy=Summarize())
        assert result.reason == "empty_selection" and rt.run.model_calls == 0
        rt.say("Draft")
        result = await rt.compact_since(marker, strategy=Identity())
        assert not result.applied and result.reason == "unchanged"
        assert (await rt.compact_since(marker, strategy=Remove())).applied
    asyncio.run(scenario())


def test_direct_conversation_mutation_during_strategy_rejects_commit():
    async def scenario():
        rt = runtime()
        marker = rt.mark_messages()
        rt.say("Original draft")

        class Misbehaving:
            async def compact(self, request):
                rt.run.messages.append(Message(role="user", content="Concurrent direct append"))
                return CompactionProposal((Message(role="assistant", content="Summary"),))

        with pytest.raises(RuntimeError, match="changed during"):
            await rt.compact_since(marker, strategy=Misbehaving())
        assert [m.content for m in rt.run.messages[-2:]] == ["Original draft", "Concurrent direct append"]
    asyncio.run(scenario())


def test_log_failure_does_not_apply_proposal():
    async def scenario():
        rt = runtime()
        marker = rt.mark_messages()
        rt.say("Draft")
        original = snapshot(rt)

        def fail(event):
            raise SessionLogError("disk failed")

        rt.run._event_sink = fail
        with pytest.raises(SessionLogError):
            await rt.compact_since(marker, strategy=Remove())
        assert rt.run.messages == original
        rt.run._event_sink = None
        assert (await rt.compact_since(marker, strategy=Remove())).applied
    asyncio.run(scenario())


def test_checkpoints_contain_handoff_and_logs_retain_removed_messages(tmp_path):
    class Planning:
        async def run(self, rt):
            marker = rt.mark_messages()
            rt.say("Generate a draft")
            await rt.complete(allow_tools=False, purpose="plan")
            await rt.compact_since(marker, strategy=Replace("Approved handoff"))
            response = await rt.complete()
            rt.finish(response.text)

    class Continue:
        async def create_message(self, messages, **kwargs):
            assert any(m.content == "Approved handoff" for m in messages)
            assert not any(m.content in {"Generate a draft", "Rejected draft"} for m in messages)
            yield {"type": "text", "text": "Continued"}

    async def scenario():
        agent = Agent(
            provider=ScriptedProvider([ModelResponse(text="Rejected draft"), ModelResponse(text="Executed")]),
            loop=Planning(), sessions=FileSessionStore(tmp_path / "sessions"), log_dir=tmp_path / "logs",
        )
        first = await agent.run("Task")
        assert first.status == "completed" and first.output == "Executed"
        checkpoint = agent.sessions.load(first.session_id)
        assert [m.content for m in checkpoint.messages] == ["Task", "Approved handoff", "Executed"]
        assert "Rejected draft" in first.log_path.read_text()
        # Default ReAct on continuation, sharing the checkpoint store.
        from mickro_agent import ReAct
        second = await replace(agent, provider=Continue(), loop=ReAct()).run("Continue", session_id=first.session_id)
        assert second.status == "completed" and second.output == "Continued"
    asyncio.run(scenario())


def test_failed_summary_during_run_preserves_checkpoint(tmp_path):
    class Planning:
        async def run(self, rt):
            marker = rt.mark_messages()
            rt.say("Draft must survive")
            await rt.compact_since(marker, strategy=Summarize())

    agent = Agent(provider=ScriptedProvider([ModelResponse(text="")]), loop=Planning(),
                  sessions=FileSessionStore(tmp_path))
    run = asyncio.run(agent.run("Task"))
    assert run.status == "failed"
    assert agent.sessions.load(run.session_id).messages[-1].content == "Draft must survive"


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_recent_turn_count(value):
    with pytest.raises(ValueError):
        OlderHistory(keep_recent_turns=value)


@pytest.mark.parametrize("strategy", [Replace, Summarize])
def test_empty_replacement_or_summary_prompt_rejected(strategy):
    with pytest.raises(ValueError):
        strategy(" ")


@pytest.mark.parametrize("summarize", [False, True])
def test_planning_example_executes_after_compaction(summarize):
    from planning_compaction import make_agent

    run = asyncio.run(make_agent(summarize=summarize).run("Process A, B, C, D with three workers."))
    assert run.status == "completed" and run.output == "Processed all four items in two batches."
    assert run.model_calls == (9 if summarize else 8) and run.tool_calls == 3
    assert not any("Draft:" in message.content for message in run.messages)
    event = next(event for event in run.events if event.kind == "context_compaction")
    assert event.data["strategy"] == ("Summarize" if summarize else "Replace")
    assert any("Draft:" in message["content"] for message in event.data["removed_messages"])
    assert "planning_start" not in run.state


def test_summary_generator_expires_and_custom_strategy_can_use_it():
    async def scenario():
        rt = runtime(ScriptedProvider([ModelResponse(text="Facts")]))
        marker = rt.mark_messages()
        rt.say("Some facts")
        saved = []

        class Custom:
            async def compact(self, request):
                saved.append(request.generate_summary)
                response = await request.generate_summary("Extract facts")
                return CompactionProposal((Message(role="assistant", content=response.text),))

        await rt.compact_since(marker, strategy=Custom())
        assert rt.run.messages[-1].content == "Facts" and rt.run.model_calls == 1
        with pytest.raises(RuntimeError, match="only valid"):
            await saved[0]("Late request")
        assert rt.run.model_calls == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("bad_count", [-1, 2.5, True, None])
def test_invalid_token_counts_leave_context_unchanged(bad_count):
    async def scenario():
        rt = runtime()
        marker = rt.mark_messages()
        rt.say("Original")
        original = snapshot(rt)

        async def counter(messages):
            return bad_count

        with pytest.raises(ValueError, match="nonnegative integer"):
            await rt.compact_since(marker, strategy=Remove(), token_counter=counter)
        assert rt.run.messages == original
    asyncio.run(scenario())
