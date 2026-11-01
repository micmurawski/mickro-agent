import asyncio
import json

import pytest

from mickro_agent import (
    Agent, Limits, Message, ModelResponse, ReActWithSummary, Run, Runtime,
    Summarize, SummaryHistory, ToolCall, Tools, tool,
)
from mickro_agent.context_ops import SUMMARY_PROMPT
from mickro_agent.history import validate_conversation
from mickro_agent.providers import ScriptedProvider


async def count_tokens(messages):
    # Deterministic test units, not a model tokenizer.
    return sum(len(message.content) + 1 for message in messages)


def make_runtime(messages, provider=None):
    return Runtime(Run(messages=messages), provider or ScriptedProvider([ModelResponse(text="Summary")]),
                   Tools(), Limits())


def conversation():
    return [
        Message(role="system", content="System"),
        Message(role="user", content="Original task"),
        Message(role="assistant", content="Old investigation. " * 30),
        Message(role="user", content="Old follow-up. " * 30),
        Message(role="assistant", content="Recent reasoning"),
        Message(role="user", content="Recent request"),
        Message(role="assistant", content="Recent answer"),
    ]


def test_summary_window_keeps_first_and_recent_messages_and_can_roll_forward():
    async def scenario():
        rt = make_runtime(conversation(), ScriptedProvider([
            ModelResponse(text="First summary"), ModelResponse(text="Second summary"),
        ]))
        original = list(rt.run.messages)
        first = await rt.compact(selection=SummaryHistory(), strategy=Summarize(), token_counter=count_tokens)
        assert first.applied and first.after_tokens < first.before_tokens
        assert rt.run.messages[:2] == original[:2]
        assert rt.run.messages[3:] == original[-3:]
        assert rt.run.messages[2].content == "First summary"
        # Once enough new history accumulates, the preceding summary is input
        # to the next summary, instead of accumulating summaries in context.
        rt.run.messages.extend([Message(role="user", content="More work. " * 30)] * 3)
        second = await rt.compact(selection=SummaryHistory(), strategy=Summarize(), token_counter=count_tokens)
        assert second.applied and rt.run.messages[2].content == "Second summary"
        requests = [e.data["request"] for e in rt.run.events if e.kind == "model_request"]
        assert requests[0]["messages"][0]["content"] == SUMMARY_PROMPT
        transcript = json.loads(requests[1]["messages"][1]["content"].split("\n", 1)[1])
        assert transcript[0]["content"] == "First summary"
        assert not any("Old investigation" in m["content"] for m in transcript)
        validate_conversation(rt.run.messages)
    asyncio.run(scenario())


@pytest.mark.parametrize("metadata", [{"compaction": {"kind": "summary"}}, {"isSummary": True}])
def test_latest_summary_supersedes_earlier_messages(metadata):
    async def scenario():
        messages = conversation()
        messages[3:4] = [Message(role="assistant", content="Last summary", metadata=metadata),
                         Message(role="user", content="New findings. " * 30)]
        rt = make_runtime(messages)
        result = await rt.compact(selection=SummaryHistory(), strategy=Summarize(prompt="Custom instructions"),
                                  token_counter=count_tokens)
        assert result.applied
        request = next(e.data["request"] for e in rt.run.events if e.kind == "model_request")
        assert request["messages"][0]["content"] == "Custom instructions"
        transcript = json.loads(request["messages"][1]["content"].split("\n", 1)[1])
        assert [m["content"] for m in transcript] == ["Last summary", "New findings. " * 30]
        assert len(rt.run.messages) == 6
    asyncio.run(scenario())


def test_summary_window_expands_tail_for_parallel_tool_results():
    async def scenario():
        messages = conversation()[:4] + [
            Message(role="assistant", tool_calls=(ToolCall(id="a", name="check"), ToolCall(id="b", name="check"))),
            Message(role="tool", tool_call_id="b", content="B"),
            Message(role="tool", tool_call_id="a", content="A"),
            Message(role="assistant", content="Latest answer"),
        ]
        rt = make_runtime(messages)
        result = await rt.compact(selection=SummaryHistory(), strategy=Summarize(), token_counter=count_tokens)
        assert result.applied and result.stop == 4
        assert rt.run.messages[3:] == messages[-4:]
        validate_conversation(rt.run.messages)
    asyncio.run(scenario())


@pytest.mark.parametrize("size", range(7))
def test_short_history_is_not_summarized(size):
    async def scenario():
        rt = make_runtime(conversation()[:size])
        original = list(rt.run.messages)
        result = await rt.compact(selection=SummaryHistory(), strategy=Summarize(), token_counter=count_tokens)
        assert not result.applied and result.reason == "not_enough_messages"
        assert rt.run.messages == original and rt.run.model_calls == 0
    asyncio.run(scenario())


def test_recent_summary_and_nonshrinking_summary_leave_history_unchanged():
    async def scenario():
        rt = make_runtime(conversation())
        rt.run.messages[-2].metadata["compaction"] = {"kind": "summary"}
        original = list(rt.run.messages)
        skipped = await rt.compact(selection=SummaryHistory(), strategy=Summarize(), token_counter=count_tokens)
        assert skipped.reason == "recent_summary" and rt.run.model_calls == 0
        assert rt.run.messages == original
        rt = make_runtime(conversation(), ScriptedProvider([ModelResponse(text="Huge summary" * 500)]))
        original = list(rt.run.messages)
        rejected = await rt.compact(selection=SummaryHistory(), strategy=Summarize(), token_counter=count_tokens)
        assert rejected.reason == "not_smaller" and rt.run.model_calls == 1
        assert rt.run.messages == original
    asyncio.run(scenario())


@tool
def lookup(key: str) -> str:
    """Return verbose evidence for the selected key."""
    return key * 300


def test_react_summarizes_between_complete_tool_batches_and_checkpoints():
    async def scenario():
        agent = Agent(
            system_prompt="System",
            provider=ScriptedProvider([
                ModelResponse(tool_calls=(ToolCall(id="a", name="lookup", arguments={"key": "a"}),)),
                ModelResponse(tool_calls=(ToolCall(id="b", name="lookup", arguments={"key": "b"}),)),
                ModelResponse(tool_calls=(ToolCall(id="c", name="lookup", arguments={"key": "c"}),)),
                ModelResponse(text="Lookup a verified."),
                ModelResponse(text="Done"),
            ]),
            tools=Tools([lookup]),
            loop=ReActWithSummary(context_window=1000, max_output_tokens=100, threshold_percent=50,
                                 token_counter=count_tokens, summary=Summarize(prompt="Keep evidence")),
        )
        run = await agent.run("Investigate")
        assert run.status == "completed" and run.output == "Done"
        assert run.model_calls == 5 and run.tool_calls == 3
        requests = [e.data["request"] for e in run.events if e.kind == "model_request"]
        assert [r["purpose"] for r in requests] == ["act", "act", "act", "compact", "act"]
        assert requests[3]["messages"][0]["content"] == "Keep evidence"
        assert any(m["content"] == "Lookup a verified." for m in requests[4]["messages"])
        assert not any(m["tool_call_id"] == "a" for m in requests[4]["messages"])
        assert any(e.kind == "tool_result" and e.data["message"]["tool_call_id"] == "a" for e in run.events)
        validate_conversation(run.messages)
        assert list(agent.sessions.load(run.session_id).messages) == run.messages[1:]
        flow = agent.loop.to_flow()
        assert flow.start == "summarize" and flow.edges["tools", "default"] == "summarize"
        assert flow.edges["call_llm", "answer"] == "finish"
    asyncio.run(scenario())


@pytest.mark.parametrize("threshold,reserve,expected", [(100, 100, 1), (50, 100, 2), (100, 400, 2)])
def test_summary_triggers_on_percentage_or_output_reserve(threshold, reserve, expected):
    async def scenario():
        rt = make_runtime(conversation(), ScriptedProvider([ModelResponse(text="Small"), ModelResponse(text="Done")]))
        # 650 measured units crosses 50%, and also the 500-unit allowed input
        # with a 400-unit output reserve; it does not cross the 800-unit limit.
        async def counter(messages):
            return 650 if len(messages) == 7 else 100
        await ReActWithSummary(context_window=1000, max_output_tokens=reserve, threshold_percent=threshold,
                              token_counter=counter).run(rt)
        assert rt.run.model_calls == expected and rt.run.status == "completed"
    asyncio.run(scenario())


def test_summary_model_call_obeys_shared_run_limit():
    async def scenario():
        agent = Agent(provider=ScriptedProvider([ModelResponse(text="Small")]), limits=Limits(max_model_calls=1),
                      loop=ReActWithSummary(context_window=1000, max_output_tokens=100,
                                           token_counter=count_tokens, threshold_percent=50))
        run = await agent.run("Continue", history=conversation()[1:])
        assert run.status == "limit" and run.model_calls == 1
        assert any(e.kind == "context_compaction" for e in run.events)
    asyncio.run(scenario())


@pytest.mark.parametrize("kwargs", [
    {"context_window": 0}, {"context_window": True}, {"max_output_tokens": 900},
    {"max_output_tokens": -1}, {"keep_recent_messages": 0}, {"keep_recent_messages": True},
    {"threshold_percent": 4}, {"threshold_percent": 101}, {"threshold_percent": float("nan")},
    {"token_counter": None}, {"tool_concurrency": 0},
])
def test_summary_loop_validates_configuration(kwargs):
    options = {"context_window": 1000, "max_output_tokens": 100, "token_counter": count_tokens, **kwargs}
    with pytest.raises((ValueError, TypeError)):
        ReActWithSummary(**options)
