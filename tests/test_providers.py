import asyncio
from types import SimpleNamespace as NS

import pytest

from mickro_agent import Agent, Message, ModelRequest, ToolCall, Tools, ToolSpec, tool
from mickro_agent.legacy import LegacyProvider, migrate_tool
from mickro_agent.providers import OpenAIChat


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def close(self):
        self.closed = True


def delta(*, text=None, calls=None, finish=None, reasoning=None):
    return NS(
        choices=[
            NS(
                finish_reason=finish,
                delta=NS(
                    content=text,
                    tool_calls=calls,
                    reasoning_content=reasoning,
                ),
            )
        ],
        usage=None,
        model="test-model",
    )


def tool_delta(index, *, id=None, name=None, arguments=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=arguments))


def test_openai_chat_roundtrip_with_injected_client():
    captured = []
    stream = FakeStream(
        [
            delta(text="Looking ", calls=[tool_delta(0, id="c1", name="look", arguments='{"key":')]),
            delta(text="up", calls=[tool_delta(0, name="up", arguments='"a"}')], finish="tool_calls"),
            NS(choices=[], usage=NS(prompt_tokens=10, completion_tokens=2)),
        ]
    )

    async def create(**kwargs):
        captured.append(kwargs)
        return stream

    client = NS(chat=NS(completions=NS(create=create)))
    provider = OpenAIChat("test-model", client=client, temperature=0.1)
    response = asyncio.run(
        provider.complete(
            ModelRequest(
                messages=(
                    Message(role="system", content="Be useful"),
                    Message(role="assistant", tool_calls=(ToolCall(id="old", name="lookup", arguments={"key": "b"}),)),
                    Message(role="tool", content="old result", tool_call_id="old"),
                ),
                tools=(ToolSpec(name="lookup", description="Lookup", parameters={"type": "object"}),),
                turn=1,
            )
        )
    )
    assert response.tool_calls[0].arguments == {"key": "a"}
    assert response.text == "Looking up" and stream.closed
    assert response.usage == {"input_tokens": 10, "output_tokens": 2}
    assert captured[0]["messages"][1]["tool_calls"][0]["id"] == "old"
    assert captured[0]["messages"][2]["tool_call_id"] == "old"
    assert captured[0]["tools"][0]["function"]["name"] == "lookup"
    assert captured[0]["temperature"] == 0.1
    assert captured[0]["stream"] is True
    assert captured[0]["stream_options"]["include_usage"] is True
    asyncio.run(provider.aclose())  # Does not close caller-owned client.


def test_openai_planning_omits_tools_and_truncation_fails():
    stream = FakeStream([delta(finish="length")])

    async def create(**kwargs):
        assert "tools" not in kwargs
        return stream

    provider = OpenAIChat("test", client=NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(ValueError, match="Incomplete"):
        asyncio.run(provider.complete(ModelRequest(messages=(), turn=1)))
    assert stream.closed


def test_legacy_provider_groups_results_and_preserves_signatures():
    class Handler:
        provider = "gemini"

        async def create_message(self, **kwargs):
            self.kwargs = kwargs
            yield {"type": "text", "text": "Look up"}
            yield {
                "type": "tool_use",
                "id": "new",
                "name": "lookup",
                "input": {"key": "a"},
                "thought_signature": b"opaque\xff",
            }
            yield {"type": "usage", "input_tokens": 3}

    handler = Handler()
    adapter = LegacyProvider(handler)
    response = asyncio.run(adapter.complete(ModelRequest(messages=(Message(role="user", content="Task"),), turn=1)))
    asyncio.run(
        adapter.complete(
            ModelRequest(
                messages=(
                    Message(role="system", content="System"),
                    Message(role="assistant", tool_calls=response.tool_calls + (ToolCall(id="other", name="lookup"),)),
                    Message(role="tool", tool_call_id="new", content="a"),
                    Message(role="tool", tool_call_id="other", content="b", is_error=True),
                ),
                turn=2,
            )
        )
    )
    assert handler.kwargs["system_prompt"] == "System"
    messages = handler.kwargs["messages"]
    assert messages[0]["content"][0]["thought_signature"] == b"opaque\xff"
    assert len(messages[1]["content"]) == 2
    assert messages[1]["content"][1]["is_error"] is True
    assert response.usage == {"input_tokens": 3}


def test_legacy_tool_keeps_injection_and_converts_result():
    def original(key: str, context: dict):
        return NS(result=context[key], error=None)

    OldTool = type(
        "OldTool",
        (),
        {
            "_func": original,
            "name": "lookup",
            "tags": {"read"},
            "_inject_params": ["context"],
        },
    )
    migrated = migrate_tool(OldTool())
    assert "context" not in migrated.spec.parameters["properties"]
    result = asyncio.run(migrated.invoke({"key": "a"}, {"context": {"a": "found"}}))
    assert result.result == "found" and result.error is None
    rejected = asyncio.run(migrated.invoke({"key": "a", "context": {"a": "bad"}}, {"context": {"a": "good"}}))
    assert rejected.error


def test_direct_stream_yields_text_before_tool_assembly_and_can_be_closed():
    stream = FakeStream(
        [
            delta(text="First token", calls=[tool_delta(0, id="c1", name="lookup", arguments='{"key":')]),
            delta(calls=[tool_delta(0, arguments='"value"}')], finish="tool_calls"),
        ]
    )

    async def create(**kwargs):
        return stream

    async def consume():
        provider = OpenAIChat("test", client=NS(chat=NS(completions=NS(create=create))))
        chunks = provider.create_message("System", [Message(role="user", content="Task")])
        assert await anext(chunks) == {"type": "text", "text": "First token"}
        assert not stream.closed
        await chunks.aclose()
        assert stream.closed

    asyncio.run(consume())


def test_interleaved_tool_fragments_preserve_order_and_reasoning():
    stream = FakeStream(
        [
            delta(reasoning="Check ", calls=[tool_delta(1, id="b", name="second", arguments='{"b":')]),
            delta(reasoning="inputs", calls=[tool_delta(0, id="a", name="first", arguments='{"a":1}')]),
            delta(calls=[tool_delta(1, arguments="2}")], finish="tool_calls"),
        ]
    )

    async def create(**kwargs):
        return stream

    provider = OpenAIChat("test", client=NS(chat=NS(completions=NS(create=create))))
    response = asyncio.run(provider.complete(ModelRequest(messages=(), turn=1)))
    assert [call.id for call in response.tool_calls] == ["a", "b"]
    assert response.tool_calls[1].arguments == {"b": 2}
    assert response.reasoning == "Check inputs"


def test_runtime_consumes_create_message_only_and_records_each_chunk():
    class StreamingOnly:
        async def create_message(self, system_prompt, messages, metadata=None, **kwargs):
            assert system_prompt == "Chef"
            assert [message.role for message in messages] == ["user"]
            assert metadata["turn"] == 1 and metadata["run_id"]
            yield {"type": "reasoning", "text": "Checking"}
            yield {"type": "text", "text": "Dinner "}
            yield {"type": "text", "text": "ready"}
            yield {"type": "usage", "input_tokens": 3, "output_tokens": 2}
            yield {"type": "usage", "output_tokens": 1}

    run = asyncio.run(Agent(provider=StreamingOnly(), system_prompt="Chef").run("Cook"))
    assert run.status == "completed" and run.output == "Dinner ready"
    assert run.usage == {"input_tokens": 3, "output_tokens": 3}
    assert run.messages[-1].metadata["reasoning"] == "Checking"
    assert len([event for event in run.events if event.kind == "model_chunk"]) == 5


def test_failed_stream_never_executes_partial_tool_turn():
    invocations = []
    closed = []

    @tool
    def action():
        invocations.append(1)

    class Provider:
        async def create_message(self, **kwargs):
            try:
                yield {"type": "tool_use", "id": "c1", "name": "action", "input": {}}
                raise ConnectionError("stream interrupted")
            finally:
                closed.append(True)

    run = asyncio.run(Agent(provider=Provider(), tools=Tools([action])).run("Task"))
    assert run.status == "failed" and "stream interrupted" in run.error
    assert not invocations and closed == [True]
    assert not any(message.role == "assistant" for message in run.messages)


def test_cancelling_a_turn_closes_provider_generator():
    closed = []

    class Provider:
        async def create_message(self, **kwargs):
            try:
                yield {"type": "text", "text": "Starting"}
                await asyncio.Event().wait()
            finally:
                closed.append(True)

    async def cancel():
        task = asyncio.create_task(Agent(provider=Provider()).run("Task"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    assert closed == [True]


@pytest.mark.parametrize("fragment,finish", [('{"a":', "tool_calls"), ("{}", None)])
def test_malformed_or_unfinished_openai_stream_fails(fragment, finish):
    stream = FakeStream([delta(calls=[tool_delta(0, id="c1", name="tool", arguments=fragment)], finish=finish)])

    async def create(**kwargs):
        return stream

    provider = OpenAIChat("test", client=NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(ValueError):
        asyncio.run(provider.complete(ModelRequest(messages=(), turn=1)))
    assert stream.closed
