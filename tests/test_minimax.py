import asyncio
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from mickro_agent import Agent, Message, ModelRequest, Tools, tool
from mickro_agent.providers import MiniMax


def block(**data):
    return NS(**data, model_dump=lambda **kwargs: deepcopy(data))


def delta(kind, index=0, **data):
    return NS(type="content_block_delta", index=index, delta=NS(type=kind, **data))


class Stream:
    def __init__(self, content, events=(), *, stop_reason="end_turn", stopped=True):
        self.closed = False
        self.events = list(events) + ([NS(type="message_stop")] if stopped else [])
        self.response = NS(
            content=content,
            stop_reason=stop_reason,
            model="MiniMax-test",
            usage=NS(input_tokens=12, output_tokens=5, cache_read_input_tokens=3, cache_creation_input_tokens=2),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def __aiter__(self):
        for event in self.events:
            yield event

    async def get_final_message(self):
        return self.response


class Client:
    def __init__(self, *streams):
        self.streams = iter(streams)
        self.requests = []
        self.messages = NS(stream=self.stream)
        self.closed = False

    def stream(self, **kwargs):
        self.requests.append(kwargs)
        return next(self.streams)

    async def close(self):
        self.closed = True


def test_streams_text_reasoning_once_and_accounts_final_usage():
    content = [
        block(type="thinking", thinking="Check inputs", signature="opaque"),
        block(type="text", text="Hello chef"),
    ]
    stream = Stream(
        content,
        [
            delta("thinking_delta", thinking="Check inputs"),
            delta("text_delta", index=1, text="Hello "),
            NS(type="text", text="Hello "),  # SDK synthetic event must not duplicate text.
            delta("text_delta", index=1, text="chef"),
            NS(type="message_delta", usage=NS(output_tokens=5)),
        ],
    )
    client = Client(stream)
    provider = MiniMax("MiniMax-test", client=client, temperature=0.7)
    response = asyncio.run(provider.complete(ModelRequest(messages=(Message(role="user", content="Hi"),), turn=1)))
    assert response.text == "Hello chef" and response.reasoning == "Check inputs"
    assert response.usage == {"input_tokens": 12, "output_tokens": 5, "cache_read_tokens": 3, "cache_write_tokens": 2}
    assert response.metadata["minimax_content"][0]["signature"] == "opaque"
    assert client.requests[0]["max_tokens"] == 4096 and client.requests[0]["temperature"] == 0.7
    assert stream.closed
    asyncio.run(provider.aclose())
    assert not client.closed  # Injected clients remain caller-owned.


def test_full_tool_roundtrip_preserves_thinking_and_groups_parallel_results():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    original = [
        block(type="thinking", thinking="Calculate both sums", signature="keep-this"),
        block(type="text", text="Calculating"),
        block(type="tool_use", id="a", name="add", input={"a": 1, "b": 2}),
        block(type="tool_use", id="b", name="add", input={"a": 3, "b": 4}),
    ]
    first = Stream(
        original,
        [
            delta("thinking_delta", thinking="Calculate both sums"),
            delta("text_delta", index=1, text="Calculating"),
            delta("input_json_delta", index=2, partial_json='{"a":1,'),
            delta("input_json_delta", index=3, partial_json='{"a":3,"b":4}'),
            delta("input_json_delta", index=2, partial_json='"b":2}'),
        ],
        stop_reason="tool_use",
    )
    second = Stream([block(type="text", text="3 and 7")], [delta("text_delta", text="3 and 7")])
    client = Client(first, second)
    run = asyncio.run(Agent(provider=MiniMax("MiniMax-test", client=client), tools=Tools([add])).run("Add"))
    assert run.status == "completed" and run.output == "3 and 7"
    assert run.tool_calls == 2
    assert "input_schema" in client.requests[0]["tools"][0]
    messages = client.requests[1]["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == [item.model_dump() for item in original]
    assert [item["tool_use_id"] for item in messages[2]["content"]] == ["a", "b"]
    assert [item["content"] for item in messages[2]["content"]] == ["3", "7"]
    assert first.closed and second.closed


@pytest.mark.parametrize("stop_reason,stopped", [("max_tokens", True), (None, True), ("end_turn", False)])
def test_truncated_streams_fail_and_close(stop_reason, stopped):
    stream = Stream(
        [block(type="text", text="Partial")],
        [delta("text_delta", text="Partial")],
        stop_reason=stop_reason,
        stopped=stopped,
    )
    run = asyncio.run(Agent(provider=MiniMax("test", client=Client(stream))).run("Task"))
    assert run.status == "failed" and "Incomplete MiniMax" in run.error
    assert run.output is None and stream.closed


def test_incomplete_tool_json_is_not_executed():
    stream = Stream(
        [block(type="tool_use", id="a", name="action", input={"a": 1})],
        [delta("input_json_delta", partial_json='{"a":1')],
        stop_reason="tool_use",
    )
    run = asyncio.run(Agent(provider=MiniMax("test", client=Client(stream))).run("Task"))
    assert run.status == "failed" and run.tool_calls == 0 and stream.closed


def test_direct_stream_is_incremental_and_closes_early():
    stream = Stream(
        [block(type="text", text="One two")], [delta("text_delta", text="One "), delta("text_delta", text="two")]
    )
    provider = MiniMax("test", client=Client(stream))

    async def consume():
        chunks = provider.create_message("System", [Message(role="user", content="Hi")])
        assert await anext(chunks) == {"type": "text", "text": "One "}
        assert not stream.closed
        await chunks.aclose()
        assert stream.closed

    asyncio.run(consume())


def test_no_tools_and_request_options():
    stream = Stream([block(type="text", text="Plan")], [delta("text_delta", text="Plan")])
    client = Client(stream)
    provider = MiniMax("test", client=client, tool_choice={"type": "auto"})

    async def consume():
        return [
            chunk
            async for chunk in provider.create_message(
                "System",
                [Message(role="user", content="Plan")],
                tools=[],
                max_tokens=8192,
                temperature=0.5,
            )
        ]

    asyncio.run(consume())
    request = client.requests[0]
    assert request["max_tokens"] == 8192 and request["temperature"] == 0.5
    assert "tools" not in request and "tool_choice" not in request


def test_owned_client_uses_minimax_key_endpoint_and_no_retries(monkeypatch):
    import sys

    constructed = []
    client = Client()

    def create(**kwargs):
        constructed.append(kwargs)
        return client

    monkeypatch.setenv("MINIMAX_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "anthropic", NS(AsyncAnthropic=create))
    provider = MiniMax("test")
    assert constructed == [{"api_key": "test-only", "base_url": "https://api.minimax.io/anthropic", "max_retries": 0}]
    asyncio.run(provider.aclose())
    assert client.closed


def test_missing_key_has_clear_error(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        MiniMax("test")
