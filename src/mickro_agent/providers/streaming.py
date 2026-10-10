"""Shared stream collection for runtime turns and optional complete() calls."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

from ..types import ModelRequest, ModelResponse, StreamChunk, ToolCall
from .base import Provider


async def close_stream(stream: Any) -> None:
    close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
    if close is not None:
        await close()


def response_chunks(response: ModelResponse) -> Iterator[StreamChunk]:
    """Emit normalized chunks from a scripted/full response."""
    if response.reasoning:
        yield {"type": "reasoning", "text": response.reasoning}
    if response.text:
        yield {"type": "text", "text": response.text}
    for call in response.tool_calls:
        yield {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments, "metadata": call.metadata}
    if response.usage:
        yield {"type": "usage", **response.usage}
    if response.metadata:
        yield {"type": "metadata", "metadata": response.metadata}


async def collect_response(
    provider: Provider,
    request: ModelRequest,
    *,
    on_chunk: Callable[[StreamChunk], None] | None = None,
) -> ModelResponse:
    system_prompt = "\n\n".join(message.content for message in request.messages if message.role == "system")
    messages = [message.model_copy(deep=True) for message in request.messages if message.role != "system"]
    stream = provider.create_message(
        system_prompt=system_prompt,
        messages=messages,
        metadata={
            "run_id": request.run_id,
            "task_id": request.run_id,
            "turn": request.turn,
            "purpose": request.purpose,
        },
        tools=[spec.model_copy(deep=True) for spec in request.tools],
    )
    text, reasoning, calls, usage, metadata = [], [], [], {}, {}
    try:
        async for chunk in stream:
            kind = chunk["type"]
            if kind in {"text", "reasoning"}:
                value = chunk["text"]
                if not isinstance(value, str):
                    raise TypeError(f"{kind} chunk requires string text")
                (text if kind == "text" else reasoning).append(value)
            elif kind == "tool_use":
                calls.append(
                    ToolCall(
                        id=chunk["id"], name=chunk["name"], arguments=chunk["input"], metadata=chunk.get("metadata", {})
                    )
                )
            elif kind == "usage":
                # Usage chunks are increments, not repeated cumulative snapshots.
                for key, value in chunk.items():
                    if key == "type":
                        continue
                    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                        raise ValueError(f"Invalid usage counter: {key}")
                    usage[key] = usage.get(key, 0) + value
            elif kind == "metadata":
                metadata.update(chunk["metadata"])
            elif kind == "grounding":
                metadata.setdefault("sources", []).extend(chunk["sources"])
            else:
                raise ValueError(f"Unknown stream chunk type: {kind!r}")
            if on_chunk is not None:
                on_chunk(chunk)
    finally:
        await close_stream(stream)
    return ModelResponse(
        text="".join(text), reasoning="".join(reasoning), tool_calls=tuple(calls), usage=usage, metadata=metadata
    )
