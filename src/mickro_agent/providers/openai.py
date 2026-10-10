"""OpenAI-compatible streaming chat; the SDK is an optional dependency."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..types import ApiHandlerCreateMessageMetadata, MessageParam, StreamChunk, ToolCall, ToolSpec
from .base import ApiHandler
from .streaming import close_stream


class OpenAIChat(ApiHandler):
    def __init__(
        self, model: str, *, client: Any = None, api_key: str | None = None, base_url: str | None = None, **options: Any
    ):
        reserved = {"model", "messages", "tools", "stream", "n"} & options.keys()
        if reserved:
            raise ValueError(f"Reserved model options: {sorted(reserved)}")
        self.model, self.options = model, options
        self._owns_client = client is None
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError("Install mickro-agent[openai] to use OpenAIChat") from exc
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.client = client

    async def create_message(
        self,
        system_prompt: str,
        messages: list[MessageParam],
        metadata: ApiHandlerCreateMessageMetadata | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        formatted = [{"role": "system", "content": system_prompt}] if system_prompt else []
        for message in messages:
            if message.role == "system":
                raise ValueError("Pass system messages through system_prompt")
            item: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            formatted.append(item)
        options = {**self.options, **kwargs}
        if {"model", "messages", "stream", "n"} & options.keys():
            raise ValueError("model, messages, stream and n are controlled by the adapter")
        specs = options.pop("tools", ())
        if specs:
            options["tools"] = [
                {"type": "function", "function": ToolSpec.model_validate(spec).model_dump()} for spec in specs
            ]
        else:
            options.pop("tool_choice", None)
            options.pop("parallel_tool_calls", None)
        options["stream_options"] = {**options.get("stream_options", {}), "include_usage": True}
        stream = await self.client.chat.completions.create(model=self.model, messages=formatted, stream=True, **options)
        buffers: dict[int, dict[str, str]] = {}
        finish_reason, usage, model = None, None, self.model
        try:
            async for chunk in stream:
                model = getattr(chunk, "model", None) or model
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta.content:
                    yield {"type": "text", "text": delta.content}
                reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield {"type": "reasoning", "text": reasoning}
                for call in delta.tool_calls or ():
                    buf = buffers.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                    if call.id:
                        buf["id"] = call.id
                    if call.function is not None:
                        buf["name"] += call.function.name or ""
                        buf["arguments"] += call.function.arguments or ""
            if finish_reason not in {"stop", "tool_calls"}:
                raise ValueError(f"Incomplete model response: {finish_reason}")
            for index in sorted(buffers):
                buf = buffers[index]
                arguments = json.loads(buf["arguments"] or "{}")
                if not isinstance(arguments, dict):
                    raise TypeError("Tool arguments must be a JSON object")
                call = ToolCall(id=buf["id"], name=buf["name"], arguments=arguments)
                yield {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            if usage is not None:
                yield {
                    "type": "usage",
                    "input_tokens": usage.prompt_tokens or 0,
                    "output_tokens": usage.completion_tokens or 0,
                }
            yield {"type": "metadata", "metadata": {"finish_reason": finish_reason, "model": model}}
        finally:
            await close_stream(stream)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()
