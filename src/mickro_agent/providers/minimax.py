"""MiniMax streaming: https://platform.minimax.io/docs/api-reference/text-anthropic-api

Full assistant content is retained for tool turns, including thinking/signatures.
The Anthropic SDK is imported only when creating a client.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

from ..types import ApiHandlerCreateMessageMetadata, MessageParam, StreamChunk, ToolCall, ToolSpec
from .base import ApiHandler

BASE_URL = "https://api.minimax.io/anthropic"
_CONTENT_KEY = "minimax_content"
_RESERVED = {"model", "messages", "system", "stream", "tools"}


def _messages(messages: list[MessageParam]) -> list[dict]:
    formatted = []
    for message in messages:
        if message.role == "system":
            raise ValueError("Pass system messages through system_prompt")
        role = "user" if message.role == "tool" else message.role
        if message.role == "tool":
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                    "is_error": message.is_error,
                }
            ]
        elif message.role == "assistant" and _CONTENT_KEY in message.metadata:
            content = deepcopy(message.metadata[_CONTENT_KEY])
            if not isinstance(content, list) or not all(isinstance(block, dict) for block in content):
                raise TypeError("minimax_content must be a list of content blocks")
        else:
            content = [{"type": "text", "text": message.content}] if message.content else []
            content.extend(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                for call in message.tool_calls
            )
        if formatted and role == "user" and formatted[-1]["role"] == "user":
            formatted[-1]["content"].extend(content)
        else:
            formatted.append({"role": role, "content": content})
    return formatted


class MiniMax(ApiHandler):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        max_tokens: int = 4096,
        client: Any = None,
        **options: Any,
    ):
        if _RESERVED & options.keys():
            raise ValueError(f"Reserved model options: {sorted(_RESERVED & options.keys())}")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self.model = model
        self.options = {"max_tokens": max_tokens, **options}
        self._owns_client = client is None
        if client is None:
            api_key = api_key or os.environ.get("MINIMAX_API_KEY")
            if not api_key:
                raise ValueError("Pass api_key or set MINIMAX_API_KEY")
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise ImportError("Install mickro-agent[minimax] to use MiniMax") from exc
            client = AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=0)
        self.client = client

    async def create_message(
        self,
        system_prompt: str,
        messages: list[MessageParam],
        metadata: ApiHandlerCreateMessageMetadata | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        specs = kwargs.pop("tools", ())
        options = {**self.options, **kwargs}
        if _RESERVED & options.keys():
            raise ValueError("model, messages, system and stream are controlled by the adapter")
        if specs:
            definitions = [ToolSpec.model_validate(spec) for spec in specs]
            options["tools"] = [
                {"name": spec.name, "description": spec.description, "input_schema": spec.parameters}
                for spec in definitions
            ]
        else:
            options.pop("tool_choice", None)
        # Runtime metadata is local bookkeeping, not an Anthropic metadata payload.
        async with self.client.messages.stream(
            model=self.model,
            system=system_prompt,
            messages=_messages(messages),
            **options,
        ) as stream:
            stopped = False
            tool_json: dict[int, str] = {}
            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "text" and block.text:
                        yield {"type": "text", "text": block.text}
                    elif block.type == "thinking" and block.thinking:
                        yield {"type": "reasoning", "text": block.thinking}
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield {"type": "text", "text": delta.text}
                    elif delta.type == "thinking_delta":
                        yield {"type": "reasoning", "text": delta.thinking}
                    elif delta.type == "input_json_delta":
                        tool_json[event.index] = tool_json.get(event.index, "") + delta.partial_json
                elif event.type == "message_stop":
                    stopped = True
                elif event.type == "error":
                    raise RuntimeError(f"MiniMax stream error: {event.error}")
                # Ignore SDK convenience events to avoid duplicating raw deltas.
            if not stopped:
                raise ValueError("Incomplete MiniMax stream: missing message_stop")
            response = await stream.get_final_message()
            if response.stop_reason not in {"end_turn", "tool_use", "stop_sequence"}:
                raise ValueError(f"Incomplete MiniMax response: {response.stop_reason}")
            content = [block.model_dump(mode="json", exclude_none=True) for block in response.content]
            for index, block in enumerate(content):
                if block["type"] != "tool_use":
                    continue
                arguments = json.loads(tool_json[index]) if index in tool_json else block["input"]
                if not isinstance(arguments, dict):
                    raise TypeError("Tool arguments must be a JSON object")
                call = ToolCall(id=block["id"], name=block["name"], arguments=arguments)
                yield {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            usage = {}
            for source, target in (
                ("input_tokens", "input_tokens"),
                ("output_tokens", "output_tokens"),
                ("cache_read_input_tokens", "cache_read_tokens"),
                ("cache_creation_input_tokens", "cache_write_tokens"),
            ):
                value = getattr(response.usage, source, None)
                if value is not None:
                    usage[target] = value
            yield {"type": "usage", **usage}
            yield {
                "type": "metadata",
                "metadata": {
                    "model": response.model,
                    "stop_reason": response.stop_reason,
                    _CONTENT_KEY: content,
                },
            }

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()
