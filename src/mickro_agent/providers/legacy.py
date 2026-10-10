"""Streaming bridge for incident-assistant ApiHandler implementations."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any

from ..types import ApiHandlerCreateMessageMetadata, MessageParam, StreamChunk, ToolSpec
from .base import ApiHandler
from .streaming import close_stream


def _tool_definitions(specs: tuple[ToolSpec, ...], provider: str) -> list[dict]:
    definitions = [spec.model_dump() for spec in specs]
    if provider in {"anthropic", "minimax"}:
        return [
            {"name": item["name"], "description": item["description"], "input_schema": item["parameters"]}
            for item in definitions
        ]
    if provider == "openai_responses":
        return [{"type": "function", **item} for item in definitions]
    if provider == "gemini":

        def strip(value):
            if isinstance(value, dict):
                return {key: strip(item) for key, item in value.items() if key != "additionalProperties"}
            if isinstance(value, list):
                return [strip(item) for item in value]
            return value

        return strip(definitions)
    if provider in {"openai", "ollama", "openrouter", "groq", "ovh"}:
        return [{"type": "function", "function": item} for item in definitions]
    raise ValueError(f"Unsupported legacy tool format: {provider!r}")


class LegacyProvider(ApiHandler):
    """Use a fresh legacy handler per run; clients remain owned by the caller."""

    def __init__(self, handler: Any):
        self.handler = handler
        self._run_id: str | None = None

    async def create_message(
        self,
        system_prompt: str,
        messages: list[MessageParam],
        metadata: ApiHandlerCreateMessageMetadata | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        run_id = (metadata or {}).get("run_id", "")
        if self._run_id is None:
            self._run_id = run_id
        elif run_id != self._run_id:
            raise ValueError("Create a fresh legacy handler/adapter per run; handlers can retain conversation state")
        formatted = []
        for message in messages:
            if message.role == "system":
                raise ValueError("Pass system messages through system_prompt")
            if message.role == "tool":
                item = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                            "is_error": message.is_error,
                        }
                    ],
                }
                if formatted and formatted[-1].get("_tool_results"):
                    formatted[-1]["content"].extend(item["content"])
                else:
                    formatted.append({**item, "_tool_results": True})
                continue
            content = [{"type": "text", "text": message.content}] if message.content else []
            for call in message.tool_calls:
                block = {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                signature = call.metadata.get("thought_signature_base64")
                if signature is not None:
                    block["thought_signature"] = base64.b64decode(signature)
                content.append(block)
            formatted.append({"role": message.role, "content": content or ""})
        for item in formatted:
            item.pop("_tool_results", None)
        specs = tuple(ToolSpec.model_validate(spec) for spec in kwargs.pop("tools", ()))
        stream = self.handler.create_message(
            system_prompt=system_prompt,
            messages=formatted,
            metadata=metadata,
            tools=_tool_definitions(specs, self.handler.provider),
            **kwargs,
        )
        try:
            async for chunk in stream:
                kind = chunk["type"]
                if kind in {"text", "reasoning"}:
                    yield {"type": kind, "text": chunk.get("text") or chunk.get("content", "")}
                elif kind == "tool_use":
                    call_metadata = dict(chunk.get("metadata", {}))
                    signature = chunk.get("thought_signature")
                    if signature is not None:
                        call_metadata["thought_signature_base64"] = base64.b64encode(signature).decode()
                    yield {
                        "type": "tool_use",
                        "id": chunk["id"],
                        "name": chunk["name"],
                        "input": chunk["input"],
                        "metadata": call_metadata,
                    }
                elif kind == "usage":
                    yield {key: value for key, value in chunk.items() if value is not None}
                else:
                    yield chunk
        finally:
            await close_stream(stream)
