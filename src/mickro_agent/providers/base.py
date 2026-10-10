"""The unified provider contract; implementations are async generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Protocol

from ..types import ApiHandlerCreateMessageMetadata, MessageParam, ModelRequest, ModelResponse, StreamChunk


class Provider(Protocol):
    # This describes an async iterator, not a coroutine awaiting an iterator.
    def create_message(
        self,
        system_prompt: str,
        messages: list[MessageParam],
        metadata: ApiHandlerCreateMessageMetadata | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]: ...


class ApiHandler(ABC):
    @abstractmethod
    async def create_message(
        self,
        system_prompt: str,
        messages: list[MessageParam],
        metadata: ApiHandlerCreateMessageMetadata | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Yield normalized chunks; tool_use contains a complete tool call."""
        raise NotImplementedError
        yield  # Makes the abstract method an async generator, matching implementations.

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Convenience collector; create_message is the sole provider primitive."""
        from .streaming import collect_response

        return await collect_response(self, request)
