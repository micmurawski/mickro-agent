from collections.abc import AsyncIterator, Sequence
from typing import Any

from ..types import ApiHandlerCreateMessageMetadata, MessageParam, ModelResponse, StreamChunk
from .base import ApiHandler
from .streaming import response_chunks


class ScriptedProvider(ApiHandler):
    """Per-turn script; metadata.turn selects a response without a shared cursor."""

    def __init__(self, responses: Sequence[ModelResponse]):
        self.responses = tuple(response.model_copy(deep=True) for response in responses)

    async def create_message(
        self,
        system_prompt: str,
        messages: list[MessageParam],
        metadata: ApiHandlerCreateMessageMetadata | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        turn = (metadata or {}).get("turn", 1)
        if not 1 <= turn <= len(self.responses):
            raise ValueError(f"Script exhausted at model turn {turn}")
        for chunk in response_chunks(self.responses[turn - 1].model_copy(deep=True)):
            yield chunk
