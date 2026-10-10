"""The common execution boundary used by every loop variation."""

from __future__ import annotations

import asyncio
from typing import Protocol

from .providers.base import Provider
from .providers.streaming import collect_response
from .tools import Tools
from .types import Limits, Message, ModelRequest, ModelResponse, Run, ToolCall


class Loop(Protocol):
    async def run(self, runtime: Runtime) -> None: ...


class LimitReached(Exception):
    pass


class Runtime:
    def __init__(self, run: Run, provider: Provider, tools: Tools, limits: Limits):
        self.run = run
        self.provider = provider
        self.tools = tools
        self.limits = limits
        self._pending: tuple[ToolCall, ...] = ()

    def _active(self) -> None:
        if self.run.status != "running":
            raise RuntimeError("Run has already finished")

    def step(self, name: str, *, metadata: dict | None = None) -> None:
        self._active()
        if self.run.steps >= self.limits.max_steps:
            raise LimitReached("Graph step limit reached")
        self.run.steps += 1
        self.run.emit("step", node=name, step=self.run.steps, metadata=metadata or {})

    def say(self, content: str) -> None:
        self._active()
        if self._pending:
            raise ValueError("Resolve pending tool calls before adding an instruction")
        self.run.messages.append(Message(role="user", content=content))
        self.run.emit("instruction", content=content)

    async def complete(self, *, allow_tools: bool = True, purpose: str = "act") -> ModelResponse:
        self._active()
        if self._pending:
            raise ValueError("Resolve pending tool calls before another model turn")
        if self.run.model_calls >= self.limits.max_model_calls:
            raise LimitReached("Model call limit reached")
        self.run.model_calls += 1
        request = ModelRequest(
            run_id=self.run.id,
            messages=tuple(message.model_copy(deep=True) for message in self.run.messages),
            tools=self.tools.specs if allow_tools else (),
            turn=self.run.model_calls,
            purpose=purpose,
        )
        self.run.emit("model_request", request=request.model_dump(mode="json"))
        response = await collect_response(
            self.provider,
            request,
            on_chunk=lambda chunk: self.run.emit("model_chunk", turn=request.turn, chunk=chunk),
        )
        if response.tool_calls and not allow_tools:
            raise ValueError("Provider returned tool calls when tools were disabled")
        ids = [call.id for call in response.tool_calls]
        if len(ids) != len(set(ids)):
            raise ValueError("Provider returned duplicate tool call IDs")
        self._pending = tuple(call.model_copy(deep=True) for call in response.tool_calls)
        self.run.messages.append(
            Message(
                role="assistant",
                content=response.text,
                tool_calls=response.tool_calls,
                metadata={**response.metadata, **({"reasoning": response.reasoning} if response.reasoning else {})},
            ).model_copy(deep=True)
        )
        for key, value in response.usage.items():
            self.run.usage[key] = self.run.usage.get(key, 0) + value
        self.run.emit("model_response", response=response.model_dump(mode="json"))
        return response

    async def execute_tools(self, *, concurrency: int = 1) -> None:
        self._active()
        if concurrency < 1:
            raise ValueError("Tool concurrency must be positive")
        calls = self._pending
        if not calls:
            raise ValueError("No pending tool calls to execute")
        if self.run.tool_calls + len(calls) > self.limits.max_tool_calls:
            raise LimitReached("Tool call limit reached; batch was not executed")
        self.run.tool_calls += len(calls)
        semaphore = asyncio.Semaphore(concurrency)

        async def execute(call: ToolCall) -> Message:
            async with semaphore:
                self.run.emit("tool_start", call=call.model_dump(mode="json"))
                # A timed-out effect has an unknown outcome: stop the run, do not retry it.
                async with asyncio.timeout(self.limits.tool_timeout):
                    result = await self.tools.invoke(call, {**self.run.context, "runtime": self})
                message = Message(
                    role="tool", tool_call_id=call.id, content=result.content(), is_error=result.error is not None
                )
                self.run.emit("tool_result", message=message.model_dump(mode="json"))
                return message

        if concurrency == 1:
            for call in calls:
                self.run.messages.append(await execute(call))
        else:
            # TaskGroup cancels siblings on failure; results retain the provider's order.
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(execute(call)) for call in calls]
            self.run.messages.extend(task.result() for task in tasks)
        self._pending = ()

    def finish(self, output: str) -> None:
        self._active()
        if self._pending:
            raise ValueError("Resolve pending tool calls before finishing")
        self.run.output = output
        self.run.status = "completed"
