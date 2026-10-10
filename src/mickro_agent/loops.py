"""Loop recipes built from the same model/tool execution primitives."""

from dataclasses import dataclass

from .flow import END, Flow, node
from .runtime import Runtime


async def model_step(runtime: Runtime) -> str:
    response = await runtime.complete()
    return "tools" if response.tool_calls else "answer"


async def tools_step(runtime: Runtime) -> None:
    await runtime.execute_tools()


def finish_step(runtime: Runtime) -> None:
    runtime.finish(runtime.run.messages[-1].content)


def _react_nodes(tool_concurrency: int):
    if tool_concurrency < 1:
        raise ValueError("Tool concurrency must be positive")

    call_llm = node(model_step, name="call_llm")
    finish = node(finish_step, name="finish")

    @node(name="tools")
    async def tools(runtime: Runtime) -> None:
        await runtime.execute_tools(concurrency=tool_concurrency)

    call_llm - "tools" >> tools
    call_llm - "answer" >> finish
    tools >> call_llm
    finish >> END
    return call_llm


def react_flow(*, tool_concurrency: int = 1) -> Flow:
    return Flow(start=_react_nodes(tool_concurrency))


@dataclass(frozen=True)
class ReAct:
    tool_concurrency: int = 1

    def to_flow(self) -> Flow:
        return react_flow(tool_concurrency=self.tool_concurrency)

    async def run(self, runtime: Runtime) -> None:
        await self.to_flow().run(runtime)


@dataclass(frozen=True)
class PlanThenAct:
    planning_prompt: str = "Write a concise action plan for the task. Do not call tools yet."
    execution_prompt: str = "Execute the plan using the available tools, then give the result."
    tool_concurrency: int = 1

    def to_flow(self) -> Flow:
        @node
        async def plan(runtime: Runtime) -> None:
            runtime.say(self.planning_prompt)
            response = await runtime.complete(allow_tools=False, purpose="plan")
            runtime.run.state["plan"] = response.text
            runtime.say(self.execution_prompt)

        call_llm = _react_nodes(self.tool_concurrency)
        plan >> call_llm
        return Flow(start=plan)

    async def run(self, runtime: Runtime) -> None:
        await self.to_flow().run(runtime)
