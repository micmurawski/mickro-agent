"""Loop recipes built from the same model/tool execution primitives."""

from dataclasses import dataclass, field

from .context_ops import Summarize, SummaryHistory, TokenCounter
from .flow import END, Flow, Node, node
from .runtime import Runtime


async def model_step(runtime: Runtime) -> str:
    response = await runtime.complete()
    return "tools" if response.tool_calls else "answer"


async def tools_step(runtime: Runtime) -> None:
    await runtime.execute_tools()


def finish_step(runtime: Runtime) -> None:
    runtime.finish(runtime.run.messages[-1].content)


def _react_nodes(tool_concurrency: int, *, before_model: Node | None = None):
    if tool_concurrency < 1:
        raise ValueError("Tool concurrency must be positive")

    call_llm = node(model_step, name="call_llm")
    finish = node(finish_step, name="finish")

    @node(name="tools")
    async def tools(runtime: Runtime) -> None:
        await runtime.execute_tools(concurrency=tool_concurrency)

    call_llm - "tools" >> tools
    call_llm - "answer" >> finish
    entry = call_llm if before_model is None else before_model
    if before_model is not None:
        before_model >> call_llm
    tools >> entry
    finish >> END
    return entry


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
class ReActWithSummary:
    """ReAct with a legacy-style summary window checked before every model turn.

    Supply a counter for the execution model, including request overhead as
    needed. Summarize at threshold_percent or within the 10% context buffer plus
    the output reserve. A summary is applied only when the full context shrinks.
    Skips leave history intact; errors propagate, with no removal fallback.
    """

    context_window: int
    token_counter: TokenCounter
    summary: Summarize = field(default_factory=Summarize)
    threshold_percent: float = 80
    max_output_tokens: int = 8192
    keep_recent_messages: int = 3
    tool_concurrency: int = 1

    def __post_init__(self) -> None:
        for name in ("context_window", "max_output_tokens", "tool_concurrency"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.threshold_percent, (int, float))
            or isinstance(self.threshold_percent, bool)
            or not 5 <= self.threshold_percent <= 100
        ):
            raise ValueError("threshold_percent must be between 5 and 100")
        if self.max_output_tokens >= self.context_window * 0.9:
            raise ValueError("Output reserve and 10% buffer must leave room for conversation")
        if not callable(self.token_counter):
            raise TypeError("token_counter must be an async callable")
        SummaryHistory(self.keep_recent_messages)

    def to_flow(self) -> Flow:
        @node(name="summarize")
        async def summarize(runtime: Runtime) -> None:
            tokens = await self.token_counter(tuple(m.model_copy(deep=True) for m in runtime.run.messages))
            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                raise ValueError("Token counter must return a nonnegative integer")
            allowed = self.context_window * 0.9 - self.max_output_tokens
            if tokens * 100 >= self.context_window * self.threshold_percent or tokens > allowed:
                await runtime.compact(
                    selection=SummaryHistory(self.keep_recent_messages),
                    strategy=self.summary,
                    token_counter=self.token_counter,
                )

        return Flow(start=_react_nodes(self.tool_concurrency, before_model=summarize))

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
