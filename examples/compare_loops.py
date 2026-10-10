"""Run two loop recipes without credentials: python examples/compare_loops.py."""

import asyncio

from mickro_agent import (
    END,
    Agent,
    Flow,
    ModelResponse,
    Runtime,
    ToolCall,
    Tools,
    finish_step,
    model_step,
    node,
    tool,
    tools_step,
)
from mickro_agent.providers import ScriptedProvider


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


async def main():
    action = ModelResponse(tool_calls=(ToolCall(id="multiply-1", name="multiply", arguments={"a": 6, "b": 7}),))
    answer = ModelResponse(text="42")
    for label, planning, script in [
        ("ReAct", False, [action, answer]),
        ("PlanThenAct", True, [ModelResponse(text="Multiply 6 by 7, then report the result."), action, answer]),
    ]:
        # Fresh graph handles per variant; Flow snapshots their wiring.
        call_llm = node(model_step, name="call_llm")
        tools = node(tools_step, name="tools")
        finish = node(finish_step, name="finish")

        call_llm - "tools" >> tools
        call_llm - "answer" >> finish
        tools >> call_llm
        finish >> END

        @node
        async def plan(runtime: Runtime) -> None:
            runtime.say("Write a concise action plan. Do not call tools yet.")
            response = await runtime.complete(allow_tools=False, purpose="plan")
            runtime.run.state["plan"] = response.text
            runtime.say("Execute the plan, then give the result.")

        if planning:
            plan >> call_llm
        loop = Flow(start=plan if planning else call_llm)
        agent = Agent(provider=ScriptedProvider(script), tools=Tools([multiply]), loop=loop)
        run = await agent.run("What is 6 times 7?")
        assert run.status == "completed", run.error
        print(f"{label:12} answer={run.output} model_calls={run.model_calls} tool_calls={run.tool_calls}")
        # Optional: run.write_events("artifacts/run.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
