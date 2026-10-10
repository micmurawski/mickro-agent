"""Insert a review step before accepting an answer, with one revision at most."""

import asyncio

from mickro_agent import END, Agent, Flow, ModelResponse, Runtime, finish_step, model_step, node, tools_step
from mickro_agent.providers import ScriptedProvider


@node(metadata={"description": "Review the first proposed answer once"})
async def review(runtime: Runtime) -> str:
    if runtime.run.state.get("reviewed"):
        return "accept"
    runtime.run.state["reviewed"] = True
    runtime.say("Check your proposed answer for errors. Return a corrected final answer if necessary.")
    return "revise"


call_llm = node(model_step, name="call_llm")
tools = node(tools_step, name="tools")
finish = node(finish_step, name="finish")

call_llm - "tools" >> tools
call_llm - "answer" >> review
tools >> call_llm
review - "revise" >> call_llm
review - "accept" >> finish
finish >> END

loop = Flow(start=call_llm)


async def main():
    agent = Agent(
        loop=loop,
        provider=ScriptedProvider(
            [
                ModelResponse(text="The result is 41."),
                ModelResponse(text="The corrected result is 42."),
            ]
        ),
    )
    run = await agent.run("What is 6 times 7?")
    assert run.status == "completed", run.error
    print(run.output)
    print(" -> ".join(event.data["node"] for event in run.events if event.kind == "step"))


if __name__ == "__main__":
    asyncio.run(main())
