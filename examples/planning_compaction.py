"""Offline plan → review → test assumptions → final plan → compact → ReAct.

Run with --summarize to make a separate, budgeted summarization call instead of
retaining the final plan directly. No external services or effects are required.
"""

import argparse
import asyncio

from mickro_agent import (
    END,
    Agent,
    Flow,
    ModelResponse,
    Replace,
    Summarize,
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
def check_capacity(items: int, workers: int) -> dict:
    """Check whether all items fit in one batch."""
    return {"fits": items <= workers, "items": items, "workers": workers}


@tool
def process_batch(items: list[str]) -> dict:
    """Simulate processing a batch of at most three items."""
    if len(items) > 3:
        raise ValueError("A batch may contain at most three items")
    return {"processed": items}


FINAL_PLAN = (
    "Process four items in two batches: [A, B, C], then [D]. "
    "check_capacity verified that four items do not fit with three workers. "
    "No items have been processed yet."
)


def planning_flow(*, summarize: bool = False) -> Flow:
    @node
    def begin(runtime):
        runtime.run.state["planning_start"] = runtime.mark_messages()

    @node
    async def plan(runtime):
        runtime.say("Plan how to process the four items.")
        await runtime.complete(allow_tools=False, purpose="plan")

    @node
    async def review(runtime):
        runtime.say("Review the plan and identify assumptions to test.")
        await runtime.complete(allow_tools=False, purpose="review")
        runtime.say("Test the capacity assumption with check_capacity, then explain the finding.")

    @node
    async def test_assumptions(runtime):
        response = await runtime.complete(purpose="test_assumptions")
        return "tools" if response.tool_calls else "done"

    assumption_tools = node(tools_step, name="assumption_tools")

    @node
    async def final_plan(runtime):
        runtime.say("Finalize the plan, preserving evidence, uncertainties, and actions already taken.")
        response = await runtime.complete(allow_tools=False, purpose="final_plan")
        runtime.run.state["final_plan"] = response.text

    @node
    async def compact(runtime):
        strategy = Summarize() if summarize else Replace(runtime.run.state["final_plan"])
        result = await runtime.compact_since(runtime.run.state.pop("planning_start"), strategy=strategy)
        runtime.run.emit("planning_handoff", applied=result.applied, strategy=result.strategy)
        runtime.say("Execute the final plan using process_batch, then report the result.")

    call_llm = node(model_step, name="call_llm")
    execute_tools = node(tools_step, name="execute_tools")
    respond = node(finish_step, name="respond")

    begin >> plan >> review >> test_assumptions
    test_assumptions - "tools" >> assumption_tools
    assumption_tools >> test_assumptions
    test_assumptions - "done" >> final_plan >> compact >> call_llm
    call_llm - "tools" >> execute_tools
    execute_tools >> call_llm
    call_llm - "answer" >> respond >> END
    return Flow(start=begin)


def make_agent(*, summarize: bool = False) -> Agent:
    responses = [
        ModelResponse(text="Draft: process A, B, C, and D in one batch."),
        ModelResponse(text="The draft assumes four items fit with three workers. Check this."),
        ModelResponse(tool_calls=(ToolCall(id="capacity", name="check_capacity",
                                         arguments={"items": 4, "workers": 3}),)),
        ModelResponse(text="Capacity check failed: four items need at least two batches."),
        ModelResponse(text=FINAL_PLAN),
    ]
    if summarize:
        responses.append(ModelResponse(text=FINAL_PLAN))
    responses.extend([
        ModelResponse(tool_calls=(ToolCall(id="batch1", name="process_batch", arguments={"items": ["A", "B", "C"]}),)),
        ModelResponse(tool_calls=(ToolCall(id="batch2", name="process_batch", arguments={"items": ["D"]}),)),
        ModelResponse(text="Processed all four items in two batches."),
    ])
    return Agent(
        provider=ScriptedProvider(responses), tools=Tools([check_capacity, process_batch]),
        loop=planning_flow(summarize=summarize),
    )


async def main(summarize: bool = False):
    run = await make_agent(summarize=summarize).run("Process A, B, C, D with three workers.")
    print(f"{run.status}: {run.output}")
    print(f"Model calls: {run.model_calls}; tool calls: {run.tool_calls}")
    print("\nConversation retained for continuation:")
    for message in run.messages:
        print(f"  {message.role}: {message.content}")
    event = next(event for event in run.events if event.kind == "context_compaction")
    print(f"\nFull trace retains {len(event.data['removed_messages'])} compacted planning messages.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", action="store_true")
    asyncio.run(main(parser.parse_args().summarize))
