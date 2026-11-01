"""Offline ReAct with rolling summaries; no external service or API key needed."""

import asyncio

from mickro_agent import Agent, ModelResponse, ReActWithSummary, Summarize, ToolCall, Tools, tool
from mickro_agent.providers import ScriptedProvider


@tool
def inspect_component(component: str) -> str:
    """Return a verbose, simulated diagnostic report."""
    return f"{component}: healthy. " * 30


async def demonstration_counter(messages) -> int:
    # Character-based test units make the offline trigger deterministic. In a
    # real agent, supply the execution model's token counter and request overhead.
    return sum(len(message.content) + 1 for message in messages)


def make_agent() -> Agent:
    return Agent(
        system_prompt="Inspect components and report the evidence.",
        tools=Tools([inspect_component]),
        provider=ScriptedProvider([
            ModelResponse(tool_calls=(ToolCall(id="a", name="inspect_component", arguments={"component": "A"}),)),
            ModelResponse(tool_calls=(ToolCall(id="b", name="inspect_component", arguments={"component": "B"}),)),
            ModelResponse(tool_calls=(ToolCall(id="c", name="inspect_component", arguments={"component": "C"}),)),
            ModelResponse(text="Component A was inspected and is healthy. No changes were made."),
            ModelResponse(text="A, B, and C are healthy. All three inspections completed; no changes were made."),
        ]),
        loop=ReActWithSummary(
            context_window=2000,
            max_output_tokens=200,
            threshold_percent=40,
            token_counter=demonstration_counter,
            summary=Summarize(prompt="Preserve inspected components, findings, and any actions already taken."),
        ),
    )


async def main() -> None:
    agent = make_agent()
    run = await agent.run("Inspect A, B, and C, then report their health.")
    assert run.status == "completed", run.error
    print(run.output)
    for event in run.events:
        if event.kind == "context_compaction":
            data = event.data
            print(f"Summary reduced context from {data['before_tokens']} to {data['after_tokens']} test units.")
    print(f"Model calls: {run.model_calls}; tool calls: {run.tool_calls}")
    print(agent.get_flow_graph())


if __name__ == "__main__":
    asyncio.run(main())
