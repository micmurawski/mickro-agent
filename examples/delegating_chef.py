"""Offline DSL example: a chef delegates, wakes on completion, then resumes the child."""

import argparse
import asyncio
import json

from mickro_agent import (
    AGENT_JOB_TOOLS,
    END,
    Agent,
    AgentJobs,
    Flow,
    Runtime,
    agent_tool,
    finish_step,
    model_step,
    node,
    tools_step,
)


class SousChef:
    async def create_message(self, messages, **kwargs):
        requests = [m.content for m in messages if m.role == "user"]
        text = "Menu: pasta with tomato sauce."
        if len(requests) > 1:
            text += f" Updated after '{requests[-1]}': use gluten-free pasta."
        yield {"type": "text", "text": text}


class HeadChef:
    """A deterministic policy demonstrating delegation, not a live model."""

    async def create_message(self, messages, metadata, **kwargs):
        last = messages[-1]
        if last.role == "tool":
            handle = json.loads(last.content)
            yield {"type": "text", "text": f"Sous chef working: {handle['job_id']}. Listening for updates."}
            return
        event = json.loads(last.content) if last.content.startswith("{") else None
        if event is None:
            arguments = {"prompt": "Plan a pasta dinner."}
        elif sum(m.role == "user" and '"AgentFinished"' in m.content for m in messages) == 1:
            arguments = {"prompt": "Make that dinner gluten-free.", "session_id": event["session_id"]}
        else:
            yield {"type": "text", "text": f"Dinner plan ready: {event['output']}"}
            return
        yield {"type": "tool_use", "id": f"delegate-{metadata['turn']}", "name": "ask_sous_chef", "input": arguments}


def child_loop():
    call_llm = node(model_step, name="call_llm")
    tools = node(tools_step, name="tools")
    finish = node(finish_step, name="finish")
    call_llm - "tools" >> tools
    call_llm - "answer" >> finish
    tools >> call_llm
    finish >> END
    return Flow(start=call_llm)


def chef_loop(mailbox, *, report):
    call_llm = node(model_step, name="call_llm")
    tools = node(tools_step, name="tools")

    @node
    async def inbox(runtime: Runtime):
        event = await mailbox.get()
        runtime.run.emit("mailbox_event", event=event)
        report(f"INBOX {json.dumps(event)}")
        runtime.run.state["finished"] = runtime.run.state.get("finished", 0) + 1
        runtime.say(json.dumps(event))

    @node
    def respond(runtime: Runtime):
        report(f"CHEF {runtime.run.messages[-1].content}")
        if runtime.run.state.get("finished") == 2:
            runtime.finish(runtime.run.messages[-1].content)
            return "closed"
        return "listen"

    call_llm - "tools" >> tools
    call_llm - "answer" >> respond
    tools >> call_llm
    respond - "listen" >> inbox
    respond - "closed" >> END
    inbox >> call_llm
    return Flow(start=call_llm)


async def demo(*, report=print, log_dir=None):
    mailbox = asyncio.Queue()
    child = Agent(name="sous_chef", provider=SousChef(), loop=child_loop(), log_dir=log_dir)
    ask_sous_chef = agent_tool(child, name="ask_sous_chef", execution="background")
    async with AgentJobs(mailbox) as jobs:
        chef = Agent(
            name="chef",
            provider=HeadChef(),
            log_dir=log_dir,
            tools=AGENT_JOB_TOOLS | ask_sous_chef,
            loop=chef_loop(mailbox, report=report),
        )
        return await chef.run("Plan dinner with your sous chef.", context={"agent_jobs": jobs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir")
    result = asyncio.run(demo(log_dir=parser.parse_args().log_dir))
    if result.status != "completed":
        raise SystemExit(result.error)
