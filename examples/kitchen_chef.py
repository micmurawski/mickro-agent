"""Offline event-driven ReAct demo: a chef, a mailbox, and real background jobs.

Run with: PYTHONPATH=src python examples/kitchen_chef.py
The provider is a deterministic demo policy, not an LLM. Cooking jobs are tiny
Python subprocesses, so cancel/kill actually terminate work. No shell is involved.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from mickro_agent import (
    END,
    Agent,
    ApiHandlerCreateMessageMetadata,
    Flow,
    Hidden,
    Limits,
    Message,
    MessageParam,
    ModelRequest,
    ModelResponse,
    Run,
    Runtime,
    StreamChunk,
    ToolCall,
    Tools,
    model_step,
    node,
    tool,
    tools_step,
)
from mickro_agent.providers import ApiHandler
from mickro_agent.providers.streaming import response_chunks

ORDER = "Please cook pasta, sauce, and toast."
STATUS = "How is dinner coming along?"
STOP = "Kill the toast job; we don't need toast anymore."

# These workers never spawn children. A general command runner would also need
# process-group termination, bounded output storage, and platform-specific handling.
COOK = """
import sys, time
dish, seconds, fail = sys.argv[1], float(sys.argv[2]), sys.argv[3] == 'True'
print(f'{dish}: preparing', flush=True)
time.sleep(seconds / 2)
print(f'{dish}: cooking', flush=True)
time.sleep(seconds / 2)
print(f'{dish}: ' + ('burnt!' if fail else 'ready!'), flush=True)
sys.exit(1 if fail else 0)
"""


@dataclass
class Job:
    id: str
    dish: str
    process: asyncio.subprocess.Process
    status: str = "running"
    requested_stop: str | None = None
    output: list[str] = field(default_factory=list)
    watcher: asyncio.Task | None = None

    def snapshot(self) -> dict:
        return {
            "job_id": self.id,
            "dish": self.dish,
            "status": self.status,
            "requested_stop": self.requested_stop,
            "exit_code": self.process.returncode,
            "output": list(self.output),
        }


class Kitchen:
    """Owns processes; its single watcher per job publishes terminal events once."""

    def __init__(self, mailbox: asyncio.Queue):
        self.mailbox = mailbox
        self.jobs: dict[str, Job] = {}
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.closed = False

    async def start(self, dish: str, seconds: float, *, fail: bool = False) -> dict:
        if self.closed:
            raise RuntimeError("Kitchen is closed")
        if not math.isfinite(seconds) or not 0 < seconds <= 60:
            raise ValueError("Cooking duration must be between 0 and 60 seconds")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            COOK,
            dish,
            str(seconds),
            str(fail),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        job = Job(id=f"job-{len(self.jobs) + 1}", dish=dish, process=process)
        self.jobs[job.id] = job
        job.watcher = asyncio.create_task(self._watch(job), name=f"watch-{job.id}")
        self.started.put_nowait(job.id)
        return job.snapshot()  # Return a handle immediately, not the cooked dish.

    async def _watch(self, job: Job) -> None:
        assert job.process.stdout is not None
        async for raw in job.process.stdout:
            line = raw.decode(errors="replace").rstrip()
            job.output.append(line)
            job.output[:] = job.output[-20:]
            self.mailbox.put_nowait({"kind": "JobProgress", "job_id": job.id, "text": line})
        exit_code = await job.process.wait()
        job.status = "succeeded" if exit_code == 0 else (job.requested_stop or "failed")
        # Success, failure, cancellation AND killing all wake the chef.
        self.mailbox.put_nowait({"kind": "JobFinished", **job.snapshot()})

    def inspect(self, job_id: str) -> dict:
        return self.jobs[job_id].snapshot()

    def list_jobs(self) -> list[dict]:
        return [job.snapshot() for job in self.jobs.values()]

    def stop(self, job_id: str, *, force: bool) -> dict:
        job = self.jobs[job_id]
        if job.process.returncode is None:
            try:
                if force:
                    job.process.kill()
                else:
                    job.process.terminate()
            except ProcessLookupError:
                pass  # The watcher will report the actual exit outcome.
            else:
                job.requested_stop = "killed" if force else "cancelled"
                job.status = "stopping"
        return job.snapshot()  # A request acknowledgement; the watcher confirms exit.

    async def wait_all(self) -> None:
        # A cancelled customer/agent must not cancel the supervisor's watchers.
        await asyncio.gather(*(asyncio.shield(job.watcher) for job in self.jobs.values()))

    async def aclose(self) -> None:
        self.closed = True
        for job in self.jobs.values():
            if job.process.returncode is None:
                self.stop(job.id, force=True)
        await self.wait_all()


@tool
async def start_job(dish: str, seconds: float, kitchen: Hidden[Kitchen]) -> dict:
    """Start cooking in the background; return a job handle immediately."""
    return {"action": "start", **await kitchen.start(dish, seconds)}


@tool
async def list_jobs(kitchen: Hidden[Kitchen]) -> dict:
    """List cooking jobs, their statuses, and recent output."""
    return {"action": "list", "jobs": kitchen.list_jobs()}


@tool
async def inspect_job(job_id: str, kitchen: Hidden[Kitchen]) -> dict:
    """Inspect one cooking job and its latest output."""
    return {"action": "inspect", **kitchen.inspect(job_id)}


@tool
async def cancel_job(job_id: str, kitchen: Hidden[Kitchen]) -> dict:
    """Request graceful termination; JobFinished will confirm the outcome."""
    return {"action": "cancel", **kitchen.stop(job_id, force=False)}


@tool
async def kill_job(job_id: str, kitchen: Hidden[Kitchen]) -> dict:
    """Force termination; JobFinished will confirm the outcome."""
    return {"action": "kill", **kitchen.stop(job_id, force=True)}


@tool
async def plate_dish(job_id: str, kitchen: Hidden[Kitchen]) -> dict:
    """Plate a dish after its cooking job has succeeded."""
    job = kitchen.inspect(job_id)
    if job["status"] != "succeeded":
        raise ValueError("Only successfully cooked dishes can be plated")
    return {"action": "plate", "job_id": job_id, "dish": job["dish"]}


KITCHEN_TOOLS = Tools([start_job, list_jobs, inspect_job, cancel_job, kill_job, plate_dish])


class DemoChef(ApiHandler):
    """Deterministic policy driven by observations, not a turn-number script."""

    async def create_message(
        self,
        system_prompt: str,
        messages: list[MessageParam],
        metadata: ApiHandlerCreateMessageMetadata | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        metadata = metadata or {}
        request = ModelRequest(
            run_id=metadata.get("run_id", ""),
            turn=metadata.get("turn", 1),
            messages=(Message(role="system", content=system_prompt), *messages),
            tools=tuple(kwargs.get("tools", ())),
            purpose=metadata.get("purpose", "act"),
        )
        for chunk in response_chunks(self._respond(request)):
            yield chunk

    def _respond(self, request: ModelRequest) -> ModelResponse:
        event = next(
            json.loads(message.content)
            for message in reversed(request.messages)
            if message.role == "user" and message.content.startswith('{"kind":')
        )

        def calls(*actions):
            return ModelResponse(
                tool_calls=tuple(
                    ToolCall(id=f"turn-{request.turn}-{i}", name=name, arguments=arguments)
                    for i, (name, arguments) in enumerate(actions)
                )
            )

        last = request.messages[-1]
        if last.role == "tool":
            if last.is_error:
                return ModelResponse(text=f"Kitchen problem: {last.content}")
            result = json.loads(last.content)
            action = result["action"]
            if action == "list":
                if event.get("text") == STOP:
                    toast = next(job for job in result["jobs"] if job["dish"] == "toast")
                    return calls(("kill_job", {"job_id": toast["job_id"]}))
                statuses = ", ".join(f"{job['dish']}={job['status']}" for job in result["jobs"])
                return ModelResponse(text=f"Dinner status: {statuses}.")
            if action == "start":
                return ModelResponse(text="Cooking jobs started. I can take questions while they cook.")
            if action in {"kill", "cancel"}:
                return ModelResponse(text=f"Stop requested for {result['dish']}; waiting for confirmation.")
            if action == "plate":
                return ModelResponse(text=f"Plated {result['dish']} after its completion event woke me up.")
            return ModelResponse(text=json.dumps(result))

        if event["kind"] == "UserMessage":
            if event["text"] == ORDER:
                return calls(
                    ("start_job", {"dish": "pasta", "seconds": 0.5}),
                    ("start_job", {"dish": "sauce", "seconds": 0.8}),
                    ("start_job", {"dish": "toast", "seconds": 30.0}),
                )
            return calls(("list_jobs", {}))
        if event["kind"] == "JobFinished":
            if event["status"] == "succeeded":
                return calls(("plate_dish", {"job_id": event["job_id"]}))
            return ModelResponse(text=f"Observed {event['dish']} {event['status']} (exit {event['exit_code']}).")
        return ModelResponse(text=f"Progress noted: {event['text']}.")


def chef_loop(mailbox: asyncio.Queue, kitchen: Kitchen, *, report=print) -> Flow:
    @node
    async def inbox(runtime: Runtime) -> str:
        event = await mailbox.get()  # Sleeps without polling; queued events wake it.
        runtime.run.emit("mailbox_event", event=event)
        if event["kind"] == "Shutdown":
            runtime.run.state["jobs"] = kitchen.list_jobs()
            runtime.finish("Kitchen service closed; all job outcomes were observed.")
            return "closed"
        report(f"INBOX {json.dumps(event)}")
        # The start_job call already got its tool result. Completion is a NEW observation.
        runtime.say(json.dumps(event))
        return "message"

    call_llm = node(model_step, name="call_llm")
    tools = node(tools_step, name="tools")

    @node
    def respond(runtime: Runtime) -> None:
        report(f"CHEF  {runtime.run.messages[-1].content}")
        # Replying does not finish the agent: it returns to listening.

    inbox - "message" >> call_llm
    inbox - "closed" >> END
    call_llm - "tools" >> tools
    call_llm - "answer" >> respond
    tools >> call_llm
    respond >> inbox
    return Flow(start=inbox)


async def demo(*, report=print, log_dir=None) -> Run:
    mailbox = asyncio.Queue()
    kitchen = Kitchen(mailbox)
    mailbox.put_nowait({"kind": "UserMessage", "text": ORDER})

    async def customer():
        # Wait for the three handles, not job completion. No timing-based polling.
        for _ in range(3):
            await kitchen.started.get()
        mailbox.put_nowait({"kind": "UserMessage", "text": STATUS})
        mailbox.put_nowait({"kind": "UserMessage", "text": STOP})
        await kitchen.wait_all()
        # Watchers enqueue every final event before returning. FIFO places this
        # explicit demo shutdown AFTER those events have been handled by the chef.
        mailbox.put_nowait({"kind": "Shutdown"})

    driver = asyncio.create_task(customer())
    try:
        agent = Agent(
            log_dir=log_dir,
            name="kitchen-chef",
            provider=DemoChef(),
            tools=KITCHEN_TOOLS,
            loop=chef_loop(mailbox, kitchen, report=report),
            limits=Limits(max_model_calls=60, max_steps=250, timeout=10),
            system_prompt="Manage kitchen jobs. Their completion events arrive as new mailbox observations.",
        )
        result = await agent.run("Open kitchen service.", context={"kitchen": kitchen})
        report(f"SERVICE {result.status}: {result.output or result.error}")
        return result
    finally:
        driver.cancel()
        await asyncio.gather(driver, return_exceptions=True)
        await kitchen.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", help="Directory for live session JSONL logs")
    result = asyncio.run(demo(log_dir=parser.parse_args().log_dir))
    if result.log_path:
        print(f"Session log: {result.log_path}")
    if result.status != "completed":
        raise SystemExit(1)
