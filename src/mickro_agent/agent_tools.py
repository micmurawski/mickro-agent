"""Expose agents as tools, with optional supervised background execution."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from .agent import Agent
from .runtime import Runtime
from .session_log import validate_session_id
from .tools import Hidden, Tool, Tools, tool
from .types import Run


def _continuation(name: str, session_id: str) -> str:
    return f'To continue, call {name}(prompt="...", session_id="{session_id}").'


def _result(run: Run, name: str) -> dict:
    return {
        "session_id": run.session_id,
        "run_id": run.id,
        "status": run.status,
        "output": run.output,
        "error": run.error,
        "continuation": _continuation(name, run.session_id),
    }


@dataclass
class AgentJob:
    id: str
    session_id: str
    parent_session_id: str
    parent_run_id: str
    tool_name: str
    task: asyncio.Task[Run] | None = None
    status: str = "running"
    result: dict | None = None

    def snapshot(self) -> dict:
        return {
            "job_id": self.id,
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "parent_run_id": self.parent_run_id,
            "status": self.status,
            "continuation": _continuation(self.tool_name, self.session_id),
            **(self.result or {}),
        }


class AgentJobs:
    """One parent's supervisor. Terminal events wake its unbounded mailbox.

    Own this with ``async with`` so shutdown cancels and awaits children. Task
    cancellation is cooperative; it cannot kill a Python thread or remote effect.
    """

    def __init__(self, mailbox: asyncio.Queue):
        if mailbox.maxsize > 0:
            raise ValueError("AgentJobs requires an unbounded mailbox for completion notifications")
        self.mailbox = mailbox
        self._jobs: dict[str, AgentJob] = {}
        self._parent_session_id: str | None = None
        self._closed = False

    def start(
        self,
        agent: Agent,
        prompt: str,
        *,
        session_id: str | None,
        runtime: Runtime,
        tool_name: str,
        context: Mapping[str, Any],
    ) -> dict:
        if self._closed:
            raise RuntimeError("AgentJobs is closed")
        parent = runtime.run
        if self._parent_session_id not in (None, parent.session_id):
            raise ValueError("Use a separate AgentJobs mailbox for each parent session")
        self._parent_session_id = parent.session_id
        session_id = uuid4().hex if session_id is None else session_id
        validate_session_id(session_id)
        if session_id == parent.session_id:
            raise ValueError("A child cannot run in its active parent session")
        job = AgentJob(uuid4().hex, session_id, parent.session_id, parent.id, tool_name)
        # If logging fails, do not launch an unacknowledged child.
        runtime.run.emit("agent_job_started", **job.snapshot())
        self._jobs[job.id] = job
        job.task = asyncio.create_task(
            agent.run(
                prompt,
                session_id=session_id,
                context=context,
                parent_session_id=parent.session_id,
                parent_run_id=parent.id,
                job_id=job.id,
            ),
            name=f"agent-{job.id}",
        )
        # A done callback also covers cancellation before the coroutine starts.
        job.task.add_done_callback(lambda task: self._finished(job, task))
        return job.snapshot()

    def _finished(self, job: AgentJob, task: asyncio.Task[Run]) -> None:
        try:
            job.result = _result(task.result(), job.tool_name)
            job.status = job.result["status"]
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.result = {"output": None, "error": "Child cancelled by supervisor"}
        except Exception as exc:  # noqa: BLE001 - Includes checkpoint/provider setup failures.
            job.status = "failed"
            job.result = {"output": None, "error": f"{type(exc).__name__}: {exc}"}
        self.mailbox.put_nowait({"kind": "AgentFinished", **job.snapshot()})

    def _owned(self, job_id: str, parent_session_id: str) -> AgentJob:
        job = self._jobs[job_id]
        if job.parent_session_id != parent_session_id:
            raise ValueError("Job belongs to another parent session")
        return job

    def inspect(self, job_id: str, *, parent_session_id: str) -> dict:
        return self._owned(job_id, parent_session_id).snapshot()

    def list_jobs(self, *, parent_session_id: str) -> list[dict]:
        return [job.snapshot() for job in self._jobs.values() if job.parent_session_id == parent_session_id]

    def cancel(self, job_id: str, *, parent_session_id: str) -> dict:
        job = self._owned(job_id, parent_session_id)
        if job.task is not None and not job.task.done() and job.status != "stopping":
            job.status = "stopping"
            job.task.cancel()
        return job.snapshot()

    async def wait_all(self) -> None:
        await asyncio.gather(
            *(asyncio.shield(job.task) for job in self._jobs.values() if job.task is not None),
            return_exceptions=True,
        )

    async def aclose(self) -> None:
        self._closed = True
        for job in self._jobs.values():
            self.cancel(job.id, parent_session_id=job.parent_session_id)
        await self.wait_all()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()


def agent_tool(
    agent: Agent,
    *,
    name: str | None = None,
    description: str | None = None,
    execution: Literal["wait", "background"] = "wait",
    context: Mapping[str, Any] | None = None,
) -> Tool:
    """Bind an agent configuration; the model supplies only prompt and session_id.

    Child resources are explicit via context, never inherited from the parent.
    Background mode requires AgentJobs in the parent's ``agent_jobs`` context.
    """
    if execution not in {"wait", "background"}:
        raise ValueError("execution must be 'wait' or 'background'")
    name = name or f"ask_{agent.name}"
    child_context = dict(context or {})

    async def call_wait(prompt: str, runtime: Hidden[Runtime], session_id: str | None = None) -> dict:
        run = await agent.run(
            prompt,
            session_id=session_id,
            context=child_context,
            parent_session_id=runtime.run.session_id,
            parent_run_id=runtime.run.id,
        )
        return _result(run, name)

    async def call_background(
        prompt: str,
        runtime: Hidden[Runtime],
        agent_jobs: Hidden[AgentJobs],
        session_id: str | None = None,
    ) -> dict:
        return agent_jobs.start(
            agent,
            prompt,
            session_id=session_id,
            runtime=runtime,
            tool_name=name,
            context=child_context,
        )

    fn = call_wait if execution == "wait" else call_background
    fn.__doc__ = (description or f"Ask {agent.name} to perform a task.") + (
        " Omit session_id to start a conversation; supply it to continue. "
        + (
            "Returns the answer and session_id."
            if execution == "wait"
            else "Returns a job handle immediately; AgentFinished arrives in the mailbox when it ends."
        )
    )
    return Tool(fn, name=name)


# Longer spelling retained for code that wants to make the factory explicit.
agent_call_tool_factory = agent_tool


@tool
async def list_agent_jobs(agent_jobs: Hidden[AgentJobs], runtime: Hidden[Runtime]) -> list[dict]:
    """List this parent's delegated agent jobs and their results."""
    return agent_jobs.list_jobs(parent_session_id=runtime.run.session_id)


@tool
async def inspect_agent_job(job_id: str, agent_jobs: Hidden[AgentJobs], runtime: Hidden[Runtime]) -> dict:
    """Inspect a delegated agent job's status and result."""
    return agent_jobs.inspect(job_id, parent_session_id=runtime.run.session_id)


@tool
async def cancel_agent_job(job_id: str, agent_jobs: Hidden[AgentJobs], runtime: Hidden[Runtime]) -> dict:
    """Request cancellation; AgentFinished will confirm the terminal outcome."""
    return agent_jobs.cancel(job_id, parent_session_id=runtime.run.session_id)


AGENT_JOB_TOOLS = Tools([list_agent_jobs, inspect_agent_job, cancel_agent_job])
