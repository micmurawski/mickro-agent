"""Generic RLM-style Python loop over application-supplied documents.

Requires Docker and a local Python image. The demonstration provider is scripted;
replace it with any streaming provider for a live experiment.
"""

import argparse
import asyncio
import json
from pathlib import Path

from mickro_agent import END, Agent, Flow, Limits, finish_step, model_step, node, tools_step
from mickro_agent.python import DockerPythonSandbox, Mount, PythonSessions, python_tools

DOCUMENTS = [
    {"id": "a", "text": "Python supports asynchronous programming."},
    {"id": "b", "text": "Containers provide separate execution environments."},
    {"id": "c", "text": "Python tools can inspect a large document collection."},
]


class DemoAnalyst:
    async def create_message(self, messages, metadata, **kwargs):
        turn = metadata["turn"]
        if messages[-1].role == "tool":
            observation = json.loads(messages[-1].content)
            if observation["error"]:
                yield {"type": "text", "text": f"Python execution failed: {observation['error']}"}
                return
        if turn == 1:
            code = "selected = [doc for doc in data if 'python' in doc['text'].lower()]\nlen(selected)"
        elif turn == 2:
            code = (
                "import json\nfrom pathlib import Path\n"
                "summary = {'matches': len(selected), 'document_ids': [doc['id'] for doc in selected]}\n"
                "Path('summary.json').write_text(json.dumps(summary))\n"
                "print(json.dumps(summary))"
            )
        else:
            yield {"type": "text", "text": f"Analysis: {observation['stdout'].strip()}. Saved summary.json."}
            return
        yield {"type": "tool_use", "id": f"python-{turn}", "name": "execute_python", "input": {"code": code}}


def analysis_loop():
    call_llm = node(model_step, name="call_llm")
    tools = node(tools_step, name="tools")
    finish = node(finish_step, name="finish")
    call_llm - "tools" >> tools
    call_llm - "answer" >> finish
    tools >> call_llm
    finish >> END
    return Flow(start=call_llm)


async def demo(*, image="python:3.12-slim", workspace=None, log_dir=None):
    mounts = []
    if workspace is not None:
        workspace = Path(workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        mounts.append(Mount(workspace, "/workspace", read_only=False))
    async with PythonSessions(lambda _: DockerPythonSandbox(image=image, data=DOCUMENTS, mounts=mounts)) as sessions:
        agent = Agent(
            name="python-analyst",
            provider=DemoAnalyst(),
            loop=analysis_loop(),
            tools=python_tools(sessions, description="data is a list of documents with id and text fields."),
            system_prompt="Inspect data using Python, retain useful variables, and report concise results.",
            limits=Limits(tool_timeout=90),
            log_dir=log_dir,
        )
        return await agent.run("Find documents about Python and save their IDs in summary.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--workspace", help="Explicit writable host directory for exported files")
    parser.add_argument("--log-dir")
    args = parser.parse_args()
    result = asyncio.run(demo(image=args.image, workspace=args.workspace, log_dir=args.log_dir))
    print(result.output or result.error)
    if result.status != "completed":
        raise SystemExit(1)
