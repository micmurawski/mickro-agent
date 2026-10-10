"""Live OpenAI or MiniMax run; install the matching extra and set its API key."""

import argparse
import asyncio
import shlex

from mickro_agent import END, Agent, FileSessionStore, Flow, Tools, finish_step, model_step, node, tool, tools_step
from mickro_agent.providers import MiniMax, OpenAIChat


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


async def main(args):
    call_llm = node(model_step, name="call_llm")
    tools = node(tools_step, name="tools")
    finish = node(finish_step, name="finish")

    call_llm - "tools" >> tools
    call_llm - "answer" >> finish
    tools >> call_llm
    finish >> END

    loop = Flow(start=call_llm)
    provider_type = MiniMax if args.provider == "minimax" else OpenAIChat
    options = {"base_url": args.base_url} if args.base_url else {}
    provider = provider_type(model=args.model, **options)
    try:
        run = await Agent(
            provider=provider,
            tools=Tools([multiply]),
            loop=loop,
            log_dir=args.log_dir,
            sessions=FileSessionStore(args.session_dir),
        ).run(args.prompt, session_id=args.session_id)
        print(run.output if run.status == "completed" else f"{run.status}: {run.error}")
        if run.log_path:
            print(f"Session log: {run.log_path}")
        if args.events:
            run.write_events(args.events)
        command = [
            "python",
            "examples/live_chat.py",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--session-dir",
            args.session_dir,
            "--session-id",
            run.session_id,
        ]
        if args.base_url:
            command.extend(["--base-url", args.base_url])
        print(f"To continue: {shlex.join([*command, 'Your next prompt'])}")
        if run.status != "completed":
            raise SystemExit(1)
    finally:
        await provider.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "minimax"], default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--events", help="Optional JSONL output path")
    parser.add_argument("--log-dir", help="Directory for the live mikro-agent-<session-id>.jsonl session log")
    parser.add_argument("--session-dir", default="artifacts/sessions", help="Directory for resumable conversations")
    parser.add_argument("--session-id", help="Continue a saved conversation (or create this named session)")
    parser.add_argument("prompt", nargs="?", default="Use multiply to compute 6 times 7.")
    asyncio.run(main(parser.parse_args()))
