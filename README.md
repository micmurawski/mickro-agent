# mickro_agent

A small Python project for experimenting with agentic loops. Migrated and redesigned
from `incident-assistant/agent`: keep explicit graphs, typed tools, provider adapters,
and model/tool turns; give each execution its own state and limits.

The first recipes are **ReAct** (model → tools → model) and **PlanThenAct** (a planning
turn followed by ReAct). A loop is ordinary Python implementing `async run(runtime)`.
`Flow` uses the familiar `node >> next_node` and `node - "action" >> next_node` DSL.
No global
agent registry, database, model catalog, or tracing service is required.

## Install and run

From this directory, using Python 3.11 or newer:

```sh
python -m pip install -e '.[dev]'
python examples/compare_loops.py
python examples/custom_loop.py
python examples/kitchen_chef.py
python examples/delegating_chef.py
python -m pytest
```

The examples above are deterministic and offline. They demonstrate mechanics, not
evidence that one prompting strategy outperforms another.

For a live model, install `.[openai]`, set `OPENAI_API_KEY`, and run:

```sh
python examples/live_chat.py --model YOUR_MODEL --events artifacts/run.jsonl
```

Use `--base-url` for an OpenAI-compatible endpoint. Model-specific options can be
passed to `OpenAIChat(model=..., temperature=..., ...)`; there are no implicit model
names or temperature settings. Client retries are disabled for clients this adapter
creates. An injected SDK client retains its own configuration and lifecycle.

## Basic API

```python
from mickro_agent import Agent, Hidden, Limits, PlanThenAct, Tools, tool
from mickro_agent.providers import OpenAIChat


@tool(tags=["lookup"])
def lookup(key: str, records: Hidden[dict]) -> str:
    """Look up a record by key."""
    return records.get(key, "Not found")


provider = OpenAIChat(model="YOUR_MODEL")
agent = Agent(
    provider=provider,
    tools=Tools([lookup]),
    loop=PlanThenAct(),  # or ReAct(), or your own Flow
    system_prompt="Answer questions using the available records.",
    limits=Limits(max_model_calls=10, max_tool_calls=20, timeout=120),
)

# Inside an async function:
run = await agent.run("Find the owner of service-a", context={"records": {"service-a": "team-a"}})
print(run.status, run.output, run.usage)
run.write_events("artifacts/service-a.jsonl")
await provider.aclose()
```

`Hidden[T]` parameters are excluded from the model's schema. Unknown arguments,
including attempts to supply hidden fields, are rejected. Pydantic validates public
arguments; nullable required arguments accept explicit `None`. Plain functions and
async functions both work, and tools can return plain JSON-compatible values or
`ToolResult(result=..., error=...)`. Compose collections with `|` and filter with
`tools.select("tag")`. Duplicate names with different implementations are rejected.

## Unified streaming providers

Providers implement the original async-generator shape:

```python
async def create_message(
    self,
    system_prompt: str,
    messages: list[MessageParam],
    metadata: ApiHandlerCreateMessageMetadata | None = None,
    **kwargs: Any,
) -> AsyncIterator[StreamChunk]:
    yield {"type": "text", "text": "Hello"}
```

Use `async for` directly; do not await `create_message()` before iterating it.
`MessageParam` aliases the canonical `Message` model, so this interface does not
depend on Anthropic or another provider's SDK types. Put the system prompt in
`system_prompt`; `messages` contains the remaining conversation. Pass canonical
tool definitions as `tools=[ToolSpec(...), ...]` (equivalent dictionaries also work).
The runtime supplies `metadata` containing `run_id`, `task_id`, `turn`, and `purpose`.

```python
from mickro_agent import Message
from mickro_agent.providers import OpenAIChat

provider = OpenAIChat(model="YOUR_MODEL")
try:
    async for chunk in provider.create_message(
        system_prompt="You are a helpful chef.",
        messages=[Message(role="user", content="Suggest a quick dinner.")],
        temperature=0.2,
    ):
        if chunk["type"] == "text":
            print(chunk["text"], end="", flush=True)
finally:
    await provider.aclose()
```

The provider package has one file per implementation:

```text
providers/
    base.py       # ApiHandler base class and structural Provider protocol
    openai.py     # OpenAIChat: native streaming, including fragmented tool calls
    minimax.py    # MiniMax: Anthropic-compatible streaming and preserved thinking blocks
    scripted.py   # ScriptedProvider: offline, per-turn responses
    legacy.py     # LegacyProvider: forwards the original handlers' streams
    streaming.py  # Shared response collector and stream cleanup
```

Chunks use `text`, `reasoning`, `tool_use`, `usage`, `metadata`, or `grounding` types.
Text and reasoning are incremental strings; each `tool_use` carries one complete
call (`id`, `name`, `input`, optional `metadata`). Adapters assemble any provider
tool-call fragments before yielding that chunk. Usage counters are additive; the
OpenAI adapter emits the final usage snapshot once. Reasoning and grounding are
retained separately from answer text.

`Runtime.complete()` consumes this stream, records `model_chunk` events as chunks
arrive, and returns the assembled `ModelResponse` to the DSL node. Tools execute
only after the stream finishes successfully. Stream errors fail the turn; partial
tool calls are not executed. Streams are closed on completion, errors, and cancellation.
For direct streaming callers that stop early, close the iterator with `aclose()`
or use `contextlib.aclosing`.

Subclass `ApiHandler` for the optional `complete(ModelRequest)` convenience method,
or simply implement `create_message()` on your class. The runtime requires only
the streaming method. Existing package-level imports such as
`from mickro_agent.providers import OpenAIChat` continue to work.

### MiniMax

Install the optional SDK and set `MINIMAX_API_KEY` (or pass `api_key=`):

```sh
python -m pip install -e '.[minimax]'
python examples/live_chat.py --provider minimax --model MiniMax-M2.7
```

```python
from mickro_agent.providers import MiniMax

provider = MiniMax(model="MiniMax-M2.7", max_tokens=4096)
# Use with Agent(provider=provider, ...) or provider.create_message(...).
# Close the owned SDK client after use: await provider.aclose()
```

The provider uses MiniMax's [Anthropic-compatible endpoint](https://platform.minimax.io/docs/api-reference/text-anthropic-api)
at `https://api.minimax.io/anthropic`, with an explicit model name and optional
`base_url` override. It streams text and reasoning, assembles tool calls, groups
parallel tool results, and reports final token/cache usage once. The full assistant
content is stored in message metadata as `minimax_content` and replayed on subsequent
turns, preserving thinking blocks and signatures as MiniMax requires. If manually
editing an assistant message, discard that metadata before reconstructing its content.

The `minimax` extra is optional; importing the core does not import the SDK. An
injected `client=` remains caller-owned. Locally created clients disable automatic
SDK retries. Truncated/error streams fail the turn and do not execute partial tool
calls. The canonical message API currently supports text and tools, not multimodal
inputs. Model-specific options such as `thinking` can be passed explicitly; no
model catalog or automatic thinking settings are applied.

## Experiment with the loop

Define ReAct directly using the DSL:

```python
from mickro_agent import END, Flow, finish_step, model_step, node, tools_step

call_llm = node(model_step, name="call_llm")
tools = node(tools_step, name="tools")
finish = node(finish_step, name="finish")

call_llm - "tools" >> tools
call_llm - "answer" >> finish
tools >> call_llm
finish >> END

loop = Flow(start=call_llm)
# Agent(provider=provider, tools=tool_collection, loop=loop)
```

Define your own nodes with `@node` (sync or async):

```python
@node(metadata={"description": "Plan before acting"})
async def plan(runtime):
    runtime.say("Write a concise action plan. Do not call tools yet.")
    response = await runtime.complete(allow_tools=False, purpose="plan")
    runtime.run.state["plan"] = response.text
    runtime.say("Execute the plan, then give the result.")


plan >> call_llm
planned_loop = Flow(start=plan)
```

- `a >> b` connects the default route; returning `None` or `"default"` follows it.
- `a - "tools" >> b` follows that route when `a` returns `"tools"`.
- Chains such as `a >> b >> c` work, as do cycles and branching.
- `finish >> END` explicitly ends routing. A leaf also ends on the default route.
  A successful run must still call `runtime.finish(output)`.
- `Flow(start=...)` snapshots reachable nodes, routes, and metadata. Wire first;
  create another Flow after changes. Existing flows retain their original routing.
- Duplicate routes and duplicate node names are errors. Use `node(fn, name="...")`
  when using the same function in multiple positions.
- `@node` also supports instance methods: `steps.call_llm - "tools" >> steps.tools`.
  Wire bound methods on the instance. Each instance has its own node handles; keep
  execution state on `runtime.run`, not on the instance.

`examples/custom_loop.py` inserts a review step before accepting an answer. Its nodes
reuse the same `model_step`, `tools_step`, and `finish_step` as ReAct. To add retrieval,
reflection, a verifier, or alternative routing, add a node and its explicit edges.
An unknown action fails the run instead of silently ending it.

The examples define their graphs with this DSL. The built-in `ReAct()` and
`PlanThenAct()` recipes use it too. The explicit `Flow(nodes=..., edges=..., start=...)`
constructor remains available for code that generates graphs.

The DSL, method-node ergonomics, and metadata support follow `thearc/flow`. Node
metadata is included in step events. This package's nodes receive `Runtime` directly;
the older decorator's signature-based shared-dictionary injection and batch options
are not part of this API. Loop nodes return actions, and update `runtime.run.state`
explicitly. Tool concurrency remains controlled by `runtime.execute_tools()`.

Each node receives `Runtime`, providing:

| API | Purpose |
| --- | --- |
| `await runtime.complete(allow_tools=True, purpose="act")` | Make a bounded model call and record the response |
| `await runtime.execute_tools(concurrency=1)` | Execute the pending calls with validation and deadlines |
| `runtime.say(text)` | Add a user instruction |
| `runtime.run.state` | Store experiment-specific state, such as a plan or review count |
| `runtime.finish(text)` | Mark the run as completed |

Every flow step, model request/response, tool call/result, transition, and terminal
status becomes an event. Requests include the actual prompt and tool schema, so runs
can be compared without reconstructing configuration. `Run.write_events()` exports
JSONL for later analysis. Context dependencies are not automatically serialized.
Events do contain model-visible inputs and tool outputs. Set `log_dir` to persist
them as they happen, or use `Run.write_events()` for an explicit snapshot export.
Conversation continuation uses the session store described below; JSONL logs remain
an execution trace, not the checkpoint format.

## Print and render flow graphs

The original agent graph API is available without running the agent:

```python
print(agent.get_flow_graph())  # Mermaid text
agent.get_flow_graph_png("artifacts/agent.png")  # Requires local Graphviz
agent.render_graph("artifacts/agent.svg")
```

Compiled DSL flows expose the same functionality directly:

```python
flow = Flow(start=call_llm)
print(flow.to_mermaid(direction="TB"))
print(flow.to_dot())
flow.render("artifacts/flow.svg")
```

`ReAct().to_flow()` and `PlanThenAct().to_flow()` return the same graph definition
used during execution. Custom loops can implement `to_flow() -> Flow` to support
export; arbitrary Python loops without a graph raise a clear error. Graph export
shows the compiled snapshot, including cycles, action labels, and actual END edges.
Dashed edges are default transitions. Directions are `LR`, `RL`, `TB`, and `BT`.

Mermaid and DOT export use only the core package. Paste Mermaid into a Mermaid
viewer or a fenced `mermaid` Markdown block. PNG/SVG rendering uses Graphviz's
`dot` executable on PATH (install Graphviz separately, e.g. `brew install graphviz`
on macOS). Rendering runs locally; no diagram is sent to a hosted service.

```sh
PYTHONPATH=src python examples/print_graph.py
PYTHONPATH=src python examples/print_graph.py --loop plan --format dot
PYTHONPATH=src python examples/print_graph.py --output artifacts/react.svg
```

The functions `build_mermaid`, `build_dot`, `render_graph`, and the original
`to_png` spelling are also available from `mickro_agent.viz` or the package root.

## Agents as tools

Bind an agent configuration to a tool with `agent_tool(...)` (also exported as
`agent_call_tool_factory(...)`):

```python
from mickro_agent import Agent, FileSessionStore, Tools, agent_tool

researcher = Agent(
    name="researcher",
    provider=research_provider,
    sessions=FileSessionStore("./sessions/researcher"),
    log_dir="./logs",
)
ask_researcher = agent_tool(
    researcher,
    name="ask_researcher",
    description="Research a topic or follow up on previous research.",
)
parent = Agent(provider=parent_provider, tools=Tools([ask_researcher]))
run = await parent.run("Ask the researcher to investigate this topic.")
```

The model sees only `prompt: str` and `session_id: str | None = None`. Omitting the
ID creates a new session; supplying a known ID restores its conversation. A supplied
unknown ID creates a named session, matching `Agent.run`. The result contains
`session_id`, `run_id`, `status`, `output`, `error`, and a human-readable `continuation`
instruction generated from those identifiers. Check `status` before treating the
output as a successful answer.

The default `execution="wait"` waits for the child. Parent tool deadlines also apply,
so configure `Limits(tool_timeout=...)` for the expected child duration. For a parent
that should keep listening, use `execution="background"`:

```python
import asyncio

from mickro_agent import AGENT_JOB_TOOLS, AgentJobs

mailbox = asyncio.Queue()
ask_researcher = agent_tool(researcher, execution="background")
async with AgentJobs(mailbox) as jobs:
    parent = Agent(
        provider=parent_provider,
        tools=AGENT_JOB_TOOLS | ask_researcher,
        loop=listening_loop,  # A Flow whose inbox awaits mailbox.get().
    )
    run = await parent.run("Investigate this topic.", context={"agent_jobs": jobs})
```

Background calls return `job_id`, `session_id`, and `status` immediately. Each child
termination queues exactly one `AgentFinished` event, including failures and
cancellation. The listening loop adds that event as a new observation and calls
the model. A plain ReAct loop finishes after its answer; use a mailbox loop for
automatic wake-up. `AGENT_JOB_TOOLS` exposes `list_agent_jobs`, `inspect_agent_job`,
and `cancel_agent_job`. The supervisor is scoped to one parent session and must use
an unbounded queue; leaving its async context cancels and awaits remaining children.
Cancellation is cooperative, not process termination: Python threads and remote
effects can outlive it. Jobs and mailbox notifications are in-memory.

`examples/delegating_chef.py` is a complete offline example. Both parent and child
loops use the DSL. The head chef delegates dinner planning, wakes on `AgentFinished`,
asks the same child session for a gluten-free revision, then wakes on its completion.
Its provider policies are deterministic, so no API keys are needed.

The runtime injects `runtime: Hidden[Runtime]` into tools and records parent/run/job
links in child logs. A factory's `context={...}` supplies child resources explicitly;
parent resources are not inherited. Ordinary tools can also request the reserved
`runtime` dependency; a user context entry cannot override it.

## Resumable conversations

```python
first = await researcher.run("Compare the options.")
second = await researcher.run("Explain the second option.", session_id=first.session_id)
assert first.session_id == second.session_id
assert first.id != second.id
```

Each Agent has its own `MemorySessionStore` by default. Reuse that Agent to continue
in the same process, or supply a shared store. `FileSessionStore(directory)` persists
JSON checkpoints for continuation after restarting the application. It uses atomic
replacement and OS locks on macOS/Linux local filesystems, serializing concurrent
invocations of the same session even through separate store instances. Different
sessions may run concurrently; provider/client/resource concurrency support remains
the caller's responsibility.

Only messages, including provider metadata such as MiniMax thinking blocks, are
restored. Each invocation starts the configured loop afresh, with new counters,
limits, state, and explicitly supplied context. A session retains its agent name,
system prompt, and log location; changing those or passing `history` during a
continuation is rejected. Model/provider changes are caller-managed: ensure their
stored message metadata is compatible. Checkpoints hold conversation content as-is.

Checkpoints are saved at invocation boundaries, including cooperative cancellation.
Interrupted tool calls receive an error observation with unknown outcome; known
results are preserved and tools are not automatically retried. A hard-killed process
leaves a `running` checkpoint that requires inspection and explicit recovery rather
than automatic replay of possibly completed effects. Checkpoint failures surface to
the caller. Session IDs identify conversations, not authentication credentials.

`examples/live_chat.py` saves checkpoints in `artifacts/sessions` by default, accepts
`--session-dir` and `--session-id`, and prints a concrete continuation command.

## Session logging

Logging is opt-in. Choose a directory on the agent, or override it for an individual
run. Each new conversation gets one session log; continuations append runs to it.
An ongoing mailbox loop such as the kitchen chef can also remain in a single run:

```python
agent = Agent(provider=provider, tools=tools, log_dir="./logs/chef")
run = await agent.run("Prepare dinner")
print(run.session_id)  # Stable conversation ID; run.id identifies this invocation
print(run.log_path)  # Absolute path: .../logs/chef/mikro-agent-<session-id>.jsonl

# Optional per-run directory and a chosen session ID:
run = await agent.run("Prepare lunch", log_dir="./logs/lunch", session_id="lunch-001")
```

With no `log_dir` on either call, no log files are created. A per-run `log_dir=None`
inherits the agent's setting. Directories are created automatically. IDs default
to fresh UUIDs; a custom ID accepts 1–128 ASCII letters/digits/underscores/hyphens
and must start with a letter or digit. Existing logs are appended only when the
locked session checkpoint identifies that log. Unrelated existing filenames and
incomplete log records are errors. Resumed sessions keep their original log directory.

The file contains one UTF-8 JSON object per event with `schema_version: 2`,
`session_id`, `run_id`, `sequence`, a UTC `timestamp`, `kind`, and `data`.
The first `run_start` record contains the agent/provider/model names, working
directory, limits, and initial messages (including supplied history). Following
records contain model requests, streamed chunks, complete responses, tool calls
and results, graph transitions, and application events such as kitchen job updates.
`run_end` records the outcome and accumulated usage, including failures, limits,
timeouts, and caller cancellation. Each invocation has its own `run_start`,
`run_end`, `run_id`, and sequence starting at zero; `session_id` stays constant.

Every line is flushed before execution continues, so readers can tail a live file
and earlier events remain available after an interrupted process. A hard kill can
leave no `run_end` record or an incomplete final line; flushing is not an `fsync`
guarantee against power loss. This is mickro_agent's versioned native log format,
not a claim of direct Codex/Claude log-format compatibility. Session checkpoints
are stored separately from logs.

The logger does not serialize injected context objects, client credentials, or
provider options. Conversation content and tool inputs/outputs are logged as-is.
File creation or write failures raise `SessionLogError` rather than silently
continuing without the requested log. `Run.write_events(path)` exports the same
record format for one run and cannot overwrite that run's session log.

Both examples expose the directory option:

```sh
python examples/live_chat.py --provider minimax --model MiniMax-M2.7 --log-dir ./logs
python examples/kitchen_chef.py --log-dir ./logs
```

## Kitchen chef: event-driven ReAct

`examples/kitchen_chef.py` is an offline toy example of an always-listening chef.
It uses a deterministic provider to demonstrate scheduling, with real Python
subprocesses representing dishes being cooked. It needs no credentials or SDK.

```sh
# From the project root, without installing the package:
PYTHONPATH=src python examples/kitchen_chef.py
```

The customer orders pasta, sauce, and toast. The chef starts three jobs through
tools and immediately receives their handles. While they cook, the customer asks
for status and asks the chef to kill the toast job. The chef lists jobs and sends
the kill request through tools. Pasta and sauce finish independently; **each
completion enqueues a new observation, wakes the chef, and causes it to plate
that dish without another user request**. The toast's confirmed termination wakes
the chef too. Progress events also enter the mailbox.

The loop is defined with the same DSL:

```python
inbox - "message" >> call_llm
inbox - "closed" >> END
call_llm - "tools" >> tools
call_llm - "answer" >> respond
tools >> call_llm
respond >> inbox
```

`inbox` awaits a queue without polling. Messages arriving while the chef is busy
remain queued. `respond` does not call `runtime.finish()`; the conversation stays
open. Only an explicit demo shutdown closes the service, queued after all job
outcomes. The entire demonstration shares one run and conversation history.

The kitchen exposes `start_job`, `list_jobs`, `inspect_job`, `cancel_job`, `kill_job`,
and `plate_dish` as tools. `start_job` returns immediately after launch; its later
completion is a fresh mailbox observation, not a second result for that tool call.
One supervisor watcher owns each job's final event. A stop request first reports
`stopping`; only observed process exit reports `cancelled` or `killed`.

This is deliberately example-local: in-memory jobs and mailbox, a finite scripted
customer, and fixed child processes that do not spawn descendants. It demonstrates
real termination and cleans up jobs on exit; it is not a durable or general-purpose
shell job manager. Cancellation uses process termination, while killing uses force
termination. Replacing the demo provider with an LLM is a separate experiment—the
demo policy itself is deterministic, not evidence of model planning ability.

## Generic Python / RLM-style tools

The Python tools adapt the persistent interpreter pattern from
`incident-assistant/agent/agent/tooling/rlm_metrics.py` to arbitrary data and tasks.
There are no Grafana clients, pandas dependencies, task classes, or global container
registry. The application owns container configuration and lifetime:

```python
from mickro_agent import Agent, Limits
from mickro_agent.python import DockerPythonSandbox, Mount, PythonSessions, python_tools

async with PythonSessions(
    lambda session_id: DockerPythonSandbox(
        image="python:3.12-slim",
        data={"documents": documents},  # Available as `data`, outside model history.
        mounts=[Mount("./inputs", "/inputs")],  # Read-only by default; source must exist.
        setup="import json, statistics",
        timeout=30,
    )
) as python:
    agent = Agent(
        provider=provider,
        tools=python_tools(python, description="data['documents'] holds the document collection."),
        limits=Limits(tool_timeout=90),
    )
    first = await agent.run("Inspect the documents with Python and summarize the patterns.")
    second = await agent.run("Check the exceptions in more detail.", session_id=first.session_id)
```

`execute_python(code)` retains variables between calls and returns structured
`stdout`, `stderr`, the last expression's `result` as repr, `error`, `truncated`,
`state_lost`, and `duration`. Python exceptions are observations, and assignments
before an exception remain in the interpreter. Ordinary Python `print`, native
stdout/stderr writes, and subprocess output are captured. Output, code size, and
history have configurable bounds. `python_history(limit=10)` reads recent code and
results; `reset_python()` discards the interpreter, temporary files, and history,
then the next execution reruns setup in a fresh container.

Each session gets a separate interpreter, and execution within one interpreter is
serialized. Keep the `PythonSessions` owner alive to preserve variables across
conversation continuations. Message checkpoints do not save Python memory; closing
the owner removes its containers. The factory can configure per-session mounts or
input data. Reuse the tool collection on child agents from `agent_tool(...)`: their
different session IDs give them separate interpreters. Explicit writable mounts can
serve as shared handover directories; files on those mounts survive container reset.
The snippets themselves do not expose a nested LLM callback; delegation uses the
existing agent tools.

The backend requires the Docker CLI, a running **local** daemon, and a preinstalled
image containing Python 3.11+. It uses the Docker attach stream, with no HTTP server,
published ports, extra Python SDK, or automatic package installation. Network access
defaults to `none`; the root filesystem is read-only, `/tmp` and `/workspace` are
bounded temporary filesystems, capabilities are dropped, and memory/CPU/process
limits are set. These Docker controls are not a guarantee for hostile multi-tenant
execution. Only explicitly configured host mounts and environment entries are
passed to the container; there is no automatic host-code fallback.

Use a custom image for dependencies such as pandas, configure `network="bridge"`
when external access is required, and pass `env={...}` and `setup="..."` to initialize
your own clients. For exported files, create a directory and mount it with
`Mount(path, "/workspace", read_only=False)`. Application-provided code/data/setup
can be large; only execution observations enter the model's conversation.

Timeout or cancellation removes the container, including its running processes.
The next call reports lost state until `reset_python` is requested. No failed code
is automatically replayed; filesystem/network effects may already have happened.
Use an enclosing `async with` so normal completion and errors clean up containers.
Cleanup failures surface rather than claiming that removal succeeded. Host tool
deadlines should allow for startup, execution, and cleanup.

The example uses the same DSL for its model → tools → model loop and a scripted
provider, so it requires no model credentials:

```sh
docker pull python:3.12-slim
PYTHONPATH=src python examples/python_rlm.py --workspace ./artifacts/python-output

# Optional real-container integration test (requires the daemon and local image):
MICKRO_TEST_DOCKER=1 python -m pytest tests/test_python.py -q
```

`PythonSandbox` is a small protocol (`execute`, `history`, `reset`, `aclose`), allowing
other execution backends without changing tools or loops. Normal tests run trusted
test snippets against the same worker in a separate local process; the opt-in test
exercises Docker mounts and container removal.

## Execution semantics

- `Agent` describes an experiment; `Run` owns its messages, counters, state, and events.
  History is copied. The context mapping is copied, but injected clients/resources
  are borrowed references: their concurrency safety remains the caller's responsibility.
- Tools execute sequentially by default. `ReAct(tool_concurrency=4)` opts into bounded
  parallel execution, retains result order, and cancels siblings on infrastructure
  failure. No shared mutable retry counters or automatic tool/node retries exist.
- Tool argument errors and tool exceptions produce error observations for the model.
  Missing injected dependencies, provider errors, and invalid graph transitions fail
  the run. A whole tool batch is refused if it would exceed the tool-call budget.
- Run/model/tool/graph limits are independent of any incident-specific `Task`. Check
  `run.status`: completion, a limit, a timeout, and failure are distinct outcomes.
  Caller cancellation propagates as `CancelledError`.
- Async tools should cooperate with cancellation. Sync tools run in worker threads;
  timing out a thread does not terminate its work. A deadline stops the run without
  retrying a potentially completed external effect.
- Custom Python loops must use the runtime methods for budgets/events. Graph step
  limits apply to `Flow` nodes; arbitrary custom loops need cooperative async code
  so the run deadline can interrupt them.

## Migration from incident-assistant

This is a selective migration, not a drop-in replacement. The original tree remains
untouched. New code depends only on Pydantic; old integrations remain opt-in.

| Original | New home |
| --- | --- |
| `LLMAgent` + shared dictionaries | `Agent`, `Run`, `Runtime` |
| `AsyncFlow`, `@node`, `>>`, `- "action" >>` | `Flow(start=node)` and `@node`, with routing snapshots and per-run state |
| `@tool`, `Hidden`, `Tools`, `ToolResult` | `mickro_agent.tools` |
| Provider-specific message shapes | Canonical `Message`, `ToolCall`, `ModelRequest`, `ModelResponse` |
| `OpenAICompatibleHandler` | `OpenAIChat`, with injected configuration/client |
| `ApiHandler.create_message(...) -> AsyncIterator[StreamChunk]` | Same streaming entry point; adapters in `mickro_agent.providers` |
| Existing provider handlers | Opt-in `LegacyProvider` bridge |
| Task histories and terminal prints | Per-run events, messages, usage, and result |

Existing decorated tools and handlers can be reused in an environment where the
old package and its dependencies are already importable:

```python
from mickro_agent import Agent, Tools
from mickro_agent.legacy import LegacyProvider, migrate_tool

# old_handler and old_tools are your existing configured objects.
agent = Agent(
    provider=LegacyProvider(old_handler),
    tools=Tools(migrate_tool(item) for item in old_tools.tools),
)
run = await agent.run("Investigate the incident", context={"cwd": workspace, "env": environment})
```

Create a fresh legacy handler/adapter for each run: some original handlers retain
conversation state. Their clients remain caller-owned. The bridge adapts complete
streamed text/tool turns and usage, including Gemini tool signature metadata. It
does not promise complete multimodal or provider-specific reasoning-block fidelity.
Infrastructure tools still require their original clients and runtime dependencies.

Kubernetes/Grafana tools, task delegation/feedback, ACE curation, database models,
automatic summarization, and tracing backends have not been copied into the core.
They can be migrated as tools, loop nodes, or event consumers as experiments need
them. Crash-safe resumption remains outside this first version.

For thearcv2 integration, the intended boundary is the exported session events:
execute here, normalize/index those events there, and apply reflection or learning
without coupling the loop runtime to a particular reflection schema. That adapter
is a follow-up; these events are not yet a thearcv2 session-source format.
