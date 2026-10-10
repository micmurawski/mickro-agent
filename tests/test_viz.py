import asyncio
import json
import shutil
import subprocess

import pytest

from mickro_agent import (
    END,
    Agent,
    Flow,
    ModelResponse,
    PlanThenAct,
    ReAct,
    build_dot,
    build_mermaid,
    node,
    render_graph,
)
from mickro_agent.providers import ScriptedProvider


def test_graph_export_shows_cycles_branches_default_edges_and_actual_end():
    def forbidden(runtime):
        raise AssertionError("Export must not execute any node")

    model = node(forbidden, name="model")
    tools = node(forbidden, name="tools")
    model - "tools" >> tools
    model - "answer" >> END
    tools >> model
    flow = Flow(model)
    assert flow.to_mermaid() == (
        'flowchart LR\n    start_node(["START"])\n    n0["model"]\n    n1["tools"]\n'
        '    end_node(["END"])\n    start_node --> n0\n    n0 -->|"answer"| end_node\n'
        '    n0 -->|"tools"| n1\n    n1 -.-> n0\n'
    )
    assert 'n0 -> n1 [label="tools"]' in flow.to_dot()
    assert "n1 -> n0 [style=dashed]" in flow.to_dot()
    assert flow.to_dot(direction="TB").startswith("digraph flow {\n    rankdir=TB;")
    cycle = node(forbidden, name="cycle")
    cycle >> cycle
    assert "end_node" not in Flow(cycle).to_mermaid()


def test_export_uses_compiled_snapshot_and_escapes_labels():
    name = 'END" ]\n<script> &#quot; | \\unicode:ą'
    action = 'a"|b\n%% comment'
    start = node(lambda _: None, name=name)
    flow = Flow(start)
    start - action >> start
    assert action not in flow.to_mermaid()
    changed = Flow(start)
    mermaid = changed.to_mermaid()
    assert "<script>" not in mermaid and "%% comment" not in mermaid
    assert "#34;" in mermaid and "#124;" in mermaid and "<br/>" in mermaid
    assert f"label={json.dumps(name, ensure_ascii=False)}" in changed.to_dot()
    assert f"label={json.dumps(action, ensure_ascii=False)}" in changed.to_dot()
    # IDs never depend on names, which may even be START or END.
    assert 'n0["' in mermaid


@pytest.mark.parametrize("loop", [ReAct(), PlanThenAct()])
def test_agent_graph_matches_recipe_and_runtime_transitions(loop):
    provider = ScriptedProvider([ModelResponse(text="Plan"), ModelResponse(text="Answer")])
    agent = Agent(provider=provider, loop=loop)
    flow = loop.to_flow()
    assert agent.get_flow_graph() == build_mermaid(flow)
    assert agent.get_flow_graph(direction="TB").startswith("flowchart TB")
    run = asyncio.run(agent.run("Task"))
    assert run.status == "completed"
    for event in run.events:
        if event.kind == "transition":
            assert flow.edges[event.data["node"], event.data["action"]] == event.data["target"]
    assert ("plan" in flow.nodes) == isinstance(loop, PlanThenAct)


def test_export_rejects_uninspectable_loops_and_invalid_options(tmp_path):
    class DynamicLoop:
        async def run(self, runtime):
            runtime.finish("done")

    agent = Agent(provider=ScriptedProvider([]), loop=DynamicLoop())
    with pytest.raises(TypeError, match="to_flow"):
        agent.get_flow_graph()

    class Broken:
        def to_flow(self):
            return "not a flow"

    with pytest.raises(TypeError, match="must return a Flow"):
        build_dot(Broken())
    with pytest.raises(ValueError, match="direction"):
        build_mermaid(ReAct(), direction="injected")
    with pytest.raises(ValueError, match="png or .svg"):
        render_graph(ReAct(), tmp_path / "graph.html")


def test_render_calls_local_graphviz_and_compatibility_methods(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=b"rendered graph")

    monkeypatch.setattr("mickro_agent.viz.subprocess.run", fake_run)
    agent = Agent(provider=ScriptedProvider([]))
    png = agent.get_flow_graph_png(tmp_path / "nested" / "flow.png")
    assert png.read_bytes() == b"rendered graph"
    assert calls[-1][0] == ["dot", "-Tpng"]
    agent.render_graph(tmp_path / "flow.svg")
    assert calls[-1][0] == ["dot", "-Tsvg"]
    ReAct().to_flow().render(tmp_path / "flow.svg", direction="BT")
    assert b"rankdir=BT" in calls[-1][1]["input"]
    assert calls[-1][1]["timeout"] == 30


@pytest.mark.parametrize("failure", ["missing", "renderer", "timeout"])
def test_failed_render_does_not_overwrite_existing_file(tmp_path, monkeypatch, failure):
    path = tmp_path / "graph.svg"
    path.write_text("original")

    def fail(command, **kwargs):
        if failure == "missing":
            raise FileNotFoundError("dot")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise subprocess.CalledProcessError(1, command, stderr=b"bad graph")

    monkeypatch.setattr("mickro_agent.viz.subprocess.run", fail)
    error = subprocess.TimeoutExpired if failure == "timeout" else RuntimeError
    with pytest.raises(error):
        render_graph(ReAct(), path)
    assert path.read_text() == "original"


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz dot is optional")
def test_real_graphviz_renders_png_and_svg(tmp_path):
    png = render_graph(PlanThenAct(), tmp_path / "plan.png")
    svg = render_graph(ReAct(), tmp_path / "react.svg")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in svg.read_text() and "call_llm" in svg.read_text()
