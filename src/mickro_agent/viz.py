"""Export compiled flow graphs without running nodes or contacting a service."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Protocol

from .flow import END, Flow


class FlowSource(Protocol):
    def to_flow(self) -> Flow: ...


def _flow(source: Flow | FlowSource) -> Flow:
    if isinstance(source, Flow):
        return source
    factory = getattr(source, "to_flow", None)
    if not callable(factory):
        raise TypeError("Graph export requires a Flow or a loop implementing to_flow() -> Flow")
    result = factory()
    if not isinstance(result, Flow):
        raise TypeError("to_flow() must return a Flow")
    return result


def _direction(direction: str) -> str:
    if direction not in {"LR", "RL", "TB", "BT"}:
        raise ValueError("direction must be LR, RL, TB, or BT")
    return direction


def _mermaid_label(text: str) -> str:
    # User labels never become Mermaid identifiers, markup, or graph statements.
    return "".join(
        "<br/>" if char == "\n" else char if char.isalnum() or char in " _-." else f"#{ord(char)};" for char in text
    )


def build_mermaid(source: Flow | FlowSource, *, direction: str = "LR") -> str:
    flow = _flow(source)
    names = sorted(flow.nodes)
    ids = {name: f"n{index}" for index, name in enumerate(names)}
    ids[END] = "end_node"
    lines = [f"flowchart {_direction(direction)}", '    start_node(["START"])']
    lines.extend(f'    {ids[name]}["{_mermaid_label(name)}"]' for name in names)
    if END in flow.edges.values():
        lines.append('    end_node(["END"])')
    lines.append(f"    start_node --> {ids[flow.start]}")
    for (name, action), target in sorted(flow.edges.items()):
        edge = "-.->" if action == "default" else f'-->|"{_mermaid_label(action)}"|'
        lines.append(f"    {ids[name]} {edge} {ids[target]}")
    return "\n".join(lines) + "\n"


def build_dot(source: Flow | FlowSource, *, direction: str = "LR") -> str:
    flow = _flow(source)
    names = sorted(flow.nodes)
    ids = {name: f"n{index}" for index, name in enumerate(names)}
    ids[END] = "end_node"
    lines = ["digraph flow {", f"    rankdir={_direction(direction)};", '    start_node [label="START", shape=oval];']
    lines.extend(f"    {ids[name]} [label={json.dumps(name, ensure_ascii=False)}, shape=box];" for name in names)
    if END in flow.edges.values():
        lines.append('    end_node [label="END", shape=oval];')
    lines.append(f"    start_node -> {ids[flow.start]};")
    for (name, action), target in sorted(flow.edges.items()):
        attrs = "style=dashed" if action == "default" else f"label={json.dumps(action, ensure_ascii=False)}"
        lines.append(f"    {ids[name]} -> {ids[target]} [{attrs}];")
    return "\n".join([*lines, "}", ""])


def render_graph(
    source: Flow | FlowSource,
    filename: str | Path,
    *,
    direction: str = "LR",
    format: str | None = None,
    timeout: float = 30,
) -> Path:
    """Render locally using Graphviz's `dot` executable; supports PNG and SVG.

    Mermaid and DOT exports do not require Graphviz. No graph is sent to a remote
    rendering service. The output file is written only after rendering succeeds.
    """
    path = Path(filename)
    format = path.suffix.lstrip(".").lower() if format is None else format
    if format not in {"png", "svg"}:
        raise ValueError("Use a .png or .svg filename, or format='png'/'svg'")
    diagram = build_dot(source, direction=direction)
    try:
        result = subprocess.run(
            ["dot", f"-T{format}"], input=diagram.encode("utf-8"), capture_output=True, check=True, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Image export requires Graphviz's 'dot' executable on PATH. Mermaid/DOT text export needs no extra dependencies."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Graphviz rendering failed: {exc.stderr.decode(errors='replace')}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.stdout)
    return path


def to_png(source: Flow | FlowSource, filename: str | Path, *, direction: str = "LR") -> Path:
    """Compatibility spelling from incident-assistant's framework.viz."""
    return render_graph(source, filename, direction=direction, format="png")
