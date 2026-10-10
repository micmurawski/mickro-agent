"""Compose nodes with >> and - action >>, then snapshot them into a runnable flow."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import Runtime

END = "__end__"
Step = Callable[["Runtime"], str | None | Awaitable[str | None]]


class Node:
    """A graph-building handle. Execution state lives on Runtime, never on this node.

    Wire bound @node methods on an instance. Their cache belongs to that instance,
    so no global descriptor cache retains the owner.
    """

    def __init__(self, function: Step, *, name: str | None = None, metadata: dict | None = None):
        if not callable(function):
            raise TypeError("A node requires a callable")
        self.function = function
        self.name = name or function.__name__
        if not self.name or self.name == END:
            raise ValueError("Node names must be nonempty; __end__ is reserved")
        self._successors: dict[str, Node | str] = {}
        self.metadata = deepcopy(metadata or {})
        self._binding_key = f"__mickro_node_{id(self)}"

    def __call__(self, runtime: Runtime):
        return self.function(runtime)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        try:
            cache = vars(instance)
        except TypeError as exc:
            raise TypeError("@node methods require an instance with __dict__") from exc
        if self._binding_key not in cache:
            cache[self._binding_key] = Node(
                self.function.__get__(instance, owner), name=self.name, metadata=self.metadata
            )
        return cache[self._binding_key]

    def next(self, target: Node | str, action: str = "default") -> Node | str:
        if not isinstance(action, str) or not action:
            raise TypeError("A transition action must be a nonempty string")
        if not isinstance(target, Node) and target != END:
            raise TypeError("A transition target must be a Node or END")
        if action in self._successors:
            raise ValueError(f"Duplicate transition from {self.name!r} for {action!r}")
        self._successors[action] = target
        return target

    def __rshift__(self, target: Node | str) -> Node | str:
        return self.next(target)

    def __sub__(self, action: str) -> _Transition:
        if not isinstance(action, str) or not action:
            raise TypeError("A transition action must be a nonempty string")
        return _Transition(self, action)


@dataclass(frozen=True)
class _Transition:
    source: Node
    action: str

    def __rshift__(self, target: Node | str) -> Node | str:
        return self.source.next(target, self.action)


def node(function: Step | None = None, *, name: str | None = None, metadata: dict | None = None):
    """Wrap a function/bound method, or use @node / @node(name='...')."""

    def decorate(fn: Step) -> Node:
        return Node(fn, name=name, metadata=metadata)

    return decorate(function) if function is not None else decorate


def _compile(start: Node) -> tuple[dict[str, Step], dict[tuple[str, str], str], dict[str, dict]]:
    nodes, edges, names, metadata = {}, {}, {}, {}
    pending = [start]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if current.name in names and names[current.name] is not current:
            raise ValueError(f"Duplicate node name: {current.name!r}; give each node a distinct name")
        names[current.name] = current
        nodes[current.name] = current.function
        metadata[current.name] = deepcopy(current.metadata)
        # A leaf ends on the default action. Named actions still need explicit routes.
        for action, target in (current._successors or {"default": END}).items():
            edges[current.name, action] = target.name if isinstance(target, Node) else target
            if isinstance(target, Node):
                pending.append(target)
    return nodes, edges, metadata


@dataclass(frozen=True, init=False)
class Flow:
    nodes: Mapping[str, Step]
    edges: Mapping[tuple[str, str], str]
    start: str
    node_metadata: Mapping[str, dict]

    def __init__(
        self,
        start: Node | str,
        *,
        nodes: Mapping[str, Step] | None = None,
        edges: Mapping[tuple[str, str], str] | None = None,
    ):
        metadata = {}
        if isinstance(start, Node):
            if nodes is not None or edges is not None:
                raise ValueError("Use a DSL start node or explicit mappings, not both")
            nodes, edges, metadata = _compile(start)
            start = start.name
        elif nodes is None or edges is None:
            raise ValueError("A string start requires explicit nodes and edges")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "nodes", MappingProxyType(dict(nodes)))
        object.__setattr__(self, "edges", MappingProxyType(dict(edges)))
        object.__setattr__(self, "node_metadata", MappingProxyType(metadata))
        if self.start not in self.nodes or END in self.nodes:
            raise ValueError("Flow needs a valid start node; __end__ is reserved")
        for (source, _), target in self.edges.items():
            if source not in self.nodes or (target != END and target not in self.nodes):
                raise ValueError(f"Invalid edge: {source!r} -> {target!r}")

    def to_mermaid(self, *, direction: str = "LR") -> str:
        from .viz import build_mermaid

        return build_mermaid(self, direction=direction)

    def to_dot(self, *, direction: str = "LR") -> str:
        from .viz import build_dot

        return build_dot(self, direction=direction)

    def render(self, filename: str | Path, *, direction: str = "LR") -> Path:
        from .viz import render_graph

        return render_graph(self, filename, direction=direction)

    async def run(self, runtime: Runtime) -> None:
        current = self.start
        while current != END:
            runtime.step(current, metadata=self.node_metadata.get(current))
            action = self.nodes[current](runtime)
            if inspect.isawaitable(action):
                action = await action
            if action is None:
                action = "default"
            if not isinstance(action, str):
                raise TypeError(f"Node {current!r} must return a string action or None")
            key = (current, action)
            if key not in self.edges:
                raise ValueError(f"No transition from {current!r} for action {action!r}")
            target = self.edges[key]
            runtime.run.emit("transition", node=current, action=action, target=target)
            current = target
