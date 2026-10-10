"""Typed tools with a strict boundary between model input and injected context."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, TypeVar, get_args, get_origin, get_type_hints

from pydantic import ConfigDict, TypeAdapter, ValidationError, create_model

from .session_log import SessionLogError
from .types import ToolCall, ToolSpec

T = TypeVar("T")
_INJECTED = object()
Hidden = Annotated[T, _INJECTED]
_JSON = TypeAdapter(Any)


@dataclass(frozen=True)
class ToolResult:
    result: Any = None
    error: str | None = None

    def content(self) -> str:
        if self.error is not None:
            return self.error
        if isinstance(self.result, str):
            return self.result
        return _JSON.dump_json(self.result, warnings="error").decode()


class Tool:
    def __init__(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        tags: Iterable[str] = (),
        injected: Iterable[str] = (),
    ):
        self.function = function
        self.name = name or function.__name__
        self.tags = frozenset(tags)
        self.signature = inspect.signature(function)
        hints = get_type_hints(function, include_extras=True)
        self.injected = set(injected)
        fields = {}
        for key, param in self.signature.parameters.items():
            if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                raise TypeError(f"Tool {self.name}: {key} must be a named parameter")
            annotation = hints.get(key, param.annotation)
            if get_origin(annotation) is Annotated and _INJECTED in get_args(annotation)[1:]:
                self.injected.add(key)
            if key in self.injected:
                continue
            if annotation is inspect.Parameter.empty:
                raise TypeError(f"Tool {self.name}: {key} needs a type annotation")
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[key] = (annotation, default)
        if self.injected - self.signature.parameters.keys():
            raise ValueError("Injected parameters must exist in the function signature")
        self.input_model = create_model(
            f"{self.name}Input", __config__=ConfigDict(extra="forbid", strict=True), **fields
        )
        self.spec = ToolSpec(
            name=self.name,
            description=inspect.getdoc(function) or "",
            parameters=self.input_model.model_json_schema(),
        )

    async def invoke(self, arguments: Mapping[str, Any], context: Mapping[str, Any]) -> ToolResult:
        try:
            validated = self.input_model.model_validate(dict(arguments))
        except ValidationError as exc:
            return ToolResult(error=f"Invalid arguments for {self.name}: {exc}")
        kwargs = {key: getattr(validated, key) for key in type(validated).model_fields}
        for key in self.injected:
            if key in context:
                kwargs[key] = context[key]
            elif self.signature.parameters[key].default is not inspect.Parameter.empty:
                kwargs[key] = self.signature.parameters[key].default
            else:
                raise ValueError(f"Tool {self.name} requires injected context {key!r}")
        # Input errors are recoverable by the model. Missing runtime dependencies are not.
        try:
            if inspect.iscoroutinefunction(self.function):
                result = await self.function(**kwargs)
            else:
                result = await asyncio.to_thread(self.function, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            result = result if isinstance(result, ToolResult) else ToolResult(result=result)
            result.content()  # Check serialization before handing output to a provider.
            return result
        except SessionLogError:
            raise  # Runtime-aware tools must not swallow a requested log's failure.
        except Exception as exc:  # noqa: BLE001 - Tool failures become observations; cancellation propagates.
            return ToolResult(error=f"{type(exc).__name__}: {exc}")

    def __or__(self, other: Tool | Tools) -> Tools:
        return Tools([self]) | other


def tool(function: Callable[..., Any] | None = None, *, name: str | None = None, tags: Iterable[str] = ()):
    """Use as @tool or @tool(tags=[...]); Hidden[T] parameters come from run context."""

    def decorate(fn: Callable[..., Any]) -> Tool:
        return Tool(fn, name=name, tags=tags)

    return decorate(function) if function is not None else decorate


class Tools:
    def __init__(self, tools: Iterable[Tool] = ()):
        registry = {}
        for item in tools:
            if item.name in registry and registry[item.name] is not item:
                raise ValueError(f"Duplicate tool name: {item.name}")
            registry[item.name] = item
        self._registry = MappingProxyType(registry)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(item.spec.model_copy(deep=True) for item in self._registry.values())

    def __or__(self, other: Tool | Tools) -> Tools:
        items = [other] if isinstance(other, Tool) else other._registry.values()
        return Tools([*self._registry.values(), *items])

    def select(self, *tags: str) -> Tools:
        return Tools(item for item in self._registry.values() if set(tags) <= item.tags)

    async def invoke(self, call: ToolCall, context: Mapping[str, Any]) -> ToolResult:
        item = self._registry.get(call.name)
        if item is None:
            return ToolResult(error=f"Unknown tool {call.name!r}; available: {', '.join(self._registry)}")
        return await item.invoke(call.arguments, context)
