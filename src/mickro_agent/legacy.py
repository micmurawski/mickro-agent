"""Opt-in bridges for incident-assistant; importing this module needs no old dependencies."""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, get_type_hints

from .providers.legacy import LegacyProvider
from .tools import Tool, ToolResult

__all__ = ["LegacyProvider", "migrate_tool"]


def migrate_tool(old_tool: Any) -> Tool:
    """Adapt an existing @tool object, preserving its injection list and ToolResult."""
    original = type(old_tool)._func

    @functools.wraps(original)
    async def execute(**kwargs):
        if inspect.iscoroutinefunction(original):
            result = await original(**kwargs)
        else:
            result = await asyncio.to_thread(original, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        if hasattr(result, "result") and hasattr(result, "error"):
            return ToolResult(result=result.result, error=result.error)
        return result

    execute.__annotations__ = get_type_hints(original, include_extras=True)
    return Tool(execute, name=old_tool.name, tags=old_tool.tags, injected=old_tool._inject_params)
