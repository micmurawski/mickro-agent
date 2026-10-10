"""Session-scoped Python tools, independent of providers and data domains."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from ..runtime import Runtime
from ..session_log import validate_session_id
from ..tools import Hidden, Tool, Tools
from .sandbox import DockerPythonSandbox, PythonSandbox


class PythonSessions:
    """Own one lazily created sandbox per agent session; no global registry.

    Keep this owner alive across Agent.run continuations to retain interpreter
    state. FileSessionStore persists messages, not running Python interpreters.
    """

    def __init__(self, factory: Callable[[str], PythonSandbox] | None = None):
        self.factory = factory or (lambda _: DockerPythonSandbox())
        self._sandboxes: dict[str, PythonSandbox] = {}
        self._closed = False

    def get(self, session_id: str) -> PythonSandbox:
        validate_session_id(session_id)
        if self._closed:
            raise RuntimeError("PythonSessions is closed")
        if session_id not in self._sandboxes:
            self._sandboxes[session_id] = self.factory(session_id)
        return self._sandboxes[session_id]

    async def aclose(self) -> None:
        self._closed = True
        results = await asyncio.gather(
            *(sandbox.aclose() for sandbox in self._sandboxes.values()), return_exceptions=True
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise ExceptionGroup("Failed to close Python sandboxes", errors)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()


def python_tools(sessions: PythonSessions, *, description: str | None = None) -> Tools:
    """Expose execution/history/reset; container configuration stays application-owned."""

    async def execute_python(code: str, runtime: Hidden[Runtime]) -> dict:
        """Execute Python with persistent variables in this session's container.

        Use print() for output; the last expression's repr is also returned. Read
        stdout, stderr, result, and error. Errors may leave partial state changes.
        If state_lost is true, reset_python is required. /workspace is the working
        directory; input supplied by the application is available as `data`.
        """
        result = await sessions.get(runtime.run.session_id).execute(code)
        return result.model_dump(mode="json")

    async def python_history(runtime: Hidden[Runtime], limit: int = 10) -> list[dict]:
        """Read recent Python code and results for this session, oldest first."""
        return [entry.model_dump(mode="json") for entry in sessions.get(runtime.run.session_id).history(limit)]

    async def reset_python(runtime: Hidden[Runtime]) -> dict:
        """Discard this session's interpreter, temporary files and history.

        The next execution creates a fresh container and reruns application setup.
        Files on explicitly writable host mounts survive reset.
        """
        await sessions.get(runtime.run.session_id).reset()
        return {"status": "reset", "session_id": runtime.run.session_id}

    if description:
        execute_python.__doc__ += "\n" + description
    return Tools([Tool(execute_python), Tool(python_history), Tool(reset_python)])
