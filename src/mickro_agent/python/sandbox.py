"""Persistent Python execution in an explicitly configured Docker container."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol
from uuid import uuid4

from pydantic import JsonValue

from ..types import Record


class PythonResult(Record):
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    error: str | None = None
    truncated: bool = False
    state_lost: bool = False
    duration: float = 0


class Execution(Record):
    code: str
    result: PythonResult


class PythonSandbox(Protocol):
    async def execute(self, code: str) -> PythonResult: ...
    async def reset(self) -> None: ...
    async def aclose(self) -> None: ...
    def history(self, limit: int = 10) -> list[Execution]: ...


@dataclass(frozen=True)
class Mount:
    source: str | Path
    target: str
    read_only: bool = True


class DockerPythonSandbox:
    """One interpreter, serialized snippets, bounded output, explicit lifecycle.

    Docker CLI and a running local daemon are required. Images must already exist;
    no SDK, port, automatic pip install, or implicit host execution is used.
    """

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        mounts: Sequence[Mount] = (),
        env: Mapping[str, str] | None = None,
        setup: str = "",
        data: JsonValue = None,
        network: str = "none",
        memory: str = "512m",
        cpus: float = 1,
        pids_limit: int = 64,
        timeout: float = 30,
        startup_timeout: float = 30,
        max_output_bytes: int = 32_768,
        max_code_bytes: int = 262_144,
        history_limit: int = 100,
        docker: str = "docker",
    ):
        for name, value in (("timeout", timeout), ("startup_timeout", startup_timeout), ("cpus", cpus)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if min(pids_limit, max_output_bytes, max_code_bytes, history_limit) < 1:
            raise ValueError("Resource and history limits must be positive")
        self.image, self.mounts, self.env = image, tuple(mounts), dict(env or {})
        self.setup, self._data_json = setup, json.dumps(data, allow_nan=False)
        self.network, self.memory, self.cpus, self.pids_limit = network, memory, cpus, pids_limit
        self.timeout, self.startup_timeout = timeout, startup_timeout
        self.max_output_bytes, self.max_code_bytes = max_output_bytes, max_code_bytes
        self.docker = docker
        self._history: deque[Execution] = deque(maxlen=history_limit)
        self._lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None
        self._create_task: asyncio.Task | None = None
        self._may_exist = False
        self._diagnostics = bytearray()
        self._name: str | None = None
        self._closed = False
        self._lost = False

    def _command(self) -> list[str]:
        worker = Path(__file__).with_name("_worker.py").resolve()
        command = [
            self.docker,
            "create",
            "--rm",
            "-i",
            "--pull=never",
            "--name",
            self._name,
            "--network",
            self.network,
            "--memory",
            self.memory,
            "--cpus",
            str(self.cpus),
            "--pids-limit",
            str(self.pids_limit),
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
            "--workdir",
            "/workspace",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--mount",
            f"type=bind,source={worker},target=/opt/mickro_worker.py,readonly",
        ]
        targets = set()
        for mount in self.mounts:
            source = Path(mount.source).expanduser().resolve(strict=True)
            target = PurePosixPath(mount.target)
            if (
                not target.is_absolute()
                or ".." in target.parts
                or str(target) in {"/", "/opt", "/opt/mickro_worker.py", "/tmp"}
                or str(target) in targets
                or "," in str(source)
                or "," in str(target)
            ):
                raise ValueError(f"Unsupported or duplicate mount target: {mount.target}")
            targets.add(str(target))
            spec = f"type=bind,source={source},target={target}"
            command.extend(["--mount", spec + (",readonly" if mount.read_only else "")])
        if "/workspace" not in targets:
            command.extend(["--tmpfs", "/workspace:rw,nosuid,nodev,size=128m,mode=1777"])
        for key, value in self.env.items():
            if not key or "=" in key or "\0" in key or "\0" in value:
                raise ValueError("Invalid container environment entry")
            command.extend(["--env", f"{key}={value}"])
        command.extend(["--entrypoint", "python", self.image, "-I", "-u", "/opt/mickro_worker.py"])
        return command

    async def _create_container(self):
        process = await asyncio.create_subprocess_exec(
            *self._command(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._may_exist = True
        try:
            async with asyncio.timeout(self.startup_timeout):
                _, error = await process.communicate()
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode:
            self._may_exist = False
            raise RuntimeError(f"Cannot create Python container: {error.decode(errors='replace')}")

    def _attach_command(self) -> list[str]:
        return [self.docker, "start", "--attach", "--interactive", self._name]

    async def _diagnostic_reader(self):
        assert self._process is not None and self._process.stderr is not None
        while chunk := await self._process.stderr.read(8192):
            self._diagnostics.extend(chunk)
            del self._diagnostics[:-8192]

    async def _start(self):
        self._name = f"mikro-python-{uuid4().hex}"
        self._diagnostics.clear()
        # Let a create request settle before cleanup, even if its caller cancels.
        # Otherwise a late Docker create could leave an orphan after `rm` returned.
        self._create_task = asyncio.create_task(self._create_container())
        await asyncio.shield(self._create_task)
        self._process = await asyncio.create_subprocess_exec(
            *self._attach_command(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=self.max_output_bytes * 32 + 4096,
        )
        self._stderr_task = asyncio.create_task(self._diagnostic_reader())
        assert self._process.stdout is not None
        ready = await self._process.stdout.readline()
        if not ready or json.loads(ready) != {"ready": True}:
            raise RuntimeError("Python container did not become ready; check Docker and the local image")
        initial = f"import json as _json\ndata = _json.loads({self._data_json!r})\ndel _json\n" + self.setup
        result = await self._exchange(initial)
        if result.error:
            raise RuntimeError(f"Python initialization failed: {result.error}")
        self._lost = False

    async def _exchange(self, code: str) -> PythonResult:
        assert self._process is not None and self._process.stdin is not None and self._process.stdout is not None
        request_id = uuid4().hex
        self._process.stdin.write(
            (
                json.dumps(
                    {
                        "id": request_id,
                        "code": code,
                        "max_output_bytes": self.max_output_bytes,
                    }
                )
                + "\n"
            ).encode()
        )
        await self._process.stdin.drain()
        line = await self._process.stdout.readline()
        if not line:
            raise RuntimeError("Python interpreter exited before returning a result")
        response = json.loads(line)
        if response.pop("id", None) != request_id:
            raise RuntimeError("Python response did not match the request")
        return PythonResult.model_validate(response)

    async def _remove_container(self):
        if self._name is None or not self._may_exist:
            return
        process = await asyncio.create_subprocess_exec(
            self.docker,
            "rm",
            "--force",
            self._name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with asyncio.timeout(10):
                _, error = await process.communicate()
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode and b"No such container" not in error:
            raise RuntimeError(f"Could not remove Python container {self._name}: {error.decode(errors='replace')}")
        self._may_exist = False

    async def _stop(self):
        if self._create_task is not None:
            await asyncio.gather(asyncio.shield(self._create_task), return_exceptions=True)
            self._create_task = None
        try:
            await self._remove_container()
        finally:
            if self._process is not None:
                if self._process.returncode is None:
                    self._process.kill()
                await self._process.wait()
                self._process = None
            if self._stderr_task is not None:
                await self._stderr_task
                self._stderr_task = None
        self._name = None

    async def _invalidate(self):
        self._lost = True
        await self._stop()

    async def execute(self, code: str) -> PythonResult:
        if len(code.encode()) > self.max_code_bytes:
            raise ValueError(f"Code exceeds {self.max_code_bytes} bytes")
        async with self._lock:
            if self._closed:
                raise RuntimeError("Python sandbox is closed")
            if self._lost:
                return PythonResult(
                    error="Python state was lost. Call reset_python before executing again.", state_lost=True
                )
            started = time.monotonic()
            try:
                if self._process is None:
                    async with asyncio.timeout(self.startup_timeout):
                        await self._start()
                async with asyncio.timeout(self.timeout):
                    result = await self._exchange(code)
            except asyncio.CancelledError:
                result = PythonResult(error="Execution cancelled; Python state lost.", state_lost=True)
                self._history.append(Execution(code=code, result=result))
                await self._invalidate()
                raise
            except Exception as exc:  # noqa: BLE001 - Execution/transport failures become observations.
                await self._invalidate()
                detail = self._diagnostics.decode(errors="replace")
                reason = (
                    "Execution deadline exceeded" if isinstance(exc, TimeoutError) else f"{type(exc).__name__}: {exc}"
                )
                result = PythonResult(error=f"{reason}. Python state lost. {detail}".rstrip(), state_lost=True)
            result = result.model_copy(update={"duration": time.monotonic() - started})
            self._history.append(Execution(code=code, result=result))
            return result

    def history(self, limit: int = 10) -> list[Execution]:
        if limit < 0:
            raise ValueError("History limit cannot be negative")
        return [entry.model_copy(deep=True) for entry in list(self._history)[-limit:]] if limit else []

    async def reset(self) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Python sandbox is closed")
            await self._stop()
            self._lost = False
            self._history.clear()

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            await self._stop()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
