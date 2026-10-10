import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from mickro_agent import Agent, Limits, ModelResponse, Run, Runtime, ToolCall, Tools
from mickro_agent.providers import ScriptedProvider
from mickro_agent.python import DockerPythonSandbox, Mount, PythonSessions, python_tools


class LocalWorker(DockerPythonSandbox):
    """Test-only transport: execute trusted test snippets in a separate process.

    This is deliberately not a public fallback for the Docker backend.
    """

    async def _create_container(self):
        pass

    def _attach_command(self):
        from mickro_agent.python import _worker

        return [sys.executable, "-I", "-u", str(Path(_worker.__file__).resolve())]

    async def _remove_container(self):
        pass


def test_state_expressions_stdout_stderr_native_output_and_exceptions():
    async def scenario():
        async with LocalWorker(data={"values": [1, 2, 3]}, setup="import math") as sandbox:
            first = await sandbox.execute("total = sum(data['values'])\nprint('hello')\ntotal")
            assert first.stdout == "hello\n" and first.result == "6" and first.error is None
            native = await sandbox.execute(
                "import os, subprocess, sys\n"
                "os.write(1, b'native\\n')\n"
                "subprocess.run([sys.executable, '-c', \"print('child')\"], check=True)\n"
                "print('warning', file=sys.stderr)\n"
                "math.sqrt(total * 6)"
            )
            assert native.stdout == "native\nchild\n"
            assert native.stderr == "warning\n" and native.result == "6.0"
            broken = await sandbox.execute("total += 1\nraise ValueError('oops')")
            assert "ValueError: oops" in broken.error and not broken.state_lost
            assert (await sandbox.execute("total")).result == "7"
            assert "SystemExit" in (await sandbox.execute("raise SystemExit(4)")).error
            assert "SyntaxError" in (await sandbox.execute("invalid ! syntax")).error
            assert "EOFError" in (await sandbox.execute("input('prompt')")).error
            assert (await sandbox.execute("total")).result == "7"

    asyncio.run(scenario())


def test_output_and_history_are_bounded_without_breaking_protocol():
    async def scenario():
        async with LocalWorker(max_output_bytes=100, history_limit=2) as sandbox:
            result = await sandbox.execute("import os\nos.write(1, b'x' * 1000000)\n'v' * 10000")
            assert len(result.stdout) == 100 and len(result.result) == 100 and result.truncated
            assert (await sandbox.execute("21 * 2")).result == "42"
            await sandbox.execute("True")
            assert [entry.code for entry in sandbox.history(100)] == ["21 * 2", "True"]
            assert sandbox.history(0) == []
            with pytest.raises(ValueError, match="negative"):
                sandbox.history(-1)

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["timeout", "cancel", "exit"])
def test_stopping_execution_loses_state_and_requires_explicit_reset(stop):
    async def scenario():
        async with LocalWorker(timeout=0.05, setup="seed = 10") as sandbox:
            assert (await sandbox.execute("seed")).result == "10"
            process = sandbox._process
            if stop == "cancel":
                task = asyncio.create_task(sandbox.execute("while True: pass"))
                await asyncio.sleep(0.02)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                code = "import os; os._exit(2)" if stop == "exit" else "while True: pass"
                result = await sandbox.execute(code)
                assert result.state_lost and result.error
            assert process.returncode is not None
            blocked = await sandbox.execute("seed")
            assert blocked.state_lost and "reset_python" in blocked.error
            await sandbox.reset()
            assert sandbox.history() == []
            assert (await sandbox.execute("seed")).result == "10"
            assert sandbox._process.pid != process.pid

    asyncio.run(scenario())


def test_concurrent_snippets_are_serialized_and_sessions_are_isolated():
    async def scenario():
        async with PythonSessions(lambda _: LocalWorker()) as sessions:
            first, second = sessions.get("first"), sessions.get("second")
            assert sessions.get("first") is first
            await first.execute("value = 0")
            results = await asyncio.gather(
                *(first.execute("import time\nold = value\ntime.sleep(.01)\nvalue = old + 1\nvalue") for _ in range(5))
            )
            assert [result.result for result in results] == ["1", "2", "3", "4", "5"]
            assert "NameError" in (await second.execute("value")).error
            processes = [first._process, second._process]
        assert all(process.returncode is not None for process in processes)
        with pytest.raises(RuntimeError, match="closed"):
            sessions.get("third")

    asyncio.run(scenario())


def test_agent_tools_keep_state_on_continuation_and_hide_configuration():
    async def scenario():
        async with PythonSessions(lambda _: LocalWorker(data=[10, 20])) as sessions:
            tools = python_tools(sessions, description="data contains the input numbers.")
            specs = {spec.name: spec for spec in tools.specs}
            assert set(specs["execute_python"].parameters["properties"]) == {"code"}
            assert "input numbers" in specs["execute_python"].description
            agent = Agent(
                provider=ScriptedProvider(
                    [
                        ModelResponse(
                            tool_calls=(
                                ToolCall(
                                    id="python",
                                    name="execute_python",
                                    arguments={
                                        "code": "count = globals().get('count', 0) + 1\n{'total': sum(data), 'visits': count}"
                                    },
                                ),
                            )
                        ),
                        ModelResponse(text="Calculated"),
                    ]
                ),
                tools=tools,
            )
            first = await agent.run("Calculate")
            second = await agent.run("Again", session_id=first.session_id)
            assert first.status == second.status == "completed"
            results = [json.loads(m.content) for m in second.messages if m.role == "tool"]
            assert results[0]["result"] == "{'total': 30, 'visits': 1}"
            assert results[1]["result"] == "{'total': 30, 'visits': 2}"

    asyncio.run(scenario())


def test_setup_error_and_missing_docker_are_reported_without_host_fallback():
    async def scenario():
        async with LocalWorker(setup="raise RuntimeError('bad setup')") as sandbox:
            result = await sandbox.execute("print('never')")
            assert "bad setup" in result.error and result.state_lost
            assert sandbox._process is None
        async with DockerPythonSandbox(docker="/does-not-exist/mikro-docker") as sandbox:
            result = await sandbox.execute("print('never')")
            assert "FileNotFoundError" in result.error and result.state_lost

    asyncio.run(scenario())


def test_docker_configuration_uses_explicit_mounts_and_no_published_ports(tmp_path):
    sandbox = DockerPythonSandbox(
        mounts=[Mount(tmp_path, "/inputs"), Mount(tmp_path, "/workspace", read_only=False)],
        env={"EXPLICIT": "configured"},
    )
    sandbox._name = "test-owned-container"
    command = sandbox._command()
    assert command[:2] == ["docker", "create"]
    assert "--pull=never" in command and "--read-only" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--cap-drop=ALL" in command and "--security-opt=no-new-privileges" in command
    assert not {"-p", "--publish", "--privileged"}.intersection(command)
    assert f"type=bind,source={tmp_path.resolve()},target=/inputs,readonly" in command
    assert f"type=bind,source={tmp_path.resolve()},target=/workspace" in command
    assert not any("/workspace:rw" in arg for arg in command)
    assert "EXPLICIT=configured" in command
    for target in ["relative", "/", "/opt/mickro_worker.py", "/some/../where"]:
        with pytest.raises(ValueError, match="mount"):
            DockerPythonSandbox(mounts=[Mount(tmp_path, target)])._command()


def test_cancel_during_creation_waits_for_creation_then_removes_container():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        actions = []

        class Delayed(LocalWorker):
            async def _create_container(self):
                entered.set()
                await release.wait()
                actions.append("created")

            async def _remove_container(self):
                actions.append("removed")

        sandbox = Delayed()
        task = asyncio.create_task(sandbox.execute("1"))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert actions == ["created", "removed"]
        assert sandbox._process is None

    asyncio.run(scenario())


def test_python_history_reset_and_outer_tool_deadline():
    async def scenario():
        async with PythonSessions(lambda _: LocalWorker(timeout=10)) as sessions:
            tools = python_tools(sessions)
            run = Run()
            context = {"runtime": Runtime(run, ScriptedProvider([]), Tools(), Limits())}
            await sessions.get(run.session_id).execute("value = 42")
            process = sessions.get(run.session_id)._process
            history = await tools.invoke(ToolCall(id="history", name="python_history"), context)
            assert history.result[0]["code"] == "value = 42"
            reset = await tools.invoke(ToolCall(id="reset", name="reset_python"), context)
            assert reset.result["status"] == "reset" and process.returncode is not None
            assert sessions.get(run.session_id).history() == []

            # Runtime cancellation must actually stop execution, even with a longer sandbox deadline.
            agent = Agent(
                provider=ScriptedProvider(
                    [
                        ModelResponse(
                            tool_calls=(
                                ToolCall(id="loop", name="execute_python", arguments={"code": "while True: pass"}),
                            )
                        )
                    ]
                ),
                tools=tools,
                limits=Limits(tool_timeout=0.03),
            )
            # Warm this session first so the assertion checks execution rather than startup.
            await sessions.get("deadline").execute("1")
            process = sessions.get("deadline")._process
            result = await agent.run("Run", session_id="deadline")
            assert result.status == "timeout"
            assert process.returncode is not None
            assert (await sessions.get("deadline").execute("1")).state_lost

    asyncio.run(scenario())


def test_python_example_uses_persistent_variables_and_exports_generic_artifact(tmp_path, monkeypatch):
    import python_rlm

    def sandbox(**kwargs):
        return LocalWorker(data=kwargs["data"], setup=f"import os\nos.chdir({str(tmp_path)!r})")

    monkeypatch.setattr(python_rlm, "DockerPythonSandbox", sandbox)
    run = asyncio.run(python_rlm.demo())
    assert run.status == "completed", run.error
    assert json.loads((tmp_path / "summary.json").read_text()) == {"matches": 2, "document_ids": ["a", "c"]}
    assert '"matches": 2' in run.output


@pytest.mark.skipif(
    os.getenv("MICKRO_TEST_DOCKER") != "1",
    reason="Set MICKRO_TEST_DOCKER=1 with Docker running and a local Python image",
)
def test_real_docker_state_mounts_timeout_and_cleanup(tmp_path):
    async def scenario():
        (tmp_path / "input.txt").write_text("hello")
        async with DockerPythonSandbox(mounts=[Mount(tmp_path, "/inputs")], timeout=0.5) as sandbox:
            first = await sandbox.execute("from pathlib import Path\nx = 40\nPath('/inputs/input.txt').read_text()")
            assert first.result == "'hello'", first.error
            assert (await sandbox.execute("x + 2")).result == "42"
            denied = await sandbox.execute("Path('/inputs/input.txt').write_text('changed')")
            assert denied.error and (tmp_path / "input.txt").read_text() == "hello"
            assert (await sandbox.execute("while True: pass")).state_lost
            assert sandbox._name is None

    asyncio.run(scenario())
