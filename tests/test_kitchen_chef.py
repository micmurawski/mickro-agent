import asyncio

import pytest
from kitchen_chef import Kitchen, demo


def test_chef_handles_messages_and_is_woken_by_every_job_finishing():
    transcript = []
    run = asyncio.run(demo(report=transcript.append))
    assert run.status == "completed", run.error
    jobs = {job["dish"]: job for job in run.state["jobs"]}
    assert jobs["pasta"]["status"] == jobs["sauce"]["status"] == "succeeded"
    assert jobs["toast"]["status"] == "killed" and jobs["toast"]["exit_code"] != 0
    observations = [event.data["event"] for event in run.events if event.kind == "mailbox_event"]
    finished = [event for event in observations if event["kind"] == "JobFinished"]
    assert len(finished) == 3 and len({event["job_id"] for event in finished}) == 3
    assert observations[-1]["kind"] == "Shutdown"
    assert any("Dinner status:" in line for line in transcript)
    assert any("Plated pasta after its completion event" in line for line in transcript)
    assert any("Plated sauce after its completion event" in line for line in transcript)
    assert any("Observed toast killed" in line for line in transcript)
    # Completion triggers a new model turn even though no further user message is sent.
    for index, event in enumerate(run.events):
        if event.kind == "mailbox_event" and event.data["event"]["kind"] == "JobFinished":
            assert any(later.kind == "model_request" for later in run.events[index + 1 :])


@pytest.mark.parametrize("fail", [False, True])
def test_completion_is_queued_even_when_agent_is_not_listening(fail):
    async def scenario():
        mailbox = asyncio.Queue()
        kitchen = Kitchen(mailbox)
        try:
            handle = await kitchen.start("soup", 0.01, fail=fail)
            assert handle["status"] == "running"
            await asyncio.wait_for(kitchen.wait_all(), timeout=5)
            events = []
            while not mailbox.empty():
                events.append(mailbox.get_nowait())
            finished = [event for event in events if event["kind"] == "JobFinished"]
            assert len(finished) == 1
            assert finished[0]["status"] == ("failed" if fail else "succeeded")
            assert finished[0]["job_id"] == handle["job_id"]
        finally:
            await kitchen.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("force,expected", [(False, "cancelled"), (True, "killed")])
def test_stop_is_acknowledged_then_confirmed_by_one_completion_event(force, expected):
    async def scenario():
        mailbox = asyncio.Queue()
        kitchen = Kitchen(mailbox)
        try:
            handle = await kitchen.start("slow roast", 30)
            receipt = kitchen.stop(handle["job_id"], force=force)
            assert receipt["status"] == "stopping"
            await asyncio.wait_for(kitchen.wait_all(), timeout=5)
            assert kitchen.inspect(handle["job_id"])["status"] == expected
            kitchen.stop(handle["job_id"], force=force)  # Repeated stop doesn't emit another outcome.
            finished = []
            while not mailbox.empty():
                event = mailbox.get_nowait()
                if event["kind"] == "JobFinished":
                    finished.append(event)
            assert len(finished) == 1 and finished[0]["status"] == expected
        finally:
            await kitchen.aclose()

    asyncio.run(scenario())


def test_cleanup_reaps_background_processes():
    async def scenario():
        kitchen = Kitchen(asyncio.Queue())
        await kitchen.start("slow roast", 30)
        await asyncio.wait_for(kitchen.aclose(), timeout=5)
        assert all(job.process.returncode is not None and job.watcher.done() for job in kitchen.jobs.values())
        with pytest.raises(RuntimeError, match="closed"):
            await kitchen.start("another dish", 1)

    asyncio.run(scenario())
