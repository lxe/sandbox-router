"""Opt-in live Orble tests: ORBLE_RUN_E2E=1 with ORBLE_API_KEY set.

Each test makes its own sandboxes and destroys them in ``finally``; snapshots
are deleted the same way. ORBLE_DOMAIN picks another Orble deployment.
"""

from __future__ import annotations

import os
import time

import pytest

from bespokelabs.sandbox import Sandbox, SandboxClient, SandboxSessionState
from bespokelabs.sandbox.exceptions import SandboxNotFoundError

pytestmark = pytest.mark.skipif(
    os.environ.get("ORBLE_RUN_E2E") != "1"
    or not os.environ.get("ORBLE_API_KEY"),
    reason="set ORBLE_RUN_E2E=1 and ORBLE_API_KEY to run live Orble tests",
)

SLEEPER = "nohup python3 -c 'import time; time.sleep(9999)' >/dev/null 2>&1 &"
COUNT_SLEEPERS = "pgrep -fc 'sleep.9999' || true"


def _sdk():
    from e2b import Sandbox as E2BSandbox

    connection = {
        "api_key": os.environ["ORBLE_API_KEY"],
        "domain": os.environ.get("ORBLE_DOMAIN") or "orble.tech",
    }
    return E2BSandbox, connection


def _assert_gone(state: SandboxSessionState) -> None:
    with pytest.raises(SandboxNotFoundError):
        SandboxClient("orble").resume(state).destroy()


def test_lifecycle(tmp_path):
    import httpx

    sb = Sandbox(
        "orble",
        timeout_secs=300,
        env_vars={"GREETING": "hi"},
        workdir="/home/user/project",
    )
    state = sb.session_state()
    try:
        assert sb.execute_code("print(6 * 7)").stdout == "42\n"
        failed = sb.execute_code("raise SystemExit(3)")
        assert failed.exit_code == 3
        result = sb.execute_command("sh", ["-c", "echo $GREETING; pwd"])
        assert result.stdout == "hi\n/home/user/project\n"
        assert sb.execute_command("nproc").stdout.strip() == "2"

        sb.write_file("/home/user/project/data.bin", b"\x00\xff")
        assert sb.read_file("/home/user/project/data.bin") == b"\x00\xff"
        names = [f.path for f in sb.list_files("/home/user/project")]
        assert "/home/user/project/data.bin" in names
        local = tmp_path / "data.bin"
        sb.download_file("/home/user/project/data.bin", str(local))
        sb.upload_file(str(local), "/tmp/copy.bin")
        assert sb.read_file("/tmp/copy.bin") == b"\x00\xff"

        assert sb.resource_usage()

        sb.execute_command(
            "sh",
            [
                "-c",
                "nohup python3 -m http.server 8000 --bind 0.0.0.0 "
                ">/dev/null 2>&1 &",
            ],
        )
        port = sb.expose_port(8000)
        assert port.url.startswith("https://")
        for _ in range(20):
            response = httpx.get(port.url, headers=port.headers)
            if response.status_code == 200:
                break
            time.sleep(0.5)
        assert response.status_code == 200
        assert "data.bin" in response.text
        assert httpx.get(port.url).status_code != 200
    finally:
        sb.destroy()
    _assert_gone(state)


def test_long_command_is_not_cut_off_at_sixty_seconds():
    with Sandbox("orble", timeout_secs=300) as sb:
        result = sb.execute_command("sh", ["-c", "sleep 65; echo done"])
        assert result.stdout == "done\n"


def test_snapshot_keeps_files_and_processes():
    sdk, connection = _sdk()
    source = Sandbox("orble", timeout_secs=300)
    snapshot_id = None
    copy = None
    try:
        source.write_file("/tmp/marker", b"from the source")
        source.execute_command("sh", ["-c", SLEEPER])
        snapshot_id = source.snapshot().snapshot_id
        copy = Sandbox("orble", snapshot_id=snapshot_id, timeout_secs=300)
        assert copy.read_file("/tmp/marker") == b"from the source"
        assert copy.execute_command("sh", ["-c", COUNT_SLEEPERS]).stdout != (
            "0\n"
        )
    finally:
        if copy is not None:
            copy.destroy()
        source.destroy()
        if snapshot_id is not None:
            sdk.delete_snapshot(snapshot_id, **connection)


def test_fork_and_suspend_through_the_sdk():
    sdk, connection = _sdk()
    sb = Sandbox("orble", timeout_secs=300)
    forks = []
    try:
        sb.write_file("/tmp/marker", b"before the fork")
        sb.execute_command("sh", ["-c", SLEEPER])
        native = sdk.connect(sb.provider_resource_id, **connection)
        forks = [
            SandboxClient("orble").resume(
                SandboxSessionState(
                    backend="orble", data={"sandbox_id": f.sandbox_id}
                )
            )
            for f in native.fork(count=2)
        ]
        for fork in forks:
            assert fork.read_file("/tmp/marker") == b"before the fork"
            count = fork.execute_command("sh", ["-c", COUNT_SLEEPERS])
            assert count.stdout != "0\n"
            fork.write_file("/tmp/marker", fork.provider_resource_id.encode())
        assert sb.read_file("/tmp/marker") == b"before the fork"

        native.pause()
        assert sdk.get_info(sb.provider_resource_id, **connection).state == (
            "paused"
        )
        woken = Sandbox.resume(sb.session_state())
        assert woken.read_file("/tmp/marker") == b"before the fork"
        count = woken.execute_command("sh", ["-c", COUNT_SLEEPERS])
        assert count.stdout != "0\n"
    finally:
        for fork in forks:
            fork.destroy()
        sb.destroy()


def test_agent_preset():
    with Sandbox("orble", preset="claude-code-codex", timeout_secs=300) as sb:
        assert sb.execute_command("claude", ["--version"]).exit_code == 0
        assert sb.execute_command("codex", ["--version"]).exit_code == 0


def test_docker_inside_the_sandbox():
    with Sandbox("orble", timeout_secs=300) as sb:
        result = sb.execute_command("docker", ["run", "--rm", "hello-world"])
        assert result.exit_code == 0
        assert "Hello from Docker!" in result.stdout


def test_suspend_on_timeout_then_resume():
    sdk, connection = _sdk()
    sb = Sandbox(
        "orble",
        timeout_secs=15,
        backend_options={"lifecycle": {"on_timeout": "pause"}},
    )
    state = sb.session_state()
    woken = None
    try:
        sb.write_file("/tmp/marker", b"before the timeout")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            info = sdk.get_info(sb.provider_resource_id, **connection)
            if info.state != "running":
                break
            time.sleep(5)
        assert info.state == "paused"
        woken = Sandbox.resume(state)
        assert woken.read_file("/tmp/marker") == b"before the timeout"
    finally:
        (woken or sb).destroy()
