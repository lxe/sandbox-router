"""Offline Orble adapter tests. No provider SDK or credentials required."""

from __future__ import annotations

import socket
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from bespokelabs.sandbox import (
    NetworkPolicy,
    Sandbox,
    SandboxClient,
    SandboxSessionState,
)
from bespokelabs.sandbox.backends import BACKENDS
from bespokelabs.sandbox.backends.orble import OrbleClient
from bespokelabs.sandbox.exceptions import (
    BackendNotInstalledError,
    ErrorOutcome,
    FeatureNotSupportedError,
    SandboxAuthenticationError,
    SandboxConfigurationError,
    SandboxCreationError,
    SandboxExecutionError,
    SandboxNotFoundError,
    SandboxTimeoutError,
)
from bespokelabs.sandbox.protocols import (
    SandboxBackendClient,
    SandboxBackendSession,
)
from bespokelabs.sandbox.types import SandboxConfig


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Offline tests must not open sockets")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


class FakeSandbox:
    """The part of the e2b 2.64 ``Sandbox`` the adapter uses."""

    creates: list[dict] = []
    connects: list[tuple[str, dict]] = []
    fail: Exception | None = None
    killed_ids: list[str] = []

    def __init__(self, sandbox_id: str, kwargs: dict) -> None:
        self.sandbox_id = sandbox_id
        self.kwargs = kwargs
        self.traffic_access_token = "traffic-token"
        self.runs: list[tuple[str, dict]] = []
        self.files_data: dict[str, bytes] = {}
        self.killed = 0
        self.exit_code = 0
        self.commands = SimpleNamespace(run=self._run)
        self.files = SimpleNamespace(
            read=self._read, write=self._write, list=self._list
        )

    @classmethod
    def create(cls, **kwargs):
        if cls.fail is not None:
            raise cls.fail
        cls.creates.append(kwargs)
        return cls(f"sbx{len(cls.creates)}", kwargs)

    @classmethod
    def connect(cls, sandbox_id, **kwargs):
        if cls.fail is not None:
            raise cls.fail
        cls.connects.append((sandbox_id, kwargs))
        return cls(sandbox_id, kwargs)

    def _run(self, cmd, **kwargs):
        self.runs.append((cmd, kwargs))
        e2b = sys.modules["e2b"]
        if self.exit_code:
            raise e2b.CommandExitException(
                stdout="out", stderr="err", exit_code=self.exit_code
            )
        return SimpleNamespace(stdout="out", stderr="", exit_code=0)

    def _read(self, path, format="text"):
        assert format == "bytes"
        return bytearray(self.files_data[path])

    def _write(self, path, content):
        self.files_data[path] = (
            content.encode() if isinstance(content, str) else content
        )

    def _list(self, path):
        e2b = sys.modules["e2b"]
        return [
            SimpleNamespace(path=f"{path}/a", type=e2b.FileType.FILE, size=2),
            SimpleNamespace(path=f"{path}/d", type=e2b.FileType.DIR, size=0),
        ]

    def get_host(self, port):
        return f"{port}-{self.sandbox_id}.e2b.orble.tech"

    def create_snapshot(self):
        return SimpleNamespace(snapshot_id="ab12-1", names=["ab12-1"])

    def kill(self):
        self.killed += 1
        FakeSandbox.killed_ids.append(self.sandbox_id)
        return True


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.setenv("ORBLE_API_KEY", "offline-placeholder")
    monkeypatch.delenv("ORBLE_DOMAIN", raising=False)
    monkeypatch.setenv("E2B_API_KEY", "must-not-be-used")
    root = ModuleType("e2b")
    for name in (
        "AuthenticationException",
        "InvalidArgumentException",
        "NotFoundException",
        "RateLimitException",
        "TimeoutException",
    ):
        setattr(root, name, type(name, (Exception,), {}))

    class CommandExitException(Exception):

        def __init__(self, *, stdout, stderr, exit_code):
            self.stdout, self.stderr, self.exit_code = stdout, stderr, exit_code

    class FileType(Enum):
        FILE = "file"
        DIR = "dir"

    root.CommandExitException = CommandExitException
    root.FileType = FileType
    FakeSandbox.creates, FakeSandbox.connects, FakeSandbox.fail = [], [], None
    FakeSandbox.killed_ids = []
    root.Sandbox = FakeSandbox
    monkeypatch.setitem(sys.modules, "e2b", root)
    return root


def test_registered_and_conforms(sdk):
    assert BACKENDS["orble"] is OrbleClient
    client = OrbleClient()
    assert isinstance(client, SandboxBackendClient)
    session = client.create(SandboxConfig(backend="orble"))
    assert isinstance(session, SandboxBackendSession)


def test_missing_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "e2b", None)
    with pytest.raises(BackendNotInstalledError, match=r"\[orble\]"):
        OrbleClient()


def test_create_uses_orble_credentials_not_e2b(sdk):
    sb = Sandbox("orble", timeout_secs=120, env_vars={"A": "1"})
    (kwargs,) = FakeSandbox.creates
    assert kwargs == {
        "api_key": "offline-placeholder",
        "domain": "orble.tech",
        "timeout": 120,
        "cpu_count": 2,
        "envs": {"A": "1"},
    }
    assert sb.provider_resource_id == "sbx1"


def test_domain_from_environment(sdk, monkeypatch):
    monkeypatch.setenv("ORBLE_DOMAIN", "staging.orble.tech")
    Sandbox("orble")
    assert FakeSandbox.creates[0]["domain"] == "staging.orble.tech"
    monkeypatch.setenv("ORBLE_DOMAIN", "https://x.test/?k=v")
    with pytest.raises(SandboxConfigurationError):
        Sandbox("orble")


def test_missing_api_key(sdk, monkeypatch):
    monkeypatch.delenv("ORBLE_API_KEY")
    with pytest.raises(SandboxAuthenticationError, match="ORBLE_API_KEY"):
        Sandbox("orble")
    assert FakeSandbox.creates == []


@pytest.mark.parametrize(
    ("cpu", "memory_mb", "vcpus"),
    [
        (None, None, 2),
        (2, 4096, 2),
        (3, None, 4),
        (None, 8192, 4),
        (0.5, 9000, 8),
        (8, 16384, 8),
    ],
)
def test_size_rounds_up(sdk, cpu, memory_mb, vcpus):
    Sandbox("orble", cpu=cpu, memory_mb=memory_mb)
    assert FakeSandbox.creates[0]["cpu_count"] == vcpus


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cpu": 16},
        {"memory_mb": 32768},
        {"image": "python:3.12"},
        {"gpu": "L4"},
        {"disk_mb": 100_000},
        {"allow_internet": False},
        {"network": NetworkPolicy(mode="none")},
    ],
)
def test_unsupported_options_rejected_before_create(sdk, kwargs):
    with pytest.raises(FeatureNotSupportedError):
        Sandbox("orble", **kwargs)
    assert FakeSandbox.creates == []


def test_backend_options_cannot_override_network(sdk):
    with pytest.raises(SandboxConfigurationError):
        Sandbox(
            "orble",
            network=NetworkPolicy(mode="public"),
            backend_options={"allow_internet_access": False},
        )


def test_backend_options_cannot_set_credentials(sdk):
    for key in ("api_key", "domain"):
        with pytest.raises(SandboxConfigurationError, match="ORBLE_"):
            Sandbox("orble", backend_options={key: "x"})
    assert FakeSandbox.creates == []


def test_snapshot_id_is_the_template(sdk):
    Sandbox("orble", snapshot_id="ab12-1")
    assert FakeSandbox.creates[0]["template"] == "ab12-1"
    with pytest.raises(SandboxConfigurationError):
        Sandbox("orble", snapshot_id="ab12-1", template="base")


def test_backend_options_merge_last(sdk):
    Sandbox("orble", backend_options={"cpu_count": 8, "metadata": {"k": "v"}})
    assert FakeSandbox.creates[0]["cpu_count"] == 8
    assert FakeSandbox.creates[0]["metadata"] == {"k": "v"}


def test_commands_and_code(sdk):
    sb = Sandbox("orble")
    raw = sb._session._sandbox
    result = sb.execute_command("echo", ["a b", "$HOME"])
    assert (result.stdout, result.exit_code) == ("out", 0)
    assert raw.runs[-1] == ("echo 'a b' '$HOME'", {"cwd": None, "timeout": 0})
    sb.execute_code("print(1)")
    assert raw.runs[-1][0] == "python3 -c 'print(1)'"
    sb.execute_code("console.log(1)", language="javascript")
    assert raw.runs[-1][0] == "node -c 'console.log(1)'"
    raw.exit_code = 3
    result = sb.execute_command("false")
    assert (result.stdout, result.stderr, result.exit_code) == (
        "out",
        "err",
        3,
    )


def test_workdir(sdk):
    sb = Sandbox("orble", workdir="/home/user/project")
    raw = sb._session._sandbox
    # The directory does not exist yet, so mkdir must not run inside it.
    assert raw.runs[0] == (
        "mkdir -p /home/user/project",
        {"cwd": None, "timeout": 0},
    )
    sb.execute_command("pwd")
    assert raw.runs[-1][1]["cwd"] == "/home/user/project"
    assert sb.session_state().data["workdir"] == "/home/user/project"


def test_files_are_binary_safe(sdk, tmp_path):
    sb = Sandbox("orble")
    sb.write_file("/tmp/x", b"\x00\xff")
    assert sb.read_file("/tmp/x") == b"\x00\xff"
    local = tmp_path / "x"
    sb.download_file("/tmp/x", str(local))
    assert local.read_bytes() == b"\x00\xff"
    sb.upload_file(str(local), "/tmp/y")
    assert sb.read_file("/tmp/y") == b"\x00\xff"
    entries = sb.list_files("/tmp")
    assert [(e.path, e.is_dir, e.size) for e in entries] == [
        ("/tmp/a", False, 2),
        ("/tmp/d", True, 0),
    ]


def test_expose_port_sends_traffic_token(sdk):
    sb = Sandbox("orble")
    port = sb.expose_port(8000)
    assert port.url == "https://8000-sbx1.e2b.orble.tech"
    assert port.headers == {"e2b-traffic-access-token": "traffic-token"}
    assert "traffic-token" not in repr(port)


def test_snapshot(sdk):
    info = Sandbox("orble").snapshot()
    assert (info.snapshot_id, info.backend) == ("ab12-1", "orble")


def test_session_state_and_resume(sdk, monkeypatch):
    state = Sandbox("orble").session_state()
    assert state.data == {"sandbox_id": "sbx1"}
    resumed = Sandbox.resume(state)
    assert FakeSandbox.connects == [
        ("sbx1", {"api_key": "offline-placeholder", "domain": "orble.tech"})
    ]
    assert resumed.provider_resource_id == "sbx1"

    monkeypatch.setenv("ORBLE_DOMAIN", "staging.orble.tech")
    state = Sandbox("orble").session_state()
    assert state.data["domain"] == "staging.orble.tech"
    monkeypatch.delenv("ORBLE_DOMAIN")
    Sandbox.resume(state)
    assert FakeSandbox.connects[-1][1]["domain"] == "staging.orble.tech"


def test_resume_rejects_bad_ids(sdk):
    for data in ({}, {"sandbox_id": "../x"}, {"sandbox_id": 3}):
        with pytest.raises(SandboxConfigurationError):
            SandboxClient("orble").resume(
                SandboxSessionState(backend="orble", data=data)
            )


def test_errors_are_translated(sdk):
    FakeSandbox.fail = sdk.NotFoundException("sandbox x not found")
    with pytest.raises(SandboxNotFoundError):
        SandboxClient("orble").resume(
            SandboxSessionState(backend="orble", data={"sandbox_id": "x"})
        )
    FakeSandbox.fail = sdk.AuthenticationException("invalid key")
    with pytest.raises(SandboxAuthenticationError):
        Sandbox("orble")
    FakeSandbox.fail = sdk.RateLimitException("slow down")
    with pytest.raises(SandboxCreationError) as err:
        Sandbox("orble")
    assert err.value.retryable
    FakeSandbox.fail = httpx.ReadTimeout("timed out")
    with pytest.raises(SandboxTimeoutError) as err:
        Sandbox("orble")
    assert err.value.outcome == ErrorOutcome.UNKNOWN
    assert not err.value.retryable


def test_execution_errors(sdk):
    sb = Sandbox("orble")

    def broken(path, format="text"):
        raise RuntimeError("envd went away")

    sb._session._sandbox.files.read = broken
    with pytest.raises(SandboxExecutionError, match="envd went away"):
        sb.read_file("/tmp/x")


def test_destroy_is_idempotent(sdk):
    sb = Sandbox("orble")
    raw = sb._session._sandbox
    sb.destroy()
    sb.destroy()
    assert raw.killed == 1


def test_session_destroy_twice_directly(sdk):
    session = OrbleClient().create(SandboxConfig(backend="orble"))
    session.destroy()
    session.destroy()
    assert session._sandbox.killed == 1


def test_workdir_failure_destroys_the_sandbox(sdk):
    def fail_mkdir(self, cmd, **kwargs):
        raise sys.modules["e2b"].CommandExitException(
            stdout="", stderr="mkdir: permission denied", exit_code=1
        )

    original = FakeSandbox._run
    FakeSandbox._run = fail_mkdir
    try:
        with pytest.raises(SandboxCreationError, match="permission denied"):
            Sandbox("orble", workdir="/root/x")
    finally:
        FakeSandbox._run = original
    assert FakeSandbox.killed_ids == ["sbx1"]


def test_presets_use_the_installed_agents(sdk):
    sb = Sandbox("orble", preset="claude-code")
    raw = sb._session._sandbox
    assert "image" not in FakeSandbox.creates[0]
    assert [cmd for cmd, _ in raw.runs] == ["claude --version"]
