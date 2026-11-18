"""Orble backend (the optional ``orble`` extra).

Orble (https://orble.tech) serves E2B's sandbox API, so this adapter drives
Orble with the stock ``e2b`` SDK. It is separate from the E2B adapter: it reads
``ORBLE_API_KEY`` and ``ORBLE_DOMAIN`` and passes them to every SDK call, so
E2B credentials in the same process are left alone.
"""

from __future__ import annotations

import os
import pathlib
import re
import shlex
from contextlib import contextmanager
from typing import Iterator

from bespokelabs.sandbox.exceptions import (
    BackendNotInstalledError,
    ErrorOutcome,
    FeatureNotSupportedError,
    SandboxAuthenticationError,
    SandboxConfigurationError,
    SandboxConnectionError,
    SandboxCreationError,
    SandboxError,
    SandboxExecutionError,
    SandboxNotFoundError,
    SandboxTimeoutError,
)
from bespokelabs.sandbox.types import (
    ExposedPort,
    FileInfo,
    SandboxConfig,
    SandboxResult,
    SnapshotInfo,
)

_DEFAULT_DOMAIN = "orble.tech"
# Orble's box sizes as (vCPUs, MiB of RAM). Every box has a 64 GiB disk.
_SIZES = ((2, 4096), (4, 8192), (8, 16384))
_DISK_MB = 64 * 1024
# Interpreter names for execute_code's ``language``; others run as given.
_INTERPRETERS = {"python": "python3", "javascript": "node", "js": "node"}


@contextmanager
def _operation(op: str) -> Iterator[None]:
    """Translate SDK failures into this library's errors."""
    import httpx
    from e2b import (
        AuthenticationException,
        InvalidArgumentException,
        NotFoundException,
        RateLimitException,
        TimeoutException,
    )

    try:
        yield
    except SandboxError:
        raise
    except Exception as exc:
        retryable = None
        if isinstance(exc, AuthenticationException):
            error_type: type[SandboxError] = SandboxAuthenticationError
        elif isinstance(exc, NotFoundException):
            error_type = SandboxNotFoundError
        elif isinstance(exc, (TimeoutException, httpx.TimeoutException)):
            error_type = SandboxTimeoutError
        elif isinstance(exc, (httpx.TransportError, ConnectionError)):
            error_type = SandboxConnectionError
        elif isinstance(exc, InvalidArgumentException):
            error_type = SandboxConfigurationError
        else:
            error_type = (
                SandboxCreationError
                if op in {"create", "resume"}
                else SandboxExecutionError
            )
            retryable = isinstance(exc, RateLimitException) or None
        # A create or destroy that timed out may still have happened.
        unknown = op in {"create", "destroy"} and error_type in {
            SandboxTimeoutError,
            SandboxConnectionError,
        }
        raise error_type(
            f"Orble {op} failed: {exc}",
            backend="orble",
            op=op,
            context={"provider_error_type": type(exc).__name__},
            outcome=ErrorOutcome.UNKNOWN if unknown else ErrorOutcome.FAILED,
            retryable=False if unknown else retryable,
        ) from exc


def _connection(op: str, domain: str | None = None) -> dict:
    """Return the SDK's api_key and domain arguments for Orble."""
    api_key = os.environ.get("ORBLE_API_KEY")
    if not api_key:
        raise SandboxAuthenticationError(
            "ORBLE_API_KEY environment variable is not set",
            backend="orble",
            op=op,
        )
    domain = domain or os.environ.get("ORBLE_DOMAIN") or _DEFAULT_DOMAIN
    if not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?", domain):
        raise SandboxConfigurationError(
            "Orble domain must be a hostname", backend="orble", op=op
        )
    return {"api_key": api_key, "domain": domain}


def _cpu_count(config: SandboxConfig) -> int:
    """Return the vCPUs of the smallest Orble size that fits the config."""
    for vcpus, memory_mb in _SIZES:
        if config.cpu <= vcpus and config.memory_mb <= memory_mb:
            return vcpus
    raise FeatureNotSupportedError(
        "Orble's largest box has 8 vCPUs and 16384 MB of RAM; "
        f"cpu={config.cpu} memory_mb={config.memory_mb} does not fit",
        backend="orble",
        op="create",
    )


def _check_config(config: SandboxConfig) -> None:
    def reject(message: str) -> None:
        raise FeatureNotSupportedError(message, backend="orble", op="create")

    if config.image:
        reject(
            "Orble boxes boot Orble's own image; pass template= or "
            "snapshot_id= instead of image="
        )
    if config.gpu:
        reject("Orble boxes have no GPUs")
    if config.disk_mb is not None and config.disk_mb > _DISK_MB:
        reject(f"Orble boxes have a {_DISK_MB} MB disk")
    if not config.allow_internet:
        reject(
            "allow_internet=False is not mapped on Orble; this adapter "
            "creates sandboxes with public internet access only"
        )
    clashes = sorted({"api_key", "domain"} & set(config.backend_options))
    if clashes:
        raise SandboxConfigurationError(
            f"backend_options {clashes} are not accepted; set ORBLE_API_KEY "
            "and ORBLE_DOMAIN instead",
            backend="orble",
            op="create",
        )
    if config.template and config.snapshot_id:
        raise SandboxConfigurationError(
            "Pass template= or snapshot_id=, not both",
            backend="orble",
            op="create",
        )


class OrbleClient:
    """Create Orble sandboxes through the e2b SDK."""

    supports_dockerfile = False
    # backend_options merge last; these would bypass the network check.
    network_option_keys = frozenset({"network", "allow_internet_access"})

    def __init__(self) -> None:
        try:
            from e2b import Sandbox as E2BSandbox
        except ImportError:
            raise BackendNotInstalledError(
                "e2b SDK not installed. Run: "
                "pip install 'bespokelabs-sandbox[orble]'"
            ) from None
        self._sandbox_cls = E2BSandbox

    def create(self, config: SandboxConfig) -> OrbleSession:
        _check_config(config)
        connection = _connection("create")
        kwargs: dict = {
            "timeout": config.timeout_secs,
            "cpu_count": _cpu_count(config),
        }
        # Orble creates a sandbox from a snapshot by its name as a template.
        template = config.snapshot_id or config.template
        if template:
            kwargs["template"] = template
        if config.env_vars:
            kwargs["envs"] = config.env_vars
        kwargs.update(config.backend_options)
        with _operation("create"):
            sandbox = self._sandbox_cls.create(**connection, **kwargs)
        session = OrbleSession(
            sandbox=sandbox, domain=connection["domain"], workdir=config.workdir
        )
        if config.workdir:
            try:
                result = session._run(
                    shlex.join(["mkdir", "-p", config.workdir]), cwd=None
                )
                if result.exit_code != 0:
                    raise SandboxCreationError(
                        f"Orble create failed to make workdir: {result.stderr}",
                        backend="orble",
                        op="create",
                    )
            except BaseException:
                session.destroy()
                raise
        return session

    def resume(self, data: dict) -> OrbleSession:
        """Reattach to a sandbox, waking it first if it is suspended."""
        sandbox_id = data.get("sandbox_id")
        if not isinstance(sandbox_id, str) or not re.fullmatch(
            r"[a-zA-Z0-9_-]+", sandbox_id
        ):
            raise SandboxConfigurationError(
                "Invalid Orble sandbox_id", backend="orble", op="resume"
            )
        connection = _connection("resume", data.get("domain"))
        with _operation("resume"):
            sandbox = self._sandbox_cls.connect(sandbox_id, **connection)
        return OrbleSession(
            sandbox=sandbox,
            domain=connection["domain"],
            workdir=data.get("workdir"),
        )


class OrbleSession:
    """One Orble sandbox."""

    def __init__(
        self, *, sandbox: object, domain: str, workdir: str | None = None
    ) -> None:
        self._sandbox = sandbox
        self._domain = domain
        self._workdir = workdir
        self._destroyed = False

    @property
    def provider_resource_id(self) -> str:
        return self._sandbox.sandbox_id

    def execute_code(
        self, code: str, language: str = "python"
    ) -> SandboxResult:
        interpreter = _INTERPRETERS.get(language, language)
        return self.execute_command(interpreter, ["-c", code])

    def execute_command(
        self, command: str, args: list[str] | None = None
    ) -> SandboxResult:
        full_command = command if not args else shlex.join([command, *args])
        return self._run(full_command, cwd=self._workdir)

    def _run(self, command_line: str, *, cwd: str | None) -> SandboxResult:
        from e2b import CommandExitException

        with _operation("execute_command"):
            try:
                # timeout=0: the SDK's 60-second default would cut off long
                # commands; the sandbox's own timeout still applies.
                result = self._sandbox.commands.run(
                    command_line, cwd=cwd, timeout=0
                )
            except CommandExitException as exc:
                result = exc
            return SandboxResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
            )

    def list_files(self, path: str = "/") -> list[FileInfo]:
        from e2b import FileType

        with _operation("list_files"):
            return [
                FileInfo(
                    path=entry.path,
                    is_dir=entry.type == FileType.DIR,
                    size=entry.size,
                )
                for entry in self._sandbox.files.list(path)
            ]

    def read_file(self, path: str) -> bytes:
        with _operation("read_file"):
            return bytes(self._sandbox.files.read(path, format="bytes"))

    def write_file(self, path: str, content: bytes | str) -> None:
        with _operation("write_file"):
            self._sandbox.files.write(path, content)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        self.write_file(remote_path, pathlib.Path(local_path).read_bytes())

    def download_file(self, remote_path: str, local_path: str) -> None:
        pathlib.Path(local_path).write_bytes(self.read_file(remote_path))

    def expose_port(self, port: int) -> ExposedPort:
        """Return the HTTPS URL for *port* and the sandbox's access token.

        Bind the service to ``0.0.0.0``. Orble accepts the token on every
        port, and the headers carry it, so do not log them.
        """
        with _operation("expose_port"):
            host = self._sandbox.get_host(port)
        token = getattr(self._sandbox, "traffic_access_token", None)
        return ExposedPort(
            port=port,
            url=f"https://{host}",
            headers={"e2b-traffic-access-token": token} if token else {},
        )

    def snapshot(self) -> SnapshotInfo:
        """Save the sandbox's memory and disk while it keeps running.

        Pass the returned ``snapshot_id`` as ``snapshot_id=`` to create
        sandboxes from it; they start with the processes that were running.
        Snapshots outlive the sandbox and are billed as storage until
        deleted (``e2b.Sandbox.delete_snapshot`` with Orble's domain).
        """
        with _operation("snapshot"):
            info = self._sandbox.create_snapshot()
        return SnapshotInfo(snapshot_id=info.snapshot_id, backend="orble")

    def session_state(self) -> dict:
        state = {"sandbox_id": self.provider_resource_id}
        if self._domain != _DEFAULT_DOMAIN:
            state["domain"] = self._domain
        if self._workdir:
            state["workdir"] = self._workdir
        return state

    def destroy(self) -> None:
        if self._destroyed:
            return
        with _operation("destroy"):
            # kill() returns False when the sandbox is already gone.
            self._sandbox.kill()
        self._destroyed = True
