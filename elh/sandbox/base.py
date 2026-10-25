"""Sandbox contract (SPEC §5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ExecResult:
    exit_code: int | None  # None when the command timed out
    stdout: str
    stderr: str
    wall_s: float
    timed_out: bool = False


@dataclass
class ResourceUsage:
    cpu_s: float = 0.0
    wall_s: float = 0.0


class Sandbox(Protocol):
    backend: str
    image: str
    workdir: str  # path of the agent's workspace as seen by commands inside the sandbox

    def start(self, workspace: dict[str, str]) -> None: ...

    def exec(self, cmd: str, timeout_s: float) -> ExecResult: ...

    def exec_argv(self, argv: list[str], timeout_s: float) -> ExecResult: ...

    def put_files(self, dest_dir: str, files: dict[str, str]) -> str:
        """Write files under a sandbox directory outside the workspace; return its path."""
        ...

    def snapshot_workspace(self) -> dict[str, str]: ...

    def resource_usage(self) -> ResourceUsage: ...

    def close(self) -> None: ...


def make_sandbox(backend: str, **kwargs) -> Sandbox:
    if backend == "local":
        from elh.sandbox.local import LocalSandbox

        return LocalSandbox(**kwargs)
    if backend == "user":
        from elh.sandbox.user import UserSandbox

        return UserSandbox(**kwargs)
    if backend == "podman":
        from elh.sandbox.podman import PodmanSandbox

        return PodmanSandbox(**kwargs)
    raise KeyError(f"unknown sandbox backend: {backend}")


MAX_SNAPSHOT_FILE_BYTES = 2_000_000
SNAPSHOT_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}
