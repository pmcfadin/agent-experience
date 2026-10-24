"""PodmanSandbox: one rootless container per episode (SPEC §5).

Not yet exercised: the dev container currently cannot create nested user namespaces, so podman
cannot start containers there. The command shapes below follow podman 4.9 and must be verified
before any confirmatory run.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
import time
import uuid

from elh.sandbox.base import (
    MAX_SNAPSHOT_FILE_BYTES,
    SNAPSHOT_SKIP_DIRS,
    ExecResult,
    ResourceUsage,
)

WORKDIR = "/workspace"


class PodmanSandbox:
    backend = "podman"

    def __init__(self, image: str, cpus: float = 2, memory_mb: int = 2048, network: str = "none", **_ignored):
        self.image = image
        self.cpus = cpus
        self.memory_mb = memory_mb
        self.network = network
        self.name = f"elh-{uuid.uuid4().hex[:12]}"
        self.workdir = WORKDIR
        self._started = False
        self._wall_s = 0.0

    def _podman(self, *args: str, input_bytes: bytes | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
        return subprocess.run(["podman", *args], input=input_bytes, capture_output=True, timeout=timeout, check=True)

    def start(self, workspace: dict[str, str]) -> None:
        self._podman(
            "run", "-d", "--rm", "--name", self.name,
            f"--network={self.network}", f"--cpus={self.cpus}", f"--memory={self.memory_mb}m",
            "--pids-limit=512", "--read-only", "--tmpfs", "/tmp:rw,size=1g",
            "--tmpfs", f"{WORKDIR}:rw,size=1g,mode=1777", "--tmpfs", "/oracle:rw,size=64m",
            "--workdir", WORKDIR, "--user", "1000:1000",
            self.image, "sleep", "infinity",
        )
        self._started = True
        self._put_tar(WORKDIR, workspace)

    def _put_tar(self, dest: str, files: dict[str, str]) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for rel, content in files.items():
                data = content.encode()
                info = tarfile.TarInfo(rel)
                info.size = len(data)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
        self._podman("exec", "-i", self.name, "tar", "-x", "-C", dest, input_bytes=buf.getvalue())

    def exec(self, cmd: str, timeout_s: float) -> ExecResult:
        return self.exec_argv(["bash", "-c", cmd], timeout_s)

    def exec_argv(self, argv: list[str], timeout_s: float) -> ExecResult:
        t0 = time.perf_counter()
        # `timeout` inside the container kills the process tree; the outer timeout is a backstop.
        inner = ["timeout", "-s", "KILL", str(int(timeout_s) + 1), *argv]
        try:
            proc = subprocess.run(["podman", "exec", "-w", WORKDIR, self.name, *inner],
                                  capture_output=True, text=True, errors="replace", timeout=timeout_s + 15)
        except subprocess.TimeoutExpired:
            wall = time.perf_counter() - t0
            self._wall_s += wall
            return ExecResult(None, "", "", wall, True)
        wall = time.perf_counter() - t0
        self._wall_s += wall
        timed_out = proc.returncode == 137 and wall >= timeout_s
        return ExecResult(None if timed_out else proc.returncode, proc.stdout, proc.stderr, wall, timed_out)

    def put_files(self, dest_dir: str, files: dict[str, str]) -> str:
        dest = "/" + dest_dir.strip("/")
        self._podman("exec", self.name, "mkdir", "-p", dest)
        self._put_tar(dest, files)
        return dest

    def snapshot_workspace(self) -> dict[str, str]:
        proc = self._podman("exec", self.name, "tar", "-c", "-C", WORKDIR, ".", timeout=300)
        snap: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
            for m in tar.getmembers():
                rel = m.name.removeprefix("./")
                if not m.isfile() or m.size > MAX_SNAPSHOT_FILE_BYTES:
                    continue
                if any(p in SNAPSHOT_SKIP_DIRS for p in rel.split("/")):
                    continue
                try:
                    snap[rel] = tar.extractfile(m).read().decode()
                except UnicodeDecodeError:
                    continue
        return dict(sorted(snap.items()))

    def resource_usage(self) -> ResourceUsage:
        # TODO(phase0-podman): read cpu usage from the container's cgroup (cpu.stat usage_usec).
        return ResourceUsage(cpu_s=0.0, wall_s=self._wall_s)

    def close(self) -> None:
        if self._started:
            subprocess.run(["podman", "rm", "-f", "-t", "0", self.name], capture_output=True)
            self._started = False
