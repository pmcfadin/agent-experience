"""LocalSandbox: temp-directory sandbox for development only (SPEC §5).

No isolation guarantee. Rows produced with it are tagged sandbox_backend='local' and excluded
from analysis.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from elh.sandbox.base import (
    MAX_SNAPSHOT_FILE_BYTES,
    SNAPSHOT_SKIP_DIRS,
    ExecResult,
    ResourceUsage,
)


class LocalSandbox:
    backend = "local"

    def __init__(self, memory_mb: int = 4096, image: str = "host-python", **_ignored):
        self.memory_mb = memory_mb
        self.image = image
        self.root: Path | None = None
        self.workdir = ""
        self._cpu_s = 0.0
        self._wall_s = 0.0

    def start(self, workspace: dict[str, str]) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="elh-sbx-"))
        ws = self.root / "workspace"
        ws.mkdir()
        self.workdir = str(ws)
        _write_tree(ws, workspace)
        # Match the podman image, where `python` is Python 3.
        bindir = self.root / "bin"
        bindir.mkdir()
        (bindir / "python").symlink_to(shutil.which("python3"))

    def _env(self) -> dict[str, str]:
        return {
            "PATH": f"{self.root / 'bin'}:/usr/local/bin:/usr/bin:/bin",
            "HOME": self.workdir,
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(self.root / "tmp"),
        }

    def _limits(self) -> None:
        limit = self.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    def exec(self, cmd: str, timeout_s: float) -> ExecResult:
        return self.exec_argv(["bash", "-c", cmd], timeout_s)

    def exec_argv(self, argv: list[str], timeout_s: float) -> ExecResult:
        assert self.root is not None, "sandbox not started"
        (self.root / "tmp").mkdir(exist_ok=True)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            argv, cwd=self.workdir, env=self._env(), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
            start_new_session=True, preexec_fn=self._limits,
        )
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
        wall = time.perf_counter() - t0
        # Kill anything the command left running in its session.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        self._cpu_s += (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
        self._wall_s += wall
        return ExecResult(None if timed_out else proc.returncode, out, err, wall, timed_out)

    def put_files(self, dest_dir: str, files: dict[str, str]) -> str:
        assert self.root is not None
        dest = self.root / dest_dir.strip("/")
        dest.mkdir(parents=True, exist_ok=True)
        _write_tree(dest, files)
        return str(dest)

    def snapshot_workspace(self) -> dict[str, str]:
        return _read_tree(Path(self.workdir))

    def resource_usage(self) -> ResourceUsage:
        return ResourceUsage(cpu_s=self._cpu_s, wall_s=self._wall_s)

    def close(self) -> None:
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None


def _write_tree(base: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _read_tree(base: Path) -> dict[str, str]:
    snap: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if not path.is_file() or any(p in SNAPSHOT_SKIP_DIRS for p in path.relative_to(base).parts):
            continue
        # Large or binary files (generated load data, caches) are not part of the solution and
        # are left out; the judge grades a fresh copy built from this snapshot.
        if path.stat().st_size > MAX_SNAPSHOT_FILE_BYTES:
            continue
        try:
            snap[str(path.relative_to(base))] = path.read_text()
        except UnicodeDecodeError:
            continue
    return snap
