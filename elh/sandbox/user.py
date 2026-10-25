"""UserSandbox: one throwaway Unix user per sandbox, in the dev container (SPEC §5).

The dev container can't run containers (no nested user namespaces), so isolation comes from
the host kernel's ordinary mechanisms instead:

- identity: every sandbox runs as a pool user (`elhsbx0`, `elhsbx1`, ...) that owns only its
  sandbox directory; `/home/dev` is closed to other users, so the harness, its oracle sources and
  credentials are unreadable;
- network: an nftables rule rejects every socket the pool users open, loopback included;
- resources: each command runs in a transient systemd scope with CPU, memory (no swap) and task
  limits from the registry;
- lifetime: after every command all of the user's processes are killed; a sandbox refuses to
  start if its user still has processes after a kill (a leftover), which is an infra error.

Weaker than a container: the pool users share the kernel, `/tmp` and the process table (they can
see other processes' command lines, not their environment or files). One-time root setup is
`elh sandbox-setup`.
"""

from __future__ import annotations

import fcntl
import io
import os
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

from elh.runner.models import InfraError
from elh.sandbox.base import MAX_SNAPSHOT_FILE_BYTES, SNAPSHOT_SKIP_DIRS, ExecResult, ResourceUsage

BASE = Path("/var/lib/elh-sandbox")
LOCK_DIR = BASE / "locks"
USER_PREFIX = "elhsbx"
NFT_TABLE = "elh_sandbox"
TASKS_MAX = 512


def _sudo(*argv: str, input: bytes | None = None, check: bool = True, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["sudo", "-n", *argv], input=input, capture_output=True, check=check, timeout=timeout,
                          cwd="/")


def pool_users() -> list[str]:
    users = []
    with open("/etc/passwd") as fh:
        for line in fh:
            name = line.split(":", 1)[0]
            if name.startswith(USER_PREFIX) and name[len(USER_PREFIX):].isdigit():
                users.append(name)
    return sorted(users, key=lambda u: int(u[len(USER_PREFIX):]))


def setup(n_users: int) -> list[str]:
    """Idempotent root setup: pool users, the nftables rules and the sandbox base directory.
    The nftables table is not persistent; rerun after a reboot (sandboxes refuse to start without it)."""
    users = [f"{USER_PREFIX}{i}" for i in range(n_users)]
    existing = set(pool_users())
    for u in users:
        if u not in existing:
            _sudo("useradd", "--system", "--user-group", "--no-create-home", "--home-dir", "/nonexistent",
                  "--shell", "/usr/sbin/nologin", u)
    rules = "\n".join(f"    meta skuid {u} counter reject" for u in pool_users())
    ruleset = (f"table inet {NFT_TABLE}\ndelete table inet {NFT_TABLE}\n"
               f"table inet {NFT_TABLE} {{\n  chain out {{\n    type filter hook output priority 0; policy accept;\n"
               f"{rules}\n  }}\n}}\n")
    _sudo("nft", "-f", "-", input=ruleset.encode())
    _sudo("install", "-d", "-o", "root", "-g", "root", "-m", "0711", str(BASE))
    _sudo("install", "-d", "-o", str(os.getuid()), "-g", str(os.getgid()), "-m", "0700", str(LOCK_DIR))
    return pool_users()


def _network_blocked(user: str) -> bool:
    out = _sudo("nft", "list", "chain", "inet", NFT_TABLE, "out", check=False)
    if out.returncode != 0:
        return False
    uid = subprocess.run(["id", "-u", user], capture_output=True, text=True).stdout.strip()
    text = out.stdout.decode()
    return f"skuid {user} " in text or f"skuid {uid} " in text


def _tar_bytes(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel, content in sorted(files.items()):
            data = content.encode()
            info = tarfile.TarInfo(rel)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class UserSandbox:
    backend = "user"

    def __init__(self, memory_mb: int = 2048, cpus: float = 2, image: str = "host-python", **_ignored):
        self.memory_mb = memory_mb
        self.cpus = cpus
        self.image = image
        self.user: str | None = None
        self.root: Path | None = None
        self.workdir = ""
        self._lock = None
        self._cpu_s = 0.0
        self._wall_s = 0.0

    # -- lifecycle ---------------------------------------------------------------------------
    def _acquire_user(self) -> None:
        users = pool_users()
        if not users:
            raise InfraError("no sandbox pool users; run `elh sandbox-setup`")
        for u in users:
            fh = open(LOCK_DIR / f"{u}.lock", "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            self.user, self._lock = u, fh
            return
        raise InfraError(f"all {len(users)} sandbox pool users are busy")

    def _kill_all(self) -> bool:
        """SIGKILL every process the pool user owns; True once none remain."""
        assert self.user
        for _ in range(20):
            _sudo("pkill", "-KILL", "-u", self.user, check=False)
            if _sudo("pgrep", "-u", self.user, check=False).returncode == 1:
                return True
            time.sleep(0.05)
        return False

    def start(self, workspace: dict[str, str]) -> None:
        self._acquire_user()
        assert self.user
        if not _network_blocked(self.user):
            self._release()
            raise InfraError(f"network block for {self.user} is missing; rerun `elh sandbox-setup`")
        if not self._kill_all():
            self._release()
            raise InfraError(f"leftover processes for {self.user} survive SIGKILL")
        self._scrub_shared_tmp()
        self.root = BASE / uuid.uuid4().hex
        _sudo("install", "-d", "-o", self.user, "-g", self.user, "-m", "0700", str(self.root))
        for sub in ("workspace", "tmp", "bin"):
            self._as_user("mkdir", "-m", "0700", str(self.root / sub))
        self._as_user("ln", "-s", "/usr/bin/python3", str(self.root / "bin" / "python"))
        self.workdir = str(self.root / "workspace")
        self._extract(self.workdir, workspace)

    def _release(self) -> None:
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
        self._lock, self.user = None, None

    def _scrub_shared_tmp(self) -> None:
        assert self.user
        _sudo("find", "/tmp", "/var/tmp", "/dev/shm", "-mindepth", "1", "-user", self.user, "-delete", check=False)

    def close(self) -> None:
        if self.user is None:
            return
        try:
            self._kill_all()
            if self.root is not None:
                _sudo("rm", "-rf", "--one-file-system", str(self.root), check=False)
                self.root = None
            self._scrub_shared_tmp()
        finally:
            self._release()

    # -- files -------------------------------------------------------------------------------
    def _as_user(self, *argv: str, input: bytes | None = None) -> subprocess.CompletedProcess:
        assert self.user
        return _sudo("-u", self.user, "--", *argv, input=input)

    def _extract(self, dest: str, files: dict[str, str]) -> None:
        self._as_user("tar", "-x", "--no-same-owner", "-C", dest, "-f", "-", input=_tar_bytes(files))

    def put_files(self, dest_dir: str, files: dict[str, str]) -> str:
        assert self.root is not None
        dest = str(self.root / dest_dir.strip("/"))
        self._as_user("mkdir", "-p", "-m", "0700", dest)
        self._extract(dest, files)
        return dest

    def snapshot_workspace(self) -> dict[str, str]:
        # Large files (generated load data) are left out in the guest, so they never cross over.
        listing = _sudo("-u", self.user, "--", "find", self.workdir, "-type", "f",
                        "-size", f"-{MAX_SNAPSHOT_FILE_BYTES + 1}c", "-printf", "%P\\0").stdout.decode()
        rels = [r for r in listing.split("\0") if r and not any(p in SNAPSHOT_SKIP_DIRS for p in r.split("/"))]
        if not rels:
            return {}
        data = _sudo("-u", self.user, "--", "tar", "-c", "-C", self.workdir, "--null", "-T", "-", "-f", "-",
                     input="\0".join(rels).encode()).stdout
        snap: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                try:
                    snap[m.name.removeprefix("./")] = tar.extractfile(m).read().decode()
                except UnicodeDecodeError:
                    continue
        return dict(sorted(snap.items()))

    # -- commands ----------------------------------------------------------------------------
    def exec(self, cmd: str, timeout_s: float) -> ExecResult:
        return self.exec_argv(["bash", "-c", cmd], timeout_s)

    def exec_argv(self, argv: list[str], timeout_s: float) -> ExecResult:
        assert self.root is not None and self.user, "sandbox not started"
        env = [f"PATH={self.root / 'bin'}:/usr/local/bin:/usr/bin:/bin", f"HOME={self.workdir}",
               "LANG=C.UTF-8", "PYTHONDONTWRITEBYTECODE=1", f"TMPDIR={self.root / 'tmp'}"]
        full = ["sudo", "-n", "systemd-run", "--scope", "--quiet", "--collect",
                f"--uid={self.user}", f"--gid={self.user}",
                "-p", f"MemoryMax={self.memory_mb}M", "-p", "MemorySwapMax=0",
                "-p", f"CPUQuota={int(self.cpus * 100)}%", "-p", f"TasksMax={TASKS_MAX}",
                "--", "/usr/bin/env", "-i", *env,
                "/bin/bash", "-c", 'cd "$0" && exec "$@"', self.workdir, *argv]
        before = os.times()
        t0 = time.perf_counter()
        proc = subprocess.Popen(full, cwd="/", stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, errors="replace")
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_all()
            out, err = proc.communicate()
        wall = time.perf_counter() - t0
        # Each command is a fresh process; nothing it started may outlive it.
        self._kill_all()
        after = os.times()
        self._cpu_s += (after.children_user - before.children_user) + (after.children_system - before.children_system)
        self._wall_s += wall
        return ExecResult(None if timed_out else proc.returncode, out, err, wall, timed_out)

    def resource_usage(self) -> ResourceUsage:
        return ResourceUsage(cpu_s=self._cpu_s, wall_s=self._wall_s)
