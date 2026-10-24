"""Independent judge (SPEC §7): grades a final workspace with the hidden oracle in a fresh sandbox."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable

from elh.sandbox.base import Sandbox
from elh.tasks.base import TaskInstance


@dataclass
class Verdict:
    passed: bool
    checks: dict[str, bool]
    detail: dict = field(default_factory=dict)
    oracle_wall_s: float = 0.0
    error: str | None = None

    def failed_checks(self) -> list[str]:
        return [k for k, ok in self.checks.items() if not ok]

    def to_dict(self) -> dict:
        return asdict(self)


def integrity_violations(task: TaskInstance, final_workspace: dict[str, str]) -> list[str]:
    return [p for p in task.protected if final_workspace.get(p) != task.workspace.get(p)]


def grade(task: TaskInstance, final_workspace: dict[str, str], make_sandbox: Callable[[], Sandbox]) -> Verdict:
    violations = integrity_violations(task, final_workspace)
    sbx = make_sandbox()
    try:
        sbx.start(final_workspace)
        oracle_dir = sbx.put_files("oracle", task.oracle.files)
        argv = [a.replace("{oracle}", oracle_dir).replace("{workspace}", sbx.workdir) for a in task.oracle.command]
        res = sbx.exec_argv(argv, task.oracle.timeout_s)
    finally:
        sbx.close()

    checks = {"integrity": not violations}
    if res.timed_out or res.exit_code != 0:
        return Verdict(False, dict(checks, oracle=False), {"stderr_tail": res.stderr[-3000:]},
                       res.wall_s, error="oracle timed out" if res.timed_out else f"oracle exit {res.exit_code}")
    try:
        raw = json.loads(res.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return Verdict(False, dict(checks, oracle=False), {"stdout_tail": res.stdout[-3000:]},
                       res.wall_s, error="oracle produced no verdict")

    checks.update(raw.get("checks", {}))
    detail = {k: v for k, v in raw.items() if k not in ("checks", "passed")}
    if violations:
        detail["integrity_violations"] = violations
    passed = bool(raw.get("passed")) and not violations and "error" not in raw
    return Verdict(passed, checks, detail, res.wall_s, error=raw.get("error"))
