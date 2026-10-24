"""Scripted agents: fixed tool-call sequences that exercise the pipeline without API spend."""

from __future__ import annotations

from elh.runner.models import ScriptedClient
from elh.tasks.base import TaskInstance

HEREDOC_TAG = "ELH_SCRIPTED_EOF"


def _write_file(path: str, content: str) -> tuple[str, dict]:
    return ("bash", {"command": f"cat > {path} <<'{HEREDOC_TAG}'\n{content}{HEREDOC_TAG}"})


def scripted_client(kind: str, task: TaskInstance, family) -> ScriptedClient:
    look = [("bash", {"command": "ls -R && cat job/core.py"})]
    verify = [("bash", {"command": "python -m unittest discover -s tests -q"})]
    if kind == "noop":
        return ScriptedClient([[("submit", {"summary": "No change."})]])
    if kind == "reference":
        core = family.solution_core(task)
    elif kind == "wrong":
        core = family.wrong_fix_core(task)
    else:
        raise KeyError(f"unknown scripted agent {kind!r}")
    return ScriptedClient([look, [_write_file("job/core.py", core)], verify,
                           [("submit", {"summary": f"scripted {kind} fix"})]])
