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
    if hasattr(family, "solution_files"):  # families that edit more than job/core.py (house_lib)
        if kind not in ("reference", "wrong"):
            raise KeyError(f"unknown scripted agent {kind!r}")
        files = family.solution_files(task) if kind == "reference" else family.wrong_files(task)
        writes = [[_write_file(path, src)] for path, src in sorted(files.items())]
        return ScriptedClient([[("bash", {"command": "ls -R"})], *writes, verify,
                               [("submit", {"summary": f"scripted {kind} solution"})]])
    if kind == "reference":
        core = family.solution_core(task)
    elif kind == "wrong":
        core = family.wrong_fix_core(task)
    else:
        raise KeyError(f"unknown scripted agent {kind!r}")
    return ScriptedClient([look, [_write_file("job/core.py", core)], verify,
                           [("submit", {"summary": f"scripted {kind} fix"})]])


# Group D stand-ins for scripted batches: a fixed induction output and a fixed reflection, so a
# scripted run exercises induction, versioning, the dev gate and snapshots without a model.
SCRIPTED_WORKFLOWS = """\
## Measure the slow job before changing it
This workflow is used to see how slow the job is on a large input before any edit.
I generate an input of production size.
python scripts/make_load.py {large_n} /tmp/big.json
Then I time the job on it.
time timeout 100 python run.py /tmp/big.json /tmp/out_before.json

## Verify the fix
This workflow is used to check behaviour is unchanged after a change.
I run the job again and compare outputs, then run the public tests.
timeout 100 python run.py /tmp/big.json /tmp/out_after.json && cmp /tmp/out_before.json /tmp/out_after.json
python -m unittest discover -s tests -q
"""

SCRIPTED_PROCEDURE = """```
1. Reproduce the slowness on a large generated input and time it.
2. Read the slow function and fix its complexity without changing output.
3. Compare outputs before and after, run the public tests, then submit.
```"""


class ScriptedInductionClient:
    """`workflows.InductionClient` that always returns SCRIPTED_WORKFLOWS with zero usage."""

    model_id = "scripted"

    def complete(self, prompt: str, *, max_tokens: int, temperature: float):
        from elh.learner.workflows import Completion

        return Completion(SCRIPTED_WORKFLOWS, {"input_tokens": 0, "output_tokens": 0}, self.model_id, None)


def scripted_reflection(prompt) -> str:
    """GEPA `LanguageModel` stand-in: always proposes SCRIPTED_PROCEDURE."""
    return SCRIPTED_PROCEDURE
