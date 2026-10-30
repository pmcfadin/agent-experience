"""Generator validation (SPEC §4.4): every task variant must behave as designed before use."""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from elh.judge import grade
from elh.sandbox.base import Sandbox
from elh.tasks.base import TaskFamily


@dataclass
class VariantResult:
    task_id: str
    variant: str
    ok: bool
    reason: str
    failed_checks: list[str]
    quality: dict


def _check(family: TaskFamily, cause: str, theme: str, seed: int,
           make_sandbox: Callable[[], Sandbox]) -> list[VariantResult]:
    task = family.generate(cause, theme, seed)
    again = family.generate(cause, theme, seed)
    results = []
    if task.content_hash != again.content_hash:
        results.append(VariantResult(task.task_id, "determinism", False, "regeneration changed content_hash", [], {}))
    for v in family.validation_variants(task):
        verdict = grade(task, v.workspace, make_sandbox)
        failed = verdict.failed_checks()
        if v.expect_passed:
            ok = verdict.passed
            reason = "passed" if ok else f"expected pass; failed {failed} {verdict.error or ''}".strip()
        else:
            # Must fail for the designed reason. Untouched buggy code must be correct and fail on
            # performance alone; wrong fixes and shortcuts must at least fail correctness.
            if v.exact:
                ok = (not verdict.passed) and failed == [v.expect_failure]
            elif v.expect_failure == "performance":
                ok = (not verdict.passed) and failed == ["performance"]
            else:
                ok = (not verdict.passed) and v.expect_failure in failed
            reason = f"failed {failed}" if ok else f"expected {v.expect_failure} to fail; got {failed}"
        results.append(VariantResult(task.task_id, v.name, ok, reason, failed, verdict.detail.get("quality", {})))
    return results


def validate_family(family: TaskFamily, make_sandbox: Callable[[], Sandbox], seeds: list[int],
                    causes: list[str] | None = None, themes: list[str] | None = None,
                    workers: int = 6) -> list[VariantResult]:
    combos = list(itertools.product(causes or family.causes, themes or family.themes, seeds))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches = pool.map(lambda c: _check(family, *c, make_sandbox), combos)
    return [r for batch in batches for r in batch]
