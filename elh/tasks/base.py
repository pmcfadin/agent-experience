"""Task contracts shared by every task family (SPEC §4)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class OracleBundle:
    files: dict[str, str]
    # Argument list; "{workspace}" and "{oracle}" are substituted by the judge.
    command: list[str]
    timeout_s: float = 600.0


@dataclass(frozen=True)
class TaskInstance:
    task_id: str
    family: str
    cause: str
    theme: str
    seed: int
    difficulty: int
    prompt: str
    workspace: dict[str, str]
    protected: tuple[str, ...]
    oracle: OracleBundle
    signature: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        payload = {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "workspace": self.workspace,
            "protected": list(self.protected),
            "oracle_files": self.oracle.files,
            "oracle_command": self.oracle.command,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class ValidationVariant:
    """A workspace variant used by generator validation (SPEC §4.4)."""

    name: str
    workspace: dict[str, str]
    expect_passed: bool
    # Check that must fail when expect_passed is False ("performance" means it must be the only one).
    expect_failure: str | None = None


class TaskFamily(Protocol):
    name: str
    causes: list[str]
    themes: list[str]

    def generate(self, cause: str, theme: str, seed: int) -> TaskInstance: ...

    def validation_variants(self, task: TaskInstance) -> list[ValidationVariant]: ...


def get_family(name: str) -> TaskFamily:
    if name == "cpu_mem":
        from elh.tasks.families.cpu_mem import CpuMemFamily

        return CpuMemFamily()
    raise KeyError(f"unknown task family: {name}")
