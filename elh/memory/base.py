"""Memory contract (SPEC §10). Phase 0 only needs NullMemory (arm A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from elh.tasks.base import TaskInstance


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    text: str
    score: float = 0.0


class Memory(Protocol):
    def recall(self, task: TaskInstance, k: int) -> list[MemoryItem]: ...

    def retain(self, episode: dict) -> list[str]: ...


class NullMemory:
    """Arm A: stateless. Recalls nothing, retains nothing."""

    def recall(self, task: TaskInstance, k: int) -> list[MemoryItem]:
        return []

    def retain(self, episode: dict) -> list[str]:
        return []


def render_memory_block(items: list[MemoryItem]) -> str:
    return "\n\n".join(f"[{it.memory_id}]\n{it.text.strip()}" for it in items)
