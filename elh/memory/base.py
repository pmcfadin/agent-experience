"""Memory contract (SPEC §10). Phase 0 only needs NullMemory (group A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from elh.tasks.base import TaskInstance


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    text: str
    score: float = 0.0
    # Source episode (Hindsight document_id == our episode_id); survives snapshot restore, so
    # learner state keyed on it (group C utilities) stays valid across snapshots.
    document_id: str | None = None
    # Per-stage retrieval scores as the memory service reported them, e.g. {"rerank": 0.78}.
    scores: dict = field(default_factory=dict, compare=False)


class Memory(Protocol):
    def recall(self, task: TaskInstance, k: int) -> list[MemoryItem]: ...

    def retain(self, episode: dict) -> list[str]: ...


class NullMemory:
    """Group A: stateless. Recalls nothing, retains nothing."""

    def recall(self, task: TaskInstance, k: int) -> list[MemoryItem]:
        return []

    def retain(self, episode: dict) -> list[str]:
        return []


class StaticMemory:
    """Static-prompt control groups (SPEC §6.2, H6): a fixed block in B's first-message slot.

    `recall` returns every item, in order, whatever `k` (P-all is deliberately not capped);
    `search` returns nothing, so the memory-group instructions and tool are kept but carry no
    retrieval. With no items it is A0: memory-group instructions with an empty block."""

    readonly = True

    def __init__(self, items: list[MemoryItem], snapshot_id: str):
        self.items = list(items)
        self.snapshot_id = snapshot_id  # provenance label ("static:..."), not a Hindsight snapshot

    def recall(self, task: TaskInstance, k: int) -> list[MemoryItem]:
        return list(self.items)

    def search(self, query: str, k: int) -> list[MemoryItem]:
        return []

    def retain(self, episode: dict) -> list[str]:
        raise RuntimeError("a static-prompt group never retains")

    def fingerprint(self) -> str:
        return render_memory_block(self.items)

    def wait_idle(self) -> None:  # no background work; the controller's read-only guard calls it
        return None


class OracleMemory(StaticMemory):
    """P-oracle (house_lib, design §6): a pool of rules, each tagged with the lessons of the training
    task that wrote it; `recall` shows, in pool order, only the rules whose lessons overlap the
    task's own (`signature["lessons"]`): retrieval with the answer key. `fingerprint` is the whole
    pool, so it is the same for every task."""

    def __init__(self, items: list[MemoryItem], lessons: dict[str, frozenset[str]], snapshot_id: str):
        super().__init__(items, snapshot_id)
        self.lessons = lessons  # memory_id -> lessons of the training task that wrote the rule

    def recall(self, task: TaskInstance, k: int) -> list[MemoryItem]:
        want = set(task.signature["lessons"])
        return [it for it in self.items if self.lessons[it.memory_id] & want]


def render_memory_block(items: list[MemoryItem]) -> str:
    return "\n\n".join(f"[{it.memory_id}]\n{it.text.strip()}" for it in items)
