"""Cost ledger (SPEC §8): every spend is priced from the pinned registry, never looked up live."""

from __future__ import annotations

from dataclasses import dataclass

from elh.registry import ExperimentConfig

CATEGORIES = ("inference", "tool_compute", "learning", "eval_overhead")


@dataclass
class LedgerEntry:
    category: str
    component: str
    units: float
    unit: str
    usd: float


def usage_entries(cfg: ExperimentConfig, model_id: str, usage: dict, category: str = "inference") -> list[LedgerEntry]:
    """Price one API response's usage block. Token fields follow the Messages API usage object."""
    assert category in CATEGORIES
    p = cfg.price(model_id)
    parts = [
        ("input_tokens", "input", p.input),
        ("output_tokens", "output", p.output),
        ("cache_creation_input_tokens", "cache_write", p.cache_write_5m),
        ("cache_read_input_tokens", "cache_read", p.cache_read),
    ]
    out = []
    for field, component, per_mtok in parts:
        n = usage.get(field) or 0
        if n:
            out.append(LedgerEntry(category, f"{model_id}:{component}", n, "tokens", n * per_mtok / 1_000_000))
    return out


def compute_entry(cfg: ExperimentConfig, cpu_s: float, category: str = "tool_compute") -> LedgerEntry:
    return LedgerEntry(category, "sandbox_cpu", cpu_s, "cpu_seconds", cpu_s / 3600 * cfg.sandbox_cpu_usd_per_hour)
