"""Experiment registration file (SPEC §3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from elh.learner.utility import UtilityConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelPin(_Strict):
    """What makes a self-hosted model 'frozen'. Recorded with every run; required before any
    non-dev split is run against an openai_compat model."""

    weights_sha256: str = ""
    weights_file: str = ""
    quantization: str = ""
    server: str = ""          # e.g. "llama.cpp b6000 --ctx-size 65536 --parallel 4"
    context_tokens: int = 0

    @property
    def complete(self) -> bool:
        return bool(self.weights_sha256 and self.quantization and self.server and self.context_tokens)

    def props_mismatches(self, props: dict) -> list[str]:
        """Compare a llama.cpp /props document with the pin: weights file, build, per-request
        context. Any difference means the backend is not the registered one."""
        out = []
        path = props.get("model_path") or ""
        if self.weights_file and path.rsplit("/", 1)[-1] != self.weights_file:
            out.append(f"model_path {path!r} != pinned {self.weights_file!r}")
        build = props.get("build_info") or ""
        if not build or build not in self.server:
            out.append(f"build_info {build!r} is not the pinned server {self.server!r}")
        n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        if n_ctx != self.context_tokens:
            out.append(f"per-request n_ctx {n_ctx} != pinned context_tokens {self.context_tokens}")
        return out


class ModelConfig(_Strict):
    provider: Literal["anthropic", "openai_compat"] = "anthropic"
    id: str
    max_tokens_per_turn: int = 16000
    # anthropic only
    effort: str = "medium"
    # openai_compat only. Names of environment variables, never the values themselves.
    base_url_env: str = "ELH_MODEL_BASE_URL"
    api_key_env: str = "ELH_MODEL_API_KEY"
    temperature: float | None = None
    top_p: float | None = None
    # URL of the serving backend's llama.cpp /props (a read-only relay; not a secret), or the name
    # of an env var holding it. Fetched, checked against the pin and stored with every episode;
    # required outside the dev split.
    props_url: str = ""
    props_url_env: str | None = None
    pin: ModelPin = ModelPin()


class MemoryLLMConfig(_Strict):
    """The model Hindsight uses for retain/reflect (SPEC §10). Its spend is charged as `learning`."""

    provider: Literal["anthropic", "openai_compat"] = "openai_compat"
    id: str = "memory"
    base_url_env: str = "ELH_MODEL_BASE_URL"
    api_key_env: str = "ELH_MODEL_API_KEY"
    props_url: str = ""
    props_url_env: str | None = None
    pin: ModelPin = ModelPin()


class RulesConfig(_Strict):
    """Learning on the Job (arXiv 2607.22157): after each train episode the agent writes one rule
    (elh/learner/rules.py)."""

    enabled: bool = False
    # Rejections before a rule is accepted with its gaps (paper: MAX_RULE_REJECTIONS = 2).
    max_retries: int = Field(2, ge=0, le=5)
    # None: the agent's own registered sampling (the paper reflects with the agent's llm_args).
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, gt=0)


class WorkflowConfig(BaseModel):  # arm D (AWM source and the dev gate); elh/learner/workflows.py
    """Arm D's registered parameters. `induce_every` is AWM's "every n tasks" (AWM itself: 1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    induce_every: int = Field(5, ge=1)            # M: induce after every M train episodes
    repeats: int = Field(2, ge=1)                 # r: dev repeats per task for the gate (temp 0.7)
    margin: float = Field(0.0, ge=0.0)            # paired mean reward must improve by MORE than this
    max_library_tokens: int = Field(1500, ge=1)   # approx (chars/4); later workflows are dropped
    rollback_margin: float = Field(0.0, ge=0.0)   # retention pass-rate/reward drop that triggers rollback
    max_output_tokens: int = Field(2048, ge=1)    # AWM WebArena max_tokens
    temperature: float = 0.0                      # AWM Mind2Web induction setting
    samples_per_cell: int = Field(1, ge=1)        # AWM --num_samples default
    max_prompt_tokens: int = Field(40_000, ge=1000)  # approx; observations shrink to fit


class GepaConfig(BaseModel):  # arm D's GEPA source; elh/learner/gepa_opt.py
    """One GEPA invocation. Defaults are the report's minimal pilot (papers/gepa.md §3.6):
    |validation| = 10, 15 iterations at ~30% acceptance -> 10 + 90 + 45 ~ 145 rollouts, so
    `max_metric_calls=150`; plus 2 x 20 rollouts for the paired promotion check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # GEPA (paper / gepa.optimize defaults unless noted)
    max_metric_calls: int = 150                 # budget in rollouts (dev episodes), GEPA's own count
    reflection_minibatch_size: int = 3          # paper App. E.4: "a minibatch size of 3"
    candidate_selection_strategy: str = "pareto"  # Table 3: Pareto +12.44 vs SelectBest +6.05
    frontier_type: str = "instance"
    acceptance_criterion: str = "strict_improvement"
    use_merge: bool = False                     # merge hurt Qwen3-8B (54.85 -> 52.40); no-op for 1 component
    skip_perfect_score: bool = False            # deviation: 1.0 is unreachable with a cost term
    seed: int = 0

    # Score (report §3.4): integrity broken = 0; else partial credit per check; passes earn
    # 1 - cost_weight * normalised_cost, so every pass outranks every failure.
    cost_ref: float = 0.05                      # cost at which normalised cost saturates (same unit
                                                # as run_dev_episode's cost; preregister ~2x median)
    cost_weight: float = Field(0.4, ge=0.0, le=0.5)
    credit_public_tests_failed: float = 0.05
    credit_correctness_failed: float = 0.10
    credit_performance_only: float = 0.30       # + perf_speed_bonus * clip(budget_s / agent_s)
    perf_speed_bonus: float = 0.20

    # Reflection
    reflection_max_tokens: int = 4096
    max_procedure_words: int = 400              # stated in the reflection prompt
    max_ticket_chars: int = 1500
    max_actions_chars: int = 6000

    # Promotion (outside GEPA: its best-of-pool is selection-biased)
    promotion_replicates: int = 1
    promotion_min_delta: float = 0.0            # promote iff mean paired delta > this, no integrity regression

    infra_retries: int = 1                      # an InfraError is retried, then raised (never scored 0)

    @model_validator(mode="after")
    def _check(self) -> "GepaConfig":
        if self.candidate_selection_strategy not in ("pareto", "current_best", "epsilon_greedy", "top_k_pareto"):
            raise ValueError(f"unknown candidate_selection_strategy {self.candidate_selection_strategy!r}")
        if self.acceptance_criterion not in ("strict_improvement", "improvement_or_equal"):
            raise ValueError(f"unknown acceptance_criterion {self.acceptance_criterion!r}")
        if self.max_metric_calls < 1 or self.reflection_minibatch_size < 1 or self.cost_ref <= 0:
            raise ValueError("max_metric_calls, reflection_minibatch_size and cost_ref must be positive")
        return self


class MemoryServiceConfig(_Strict):
    kind: Literal["hindsight"] = "hindsight"
    base_url: str = ""                 # loopback URL of the dedicated server; not a secret
    base_url_env: str = "ELH_HINDSIGHT_URL"
    version: str = ""
    llm: MemoryLLMConfig = MemoryLLMConfig()
    embedder: str = ""                 # repo@revision; files hashed in services/hindsight/models.lock.json
    reranker: str = ""
    # What one retained episode looks like (elh/memory/hindsight.py render_episode/render_context).
    # lesson-v2: outcome, judge timings, the agent's summary and source diff; ticket as context.
    # lesson-v1: same, but a failed check was described by its pass wording (inverted lesson); do not use.
    # ticket-v0: the first format (ticket first, fixed context), kept for comparison experiments.
    # lesson-v3: short outcome context (no ticket text: Hindsight reranks "{context}: {text}", so a
    #   ticket context made recall rank by ticket similarity), outcome tags, a distilled rule first
    #   when there is one (Learning on the Job), diff optional (Moves to Memory: raw detail transfers badly).
    record_format: Literal["lesson-v3", "lesson-v2", "lesson-v1", "ticket-v0"] = "lesson-v2"
    include_diff: bool = True              # lesson-v3 only
    ticket_metadata: bool = True           # lesson-v3 only: ticket kept as item metadata
    recall_max_tokens: int = 2048
    recall_types: list[str] | None = None  # Hindsight fact types for the ticket query (None: all)
    lesson_query: str | None = None        # fixed, preregistered second query filling lesson_k slots
    lesson_k: int = 2
    lesson_types: list[str] | None = None
    retain_mission: str | None = None
    observations_mission: str | None = None
    retain_extraction_mode: str | None = None
    # Learning on the Job's read path: the agent may call search_memory(query) during the episode.
    agent_search: bool = False

    def hindsight_options(self) -> dict:
        """HindsightMemory constructor kwargs for this registration."""
        return {k: getattr(self, k) for k in (
            "record_format", "include_diff", "ticket_metadata", "recall_max_tokens", "recall_types",
            "lesson_query", "lesson_k", "lesson_types", "retain_mission", "observations_mission",
            "retain_extraction_mode")}


class Price(_Strict):
    """USD per million tokens. For a self-hosted model these are preregistered shadow prices
    (derived from measured GPU time and a fixed hardware rate), not invoices."""

    input: float
    output: float
    cache_write_5m: float = 0.0
    cache_read: float = 0.0


class Budgets(_Strict):
    max_turns: int = 40
    max_total_tokens: int = 400_000
    wall_clock_seconds: float = 1200
    bash_timeout_seconds: float = 120
    tool_output_chars: int = 12_000


class SandboxConfig(_Strict):
    backend: str = "podman"
    image: str = "host-python"
    cpus: float = 2
    memory_mb: int = 2048
    network: str = "none"


class Schedule(_Strict):
    train_episodes: int = 0
    checkpoints: list[int] = Field(default_factory=list)
    checkpoint_tasks: int = 0
    sealed_tasks: int = 0
    seed: int = 0


class Amortization(_Strict):
    horizon_tasks: int = 1000
    sensitivity: list[int] = Field(default_factory=list)


class ExperimentConfig(_Strict):
    name: str
    version: int
    model: ModelConfig
    prices_usd_per_mtok: dict[str, Price]
    budgets: Budgets = Budgets()
    sandbox: SandboxConfig = SandboxConfig()
    sandbox_cpu_usd_per_hour: float = 0.0
    arms: list[str]
    histories: int = 1
    families: list[str]
    schedule: Schedule = Schedule()
    amortization: Amortization = Amortization()
    memory_service: MemoryServiceConfig | None = None
    utility: UtilityConfig | None = None   # arm C (MemRL-style utility ranking); required when C runs
    rules: RulesConfig | None = None       # Learning on the Job: one reflected rule per train episode
    # Arm D = arm C + a procedure library shown first in the memory block (SPEC §10). The library
    # comes from AWM induction over verified train successes (`workflows`) or from GEPA over dev
    # episodes (`gepa`); either way a candidate is promoted only by the paired dev gate in `workflows`.
    workflows: WorkflowConfig | None = None
    gepa: GepaConfig | None = None
    procedure_source: Literal["awm", "gepa"] = "awm"

    @model_validator(mode="after")
    def _arm_d(self) -> "ExperimentConfig":
        if "D" in self.arms:
            missing = [k for k in ("utility", "workflows") if getattr(self, k) is None]
            if self.procedure_source == "gepa" and self.gepa is None:
                missing.append("gepa")
            if missing:
                raise ValueError(f"arm D needs registration sections {missing} (utility: as arm C; workflows: "
                                 "induction cadence and the dev promotion gate; gepa: when procedure_source is gepa)")
        return self

    @property
    def config_hash(self) -> str:
        blob = json.dumps(_hashable(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def price(self, model_id: str) -> Price:
        try:
            return self.prices_usd_per_mtok[model_id]
        except KeyError:
            raise KeyError(f"no pinned price for model {model_id!r}; add it to the registration file") from None


# Fields added after registrations were in use. Left out of the hash while at their default, so a
# registration that doesn't use them keeps its hash (and its run id and banks).
_LATE_FIELDS = {
    "": {"utility": None, "rules": None, "workflows": None, "gepa": None, "procedure_source": "awm"},
    "memory_service": {"include_diff": True, "ticket_metadata": True, "recall_max_tokens": 2048,
                       "recall_types": None, "lesson_query": None, "lesson_k": 2, "lesson_types": None,
                       "retain_mission": None, "observations_mission": None, "retain_extraction_mode": None,
                       "agent_search": False},
}


def _hashable(cfg: "ExperimentConfig") -> dict:
    d = cfg.model_dump(mode="json")
    for section, defaults in _LATE_FIELDS.items():
        target = d if not section else d.get(section)
        if isinstance(target, dict):
            for k, v in defaults.items():
                if k in target and target[k] == v:
                    del target[k]
    return d


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        return ExperimentConfig.model_validate(yaml.safe_load(f))
