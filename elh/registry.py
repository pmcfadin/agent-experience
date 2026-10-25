"""Experiment registration file (SPEC §3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


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
    record_format: Literal["lesson-v2", "lesson-v1", "ticket-v0"] = "lesson-v2"


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

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def price(self, model_id: str) -> Price:
        try:
            return self.prices_usd_per_mtok[model_id]
        except KeyError:
            raise KeyError(f"no pinned price for model {model_id!r}; add it to the registration file") from None


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        return ExperimentConfig.model_validate(yaml.safe_load(f))
