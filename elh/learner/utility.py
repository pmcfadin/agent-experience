"""Group C: UtilityLearner, MemRL two-phase retrieval over the memory service's candidates (SPEC §10).

Method: MemRL (arXiv 2601.03192 v2, §4.2-§4.3, Table 8; code github.com/MemTensor/MemRL @ c1b322ca,
`memrl/service/memory_service.py:1254-1506`, `memrl/service/value_driven.py:181-220`). Variant
**C-SPEC**: the candidate pool is the list Hindsight's recall already returned for test_group B, so test_group C
differs from test_group B only in which candidates are injected and in what order.

Unit of utility. MemRL keeps one utility Q per memory (one memory per episode). Hindsight extracts
several memory units from each retained episode and gives them new ids on every snapshot restore,
but keeps `document_id` (= our episode_id) stable. So Q is keyed by **document_id**: all units of
one episode share that episode's Q. Units without a document_id (Hindsight observations, which are
consolidated across documents) have no utility and are dropped from test_group C's pool (logged).

Retrieval (per call to `rank`):

1. Group candidate units by document_id. An episode's similarity to the ticket is the **max** of its
   units' similarity scores (`doc_sim_agg`, default "max"): MemRL compares the query with one
   intent per memory; the best-matching unit is the closest analogue, and a mean would penalise
   episodes from which the service extracted more facts. Units with no similarity score (the
   semantic test_group did not surface them) do not count towards the max; an episode with no scored unit
   is dropped.
2. Phase A (paper Eq. 6): keep episodes with sim > delta (`gate_inclusive=True` gives the code's
   >=), sort by sim, keep the top k1.
3. Phase B (paper Eq. 7, normalisation as in the code, which the paper leaves unspecified):
   sim_z = (sim - sim_mean) / sim_std with **fixed** corpus statistics; q_z = clip((Q - mean_C) /
   pstd_C, -3, 3) over the Phase-A survivors (std 1 when there is one survivor or std <= 1e-9, so
   equal Qs give q_z = 0 and ranking falls back to similarity); score = (1 - lambda) sim_z +
   lambda q_z. Top k2 episodes by score (ties: higher sim, then document_id).
4. Injection: each selected episode contributes its units in the service's order
   (`units_per_doc` caps it; default None = every unit the service returned for it, the closest
   analogue of MemRL injecting the whole experience). Episodes appear in score order.

Update (paper Eq. 4, §4.3): after a verified train episode, every injected episode gets
Q <- Q + alpha (r - Q), with the same reward for all (no per-memory credit, as in the paper). An
episode never seen before starts at q_init. Updates are applied per episode (MemRL's code batches
5-32 episodes; ours is sequential, so per-episode is the on-policy limit of that).

Paper vs code (config knobs; defaults follow the PAPER):
- `epsilon`: code BCB 0.1 / LLB 0.01 epsilon-greedy selection; not in the paper (default 0).
- `q_floor`: code clamps Q at a floor (LLB 0.0) at read and update time; not in the paper (None).
- `q_min`: code drops candidates whose Q < q_min_threshold (BCB/LLB -0.8); not in the paper
  (None). With +/-1 rewards and alpha 0.3 a memory that fails ~3 times in a row falls below -0.8
  and is never retrieved again, so its Q can never recover.
- `gate_inclusive`: code gates sim >= delta; the paper writes sim > delta (default False).
- `q_init`: 0.0 on every benchmark in the paper (Table 8); the code uses 0.5 (LLB) or 1.0/0.0 by
  outcome (HLE). Only the constant is implemented.
- Thresholds: MemRL's delta (0.25-0.62) and similarity constants come from its embedder and
  corpora; our e5-small scores sit around 0.78-0.85, so `sim_mean`, `sim_std` and `delta` must be
  estimated on our corpus (`estimate_similarity_stats`) and frozen before the run. The paper sets
  delta at the top-20% quantile of pairwise similarity; the code's sim_mean / sim_std are the mean
  and std of that same distribution.
- Deviation of C-SPEC from the paper: similarity is ticket-vs-extracted-unit (Hindsight's score),
  not ticket-vs-stored-ticket (MemRL's intent).

State: the Q table lives in SQLite (`learner_utility`), one row per (run_id, history, test_group,
document_id), with an append-only log of updates (`learner_utility_updates`). `state_sha256()`
hashes the canonical JSON of the config and the table; `snapshot()` stores that same JSON as a
content-addressed blob whose digest *is* the state hash, so `load_snapshot(store, sha)` gives a
frozen read-only learner for checkpoint and sealed episodes. A read-only learner ranks with frozen Q
and raises on `observe`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from elh.memory.base import MemoryItem

TABLES = """
CREATE TABLE IF NOT EXISTS learner_utility (
    run_id TEXT NOT NULL, history INTEGER NOT NULL, test_group TEXT NOT NULL, document_id TEXT NOT NULL,
    q REAL NOT NULL, visits INTEGER NOT NULL,
    PRIMARY KEY (run_id, history, test_group, document_id)
);
CREATE TABLE IF NOT EXISTS learner_utility_updates (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, history INTEGER NOT NULL,
    test_group TEXT NOT NULL, episode_id TEXT, document_id TEXT NOT NULL,
    reward REAL NOT NULL, q_before REAL NOT NULL, q_after REAL NOT NULL
);
"""

STATE_FORMAT = "utility-v1"
RewardMode = Literal["spec", "memrl", "spec-signed"]


# -- reward ------------------------------------------------------------------------------------
def normalised_cost(episode_cost: float, budget_cost: float) -> float:
    """Episode cost as a share of the cost of the registered episode budget, clipped to [0, 1].

    Both in the same unit (USD at the pinned shadow prices). Count what the injected memories can
    influence: `inference` + `tool_compute` (`episode_cost_usd`); not `learning` or `eval_overhead`."""
    if not budget_cost > 0 or math.isinf(budget_cost):
        raise ValueError(f"budget_cost must be positive and finite, got {budget_cost!r}")
    if math.isnan(episode_cost):
        raise ValueError("episode_cost is NaN")
    return min(1.0, max(0.0, episode_cost / budget_cost))


def reward(passed: bool, normalised_cost: float, mode: RewardMode = "spec-signed") -> float:
    """Verified reward for the Q update.

    - "spec": r = passed x (1 - cost), as SPEC §10 states it. r is in [0, 1] and a failure gives 0.
    - "memrl": +1 / -1 (MemRL's BCB, ALFWorld and HLE configs; its code takes no scalar reward).
    - "spec-signed" (default): pass -> 1 - cost, fail -> -1.

    Why "spec-signed" is the default: with r = passed x (1 - cost) >= 0 and Q_init = 0, every
    update moves Q towards a non-negative target, so Q can never fall below 0. A memory that keeps
    being injected into failures sits at 0, exactly like one that was never used, and is never
    demoted below it; a memory injected into one cheap success outranks it for good. MemRL demotes
    failure-linked memories (failure reward -1), which is half its mechanism. "spec-signed" keeps
    the SPEC's cost-sensitive success reward (cheaper successes are worth more) and restores the
    paper's negative signal on failure, and stays inside the [-1, 1] range the paper's analysis
    assumes (App. A.1).
    """
    if not 0.0 <= normalised_cost <= 1.0:
        raise ValueError(f"normalised_cost must be in [0, 1], got {normalised_cost!r}")
    passed = bool(passed)
    if mode == "spec":
        return (1.0 - normalised_cost) if passed else 0.0
    if mode == "memrl":
        return 1.0 if passed else -1.0
    if mode == "spec-signed":
        return (1.0 - normalised_cost) if passed else -1.0
    raise ValueError(f"unknown reward mode {mode!r}")


def episode_cost_usd(db: sqlite3.Connection, episode_id: str,
                     categories: Sequence[str] = ("inference", "tool_compute")) -> float:
    """Sum of an episode's ledger rows in `categories` (call after the controller wrote them)."""
    marks = ", ".join("?" for _ in categories)
    row = db.execute(f"SELECT COALESCE(SUM(usd), 0) FROM ledger WHERE episode_id = ? AND category IN ({marks})",
                     (episode_id, *categories)).fetchone()
    return float(row[0])


# -- config ------------------------------------------------------------------------------------
class UtilityConfig(BaseModel):
    """Group C hyperparameters. Defaults follow the paper (Table 8, BCB/LLB column for k1/k2).
    `sim_mean`, `sim_std` and `delta` have no default: they depend on our embedder and corpus and
    must be estimated (`estimate_similarity_stats`) and registered before the run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k1: int = Field(10, ge=1, description="Phase A: episodes kept after the similarity gate")
    k2: int = Field(5, ge=1, description="Phase B: episodes injected")
    delta: float = Field(description="similarity gate (paper: top-20% quantile of pairwise similarity)")
    gate_inclusive: bool = Field(False, description="paper: sim > delta; code: sim >= delta")
    sim_mean: float = Field(description="fixed corpus mean for sim z-scores")
    sim_std: float = Field(gt=0, description="fixed corpus std for sim z-scores")
    lam: float = Field(0.5, ge=0, le=1, description="lambda: weight of Q in the Phase-B score")
    alpha: float = Field(0.3, gt=0, le=1, description="EMA step of the Q update")
    q_init: float = 0.0
    q_clip: float = Field(3.0, gt=0, description="q_z is clipped to +/- q_clip (code)")
    epsilon: float = Field(0.0, ge=0, le=1, description="code-only epsilon-greedy; paper: 0")
    epsilon_seed: int = 0
    q_floor: float | None = Field(None, description="code-only Q clamp (LLB 0.0); paper: none")
    q_min: float | None = Field(None, description="code-only exclusion of Q < q_min (BCB -0.8); paper: none")
    sim_score: str = Field("semantic", description="key of MemoryItem.scores used as similarity")
    doc_sim_agg: Literal["max", "mean"] = "max"
    units_per_doc: int | None = Field(None, ge=1, description="units injected per episode; None = all returned")
    reward_mode: RewardMode = "spec-signed"

    @model_validator(mode="after")
    def _check(self) -> "UtilityConfig":
        if self.k2 > self.k1:
            raise ValueError("k2 must not exceed k1")
        return self


@dataclass(frozen=True)
class SimilarityStats:
    n: int
    mean: float
    std: float      # population std
    delta: float    # the `quantile` point of the distribution
    quantile: float


def _quantile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default 'linear' method)."""
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def estimate_similarity_stats(similarities: Iterable[float], quantile: float = 0.8) -> SimilarityStats:
    """sim_mean, sim_std and delta from a pool of ticket-vs-memory similarities.

    MemRL's sim_norm_mean/std are the mean and std of pairwise task similarity over the benchmark,
    and delta is its top-20% quantile (App. E.2). In C-SPEC the similarity being gated is the
    episode-level score (`doc_similarities`), so pool `doc_similarities(recall(ticket))` over the
    train-split tickets (a calibration bank, never the sealed tickets) and freeze the result."""
    xs = sorted(float(x) for x in similarities if x is not None and not math.isnan(x))
    if len(xs) < 2:
        raise ValueError("need at least two similarities")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    mean = math.fsum(xs) / len(xs)
    std = math.sqrt(math.fsum((x - mean) ** 2 for x in xs) / len(xs))
    return SimilarityStats(n=len(xs), mean=mean, std=std, delta=_quantile(xs, quantile), quantile=quantile)


# -- grouping ----------------------------------------------------------------------------------
@dataclass
class _Doc:
    document_id: str
    units: list[MemoryItem]
    sim: float | None


def _unit_sim(item: MemoryItem, key: str) -> float | None:
    v = (item.scores or {}).get(key)
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


def group_by_document(candidates: Sequence[MemoryItem], sim_score: str = "semantic",
                      agg: str = "max") -> tuple[list[_Doc], list[str]]:
    """Episodes in order of first appearance, each with its units in service order and its
    aggregated similarity; plus the ids of units with no document_id (dropped)."""
    docs: dict[str, _Doc] = {}
    unattributed: list[str] = []
    for it in candidates:
        if not it.document_id:
            unattributed.append(it.memory_id)
            continue
        docs.setdefault(it.document_id, _Doc(it.document_id, [], None)).units.append(it)
    for d in docs.values():
        sims = [s for s in (_unit_sim(u, sim_score) for u in d.units) if s is not None]
        if sims:
            d.sim = max(sims) if agg == "max" else math.fsum(sims) / len(sims)
    return list(docs.values()), unattributed


def doc_similarities(candidates: Sequence[MemoryItem], cfg: UtilityConfig | None = None, *,
                     sim_score: str = "semantic", agg: str = "max") -> list[float]:
    """Episode-level similarities of one recall, for `estimate_similarity_stats`."""
    if cfg is not None:
        sim_score, agg = cfg.sim_score, cfg.doc_sim_agg
    docs, _ = group_by_document(candidates, sim_score, agg)
    return [d.sim for d in docs if d.sim is not None]


# -- trace -------------------------------------------------------------------------------------
@dataclass
class DocScore:
    document_id: str
    sim: float | None
    n_units: int
    q: float | None = None
    visits: int | None = None
    sim_z: float | None = None
    q_z: float | None = None
    score: float | None = None
    status: str = ""   # no_sim | below_delta | beyond_k1 | below_q_min | selected | not_selected


@dataclass
class RankTrace:
    """Everything one `rank` call used, for the episode log."""
    state_sha256: str
    candidate_ids: list[str]                 # unit ids as the service returned them
    unattributed_ids: list[str]              # units without document_id (dropped)
    docs: list[DocScore]                     # every candidate episode, first-appearance order
    phase1_document_ids: list[str]           # Phase-A survivors, by sim
    ranked_document_ids: list[str]           # Phase-B order of the survivors (after q_min)
    selected_document_ids: list[str]         # injected episodes, in injection order
    injected_ids: list[str]                  # injected unit ids, in injection order
    q_mean: float | None = None
    q_std: float | None = None               # std used for q_z (after the degenerate-case rule)
    epsilon_fired: bool = False
    readonly: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# -- learner -----------------------------------------------------------------------------------
class ReadOnlyLearnerError(RuntimeError):
    pass


def _canonical(cfg: UtilityConfig, table: dict[str, tuple[float, int]]) -> bytes:
    body = {"format": STATE_FORMAT, "config": cfg.model_dump(mode="json"),
            "q": {doc: [q, v] for doc, (q, v) in sorted(table.items())}}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class UtilityLearner:
    """Q table for one (run_id, history, test_group), persisted in the experiment store's SQLite DB."""

    def __init__(self, db: sqlite3.Connection, run_id: str, history: int, group: str, config: UtilityConfig):
        self.db = db
        self.run_id, self.history, self.group = run_id, history, group
        self.config = config
        self.readonly = False
        self.snapshot_sha256: str | None = None
        self.last_trace: RankTrace | None = None
        with db:
            db.executescript(TABLES)
        self._q: dict[str, tuple[float, int]] = {
            r[0]: (float(r[1]), int(r[2])) for r in db.execute(
                "SELECT document_id, q, visits FROM learner_utility WHERE run_id = ? AND history = ? AND test_group = ?",
                (run_id, history, group))}

    @classmethod
    def for_store(cls, store, run_id: str, history: int, group: str, config: UtilityConfig) -> "UtilityLearner":
        return cls(store.db, run_id, history, group, config)

    # -- state ---------------------------------------------------------------------------------
    def q(self, document_id: str) -> float:
        return self._q.get(document_id, (self.config.q_init, 0))[0]

    def visits(self, document_id: str) -> int:
        return self._q.get(document_id, (self.config.q_init, 0))[1]

    def table(self) -> dict[str, tuple[float, int]]:
        return dict(self._q)

    def state_bytes(self) -> bytes:
        return _canonical(self.config, self._q)

    def state_sha256(self) -> str:
        """Hash of config + Q table; recorded as memory_snapshots.learner_state_sha256."""
        return hashlib.sha256(self.state_bytes()).hexdigest()

    def snapshot(self, store) -> str:
        """Store the state as a blob ("learner" kind) and return its digest == state_sha256()."""
        sha = store.put_blob("learner", self.state_bytes())
        assert sha == self.state_sha256()
        return sha

    def readonly_view(self) -> "UtilityLearner":
        """A frozen copy of the current state (e.g. `dev` reading the live learner)."""
        view = object.__new__(UtilityLearner)
        view.db, view.run_id, view.history, view.group = None, self.run_id, self.history, self.group
        view.config, view._q = self.config, dict(self._q)
        view.readonly, view.snapshot_sha256, view.last_trace = True, None, None
        return view

    @classmethod
    def load_snapshot(cls, store, state_sha256: str, *, config: UtilityConfig | None = None,
                      run_id: str = "", history: int = -1, group: str = "") -> "UtilityLearner":
        """Read-only learner from a stored state blob. `config`, if given, must equal the stored one."""
        body = json.loads(store.get_blob("learner", state_sha256))
        if body.get("format") != STATE_FORMAT:
            raise ValueError(f"learner state {state_sha256} has format {body.get('format')!r}")
        stored_cfg = UtilityConfig.model_validate(body["config"])
        if config is not None and config != stored_cfg:
            raise ValueError(f"learner state {state_sha256} was taken under a different UtilityConfig")
        view = object.__new__(cls)
        view.db, view.run_id, view.history, view.group = None, run_id, history, group
        view.config = stored_cfg
        view._q = {doc: (float(q), int(v)) for doc, (q, v) in body["q"].items()}
        view.readonly, view.snapshot_sha256, view.last_trace = True, state_sha256, None
        if view.state_sha256() != state_sha256:
            raise ValueError(f"learner state {state_sha256} does not re-hash to itself")
        return view

    # -- retrieval -----------------------------------------------------------------------------
    def _read_q(self, document_id: str) -> float:
        q = self.q(document_id)
        if self.config.q_floor is not None:
            q = max(self.config.q_floor, q)
        return q

    def _rng(self, task) -> random.Random:
        key = getattr(task, "content_hash", None) or getattr(task, "task_id", None) or str(task)
        seed = hashlib.sha256(f"{self.config.epsilon_seed}:{key}:{self.state_sha256()}".encode()).digest()
        return random.Random(int.from_bytes(seed[:8], "big"))

    def rank_with_trace(self, task, candidates: Sequence[MemoryItem],
                        k2: int | None = None) -> tuple[list[MemoryItem], RankTrace]:
        """`k2` overrides the configured number of injected episodes (when other slots of the
        memory block are already taken, e.g. by verbatim rules)."""
        cfg = self.config
        k2_max = cfg.k2 if k2 is None else max(0, min(k2, cfg.k2))
        docs, unattributed = group_by_document(candidates, cfg.sim_score, cfg.doc_sim_agg)
        rows = {d.document_id: DocScore(d.document_id, d.sim, len(d.units)) for d in docs}

        # Phase A: similarity gate, top-k1 by sim.
        gated = []
        for d in docs:
            if d.sim is None:
                rows[d.document_id].status = "no_sim"
            elif d.sim > cfg.delta or (cfg.gate_inclusive and d.sim == cfg.delta):
                gated.append(d)
            else:
                rows[d.document_id].status = "below_delta"
        gated.sort(key=lambda d: (-d.sim, d.document_id))
        for d in gated[cfg.k1:]:
            rows[d.document_id].status = "beyond_k1"
        phase1 = gated[:cfg.k1]

        # Phase B: z(sim) with fixed stats, z(Q) within the candidate set, clipped.
        pool = []
        for d in phase1:
            r = rows[d.document_id]
            r.q, r.visits = self._read_q(d.document_id), self.visits(d.document_id)
            if cfg.q_min is not None and r.q < cfg.q_min:
                r.status = "below_q_min"
            else:
                pool.append(d)
        qs = [rows[d.document_id].q for d in pool]
        q_mean = q_std = None
        if qs:
            q_mean = math.fsum(qs) / len(qs)
            q_std = math.sqrt(math.fsum((q - q_mean) ** 2 for q in qs) / len(qs)) if len(qs) > 1 else 1.0
            if q_std <= 1e-9:
                q_std = 1.0
        for d in pool:
            r = rows[d.document_id]
            r.sim_z = (d.sim - cfg.sim_mean) / cfg.sim_std
            r.q_z = max(-cfg.q_clip, min(cfg.q_clip, (r.q - q_mean) / q_std))
            r.score = (1.0 - cfg.lam) * r.sim_z + cfg.lam * r.q_z
        pool.sort(key=lambda d: (-rows[d.document_id].score, -d.sim, d.document_id))

        k2 = min(k2_max, len(pool))
        fired = False
        if cfg.epsilon > 0 and pool:
            rng = self._rng(task)
            if rng.random() < cfg.epsilon:
                fired = True
                picked = {d.document_id for d in rng.sample(pool, k2)}
                selected = [d for d in pool if d.document_id in picked]  # injected in score order
        if not fired:
            selected = pool[:k2]
        for d in pool:
            rows[d.document_id].status = "not_selected"
        out: list[MemoryItem] = []
        for d in selected:
            rows[d.document_id].status = "selected"
            out += d.units[:cfg.units_per_doc] if cfg.units_per_doc else d.units

        trace = RankTrace(
            state_sha256=self.state_sha256(), candidate_ids=[c.memory_id for c in candidates],
            unattributed_ids=unattributed, docs=[rows[d.document_id] for d in docs],
            phase1_document_ids=[d.document_id for d in phase1],
            ranked_document_ids=[d.document_id for d in pool],
            selected_document_ids=[d.document_id for d in selected],
            injected_ids=[it.memory_id for it in out], q_mean=q_mean, q_std=q_std,
            epsilon_fired=fired, readonly=self.readonly)
        self.last_trace = trace
        return out, trace

    def rank(self, task, candidates: Sequence[MemoryItem], k2: int | None = None) -> list[MemoryItem]:
        """Learner.rank (SPEC §10). The full record of the call is left in `self.last_trace`."""
        return self.rank_with_trace(task, candidates, k2)[0]

    # -- learning ------------------------------------------------------------------------------
    def observe(self, injected_document_ids: Iterable[str], reward: float, *,
                episode_id: str | None = None) -> dict[str, float]:
        """Q <- Q + alpha (r - Q) for every distinct injected episode (new ones start at q_init).
        Returns the new Q per episode. Raises on a read-only learner."""
        if self.readonly:
            raise ReadOnlyLearnerError("this UtilityLearner is a read-only view (checkpoint/sealed/dev); "
                                       "only train episodes may observe")
        reward = float(reward)
        if math.isnan(reward) or math.isinf(reward):
            raise ValueError(f"reward must be finite, got {reward!r}")
        cfg = self.config
        new: dict[str, float] = {}
        seen = []
        for doc in injected_document_ids:
            if doc and doc not in seen:
                seen.append(doc)
        with self.db:
            for doc in seen:
                before, visits = self._q.get(doc, (cfg.q_init, 0))
                after = before + cfg.alpha * (reward - before)
                if cfg.q_floor is not None:
                    after = max(cfg.q_floor, after)
                self.db.execute(
                    "INSERT INTO learner_utility (run_id, history, test_group, document_id, q, visits) VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (run_id, history, test_group, document_id) DO UPDATE SET q = excluded.q, visits = excluded.visits",
                    (self.run_id, self.history, self.group, doc, after, visits + 1))
                self.db.execute(
                    "INSERT INTO learner_utility_updates (run_id, history, test_group, episode_id, document_id, reward, q_before, q_after) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (self.run_id, self.history, self.group, episode_id, doc, reward, before, after))
                new[doc] = after
        for doc, q in new.items():
            self._q[doc] = (q, self._q.get(doc, (cfg.q_init, 0))[1] + 1)
        return new

    def observe_episode(self, injected_document_ids: Iterable[str], *, passed: bool, episode_cost: float,
                        budget_cost: float, episode_id: str | None = None) -> tuple[float, dict[str, float]]:
        """Convenience: reward from the verdict and cost under the configured mode, then observe."""
        r = reward(passed, normalised_cost(episode_cost, budget_cost), self.config.reward_mode)
        return r, self.observe(injected_document_ids, r, episode_id=episode_id)

    def propose_skill(self):
        return None
