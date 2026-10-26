"""Group D's procedure optimiser: GEPA (Agrawal et al., ICLR 2026) over one text component.

The component is a `<procedure>` block shown to the frozen agent. GEPA itself is the `gepa`
package, pinned in pyproject.toml (0.1.4; the paper's algorithm is its default configuration).
This module supplies what GEPA leaves to the user (papers/gepa.md §3):

- `ProcedureAdapter`: GEPA's adapter. `evaluate` runs dev-split episodes through an injected
  `run_dev_episode(task, procedure_text)` (the controller's job: a fixed read-only memory snapshot,
  ledger category `learning`) and scores each with partial credit per judge check plus a cost term;
  an integrity violation scores 0. `make_reflective_dataset` writes the "Feedback" with group B's
  lesson-v2 wording (`render_episode`), from a whitelisted verdict: never reference timings, hidden
  test inputs, the oracle or a reference solution.
- `ReflectionLM`: text in, text out over our `ModelClient` (the same frozen agent model, as in the
  paper's Qwen3-8B runs), recording every call's usage so it is charged to `learning`.
- `REFLECTION_PROMPT`: replaces GEPA's default meta-prompt (see the deviation note there).
- `run_gepa`: GEPA with the paper's settings (Pareto selection, minibatch 3, merge off, budget in
  rollouts) on two dev pools, then the paired promotion check on a third, returning the winner,
  the decision and the full candidate history for provenance.

Settings that differ from GEPA's defaults, and why:
- `skip_perfect_score=False`: our score reaches 1.0 only at zero cost, so "perfect" never means
  "nothing to learn" (report §3.4).
- `cache_evaluation=False` (GEPA default) and `use_merge=False` (GEPA default; merge hurt Qwen3-8B,
  Table 1, and is a no-op with one component, report §1.6).
- The reflection prompt (below) and the score (`score_rollout`), which are ours by design.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version as _pkg_version
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from elh.judge import Verdict
from elh.registry import GepaConfig  # noqa: F401  (re-export)
from elh.memory.hindsight import render_episode
from elh.runner.agent import EpisodeResult
from elh.runner.models import InfraError, ModelClient
from elh.tasks.base import TaskInstance

COMPONENT = "procedure"
GEPA_VERSION = _pkg_version("gepa")

# Splits a GEPA rollout may never touch (SPEC §4.3); only `dev` is allowed.
FORBIDDEN_SPLITS = ("train", "checkpoint", "sealed_familiar", "sealed_novel_combo", "sealed_novel_cause",
                    "retention")


def procedure_block(text: str) -> str:
    """How a procedure is shown to the agent (first user message, before `<experience>`). Group D
    only; an empty procedure adds nothing, so groups A-C stay byte-identical."""
    text = text.strip()
    return f"<procedure>\n{text}\n</procedure>" if text else ""


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------

# GepaConfig lives in elh.registry (the registration file holds it); re-exported here.


class DevPools(BaseModel):
    """Three disjoint dev task-id pools (report §3.3): `reflection` feeds GEPA's minibatches (its
    trainset, D_feedback), `validation` is GEPA's valset (D_pareto, scores only), `promotion` is the
    paired incumbent-vs-winner check run after GEPA (D_confirm)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reflection: list[str]
    validation: list[str]
    promotion: list[str]

    @model_validator(mode="after")
    def _disjoint(self) -> "DevPools":
        pools = {"reflection": self.reflection, "validation": self.validation, "promotion": self.promotion}
        for name, ids in pools.items():
            if not ids:
                raise ValueError(f"dev pool {name!r} is empty")
            if len(set(ids)) != len(ids):
                raise ValueError(f"dev pool {name!r} repeats a task id")
        names = list(pools)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                both = set(pools[a]) & set(pools[b])
                if both:
                    raise ValueError(f"dev pools {a!r} and {b!r} share tasks: {sorted(both)[:5]}")
        return self

    def all_ids(self) -> list[str]:
        return self.reflection + self.validation + self.promotion


def split_dev_pools(task_ids: Sequence[str], cfg: GepaConfig, seed: str | int) -> DevPools:
    """Deterministic three-way split of a dev panel: the ids are sorted, shuffled with `seed`, then
    reflection takes max(minibatch, ceil(n/3)), validation half the rest (rounded up), promotion the
    remainder. Six tasks give 3/2/1. Raises if any pool would be empty."""
    ids = sorted(set(task_ids))
    random.Random(str(seed)).shuffle(ids)
    n = len(ids)
    n_ref = max(cfg.reflection_minibatch_size, -(-n // 3))
    rest = n - n_ref
    n_val = -(-rest // 2)
    if n_ref > n or n_val < 1 or rest - n_val < 1:
        raise ValueError(f"a dev panel of {n} tasks is too small for GEPA's three pools "
                         f"(reflection needs {cfg.reflection_minibatch_size}, validation and promotion one each)")
    return DevPools(reflection=ids[:n_ref], validation=ids[n_ref:n_ref + n_val], promotion=ids[n_ref + n_val:])


def check_dev_only(pools: DevPools, split_of: Callable[[str], str]) -> None:
    """Every pool task must be a `dev` task; sealed, checkpoint, retention (and train) never are."""
    bad = {tid: split_of(tid) for tid in pools.all_ids()}
    bad = {tid: s for tid, s in bad.items() if s != "dev"}
    if bad:
        raise ValueError(f"GEPA may only use dev tasks; got {dict(sorted(bad.items())[:5])}"
                         f"{' ...' if len(bad) > 5 else ''}")


# --------------------------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------------------------

@dataclass
class DevRollout:
    """What `run_dev_episode` returns. A plain `(verdict, result, cost)` tuple is accepted too.
    `diff` is the agent's change (`workspace_diff(task.workspace, final_ws)`), shown in the feedback
    as in lesson-v2; `episode_id` is kept for provenance (`skills.source_episode_ids_json`)."""

    verdict: Verdict
    result: EpisodeResult
    cost: float
    episode_id: str | None = None
    diff: str | None = None


RunDevEpisode = Callable[[TaskInstance, str], "DevRollout | tuple"]


def _as_rollout(out: Any) -> DevRollout:
    if isinstance(out, DevRollout):
        return out
    return DevRollout(*out)


def integrity_broken(verdict: Verdict) -> bool:
    return (not verdict.checks.get("integrity", True)) or bool((verdict.detail or {}).get("integrity_violations"))


def score_rollout(verdict: Verdict, cost: float, cfg: GepaConfig) -> float:
    """Graded reward in [0, 1] (report §3.4). SPEC §10's r = passed x (1 - normalised_cost) is the
    pass branch with a floor (1 - cost_weight); failures get partial credit for the checks they got
    furthest on, so GEPA has a gradient at an 80-90% pass rate."""
    if integrity_broken(verdict):
        return 0.0
    if verdict.passed:
        norm = min(max(cost / cfg.cost_ref, 0.0), 1.0)
        return 1.0 - cfg.cost_weight * norm
    failed = {k for k, ok in verdict.checks.items() if not ok}
    if failed == {"performance"}:
        q = (verdict.detail or {}).get("quality") or {}
        agent_s, budget_s = q.get("agent_s"), q.get("budget_s")
        speed = min(budget_s / agent_s, 1.0) if agent_s and budget_s else 0.0
        return cfg.credit_performance_only + cfg.perf_speed_bonus * speed
    if "public_tests" in failed or "oracle" in failed:
        return cfg.credit_public_tests_failed
    return cfg.credit_correctness_failed


# --------------------------------------------------------------------------------------------
# Feedback (what reflection may see)
# --------------------------------------------------------------------------------------------

# Verdict fields the reflection may see: the same ones lesson-v2 renders. Everything else in
# `detail` (reference_s, agent_over_reference, cases with hidden inputs, oracle stderr) is dropped.
_QUALITY_KEEP = ("agent_s", "budget_s", "large_status")


def safe_verdict(verdict: Verdict) -> dict:
    detail = verdict.detail or {}
    q = {k: v for k, v in (detail.get("quality") or {}).items() if k in _QUALITY_KEEP}
    safe_detail: dict = {}
    if q:
        safe_detail["quality"] = q
    if detail.get("integrity_violations"):
        safe_detail["integrity_violations"] = list(detail["integrity_violations"])
    checks = {k: bool(v) for k, v in verdict.checks.items() if k != "oracle"}
    broken = integrity_broken(verdict)
    if broken:
        checks["integrity"] = False  # as the judge does: a violation is never a pass
    return {"passed": bool(verdict.passed) and not broken, "checks": checks, "detail": safe_detail}


def _tail(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else "..." + text[-n:]


def compact_actions(result: EpisodeResult, max_chars: int) -> str:
    """The agent's own visible work: its short remarks and each command with exit status and the
    tail of its output. The first message (procedure, memory) and all usage data are left out."""
    entries: list[str] = []
    for ev in result.trace:
        if ev.get("event") == "response":
            said = " ".join(b.get("text", "") for b in ev.get("content") or [] if b.get("type") == "text").strip()
            if said:
                entries.append(f"[turn {ev.get('turn')}] agent: {_tail(said, 300)}")
        elif ev.get("event") == "tool_exec":
            status = "TIMED OUT" if ev.get("timed_out") else f"exit {ev.get('exit_code')}"
            out = _tail(ev.get("stdout", ""), 300)
            err = _tail(ev.get("stderr", ""), 300)
            entries.append(f"[turn {ev.get('turn')}] $ {(ev.get('command') or '')[:400]}\n-> {status}, "
                           f"{ev.get('wall_s')}s" + (f"\nstdout: {out}" if out else "") +
                           (f"\nstderr: {err}" if err else ""))
    if not entries:
        return "(no commands run)"
    text = "\n".join(entries)
    if len(text) <= max_chars:
        return text
    kept: list[str] = []
    size = 0
    for e in reversed(entries):  # the last actions matter most; keep as many as fit
        if size + len(e) + 1 > max_chars - 80:
            break
        kept.append(e)
        size += len(e) + 1
    return f"({len(entries) - len(kept)} earlier actions omitted)\n" + "\n".join(reversed(kept))


def feedback_text(task: TaskInstance, rollout: DevRollout, score: float, cfg: GepaConfig) -> str:
    """lesson-v2's record (`render_episode`: failed checks in failure wording, protected files
    named, timing against the budget, outcome, summary, diff) plus the score GEPA optimises."""
    r = rollout.result
    ep = {"family": task.family, "title": task.prompt.strip().splitlines()[0] if task.prompt.strip() else None,
          "verdict": safe_verdict(rollout.verdict), "outcome": r.outcome, "turns": r.turns,
          "tool_calls": r.tool_calls, "submit_summary": r.submit_summary, "diff": rollout.diff}
    text = render_episode(ep)
    if integrity_broken(rollout.verdict):
        text += ("\n\nScore: 0 (integrity). Changing protected files voids the attempt, whatever else "
                 "it achieved.")
    else:
        text += (f"\n\nScore: {score:.2f} of 1 (a pass scores {1 - cfg.cost_weight:.1f}-1.0, cheaper is higher; "
                 f"a failure at most {cfg.credit_performance_only + cfg.perf_speed_bonus:.1f}).")
    return text


def _assert_no_leak(text: str, task: TaskInstance) -> None:
    """Belt and braces on top of the whitelist: no oracle file may appear in the feedback."""
    for name, content in task.oracle.files.items():
        if content.strip() and content.strip() in text:
            raise AssertionError(f"oracle file {name} leaked into the reflective dataset of {task.task_id}")


# --------------------------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------------------------

@dataclass
class _Trajectory:
    task_id: str
    rollout: DevRollout
    score: float


class ProcedureAdapter:
    """GEPAAdapter for one component. Data instances are task ids; rollouts run serially (the
    model alias serves one request at a time)."""

    propose_new_texts = None  # use GEPA's reflective proposer with REFLECTION_PROMPT

    def __init__(self, cfg: GepaConfig, tasks: Mapping[str, TaskInstance], run_dev_episode: RunDevEpisode,
                 stage: str = "gepa"):
        self.cfg = cfg
        self.tasks = tasks
        self.run_dev_episode = run_dev_episode
        self.stage = stage
        self.rollouts: list[dict] = []  # provenance: one row per dev episode

    def _run(self, task: TaskInstance, procedure: str) -> DevRollout:
        for attempt in range(self.cfg.infra_retries + 1):
            try:
                return _as_rollout(self.run_dev_episode(task, procedure))
            except InfraError:
                if attempt == self.cfg.infra_retries:
                    raise
        raise AssertionError("unreachable")

    def rollout(self, task_id: str, procedure: str, capture: bool) -> tuple[float, _Trajectory]:
        task = self.tasks[task_id]
        ro = self._run(task, procedure)
        s = score_rollout(ro.verdict, ro.cost, self.cfg)
        self.rollouts.append({"stage": self.stage, "task_id": task_id, "procedure_sha256": text_sha(procedure),
                              "episode_id": ro.episode_id, "passed": bool(ro.verdict.passed),
                              "integrity": not integrity_broken(ro.verdict), "cost": ro.cost, "score": s,
                              "captured": capture})
        return s, _Trajectory(task_id, ro, s)

    def evaluate(self, batch: list[str], candidate: dict[str, str], capture_traces: bool = False):
        from gepa import EvaluationBatch

        procedure = candidate[COMPONENT]
        outputs, scores, trajs = [], [], []
        for task_id in batch:
            s, traj = self.rollout(task_id, procedure, capture_traces)
            outputs.append({"task_id": task_id, "episode_id": traj.rollout.episode_id,
                            "passed": bool(traj.rollout.verdict.passed)})
            scores.append(s)
            trajs.append(traj)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs if capture_traces else None)

    def make_reflective_dataset(self, candidate: dict[str, str], eval_batch, components_to_update: list[str]):
        records = []
        for traj in eval_batch.trajectories or []:
            task = self.tasks[traj.task_id]
            rec = {
                "Inputs": {"Ticket": _clip(task.prompt.strip(), self.cfg.max_ticket_chars)},
                "Generated Outputs": {"Agent actions": compact_actions(traj.rollout.result, self.cfg.max_actions_chars)},
                "Feedback": feedback_text(task, traj.rollout, traj.score, self.cfg),
            }
            _assert_no_leak(json.dumps(rec), task)
            records.append(rec)
        return {c: records for c in components_to_update}


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + " [...]"


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------------------------
# Reflection
# --------------------------------------------------------------------------------------------

# DEVIATION from GEPA's default meta-prompt (gepa.strategies.instruction_proposal, = paper App. C).
# The default tells the reflector to "identify all niche and domain specific factual information
# about the task and include it in the instruction". Our feedback comes from specific dev tickets;
# copying their facts (names, field names, sizes, timings, the exact fix) into the procedure would
# carry ticket specifics forward and teach the cause rather than a way of working, which would
# contaminate the novel-cause/novel-combo splits (H3). This template, adapted from gskill's
# (gskill/train_optimize_anything.py), asks for a general procedure, forbids ticket specifics and
# caps length (prompt tokens are cost charged to the group). It keeps GEPA's two placeholders and
# its single ``` output block, so GEPA's own renderer and extractor are used unchanged.
REFLECTION_PROMPT = """You are improving a short working procedure for an AI coding agent.

The agent fixes operations tickets about slow Python batch jobs. Each ticket is a different job with \
a different root cause. The agent works in an isolated sandbox: each bash command runs in a fresh \
process in the workspace (files persist, shell state does not), there is no network, commands are \
killed after a timeout, and it has a limited number of turns. When it submits, hidden tests check \
that protected files are unchanged, that public and hidden correctness tests pass, and that the \
code is fast enough on a large input.

The agent is shown this procedure before every ticket:
```
<curr_param>
```

Below are some of its recent attempts while following that procedure: the ticket, the agent's \
actions, and the verified result with a score (higher is better; changing protected files scores 0).
```
<side_info>
```

Write an improved procedure. It will be used on OTHER tickets, with other causes, data and file \
contents, never these ones, so it must generalise:
- Keep what led to passing, cheap attempts; fix the habits behind failures, timeouts, wasted turns \
and any change to protected files.
- Describe how to work: how to reproduce and measure, locate the cause, change code safely, verify \
correctness and speed within the time limits, and when to submit.
- Do NOT include anything specific to these tickets: no job, file, function or field names, no \
numbers, sizes or timings taken from them, and no particular fix. Say "the slow function", not its name.
- Keep earlier steps that are still useful; drop ones that did not help. Adding steps blindly does not help.
- At most <max_words> words, as a short numbered list of imperative steps. No code blocks inside it.

Return the complete new procedure inside a single ``` block, and nothing else."""


def reflection_prompt(cfg: GepaConfig) -> str:
    return REFLECTION_PROMPT.replace("<max_words>", str(cfg.max_procedure_words))


REFLECTION_SYSTEM = "You write concise, general working procedures for software agents."


class ReflectionTruncated(RuntimeError):
    """The reflection hit its token limit; GEPA retries the reflection once, then skips it."""


class ReflectionLM:
    """GEPA's `LanguageModel`: (str | chat messages) -> str, over our ModelClient. Every call's
    usage is kept in `calls` and passed to `on_usage`, so the controller charges it to `learning`
    (see `reflection_ledger_entries`)."""

    def __init__(self, client: ModelClient, *, max_tokens: int = 4096,
                 on_usage: Callable[[dict, dict], None] | None = None, system: str = REFLECTION_SYSTEM):
        self.client = client
        self.max_tokens = max_tokens
        self.on_usage = on_usage
        self.system = system
        self.calls: list[dict] = []

    def __call__(self, prompt: str | list[dict]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": m["role"], "content": m["content"]} for m in prompt if m.get("role") != "system"]
        resp = self.client.create(system=self.system, messages=messages, tools=[], max_tokens=self.max_tokens)
        text = "".join(b.get("text", "") for b in resp.content if b.get("type") == "text")
        call = {"model": resp.model, "request_id": resp.request_id, "stop_reason": resp.stop_reason,
                "usage": dict(resp.usage), "latency_s": resp.latency_s, "served": resp.served,
                "prompt_sha256": text_sha(json.dumps(messages, sort_keys=True)), "output": text,
                "at": time.time()}
        self.calls.append(call)
        if self.on_usage is not None:
            self.on_usage(dict(resp.usage), call)
        if resp.stop_reason == "max_tokens" or ("<think>" in text and "</think>" not in text):
            raise ReflectionTruncated(f"reflection output truncated at {self.max_tokens} tokens")
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    def usage_total(self) -> dict:
        out: dict[str, int] = {}
        for c in self.calls:
            for k, v in c["usage"].items():
                out[k] = out.get(k, 0) + (v or 0)
        return out


def reflection_ledger_entries(cfg, model_id: str, calls: Sequence[dict]):
    """Ledger rows (category `learning`) for reflection calls, priced under the registered id."""
    from elh.ledger import usage_entries

    return [e for c in calls for e in usage_entries(cfg, model_id, c["usage"], category="learning")]


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------

class _ListLogger:
    def __init__(self, echo: bool = False):
        self.lines: list[str] = []
        self.echo = echo

    def log(self, message: str) -> None:
        self.lines.append(message)
        if self.echo:
            print(message, flush=True)


@dataclass
class PromotionCheck:
    incumbent_sha256: str
    challenger_sha256: str
    pairs: list[dict]                 # {task_id, replicate, incumbent, challenger}
    mean_delta: float
    incumbent_integrity_failures: int
    challenger_integrity_failures: int
    promote: bool
    rule: str


@dataclass
class GepaOutcome:
    best_procedure: str
    best_idx: int
    promote: bool
    promotion: PromotionCheck | None
    history: list[dict]               # every candidate: text, sha, parents, validation scores
    rollouts: list[dict]              # every dev episode: stage, task, procedure sha, episode id, score
    reflection_calls: list[dict]      # every reflection call: usage, output (for the ledger)
    total_metric_calls: int
    provenance: dict = field(default_factory=dict)

    def source_episode_ids(self) -> list[str]:
        return [r["episode_id"] for r in self.rollouts if r["episode_id"]]

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def paired_promotion_check(cfg: GepaConfig, incumbent: str, challenger: str, task_ids: Sequence[str],
                           adapter: ProcedureAdapter) -> PromotionCheck:
    """Incumbent vs challenger on the same fresh dev tasks (x replicates), interleaved per task in a
    seeded order. Preregistered rule: mean paired delta > promotion_min_delta and no more integrity
    failures than the incumbent."""
    adapter.stage = "promotion"
    pairs, inc_bad, ch_bad = [], 0, 0
    for rep in range(cfg.promotion_replicates):
        for task_id in task_ids:
            order = [("incumbent", incumbent), ("challenger", challenger)]
            random.Random(f"{cfg.seed}/{rep}/{task_id}").shuffle(order)
            got = {}
            for who, text in order:
                s, traj = adapter.rollout(task_id, text, capture=False)
                got[who] = s
                if integrity_broken(traj.rollout.verdict):
                    if who == "incumbent":
                        inc_bad += 1
                    else:
                        ch_bad += 1
            pairs.append({"task_id": task_id, "replicate": rep, **got})
    delta = statistics.fmean(p["challenger"] - p["incumbent"] for p in pairs)
    promote = delta > cfg.promotion_min_delta and ch_bad <= inc_bad
    return PromotionCheck(text_sha(incumbent), text_sha(challenger), pairs, delta, inc_bad, ch_bad, promote,
                          f"mean paired delta > {cfg.promotion_min_delta} and challenger integrity failures "
                          f"<= incumbent's")


def run_gepa(config: GepaConfig, seed_procedure: str, pools: DevPools, run_dev_episode: RunDevEpisode,
             reflect_client: ModelClient | Callable[[str | list[dict]], str], *,
             tasks: Mapping[str, TaskInstance], split_of: Callable[[str], str],
             on_reflection_usage: Callable[[dict, dict], None] | None = None,
             run_dir: str | None = None, echo: bool = False, promotion_check: bool = True) -> GepaOutcome:
    """One GEPA invocation, warm-started from the incumbent `seed_procedure`.

    `run_dev_episode(task, procedure_text)` runs one dev episode with the procedure shown to the
    agent and returns `DevRollout` or `(verdict, result, cost)`. `reflect_client` is a ModelClient
    (wrapped in `ReflectionLM`) or any GEPA LanguageModel callable. `tasks` maps every pool id to
    its TaskInstance; `split_of(task_id)` must say "dev" for all of them.

    `promotion_check=False` skips the paired check on the promotion pool (`promote` is then False
    and `promotion` None): the batch runner gates every group-D candidate, AWM or GEPA, with
    `workflows.promote` on that pool instead, so there is one promotion rule."""
    import gepa

    check_dev_only(pools, split_of)
    missing = [t for t in pools.all_ids() if t not in tasks]
    if missing:
        raise KeyError(f"no TaskInstance for pool tasks {missing[:5]}")
    if len(pools.reflection) < config.reflection_minibatch_size:
        raise ValueError("the reflection pool is smaller than one minibatch")

    lm = reflect_client
    if hasattr(reflect_client, "create"):
        lm = ReflectionLM(reflect_client, max_tokens=config.reflection_max_tokens, on_usage=on_reflection_usage)
    adapter = ProcedureAdapter(config, tasks, run_dev_episode)
    logger = _ListLogger(echo)

    result = gepa.optimize(
        seed_candidate={COMPONENT: seed_procedure},
        trainset=list(pools.reflection),
        valset=list(pools.validation),
        adapter=adapter,
        reflection_lm=lm,
        reflection_prompt_template=reflection_prompt(config),
        candidate_selection_strategy=config.candidate_selection_strategy,
        frontier_type=config.frontier_type,
        skip_perfect_score=config.skip_perfect_score,
        reflection_minibatch_size=config.reflection_minibatch_size,
        module_selector="round_robin",
        use_merge=config.use_merge,
        max_metric_calls=config.max_metric_calls,
        acceptance_criterion=config.acceptance_criterion,
        cache_evaluation=False,
        seed=config.seed,
        raise_on_exception=True,
        logger=logger,
        run_dir=run_dir,
        display_progress_bar=False,
    )

    history = []
    for i, cand in enumerate(result.candidates):
        history.append({"idx": i, "procedure": cand[COMPONENT], "sha256": text_sha(cand[COMPONENT]),
                        "parents": list(result.parents[i]), "val_mean": result.val_aggregate_scores[i],
                        "val_scores": {pools.validation[k]: v for k, v in result.val_subscores[i].items()},
                        "discovered_at_metric_call": result.discovery_eval_counts[i]})
    best = result.best_candidate[COMPONENT]
    promotion = None
    if promotion_check and text_sha(best) != text_sha(seed_procedure):
        promotion = paired_promotion_check(config, seed_procedure, best, pools.promotion, adapter)
    calls = getattr(lm, "calls", [])
    return GepaOutcome(
        best_procedure=best, best_idx=result.best_idx, promote=bool(promotion and promotion.promote),
        promotion=promotion, history=history, rollouts=adapter.rollouts, reflection_calls=calls,
        total_metric_calls=result.total_metric_calls or 0,
        provenance={"gepa_version": GEPA_VERSION, "config": config.model_dump(mode="json"),
                    "pools": pools.model_dump(mode="json"), "seed_procedure_sha256": text_sha(seed_procedure),
                    "reflection_prompt_sha256": text_sha(reflection_prompt(config)),
                    "reflection_model": getattr(reflect_client, "model_id", None), "gepa_log": logger.lines},
    )
