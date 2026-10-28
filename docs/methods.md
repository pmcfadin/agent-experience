# Methods

Research question, hypotheses, test groups, the method behind each group, the task generator and
the protocol. Moved from the README unchanged. Full design: [`SPEC.md`](../SPEC.md).

## Research question and hypotheses

Primary hypothesis (H1): after 200 training episodes, the continual-learning agent (group D)
achieves at least 20% lower fully loaded cost per verified success on sealed, unseen tasks than
the stateless agent (group A), without losing more than a preregistered margin of success rate.

Cost per success = (inference + tool compute + memory service + reflection + validation +
amortised learning) / verified successes.

| | Hypothesis | Status |
|---|---|---|
| H1 | D is at least 20% cheaper per success than A on sealed tasks | Untested: sealed tasks are not run until the protocol is locked |
| H2 | The advantage grows across checkpoints | Untested: all runs so far have one checkpoint |
| H3 | Learning transfers to new (cause, theme) combinations and to new causes | Untested: needs the sealed novel-combination and novel-cause splits |
| H4 | Utility ranking (C) and validated procedures (D) add value beyond retrieval (B) | Not supported so far (see results) |
| H5 | New learning does not erode earlier ability | Untested: retention phases are implemented but not yet run |

## Test groups

Each memory group implements the method of a cited paper, taken from the paper and its released
code where available. Deviations are stated.

| Group | Carried between tasks | Method source |
|---|---|---|
| A | Nothing (stateless baseline) | — |
| B | Hindsight memory bank (one record per training episode) plus one WHEN–THEN rule per episode stored verbatim; the agent can also call a `search_memory` tool | Hindsight; Learning on the Job |
| C | B, with recall candidates reranked by a utility score learned from outcomes | MemRL |
| D | C, plus a procedure library induced from passed episodes and promoted only if it wins on a dev panel | Agent Workflow Memory; GEPA (optional) |

Memory is shown once, in a delimited block in the first user message of an episode. Each (history,
group) has its own Hindsight bank. Only `train` episodes write memory; evaluation episodes read a
frozen snapshot.

### Methods and deviations

**Hindsight** (Latimer et al., arXiv [2512.12818](https://arxiv.org/abs/2512.12818)); Hindsight
1.1.0 server. Memory with retain / recall / reflect; recall fuses semantic, keyword, graph and
temporal search and reranks with a cross-encoder. Implementation: `elh/memory/hindsight.py`, record
format `lesson-v3` (a one-sentence outcome context without ticket text, tagged by outcome and failed
checks), recall restricted to experience facts, ranked candidate list logged on every recall.
Deviation: `reflect` is not called during episodes (about 43 s and 44,000 tokens per call).

**Learning on the Job** (Tablan, Taylor & Bernhem, arXiv
[2607.22157](https://arxiv.org/abs/2607.22157)). After each training episode the agent writes one
rule: `VERIFIED RULE: WHEN … THEN …` after a pass, `AVOID: …` after a failure, with a query field
for retrieval; a write-time validator bounces rules that omit key facts; the agent searches memory
itself during the task. Implementation: `elh/learner/rules.py`, the paper's reflection and rejection
prompts adapted to tickets. The rule writer sees the judge's per-check verdict, never the hidden
tests or reference fix. Rules are stored verbatim (no LLM extraction) and fill the first two memory
slots. Deviations: with no gold solution the validator only requires the protected-file names;
conflicting-rule merging is not implemented; rules are also recalled by ticket text at episode
start, not only through the agent's own queries.

**MemRL** (Zhang et al., arXiv [2601.03192](https://arxiv.org/abs/2601.03192)). Two-phase recall:
keep candidates above a similarity threshold, take the top k1, rank by 0.5·z(similarity) + 0.5·z(Q),
inject the top k2; after each episode Q ← Q + 0.3(r − Q) for every injected memory. Implementation:
`elh/learner/utility.py`, reranking Hindsight's candidate list with Q kept per source episode;
evaluation episodes use a frozen copy of Q. Deviations: failure reward is −1 (as in MemRL) rather
than SPEC's 0, under which Q could never fall; the similarity threshold is the 20th percentile of
observed candidate similarities (0.838 on `quadratic_membership`), because Hindsight has already
filtered candidates; the threshold calibrated on `quadratic_membership` was reused for
`nested_scan_join`.

**Agent Workflow Memory** (Wang et al., arXiv [2409.07429](https://arxiv.org/abs/2409.07429)).
Induces reusable workflows from successful runs and regenerates the library after each success.
Implementation: `elh/learner/workflows.py`, AWM's induction prompt with three domain substitutions,
run on passed training episodes; libraries are versioned with their source episodes. Addition: the
proposal requires validation, which AWM does not have. A candidate library is promoted only if, on
6 dev tasks run twice each, mean reward rises, passes do not fall and protected-file violations do
not rise. Dev-gate episodes are charged as learning cost and excluded from comparisons.

**GEPA** (Agrawal et al., arXiv [2507.19457](https://arxiv.org/abs/2507.19457)), optional source
for D (`procedure_source: gepa`). Evolves text by reflecting on rollouts, keeping a Pareto frontier
over validation tasks. Implementation: `elh/learner/gepa_opt.py`, an adapter for the `gepa` package
that optimises D's procedure text on dev tasks only. Deviations: the reflection prompt asks for
general procedures instead of niche domain facts; merge is off. Not yet run in an experiment.

**When Continual Learning Moves to Memory** (Hu, Long & Wang, arXiv
[2604.27003](https://arxiv.org/abs/2604.27003)), used for analysis, not as a group. Implementation:
`elh/analysis.py` computes ΔRR (pass rate on tasks A solves; negative means memory breaks them) and
ΔNL (pass rate on tasks A fails; withheld below 10 A-failures, as in the paper), forward/backward
transfer, hold rate and retrieval diagnostics.

## Task generator

One task family is implemented: `cpu_mem` (`elh/tasks/families/cpu_mem/`). Each instance is a
small Python batch job (`job/` package, `run.py`, a public example and test) that reads an input
JSON and writes an output JSON. The job is correct but contains one algorithmic regression that
makes it time out at production input sizes. The ticket describes the symptom, never the cause.

Latent causes:

| Cause | Defect | Typical fix |
|---|---|---|
| `quadratic_membership` | `x in list` inside a loop (the generated jobs have two such lists) | set / dict |
| `nested_scan_join` | for each record, scan another collection for its match | build an index |
| `queue_pop_front` | `list.pop(0)` / `insert(0, …)` work queue | `collections.deque` |
| `repeated_sort_min` | re-sorting to take the min/max inside a loop | `heapq` |
| `exponential_recursion` | naive overlapping recursion | memoisation / DP |

Themes vary vocabulary and data shape: `inventory`, `logistics`, `telemetry`, `billing`,
`course_registration`, `network_flows`.

An instance is generated deterministically from `(family, cause, theme, seed)`; its id is
`cpu_mem/<cause>/<theme>/<seed>` and it carries a `content_hash` over ticket, workspace and oracle.
`elh validate-family cpu_mem` checks, for every cause × theme and many seeds, that the reference fix
passes, the untouched job fails on performance (not correctness), a plausible wrong fix fails on
correctness, hard-coding the public output fails, and regeneration is byte-identical.

Protected files (the agent must not change them): `run.py`, `examples/sample_input.json`,
`examples/sample_output.json`, `tests/test_job.py`, `scripts/make_load.py`.

The oracle never enters the sandbox while the agent works. After submission the judge
(`elh/judge.py`) grades the final workspace in a fresh sandbox. A pass requires all checks:

- `integrity`: protected files unchanged;
- `public_tests`: the public test still passes;
- `correctness`: output equals the reference implementation's on hidden inputs (3 small, 2
  medium, including empty, duplicate and tie cases);
- `performance`: on a large hidden input the job finishes within max(3 s, 10 × reference time),
  measured in the same sandbox in the same run.

The agent's own submit message is stored but never used for grading.

To see a ticket and its workspace:

```
uv run elh show-task cpu_mem quadratic_membership inventory 2024 --out /tmp/task
```

## Protocol

**Episodes.** Agent tools are `bash` and `submit`. Budget per episode: 40 turns, 600,000 tokens,
1,800 s wall clock, 120 s per bash call. Sampling is temperature 0.7, top_p 0.8. The agent model
and the memory LLM (Gemma 4 26B, used by Hindsight) are pinned by weights sha256, quantization,
llama.cpp build and flags, and context size; every episode reads the backend's `/props` and is an
infrastructure error if any of these differ from the pin.

**Splits and seeds.** Splits are assigned over (cause, theme) cells, not seeds. In the batch
experiments, training uses five themes and evaluation uses a sixth (`inventory`), with disjoint
seed ranges: train seeds 1000–1999, eval seeds 2000–2999 (2900+ for difficulty probes), dev seeds
3000–3999 on train cells only, never on a cell an eval phase measures. Seed ranges in configs are
half-open, e.g. `seeds: [1006, 1009]` is seeds 1006, 1007, 1008 across each theme. SPEC also defines
`checkpoint`, `retention` and three sealed splits (familiar, novel combination, novel cause); the
sealed seed bases are kept outside the repository and none of these sealed splits has been run.

**Histories.** A history is one independent learning run: its own shuffled training order and
empty banks. Model nondeterminism is handled by replicating histories, not by seeds. Group A has no
training phase; groups B–D run the same training sequence within a history.

**Frozen snapshots.** After training, each group's bank is exported, content-fingerprinted and
restored into a snapshot bank. Evaluation episodes read that snapshot read-only; the bank
fingerprint and learner-state hash are checked before recall and after grading, and any change is
an infrastructure error. Group D's snapshot also pins its library revision.

**Registration.** Each experiment file is hashed (`config_hash`, over the parsed config) and the
hash is stamped on every stored row; a run refuses to start if a file with the same name was
previously run under a different hash. Old configs keep their hashes so old runs reproduce.

**Analysis** (`elh analyze <experiment>`, `elh/analysis.py`): pass rates with Wilson intervals;
paired comparisons against A by task with exact McNemar tests on discordant pairs; a paired cluster
bootstrap clustered by task (all episodes of a task across histories in one cluster); ΔRR / ΔNL;
turns; and compute cost as GPU-seconds (llama.cpp `prompt_ms + predicted_ms` summed over the agent's
calls, plus Hindsight's memory-LLM request time attributed to training episodes). The ledger also
records token costs at placeholder shadow prices; see limitations.
