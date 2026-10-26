# Experiential Learning Harness (ELH)

Research harness for the study *Can AI Agents Become More Experienced?* It asks whether a coding
agent whose weights never change gets better at a class of work because of what it carries from
one task to the next: stored episodes, rules it wrote for itself, learned rankings over those
memories, or validated procedures.

A frozen local model (Qwen3-Coder 30B-A3B, Q4_K_M, served by llama.cpp) fixes generated
performance-bug tickets in a sandbox. A hidden oracle grades every submission. Test groups differ
only in what they carry between tasks; model, prompt, tools, sandbox, budget and judge are the same
for all of them.

This is a pilot. The confirmatory study (sealed tasks, more histories, power analysis) has not
been run. See [Status and limitations](#status-and-limitations).

- Full design: [`SPEC.md`](SPEC.md)
- Current results write-up: [`docs/study-report.html`](docs/study-report.html)
- Implementation notes on each cited paper: [`docs/papers/`](docs/papers/)

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

## Results so far

Each memory group trains on 15 tickets across five themes, then is tested on 10 or 12 new tickets in
the sixth theme. Counts are passes on test tickets. Single history unless stated.

| Exp. | Bug type | Setting | A | B | C | D |
|---|---|---|---|---|---|---|
| 1 | `queue_pop_front` | early memory format | 9/10 | 9/10 | – | – |
| 6 | `queue_pop_front` | groups built from the papers, no rules | 8/10 | 6/10 | 5/10 | – |
| 8a | `quadratic_membership` | no rules | 1/12 | 5/12 | – | – |
| 8b | `quadratic_membership` | rules, paraphrased by Hindsight | 3/12 | 7/12 | – | – |
| 10 | `quadratic_membership` | rules stored verbatim | 4/12 | 8/12 | 12/12 | 10/12 |
| 11 | `quadratic_membership` | as 10, new tickets | 3/12 | 12/12 | 8/12 | – |
| 12 | `nested_scan_join` | as 11 | 9/12 | 9/12 | 10/12 | – |
| 13 | `quadratic_membership` | as 11, three histories, new tickets | 6/36 | 26/36 | 23/36 | – |

Experiment 13 in detail (same 15 training and 12 test tickets, three histories):

| Measure | A | B | C |
|---|---|---|---|
| Training passed, by history (of 15) | – | 10 · 7 · 3 | 9 · 8 · 3 |
| Test passed, by history (of 12) | 2 · 3 · 1 | 11 · 12 · 3 | 11 · 10 · 2 |
| Paired wins / losses against A (McNemar p) | – | 21 / 1 (< 0.001) | 18 / 1 (< 0.001) |
| Pass-rate difference against A [95% CI] | – | +56% [+42, +67] | +47% [+33, +61] |
| Agent GPU time per success against A [95% CI] | – | 0.21× [0.07, 0.35] | 0.20× [0.07, 0.33] |
| Mean turns per test episode | 34.2 | 28.8 | 27.8 |

Findings, with their caveats:

- On a bug the baseline usually fails (`quadratic_membership`), memory with verbatim rules raised
  the pass rate substantially in every run since experiment 10.
- The size of the gain depends on what the first training tickets teach. In experiment 13, B fixed
  11, 12 and 3 of 12 test tickets in the three histories. In the third history no stored rule
  mentioned the ticket's second slow list lookup (0 of 30 rules, against 31 of 60 in the other two),
  and in 45 of its 47 timeouts the agent had fixed only the first lookup.
- MemRL reranking (C) did not improve on plain memory (B): B against C in experiment 13 was 6 wins,
  3 losses (p = 0.51); across experiments 10, 11 and 13, B fixed 46 of 60 and C 43 of 60. B and C
  swapped order between experiments 10 and 11.
- Where the baseline already succeeds, memory reduced cost but not failures: on `nested_scan_join`
  (experiment 12) C used 0.67× A's GPU time per success (95% CI 0.44–0.93), with about 8 fewer turns.
  On the easy `queue_pop_front` bug (experiment 6) the memory groups did worse than A.
- Validated procedures (D) did not improve on C in the one run that included them (experiment 10:
  D 10/12, C 12/12). The dev gate rejected two of three AWM candidates; its 72 dev episodes cost
  2,877 agent GPU-seconds, charged as learning.
- Memory did not prevent protected-file violations: agents with memory still overwrote
  `examples/sample_output.json`, including right after recalling a rule that named the file.
- Experiment 1's first run was voided because failed checks were recorded in passing wording;
  experiment 9 was stopped after 6 episodes when rules were found to be paraphrased by Hindsight's
  extraction. The other configs in `experiments/batches/` (exp2–5, exp7, probe-difficulty) were
  earlier pilots and are not summarised above.

See [`docs/study-report.html`](docs/study-report.html) for the full write-up, including harness
fixes made along the way.

## Reproducing

### Requirements

- Linux with systemd and nftables, Python 3.12 and [uv](https://docs.astral.sh/uv/). The
  confirmatory `UserSandbox` runs each command as a throwaway system user through `sudo -n` (no
  password prompt), so the harness user needs passwordless sudo for `useradd`, `nft`,
  `systemd-run`, `install`, `pkill`, `find`, `rm`, `tar` and running commands as the pool users.
- An OpenAI-compatible endpoint (llama.cpp `llama-server`, directly or behind a gateway) serving
  two models under these ids:
  - `study-qwen3-coder-30b-a3b`: `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`, the agent;
  - `study-gemma4-26b-a4b`: `gemma-4-26B_q4_0-it.gguf`, Hindsight's memory LLM.

  Weights hashes, the llama.cpp build and the server flags are pinned in
  `experiments/phase0-local.yaml` (`model.pin`, `memory_service.llm.pin`). Outside the `dev` split the
  controller checks each backend's llama.cpp `/props` (at `props_url`) against the pin, so a
  different build, weights file or context size needs a new registration rather than an edit.
- A dedicated Hindsight 1.1.0 server on `127.0.0.1:8901`: see `services/hindsight/` (`serve.sh`,
  its own `uv.lock`, embedder and reranker pinned in `models.lock.json` and fetched by
  `fetch_models.py`; `install-service.sh` installs a systemd user unit). The unit file assumes the
  checkout is at `~/projects/agent-experience`; adjust `ExecStart` if it is not.

### Environment variables (names only)

| Variable | Used by | Purpose |
|---|---|---|
| `WINTERMUTE_MODEL_API_BASE` | harness, `serve.sh` | base URL of the OpenAI-compatible endpoint (name registered in the config) |
| `WINTERMUTE_MODEL_API_KEY` | harness, `serve.sh` | API key for that endpoint |
| `ELH_MODEL_ENV_FILE` | `serve.sh` | env file that defines the two variables above |
| `ELH_HINDSIGHT_PORT`, `ELH_HINDSIGHT_PG_PORT` | `serve.sh` | Hindsight and embedded PostgreSQL ports (8901, 8902) |
| `ELH_HINDSIGHT_URL` | tests | Hindsight base URL for integration tests |
| `ELH_LOCK_DIR` | harness | per-model-alias lock directory |

Keep keys in a private env file outside the repository; `.env` files are gitignored.

### Commands

```
uv sync
uv run pytest -q                                   # unit tests; some need Hindsight / the sandbox
uv run elh validate-family cpu_mem --sandbox local # generator validation (SPEC §4.4)
uv run elh sandbox-setup                           # once per boot: UserSandbox pool users + nftables rule
uv run elh run-batch experiments/batches/exp13-histories.yaml --dry-run       # print the schedule
uv run elh run-batch experiments/batches/exp13-histories.yaml --client model  # run it
uv run elh analyze exp13-histories                 # comparisons, bootstrap, ΔRR/ΔNL, GPU-seconds
```

Runs write a SQLite database, content-addressed JSONL traces, workspaces and memory snapshots under
`runs/` (gitignored; `--runs-dir` to change). `elh regrade` re-grades a stored episode and compares
verdicts. `--sandbox local` uses the `LocalSandbox` (a temp directory with resource limits, no isolation);
it is for development only, and each episode row records which backend it ran on.

## Repository layout

```
SPEC.md                    design specification: schemas, contracts, decisions, phase plan
experiments/
  phase0-local.yaml        base registration (models, pins, budgets, sandbox, memory service)
  phase0-sonnet.yaml       optional Claude Sonnet comparison configuration (dev only)
  batches/                 one file per experiment (exp1..exp13, probe-difficulty); the record of what was run
elh/
  cli.py                   `elh` entry point
  registry.py              experiment config schema, config hash, prices
  batch.py                 expands a batch file into a schedule of phases × histories × groups
  controller.py            runs episodes, enforces splits, snapshots, pins
  judge.py                 hidden-oracle grading and integrity checks
  ledger.py, store.py      cost ledger; SQLite schema and trace store
  analysis.py              comparisons and continual-learning metrics
  tasks/                   task contracts and the cpu_mem family (causes, oracle runner)
  sandbox/                 UserSandbox (confirmatory), LocalSandbox (dev), PodmanSandbox (unused)
  runner/                  tool-use loop, model clients, frozen prompts
  memory/                  NullMemory, HindsightMemory
  learner/                 rules.py (LotJ), utility.py (MemRL), workflows.py (AWM), gepa_opt.py (GEPA)
services/hindsight/        pinned, dedicated Hindsight server for the experiment
docs/study-report.html     results write-up
docs/papers/               implementation notes per cited paper (they reference local copies of
                           papers and code that are not included here)
tests/
```

## Status and limitations

- Pilot stage: one task family, one or two bug types per experiment, 10–12 test tickets per group
  per history. At a 90% pass rate the 95% interval for 12 tickets spans roughly 60–98%.
- Only experiment 13 has more than one history, and its histories ranged from 3 to 12 of 12 for the
  same group, so single-history results carry similar variation. Its 12 test tickets are shared
  across histories; three histories show that outcomes vary but do not estimate how often an
  incomplete lesson occurs.
- Sealed splits, multiple checkpoints and retention panels are implemented but not run, so H1, H2,
  H3 and H5 are untested.
- Token prices in the ledger are placeholders; cached input (about 97% of input tokens) is priced at
  zero. Measured GPU time puts learning at 80–90% of inference cost, while the ledger says 28%. The
  GPU-seconds figures above are the more reliable cost measure.
- Published effects for these methods are a few points on fresh tasks; the confirmatory study needs
  a power analysis from pilot variance.
- Reproduction is tied to specific hardware and pinned model builds; results on other hardware need
  a new registration and are not directly comparable on timing.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
