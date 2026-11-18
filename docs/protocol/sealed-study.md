# Sealed confirmatory study: protocol

> **DRAFT for the owner's review. Not frozen, not preregistered.** Drafted 2026-11-18 from the
> repository state at commit 88f7b86. Every item still undecided is marked **OPEN — decision
> pending** and collected in §10. Nothing here is a result; stage 4.1 and exp28 are exploratory
> and are cited only for the design choices they informed. Q14 and Q15 are recorded in
> `docs/design/house-lib-core-grid.md` §10 and SPEC §13; Q3, Q4, Q8 and Q12 are the owner's
> question numbers as given in the drafting brief and are not defined in the repository.

Terse references: SPEC = `SPEC.md`; grid = `docs/design/house-lib-core-grid.md`; family =
`docs/design/house-lib-family.md`; methods = `docs/methods.md`; repro = `docs/reproducing.md`.

## 1. Status and scope

- This document is the protocol that the roadmap (SPEC §13, "Roadmap to the sealed study", step 5;
  Q12) freezes at git tag `prereg-v1`, together with the sealed batch files, their base config and
  the analysis script, before any w2 evaluation episode runs.
- The harness enforces the freeze for files, not for this prose: `elh prereg-lock` hashes the
  batches, base config, analysis scripts, the whole `elh/` tree, `uv.lock`, `pyproject.toml`,
  `experiments/calibration/` and the Hindsight pins; docs and tests are not covered
  (`elh/prereg.py` docstring). Whether this document is nevertheless added to the hashed set:
  **OPEN — decision pending** (see §8).
- Claim made: the H6 primary test of §2, on the w2 worlds, with the Qwen3-Coder agent. Not claimed:
  H1-H5, H7, groups C and D, any other model, any other task family.
- The owner files the public registry entry (OSF or AsPredicted) carrying the tag's commit hash
  (SPEC §13 "Lock"). Which registry and what the entry contains beyond the commit hash: **OPEN —
  decision pending**.
- Stage 4.3 is an estimation run, not confirmatory; it supplies variances, not the claim (SPEC §13;
  grid §10). This protocol cannot be frozen until 4.3 is analysed and the items in §10 are decided.

## 2. Hypotheses

| | Statement | Status in this protocol |
|---|---|---|
| H6 primary | Retrieved memory (B) differs from a static prompt carrying the same rules and records (P-all-ep) in lesson-check pass at L64-c50; smallest effect of interest 10 points (B − P-all-ep) | Confirmatory (SPEC §13 roadmap, "Primary test") |
| H6 secondary | The B − P-all-ep gap varies with L and c: mixed model `pass ~ group × log2(L) × c + (1 \| ticket) + (1 \| history)` (grid §4; SPEC §12) | Tested only if the primary test succeeds (SPEC §13). Exact terms tested and multiplicity handling: **OPEN — decision pending** |
| H7 | Carried memory changes the frozen model's working style (SPEC §12) | Not part of the sealed claim as the roadmap defines it (its primary and secondary are H6 only). Whether H7's repeatability / between-history ICC analysis is preregistered on the sealed data as secondary or exploratory: **OPEN — decision pending**. SPEC §12 says it is answered from "step 4's replicated histories", i.e. 4.3 |

H6's original form (SPEC §12) is "B beats P-all once lessons are numerous or conflict; P-all matches
or beats B at few lessons". The sealed primary is narrower: one world, one contrast. The test is
two-sided (α = 0.05; §4), so a significant P-all-ep advantage is a result, not a failed test. Stage
4.1 had B behind P-all at L16 (grid §8) and exp28 reduced the gap by retuning B's recall (grid §9);
both are exploratory, and tuning B without matching tuning for P-all is a stated deviation (§7).

## 3. Design

### 3.1 Worlds

World seed 2 (w2), six cells L ∈ {4, 16, 64} × c ∈ {0, 0.5}: `house_lib/L{4,16,64}-c{0,50}-w2`.
Freshly generated; no generator change (the world seed is the id's `-w<seed>` part); w1 output is
pinned by a digest in `tests/test_house_lib.py` (grid §10, "Sealed study worlds (w2)"). Templates,
contexts, choice sets, themes and the library API are shared with w1: w2 is a fresh draw from the
same lesson universe, not a new domain.

Generator validation (`validate-family`, 3 seeds, all checks passed; grid §10): L4-c0 314/314,
L4-c50 324/324, L16-c0 1309/1309, L16-c50 1729/1729, L64-c0 7192/7192, L64-c50 8637/8637.

Calibration (exp40: A and P-hand panel; exp41: P-task learnability control; same panel and
thresholds as w1's exp21 / exp24: guessable if A passes > 20%, dropped if P-task passes < 60%;
family §5, §10; verdicts in `experiments/calibration/<world>-w2.json`):

| World | Kept | Dropped (reason) |
|---|---|---|
| L4-c0-w2 | 4/4 | none |
| L4-c50-w2 | 4/4 | none |
| L16-c0-w2 | 15/16 | `feature_flag.payments` (P-task 47% of 17 < 60%) |
| L16-c50-w2 | 16/16 | none |
| L64-c0-w2 | 64/64 | none |
| L64-c50-w2 | 62/64 | `partner_format.elm` (58% of 12), `partner_format.fjord` (50% of 12), both < 60% |

A passed 0 on every lesson of every w2 world in the verdict files (`a_rate`). Dropped lessons stay
in the world (library docs and handbook unchanged) but are never an evaluation ticket's primary
lesson and their checks are left out of every score (grid §3, decision 2026-11-01;
`grid.kept_lessons`, `grid.scored_checks`). The L64-c50 primary world therefore has 62 kept lessons.

Calibration remainder batches `exp40-calib-{L16-c0,L64-c0,L64-c50}-w2-rest-{a,phand}` (6, 22 and
930 episodes) hold the (group, task) pairs with no row in the parent batch (commit ccf58f8); the
verdicts pool parent and remainder (§6).

**Q14: L64 overlap limitation (owner, 2026-11-07/08).** L64-w2 shares all 64 lesson ids with w1
(the pool is exactly 64 lessons); only choices (24-30 of 64 differ), visibility and conflict pairs
are redrawn, with no conflict pair in common. L4 shares no lesson ids with w1, L16 shares 2-4
(grid §10). Stated limitation: the L64 sealed world is not held out at the lesson level, and w1
informed every choice so far. The primary result is also reported on the subset of L64-w2 lessons
whose choice differs from w1 (differs-from-w1 subset analysis). Its test statistic, whether it
reuses §4's estimator on the subset's checks, and what counts as agreement with the full result:
**OPEN — decision pending**. The subset is fixed by the world definitions, not by results, and
excludes dropped lessons.

### 3.2 Groups

Defined in grid §2 (family §6). All share model, sandbox, budgets, tools and judge; each memory
group shows its block after the ticket in the first message (`experience_position: after_ticket`,
family §10, decision 2026-11-01).

| Group | Content | In the sealed study |
|---|---|---|
| A | nothing | Calibration only so far. Whether A and P-hand run on the sealed panel: **OPEN — decision pending** (grid §5 includes them in the full grid; the primary test needs only B and P-all-ep) |
| B | Hindsight memory: verbatim rules + episode records, retrieved per ticket, k14r12 (§3.4) | Primary |
| P-all-ep | all of that history's B rules and all records, every ticket | Primary comparator |
| P-all | the same rules only | Full grid / secondary |
| P-oracle | only rules from training episodes sharing a lesson with the ticket | Full grid / secondary |
| P-hand | world handbook (gold text of every lesson) | Full grid / secondary |

Which groups run in each of the six cells, and so the secondary analysis's exact data: **OPEN —
decision pending**. Extensions (C, D, G-opt, hidden lessons, B-rules, P-wrong, W-hand, few-shot) are
not in the sealed study (SPEC §13, decided before 4.3; they are the Q8 list, see §9).

### 3.3 Panels and seeds

From `grid.grid_tasks` / `grid.batch_docs` (grid §3, §5; SPEC §4.3):

- Training (B only): N = 2L tickets (8, 32, 128), each lesson the primary lesson of 2 tickets
  (`TRAIN_PER_LESSON = 2`), train seeds from 1000, themes shuffled and order shuffled per history.
  Only train episodes write memory.
- Evaluation: 24 tickets per world (`EVAL_TASKS = 24`) on checkpoint seeds from 2000, never
  retained, the same tickets for every group and history (paired by ticket); primary lessons spread
  evenly over kept lessons. Whether the sealed study uses 24 tickets or more: **OPEN — decision
  pending** (set by the power analysis, §5).
- Dev seeds 3000-3999 are never reported (SPEC §4.3). exp28's tuning tickets used seeds 3000+ and
  stay out of every reported comparison (grid §9).
- Sealed seed bases (SPEC §4.3: "kept in a file outside the repository until the sealed test is
  run"): whether the w2 evaluation seeds are fresh bases chosen before the lock, disjoint from w1's
  2000-series, or the same series, and how they are recorded in the registry entry: **OPEN —
  decision pending**. The w2 worlds already differ from w1 in every task's records, library and
  oracle (grid §10).
- Histories: independent learning runs with their own shuffled order and empty banks (methods,
  "Histories"). Count: **OPEN — decision pending** until 4.3 (§5). Frozen snapshots are taken when
  training ends, before the first evaluation episode; bank fingerprint and learner-state hash are
  checked before recall and after grading (methods, "Frozen snapshots").

### 3.4 Recall setting and report

B's recall is `(memory_service.k, rules.recall_k, budget) = (14, 12, 8192)` ("k14r12",
`grid.GRID_RECALL`), fixed across worlds and not scaled with L (grid §9, §10; decision 2026-11-07,
Q4 per the brief). B recalls during training too. At L64 B's block is bounded by its 14 items, so it
stays a few thousand tokens; P-all and P-all-ep are not bounded (§3.6).

Limitation: a fixed 14 items is a smaller share of the bank at L64 than at L16. Per world the
report states how often a rule for the ticket's primary lesson was shown, so a recall drop at L64
is visible rather than tuned away (grid §10 item 3). The sealed study does not re-tune recall;
doing so would be a further stated deviation. Exact form of the per-world primary-lesson-rule-shown
report (denominator, per-history breakdown, whether it enters the secondary model as a covariate):
**OPEN — decision pending**.

Rule-writing settings are the grid's `OVERRIDES` (`rules.context`, `rules.feedback_where`,
`record_format: lesson-v3`, verbatim rule storage, agent search on; `grid.py`); the context and
feedback additions are stated deviations from the Learning on the Job paper (family §10).

### 3.5 Replicates

Q15 (owner, 2026-11-07/08) gave 4.3 replicate evaluation runs at L64-c50-w1 only (B and P-all-ep, 3
runs per history and ticket) to split episode noise from history × ticket variance for the power
analysis (grid §10). Replicates share the bank snapshot and ticket and differ only by sampling; they
are not independent histories (grid §10). Whether the sealed study keeps replicates (R runs per
cell at the primary world; `power_sealed.py` tabulates R ∈ {1, 2, 3}): **OPEN — decision pending**,
to be decided from 4.3's variance split. The test statistic is the same for every R (§4).

### 3.6 Model, registration and context

- Agent: `study-qwen3-coder-30b-a3b` (Qwen3-Coder-30B-A3B-Instruct, Q4_K_M), temperature 0.7, top_p
  0.8 (SPEC §1; `experiments/phase0-local.yaml`). Memory LLM: `study-gemma4-26b-a4b`. Tools `bash` +
  `submit`, identical across groups; budgets 40 turns, 600,000 tokens, 1,800 s wall clock, 120 s per
  bash call (methods, "Protocol"; `phase0-local.yaml` v6; the SPEC §3 example registration shows
  other values, the registration file governs). Qwen3-Coder only; no other model in the sealed study
  (SPEC §13 "Model").
- Frozen by `model.pin` (weights sha256, quantization, llama.cpp build and flags, context size);
  every episode reads `/props` and a mismatch is an infrastructure error; the controller refuses any
  non-dev split until the pin is complete (SPEC §1, §14 items 2-3).
- **Context size: OPEN — decision pending the context increase.** The current pin is 65,536 tokens
  per request. Projections from L16 put P-all-ep at L64 near 52k tokens and P-all near 36k, which
  does not leave room for an episode (grid §9; SPEC §13 4.3 "L64 gate"; family §9 item 4 decided to
  raise the context rather than truncate). The L64 gate (`elh measure-block`, grid §10) measures the
  real block; if it does not fit, a Forgejo infrastructure issue asks for a larger context, and 4.3
  and the sealed study run under the new registration. Pins to fill in:
  - registration name and version: **OPEN — pending the context increase**
  - agent `context_tokens`: **OPEN — pending the context increase**
  - weights sha256, quantization, llama.cpp build and flags (from the host administrator, matched
    against `/props`): **OPEN — pending the context increase**
  - memory LLM pin (its per-request context may also change): **OPEN — pending the context increase**
  - whether the grid's recall budget of 8192 and the `measure-block --reserve` (default 16,384) stay:
    **OPEN — decision pending**
  - Hindsight: dedicated 1.1.0 instance; pins in `services/hindsight/` (`models.lock.json` for the
    embedder and reranker), hashed by the lock.
- Cost ledger: token prices are preregistered shadow prices (SPEC §8); cost per success is not part
  of the sealed primary. Amortisation horizon for any secondary cost report: **OPEN — decision
  pending** (SPEC §8.2 requires `horizon_tasks` to be preregistered for a cost-per-success claim).

### 3.7 Harness and sandbox

`UserSandbox` only for confirmatory data (SPEC §1, §5): throwaway pool user per sandbox, nftables
network block, systemd-scope CPU/memory/task limits, fresh sandbox for the judge; `LocalSandbox`
rows are tagged and refused by analysis. Hindsight work runs strictly between episodes because
grading shares the host. `elh sandbox-setup` must be rerun after a reboot (the nftables table is not
persistent). The judge is the hidden oracle of family §4.3 (`integrity`, `public_tests`, `feature`,
one `lesson:<id>` check per required lesson); the agent's submit message is never used for grading
(SPEC §7). The harness is the `elh` tree at the tag; its hash is in the manifest.

## 4. Primary outcome and test

**Outcome.** Lesson-check pass on kept lessons: each required lesson's `lesson:<id>` check, with
dropped lessons' checks left out (`grid.scored_checks`); the unit of observation is a check (history
h, eval ticket t, check k). Task pass over kept lessons is secondary (grid §4).

**Test** (`analysis/power_sealed.py`, whose docstring calls it "what is preregistered"). Not a
fitted mixed logistic model (slow, convergence failures, no exact small-sample df with ~10
clusters). Instead a linear mixed model on the paired contrast, by crossed-random-effects ANOVA.
Every history runs every ticket in both groups (P-all-ep is built from the same history's B bank),
so (h, t, k) cells pair:

    z_ht = (n_t / mean n) · mean_k (y_B − y_P)     one number per (history, ticket) cell; with R runs,
                                                   the run-averaged outcomes of each group are used
    z_ht = δ + u_h + v_t + e_ht                    histories and tickets crossed, one observation per cell
    δ̂   = mean z                                  = the pooled lesson-check pass difference
    Var(δ̂) = (MS_h + MS_t − MS_e) / (H·T), floored at MS_e / (H·T)
    t test with Satterthwaite df (floored case: df = (H−1)(T−1))

`n_t` is ticket t's number of scored checks, so tickets with more kept-lesson checks weigh more, as
in the pass rate. Reject if the two-sided p < 0.05 (`primary_test`; the same function for every R).

| Item | Value | Source |
|---|---|---|
| Contrast | B − P-all-ep, L64-c50-w2, kept lessons | SPEC §13 |
| Estimand | pooled-over-checks mean difference δ̂ (marginal pass-rate difference) | `power_sealed.py` |
| Effect scale | pass-rate difference; smallest effect of interest 10 points | `EFFECT = 0.10`; SPEC §13 |
| α | 0.05 two-sided | `ALPHA`; SPEC §13 |
| Power target | 0.80 | `TARGET`; SPEC §13 |
| Paired by | ticket, and history (P-all-ep uses that history's B bank) | grid §2 |
| Type-I error (simulated) | ≈ 0.05 for ≥ 6 histories, ≈ 0.07 at 3 histories with many tickets | grid §10 |

The 10 points are fixed, not taken from 4.3's observed effect; 4.3 supplies only the ticket and
history variances (SPEC §13). The confirmatory claim for each outcome (significant in either
direction; non-significant, and whether any equivalence margin is stated), and its wording:
**OPEN — decision pending** (SPEC §12 phrases the refutation as P-all ≥ B; the power analysis is
two-sided).

**Four statistical choices proposed but not yet approved by the owner** (grid §10, "Proposed, pending
the owner's review before preregistration"):

1. Estimand: pooled over checks. **OPEN — decision pending.**
2. Effect scale: the 10 points as a marginal pass-rate difference. **OPEN — decision pending.**
3. Baseline for planning: P-all-ep's 4.3 pass rate at L64-c50, B planned at baseline + 10 points.
   **OPEN — decision pending.**
4. Variances: conservative, i.e. the upper end of the history SDs (e.g. doubled), since 4.3's
   estimates rest on 3 histories. **OPEN — decision pending.**

Handling of tickets with zero kept-lesson checks, missing cells (§6) and unequal runs per cell (the
function assumes a complete history × ticket grid): **OPEN — decision pending**.

## 5. Sample size

Rule (SPEC §13; grid §10, "Power analysis"): fix δ = 10 points, α = 0.05 two-sided, power 0.80; take
variances (ticket, history, ticket × history, episode) from 4.3's L64-c50 data; simulate the logistic
model (random intercepts and B-by-cluster slopes for history, ticket, check and ticket × history
cell, plus an episode effect per run) and apply §4's exact test:

    uv run python analysis/power_sealed.py --from-batches <B batch> <P-all-ep batch> --world L64-c50-w1 \
        --baseline <P-all-ep pass rate to plan for> --sims 5000

It tabulates 3-10 histories × 24 / 36 / 48 tickets × 1 / 2 / 3 runs per cell. Choose the fewest
histories reaching 0.80, cap 10 (≈ 7 days, SPEC §13). If none do at 24 tickets, take the smallest
ticket count that does and say so here. If more than 10 histories would be needed, change the design
(e.g. more eval tickets per world), not the history count. Also run the table at the upper end of
the history SDs (doubled) and plan for that, because 4.3 has 3 histories.

**Warning (demo, not the decision).** On 4.1's L16-c50 data (1 run per cell, 3 histories, 24
tickets, P-all-ep pass 0.43), power at 10 histories for 1 / 2 / 3 runs per cell was 0.46 / 0.55 /
0.56 at 24 tickets and 0.59 / 0.65 / 0.68 at 48 tickets when all cell variance is ticket × history,
and at best 0.76 (48 tickets, 3 runs) when it is all episode noise (grid §10). No design reached
0.80 within 10 histories. The sealed design may therefore need revisiting after 4.3 (more tickets, a
larger effect of interest, or a different primary estimand); the owner deferred that decision
(SPEC §13, "Open").

- Number of histories: **OPEN — decision pending** until 4.3's variances (SPEC §14 item 8).
- Eval tickets per world / at the primary world: **OPEN — decision pending.**
- Runs per cell (§3.5): **OPEN — decision pending.**
- What happens if 0.80 is unreachable within the cap (accept the lower power and state it, enlarge
  the effect of interest, or change the estimand): **OPEN — decision pending.** Changing the effect
  of interest after seeing 4.3 departs from "fixed, not 4.3's observed effect" (SPEC §13) and would be
  recorded as a deviation.
- Interim look or stopping rule on the sealed run: none planned; **OPEN — decision pending** (§6).

## 6. Data quality and exclusions

Rules apply identically to all groups and are fixed before the lock; no outcome-dependent exclusion.

**Valid episode.** A stored episode row with a judge verdict from the confirmatory `UserSandbox`, the
pinned registration (`/props` matched), a constant backend identity across the episode (gateway
deployment id and `system_fingerprint`; a change mid-episode is an infrastructure error, SPEC §1), a
frozen snapshot unchanged before and after grading, and no degenerate reply without a clean retry.
Agent failures (wrong code, timeouts, budget exhaustion, refusals) are valid episodes and count as
failed checks.

**Infrastructure errors.** An `InfraError` stores no episode row (`elh/batch.py`), so the (group,
history, ticket, run) cell is missing; it is filled by a remainder batch, never by dropping only its
partner group (§4's paired design needs both). A cell that stays missing for one group (rerun until
filled, or drop the (history, ticket) cell for both): **OPEN — decision pending**.

**Degenerate replies** (methods, "Degenerate replies and the prompt-cache retry"; decided
2026-11-11 with the server admin). The coder occasionally returns a degenerate reply (e.g. 8,192 `?`
characters) when a reused prompt-cache prefix on the llama.cpp server is corrupted; the same request
is fine with the cache off. The prompt cache stays on for normal requests. On a degenerate reply the
client retries that one request once with `cache_prompt=false` (same messages, tools, sampling and
`max_tokens`); a clean retry is the turn's response, a fresh sample of the same request, not a
replay. A degenerate retry raises `DegenerateOutputError`, an infrastructure error, and the episode
is skipped. Each retry is recorded as `cache_retry` in the trace. After every batch the operator
runs `uv run elh check-degenerate <batch-prefix>...` (`elh/degenerate.py`), which lists stored
episodes holding degenerate replies and the cache-retried turns with their `retry_clean` flags.
The guard predates every sealed episode. Whether episodes with a clean retry stay in the primary
analysis (planned: yes) and whether a sensitivity analysis without them is reported: **OPEN —
decision pending**.

**Remainder batches.** A batch that aborts or loses episodes is completed by a `-rest*` batch holding
exactly the (group, task) pairs with no row in the parent (commit ccf58f8; a test checks each is a
subset of its parent's panel), run under a new name; the analysis pools parent and remainders
(`analysis/exp17_calibration.py` does this for calibration). For sealed batches, whether remainder
files are pre-generated at the lock or added under a documented amendment (a new hashed file needs a
new lock, repro "Protocol lock"): **OPEN — decision pending**.

**Abort rules** (`elh/batch.py`, commit 7d197f0): a `BackendUnreachableError` (`/props` fetch
failure, connection error or timeout, HTTP 5xx after retries) aborts the batch at once;
`MAX_CONSECUTIVE_INFRA_ERRORS` = 3 infra errors in a row abort it; an isolated degenerate reply is
skipped. The abort message prints the number of recorded episodes; the remainder runs after the
backend is fixed.

**Exclusion list** (`experiments/calibration/excluded-episodes.json`, dated 2026-11-11): 17
calibration episodes (10 at L64-c50-w1, 7 at L64-c0-w1) recorded while the server was in a degenerate
state (found by `check-degenerate`); database rows are untouched, `exp17_calibration.py` leaves them
out, and the replacements are the `-rest*` batches named in the file. All 17 are w1 calibration
episodes. The file sits under `experiments/calibration/`, so the lock hashes it. For the sealed run:
the list's format, who may add to it, and whether it must be closed before the sealed analysis
script first runs: **OPEN — decision pending**. Proposed principle (not yet decided): only episodes
flagged mechanically by `check-degenerate` are excludable, never by outcome.

**Further stopping rules for training batches.** Training is strictly sequential within a history and
learning depends on order (SPEC §11). Whether a training episode lost to an infra error is rerun in
order, skipped (a gap in the bank) or the history restarted, and any threshold at which a history
is discarded: **OPEN — decision pending**. Whether an analogue of 4.1's go/no-go infra bound (≤ 2%,
grid §6) binds the sealed study: **OPEN — decision pending**. A history may be discarded only for
infrastructure reasons, never from its evaluation outcomes.

## 7. Deviations already known

| Item | What happened | Treatment | Source |
|---|---|---|---|
| exp28 recall tuning after 4.1 (Q3 per the brief) | 4.1 had B behind P-all at L16 (5 items shown, 2 rules, of 32); recall was tuned on read-only replays of 4.1's banks (12 tuning tickets × 3 histories × 2 worlds, seeds 3000+) to k14r12, with P-all not tuned | Stated deviation: B's recall tuned, P-all's not; exploratory; B's histories rerun with the tuned recall; tuning tickets never enter a reported comparison; 4.3 is an estimation run; the H6 claim is made only in the sealed study | grid §8-§10; SPEC §13 |
| L64-c50 gate crash and exp29b rerun | `exp29-gate-L64-c50-w1` crashed 2026-11-10 after 7 training episodes (rule-writer bug, fixed in 3807223); a crashed run's bank is named after its run id, so the rerun is `exp29b`/`exp30b` (`grid.RERUN`); the 7 episodes stay in the database | Affects 4.3 (w1) only; the fix is in the `elh/` tree the lock hashes | grid §10 |
| Server degenerate-output bursts and exclusions | 17 w1 calibration episodes lost to degenerate replies; excluded and rerun | Mechanical detection (`check-degenerate`), prompt-cache retry from 2026-11-11, exclusion file; so far calibration only | methods; `excluded-episodes.json` |
| Calibration remainders | w2 calibration lost episodes to degenerate replies and the 2026-11-14 outage; completed by `exp40-calib-{L16-c0,L64-c0,L64-c50}-w2-rest-{a,phand}` (6, 22, 930 episodes) | Verdicts pool parent and remainder | commits ccf58f8, 88f7b86 |
| Learnability filter uses P-task, not P-hand | P-hand mixed the handbook's length into learnability and would have dropped lessons where memory could gain | Decision 2026-11-02; P-hand stays in the grid | family §10 |
| Rule writer and feedback differ from the cited paper | `rules.context` keeps ticket context names; `rules.feedback_where` adds the file that defines the convention to failure feedback (never its content) | Decided 2026-11-02 after exp25/26; used by the grid | family §10 |
| Memory placement | block shown after the ticket in `house_lib` (handbook before the ticket was ignored, exp17-19) | Decided 2026-11-01 | family §10 |
| L64 lesson overlap with w1 (Q14) | §3.1 | Limitation + subset analysis | grid §10 |
| Context size | changes from 65,536 (§3.6) | New registration | grid §9, §10 |
| Hindsight `reflect` | not called during episodes (≈ 43 s, 44,000 tokens per call) | Stated in methods | methods |

Stage 4.1 and exp28 results are exploratory; they informed design choices and are not evidence for
H6. A deviation found during 4.3 (for example a further rule-writer or harness fix) is added to
this table before the lock; after the tag, deviations go in a separate dated log reported with the
analysis, because the tagged files cannot change quietly (repro "Protocol lock").

## 8. Lock procedure

Per SPEC §3, §13 and repro "Protocol lock (sealed batches)":

1. Preconditions: 4.3 analysed; power analysis done and history, ticket and run counts decided;
   context increase done and the new registration file written with pins matching `/props`; every
   item in §10 closed; w2 worlds validated and calibrated (done, §3.1).
2. Generate the sealed batch files (`grid.batch_docs` / `grid43_docs` with `sealed: true`) for the
   w2 worlds, and settle the remainder-batch policy of §6. Batch names and sealed seed bases:
   **OPEN — decision pending**.
3. `uv run elh prereg-lock --batches experiments/batches/<sealed>.yaml ... --analysis
   analysis/power_sealed.py analysis/<sealed analysis script>.py` writes
   `experiments/prereg/manifest.json`: sha256 of the batches, their resolved `base_config`, the
   analysis scripts, the whole `elh/` tree, `uv.lock`, `pyproject.toml`, `experiments/calibration/`
   and the `services/hindsight/` pins, one combined digest, and the commit. The sealed analysis
   script (the confirmatory test on episode data, as opposed to the power simulation) is not written
   yet: **OPEN — decision pending** (engineering to do; SPEC §13).
4. `uv run elh prereg-verify` lists mismatched, added or missing files and uncommitted changes.
5. Commit the manifest and this document in its final, non-draft form; the owner tags that commit
   `prereg-v1` by hand (the tool never tags). Whether this document is in the hash set: **OPEN —
   decision pending** (`elh/prereg.py` does not cover `docs/`; adding `docs/protocol/` to the hashed
   trees needs a code change under `elh/` before the lock).
6. The owner files the registry entry (OSF or AsPredicted) with the tag's commit hash.
7. `run-batch` on a sealed batch then runs only if the manifest lists the batch, all hashes match,
   hashed files are committed, tag `prereg-v1` exists, HEAD contains it and the manifest equals the
   one at the tag (`--dry-run` prints the verdict and the schedule). No w2 evaluation episode runs
   before the tag. Changing a hashed file afterwards needs a new lock under a new tag name.
8. After the tag the sealed analysis is run as locked; any deviation is logged (§7).

## 9. Planned follow-ups (not part of the sealed claim)

- **Sonnet 5.5 replication** of the primary test, using the `experiments/phase0-sonnet.yaml`
  configuration (adaptive thinking, `effort: medium`, refusal fallbacks off, refusals reported as
  `outcome=refused`; SPEC §1, §13 "Model", §14 item 6; needs Anthropic credentials). Decided
  2026-11-07 as a follow-up, not part of the sealed study. Whether its design (histories, tickets)
  is stated here or in a later registry amendment: **OPEN — decision pending**.
- **Second, independently written agent harness** (for example a pi-based agent) running the primary
  test as a robustness check against harness-specific effects. It appears only in the drafting
  brief, not in the repository or SPEC. Scope, whether it uses Qwen3-Coder and the w2 worlds, and
  its registry status: **OPEN — decision pending**.
- **Extensions** (Q8 per the brief; the list is in SPEC §13 and grid §2): groups C and D, G-opt,
  hidden lessons, B-rules, P-wrong, W-hand, few-shot (F-fix / F-ret), A0, a mid-episode `recall`
  tool (SPEC §6.2), novel-combination and novel-cause splits and retention (SPEC §4.3), and H1-H5
  (methods). None runs before the sealed study; any later run is exploratory unless separately
  registered.

## 10. Open decisions checklist

All items marked OPEN above, in section order.

- [ ] §1 Whether this document is covered by the lock hash set (and the code change to add `docs/protocol/`).
- [ ] §1 Registry choice (OSF or AsPredicted) and the entry's contents beyond the tag commit.
- [ ] §2 Secondary L × c test: terms tested and multiplicity handling.
- [ ] §2 H7's status on the sealed data (secondary, exploratory, or out of scope).
- [ ] §3.1 Differs-from-w1 subset analysis: statistic and what counts as agreement.
- [ ] §3.2 Whether A and P-hand run on the sealed panel; which groups run in each of the six cells.
- [ ] §3.3 Number of eval tickets per world (24 or more).
- [ ] §3.3 Sealed evaluation seed bases and how they are recorded.
- [ ] §3.3 / §5 Number of histories (after 4.3).
- [ ] §3.4 Form of the per-world primary-lesson-rule-shown report.
- [ ] §3.5 / §5 Whether the sealed study keeps replicates, and R.
- [ ] §3.6 Context size (pending the context increase).
- [ ] §3.6 New registration: name/version, weights sha256, quantization, build and flags, memory-LLM pin (pending the context increase).
- [ ] §3.6 Whether the recall budget (8192) and the measure-block reserve stay.
- [ ] §3.6 Amortisation horizon for any secondary cost report.
- [ ] §4 Pooled-over-checks estimand (not yet approved).
- [ ] §4 Marginal 10-point effect scale (not yet approved).
- [ ] §4 P-all-ep's 4.3 rate as the planning baseline (not yet approved).
- [ ] §4 Conservative (upper-end) history variances (not yet approved).
- [ ] §4 Wording of the conclusion for each outcome; any equivalence margin.
- [ ] §4 Handling of zero-check tickets, missing cells, unequal runs per cell.
- [ ] §5 Eval tickets per world and runs per cell from the power analysis.
- [ ] §5 What to do if 0.80 is unreachable within 10 histories.
- [ ] §5 / §6 Any interim look or stopping rule on the sealed run.
- [ ] §6 A cell missing for one group after reruns: fill or drop for both.
- [ ] §6 Sensitivity analysis without cache-retried episodes.
- [ ] §6 Remainder batches: pre-generated at the lock or added by amendment.
- [ ] §6 The sealed exclusion list: format, who may add, closure before the first analysis run.
- [ ] §6 Training-batch stopping rules: infra error in training, history-discard threshold, analogue of the 2% bound.
- [ ] §8 Sealed batch names, and the sealed analysis script (to be written).
- [ ] §9 Sonnet 5.5 replication design.
- [ ] §9 Second-harness follow-up scope.
