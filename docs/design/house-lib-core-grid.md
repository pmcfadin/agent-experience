# Plan: `house_lib` core grid (SPEC §13 step 4)

Status: reviewed 2026-11-01 (decisions in §7). Nothing here has run. Builds on
`docs/design/house-lib-family.md` §6-8 and the calibration results in its §10.

## 1. What the grid answers

H6: does retrieved memory (B) beat a static prompt carrying the same content once there are many
lessons, or lessons that conflict across contexts? The design predicts the B − P-all gap grows with
the number of lessons `L` and with the conflict rate `c`, and that P-all matches B at small `L`.

## 2. Groups

| Group | What the agent gets | Built from | Varies by history |
|---|---|---|---|
| A | nothing | — | no |
| B | Hindsight memory: verbatim rules + episode records, retrieved per ticket | its own training | yes |
| P-all | all of B's rules, frozen, every ticket | that history's B bank after training | yes |
| P-all-ep | all of B's rules and all records, every ticket | same | yes |
| P-oracle | only B's rules from training episodes that shared a lesson with this ticket | same, plus `signature.lessons` | yes |
| P-hand | the world handbook (our gold text for every lesson) | the world | no |

B vs P-all-ep is the cleanest H6 test: same content, the only difference is selection per ticket.
P-all asks the same with rules only. P-oracle is the best selection could do; P-hand the best any
static prompt could do. Every memory group shows its block after the ticket (design §10).

Not in the core grid: A0 (cpu_mem already showed A0 ≈ A), B-rules, P-wrong, W-hand, few-shot,
G-opt, C, D (design §8 extensions). G-opt and D still need house_lib prompts.

## 3. Worlds and tasks

- Worlds: L ∈ {4, 16, 64} × c ∈ {0, 0.5}, world seed 1, `source` and `doc` lessons only.
- Training (B only): N = 2L tickets (8, 32, 128), each lesson the primary lesson of 2 tickets, so
  about 3 exposures per lesson counting secondary and module-wide ones. Train seeds, shuffled per
  history.
- Evaluation: 24 tickets per world on checkpoint seeds (never retained), the same 24 for every group
  and history, so comparisons are paired by ticket. Primary lessons spread evenly over kept lessons.
- **Dropped lessons (Decision, 2026-11-01)** (calibration filter): keep them in the world (the library docs and
  the handbook are unchanged) but never as an evaluation ticket's primary lesson, and leave their
  checks out of every score. Regenerating a world without them would change the library and need
  recalibration.

## 4. Outcomes and analysis

- Primary: per-lesson check pass on kept lessons (more observations than task pass).
- Secondary: task pass over kept lessons; cost per success; retrieval recall (did B's block contain
  a rule for each required lesson); P-all pass by rule position (long-prompt effect).
- Model: mixed-effects logistic, `pass ~ group × log2(L) × c + (1 | ticket) + (1 | history)`, plus
  the per-world McNemar and bootstrap comparisons already in `analysis/`.

## 5. Stages and cost

Measured on house_lib so far: A ≈ 35 s and P-hand ≈ 45 s per episode, including grading. B training
episodes also retain and consolidate memory; not yet measured on house_lib, assumed ≈ 90 s (cpu_mem:
≈ 120 s with slower tasks).

| Stage | Content | Episodes | GPU time |
|---|---|---|---|
| 4.0 Prerequisites | P-oracle group; dropped-lesson handling; train/eval panel generator; B smoke run (≈ 10 train + 4 eval on L16-c50) | ≈ 14 | ≈ 20 min |
| 4.1 Go / no-go | L16-c0 and L16-c50 (after their exp21 recalibration), 3 histories, all six groups | 192 train + 864 eval = 1,056 | ≈ 15 h |
| 4.2 Calibrate | exp21 (A, P-hand) and exp24 (P-task, the learnability control), 8 tasks per lesson: L4 ×2 and L16 ×2 (before 4.1), then L64 ×2 (before 4.3) | 960 + 3,072 | ≈ 12 h + ≈ 36 h |
| 4.3 Full grid | 6 worlds; B's histories rerun with k14r12 (§9, §10); A and P-hand reused at L16 | 3,312 (§5 below, §10) | ≈ 2.3 days |

Stage 4.0 status (2026-11-01): P-oracle (`OracleMemory`, `elh/batch.py oracle_rules`), kept-lesson
scoring and the panel and batch generator (`elh/tasks/families/house_lib/grid.py`) are built and
tested; each world's verdict is written by `analysis/exp17_calibration.py --write` to
`experiments/calibration/<world>.json`, from which `grid.batch_docs` makes `exp22-grid-<world>` (A, B,
P-hand; B trains) and `exp23-grid-static-<world>` (P-all, P-all-ep, P-oracle from exp22's B). Left: the
B smoke runs (exp25-27, house-lib-family.md §10: they led to `rules.context` and
`rules.feedback_where`, now in `grid.OVERRIDES`). Stage 4.1 started 2026-11-02 with both L16 worlds keeping
16 of 16 lessons (exp24); the go / no-go analysis is `analysis/exp22_grid.py`.

Full grid per history: 336 train + 576 eval (B, P-all, P-all-ep, P-oracle × 6 worlds × 24).
A and P-hand do not depend on the history, so they run as 3 replicates (864 episodes in all) rather
than once per history. Exact counts for the batches of §10 (3 histories, 24 evaluation tickets):

| World | Learn (A, B, P-hand) | Static (P-all, P-all-ep, P-oracle) | Total |
|---|---|---|---|
| L4 (each of 2) | 24 train + 216 eval = 240 | 216 | 456 |
| L16 (each of 2) | 96 train + 72 eval (B only; 4.1's A and P-hand reused) = 168 | 216 | 384 |
| L64 (each of 2) | 384 train + 216 eval = 600 | 216 | 816 |
| All six | | | **3,312** (3,600 without the L16 reuse) |

GPU time ≈ 2.3 days (the former ≈ 2.5 days scaled by 3,312 / 3,600). Counts for another history count
come from `grid.grid43_docs(..., histories)`.

The history count for the confirmatory study (SPEC §14 item 8) is set by the power analysis in
§10, with variances from 4.3.

## 6. Go / no-go rule

Decided before 4.1 runs, and not on the direction of B vs P-all (that would bias the study toward
whichever result we hoped for). Go if:

1. no more than 2% infra errors;
2. B learns: B's lesson-check pass on L16 is clearly above A's (A is near 0%);
3. P-all and P-all-ep fit in the context at L16, and a dry build of B's block at L64 projects under
   the pinned 65,536-token context with room for an episode (else raise the context, design §9.4).

No-go means fixing the pipeline, not changing the hypothesis.

## 7. Decisions

1. ~~Go / no-go worlds~~: decided 2026-11-01, the two calibrated L16 worlds (not design §8's
   L ∈ {4, 16} at c = 0.5: L4-c50 has a single conflict pair).
2. ~~Dropped lessons~~: decided 2026-11-01, kept in the world and left out of scores (§3).
3. ~~P-oracle in 4.1~~: decided 2026-11-01, yes (144 of 4.1's 864 eval episodes, ≈ 2 h).
4. ~~Histories in 4.1~~: decided 2026-11-01, 3.

## 8. Stage 4.1 results (2026-11-03; analysis/exp22_grid.py, runs/logs/exp22-analysis.md)

1,056 episodes, 0 infra errors. **Go** on all three criteria: infra 0%; B above A (L16-c0 lesson
checks 47/135 vs 0/135, L16-c50 39/141 vs 1/141, McNemar p < 0.001); largest block ≈ 13k tokens
(P-all-ep, L16-c50), well under the 65,536-token context. Lesson checks (task pass):

| World | A | B | P-all | P-all-ep | P-oracle | P-hand |
|---|---|---|---|---|---|---|
| L16-c0 | 0/135 (0/72) | 47/135 (11/72) | 95/135 (42/72) | 98/135 (39/72) | 87/135 (38/72) | 97/135 (38/72) |
| L16-c50 | 1/141 (1/72) | 39/141 (14/72) | 58/141 (23/72) | 61/141 (23/72) | 64/141 (27/72) | 109/141 (45/72) |

- At L = 16 B loses to the same content shown whole (B vs P-all lesson checks p < 0.001 at c = 0,
  p = 0.011 at c = 0.5); the gap narrows with conflicts (48 vs 19 checks), the predicted direction
  but not a reversal. P-all-ep ≈ P-all (records add nothing). P-hand stays far ahead at c = 0.5.
- Retrieval is B's bottleneck: B shows 5 items, 2 of them rules, out of 32; only 23/72 (c0) and
  26/72 (c50) eval tickets were shown a rule from a training ticket with their primary lesson.
- `event_fields` at c = 0.5: 0/12 for every memory-derived group (B's rules never captured the key
  recipe), 12/12 with the handbook. `feature_flag`: the handbook is weakest (3/24, 7/27).

## 9. Recall tuning, exp28 (2026-11-07; analysis/exp28_tune.py, runs/logs/exp28-analysis.md)

B's exp22 banks replayed read-only (`B-replay`, after 32 training tickets) on 12 tuning tickets per
history (seeds 3000+, never in a reported panel), 3 histories, under (`memory_service.k`,
`rules.recall_k`, recall budget), with P-all on the same tickets. 288 episodes, 0 infra errors. The
first k8r6 L16-c0 run was cut off by a host reboot after 15 episodes; they are kept under
`exp28-tune-k8r6-L16-c0-w1-aborted` and the batch was rerun in full.

| World | Setting | Task pass | Lesson checks | Primary-lesson rule shown | Lesson checks vs P-all (McNemar) |
|---|---|---|---|---|---|
| L16-c0 | k5r2 (4.1) | 6/36 | 13/54 | 6/36 | 3 vs 27, p < 0.001 |
| L16-c0 | k8r6 | 13/36 | 24/54 | 18/36 | 8 vs 21, p = 0.024 |
| L16-c0 | k14r12 | 15/36 | 28/54 | 26/36 | 6 vs 15, p = 0.078 |
| L16-c0 | P-all | 20/36 | 37/54 | - | - |
| L16-c50 | k5r2 (4.1) | 3/36 | 19/78 | 16/36 | 9 vs 22, p = 0.029 |
| L16-c50 | k8r6 | 8/36 | 36/78 | 25/36 | 15 vs 11, p = 0.56 |
| L16-c50 | k14r12 | 6/36 | 31/78 | 30/36 | 9 vs 10, p = 1.0 |
| L16-c50 | P-all | 9/36 | 32/78 | - | - |

- Retrieval was the bottleneck §8 named: showing more rules raises how often the ticket's primary
  lesson's rule is shown, and lesson checks follow. At c = 0 the gap to P-all shrinks from 24 to 9
  checks; at c = 0.5 B matches P-all.
- k8r6 and k14r12 are within noise of each other at c = 0.5; k14r12 is ahead at c = 0 and shows the
  primary lesson's rule most often.
- Blocks (≈ 4 characters per token, largest / mean): k14r12 4.5k / 3.4k at L16-c50, P-all 9.0k / 8.3k.
  B's block is bounded by its 14 items, not by the bank, so at L64 it projects to the same few
  thousand tokens: criterion 3 of §6 holds for B. P-all is not bounded: ≈ 4× the rules at L64
  projects to ≈ 36k tokens for P-all and ≈ 52k for P-all-ep (13k at L16 in 4.1), which leaves too
  little of the 65,536-token context for an episode. To settle before 4.3 (design §9.4).

Decision (2026-11-07): the full grid runs B with (k, recall_k, budget) = (14, 12, 8192),
`grid.GRID_RECALL`, via `grid.batch_docs(..., recall=GRID_RECALL)`. B recalls during training as
well, so 4.1's B histories are not reused in 4.3; A and P-hand do not depend on recall. The tuning
tickets were chosen for this decision only and stay out of every reported comparison.

## 10. Next steps (decided 2026-11-07; roadmap in SPEC §13 "Roadmap to the sealed study")

**4.3 is an estimation run, not confirmatory.** It gives effect sizes and variances, not the H6
claim. B's recall was tuned (exp28) after 4.1 showed B behind P-all, with no matching tuning for
P-all; this is a stated deviation in the methods. The H6 claim is made only in the sealed study.

1. **L64 calibration first** (exp21 + exp24 at L64-c0 and L64-c50, ≈ 3,072 episodes, started
   2026-11-07, finishing ≈ 2026-11-09), then `analysis/exp17_calibration.py --write`.
2. **L64 gate.** Train L64 history 0 for B and measure the real P-all-ep block. If it plus an
   episode does not fit the 65,536-token pinned context (projections from L16: ≈ 36k P-all, ≈ 52k
   P-all-ep, §9), open a Forgejo infrastructure issue for a larger context (new registration and pins
   from the host administrator; SPEC §14 item 7 already decided to raise the context rather than cap
   P-all). The rest of 4.3 waits, and all of 4.3 runs under the new registration.
3. **Full grid.** 6 worlds (L ∈ {4, 16, 64} × c ∈ {0, 0.5}, w1), 3 histories, B with recall k14r12
   (`grid.GRID_RECALL` = (14, 12, 8192)) fixed across every world, not scaled with L; this is a stated
   limitation. Report per world how often a rule for the ticket's primary lesson was shown, so a
   recall drop at L64 is visible rather than tuned away.
4. **Reuse at L16.** 4.1's A and P-hand (`exp22-grid-*`) are reused: they do not depend on recall,
   and the eval tickets and registration are the same. B is rerun, and so are the groups built from
   its banks (P-all, P-all-ep, P-oracle), since B recalls during training. If the context changes,
   everything reruns.
5. **No extensions** (C, D, G-opt, hidden, B-rules, P-wrong, W-hand, few-shot) before the sealed
   study. Decided before 4.3 so the choice cannot depend on its results.

Sealed study (SPEC §13): primary test B vs P-all-ep at L64-c50 (lesson-check pass on kept lessons,
paired by ticket); the §4 mixed model on the B − P-all-ep gap across L × c is secondary, tested only
if the primary succeeds. Power: smallest effect of interest 10 points B − P-all-ep (fixed, not 4.3's
observed effect), α = 0.05 two-sided, power 0.80, by simulation from the §4 model with 4.3's ticket
and history variances; cap 10 histories, and if more are needed, add eval tickets per world rather
than histories. New worlds use seed 2 (w2), generated, validated and calibrated while 4.3 is
analysed.

### 4.3 batches (generated 2026-11-07; `grid.grid43_docs`, `grid.write_43`)

`uv run python -m elh.tasks.families.house_lib.grid` writes the batch files of every world that has a
calibration verdict and names the worlds that do not (L64 until `exp17_calibration.py --write` has
run; a test fails if a calibrated world lacks its files). Learn batches `exp29-*` (B, with A and
P-hand except at L16, recall `GRID_RECALL`); static batches `exp30-*` (P-all, P-all-ep, P-oracle from
the `exp29` batch's B after the whole training panel). Same panels and seeds as 4.1.

| World | Batches, in run order | Episodes |
|---|---|---|
| L4-c0, L4-c50 | `exp29-grid-<w>`, `exp30-grid-static-<w>` | 240 + 216 each |
| L16-c0, L16-c50 | `exp29-grid-<w>` (B only), `exp30-grid-static-<w>` | 168 + 216 each |
| L64-c0, L64-c50 | `exp29-gate-<w>`, `exp30-gate-static-<w>`, `exp29-grid-<w>`, `exp30-grid-static-<w>` | 200 + 72 + 400 + 144 each |

Total 3,312 episodes (counts and the L16 saving in §5).

**L64 gate procedure.** A batch refuses to start if its run already has episodes (no resume), and a
bank belongs to one batch's run, so history 0 cannot be trained once and then extended by a later
batch. Instead the L64 grid is split by history, with a new `history_start` field (a batch runs
histories `[history_start, history_start + histories)`, with the seeds, shuffles and checkpoints of
one batch of all three; tested). Nothing run in the gate is repeated:

1. Run `exp29-gate-L64-c50-w1` (history 0: B trains 128 tickets, then A, B and P-hand evaluate 24
   tickets). c50 is the sealed study's primary world and has the most to remember; the c0 gate is the
   same batch for the other world and follows it. The bank is snapshotted when training ends, before
   the first evaluation episode.
2. As soon as the first evaluation episode appears (≈ 128 × 90 s ≈ 3.2 h in), run
   `uv run elh measure-block experiments/batches/exp30-gate-static-L64-c50-w1.yaml`. It builds P-all,
   P-all-ep and P-oracle from that bank with no episode and prints their blocks and whole first
   message (system prompt, tools, ticket) in tokens (4 characters each, as in `analysis/exp22_grid.py`)
   against the pinned 65,536 with `--reserve` (default 16,384) kept for the episode; it exits non-zero
   if P-all-ep does not fit. It reads the main `runs/` through `--runs-dir`.
3. If it fits, let the gate finish, run `exp30-gate-static-L64-c50-w1`, then the same two for c0 and
   `exp29-grid-<w>`, `exp30-grid-static-<w>` (histories 1 and 2). If it does not, interrupt the batch
   (the evaluation episodes are not needed; everything reruns under a new registration) and open
   the Forgejo issue for a larger context (item 2).

Engineering for 4.3: analysis combining 4.1's A and P-hand (L16, `exp22-grid-*`) with 4.3's groups
(extends `analysis/exp22_grid.py`); it must pool a world's gate and grid batches by history.
