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
| 4.3 Full grid | 6 worlds; 4.1's histories reused if nothing changed | see below | see below |

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
than once per history.

| Histories | Episodes | GPU time |
|---|---|---|
| 3 | 3,600 | ≈ 2.5 days |
| 8 | 8,160 | ≈ 5.5 days |

The history count for the confirmatory study (SPEC §14 item 8) is set from 4.1's variance, not
before.

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
