# Results

Pilot results and limitations, moved from the README unchanged. The full write-up is
[`study-report.html`](study-report.html).

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

See [`docs/study-report.html`](study-report.html) for the full write-up, including harness
fixes made along the way.

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
