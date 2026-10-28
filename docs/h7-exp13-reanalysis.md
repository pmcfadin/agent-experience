# H7 step 1: exp13 trace reanalysis

Date: 2026-10-28. Script: [`analysis/exp13_traits.py`](../analysis/exp13_traits.py) (stdlib only;
re-run with `python3 analysis/exp13_traits.py`). Plan: `SPEC.md` §13, step 1.

## Result: no-go for history-specific personality

The question was whether exp13's three memory histories developed distinct working styles, beyond
the run-to-run noise seen in group A (no memory). They did not, on any test:

- **Per feature:** after removing ticket and pass/fail effects, 0 of 13 features differed by history
  in B, 1 of 13 in C (`reads_before_edit`, p = 0.014 uncorrected) and 0 of 13 in A. With 39 tests,
  about 2 would pass p < 0.05 by chance.
- **Attribution:** a held-out episode could not be matched to its history above chance (B 0.25, C 0.44,
  A 0.36; chance 0.33; all permutation p > 0.2).
- **Shared training order:** B and C from the same history did not look alike (mean profile
  correlation −0.27, against −0.01 across histories).

By the exit criterion in `SPEC.md` §13, H7 as "lineage-specific personality" is not supported here.
History 2's poor results are a missing lesson (see the README), not a different working style.

## What memory did change: style at group level

Memory shifted how *every* history worked, in the same direction, including within passed and within
failed episodes (test tickets, means):

| Feature | A pass | B pass | C pass | A fail | B fail | C fail |
|---|---|---|---|---|---|---|
| Source edits | 3.2 | 1.9 | 1.2 | 4.4 | 1.0 | 1.4 |
| Characters edited | 1,468 | 843 | 779 | 962 | 710 | 885 |
| Verifications | 8.3 | 5.5 | 6.0 | 6.3 | 3.8 | 5.1 |
| Turns | 33.3 | 28.3 | 28.7 | 34.4 | 30.0 | 26.4 |
| (episodes) | 6 | 26 | 23 | 30 | 10 | 13 |

With memory, the agent makes one or two compact rewrites instead of several incremental edits, and
verifies less. That is a memory-caused change of behaviour shared by all histories: closer to
"memory teaches a procedure" than to individuality. It is descriptive only (A has 6 passed episodes).

## Limits

- Three histories per group and one run per ticket per snapshot: only large history effects could
  be detected. A null here does not rule out moderate ones.
- One bug type, one theme, one checkpoint; tickets with a single correct fix leave little room for
  style.
- Features come from regular expressions over bash commands; they were spot-checked on sample
  traces, not validated against human ratings.
- A's and the memory groups' system prompts differ (the `search_memory` tool), so group-level
  differences are memory plus scaffolding; the planned A0 control separates them.

## Options for H7

1. **Reframe** H7 as "memory changes working style" (a group-level effect), tested in `house_lib` with
   A0 and the placebo bank, and drop the individuality claim.
2. **Keep** individuality as a secondary question, answered only by the replicated-history design
   (8–10 histories, repeat runs) and tickets with room for style, at no extra cost because step 4
   collects those data anyway.
3. **Drop** H7.

**Decision (2026-10-28): 1 and 2.** H7's primary claim is now the group-level style change; history
individuality is a secondary question answered from step 4's replicated histories. See `SPEC.md` §12.

## Generated output

Episodes: 198. Features are residualised on ticket and pass/fail before every test. p-values come from 5000 within-ticket permutations of history labels.

### Pass rate by group and history (test tickets)

| Group | h0 | h1 | h2 |
|---|---|---|---|
| A | 2/12 | 3/12 | 1/12 |
| B | 11/12 | 12/12 | 3/12 |
| C | 11/12 | 10/12 | 2/12 |

### 1. Per-feature history effect (test tickets, 36 episodes per group)

F = between-history variance over within-history variance after adjustment; p = permutation.

| Feature | Trait | A: F (p) | B: F (p) | C: F (p) |
|---|---|---|---|---|
| turns | persistence | 0.57 (0.673) | 0.78 (0.608) | 2.83 (0.174) |
| budget_exhausted | persistence | 3.00 (0.153) | 1.50 (1.000) | 0.00 (1.000) |
| cmds_before_edit | exploration | 1.30 (0.431) | 0.54 (0.710) | 4.85 (0.054) |
| reads_before_edit | exploration | 2.58 (0.206) | 1.42 (0.401) | 8.01 (0.014) |
| n_edits | boldness | 0.80 (0.671) | 1.67 (0.335) | 1.98 (0.270) |
| edit_chars | boldness | 2.91 (0.172) | 1.16 (0.540) | 2.49 (0.198) |
| protected_writes | boldness | 0.00 (1.000) | 1.50 (1.000) | 0.00 (1.000) |
| verify_count | caution | 1.52 (0.360) | 0.20 (0.872) | 0.78 (0.595) |
| verified_at_end | caution | 0.81 (0.488) | 0.91 (0.538) | 0.72 (1.000) |
| perf_checks | caution | 1.57 (0.388) | 0.29 (0.829) | 1.55 (0.369) |
| error_rate | caution | 1.57 (0.365) | 2.71 (0.184) | 0.25 (0.855) |
| text_per_turn | style | 1.69 (0.348) | 0.64 (0.690) | 0.39 (0.773) |
| submit_len | style | 1.43 (0.400) | 2.34 (0.209) | 0.10 (0.938) |

- Group A: 0 of 13 features at p < 0.05 (about 0.7 expected by chance).
- Group B: 0 of 13 features at p < 0.05 (about 0.7 expected by chance).
- Group C: 1 of 13 features at p < 0.05 (about 0.7 expected by chance).

### 2. Can an episode be attributed to its history? (test tickets)

Leave-one-ticket-out nearest centroid on all standardised residual features. Chance is 1/3.

| Group | Accuracy | Permutation p |
|---|---|---|
| A | 0.36 | 0.463 (n=200) |
| B | 0.25 | 0.796 (n=200) |
| C | 0.44 | 0.209 (n=200) |

### 3. Do B and C that shared a training order look alike? (test tickets)

Mean profile correlation, same history: -0.27 (0.12, -0.24, -0.67); different histories: -0.01.

### Raw feature means by group and history (test tickets)

| Feature | A0 | A1 | A2 | B0 | B1 | B2 | C0 | C1 | C2 |
|---|---|---|---|---|---|---|---|---|---|
| turns | 34.58 | 34.75 | 33.33 | 28.00 | 27.75 | 30.58 | 26.75 | 29.42 | 27.33 |
| budget_exhausted | 0.25 | 0.00 | 0.33 | 0.00 | 0.00 | 0.08 | 0.00 | 0.00 | 0.00 |
| cmds_before_edit | 13.17 | 14.75 | 14.50 | 13.33 | 14.08 | 12.08 | 12.50 | 15.08 | 10.58 |
| reads_before_edit | 7.67 | 8.75 | 8.17 | 7.50 | 8.17 | 7.25 | 6.75 | 8.50 | 6.58 |
| n_edits | 5.08 | 3.33 | 4.25 | 1.00 | 1.17 | 2.75 | 1.00 | 1.08 | 1.75 |
| edit_chars | 1011.75 | 1196.25 | 930.75 | 767.67 | 824.58 | 825.08 | 693.08 | 710.67 | 1048.08 |
| protected_writes | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | 0.00 | 0.00 | 0.00 |
| verify_count | 6.42 | 7.67 | 5.92 | 5.75 | 5.67 | 3.75 | 5.50 | 6.00 | 5.42 |
| verified_at_end | 0.75 | 0.92 | 0.83 | 0.92 | 0.83 | 0.42 | 0.92 | 0.92 | 1.00 |
| perf_checks | 5.33 | 6.58 | 7.25 | 4.92 | 4.67 | 6.08 | 5.33 | 6.67 | 5.42 |
| error_rate | 0.11 | 0.15 | 0.12 | 0.10 | 0.13 | 0.11 | 0.12 | 0.11 | 0.10 |
| text_per_turn | 109.56 | 130.75 | 114.51 | 108.04 | 121.95 | 125.71 | 102.05 | 102.91 | 111.79 |
| submit_len | 490.42 | 560.08 | 405.00 | 592.42 | 681.83 | 520.67 | 625.67 | 610.25 | 584.25 |

Feature definitions: `turns`: model turns used; `budget_exhausted`: ran out of turns (0/1); `cmds_before_edit`: commands before the first source edit; `reads_before_edit`: read-only commands before the first source edit; `n_edits`: source-editing commands; `edit_chars`: characters in source-editing commands; `protected_writes`: write commands touching protected paths; `verify_count`: test runs and output comparisons; `verified_at_end`: a verification in the last 5 commands (0/1); `perf_checks`: load generation or timing commands; `error_rate`: share of commands with a non-zero exit; `text_per_turn`: assistant text characters per turn; `submit_len`: characters in the submit summary.

