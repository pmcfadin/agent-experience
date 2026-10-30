# H6 step 2: all-rules prompt against retrieved memory (exp14)

Date: 2026-10-29. Experiment: [`exp14-static-prompt`](../experiments/batches/exp14-static-prompt.yaml).
Script: [`analysis/exp14_static.py`](../analysis/exp14_static.py) (stdlib only). Plan: `SPEC.md` §13, step 2.

## Result: memory beat the all-rules prompt (a tie was expected)

Each history's 15 B rules, all placed in the prompt (P-all), passed 18 of 36 test tickets. Retrieved
memory (B) passed 26 of 36 on the same tickets: 9 tickets passed only with B and 1 only with P-all
(exact paired test, p = 0.021). P-all still clearly beat no memory (18 against 6, p = 0.004) and
its own empty-block control (A0, 18 against 8, p = 0.041).

Memory-group instructions alone do nothing: A0 (B's system prompt and search tool, empty memory)
passed 8 of 36 against A's 6 (p = 0.75). This also removes step 1's concern that A's slightly
different system prompt explains the memory groups' gains in pass rate.

| Group | h0 | h1 | h2 | Total | Mean turns | Mean wall (s) |
|---|---|---|---|---|---|---|
| A (exp13) | 2 | 3 | 1 | 6/36 | 34.2 | 168 |
| A0 | 4 | 2 | 2 | 8/36 | 35.1 | 148 |
| P-all | 9 | 9 | 0 | 18/36 | 27.7 | 89 |
| B (exp13) | 11 | 12 | 3 | 26/36 | 28.8 | 99 |

Failures by check: P-all failed `performance` 15 times and `integrity` (changed a protected file) 7
times; B failed them 9 and 2 times. Every history's rules include an AVOID rule against changing
`examples/`, yet P-all changed `examples/sample_output.json` more often than B. A0 ran out of budget
10 times, P-all once.

## Why this is not yet "retrieval beats a prompt"

1. **Content differs, not only selection.** At episode start B was shown 5 items, 2 rules and 3
   episode records (72 rules among 180 items over 36 episodes), and its 29 memory searches returned
   both kinds. P-all held only rules. B's advantage may come from the episode records, which show a
   concrete past fix, rather than from choosing which rules to show.
2. **Not run together.** exp13 ran on 2026-10-26, exp14 on 2026-10-29, not interleaved. The pinned
   model, build and settings are identical (the stored `/props` differ only in a per-start random
   marker), and A0 ≈ A suggests no drift, but drift is not excluded.
3. **Small and uneven.** Three histories, 12 tickets each; history 2 (whose rules missed the second
   slow lookup) contributes 3 of the 8-ticket gap.

## What it means for the plan

The cheapest way to settle point 1 is a **B-rules** group: B's retrieval restricted to rules (rule
slots fill all 5 places, searches return only rules), from the same exp13 snapshots, 36 episodes
(about 1 GPU hour). If B-rules ≈ B, selection is what helps and H6 is supported earlier than
expected; if B-rules ≈ P-all, the gain is the episode records, and `house_lib` needs a P-all
variant that includes them.

## Generated output

### Pass counts per history (of 12)

| Group | h0 | h1 | h2 | Total | Mean turns | Mean wall (s) | Mean turns, passed |
|---|---|---|---|---|---|---|---|
| A | 2/12 | 3/12 | 1/12 | 6/36 | 34.2 | 168 | 33.3 |
| A0 | 4/12 | 2/12 | 2/12 | 8/36 | 35.1 | 148 | 36.2 |
| B | 11/12 | 12/12 | 3/12 | 26/36 | 28.8 | 99 | 28.3 |
| C | 11/12 | 10/12 | 2/12 | 23/36 | 27.8 | 97 | 28.7 |
| P-all | 9/12 | 9/12 | 0/12 | 18/36 | 27.7 | 89 | 27.2 |

### Paired comparisons (same history, same ticket)

| Comparison | Pairs | Only first passed | Only second passed | p (exact, two-sided) |
|---|---|---|---|---|
| P-all vs B | 36 | 1 | 9 | 0.021 |
| P-all vs A0 | 36 | 15 | 5 | 0.041 |
| A0 vs A | 36 | 6 | 4 | 0.754 |
| B vs A | 36 | 21 | 1 | 0.000 |
| P-all vs A | 36 | 14 | 2 | 0.004 |
