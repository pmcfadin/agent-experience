# H6 step 2: all-rules prompt against retrieved memory (exp14, exp15, exp16)

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

## Follow-up: rules-only retrieval (exp15, 2026-10-30)

B-rules used each history's own B snapshot, with every start slot and search hit a rule (no episode
records): [`exp15-rules-only`](../experiments/batches/exp15-rules-only.yaml), 36 episodes, no infra
errors. It passed 21 of 36 (10, 10, 1 per history), between P-all (18) and B (26), and neither
difference is significant: 2 tickets only B-rules passed against 7 only B passed (p = 0.18), and 6
against 3 against P-all (p = 0.51).

So of B's 8-ticket lead over the all-rules prompt, about 5 tickets go with the episode records and
about 3 with selecting among rules, and this design cannot separate either from noise. Three
histories and 12 tickets are too few. The question moves to `house_lib`, where the grid has more
tickets and histories and where selection should matter more (many lessons, some conflicting); its
P-all group needs a variant that also carries episode records (P-all+ep) so that content and
selection are tested separately.

## Follow-up: all rules plus all records, no selection (exp16, 2026-10-31)

P-all-ep showed each history's whole retrievable B pool in the prompt: the 15 rules, each followed by
the experience facts Hindsight had extracted from that episode's record (49 items, ~5k tokens), with
no selection: [`exp16-all-rules-and-records`](../experiments/batches/exp16-all-rules-and-records.yaml),
36 episodes, no infra errors. It passed **26 of 36 (12, 10, 4 per history), the same as B**: 5
tickets only P-all-ep passed and 5 only B passed (p = 1.0). Against the rules-only prompt P-all it
gained 10 tickets and lost 2 (p = 0.039).

| Group | h0 | h1 | h2 | Total | Failed `performance` | Failed `integrity` |
|---|---|---|---|---|---|---|
| P-all (rules only) | 9 | 9 | 0 | 18/36 | 15 | 7 |
| B-rules (rules, selected) | 10 | 10 | 1 | 21/36 | 11 | 4 |
| P-all-ep (rules + records) | 12 | 10 | 4 | 26/36 | 9 | 3 |
| B (rules + records, selected) | 11 | 12 | 3 | 26/36 | 9 | 2 |

**On `cpu_mem`, B's lead over the all-rules prompt comes from what it holds, not from choosing what to
show.** Adding the episode records to the prompt closed the whole gap; selecting among rules alone
closed about 3 of 8 tickets, not significantly. Here a static prompt carrying the same content
matches retrieved memory, which is what H6's null predicted for 5 causes. P-all-ep was also the
fastest group (mean 74 s, 27.3 turns).

Caveats as before: run on a different day from exp13 (same pinned model, build and settings), three
histories, 12 tickets. The content in P-all-ep was written by memory (B's training run), so this
says the *static form* of a memory's content can match the memory, not that a prompt written without
experience can. Whether selection starts to matter when there are many lessons, some contradicting,
is the `house_lib` question; its core grid should include P-all-ep next to P-all.

## Generated output

### Pass counts per history (of 12)

| Group | h0 | h1 | h2 | Total | Mean turns | Mean wall (s) | Mean turns, passed |
|---|---|---|---|---|---|---|---|
| A | 2/12 | 3/12 | 1/12 | 6/36 | 34.2 | 168 | 33.3 |
| A0 | 4/12 | 2/12 | 2/12 | 8/36 | 35.1 | 148 | 36.2 |
| B | 11/12 | 12/12 | 3/12 | 26/36 | 28.8 | 99 | 28.3 |
| B-rules | 10/12 | 10/12 | 1/12 | 21/36 | 28.7 | 88 | 28.6 |
| C | 11/12 | 10/12 | 2/12 | 23/36 | 27.8 | 97 | 28.7 |
| P-all | 9/12 | 9/12 | 0/12 | 18/36 | 27.7 | 89 | 27.2 |
| P-all-ep | 12/12 | 10/12 | 4/12 | 26/36 | 27.3 | 74 | 26.5 |

### Paired comparisons (same history, same ticket)

| Comparison | Pairs | Only first passed | Only second passed | p (exact, two-sided) |
|---|---|---|---|---|
| P-all vs B | 36 | 1 | 9 | 0.021 |
| P-all vs A0 | 36 | 15 | 5 | 0.041 |
| A0 vs A | 36 | 6 | 4 | 0.754 |
| B vs A | 36 | 21 | 1 | 0.000 |
| P-all vs A | 36 | 14 | 2 | 0.004 |
| B-rules vs B | 36 | 2 | 7 | 0.180 |
| B-rules vs P-all | 36 | 6 | 3 | 0.508 |
| P-all-ep vs B | 36 | 5 | 5 | 1.000 |
| P-all-ep vs P-all | 36 | 10 | 2 | 0.039 |
| P-all-ep vs B-rules | 36 | 8 | 3 | 0.227 |
