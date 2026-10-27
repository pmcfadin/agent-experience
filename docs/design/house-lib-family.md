# Design: `house_lib` task family (memory vs static prompt)

Status: draft for review, 2026-10-27. Nothing here is implemented.

## 1. Why a new family

The `cpu_mem` results show memory groups beating the stateless group A, but they cannot show that
memory beats a *better prompt*. The fixes (use a set, build an index) are already known to the
model, so a short generic hint may recover most of the gain, and with 5 causes a full list of
distilled rules fits easily in the system prompt.

The question this family is built to answer (proposed **H6**):

> Retrieved memory beats a static prompt carrying the same lessons once the number of
> conditional, unguessable lessons is large enough or the lessons conflict across contexts; at
> small numbers of lessons the static prompt matches or beats it.

No public dataset has the needed properties (see `reports/Memory vs prompt datasets.md`), so we
generate our own, reusing the SPEC §4 task contract, judge and split machinery.

Prior designs this follows (verified citations in
`research_notes/Memory vs prompt datasets/methodology_citations.md`):

- Task format: code written against a library the model cannot know (Zan et al. 2022,
  "When Language Model Meets Private Library"; CodeUpdateArena, arXiv 2407.06249).
- Unguessable lessons: counterfactual changes to familiar behaviour (Wu et al. 2023,
  "Reasoning or Reciting?").
- Splits: held-out combinations of familiar parts (SCAN, COGS, gSCAN); held-out seeds (Procgen).
- Context controls: gold / retrieved / distractor / none ("Lost in the Middle"; Cuconasu et al.
  2024, "The Power of Noise"; Yoran et al. 2024); long-prompt degradation (RULER).
- Transfer and forgetting metrics: GEM (Lopez-Paz & Ranzato 2017); van de Ven, Tuytelaars &
  Tolias 2022.

## 2. Concept

Each experiment runs in one **world**: a fixed, generated in-house Python library (`acmecore`) plus a
service codebase that uses it, plus a catalogue of **lessons**, the house conventions every change
must follow. The world is the "company codebase": it is identical for every task in a world, which
is what lets a lesson learned on one ticket apply to the next.

A **task** is a feature ticket against that codebase ("add an overdue-invoice export for partner
Halden"). Doing the feature is easy; doing it the way the house requires is the hard part. Each
ticket needs 1–3 lessons. The ticket names the context (partner, region, module) but never the
convention.

Mapping to the SPEC §4 contract:

| SPEC term | `house_lib` meaning |
|---|---|
| family | `house_lib` |
| latent cause | a lesson id (the task's primary lesson; all required lessons are in `signature.lessons`) |
| theme / surface | business domain and feature (reuse the `cpu_mem` themes: billing, logistics, …) |
| instance | `(world_seed, lesson set, theme, seed)` → one ticket + workspace + oracle |

`task_id` = `house_lib/<world>/<primary_lesson>/<theme>/<seed>`. The world id (`L16-c50-w7`, i.e.
16 lessons, 50% conflict rate, world seed 7) and its library fingerprint go into every task's
`signature` and `content_hash`.

## 3. Lessons

### 3.1 Schema

Each lesson in a world's catalogue has:

| Field | Meaning |
|---|---|
| `id` | e.g. `partner_date_format.halden` |
| `template` | the lesson template it was instantiated from (§3.2) |
| `trigger` | the WHEN: an observable property of the ticket or workspace (partner, region, module, data type) |
| `action` | the THEN: what the change must do |
| `visibility` | `hidden` (only judge feedback reveals it), `source` (inferable by reading `acmecore` closely), `doc` (stated in `acmecore` docs) |
| `conflict_group` | lessons with the same action but opposite conventions in different contexts, or none |
| `gold_text` | one-sentence statement written by us; used by the handbook and oracle controls |
| `check` | the oracle check that fails if the lesson is ignored |
| `blind_variant` | generator for a functionally correct solution that ignores only this lesson (validation) |

### 3.2 Templates

Lessons are instantiated from templates. Each instantiation makes arbitrary choices from the world
seed (which partner wants which format, which module uses which unit). That is what makes them
unguessable: there is no "right" answer to infer, only the house's choice.

| Template | Example lesson | Check |
|---|---|---|
| `partner_format` | Halden's feed wants dates as `DD.MM.YYYY`; Orsk's wants epoch ms | output field format on hidden inputs |
| `call_order` | `audit.record()` must be called before `store.write()` | instrumented `acmecore` records the call order |
| `counterfactual_default` | `acmecore.page()` is 1-based in `ledger`, 0-based in `catalog` | correct rows on multi-page hidden inputs |
| `required_wrapper` | time comes from `acmecore.clock.now()`, never `datetime.now()` | frozen clock in the oracle; output must match |
| `sentinel_return` | `cache.get()` returns `MISSING`, not `None`, and `MISSING` is truthy | hidden input with a cache miss |
| `unit_convention` | `billing` uses integer cents, `ledger` uses `Decimal` strings | output type and value |
| `event_fields` | every emitted event carries an `idem_key` built by `ids.idem()` | event schema and key on hidden inputs |
| `region_redaction` | EU records must pass email through `pii.mask()`; US must not | output per region |
| `retry_policy` | `TransientError` must go through `retry.call()`; `FatalError` must not be retried | fault-injecting `acmecore` in the oracle |
| `feature_flag` | new code paths in `payments` check `flags.on("payments_v2", tenant)` | oracle toggles the flag |

Ten templates instantiated across contexts give up to ~100 distinct lessons. Lessons from the same
template share a meta-lesson ("check the partner's format"). A static prompt can state that
meta-lesson but cannot carry every specific choice unless it lists them all, which is exactly the
memory vs prompt contrast. Analysis reports results by template so meta-lesson transfer is visible.

### 3.3 Conflicts

Within a `conflict_group`, contexts require opposite actions (two partners, two regions, two
modules). The conflict rate `c` is the share of lessons that belong to a group. Conflicts are the
sharpest test: a static prompt listing both rules forces the model to pick the right one by context
on every task; retrieval keyed on the ticket shows only the rule that applies.

## 4. Task instance

### 4.1 Workspace

```
acmecore/            # the world's library: identical across tasks in a world (protected)
service/             # app code using acmecore; the feature goes here
  <module>/…
  partners.py        # partner and region registry (gives the triggers their context)
tests/test_public.py # public tests: the feature's basic behaviour, no lesson checks (protected)
examples/            # sample input/output for the feature, lesson-neutral fields only (protected)
TICKET.md
```

Protected: `acmecore/**`, `tests/test_public.py`, `examples/**`. The integrity check is as in
`cpu_mem`.

### 4.2 Ticket

Generated from a template per (theme, feature), naming the context and the expected behaviour, never
the convention. For example: "Partner Halden needs a nightly CSV of overdue invoices for EU accounts.
Columns: invoice id, account, due date, amount. See `examples/` for the shape." This needs
`partner_format.halden`, `region_redaction.eu` and `unit_convention.billing`.

### 4.3 Oracle

Run in a fresh sandbox after submission, as in SPEC §7. The oracle swaps in an instrumented copy of
`acmecore` (call recording, frozen clock, fault injection, flag toggles); this is allowed because
`acmecore` is protected. A pass requires all of:

- `integrity`: protected files unchanged;
- `public_tests`: still pass;
- `feature`: output correct on hidden inputs on every lesson-neutral field (3 small, 2 medium; edge cases);
- `lesson:<id>`: one check per required lesson.

The verdict reports every check separately. Lesson checks are the finer-grained outcome used in
analysis (§7) and the per-check feedback the rule writer sees (SPEC §6.2; it never sees the oracle,
hidden inputs or `gold_text`).

## 5. Generator validation

`elh validate-family house_lib` extends SPEC §4.4. For every lesson × theme and ≥ 20 seeds:

1. the reference solution passes;
2. the untouched workspace fails `feature`;
3. **necessity:** each lesson's `blind_variant` passes `feature` and fails exactly `lesson:<id>`;
4. **conflict:** for each conflict group, a solution that applies a sibling context's rule fails `lesson:<id>`;
5. hard-coding the public example output fails;
6. regeneration is byte-identical; the world's library fingerprint is stable.

Plus an **empirical calibration** on `dev` seeds before a world is used (not in CI; costs episodes):

- **guessability:** group A (no hints), per lesson. A lesson whose check A passes in > 20% of
  episodes is guessable and is dropped or redrawn.
- **learnability:** handbook control P-hand (§6), per lesson. A lesson passing < 60% even with
  `gold_text` in the prompt is too hard for the model to follow and is dropped.

This two-sided filter makes every lesson one the model can't guess but can follow once told, so
differences between groups come from how the lesson reaches the model, not from the lesson itself.

## 6. Test groups and controls

All groups share model, sandbox, budgets, tools and judge (as now). Only the carried content and
where it appears differ.

| Group | Carries | Where | Role |
|---|---|---|---|
| A | nothing | — | stateless baseline |
| A0 | nothing, but memory-group instructions and an empty `<experience>` block | first message | separates the scaffolding from the content |
| B | Hindsight bank + verbatim rules (as exp10–13) | first message, retrieved per task | memory |
| C / D | as now | as now | secondary; not in the core grid |
| **P-all** | all of B's rules at the end of training, frozen, in training order | system prompt | **key control**: same content as B, no per-task retrieval |
| P-oracle | B's rules for the task's own lessons only (by `signature.lessons`) | system prompt | upper bound on retrieval quality |
| P-wrong | same number of tokens of B's rules for other lessons | system prompt | rules out "more text helps" |
| P-hand | `gold_text` of every lesson in the world | system prompt | best possible static prompt |
| W-hand | `gold_text` as `HANDBOOK.md` in the workspace | workspace | static docs the agent must look up itself |
| F-fix / F-ret | 3 passed training trajectories, fixed vs retrieved by ticket similarity | first message | few-shot comparison |
| G-opt | prompt optimised with GEPA (and an ACE playbook as a variant) on train episodes, same episode budget as B's training | system prompt | prompt optimisation comparison |

P-all, P-oracle and P-wrong are built per history from that history's B bank, so content is matched
exactly. P-hand and W-hand are the same for every history. All system-prompt variants keep the
frozen system prompt byte-identical and append one delimited block; this is an addition to SPEC
§6.2, which today fixes the system prompt across groups.

Prompt cost is counted: P-all and P-hand pay their block on every turn (cache reads included), and
cost per success (H1's endpoint) shows it.

## 7. Experimental grid and analysis

### 7.1 World parameters

| Parameter | Values (core grid) |
|---|---|
| lessons per world `L` | 4, 16, 64 |
| conflict rate `c` | 0, 0.5 |
| lessons per task `m` | 1–3, mean 1.5 |
| training exposures per lesson `e` | 3 (training length N = e·L/m ≈ 2L: 8, 32, 128) |
| evaluation tasks per world | 24, fixed across `L` so power doesn't change |
| histories | 3 |

Visibility mix is fixed (e.g. ⅓ each) in the core grid and reported by stratum.

### 7.2 Splits

As in SPEC §4.3, over (lesson, theme) cells:

- `sealed_familiar`: trained lessons, new seeds;
- `sealed_novel_combo`: trained lessons in combinations or themes not seen together in training;
- `sealed_novel_cause`: lessons never seen in training (expect no gain; measures harm, ΔRR);
- `retention`: a fixed panel of training lessons re-measured as training continues (forgetting,
  backward transfer).

### 7.3 Primary analysis

- Outcome: task pass, and per-lesson check pass (more observations, more power).
- Model: mixed-effects logistic regression, `pass ~ group × log2(L) × c + (1 | task) + (1 | history)`.
- H6 is supported if the B − P-all gap rises with `L` and with `c`, and B ≥ P-all at the largest `L`.
  It is refuted if P-all ≥ B at every `L`; that result is publishable too.
- The existing McNemar and bootstrap comparisons are kept per world.
- Diagnostics: retrieval recall (did B show the rule for each required lesson?), lesson-check pass by
  rule position in P-all (long-prompt degradation), cost per success.

## 8. Phases and cost

Measured episode wall time on `cpu_mem` (exp6–13, `runs/elh.sqlite3`): median ≈ 150 s for A, ≈ 90 s
for B. Assume ≈ 2 min per episode on average.

| Phase | Content | Episodes (approx.) | Wall time |
|---|---|---|---|
| 1. Generator | templates, world builder, oracle instrumentation, `validate-family` | 0 | engineering only |
| 2. Calibration | A and P-hand on dev seeds, 1 world per `L` × `c`, ~4 tasks per lesson | ~1,000 | ~1.5 days |
| 3. Core grid | A, B, P-all, P-oracle, P-hand × 6 worlds × 3 histories (B also trains) | ~3,200 | ~4.5 days |
| 4. Extensions | A0, P-wrong, W-hand, F-*, G-opt, C/D, novel splits, retention | depends on selection | — |

Phase 3 can start smaller (L ∈ {4, 16}, c = 0.5, 2 histories: ~600 episodes, under a day) as a go /
no-go before running the full grid.

## 9. Open decisions

1. **Visibility.** Should `hidden` lessons be in the main grid? They are learnable only from judge
   feedback, so A can never pass them. That is the purest memory test but the least realistic.
   Alternative: main grid uses `source` and `doc` only; `hidden` is an extension.
2. **Rule pool for P-all.** One rule per training episode (verbatim, as now) means 2L rules, some
   duplicates or wrong. Use them as is (matched to B), or also test a condensed version (an
   ACE-style merged playbook)? Proposed: as is for P-all, condensed as part of G-opt.
3. **System-prompt arms.** These need a SPEC §6.2 change (system prompt fixed except one appended
   block). Alternative: put the static block in the first message like B, which makes position the
   same and isolates retrieval, at the cost of being less like a real "system prompt" deployment.
   Proposed: first message, same position as B's block, so only selection differs.
4. **Context size.** At L = 64, P-all is about 128 rules (~8–10k tokens). Check against the pinned
   context size and `memory_token_budget`; B stays capped, and P-all deliberately is not.
5. **Library realism.** Generated `acmecore` will be small (~1–2k lines). Bigger makes lookup costlier
   (helps the memory case, raises cost); proposed: start small, and add size as a later parameter.
