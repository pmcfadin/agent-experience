# GEPA: Reflective Prompt Evolution, as an implementation guide for ELH Arm D

**Sources read**

- Paper: Agrawal et al., *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*, arXiv 2507.19457 **v2 (14 Feb 2026)**, marked "Accepted at ICLR 2026 (Oral)". I read the full PDF text (96 pp.). Citations like "§3.1" or "App. E.4" refer to this version.
- Code: `github.com/gepa-ai/gepa` at HEAD `7af5f0c` (2026-10-22). The latest PyPI release is `gepa 0.1.4` (MIT). HEAD may contain features that 0.1.4 lacks, so pin a commit or check before relying on anything below. Files are cited relative to `src/gepa/`.
- DSPy wrapper: `stanfordnlp/dspy` main, `dspy/teleprompt/gepa/gepa.py` (used only for the auto-budget formula).
- gskill (GEPA used to learn skill files for coding agents): `src/gepa/gskill/` plus the blog post `docs/docs/blog/posts/2026-02-18-automatically-learning-skills-for-coding-agents/index.md`.
- ELH: `SPEC.md` §3, §4.3, §8, §9, §10; `elh/judge.py`; `elh/runner/prompts.py`; `elh/tasks/families/cpu_mem/__init__.py`; episode statistics read read-only from `runs/elh.sqlite3`.

Labels used below: **[read]** means stated in the paper or code; **[inf]** means my inference or design proposal.

---

## 1. Algorithm, step by step

### 1.1 Objects

- **Candidate**: `dict[str, str]`, mapping each named component to its text (`core/adapter.py`, `Candidate = dict[str, str]`). Weights are frozen. GEPA evolves only text (§3) [read].
- **Inputs**: system Φ; `D_train`; metric μ; feedback function μ_f; rollout budget B; hyperparameters minibatch size b and Pareto-set size n_pareto (Alg. 1) [read].
- **Splits inside GEPA**: `D_train` is split into `D_feedback` (minibatches for reflection) and `D_pareto` (validation used *only* for selection). In the paper, the benchmark train split is `D_feedback` and the benchmark val split is `D_pareto` (App. E.4) [read]. In code, `trainset` is `D_feedback` and `valset` is `D_pareto`. If `valset` is None, the trainset is reused (`api.py`) [read].
- **Reflection sees only `D_feedback` traces.** Validation instances contribute scores only. The paper says "direct access to the content of validation instances is restricted" (§4). In code, the merge path evaluates valset with `capture_traces=False`, and only trainset minibatches feed `make_reflective_dataset` [read].

### 1.2 Main loop (Alg. 1, `core/engine.py`, `proposer/reflective_mutation/reflective_mutation.py`)

1. Initialise the pool `P = [seed]` and parents `A = [None]`. Evaluate the seed on all of `D_pareto` to get the per-instance score vector `S_seed` [read].
2. **While budget remains** (`max_metric_calls`, which counts every per-example evaluation; or other stoppers such as timeout, no-improvement, a `gepa.stop` file, or reflection cost):
   1. If merge is enabled, a merge is due, and the last iteration found a new program, try a merge first (§1.6). Either way, that iteration does no reflective mutation [read, engine.py ~L1007-1093].
   2. **SelectCandidate**: Pareto-based sampling (§1.3) gives parent index k.
   3. **SelectModule**: round-robin over components per candidate (`RoundRobinReflectionComponentSelector`). The alternative `'all'` updates every component at once [read].
   4. **Minibatch**: take b ids from `D_feedback` with `EpochShuffledBatchSampler`. It shuffles the ids each epoch, pads to a multiple of b with the least-frequent ids, and walks through them in order (`state.i * b`), so all training examples are covered evenly. It is seeded [read, `strategies/batch_sampler.py`].
   5. **Run the parent on the minibatch with traces** (`adapter.evaluate(..., capture_traces=True)`). The parent is **re-run** here; minibatch scores are not cached unless `cache_evaluation=True`. Cost: b rollouts [read].
   6. **Skip if perfect**: if `skip_perfect_score` (default True) and every minibatch score is ≥ `perfect_score` (default 1.0), no reflection happens. The b rollouts are still spent [read].
   7. **Build the reflective dataset**: `adapter.make_reflective_dataset(candidate, eval_batch, [component])` returns `{component: [record, ...]}`. The recommended record is `{"Inputs", "Generated Outputs", "Feedback"}` (`core/adapter.py` docstring). This is where μ_f's text goes [read].
   8. **Reflect**: the reflection LM gets the meta-prompt (§1.4) and returns new text in a ``` fence. Truncated output (finish_reason `length`, or an unclosed `<think>`) is rejected (`strategies/instruction_proposal.py` `ProposalAdapter`) [read].
   9. **Run the child on the same minibatch**. Cost: b rollouts [read].
   10. **Acceptance** (`StrictImprovementAcceptance`, the default): accept iff `sum(child scores) > sum(parent scores)` on that minibatch. `improvement_or_equal` is an option [read, `strategies/acceptance.py`].
   11. **If accepted**: evaluate the child on **all of `D_pareto`** (|D_pareto| rollouts), append it to the pool with its parent, and update the per-instance frontier. For each val id, the frontier set holds the program(s) with the max score, and ties are added (`core/state.py::_update_pareto_front_for_val_id`) [read].
3. **Return** the candidate with the highest **mean** `D_pareto` score. Ties go to the one with more coverage (`FullEvaluationPolicy.get_best_program`) [read].

**Rollout cost per iteration** [read, derived]: `2b` (parent re-run plus child) + `|D_pareto|` if accepted. The paper notes that "the majority of GEPA's rollout budget is spent on validation" (§4, Obs. 1).

### 1.3 Pareto-based candidate selection (exact procedure)

Paper, Alg. 2 [read]:
1. For each val instance i, `s*[i] = max_k S[k][i]`, and `P*[i] = {k : S[k][i] = s*[i]}`.
2. `C` is the union of all `P*[i]`.
3. Iteratively remove candidates that are dominated by another member of `C`.
4. `f[Φ]` is the number of instances where Φ is still in the pruned `P*[i]`.
5. Sample Φ with probability ∝ `f[Φ]`.

Code (`gepa_utils.py::remove_dominated_programs`, `select_program_candidate_from_pareto_front`) [read]. The implementation is a greedy redundancy prune, not a textbook dominance check:
- Programs are sorted by **ascending aggregate val score**.
- In that order, a program is dropped if *every* front it belongs to still has another surviving member. Fronts' remaining counts are decremented as programs are dropped.
- Then each survivor is listed once per front it is on, and `rng.choice` picks from that list (probability ∝ front count).
- Default `frontier_type="instance"` (one front per val example). Other options: `"objective"` (one per named objective from `objective_scores`), `"hybrid"`, and `"cartesian"` (one per (example, objective)) [read, `api.py`].
- Ablations available in code: `current_best`, `epsilon_greedy` (ε=0.1), `top_k_pareto` (k=5) [read].

### 1.4 Reflection meta-prompt, verbatim from code

`strategies/instruction_proposal.py::InstructionProposalSignature.default_prompt_template` [read]. It matches App. C of the paper.

```
I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks.
```

- `<side_info>` is the reflective dataset rendered as markdown: `# Example n`, then `## <key>`, with nested dicts and lists becoming `###` and deeper headers [read].
- A custom template can be passed (`reflection_prompt_template`, either one string or a per-component dict). It must contain `<curr_param>` and `<side_info>` [read].
- gskill replaces it with an agent-skill template (`gskill/gskill/train_optimize_anything.py::LoggingProposer.DEFAULT_PROMPT_TEMPLATE`). That template asks for terminal skills, repo knowledge, debugging strategies and editing patterns, and says "the agent will not be evaluated on the same task", "keep the overall skills concise", and "Adding skills blindly will not help". This is the better starting point for ELH [read].

### 1.5 Feedback function μ_f

- μ_f extends μ to return `(score, feedback_text)`. It exposes the **evaluation trace**, such as compiler errors, failed rubrics, or which constraints were met or failed, and can be module-specific (§3, "Evaluation traces as diagnostic signals") [read].
- Paper examples (App. E.1) [read]:
  - HotpotQA/HoVer: which gold docs were retrieved and which remain.
  - IFBench: which constraints were satisfied and which failed.
  - PUPA: a breakdown of the quality score and the PII-leak score.
  - Kernels: compiler and profiler output, plus retrieved manual sections chosen from the error (§5.1).
- In the adapter API, μ_f is **not a separate hook**. Scores come from `evaluate()`. Feedback text is whatever `make_reflective_dataset` puts in the `"Feedback"` field, built from the trajectories that `evaluate(capture_traces=True)` returned [read].

### 1.6 System-aware merge (App. D.1, Alg. 3-4; `proposer/merge.py`)

- **Preconditions** [read]: two Pareto survivors i and j, neither an ancestor of the other, sharing a common ancestor a with `agg(a) ≤ min(agg(i), agg(j))`, and at least one component where exactly one descendant still equals the ancestor.
- **Construction** [read]: for each component, take the descendant's version that changed. If both changed, take the one with the higher aggregate score. The merged candidate is evaluated on a **5-example val subsample** stratified across "i better / j better / tie". It is accepted if `sum ≥ max(parent sums)`, then fully evaluated on `D_pareto`.
- **Limits** [read]: at most `max_merge_invocations` (default 5). Off by default (`use_merge=False`).
- **For ELH** [inf, verified against the code path]: with a **single component**, merge cannot produce anything new. When both descendants changed the one component, the rule copies the better parent's text, the result equals an existing candidate, and it is skipped (`any(new_program == candidate ...)`). Merge only matters if the candidate has two or more components, such as a procedure plus a memory preamble.
- **Paper results** [read]: merge helped GPT-4.1-mini (+1.1 aggregate) and hurt Qwen3-8B (54.85 → 52.40 aggregate), and its budget allocation "needs further study" (Obs. 5).

### 1.7 Defaults (`api.py::optimize`) [read]

| Parameter | Default |
|---|---|
| `candidate_selection_strategy` | `"pareto"` |
| `frontier_type` | `"instance"` |
| `reflection_minibatch_size` | 3 (paper: "All GEPA optimization runs use a minibatch size of 3", App. E.4) |
| `module_selector` | `"round_robin"` |
| `acceptance_criterion` | `"strict_improvement"` (minibatch sum) |
| `skip_perfect_score` / `perfect_score` | True / 1.0 |
| `use_merge` / `max_merge_invocations` | False / 5 (paper: ≤5 merges when enabled) |
| `val_evaluation_policy` | `"full_eval"` (every accepted child scored on all of D_pareto) |
| `cache_evaluation` | False |
| `seed` | 0 (one `random.Random(seed)` drives selection, minibatches and merge) |
| Budget | `max_metric_calls` and/or `stop_callbacks` and/or `max_reflection_cost` (one is required) |
| Sampling | `SingleMutationSampling` (1 parent and 1 child per iteration). Options: `SameParentSampling(n)`, `IndependentSampling(n)`, `PxNSampling(p,n)` |
| Resume | `run_dir/gepa_state.bin`. A new `seed_candidate` on resume is added as a child of the saved seed |

DSPy `auto` presets (`dspy/teleprompt/gepa/gepa.py`) [read]: `light/medium/heavy` set n = 6/12/18 candidates. The budget is computed as `V + 5n + N·M + (periodic full evals)·V`, with `N = max(2·(2·num_preds)·log2 n, 1.5n)` and M = 35. This is a heuristic for cheap QA programs and does not fit ELH.

---

## 2. Methodology and results

**Benchmarks, systems and splits** (App. E.1) [read]

| Task | System | Train / Val / Test | μ_f text |
|---|---|---|---|
| HotpotQA | multi-hop retrieve + answer (DSPy) | 150 / 300 / 300 | docs still to retrieve, per hop |
| IFBench | 2-stage answer, then rewrite under constraints | 150 / 300 / 294 (IFBench OOD constraints as test) | constraints met / failed |
| HoVer | 3-hop, 2 query writers + 2 summarisers | 150 / 300 / 300 | correct docs retrieved / remaining |
| PUPA (PAPILLON) | query rewriter → untrusted LM → response rewriter | 111 / 111 / 221 | quality score + PII-leak breakdown |
| AIME-2025 | single CoT | AIME 2022-24 (90) split 45/45; test = 30 AIME-2025 questions × 5 | — |
| LiveBench-Math | single CoT | 368 questions split in thirds | — |

**Models** (App. E.2) [read]: Qwen3-8B (T=0.6, top-p 0.95, top-k 20) and GPT-4.1-mini (T=1.0), with a 16k context. All modules use the same model. The paper does not separately name the reflection LM in v2. Table 4 reports reflection calls per task model (17-90 per run), which suggests the task model reflected on itself [inf].

**Baselines** (App. E.4) [read]:
- **MIPROv2** with `auto=heavy` (18 instruction candidates and 18 few-shot sets; 2,270-6,926 rollouts). GEPA's budget was capped to MIPROv2's per benchmark, within 10.15%.
- **GRPO**: LoRA r16/α64, lr 1e-5, group size 12, 4 instances per step, 500 steps, which is **24,000 rollouts**. Early stopping on val every 20 steps. A full-finetune variant is shown in Fig. 11.
- **Trace (OptoPrime)** and **TextGrad**, at the same budget and with the same feedback.

**Key numbers** (Tables 1-2; test scores) [read]:
- Qwen3-8B aggregate: baseline 45.23, GRPO 48.91, MIPROv2 47.84, **GEPA 54.85 (+9.62)**, GEPA+Merge 52.40.
  - GEPA beats GRPO on 5 of 6 tasks (not on AIME: 32.0 vs 38.0).
  - GEPA's total rollouts were 1,839-7,051, against 24,000 for GRPO.
- GPT-4.1-mini aggregate: baseline 53.03, MIPROv2 58.67, TextGrad 59.14, **GEPA 65.22**, **GEPA+Merge 66.36 (+13.33)**.
- Prompts optimised on Qwen and evaluated on GPT-4.1-mini gave +9.00 aggregate, which is cross-model transfer (Obs. 6).

**Sample efficiency** (Obs. 1) [read]:
- GEPA matched GRPO's best validation score after 243, 402, 330, 1143, 1179 and 306 rollouts (up to 78× fewer).
- Counting **train (reflection) rollouts only**, it needed **79-737 rollouts** to reach its own optimum, and only **102, 32, 6 and 179** train rollouts to match GRPO on four tasks.
- Fig. 5 shows a single reflective update can give large gains (PUPA 82.26 → 90.99 at the first child).

**Ablations** [read]:
- Candidate selection (Table 3, Qwen3-8B, 4 tasks): SelectBest +6.05, BeamSearch(N=4) +5.11, **Pareto +12.44**.
- Merge: mixed results (above).
- GEPA prompts are up to 9.2× shorter than MIPROv2's, about 33% shorter on aggregate (Obs. 4, App. I).

**Cost** (App. E.3) [read]: all GPT-4.1-mini GEPA runs together cost $86.

**Extended uses** (§5) [read]:
- Inference-time search for code (NPUEval mean vector utilisation 4.25% → 30.52%; KernelBench fast₁ from about 0 to >20%), with temperature removed via caching.
- Adversarial prompt search: a GEPA-evolved adversarial prompt cut GPT-5-mini's AIME-2025 pass@1 from 76% to 10%. This shows how strongly a single text artefact can move a frozen model, in either direction.

**gskill**, the closest analogue to Arm D (blog post) [read]:
- Setup: Mini-SWE-agent with gpt-5-mini, a single component `{"skills": ""}` starting empty and injected into the system prompt template, SWE-smith tasks split about 200 train / 50 val / 60 test, reflection by a *stronger* model (`gpt-5.2-pro` default), minibatch 3, Pareto selection, default budget `--max-metric-calls 600`.
- Results: "Under 300 rollouts", resolve rate went from 55% to 82% (Jinja) and from 24% to 93% (Bleve). Skills transferred to Claude Code (Haiku 4.5: 79.3% → 98.3% on Bleve) and also cut task duration.
- μ_f: per task `{"Input": {Task ID, Problem[:200]}, "Generated Outputs": {Patch[:500], Agent Trace}, "Feedback": {Status: no_patch|f2p_failed|p2p_regression|all_passed, Test Output (head and tail 500 chars)}, "scores": {...}}` (`gskill/gskill/swe_fitness_fn.py`).
- Its "loop" proposer reflects one item at a time, then runs an LLM merge into one skill set (`LoopProposer.MERGE_TEMPLATE`).

---

## 3. Gap analysis and design for ELH Arm D

### 3.1 What ELH has and lacks

**Present in ELH** [read]:
- Frozen agent (`study-qwen3-coder-30b-a3b`, T=0.7, one request at a time).
- Byte-identical system prompt. Arm-specific text is allowed **only in the first user message** (`elh/runner/prompts.py::first_user_message`, which wraps memory in `<experience>`; SPEC §6.1 steps 1-2).
- Judge `Verdict{passed, checks{integrity, public_tests, correctness, performance}, detail{cases[], quality{reference_s, budget_s, agent_s, agent_over_reference, large_status, large_correct}}}` (`elh/judge.py`; example rows in `runs/elh.sqlite3`).
- Ledger category `learning` explicitly covers "SkillLearner reflection and validation episodes" (SPEC §8.1).
- `skills` table schema (SPEC §9).
- `dev` split, defined as "SkillLearner validation and harness tuning; never reported" (SPEC §4.3).
- Arm D contract: "every M episodes, reflect over clusters of successful traces → candidate procedure → validate on dev → promote only if it beats the incumbent; versioned; auto-rollback" (SPEC §10).

**Missing** [read]:
- `elh/learner/` does not exist.
- There is no slot for a procedure or skill in the prompt.
- I found no concrete `dev` cell/seed assignment in `elh/`. Batch files choose seed ranges by hand (e.g. `experiments/batches/exp1-memory-helps.yaml`).

### 3.2 What to optimise [inf]

- **Component `procedure`** (start here). A single text block injected into the first user message, e.g. `<procedure>…</procedure>`, placed before `<experience>`. The wrapper text is part of the frozen prompt and **needs a registration bump** (prompts.py says edits require a new registration).
  - Seed it with an empty string (as gskill does), or with a short hand-written checklist.
  - Arms A-C would see no block, so A/B/C stay byte-identical.
- **Optional second component `experience_preamble`**: replace the fixed sentence "Notes retrieved from your previous work… may or may not apply here." with an evolvable text that tells the agent how to use recalled memories.
  - Only with two components do round-robin and merge mean anything.
  - Start with one component. Fewer moving parts and less budget.

### 3.3 Data: dev only [inf]

**Rules**:
- **Never** sealed, checkpoint or retention tasks.
- Draw dev tasks only from **train (cause, theme) cells, with a disjoint dev seed range**. Do not use held-out cells or `sealed_novel_cause` causes, or GEPA will learn about the novel-cause split, which defeats H3. If dev cells are defined separately, they must be a subset of the train cells.
- GEPA's reflection sees `D_feedback` traces, so the skill text can absorb cause-specific idioms. That is the intended within-cause transfer. Cross-cause claims stay clean only if the dev causes ⊆ train causes.

**Three disjoint dev pools per GEPA invocation**:
- `D_feedback`: 15-30 tasks covering all train causes × several themes. Reflection minibatches of 3 cycle through them.
- `D_pareto`: 8-12 tasks, stratified so each train cause appears about twice. Used for selection only.
- `D_confirm`: 15-20 fresh dev tasks. This is the SPEC's "promote only if it beats the incumbent" gate, run **outside** GEPA as a paired comparison: incumbent vs GEPA winner on the same tasks, ideally ×2 replicates. It is needed because GEPA's best-of-pool on `D_pareto` is selection-biased (winner's curse, worse with a stochastic agent at T=0.7).

**Memory state**:
- Arm D is "as C", so episodes include recalled memory. Run each GEPA invocation against **one fixed read-only snapshot** (`open_snapshot`, created at the trigger point). Otherwise candidate scores drift as the bank changes. SPEC lets dev read the live bank, but a snapshot is required here for comparability.
- Record `memory_snapshot_id` on every GEPA episode.

**Trigger**: SPEC says "every M episodes". Treat each trigger as one GEPA run, warm-started with `seed_candidate = incumbent`. A fresh run_dir per trigger is cleaner than resuming. Larger M (e.g. 25-50) keeps learning spend bounded.

### 3.4 Score μ and feedback μ_f from the judge [inf]

**Ceiling problem** [read, `runs/elh.sqlite3`]: Qwen arm A passes 0.90 of checkpoint episodes (n=20) and arm B 0.80 (n=20); B train is 0.78 (n=32). With binary scores and b=3:
- Most minibatches are all-pass. P(all 3 pass) ≈ 0.73 at p=0.9.
- `skip_perfect_score` then skips the reflection but still spends 3 rollouts.
- Strict-improvement acceptance has nothing to improve.

**Fixes**:
- (a) **Graded score.** Reuse arm C's reward so that GEPA optimises the same thing the primary endpoint measures: `r = passed × (1 − normalised_cost)` (SPEC §10), plus partial credit for failures. Example: integrity broken = 0.0, public tests fail = 0.05, correctness fail = 0.1, performance-only fail = 0.3 + 0.2·clip(budget_s/agent_s). With this, a 1.0 score is essentially unreachable, so set `perfect_score` above the maximum or `skip_perfect_score=False`.
- (b) **Weight `D_feedback` toward harder dev cells.** First measure a per-cause dev baseline. The single dev failure in `runs-dev-v1-20261025` was performance: `large_status: timeout`, agent_over_reference 18×. This biases training only, which is acceptable, because promotion is still decided on stratified `D_confirm`.
- Optional: `objective_scores = {correctness, performance, integrity, cost}` with `frontier_type="hybrid"`. This keeps candidates that are best on one axis.

**Feedback text**: reuse the `lesson-v2` discipline (SPEC §10).
- State passed and failed checks separately, with failure wording, not pass wording. This is the exp1 bug, which voided that run's arm B result.
- Name any protected files changed.
- Give the agent's time against `budget_s`, the outcome (submitted / budget_exhausted …), turns and tokens.
- **Exclude** the oracle files, hidden case sizes/inputs, and `reference_s`, matching the memory policy. Integrity failures are the most important signal to keep explicit, because GEPA will happily evolve text that games anything ungated.

**Trajectory for reflection**:
- Ticket (≤1.5k chars).
- Compacted tool-call log: the commands run, truncated outputs, and the last N turns in full.
- The agent's submit summary and its diff (≤80 lines, as in lesson-v2).
- The verdict feedback.
- With b=3, three such records must fit the reflection LM's context, so cap each at about 6-8k tokens.

**Reflection prompt**: adapt gskill's template. Changes:
- Replace the Docker/repo text with ELH's sandbox facts (fresh bash process per call, no network, time limits).
- Tell it the procedure will be used on *other tickets of other causes and themes*.
- Forbid ticket-specific identifiers and numbers.
- Cap length (e.g. ≤ 400 words). Prompt length is inference cost charged to the arm.
- Keep the `<curr_param>` and `<side_info>` placeholders.

### 3.5 Can the `gepa` package be used directly? Yes. [read + inf]

- Core `gepa` has **no required dependencies** (`pyproject.toml: dependencies = []`; litellm only in the `full` extra).
- `reflection_lm` can be **any callable `(str | list[dict]) -> str`** (`LanguageModel` protocol, `proposer/reflective_mutation/base.py`).
- Recommended: pass a thin wrapper around ELH's own OpenAI-compatible client instead of `gepa.lm.LM`/litellm. The wrapper keeps the per-alias lock, `/props` and backend-identity capture, and **learning-ledger charging**.
- `gepa.lm.LM("openai/<alias>", api_base=…, api_key=…)` via litellm also works, but bypasses the ledger.

**Adapter sketch** [inf]:

```python
class ElhProcedureAdapter:          # satisfies gepa.core.adapter.GEPAAdapter
    propose_new_texts = None        # use GEPA's reflective proposer
    def evaluate(self, batch: list[TaskInstance], candidate: dict[str, str], capture_traces=False):
        outs, scores, trajs = [], [], []
        for task in batch:          # serial: the alias takes 1 request at a time
            ep = run_episode(cfg, store, task, client, arm="D", split="dev",
                             memory=snapshot_view, procedure=candidate["procedure"],
                             ledger_category="learning")        # retry infra_error; raise if persistent
            scores.append(score_from_verdict(ep.verdict, ep.cost))
            outs.append({"episode_id": ep.episode_id, "passed": ep.verdict.passed})
            trajs.append(compact_trace(ep) if capture_traces else None)
        return EvaluationBatch(outputs=outs, scores=scores,
                               trajectories=trajs if capture_traces else None)
    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        recs = [{"Ticket": t.ticket, "Agent actions": t.actions, "Agent summary and diff": t.diff,
                 "Feedback": t.feedback_text, "Score": s}
                for t, s in zip(eval_batch.trajectories, eval_batch.scores)]
        return {c: recs for c in components_to_update}

result = gepa.optimize(seed_candidate={"procedure": incumbent},
    trainset=d_feedback, valset=d_pareto, adapter=ElhProcedureAdapter(...),
    reflection_lm=LedgeredLM("study-gemma4-26b-a4b"),          # or the Qwen alias; see below
    reflection_prompt_template=ELH_TEMPLATE, reflection_minibatch_size=3,
    candidate_selection_strategy="pareto", use_merge=False,
    skip_perfect_score=False, max_metric_calls=BUDGET, run_dir=..., seed=history_seed,
    raise_on_exception=True)
```

**Notes** [read: adapter contract; inf: ELH mapping]:
- **Infra errors must not score 0.** The adapter contract says to return 0 for *task* failures and to raise for systemic ones. Map ELH `infra_error` (backend identity changed mid-episode, sandbox failure) to retry-then-raise.
- **Ledger**: every GEPA episode is a normal `episodes` row with `split='dev'`. Its inference and the reflection tokens go to `learning`, not to the arm's `inference` (SPEC §8.1).
- **Versioning**: on promotion, write a `skills` row with `body_sha256`, `validation_json` (the GEPA run summary, D_pareto scores and D_confirm paired result) and `source_episode_ids_json` (the D_feedback episode ids). Store `gepa_state.bin` content-addressed, along with the GEPA commit and config, in the preregistration.
- **Reflection LM choice**:
  - Gemma (the memory LLM alias) keeps "learning" on one pinned model, consistent with Hindsight.
  - Qwen self-reflection follows the paper's same-model setup. gskill used a much stronger reflector.
  - Both are local and pinned. Pick one and preregister it. A frontier reflector would make Arm D's gain partly "distillation from a stronger model" and is a confound.
- **Parallelism**: GEPA's `batch_evaluate` hook allows parallel rollouts, but the alias serialises requests, so expect strictly serial throughput.

### 3.6 Rollout cost and a realistic minimal budget

**Episode time** [read]:
- `runs/elh.sqlite3`: Qwen episodes average **55-67 s wall** (arm A checkpoint 67.3 s, n=20; arm B train 55.4 s, n=32; range 28-116 s), including about 2-4 s of oracle time.
- The caller's ~35 s is GPU time. Plan on **~60 s wall per rollout, serial**, i.e. about 60 rollouts/hour.

**Formula** [read, derived from §1.2]: `R ≈ |D_pareto| + N·2b + a·N·|D_pareto|`, where N = iterations and a = acceptance rate.

**Minimal per-trigger budget** [inf]: b=3, |D_pareto|=10, N=15, a≈0.3.
- GEPA: 10 + 90 + 45 ≈ **145 rollouts ≈ 2.4 h** (about 1.4 h if 35 s holds). Set `max_metric_calls=150`.
- Reflection: about 15 calls (Table 4 shows 17-90 per full run). Negligible next to rollouts.
- Promotion gate: incumbent vs winner on 20 `D_confirm` tasks, ×1 = 40 rollouts (≈ 40 min); ×2 replicates = 80.
- **Total ≈ 185-225 rollouts ≈ 3-4 h GPU-serial per trigger**, all charged to `learning`.

**Context for that number** [read]: it is in the same range as gskill's "under 300 rollouts" and the paper's 6-179 train rollouts to match GRPO. Those gains came from far lower baselines (24-55% resolve), not 80-90%.

**Fit to the schedule** [inf]:
- With M=50 and a 200-episode train horizon, that is 4 triggers ≈ 750-900 learning rollouts per (history, arm).
- That is 4-5× the train episodes themselves. `amortised_learning` (SPEC §8.2, `horizon_tasks` 1000) will make it visible in cost_per_success.
- Cheaper options:
  - A single GEPA run at the midpoint.
  - |D_pareto|=6 with `val_evaluation_policy` subsets. The paper flags "dynamically selected validation subsets" as future work (Obs. 1).
  - `cache_evaluation=True`, which only helps on exact repeats.
- **Smallest meaningful pilot**: one GEPA run at 150 metric calls plus a 40-rollout paired confirm, about 3 h, before committing Arm D's protocol.

**Statistical power** [inf]:
- At p≈0.85, a 20-task paired confirm cannot resolve gains under about 15-20 pp on pass rate.
- The graded or cost-aware score (§3.4) is what makes promotion decisions informative. Preregister the promotion rule, e.g. "mean paired Δr > 0 and no integrity regressions".

---

## 4. Caveats

1. **Noise and acceptance.** [read: the rule; inf: impact] The paper's tasks are mostly single-turn at T≈0.6-1.0. ELH episodes are about 30-turn agent runs at T=0.7. The strict minibatch test (3 vs 3 samples) will accept many lucky children and reject good ones. Full `D_pareto` evaluation filters only partly, and the final argmax over the pool is optimistic. The external `D_confirm` gate is mandatory, not optional. Consider `improvement_or_equal` only with a graded score.
2. **Ceiling.** [read: rates; inf: impact] At 80-90% baseline pass, binary μ makes GEPA mostly burn rollouts on all-pass minibatches (§3.4).
3. **Merge is a no-op** for one component, and hurt Qwen3-8B in the paper. Keep `use_merge=False` [read].
4. **Overfitting and leakage.** [inf] The default meta-prompt explicitly asks to "include all niche and domain specific factual information". For ELH that pushes ticket/theme specifics and oracle-ish facts into the procedure. Use a generalisation-focused template, strip `reference_s` and hidden-case data from feedback, and keep dev cells ⊆ train cells so the `sealed_novel_*` splits stay clean.
5. **Reward hacking and integrity.** [inf, with exp1 precedent read in SPEC §10] A text artefact can steer the agent toward editing tests or harness files. Integrity must score 0, be stated explicitly in feedback, and block promotion. The paper's §5.2 adversarial result shows how much a single prompt can move a model.
6. **Interaction with memory.** [inf] The procedure is optimised against one memory snapshot. Later recalls differ, so the promoted skill can degrade. SPEC's "auto-rollback if checkpoint retention drops" covers this, but needs the retention panel scored for D with and without the skill to attribute any drop.
7. **Version drift.** [read] HEAD (`7af5f0c`) has many post-paper features (optimize_anything, ComBEE reflectors, parallel sampling strategies, agent-state dirs), and PyPI is 0.1.4. The paper's algorithm corresponds to the defaults described here. Pin the exact version in the preregistration and record it in `runs.config_json`.
8. **Paper scope.** [read] The headline gains are on QA, IF, privacy and math pipelines with 111-150 training examples and thousands of rollouts. Agentic evidence (gskill, kernels) comes from a blog and from preliminary §5 results, not from controlled comparisons. The reflection-model identity for the main tables is not stated explicitly in v2 (my inference: same as the task model). gskill used a far stronger reflector.
9. **SPEC mismatch.** [read] SPEC §10 describes D as "reflect over clusters of *successful* traces". GEPA reflects over fresh minibatch rollouts of the *current candidate*, both successes and failures, on dev tasks. ELH `train` traces are not consumed directly, though they could seed the first candidate via one offline reflection over train clusters [inf]. If GEPA is adopted, SPEC §10's D row needs rewording and re-registration.
10. **Cost accounting.** [inf] GEPA rollouts are real agent episodes on the shared single GPU. Keep them out of timed train/eval windows, as is already done for Hindsight consolidation, so wall-clock and latency measurements of the arms are not disturbed.
