# Two continual-learning-via-memory papers: implementation-grade notes

Sources were read in full from the arXiv HTML (LaTeXML) renderings, which have no page numbers,
so citations give section, table, figure or appendix. Local copies:
`scratchpad/papers/2604.27003.{pdf,html,txt}` and `2607.22157.{pdf,html,txt}`. The released code
for paper 2 is cloned at `scratchpad/papers/spark/`. Its run data is stored in Git LFS and was
not pulled, because `git lfs` is not installed.

Tags: **[read]** means stated in the paper or code. **[inf]** means my inference.

| arXiv id | Title | Authors | Code |
|---|---|---|---|
| **2604.27003** (v1, 29 Apr 2026) | *When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents* | Qisheng Hu, Quanyu Long, Wenya Wang (NTU) | No repo linked. Built on AgentGym and the ReMe memory module (Cao et al. 2025, arXiv 2512.10696) (§A.1.3) |
| **2607.22157** (v1, 24 Jul 2026) | *Learning on the Job: Continual Learning from Deployment Feedback for Frozen-Weights Agents* | Valentin Tablan, Scott Taylor, Kristoffer Bernhem (The Memory Company) | https://github.com/memcoai/spark-continual-learning-paper-data (MIT code, CC-BY data). The Spark memory server itself is proprietary and not released |

The two ids are the reverse of the order in the request: 2604.27003 is "Moves to Memory" and
2607.22157 is "Learning on the Job".

---

## Paper 1: "Learning on the Job" (Tablan, Taylor, Bernhem; arXiv 2607.22157)

### 1.1 Method

**Setting [read §1, §3.3].** A frozen agent works τ-bench banking (τ-Knowledge) tasks. There are
97 tasks, and the policy (about 700 documents) must be found with a BM25 `KB_search` tool. The
memory store is external (Spark, reached over MCP).

**Feedback grades [read §3.3.1–3.3.2].**
- *experience*: a one-bit verdict from the deterministic evaluator (final DB state equals ground
  truth). On a failure the lesson is restricted to avoidance: the agent may record "do NOT X" and
  may not assert a correct action.
- *instruction*: the verdict, plus, after a failed episode only, the verified gold action
  sequence, delivered once.

**Unit of memory [read §3.1].** An *insight* carries:
- a `title`;
- `content`;
- a `query`, "phrased as the question a future agent would ask mid-task; the query is the
  primary retrieval hook";
- optional `tags`.

The memory-tool schema is in App. A.2. `memory_create(query, title, content, tags?)` describes
the query as "What a future agent would search to find this lesson."

**Rule format: one rule in WHEN–THEN form per episode [read §3.3.3, App. A.5].** The required
first lesson depends on the outcome.

On success (both modes):
- title `VERIFIED RULE: <short name for this customer situation>`
- content `"WHEN <the customer situation, as it appeared in this conversation> THEN <the DECISIVE
  action(s), chosen from the CONSEQUENTIAL ACTIONS list above -- tool names and every
  decision-critical value ... verbatim; customer identifiers as placeholders like
  customer_name=<their name>>"`
- The WHEN clause "must name the customer's goal AND the observable behavior that distinguishes
  this situation from similar-looking ones".
- "Do NOT soften it with alternatives or conditions you did not verify ... hedged rules do not
  survive the next conversation."

On failure under instruction:
- Same title form. THEN holds "the exact correct action(s), copied VERBATIM from the CORRECT
  solution above".
- If the ACTION DIFF shows the decisive divergence, add one sentence naming it.

On failure under experience:
- title `AVOID: <the mistaken action, short>`
- content `"WHEN <situation> do NOT <the exact action(s) taken ...> -- verified wrong against the
  ground-truth outcome."`
- "You do NOT know what the correct answer was: do not invent one, and do not guess."

After the required lesson the agent may save further lessons: "a separate `memory_create` per
genuinely DISTINCT, reusable lesson. Prefer a few high-value lessons over many marginal ones."

**Reflection turn [read App. A.4; code `spark_agent.py::_agent_reflection_writeback`,
`_build_reflection_nudge`].**
- The same agent runs it after the episode, with the real conversation seeded as message
  history.
- The prompt template:

```
The customer interaction is now COMPLETE.
{verdict}
{correction_block}{diff_block}{consequential_block}{outcome_block}{diagnosis_hint}
Now update your experience memory so future similar interactions go better. This is the ONLY thing to do now -- do NOT message the customer and do NOT call any other tools ...:
{save_instructions}
Rules for EVERY memory you write:
- NEVER contradict, weaken, or add exceptions to an existing memory whose title starts with "VERIFIED RULE:". If your experience seems to conflict with one, save a SEPARATE note describing the conflict -- do not edit the rule. (One exception: merging ... into a single branch rule ...)
- Do NOT state facts (rates, fees, eligibility, program terms) that no tool result in THIS conversation supports. If you cannot point to the evidence for a reason, write "Reason not documented in the KB; rule verified against the ground-truth outcome." instead of a reason.
- Do not save customer-specific facts (names, account numbers).
{rate_line}{transcript_block}
```

- `{verdict}` = `TASK OUTCOME (ground truth): the desired outcome WAS / was NOT achieved.`
- On any failure, `{diagnosis_hint}` = "A wrong outcome doesn't always mean wrong facts. The cause
  may be the wrong action or procedure, a missing or mis-stated detail, the wrong tool -- or, when
  the customer is the one who must act, not persuading them to take the right step. Work out which
  applied here and capture that lesson."

**Harness-computed credit assignment [read §3.3.3; code `reflection_diff.py`].**
- *On success*, a `CONSEQUENTIAL ACTIONS` block lists every state-changing (non-read-only) action
  taken, in order. Its purpose is that "the banked rule credits the decisive action rather than
  defaulting to the last one". The code comment records a real failure this fixed: a success was
  banked as "transfer to team" because the conversation ended with a transfer.
- *On an instruction failure*, a deterministic ACTION DIFF between gold and executed actions
  lists missing actions (with counts), wrong-args actions and extra consequential actions.

**Write-time validator [read App. A.7; code `spark_agent.py` lines ~1108–1140,
`reflection_diff.rule_required_strings/rule_content_gaps`].**
- From the diff it collects the required strings, capped at 12:
  - gold string values of wrong arguments (customer identifiers excluded);
  - names of extra tools;
  - names of missing tools.
- It refuses a `VERIFIED RULE` write when any required string is absent, or appears only after
  "e.g."/"for example"/"such as".
- The refusal message is `NOT SAVED -- the VERIFIED RULE is missing decision-critical content ...:
  {missing strings}. Each of these must appear VERBATIM in the rule (in the THEN clause as a
  directive ...)`.
- `MAX_RULE_REJECTIONS = 2`; after that the rule is accepted with its gaps and a warning is logged.

**Merge / contradiction / update policy [read §3.3.3, App. A.4–A.5 "Twin situations"].**
- Verified rules are never contradicted or weakened. A conflicting experience goes into a
  separate note.
- The single permitted merge is a *branch rule*. It applies when a retrieved VERIFIED RULE covers
  a look-alike situation but prescribes a different action. The format is: `VERIFIED RULE: <the
  shared situation> -- which action applies`; content `"WHEN <shared situation>: IF <observable
  customer behaviour here> THEN <this task's verified action>; IF <behaviour from retrieved rule>
  THEN <retrieved rule's action, unchanged>." End with "Supersedes: <title of the retrieved
  rule>"`.
- Each branch must come verbatim from a verified source, and conditions must be observable.
- `memory_enrich(memory_idx, ...)` extends a retrieved memory instead of duplicating it.
- `memory_feedback` rates each retrieved insight relevant/correct, using rubrics anchored to the
  outcome (App. A.6). Examples:
  - On an instruction failure: "an insight is correct=true ONLY if it points toward the CORRECT
    solution above. An insight that endorsed what you actually did this time is correct=false".
  - "Never rate a VERIFIED RULE insight incorrect for being too strict."
- Feedback feeds a *trust estimate* which, "together with relevance, determines retrieval
  ranking" (§3.1).
- Server-side curation (§3.1, proprietary, described only in prose):
  - a safety filter and a quality gate;
  - deduplication that folds restatements into an existing insight as "supporting evidence";
  - continuous pruning of decayed insights and reconciliation of conflicts;
  - synthesis operators that abstract over clusters.

  None of this is specified in detail and none is in the repo **[read: code only calls the MCP
  API]**.
- Mid-conversation writes are "acknowledged but not committed" (§3.3.3), so a confident mistake
  cannot poison the store.

**Retrieval and injection [read §3.3.3 "Read path", App. A.1–A.3; README "Memory settings"].**
- The paper runs use `SPARK_RETRIEVAL_MODE=nudge`, which is **agent-driven retrieval**. The
  harness never injects anything.
- The system policy advertises `memory_search` next to `KB_search` (A.1).
- A one-time `<memory_check>` reminder is appended to the first customer message (A.3):

  > After you've collected all the data, and before you respond to the customer, check your
  > experience memory. Call the `memory_search` tool with a query describing this customer's
  > request and your intended solution ... If the situation changes later in the conversation (a
  > new procedure, code, or request appears), call `memory_search` again with the new specifics
  > before acting on it. Before you commit to a recommendation or perform a consequential account
  > action, re-read any VERIFIED RULE from your memory search results: follow it, or state
  > explicitly why it does not apply to this customer before deviating.

- So the **query is written by the agent** and describes the request *plus its intended
  solution*, after it has gathered data. It is not the raw opening message.
- The agent re-queries whenever the situation changes, and memories are matched against their
  stored `query` hook, which is phrased as a future agent's mid-task question.
- The code comment (`spark_agent.py` ~line 474) gives the reason **[read]**: "the store's one
  bypass-code lesson was unreachable by construction, because the single start-of-task search used
  the opening request's wording while the decisive situation ... appeared many turns later." A
  second failure: a correct VERIFIED RULE "sat at the top of its retrieval and it still recommended
  a different card", which is why the decision-point re-read cue exists.
- The code also has an `inject` mode: the harness searches with the first user message and folds
  the results into that turn inside `<retrieved_memory>`, framed as "NOT from the customer and may
  be outdated or incorrect — the policy and the knowledge base are authoritative". It is the code
  default, but **it was not the configuration of the paper runs** [read README].
- Top-k is decided server-side and is not stated anywhere **[read: not specified]**.
- Memory is framed as subordinate to the KB on any conflict.

### 1.2 Experimental methodology

- **Tasks [read §3.2, §3.3.4]:**
  - All 97 τ-bench banking tasks (v1.0.1). Grading is deterministic, except for one task.
  - k=4 trials per task, 388 conversations per condition.
  - **Trial-major scheduling**: every task runs trial 1 before any task runs trial 2.
  - Each condition gets its own fresh store, starting empty. The baseline touches no store.
- **There is no train/test split.** Learning is measured *within task across repeated trials*:
  repetition with variation, because a stochastic GPT-5.2 user simulator plays the customer
  (§5 "Learning or answer caching?").
  - They state outright that the design "does not measure ... generalisation to unseen task types"
    (§5).
  - Cross-task coupling through the shared store is noted to "appear weak" (§5 Limitations).
- **Models:** Mistral Large 2512 runs baseline, experience and instruction. Claude Sonnet 5
  (medium effort) runs baseline and instruction.
- **Baseline:** the stock agent with BM25 RAG over the whole policy corpus (static RAG).
- **Metrics [read §3.3.4, App. E]:**
  - pass^k with the unbiased estimator C(c,k)/C(n,k).
  - Per-trial success curve: pooled success at trial index t.
  - **Hold rate** = P(pass at t+1 | pass at t), pooled over all transitions. This is the
    operational *retention* measure, and its complement is the regression rate.
  - **Floor stratum** = tasks the baseline never solves in any of 4 trials. **Competent stratum** =
    the rest. Labels are fixed mechanically from the baseline before the memory groups are examined.
  - **Conversion** = a floor task solved in at least one trial.
  - For no-memory or frozen-store conditions, hold measures "re-draw persistence"
    (≈ Σp_i²/Σp_i) and serves as the reference level (App. D).
- **Statistics [read App. E]:**
  - Cluster bootstrap over tasks, B = 10,000, percentile intervals.
  - The same resampled task indices are used for every condition in a replicate, so contrasts are
    paired.
  - Replicates with an undefined hold rate are excluded and counted.
  - The seed is fixed and published.
  - **One run per condition** (§5 Limitations), so intervals reflect task sampling only.
- **Transfer experiment [read §3.4, App. B]:**
  - Each model reads the other model's instruction-built store, frozen at the end of the run.
    Consumers are read-only (search tool only; no writes, feedback or reflection).
  - Store integrity (insight count, latest write timestamp) is verified before and after each
    cell.
  - Transfer stratum = the consumer's floor ∩ the producer's coverage (tasks the producer solved at
    least once). It is pre-registered before the consumer cells run (`data/strata.md`).

**Key results [read Table 1, §4, App. D].**

| Condition | pass^1 | pass^4 | Solved | Hold |
|---|---|---|---|---|
| Mistral baseline | 0.064 | 0.031 | 13/97 | 0.61 (re-draw) |
| Mistral experience | 0.103 | 0.031 | 16/97 | **0.88** [0.75, 1.00] |
| Mistral instruction | 0.170 | 0.031 | 32/97 | 0.65 [0.47, 0.80] |
| Sonnet baseline | 0.248 | 0.115 | 40/97 | 0.70 (re-draw) |
| Sonnet instruction | 0.397 | 0.165 | 62/97 | 0.83 |
| Mistral reads Sonnet store | 0.289 | 0.093 | 51/97 | 0.65 |
| Sonnet reads Mistral store | 0.314 | 0.165 | 46/97 | 0.78 |

- **Paired pass^1 contrasts:** I−B +0.106 [0.057, 0.157]; E−B +0.039 [0.008, 0.072]; I−E +0.067
  [0.023, 0.111]. Sonnet I−B +0.149 [0.088, 0.211].
- **Floor conversions (Mistral):** instruction 22/84, experience 5/84. I−E conversion +0.202
  [0.119, 0.291]. Sonnet instruction converts 28/57.
- **Cold first trial is at parity:** 6, 5 and 9 tasks solved. The gains appear only between trials
  (§4.1).
- **Retention:** experience holds 0.88 against instruction's 0.65, contrast −0.23 [−0.47, −0.02]
  (§4.1).
- **Competent stratum (negative-transfer check, §5):** of the 13 tasks the baseline solves,
  experience keeps 11 and instruction 10, "within what re-draw noise alone would produce".
- **Transfer:** Mistral +0.224 [0.160, 0.289] and Sonnet +0.066 [0.018, 0.119]. Conditional
  conversion on the transfer stratum is about 31% in both directions.
- **pass^4 cannot see learning:** a task with grid 0111 contributes 0 at pass^4, so hold rate and
  the per-trial curve are the right measures (§5).

---

## Paper 2: "When Continual Learning Moves to Memory" (Hu, Long, Wang; arXiv 2604.27003)

### 2.1 Method

**Framing [read §1, §2.1, Table 1].** With a frozen backbone the interference moves from weight
overwrites to *retrieval*. They name three failure modes:
- *retrieval pollution*: irrelevant memories are recalled;
- *context competition*: useful items are displaced within a finite context budget;
- *memory dilution*: the right item gets harder to find as the store grows.

**Loop [read §2.1, App. A].**
- A ReAct agent runs on AgentGym with the ReMe memory module.
- Per instance: retrieve → insert into context **before the first action step** → act → store the
  experience. Both successes and failures are stored ("Success-only filtering: false", Table 8).
- Retrieval is **BM25 throughout** (§A.1.3).
- Backbone Qwen-Plus, temperature 0.5, 4096 max tokens, seed 42 (Table 8).

**(k,v) decomposition [read §2.2].** v is the representation (Study 1). k is the organisation:
storage granularity and retrieval frequency (Study 2).

**Conditions [read §3, §4, §A.1.4–A.1.5, App. B].**

| Name | Value | Key | When retrieved | top-k |
|---|---|---|---|---|
| Raw | full action–observation trajectory | task instruction + initial observation | once, at episode start | 1 |
| Insight (= Cond-Agg) | about 3 LLM-distilled "abstract strategy insights" bundled into one unit | task instruction (+ initial observation) | once, at episode start | 1 |
| Cond-Ind | each insight a separate unit | a **`when_to_use` descriptor** generated alongside the insight | once, at episode start | 3 |
| Cond-Step | same as Cond-Ind | `when_to_use` | **re-query every 4 environment steps** | 3 |

All Study 2 conditions surface about 3 insights per retrieval event, which keeps the budget
roughly constant (§4, §A.1.5).

**Distillation prompt:** *not given in the paper* **[read: absent]**. It is ReMe's insight
extractor. Example output (App. B.1):

> "Insight 1. Check countertops and tables first, as they are the most common locations for
> portable objects like bowls and cups. Insight 2. If the target object is not visible,
> systematically visit receptacles in order of likelihood ... Insight 3. Once the object is picked
> up, navigate directly to the target receptacle ..."

Cond-Ind key examples (App. B.2):
- "When the target object is absent from all observations after multiple reorientations, and
  multiple closed doors are present" → "Prioritize attempting all available closed-door
  traversals ..."
- "After entering an empty room with no objects or doors visible" → "... backtracking should be
  initiated immediately".

**[inf]** The exact prompt would have to come from the ReMe repository (modelscope/ReMe). I did
not fetch it.

**Update, merge, deprecate:** the paper does not discuss any of these. The pool only grows. No
deduplication, utility scoring or pruning is described **[read: absent; inf: ReMe's defaults may
include some, but the paper attributes none of its results to it]**.

### 2.2 Experimental methodology

- **Task pairs [read §2.4, Table 2]:**
  - ALFWorld. A = pick_and_place ("put X in Y"), baseline 95%. B = pick_clean_place ("put a clean
    X in Y"), baseline 54%. The instructions are lexically near-identical, but the procedures
    differ.
  - BabyAI. A = action_obj_door (3 heterogeneous subtypes), baseline 70%. B = find_obj
    (homogeneous "find [object]"), baseline 49%.
- **Protocol [read §2.1, Fig. 1, §A.1.6, Table 8]:**
  - Two sequential phases, 200 *training* instances per phase, streamed online. Memory starts
    empty and grows across the phase switch.
  - A **held-out test set of 100 instances per task**.
  - One evaluation checkpoint at the end of each phase.
  - **2 independent runs per condition, averaged.** No intervals or significance tests are
    reported **[read: none present]**.
- **Runs needed per sequence [read §A.1.1]:**
  - no-memory baseline;
  - *scratch* (memory-enabled, trained on one task only, ∅→T);
  - *cross-task* (∅→X→T);
  - *eval-only probe* (re-evaluating the earlier task after the full sequence).
  - Both directions (A→B and B→A).
- **Metrics [read §2.3]:**
  - **FWT(A→B) = Acc_B(∅→A→B) − Acc_B(∅→B)**, where ∅→B is the memory-enabled scratch run, not the
    no-memory baseline.
  - **BWT(A→B) = Acc_A(∅→A→B) − Acc_A(∅→A)**.
  - **RR/NL diagnostic.** Split the test set by the *no-memory baseline* into a baseline-success
    subset (easy) and a baseline-fail subset (hard).
    - RR = success on the easy subset; NL = success on the hard subset.
    - ΔRR = RR(∅→X→T) − RR(∅→T); ΔNL = NL(∅→X→T) − NL(∅→T).
    - ΔRR < 0 is *harmful reuse* (breaks solved cases). ΔNL < 0 is *ineffective reuse*.
    - NL is omitted when the hard subset has fewer than 10 items (ALFWorld A has 5).
  - **Retrieval diversity diagnostics [read App. D, Fig. 11]:**
    - key diversity (TF-IDF dissimilarity of stored keys) vs **context diversity** (of the items
      actually retrieved over all test queries);
    - **retrieval coverage** = unique retrieved items / all retrievals;
    - **top-1 concentration** = the share of queries whose top-1 is the single most common item.
- **Learning dynamics [read App. E]:** cumulative training success curves; RR/NL tracked over
  training.

### 2.3 Key results

**Study 1, forward transfer (Table 3):**

| Env | Direction | Raw FWT | Insight FWT |
|---|---|---|---|
| ALFWorld | A→B | **−9.5** (ΔRR +4.6, ΔNL **−26.1**) | **+6.5** (ΔRR +9.3, ΔNL +3.3) |
| BabyAI | A→B | −7.5 (ΔNL −12.7) | +9.0 (ΔNL +9.8) |
| ALFWorld | B→A | −0.5 | −2.5 |
| BabyAI | B→A | +7.0 | +7.5 |

- Transferring from the easier to the harder task is where raw trajectories hurt (§3.1).
- **Negative transfer lands on hard cases** (§3.2). Case F.1: a retrieved Task A bowl trajectory
  leads the agent to reason "it's reasonable to assume it's clean" and skip cleaning.
- **Within-task advantage does not predict cross-task benefit** (§3.3, Fig. 4). Raw is far better
  within-task on ALFWorld B, but that advantage vanishes or reverses across tasks.

**Study 1, backward transfer (Table 4):**
- ALFWorld B→A: Raw −5.5, Insight −3.5.
- BabyAI B→A: Raw −4.0, Insight **+3.5**.
- The authors' reading: abstraction reduces forgetting (§3.4).

**Study 2, forward transfer (Table 5):**
- BabyAI A→B: Ind **+15.0** (ΔRR +19.4, ΔNL +10.8) > Agg +9.0 > Step −2.5.
- BabyAI B→A: Agg +7.5 = Step +7.5 > Ind −1.0.
- ALFWorld A→B: Agg +6.5 > Step −1.5 > Ind −2.0.
- No configuration dominates (§4, Fig. 7).

**Study 2, backward transfer (Table 6):**
- BabyAI A→B: **Cond-Ind BWT −10.0** (ΔRR −6.4). This is the run with the best FWT, which is the
  stability–plasticity trade-off "through memory dilution" (§4.3, Case 6).
- Cond-Step +3.0; Agg 0.0.

**Retrieval diversity collapse (App. D, Fig. 11):**
- Under Cond-Ind on homogeneous Task B, key diversity is >0.91 but context diversity is 0.38.
- Only 13 unique items appear among 240 retrievals (5.4% coverage).
- **88.5% of queries return the same top-1 item.**
- Within-task performance is −17 pp.
- Case F.4: the most common top-1 item is returned for 82 of 200 queries.

**Retrieval frequency (§4.2):** periodic re-query helps when a task has distinct phases and adds
noise when execution is uniform. Step reaches 79.5% against Ind's 55.0% *within-task* on ALFWorld
B, but the gain does not transfer.

### 2.4 How they measure negative transfer, interference and forgetting, and what to copy

**What they measure [read].**
- *Negative transfer:* FWT < 0 against a memory-enabled **scratch** control, decomposed by
  ΔRR/ΔNL on subsets that the no-memory baseline defines.
- *Forgetting:* BWT < 0 from an eval-only probe of the earlier task after the later phase,
  against the scratch-on-earlier-task control.
- *Interference / competition mechanism:* measured at retrieval:
  - coverage and top-1 concentration;
  - key vs context diversity;
  - qualitative inspection of retrieved items (App. F).

  Plus RR/NL learning curves over training (App. E).

**Mitigations found [read].**
- Abstract insights instead of raw trajectories: safer FWT and less forgetting.
- **Aggregate (bundled) units keep the pool compact** and resist dilution.
- Fine-grained units only help when the source insights are genuinely diverse.
- Step-level re-query can partly compensate for dilution, because it surfaces different items
  over the course of an episode.
- They propose no explicit dedup, utility weighting or pruning.

**Protocol to copy for our H5 (retention) [inf, adapting their definitions].**
1. **Apply BWT to our retention panel.** At each checkpoint c, compute
   `BWT_c = Acc_retention(snapshot_c) − Acc_retention(snapshot_at_first_learn)`, where
   "first learn" is the checkpoint right after the panel's cells were trained. For a per-cause
   view, compute it per cause: train on cause X, then on cause Y, then re-probe X.
2. **ΔRR/ΔNL on every evaluation panel.** Split each eval panel by **group A's** outcome on the same
   tasks (A-pass = easy, A-fail = hard), with labels fixed from A before B is examined, as both
   papers do.
   - Report ΔRR (B − A on the A-pass subset) as *harmful reuse*. Exp1's integrity breaks fall
     exactly here.
   - Report ΔNL as *new learning*.
   - This needs A run on the identical tasks, which SPEC §11 already does.
3. **Hold rate (from paper 1) for repeated measurement of the retention panel.** P(pass at
   checkpoint c+1 | pass at c) per task, against group A's re-draw persistence Σp²/Σp as the
   reference.
4. **Log retrieval diagnostics for every recall:**
   - memory ids returned and their rank;
   - coverage (unique ids / total);
   - top-1 concentration;
   - the share of recalls where a memory from the *same cause* is in the top-k;
   - "lesson vs restatement" labels.

   This is how they localise dilution, and it is directly relevant to our "lesson surfaced in 1 of
   10" finding.

**Protocol to copy for H3 (transfer) [inf].**
1. Define FWT against a **memory-enabled scratch control**, not just against A. For
   `sealed_novel_cause` cause Y, `FWT = Acc_Y(bank trained on train causes, then Y episodes) −
   Acc_Y(empty bank, Y episodes only)`. In a frozen-eval design the simpler `Acc_Y(B frozen
   snapshot) − Acc_Y(A)` is paper 1's "consumer vs own no-memory baseline". Report both if we can
   afford it.
2. Run **both directions** for at least one cause pair (train on X then Y, and Y then X), since
   their effects are strongly directional, from easier to harder (§3.1, F.2).
3. **Pre-register the strata.** "Transfer stratum" = sealed tasks A never solves whose cause was
   solved at least once in training (paper 1 App. B). Commit it before the B cells run.
4. **Report within-task gains (sealed_familiar) separately from cross-task gains
   (novel_combo / novel_cause).** Paper 2 §3.3 shows the two can rank designs oppositely.

---

## 3. Gap analysis against our harness

What we have **[read: SPEC §10, `elh/memory/hindsight.py`]**:
- One lesson-v2 record per train episode: outcome first, failed and passed checks, timing against
  the budget, the agent's summary, a diff capped at 80 lines.
- The ticket goes in Hindsight's extraction `context`.
- Hindsight's LLM extracts facts.
- Recall uses `query = task.prompt` (the ticket), `max_tokens = 2048`, top-5, injected once at
  episode start.
- No utility signal, no rule structure, no merge policy.
- Eval panels: checkpoint, retention and sealed against read-only snapshots.

### 3.1 Why our lessons do not surface (both papers bear on this)

- **Neither paper retrieves with the raw task text against a free-text episode record.**
  - Paper 1 matches an **agent-authored query** ("this customer's request **and your intended
    solution**", issued after data gathering and re-issued when the situation changes) against a
    **stored `query` hook** written as the future agent's mid-task question.
  - Paper 2's best general setting either keys a bundle **by task instruction** (Agg, top-1) or
    keys each insight **by a `when_to_use` applicability descriptor** (Ind, top-3).
- **Neither injects globally.** Both are similarity-retrieved. Paper 1 is agent-pulled; paper 2 is
  harness-pushed at step 0 (and every 4 steps under Step).
- **[inf] Our mismatch is query/key vocabulary.**
  - Our query is a ticket, written in theme vocabulary (inventory, telemetry...).
  - Our stored units are LLM-extracted facts from an outcome narrative and a diff.
  - The ticket sits only in `context`, so the facts do not carry the ticket's surface words. That
    was intended, to avoid restatements.
  - The cause vocabulary ("list membership in a loop → set") appears in the diff and the lesson
    but *not in the ticket*, which describes symptoms.
  - So similarity ranks restatement-like or theme-matching facts above the lesson. This is paper
    2's "retrieval pollution" plus "diversity collapse" (similar tickets → the same few items).
  - **Measure this:** log ids and ranks per recall, and compute coverage and top-1 concentration
    (paper 2 App. D).
- **[inf] Both papers point at the same fix: give each memory a *retrieval key in the query's
  language*, separate from its *content*.**
  - The key should describe the **observable situation** (paper 1's WHEN clause; paper 2's
    `when_to_use`).
  - For us that means the symptom as it appears in a ticket or profile ("slow on large input;
    nested loop over a list with `in` checks"), not the theme.

### 3.2 Changes to be faithful to paper 1 (Learning on the Job)

1. **Record format → `rule-v1`.** One required rule per train episode, plus optional distinct
   lessons. Fields:
   - `title`: `VERIFIED RULE: …` on PASS; `AVOID: …` on FAIL. We are in the "experience" grade,
     since we have the verdict and check detail but not the reference solution; see item 4.
   - `content`: one sentence, `WHEN <observable situation: ticket symptom + what the agent observed
     when profiling/reading code that distinguishes this from look-alikes> THEN <decisive change,
     with verbatim decision-critical identifiers/constructs>`, or `do NOT <exact action> --
     verified wrong against the judge`.
   - `query`: "what a future agent would search", written by the reflector.
   - tags: family, and cause **only if the agent itself names it [inf: the cause label is hidden
     from the agent, so tagging with the true cause would leak oracle-side information; do not]**.
2. **Who writes it.** The *agent model itself*, in a post-episode reflection turn with its own
   conversation as history, as in A.4. Do not let Hindsight's fact extraction paraphrase it.
   - **[inf]** Store the rule verbatim. Hindsight's retain runs LLM extraction; check whether
     Hindsight can store a pre-formed unit without re-extraction (e.g. as a "mental model" or
     verbatim document). If it cannot, a thin store of our own is more faithful.
   - **Done (2026-10-25, `rules.store: verbatim`, first used by exp10).** exp8b/exp9 put the rule
     inside the lesson-v3 record and extraction paraphrased it (0 of 587 exp8b recall candidates
     contained "AVOID", "VERIFIED" or "WHEN"). Now the record is retained without the rule and
     the rule is its own retain item, `<VERIFIED RULE|AVOID>: WHEN … THEN …` plus `Situation:
     <query>`, under a bank retain strategy with `retain_extraction_mode: chunks` (Hindsight
     1.1.0 stores each chunk as one `world` unit, no LLM call), tagged `rule`, document
     `<episode>:rule`. Recall reserves `rules.recall_k` (2) slots for a rule-only recall
     (tags `rule`, `any_strict`); `search_memory` returns rule hits first.
     **Remaining deviation:** the paper retrieves rules only through the agent's own queries;
     we also recall them by the ticket text at episode start (and the agent's queries hit the
     rule text plus its situation key, not the key alone). The stored text is `render_rule`
     (WHEN/THEN with its head), not the agent's `title` line.
   - Charge the reflection tokens as `learning`.
3. **Credit assignment from the harness.**
   - On PASS: give the reflection the list of *consequential actions* (file edits / the diff
     hunks) so it credits the decisive change rather than the last one.
   - On FAIL: give it the failed checks, which lesson-v2 already does. Under `integrity` failure,
     the AVOID rule must name the protected files changed. This matches the validator's role.
4. **Instruction grade (optional group/ablation).** Paper 1's largest gain (2.6×) comes from handing
   the *verified solution* to the reflection after a failure. For us that would be the reference
   diff after a failed train episode. **[inf]** This conflicts with "never the oracle" in SPEC §10,
   so treat it as a separate pre-registered condition ("B-instr"), never the default group B.
5. **Write-time validator.**
   - Reject a rule whose THEN clause lacks the decision-critical strings drawn from the
     harness-known facts: e.g. the names of the protected files touched, or the function/construct
     changed in the diff.
   - Allow at most 2 retries with the exact missing strings.
   - Reject placeholders and "e.g.".
6. **Never contradict verified rules; branch-merge twins.**
   - The reflection receives the rules recalled in this episode. On conflict it writes a separate
     note or the single branch rule `WHEN …: IF <obs> THEN a; IF <obs'> THEN b. Supersedes: …`.
   - **[inf]** Hindsight's background consolidation may rewrite or merge memories on its own. That
     breaks "never weaken a VERIFIED RULE", so either disable consolidation for rule units or log
     and diff the bank to detect it.
7. **Retrieval: agent-driven, mid-task, decision-point** (paper 1's actual configuration).
   - Expose `memory_search(query)` as a tool. Inject paper 1's A.3-style reminder once:
     - search *after* inspecting/profiling the code, with a query describing the problem and the
       intended fix;
     - re-search when a new situation appears;
     - before `submit`, re-read any VERIFIED RULE.
   - **[inf]** For group B as specified (a harness-injected top-5 at the start), add a variant
     "B-nudge" rather than replacing B.
   - Note that agent-driven search adds model tokens to the *episode*, not to learning, which
     affects cost-per-success accounting.
8. **Feedback / trust (this is group C territory).**
   - Paper 1's `memory_feedback`, with outcome-anchored rubrics (A.6), feeds a trust score used in
     ranking. That maps onto our UtilityLearner's Q.
   - **[inf]** Paper 1 shows the signal can be the agent's outcome-anchored per-item rating instead
     of a learned Q. Consider it as C's update rule or as an ablation.
9. **Evaluation protocol.**
   - Adopt **hold rate** and the **floor/competent strata** (labels fixed from group A).
   - Use a **cluster bootstrap over tasks with paired resampling across groups**, B = 10,000, and a
     published seed. SPEC §12 already has a hierarchical bootstrap; add the pairing.
   - Report pass^1 and the per-checkpoint curve. pass^k is not useful under learning.
   - **[inf]** Paper 1's within-task repeated-trial design does not match our sealed-split design.
     Its closest analogue for us is the retention panel (repeat the *same* tasks across
     checkpoints), so hold rate belongs to H5, not H3.

### 3.3 Changes to be faithful to paper 2 (Moves to Memory)

1. **Representation.** Store abstract insights, not trajectories. lesson-v2 includes the raw diff
   (up to 80 lines), which is "Raw"-like content. That is the representation paper 2 shows causes
   negative cross-task FWT, concentrated on hard cases.
   - **[inf]** For H3 (novel cause/combination), ablate `diff` in vs out of the record.
2. **Organisation conditions to reproduce (one ablation each):**
   - *Agg*: one bundle of about 3 insights per episode, key = the ticket text, top-1. **[inf]**
     Ironically, with ticket-keyed bundles, ticket-similarity recall becomes appropriate.
   - *Ind*: each insight a unit with a `when_to_use` key, top-3.
   - *Step*: Ind plus re-query every N agent turns (they use 4 env steps), with the query built
     from the recent trajectory.
   - Hold the budget at about 3 insights per retrieval event across conditions. We use top-5 at
     2048 tokens, so match it.
3. **Retrieval.** They use BM25 (lexical). Our Hindsight uses an e5-small embedding plus a
   MiniLM cross-encoder plus its own graph. **[inf]** A BM25-over-keys baseline would be cheap and
   faithful, and it isolates whether Hindsight's semantic recall is part of the failure.
4. **Evaluation protocol (H3/H5).**
   - Add **scratch controls**: a memory-enabled run on target task T from an empty bank.
   - Add **eval-only probes** of earlier causes after later training.
   - Report **FWT, BWT, ΔRR, ΔNL** (labels from group A), plus **coverage, top-1 concentration and
     key-vs-context diversity** of retrievals.
   - Run sequences in both directions for at least one cause pair.
   - They used 2 runs; our histories play the same role. Keep intervals, which they did not
     report.
5. **Dilution watch.** cpu_mem tickets are templated and homogeneous within a cause, which is
   exactly the BabyAI-B regime where Ind collapses (88.5% identical top-1).
   - **[inf]** Expect collapse onto a few memories as the bank grows. Monitor coverage per
     checkpoint, and prefer compact/aggregated units or a dedup step.

### 3.4 Directly actionable for the 1-in-10 recall finding (ranked) [inf]

1. **Instrument first.** For each eval episode, log the recalled ids, ranks and texts. Also
   record whether a lesson for the *same cause* exists in the bank and at what rank it sits under
   the ticket query; use a larger k such as 50 to find its rank. Compute coverage and top-1
   concentration. This separates "not retrieved" from "retrieved but ignored". Paper 1 saw the
   second case too: a correct rule was at the top and the agent still deviated.
2. **Add a situation key at write time** (`when_to_use` / `query`) in symptom vocabulary, and
   retrieve against keys, not extracted facts.
3. **Move the query later or let the agent write it:** after reading and profiling, use "problem
   + intended fix" (paper 1 A.3), or re-query mid-episode (paper 2 Step).
4. **Store pre-formed rules verbatim** instead of letting Hindsight re-extract them. Done as
   `rules.store: verbatim` (see §3.2 item 2); exp8b/exp9 used the extracted form.
5. **Add a decision-point reminder before `submit`**: re-read VERIFIED RULEs.
6. **Evaluate with ΔRR/ΔNL against A**, not only the aggregate (9/10 vs 9/10 gives no power). A
   9/10 baseline leaves only 1 hard task, so per paper 2 we need far larger panels, or harder
   tasks, to see NL at all. Paper 2 omits NL with fewer than 10 hard items.
