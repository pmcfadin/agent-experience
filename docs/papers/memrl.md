# MemRL: implementation-grade notes for ELH Group C

Sources read (2026-10-25):
- Paper: arXiv 2601.03192 **v2** (12 Feb 2026), full HTML (main text and Appendices A–I). Cited as `§x.y`, `Eq. n`, `Table n`, `App. X`.
- Code: `github.com/MemTensor/MemRL` at commit `c1b322ca43de` (2026-07-18), cloned to
  `scratchpad/memrl-repo`. Cited as `path:line`.
- Our harness: `SPEC.md`, `elh/memory/hindsight.py`, `elh/memory/base.py`, `elh/controller.py`.
- Our Hindsight 1.1.0 server: `GET /openapi.json` plus one read-only recall against snapshot bank
  `elh-phase0-local-exp1-memory-helps-r2-v6-10112455fc-h0-B-e15`, and a read-only lookup in `runs/elh.sqlite3`.

Labels: **[READ]** means stated in the paper or code. **[INFERRED]** means my reading or a
calculation. **[OBSERVED]** means I ran it against our server.

---

## 0. TL;DR

- A memory is a triplet (intent `z` = task-description embedding, experience `e` = an
  LLM-processed record of the episode, utility `Q` = a scalar). In the code, `Q` belongs to the
  memory alone, not to a (state, memory) pair.
- Retrieval works in two phases. **Phase A**: cosine similarity of the query against stored
  *intents* (task descriptions, not the content), filtered by `sim > δ`, then the top-k1 intents.
  **Phase B**: score = `0.5·z(sim) + 0.5·z(Q)`, then the top-k2. Similarity is z-scored with
  **fixed corpus constants** (the mean and std of pairwise task similarity). Q is z-scored
  **within the candidate set**, with std clipped and the result clipped to ±3. The code also has
  ε-greedy, which the paper never mentions.
- Update: after a verified outcome, **every memory injected into that episode** gets the
  Monte-Carlo EMA update `Q ← Q + α(r − Q)`, with α = 0.3. In the code the reward is binary: ±1 in
  BCB/HLE/ALFWorld, 1/0 in LLB. New memories start at Q = 0 per the paper; the code differs by benchmark.
- Write path: every train episode, success or failure, writes one new memory. Successes get an
  LLM "high-level script" plus the raw trajectory. Failures get an LLM reflection plus the failed
  trajectory. There is no dedup or merge, except that a new memory is filed under an existing intent
  key when that key's similarity is ≥ 0.9.
- Evidence: a single run per configuration, with no seeds, CIs or significance tests. The gains
  over MemP are small (+3.8 pp CSR runtime, +2.8 pp transfer). Most of the "runtime" gain comes
  from re-solving **the same tasks** over 10 epochs. Our setting resembles their *Transfer*
  setting, where the gains are smallest.
- For our harness: Hindsight recall **does** return candidates with ids, `document_id` and
  per-stage scores. The `document_id` (= our episode_id) **survives snapshot restore** [OBSERVED].
  That makes a doc-level Q table outside Hindsight, keyed by episode_id, workable.

---

## 1. Memory item structure

### Paper
- `M = {(z_i, e_i, Q_i)}` (Eq. 5, §4.1). `z_i` is the **Intent**: "the User Intent, encapsulated by
  the current query embedding" (§3.2). `e_i` is the raw experience, "e.g., a successful solution
  trace or trajectory". `Q_i` "approximates the expected return of applying experience e_i to
  intents similar to z_i" (§4.1). [READ]
- New triplet `(z, e_new, Q_init)` is written "for each sampled trajectory", with `e_new` an LLM
  summary (§4.3). [READ]
- The paper writes `Q(s,m)`, but in practice it is per-memory: `E[Q(m)] = Σ_{s∈S(m)} E[r|s,m]·Pr(s|m)`
  (Eq. 9/14, App. A.1). The state dependence comes only through the Phase-A gate. [READ]

### Code (a MemOS `TextualMemoryItem`)
- `memory` = **task_description** (the retrieval key, embedded) (`memrl/service/memory_service.py:667-670`,
  `memrl/service/updater.py:234-243`). Task text is truncated to 4096 chars before write
  (`memory_service.py:1603`).
- `metadata` fields written (`memory_service.py:638-666`, `1571-1600`; `updater.py:46-68`):
  - `type`: `"procedure"` for a success or `"adjustment"` for a failure reflection. `source`, `source_benchmark`.
  - `success`: bool.
  - `full_content`: **the experience text that gets injected** (format in §4).
  - `q_value`: float. `q_visits`: int. `q_updated_at`, `last_used_at`. `reward_ma`: EMA of reward.
    `last_reward` is added on update (`memrl/service/value_driven.py:204-216`).
  - `related_memory_ids`: the ids retrieved for the episode that produced this memory (`memory_service.py:1564-1567`).
  - `confidence` (100, or ×0.8 for reflections; unused by retrieval), `strategy_*`, `task_description`,
    `updated_at`, `memory_time`.
  - Runner-specific: `task_id`, `outcome`, `eval_status`, `eval_error`, `bcb_epoch`, `phase`, `model`
    (`memrl/run/bcb_runner.py:567-584`); `sample_index`, `lb_epoch` (`memrl/run/llb_rl_runner.py:1226-1250`).
- **Intent index** (outside MemOS): `dict_memory: {intent_text → [mem_id, ...]}` and
  `query_embeddings: {intent_text → vector}` (`memory_service.py:436-437`, `1637-1657`). One intent key
  can hold **many** memories (one per attempt at that task, across epochs).
- Q is also cached in `_q_cache {mem_id: q}`, which takes precedence over metadata at retrieval
  (`memory_service.py:1378-1402`, `843`).

---

## 2. Retrieval algorithm

### Paper (§4.2)
1. **Phase A**: `C(s) = TopK_{k1}({ i | sim(Emb(s), Emb(z_i)) > δ })` (Eq. 6). If `C(s)=∅`, the
   frozen LLM runs with no memory.
2. **Phase B**: `score = (1−λ)·ŝim + λ·Q̂`, where `^` is z-score normalization; then the top-k2 (Eq. 7).
3. In theory (App. A.4.3) the optimal μ is a Boltzmann distribution `∝ π_sim·exp(βQ)`. The
   implemented rule is the deterministic log-linear score (Eq. 7). [READ]

Hyperparameters (Table 8, App. E.2) [READ]:

| | BCB | LLB-OS | LLB-DB | ALFWorld | HLE |
|---|---|---|---|---|---|
| α | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| λ | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| δ | 0.38 | 0.50 | 0.37 | 0.62 | 0.25 |
| k1 | 10 | 10 | 10 | 5 | 5 |
| k2 | 5 | 5 | 5 | 3 | 3 |
| Q_init | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

- δ is chosen as "the top 20% quantile" of the pairwise cosine similarity distribution of task
  descriptions within each benchmark (App. E.2). [READ]
- Embedder: OpenAI `text-embedding-3-large` (Table 7). [READ]

### Code: `MemoryService.retrieve_query` (`memory_service.py:1254-1506`), used by all four runners
1. Embed the query (the task description / ticket) (`:1290`).
2. Compute cosine similarity against **every intent key** in `dict_memory` (brute force) (`:1306-1312`).
   Keep `sim >= threshold` (note `>=`; the paper writes `>`).
3. Sort descending and keep the top **k intents** (`k = memory.k_retrieve`) (`:1313-1316`). Then expand
   each intent to **all** its memory ids (`:1321-1358`). **The candidate pool can be larger than k1.** [READ]
4. For each candidate, Q = `_q_cache` or else `metadata.q_value` or else `q_init_pos` (`:1378-1390`).
   Optional `recency_boost` (0 in every config) and `q_floor` (LLB only: 0.0) (`:1404-1416`).
5. Optional hard filter `q_estimate >= q_min_threshold` (BCB −0.8, LLB −0.8, ALF/HLE −10) (`:1424-1427`).
6. Normalization:
   - `sim_z = (sim − sim_norm_mean) / sim_norm_std`, using **fixed per-benchmark constants** (`:1239-1244`).
   - `q_z = clip((q − mean_q)/std_q, −3, 3)`, with mean and **population** std over the current
     candidates. If there is a single candidate or std ≤ 1e-9, std becomes 1 (`:1246-1252`, `:1443-1447`).
     So when all candidates share a Q (for example all at Q_init), `q_z = 0` and ranking is by
     similarity only.
   - LLB can switch normalization off (`use_z_score_normalization`, `:316`).
7. `score = w_sim·sim_z + w_q·q_z`, with `w_sim = w_q = 0.5` (`:1449-1460`). Sort descending.
8. Selection of `topk` (= k2, `rl_config.topk`): with probability **ε** take `random.sample(all, k2)`;
   otherwise the top-k2 (`:1467-1473`). LLB can dedupe by `task_id` (`:1474-1493`; off by default).
9. Returns `selected` (injected), `candidates`, `simmax`, plus the `(intent, sim)` list used later by
   the write path (`:1495-1500`).

Config values (`configs/rl_*.yaml`) [READ]:

| knob | BCB | LLB | ALFWorld | HLE |
|---|---|---|---|---|
| `sim_threshold` (δ) | 0.38 | os 0.50 / db 0.37 | 0.62 | 0.25 |
| `k_retrieve` (k1, intents) | **5** | 10 | 5 | 5 |
| `topk` (k2) | 5 | 5 | 3 | 3 |
| `epsilon` | **0.1** | **0.01** | 0 | 0 |
| `sim_norm_mean / std` | 0.31 / 0.10 | 0.39 / 0.14 (os 0.3896/0.1429, db 0.2748/0.1127) | 0.5187 / 0.1203 | 0.1857 / 0.0941 |
| `q_min_threshold` | −0.8 | −0.8 (moot with `q_floor` 0) | −10 | −10 |
| `weight_sim / weight_q` | 0.5/0.5 | 0.5/0.5 | 0.5/0.5 | 0.5/0.5 |

How the constants relate [INFERRED, numbers READ]:
- `sim_norm_mean` equals the "intra-dataset semantic similarity" in App. B.3 (HLE 0.186, ALF 0.518,
  OS 0.390, BCB 0.308). So the similarity z-score uses the mean and std of **pairwise task-description
  similarity over the whole benchmark**, not of the candidate set.
- δ ≈ mean + 0.84·std, which is the 80th percentile under normality. ALF: 0.5187 + 0.84·0.1203 = 0.620 ✓.
  OS: 0.51 vs 0.50. BCB: 0.394 vs 0.38. HLE: 0.265 vs 0.25.

### Paper vs code discrepancies (retrieval)
- **ε-greedy** exists in code (BCB 0.1, LLB 0.01) and appears nowhere in the paper.
- **BCB k1**: the paper says 10; `rl_bcb_config.yaml:42` says `k_retrieve: 5`. In the code k1 counts
  *intent keys*, not memories.
- **"z-score normalization"** in the paper is unspecified. The code uses fixed corpus statistics for
  similarity and per-candidate-set statistics, clipped to ±3, for Q.
- The **`q_min_threshold` hard exclusion** is not in the paper. With ±1 rewards and α = 0.3, a memory
  that fails about 3 times in a row drops below −0.8 and is never retrieved again, so its Q can never
  recover (an absorbing state) [INFERRED].
- `ValueAwareSelector` (sort by Q only, `value_driven.py:81-152`) and `retrieve_value_aware`
  (`memory_service.py:678-711`) are **dead code**: no runner calls them [READ, grep].
- Backbone: the paper lists GPT-4o for BCB (Table 7); `rl_bcb_config.yaml:21` has `gpt-4o-mini`.

---

## 3. Utility (Q) update

### Paper
- `Q_new ← Q_old + α(r − Q_old)` (Eq. 4), a Monte-Carlo / one-step-terminal special case of the TD
  rule (Eq. 3) with γ = 0 (§3.2). [READ]
- **Which memories**: "For the memories actually injected into the input context M_ctx(s), we
  update their utilities" (§4.3). That is **all k2 selected**, with the same reward and no
  per-memory credit. The authors flag this as a limitation (§6, App. G.2). [READ]
- Reward: "environmental reward r (e.g., execution success or scalar score)" (§4.3). The analysis
  assumes `r_t ∈ [−1, 1]` (App. A.1). [READ]
- Theory: `E[Q_t] → β(s,m)` at rate `(1−α)^t` (Eq. 11-12), and
  `limsup Var(Q) ≤ α/(2−α)·Var(r)` (Eq. 13/18). For α = 0.3 that is 0.176·Var(r). [READ]
- α = 0.3, Q_init = 0.0 on every benchmark (Table 8). [READ]

### Code
- `QValueUpdater.update` (`value_driven.py:181-220`):
  `target = r + γ·next_max_q` (γ = 0), `new_q = (1−α)·old_q + α·target`, optional `q_floor`;
  `q_visits += 1`; `reward_ma` EMA; timestamps; writes the metadata back. [READ]
- `MemoryService.update_values(successes, retrieved_ids_list)` (`memory_service.py:793-848`): the
  reward is **binary on truthiness**, `success_reward if success else failure_reward`
  (`:812-818`). **Scalar rewards are not supported** even though the paper mentions them. Every id
  in the episode's injected list is updated (`:819-820`). [READ]
- The ids updated are exactly the injected ones:
  - BCB: `selected_ids` (`bcb_runner.py:511-515`, `604`).
  - LLB/ALF: every memory in the success and failure buckets that was formatted into the prompt
    (`llb_rl_runner.py:1189-1197`).
  - HLE: `retrieved_ids` from `_build_memory_context` (`hle_runner.py:596-617`, `973`).
- **Timing**: Q updates are **batched**, not per episode.
  - BCB flushes every 25 tasks (`bcb_runner.py:610-612`).
  - LLB per mini-batch of 5 (`rl_llb_config.yaml:114`).
  - ALF and HLE per batch (`batch_size: 32`).
  So within a batch, retrieval sees stale Q. Order: Q update, then add new memories (BCB/LLB/ALF);
  HLE adds first, then updates (`hle_runner.py:964-973`).
- Only train phases update. BCB `val` runs with `update_memory=False` (`bcb_runner.py:716-722`).

Rewards and initialization actually used (code) [READ]:

| | success_reward | failure_reward | q_init_pos (success mem) | q_init_neg (failure mem) |
|---|---|---|---|---|
| BCB (`rl_bcb_config.yaml:104-121`) | 1.0 | **−1.0** | 0.0 | 0.0 |
| ALF (`rl_alf_config.yaml:67-83`) | 1.0 | −1.0 | 0 | 0 |
| HLE (`rl_hle_config.yaml:66-82`) | 1.0 | −1.0 | **overridden by runner: 1.0 / 0.0** (`hle_runner.py:957`) | |
| LLB (`rl_llb_config.yaml:148-169`) | 1.0 | **0.0** | **0.5** | **0.5** (+ `q_floor` 0.0) |

The paper says Q_init = 0 for all (Table 8). LLB and HLE disagree with it.

- Unused: `MemoryCurator.find_merge_target / attribute_reward` (novelty merge with
  `reward_merge_gain` 0.1) is defined (`value_driven.py:225-273`) but **never called** [READ, grep].
- LLB's `algorithm: mdp/rlmdp` calls `memory_service.update_values_chain_mdp`, which **does not
  exist** in the repo (`llb_rl_runner.py:1333`). Only `rl` works. [READ, grep]

---

## 4. Write path

### When
- After each (mini-)batch of **train** episodes, one new memory is written **per episode, success
  or failure**. Strategy `proceduralization + query + adjustment` in every config
  (`configs/rl_*.yaml`, `memory.*_strategy`). [READ]
- Flow: `add_memories` → `update_memories` → `AdjustmentUpdater.prepare_update_op`
  (`memory_service.py:1508-1665`; `updater.py:316-343`).

### What
- **Success** goes through `ProceduralizationBuilder` (`updater.py:322-334`; `memrl/service/builders.py:155-205`). Content:
  `"Task: {task_description}\n\nSCRIPT:\n{script}\n\nTRAJECTORY:\n{trajectory}"`, where the script comes from
  `generate_script` (`memrl/providers/llm.py:354-380`, temperature = the provider default, 0.7 unless
  configured), verbatim:
  ```
          Analyze the following detailed task trajectory and create a concise, 
          high-level script that captures the essential steps and decision points.
          
          The script should be:
          1. Generic enough to apply to similar tasks
          2. Specific enough to provide useful guidance
          3. 3-5 high-level steps maximum
          4. Focus on the strategy and key decisions, not detailed actions
          
          Trajectory:
          {trajectory}
          
          High-level script:
  ```
- **Failure** goes through `_generate_reflection` (`updater.py:247-267`, temperature 0.3), verbatim:
  ```
  Task: {task_description}

  Failed trajectory:
  {failed_trajectory}

  This task failed. Analyze what went wrong and suggest improvements for future similar tasks.
  Focus on:
  1. Incorrect assumptions
  2. Steps to improve
  3. What to avoid next time

  Provide a brief reflection:
  ```
  Stored content, append mode (`updater.py:345-361`):
  `"TASK REFLECTION:\nTask: {task}\n\nWhat went wrong:\n{reflection}\n\nFailed approach:\n{failed_trajectory}\n"`,
  `type="adjustment"`, confidence ×0.8.
- The summarizer LLM appears to be the same backbone as the agent (`run/run_bcb.py:211-214` passes the
  same `llm`) [INFERRED from wiring].
- What "trajectory" means per benchmark:
  - BCB: the model's raw response, or a structured step blob as fallback (`bcb_runner.py:276-326`).
  - LLB: the serialized chat.
  - HLE: `{question}\n\nSOLUTION\n\n{model_output}` (App. I.1).
- Paper App. I.1 reproduces the same templates (its HTML rendering drops the first line of each listing).
- The appendix case studies (App. H) show a different, structured reflection format:
  `[TASK] / [REFLECTION] - ROOT CAUSE / PATTERN TO AVOID / CORRECT APPROACH`. BCB *coerces* stored content
  at injection time into `[MEMORY TYPE] SUCCESS_PROCEDURE|FAILURE_REFLECTION / [TASK] / [EXECUTION
  TRAJECTORY]|[REFLECTION]` (`bcb_runner.py:220-274`). [READ]

### Injection format
- Successes and failures are split into labeled sections.
  - LLB: `=== SUCCESSFUL EXPERIENCES (Learn from these) ===` / `=== FAILED EXPERIENCES (Avoid these
    mistakes) ===`, and failure memories have the long "Failed approach:" tail **stripped**
    (`memrl/lifelongbench_eval/memory_context.py:15-87`).
  - HLE: `=== Successful Memories ===` / `=== Failed Memories (for caution) ===` (`hle_runner.py:612-616`).
- BCB: `# Relevant Code Examples from Memory` / `## Example i [OUTCOME]` (`bcb_runner.py:166-218`). The
  system prompt explains the memory types (`bcb_runner.py:41-59`).
- Memory goes into the **system** message (BCB `_generate_raw`, `bcb_runner.py:328-349`; App. I.2).

### Dedup / merge
- There is **no content dedup and no merge**.
- The only grouping is the intent key. If any of the episode's own retrieved intents had
  `sim >= add_similarity_threshold` (0.9 BCB/ALF/HLE, **0.99** LLB), the new memory id is appended
  under that existing intent key and reuses its embedding. Otherwise a new intent key is created and
  embedded (`memory_service.py:1629-1657`). [READ]
- Consequence [INFERRED]: on epoch 2+ the same task returns with sim ≈ 1, so all attempts at a task
  pile up under one key. That is how HLE's "runtime memorization" works (App. B.3).
- There is no eviction or forgetting. MemOS stores everything.

---

## 5. Experimental methodology and results

- **Benchmarks** (App. D.2, Table 9):
  - BigCodeBench-Instruct full: 1,140 tasks.
  - LifelongAgentBench OS and DB: 500 each.
  - ALFWorld: 3,553 train / 140 valid_seen.
  - HLE: 2,500, runtime only.
- **Backbones**, one per benchmark "to avoid no-signal or ceiling" (§5.1, App. E.4): GPT-4o (BCB),
  GPT-4o-mini (LLB), GPT-5-mini (ALF), Gemini-3-pro (HLE). Temperature 0, except 0.6 for HLE (Table 7).
- **Protocols**:
  - *Runtime learning*: 10 epochs **re-running the same task set**, writing memory and updating Q
    throughout. They report **last-epoch SR** and **CSR** (the share of tasks solved at least once
    across epochs) (§5.1, Table 1). Configs: `num_sections: 10`, "re-uses the same training set each
    section" (`rl_alf_config.yaml:49`).
  - *Transfer*: freeze the memory after training and test on a held-out 30% (7:3 split, seed 42; for
    ALF, 140 valid_seen instances). Results use "the best validation results" (Table 2 caption)
    [READ]. Checkpoint selection on the evaluated split is optimistic [INFERRED].
- **Baselines**: No Memory, Pass@10, RAG, Self-RAG, Mem0, MemP (App. D.1). Baseline top-k matches k2 (Table 8).
- **Statistics**: **single run per cell. No seeds across runs, no CIs, no error bars, no
  significance tests** (I searched the whole paper) [READ]. The only seed is the 7:3 split (42).
- **Key results**:
  - Runtime (Table 1), MemRL vs MemP CSR: BCB 0.627 vs 0.602, OS 0.804 vs 0.742, DB 0.972 vs 0.966,
    ALF 0.981 vs 0.919, HLE 0.606 vs 0.570. Average +3.8 pp.
  - Transfer SR (Table 2): BCB **0.508** vs MemP 0.494, Self-RAG 0.500, NoMem 0.485. OS 0.746 vs 0.720.
    DB 0.942 vs 0.935 (Mem0). ALF 0.979 vs 0.950. Average +2.8 pp over MemP.
  - GPT-4o-mini on ALF (Tables 10-11): runtime CSR 0.680 vs MemP 0.413; transfer 0.457 vs 0.314. This
    is the largest gap they report.
- **Ablations, and what mattered**:
  - RL vs no RL (Fig. 4, OS only): curves diverge over epochs, with the CSR gap widening (§5.3.1).
  - λ ∈ {0, .25, .5, .75, 1} (Fig. 5): concave, peaking at **0.5**. λ = 0 (pure similarity) plateaus.
    λ → 1 is volatile and loses context (§5.3.2). Run on an unnamed benchmark, probably OS.
  - Cross-task vs single-task reflection (Table 3): +9.0 pp on OS, +5.1 pp on ALF, and **tied on HLE**
    (0.610 vs 0.606) because HLE's intra-similarity is low (§5.3.3).
  - k1/k2 (Fig. 6, HLE CS/AI subset): 3/1 < **5/3** > 10/5. An inverted U (§5.3.4).
  - Removing **z-norm and similarity gating** raises the forgetting rate from 0.041 to 0.073
    (MemP 0.051) (§5.4.2, App. B.1).
  - Q-critic validity (Fig. 7): Pearson r = 0.861 between the Q bin and empirical success
    (21.5% → 88.1%). About 12% of memories in the high-Q bins are *failure* memories ("near-misses")
    (§5.4.1, App. H).
  - Gains are larger on multi-step tasks (ALF/OS +6.2 pp) than single-step (BCB +2.5, HLE +3.6) (Table 4).
  - Merging OS and DB banks causes negligible interference (Table 6). Cross-model transfer of an HLE
    bank to Qwen3-235B: 0.150 → 0.531 (Table 5).
  - Cost: about 32K tokens per HLE question, similar to MemP. Retrieval and update overhead is
    negligible (App. F).

---

## 6. Gap analysis: a faithful Group C in ELH

### 6.1 What exists
- `HindsightMemory.recall` sends `{"query": task.prompt, "max_tokens": 2048}` and keeps
  `results[:k]` **units**. It drops `scores`, `document_id` and `type` (`elh/memory/hindsight.py:96-98`).
  `MemoryItem` has a `score` field that is never filled (`elh/memory/base.py:11-15`).
- Retain: one document per train episode, `document_id = episode_id`, tags `family:*`, content
  `lesson-v2` (`hindsight.py:100-111`, `218-249`). Hindsight's LLM then extracts several
  world/experience units per document and consolidates observations.
- The controller recalls once before the episode, injects the units, stores
  `retrieved_memory_ids_json` (unit ids), and retains only on non-read-only splits
  (`elh/controller.py:185`, `221`, `243-249`).
- `memory_snapshots.learner_state_sha256` exists and is always `None` (`controller.py:129`; SPEC §9).
- SPEC group C: "two-phase retrieval (relevance top-N → rerank by learned Q); Q updated from verified
  reward r = passed × (1 − normalised_cost)" (SPEC.md:353). `Learner.observe/rank` (SPEC.md:341-345).
  `elh/learner/` does not exist yet.

### 6.2 What our Hindsight 1.1.0 recall gives [OBSERVED]
- `POST /v1/default/banks/{bank}/memories/recall` returns `results[]` with `id`, `type`
  (world/experience/observation), **`document_id`**, `tags`, `metadata`, and
  **`scores {final, reranker, semantic, keyword}`**. `semantic` is the vector cosine (null if the
  semantic group did not surface the result). `final` is the reranker score plus recency/temporal
  boosts. `min_scores {semantic, keyword, reranker, final}` floors exist (openapi `RecallScores`, `MinScores`).
- Probe with a generic cpu_mem ticket, `max_tokens 4096`, on the 15-episode snapshot bank: **80
  candidates** (33 experience, 30 world, 17 observation) spanning **16 distinct document_ids**.
  Observations have `document_id: null` (their `source_fact_ids` need `include.source_facts`).
- **The integrity-failure lesson was the #2 unit by `final`** (doc `7d9f3a62…`: "A coding agent failed
  the performance ticket because it modified forbidden files, specifically `run.py`"). Two more
  integrity-failure docs (`6deb4062…`, `e12031c9…`) appear lower. So the candidate pool **does contain
  the right lessons**. The problem lies in the final top-5 truncation and ordering, which Phase B is
  meant to fix.
- **document_id survives snapshot restore**: the snapshot bank's `document_id`s are the training
  episode ids. `7d9f3a62…`, `6deb4062…` and `e96b5dcf…` are group-B train episodes in `runs/elh.sqlite3`.
  The first two failed on integrity; the third passed. SPEC says unit ids **do not** survive restore
  (SPEC.md:397-398).
- `semantic` scores are compressed: 0.776–0.849, mean 0.823 over the 80 candidates
  (multilingual-e5-small). **MemRL's δ values (0.25–0.62) and z constants cannot be carried over.**
  They must be re-estimated on our corpus.
- `GET /documents/{document_id}` returns `original_text`, the full `lesson-v2` record. That is the
  analogue of MemRL's `full_content`.
- There is no embedding endpoint in the API (path list). To embed ticket intents ourselves we would
  need the pinned e5 model in the harness.

### 6.3 Required design decisions (choose and preregister)

**(a) Unit of utility (the "memory m").** Key Q by **document_id = episode_id** (one experience per
episode, as in MemRL), not by unit id. Unit ids change on restore, and units are LLM-extracted
fragments of one episode. All units of a document share that document's Q.

**(b) Phase A, two options:**
- **C-faithful.** Keep our own intent index `{episode_id: emb(ticket)}` (z = ticket text, like
  MemRL's task_description key). Phase A is cosine(new ticket, stored tickets), `> δ`, top-k1
  intents. Needs a pinned embedder in the harness (the same e5-small files from `models.lock.json`,
  with e5's `query:`/`passage:` prefixes handled consistently). Inject each selected episode's
  `original_text`, split into "successful" and "failed" sections as MemRL does.
- **C-SPEC (rerank Hindsight's list).** Phase A is Hindsight recall with a large `max_tokens`,
  grouped by `document_id`. Doc relevance = max over its units of `scores.semantic` (or `final`).
  Drop units with null `document_id` (observations), or map them through `source_facts`. Apply δ and
  top-k1 docs, then Phase B over docs, then inject that doc's units or record.
  - Deviation: similarity is ticket-vs-extracted-fact, not ticket-vs-intent.
  - Advantage: **C differs from B only in ranking**, which is the cleanest C-vs-B ablation (SPEC §12).
  - Recommendation [INFERRED]: C-SPEC for the primary ablation, and state the deviation. Ticket
    texts across causes look alike ("throughput collapses, CPU pegged"), so a ticket-intent gate may
    not separate causes anyway.

**(c) Phase-B normalization.** Copy the code:
- `sim_z = (sim − μ_corpus)/σ_corpus`, with μ and σ taken from pairwise similarities on the
  **train-split tickets** under our scorer, frozen before the run.
- `q_z = clip((Q − mean_C)/pstd_C, ±3)` over the candidate docs, with std set to 1 when it is
  degenerate.
- `score = 0.5·sim_z + 0.5·q_z`.
- δ = the 80th percentile of the same pairwise distribution (App. E.2). k1/k2: k2 = our `memory_k`
  (5). k1 = 10 docs (BCB/LLB setting), or 5/3 per the HLE ablation optimum.
- ε: the paper says nothing and the code uses 0–0.1. Decide explicitly. With Q_init = 0 and no
  exploration, early ranking equals B's, and a memory rises only after it has been injected. That is
  the mechanism behind the 1/10 surfacing problem.

**(d) Update rule.** After each **train** episode's verdict, apply to every distinct `document_id`
injected into that episode: `Q ← Q + 0.3·(r − Q)`; `visits += 1`. A new episode's doc gets `Q_init`.
- Paper: 0.0.
- Code variants: 0.5 (LLB), or 1.0/0.0 by outcome (HLE).
- We are sequential, so update per episode. MemRL batches (5–32), so ours is strictly more on-policy;
  note this as a deviation.
- Only `train` episodes call `observe`. Read-only episodes must leave the Q table byte-identical.
- Hash the Q table, visits and intent index into `memory_snapshots.learner_state_sha256`. Store the
  learner state alongside the bank archive at each checkpoint so `open_snapshot` restores both.

**(e) Reward** (SPEC.md:353: `r = passed × (1 − normalised_cost)`):
- MemRL's code uses binary ±1 (or 1/0) and **cannot** take a scalar. Ours is a deliberate
  extension (the paper allows a "scalar score", §4.3).
- `normalised_cost` must be in [0, 1] and preregistered. Suggestion: `c = clip(episode_cost_usd /
  cost_cap_usd, 0, 1)`, where `cost_cap_usd` is the cost of the registered episode budget
  (`max_total_tokens` at the pinned shadow prices), inference plus tool_compute. **Exclude**
  `learning` and `eval_overhead`, which are not attributable to the injected memories.
- Consequences to note [INFERRED]:
  1. With failures giving r = 0 and Q_init = 0, **Q can only rise**. An injected memory that keeps
     co-occurring with failures is never pushed below an unused one. MemRL's ±1 configs do demote
     such memories.
  2. A memory injected into an *integrity* failure receives the same 0 as one injected into a
     near-miss performance failure.
  3. Among successes, (1 − c) may vary only a little. The per-candidate z-score of Q *amplifies*
     small differences, so cost noise becomes ranking signal.

  Options, to decide before the run:
  - `r_fail = −1`, matching MemRL BCB/ALF/HLE.
  - Keep the proposal's reward and set Q_init above the typical failure return.
  - Report both as a sensitivity analysis.

**(f) Write path.** We keep Hindsight's retain of `lesson-v2` (no MemRL script/reflection prompts).
`lesson-v2` already leads with verified outcome and failed checks, the analogue of the reflection.
A stricter replication would add MemRL's script/reflection LLM step (charged as `learning`). This
is optional, and it would make C's records differ from B's.

**(g) Injection.** MemRL separates SUCCESS and FAILURE sections and labels them. Our block is a flat
list of units. If C injects document records, mirror the labeling. To keep the C-vs-B ablation clean,
keep B's renderer and only reorder or select differently.

**(h) Ledger.** Phase B is negligible compute. Intent-index embeddings (C-faithful) are `learning`.
Q updates cost nothing.

### 6.4 Mapping to the SPEC interfaces
- `Memory.recall(task, k)` → a new `recall_candidates(task) -> list[Candidate(unit_id, document_id,
  type, text, semantic, final)]`. The current `recall` keeps working for B.
- `Learner.rank(task, candidates)` runs Phase A (gate, top-k1 docs) and Phase B (z-scored score,
  top-k2), returning the selected units or docs. Log for each episode: candidate doc ids, sim, Q,
  sim_z, q_z, score, and whether ε fired (MemRL logs the same: `llb_rl_runner.py:624-670`).
- `Learner.observe(episode, verdict)` applies the EMA update for the injected doc ids and adds the
  new doc with Q_init. This needs the injected **document ids** on the episode row; today only unit
  ids are stored.

---

## 7. What did not work, and caveats

- **Pure similarity (λ = 0) plateaus; λ → 1 is volatile** (§5.3.2).
- **Too many memories (k = 10/5) hurt; too few (3/1) hurt** (§5.3.4).
- **No z-norm or no similarity gate**: forgetting rises 0.041 → 0.073, with spikes (§5.4.2, App. B.1).
- **Low task similarity**: cross-task memory gives no benefit, and MemRL "degrades into reflection-like
  behavior" (HLE tie in Table 3; §6(iii); App. G.3). Their HLE gain is attributed to *memorizing the
  same questions* over epochs (App. B.3), which our fresh-seed design does not allow.
- **Weak backbones give no learning signal** (GPT-4o-mini at about 4% on HLE; App. E.4).
  Benchmark-specific backbone choice is itself a methodological caveat.
- **Credit assignment**: all injected memories get the same reward (§6(ii), App. G.2).
- **High variance in long horizons** from step-wise updates (§6(i), App. G.1).
- **Reward hacking or verifier false positives** "quickly solidify" wrong high-Q memories (App. G.4).
  For us, the judge's integrity check is the guard.
- **Statistics**: single runs, no variance estimates. Transfer numbers use the "best validation"
  checkpoint. The gains in our closest analogue (BCB transfer, single-step code) are about 1–2 pp. Our
  power analysis should not assume MemRL-sized effects.
- **Code vs paper**:
  - ε-greedy is undocumented.
  - Q_init and the reward scale differ by benchmark.
  - BCB k1 is 5 vs 10.
  - The scalar reward is unsupported.
  - A hard Q floor and exclusion (`q_min_threshold`) is undocumented and can be absorbing.
  - Updates are batched, not per episode.
  - Dead or missing code: `ValueAwareSelector`, `MemoryCurator`, `update_values_chain_mdp`.
  - The paper's Boltzmann derivation (App. A.4.3) is not implemented as sampling; the code selects
    deterministically with ε-greedy.
- Paper HTML caveat: App. I prompt listings lost their first lines in rendering. The verbatim
  prompts above come from the code.
