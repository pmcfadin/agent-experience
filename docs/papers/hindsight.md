# Hindsight (Latimer et al.): paper, Hindsight 1.1.0 API, and what it means for Group B

Date: 2026-10-25. Scope: read-only. No repo file was edited and no bank was written. The snapshot bank's
content fingerprint (`b9db0adbd5d3772a`, 85 units) was the same before and after every recall and the reflect call.

Labels used throughout:
- **[READ]** stated in the named source.
- **[OBS]** observed by running something here.
- **[INF]** my inference.

Sources:
- **ACL:** "HINDSIGHT: Structured Agent Memory that Retains, Recalls, and Reflects", ACL 2026 System Demonstrations, pp. 275–285. https://aclanthology.org/2026.acl-demo.27.pdf
- **arXiv:** "Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects", arXiv:2512.12818 (v1, 14 Dec 2025, 28 pp.). It has the formulas and prompts. The ACL paper cites it as "ArXiv:2505.XXXXX", a placeholder.
- **Installed server source:** `services/hindsight/.venv/lib/python3.12/site-packages/hindsight_api/` (hindsight-api-slim 1.1.0).
- **Live OpenAPI:** `GET http://127.0.0.1:8901/openapi.json` (version 1.1.0). Saved copy: `scratchpad/openapi.json`.
- **Upstream docs:** github.com/vectorize-io/hindsight `hindsight-docs/` (mental-models.mdx, observations.mdx, api/recall.mdx, blog 2026-07-24-recall-vs-reflect.md). Copies are in `scratchpad/docs/`.
- **Harness:** `elh/memory/hindsight.py` and SPEC.md §6.2 and §10.

---

## 0. Headline findings

1. **[OBS] I reproduced the 1/10 result exactly.** Running the harness's own recall call (`{"query": ticket, "max_tokens": 2048}`) on the e15 snapshot bank for the 10 checkpoint tickets (`inventory/2000–2009`):
   - It returns the **same top-5 ids as logged in `episodes.retrieved_memory_ids_json` in 10/10 episodes**.
   - A lesson unit appears in the top 5 only for ticket 2005, at rank 4.
2. **[OBS] Root cause: the reranker is scoring our `context` field, not the lesson.**
   - Hindsight's cross-encoder scores each candidate as `"{context}: {text}"` (`engine/search/reranking.py` ~L410–414).
   - Our retain `context` is `render_context()`: a preamble plus **the full ticket**.
   - So every extracted unit carries roughly 230 tokens of ticket in front of its text.
   - The query is also a ticket of roughly 240 tokens. Together they exceed MiniLM's 512-token pair limit, so the unit's own text is truncated.
   - In effect the cross-encoder ranks "how similar was the past ticket to this ticket", whatever the unit actually says.
   - I measured this by running the pinned ONNX reranker directly:
     - The lesson unit `adfaed81` scores logit **+0.78** with the context prefix and **−9.32** on its text alone.
     - A ticket-restatement world fact scores **+3.06** with the context prefix, matching `cross_encoder_score` in the server trace exactly.
     - Observations have no `context`, score about −9.4, end up with `final` ≈ 0.01, and never reach the top 5 of a mixed-type recall.
   - **Practical effect:** whether the agent sees a lesson depends on luck. It happens only when the failed episode's ticket happens to resemble the current one, as with ticket 2005 (#6864 inventory) against the failed episode on #6853 invoice.
3. **[OBS] BM25 contributes nothing for ticket-shaped queries.** The trace shows `bm25_count: 0` in every group.
   - Hindsight keeps only the 16 lowest-document-frequency query tokens (`bm25_term_selection.py`, `bm25_max_query_terms=16`).
   - For a new ticket those are the novel words ("inventory", "stock", "#6864"), which by construction are not in the bank.
   - Semantic and graph retrieval do all the work.
4. **[OBS] `budget` has no effect at this bank size.**
   - `fixed` budgets are low=100, mid=300, high=1000 graph nodes, and recall already visits all 85 nodes (`budget_used: 85`).
   - low, mid and high give identical results.
5. **[OBS] Two cheap changes make lessons appear in the top 5:**
   - **`types: ["experience"]`:** lessons in the top 5 for **8/10** tickets (rank 0–1 in most), up from 1/10.
   - **Split recall (3 by ticket + 2 from a fixed lesson-oriented query over `experience` and `observation`):** **10/10**.
6. **[OBS] Reflect is read-only in 1.1.0.**
   - The docstring says "Reflect is read-only … persists nothing", and the fingerprint was unchanged after the call.
   - One call with the ticket plus "what mistakes should be avoided" took **42.9 s and 44k input / 588 output memory-LLM tokens**.
   - It produced an explicit "Mistakes to Avoid: Modifying Protected Files … must not modify run.py, examples/, tests/, scripts/" section.
   - This is the arXiv paper's reflect (§6.1), which wrote opinions back to memory, redesigned. 1.1.0 no longer writes them.

---

## 1. Architecture as the paper describes it

### 1.1 Components and memory networks
**[READ]** arXiv §1 and §3.3; ACL §1 and §3.3:
- **TEMPR** ("Temporal Entity Memory Priming Retrieval") implements retain and recall.
- **CARA** ("Coherent Adaptive Reasoning Agents") implements reflect.
- A **bank** is an isolated container holding the networks plus a profile (arXiv §3.2; ACL §4.5).

**[READ]** arXiv §3.1 defines M = {W, B, O, S}:
- **World (W):** objective facts about the external world.
- **Experience / "bank" (B):** first-person facts about the agent's own actions, experiences and recommendations.
- **Opinion (O):** subjective judgments, o = (text with rationale, confidence c ∈ [0,1], τ, bank, entities) (§5.4.1).
- **Observation (S):** preference-neutral entity summaries synthesized from W and B in the background, regenerated when the underlying facts change (§4.1.5).

**[READ] The 1.1.0 API differs from the arXiv paper (ACL App. C; OpenAPI).**
- Recall exposes three types: `world`, `experience` and `observation`.
- **Opinions are gone from the API.**
- Observations were redefined (ACL §3.3) as "evidence-grounded beliefs consolidated from multiple facts", with:
  - evidence quotes
  - a proof count
  - a computed trend (new / strengthening / stable / weakening / stale)
- ACL adds three newer layers:
  - **mental models**, which neither paper defines
  - **directives**, "prioritized hard rules injected into every response"
  - an **agentic reflect loop**

### 1.2 Memory unit and links
**[READ]** arXiv Eq. 1: f = (id, bank, narrative text, embedding, occurrence interval τs–τe, mention time τm, fact type, confidence, aux = context, access count, full-text vector). Each unit is one Postgres row, with pgvector HNSW and GIN full-text indexes (ACL §4.1).

Links, G = (V, E), arXiv §4.1.3–4.1.4:

| Link | Weight |
|---|---|
| entity | 1.0 between units that share a canonical entity |
| temporal | exp(−Δt/σt) |
| semantic | cosine, kept if ≥ θs |
| causal | 1.0, extracted by the LLM, up-weighted in traversal |

**[OBS]** The e15 bank holds 3150 semantic, 630 temporal, 96 entity and 24 caused_by links over 85 units.

### 1.3 Retain pipeline
**[READ]** arXiv §4.1.2–4.1.4, §6.1, App. A.1 and A.5.1:
- **Chunking:** coarse, "2–5 comprehensive facts per conversation".
- **Extraction:** one LLM call with structured output, covering:
  - coreference resolution
  - temporal normalization
  - participant attribution
  - preserving the reasoning (the "why")
  - fact-type classification
  - entities (PERSON/ORG/LOC/PRODUCT/CONCEPT/OTHER)
- **Schema:** what/when/where/who/why, plus `causal_relations`.
- **Entity resolution:** string + co-occurrence + temporal similarity (Eq. 2).
- **Link construction.**
- **Background consolidation:** observations, plus opinion reinforcement in the arXiv version.

**[READ]** In 1.1.0, extraction modes are `concise` (our bank's setting), `verbose`, `custom`, `verbatim` and `chunks` (config.py L1819). A bank can also carry `retain_mission` and `retain_custom_instructions`. The mission is sent in the per-request user message (`fact_extraction.py` ~L1585).

**[OBS]** What our bank holds: 15 documents produced 30 world, 33 experience and 22 observation units. For a typical episode:
- 1 world unit restating the ticket
- 1 world unit with the root cause
- 1–3 experience units: the fix, the timing, and, when the episode failed, the failure reason

### 1.4 Recall (TEMPR)
**[READ]** arXiv §4.2:
- Recall(B, Q, k), where k is a **token budget**, not top-k. An optional "cost/latency budget caps how aggressively to expand search".
- **Four channels, run in parallel for each fact type:**
  - semantic (HNSW cosine)
  - BM25 (Postgres FTS)
  - graph spreading activation from the top semantic hits: A(fj,t+1) = max A(fi,t)·w·δ·µ(ℓ), with µ > 1 for causal and entity links; 2–3 hops per ACL §4.1
  - temporal, only when the query contains a time expression
- **Fusion:** RRF, Σ 1/(k + rank), with k ≈ 60.
- **Rerank:** cross-encoder `ms-marco-MiniLM-L-6-v2`, with dates included in its input.
- **Packing:** greedy, until the token budget is used.

**[READ] What 1.1.0 adds** (`engine/search/reranking.py` L176–290):
- The final score is the cross-encoder score (normalized) × a recency boost (±10%) × a temporal boost (±10%) × a proof-count boost (±5%).
- The cross-encoder input is `[Date: …] {context}: {text}`, which is the root cause in §0.2.

**[OBS]** Trace of the default recall (`scratchpad/trace_default.json`):

| Stage | Count / time |
|---|---|
| semantic | 85 |
| bm25 | 0 |
| graph | 25 |
| temporal | 0 |
| RRF merged | 85 |
| cross-encoder reranked | all 85, 1.50 s on CPU |
| kept within max_tokens = 2048 | 46 (the harness then takes the first 5) |

Total time was about 1.6 s.

### 1.5 Reflect (CARA), disposition and mission
**[READ] arXiv §5–6.**
- Profile Θ = (skepticism 1–5, literalism 1–5, empathy 1–5, bias strength β ∈ [0,1]) plus a first-person background, turned into the system prompt.
- Reflect steps:
  1. recall over all networks
  2. answer conditioned on the profile
  3. extract new opinions with confidence
  4. **store them**
- Opinion confidence is reinforced on later retains: reinforce +α, weaken −α, contradict −2α (Eq. 26).

**[READ] ACL §3.3 version.**
- Reflect is an agentic loop over the tools `search_mental_models` → `search_observations` → `recall` → `expand`, and "Hierarchical retrieval queries mental models first, then observations, then raw facts."
- Directives are injected as hard rules.

**[READ] 1.1.0 implementation.**
- The `reflect_async` docstring (memory_engine.py L15768ff): "Reflect is read-only: it synthesizes an answer from the bank's stored memories and persists nothing."
- Disposition and mission are now bank config: `disposition_skepticism`, `disposition_literalism`, `disposition_empathy`, `reflect_mission`. The `/profile` endpoint returns a deprecation message saying so.
- **There is no β in 1.1.0. Disposition affects reflect only, not recall.** [INF from code layout]

### 1.6 Observations, consolidation and mental models
**[READ] Observations / consolidation** (observations.mdx; consolidation/prompts.py):
- After each retain, a background consolidator merges new facts into observations.
- It is guided by `observations_mission`. Its default tells the consolidator "to track anything worth remembering".
- Scope can be set by tag: `observation_scopes` per item (`per_tag`, `combined`, `shared`), or `consolidation_strategies` with a per-scope mission and cap.

**[READ] Mental models** (mental-models.mdx; OpenAPI `CreateMentalModelRequest`):
- "A mental model is a standing answer to a question about a bank. You define the question once; Hindsight writes the answer … and rewrites it in the background as the bank learns more."
- Reading one "is a database read. No retrieval, no synthesis, no LLM call."
- Refresh can run after consolidation, on a cron, or in delta mode, and is scoped by tags.
- Each model records provenance and keeps its history.

**[READ]** Bank transfer exports `mental_models` and `directives` (`engine/transfer/export.py` L89–95), so snapshots would carry them.

---

## 2. Evaluation in the paper

**[READ]** arXiv §7, Tables 2–4; ACL §6, Tables 2 and 4.

**Benchmarks:**
- **LongMemEval-S:** 500 questions, about 115k tokens and about 50 sessions each. The M setting is not reported.
- **LoCoMo:** 50 conversations, about 9.2k tokens and 19 sessions each.

**Setup:**
- **Judge:** GPT-OSS-120B at temperature 0, binary scoring with category-specific prompts.
- **Backbones:** GPT-OSS-20B; GPT-OSS-120B; GPT-OSS-120B for memory with Gemini-3 Pro answering.
- **Profile:** neutral (3/3/3, β = 0.2).
- **Recall token budgets:** the text literally says "<add> tokens". They were never filled in.

**Results:**

| | Full context, GPT-4o | Zep | Supermemory, Gemini-3 | Hindsight 20B | Hindsight 120B | Hindsight Gemini-3 |
|---|---|---|---|---|---|---|
| LongMemEval-S overall | 60.2 | 71.2 | 85.2 | 83.6 | 89.0 | **91.4** |
| LoCoMo overall | — | 75.1 | — | 83.2 | 85.7 | 89.6 |

- On LongMemEval, the largest gains are in multi-session (87.2), temporal reasoning (91.0) and knowledge update (94.9).
- On LoCoMo, Backboard's reported 90.0 is higher than Hindsight's 89.6.

**Caveats [READ]:**
- Baseline numbers are copied from vendor reports that used different judges (Supermemory used GPT-4o). The authors say the Backboard numbers "could not be independently reproduced".
- **There are no component ablations** (channels, RRF, reranker, network types) in either version.
- Latency: under 200 ms recall on a 10k-unit bank, excluding the LLM (ACL §4.2).
- No cost figures.
- Every evaluation is conversational QA. Tool use and workflows, and an RL loop over retain/recall/reflect, are listed as future work (arXiv §9).

**[INF] Relevance to us:**
- The paper's evidence is about recalling *facts from conversations*, not about *learning from task outcomes*.
- The "experience" network was designed for "what I did/recommended", and the extraction prompt requires recording "what the user asked … that triggered this".
- There is **no outcome or failure field and no mistake-learning mechanism** (arXiv A.1 and A.5.1). Our harness puts the outcome into the narrative text, which is the only option.

**[READ] Usage guidance in the papers and docs is thin:**
- Recall "before each LLM call", retain "after each response" (ACL App. E, LiteLLM wrapper).
- Recall is for loading context; reflect is for "synthesis, not on every turn" (docs blog 2026-07-24).
- Nothing on tags or query formulation.

---

## 3. 1.1.0 parameters we do not use, and which matter for us

What we send today (hindsight.py):
- **retain:** `{items: [{content, context, document_id, tags: ["family:<f>"]}]}`
- **recall:** `{query: ticket, max_tokens: 2048}`, then the client keeps the first k = 5.

### 3.1 Recall (`POST /v1/default/banks/{bank}/memories/recall`, schema `RecallRequest`)

| Param | Default | What it does [READ OpenAPI] | Matters for us? |
|---|---|---|---|
| `types` | all (`world`, `experience`, `observation`) | Each type runs the full pipeline separately. | **Yes, highest.** `["experience"]` removes the 30 ticket-restatement world units. [OBS] 1/10 → 8/10. |
| `prefer_observations` | bank config (false) | With observations plus raw types, drops raw facts an observation in the results already covers, then backfills. | Medium. Our observations get very low reranker scores (no context), so on their own they lose. [OBS] No change in our test. |
| `budget` | `mid` | Node budget for graph/candidates: fixed low/mid/high = 100/300/1000, or `adaptive` as a fraction of max_tokens. | **No** at 85 units. [OBS] Identical results. Only matters above about 300 units. |
| `max_tokens` | 4096 (we pass 2048) | Token cap on returned facts. | Low. We truncate to 5 ourselves. |
| `tags`, `tags_match` (`any`, `all`, `any_strict`, `all_strict`, `exact`) | none / `any` | Filter by tags. | **Not useful today.** All 85 units carry the same `family:cpu_mem` tag. [OBS] It becomes useful if retain adds outcome tags (see §5). |
| `tag_groups` | none | Boolean and/or/not tag expressions; `resolve: "fuzzy"`. | Same as above. Enables things like "outcome:failed AND family:X". |
| `min_scores` | none | Per-stage score floors (semantic, keyword, reranker, final). | Low. |
| `query_timestamp`, `temporal_window` | now / none | Temporal anchor and temporal-group window; ranks, does not filter. | Low. All units are dated the same day, and recency changes scores by at most ±10%. |
| `include.entities` | `{max_tokens: 500}` | Adds entity observations to the response. | Low (we ignore it). Turning it off saves a little time. |
| `include.chunks` | off | Returns the raw source chunks, i.e. our full rendered episode record. | **Medium.** Lets us inject the verbatim record, including the "Protected files the agent changed: …" line, instead of the extracted paraphrase. |
| `include.source_facts` | off | Source facts for observation results. | Low. |
| `trace` | false | Per-stage diagnostics. | Useful for audits. |

Example from the OpenAPI spec (`RecallRequest`):
```json
{"budget":"mid","include":{"entities":{"max_tokens":500}},"max_tokens":4096,
 "query":"What did Alice say about machine learning?","query_timestamp":"2023-05-30T23:40:00",
 "tags":["user_a"],"tags_match":"any",
 "temporal_window":{"end":"2023-06-30T23:59:59Z","start":"2023-04-01T00:00:00Z"},
 "trace":true,"types":["world","experience"]}
```

What a lesson-surfacing recall would look like for us [INF]:
```json
{"query":"<ticket>","types":["experience"],"max_tokens":2048,"include":{"entities":null}}
```

### 3.2 Retain (`POST /memories`, `RetainRequest` / `MemoryItem`)

| Param | What it does [READ] | Matters? |
|---|---|---|
| `context` (we use it) | Extraction context, **stored on every extracted unit and prepended to the reranker input**. | **Yes, highest.** It is the root cause in §0.2. Keep it short and free of the ticket. [OBS/INF] |
| `tags` (we use `family:`) | Visibility scoping; also drives observation scoping. | Add `outcome:passed|failed` and `check:<name>-failed`. [INF] |
| `metadata` | Free key/value pairs, returned in results. | Lets us carry the verdict in machine-readable form without exposing it to the reranker. |
| `timestamp` | When the event occurred. Default now; `"unset"` means none. | Low. |
| `entities` / `resolve_entities` | Seed entities, e.g. `run.py`, `tests/`, "protected files". | Medium. Strengthens the entity graph between lessons. [INF] |
| `observation_scopes` | `per_tag` / `combined` / `shared`. | Medium. With outcome tags, `per_tag` makes a "failed" scope consolidate on its own. |
| `strategy`, `update_mode` | Named retain strategy; replace or append. | Low. |
| `async`, `operation_id` | Background retain; idempotency. | Low (we already wait until the bank is idle). |

### 3.3 Bank config (`PATCH /v1/default/banks/{bank}/config`, `{"updates": {...}}`)
[READ] Our bank currently runs all defaults (`overrides: {}`).

| Key | Purpose | For us |
|---|---|---|
| `retain_mission` | Tells extraction what matters. | **High.** For example: "Record what caused each attempt to fail or pass, as reusable lessons." |
| `retain_extraction_mode` + `retain_custom_instructions` | `concise` (current), `verbose`, `custom`, `verbatim`, `chunks`. | Medium. With `custom`, we specify the lesson format. |
| `observations_mission` | Guides consolidation. | **High.** For example: "Maintain beliefs about which agent behaviours caused verified failures (e.g., modifying protected files) and which fixes worked." |
| `consolidation_strategies` | Per-tag-scope mission and cap. | Medium, once outcome tags exist. |
| `reflect_mission`, `disposition_*` | Reflect only. | Only if we use reflect or mental models. |
| `recall_prefer_observations`, `recall_budget_*` | Recall defaults. | Low. |
| `enable_text_search`, `enable_graph_retrieval`, `enable_temporal_retrieval`, `enable_reranking` | Turn individual pipeline stages off. | `enable_reranking: false` is a cheap ablation of the root cause. [INF] |

Example from the OpenAPI spec: `{"updates":{"retain_custom_instructions":"Extract technical details carefully","retain_extraction_mode":"custom"}}`

### 3.4 Reflect (`POST /reflect`, `ReflectRequest`)

Parameters [READ]:
- `query`
- `budget`: default `low`; controls the iteration count
- `max_tokens`
- `response_schema`: structured output
- `tags`, `tags_match`, `tag_groups`
- `fact_types`
- `apply_all_directives`
- `exclude_mental_models`, `exclude_mental_model_ids`
- `include.facts`, `include.tool_calls`
- `reflect_search_observations_max_tokens`, `reflect_search_observations_include_entities`

Read-only in 1.1.0 (§0.6). Example from the spec:
```json
{"budget":"low","include":{"facts":{}},"max_tokens":4096,
 "query":"What do you think about artificial intelligence?",
 "response_schema":{"properties":{"summary":{"type":"string"},"key_points":{"items":{"type":"string"},"type":"array"}},"required":["summary","key_points"],"type":"object"},
 "tags":["user_a"],"tags_match":"any"}
```

[OBS] With the ticket plus a "mistakes to avoid" question:
- Latency 42.9 s; 44,054 input / 588 output tokens on local Gemma.
- `based_on`: 65 memories, 0 mental models, 0 directives.
- The output contained the right lesson (see §0.6). Full output: `scratchpad/reflect_out.json`.

**Cost of using reflect per episode [INF]:**
- 43 s and 44k memory-LLM tokens, which must be charged to the ledger.
- The output is nondeterministic, because it is LLM synthesis.
- That conflicts with SPEC §6.2 ("exact items shown … are known"). We would have to log the reflect text as the injected block.

### 3.5 Mental models (`POST /mental-models`, `CreateMentalModelRequest`)

Fields [READ]: `id`, `name`, `source_query`, `tags`, `max_tokens`, and `trigger`, where trigger takes:
- `refresh_after_consolidation` or `refresh_cron`, plus `full_refresh_cron`
- `mode`: `full` / `delta`
- `budget`
- `fact_types`
- `tags_match`, `tag_groups`
- `response_schema`
- `min_refresh_interval_seconds`
- `keep_trace`

Example from the spec:
```json
{"id":"team-communication","max_tokens":2048,"name":"Team Communication Preferences",
 "source_query":"How does the team prefer to communicate?","tags":["team"],
 "trigger":{"refresh_after_consolidation":false}}
```

**Yes, we can define a standing "lessons about mistakes" model.** [READ semantics; INF design] Example:
```json
{"id":"lessons-from-failures","name":"Lessons from verified failures",
 "source_query":"Across all past attempts, which agent actions caused the automated judge to fail an attempt (integrity, public tests, correctness, performance), and what should a future attempt do or avoid? List concrete rules.",
 "max_tokens":800,
 "trigger":{"refresh_after_consolidation":true,"mode":"delta","fact_types":["experience","observation"],"keep_trace":true}}
```

How it would fit the harness:
- Create it on the **training** bank before episode 1.
- Each refresh is a reflect, costing about 40 s and about 40k tokens on Gemma. It should be charged as `learning`, and `wait_idle()` already blocks until operations finish.
- At episode start, read it with `GET /mental-models/{id}`: a DB read, no LLM call, deterministic for a snapshot.
- Inject its `content` alongside the top-k recall.
- It travels with the bank-transfer snapshot.
- **`fingerprint()` currently hashes only memory units.** It must also hash mental-model content (and directives), or a drifted model would go undetected.

### 3.6 Directives (`POST /directives`)

Fields [READ]: `name`, `content`, `priority`, `is_active`, `tags`. They are "injected into prompts" of **reflect and mental-model refresh**; recall does not use them.

**[INF] They are hand-written rules, not learned memory**, so putting "Do not modify protected files" in a directive would contaminate the A/B comparison. The only legitimate use is a neutral formatting directive for the mental model, e.g. "state rules as imperatives; cite the failure that taught each rule".

---

## 4. Experiment on the read-only snapshot bank

Setup:
- Bank: `elh-phase0-local-exp1-memory-helps-r2-v6-10112455fc-h0-B-e15` (85 units: 30 world, 33 experience, 22 observation; all tagged `family:cpu_mem`; no mental models or directives; all bank config at defaults).
- Lesson units (`scratchpad/lesson_ids.json`):
  - experience `adfaed81` (examples/sample_output.json), `79b1bdc9` (run.py), `7522f220` (tests/test_job.py)
  - observation `11cbd279` (run.py), `7626fb0a` (tests/test_job.py)
  - mixed observation `11266807`: the #6853 summary, which ends with the protected-file failure
- Scripts and output: `scratchpad/recall_exp.py` → `recall_exp.out`; `scratchpad/sweep.py`.

### 4.1 Single ticket `cpu_mem/queue_pop_front/inventory/2005` (#6864)

| Setting | Lesson ranks in top 5 (all returned) | Notes |
|---|---|---|
| default (harness) | [4] (4, 39) | Ranks 0–3 are world ticket restatements. Its lesson is the #6853 one, the most similar ticket. |
| budget=high / low | [4] | Identical to default. |
| types=[experience] | **[0]** | |
| types=[observation] | [0] (mixed obs) | Every final score ≈ 0.01, so ranking is by RRF. |
| types=[exp,obs] + prefer_observations | [0] | Observations are all ≈ 0.01. |
| lesson query (generic "What mistakes made a coding agent fail…") | **[0,1,2,3,4]** | Cross-encoder 0.97–0.999, BM25 fires (kw 0.6–1.6). |
| lesson query, types=[experience] | [0,1,2] | |
| ticket + appended lesson question | [] (6) | The long ticket still dominates. |
| keyword list ("failed modified protected forbidden files run.py tests/ examples/") | [0–4] | |

Latency is 0.4–1.8 s per recall, dominated by CPU cross-encoder time.

### 4.2 All 10 checkpoint tickets (inventory/2000–2009)

| Strategy | Tickets with a lesson in the top 5 |
|---|---|
| A default (harness) | **1/10**, which reproduces the logged result |
| B `types:["experience"]` | **8/10** (misses 2002 and 2003) |
| C `types:["experience","observation"]` | 8/10 |
| D `budget:"high"` | 1/10 |
| E split: top 3 by ticket + top 2 from a fixed lesson query (`types` = exp+obs, deduplicated) | **10/10** |

**Reflect:** one call (see §3.4): 42.9 s, surfaced the lesson, bank unchanged.

---

## 5. Recommendation: what "using Hindsight as the paper intends" means for Group B

Ranked by expected impact on surfacing lessons, with cost and protocol risk noted. Everything here is [INF], grounded in the observations above.

1. **Stop putting the ticket in `context`. Expected impact: very high. Cost: none.**
   - The paper treats `context` as short auxiliary metadata (Eq. 1 "aux"; the API examples use `"context": "team meeting"`, `"work info"`).
   - 1.1.0 prepends it to the reranker input, so ours turns the reranker into a ticket-similarity matcher.
   - Use a short, outcome-bearing context, for example: `"Verified result of a coding agent's attempt at a cpu_mem performance ticket: FAILED (integrity)"`.
   - To keep the anti-restatement benefit that motivated lesson-v1 (SPEC §10), either:
     - put the ticket in `metadata`, or
     - set `retain_mission` / `retain_custom_instructions` so extraction does not produce ticket restatements.
   - This is a new record format (`lesson-v3`) and needs registration. Verify with the same 10-ticket sweep.
2. **Recall over the right memory type: `types: ["experience"]`, or experience plus observation. Impact: high (1/10 → 8/10 measured). Cost: none.**
   - This matches the paper's split: world = facts about the environment, experience = what the agent did and what happened. Lessons are experience.
   - The world units are mostly ticket restatements and add little the agent does not already see in its prompt.
3. **Add a standing "lessons from failures" layer, which is how the ACL paper's design is meant to work. Impact: high and robust. Cost: learning tokens.**
   - Option (a), preferred: a **mental model** (§3.5), refreshed after consolidation and read at episode start (a DB read, deterministic per snapshot).
   - Option (b): a second, fixed, generic recall query (strategy E, 10/10 measured).
   - Either one must be pre-registered and generic: no mention of protected files, and no hand-written rules.
   - Option (a) is closest to the ACL paper's "mental models first, then observations, then raw facts".
   - Extend `fingerprint()` and the snapshot checks to cover mental-model content.
4. **Set `observations_mission` (and `retain_mission`) on the bank. Impact: medium.**
   - The default consolidator mission tracks "anything worth remembering". Here it produced 22 observations, mostly ticket summaries.
   - A mission focused on causes of verified failures and success makes the observation layer, and any mental model built on it, about lessons.
   - Apply it at bank creation (`PATCH /config`). It travels with the bank-transfer config.
5. **Tag outcomes at retain (`outcome:failed`, `check:integrity-failed`) and use `per_tag` observation scopes. Impact: medium; enables precise retrieval.**
   - Gives a deterministic `tag_groups` recall such as `{"tags":["outcome:failed"],"match":"any_strict"}` for a "known failure modes" slot.
   - Also lets consolidation build failure-only observations.
6. **Use `include.chunks` to inject the verbatim record. Impact: low to medium.** The chunk carries the explicit "Protected files the agent changed …" line rather than Gemma's paraphrase.
7. **Use reflect only as an ablation, not as the primary Group B mechanism.**
   - It works (§0.6), but it costs about 43 s and about 44k tokens per episode, and its injected text is nondeterministic.
   - If tested, log the exact text and charge it as `learning`. A mental model gives most of the benefit with refresh cost paid at retain time.
8. **No-ops:** `budget` (irrelevant below about 300 units), `query_timestamp`/`temporal_window`, disposition traits (reflect only), and directives (hand-written rules would confound the groups).

**Protocol notes:**
- Options 1, 4 and 5 change what is stored, so they need a new registration and a fresh training run.
- Options 2, 3b and 6 change only recall, so they can be evaluated on existing snapshots in read-only mode, as done here.
- Option 3a needs a training bank, because creating a mental model writes to the bank.
