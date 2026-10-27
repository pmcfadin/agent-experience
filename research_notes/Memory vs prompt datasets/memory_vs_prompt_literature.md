# Memory vs Prompt: Literature on Separating Retrieval-per-Task Memory from a "Better Prompt"

Context: frozen Qwen3-Coder 30B coding agent; memory groups (Hindsight + agent-written WHEN-THEN rules) beat a stateless baseline on Python performance bugs. Question: would a fixed or optimized prompt holding the same lessons match it? These notes collect the baselines, datasets and results prior work used to tell the two apart.

## Q1. Which papers directly compare agent memory against static-prompt / prompt-optimization baselines, on which datasets, with what results?

### Takeaway
Only a few papers run memory and prompt optimization head to head. The cleanest is ACE: one evolving "playbook" context beat ICL few-shot, MIPROv2 and GEPA on AppWorld and finance tasks. Note that ACE's playbook is itself a single context that grows over time, not retrieval per task. The cleanest "retrieve vs dump everything" comparisons come from the guideline and insight lineage: ExpeL's all-insights-in-prompt approach lost to AutoGuide's context-conditioned retrieval of guidelines, and Dynamic Cheatsheet's full-history baseline lost to curated memory. "Learning on the Job" (2607.22157) is the closest design to ours: WHEN-THEN rules plus a static-RAG control on tau-bench banking. Even so, it does not test "all learned rules placed in the system prompt".

### Cited Findings

**ACE: Agentic Context Engineering (arXiv 2510.04618)**
- Datasets: AppWorld (test-normal and test-challenge splits), FiNER (XBRL financial entity tagging) and Formula (financial numerical reasoning). DDXPlus and BIRD-SQL also appear in the paper. — [ACE html](https://arxiv.org/html/2510.04618)
- Baselines: Base LLM, ICL (in-context demonstrations), MIPROv2, GEPA and Dynamic Cheatsheet (online). Backbone: DeepSeek-V3.1. — [ACE html](https://arxiv.org/html/2510.04618)
- AppWorld offline averages: ReAct base 42.4, ICL 46.0 (+3.6), GEPA 46.4 (+4.0), ACE 59.4 (+17.0). Online without labels: DC 51.9 (+9.5), ACE 59.5 (+17.1). — [ACE html](https://arxiv.org/html/2510.04618)
- FiNER and Formula average, offline with labels: base 69.1, MIPROv2 70.9, GEPA 72.5, ACE 81.9. Online: DC 71.8, ACE 76.6. — [ACE html](https://arxiv.org/html/2510.04618)
- Ablation on AppWorld: with no reflector and no multi-epoch, ACE reached 55.1. Removing multi-epoch alone gave 56.8. Full ACE scored 59.4. — [ACE html](https://arxiv.org/html/2510.04618)
- "Context collapse" in Dynamic Cheatsheet: a context of 18,282 tokens scoring 66.7% was rewritten down to 122 tokens and fell to 57.1%, below the no-adaptation baseline of 63.7%. ACE's argument is that monolithic rewriting and "brevity bias" (i.e. GEPA-style short prompts) throw away domain detail. — [ACE html](https://arxiv.org/html/2510.04618); [ACE abs](https://arxiv.org/abs/2510.04618)
- Cost: against GEPA on AppWorld, ACE cut adaptation latency by 82.3% and rollouts by 75.1%. — [ACE html](https://arxiv.org/html/2510.04618)
- ACE degraded when reliable feedback signals were missing. — [ACE html](https://arxiv.org/html/2510.04618)

**Dynamic Cheatsheet (arXiv 2504.07952; EACL 2026)**
- This paper has the richest set of baselines for our question:
  - BL: plain prompting.
  - DC-empty: the structured DC instructions with no memory, which isolates the prompt-scaffold effect.
  - Full-History appending (FH).
  - DR: verbatim retrieval of past examples, no synthesis.
  - DC-Cu: one cumulative cheatsheet, no retrieval.
  - DC-RS: retrieve the top 3 similar past items and synthesize.
  
  — [DC html](https://arxiv.org/html/2504.07952v1)
- Datasets: AIME 2020–2025, GPQA-Diamond (198), Game of 24 (100), Math Equation Balancer (250), MMLU-Pro Engineering and Physics (250 each). — [DC html](https://arxiv.org/html/2504.07952v1)
- Results by model:
  - Claude 3.5 Sonnet on AIME 2024: 23% to 50% with DC-Cu, versus 47% with DC-RS.
  - GPT-4o on Game of 24: 10% to 99% with DC-RS, because it learned and reused a Python brute-force solver.
  - GPQA-Diamond: Claude went from 60% to 69% (DC-RS). GPT-4o went from 57% to 58%, and retrieval sometimes confused it.
  
  — [DC html](https://arxiv.org/html/2504.07952v1)
- Full-history appending did badly. On AIME 2024 it scored 27% for Claude and 13% for GPT-4o, versus 47% and 40% for DC-RS. — [DC html](https://arxiv.org/html/2504.07952v1)
- Smaller models (Claude 3.5 Haiku, GPT-4o-mini) gained little: they produce few correct seed solutions and apply retrieved knowledge poorly. — [DC html](https://arxiv.org/html/2504.07952v1)

**GEPA (arXiv 2507.19457, ICLR 2026 Oral)**
- This is a pure prompt optimizer; it has no memory comparison. Tasks: HotpotQA, IFBench, HoVer and Papillon, on Qwen3 8B and GPT-4.1 Mini. It beats GRPO by about 10% on average (up to 20%) with 35x fewer rollouts, and beats MIPROv2 by about 14% with prompts 9.2x shorter. — [GEPA pdf](https://arxiv.org/pdf/2507.19457); [secondary summary](https://www.morphllm.com/gepa-prompt-optimization)
- On multi-agent systems GEPA's effect was highly variable, ranging from large gains to severe drops (MAS-PromptBench). — [MAS-PromptBench](https://arxiv.org/html/2606.23664)

**ExpeL (arXiv 2308.10144)**
- ExpeL places all extracted insights in the prompt (a static rule list) and also retrieves similar successful trajectories as few-shot examples. Datasets: HotpotQA, ALFWorld, WebShop, and transfer to FEVER. — [ExpeL html](https://arxiv.org/html/2308.10144)
- Ablation separating the two components (success rate):

  | Dataset | ReAct | Insights-only | Retrieve-only | Full ExpeL |
  |---|---|---|---|---|
  | HotpotQA | 28.0 | ~36 | ~31 | 39.0 |
  | ALFWorld | 40.0 | ~50 | ~55 | 59.0 |
  | WebShop | — | ~37 | ~38 | ~43 |

  Read from the paper's figures, so the values are approximate. Static insights helped more on QA; per-task exemplar retrieval helped more on ALFWorld. — [ExpeL html](https://arxiv.org/html/2308.10144)
- **Hand-crafted insights control (HotpotQA):** hand-written insights scored 32.0, insights extracted by GPT-3.5 scored 32.0, and insights extracted by GPT-4 scored 39.0. A human-written static prompt was a real baseline and was beaten. — [ExpeL html](https://arxiv.org/html/2308.10144)
- Retrieval-key ablation on ALFWorld: random exemplars 42.5, reason-similarity 48.5, task-similarity 59.0. — [ExpeL html](https://arxiv.org/html/2308.10144)

**AutoGuide (arXiv 2403.08978)**
- This paper tests retrieval of guidelines by context against ExpeL's all-guidelines approach. Its guidelines are conditional "when in context X, do Y" rules, close to our WHEN-THEN rules. — [AutoGuide html](https://arxiv.org/html/2403.08978)
- Results (ReAct / ExpeL / AutoGuide):
  - ALFWorld: 54.5 / 59.0 / 79.1
  - WebShop: 30 / 35 / 46
  - WebArena: 8.0 / 21.8 / 47.1
  
  The authors attribute the bigger WebArena gap to its "wide variety of tasks across different domains", which "requires a larger number of guidelines". (Check the WebArena figure against the PDF; it may be a subset.) — [AutoGuide html](https://arxiv.org/html/2403.08978)
- WebShop ablation: ReAct 30, plus context identification only 36, plus guidelines without contexts 37, full AutoGuide 46. — [AutoGuide html](https://arxiv.org/html/2403.08978)
- Top-k guidelines on WebShop: k=0 gives 30, k=1 gives 42, k=2 gives 46, k=3 gives 47, k=5 gives 43. More rules eventually hurt. — [AutoGuide html](https://arxiv.org/html/2403.08978)

**Agent Workflow Memory (arXiv 2409.07429)**
- WebArena (GPT-4): BrowserGym 23.5, **SteP 33.0** (14 workflows written by human experts for WebArena), AWM 35.5 at 5.9 steps. Induced memory roughly matched or slightly beat a hand-written expert prompt. — [AWM html](https://arxiv.org/html/2409.07429)
- Mind2Web step success rate:
  - Cross-task: MindAct 36.2, AWM offline 45.1.
  - Cross-website: MindAct 30.1, AWM offline 33.7, AWM online 33.9.
  - Cross-domain: MindAct 18.6, AWM offline 32.6, AWM online 35.5.
  
  — [AWM html](https://arxiv.org/html/2409.07429)
- AWM puts **all induced workflows** (per website) into memory rather than retrieving per task. That makes it closer to a learned static prompt, scoped by website. — [AWM html](https://arxiv.org/html/2409.07429)

**ReasoningBank (arXiv 2509.25140)**
- Datasets: WebArena (shopping, admin, gitlab, reddit), Mind2Web and SWE-Bench-Verified. Baselines: no memory, Synapse (trajectory memory) and AWM. There is no prompt-optimization baseline. — [ReasoningBank html](https://arxiv.org/html/2509.25140)
- Results: +8.3 success-rate points over no memory on WebArena with Gemini-2.5-flash. Adding failure memories took WebArena-Shopping from 46.5 to 49.7. MaTTS (k=5) reached 51.8. — [ReasoningBank html](https://arxiv.org/html/2509.25140)

**Learning on the Job (arXiv 2607.22157; Tablan, Taylor, Bernhem)**
- Setup: tau-bench banking domain with about 700 interlinked policy documents; published frontier results sit around 25% single-trial success. — [LotJ html](https://arxiv.org/html/2607.22157)
- Memory stores WHEN-THEN rules. Retrieval is driven by the agent through tool calls, with trust-weighted ranking; the harness never injects memory into the context. — [LotJ html](https://arxiv.org/html/2607.22157)
- Control: static-RAG, i.e. BM25 over the full policy corpus with no memory. — [LotJ html](https://arxiv.org/html/2607.22157)
- Results with Mistral Large: baseline 0.064, verdict-only learning 0.103 (about 1.6x), learning from corrections 0.170 (about 2.6x). Memory converted 22 of the 84 tasks the baseline never solved. — [LotJ html](https://arxiv.org/html/2607.22157)
- Memory transferred across models (Mistral Large and Claude Sonnet 5): each model using the other's memory beat its own no-memory baseline. — [LotJ abs](https://arxiv.org/abs/2607.22157)

**When Continual Learning Moves to Memory (arXiv 2604.27003; Hu, Long, Wang)**
- Setup: ALFWorld and BabyAI (one task pair each), frozen Qwen-Plus, BM25 retrieval, 200 instances per phase, 100 test instances, 2 runs. — [2604.27003 html](https://arxiv.org/html/2604.27003)
- **No static-prompt or prompt-optimization baselines.** The comparison is over memory representation (Raw trajectory vs Insight) and organization (Cond-Agg, Cond-Ind, Cond-Step). An abstract-page summary claiming otherwise was not borne out by the full text. — [2604.27003 html](https://arxiv.org/html/2604.27003)
- Forward transfer (FWT): Raw was −9.5 on ALFWorld and −7.5 on BabyAI. Insight was +6.5 and +9.0. Cond-Ind reached +15.0 on BabyAI. — [2604.27003 html](https://arxiv.org/html/2604.27003)
- Cross-task noise mostly hurts hard cases (ΔNL −26.1 for ALFWorld Raw). The condition with strong forward transfer also showed significant forgetting. — [2604.27003 html](https://arxiv.org/html/2604.27003)

**Training-Free GRPO (arXiv 2510.08191)**
- The learned "experiential knowledge" is a fixed token prior placed in the prompt after training, so it is effectively an optimized static prompt built from experience.
- Results with DeepSeek-V3.1-Terminus: AIME24 80.0 to 82.7, AIME25 67.9 to 73.3. Gains also on WebWalkerQA. Training used about 100 DAPO problems and cost about $18.

— [TF-GRPO](https://arxiv.org/html/2510.08191v1)

**Evo-Memory (arXiv 2511.20857)**
- Datasets: AIME-24/25, GPQA-Diamond, MMLU-Pro, ToolBench, AlfWorld, BabyAI, ScienceWorld, PDDL and Jericho, all restructured into streams of tasks.
- Baselines: ReAct, A-Mem, SelfRAG, MemOS, Mem0, LangMem, DC-Cu/RS, AWM, ExpRecent, ExpRAG and ReMem.
- There is no prompt-optimization baseline.

— [Evo-Memory html](https://arxiv.org/html/2511.20857v1)

**MemRL (arXiv 2601.03192)**
- Two-phase retrieval: filter by semantic similarity, then rerank by learned Q-value utility. Evaluated on HLE, BigCodeBench, ALFWorld and Lifelong Agent Bench. — [MemRL](https://arxiv.org/html/2601.03192v1)
- Numbers and baselines were not retrieved; see Gaps.

### Inferences
- Across papers, the static-prompt controls were: few-shot ICL, MIPROv2, GEPA, hand-crafted insights (ExpeL), human-expert workflows (SteP in AWM), a single cumulative cheatsheet (DC-Cu), all insights in the prompt (ExpeL vs AutoGuide), full history (DC), structured scaffold with no memory (DC-empty), and static RAG over documentation (LotJ).
- No paper found runs our exact control: the final distilled rules from the memory run, frozen and placed in the system prompt, then compared with per-task retrieval of the same rules. AutoGuide's "guidelines without contexts" ablation and ExpeL's insights-only condition come closest.
- ACE's win over GEPA is "rich, growing context" beating "short optimized prompt". It is not "retrieval" beating "prompt", because ACE's playbook is fully in context. So a "dump all rules" control could plausibly be strong when the rule set is small.
- Two hand-written baselines were already competitive with learned memory: SteP (33.0 vs AWM 35.5) and ExpeL's hand-crafted insights (32 vs 39). A hand-written or oracle prompt is therefore a meaningful control, not a strawman.

### Gaps
- Not verified from full text: Mem-p/Memp (procedural memory), Memento (case-based memory, GAIA/DeepResearcher), SkillWeaver (WebArena skill APIs), Voyager (skill-library ablation in Minecraft), MemRL numbers. None of these were fetched in this pass, so their baselines are not reported here.
- Could not confirm whether any 2026 paper (e.g. "The Optimizer Is the Agent", 2608.06714, or MemPro, 2606.00619) runs GEPA against episodic memory on the same tasks. They appeared in search but were not read.
- AutoGuide's WebArena numbers should be checked against the PDF for the subset definition.

## Q2. When does a single static prompt saturate, and when does per-task retrieval win?

### Takeaway
Per-task retrieval wins when there are many heterogeneous lessons, so that putting all of them in context adds noise. Examples: AutoGuide's biggest gap on WebArena, ExpeL's task-similarity retrieval beating random exemplars, the top-k rules curve turning down past k=3. A single cumulative context does as well or better when tasks are homogeneous and the lesson set is small: DC-Cu beat DC-RS on AIME 2024, AWM's all-workflows-per-website approach worked, and ACE's single playbook won. Gains also rise with how similar tasks are within a dataset (Evo-Memory, r ≈ 0.56–0.72).

### Cited Findings
- **Within-dataset similarity.** ReMem's improvement correlates with within-dataset task similarity, measured by embedding-cluster coherence: Pearson r=0.717 on Gemini 2.5 Flash and r=0.563 on Claude 3.7 Sonnet. High-similarity sets (PDDL, AlfWorld) gained most; diverse sets (AIME-25, GPQA) gained least. — [Evo-Memory](https://arxiv.org/html/2511.20857v1)
- **Pruning tracks heterogeneity.** Pruning was higher on diverse GPQA (36.8%) than on AIME (10.8–17.5%), meaning more stored experience was judged irrelevant. — [Evo-Memory](https://arxiv.org/html/2511.20857v1)
- **Domain variety.** AutoGuide's advantage over all-insights ExpeL was largest on WebArena (47.1 vs 21.8), which the authors attribute to domain variety requiring more guidelines. — [AutoGuide](https://arxiv.org/html/2403.08978)
- **Too many rules hurts.** Performance peaks at k≈3 retrieved guidelines and falls at k=5 (47 to 43). — [AutoGuide](https://arxiv.org/html/2403.08978)
- **Cumulative vs retrieval.** On sequential AIME, the cumulative cheatsheet (DC-Cu, 50%) beat retrieval-and-synthesis (DC-RS, 47%). DC-RS did better on more diverse GPQA. Full-history appending was worst, from context bloat. — [DC](https://arxiv.org/html/2504.07952v1)
- **Raw vs abstract memory.** Raw trajectory memory gives negative cross-task transfer; abstract insights give positive transfer. Fine-grained per-insight storage helps only when source memories are diverse ("retrieval diversity collapse" with homogeneous queries). — [2604.27003](https://arxiv.org/html/2604.27003)
- **Context collapse.** Short, compressed prompts lose domain detail; DC's rewrite from 18k to 122 tokens fell below baseline. This "brevity bias" is ACE's critique of GEPA. — [ACE](https://arxiv.org/html/2510.04618)
- **Retrieval key matters.** Task-similarity retrieval beat random exemplars by 16.5 points on ALFWorld. Much of the "memory" benefit is task-matched few-shot examples rather than rules. — [ExpeL](https://arxiv.org/html/2308.10144)
- **Weak models.** Small or weak models gain little from memory in either form: DC with Haiku and 4o-mini. Relevant to a 30B coder model. — [DC](https://arxiv.org/html/2504.07952v1)

### Inferences
- The likely tipping-point variables are:
  - the number of distinct lessons relative to the context the model can use well;
  - how heterogeneous the task categories are;
  - how many rules are injected per task. AutoGuide suggests about 3; past that, noise dominates.
- If our perf-bug set has few bug categories and a short distilled rule list, expect "all rules in the system prompt" to recover most of the memory gain. If it has many categories, each needing a different rule, per-task retrieval should keep an edge. This is testable by sweeping category count.
- Recommended controls for our study, in order of priority:
  1. **All-rules-static.** Freeze the final rule set from the memory run and put it all in the system prompt.
  2. **Oracle-rule.** Inject only the rules relevant to the task's ground-truth bug category. This is the upper bound on retrieval.
  3. **Shuffled/wrong-category rules.** Same length, wrong content. Controls for prompt length and attention effects.
  4. **Scaffold-only.** The memory-group instructions with an empty memory, as in DC-empty.
  5. **Hand-written lessons prompt.** As in ExpeL hand-crafted insights and AWM's SteP.
  6. **Few-shot.** Task-similar exemplars retrieved (ExpeL retrieve-only) vs fixed random exemplars.
  7. **Optimized prompt.** GEPA, MIPROv2 or ACE on a train split, judged on a held-out split with the same rollout budget as the memory run.
  8. **Category sweep.** Vary the number of bug categories (e.g. 2, 5, 10, 20) and the within-set similarity, then plot the memory-minus-static gap.
  9. **Top-k sweep.** k = 0, 1, 3, 5, all.
- Report results on held-out categories, since forward transfer (FWT), as in 2604.27003, separates generalization from memorization. Also report the gain on tasks the baseline never solves, as in LotJ's 22/84.

### Gaps
- No study found plots the memory-vs-static gap directly against the number of distinct lessons or categories. The Evo-Memory similarity correlation is the nearest evidence.
- No study found does the oracle-rule control explicitly in agentic coding.

## Q3. Datasets built to have many distinct latent "lessons" with category labels

### Takeaway
No benchmark found was designed with labelled latent lessons for memory-vs-prompt studies. The closest are benchmarks with natural category or template structure: Mind2Web (cross-task, cross-website, cross-domain splits), WebArena (task templates and sites), MMLU-Pro and GPQA (subject labels), ALFWorld (6 task types), and the Evo-Memory streams. Each gives category labels usable for stratified analysis.

### Cited Findings
- **Mind2Web** has explicit generalization splits (cross-task, cross-website, cross-domain). AWM's gains grew with distribution gap: cross-domain 18.6 to 35.5 step success rate. — [AWM](https://arxiv.org/html/2409.07429)
- **Evo-Memory** reorganizes 9 datasets into task streams and measures within-dataset similarity. It also tests easy-to-hard vs hard-to-easy ordering: ReMem scored 0.77 and 0.81 success. — [Evo-Memory](https://arxiv.org/html/2511.20857v1)
- **2604.27003** builds task-pair (A→B) splits in ALFWorld and BabyAI to measure forward transfer and forgetting. — [2604.27003](https://arxiv.org/html/2604.27003)
- **tau-bench banking** has about 700 policy documents. The latent "lessons" are policy rules and customer-behavior patterns, which makes it suitable for a static-RAG-vs-memory comparison. — [LotJ](https://arxiv.org/html/2607.22157)
- **DC's Game of 24 and Math Equation Balancer** are a single-lesson extreme: one reusable solver gives almost all of the gain. A useful contrast with heterogeneous sets. — [DC](https://arxiv.org/html/2504.07952v1)
- **EvoAgentBench** (arXiv 2607.05202) benchmarks agent self-evolution via ability transfer. It appeared in search but its design was not read. — [EvoAgentBench](https://arxiv.org/pdf/2607.05202)

### Inferences
- For our study, building our own labelled set is likely necessary. Tag each Python performance bug with its fix category (e.g. quadratic membership test to set, repeated regex compile, pandas iterrows to vectorize, string concatenation in a loop, missing cache). Then:
  - control the number of categories and the instances per category;
  - hold out instances within a category, and separately hold out whole categories;
  - keep a category-to-rule map so the oracle-rule and wrong-rule controls are cheap.
- Single-lesson tasks like Game of 24 predict that a static prompt will saturate. Categorical diversity is the lever that should separate retrieval from a dumped prompt.

### Gaps
- No dataset found was explicitly built with labelled latent lessons and a known lesson-to-task mapping for memory research. EvoAgentBench and Lifelong Agent Bench (used by MemRL) may come close but were not checked.
