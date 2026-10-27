# Benchmarks for frozen-weight agents that learn from a task stream (memory, test-time learning, self-evolving agents)

Scope: candidate benchmarks for testing whether carried memory (episodic memory, self-written WHEN-THEN rules, MemRL-style utility reranking, AWM-style workflow induction) beats an equivalent static prompt for a frozen local coding agent (Qwen3-Coder 30B). Researched 2026-10-27. Paper dates are given inline. Sizes and numbers come from the papers' own text as returned by fetch tools. Where I could not open a paper's full text (several 2026 arXiv entries), I say so.

## Streaming/continual agent benchmarks: what each one is, its size, stream structure, labels, grading, local executability, and memory baselines

### Takeaway
The benchmarks most directly usable for a frozen coding agent with category splits are LifelongAgentBench (DB/OS/KG, with skill-labelled tasks), StreamBench (Spider/CoSQL/BIRD/DS-1000/ToolBench streams with binary feedback), AppWorld (750 API-and-code tasks with state-based unit tests, plus the ACE paper's memory-vs-GEPA comparison), BigCodeBench and ALFWorld as used by MemRL, and SWE-Bench-CL (273 chronologically ordered SWE-bench Verified tasks, but it reports no validated memory results). Evo-Memory is the broadest umbrella: 10 datasets and more than 10 memory methods, including AWM and Dynamic Cheatsheet. Its single-turn and multi-turn tasks, however, are mostly not coding tasks.

### Cited Findings

**StreamBench (NeurIPS 2024 Datasets & Benchmarks; arXiv 2406.08747, June 2024)**
- It is a benchmark for "continuous improvement of LLM agents over an input-feedback sequence", in an online setting where the feedback is a binary correct/incorrect signal rather than the ground-truth label. — [StreamBench arXiv](https://arxiv.org/html/2406.08747v1)
- Stream sizes: Spider 2,147; CoSQL 1,007; BIRD 1,534; DS-1000 1,000 (Python, from StackOverflow); ToolBench 750 (a stable subset of APIs); DDXPlus 1,764 (medical diagnosis, 49 diagnoses); HotpotQA 1,500. — [StreamBench arXiv](https://arxiv.org/html/2406.08747v1); [search summary](https://arxiv.org/pdf/2406.08747)
- Spider, CoSQL and BIRD were chosen to form a progressive difficulty curve. — [StreamBench PDF](https://arxiv.org/pdf/2406.08747)
- Baselines: non-streaming zero-shot, few-shot, CoT and Self-Refine, which are effectively static-prompt baselines. Streaming baselines: GrowPrompt, MemPrompt, Self-StreamICL and MAM-StreamICL. — [StreamBench arXiv](https://arxiv.org/html/2406.08747v1)
- Findings:
  - Storing only correctly generated examples in memory consistently helped.
  - Storing incorrect examples sometimes hurt, even when they were marked as wrong.
  - MAM-StreamICL (three LLMs sharing one memory) was best.
  - Baseline rankings held across five random stream orderings. — [StreamBench arXiv](https://arxiv.org/html/2406.08747v1)
- The models used were API models: GPT-3.5-turbo, Gemini 1.0-pro and Claude 3 Haiku, plus GPT-4o and Gemini 1.5-flash. Code and data: https://github.com/stream-bench/stream-bench — [StreamBench arXiv](https://arxiv.org/html/2406.08747v1)

**Evo-Memory (arXiv 2511.20857, 25 Nov 2025)**
- Each dataset is restructured into a sequential task stream, "in which earlier tasks provide essential information or strategies for later ones". Task order is unified within each dataset. — [Evo-Memory arXiv](https://arxiv.org/html/2511.20857v1)
- Datasets:
  - Single-turn: MMLU-Pro (with domain subsets), GPQA-Diamond, AIME-24/25, ToolBench.
  - Multi-turn: ALFWorld, BabyAI, ScienceWorld, PDDL. — [Evo-Memory arXiv](https://arxiv.org/html/2511.20857v1)
- Methods compared: no-memory/history, ReAct, A-Mem, SelfRAG, MemOS, Mem0, LangMem, Dynamic Cheatsheet (two variants), AWM, ExpRecent, ExpRAG, and the paper's own ReMem. — [Evo-Memory arXiv](https://arxiv.org/html/2511.20857v1)
- Backbones were Gemini-2.5 Flash, Flash-Lite and Pro, plus Claude 3.5-Haiku and 3.7-Sonnet. No open-weight backbone is reported. — [Evo-Memory arXiv](https://arxiv.org/html/2511.20857v1)
- Average single-turn scores (Gemini 2.5 Flash): baseline 0.59, ReAct 0.37, ExpRAG 0.60, ReMem 0.65. On ALFWorld with Claude 3.7 Sonnet, ReMem reached 0.92 success against 0.50 for the baseline. — [Evo-Memory arXiv](https://arxiv.org/html/2511.20857v1)
- At v1 the code was "will release"; no GitHub link appears in the paper. — [Evo-Memory arXiv](https://arxiv.org/html/2511.20857v1)

**LifelongAgentBench (arXiv 2505.11942, May 2025; v3)**
- Three environments, each with skill labels:
  - Database (SQL): 500 tasks, 22 SQL skills.
  - Operating System (Bash): 500 tasks, 29 Bash skills.
  - Knowledge Graph: 396 tasks, 7 action types. — [LifelongAgentBench arXiv HTML](https://arxiv.org/html/2505.11942)
- Tasks are "skill-grounded and explicitly designed to quantify inter-task relatedness". They run in sequence, have simple, intermediate and complex difficulty levels, and come with automatic label verification. — [LifelongAgentBench arXiv HTML](https://arxiv.org/html/2505.11942); [search summary](https://arxiv.org/pdf/2505.11942)
- Experience replay with Llama-3.1-8B:
  - DB rose from 19% to 78% with 64 replayed trajectories.
  - OS rose from 43% to 50% at 4–16 trajectories, then degraded.
  - KG rose from 28% to 35% with one trajectory, then plateaued.
  - Replaying more than the optimum gives "diminishing or negative returns due to excessive input length". — [LifelongAgentBench arXiv HTML](https://arxiv.org/html/2505.11942)
- Models included Llama-3.1 8B/70B, Qwen2.5 7B/32B and DeepSeek-R1 variants. Open models were evaluated, so local execution is the expected use. — [LifelongAgentBench arXiv HTML](https://arxiv.org/html/2505.11942)
- Code: https://github.com/caixd-220529/LifelongAgentBench. Data: https://huggingface.co/datasets/csyq/LifelongAgentBench — [LifelongAgentBench PDF](https://arxiv.org/pdf/2505.11942)

**SWE-Bench-CL (arXiv 2507.00014, mid-2025)**
- 273 SWE-bench Verified tasks from 8 Python repos: django, sympy, sphinx, matplotlib, scikit-learn, astropy, xarray, pytest. — [SWE-Bench-CL arXiv](https://arxiv.org/html/2507.00014v1)
- Ordering and labels:
  - Tasks are in chronological order within each repo, with a curriculum by human fix-time label (<15 min, 15 min–1 h, 1–4 h, >4 h).
  - Dependencies were found by file-path overlap: 13–59% of tasks per repo have one. — [SWE-Bench-CL arXiv](https://arxiv.org/html/2507.00014v1)
- Metrics: ACC, forgetting, forward and backward transfer, CL-F1/CL-Fβ, and tool-use efficiency. The paper provides a LangGraph agent with a FAISS memory. — [SWE-Bench-CL arXiv](https://arxiv.org/html/2507.00014v1)
- **No validated memory results.** The harness experiments had pass rates below 8.5%, and the authors say this "made it impossible to draw reliable conclusions". Repo: https://github.com/thomasjoshi/agents-never-forget/ — [SWE-Bench-CL arXiv](https://arxiv.org/html/2507.00014v1)

**MemoryAgentBench (arXiv 2507.05257; ICLR 2026 poster)**
- Tests four competencies: accurate retrieval, test-time learning, long-range understanding and selective forgetting. It splits existing datasets into dialogue chunks and adds two new datasets, EventQA and FactConsolidation. — [MemoryAgentBench arXiv](https://arxiv.org/pdf/2507.05257); [ICLR page](https://iclr.cc/virtual/2026/poster/10010781)
- It measures long-context and conversational memory, not procedural or coding experience. That makes it a poor fit for "lessons learned" in a coding agent (my inference from the competency list).

**MemRL (arXiv 2601.03192; v2 12 Feb 2026)**
- Benchmarks are HLE, BigCodeBench, ALFWorld, and LifelongAgentBench OS and DB. Each runs for 10 epochs and is scored by last-epoch success rate and cumulative success rate. — [MemRL arXiv v2](https://arxiv.org/html/2601.03192v2)
- Baselines: No Memory, RAG, Self-RAG, MemP, Mem0 and Pass@k. There is no static-prompt or prompt-optimization baseline. — [MemRL arXiv v2](https://arxiv.org/html/2601.03192v2)
- Results (no-memory / best baseline / MemRL):

  | Benchmark | No memory | Best baseline | MemRL |
  |---|---|---|---|
  | BigCodeBench | 0.485 | 0.578 (MemP) | 0.595 |
  | OS | 0.860 | 0.736 (MemP) | 0.788 |
  | DB | 0.777 | 0.960 (MemP) | 0.960 |
  | ALFWorld | 0.357 | 0.522 (MemP) | 0.570 |

  The gain on HLE was only +0.4%. HLE has low internal semantic similarity (0.186). — [MemRL arXiv v2](https://arxiv.org/html/2601.03192v2)
- On OS tasks, the no-memory baseline (0.860) beat every memory method as reported (MemRL 0.788). The fetch summary's row is internally odd, so check it against the paper table before citing. — [MemRL arXiv v2](https://arxiv.org/html/2601.03192v2)
- Backbones were GPT-4o, GPT-4o-mini, GPT-5-mini and Gemini-3-pro, chosen per benchmark. Code: https://github.com/MemTensor/MemRL — [MemRL arXiv v2](https://arxiv.org/html/2601.03192v2)

**ReasoningBank (arXiv 2509.25140, 29 Sep 2025, Google Cloud AI Research)**
- WebArena has 684 tasks: Shopping 187, Admin 182, GitLab 180, Reddit 106, Multi 29. — [ReasoningBank arXiv](https://arxiv.org/html/2509.25140v1)
- Mind2Web has 1,341 tasks: Cross-Task 252, Cross-Website 177, Cross-Domain 912. — [ReasoningBank arXiv](https://arxiv.org/html/2509.25140v1)
- SWE-Bench-Verified has 500 tasks. — [ReasoningBank arXiv](https://arxiv.org/html/2509.25140v1)
- Baselines: No Memory, Synapse and AWM. Backbones: Gemini-2.5-Flash/Pro and Claude-3.7-Sonnet. — [ReasoningBank arXiv](https://arxiv.org/html/2509.25140v1)
- WebArena gains over no-memory were +8.3, +7.2 and +4.6 points across the three backbones. On SWE-Bench-Verified the gain was +4.6 (Gemini-2.5-Flash), with about 3 fewer steps per task. MaTTS (k=5) added roughly 3 more points on WebArena. — [ReasoningBank arXiv](https://arxiv.org/html/2509.25140v1); [Google Research blog](https://research.google/blog/reasoningbank-enabling-agents-to-learn-from-experience/)

**Agent Workflow Memory, AWM (arXiv 2409.07429; ICML 2025)**
- Induces reusable workflows either offline (from training examples) or online (from test queries, filtered by an evaluator module that judges correctness). — [arXiv](https://arxiv.org/html/2409.07429v1); [ICML](https://icml.cc/virtual/2025/poster/45496)
- Relative success-rate gains over baseline were +24.6% on Mind2Web and +51.1% on WebArena. Online AWM beat baselines by 8.9 to 14.0 absolute points as the cross-task, website and domain gap widened. — [Agent Workflow Memory review](https://hub.athina.ai/research-papers/agent-workflow-memory); [themoonlight review](https://www.themoonlight.io/review/agent-workflow-memory)

**ExpeL (arXiv 2308.10144; AAAI 2024)**
- Extracts natural-language insights from training tasks and recalls insights plus similar trajectories at inference. Benchmarks are HotpotQA, ALFWorld, WebShop, plus FEVER for transfer. — [ExpeL arXiv](https://arxiv.org/html/2308.10144v3); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29936)
- Reported results: ALFWorld "put" tasks 83% against ReAct's 50%; WebShop 0.701 against 0.665. — [ExpeL search summary via arXiv](https://arxiv.org/pdf/2308.10144)
- Insight extraction and task similarity both contribute to the gains. — [ExpeL arXiv](https://arxiv.org/html/2308.10144v3)

**Agent KB (arXiv 2507.06229, OPPO)**
- A cross-framework experience knowledge base with a "Reason-Retrieve-Refine" loop and a disagreement gate. Evaluated on GAIA, HLE, GPQA and SWE-bench. — [Agent KB arXiv](https://arxiv.org/pdf/2507.06229)
- Up to +16.36 points on GAIA. GPT-4.1 on Level 2 went from 53.49 to 73.26. — [Agent KB arXiv](https://arxiv.org/pdf/2507.06229)

**Dynamic Cheatsheet (arXiv 2504.07952; EACL 2026)**
- Evolving memory of strategies and code snippets, used with a black-box LM. — [DC arXiv](https://arxiv.org/html/2504.07952v1)
- Results:
  - Game of 24 (GPT-4o): about 10% to 99%, by reusing a Python solver.
  - AIME (Claude 3.5 Sonnet): more than doubled.
  - Equation balancing: near-perfect against a baseline around 50%.
  - GPQA-Diamond: +9%. MMLU-Pro Engineering and Physics: +8%. — [DC arXiv](https://arxiv.org/html/2504.07952v1); [ACL Anthology](https://preview.aclanthology.org/ingest-eacl/2026.eacl-long.333)

**AppWorld (arXiv 2407.18901; ACL 2024) and ACE (arXiv 2510.04618; ICLR 2026)**
- AppWorld has 9 apps, 457 APIs, about 100 simulated users and 750 tasks. Grading uses state-based unit tests that also detect collateral damage. Tasks are split into "normal" and "challenge". GPT-4o solved about 49% of normal tasks and 30% of challenge tasks. — [AppWorld arXiv](https://arxiv.org/abs/2407.18901); [appworld.dev](https://appworld.dev/)
- ACE results with DeepSeek-V3.1, an open-weight model. Columns are TGC/SGC on test-normal, then test-challenge, then the average.

  | Method | Test-normal TGC/SGC | Test-challenge TGC/SGC | Average |
  |---|---|---|---|
  | ReAct | 63.7/42.9 | 41.5/21.6 | 42.4 |
  | +ICL | 64.3/46.4 | 46.0/27.3 | 46.0 |
  | +GEPA (static prompt optimization) | 64.9/44.6 | 46.0/30.2 | 46.4 |
  | +ACE offline (labels) | 76.2/64.3 | 57.3/39.6 | 59.4 |
  | +DC online (no labels) | 65.5/58.9 | 52.3/30.8 | 51.9 |
  | +ACE online (no labels) | 69.6/53.6 | 66.0/48.9 | 59.5 |

  — [ACE arXiv HTML](https://arxiv.org/html/2510.04618v1)
- ACE says it cuts adaptation latency by 86.9% and rollout cost by 75% compared with GEPA and DC. — [ACE (HF papers)](https://huggingface.co/papers/2510.04618)

**Terminal-Bench 2.0, τ-bench, and others named in the brief**
- Terminal-Bench 2.0 has 89 tasks spanning software engineering, security, biology and gaming. Each task has its own Docker container, a human oracle solution, and tests on the final container state. Harbor is the official harness. Local models must be served behind an API, for example with vLLM. — [evalscope docs](https://evalscope.readthedocs.io/en/v1.5.1/third_party/terminal_bench.html); [Harbor TB example](https://mintlify.com/harbor-framework/harbor/examples/terminal-bench)
- τ-bench v1 has 165 tasks (115 retail, 50 airline), with domain policies and a pass^k reliability metric. — [τ-bench arXiv](https://arxiv.org/pdf/2406.12045); [benchlm summary](https://benchlm.ai/benchmarks/tau-bench)

**Newer 2026 continual-memory benchmarks** (I mostly saw abstracts only)
- AgentCL (arXiv 2606.02461, June 2026; revised 26 Sep 2026)
  - Covers coding, deep research, and language understanding/reasoning streams.
  - Distinguishes three stream types: "dependent" (later tasks reuse earlier knowledge), "conventional" and "independent".
  - Dependent streams separate memory designs better. Conventional and independent streams show limited gains and "memory-induced degradation".
  - Proposes the MemProbe method. — [AgentCL arXiv](https://arxiv.org/abs/2606.02461)
- VibeMemBench (arXiv 2609.23570, Sep 2026)
  - 111 coding targets from 90 SWE-bench V2 repos, with 3,634 history trajectories. Targets were kept only if injecting the history verifiably helped.
  - Direct injection of experience improved resolution by 1.1–4.5 points on 4 of 5 solvers.
  - "Eleven of twelve solver and system pairings fail to exceed the matched memory-off baseline" when the memory systems build and retrieve experience themselves. — [VibeMemBench arXiv](https://arxiv.org/abs/2609.23570)
- AgentMemoryBench (ICLR 2026)
  - Settings: code-centric tool use, embodied, web and long-horizon dialogue, including interleaved task streams.
  - Five modes: improvement, retention, forgetting, generalization and knowledge-conflict resolution.
  - Code at an anonymous link: https://anonymous.4open.science/r/AgentMemoryBench-12FD — [ICLR page](https://iclr.cc/virtual/2026/10012519)
- Continual Learning Bench (arXiv 2606.05661, June 2026)
  - Six domains with learnable latent structure: software engineering, signal processing, outbreak forecasting, database querying, game-playing and demand forecasting.
  - Finding: "naive ICL outperforms systems dedicated to memory management". — [CL Bench arXiv](https://arxiv.org/abs/2606.05661)
- PATH-Bench (arXiv 2608.01149, Aug 2026)
  - Single-turn code generation and multi-turn tool use.
  - Uses probe-centred sequences with controlled helpful and interfering histories. Eight lifelong agents were tested.
  - Findings: "strong transfer doesn't guarantee retention" and "later experiences can reshape earlier gains". — [PATH-Bench arXiv](https://arxiv.org/abs/2608.01149)
- Other entries surfaced by search but not opened:
  - PAST-Bench (arXiv 2608.04003): evaluates "task families" of fresh-session episodes that share a latent rule, a reusable artifact, a correction or a reference.
  - MemoryBench (arXiv 2510.17281).
  - StreamMemBench (arXiv 2606.14571). — [search results: PAST-Bench](https://arxiv.org/pdf/2608.04003); [MemoryBench](https://arxiv.org/pdf/2510.17281)

### Inferences
- **Best fit for a local Qwen3-Coder-30B coding agent, in priority order:**
  1. LifelongAgentBench DB+OS. Skill labels (22+29) allow familiar / novel-combination / novel-skill splits. It is locally executable (open models are already in the paper). MemRL reports results on it.
  2. AppWorld. It is code-as-action, has state-based unit tests, and its 9 apps / 457 APIs supply many varied API "gotchas" that are hard to fit into one prompt. ACE already provides ICL and GEPA static baselines with an open model.
  3. StreamBench DS-1000 and BIRD/Spider. These are cheap single-turn streams that already include zero-shot, few-shot and CoT static baselines.
  4. BigCodeBench, following the MemRL protocol.
  5. SWE-Bench-CL or VibeMemBench for realism. They are expensive, and the 30B solve rates will be low.
- **Category labels:** LifelongAgentBench has skills; SWE-Bench-CL has repo and difficulty; WebArena has domain; Mind2Web has cross-task / website / domain; AppWorld has app and difficulty; StreamBench has dataset-level difficulty; τ-bench has domain.
  - Mind2Web's three-way split is the only ready-made "novel category" split. It is a web-navigation (non-coding) benchmark.
  - Novel-combination splits would need to be built from LifelongAgentBench skill co-occurrence (multi-skill tasks) or AppWorld multi-app tasks.

### Gaps
- I could not confirm whether Evo-Memory code has since been released. Its v1 says "will release".
- I did not obtain AppWorld split sizes (train/dev/test-normal/test-challenge), OSWorld, BrowseComp-Plus or ScienceWorld sizes from fetched sources.
- I did not verify which memory baselines AgentMemoryBench, PAST-Bench, MemoryBench or StreamMemBench include. Only abstracts were seen.
- The MemRL OS-task row (no memory 0.860 beating all memory methods) needs checking against the PDF table.

## Which benchmarks provide many recurring-but-varied task families (category taxonomies)?

### Takeaway
LifelongAgentBench has the richest explicit taxonomy: 22 SQL skills, 29 Bash skills and 7 KG action types, with designed inter-task relatedness. AppWorld (9 apps, 457 APIs) and WebArena/Mind2Web (domains and websites) provide natural families. PAST-Bench and AgentCL deliberately construct "task families" and "dependent streams". Evo-Memory and MemRL show that gains track within-stream similarity, so families must recur for memory to pay off.

### Cited Findings
- LifelongAgentBench: tasks are skill-grounded with explicit inter-task relatedness. Counts are DB 500/22 skills, OS 500/29 skills and KG 396/7 action types. — [LifelongAgentBench](https://arxiv.org/html/2505.11942)
- Evo-Memory: ReMem's improvement correlates with within-dataset task similarity (Pearson r=0.717 on Gemini 2.5 Flash, r=0.563 on Claude 3.7). PDDL has high coherence and showed larger gains. AIME-25 is diverse and showed smaller gains. — [Evo-Memory](https://arxiv.org/html/2511.20857v1)
- MemRL: gains "correlate strongly with task similarity". HLE (similarity 0.186) gained only +0.4%. — [MemRL](https://arxiv.org/html/2601.03192v2)
- AgentCL: "dependent" streams, where later tasks reuse earlier knowledge, discriminate memory designs. Independent streams expose memory-induced degradation. — [AgentCL](https://arxiv.org/abs/2606.02461)
- PAST-Bench's unit is a task family sharing a latent rule, a reusable artifact, a correction or a pre-seeded reference, with context wiped between episodes. — [PAST-Bench](https://arxiv.org/pdf/2608.04003)
- SWE-Bench-CL groups tasks by repo (8 families) and fix-time difficulty. 13–59% of tasks per repo depend on earlier tasks through file overlap. — [SWE-Bench-CL](https://arxiv.org/html/2507.00014v1)
- AppWorld spans 9 apps and 457 APIs across 750 tasks. — [AppWorld](https://arxiv.org/abs/2407.18901)
- Terminal-Bench 2.0's 89 tasks fall into categories (software engineering, security, biology, gaming). That is likely too few per category for a stream study (my inference from the size). — [evalscope docs](https://evalscope.readthedocs.io/en/v1.5.1/third_party/terminal_bench.html)

### Inferences
- The "many varied lessons" requirement favours AppWorld (API-specific quirks across 457 APIs) and LifelongAgentBench OS (29 Bash skills). One static prompt would have to encode many API- or skill-specific facts. In AppWorld, ACE's evolving playbook beats GEPA's optimized static prompt by about 13 points average.
- The similarity correlations imply a design rule: hold out whole skills or apps for the "novel category" split, and expect memory gains near zero there. A static prompt could match memory on the novel split, which is a useful control.

### Gaps
- I found no benchmark that ships ready-made "familiar / novel combination / novel category" splits for coding. Only Mind2Web (cross-task/website/domain) does this, and only for the web.
- I did not obtain per-skill task counts for LifelongAgentBench, which are needed to size the splits.

## Which show memory gains that persist beyond what a prompt hint / static prompt provides?

### Takeaway
Only a few papers include a static-prompt or prompt-optimization control:
- ACE on AppWorld: evolving context 59.5 against GEPA 46.4 and ICL 46.0, with the open DeepSeek-V3.1.
- StreamBench: streaming ICL beats zero-shot, few-shot and CoT.
- Continual Learning Bench: naive ICL beats dedicated memory systems.

Most memory papers compare only against no-memory (MemRL, ReasoningBank, AWM, Evo-Memory, Agent KB). Several 2026 benchmarks (VibeMemBench, AgentCL, CL Bench, PATH-Bench) report that memory systems often fail to beat simple baselines, or degrade performance.

### Cited Findings
- ACE (AppWorld, DeepSeek-V3.1):
  - Static prompt optimization (GEPA) gives an average of 46.4 and ICL gives 46.0, against 42.4 for plain ReAct.
  - Online ACE gives 59.5 and online Dynamic Cheatsheet gives 51.9.
  - On test-challenge TGC, online ACE scores 66.0 against GEPA's 46.0. — [ACE](https://arxiv.org/html/2510.04618v1)
- StreamBench includes zero-shot, few-shot, CoT and Self-Refine as non-streaming baselines. Streaming methods (Self-StreamICL, MAM-StreamICL) were best. — [StreamBench](https://arxiv.org/html/2406.08747v1)
- Continual Learning Bench: "naive ICL outperforms systems dedicated to memory management". Agents "overfit to immediate observations or fail to reuse knowledge across instances". — [CL Bench](https://arxiv.org/abs/2606.05661)
- VibeMemBench: 11 of 12 solver-and-memory-system pairings fail to beat memory-off. Only directly injected, pre-verified experience helps (+1.1–4.5 points). — [VibeMemBench](https://arxiv.org/abs/2609.23570)
- AgentCL: conventional and independent streams show "memory-induced degradation". — [AgentCL](https://arxiv.org/abs/2606.02461)
- LifelongAgentBench: replay gains on OS peak at 4–16 examples and then decline. Too much context hurts. — [LifelongAgentBench](https://arxiv.org/html/2505.11942)
- Evo-Memory: ReAct (0.37) scored below the plain baseline (0.59) on single-turn tasks, and ExpRAG was only +0.01. Large gains appear mainly on multi-turn, high-similarity environments (ALFWorld 0.50 to 0.92). — [Evo-Memory](https://arxiv.org/html/2511.20857v1)
- StreamBench: incorrect self-generated examples can degrade performance even when labelled as wrong. — [StreamBench](https://arxiv.org/html/2406.08747v1)
- A paper titled "Large Language Model Agents Are Not Always Faithful Self-Evolvers" (arXiv 2601.22436) surfaced in search. I did not open it. — [arXiv](https://arxiv.org/pdf/2601.22436)

### Inferences
- **The strongest prior evidence that evolving memory beats an optimized static prompt is ACE vs GEPA on AppWorld.** It uses an open model, but a larger one than 30B. This makes AppWorld the most informative replication target for the hypothesis.
- **A fair static-prompt control is rarely run.** The project's "equivalent static prompt" arm would be a novel contribution in most of these benchmarks. A useful candidate is the final memory contents frozen into a prompt, or GEPA/MIPRO-optimized on a training slice.
- The 2026 negative results (CL Bench, VibeMemBench, AgentCL) suggest the null hypothesis is plausible. Effects may be small for a 30B model unless streams are "dependent".

### Gaps
- I found no paper that compares "frozen final memory as a static prompt" against "per-task retrieved memory" directly. That is the exact control this project needs.
- ACE's offline/online numbers use DeepSeek-V3.1 (a large MoE model). I found no results for a model around 30B.

## Practical cost to run with a 30B local model

### Takeaway
The cheapest options are StreamBench (single-turn SQL and Python, with execution grading), LifelongAgentBench (open models already tested, Docker DB/OS) and BigCodeBench. AppWorld is local and self-contained but multi-turn and long-context. SWE-bench-based streams (SWE-Bench-CL, VibeMemBench) and Terminal-Bench need per-task Docker images and long trajectories. WebArena needs hosted web apps. Several leading papers used only API models.

### Cited Findings
- LifelongAgentBench evaluated Llama-3.1-8B/70B, Qwen2.5-7B/32B and DeepSeek-R1 variants. In KG, group self-consistency cut tokens from 56,409 to 11,002 per task for Llama-3.1-8B without accuracy loss. — [LifelongAgentBench](https://arxiv.org/html/2505.11942)
- Terminal-Bench 2.0 only supports API-served models, so local models need vLLM or a similar server. Each task runs in its own Docker container. — [evalscope docs](https://evalscope.readthedocs.io/en/v1.5.1/third_party/terminal_bench.html)
- SWE-Bench-CL baseline pass rates were below 8.5% in the authors' harness. The models they listed for future tests include Gemma3-27B. — [SWE-Bench-CL](https://arxiv.org/html/2507.00014v1)
- AppWorld runs as a 60k-line local engine with 9 apps and needs no external web services. — [AppWorld](https://arxiv.org/abs/2407.18901)
- MemRL runs 10 epochs per benchmark, which multiplies cost by 10 compared with a single pass. — [MemRL](https://arxiv.org/html/2601.03192v2)
- StreamBench streams range from 750 to 2,147 single-turn items, graded by execution or exact match. — [StreamBench](https://arxiv.org/html/2406.08747v1)
- The backbones in Evo-Memory, ReasoningBank and MemRL were all API models (Gemini, Claude, GPT). — [Evo-Memory](https://arxiv.org/html/2511.20857v1); [ReasoningBank](https://arxiv.org/html/2509.25140v1); [MemRL](https://arxiv.org/html/2601.03192v2)

### Inferences
- **Rough cost tiers for Qwen3-Coder-30B on local GPUs:**
  - Single-turn StreamBench and BigCodeBench: hours.
  - LifelongAgentBench: about 1,000 short multi-turn tasks, also hours.
  - AppWorld: hundreds of long multi-turn tasks, a day or more per arm.
  - SWE-bench-derived streams: about 273–500 tasks with heavy Docker images and long contexts, days per arm.

  These are estimates, not measured.
- Absolute performance from the API-model papers will not carry over. The 30B baseline may sit near the floor on SWE-Bench-CL (the authors already saw below 8.5%), which would hide any memory effect. Prefer benchmarks where the 30B baseline sits in the 30–70% range.

### Gaps
- I found no published Qwen3-Coder-30B numbers on AppWorld, LifelongAgentBench or StreamBench.
- I have no measured token or GPU-hour costs per benchmark for local models.
