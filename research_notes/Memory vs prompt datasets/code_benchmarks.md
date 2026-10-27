# Code-repair and code-efficiency datasets for memory-vs-static-prompt experiments (ELH)

Scope: candidate task sources for a frozen Qwen3-Coder-30B-A3B agent (bash + submit, hidden oracle grading integrity / correctness / performance), where the experiment needs (a) a labelled cause/category so we can split train/eval by cause and test transfer to a new cause, (b) local executability, (c) low contamination, (d) a difficulty a ~30B local model can reach. Research run 2026-10-27. Facts carry inline sources; anything not checked this session is listed under Gaps.

## Q1. Performance and efficiency benchmarks: what exists, and which have inefficiency category labels?

### Takeaway
The repository-level perf benchmarks (GSO, SWE-Perf, SWE-fficiency, plus the June 2026 SWE-Pro preprint) are real and agentic, but they are small (102-498 tasks), dominated by numpy/pandas/scipy-style repos, very hard for frontier models (GSO Opt@1 under 5 percent; SWE-fficiency under 0.23x expert speedup), and a July 2026 audit found large timing-noise problems. Only SWE-fficiency ships a category label (9 LM-assigned optimization types); none label the root-cause *inefficiency pattern* the way ELH's latent causes do. Function-level efficiency sets (PIE, ECCO, Mercury, EffiBench(-X), EvalPerf, COFFE) are large and cheap to run but are "write or rewrite one solution" tasks, not repo editing, and also lack cause labels.

### Cited Findings

**GSO (UC Berkeley; NeurIPS 2025 D&B)**
- 102 optimization tasks from 10 codebases, 5 languages (Python, C, C++, Cython, Rust), 8 domains. Tasks were mined from commit histories, with generated performance tests — [arXiv 2505.23671](https://arxiv.org/abs/2505.23671)
- Format: the agent gets the codebase and a performance test as the spec and must improve runtime. Metric is Opt_p@K: the patch must pass the correctness tests and reach at least p (0.95) of the human speedup, using the harmonic mean over tests — [arXiv 2505.23671v3](https://arxiv.org/html/2505.23671v3)
- Opt@1 is under 5 percent for all models (best about 4 percent, Claude-4.0); Opt@10 is about 15 percent. Python-only tasks reach 21.4 percent, non-Python tasks 4.0 percent. Failure modes: avoiding low-level languages, "lazy" input-specific fast paths, localizing the wrong bottleneck, stopping early (75 percent of trajectories end before 100 steps) — [arXiv 2505.23671v3](https://arxiv.org/html/2505.23671v3)
- Gold commits average 250 changed lines (median 110). No optimization-type taxonomy is reported — [arXiv 2505.23671v3](https://arxiv.org/html/2505.23671v3)
- Availability: [github.com/gso-bench/gso](https://github.com/gso-bench/gso), [HF gso-bench/gso](https://huggingface.co/datasets/gso-bench/gso). Source repos are under permissive licenses (BSD/MIT/Apache/HPND) — [arXiv 2505.23671v3](https://arxiv.org/html/2505.23671v3); [project page](https://sky.cs.berkeley.edu/project/gso/)

**SWE-Perf (July 2025)**
- 140 instances from performance-improving PRs in 9 repos (astropy, matplotlib, sklearn, xarray, sympy, ...). Each ships the codebase, target functions, perf tests, an expert patch and a Docker environment — [arXiv 2507.12415](https://arxiv.org/html/2507.12415v1)
- Measurement: 20 runs per test with warm-up, IQR outlier removal, Mann-Whitney U at p < 0.1, single CPU core. Two settings: "oracle" (file-level) and "realistic" (repo-level agent) — [arXiv 2507.12415](https://arxiv.org/html/2507.12415v1)
- Open-model results (oracle): Qwen3-235B 0.68 percent performance, DeepSeek-R1 0.90 percent, DeepSeek-V3 0.54 percent, with correctness around 43-51 percent. Agentic: OpenHands 2.26 percent vs Agentless 0.41 percent. No explicit optimization-type taxonomy — [arXiv 2507.12415](https://arxiv.org/html/2507.12415v1); homepage [swe-perf.github.io](https://swe-perf.github.io)

**SWE-fficiency (Google DeepMind et al.; ICML 2026)**
- 498 tasks across 9 repos (astropy, dask, matplotlib, numpy, pandas, scikit-learn, scipy, sympy, xarray). The agent gets a full repo and a slow workload and must match or beat the expert speedup while passing the same unit tests — [arXiv 2511.06090](https://arxiv.org/pdf/2511.06090); [ICML 2026 poster](https://icml.cc/virtual/2026/poster/66745)
- Metric: Speedup Ratio (SR = LM speedup / expert speedup), harmonic mean across tasks. Top agents get under 0.23x the expert speedup — [arXiv 2511.06090](https://arxiv.org/pdf/2511.06090)
- **Has category labels**: an LM assigns each task one of 9 types (Algorithmic/Data-Structure; Memory; Concurrency/Parallelism; I/O; Code simplification/dead code; Compiler/build/low-level; Config/parameter tuning; Caching & reuse; Unknown). Per-category counts were not in the version fetched — [arXiv 2511.06090 HTML](https://arxiv.org/html/2511.06090)
- Open-weight SR: Qwen3 Coder Plus 0.068x, Kimi K2-0905 0.054x, DeepSeek V3.1 0.043x, GLM-4.6 0.042x. Runs in Docker with CPU core pinning — [arXiv 2511.06090 HTML](https://arxiv.org/html/2511.06090)

**SWE-Pro (preprint, 24 June 2026)**
- Repository-level benchmark from 102 expert-written optimizations, graded on runtime, peak memory and time-weighted memory usage. Experts reach an aggregate 15.5x speedup and 171.3x memory reduction. No inefficiency taxonomy and no public repo were found in the abstract page — [arXiv 2606.25530](https://arxiv.org/abs/2606.25530)

**Reliability audit of repo-level perf benchmarks (16 July 2026)**
- Replaying 740 reference patches on 4 GCP machine types, only 39/102 GSO, 11/140 SWE-Perf and 411/498 SWE-fficiency tasks kept their validity signal on every machine. SWE-Perf reference patches cluster near zero change (median -0.03 percent); its noise-to-signal ratio is 43.23x, against 0.07x for GSO — [arXiv 2607.01211](https://arxiv.org/html/2607.01211)
- In SWE-fficiency's harmonic-mean score, the worst 10 tasks carry 58.5-82.8 percent of the weight. GSO and SWE-fficiency rankings disagree on 9 of 28 pairwise comparisons. 384 of 450 replay-valid tasks are already matched or beaten by some public submission — [arXiv 2607.01211](https://arxiv.org/html/2607.01211)

**Function-level efficiency benchmarks**
- PIE: more than 77K C++ pairs of human performance-improving edits from CodeNet competitive programming, with unit tests and execution-time annotations (ICLR 2024) — [arXiv 2302.07867](https://arxiv.org/pdf/2302.07867)
- ECCO: Python, sourced from CodeNet. Measures correctness, runtime and memory under two paradigms (history-based editing and NL-based generation) — [arXiv 2407.14044](https://arxiv.org/pdf/2407.14044v2). The size given in a comparison table is 48 tasks, about 20 tests each — [arXiv 2607.01211 / survey table via search](https://arxiv.org/html/2607.01211) (this is low against my recollection of ECCO's size; check it)
- Mercury: Python, from LeetCode, metrics Pass and "Beyond"; listed as 256 tasks (eval split) with about 18 solutions each — [Mercury](https://www.researchgate.net/publication/397196083_Mercury_A_Code_Efficiency_Benchmark_for_Code_Large_Language_Models)
- EvalPerf: 121 perf-exercising tasks from HumanEval/MBPP (EvalPlus), metric DPS / DPS_norm — [search summary of comparison table](https://aclanthology.org/2025.acl-long.1387.pdf)
- EffiBench / Mercury both use LeetCode problems, scored on runtime and memory — [EffiBench-X arXiv 2505.13004](https://arxiv.org/pdf/2505.13004)
- EffiBench-X (NeurIPS 2025 D&B): the first multi-language efficiency benchmark (Python, C++, Java, JavaScript, Ruby, Go) — [arXiv 2505.13004](https://arxiv.org/pdf/2505.13004); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/6b8198f777d1ba87a8250368db2ea16a-Paper-Datasets_and_Benchmarks_Track.pdf); [HF EffiBench/effibench-x](https://huggingface.co/datasets/EffiBench/effibench-x)
- COFFE (FSE 2025): 398 function-level and 358 file-level problems. Its efficient@k metric uses CPU instruction counts for stable comparison — [arXiv 2502.02827](https://arxiv.org/pdf/2502.02827); [FSE 2025](https://conf.researchr.org/details/fse-2025/fse-2025-research-papers/34/COFFE-A-Code-Efficiency-Benchmark-for-Code-Generation)

### Inferences
- None of these labels the *root-cause pattern* (quadratic membership, nested-scan join, O(n) queue pop, and so on) in a form that supports "train on causes A-D, test on novel cause E". The SWE-fficiency 9-way labels are coarse: most ELH-style causes would land in "Algorithmic/Data Structure", and the labels are LM-assigned. ELH would have to relabel tasks itself (for example, LM-classify gold diffs into a fine-grained taxonomy and then hand-check them).
- GSO, SWE-Perf and SWE-fficiency are probably at floor for a 30B-A3B model. SWE-Perf's 0.5-2 percent "performance" rates for much larger models, and SWE-fficiency's 0.04-0.07x SR for large open models, mean memory-vs-prompt differences would be lost in noise. The audit also shows SWE-Perf timing is mostly noise. Of the three, SWE-fficiency (largest, most replay-stable at 411/498) is the best candidate for a stretch eval split, ideally restricted to the "Algorithmic/Data Structure" and "Caching & reuse" Python tasks.
- The function-level sets (PIE, Mercury, EffiBench, COFFE, ECCO "history-based editing") can be wrapped as "here is a slow correct solution in a file, make it faster without breaking tests". That gives hundreds to thousands of feasible tasks, but the agentic / repo-navigation part is trivial, and contamination risk is high because LeetCode and CodeNet are widely trained on.
- Hidden-oracle note: GSO's design (perf test given as spec, correctness tests hidden) is closest to ELH's oracle. Its "lazy optimization" failure mode (input-specific fast paths) is exactly the integrity dimension ELH already grades.

### Gaps
- Per-category task counts for SWE-fficiency; whether its dataset and Docker images are now public on HF/GitHub (the fetched paper said "will be open-sourced").
- Exact sizes and licenses for EffiBench (original), EffiBench-X, Mercury (full train set), ECCO (full), EvalPerf, and KernelBench. These were not verified this session. KernelBench (CUDA kernels, arXiv 2502.10517) was not researched; it is likely a poor fit (GPU, not Python repo editing).
- No per-model 30B-class numbers on GSO / SWE-Perf / SWE-fficiency were found.
- SWE-Pro (2606.25530) has the same task count (102) as GSO. It was not checked whether it overlaps GSO or is independent.

## Q2. Synthetic / generated bug corpora with labelled bug types: which allow unlimited fresh instances per category?

### Takeaway
SWE-smith is the strongest fit for "unlimited fresh instances per labelled category". Its generator is open (MIT), runs per repo in Docker, and every instance is created by a named strategy (LM-modify, LM-rewrite, specific AST procedural transforms, PR-mirror, combine), so the strategy name is a usable category label. It only generates correctness bugs, not performance regressions; ELH would need to add perf-mutation operators. R2E-Gym, SWE-Gym, SWE-bench-Live and BugPilot are executable corpora without bug-type labels. Defects4J and BugsInPy are classic, small and heavily contaminated.

### Cited Findings
- SWE-smith: 52K task instances, 250+ Docker images (one per repo), 26K SWE-agent trajectories; MIT license; NeurIPS 2025 D&B spotlight. SWE-agent-LM-32B, trained on it, reaches 40.2 percent pass@1 on SWE-bench Verified — [GitHub SWE-bench/SWE-smith](https://github.com/SWE-bench/SWE-smith)
- SWE-smith generation strategies: (1) LM Modify (55.98 percent yield) and LM Rewrite (35.04 percent yield); (2) procedural AST modifications such as removing a conditional or changing an operator, with per-strategy yield varying (class-function shuffle 1.93 percent, conditional inversion 47.04 percent); (3) PR Mirror (an LM reverts a real PR's change); (4) combining bugs from the same file or module. Only candidates that break at least one existing test are kept — [arXiv 2504.21798](https://arxiv.org/pdf/2504.21798)
- SWE-smith was extended to JavaScript in January 2026 — [swebench.com post](https://www.swebench.com/post-260113-swesmith-javascript.html)
- R2E-Gym: more than 8.1K procedurally curated problems with executable environments, unit tests and NL descriptions. 4,578 remain after removing repos that overlap SWE-bench (R2E-Gym-Subset). CC BY 4.0 — [arXiv 2504.07164](https://arxiv.org/html/2504.07164v1); [HF R2E-Gym](https://huggingface.co/datasets/R2E-Gym/SWE-Bench-Verified)
- SWE-Gym: 2,438 instances from 11 Python repos, MIT — [HF SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym)
- SWE-bench-Live: 1,565 task instances across 164 repos, MIT, continuously refreshed to limit contamination — [HF SWE-bench-Live](https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live)
- BugPilot (October 2025): SWE agents are told to *add features*, and the resulting accidental test breakages become bugs. The authors say these match human edit patterns better than deliberate perturbations. 1.2K such bugs outperform 3K bugs from other sources for training. Their 32B "FrogBoss" model gets 54.6 percent pass@1 on SWE-bench Verified. No bug-type labelling was reported — [arXiv 2510.19898](https://arxiv.org/abs/2510.19898)
- SWE-Hub (March 2026) describes itself as a unified production system for scalable, executable SWE tasks (not examined in depth) — [arXiv 2603.00575](https://arxiv.org/pdf/2603.00575)
- Active-SWE (August 2026): coding agents asked to fix bugs proactively without issue reports (not examined in depth) — [arXiv 2608.04682](https://arxiv.org/pdf/2608.04682)

### Inferences
- **Recommended path for ELH:** reuse SWE-smith's infrastructure (repo-to-Docker env, validation that the change breaks something, issue-text generation), but add *performance-mutation* operators corresponding to labelled causes. Examples: replace a set/dict lookup with list membership; replace a hash join with a nested scan; use list.pop(0) instead of deque; re-sort inside a loop; remove memoization (exponential recursion); repeated string concatenation; N+1 queries; recomputing an invariant inside a loop; quadratic regex/backtracking. Validate each mutation by asserting tests still pass and a workload slows by more than kx. This gives unlimited fresh, uncontaminated instances per cause, in real repos, with known gold (the un-mutated code). Held-out causes then become clean "novel cause transfer" tests. SWE-smith's per-strategy yield table shows that operator-level yields vary a lot, so expect to tune operators.
- SWE-smith's strategy names (for example a remove-conditional procedural transform versus lm_rewrite) work as correctness-bug categories for a secondary repair experiment: split train/eval by strategy and test whether memory transfers across mutation types.
- R2E-Gym, SWE-Gym and SWE-bench-Live give realistic held-out *repair* tasks without labels. SWE-bench-Live's rolling freshness is the best contamination control among real-issue sets.

### Gaps
- Multi-SWE-bench (multi-language; I recall about 1.6K instances in 7 languages, arXiv 2504.02605), Defects4J (Java; I recall 835 bugs in v2.0) and BugsInPy (Python; I recall 493 bugs from 17 projects) were not verified this session. They have bug-type information only through follow-up taxonomies, are small, and are old enough to be likely contaminated.
- Whether SWE-smith's released HF instances carry the strategy name as an explicit field was not confirmed (the README fetch did not show per-instance labels; instance IDs are believed to encode the strategy, which is unverified).
- No public generator was found that produces *performance* regressions with labelled causes. A targeted search returned nothing relevant, which suggests this is an open niche, but absence is not proven.

## Q3. Feasibility for a 30B local model

### Takeaway
On repair benchmarks, a 30B-class open model is in a workable range (roughly 40-60 percent on SWE-bench Verified for 32B fine-tunes and Qwen3-Coder-30B-A3B), so SWE-smith-style or SWE-Gym tasks give a measurable signal. On repo-level performance benchmarks, even 235B-480B-class open models are near floor, so GSO, SWE-Perf and SWE-fficiency are too hard for a clean memory-vs-prompt comparison unless they are filtered to an easy Python subset.

### Cited Findings
- Qwen3-Coder-30B-A3B-Instruct: Apache-2.0. The HF card does not list SWE-bench numbers inline — [HF model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct). One aggregator claims 60.4 percent on SWE-bench Verified — [llmrun.dev](https://llmrun.dev/model/qwen-qwen3-coder-30b-a3b-instruct/benchmarks). This is unverified and conflicts with my recollection of Qwen's own blog figure (about 51.6 percent with OpenHands), so treat it as uncertain.
- 32B fine-tunes: SWE-agent-LM-32B 40.2 percent ([SWE-smith](https://github.com/SWE-bench/SWE-smith)); FrogBoss-32B 54.6 percent ([BugPilot](https://arxiv.org/abs/2510.19898)) on SWE-bench Verified.
- Perf benchmarks: SWE-Perf oracle performance is 0.54-0.90 percent for DeepSeek-V3/R1 and Qwen3-235B ([arXiv 2507.12415](https://arxiv.org/html/2507.12415v1)); SWE-fficiency SR is 0.042-0.068x for Qwen3 Coder Plus, Kimi K2, DeepSeek V3.1 and GLM-4.6 ([arXiv 2511.06090](https://arxiv.org/html/2511.06090)); GSO Opt@1 is under 5 percent for frontier models but 21.4 percent on Python-only tasks ([arXiv 2505.23671](https://arxiv.org/html/2505.23671v3)).

### Inferences
- For ELH, the most informative difficulty band is about 30-70 percent baseline success, so that both memory and static-prompt arms have room to move. Generated perf regressions in real repos, with one injected cause per task, should land there for a 30B-A3B model because localization is the main difficulty and the fix is a known idiom. Real-world perf PRs will not.
- A static prompt can list the ELH causes as hints. To separate memory from prompt, the eval set needs causes that are absent from any fixed prompt (novel-cause split) and repo-specific conventions that only episodic memory captures (the same repo seen repeatedly, as in SWE-Bench-CL).

### Gaps
- No published 30B-A3B results on GSO, SWE-Perf, SWE-fficiency, EffiBench-X or Mercury were found.
- Qwen3-Coder-30B-A3B's official SWE-bench Verified score could not be confirmed from a primary source this session.

## Q4. Benchmarks designed for recurring task families / experience reuse in coding

### Takeaway
Several 2025-2026 benchmarks target experience reuse directly. SWE-Bench-CL (chronological SWE-bench Verified sequences), SWE-ContextBench (related-task families with oracle vs retrieved experience), VibeMemBench (memory systems over repo history, September 2026), and the non-coding-specific SkillEvolBench and Evo-Memory. The headline result in VibeMemBench is negative: 11 of 12 memory/solver pairings fail to beat memory-off. That supports ELH's concern that a good static prompt may match memory.

### Cited Findings
- SWE-Bench-CL (July 2025): 273 SWE-bench Verified tasks from 8 Python repos, ordered chronologically into per-repo sequences. Ships a LangGraph evaluation framework with a FAISS semantic-memory module and CL metrics (average accuracy, forgetting, forward/backward transfer, tool-use efficiency, a composite CL score, CL-F-beta) — [arXiv 2507.00014](https://arxiv.org/html/2507.00014v1)
- SWE-ContextBench (arXiv 2602.08316, v3 May 2026): evaluates reuse of prior-task experience from a *static, pre-built* pool of trajectories. Oracle-provided summaries improve resolution by about 8 percentage points — [VibeMemBench's description, via search](https://arxiv.org/pdf/2609.23570); [arXiv 2602.08316](https://arxiv.org/pdf/2602.08316)
- VibeMemBench (September 2026): 111 coding targets from 90 SWE-rebench V2 repos, plus 3,634 history trajectories from those repos. Covers bug fixes, features, interface changes and config work. When memory systems must build and retrieve experience from the same history, 11 of 12 solver/system pairings fail to beat the matched memory-off baseline — [arXiv 2609.23570](https://arxiv.org/pdf/2609.23570)
- SkillEvolBench (May 2026): 180 tasks across 6 agent environments, 5 task families per environment, 6 tasks per family. Each family shares an underlying procedure but varies failure modes and surface forms; it tests turning episodic experience into procedural skills — [arXiv 2605.24117](https://arxiv.org/pdf/2605.24117)
- Other related work: Evo-Memory (test-time learning with self-evolving memory) — [alphaXiv 2511.20857](https://www.alphaxiv.org/abs/2511.20857); AgentCL (rigorous CL evaluation for language agents) — [arXiv 2606.02461](https://arxiv.org/html/2606.02461v2); Metis (text + code memory for self-evolving agents) — [arXiv 2606.24151](https://arxiv.org/pdf/2606.24151)

### Inferences
- SkillEvolBench's design (families share a procedure, varied failure modes and surface forms) is the right template for ELH's cause families. The ELH generator could follow it directly: per cause, vary repo, surface form and how the slowdown is exposed.
- SWE-Bench-CL's 273 tasks are a natural real-world repair complement: per-repo sequences let "repo-specific memory" beat a generic static prompt. Baseline pass rates for a 30B model on Verified are in a workable band (Q3).
- The VibeMemBench negative result is a useful prior and a citable motivation: ELH should include a strong static-prompt arm (distilled rules) and a "memory-off but same tokens" control.

### Gaps
- SWE-ContextBench's size, family structure and release link could not be extracted (the PDF fetch summary was vague).
- Whether VibeMemBench, SWE-Bench-CL and SkillEvolBench data and code are publicly released with Docker environments, and their licenses, were not verified.
- No benchmark was found that combines recurring families with *performance* grading.
