# Emergent individuality in identical LLM agents: prior evidence and experimental designs

Scope: whether agents with the same frozen weights diverge into distinct, stable behavioural styles because of different experience histories, memory or social interaction, plus analogues and measurement designs from other fields. Searched October 2026. Items marked **[UNVERIFIED]** were not confirmed against a primary source in this session.

## Q1. LLM-agent work on spontaneous emergence of individuality and personality

### Takeaway
There is direct but thin evidence that identical LLM agents become different through interaction. Takata et al. (2024) is the canonical citation, but it is small (10 agents, Llama-2-7B, 100 steps, one run) and its "stability" claim rests on two MBTI snapshots. A September 2026 preprint claims stable symmetry-breaking with matched controls. Neither study tests task performance or coding-agent working styles, and neither runs repeated replicates per history, which is what the path-dependence question needs.

### Cited Findings
- **Takata, Masumori & Ikegami. "Spontaneous Emergence of Agent Individuality through Social Interactions in LLM-Based Communities."** The citation checks out: arXiv:2411.03252 (submitted 5 Nov 2024), published in *Entropy* 26(12):1092 (Dec 2024), DOI 10.3390/e26121092. — [arXiv](https://arxiv.org/pdf/2411.03252); [MDPI](https://www.mdpi.com/1099-4300/26/12/1092); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11675631/)
- Takata design: 10 homogeneous agents running Llama 2 7b-chat on a 50×50 periodic grid for 100 steps, at temperature 0.7 and top-p 0.95. Agents differed only in starting position and name, with no predefined personality or memory. Each step an agent could move, send messages and update its memory. — [arXiv HTML](https://arxiv.org/html/2411.03252v1)
- Takata results:
  - Movement-command distributions became biased per agent (some agents mostly chose "stay" after clustering).
  - MBTI at step 0: nine agents were INFJ. At step 100 they had split into five types (ESFJ, ISTJ ×4, ENTJ ×3, ISFJ).
  - BERT emotion scores moved together within some clusters but not others.
  - Sentence-BERT/UMAP showed agent memories spread apart while their messages stayed similar.
  - The agents produced hallucinations and hashtags that kept the conversation going and increased word diversity.
  — [arXiv HTML](https://arxiv.org/html/2411.03252v1); [arXiv abstract](https://arxiv.org/abs/2411.03252v1)
- **Liu (2026). "Self-Emergence Agent Architecture: Behavior-Inertia HMM, Reflexive Metacognition, and Social-Contrastive Self-Modeling."** arXiv:2609.17331 (15 Sep 2026), single author.
  - Behavioural inertia is encoded as an editable HMM transition matrix, which a Reflexion-style loop updates.
  - Reports that "initially identical agents consolidate distinct, stable personalities whereas matched controls do not". This also holds in prototypes with no LLM.
  - In LLM runs, agents produced distinct first-person self-narratives, and five-agent deliberations formed a hierarchy with "a consensus hub and a unanimously rejected outlier".
  — [Emergent Mind summary](https://www.emergentmind.com/papers/2609.17331) (read via an aggregator, not the PDF; the details are summary-level)
- **Xiong et al. "How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior."** arXiv:2505.16067; ACL 2026 long paper.
  - Identifies an "experience-following property": when a task input closely matches a retrieved memory record's input, the agent's output closely matches that record's output.
  - Names two failure modes, error propagation and misaligned experience replay.
  - Suggests future task evaluations can serve as free quality labels for memory.
  - An aggregator summary gives correlations of 0.52–0.91 between input similarity and output similarity. I did not check this range against the paper.
  — [arXiv](https://arxiv.org/abs/2505.16067); [ACL Anthology](https://aclanthology.org/2026.acl-long.27.pdf)
- Generative Agents (Park et al. 2023, arXiv:2304.03442) and Voyager (Wang et al. 2023, arXiv:2305.16291; ever-growing skill library in Minecraft) are the standard precedents for memory/reflection and skill-library agents. **[UNVERIFIED in this session: IDs from prior knowledge, not re-fetched.]** In both, agents start with authored seeds (Generative Agents uses hand-written persona descriptions), so neither isolates divergence driven only by history.

### Inferences
- Experience-following (Xiong et al.) is a plausible mechanism for history-specific "styles" in a coding agent. If early tickets write memory records, later similar tickets get pulled toward those trajectories. Divergence would then be strongest on tasks similar to the early tickets and weak on novel ones, and that prediction can be tested.
- Takata's MBTI measurement is weak evidence of a stable trait: self-report from a 7B model, two time points, and no test-retest under resampling. For coding agents, measure behaviour (tool-call mix, test-first rate, edit size, exploration depth) rather than self-reported personality.
- The Liu 2026 "matched controls" design is the template to copy: identical agents with and without the history or self-model mechanism, so that symmetry-breaking can be attributed to it.

### Gaps
- I found no study that replicates the same experience history several times to separate history-driven divergence from sampling noise. No study in this set covers coding agents or working styles.
- I could not verify the Liu 2026 numbers or sample sizes from the PDF.
- Voyager-style work comparing divergent skill libraries across seeds was not located in this pass.

## Q2. Path dependence and history effects in memory-augmented agents

### Takeaway
2026 work confirms that, for memory and skill agents, the order of accumulated experience changes performance and agent rankings. Recent experience dominates retrieval, and early gains can be overwritten. This supports the hypothesis that the first tickets shape outcomes, but the same work finds recency effects that would undermine lock-in. Whether early experience locks in or later experience reshapes it is an empirical question the study can test.

### Cited Findings
- **PATH-Bench: Path-Dependent Evaluation of Lifelong Agents.** Yang, Zhang, Li et al. (East China Normal University), arXiv:2608.01149, 2 Aug 2026.
  - Task relations are estimated by multi-model voting (DeepSeek-V4-Flash, GPT-5.4-mini, GLM-5.0).
  - Each probe task is preceded by a controlled history of 100 intervening tasks: positive-dominant (70% helpful) or negative-dominant (70% interfering).
  - Uses 120 tasks each from BigCodeBench and WildToolBench.
  - Reports forgetting (FGT), forward transfer (FWT) and backward transfer (BWT).
  — [alphaXiv](https://www.alphaxiv.org/abs/2608.01149)
- PATH-Bench findings:
  - "Performance rankings of lifelong agents can fluctuate significantly depending on the task sequence."
  - Memory agents often underperformed memory-free baselines because of distraction from irrelevant or conflicting memories.
  - Concrete trajectories helped single-turn tasks, while abstract rules helped multi-turn tasks.
  - Recent experience disproportionately shaped retrieval, and agents recovered when the sequence turned positive.
  - "Strong transfer does not ensure retention": later experience reshaped earlier gains.
  - Their Selective Experience Use (preserve / abstract / ignore) cut forgetting by up to 16.84 pp on BigCodeBench.
  — [alphaXiv](https://www.alphaxiv.org/abs/2608.01149)
- A search snippet (source page not fetched) says default task orderings in prior memory-agent work impose an implicit curriculum, and that shuffling the order shrinks the reported memory gains. — [Memory for Autonomous LLM Agents survey, arXiv:2603.07670](https://www.alphaxiv.org/abs/2603.07670) **[attribution of this snippet to this exact source UNVERIFIED]**
- An order effect without memory, within a single context: models had lower success when ambiguous information came first than when clarified information came first, even with the same final information. The paper frames this as early "state commitment". — [Clarification Is Not Correction: LLMs Fail to Let Go, arXiv:2609.25337](https://arxiv.org/pdf/2609.25337) (snippet-level; not fetched)

### Inferences
- PATH-Bench's probe-centred design maps directly onto the study:
  1. Fix a set of probe tickets.
  2. Build histories of controlled composition (helpful-first versus interfering-first).
  3. Re-evaluate the probes repeatedly after each history.
  This makes "what the first tickets taught" an explicit manipulated factor rather than an accident of three runs.
- The recency finding predicts that history-specific styles fade as later experience accumulates unless memory curation preserves them. That fading is the retention test.
- Memory can hurt relative to no memory (PATH-Bench). The observed divergence among our three histories could therefore partly be one history polluting its memory, not a "style". Include a memory-free baseline.

### Gaps
- I found no work showing permanent lock-in (early experience fixing a strategy that later experience cannot overturn) in LLM memory agents. Current evidence leans toward recency and reshaping.
- I found no in-context-learning order-effect study that I could verify beyond snippets in this pass. The classic ICL example-order sensitivity literature (for example Lu et al. 2022, "Fantastically Ordered Prompts") was not re-verified here **[UNVERIFIED]**.

## Q3. Run-to-run behavioural variance: the noise baseline

### Takeaway
SWE-bench-style agents vary a lot from run to run even at temperature 0. Pass@1 moves 2.2–6.0 pp depending on the run, and trajectories diverge within the first few percent of tokens. Three single histories cannot separate history effects from this noise. Each history needs replicate runs, and differences need to be compared against within-history variance.

### Cited Findings
- **Bjarnason, Silva & Monperrus. "On Randomness in Agentic Evals."** arXiv:2602.07150, submitted 6 Feb 2026; listed at ICLR 2026.
  - 60,000 trajectories on SWE-Bench-Verified, covering three models and two scaffolds.
  - Single-run pass@1 varies by 2.2–6.0 pp depending on which run is chosen.
  - SD exceeds 1.5 pp even at temperature 0, because of infrastructure nondeterminism.
  - Trajectories diverge "often within the first few percent of tokens", and the divergence cascades into different solution strategies.
  - "Reported improvements of 2–3 percentage points may reflect evaluation noise."
  - Recommendations: multiple runs per task, power analysis, and pass@k / pass^k.
  — [arXiv](https://arxiv.org/abs/2602.07150v1); [ICLR 2026](https://iclr.cc/virtual/2026/10016323)
- RL analogue: **Henderson et al. "Deep Reinforcement Learning that Matters"** (AAAI 2018, arXiv:1709.06560). Results vary substantially across random seeds and environment stochasticity, and averaging over a few seeds can mislead. — [arXiv](https://arxiv.org/pdf/1709.06560); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/11694/11553)
- Clary et al., "Let's Play Again: Variability of Deep RL Agents in Atari Environments" (arXiv:1904.06312), studies variability of trained agents across evaluation runs. **[found in search, not fetched]** — [arXiv](https://arxiv.org/pdf/1904.06312)

### Inferences
- Early token divergence (Bjarnason et al.) means one replay of the same history with the same memory can drift into a different style by chance. "Individuality" must be shown as between-history variance above within-history variance. This is exactly the repeatability (ICC) framing in Q4.
- For a frozen local model (Qwen3-Coder 30B), fixing seeds and inference settings reduces the noise but does not remove it (batching and kernel nondeterminism). Measure the noise empirically by replaying identical histories.

### Gaps
- I found no variance study specific to memory-augmented agents, that is, how much two replays of the same experience history diverge in their memory contents and behaviour.

## Q4. Analogues from biology and RL, and behavioural-individuality measurement

### Takeaway
Behavioural ecology already has the designs this study needs: isogenic organisms in standardised environments, longitudinal tracking, and repeatability (ICC) estimated with mixed models. Two key findings: individuality grows with time in isogenic mice (Freund 2013), and it arises even without social experience in clonal fish (Bierbach 2017). The second warns that divergence can come from developmental noise, the analogue of sampling noise, and not only from distinct experiences.

### Cited Findings
- **Freund et al. (2013). "Emergence of Individuality in Genetically Identical Mice."** *Science* 340(6133):756–759.
  - 40 inbred mice lived in one large enriched environment and were tracked longitudinally.
  - Exploratory activity diverged over time, so individual differences increased with age.
  - Individual differences in cumulative "roaming entropy" (active territory coverage) correlated positively with adult hippocampal neurogenesis.
  — [PubMed 23661762](https://pubmed.ncbi.nlm.nih.gov/23661762/); [TU Dresden record](https://fis.tu-dresden.de/portal/en/publications/emergence-of-individuality-in-genetically-identical-mice(bb4df309-d366-46b3-9aed-18227624792a).html)
- **Bierbach, Laskowski et al. (2017). "Behavioural individuality in clonal fish arises despite near-identical rearing conditions."** *Nature Communications* 8:15361 (17 May 2017).
  - Clonal Amazon mollies were isolated at birth in standardised conditions, or kept in groups of four for 1 or 3 weeks before isolation.
  - Substantial individual behavioural variation emerged in every treatment.
  - More social experience did not change the level of individual variation.
  - The authors suggest individuality may be "an inevitable and potentially unpredictable outcome of development". Data are on Dryad.
  — [RePEc record](https://ideas.repec.org/a/nat/natcom/v8y2017i1d10.1038_ncomms15361.html); [IGB press release](https://www.igb-berlin.de/en/node/6666); [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.td3sj)
- **Bell, Hankison & Laskowski (2009). "The repeatability of behaviour: a meta-analysis."** *Animal Behaviour* 77(4):771–783.
  - Repeatability was higher in the field than in the lab.
  - It was higher when the interval between observations was short.
  - Mate preference was among the least repeatable behaviours.
  - The commonly cited figures, average R ≈ 0.37 across ~759 estimates, are **[UNVERIFIED]**: the PDF could not be text-extracted in this session.
  — [Bell lab PDF](https://alisonbelllab.web.illinois.edu/wp-content/uploads/2021/05/Bell-et-al-2009-repeatability.pdf)
- **Nakagawa & Schielzeth (2010). "Repeatability for Gaussian and non-Gaussian data: a practical guide for biologists."** *Biological Reviews* 85(4):935–956.
  - Repeatability is the ICC: the share of phenotypic variance due to between-individual variance. The non-repeatable share is measurement error plus within-individual flexibility.
  - Recommends LMM/GLMM-based estimation with SEs and CIs so confounders can be controlled. This applies to binary outcomes such as ticket pass/fail.
  - Implemented in the R package `rptR`.
  — [OpenAlex](https://explore.openalex.org/works/w2045443942); [rptR docs](https://rdrr.io/cran/rptR/man/rpt.html)

### Inferences
Design translation for the coding-agent study:
- **"Individual"** = one memory lineage (one experience history). **"Repeated measures"** = several runs of that lineage on the same fixed probe tickets, at several checkpoints.
- **Repeatability R** = between-lineage variance divided by (between-lineage + within-lineage variance) for each style metric, fitted with a GLMM through `rptR`. This includes binary metrics such as "ran tests before editing" and pass/fail. R significantly above 0 means stable history-specific styles. R near 0 means the divergence is noise.
- **Freund-style longitudinal design**: check whether between-lineage variance grows with the number of tickets experienced, the analogue of "differences increase with age".
- **Bierbach-style control**: run several lineages that receive the *same* ticket sequence. If they also diverge, part of the individuality is "developmental noise" (sampling) rather than experience content. The between-sequence versus within-sequence comparison separates the two sources.
- **Reaction norms**: plot each lineage's behaviour across a task-difficulty or task-type gradient. Lineages can differ in intercept (overall style) and slope (plasticity). This is the behavioural-ecology way to show "personality × environment".
- Bell 2009's finding that repeatability declines as the interval lengthens implies that retention (Q5) should be tested at several delays.

### Gaps
- Exact Bell 2009 summary statistics are unverified.
- I did not locate RL work that explicitly measures stable seed-specific *behavioural styles*, as opposed to performance variance. The "seed individuality" analogue is supported only for performance variance.

## Q5. Retention of learned behaviour or identity after interference, compaction or snapshot restore

### Takeaway
Evidence is limited and comes mostly from persona-drift work, not memory-agent work. Identity drifts within long sessions, in-context compaction does not reset it, and a single anchor prompt restores the baseline. I found no study testing whether a memory-derived style survives interference or a restore of a memory snapshot.

### Cited Findings
- **Ding et al. "ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions."** arXiv:2605.24279 (22 May 2026; revised 27 Aug 2026).
  - Uses a "snapshot-then-probe" protocol that forks the conversation state to probe without perturbing the main session.
  - Drift is widespread across 23 frontier models in three anonymised Claude Code sessions of 3,746–9,716 turns.
  - "In-session compaction does not reliably reset it", while "a single-shot anchor restores the trained register".
  - Drift may help tool-use continuation but breaks formatting contracts and inflates output length in tool-free chat.
  — [arXiv](https://arxiv.org/abs/2605.24279)
- **Li et al. "Measuring and Controlling Persona Drift in Language Model Dialogs"** (arXiv:2402.10962). Self-chat between two persona chatbots shows significant drift within eight rounds for LLaMA2-chat-70B, which the authors attribute partly to attention decay. — [arXiv](https://arxiv.org/pdf/2402.10962v1); [Harvard VCG](https://vcg.seas.harvard.edu/publications/measuring-and-controlling-persona-drift-in-language-model-dialogs)
- **"Examining Identity Drift in Conversations of LLM Agents"** (arXiv:2412.00804). Larger models showed more identity drift, and assigning a persona did not reliably maintain identity. — [arXiv](https://arxiv.org/html/2412.00804v2)
- PATH-Bench (Q2): backward transfer and forgetting show that later interfering experience reshapes earlier gains, and that selective memory curation reduces forgetting (up to 16.84 pp). — [alphaXiv](https://www.alphaxiv.org/abs/2608.01149)

### Inferences
- ContextEcho's snapshot-then-probe protocol is directly reusable:
  1. Snapshot each lineage's memory store at checkpoints.
  2. Probe forks with the fixed ticket battery.
  3. Restore the snapshot after an interference phase (for example, tickets that teach an opposite strategy).
  4. Test whether the style returns.
- With frozen weights, all "individuality" lives in memory and context. So restoring a snapshot should restore the style exactly, up to sampling noise. A failure to restore would point to uncontrolled state or nondeterminism, not to the model having learned anything. The more interesting test is interference *without* restore: does the style persist (lock-in) or fade (recency, as PATH-Bench predicts)?

### Gaps
- I found no work on identity or style retention specifically for external-memory agents after interference or snapshot restore. This appears to be an open gap the study could fill.
- Persona-drift studies measure in-context drift from an assigned persona, not retention of a style the agent acquired itself.
