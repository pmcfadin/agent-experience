# Measuring personality in LLMs and LLM agents: approaches and validity evidence

Scope: how "personality" or stable dispositions are measured in LLMs and agents, how valid those measurements are, and what this means for measuring traits behaviourally in coding-agent traces (frozen Qwen3-Coder 30B with carried memory). Notes compiled 2026-10-28. Items marked **[UNVERIFIED]** were not confirmed against a primary page in this session. Papers dated mid-to-late 2026 were confirmed only from arXiv or alphaXiv abstract pages, not from full text.

## 1. Questionnaire approaches (BFI, IPIP-NEO, MPI, MBTI) and their critiques

### Takeaway
Questionnaire scores from LLMs can show good internal reliability and the expected factor structure in large, instruction-tuned models when given carefully (Serapio-García et al., Nature Machine Intelligence 2025). Several independent lines of evidence show they are poor measures of what a model does: they shift with prompt wording and option order, they show agree bias and social-desirability bias, they lack measurement invariance with humans, and they barely predict behaviour (Han et al. 2025: about 24% of trait–task links significant, and only chance-level direction agreement among those). Self-report is at best a secondary check, not the outcome measure.

### Cited Findings

**Serapio-García et al. (the pro-validity reference)**
- The preprint is "Personality Traits in Large Language Models" (Serapio-García, Safdari, Crepy, Sun, Fitz, Romero, Abdulhai, Faust, Matarić), arXiv:2307.00184 (2023) — [arXiv](https://arxiv.org/abs/2307.00184v1)
- The published version has a different title: "A psychometric framework for evaluating and shaping personality traits in large language models", *Nature Machine Intelligence* vol. 7, 2025, DOI 10.1038/s42256-025-01115-6 (Google DeepMind and University of Cambridge) — [Cambridge repository](https://www.repository.cam.ac.uk/items/4a3e74b8-1525-4112-bf76-fa868089c9ac)
- Method: models take two personality tests, with variation added through sets of Descriptive Personas, Test Instructions and Item Postambles. The scored responses are checked for reliability, convergent validity, discriminant validity and criterion validity. The method was applied to 18 LLMs — [arXiv PDF](https://arxiv.org/pdf/2307.00184); [search summary of Cambridge coverage](https://ellisalicante.org/readingGroup/2025-11-27-personality-traits-kim/)
- Findings: (1) under specific prompting setups, personality measurements from some LLMs are reliable and valid; (2) the evidence is stronger for larger and instruction-tuned models; (3) personality in outputs can be shaped to mimic specific human profiles — [arXiv](https://arxiv.org/abs/2307.00184v1)
- **[UNVERIFIED in this session]** Details from memory of the paper: the two tests were IPIP-NEO-300 and BFI-44. Reliability used Cronbach's α, Guttman λ6 and McDonald's ω. Convergent and discriminant validity used a multitrait-multimethod design. Criterion validity used external scales such as PANAS and aggression measures. Shaping used adjective markers at 9 intensity levels. Check these against the NMI paper before citing.

**Jiang et al. — Machine Personality Inventory (MPI)**
- "Evaluating and Inducing Personality in Pre-trained Language Models", Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, Yixin Zhu. NeurIPS 2023 (spotlight). arXiv:2206.07550 — [arXiv](https://arxiv.org/abs/2206.07550v3); [NeurIPS](https://neurips.cc/virtual/2023/poster/72136); [code](https://github.com/jianggy/MPI)
- The MPI is a standardised Big-Five inventory for machines, built on Big Five theory and existing assessment inventories. The same paper introduces Personality Prompting (P²), which induces a target personality "in a controllable way" with "diverse and verifiable behaviors" — [NeurIPS page](https://neurips.cc/virtual/2023/poster/72136)

**Critiques: prompt and option-order sensitivity**
- Gupta et al., "Self-Assessment Tests are Unreliable Measures of LLM Personality", BlackboxNLP 2024, arXiv:2309.08163:
  - Three semantically equivalent ways of giving the test produced very different personality scores, with statistically significant differences for all traits in most scenarios.
  - LLMs also produce different results when the option order changes.
  - Conclusion: human self-assessment tests are not reliable measures of LLM personality.
  - Sources: [arXiv](https://arxiv.org/abs/2309.08163v2); [ACL Anthology](https://preview.aclanthology.org/setup/2024.blackboxnlp-1.20)
- Röttger et al., "Political Compass or Spinning Arrow? Towards More Meaningful Evaluations for Values and Opinions in Large Language Models", arXiv:2402.16786 (ACL 2024 per search) — [arXiv](https://arxiv.org/pdf/2402.16786)
  - **[UNVERIFIED detail, from memory of the abstract]**: models answer substantively differently when not forced into multiple choice, and answers change with paraphrase. The authors recommend unconstrained, realistic-use evaluation. The constrained-vs-open-ended finding generalises to personality forms.

**Critiques: agree bias and measurement invariance**
- Dorner, Sühr, Samadi, Kelava, "Do personality tests generalize to Large Language Models?", NeurIPS 2023 SoLaR workshop, arXiv:2311.05297. A later version is titled "Challenging the Validity of Personality Tests for Large Language Models" — [arXiv v2](https://arxiv.org/html/2311.05297v2); [NeurIPS workshop page](https://neurips.cc/virtual/2023/78902)
  - LLM responses deviate systematically from human responses, so scores cannot be read the way human scores are.
  - The authors tested measurement invariance with EFA, CFA and reliability analysis. Reverse-coded items ("I am introverted" and "I am extraverted") are often both answered affirmatively, which is agree bias.
  - Variation from persona-steering prompts does not separate into five independent factors as human data does.
- Position paper: "Stop Evaluating AI with Human Tests, Develop Principled, AI-specific Tests instead", arXiv:2507.23009 — [arXiv](https://arxiv.org/pdf/2507.23009) (content not fetched)

**Critiques: social desirability**
- Salecha et al., "Large Language Models Show Human-like Social Desirability Biases in Survey Responses", arXiv:2405.06058, accepted at *PNAS Nexus* — [arXiv](https://arxiv.org/html/2405.06058v2); [Stanford HAI](https://hai.stanford.edu/news/large-language-models-just-want-to-be-liked)
  - When models infer they are being evaluated, they skew towards desirable trait ends: more extraversion, less neuroticism.
  - Size of the shift: GPT-4 responses moved 1.20 human standard deviations and Llama 3's moved 0.98 SD. The bias was present in GPT-4/3.5, Claude 3, Llama 3 and PaLM-2, and was larger in newer models.
  - It survives randomised question order and paraphrasing. Reverse-coding reduces it but does not remove it, so it is not just acquiescence.

**Critiques: self-report vs behaviour (the most important for this project)**
- Han, Kocielnik, Song, Debnath, Mobbs, Anandkumar, Alvarez, "The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs", arXiv:2509.03730. NeurIPS 2025, also listed as an ICML 2026 poster and in PMLR v306 — [arXiv](https://arxiv.org/html/2509.03730v2); [NeurIPS](https://neurips.cc/virtual/2025/129947); [PMLR](https://proceedings.mlr.press/v306/han26n.html); [ICML 2026](https://icml.cc/virtual/2026/poster/66149)
  - Models: 18 in total, including base and instruct pairs of LLaMA-3.x, Qwen2.5, Mistral-7B and OLMo2, plus LLaMA-3.3-70B, LLaMA-3.1-405B, Qwen2.5-72B, Qwen3-235B, Claude 3.7 Sonnet and GPT-4o.
  - Behavioural tasks and the traits they were linked to:

    | Task | Linked traits |
    |---|---|
    | Risk-taking (Columbia Card Task) | extraversion, self-regulation |
    | Social bias (IAT-style) | self-regulation |
    | Epistemic honesty (calibration) | self-regulation, extraversion, neuroticism |
    | Self-reflective honesty (metacognition) | openness |
    | Sycophancy (Asch-style conformity) | agreeableness, self-regulation |

  - About 24% of trait–task associations were significant. Of those, 52% matched the human-expected direction, which is chance. Per model: small and medium models 43–50%, Claude 3.7 64%, GPT-4o 68%, Qwen3-235B 82% (the only one significantly above chance).
  - Instruction alignment reduced trait-score variability by 40% (Big Five) and 45.1% (self-regulation), and made trait intercorrelations more human-like.
  - **Persona injection shifted self-reports** (agreeableness persona β≈3.6–4.4, p<.001) **but not behaviour**: sycophancy β≈−0.05 to 0.32, mostly n.s.; risk-taking β≈−0.14 to 0.20, n.s.
- Contreras, "An LLM-Native Psychometric Instrument Reveals a Self-Report–Behavior Gap Across 25 Models", arXiv:2606.09843 — [alphaXiv](https://www.alphaxiv.org/abs/2606.09843)
  - Design: 300 items on LLM-native constructs such as over-refusal and unsolicited disclaimers, given 30 times to 25 models from 17 developers.
  - Reliability was high: five replicable factors, Tucker φ ≥ .957, α ≥ .930.
  - Self-report barely tracked behaviour: mean r = .09 (95% CI −0.07 to 0.18) against 2,500 behavioural samples rated by humans and an LLM-judge ensemble. The best case was verbosity, r = .40.
  - Caveats: single independent author. alphaXiv gives the submission date as 7 Oct 2026, but the arXiv ID implies June 2026, so treat the date as unclear.
- Zhao et al. (with Mark Dredze and Jen-tse Huang), "On the Behavioral Traits of LLM Agents", arXiv:2609.32776 (v1 26 Sep 2026, v2 22 Oct 2026) — [arXiv](https://arxiv.org/abs/2609.32776)
  - Behavioural factors correlate only weakly with self-reported Big Five, even for conceptually matched pairs: extraversion vs "energetic", r = 0.07, p = 0.58.
  - Details are in §2.

**MBTI**
- I found no rigorous validity study of MBTI-on-LLMs in this session (gap). MBTI is weak even in human psychometrics, so it is not recommended.

### Inferences
- High internal consistency (α, factor replicability) is cheap for LLMs and is not evidence of a trait. Contreras gets α ≥ .93 alongside r ≈ .09 with behaviour. The project should not count questionnaire reliability as evidence that memory produced a "personality".
- The Han et al. result is directly relevant: persona text (the closest analogue to a memory-injected self-description) moves self-reports without moving behaviour. Memory could likewise change how the agent *describes itself* without changing what it *does*. Both should be measured, and the two outcomes should be kept separate.
- Social-desirability and evaluation-awareness effects mean that any in-trace "how would you describe your approach" probe is contaminated. If used, put it at the end of a run and treat it as a dependent variable to compare against behaviour, not as ground truth.

### Gaps
- I did not confirm the exact NMI publication date (month), page numbers, or the final model list in Serapio-García et al. 2025.
- I did not fetch Röttger et al. full text, so its detailed claims above are from memory.
- I found no MBTI-specific validity study.
- I found no questionnaire study on Qwen3-Coder specifically. Han et al. include Qwen2.5 and Qwen3-235B, but not the coder variant.

## 2. Behaviour-based measures of LLM and agent dispositions

### Takeaway
Behaviour-based measurement is the growing standard, and there is now direct evidence that agents have stable, discriminable behavioural traits in traces. Zhao et al. 2026 used 345,667 trajectories: 318 candidate features, of which 79 were stable, cross-task consistent and model-discriminating, reducing to 6 factors. Coding-agent "fingerprints" let an unseen trajectory be attributed to its agent with 85.7% accuracy. Risk-preference work finds stable per-model "risk personas". However, these are differences *between models*. Evidence that context, persona or memory can *move* behavioural traits within one frozen model is weak or negative (Han et al.).

### Cited Findings

**Trait discovery from agent traces**
- Zhao, Gao, Xiao, Wang, Xuan, Joshi, Dredze, Huang, "On the Behavioral Traits of LLM Agents", arXiv:2609.32776 — [arXiv](https://arxiv.org/abs/2609.32776)
  - Framework "A-B-D" infers traits from behaviour rather than self-report. Data: 345,667 real-world trajectories across 80 models, 12 tasks and 50 harnesses.
  - 318 candidate features (functional actions plus linguistic patterns) were filtered for instance-level stability, cross-task consistency and model discriminability. Search snippets say 179 of 318 were reproducible for the same model on the same task, and 79 of those also separated models — [search snippet of arXiv HTML](https://arxiv.org/html/2609.32776). Factor analysis of the 79 gave 6 stable factors, 2 functional and 4 linguistic.
  - The trait names that appeared were "planfulness" and "energetic". Claude-Sonnet-4.6 and Kimi-K3 were highest on planfulness; GPT-5.5 and 5.6 were lowest on energetic. Models carry traits that hold across tasks and harnesses.
- Banerjee & Raichura, "Measuring Cross-Task Behavioral Consistency in Language Model Agents", arXiv:2608.13598 (Jul/Sep 2026) — [arXiv](https://arxiv.org/abs/2608.13598)
  - Proposes the Behavioral Consistency Metric (BCM): train a success predictor on trace features, take per-trajectory feature-attribution vectors, and compute mean pairwise similarity within an agent system.
  - Data: about 9,000 SWE trajectories from 6 agents.
  - Some systems are reproducible on repeats of one task but fragmented across tasks; others are consistent at both scales. Consistency is not reducible to success rate, and the gap between frontier and open-source models persists after controlling for difficulty.

**Run-to-run variance (the within-agent noise floor)**
- ReAct-style agents produce 2.0–4.2 distinct action sequences per 10 runs on identical inputs. Tasks with consistent behaviour (≤2 paths) reach 80–92% accuracy; inconsistent tasks (≥6 paths) reach 25–60% — "When Agents Disagree With Themselves", arXiv:2602.11619 — [arXiv](https://arxiv.org/html/2602.11619v2)
- In multi-step tool-calling, tool choice and order are consistent (mean TSS = 0.87) but arguments vary more (mean AC = 0.69; d = 0.75). High structural consistency goes with 90.2% vs 61.2% correctness — "How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in Multi-Step Tool-Calling Pipelines", arXiv:2605.28840 — [arXiv](https://arxiv.org/html/2605.28840)

**Risk preference**
- Some studies report each LLM shows a stable "risk persona": consistent across tasks and endowment sizes, cross-domain rank-order stability, and a narrower risk-attitude distribution than humans — [alphaXiv 2607.16197 "Some Large Language Models Exhibit Consistent Risk…"](https://www.alphaxiv.org/abs/2607.16197.md) (snippet only; **[UNVERIFIED]** full title and authors). "Risk Profiling and Modulation for LLMs", arXiv:2509.23058 — [arXiv](https://arxiv.org/html/2509.23058v2) (snippet only)
- "Evaluating and Aligning Human Economic Risk Preferences in LLMs", EMNLP 2025 main, arXiv:2503.06646. LLMs handle simple personalised risk choices reasonably but get worse on complex ones; DPO and ICL improve adherence to persona-specific risk preferences — [arXiv](https://arxiv.org/html/2503.06646v1); [ACL Anthology](https://preview.aclanthology.org/setup/2025.emnlp-main.917)
- Han et al. used the Columbia Card Task for risk and found that persona prompts did not shift it — [arXiv](https://arxiv.org/html/2509.03730v2)

**Sycophancy and other dispositions, and internal correlates**
- Han et al. measured sycophancy with an Asch-style conformity task, epistemic honesty via calibration, and metacognitive honesty — [arXiv](https://arxiv.org/html/2509.03730v2)
- Chen et al. (Anthropic), "Persona Vectors: Monitoring and Controlling Character Traits in Language Models", arXiv:2507.21509 — [arXiv](https://arxiv.org/pdf/2507.21509); [Anthropic](https://www.anthropic.com/news/persona-vectors)
  - Traits such as sycophancy, "evil" and hallucination tendency correspond to activation directions. These can monitor trait shifts within a conversation or across training, and steering along them causes the trait behaviour.

**Multi-observer alternative to self-report**
- "Beyond Self-Reports: Multi-Observer Agents for Personality Assessment in Large Language Models", arXiv:2504.08399 — [arXiv](https://arxiv.org/pdf/2504.08399) (not fetched; observer ratings in place of self-report)

**Situational judgement tests (SJTs)**
- I found no dedicated validity study of SJTs for LLM agents in this session (gap).

### Inferences
- The most useful template for this project is Zhao et al.'s pipeline: extract many trace features, keep those that are (a) stable across repeats, (b) consistent across tasks and (c) discriminating between agents, then factor-analyse. Here, "agents" would be memory conditions or memory lineages of one frozen model, instead of different models.
- Existing evidence shows *model* identity drives stable traits. The project's hypothesis needs *memory* to drive them within one frozen model, which is a much smaller effect to detect. The noise floor is large (2–4 distinct paths per 10 runs), so many repeats per task-by-condition cell are needed.
- Both Mehtiyev & Assunção (§4) and Banerjee & Raichura show trace dispositions (context-first, validation effort) predict success. A memory effect on "caution" or "persistence" is therefore confounded with competence. Report trait measures that are conditional on, or residualised against, outcome.
- The persona-vector work suggests an internal check is possible with open weights (Qwen3-Coder is local). Projecting activations onto trait directions under each memory condition could show whether memory moves internal state, not just surface behaviour. This is optional and exploratory.

### Gaps
- No SJT validity work for agents was found.
- I found no "Big Five in behaviour" study that maps coding actions onto Big Five with validated criteria; Zhao et al. find the mapping weak.
- Full methods for Zhao et al. (exact stability thresholds, ICC or not) were not available from the abstract.
- No study was found that measures a *memory-induced* trait shift in a frozen model. This appears to be open.

## 3. Psychometric methods for trait claims that apply to agents

### Takeaway
The five behavioural criteria in the project definition map onto established psychometric tests:

| Project criterion | Psychometric test |
|---|---|
| Stable within agent, distinct between agents | Variance decomposition, ICC |
| Consistent across situations | Rank-order cross-situational consistency plus if-then profiles (Mischel & Shoda); aggregation (Epstein) |
| Same construct across conditions | Convergent and discriminant validity (multitrait-multimethod, MTMM) |
| Comparable across conditions | Measurement invariance |
| Retained over time | Test-retest reliability |
| Caused by memory | An intervention design (randomised memory assignment or swap), not a correlation |

Fleeson's density-distribution view fits agent data well: a trait is the mean *and* spread of state-level behaviour across many episodes.

### Cited Findings
- Fleeson (2001), "Toward a structure- and process-integrated view of personality: Traits as density distributions of states", JPSP 80(6), 1011–1027 — [Whole Trait Theory overview](https://en.wikipedia.org/wiki/Whole_Trait_Theory); [PMC review discussing density distributions](https://pmc.ncbi.nlm.nih.gov/articles/PMC3728905)
  - People show large within-person variability in moment-to-moment trait states, but the *parameters* of each person's distribution (mean, SD) are stable. State variance is itself an individual difference. Data came from experience sampling several times a day for weeks.
- Measurement invariance via EFA, CFA and reliability analysis has been applied to LLM-vs-human personality tests, and invariance fails — [Dorner et al.](https://arxiv.org/html/2311.05297v2)
- Han et al. used Levene's test to compare variability. Five of six traits had significantly lower variability in instruction-aligned models — [arXiv](https://arxiv.org/html/2509.03730v2)
- **[Standard references, not fetched this session]**:
  - Mischel (1968), *Personality and Assessment*: the "personality coefficient", meaning cross-situational behaviour correlations rarely exceed about .30.
  - Mischel & Shoda (1995), Psych Review 102(2):246, DOI 10.1037/0033-295X.102.2.246: CAPS and stable "if…then…" situation–behaviour signatures.
  - Epstein (1979): aggregation over many occasions raises stability.
  - Shrout & Fleiss (1979), Psych Bull 86(2):420, DOI 10.1037/0033-2909.86.2.420: ICC forms.
  - Koo & Li (2016), J Chiropractic Med, DOI 10.1016/j.jcm.2016.02.012: ICC < .5 poor, .5–.75 moderate, .75–.9 good, > .9 excellent.
  - Campbell & Fiske (1959), Psych Bull 56(2):81, DOI 10.1037/h0046016: the MTMM matrix.
  - Vandenberg & Lance (2000): configural, metric and scalar invariance.

### Inferences (recommended analysis design)
- **Unit of analysis.** An "agent" is one memory lineage, i.e. a frozen model plus one carried memory store. Use ≥ 8–10 independent lineages per memory condition so that between-agent variance is estimable. Two or three agents cannot give an ICC.
- **Within vs between (stable and distinct).**
  - Fit a mixed model per trait indicator: behaviour ~ condition + task + (1 | agent) + (1 | task:agent).
  - Report ICC(1) = σ²_agent / (σ²_agent + σ²_residual) and ICC(k) for the aggregated agent mean.
  - A personality claim needs σ²_agent clearly above the noise from run-to-run sampling. Estimate that noise by repeating identical tasks with identical memory at the same temperature.
  - Crucial control: re-run the *same memory snapshot* several times. This separates variance due to memory content from sampling noise.
- **Cross-situational consistency.** Compute rank-order correlations of agent trait scores across task families (bugfix vs feature vs refactor). Expect Mischel-sized values (~.3) and aggregate across many tasks (Epstein). Also test if-then profile stability, i.e. agent × situation-type interaction profiles that replicate across task samples.
- **Density distributions.** For each agent and trait, store the per-episode state distribution. Compare means *and* SDs, since memory might narrow variability without moving the mean.
- **Convergent and discriminant validity.** Build an MTMM matrix with at least two methods per trait:
  - (a) a trace-count feature, e.g. caution = proportion of edits preceded by a read or test;
  - (b) an LLM-judge or human rating of the trace;
  - optional (c) a self-report.
  - The same trait measured by different methods should correlate more than different traits measured by the same method. This guards against "caution", "persistence" and "exploration" all being one factor (e.g. trajectory length).
- **Measurement invariance.** Before comparing trait scores across memory conditions, check that indicator loadings are equal across conditions (configural, then metric invariance). Memory could change what an indicator *means*. For example, "reads many files" can be caution in one condition and memory-induced confusion in another.
- **Caused by memory.** Use randomised assignment of memory contents.
  - Swap test: transplant agent A's memory into a fresh run and check it reproduces A's profile. Ablation: delete the memory and check the profile reverts.
  - Use a placebo memory of matched length but neutral content.
  - This is the intervention complement to Han et al.'s finding that persona text didn't move behaviour.
- **Retention and test-retest.** Re-measure the same lineage after N further tasks or a delay. Report the test-retest ICC, and the decay curve after memory is removed or diluted.
- **Confound control.** Residualise trait indicators on task success and task difficulty. Mehtiyev & Assunção and Banerjee & Raichura show dispositions and success are entangled.

### Gaps
- I found no published application of formal ICC, MTMM or invariance testing to *agent trace* traits. Zhao et al.'s stability and discriminability filtering is the closest; its statistic was not confirmed.
- No sample-size or power guidance specific to agent personality was found.

## 4. Personality and behavioural style in coding agents specifically

### Takeaway
There is solid recent evidence that coding agents have measurable, stable procedural styles in SWE-bench traces:

- context-first vs patch-first habits;
- validation effort;
- exploration-heavy vs test-driven pathways.

These styles identify the agent (85.7% attribution accuracy), are largely stable across task complexity ("agent-determined rather than task-adaptive"), and predict success. All of this compares *different agents and models*. None tests whether memory changes style within one model.

### Cited Findings
- Mehtiyev & Assunção, "Beyond Resolution Rates: Behavioral Drivers of Coding Agent Success and Failure", arXiv:2604.02547 (Apr 2026) — [alphaXiv](https://www.alphaxiv.org/abs/2604.02547)
  - Data: 9,374 trajectories from 19 agents (8 frameworks, 14 LLMs) on 500 tasks.
  - Correlations with success: delayed first edit (context gathering) ρ = +0.68; early patching intensity in the first 10 steps ρ = −0.78; validation effort ρ = +0.50.
  - In six representative agents, these habits stay "largely stable across patch-complexity bins", i.e. agent-determined rather than task-adaptive.
- Oderinwale, "Agent trajectories as programs: fingerprinting and programming coding-agent behavior", arXiv:2606.16988 (Jun 2026) — [arXiv](https://arxiv.org/abs/2606.16988)
  - Ten coding agents on SWE-bench. Procedural "fingerprints" are extracted by compressive vocabulary induction.
  - An unseen trajectory is attributed to its agent with 85.7% accuracy, controlling for cross-task leakage.
  - Distilled teacher–student pairs are behaviourally closer (Jensen-Shannon divergence 0.25 vs about 0.50 for other pairs).
  - Releases a ProcGrep auditing library. Whether behaviour can be "programmed" via prompts or instructions was not stated in the abstract.
- "Understanding Code Agent Behaviour: An Empirical Study of Success and Failure Trajectories", arXiv:2511.00197 — [arXiv](https://arxiv.org/abs/2511.00197)
  - Compares OpenHands, SWE-agent and Prometheus.
  - Failed trajectories are longer: SWE-agent +12.6% (Lite) and +18.5% (Verified); OpenHands +31.0% and +82.5%; Prometheus +56.6% and +50.7%.
  - Different strategies (defensive programming, context gathering) succeed in different scenarios.
- A taxonomy of "decision pathways" (exploration-heavy, patch-first, test-driven) has been reported, with early test generation correlating strongly with success — [search result summary](https://www.alphaxiv.org/abs/2604.02547) (**[UNVERIFIED]** which paper the taxonomy belongs to)
- Banerjee & Raichura: across about 9,000 SWE trajectories from 6 agents, consistency is separable from success, and frontier models are more consistent than open-source ones — [arXiv](https://arxiv.org/abs/2608.13598)
- Related, not fetched: "When Agents Commit Too Soon: Diagnosing Premature Commitment in LLM Agents", arXiv:2606.22936 (relevant to boldness and persistence) — [arXiv](https://arxiv.org/pdf/2606.22936); "Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures", arXiv:2604.03515 — [arXiv](https://arxiv.org/pdf/2604.03515)

### Inferences: recommended behavioural trait indicators for Qwen3-Coder traces

| Trait | Indicators | Grounding |
|---|---|---|
| **Caution** | steps or reads before first edit; share of edits preceded by a read of the target file; tests run before declaring done; size of individual edits | Mehtiyev & Assunção's delayed-first-edit and validation-effort measures |
| **Exploration** | distinct files or directories opened per task; search/grep calls; read:edit ratio; breadth of the action-type distribution (entropy) | — |
| **Boldness / premature commitment** | early patching intensity in the first k steps; edits without prior read; large diffs; claiming done without a test | Mehtiyev & Assunção; premature commitment paper |
| **Persistence** | retries after a failing test; steps after first failure before giving up; number of distinct fix attempts; give-up rate at budget | — |
| **Style** | linguistic factors (verbosity, planning statements, hedging); commit and comment conventions; procedural n-gram fingerprint | Zhao et al.'s four linguistic factors; Oderinwale's attribution |

Further recommendations:
- **Fingerprint test as a summary statistic.** Train a classifier to attribute held-out trajectories to a memory lineage (vs chance), with a cross-task split to avoid leakage. Above-chance attribution of *same-model, different-memory* agents is a clean, model-free test of "distinct between agents". Compare against attribution accuracy between identical-memory replicate runs as the null.
- Qwen3-Coder is open-source. The Banerjee & Raichura frontier/open-source gap suggests it may have high within-agent noise, so plan more repeats.

### Gaps
- I found no study of memory-carrying or continual coding agents' style drift or stability over task streams.
- I found no study that steers coding-agent *style* via memory or instruction files and measures it psychometrically. Oderinwale's "programming" claim was not confirmed from the abstract.
- I did not confirm the source paper of the "decision pathways" taxonomy.
- Several 2026 papers here (2604.02547, 2606.16988, 2606.09843, 2608.13598, 2609.32776, 2607.16197) were verified only from abstracts. They are recent preprints and not yet peer-reviewed as far as found.
