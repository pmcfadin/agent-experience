# Literature

Related work for each research question, with what it means for this study. The full reviews are in
[`reports/`](../reports/) and their source notes in [`research_notes/`](../research_notes/). Many
2026 entries are preprints; those read only from the abstract or a summary are marked
*(abstract only)*.

## 1. Can memory make a frozen agent better? (the current study)

The test groups implement these papers; the method and deviations for each are in
[`methods.md`](methods.md#methods-and-deviations).

| Paper | Used for |
|---|---|
| Hindsight, Latimer et al. ([2512.12818](https://arxiv.org/abs/2512.12818)) | Memory bank (group B) |
| Learning on the Job, Tablan, Taylor & Bernhem ([2607.22157](https://arxiv.org/abs/2607.22157)) | Agent-written WHEN–THEN rules (B) |
| MemRL, Zhang et al. ([2601.03192](https://arxiv.org/abs/2601.03192)) | Utility reranking (C) |
| Agent Workflow Memory, Wang et al. ([2409.07429](https://arxiv.org/abs/2409.07429)) | Procedure library (D) |
| GEPA, Agrawal et al. ([2507.19457](https://arxiv.org/abs/2507.19457)) | Optional procedure optimiser (D) |
| When Continual Learning Moves to Memory, Hu, Long & Wang ([2604.27003](https://arxiv.org/abs/2604.27003)) | ΔRR / ΔNL analysis |

Two results bear on how far the pilot findings can be trusted:

- Coding-agent runs are noisy. Across 60,000 SWE-bench Verified runs, single-run pass rates varied
  by 2.2–6.0 points, even at temperature 0 (Bjarnason, Silva & Monperrus, ICLR 2026,
  [2602.07150](https://arxiv.org/abs/2602.07150)). Single-history comparisons need replication.
- Experience order matters, and recent experience dominates what is retrieved (PATH-Bench,
  [2608.01149](https://www.alphaxiv.org/abs/2608.01149), *abstract only*). This fits experiment 13,
  where what the first tickets taught decided the outcome.

## 2. Does memory beat a better prompt?

Full review: [`reports/Memory vs prompt datasets.md`](../reports/Memory%20vs%20prompt%20datasets.md).
Design that follows from it: [`design/house-lib-family.md`](design/house-lib-family.md).

**The comparison is mostly missing.** No paper found freezes an agent's final memory into a static
prompt and compares it with retrieving the same rules per task.

**What the nearest work shows:**

- Retrieving a few conditional guidelines per task beat putting all learned insights in the prompt,
  most clearly on diverse tasks, with a peak at about 3 retrieved guidelines (AutoGuide,
  [2403.08978](https://arxiv.org/abs/2403.08978); compared against ExpeL,
  [2308.10144](https://arxiv.org/abs/2308.10144)).
- Hand-written prompts are a strong control: ExpeL's hand-written insights scored 32 against 39 for
  learned ones; AWM's learned workflows scored 35.5 against 33.0 for expert-written ones.
- A growing in-context playbook beat GEPA-optimised and few-shot prompts on AppWorld with an open
  model, 59.5 against 46.4 and 46.0 (ACE, [2510.04618](https://arxiv.org/abs/2510.04618)). That is
  rich context against a short prompt, not retrieval against a prompt.
- Dynamic Cheatsheet ([2504.07952](https://arxiv.org/abs/2504.07952)) has the most useful controls
  (empty memory, full history, retrieval without summarising); smaller models gained little.
- Memory gains rise with how similar tasks are within a dataset (Evo-Memory,
  [2511.20857](https://arxiv.org/abs/2511.20857)).
- Recent benchmarks lean negative: 11 of 12 memory-system and model pairings failed to beat memory-off
  (VibeMemBench, [2609.23570](https://arxiv.org/abs/2609.23570)); plain in-context learning beat
  dedicated memory systems (Continual Learning Bench, [2606.05661](https://arxiv.org/abs/2606.05661)).

**Datasets.** None fits the question as is:

- Repository-level performance benchmarks are near zero for a 30B model: GSO
  ([2505.23671](https://arxiv.org/abs/2505.23671)), SWE-Perf
  ([2507.12415](https://arxiv.org/abs/2507.12415); an audit found 11 of 140 tasks stable across
  machines, [2607.01211](https://arxiv.org/abs/2607.01211)), SWE-fficiency
  ([2511.06090](https://arxiv.org/abs/2511.06090)).
- Generated bug corpora give unlimited fresh tasks but only correctness bugs: SWE-smith
  ([2504.21798](https://arxiv.org/abs/2504.21798), [code](https://github.com/SWE-bench/SWE-smith)).
- Continual-learning benchmarks usable as outside checks: AppWorld
  ([2407.18901](https://arxiv.org/abs/2407.18901)), LifelongAgentBench
  ([2505.11942](https://arxiv.org/abs/2505.11942)), StreamBench
  ([2406.08747](https://arxiv.org/abs/2406.08747)), SWE-Bench-CL
  ([2507.00014](https://arxiv.org/abs/2507.00014)).

**Methods the `house_lib` design borrows:**

| Source | Borrowed |
|---|---|
| Zan et al., When Language Model Meets Private Library ([2210.17236](https://arxiv.org/abs/2210.17236)) | Tasks against a library the model cannot know; no-docs / oracle-docs / retrieved-docs comparison |
| CodeUpdateArena ([2407.06249](https://arxiv.org/abs/2407.06249)) | Synthetic API changes graded by tests; docs in the prompt were not enough for open models |
| Wu et al., Reasoning or Reciting? ([2307.02477](https://arxiv.org/abs/2307.02477)) | Counterfactual variants, so lessons can't come from pretraining |
| SCAN ([1711.00350](https://arxiv.org/abs/1711.00350)), COGS ([EMNLP 2020](https://aclanthology.org/2020.emnlp-main.731)), gSCAN ([2003.05161](https://arxiv.org/abs/2003.05161)) | Held-out combination splits |
| Procgen ([1912.01588](https://arxiv.org/abs/1912.01588)) | Train on generated seeds, test on unseen ones |
| Lost in the Middle ([TACL 2024](https://aclanthology.org/2024.tacl-1.9/)), The Power of Noise ([2401.14887](https://arxiv.org/abs/2401.14887)), Yoran et al. ([2310.01558](https://arxiv.org/abs/2310.01558)) | Gold / retrieved / distractor / no-context controls |
| RULER ([2404.06654](https://arxiv.org/abs/2404.06654)) | Degradation as the prompt grows |
| GEM ([1706.08840](https://arxiv.org/abs/1706.08840)); van de Ven, Tuytelaars & Tolias ([Nat. Mach. Intell. 2022](https://doi.org/10.1038/s42256-022-00568-3)) | Transfer and forgetting metrics; incremental-learning scenarios |

## 3. Can memory give an agent a personality it keeps?

Full review: [`reports/Agent personality through memory.md`](../reports/Agent%20personality%20through%20memory.md).

**Questionnaires don't measure it.** LLM personality-test scores can be internally reliable
(Serapio-García et al., [Nat. Mach. Intell. 2025](https://arxiv.org/abs/2307.00184)) but shift with
wording and option order (Gupta et al., [2309.08163](https://arxiv.org/abs/2309.08163)), agree with
opposite statements (Dorner et al., [2311.05297](https://arxiv.org/abs/2311.05297)) and drift toward
desirable answers when tested (Salecha et al., [2405.06058](https://arxiv.org/abs/2405.06058)). Most
important here: persona prompts changed what models said about themselves but not their behaviour
(Han et al., The Personality Illusion, NeurIPS 2025,
[2509.03730](https://arxiv.org/abs/2509.03730)).

**Behaviour traces do show stable traits, between models.** 79 of 318 trace features were stable,
consistent across tasks and model-distinguishing, reducing to 6 factors that barely matched
self-reported Big Five (Zhao et al., [2609.32776](https://arxiv.org/abs/2609.32776), *abstract only*).
Unseen coding-agent trajectories can be traced to their agent 85.7% of the time (Oderinwale,
[2606.16988](https://arxiv.org/abs/2606.16988), *abstract only*), and gathering context before the first
edit predicts SWE-bench success (Mehtiyev & Assunção,
[2604.02547](https://www.alphaxiv.org/abs/2604.02547), *abstract only*).

**Identical agents diverge, but the evidence is thin.** Ten identical 7B agents split into several
MBTI types through social interaction (Takata et al., [Entropy 2024](https://www.mdpi.com/1099-4300/26/12/1092)).
Animal analogues: genetically identical mice became more individual with experience (Freund et al.,
[Science 2013](https://pubmed.ncbi.nlm.nih.gov/23661762/)), and clonal fish developed individuality even
when raised alone (Bierbach et al., Nat. Commun. 2017), a warning that some divergence is noise. The
standard measure is repeatability, the share of variance between individuals (Nakagawa & Schielzeth
2010; R package [rptR](https://rdrr.io/cran/rptR/man/rpt.html)).

**Persona stability depends on where the persona lives:**

- Prompt: drifts within about 8 rounds (Li et al., [2402.10962](https://arxiv.org/abs/2402.10962)) and
  fades over 100+ rounds (Luz de Araujo et al., EACL 2026,
  [2512.12775](https://arxiv.org/abs/2512.12775)).
- Activations: trait directions predict and steer persona (Persona Vectors,
  [2507.21509](https://arxiv.org/abs/2507.21509)); coding conversations keep models near the default
  assistant persona (Assistant Axis, [2601.10387](https://arxiv.org/abs/2601.10387)).
- Weights: narrow fine-tuning can switch on a broad persona (Wang et al., ICLR 2026,
  [2506.19823](https://arxiv.org/abs/2506.19823)).
- Memory: personality built into memory operations kept 89% of its effect under a conflicting
  instruction against 58% for a prompt persona (PersMem,
  [2609.34372](https://arxiv.org/abs/2609.34372), preprint, single instruction, not a task stream).
  Recall is not enactment (PAI-Bench, [2609.13637](https://arxiv.org/abs/2609.13637)), and
  self-written identity drifts in a consistent direction (MicroVerse,
  [2608.15844](https://arxiv.org/abs/2608.15844)).

**What no study has done**, and this one could: replicate the same experience history to separate
history from noise; show a memory-caused trait shift inside one frozen model with psychometric
criteria; prove cause by moving or replacing a memory bank; test whether a self-acquired style
survives later interference; compare one persona as a static prompt, a re-inserted prompt and
self-written memory over a long task stream.
