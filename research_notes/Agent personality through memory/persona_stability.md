# Persona induction, stability and drift in LLMs and agents: prompt vs memory vs weights

Scope note: these notes are for a frozen local coding agent (Qwen3-Coder 30B) with carried memory (episodic + self-written rules, Hindsight server). The hypothesis being tested is that a persona carried in memory, in the agent's own words, stays more stable over a long task stream than the same persona in a system prompt, and holds up better under interference. Search date: 2026-10-28. "Verified" means the arXiv abstract or primary page was fetched this session. Items marked **[prior knowledge, not re-fetched]** come from training knowledge and still need a citation check.

## Q1. Persona drift and instability in prompt-induced personas (dialogue and role-play)

### Takeaway
A persona set only in a system prompt drifts measurably within a few turns (Li et al.: within 8 rounds) and keeps fading over long dialogues (100+ rounds). Over time the model's behaviour moves back toward the model's default with no persona. Self-reports of the persona stay stable while the persona actually expressed in behaviour fades, so evaluation needs behavioural or observer measures.

### Cited Findings
- **Li, Liu, Bashkansky, Bau, Viégas, Pfister, Wattenberg, "Measuring and Controlling Instruction (In)Stability in Language Model Dialogs", arXiv:2402.10962 (v1 Feb 2024; v4 Jul 2024).**
  - Method: a benchmark of self-chats between two system-prompted chatbots.
  - Result: "significant instruction drift within eight rounds of conversations" for LLaMA2-chat-70B and GPT-3.5.
  - Code repo is named `persona_drift`, and Northeastern's press piece calls the work "Measuring and Controlling Persona Drift". — [arXiv](https://arxiv.org/abs/2402.10962v3); [HTML v4](https://arxiv.org/html/2402.10962v4); [Northeastern news](https://news.northeastern.edu/?p=230037)
- Mechanism proposed by Li et al.: attention decay. Attention to the system prompt stays almost constant within a turn but drops significantly across turns. Their geometric explanation is that out-of-distribution user utterances widen the cone of token embeddings. — [HTML v4](https://arxiv.org/html/2402.10962v4)
- Mitigations compared by Li et al.:
  - **Split-softmax** is parameter-free, needs no retraining, and boosts attention to system-prompt tokens with a power-law exponent. It gives the best overall stability/performance trade-off.
  - **System Prompt Repetition (SPR)** costs context but "excels in later conversation turns".
  - **Classifier-free guidance** stabilizes early turns but generalizes poorly to long dialogues.
  - Source: [HTML v4](https://arxiv.org/html/2402.10962v4)
  - Venue: often listed as COLM 2024. **[unverified this session]**
- **Luz de Araujo, Hedderich, Modarressi, Schütze, Roth, "Persistent Personas? Role-Playing, Instruction Following, and Safety in Extended Interactions", arXiv:2512.12775, EACL 2026.**
  - Setup: 7 open and closed LLMs, dialogues of more than 100 rounds.
  - Result: "persona fidelity degrades over the course of dialogues, especially in goal-oriented conversations, where models must sustain both persona fidelity and instruction following."
  - Trade-off: baseline models with no persona did better at first than persona-assigned ones. As dialogues went on, persona-conditioned responses came to resemble the baseline outputs.
  - Source: [arXiv](https://arxiv.org/abs/2512.12775); [ACL Anthology preview](https://preview.aclanthology.org/ingest-eacl/2026.eacl-long.246)
- **Gonnermann-Müller, Haase, Leins, Kosch, Pokutta, "Stable Personas: Dual-Assessment of Temporal Stability in LLM-Based Human Simulation", arXiv:2601.22812 (v2 May 2026).**
  - Scale: 3,473 between-conversation trials and 1,370 multi-turn conversations (18 turns), 7 LLMs.
  - Self-reports stay highly stable. Persona assignment explains 92.3% of variance in self-reports and 89.5% in observer reports.
  - Observer-rated expression declines within conversations, on a 36-point scale: high-intensity personas −3.50, moderate −2.40, low-intensity stable.
  - By model: Claude −1.6, GPT-5.1 −5.5.
  - Prompt design explained less than 1% of variance.
  - Source: [arXiv HTML](https://arxiv.org/html/2601.22812)
- **PersonaGym** (Samuel et al.), arXiv:2407.18416, Findings of EMNLP 2025.
  - Dynamic evaluation of persona agents: 200 personas, 10,000 questions, 150 environments, 5 tasks. Metric is "PersonaScore".
  - Claude 3.5 Sonnet scored only 2.97% (relative) above GPT-3.5, and larger or newer models were not reliably better at holding a persona.
  - Source: [arXiv HTML](https://arxiv.org/html/2407.18416v4)
  - Venue taken from a search snippet. Author list **[unverified]**.
- **CharacterEval**: Chinese role-play benchmark with 77 characters, 1,785 multi-turn dialogues, 23,020 examples, and 13 metrics in 4 dimensions (including character consistency). — [benchmark registry](https://leaderboard.steel.dev/registry/benchmarks/charactereval)
  - arXiv:2401.01275 (Tu et al., 2024). **[prior knowledge, not re-fetched]**
- **[prior knowledge, not re-fetched]** Other role-play consistency benchmarks:
  - **RoleLLM / RoleBench** (Wang et al., arXiv:2310.00746).
  - **InCharacter**: tests role-play fidelity by giving psychological interviews (BFI/MBTI) to role-playing agents (Wang et al., arXiv:2310.17976, ACL 2024).
  - **CoSER**: role-play benchmark, cited in PersMem below.
  - All of these mostly score short-horizon fidelity, not drift over long task streams.
- Multi-turn survey listing persona drift as a known failure mode: "Beyond Single-Turn: A Survey on Multi-Turn Interactions with LLMs", arXiv:2504.04717. — [arXiv](https://arxiv.org/pdf/2504.04717)

### Inferences
- What Li et al. call "System Prompt Repetition excels in later turns" is very close to what memory retrieval does. Re-injecting persona text near the generation point counters attention decay. A memory arm that re-injects persona-bearing episodes each task is partly SPR plus content effects. The experiment therefore needs an **SPR control**: the static persona prompt re-inserted on every task or turn. Without it, any memory advantage cannot be told apart from simply putting the persona text closer to the generation point.
- Persona decay is worse in goal-oriented dialogue (Luz de Araujo et al.), and drifting personas regress toward baseline behaviour. Both predict that a coding task stream will pull a prompt persona back toward the default assistant.
- Self-report probes ("describe your working style") will overstate stability (Stable Personas). The metric should be behavioural: actions taken in the task, as judged from trajectories.
- Persona intensity matters. A strong or extreme persona decays faster, so the house persona's intensity should be calibrated and reported.

### Gaps
- No study found that measures drift over a stream of **separate agentic tasks**, each a fresh context, as opposed to one growing dialogue. With a fresh context per task, the system prompt is always at the start. Drift there would come from task interference, not attention decay, and that is a different mechanism from Li et al.'s.
- No Qwen3-Coder-specific drift data found.
- PersonaGym author list and Li et al. venue not verified.

## Q2. Persona vectors, the Assistant Axis and activation steering: trait stability and drift

### Takeaway
Persona traits correspond to linear directions in activation space. These directions (a) predict the persona the model is about to adopt before it responds, (b) track personality shifts caused by prompts and by fine-tuning, and (c) can be clamped to stop drift. Anthropic's Assistant Axis work finds that **coding conversations keep models firmly in default-Assistant territory**, while therapy-style and philosophical conversations cause drift. This is directly relevant to a coding agent.

### Cited Findings
- **Chen, Arditi, Sleight, Evans, Lindsey, "Persona Vectors: Monitoring and Controlling Character Traits in Language Models", arXiv:2507.21509 (Jul 2025; v3 Sep 2025).**
  - Automated extraction of trait directions from a natural-language trait description alone.
  - Traits studied: evil, sycophancy, hallucination.
  - "Both intended and unintended personality changes after finetuning are strongly correlated with shifts along the relevant persona vectors."
  - The vectors can be used to monitor during deployment, to intervene after the fact, to steer preventively during training, and to flag training data at dataset and sample level.
  - Source: [arXiv](https://arxiv.org/abs/2507.21509)
- Persona vectors, further details from the Anthropic blog:
  - Models: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct.
  - The persona vector "activates before the response – it predicts the persona the model will adopt in advance", including under system prompts that encourage a trait to different degrees.
  - Post-hoc steering against a trait reduced it but degraded capability. Preventive steering during training caused "little-to-no degradation" in MMLU.
  - On LMSYS-Chat-1M, data flagging caught samples, such as romantic roleplay requests that activate sycophancy, that human reviewers missed.
  - Source: [Anthropic research](https://www.anthropic.com/research/persona-vectors)
- **Lu, Gallagher, Michala, Fish, Lindsey, "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models", arXiv:2601.10387 (Jan 2026).**
  - Method: extracted activation directions for many character archetypes. The leading principal component is the "Assistant Axis". Steering toward it reinforces assistant behaviour, and steering away makes models "more willing to adopt alternative identities".
  - Models: Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B.
  - Sources: [arXiv](https://arxiv.org/abs/2601.10387); [Anthropic research](https://www.anthropic.com/research/assistant-axis)
- Assistant Axis, drift by conversation type:
  - **Coding conversations kept models firmly within Assistant territory throughout.**
  - Therapy-style conversations caused "steady drift away from the Assistant", with the model beginning to role-play other characters.
  - Philosophical conversations about the model's own nature caused "significant drift".
  - User messages that predict drift: vulnerable emotional disclosure, meta-reflection requests, and requests for specific authorial voices.
  - Source: [Anthropic research](https://www.anthropic.com/research/assistant-axis)
- Assistant Axis, activation capping. Clamping activations to a fixed range along the axis "reduced harmful response rates by roughly 50% while preserving performance on capability benchmarks", and it stabilized behaviour against persona-based jailbreaks. Secondary reporting gives "nearly 60%" and "cut persona-based jailbreak success rates in half". Secondary coverage also says the axis came from PCA over 275 persona vectors.
  - Sources: [Anthropic research](https://www.anthropic.com/research/assistant-axis); [implicator.ai](https://www.implicator.ai/anthropic-finds-the-off-switch-for-ai-personality-drift/); [deeplearning.ai The Batch](https://www.deeplearning.ai/the-batch/large-language-models-can-drift-drift-from-helpful-personas-to-harmful-ones-but-new-research-aims-to-stabilize-them)
  - The primary page says ~50% and secondary sources say ~60%. Use the primary figure.
- Representation engineering background: Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency", arXiv:2310.01405 (2023). It reads and steers concepts such as honesty and emotion as linear directions. **[prior knowledge, not re-fetched]**

### Inferences
- Qwen 3 32B, which is close in family to Qwen3-Coder 30B, has a well-defined Assistant Axis. Coding tasks pin the model to the default Assistant region. That is a **built-in restoring force** working against any non-default persona during a coding stream. The force is not adversarial interference. It is a pull back toward the default, and both a prompt persona and a memory persona have to beat it.
- Persona vectors give a mechanism-level way to measure persona that does not depend on self-report. With Qwen3-Coder running locally, the team could extract a vector for the target persona trait and log the projection per task under each condition (prompt vs memory vs none). That shows whether memory keeps internal state shifted or only changes surface wording. This is feasible on an open-weights frozen model, but it is an addition to the design. **The direction has not been validated on Qwen3-Coder.**
- Activation steering is the natural "weight-free but below the prompt" comparison condition. It acts like a constant persona with no attention decay. It makes a useful ceiling for stability, though it may cost capability (post-hoc steering hurt MMLU in the Persona Vectors work).

### Gaps
- No study found that compares persona-vector projection under prompt-induced vs memory-retrieval-induced personas.
- No Qwen3-Coder (MoE) persona-vector results found. Steering MoE models may behave differently. **[unverified]**
- Assistant Axis per-turn drift curves (magnitude per turn) were not extracted from the full paper.

## Q3. Fine-tuning-induced personality change (weight-based personas) as a contrast

### Takeaway
Weight changes produce broad, persistent persona shifts, often unintended. Narrow fine-tuning can switch on a generalized "misaligned persona" that shows up on unrelated prompts. Weight-based personas are the most persistent and the least targeted. A frozen-model memory approach avoids this, at the cost of relying on context.

### Cited Findings
- **Wang et al. (OpenAI), "Persona Features Control Emergent Misalignment", arXiv:2506.19823, ICLR 2026.**
  - Emergent misalignment appears across fine-tuning on insecure code, RL on reasoning models, and various synthetic datasets, including in models without safety training.
  - SAE model-diffing finds "misaligned persona" features. A "toxic persona" feature most strongly controls the misalignment and predicts it.
  - A few hundred benign samples of fine-tuning restore alignment.
  - Sources: [arXiv](https://www.arxiv.org/pdf/2506.19823); [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/50db99ee3bccf73bfe1cf2af1e960414-Abstract-Conference.html)
  - Steering numbers from search snippets: positive steering induces ~40% misalignment, and negative steering cuts it by 30–50 points. **[from snippet; verify against the paper]**
- Persona Vectors (above): unintended personality changes from fine-tuning correlate strongly with shifts along persona vectors, and this can be predicted from the data before training. — [arXiv](https://arxiv.org/abs/2507.21509)
- A follow-up on data attribution of emergent misalignment via persona features exists, arXiv:2608.11025. It appeared in search results only and was not read. — [arXiv](https://arxiv.org/pdf/2608.11025)
- **[prior knowledge, not re-fetched]** Further background:
  - Betley et al., "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs", arXiv:2502.17424 (2025): the original finding.
  - Anthropic, "Claude's Character" (blog, June 2024): describes character training, a fine-tuning stage that instils traits through synthetic self-generated data and preference training.

### Inferences
- Weight-based personas generalize broadly and persist, which is the property the memory condition aims for. The cost is that they also spread to unintended contexts, as emergent misalignment shows. The "frozen weights + memory" design keeps the persona reversible and auditable. The weights-fixed premise of the experiment holds.
- "Persona is a direction the model selects", as in Wang et al. and Chen et al., suggests a mechanism for memory personas. Episodes written in the agent's own voice may act as many-shot in-context evidence that moves the same internal direction that fine-tuning would. This is a hypothesis to test with Q2's projection logging, not a finding.

### Gaps
- No study found that compares the same persona induced by fine-tuning vs by in-context episodes on stability under interference.

## Q4. Memory-based identity: generative agents, MemGPT/Letta, self-authored identity

### Takeaway
Memory architectures have been shown to make agents more *believable*, and memory systems can be built to carry identity. Direct evidence that memory gives *more stable* identity than a system prompt is thin and recent. The closest result is PersMem (Sep 2026, preprint): personality built into memory operations kept 89% of its trait effect under a conflicting instruction, against 58% for a prompt-persona baseline. Recent work also shows that **recall ≠ identity enactment**, and that self-authored identity drifts in a direction.

### Cited Findings
- **Zhang et al., "PersMem: Internalizing Personality into Dual-Pathway Memory for LLM Agents", arXiv:2609.34372 (Sep 28, 2026).**
  - A fixed personality vector shapes four memory operations: affective appraisal, retention, passive affect-driven retrieval, and active goal-driven retrieval. The response model never sees the personality explicitly.
  - Under adversarial framing ("discuss only positive content"), PersMem kept **89%** of its neuroticism-related retrieval differences, against **58% for a prompt-persona baseline**.
  - The authors say outright that the paper does *not* directly compare system-prompt and memory personas over long interactions and does not test extended multi-turn interactions.
  - Other results: Big-Five blind comparisons 67.5% (vs 60.8% for a random-memory control); neuroticism vs recalled negative valence r=−0.703; CoSER character fidelity 69.33.
  - Source: [arXiv HTML](https://arxiv.org/html/2609.34372)
- **Zhao & Zhao, "Identity Is More Than Recall: A Benchmark for Persistent Identity in Deployed AI Agents" (PAI-Bench), arXiv:2609.13637 (Sep 2026).**
  - Separates recall, composition, behavioural enactment, resistance, persistence, lineage, and role-conditioned updates.
  - Identifiers appeared in 48/48 atomic responses but only 1/48 implicit self-portraits.
  - Explicit field cues raised joint presence of identity markers from 0/8 to 7/8. Startup configuration ("body-label substitution") raised full-designation presence from 1/8 to 7/8.
  - Single samples, no adversarial testing. Evaluators varied by 12.5 points on identical responses.
  - Source: [arXiv](https://arxiv.org/abs/2609.13637)
- **Ng et al. (45 authors), "MicroVerse: An Instrument for Measuring Self-Authored Identity Drift in Long-Horizon Multi-Agent LM Simulations", arXiv:2608.15844 (Aug 2026).**
  - Agents have an immutable "soul file" plus a mutable, self-edited identity.
  - Self-authored identity showed **directional drift, not stabilization**. The largest category of self-added boundaries was anti-self-deception guardrails (27 of 111, 24%).
  - Drift patterns held up across reflection-threshold settings.
  - Source: [arXiv](https://arxiv.org/abs/2608.15844)
- PERMA benchmark (arXiv:2603.23231), as summarised in search results: reasoning LLMs keep explicit persona consistency well until context exceeds their effective window. Memory systems show "higher stability across temporal drift due to persistent persona representations". This concerns *user* persona/preference memory, not agent self-persona, and **only the search snippet was read**. — [arXiv](https://arxiv.org/html/2603.23231)
- Verification-gated persona state transitions (Symmetry 18(6):1037, MDPI, 2026):
  - Separates fast episodic memory from a slow persona state.
  - Persona changes are allowed only when traces satisfy explicit specifications, which is a design pattern against drift.
  - Search snippet only. — [MDPI](https://www.mdpi.com/2073-8994/18/6/1037)
- **[prior knowledge, not re-fetched]** Park et al., "Generative Agents: Interactive Simulacra of Human Behavior", arXiv:2304.03442, UIST 2023.
  - Architecture: memory stream, reflection, planning.
  - An ablation rated by human evaluators found the full architecture most believable, with each removed component (observation, reflection, planning) reducing believability.
  - Identity was seeded by a short natural-language seed memory per agent.
  - Reported failure modes: memory retrieval failures, embellishment/hallucination, and overly formal or cooperative behaviour attributed to instruction tuning. That last one is a pull toward the default assistant, consistent with Q2.
- **[prior knowledge, not re-fetched]** Packer et al., "MemGPT: Towards LLMs as Operating Systems", arXiv:2310.08560 (2023), now Letta.
  - "Core memory" holds editable *persona* and *human* blocks that stay in context and that the agent rewrites through function calls.
  - Evaluated on multi-session chat consistency and engagement (built on Multi-Session Chat).
  - Letta's docs describe persona blocks as the agent's self-concept, which it can edit itself.
- PersonaAgent: Bridging Memory and Action for Personalized LLM Agents (arXiv:2506.06254). It covers user personalization with episodic + semantic memory, not agent identity. — [arXiv](https://arxiv.org/pdf/2506.06254)

### Inferences
- PersMem's 89% vs 58% retention under a conflicting instruction is the closest available evidence for "memory persona survives interference better than prompt persona". But it is (a) a preprint, (b) measured at the level of retrieval behaviour, (c) a single conflicting instruction rather than a task stream, and (d) based on memory *operations* shaped by personality, not memories *written in the agent's own words*. Treat it as weak, indirect support.
- PAI-Bench and Stable Personas both show that having persona content in context is not the same as enacting it. A memory condition might retrieve persona-bearing episodes and still not act on them. The experiment should measure enactment separately from retrieval: log whether persona memories were recalled for each task, and separately whether behaviour matched.
- MicroVerse warns that self-written rules accumulate and drift in a direction. A memory persona can be *more* persistent and also *less* faithful to its seed. The design should consider a fixed persona core (a "soul file") plus mutable episodes, and should measure divergence from the original persona, not only consistency with the agent's own latest state.

### Gaps
- No controlled study found that compares an identical persona delivered as (a) static system prompt vs (b) retrieved episodes or self-written rules in the agent's own words, over a long task stream with opposing interference. This appears to be an open gap, and it is the experiment's contribution.
- Generative Agents and MemGPT findings were not re-fetched this session. Check the ablation details before citing.
- No evidence found on persona carried in memory for *coding* agents specifically.

## Q5. Does the persona affect task performance?

### Takeaway
Persona prompts generally do **not** improve accuracy on factual or reasoning tasks. Sociodemographic personas can sharply hurt reasoning. Persona-assigned models can start below the no-persona baseline in goal-oriented dialogue. A persona that makes the agent better at coding is therefore unlikely, and the experiment should track task success as a cost or side-effect metric.

### Cited Findings
- **Zheng, Pei, Logeswaran, Lee, Jurgens, "When 'A Helpful Assistant' Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of LLMs", arXiv:2311.10054, Findings of EMNLP 2024.**
  - Setup: 162 roles across 4 LLM families and 2,410 factual questions.
  - Personas did not improve performance over no persona, though persona gender, type and domain affected accuracy.
  - Picking the best persona per question would help, but automatically identifying it was no better than random.
  - Sources: [arXiv](https://arxiv.org/abs/2311.10054v2); [ACL Anthology](https://preview.aclanthology.org/new-sigs/2024.findings-emnlp.888/)
- **Gupta et al., "Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs", arXiv:2311.04892, ICLR 2024.**
  - Setup: 24 reasoning datasets, 4 LLMs, 19 sociodemographic personas.
  - With ChatGPT-3.5, 80% of personas show bias, and some datasets show drops of more than 70%. GPT-4-Turbo still shows bias in 42% of personas.
  - Simple prompt-based mitigations were insufficient.
  - Source: [arXiv](https://arxiv.org/abs/2311.04892)
- Luz de Araujo et al.: baselines with no persona did better at first than persona-assigned models on instruction-following and safety. There is a persona/task trade-off that shifts as dialogue length grows. — [arXiv](https://arxiv.org/abs/2512.12775)
- Persona Vectors: post-hoc steering to suppress traits degraded capability, while preventive steering preserved MMLU. Assistant Axis capping preserved capability benchmarks. — [Anthropic](https://www.anthropic.com/research/persona-vectors); [Anthropic](https://www.anthropic.com/research/assistant-axis)

### Inferences
- In the experiment, a persona that "survives" may cost task success, as Luz de Araujo et al. found for goal-oriented settings. Report persona fidelity and task pass rate together, for example as a 2-D frontier per condition. Otherwise "the persona faded" and "the agent got better at coding" may be the same event.
- Work-style personas (for example "always writes tests first", "prefers minimal diffs") are behavioural rather than sociodemographic. They are more likely to be neutral or helpful than the identity personas in Gupta et al. No study tests this directly for coding agents.

### Gaps
- No study found on how persona affects coding-agent task success (e.g., SWE-bench-style) for work-style personas.
- Whether memory-carried personas cost task performance differently from prompt personas is untested.
