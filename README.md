# Experiential Learning Harness (ELH)

Can an AI agent become more experienced without changing its weights? ELH is a research harness
that tests this on coding work: a frozen model carries memory from one task to the next, and a
hidden grader checks whether it gets better.

## What

A local model (Qwen3-Coder 30B, weights never change) fixes generated Python performance-bug
tickets in a sandbox. Test groups differ only in what they carry between tasks:

| Group | Carries |
|---|---|
| A | nothing (baseline) |
| B | a memory bank of past episodes plus one rule the agent wrote after each one |
| C | B, with memories ranked by how useful they proved |
| D | C, plus reusable procedures that must pass a validation gate |

Each group implements a published method: Hindsight, Learning on the Job, MemRL, Agent Workflow
Memory. Everything else is identical across groups: model, prompt, tools, budget and grader.

**Headline result.** On the bug type the baseline usually fails, memory with the agent's own rules
fixed 26 of 36 test tickets against 6 of 36 for the baseline (experiment 13, three learning
histories, p < 0.001). It used about a fifth of the GPU time per fixed ticket.

## So what

- **Memory helps where the agent would otherwise fail.** Where the baseline already succeeds, memory
  cut cost (about 8 fewer turns) but not failures. On the easiest bug the memory groups did worse than the baseline.
- **Plain memory is enough.** Ranking memories by usefulness (C) and validated procedures (D) did not
  beat plain memory (B).
- **What the agent learns first decides a lot.** One of three histories fixed only 3 of 12 test
  tickets because none of its rules covered a second slow lookup. Learning is path-dependent and
  noisy, so single runs mislead.
- **This is a pilot.** One model, mostly one bug type, 10–12 test tickets per group per run. The sealed
  tests that would confirm the main hypothesis have not been run.

Details: [results and limitations](docs/results.md) · [full write-up](docs/study-report.html).

## Now what

Two questions come next. For each, the literature review is done and the experiment is planned.

**1. Is it memory, or just a better prompt?** The gains so far could come from a well-written
prompt. With only five bug types, every lesson fits in one prompt. The plan is a generated task
family built on a made-up in-house library with many house conventions, some of which conflict.
It compares retrieved memory with the same rules frozen into a prompt, as the number of lessons
grows. No published study has run that comparison.
[Design](docs/design/house-lib-family.md) · [review](reports/Memory%20vs%20prompt%20datasets.md)

**2. Can memory give an agent a personality it keeps?** Measured by behaviour, not questionnaires.
Experiment 13's traces showed no style specific to one learning history, but memory moved every
history the same way: fewer, larger edits and fewer test runs. The question is now whether memory
changes a frozen model's working style, beyond its effect on success, and whether that change is
caused by the memory and kept over time. Whether each history gets a style of its own stays a
secondary question for the replicated runs. Prior work shows stable styles between different models
and persona drift within one, but has never tested a memory-made style in a single model.
[Review](reports/Agent%20personality%20through%20memory.md)

Related work for all three questions: [literature](docs/literature.md).

**Order of work.** Both questions change how the main result should be read, so their controls go
into the confirmatory study before its protocol is locked. One replicated set of 8–10 histories
serves all three questions.

1. ~~Reanalyse experiment 13's traces for personality signals.~~ Done: no history-specific style,
   but memory shifted every history's working style the same way, so question 2 now asks whether
   memory changes working style ([note](docs/h7-exp13-reanalysis.md)).
2. ~~Run the all-rules prompt on the current bug tickets.~~ Done: memory passed 26 of 36 against 18
   of 36 for all of its rules in the prompt, against an expected tie. Memory also showed past
   episodes, so that gap is not yet proof that retrieval beats a prompt
   ([note](docs/h6-exp14-static-prompt.md)).
3. Build, validate and calibrate the `house_lib` task family.
4. Run the core `house_lib` grid with replicated histories (several GPU days).
5. Lock the protocol and run the sealed confirmatory study.

Details: [`SPEC.md` §13](SPEC.md#13-phase-plan-and-exit-criteria).

## Find your way around

| Page | Contents |
|---|---|
| [Methods](docs/methods.md) | hypotheses, test groups, each paper's method and our deviations, task generator, protocol |
| [Results](docs/results.md) | all experiments so far, status and limitations |
| [Literature](docs/literature.md) | related work for each research question |
| [Reproducing](docs/reproducing.md) | requirements, configuration, commands, repository layout |
| [`SPEC.md`](SPEC.md) | full design specification |
| [`docs/papers/`](docs/papers/) | implementation notes per cited paper |

Quick start (needs the pinned model endpoint and Hindsight server, see
[Reproducing](docs/reproducing.md)):

```
uv sync
uv run elh run-batch experiments/batches/exp13-histories.yaml --dry-run
uv run elh analyze exp13-histories
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
