# Experiential Learning Harness (ELH) — Repository Specification

Build target for the study described in *Can AI Agents Become More Experienced?* (research
proposal v1.0, 2026-10-24). This spec turns that design into concrete schemas, interfaces and
exit criteria. Where the proposal leaves a choice open, the decision taken here is marked
**Decision** and the alternative is noted.

Status: draft v0.1 (2026-10-24). Covers Phase 0 in full detail and Phases 1–2 at contract level.
Phase 0 code is implemented; see §13 for what has and has not been verified.
2026-10-28: two hypotheses added (H6 memory vs static prompt, H7 memory-made personality; §12),
static-prompt control groups planned (§6.2), and the phase plan reordered (§13). Step 1 done the
same day: no history-specific style in exp13, so H7 was reframed (§12).

---

## 1. Fixed choices (frozen for the confirmatory study)

| Item | Value | Notes |
|---|---|---|
| Agent model | **Self-hosted Qwen3 Coder 30B A3B** via a self-hosted OpenAI-compatible gateway in front of llama.cpp (Decision, 2026-10-24) | Frozen by weights file hash, quantization, server build/flags and context size (`model.pin`); the controller refuses any non-dev split until the pin is complete. Served through the dedicated alias `study-qwen3-coder-30b-a3b`, which takes one request at a time (the client holds a per-alias lock). The alias shares its backend unit with other users' aliases, so every response's backend identity (gateway deployment id, `system_fingerprint`) is recorded and a change mid-episode is an infrastructure error; the backend's `/props` is stored with every episode. `claude-sonnet-5-5` is kept as an optional comparison configuration (`experiments/phase0-sonnet.yaml`). |
| Thinking / effort | n/a for the local model; Sonnet comparison: adaptive thinking, `effort: medium` | Any `reasoning_content` the gateway returns is kept in the trace and never sent back. |
| Refusal fallbacks (Sonnet comparison) | **off** | Server-side fallback would silently switch models mid-episode and break the frozen-model control. A `refusal` stop ends the episode as `outcome=refused` and is reported separately. |
| Sampling | local model: `temperature 0.7, top_p 0.8` (Qwen3-Coder recommended), preregistered | Model nondeterminism is handled by replicated learning histories, not by seeds. |
| Agent tools | `bash` + `submit` (local model: function tools; Sonnet: Anthropic-defined `bash_20250124` + strict `submit`) | Identical across groups. Each bash call is a fresh process in the workspace (files persist, shell state does not); the system prompt says so. Memory never appears as a tool in v1 (see §6.2). |
| Sandbox | `UserSandbox`: a throwaway Unix user per sandbox in `dev`, with an nftables network block and systemd-scope CPU/memory/task limits (§5) | Containers don't run in `dev` (nesting stays off, by owner decision 2026-10-25). `LocalSandbox` exists for development only and is never used for confirmatory data. |
| Memory service | dedicated Hindsight instance for the experiment | Pinned version and embedder, isolated banks. Its LLM is the `study-gemma4-26b-a4b` alias (pinned the same way), charged as `learning`. Never the shared household instance. |
| Language / tooling | Python 3.12, `uv`, SQLite, JSONL traces | Parquet export for analysis. |

## 2. Repository layout

```
SPEC.md                     this file
experiments/
  phase0-local.yaml         registration files (one per run family; immutable once a run starts)
elh/
  registry.py               ExperimentConfig (pydantic) + load/hash; price table
  controller.py             schedules (history × group × episode); split enforcement; snapshots
  cli.py                    `elh ...` entry point
  tasks/
    base.py                 TaskFamily / TaskInstance / Oracle contracts
    families/cpu_mem/       Phase 0 family: algorithmic CPU/memory regressions
  sandbox/
    base.py                 Sandbox contract
    local.py                LocalSandbox (dev only)
    user.py                 UserSandbox (confirmatory)
    podman.py               PodmanSandbox (unused; kept for a host with containers)
  runner/
    agent.py                manual tool-use loop; budgets; trace capture
    models.py               ModelClient: AnthropicClient, ScriptedClient (tests / dry runs)
    prompts.py              frozen system prompt + episode prompt rendering
  judge.py                  runs hidden oracle after submission; tamper checks
  ledger.py                 cost accounting (inference vs learning overhead)
  store.py                  SQLite schema + content-addressed trace store
  memory/                   Memory contract; NullMemory (A); HindsightMemory (B–D, Phase 1)
  learner/                  UtilityLearner (C), SkillLearner (D) — Phase 1–2
  analysis/                 curves, stratified bootstrap, non-inferiority — Phase 1
tests/
runs/                       (gitignored) SQLite DB, traces, snapshots per run
```

## 3. Experiment registration file

`experiments/<name>.yaml` is the preregistration record. The controller computes `config_hash`
(sha256 of the canonicalised file) and stamps it on every row it writes. A run refuses to start if a
file with the same `name` was previously run under a different hash.

```yaml
name: phase0-local
version: 1
model:
  provider: openai_compat         # or: anthropic
  id: study-qwen3-coder-30b-a3b
  base_url_env: WINTERMUTE_MODEL_API_BASE  # env var NAMES only; values live in a private env file
  api_key_env: WINTERMUTE_MODEL_API_KEY
  props_url: http://127.0.0.1:4120/coder/props   # llama.cpp /props via the read-only relay
  max_tokens_per_turn: 8192
  temperature: 0.7
  top_p: 0.8
  pin: {weights_sha256: "...", weights_file: "...", quantization: "...", server: "...", context_tokens: 65536}
prices_usd_per_mtok:            # pinned; ledger never looks prices up live (shadow prices for local)
  study-qwen3-coder-30b-a3b: {input: 0.05, output: 0.20}
budgets:                        # identical for every group
  max_turns: 40
  max_total_tokens: 400000
  wall_clock_seconds: 1200
  bash_timeout_seconds: 120
sandbox: {backend: podman, image: "localhost/elh-cpu-mem@sha256:...", cpus: 2, memory_mb: 2048, network: none}
groups: [A]                       # Phase 0: A only
histories: 1                    # independent learning histories (task orders)
families: [cpu_mem]
schedule:
  train_episodes: 0
  checkpoints: []               # e.g. [0, 25, 50, 100, 150, 200]
  checkpoint_tasks: 0
  sealed_tasks: 0
  seed: 20261024
amortization:                   # how learning spend is charged (§8.2)
  horizon_tasks: 1000
  sensitivity: [100, 500, 5000]
```

## 4. Task contract

### 4.1 Concepts

- **Family**: a domain of problems (`cpu_mem`, `io_query`, `containers`, `code_repair`).
- **Latent cause**: the underlying defect class within a family (e.g. `quadratic_membership`).
  Learning is expected to transfer *within* a cause and partly *across* causes.
- **Surface**: everything that varies without changing the cause: theme/domain vocabulary,
  identifiers, module layout, data sizes, the wording of the ticket.
- **Instance**: one concrete, deterministic task generated from `(family, cause, surface, seed)`.

### 4.2 Interfaces

```python
class TaskFamily(Protocol):
    name: str
    causes: list[str]
    themes: list[str]
    def generate(self, cause: str, theme: str, seed: int) -> TaskInstance: ...

@dataclass(frozen=True)
class TaskInstance:
    task_id: str                 # "{family}/{cause}/{theme}/{seed}" — stable, reproducible
    family: str; cause: str; theme: str; seed: int; difficulty: int
    prompt: str                  # the ticket shown to the agent
    workspace: dict[str, str]    # relpath -> file content placed in /workspace
    oracle: OracleBundle         # never enters the sandbox while the agent runs
    signature: dict              # normalised features for memory/task_signature
    content_hash: str            # sha256 over prompt + workspace + oracle

@dataclass(frozen=True)
class OracleBundle:
    files: dict[str, str]        # relpath -> content, copied to /oracle at grading time only
    command: list[str]           # run inside the sandbox; prints one JSON verdict on stdout
```

### 4.3 Splits (Decision)

Splits are assigned over **(cause, theme)** cells, not over seeds, so the sealed test contains
combinations the agent has never seen (H3):

| Split | Contents |
|---|---|
| `train` | most (cause, theme) cells; fresh seeds per episode |
| `checkpoint` | same cells as train, disjoint seeds; never written to memory |
| `sealed_familiar` | train cells, sealed seeds |
| `sealed_novel_combo` | held-out (cause, theme) cells whose cause and theme each appear in train |
| `sealed_novel_cause` | causes never seen in training (family shift) |
| `retention` | fixed panel of train-split tasks re-measured at each checkpoint |
| `dev` | used for SkillLearner validation and harness tuning; never reported (seeds 3000–3999 in batch experiments, train cells only) |

Seed ranges are disjoint per split (`seed = split_base + i`); `split_base` values for sealed splits
are kept in a file outside the repository until the sealed test is run.

### 4.4 Generator validation (required before any family is used)

For every cause × theme and ≥ 20 seeds:
1. the reference solution passes the oracle;
2. the untouched buggy workspace fails the oracle **on performance**, not correctness;
3. at least one plausible-but-wrong "fix" (provided per cause) fails on correctness;
4. hard-coding the public example output fails (hidden inputs are drawn from the oracle's own seed);
5. two generations with the same arguments are byte-identical (`content_hash` equal).

`elh validate-family <name>` runs this and must pass in CI.

### 4.5 Phase 0 family: `cpu_mem`

Each instance is a small Python batch job (`job/` package + `run.py` + a public example and test).
It reads an input JSON and writes an output JSON. The job is correct but contains one algorithmic
regression that makes it time out at production input sizes. The ticket describes the symptom
("nightly reconciliation runs for hours, CPU pinned") and never the cause.

Initial causes:

| Cause | Defect | Typical fix |
|---|---|---|
| `quadratic_membership` | `x in list` inside a loop | set / dict |
| `nested_scan_join` | for each record, scan another collection for its match | build an index |
| `queue_pop_front` | `list.pop(0)` / `insert(0, …)` work queue | `collections.deque` |
| `repeated_sort_min` | re-sorting to take the min/max inside a loop | `heapq` |
| `exponential_recursion` | naive overlapping recursion | memoisation / DP |

Planned for later: `busy_poll_loop`, `leaked_subprocesses`, `unbounded_cache_growth` (process-level
causes; need cgroup-level measurement from the podman backend).

Themes vary vocabulary and data shape: `inventory`, `logistics`, `telemetry`, `billing`,
`course_registration`, `network_flows`.

Oracle verdict (all must hold for `passed=true`):
- **correctness**: output equals the reference implementation's output on K hidden inputs
  (including edge cases: empty, duplicates, ties);
- **performance**: on the large hidden input the agent's job finishes within
  `max(perf_floor_s, perf_factor × reference_time)` measured in the same sandbox in the same run;
- **invariants**: public test still passes; the job's CLI contract (`run.py IN OUT`) is unchanged;
- **integrity**: protected files (input fixtures, `run.py` interface) unchanged.

Verdict also carries `quality` metrics: speed-up vs buggy baseline, agent runtime / reference runtime.

## 5. Sandbox contract

```python
class Sandbox(Protocol):
    def start(self, workspace: dict[str, str]) -> None: ...
    def exec(self, cmd: str, timeout_s: float) -> ExecResult: ...       # bash -lc in /workspace
    def put_files(self, dest_dir: str, files: dict[str, str]) -> None: ...
    def snapshot_workspace(self) -> dict[str, str]: ...                 # for trace + tamper checks
    def resource_usage(self) -> ResourceUsage: ...                      # cpu_s, peak_rss, wall_s
    def close(self) -> None: ...
```

- **UserSandbox** (confirmatory, see below). **PodmanSandbox** (unused in dev): one container per episode from a pinned image digest,
  `--network none`, CPU/memory limits from the registry, read-only rootfs except `/workspace` and
  `/tmp`. The judge runs in a **fresh** container that receives the agent's final workspace, so the
  agent cannot leave processes running or tamper with grading.
- `snapshot_workspace()` keeps text files ≤ 2 MB and skips caches; generated load data is not part
  of a solution and is not graded or stored.
- **LocalSandbox** (development only): a temp directory, subprocess with `prlimit`, scrubbed
  environment, timeouts. No isolation guarantee; rows produced with it are tagged
  `sandbox=local` and the analysis refuses to include them.

**Environment decision (2026-10-25):** the `dev` Incus container cannot create nested user
namespaces, and the owner is keeping nesting off; podman has been removed from dev. `wm-test`
leases were considered and rejected: test leases are capped at 30 minutes, the same as one
episode's wall-clock budget. Instead, the owner approved isolating episodes in `dev` itself:

- **UserSandbox** (`elh sandbox-setup` once, as root via sudo; rerun after a reboot because the
  nftables table isn't persistent). A pool of system users `elhsbx0..N`; each sandbox locks one,
  owns a fresh `0700` directory under `/var/lib/elh-sandbox/`, and runs each command through
  `sudo systemd-run --scope` as that user with `MemoryMax`, `MemorySwapMax=0`, `CPUQuota` and
  `TasksMax`, a scrubbed environment, and the workspace as cwd. An nftables `meta skuid` rule
  rejects every socket the pool users open, loopback included. `/home/dev` is `0750`, so the
  harness, oracle sources and credentials are unreadable. After every command all the user's
  processes are SIGKILLed; a sandbox refuses to start (infra error) if its user has processes that
  survive a kill or if its network rule is missing. Closing removes the directory and the user's
  files in `/tmp`, `/var/tmp` and `/dev/shm`. The judge gets its own fresh sandbox (usually another
  pool user). Tested in `tests/test_user_sandbox.py`.
- Weaker than a container: the pool users share the kernel, the process table (they can see other
  processes' command lines, not their files or environment) and the dev container's performance.
  Timing-sensitive grading shares the host with the model backends and with Hindsight, so
  Hindsight work runs strictly between episodes.

## 6. Agent runner

### 6.1 Loop

A manual tool-use loop over the Anthropic Messages API (no tool runner, no Agent SDK) so that every
request body is under harness control and recorded:

1. Build `system` = frozen system prompt (byte-identical across groups) + nothing else.
2. First user message = memory block (groups B–D; empty for A) + task ticket.
3. Loop: call model → append the full `response.content` (append-only history; never edit earlier
   turns) → execute every `tool_use` block → return all `tool_result`s in one user message.
4. Stop on: `submit` tool call, `end_turn` without tool calls (counts as implicit submit), budget
   exhaustion (`outcome=budget_exhausted`), `refusal` (`outcome=refused`), or API error after
   retries (`outcome=infra_error`, episode re-queued, never counted).
5. Prompt caching: top-level automatic `cache_control` so the growing transcript is cached; cache
   read/write tokens are recorded and priced separately.

### 6.2 Memory injection (Decision)

v1 injects recalled memory **once**, at episode start, as a fixed, delimited block in the first user
message, capped at `memory_token_budget` tokens. Rationale: the exact items shown and their order
are known (`retrieved_memory_ids`), and groups differ only in that block. A mid-episode `recall` tool
is a planned ablation, not part of the primary comparison.

**Agent-issued search (implemented, off by default, 2026-10-25).** `run_episode(..., search=fn)`
adds a `search_memory(query)` tool, following "Learning on the Job" (arXiv 2607.22157, retrieval mode
`nudge`, App. A.1–A.3 adapted). The system prompt advertises the tool, a `<memory_check>` reminder
is appended to the first message (search after gathering data with "problem + intended solution";
re-search when the situation changes; re-read VERIFIED RULEs before committing), and each call goes
to the read-only `fn(query, k)`. Every call's query, returned ids and ranks are recorded in
`EpisodeResult.memory_searches`. With `search=None` every request is byte-identical to before
(`tests/test_memory_search_tool.py` pins the hashes). Search tokens are episode `inference`.

**Rules (`elh/learner/rules.py`).** After a train episode the agent model reflects with its own
conversation as history and writes one `memory_create(query, title, content)` rule: `VERIFIED
RULE: … / WHEN … THEN …` on pass, `AVOID: … / WHEN … do NOT … -- verified wrong against the
ground-truth outcome.` on fail (paper App. A.4–A.5, experience grade). It gets the judge's checks
in lesson-v2 wording, protected-file violations, timing against the budget and the
workspace-writing commands, never the oracle, reference or hidden inputs. The write-time validator (App. A.7)
requires the protected files changed verbatim, with up to `max_retries` (2) bounces before the rule
is accepted with its gaps. Reflection tokens are `learning`.

*Storage (`rules.store`).* `extract` (default; exp8b, exp9): the rule leads the lesson-v3 episode
record, so Hindsight's LLM extraction paraphrases it (in exp8b no recalled unit kept "AVOID",
"VERIFIED" or "WHEN"). `verbatim` (exp10 on): the paper stores rules as written, so the record is
retained without the rule and the rule is a second retain item (`render_rule` + `Situation:
<query>`, tags `family:…`, `rule`, `rule:<kind>`, document `<episode_id>:rule`) under the bank
retain strategy `rule-verbatim` = `{retain_extraction_mode: chunks}`: one `world` unit holding the
text as written, no LLM call. Recall fills its first `rules.recall_k` (2) slots from a rule-only
recall by the ticket text (`tags: [rule]`, `tags_match: any_strict`), the rest as before; groups C/D
show those rules first and let the utility learner rerank the other `k2 - len(rules)` episodes
(shown rules are logged in `retrieval_json.rules` and get no Q update). `search_memory` returns rule
hits first. Remaining deviation: the paper retrieves rules only through the agent's own queries;
here they are also recalled by the ticket text at episode start.

**Static-prompt controls (planned, 2026-10-28; for H6).** Groups that carry a fixed block instead
of retrieved memory, so that "memory" and "a better prompt" can be told apart. The block sits in the
same first-message slot as B's `<experience>` block, so position is the same and only selection
differs (Decision; the alternative, appending it to the system prompt, is noted in
`docs/design/house-lib-family.md` §9). Groups:

- `P-all`: all of a history's B rules at the end of training, frozen, in training order, not capped
  by `memory_token_budget`;
- `P-oracle`: B's rules for the task's own lessons only; `P-wrong`: the same number of tokens of rules
  for other lessons;
- `P-hand`: hand-written statements of every lesson (`house_lib` only);
- `A0`: memory-group instructions with an empty block;
- for H7: `SPR`, a persona as a static block re-inserted every episode, and a placebo bank (matched
  length and format, neutral content).

P-all, P-oracle and P-wrong are built per history from that history's frozen B snapshot. The system
prompt stays byte-identical across all groups. Block tokens are charged as `inference` on every turn.

*Implemented (2026-10-28): P-all and A0* (`elh/batch.py` `static_memory`, `StaticMemory` in
`elh/memory/base.py`; experiment `exp14-static-prompt`). P-all's block is the source group's rules
from its first `static.after` train episodes, read from `episodes.rule_json` and rendered exactly
as B's verbatim rule items (`render_rule_verbatim`), in training order; episodes that wrote no rule
are skipped. Decision: when `agent_search` is on, both groups keep B's system prompt, tool list and
`<memory_check>` reminder, and `search_memory` returns "No matching memories found.", so the
instructions match B's and only the block differs. Their episodes log `ranker: static` and
`checkpoint = static.after`.

*B-rules (2026-10-30, experiment `exp15-rules-only`).* Opens the source group's own snapshot (under
the source run's stored registration) and fills every start slot and every search hit with verbatim
rules (`RulesOnlyMemory`), so it differs from B only in showing no episode records and from P-all
only in selecting among the rules.

### 6.3 Model client

`ModelClient.create(**request) -> Response` with two implementations: `AnthropicClient` (real API,
records `_request_id`, asserts `response.model` matches the pinned ID) and `ScriptedClient`
(replays a fixed list of tool calls; used by tests and generator validation so the full pipeline
runs without spending money).

## 7. Judge

`Judge.grade(task, final_workspace) -> Verdict`:
1. start a fresh sandbox with the agent's final workspace;
2. copy `oracle.files` to `/oracle` (first time they exist anywhere the agent's code can see);
3. run `oracle.command` with its own timeout; parse the single JSON verdict line;
4. integrity checks against the original workspace (protected files, interface);
5. return `Verdict{passed, correctness, performance, invariants, integrity, quality{…}, raw}`.

The agent's `submit` message is stored but never consulted for grading.

## 8. Cost ledger

For the self-hosted model, token prices in the registry are **preregistered shadow prices**: measured
GPU-seconds per input/output token × a fixed hardware rate, set from the pilot and changed only by
re-registering. Wall-clock latency per call is recorded alongside so the shadow price can be audited.

### 8.1 Entries

Every spend is a ledger row `{run_id, episode_id?, category, component, units, usd}` with
categories:
- `inference` — agent model calls (input, output, cache write, cache read tokens);
- `tool_compute` — sandbox CPU-seconds, priced at a registry constant (default $0 in Phase 0, but
  recorded);
- `learning` — Hindsight retain/reflect LLM + embedding calls, reranking, SkillLearner reflection
  and validation episodes, storage;
- `eval_overhead` — judge runs (reported, never charged to groups).

### 8.2 Primary endpoint

`cost_per_success(group) = (Σ inference + Σ tool_compute + amortised_learning) / verified_successes`
on the sealed split, where `amortised_learning = total_learning_spend × (n_sealed / horizon_tasks)`.
`horizon_tasks` is preregistered; the report also shows the sensitivity values and the
inference-only cost.

## 9. Data model (SQLite)

```sql
runs(run_id PK, name, config_hash, config_json, started_at, git_rev, harness_version)
episodes(episode_id PK, run_id, history, group, split, checkpoint, ordinal,
         task_id, family, cause, theme, seed, task_hash,
         model_id, sandbox_backend, sandbox_image,
         outcome,           -- submitted | implicit_submit | budget_exhausted | refused | infra_error
         passed, verdict_json,
         turns, tool_calls, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
         wall_s, sandbox_cpu_s,
         trace_sha256, retrieved_memory_ids_json, memory_snapshot_id,
         started_at, finished_at)
llm_calls(call_id PK, episode_id, turn, request_id, model, stop_reason,
          input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, latency_s)
ledger(entry_id PK, run_id, episode_id NULL, category, component, units, unit, usd,
       history NULL, group NULL, purpose NULL)  -- last three: spend with no episode (group D induction/reflection)
memory_snapshots(snapshot_id PK, run_id, history, group, after_episode, source_bank_id, bank_id,
                 bank_export_sha256, content_sha256, learner_state_sha256, library_revision NULL,
                 created_at)
skills(skill_id, revision, run_id, history, status, body_sha256, source_episode_ids_json,
       validation_json, promoted_at, rolled_back_at)          -- Phase 2
```

Traces are JSONL (one event per line: request, response, tool_exec, verdict), stored
content-addressed at `runs/<run>/traces/<sha256[:2]>/<sha256>.jsonl`.

## 10. Memory and learning contracts (Phase 1–2)

```python
class Memory(Protocol):
    def retain(self, episode: EpisodeRecord) -> list[str]: ...              # memory ids
    def recall(self, task: TaskInstance, k: int) -> list[MemoryItem]: ...
    def reflect(self, query: str) -> str: ...
    def export_snapshot(self) -> bytes: ...
    def open_readonly(self, snapshot_id: str) -> "Memory": ...              # for checkpoints/sealed

class Learner(Protocol):
    def observe(self, episode: EpisodeRecord, verdict: Verdict) -> None: ...
    def rank(self, task: TaskInstance, candidates: list[MemoryItem]) -> list[MemoryItem]: ...
    def propose_skill(self) -> SkillCandidate | None: ...                    # D only
```

Group wiring:

| Group | Memory | Learner |
|---|---|---|
| A | `NullMemory` | none |
| B | `HindsightMemory` (relevance recall) | none |
| C | `HindsightMemory` | `UtilityLearner`: two-phase retrieval (relevance top-N → rerank by learned Q); Q updated from verified reward `r = passed × (1 − normalised_cost)` |
| D | as C (own bank `…-D`, own utility table) | + procedure library shown first inside the `<experience>` block (system prompt unchanged; `workflows.experience_block`). Source `procedure_source: awm` (default): every `workflows.induce_every` train episodes, if a verified success joined the pool, AWM regenerates the whole library from D's verified `train` successes, one sampled per (cause, theme) cell (`elh/learner/workflows.py`); `gepa`: GEPA, seeded with the incumbent's text, searches on dev episodes (`elh/learner/gepa_opt.py`). Either way the candidate is versioned (`skills`) and promoted only by the paired dev gate below; auto-rollback if the retention panel drops |

**Group D's dev validation loop (2026-10-25).** Registration: `workflows` (cadence, gate, caps;
required), `utility` (as C; required), `gepa` (required when `procedure_source: gepa`). An experiment
with group D names a `dev` panel (train cells only, never a cell an eval or retention phase measures,
seeds 3000–3999). After D's n-th train episode, when `should_induce` fires: AWM induces a candidate
(its model calls are ledger rows `learning` with no episode, `purpose = awm_induction`), or GEPA runs
on the dev panel split deterministically into reflection / validation / promotion pools (its dev
episodes read a read-only view of D's live bank and utility table; reflection is `learning`,
`purpose = gepa_reflection`). The gate then runs incumbent (the promoted library, or none) and
candidate on the gate tasks × `workflows.repeats` (AWM: the whole dev panel; GEPA: only its
promotion pool, which its search never saw), interleaved per task in a seeded order, as `dev`
episodes charged as `learning` (`run_single_episode(charge_as="learning", procedures=…)`), with
`checkpoint` = n. Rewards r = passed × (1 − tokens / `budgets.max_total_tokens`); `workflows.promote`
(paired mean gain > `margin`, no fewer passes, no more integrity violations) is the one promotion
rule for both sources (GEPA's internal paired check is skipped), and its decision and the episode
ids are stored in `skills.validation_json`. Train and dev episodes show the library promoted when
they start; checkpoint, sealed and retention episodes show the revision their snapshot pinned. Every
D episode records `procedure_ids` (`skill_id@revision`), `procedure_revision` and the library hash in
`retrieval_json` (`retrieved_memory_ids_json` stays memory items only). Rollback: after D's last
episode of a `retention` phase, its panel is compared (paired by task) with the previous retention
checkpoint by `workflows.rollback`; the promoted revision is rolled back only if the rule fires, the
library differs between the two checkpoints and that revision is still the promoted one. Dev
episodes are excluded from the analysis comparisons and reported in the group D section (revisions,
decisions, learning spend).

Only `train` episodes call `retain`/`observe`. Checkpoint, sealed and retention episodes use
`open_readonly(snapshot)`; the controller passes them a read-only view and asserts that bank
contents and learner state hashes are unchanged afterwards.

Each (history, group) gets its own Hindsight bank: `elh-<run>-h<history>-<group>`.

**Deployment (2026-10-25).** `services/hindsight/` runs Hindsight natively in dev, in its own
locked environment (`hindsight-api-slim==1.1.0`, `uv.lock`), started by `serve.sh` under the systemd
user unit `elh-hindsight` (`install-service.sh`; lingering keeps it up; `journalctl --user -u elh-hindsight`):
loopback only (`127.0.0.1:8901`), embedded PostgreSQL instance `elh-hindsight` on port 8902 (data
in `~/.pg0/instances/elh-hindsight`), one worker, MCP off. Memory LLM: `study-gemma4-26b-a4b`, one
request at a time, **thinking off** (with it on, structured fact extraction spends its whole
budget reasoning and retain stalls). Embedder `intfloat/multilingual-e5-small` and reranker
`cross-encoder/ms-marco-MiniLM-L6-v2` run in-process on ONNX/CPU from local copies pinned by
revision and sha256 (`models.lock.json`, `fetch_models.py`); `HF_HUB_OFFLINE=1` at run time.

`HindsightMemory` (`elh/memory/hindsight.py`): recall queries with the ticket text and injects the
top k; retain stores one record per `train` episode, tagged by family, with the episode id as
document id. Record format `lesson-v1` (registered as `memory_service.record_format`): the content
leads with the outcome (failed checks, the agent's time against the budget, how the episode
ended), then the agent's own summary and its diff to source files (other files it created are
named, not shown; capped at 80 lines / 6000 chars); the ticket goes in Hindsight's extraction
`context`. Never the oracle or the reference timing. Chosen 2026-10-25 because the first format
(ticket first) made recall return ticket restatements: on two real episodes and three probe
tickets, top-5 recall held 12/15 restatements and 3/15 lessons; `lesson-v1` gave 5/15 and 10/15.
That first format stays selectable as `record_format: ticket-v0` (ticket first, fixed context; never
the oracle or reference timing) for experiments comparing the two formats.
**`lesson-v2` (2026-10-25, registration v6)** replaces `lesson-v1`, which described a failed check
with its pass wording: a failed integrity check was stored as "failed because it left the tests and
harness untouched", the opposite of what happened. In exp1 (first run) Hindsight extracted exactly
that lesson and group B broke integrity on 3/10 eval tickets against 1/10 for A, so that run's group B
result is void. v2 states passed and failed checks separately and names the protected files the
agent changed. `lesson-v1` stays selectable only to reproduce old runs. The
memory LLM's tokens from each retain are charged as `learning`. The controller waits until the
bank has no active operations (consolidation runs in the background after a retain) before the
next recall, so memory work never overlaps a timed episode.

**Snapshots (2026-10-25).** `snapshot_memory` (CLI `elh snapshot --group B --after n`) waits for the
training bank to go idle, exports it with Hindsight's bank transfer (data and bank config, no
audit/LLM history), stores the archive as a content-addressed blob (`runs/snapshots/`) and
restores it into the snapshot bank `elh-<run>-h<history>-<group>-e<n>` (re-embedding only, no LLM
call; imports run against the empty host bank `elh-transfer`). A restore gives memories new ids
and recomputes similarity links, so banks are compared by a content fingerprint (every memory
unit's type, state, tags and text, plus the document count), recorded as `content_sha256`; the
snapshot is refused unless the restored bank matches the live one. `open_snapshot` returns a
read-only view, re-restoring from the archive if the bank is gone and refusing if its fingerprint
has drifted. Group D's snapshot also pins its library: `library_revision`, and `learner_state_sha256`
is the sha256 of a stored "learner" blob, canonical JSON `{format: group-d-state-v1,
utility_state_sha256, library: {family, revision, library_sha256, state_sha256}}`
(`workflows.group_d_state`), so the hash changes if the utility table or the library changes;
`open_snapshot` restores both and re-checks the library against it, and read-only D episodes
re-check that the library they showed is unchanged. Checkpoint, sealed and retention episodes must name a snapshot (`--snapshot`); `dev`
may read the live bank read-only. Every read-only episode fingerprints the bank before recall and
after grading and is an infra error if it changed; `episodes.memory_snapshot_id` records the
snapshot used.

## 11. Controller and schedule

For each history h (independent seed → independent task order):
1. Group A runs only on evaluation splits (it doesn't learn, so it has no training phase).
2. Groups B–D run the same ordered train sequence; after episode n ∈ checkpoints, snapshot and run
   the checkpoint panel + retention panel against the snapshot.
3. After the final train episode, run the sealed splits (all groups, same tasks).

Episodes of different (history, group) streams run in parallel; within a stream they are strictly
sequential (learning depends on order).

## 12. Analysis (Phase 1+)

- Primary: D vs A cost-per-success ratio on sealed splits, with a hierarchical bootstrap
  (resample histories, then tasks stratified by family/cause); one-sided test of ≥20% reduction.
- Correctness gate: one-sided non-inferiority on pass rate, margin preregistered (proposal: 2 pp)
  — the margin and sample size are set from pilot variance before the confirmatory run.
- Ablations: B vs A, C vs B, D vs C. Curves: metric vs checkpoint per group (H2). Split-wise results
  for H3. Retention deltas for H5.
- **H6 (memory vs static prompt, added 2026-10-28).** Retrieved memory (B) beats a static prompt with
  the same content (P-all) once lessons are numerous or conflict across contexts; at few lessons
  P-all matches or beats B. Test: mixed-effects logistic regression `pass ~ group × log2(L) × c +
  (1 | task) + (1 | history)` over the `house_lib` grid (`docs/design/house-lib-family.md` §7), with
  per-lesson check pass as a second outcome. Refuted if P-all ≥ B at every lesson count.
- **H7 (memory-made working style, added 2026-10-28, reframed the same day).** Primary: carried
  memory changes a frozen model's working style, measured from trace features (caution,
  exploration, boldness, persistence, style), beyond what its effect on pass rate explains. Test:
  B vs A, A0 and a placebo bank on the `house_lib` grid, trait scores adjusted for pass/fail and
  ticket difficulty, with (d) causation by bank swap or deletion and (e) retention after later
  interfering episodes. Secondary: whether a history gives the model a style of its own, i.e.
  (a) repeatable across replicate runs of one snapshot, (b) distinct between histories beyond
  group A's run-to-run noise (between-history ICC) and (c) consistent across task families;
  answered from step 4's replicated histories. Reason: the step 1 reanalysis of exp13
  (`docs/h7-exp13-reanalysis.md`) found no history-specific signal (0–1 of 13 features per
  group, history attribution at chance) but a shared shift in all memory histories: fewer, larger
  source edits and fewer verification runs. Questionnaire self-reports are not used as evidence.
  Background: `reports/Agent personality through memory.md`.

## 13. Phase plan and exit criteria

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 | registry, cpu_mem family + validator, Local/Podman sandboxes, judge, agent loop, ledger, store, CLI | `validate-family cpu_mem` passes; one real episode with the registered model runs end-to-end in the confirmatory sandbox, with its trace, ledger rows and verdict; re-generating the task and re-grading the stored final workspace reproduces the verdict |
| 1 | Hindsight adapter (dedicated instance), groups A/B/basic C, second family, checkpoints, analysis v1 | 50-episode pilot with an interpretable learning curve and variance estimates → power analysis |
| 2 | SkillLearner, all four groups, four families, preregistration, sealed run | protocol locked; sealed test completed |

### Revised order (2026-10-28)

H6 and H7 bear on how H1 should be read: if a static prompt matches memory, a cheaper-per-success
result means less. Their controls therefore go into the confirmatory protocol before it is locked.
All three questions need more than three histories, so one replicated set of histories serves them
together. Groups C and D did not beat B in the pilot and move out of the core grid into extensions.

| Step | Work | Exit criterion | Cost (≈ 2 min per episode) |
|---|---|---|---|
| 1 | H7 go/no-go: trace-feature reanalysis of exp13 (3 histories × A/B/C), A's histories as the noise null | B/C histories differ more than A's after competence adjustment; if not, H7 is dropped or reframed. **Done 2026-10-28: no-go; H7 reframed** (§12, `docs/h7-exp13-reanalysis.md`) | hours, no episodes |
| 2 | H6 on `cpu_mem`: P-all (and A0) from exp13's frozen B snapshots, same test tickets | P-all vs B per history reported; a tie is the expected result at 5 causes. **Done 2026-10-29: B 26/36, P-all 18/36 (paired p = 0.021), A0 8/36 ≈ A 6/36**; B also saw episode records, so a B-rules group is needed to separate selection from content (`docs/h6-exp14-static-prompt.md`) | 72 episodes, ≈ 2.5 GPU hours |
| 3 | `house_lib` family: generator, oracle instrumentation, `validate-family`, then calibration (guessability and learnability filters) | validation passes; every lesson kept is unguessable by A and followable with its gold text | engineering, then ≈ 1.5 days |
| 4 | Core `house_lib` grid: A, B, P-all, P-oracle, P-hand × L ∈ {4, 16, 64} × c ∈ {0, 0.5}, 8–10 histories, 3–5 replicate runs per probe ticket per snapshot | H6 test as in §12; H7 repeatability and between-history ICC | ≥ 4.5 days (that figure is for 3 histories; scales with history count); a 2-world go/no-go first |
| 5 | Lock the protocol, including the H6/H7 controls and the history count from a power analysis, then run the sealed confirmatory study | protocol locked; sealed test completed | as Phase 2, larger |

### Phase 0 status (2026-10-24)

Verified on the LocalSandbox: task generation is deterministic; generator validation (§4.4) runs
for every cause × theme; scripted reference / wrong-fix / no-op agents produce the designed
verdicts end to end; stored episodes re-grade to identical verdicts; the Anthropic client's request
and response handling were checked against a local stub. 
2026-10-25: first real episode (dev split, LocalSandbox) with `study-qwen3-coder-30b-a3b` on
`quadratic_membership/inventory/1`: 28 turns, 146 s, submitted, failed only `performance`. The
model turned the `seen` list into a set but left `blocked` as a list, so the large input still timed
out; the verdict re-grades identically. Every call recorded the same backend identity. Then, on
UserSandbox with the v2 registration: `queue_pop_front/telemetry/2` passed every check (30 turns,
75 s, agent 0.97× the reference time) and re-grades identically; the 360-check family validation
also passes on UserSandbox. Group B: a scripted `train` episode retained into its bank (memory-LLM tokens
in the ledger as `learning`), and a later read-only dev episode on a related ticket got five
recalled memories while the bank's counts stayed unchanged. Recall mostly returns restatements of
the earlier ticket; the useful verdict detail is thin, which is worth watching in the pilot. Not yet verified: `/props` capture (§14).

## 14. Open items needing the owner

1. ~~Local model access from dev~~ (done 2026-10-25: loopback access, API key in a private env
   file outside the repository, aliases `study-qwen3-coder-30b-a3b` and `study-gemma4-26b-a4b`).
2. ~~Pin facts~~ (done 2026-10-25: supplied by the host administrator and matched against
   `/props`; registration v4 pins both the agent and the memory LLM. Gemma's
   `context_tokens` is the per-request 65536: `--ctx-size 131072` over 2 slots).
3. ~~Backend `/props`~~ (done 2026-10-25: a read-only `/props` relay on `127.0.0.1:4120`,
   `GET /coder/props` and `/gemma/props` only). Every episode fetches the agent's `/props` (and the
   memory LLM's on group B), stores it, and is an infra error if the weights file, build or
   per-request context differs from the pin; the relay is sent no credentials.
4. ~~Confirmatory sandbox~~ (decided 2026-10-25: UserSandbox in dev, §5).
5. ~~Dedicated Hindsight~~ and ~~checkpoint snapshots~~ (done 2026-10-25, §10). First real-model
   `train` episodes on group B ran 2026-10-25 (retain, learning ledger, recall into the next episode).
6. Optional: Anthropic credentials for the Sonnet comparison configuration.
7. ~~`house_lib` design decisions~~ (decided 2026-10-28, `docs/design/house-lib-family.md` §9):
   `hidden` lessons are an extension, not in the main grid; P-all uses raw rules, condensed rules
   belong to G-opt; the context size is raised if P-all at 64 lessons does not fit; the library
   starts small, with size as a later parameter.
8. Number of histories for the confirmatory study (8–10 proposed, set by power analysis after
   step 4) and the GPU time that implies.
