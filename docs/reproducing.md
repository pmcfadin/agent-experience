# Reproducing

Requirements, configuration, commands and repository layout, moved from the README unchanged.

## Requirements

- Linux with systemd and nftables, Python 3.12 and [uv](https://docs.astral.sh/uv/). The
  confirmatory `UserSandbox` runs each command as a throwaway system user through `sudo -n` (no
  password prompt), so the harness user needs passwordless sudo for `useradd`, `nft`,
  `systemd-run`, `install`, `pkill`, `find`, `rm`, `tar` and running commands as the pool users.
- An OpenAI-compatible endpoint (llama.cpp `llama-server`, directly or behind a gateway) serving
  two models under these ids:
  - `study-qwen3-coder-30b-a3b`: `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`, the agent;
  - `study-gemma4-26b-a4b`: `gemma-4-26B_q4_0-it.gguf`, Hindsight's memory LLM.

  Weights hashes, the llama.cpp build and the server flags are pinned in
  `experiments/phase0-local.yaml` (`model.pin`, `memory_service.llm.pin`). Outside the `dev` split the
  controller checks each backend's llama.cpp `/props` (at `props_url`) against the pin, so a
  different build, weights file or context size needs a new registration rather than an edit.
- A dedicated Hindsight 1.1.0 server on `127.0.0.1:8901`: see `services/hindsight/` (`serve.sh`,
  its own `uv.lock`, embedder and reranker pinned in `models.lock.json` and fetched by
  `fetch_models.py`; `install-service.sh` installs a systemd user unit). The unit file assumes the
  checkout is at `~/projects/agent-experience`; adjust `ExecStart` if it is not.

## Environment variables (names only)

| Variable | Used by | Purpose |
|---|---|---|
| `WINTERMUTE_MODEL_API_BASE` | harness, `serve.sh` | base URL of the OpenAI-compatible endpoint (name registered in the config) |
| `WINTERMUTE_MODEL_API_KEY` | harness, `serve.sh` | API key for that endpoint |
| `ELH_MODEL_ENV_FILE` | `serve.sh` | env file that defines the two variables above |
| `ELH_HINDSIGHT_PORT`, `ELH_HINDSIGHT_PG_PORT` | `serve.sh` | Hindsight and embedded PostgreSQL ports (8901, 8902) |
| `ELH_HINDSIGHT_URL` | tests | Hindsight base URL for integration tests |
| `ELH_LOCK_DIR` | harness | per-model-alias lock directory |

Keep keys in a private env file outside the repository; `.env` files are gitignored.

## Commands

```
uv sync
uv run pytest -q                                   # unit tests; some need Hindsight / the sandbox
uv run elh validate-family cpu_mem --sandbox local # generator validation (SPEC §4.4)
uv run elh sandbox-setup                           # once per boot: UserSandbox pool users + nftables rule
uv run elh run-batch experiments/batches/exp13-histories.yaml --dry-run       # print the schedule
uv run elh run-batch experiments/batches/exp13-histories.yaml --client model  # run it
uv run elh analyze exp13-histories                 # comparisons, bootstrap, ΔRR/ΔNL, GPU-seconds
```

Runs write a SQLite database, content-addressed JSONL traces, workspaces and memory snapshots under
`runs/` (gitignored; `--runs-dir` to change). `elh regrade` re-grades a stored episode and compares
verdicts. `--sandbox local` uses the `LocalSandbox` (a temp directory with resource limits, no isolation);
it is for development only, and each episode row records which backend it ran on.

## Repository layout

```
SPEC.md                    design specification: schemas, contracts, decisions, phase plan
experiments/
  phase0-local.yaml        base registration (models, pins, budgets, sandbox, memory service)
  phase0-sonnet.yaml       optional Claude Sonnet comparison configuration (dev only)
  batches/                 one file per experiment (exp1..exp13, probe-difficulty); the record of what was run
elh/
  cli.py                   `elh` entry point
  registry.py              experiment config schema, config hash, prices
  batch.py                 expands a batch file into a schedule of phases × histories × groups
  controller.py            runs episodes, enforces splits, snapshots, pins
  judge.py                 hidden-oracle grading and integrity checks
  ledger.py, store.py      cost ledger; SQLite schema and trace store
  analysis.py              comparisons and continual-learning metrics
  tasks/                   task contracts and the cpu_mem family (causes, oracle runner)
  sandbox/                 UserSandbox (confirmatory), LocalSandbox (dev), PodmanSandbox (unused)
  runner/                  tool-use loop, model clients, frozen prompts
  memory/                  NullMemory, HindsightMemory
  learner/                 rules.py (LotJ), utility.py (MemRL), workflows.py (AWM), gepa_opt.py (GEPA)
services/hindsight/        pinned, dedicated Hindsight server for the experiment
docs/methods.md            hypotheses, test groups, method sources and deviations, tasks, protocol
docs/results.md            results so far, status and limitations
docs/reproducing.md        this page
docs/literature.md         related work for each research question
docs/study-report.html     results write-up
docs/design/               designs for planned work (house_lib family)
docs/papers/               implementation notes per cited paper (they reference local copies of
                           papers and code that are not included here)
reports/                   literature reviews behind the planned work
research_notes/            source notes for those reviews
tests/
```
