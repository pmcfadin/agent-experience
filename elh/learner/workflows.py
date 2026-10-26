"""Group D procedures: Agent Workflow Memory (Wang et al., ICML 2025), online mode, plus a dev gate.

What follows AWM (github.com/zorazrw/agent-workflow-memory, `webarena/induce_prompt.py`):
- only verified successes are induced from (our oracle verdict = AWM's `--criteria gt`);
- the pool is cumulative and one episode is sampled per group (`random_group_sample(..., 1)`);
  the group is the (cause, theme) cell, as AWM's was the WebArena `intent_template_id`;
- each example is `Query:` + steps of the model's reasoning text and its actions, with invalid and
  no-op actions dropped (`remove_invalid_steps`) and steps left without an action dropped;
- the prompt is AWM's `instruction.txt`, adapted only in its domain words, then a one-shot, then
  `## Concrete Examples ... ## Summary Workflows`, sent as one user message;
- the whole library is regenerated at every induction and injected whole (no retrieval);
- the inducing model is the acting model; temperature 0 (AWM's Mind2Web setting);
- output blocks that copy the one-shot are dropped (AWM `filter_workflows`'s "delta" guard).

Deliberate departures, each noted where it happens: our examples also carry a trimmed observation
(Mind2Web's induction input carried the observation, WebArena's did not); the library is versioned
with provenance; a candidate is promoted only through a paired dev-split gate and rolled back if the
retention panel drops (the proposal's additions; AWM has neither).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from elh.registry import WorkflowConfig  # noqa: F401  (re-export)
from elh.store import Store, utcnow

FORMAT_VERSION = "awm-v1"

# --------------------------------------------------------------------------------------------
# Prompt (module constants; any edit is a re-registration)
# --------------------------------------------------------------------------------------------

# AWM `webarena/prompt/instruction.txt`, verbatim except for the domain words:
#   "web navigation tasks" -> "terminal-based software performance tickets";
#   "(input text, button strings)" -> "(file paths, module/function/variable names, domain vocabulary, data sizes)";
#   the invariant-element examples (`id of "Search" or "Customers"`) -> commands every task shares.
INDUCTION_INSTRUCTION = """\
Given a list of terminal-based software performance tickets, your task is to extract the common workflows to solve these tasks.
Each given task contains a natural language instruction, and a series of actions to solve the task. You need to find the repetitive subset of actions across multiple tasks, and extract each of them out as a workflow.
Each workflow should be a commonly-reused sub-routine of the tasks. Do not generate similar or overlapping workflows. Each workflow should have at least two steps. Represent the non-fixed elements (file paths, module/function/variable names, domain vocabulary, data sizes) with descriptive variable names as shown in the example.
Keep the values of invariant elements, e.g., `python run.py INPUT.json OUTPUT.json` or `python -m unittest discover -s tests`, as they will share and stay invariant across tasks.
Try to generate as many workflows that can cover all the tasks in the input list."""

# The one-shot plays the role of AWM's `webarena/prompt/one_shot.txt`: concrete trajectories, then
# the summary workflows a reader should extract from them. It is written for a defect that is not a
# cpu_mem cause and must never become one (`per-record deepcopy`), in a theme no family uses
# (`greenhouse`), so it teaches the format without teaching any task's fix. Workflow blocks that
# mention its distinctive tokens are dropped from the output (AWM drops blocks that mention "delta").
ONE_SHOT = """\
## Concrete Examples

Query: Incident follow-up #5121 — Greenhouse climate report performance regression
The nightly greenhouse climate report takes hours once the sensor log is large, with one core at 100%. Output must stay byte-for-byte identical and `python run.py INPUT.json OUTPUT.json` must not change.
Actions:
<think>
I'll start by looking at the workspace layout and the README.
</think>
<action>
ls -R && cat README.md
</action>
<observation>
.:
README.md  examples  job  run.py  scripts  tests
...
- Synthetic load: `python scripts/make_load.py N OUT.json`
</observation>

<think>
Let me read the job's main module to see how readings are processed.
</think>
<action>
cat job/climate.py
</action>
<observation>
import copy
...
    for reading in readings:
        profiles = copy.deepcopy(sensor_profiles)
...
</observation>

<think>
Every reading deep-copies the whole profile table, so the work grows with readings x profiles. I'll measure on a larger input first.
</think>
<action>
python scripts/make_load.py 20000 big.json && time timeout 100 python run.py big.json out_before.json
</action>
<observation>
real	0m41.802s
</observation>

<think>
The copy is never mutated, so I can read the table directly instead of copying it per reading.
</think>
<action>
sed -i 's/profiles = copy.deepcopy(sensor_profiles)/profiles = sensor_profiles/' job/climate.py
</action>

<think>
Now check that the output is unchanged and the time dropped, and run the tests.
</think>
<action>
time timeout 100 python run.py big.json out_after.json && cmp out_before.json out_after.json && python -m unittest discover -s tests -q
</action>
<observation>
real	0m0.391s
...
OK
</observation>

Query: Incident follow-up #5188 — Greenhouse irrigation planner performance regression
The irrigation planner never finishes for large greenhouse zone lists; CPU is pinned. The schedule it writes must not change.
Actions:
<think>
First I'll find the source files.
</think>
<action>
find . -name "*.py" | sort
</action>
<observation>
./job/__init__.py
./job/plan.py
./run.py
./scripts/make_load.py
./tests/test_job.py
</observation>

<think>
Let me profile the planner on a mid-sized input to find the hot function.
</think>
<action>
python scripts/make_load.py 5000 mid.json && timeout 100 python -m cProfile -s cumtime run.py mid.json out.json | head -15
</action>
<observation>
   ncalls  tottime  cumtime filename:lineno(function)
...
     5000    0.210   38.950 plan.py:31(schedule_zone)
  5000    38.600   38.600 {built-in method copy.deepcopy}
</observation>

<think>
schedule_zone deep-copies the valve table for each zone. Let me read it.
</think>
<action>
sed -n 25,45p job/plan.py
</action>
<observation>
def schedule_zone(zone, valve_table):
    valves = copy.deepcopy(valve_table)
...
</observation>

<think>
The valves are only read, so I'll drop the per-zone copy, then compare outputs and run the tests.
</think>
<action>
cp out.json out_before.json && sed -i 's/valves = copy.deepcopy(valve_table)/valves = valve_table/' job/plan.py && timeout 100 python run.py mid.json out.json && cmp out_before.json out.json && python -m unittest discover -s tests -q
</action>
<observation>
OK
</observation>

## Summary Workflows

## Locate the hot path of a slow batch job
This workflow is used to find the function that dominates the runtime of a job that is slow on large inputs.
To see where the time goes, I need an input larger than the example but small enough to finish.
python scripts/make_load.py {medium_size} {input_file}.json
Next, I profile the job on it and read the top cumulative entries.
timeout {seconds} python -m cProfile -s cumtime run.py {input_file}.json out.json | head -15
The top entry names {hot_function} in {module_path}; I read that function.
sed -n {start_line},{end_line}p {module_path}

## Verify a performance fix keeps the output identical
This workflow is used to check that a change made the job faster without changing its output.
Before editing, I keep the output of the original code on a large input.
timeout {seconds} python run.py {input_file}.json out_before.json
After the edit, I time the job on the same input and compare the outputs byte for byte.
time timeout {seconds} python run.py {input_file}.json out_after.json && cmp out_before.json out_after.json
Finally, I run the public tests.
python -m unittest discover -s tests -q"""

# Tokens unique to the one-shot; an output block containing one is a copy of the example.
ONE_SHOT_TOKENS = ("greenhouse", "deepcopy", "irrigation", "valve", "sensor_profiles", "climate")



def _one_shot_lines() -> frozenset[str]:
    return frozenset(_norm(ln).lower() for ln in ONE_SHOT.splitlines() if ln.strip() and not ln.startswith("<"))


def copies_one_shot(text: str, threshold: float = 0.5) -> bool:
    """A workflow copied from the one-shot: it names one of the one-shot's distinctive tokens (AWM's
    "delta" filter), or at least `threshold` of its lines are lines of the one-shot verbatim. The
    one-shot's summary workflows are generic on purpose, so the token check alone misses copies of them."""
    low = text.lower()
    if any(tok in low for tok in ONE_SHOT_TOKENS):
        return True
    lines = [_norm(ln).lower() for ln in text.splitlines() if ln.strip()]
    ref = _one_shot_lines()
    return bool(lines) and sum(ln in ref for ln in lines) / len(lines) >= threshold


PROCEDURE_HEADER = ("Workflows: reusable procedures distilled from your earlier successful tickets. "
                    "Values in {braces} stand for task-specific names.")

# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------


# WorkflowConfig lives in elh.registry (the registration file holds it); re-exported here.


def approx_tokens(text: str) -> int:
    """Token estimate without a tokenizer (about 4 characters per token for code and English)."""
    return (len(text) + 3) // 4


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------------------------
# Inputs: which episodes, and what an example looks like
# --------------------------------------------------------------------------------------------


class LeakageError(ValueError):
    """An episode that induction may not see (not a verified `train` success) was passed to it."""


def check_inducible(row) -> None:
    """Induction sees only `train` episodes that the judge verified: passed, submitted, integrity
    clean. Dev, checkpoint, sealed and retention episodes never enter the pool (SPEC §10)."""
    eid = row["episode_id"]
    if row["split"] != "train":
        raise LeakageError(f"episode {eid} is from split {row['split']!r}; induction reads only train episodes")
    if not row["passed"] or row["outcome"] != "submitted":
        raise LeakageError(f"episode {eid} is not a verified success (passed={row['passed']}, "
                           f"outcome={row['outcome']})")
    checks = (json.loads(row["verdict_json"] or "{}").get("checks") or {})
    if checks.get("integrity") is not True:
        raise LeakageError(f"episode {eid} has no clean integrity check")


def eligible_pool(store: Store, run_id: str, history: int, group: str, family: str,
                  max_ordinal: int | None = None) -> list[sqlite3.Row]:
    """The cumulative success pool of one (run, history, group, family) stream, in train order."""
    sql = ("SELECT * FROM episodes WHERE run_id = ? AND history = ? AND test_group = ? AND family = ? "
           "AND split = 'train' AND passed = 1 AND outcome = 'submitted'")
    args: list = [run_id, history, group, family]
    if max_ordinal is not None:
        sql += " AND ordinal <= ?"
        args.append(max_ordinal)
    rows = store.db.execute(sql + " ORDER BY ordinal, started_at, episode_id", args).fetchall()
    out = []
    for r in rows:
        try:
            check_inducible(r)
        except LeakageError:
            continue
        out.append(r)
    return out


def sample_per_cell(rows: Sequence, seed: int, n: int = 1) -> list:
    """AWM `random_group_sample(template_dict, n)` over (cause, theme) cells, seeded."""
    cells: OrderedDict[tuple, list] = OrderedDict()
    for r in sorted(rows, key=lambda r: (r["cause"], r["theme"], r["episode_id"])):
        cells.setdefault((r["cause"], r["theme"]), []).append(r)
    rng = random.Random(seed)
    return [ex for group in cells.values() for ex in rng.sample(group, min(n, len(group)))]


def induction_seed(run_id: str, history: int, group: str, family: str, index: int) -> int:
    return int(_sha(f"{run_id}|{history}|{group}|{family}|{index}")[:12], 16)


@dataclass(frozen=True)
class Step:
    think: str
    actions: tuple[str, ...]
    observation: str = ""


@dataclass(frozen=True)
class Example:
    episode_id: str
    cell: tuple[str, str]
    query: str
    steps: tuple[Step, ...]


_NOOP = re.compile(r"^\s*(true|:|pwd|clear|sleep\s+[\d.]+|echo(\s+.*)?)\s*;?\s*$", re.S)
_TICKET = re.compile(r"<ticket>\n?(.*?)\n?</ticket>", re.S)


def _clip_lines(text: str, head: int, tail: int, width: int = 200) -> str:
    lines = [ln if len(ln) <= width else ln[:width] + " ..." for ln in text.rstrip("\n").splitlines()]
    if len(lines) <= head + tail:
        return "\n".join(lines)
    if head + tail == 0:
        return f"[{len(lines)} lines]"
    return "\n".join(lines[:head] + [f"... [{len(lines) - head - tail} lines omitted] ..."] +
                     (lines[-tail:] if tail else []))


def _clip_chars(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def _norm(text: str) -> str:
    return " ".join(text.split())


def contains_reference(text: str, forbidden: Iterable[str]) -> bool:
    """True if `text` holds any forbidden (reference / oracle) text, ignoring whitespace layout."""
    t = _norm(text)
    return any(f and _norm(f) in t for f in forbidden)


def _observation(exe: dict | None, obs_head: int, obs_tail: int) -> str:
    if exe is None:
        return ""
    out = exe.get("stdout") or ""
    if exe.get("stderr"):
        out += ("\n" if out and not out.endswith("\n") else "") + exe["stderr"]
    body = _clip_lines(out, obs_head, obs_tail) if out.strip() else ""
    if exe.get("timed_out"):
        body += ("\n" if body else "") + "[command timed out and was killed]"
    elif exe.get("exit_code"):
        body += ("\n" if body else "") + f"[exit code {exe['exit_code']}]"
    return body


_HEREDOC_WRITE = re.compile(r"^\s*cat\s*>\s*(\S+)\s*<<-?\s*['\"]?(\w+)['\"]?\n(.*?)\n?\2\s*$", re.S)


def _edit_as_diff(cmd: str, originals: dict[str, str], forbidden: Sequence[str]) -> str:
    """A whole-file write whose content equals the reference fix is shown as the agent's change
    (a zero-context unified diff against the original file, like `lesson-v2`'s diff), never as the
    file. Anything else that holds reference text is withheld."""
    import difflib

    m = _HEREDOC_WRITE.match(cmd)
    if m and m.group(1).removeprefix("./") in originals:
        path = m.group(1).removeprefix("./")
        new = m.group(3) + "\n"
        diff = "".join(difflib.unified_diff(originals[path].splitlines(True), new.splitlines(True),
                                            f"a/{path}", f"b/{path}", n=0))
        shown = f"# (file write of {path}, shown as its diff)\n{diff.rstrip()}"
        if not contains_reference(shown, forbidden):
            return shown
    return "[file edit withheld]"


def example_from_trace(episode_id: str, cell: tuple[str, str], events: list[dict], *,
                       forbidden: Sequence[str] = (), originals: dict[str, str] | None = None,
                       obs_head: int = 5, obs_tail: int = 5,
                       max_think_chars: int = 800, max_command_lines: int = 40,
                       max_command_chars: int = 2500) -> Example:
    """One AWM example from our trace: Query = the ticket (never the <experience> block); each step
    = the turn's visible text (AWM's <think>), its valid bash commands (AWM's <action>), and the
    command's trimmed output. Dropped like AWM's `remove_invalid_steps`: invalid tool calls,
    `restart`, `submit`, empty or no-op commands, and a command repeating the previous one verbatim.
    The verdict event is never read. Text holding forbidden (reference) content is withheld, except
    that a whole-file write of it is shown as a diff against `originals` (the task's workspace)."""
    start = next(e for e in events if e["event"] == "episode_start")
    m = _TICKET.search(start["first_message"])
    query = (m.group(1) if m else start["first_message"]).strip()
    execs = {e["tool_use_id"]: e for e in events if e["event"] == "tool_exec"}
    steps: list[Step] = []
    previous: str | None = None
    for e in events:
        if e["event"] != "response":
            continue
        think = " ".join(b.get("text", "").strip() for b in e["content"] if b.get("type") == "text").strip()
        actions, obs = [], []
        for b in e["content"]:
            if b.get("type") != "tool_use" or b.get("name") != "bash":
                continue
            inp = b.get("input") or {}
            cmd = inp.get("command")
            if inp.get("restart") or not isinstance(cmd, str) or not cmd.strip() or _NOOP.match(cmd):
                continue
            if b.get("id") not in execs:  # never executed: not a valid action
                continue
            if cmd.strip() == previous:
                continue
            previous = cmd.strip()
            if contains_reference(cmd, forbidden):
                shown = _edit_as_diff(cmd, originals or {}, forbidden)
            else:
                shown = _clip_lines(cmd.strip(), max_command_lines, 0, width=10_000)
            shown = _clip_chars(shown, max_command_chars)
            o = _observation(execs[b["id"]], obs_head, obs_tail)
            if contains_reference(o, forbidden):
                o = "[output withheld]"
            actions.append(shown)
            if o:
                obs.append(o)
        if not actions:
            continue
        if contains_reference(think, forbidden):
            think = ""
        steps.append(Step(_clip_chars(think, max_think_chars), tuple(actions), "\n".join(obs)))
    return Example(episode_id, cell, query, tuple(steps))


def _task_for(row):
    from elh.tasks.base import get_family

    family = get_family(row["family"])
    task = family.generate(row["cause"], row["theme"], row["seed"])
    if task.content_hash != row["task_hash"]:
        raise RuntimeError(f"task regeneration is not deterministic for {row['task_id']}")
    return family, task


def reference_texts(row) -> list[str]:
    """Reference text an induced workflow must never carry, for the guard: the family's reference
    fix for this episode's task (regenerated deterministically) and the oracle's own files."""
    family, task = _task_for(row)
    out = []
    if hasattr(family, "solution_core"):
        out.append(family.solution_core(task))
    out += [v for v in task.oracle.files.values() if len(v) > 200]
    return out


def format_example(ex: Example) -> str:
    """AWM `format_examples`/`format_trajectory`, with an <observation> per step (see module doc)."""
    parts = []
    for s in ex.steps:
        block = f"<think>\n{s.think}\n</think>\n<action>\n" + "\n".join(s.actions) + "\n</action>"
        if s.observation:
            block += f"\n<observation>\n{s.observation}\n</observation>"
        parts.append(block)
    return f"Query: {ex.query}\nActions:\n" + "\n\n".join(parts)


def build_prompt(examples: Sequence[Example]) -> str:
    """AWM `llm_generate`: INSTRUCTION, ONE_SHOT, then the examples between the two headers."""
    body = "\n\n".join(["## Concrete Examples"] + [format_example(e) for e in examples] + ["## Summary Workflows"])
    return "\n\n".join([INDUCTION_INSTRUCTION, ONE_SHOT, body])


# --------------------------------------------------------------------------------------------
# The library
# --------------------------------------------------------------------------------------------

# A workflow starts at a `## title` (the one-shot's form) or AWM's `Workflow N: title`. A single `#`
# is not a head: inside a workflow it is usually a bash comment.
_HEAD = re.compile(r"^\s*(#{2,3}\s+\S|(\*\*)?workflow\s*\d+\s*[:.])", re.I)
_PLACEHOLDER = re.compile(r"\{[A-Za-z][A-Za-z0-9_ -]*\}")


@dataclass(frozen=True)
class Workflow:
    title: str
    description: str
    steps: tuple[str, ...]

    @property
    def body(self) -> str:
        return "\n".join([f"## {self.title}", self.description, *self.steps])

    @property
    def body_sha256(self) -> str:
        return _sha(self.body)

    @property
    def skill_id(self) -> str:
        return "wf-" + self.body_sha256[:12]

    @property
    def placeholders(self) -> list[str]:
        return sorted(set(_PLACEHOLDER.findall(self.body)))


@dataclass(frozen=True)
class WorkflowLibrary:
    family: str
    workflows: tuple[Workflow, ...] = ()

    @property
    def text(self) -> str:
        return "\n\n".join(w.body for w in self.workflows)

    @property
    def library_sha256(self) -> str:
        """Content hash of the library (the learner state frozen into a snapshot)."""
        return _sha(f"{FORMAT_VERSION}\n{self.family}\n{self.text}")

    def __len__(self) -> int:
        return len(self.workflows)


EMPTY_LIBRARY_SHA256 = WorkflowLibrary("").library_sha256


def _parse_block(lines: list[str]) -> Workflow | None:
    lines = [ln.rstrip() for ln in lines if ln.strip() and ln.strip() not in ("```", "```bash", "```text")]
    if len(lines) < 4:  # AWM `load_workflows` keeps blocks of >= 4 lines
        return None
    title = re.sub(r"^#*\s*(\*\*)?(workflow\s*\d+\s*[:.]\s*)?", "", lines[0].strip(), flags=re.I).strip().strip("*").strip()
    return Workflow(title=title, description=lines[1].strip(), steps=tuple(lines[2:]))


def parse_workflows(text: str) -> list[Workflow]:
    """Segment the model output into workflows. AWM segments on blank lines (§2.3); we split on the
    `## title` lines the one-shot uses, falling back to blank-line blocks if there are none."""
    text = text.split("## Summary Workflows", 1)[-1] if "## Summary Workflows" in text else text
    lines = text.strip().splitlines()
    heads = [i for i, ln in enumerate(lines) if _HEAD.match(ln)]
    if heads:
        blocks = [lines[a:b] for a, b in zip(heads, heads[1:] + [len(lines)])]
    else:
        blocks = [b.splitlines() for b in re.split(r"\n\s*\n", text.strip())]
    return [w for w in (_parse_block(b) for b in blocks) if w is not None]


def assemble_library(family: str, raw: str, *, max_library_tokens: int,
                     forbidden: Sequence[str] = ()) -> tuple[WorkflowLibrary, list[dict]]:
    """Parse, then drop: one-shot copies, reference leaks, duplicates, and whatever exceeds the
    token cap (in output order). Returns the library and a log of what was dropped and why."""
    kept: list[Workflow] = []
    dropped: list[dict] = []
    seen: set[str] = set()
    used = approx_tokens(PROCEDURE_HEADER)
    for w in parse_workflows(raw):
        reason = None
        if copies_one_shot(w.body):
            reason = "copies the one-shot example"
        elif contains_reference(w.body, forbidden) or "def reference(" in w.body or "/oracle" in w.body:
            reason = "contains reference or oracle text"
        elif w.body_sha256 in seen:
            reason = "duplicate"
        elif used + approx_tokens(w.body) + 1 > max_library_tokens:
            reason = "over max_library_tokens"
        if reason:
            dropped.append({"title": w.title, "reason": reason, "body_sha256": w.body_sha256})
            continue
        seen.add(w.body_sha256)
        used += approx_tokens(w.body) + 1
        kept.append(w)
    return WorkflowLibrary(family, tuple(kept)), dropped


def render_procedure_block(library: WorkflowLibrary | None) -> str:
    """The text placed first inside the agent's <experience> block (the system prompt is untouched).
    AWM puts the whole workflow file in context with no per-query retrieval; so do we."""
    if library is None or not library.workflows:
        return ""
    return PROCEDURE_HEADER + "\n\n" + library.text


def procedure_ids(library: WorkflowLibrary | None, revision: int | None) -> list[str]:
    """Ids for `episodes.retrieved_memory_ids_json`: `skill_id@revision`, in rendered order."""
    if library is None or revision is None:
        return []
    return [f"{w.skill_id}@{revision}" for w in library.workflows]


def experience_block(library: WorkflowLibrary | None, memory_block: str = "") -> str:
    """Procedures first, then group C's recalled notes (what `run_episode(memory_block=...)` takes)."""
    return "\n\n".join(p for p in (render_procedure_block(library), memory_block.strip()) if p)


# GEPA optimises one free-text procedure (elh/learner/gepa_opt.py). It is stored and shown as a
# one-workflow library under this title, so both procedure sources share versioning, snapshots,
# rendering and the promotion gate.
GEPA_TITLE = "Working procedure"


def library_from_procedure(family: str, text: str) -> WorkflowLibrary:
    """A GEPA procedure as a library: one workflow, its first non-blank line the description and the
    rest its steps (blank lines dropped). Empty text is the empty library (no procedures shown)."""
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return WorkflowLibrary(family)
    return WorkflowLibrary(family, (Workflow(GEPA_TITLE, lines[0].strip(), tuple(lines[1:])),))


def procedure_text(library: WorkflowLibrary | None) -> str:
    """The text GEPA is seeded with for an incumbent library: the procedure itself for a GEPA
    library, the whole rendered library for an AWM one, "" when nothing is promoted."""
    if library is None or not library.workflows:
        return ""
    if len(library.workflows) == 1 and library.workflows[0].title == GEPA_TITLE:
        w = library.workflows[0]
        return "\n".join([w.description, *w.steps])
    return library.text


# --------------------------------------------------------------------------------------------
# Induction
# --------------------------------------------------------------------------------------------


@dataclass
class Completion:
    text: str
    usage: dict  # Messages-API usage keys, so elh.ledger.usage_entries prices it
    model: str
    request_id: str | None = None
    served: dict = field(default_factory=dict)


class InductionClient(Protocol):
    model_id: str

    def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> Completion: ...


class OpenAICompatInductionClient:
    """One user message, no tools, to the study alias (AWM: one chat call, same model as the agent).
    Holds the per-alias lock, like the agent client, so induction never overlaps an episode's call."""

    def __init__(self, model_id: str, base_url: str, api_key: str, timeout_s: float = 900):
        import openai

        self.model_id = model_id
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=4, timeout=timeout_s)

    def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> Completion:
        from elh.runner.models import SERVED_HEADERS, alias_lock

        with alias_lock(self.model_id):
            raw = self.client.chat.completions.with_raw_response.create(
                model=self.model_id, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens)
        resp = raw.parse()
        served = {v: raw.headers[k] for k, v in SERVED_HEADERS.items() if k in raw.headers}
        served["system_fingerprint"] = getattr(resp, "system_fingerprint", None)
        u = resp.usage
        cached = 0
        if u is not None and getattr(u, "prompt_tokens_details", None) is not None:
            cached = u.prompt_tokens_details.cached_tokens or 0
        usage = {"input_tokens": ((u.prompt_tokens if u else 0) or 0) - cached,
                 "output_tokens": (u.completion_tokens if u else 0) or 0,
                 "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}
        return Completion(resp.choices[0].message.content or "", usage, resp.model or self.model_id,
                          raw.headers.get("x-litellm-call-id") or resp.id, served)


def make_induction_client(model_cfg) -> InductionClient:
    """The acting model (AWM §3.2: same LM for induction and acting), from the registry's ModelConfig."""
    import os

    if model_cfg.provider != "openai_compat":
        raise NotImplementedError("workflow induction is wired for the self-hosted model only")
    base_url, api_key = os.environ.get(model_cfg.base_url_env), os.environ.get(model_cfg.api_key_env)
    if not base_url or not api_key:
        raise RuntimeError(f"set {model_cfg.base_url_env} and {model_cfg.api_key_env} for the local model gateway")
    return OpenAICompatInductionClient(model_cfg.id, base_url, api_key)


@dataclass
class Induction:
    library: WorkflowLibrary
    raw_output: str
    prompt_sha256: str
    source_episode_ids: list[str]
    usage: dict
    model: str
    seed: int
    dropped: list[dict]
    request_id: str | None = None
    served: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)   # source-specific provenance (GEPA: run summary)

    def provenance(self) -> dict:
        return {"format": FORMAT_VERSION, "model": self.model, "seed": self.seed, "usage": self.usage,
                "prompt_sha256": self.prompt_sha256, "raw_output_sha256": _sha(self.raw_output),
                "request_id": self.request_id, "served": self.served, "dropped": self.dropped, **self.extra}


def _load_trace(store: Store, row) -> list[dict]:
    return [json.loads(ln) for ln in store.get_blob("traces", row["trace_sha256"]).decode().splitlines() if ln]


def induce(store: Store, episodes: Sequence, client: InductionClient, *, cfg: WorkflowConfig = WorkflowConfig(),
           seed: int = 0, family: str | None = None) -> Induction:
    """Regenerate the whole workflow library from the success pool (AWM online, one induction step).

    `episodes` is the cumulative pool (episode ids or `episodes` rows), e.g. `eligible_pool(...)`.
    Every one must be a verified `train` success (else LeakageError); one is sampled per
    (cause, theme) cell with `seed`. Returns the candidate library and the model usage, which the
    caller charges as `learning` (see `learning_entries`)."""
    rows = [store.episode(e) if isinstance(e, str) else e for e in episodes]
    if not rows:
        raise ValueError("induction needs at least one verified train success")
    for r in rows:
        check_inducible(r)
    families = {r["family"] for r in rows}
    if len(families) != 1 or (family is not None and families != {family}):
        raise ValueError(f"induction is scoped to one family (AWM: one website); got {sorted(families)}")
    family = families.pop()
    chosen = sample_per_cell(rows, seed, cfg.samples_per_cell)
    forbidden = [t for r in chosen for t in reference_texts(r)]

    traces = {r["episode_id"]: _load_trace(store, r) for r in chosen}
    prompt = ""
    for head, tail in ((5, 5), (2, 2), (0, 0)):  # shrink observations until the prompt fits
        examples = [example_from_trace(r["episode_id"], (r["cause"], r["theme"]), traces[r["episode_id"]],
                                       forbidden=forbidden, originals=_task_for(r)[1].workspace,
                                       obs_head=head, obs_tail=tail) for r in chosen]
        prompt = build_prompt(examples)
        if approx_tokens(prompt) <= cfg.max_prompt_tokens:
            break
    else:
        raise ValueError(f"induction prompt (~{approx_tokens(prompt)} tokens) exceeds max_prompt_tokens")

    out = client.complete(prompt, max_tokens=cfg.max_output_tokens, temperature=cfg.temperature)
    library, dropped = assemble_library(family, out.text, max_library_tokens=cfg.max_library_tokens,
                                        forbidden=forbidden)
    return Induction(library=library, raw_output=out.text, prompt_sha256=_sha(prompt),
                     source_episode_ids=[r["episode_id"] for r in chosen], usage=out.usage, model=out.model,
                     seed=seed, dropped=dropped, request_id=out.request_id, served=out.served)


def learning_entries(exp_cfg, induction: Induction, model_id: str | None = None):
    """Ledger rows (category `learning`) for one induction call, priced under the registered model."""
    from elh.ledger import usage_entries

    return usage_entries(exp_cfg, model_id or exp_cfg.model.id, induction.usage, category="learning")


def should_induce(train_episodes_done: int, pool_ids: Iterable[str], last_source_pool: Iterable[str],
                  cfg: WorkflowConfig) -> bool:
    """Induce after every M train episodes, and only if a success joined the pool since the last
    induction (otherwise AWM's input would be unchanged)."""
    return (train_episodes_done > 0 and train_episodes_done % cfg.induce_every == 0
            and bool(set(pool_ids) - set(last_source_pool)))


# --------------------------------------------------------------------------------------------
# Versioning (store.skills): one row per workflow; a revision is one whole library
# --------------------------------------------------------------------------------------------

STATUSES = ("candidate", "promoted", "rejected", "superseded", "rolled_back")
_TRANSITIONS = {"candidate": {"promoted", "rejected"}, "promoted": {"superseded", "rolled_back"},
                "superseded": {"promoted"}}  # superseded -> promoted only by rollback


def _scope(run_id: str, history: int, group: str, family: str) -> tuple:
    return (run_id, history, group, family)


def record_candidate(store: Store, induction: Induction, *, run_id: str, history: int, group: str = "D") -> int | None:
    """Store a regenerated library as the next candidate revision. Returns the revision, or None if
    the library came out empty (nothing to validate; the spend is still charged by the caller)."""
    lib = induction.library
    if not lib.workflows:
        return None
    scope = _scope(run_id, history, group, lib.family)
    rev = store.db.execute("SELECT COALESCE(MAX(revision), 0) + 1 FROM skills WHERE run_id = ? AND history = ? "
                           "AND test_group = ? AND family = ?", scope).fetchone()[0]
    now = utcnow()
    prov = json.dumps(induction.provenance(), sort_keys=True)
    src = json.dumps(induction.source_episode_ids)
    with store.db:
        for pos, w in enumerate(lib.workflows):
            store.db.execute(
                "INSERT INTO skills (skill_id, revision, run_id, history, test_group, family, position, status, title, "
                "body, body_sha256, library_sha256, source_episode_ids_json, induction_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'candidate', ?, ?, ?, ?, ?, ?, ?)",
                (w.skill_id, rev, run_id, history, group, lib.family, pos, w.title, w.body, w.body_sha256,
                 lib.library_sha256, src, prov, now))
    return rev


def load_library(store: Store, *, run_id: str, history: int, group: str, family: str, revision: int) -> WorkflowLibrary:
    rows = store.db.execute("SELECT * FROM skills WHERE run_id = ? AND history = ? AND test_group = ? AND family = ? "
                            "AND revision = ? ORDER BY position", (*_scope(run_id, history, group, family), revision)
                            ).fetchall()
    if not rows:
        raise KeyError(f"no library revision {revision} for {run_id}/h{history}/{group}/{family}")
    wfs = []
    for r in rows:
        lines = r["body"].splitlines()
        w = Workflow(title=lines[0][3:], description=lines[1], steps=tuple(lines[2:]))
        if w.body_sha256 != r["body_sha256"]:
            raise ValueError(f"skill {r['skill_id']}@{revision} body does not match its hash")
        wfs.append(w)
    lib = WorkflowLibrary(family, tuple(wfs))
    if lib.library_sha256 != rows[0]["library_sha256"]:
        raise ValueError(f"library revision {revision} does not match its recorded hash")
    return lib


def revision_status(store: Store, *, run_id: str, history: int, group: str, family: str, revision: int) -> str:
    row = store.db.execute("SELECT DISTINCT status FROM skills WHERE run_id = ? AND history = ? AND test_group = ? "
                           "AND family = ? AND revision = ?", (*_scope(run_id, history, group, family), revision)
                           ).fetchall()
    if len(row) != 1:
        raise KeyError(f"revision {revision} missing or inconsistent")
    return row[0][0]


def _set_status(store: Store, scope: tuple, revision: int, status: str, *, validation: dict | None = None) -> None:
    current = revision_status(store, run_id=scope[0], history=scope[1], group=scope[2], family=scope[3],
                              revision=revision)
    if status not in _TRANSITIONS.get(current, set()):
        raise ValueError(f"revision {revision}: {current} -> {status} is not allowed")
    sets, args = ["status = ?"], [status]
    if status == "promoted":
        sets.append("promoted_at = ?")
        args.append(utcnow())
    if status == "rolled_back":
        sets.append("rolled_back_at = ?")
        args.append(utcnow())
    if validation is not None:
        sets.append("validation_json = ?")
        args.append(json.dumps(validation, sort_keys=True))
    store.db.execute(f"UPDATE skills SET {', '.join(sets)} WHERE run_id = ? AND history = ? AND test_group = ? "
                     "AND family = ? AND revision = ?", (*args, *scope, revision))


def promoted_revision(store: Store, *, run_id: str, history: int, group: str, family: str) -> int | None:
    row = store.db.execute("SELECT MAX(revision) FROM skills WHERE run_id = ? AND history = ? AND test_group = ? "
                           "AND family = ? AND status = 'promoted'", _scope(run_id, history, group, family)).fetchone()
    return row[0]


def promoted_library(store: Store, *, run_id: str, history: int, group: str, family: str
                     ) -> tuple[int | None, WorkflowLibrary]:
    """The library group D's episodes see now (empty before the first promotion)."""
    rev = promoted_revision(store, run_id=run_id, history=history, group=group, family=family)
    if rev is None:
        return None, WorkflowLibrary(family)
    return rev, load_library(store, run_id=run_id, history=history, group=group, family=family, revision=rev)


def library_state_sha256(revision: int | None, library: WorkflowLibrary) -> str:
    return _sha(json.dumps({"revision": revision, "library_sha256": library.library_sha256}, sort_keys=True))


def learner_state_sha256(store: Store, *, run_id: str, history: int, group: str, family: str) -> str:
    """Which library is promoted now (revision and content); the library half of group D's
    `memory_snapshots.learner_state_sha256` (see `group_d_state`)."""
    rev, lib = promoted_library(store, run_id=run_id, history=history, group=group, family=family)
    return library_state_sha256(rev, lib)


GROUP_D_STATE_FORMAT = "group-d-state-v1"


def group_d_state(utility_state_sha256: str, revision: int | None, library: WorkflowLibrary) -> bytes:
    """Group D's learner state as canonical JSON: the utility table's state hash (itself a stored
    "learner" blob, elh/learner/utility.py) and the promoted library (revision, content hash, and
    `library_state_sha256` of the two). The controller stores these bytes as a "learner" blob; their
    sha256 is the snapshot's `learner_state_sha256`, so it changes if either half changes, and the
    blob says how to restore both."""
    body = {"format": GROUP_D_STATE_FORMAT, "utility_state_sha256": utility_state_sha256,
            "library": {"family": library.family, "revision": revision, "library_sha256": library.library_sha256,
                        "state_sha256": library_state_sha256(revision, library)}}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def group_d_state_sha256(utility_state_sha256: str, revision: int | None, library: WorkflowLibrary) -> str:
    return hashlib.sha256(group_d_state(utility_state_sha256, revision, library)).hexdigest()


def decide_candidate(store: Store, *, run_id: str, history: int, group: str, family: str, revision: int,
                     decision: "GateDecision") -> None:
    """Apply a gate decision: promote (the old incumbent becomes `superseded`) or reject."""
    scope = _scope(run_id, history, group, family)
    with store.db:
        if decision.promote:
            old = promoted_revision(store, run_id=run_id, history=history, group=group, family=family)
            if old is not None:
                _set_status(store, scope, old, "superseded")
            _set_status(store, scope, revision, "promoted", validation=decision.to_dict())
        else:
            _set_status(store, scope, revision, "rejected", validation=decision.to_dict())


def apply_rollback(store: Store, *, run_id: str, history: int, group: str, family: str,
                   evidence: dict | None = None) -> int | None:
    """Mark the promoted revision `rolled_back` and re-promote the most recent `superseded` one
    (None if there is none: group D then runs with no procedures). Returns the revision now active."""
    scope = _scope(run_id, history, group, family)
    with store.db:
        cur = promoted_revision(store, run_id=run_id, history=history, group=group, family=family)
        if cur is None:
            raise ValueError("nothing is promoted; nothing to roll back")
        _set_status(store, scope, cur, "rolled_back", validation=evidence)
        prev = store.db.execute("SELECT MAX(revision) FROM skills WHERE run_id = ? AND history = ? AND test_group = ? "
                                "AND family = ? AND status = 'superseded' AND revision < ?", (*scope, cur)).fetchone()[0]
        if prev is not None:
            _set_status(store, scope, prev, "promoted")
        return prev


# --------------------------------------------------------------------------------------------
# Promotion gate and rollback rule (the proposal's additions; pure functions)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DevResult:
    """One validation episode. `reward` is SPEC §10's r = passed × (1 − normalised_cost)."""

    task_id: str
    repeat: int
    passed: bool
    reward: float
    integrity_violation: bool = False


def episode_reward(passed: bool, cost: float, cost_cap: float) -> float:
    """r = passed × (1 − normalised_cost), with cost normalised by a registered cap and clipped to [0, 1]."""
    if cost_cap <= 0:
        raise ValueError("cost_cap must be positive")
    return float(passed) * (1.0 - min(max(cost / cost_cap, 0.0), 1.0))


def dev_result_from_row(row, repeat: int, cost_cap_tokens: int) -> DevResult:
    """A DevResult from an `episodes` row, using total tokens as the cost (cap: budgets.max_total_tokens)."""
    tokens = sum((row[k] or 0) for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"))
    checks = json.loads(row["verdict_json"] or "{}").get("checks") or {}
    return DevResult(row["task_id"], repeat, bool(row["passed"]), episode_reward(bool(row["passed"]), tokens,
                     cost_cap_tokens), checks.get("integrity") is False)


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reasons: tuple[str, ...]
    stats: dict

    def to_dict(self) -> dict:
        return {"promote": self.promote, "reasons": list(self.reasons), **self.stats}


def _paired(a: Sequence[DevResult], b: Sequence[DevResult], repeats: int | None) -> list[tuple[DevResult, DevResult]]:
    ka = {(r.task_id, r.repeat): r for r in a}
    kb = {(r.task_id, r.repeat): r for r in b}
    if len(ka) != len(a) or len(kb) != len(b):
        raise ValueError("duplicate (task_id, repeat) in results")
    if set(ka) != set(kb):
        raise ValueError("results are not paired: the two sides ran different (task, repeat) sets")
    if not ka:
        raise ValueError("no results")
    if repeats is not None:
        per_task: dict[str, int] = {}
        for t, _ in ka:
            per_task[t] = per_task.get(t, 0) + 1
        short = {t: n for t, n in per_task.items() if n != repeats}
        if short:
            raise ValueError(f"expected {repeats} repeats per task, got {short}")
    return [(ka[k], kb[k]) for k in sorted(ka)]


def _summary(pairs: list[tuple[DevResult, DevResult]]) -> dict:
    n = len(pairs)
    inc = [p[0] for p in pairs]
    cand = [p[1] for p in pairs]
    return {
        "n": n,
        "tasks": len({r.task_id for r in inc}),
        "incumbent_mean_reward": sum(r.reward for r in inc) / n,
        "candidate_mean_reward": sum(r.reward for r in cand) / n,
        "mean_paired_diff": sum(c.reward - i.reward for i, c in pairs) / n,
        "incumbent_passes": sum(r.passed for r in inc),
        "candidate_passes": sum(r.passed for r in cand),
        "incumbent_integrity_violations": sum(r.integrity_violation for r in inc),
        "candidate_integrity_violations": sum(r.integrity_violation for r in cand),
    }


def promote(incumbent_results: Sequence[DevResult], candidate_results: Sequence[DevResult],
            cfg: WorkflowConfig) -> GateDecision:
    """Promote the candidate library only if, on the same dev tasks × `cfg.repeats`:
    (a) its paired mean reward beats the incumbent's by more than `cfg.margin`;
    (b) its pass count is not lower; (c) its integrity violations are not higher.
    The incumbent is the promoted library, or no procedures at all before the first promotion."""
    s = _summary(_paired(incumbent_results, candidate_results, cfg.repeats))
    reasons = []
    if not s["mean_paired_diff"] > cfg.margin:
        reasons.append(f"mean reward gain {s['mean_paired_diff']:.4f} is not above margin {cfg.margin}")
    if s["candidate_passes"] < s["incumbent_passes"]:
        reasons.append(f"pass count dropped {s['incumbent_passes']} -> {s['candidate_passes']}")
    if s["candidate_integrity_violations"] > s["incumbent_integrity_violations"]:
        reasons.append(f"integrity violations rose {s['incumbent_integrity_violations']} -> "
                       f"{s['candidate_integrity_violations']}")
    return GateDecision(not reasons, tuple(reasons) or ("all criteria met",), {"gate": "paired-dev-v1", **s,
                        "margin": cfg.margin, "repeats": cfg.repeats})


def rollback(previous_retention: Sequence[DevResult], current_retention: Sequence[DevResult],
             cfg: WorkflowConfig) -> GateDecision:
    """Auto-rollback rule at a checkpoint: the retention panel (same tasks, paired) is compared with
    the previous checkpoint's. Roll back if its pass rate or mean reward fell by more than
    `cfg.rollback_margin`. `GateDecision.promote` is True when a rollback is due."""
    s = _summary(_paired(previous_retention, current_retention, None))
    n = s["n"]
    pass_drop = (s["incumbent_passes"] - s["candidate_passes"]) / n
    reward_drop = -s["mean_paired_diff"]
    reasons = []
    if pass_drop > cfg.rollback_margin:
        reasons.append(f"retention pass rate fell by {pass_drop:.4f}")
    if reward_drop > cfg.rollback_margin:
        reasons.append(f"retention mean reward fell by {reward_drop:.4f}")
    return GateDecision(bool(reasons), tuple(reasons) or ("retention held",),
                        {"rule": "retention-rollback-v1", **s, "rollback_margin": cfg.rollback_margin})
