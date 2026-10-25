"""Post-episode rule writing, after "Learning on the Job" (Tablan, Taylor, Bernhem; arXiv 2607.22157).

After a `train` episode the frozen agent model reflects once more, with its own conversation as
history, and writes ONE rule through a `memory_create` tool call (paper §3.3.3, App. A.4-A.5):

- passed  -> kind "verified": title `VERIFIED RULE: ...`, content `WHEN <situation> THEN <decisive action>`
- failed  -> kind "avoid":    title `AVOID: ...`, content `WHEN <situation> do NOT <exact action> --
  verified wrong against the ground-truth outcome.`

We are in the paper's *experience* grade (one verified bit plus the judge's per-check detail), never
the *instruction* grade: the reflection never sees the hidden oracle, the reference solution, the
reference timing or the hidden test inputs. Harness assistance, adapted from the paper:

- the CONSEQUENTIAL ACTIONS list (paper: non-read-only tool calls; here: the commands that wrote to
  the workspace, from the trace, plus the files that differ from the original workspace);
- the judge's per-check verdict in lesson-v2 wording (`CHECK_PASSED` / `CHECK_FAILED`), the protected
  files the agent changed, and its time on the large input against the budget;
- a write-time validator (App. A.7, `reflection_diff.rule_content_gaps`): a rule missing
  decision-critical strings is bounced back with the exact gaps, at most `max_retries` times, and
  then accepted with its gaps recorded (the paper's MAX_RULE_REJECTIONS = 2).

Prompt texts are the paper's (App. A.4, A.5, A.7 and the released `spark_agent.py`), with the
customer-service wording swapped for ours. Left out on purpose: optional further lessons (we keep
one rule per episode), memory feedback ratings (arm C territory) and twin-situation branch rules
(they need a store that honours "Supersedes").

Every model call's usage is returned so the controller can charge it to the ledger as `learning`.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from elh.judge import Verdict
from elh.memory.hindsight import CHECK_FAILED, CHECK_PASSED, OUTCOME_WORDS
from elh.runner.agent import EpisodeResult
from elh.runner.models import ContextLimitError, ModelClient
from elh.runner.prompts import SYSTEM_PROMPT, TOOLS
from elh.tasks.base import TaskInstance


# RulesConfig lives in the registry (it is part of a registration); re-exported here.
from elh.registry import RulesConfig  # noqa: E402


# --- prompts (module constants; any edit is a new registration) ------------------------------

# App. A.2 `memory_create(query, title, content, tags?)`, without tags.
MEMORY_CREATE_TOOL = {
    "name": "memory_create",
    "description": (
        "Save a new lesson to memory so future agents handle similar tickets better. Record one "
        "concise, reusable takeaway from this ticket -- a rule on how to resolve a problem, the "
        "correct action for a situation, or a pitfall to avoid. Capture the generalizable pattern "
        "(situation -> right action), not one-off facts like ticket numbers or this ticket's "
        "record field names."),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What a future agent would search to find this lesson."},
            "title": {"type": "string", "description": "A short title."},
            "content": {"type": "string", "description": "The concise, reusable lesson."},
        },
        "required": ["query", "title", "content"],
        "additionalProperties": False,
    },
    "strict": True,
}

# App. A.4 / `REFLECTION_NUDGE_TEMPLATE_V2`. {checks_block} takes the place of the paper's
# {outcome_block} ("What the customer ultimately did"); there is no correction, diff or rate line.
REFLECTION_TEMPLATE = """\
The task is now COMPLETE.

{verdict}
{checks_block}{consequential_block}{diagnosis_hint}
Now update your experience memory so future similar tasks go better. This is the ONLY thing to \
do now -- do NOT run any more commands and do NOT call any other tools (no bash, no submit):

{save_instructions}

Rules for the memory you write:
- NEVER contradict, weaken, or add exceptions to an existing memory whose title starts with \
"VERIFIED RULE:". If your experience seems to conflict with one, describe the conflict in \
your rule -- do not edit the existing rule.
- Do NOT state facts (timings, causes, library behaviour) that no command output in THIS \
conversation supports. If you cannot point to the evidence for a reason, write "Reason not \
documented in this task; rule verified against the ground-truth outcome." instead of a reason.
- Do not save ticket-specific facts (ticket numbers, this ticket's record field or variable names); \
write them as placeholders like <record id field>."""

# `_format_assessment` for the evaluator source.
VERDICT_LINE = "TASK OUTCOME (ground truth): the desired outcome {achieved} achieved."

# App. A.4 {diagnosis_hint}, on any failure (the customer-persuasion clause does not apply).
DIAGNOSIS_HINT = (
    "\nA wrong outcome doesn't always mean wrong facts. The cause may be the wrong action or "
    "procedure, a missing or mis-stated detail, the wrong file or tool. Work out which applied "
    "here and capture that lesson.\n")

# `_format_consequential_block`.
CONSEQUENTIAL_HEADER = ("CONSEQUENTIAL ACTIONS taken in this conversation, in order (the candidate "
                        "causes of the outcome):")
FILES_CHANGED_HEADER = "FILES CHANGED by the end of the task, compared with the original workspace:"
CHECKS_HEADER = "THE JUDGE'S CHECKS (hidden tests run after you finished):"

_QUERY_BULLET = ("- query: the question a future agent would ask mid-task when it meets this "
                 "situation (describe the observable problem, not this ticket's names).")
_WHEN_BULLET = ("- The WHEN clause must name the ticket's goal AND the observable behavior that "
                "distinguishes this situation from similar-looking ones (what the code, the profiling "
                "or the test output showed in THIS task). Describe only behavior that actually occurred.")
_DETAILS = ("file names and every decision-critical detail (functions, data structures, library calls) "
            "verbatim; ticket-specific identifiers as placeholders like <record id field>")

# App. A.5 success instruction, grounded variant (`SAVE_SUCCESS_GROUNDED`).
SAVE_SUCCESS_GROUNDED = f"""\
**1. Bank the win (required -- do this FIRST).** Call `memory_create` with the ONE decisive rule \
that made this task succeed:
- title: `VERIFIED RULE: <short name for this situation>`
- content: one sentence of the form "WHEN <the situation, as it appeared in this task> THEN <the \
DECISIVE action(s), chosen from the CONSEQUENTIAL ACTIONS list above -- {_DETAILS}>".
- The decisive action is the one that produced the graded outcome. It is often NOT the last thing \
that happened: a closing test run, a cleanup, or a summary at the end of the task is rarely what \
made the task succeed. Pick from the list above the action(s) that actually resolved the ticket.
{_WHEN_BULLET}
- Record the rule that actually WON. Do NOT soften it with alternatives or conditions you did not \
verify this time ("other options may be better if...") -- hedged rules do not survive the next task.
{_QUERY_BULLET}"""

# The variant without a consequential-actions list (`SAVE_SUCCESS_V2`).
SAVE_SUCCESS = f"""\
**1. Bank the win (required -- do this FIRST).** Call `memory_create` with the ONE decisive rule \
that made this task succeed:
- title: `VERIFIED RULE: <short name for this situation>`
- content: one sentence of the form "WHEN <the situation, as it appeared in this task> THEN <the \
exact action(s) you took that produced the successful outcome -- {_DETAILS}>".
{_WHEN_BULLET}
- Record the rule that actually WON. Do NOT soften it with alternatives or conditions you did not \
verify this time ("other options may be better if...") -- hedged rules do not survive the next task.
{_QUERY_BULLET}"""

# App. A.5 failure under experience (`SAVE_EXPERIENCE_FAIL_V2`).
SAVE_FAIL = f"""\
**1. Save the warning (required -- do this FIRST).** Call `memory_create` recording what to \
AVOID, from what actually happened:
- title: `AVOID: <the mistaken action, short>`
- content: one sentence of the form "WHEN <the situation, as it appeared in this task> do NOT \
<the exact action(s) taken that led to the failed outcome -- {_DETAILS}> -- verified wrong against \
the ground-truth outcome."
- Name the concrete action from THIS task, not a vague behavior ("be more careful").
{_WHEN_BULLET}
- You do NOT know what the correct answer was: do not invent one, and do not guess.
{_QUERY_BULLET}"""

# App. A.7. The paper's source is "the CORRECT solution / ACTION DIFF"; ours is the judge's verdict
# and the agent's own changes.
REJECT_GAPS = (
    "NOT SAVED -- the {label} is missing decision-critical content from the JUDGE'S CHECKS / "
    "CONSEQUENTIAL ACTIONS: {missing}. Each of these must appear VERBATIM in the rule (in the "
    "{clause} clause as a directive -- not as a placeholder like <file>, and not inside an 'e.g.' "
    "example). Call memory_create again now with the corrected content.")
# The paper's required-field guard, extended to the fields our rule needs.
REJECT_FIELDS = ("NOT SAVED -- memory_create requires non-empty {fields}. Call memory_create again "
                 "with the complete memory (title, content, query).")
REJECT_TITLE = ("NOT SAVED -- the desired outcome {achieved} achieved, so the title must start with "
                "\"{prefix}\". Call memory_create again now with the corrected title.")
REJECT_FORM = ("NOT SAVED -- the content must be one sentence of the form \"{form}\". Call "
               "memory_create again now with the corrected content.")
REJECT_NO_CALL = ("NOT SAVED -- you did not call memory_create. Call memory_create now with the "
                  "required rule.")
REJECT_OTHER_TOOL = "NOT EXECUTED -- the task is over; the only thing to do now is call memory_create."
REJECT_EXTRA = "NOT SAVED -- only one rule is kept per task; the first memory_create call is the one considered."

PREFIX = {"verified": "VERIFIED RULE:", "avoid": "AVOID:"}
FORM = {"verified": "WHEN <situation> THEN <decisive action(s)>",
        "avoid": "WHEN <situation> do NOT <exact action(s)> -- verified wrong against the ground-truth outcome."}
AVOID_SUFFIX = "-- verified wrong against the ground-truth outcome."

# Fallback when the seeded conversation no longer fits the context (`_render_conversation`).
TRANSCRIPT_MAX_CHARS = 16000
TRANSCRIPT_RESULT_CHARS = 2000
MAX_REQUIRED = 12  # `rule_required_strings(cap=12)`


@dataclass
class Rule:
    kind: Literal["verified", "avoid"]
    when: str
    then: str  # the action clause: "<action>" for verified, "do NOT <action> -- verified wrong ..." for avoid
    query: str
    title: str = ""

    @property
    def content(self) -> str:
        return f"WHEN {self.when} THEN {self.then}" if self.kind == "verified" else f"WHEN {self.when} {self.then}"

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.content}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RuleResult:
    rule: Rule | None
    calls: list[dict] = field(default_factory=list)  # per model call: request_id, model, stop_reason, usage, latency_s, served
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0,
                                                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})
    required: list[str] = field(default_factory=list)       # strings the rule must carry verbatim
    gaps: list[str] = field(default_factory=list)           # still missing when the rule was accepted
    rejections: list[str] = field(default_factory=list)     # the NOT SAVED messages sent back
    prompt: str = ""                                        # the reflection message
    fallback_transcript: bool = False
    error: str | None = None
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rule"] = self.rule.to_dict() if self.rule else None
        return d


# --- harness-computed context -----------------------------------------------------------------

_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1[^\n]*\n.*?\n\s*\2[ \t]*(?=\n|$)", re.S)
_QUOTED = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"")
_REDIRECT = re.compile(r"(?<![<>&])(?:\d|&)?>>?(?!&)\s*(?!/dev/null\b)[^\s;|&)]")
_WRITE_CMD = re.compile(r"(?:^|[\s;|&(])(?:tee|sed\s+(?:-\w+\s+)*-i\S*|perl\s+-\S*i\S*|mv|cp|rm|rmdir|touch|mkdir|"
                        r"patch|truncate|ln|install|dd|git\s+(?:apply|checkout|reset|restore|stash|mv|rm)|"
                        r"pip\s+install)(?=\s|$)")
_PY_WRITE = re.compile(r"\bpython[0-9.]*\b[\s\S]*?(?:\.write_text\(|\.write_bytes\(|\.write\(|json\.dump\(|"
                       r"open\([^)]*['\"][wax]\+?b?['\"]|os\.(?:remove|unlink|rename|replace|makedirs|mkdir)\(|"
                       r"shutil\.)")


def is_state_changing(command: str) -> bool:
    """Whether a bash command writes to the workspace, judged from its text: output redirection
    (not to /dev/null), file-writing commands, or Python code that writes files."""
    if _PY_WRITE.search(command):
        return True
    shell = _QUOTED.sub("''", _HEREDOC.sub("<<EOF", command))
    return bool(_REDIRECT.search(shell) or _WRITE_CMD.search(shell))


def _short_command(command: str, limit: int = 240) -> str:
    lines = command.strip().splitlines() or [""]
    head = lines[0] if len(lines[0]) <= limit else lines[0][:limit] + " ..."
    return head + (f"  [+{len(lines) - 1} more lines]" if len(lines) > 1 else "")


def consequential_actions(result: EpisodeResult) -> list[str]:
    """The workspace-writing commands the agent ran, in order (from the trace)."""
    out = []
    for ev in result.trace:
        if ev.get("event") == "tool_exec" and is_state_changing(ev.get("command") or ""):
            status = (" [timed out]" if ev.get("timed_out")
                      else f" [exit {ev['exit_code']}]" if ev.get("exit_code") else "")
            out.append(f"turn {ev['turn'] + 1}: {_short_command(ev['command'])}{status}")
    return out


def files_changed(before: dict[str, str], after: dict[str, str]) -> list[tuple[str, str]]:
    return [(p, "deleted" if p not in after else "created" if p not in before else "modified")
            for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]


def _checks_block(verdict: Verdict, result: EpisodeResult) -> str:
    lines = [CHECKS_HEADER]
    passed = [CHECK_PASSED.get(k, k) for k, ok in verdict.checks.items() if ok]
    failed = [CHECK_FAILED.get(k, k) for k, ok in verdict.checks.items() if not ok]
    if failed:
        lines.append(f"- FAILED: {'; '.join(failed)}")
    if passed:
        lines.append(f"- passed: {'; '.join(passed)}")
    violations = (verdict.detail or {}).get("integrity_violations") or []
    if violations:
        lines.append(f"- Protected files you changed (the ticket forbids this): {', '.join(violations)}")
    q = (verdict.detail or {}).get("quality") or {}
    if q.get("agent_s") is not None and q.get("budget_s"):
        status = q.get("large_status")
        lines.append(f"- On the large input your code took {q['agent_s']:.2f}s against a budget of "
                     f"{q['budget_s']:.2f}s ({q['agent_s'] / q['budget_s']:.0%} of budget"
                     f"{', ' + status if status and status != 'ok' else ''}).")
    ended = OUTCOME_WORDS.get(result.outcome, result.outcome).replace("its ", "your ")
    lines.append(f"- You {ended} after {result.turns} turns and {result.tool_calls} tool calls.")
    return "\n" + "\n".join(lines) + "\n"


def _consequential_block(result: EpisodeResult, changed: list[tuple[str, str]]) -> str:
    acts = consequential_actions(result)
    if not acts and not changed:
        return ""
    lines = [CONSEQUENTIAL_HEADER] + [f"- {a}" for a in acts] if acts else []
    if changed:
        lines += [FILES_CHANGED_HEADER] + [f"- {p} ({how})" for p, how in changed[:30]]
        if len(changed) > 30:
            lines.append(f"- ... and {len(changed) - 30} more")
    return "\n" + "\n".join(lines) + "\n"


def required_strings(verdict: Verdict) -> list[str]:
    """What a rule must carry verbatim (App. A.7's `rule_required_strings`). The paper draws them
    from the gold-vs-executed action diff: wrong argument values, EXTRA consequential actions (the
    thing to prohibit) and MISSING actions. Without the oracle we know only the EXTRA analogue:
    the protected files the agent changed. A passed episode has none, as in the paper, where a
    success is never validated for content."""
    return list((verdict.detail or {}).get("integrity_violations") or [])[:MAX_REQUIRED]


# `reflection_diff.rule_content_gaps` (MIT, memcoai/spark-continual-learning-paper-data): a
# requirement is unmet when absent, or when every occurrence follows "e.g."/"for example"/"such as".
_EXAMPLE_MARKER = re.compile(r"(?:e\.?g\.?|for example|such as)[^A-Za-z0-9]{0,8}$", re.I)


def rule_content_gaps(rule_text: str, required: list[str]) -> list[str]:
    haystack = " ".join((rule_text or "").split()).lower()
    gaps = []
    for req in required:
        needle = " ".join(req.split()).lower()
        if not needle:
            continue
        starts = [m.start() for m in re.finditer(re.escape(needle), haystack)]
        if not starts or all(_EXAMPLE_MARKER.search(haystack[max(0, s - 24):s]) for s in starts):
            gaps.append(req)
    return gaps


def parse_rule(kind: str, title: str, content: str, query: str) -> Rule | None:
    """Split `WHEN ... THEN ...` / `WHEN ... do NOT ...` content; None if it has neither form."""
    text = " ".join(content.split())
    if kind == "verified":
        m = re.match(r"^WHEN\s+(.+?)\s+THEN\s+(.+)$", text) or re.match(r"(?i)^when\s+(.+?)[\s,]+then\s+(.+)$", text)
    else:
        m = re.match(r"(?i)^when\s+(.+?)[\s,]+(do\s+not\s+.+)$", text)
    if not m or not m.group(1).strip(" ,:") or not m.group(2).strip():
        return None
    return Rule(kind, m.group(1).strip(" ,:"), m.group(2).strip(), query.strip(), title.strip())


# --- reflection -----------------------------------------------------------------------------------

def reflection_message(task: TaskInstance, result: EpisodeResult, verdict: Verdict,
                       final_ws: dict[str, str]) -> str:
    changed = files_changed(task.workspace, final_ws)
    consequential = _consequential_block(result, changed)
    if verdict.passed:
        save = SAVE_SUCCESS_GROUNDED if consequential_actions(result) else SAVE_SUCCESS
    else:
        save = SAVE_FAIL
    return REFLECTION_TEMPLATE.format(
        verdict=VERDICT_LINE.format(achieved="WAS" if verdict.passed else "was NOT"),
        checks_block=_checks_block(verdict, result),
        consequential_block=consequential,
        diagnosis_hint="" if verdict.passed else DIAGNOSIS_HINT,
        save_instructions=save)


def _seed_history(messages: list[dict], reflection: str) -> list[dict]:
    """The episode's conversation with the reflection appended as the next user turn. Pending tool
    results stay in that turn (every tool call keeps its result); a trailing plain user turn (the
    truncation nudge) is dropped, as the paper trims trailing non-assistant messages."""
    hist = copy.deepcopy(messages)
    while hist and hist[-1]["role"] == "user" and isinstance(hist[-1]["content"], str):
        hist.pop()
    text = {"type": "text", "text": reflection}
    if hist and hist[-1]["role"] == "user":
        hist[-1] = {"role": "user", "content": list(hist[-1]["content"]) + [text]}
    elif hist and any(b.get("type") == "tool_use" for b in hist[-1]["content"] if isinstance(b, dict)):
        pending = [{"type": "tool_result", "tool_use_id": b["id"], "content": REJECT_OTHER_TOOL, "is_error": True}
                   for b in hist[-1]["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        hist.append({"role": "user", "content": pending + [text]})
    else:
        hist.append({"role": "user", "content": reflection})
    return hist


def render_transcript(messages: list[dict], max_chars: int = TRANSCRIPT_MAX_CHARS) -> str:
    lines = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            lines.append(f"{'User' if m['role'] == 'user' else 'Agent'}: {content.strip()}")
            continue
        for b in content:
            t = b.get("type")
            if t == "text" and b.get("text", "").strip():
                lines.append(f"{'User' if m['role'] == 'user' else 'Agent'}: {b['text'].strip()}")
            elif t == "tool_use":
                lines.append(f"Agent action: {b.get('name')}({json.dumps(b.get('input') or {})})")
            elif t == "tool_result":
                body = b["content"] if isinstance(b["content"], str) else json.dumps(b["content"])
                lines.append(f"Tool result: {body[:TRANSCRIPT_RESULT_CHARS]}")
    text = "\n".join(lines)
    return "...\n" + text[-max_chars:] if len(text) > max_chars else text


def _with_temperature(client: ModelClient, temperature: float | None) -> ModelClient:
    if temperature is None or not hasattr(client, "temperature"):
        return client
    c = copy.copy(client)
    c.temperature = temperature
    return c


def _check(kind: str, inp: dict, required: list[str]) -> tuple[Rule | None, str | None, list[str]]:
    """(rule, rejection, gaps) for one memory_create call."""
    vals = {f: inp.get(f) if isinstance(inp.get(f), str) else "" for f in ("title", "content", "query")}
    missing = [f for f, v in vals.items() if not v.strip()]
    if missing:
        return None, REJECT_FIELDS.format(fields=" and ".join(missing)), []
    if not vals["title"].lstrip().upper().startswith(PREFIX[kind].upper()):
        return None, REJECT_TITLE.format(achieved="WAS" if kind == "verified" else "was NOT",
                                         prefix=PREFIX[kind]), []
    rule = parse_rule(kind, vals["title"], vals["content"], vals["query"])
    if rule is None:
        return None, REJECT_FORM.format(form=FORM[kind]), []
    gaps = rule_content_gaps(f"{vals['title']} {vals['content']}", required)
    if gaps:
        return rule, REJECT_GAPS.format(label=PREFIX[kind].rstrip(":"), clause="THEN" if kind == "verified" else "do NOT",
                                        missing="; ".join(f"'{g}'" for g in gaps)), gaps
    return rule, None, []


def write_rule(client: ModelClient, task: TaskInstance, result: EpisodeResult, verdict: Verdict,
               final_ws: dict[str, str], cfg: RulesConfig | None = None) -> RuleResult:
    """Run the reflection turn(s) and return the validated rule (or None) with every call's usage.

    The model sees its own episode conversation (system prompt and tools as in the episode, plus
    `memory_create`), then the reflection message. A rule missing required strings is bounced
    back up to `cfg.max_retries` times, then accepted with `gaps` recorded; a call with no usable
    rule at all after the last retry gives `rule=None` and `error`."""
    cfg = cfg or RulesConfig()
    kind = "verified" if verdict.passed else "avoid"
    start = result.trace[0] if result.trace and result.trace[0].get("event") == "episode_start" else {}
    system = start.get("system", SYSTEM_PROMPT)
    tools = list(start.get("tools") or TOOLS) + [MEMORY_CREATE_TOOL]

    required = required_strings(verdict)
    reflection = reflection_message(task, result, verdict, final_ws)
    out = RuleResult(rule=None, required=required, prompt=reflection)
    llm = _with_temperature(client, cfg.temperature)
    messages = _seed_history(result.messages, reflection) if result.messages else \
        [{"role": "user", "content": reflection}]
    last_rule: Rule | None = None
    last_gaps: list[str] = []

    for attempt in range(cfg.max_retries + 1):
        try:
            try:
                resp = llm.create(system=system, messages=messages, tools=tools, max_tokens=cfg.max_tokens)
            except ContextLimitError:
                if out.fallback_transcript or attempt > 0:
                    raise
                # The paper's fallback: an empty history with the transcript in the reflection message.
                out.fallback_transcript = True
                reflection = reflection + "\n\n--- Conversation transcript ---\n" + render_transcript(result.messages)
                out.prompt = reflection
                messages = [{"role": "user", "content": reflection}]
                resp = llm.create(system=system, messages=messages, tools=tools, max_tokens=cfg.max_tokens)
        except ContextLimitError as exc:
            out.error = f"reflection does not fit the context: {exc}"
            out.rule, out.gaps = last_rule, (last_gaps if last_rule else [])
            break
        for k in out.usage:
            out.usage[k] += resp.usage.get(k) or 0
        out.calls.append({"attempt": attempt, "request_id": resp.request_id, "model": resp.model,
                          "stop_reason": resp.stop_reason, "usage": resp.usage, "latency_s": resp.latency_s,
                          "served": resp.served})
        out.trace.append({"event": "rule_response", "attempt": attempt, "request_id": resp.request_id,
                          "stop_reason": resp.stop_reason, "usage": resp.usage, "content": resp.content})
        messages.append({"role": "assistant", "content": resp.content})

        uses = [b for b in resp.content if b.get("type") == "tool_use"]
        creates = [b for b in uses if b.get("name") == "memory_create"]
        if not creates:
            rejection = REJECT_NO_CALL
            results = [{"type": "tool_result", "tool_use_id": b["id"], "content": REJECT_OTHER_TOOL, "is_error": True}
                       for b in uses]
            messages.append({"role": "user", "content": results + [{"type": "text", "text": rejection}]}
                            if results else rejection)
        else:
            rule, rejection, gaps = _check(kind, creates[0].get("input") or {}, required)
            if rule is not None:
                last_rule, last_gaps = rule, gaps
            if rejection is None:
                out.rule = rule
                break
            results = []
            for b in uses:
                body = rejection if b is creates[0] else REJECT_EXTRA if b.get("name") == "memory_create" \
                    else REJECT_OTHER_TOOL
                results.append({"type": "tool_result", "tool_use_id": b["id"], "content": body, "is_error": True})
            messages.append({"role": "user", "content": results})
        out.rejections.append(rejection)
        out.trace.append({"event": "rule_rejected", "attempt": attempt, "message": rejection})
    else:
        # Out of retries: a well-formed rule is kept with its gaps (a lost write is worse than an
        # imperfect one); without one there is nothing to keep.
        if last_rule is not None and last_gaps:
            out.rule, out.gaps = last_rule, last_gaps
        else:
            out.error = f"no well-formed {PREFIX[kind]} rule after {cfg.max_retries + 1} attempts"
    if out.rule is not None:
        out.trace.append({"event": "rule_accepted", "rule": out.rule.to_dict(), "gaps": out.gaps})
    return out
