"""Frozen prompts (SPEC §6). Byte-identical across arms; any edit requires a new registration."""

from __future__ import annotations

from elh.tasks.base import TaskInstance

SYSTEM_PROMPT = """\
You are a software engineer resolving an operations ticket in an isolated Linux sandbox.

Environment:
- The `bash` tool runs each command in a new bash process whose working directory is the task \
workspace. Files persist between calls; shell state (cd, exported variables, background jobs) does not.
- There is no network access. Python 3 is available as `python`.
- Long-running commands are killed after a timeout, so bound any experiment you run.

Work the way a careful engineer would: reproduce the problem, find the root cause, make the \
smallest change that fixes it, and verify that behaviour is unchanged before you finish. Your work \
is graded independently with hidden tests after you submit.

When you are done, call the `submit` tool once."""

SUBMIT_TOOL = {
    "name": "submit",
    "description": "Finish the task. Call exactly once, after your fix is complete and verified.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Root cause and fix in 1-5 sentences."},
        },
        "required": ["summary"],
        "additionalProperties": False,
    },
    "strict": True,
}

BASH_TOOL = {"type": "bash_20250124", "name": "bash"}

TOOLS = [BASH_TOOL, SUBMIT_TOOL]

# --- Optional agent-issued memory search (off by default) -----------------------------------
# Paper 2607.22157 ("Learning on the Job"), retrieval mode `nudge`: the agent writes its own
# query after gathering data and re-reads VERIFIED RULEs before acting. Texts are App. A.1-A.3,
# adapted minimally from customer service to tickets (the tool is `search_memory`, no tags).
# None of this is sent unless the episode runs with a search callable; with it off, the system
# prompt, tool list and first message are byte-identical to the ones above.

# App. A.1, inserted beside the instructions on how to work (the paper inserts it right after the
# policy's knowledge-base instruction).
MEMORY_SEARCH_ADDENDUM = """\
## Experience memory

**Search your experience memory** using the provided `search_memory` tool before completing a \
task. This contains knowledge learned in the course of resolving tickets, and lessons from past \
similar tickets."""

# App. A.2 (`memory_search(query, tags?)`).
SEARCH_MEMORY_TOOL = {
    "name": "search_memory",
    "description": (
        "Search your memory of past tickets for relevant experiences (what worked or didn't go well "
        "in the past). Use it whenever you are starting to handle a ticket, or before taking an "
        "important action -- to reuse what worked before and avoid known pitfalls."),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": (
                "What you want to recall, phrased as a question or situation (e.g. \"batch job too "
                "slow on large input; profiler shows time spent in one loop\").")},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

# App. A.3, appended once to the first user message.
MEMORY_CHECK_REMINDER = """\
<memory_check>
After you've collected all the data, and before you change any code, check your experience \
memory. Call the `search_memory` tool with a query describing this ticket's problem and your \
intended solution to check if you need to change anything before acting.
If the situation changes later in the task (a new symptom, cause, or approach appears), call \
`search_memory` again with the new specifics before acting on it.
Before you commit to a fix or submit, re-read any VERIFIED RULE from your memory search \
results: follow it, or state explicitly why it does not apply to this ticket before deviating.
</memory_check>"""

_ADDENDUM_ANCHOR = "\n\nWhen you are done, call the `submit` tool once."


def system_prompt(search_memory: bool = False) -> str:
    if not search_memory:
        return SYSTEM_PROMPT
    assert _ADDENDUM_ANCHOR in SYSTEM_PROMPT
    return SYSTEM_PROMPT.replace(_ADDENDUM_ANCHOR, "\n\n" + MEMORY_SEARCH_ADDENDUM + _ADDENDUM_ANCHOR)


def tools(search_memory: bool = False) -> list[dict]:
    return TOOLS + [SEARCH_MEMORY_TOOL] if search_memory else TOOLS


def first_user_message(task: TaskInstance, workdir: str, memory_block: str = "",
                       memory_check: bool = False) -> str:
    parts = []
    if memory_block:
        parts.append(
            "<experience>\nNotes retrieved from your previous work on other tickets. They may or may "
            "not apply here.\n\n" + memory_block.strip() + "\n</experience>"
        )
    parts.append(f"Workspace: {workdir}\n\n<ticket>\n{task.prompt.strip()}\n</ticket>")
    if memory_check:
        parts.append(MEMORY_CHECK_REMINDER)
    return "\n\n".join(parts)
