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


def first_user_message(task: TaskInstance, workdir: str, memory_block: str = "") -> str:
    parts = []
    if memory_block:
        parts.append(
            "<experience>\nNotes retrieved from your previous work on other tickets. They may or may "
            "not apply here.\n\n" + memory_block.strip() + "\n</experience>"
        )
    parts.append(f"Workspace: {workdir}\n\n<ticket>\n{task.prompt.strip()}\n</ticket>")
    return "\n\n".join(parts)
