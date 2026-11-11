"""Read-only scan of stored traces for degenerate model replies (elh.runner.models.degenerate_reason).

Run after each batch: `uv run elh check-degenerate <batch-name-prefix>...`. Episodes recorded before
the client raised DegenerateOutputError may hold such replies; they count as agent failures unless
excluded (experiments/calibration/excluded-episodes.json)."""

from __future__ import annotations

import json

from elh.runner.models import degenerate_reason
from elh.store import Store


def scan(store: Store, prefixes: list[str], since: str | None = None) -> list[dict]:
    """Episodes whose batch_id starts with any prefix (all when none) and that started at or after
    `since` (ISO date), with one finding per degenerate response turn."""
    where, args = [], []
    if prefixes:
        where.append("(" + " OR ".join("batch_id LIKE ?" for _ in prefixes) + ")")
        args += [p + "%" for p in prefixes]
    if since:
        where.append("started_at >= ?")
        args.append(since)
    rows = store.db.execute(
        "SELECT episode_id, batch_id, test_group, task_id, replicate, ordinal, started_at, trace_sha256 FROM episodes"
        + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY started_at", args).fetchall()
    found = []
    for r in rows:
        if not r["trace_sha256"]:
            continue
        trace = store.get_blob("traces", r["trace_sha256"]).decode().splitlines()
        hits = []
        for line in trace:
            if '"event": "response"' not in line:
                continue
            ev = json.loads(line)
            for b in ev.get("content", []):
                why = b.get("type") == "text" and degenerate_reason(b["text"], ev.get("stop_reason"))
                if why:
                    hits.append({"turn": ev["turn"], "reason": why})
        if hits:
            found.append({"episode_id": r["episode_id"], "batch_id": r["batch_id"], "group": r["test_group"],
                          "task_id": r["task_id"], "replicate": r["replicate"], "ordinal": r["ordinal"],
                          "started_at": r["started_at"], "turns": [h["turn"] for h in hits], "reason": hits[0]["reason"]})
    return found
