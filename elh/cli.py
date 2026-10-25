"""`elh` command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from elh.registry import load_config


def cmd_validate_family(args) -> int:
    from elh.sandbox.base import make_sandbox
    from elh.tasks.base import get_family
    from elh.validation import validate_family

    family = get_family(args.family)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    results = validate_family(family, lambda: make_sandbox(args.sandbox), seeds,
                              causes=args.cause or None, themes=args.theme or None, workers=args.workers)
    bad = [r for r in results if not r.ok]
    by_variant = Counter((r.variant, r.ok) for r in results)
    for (variant, ok), n in sorted(by_variant.items()):
        print(f"{variant:18s} {'ok' if ok else 'FAIL':4s} {n}")
    for r in bad:
        print(f"FAIL {r.task_id} {r.variant}: {r.reason}")
    print(f"{len(results) - len(bad)}/{len(results)} checks passed")
    return 1 if bad else 0


def cmd_show_task(args) -> int:
    from elh.tasks.base import get_family

    task = get_family(args.family).generate(args.cause, args.theme, args.seed)
    print(f"task_id: {task.task_id}\ncontent_hash: {task.content_hash}\n")
    print(task.prompt)
    if args.out:
        out = Path(args.out)
        for rel, content in task.workspace.items():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_text(content)
        print(f"\nworkspace written to {out}")
    else:
        print("\nfiles: " + ", ".join(sorted(task.workspace)))
    return 0


def _client(kind: str, cfg, task, family):
    if kind == "model":
        from elh.runner.models import make_client

        return make_client(cfg.model)
    from elh.runner.scripts import scripted_client

    return scripted_client(kind.removeprefix("scripted-"), task, family)


def cmd_run_episode(args) -> int:
    from elh.controller import run_single_episode
    from elh.store import Store
    from elh.tasks.base import get_family

    cfg = load_config(args.config)
    family = get_family(args.family)
    task = family.generate(args.cause, args.theme, args.seed)
    store = Store(args.runs_dir)
    from elh.controller import memory_for

    episode_id, result, verdict = run_single_episode(
        cfg, store, task, _client(args.client, cfg, task, family), arm=args.arm, split=args.split,
        memory=memory_for(cfg, args.arm, 0, args.split), sandbox_backend=args.sandbox)
    print(json.dumps({
        "episode_id": episode_id, "task_id": task.task_id, "outcome": result.outcome,
        "passed": verdict.passed, "checks": verdict.checks, "turns": result.turns,
        "tool_calls": result.tool_calls, "usage": result.usage, "wall_s": round(result.wall_s, 1),
        "quality": verdict.detail.get("quality"), "submit_summary": result.submit_summary,
    }, indent=2))
    return 0


def cmd_regrade(args) -> int:
    from elh.controller import regrade_episode
    from elh.store import Store

    cfg = load_config(args.config)
    same, verdict, recorded = regrade_episode(cfg, Store(args.runs_dir), args.episode_id, args.sandbox)
    print(json.dumps({"reproduced": same, "recorded": {"passed": recorded["passed"], "checks": recorded["checks"]},
                      "regraded": {"passed": verdict.passed, "checks": verdict.checks}}, indent=2))
    return 0 if same else 1


def cmd_sandbox_setup(args) -> int:
    from elh.sandbox.user import setup

    users = setup(args.users)
    print(f"{len(users)} pool users: {', '.join(users)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="elh")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate-family", help="run generator validation (SPEC §4.4)")
    p.add_argument("family")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--cause", action="append")
    p.add_argument("--theme", action="append")
    p.add_argument("--sandbox", default="local")
    p.add_argument("--workers", type=int, default=6)
    p.set_defaults(fn=cmd_validate_family)

    p = sub.add_parser("sandbox-setup", help="create the UserSandbox pool users and network block (sudo)")
    p.add_argument("--users", type=int, default=8)
    p.set_defaults(fn=cmd_sandbox_setup)

    p = sub.add_parser("show-task", help="print a task's ticket; optionally write its workspace")
    p.add_argument("family")
    p.add_argument("cause")
    p.add_argument("theme")
    p.add_argument("seed", type=int)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_show_task)

    for name, fn, help_ in [("run-episode", cmd_run_episode, "run and grade one episode"),
                            ("regrade", cmd_regrade, "re-grade a stored episode and compare verdicts")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", required=True)
        p.add_argument("--runs-dir", default="runs")
        p.add_argument("--sandbox", help="override the registered sandbox backend (dev only)")
        if name == "run-episode":
            p.add_argument("--family", default="cpu_mem")
            p.add_argument("--cause", required=True)
            p.add_argument("--theme", required=True)
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--arm", default="A")
            p.add_argument("--split", default="dev")
            p.add_argument("--client", default="scripted-reference",
                           choices=["model", "scripted-reference", "scripted-wrong", "scripted-noop"])
        else:
            p.add_argument("episode_id")
        p.set_defaults(fn=fn)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
