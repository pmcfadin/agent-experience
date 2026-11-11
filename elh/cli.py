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

    memory = memory_for(cfg, args.group, args.history, args.split, store=store, snapshot_id=args.snapshot)
    episode_id, result, verdict = run_single_episode(
        cfg, store, task, _client(args.client, cfg, task, family), group=args.group, split=args.split,
        history=args.history, checkpoint=args.checkpoint, memory=memory, sandbox_backend=args.sandbox)
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


def cmd_snapshot(args) -> int:
    from elh.controller import memory_for, snapshot_memory
    from elh.store import Store

    cfg = load_config(args.config)
    store = Store(args.runs_dir)
    live = memory_for(cfg, args.group, args.history, "train", store=store)
    sid = snapshot_memory(cfg, store, live, history=args.history, group=args.group, after_episode=args.after)
    row = store.snapshot(sid)
    print(json.dumps({k: row[k] for k in row.keys()}, indent=2))
    return 0


def cmd_run_batch(args) -> int:
    from elh.batch import load_experiment, print_schedule, report, run_batch, schedule
    from elh.tasks.base import get_family

    exp, cfg = load_experiment(args.experiment)
    digest = None
    if exp.sealed:  # SPEC §13: the protocol lock must verify, also for a dry run (which only reports it)
        from elh import prereg

        res = prereg.verify(Path.cwd(), args.experiment)
        print(prereg.format_result(res))
        digest = res.digest
        if not res.ok and not args.dry_run:
            raise SystemExit("run-batch: sealed batch refused (see above)")
    if args.dry_run:
        print_schedule(exp, cfg, schedule(exp, cfg))
        return 0
    if args.client is None:
        raise SystemExit("run-batch: --client is required (model or scripted-*)")
    from elh.store import Store

    family = get_family(exp.family)
    shared = _client("model", cfg, None, family) if args.client == "model" else None
    store = Store(args.runs_dir)
    extra = {}
    if args.client != "model":  # group D's induction / reflection stand-ins (elh/runner/scripts.py)
        from elh.runner.scripts import ScriptedInductionClient, scripted_reflection

        extra = {"make_induction_client": ScriptedInductionClient,
                 "make_reflection_client": lambda: scripted_reflection}
    results = run_batch(exp, cfg, store, lambda task: shared or _client(args.client, cfg, task, family),
                        sandbox_backend=args.sandbox, sealed_digest=digest, **extra)
    report(exp, store, results)
    return 0


def cmd_prereg_lock(args) -> int:
    from elh import prereg

    root = Path.cwd()
    m = prereg.lock(root, args.batches, args.analysis or [])
    path = prereg.write_manifest(root, m)
    print(f"{path}: {len(m['files'])} files, digest {m['digest']}, commit {m['commit']}")
    print(f"commit it, then tag that commit {prereg.TAG} (not done here)")
    return 0


def cmd_prereg_verify(args) -> int:
    from elh import prereg

    res = prereg.verify(Path.cwd(), args.batch)
    print(prereg.format_result(res))
    return 0 if res.ok else 1


def cmd_measure_block(args) -> int:
    from elh.batch import load_experiment, measure_blocks
    from elh.store import Store

    exp, cfg = load_experiment(args.experiment)
    rows = measure_blocks(Store(args.runs_dir), exp, cfg, args.history, reserve=args.reserve)
    for r in rows:
        print(f"{r['group']:9s} {r['items']:4d} items  block {r['block_tokens']:6,d} tokens  first message + system + tools "
              f"{r['prompt_tokens']:6,d} of {r['context_tokens']:,d}  (+ {r['reserve']:,d} reserved: "
              f"{'fits' if r['fits'] else 'DOES NOT FIT'})")
    return 0 if all(r["fits"] for r in rows) else 1


def cmd_analyze(args) -> int:
    from elh.analysis import main as analysis_main

    return analysis_main(args.rest)


def cmd_sandbox_setup(args) -> int:
    from elh.sandbox.user import setup

    users = setup(args.users)
    print(f"{len(users)} pool users: {', '.join(users)}")
    return 0


def cmd_check_degenerate(args) -> int:
    from collections import Counter

    from elh.degenerate import scan, scan_retries
    from elh.store import Store

    found = scan(Store(args.runs_dir), args.prefixes, args.since)
    for f in found:
        print(f"{f['batch_id']}\t{f['group']}\t{f['task_id']}\trep{f['replicate']}\t{f['started_at']}\t"
              f"{f['episode_id']}\tturns {f['turns']}\t{f['reason']}")
    per = Counter(f["batch_id"] for f in found)
    print(f"{len(found)} degenerate episode(s)" + "".join(f"\n  {b}: {n}" for b, n in sorted(per.items())))
    retries = scan_retries(Store(args.runs_dir), args.prefixes, args.since)
    for f in retries:
        print(f"{f['batch_id']}\t{f['group']}\t{f['task_id']}\trep{f['replicate']}\t{f['episode_id']}\t"
              f"{f['event']} {f['turn']}\tfirst {f['first_request_id']} ({f['first_output_tokens']} tok, {f['first_finish']})"
              f"\tretry_clean={f['retry_clean']} cached={f['retry_cached_tokens']}")
    clean = sum(1 for f in retries if f["retry_clean"])
    print(f"{len(retries)} cache-retried turn(s): {clean} clean, {len(retries) - clean} still degenerate")
    per = Counter((f["batch_id"], f["retry_clean"]) for f in retries)
    print("".join(f"  {b}: retry_clean={c}: {n}\n" for (b, c), n in sorted(per.items())), end="")
    return 1 if found else 0


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

    p = sub.add_parser("snapshot", help="freeze a group's training bank for checkpoint episodes")
    p.add_argument("--config", required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--group", default="B")
    p.add_argument("--history", type=int, default=0)
    p.add_argument("--after", type=int, required=True, help="number of train episodes the bank has seen")
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser("run-batch", help="run an experiment file (experiments/batches/*.yaml) in sequence")
    p.add_argument("experiment")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--sandbox", help="override the registered sandbox backend (dev only)")
    p.add_argument("--client", help="required unless --dry-run (no default: a scripted batch would claim the run)",
                   choices=["model", "scripted-reference", "scripted-wrong", "scripted-noop"])
    p.add_argument("--dry-run", action="store_true", help="print the expanded schedule and exit")
    p.set_defaults(fn=cmd_run_batch)

    p = sub.add_parser("prereg-lock", help="write the protocol-lock manifest for sealed batches (SPEC §13); run at the repo root")
    p.add_argument("--batches", nargs="+", required=True, help="sealed batch YAMLs (must say `sealed: true`)")
    p.add_argument("--analysis", nargs="*", help="analysis scripts to lock")
    p.set_defaults(fn=cmd_prereg_lock)

    p = sub.add_parser("prereg-verify", help="check the working tree against the protocol-lock manifest; run at the repo root")
    p.add_argument("batch", nargs="?", help="also require this batch to be listed")
    p.set_defaults(fn=cmd_prereg_verify)

    p = sub.add_parser("measure-block", help="size a static batch's blocks from its source bank, without an episode")
    p.add_argument("experiment", help="a static batch file (groups P-all, P-all-ep, P-oracle)")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--history", type=int, default=0)
    p.add_argument("--reserve", type=int, default=16384, help="tokens kept free for the episode itself")
    p.set_defaults(fn=cmd_measure_block)

    p = sub.add_parser("analyze", help="compare groups for a batch (see python -m elh.analysis --help)")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("check-degenerate", help="list stored episodes holding a degenerate model reply (read-only; exit 1 if any)")
    p.add_argument("prefixes", nargs="*", help="batch-name prefixes (default: every batch)")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--since", help="only episodes started on/after this ISO date")
    p.set_defaults(fn=cmd_check_degenerate)

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
            p.add_argument("--group", default="A")
            p.add_argument("--split", default="dev")
            p.add_argument("--history", type=int, default=0)
            p.add_argument("--snapshot", help="snapshot id (required for checkpoint, sealed and retention splits)")
            p.add_argument("--checkpoint", type=int, help="checkpoint this episode belongs to")
            p.add_argument("--client", default="scripted-reference",
                           choices=["model", "scripted-reference", "scripted-wrong", "scripted-noop"])
        else:
            p.add_argument("episode_id")
        p.set_defaults(fn=fn)

    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["analyze"]:  # everything after it belongs to elh.analysis
        from elh.analysis import main as analysis_main

        return analysis_main(argv[1:])
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
