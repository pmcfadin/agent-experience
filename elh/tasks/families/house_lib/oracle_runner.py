"""Hidden oracle for house_lib tasks. Runs inside the judge sandbox; stdlib only.

Usage: python3 oracle.py --workspace DIR --oracle DIR
       python3 oracle.py --harness --oracle DIR WS CASE.json [OFF_FLAG]   (one export run)

Grades a copy of the workspace whose `acmecore/` is replaced by the oracle's own copy (design §4.3;
allowed because `acmecore` is protected), with the clock frozen and, for flag lessons, one flag off.
Prints exactly one JSON verdict line on stdout. Checks: `public_tests`, `feature` (lesson-neutral
columns on hidden inputs) and one `lesson:<id>` per required lesson.
"""

import argparse
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile


def harness(oracle_dir, ws, case_path, off_flag):
    """Run export() once on a case; print the store/audit log as JSON."""
    with open(os.path.join(oracle_dir, "params.json")) as f:
        P = json.load(f)
    sys.path.insert(0, ws)
    out = {"log": [], "error": None}
    try:
        from acmecore import clock, flags, store

        clock._FROZEN.update(P["frozen_clock"])
        if off_flag:
            flags._OFF.add(off_flag)
        store.reset()
        mod = importlib.import_module(P["spec"]["feature_mod"])
        with open(case_path) as f:
            records = json.load(f)
        mod.export(records)
        out["log"] = store._LOG
    except BaseException as exc:  # noqa: BLE001 - any failure of agent code is a verdict, not a crash
        out["error"] = f"{type(exc).__name__}: {exc}"[:500]
    print(json.dumps(out, default=repr))


def run_case(oracle_dir, ws, case_path, off_flag, timeout_s):
    cmd = [sys.executable, os.path.abspath(__file__), "--harness", "--oracle", oracle_dir, ws, case_path]
    if off_flag:
        cmd.append(off_flag)
    try:
        proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=timeout_s, start_new_session=True)
    except subprocess.TimeoutExpired:
        return {"log": [], "error": "timeout"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"log": [], "error": f"no harness output (exit {proc.returncode}): {proc.stderr[-500:]}"}


def same(a, b):
    """Equal including JSON type (1 != 1.0 != "1")."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def expected_value(les, rec, spec, frozen, acme):
    t, c = les["template"], les["choice"]
    if t == "partner_format":
        return acme["formats"].format_date(rec["due"], c)
    if t == "unit_convention":
        return acme["units"].convert(rec["amount_cents"], c)
    if t == "region_redaction":
        return getattr(acme["pii"], c)(rec["email"])
    if t == "required_wrapper":
        return frozen[c]
    if t == "event_fields":
        src = {"key": rec[spec["key"]], "group": rec[spec["group"]], "module": spec["module"], "due": rec["due"]}
        return acme["ids"].idem(*[src[p] for p in c.split("+")])
    raise KeyError(t)


FIELD = {"partner_format": "due_date", "unit_convention": "amount", "region_redaction": "contact",
         "required_wrapper": "generated_at", "event_fields": "idem_key"}


# What the agent and the rule writer are told about each check (the verdict's `check_text`, read by
# elh.memory.hindsight.check_phrase). It says what was observed to be wrong, never the expected
# value or the convention, and never the lesson id (which names the template and context).
FEATURE_TEXT = {"label": "feature", "passed": "rows correct on hidden inputs",
                "failed": "wrong, missing or extra rows on hidden inputs (row keys, groups or which records qualify)"}


def lesson_text(les):
    t = les["template"]
    if t in FIELD:
        f = FIELD[t]
        return {"label": f"{f} column", "passed": f"`{f}` values correct",
                "failed": f"wrong or missing `{f}` values in the written rows"}
    if t == "call_order":
        return {"label": "audit trail", "passed": "audit records in the expected place",
                "failed": "audit records missing or in the wrong place relative to the store writes"}
    if t == "feature_flag":
        return {"label": "rollout gate", "passed": "wrote nothing while the module's rollout was switched off",
                "failed": "wrote rows while the module's rollout was switched off"}
    raise KeyError(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--harness", action="store_true")
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args()
    if args.harness:
        harness(args.oracle, args.paths[0], args.paths[1], args.paths[2] if len(args.paths) > 2 else None)
        return

    sys.path.insert(0, args.oracle)
    import records as R
    acme = {m: importlib.import_module(f"acmecore.{m}") for m in ("formats", "units", "pii", "ids")}
    with open(os.path.join(args.oracle, "params.json")) as f:
        P = json.load(f)
    with open(os.path.join(args.oracle, "fields.json")) as f:
        F = json.load(f)
    spec, lessons, frozen = P["spec"], P["lessons"], P["frozen_clock"]
    key, table = spec["key"], spec["table"]

    tmp = tempfile.mkdtemp(prefix="oracle-")
    ws = os.path.join(tmp, "ws")
    shutil.copytree(args.workspace, ws, symlinks=True)
    shutil.rmtree(os.path.join(ws, "acmecore"), ignore_errors=True)
    shutil.copytree(os.path.join(args.oracle, "acmecore"), os.path.join(ws, "acmecore"))
    verdict = {"passed": False, "checks": {}, "cases": []}

    # 1. Public tests still pass (against the oracle's acmecore).
    try:
        pub = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
                             cwd=ws, capture_output=True, text=True, timeout=P["public_test_timeout_s"])
        public_ok, public_tail = pub.returncode == 0, pub.stderr[-1500:]
    except subprocess.TimeoutExpired:
        public_ok, public_tail = False, "timeout"
    verdict["checks"]["public_tests"] = public_ok
    if not public_ok:
        verdict["public_tests_tail"] = public_tail

    # 2. Hidden inputs: edge cases, small, medium. One run per case, plus a flag-off run per flag lesson.
    rng = random.Random(P["hidden_seed"])
    sizes = [0, 1] + [P["sizes"]["small"]] * P["k_small"] + [P["sizes"]["medium"]] * P["k_medium"]
    flag_lessons = [les for les in lessons if les["template"] == "feature_flag"]
    feature_ok = True
    lesson_ok = {les["id"]: True for les in lessons}
    for i, n in enumerate(sizes):
        recs = R.gen_records(random.Random(rng.randrange(2**32)), n, F, spec["region"], spec["min_cents"])
        path = os.path.join(tmp, f"case{i}.json")
        with open(path, "w") as f:
            json.dump(recs, f)
        want = [r for r in recs if R.qualifies(r, spec["region"], spec["min_cents"])]
        res = run_case(args.oracle, ws, path, None, P["case_timeout_s"])
        log = res["log"]
        writes = [(j, e[2]) for j, e in enumerate(log) if e[0] == "write" and e[1] == table]
        got_neutral = [{key: row.get(key), spec["group"]: row.get(spec["group"])} for _, row in writes]
        want_neutral = [{key: r[key], spec["group"]: r[spec["group"]]} for r in want]
        ok = res["error"] is None and same(got_neutral, want_neutral)
        verdict["cases"].append({"n": n, "rows": len(want), "ok": ok, "error": res["error"]})
        feature_ok &= ok
        by_key = {row.get(key): (j, row) for j, row in writes}
        for les in lessons:
            t, lid = les["template"], les["id"]
            if res["error"] is not None or any(r[key] not in by_key for r in want):
                lesson_ok[lid] = False
            elif t in FIELD:
                lesson_ok[lid] &= all(FIELD[t] in by_key[r[key]][1]
                                      and same(by_key[r[key]][1][FIELD[t]], expected_value(les, r, spec, frozen, acme))
                                      for r in want)
            elif t == "call_order":
                step = -1 if les["choice"] == "before" else 1
                lesson_ok[lid] &= all(0 <= by_key[r[key]][0] + step < len(log)
                                      and log[by_key[r[key]][0] + step] == ["audit", table, r[key]] for r in want)
        for les in flag_lessons:
            if want:
                off = run_case(args.oracle, ws, path, les["choice"], P["case_timeout_s"])
                off_writes = [e for e in off["log"] if e[0] == "write" and e[1] == table]
                lesson_ok[les["id"]] &= off["error"] is None and not off_writes
    verdict["checks"]["feature"] = feature_ok
    verdict["check_text"] = {"feature": FEATURE_TEXT}
    for lid, ok in lesson_ok.items():
        verdict["checks"][f"lesson:{lid}"] = ok
    for les in lessons:
        verdict["check_text"][f"lesson:{les['id']}"] = lesson_text(les)
    verdict["passed"] = all(verdict["checks"].values())
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
