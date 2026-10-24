"""Hidden oracle for cpu_mem tasks. Runs inside the judge sandbox; stdlib only.

Usage: python3 oracle.py --workspace DIR --oracle DIR
       python3 oracle.py --reference --oracle DIR IN OUT   (reference job, used for timing)

Prints exactly one JSON verdict line on stdout.
"""

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import time


def load_cause(oracle_dir):
    spec = importlib.util.spec_from_file_location("cause", os.path.join(oracle_dir, "cause.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def canon(obj):
    return json.loads(json.dumps(obj, sort_keys=True))


def run_job(cmd, cwd, in_path, out_path, timeout_s):
    if os.path.exists(out_path):
        os.remove(out_path)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd + [in_path, out_path], cwd=cwd, capture_output=True, text=True,
            timeout=timeout_s, start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_s": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return {"status": "error", "wall_s": wall, "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-2000:]}
    try:
        with open(out_path) as f:
            out = json.load(f)
    except Exception as exc:  # noqa: BLE001 - any unreadable output is a failure
        return {"status": "bad_output", "wall_s": wall, "error": repr(exc)}
    return {"status": "ok", "wall_s": wall, "output": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args()

    cause = load_cause(args.oracle)
    with open(os.path.join(args.oracle, "fields.json")) as f:
        F = json.load(f)

    if args.reference:
        in_path, out_path = args.paths
        with open(in_path) as f:
            data = json.load(f)
        with open(out_path, "w") as f:
            json.dump(cause.reference(data, F), f)
        return

    with open(os.path.join(args.oracle, "params.json")) as f:
        P = json.load(f)

    ws = args.workspace
    tmp = tempfile.mkdtemp(prefix="oracle-")
    verdict = {"passed": False, "checks": {}, "cases": [], "quality": {}}
    agent_cmd = [sys.executable, "run.py"]
    ref_cmd = [sys.executable, os.path.abspath(__file__), "--reference", "--oracle", args.oracle]

    # 1. Public tests still pass.
    try:
        pub = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
                             cwd=ws, capture_output=True, text=True, timeout=P["public_test_timeout_s"])
        public_ok = pub.returncode == 0
        public_tail = pub.stderr[-1500:]
    except subprocess.TimeoutExpired:
        public_ok, public_tail = False, "timeout"
    verdict["checks"]["public_tests"] = public_ok
    if not public_ok:
        verdict["public_tests_tail"] = public_tail

    # 2. Correctness on hidden inputs (edge cases + small + medium).
    rng = random.Random(P["hidden_seed"])
    sizes = [0, 1] + [cause.SIZES["small"]] * P["k_small"] + [cause.SIZES["medium"]] * P["k_medium"]
    correct = True
    for i, n in enumerate(sizes):
        data = cause.gen_input(random.Random(rng.randrange(2**32)), n, F)
        in_path = os.path.join(tmp, f"case{i}.json")
        out_path = os.path.join(tmp, f"case{i}.out.json")
        with open(in_path, "w") as f:
            json.dump(data, f)
        expected = canon(cause.reference(data, F))
        res = run_job(agent_cmd, ws, in_path, out_path, P["correctness_timeout_s"])
        ok = res["status"] == "ok" and canon(res.pop("output")) == expected
        res.pop("output", None)
        verdict["cases"].append({"n": n, "ok": ok, **res})
        if not ok:
            correct = False
            break
    verdict["checks"]["correctness"] = correct

    # 3. Performance on a production-sized hidden input, relative to the reference job.
    perf_ok = False
    if correct:
        data = cause.gen_input(random.Random(rng.randrange(2**32)), cause.SIZES["large"], F)
        in_path = os.path.join(tmp, "large.json")
        with open(in_path, "w") as f:
            json.dump(data, f)
        ref_times = []
        ref_out = None
        for _ in range(2):
            r = run_job(ref_cmd, tmp, in_path, os.path.join(tmp, "large.ref.json"), P["reference_timeout_s"])
            if r["status"] != "ok":
                verdict["error"] = f"reference job failed: {r}"
                print(json.dumps(verdict))
                return
            ref_times.append(r["wall_s"])
            ref_out = r["output"]
        ref_t = min(ref_times)
        budget = max(P["perf_floor_s"], P["perf_factor"] * ref_t)
        res = run_job(agent_cmd, ws, in_path, os.path.join(tmp, "large.out.json"), budget)
        large_correct = res["status"] == "ok" and canon(res.pop("output")) == canon(ref_out)
        perf_ok = res["status"] == "ok" and large_correct
        verdict["quality"] = {
            "reference_s": round(ref_t, 4),
            "budget_s": round(budget, 4),
            "agent_s": round(res["wall_s"], 4),
            "agent_over_reference": round(res["wall_s"] / ref_t, 3) if ref_t > 0 else None,
            "large_status": res["status"],
            "large_correct": large_correct,
        }
        if res["status"] == "ok" and not large_correct:
            verdict["checks"]["correctness"] = False
        verdict["checks"]["performance"] = perf_ok
    else:
        verdict["performance_skipped"] = "correctness failed"

    verdict["passed"] = all(verdict["checks"].values())
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
