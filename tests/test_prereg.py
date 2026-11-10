"""Protocol lock (elh/prereg.py): manifest round trip and sealed-batch enforcement in `run-batch`."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from elh import prereg
from elh.batch import load_experiment, run_batch
from elh.cli import main
from elh.store import Store

ROOT = Path(__file__).parent.parent
BATCH = "experiments/batches/sealed1.yaml"


def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                   capture_output=True)


def commit(repo, msg="c"):
    git(repo, "add", "-A")
    git(repo, "commit", "-m", msg)


def make_batch(repo, name="sealed1", **kw):
    doc = {"name": name, "base_config": "experiments/phase0-local.yaml", "family": "cpu_mem", "histories": 1,
           "groups": ["A"], "sealed": True,
           "phases": [{"eval": {"tasks": {"causes": ["nested_scan_join"], "themes": ["inventory"], "seeds": [2000, 2002]}}}]}
    doc.update(kw)
    (repo / "experiments/batches" / f"{name}.yaml").write_text(yaml.safe_dump(doc))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    (r / "experiments/batches").mkdir(parents=True)
    (r / "elh").mkdir()
    (r / "elh/mod.py").write_text("X = 1\n")
    (r / "analysis").mkdir()
    (r / "analysis/an.py").write_text("print('an')\n")
    (r / "uv.lock").write_text("lock\n")
    shutil.copy(ROOT / "experiments/phase0-local.yaml", r / "experiments/phase0-local.yaml")
    make_batch(r)
    subprocess.run(["git", "init", "-q", "-b", "main", str(r)], check=True)
    monkeypatch.chdir(r)
    return r


def lock_and_commit(repo, tag=True):
    assert main(["prereg-lock", "--batches", BATCH, "--analysis", "analysis/an.py"]) == 0
    commit(repo, "lock")
    if tag:
        git(repo, "tag", prereg.TAG)


def test_manifest_round_trip(repo):
    lock_and_commit(repo)
    m = json.loads((repo / prereg.MANIFEST).read_text())
    assert m["batches"] == [BATCH] and m["analysis"] == ["analysis/an.py"] and m["tag"] == prereg.TAG
    assert {"elh/mod.py", "uv.lock", BATCH, "experiments/phase0-local.yaml", "analysis/an.py"} <= set(m["files"])
    assert m["digest"] == prereg.combined_digest(m["files"]) and m["commit"] is None  # locked before the first commit
    res = prereg.verify(repo, BATCH)
    assert res.ok, res.problems
    assert res.digest == m["digest"]
    assert main(["prereg-verify", BATCH]) == 0


def test_manifest_records_the_commit(repo):
    commit(repo, "base")
    lock_and_commit(repo)
    head = subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert json.loads((repo / prereg.MANIFEST).read_text())["commit"] == head


def test_lock_rejects_unsealed_batch(repo):
    make_batch(repo, "plain", sealed=False)
    with pytest.raises(ValueError, match="sealed: true"):
        prereg.lock(repo, ["experiments/batches/plain.yaml"], [])


def test_modified_hashed_file_refuses(repo, capsys):
    lock_and_commit(repo)
    (repo / "elh/mod.py").write_text("X = 2\n")
    assert main(["prereg-verify"]) == 1
    out = capsys.readouterr().out
    assert "hash mismatch: elh/mod.py" in out and "uncommitted change: elh/mod.py" in out
    with pytest.raises(SystemExit, match="refused"):
        main(["run-batch", BATCH, "--client", "scripted-reference", "--sandbox", "local", "--runs-dir", str(repo / "runs")])
    assert not (repo / "runs").exists()
    commit(repo)  # committing the edit still fails: the hash no longer matches the manifest
    assert "hash mismatch: elh/mod.py" in prereg.verify(repo, BATCH).problems


def test_added_file_in_tree_refuses(repo):
    lock_and_commit(repo)
    (repo / "elh/new.py").write_text("Y = 1\n")
    assert "not in manifest (added): elh/new.py" in prereg.verify(repo).problems


def test_unlisted_sealed_batch_refuses(repo):
    lock_and_commit(repo)
    make_batch(repo, "sealed2")
    commit(repo)
    other = "experiments/batches/sealed2.yaml"
    assert any("not a locked batch" in p for p in prereg.verify(repo, other).problems)
    with pytest.raises(SystemExit):
        main(["run-batch", other, "--client", "scripted-reference", "--runs-dir", str(repo / "runs")])


def test_missing_manifest_and_missing_tag_refuse(repo):
    assert "no manifest" in prereg.verify(repo, BATCH).problems[0]
    lock_and_commit(repo, tag=False)
    assert any("tag prereg-v1 does not exist" in p for p in prereg.verify(repo, BATCH).problems)
    with pytest.raises(SystemExit):
        main(["run-batch", BATCH, "--client", "scripted-reference", "--runs-dir", str(repo / "runs")])


def test_head_must_contain_tag(repo):
    lock_and_commit(repo)
    assert prereg.verify(repo, BATCH).ok
    git(repo, "checkout", "-q", "--orphan", "other")
    commit(repo, "unrelated")  # same files, but the tag is not an ancestor
    assert any("does not contain tag" in p for p in prereg.verify(repo, BATCH).problems)


def test_manifest_changed_after_tag_refuses(repo):
    lock_and_commit(repo)
    (repo / "analysis/an.py").write_text("print('changed')\n")
    prereg.write_manifest(repo, prereg.lock(repo, [BATCH], ["analysis/an.py"]))
    commit(repo)
    assert any("differs from the one at tag" in p for p in prereg.verify(repo, BATCH).problems)


def test_dry_run_of_sealed_batch_reports_verification(repo, capsys):
    assert main(["run-batch", BATCH, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "prereg: FAILED" in out and "no manifest" in out and "nested_scan_join" in out
    lock_and_commit(repo)
    assert main(["run-batch", BATCH, "--dry-run"]) == 0
    assert "prereg: OK" in capsys.readouterr().out


def test_run_batch_function_refuses_sealed_without_digest(repo):
    exp, cfg = load_experiment(BATCH)
    assert exp.sealed
    with pytest.raises(RuntimeError, match="sealed"):
        run_batch(exp, cfg, Store(repo / "runs"), lambda t: None, sandbox_backend="local")


def test_sealed_batch_runs_when_verified(repo, capsys):
    lock_and_commit(repo)
    assert main(["run-batch", BATCH, "--client", "scripted-reference", "--sandbox", "local",
                 "--runs-dir", str(repo / "runs")]) == 0
    out = capsys.readouterr().out
    assert "prereg: OK" in out and out.count("passed=1") == 2


def test_unsealed_batch_ignores_the_lock(repo, capsys):
    make_batch(repo, "plain", sealed=False)  # no manifest, no tag, dirty tree
    assert main(["run-batch", "experiments/batches/plain.yaml", "--client", "scripted-reference",
                 "--sandbox", "local", "--runs-dir", str(repo / "runs")]) == 0
    out = capsys.readouterr().out
    assert "prereg" not in out and out.count("passed=1") == 2
