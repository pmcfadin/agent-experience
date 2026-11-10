"""Protocol lock for the sealed study (SPEC §3, §13).

`elh prereg-lock` writes `experiments/prereg/manifest.json`: the sha256 of every file that defines a
sealed batch's behaviour, plus one combined digest. The owner commits the manifest and tags that
commit `prereg-v1`. A batch whose YAML says `sealed: true` runs only if `verify` passes: the batch is
listed, every hashed file matches the manifest and is committed, the manifest is the one at the tag,
and HEAD contains the tag. Non-sealed batches never call into this module.

Covered: the listed batch YAMLs, their resolved `base_config`, the listed analysis scripts, the whole
`elh/` tree (episode behaviour, prompts, generators, judge), `uv.lock` and `pyproject.toml`
(dependency drift), `experiments/calibration/` (read by the house_lib grid) and the Hindsight
service pins (`services/hindsight/`). Whole-tree hashing is deliberate: a per-module whitelist
would silently miss a transitively imported file. Tests, docs and reports are not covered.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TAG = "prereg-v1"
MANIFEST = Path("experiments/prereg/manifest.json")
FORMAT = 1
TREES = ["elh", "experiments/calibration"]
FIXED = ["uv.lock", "pyproject.toml", "services/hindsight/pyproject.toml", "services/hindsight/uv.lock",
         "services/hindsight/models.lock.json", "services/hindsight/serve.sh"]
_SKIP = {"__pycache__", ".pytest_cache"}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def combined_digest(files: dict[str, str]) -> str:
    """Digest over sorted `path NUL sha256 LF` lines: any added, removed, renamed or edited file changes it."""
    blob = "".join(f"{p}\0{h}\n" for p, h in sorted(files.items()))
    return hashlib.sha256(blob.encode()).hexdigest()


def _rel(root: Path, p: str | Path) -> str:
    """Repo-relative path of `p` (relative paths are taken from the current directory)."""
    return str(Path(p).resolve().relative_to(root.resolve()))


def collect(root: Path, extra: list[str]) -> dict[str, str]:
    """{repo-relative path: sha256} for the trees, the fixed files and `extra` (repo-relative)."""
    paths: set[str] = set(extra)
    for t in TREES:
        for p in (root / t).rglob("*"):
            if p.is_file() and not (_SKIP & set(p.parts)) and p.suffix != ".pyc":
                paths.add(str(p.relative_to(root)))
    paths |= {f for f in FIXED if (root / f).is_file()}
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sorted(paths)}


def base_config_of(batch: Path) -> Path:
    """The batch's `base_config`, resolved as `load_experiment` does."""
    from elh.batch import _resolve

    return _resolve(yaml.safe_load(batch.read_text())["base_config"], batch.parent)


def lock(root: Path, batches: list[str], analysis: list[str]) -> dict:
    """Build the manifest dict (the caller writes it). Every batch must declare `sealed: true`."""
    root = Path(root).resolve()
    rels, extra = [], []
    for b in batches:
        if not yaml.safe_load(Path(b).read_text()).get("sealed"):
            raise ValueError(f"{b} does not declare `sealed: true`")
        rels.append(_rel(root, b))
        extra += [rels[-1], _rel(root, base_config_of(Path(b)))]
    ana = sorted(_rel(root, a) for a in analysis)
    files = collect(root, extra + ana)
    head = _git(root, "rev-parse", "--verify", "-q", "HEAD")
    return {"format": FORMAT, "tag": TAG, "commit": None if head.returncode else head.stdout.strip(),
            "digest": combined_digest(files), "batches": sorted(rels), "analysis": ana, "files": files}


def write_manifest(root: Path, manifest: dict) -> Path:
    path = Path(root) / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return path


@dataclass
class Result:
    problems: list[str] = field(default_factory=list)
    digest: str | None = None

    @property
    def ok(self) -> bool:
        return not self.problems


def verify(root: Path, batch: str | Path | None = None) -> Result:
    """Check the working tree against the manifest. `batch` (if given) must be listed in it."""
    root = Path(root).resolve()
    res = Result()
    mp = root / MANIFEST
    if not mp.is_file():
        res.problems.append(f"no manifest at {MANIFEST}; run `elh prereg-lock`")
        return res
    raw = mp.read_text()
    m = json.loads(raw)
    res.digest = m["digest"]
    if m.get("format") != FORMAT:
        res.problems.append(f"manifest format {m.get('format')} != {FORMAT}")
    if combined_digest(m["files"]) != m["digest"]:
        res.problems.append("manifest digest does not match its own file list (edited by hand?)")
    if batch is not None:
        rel = _rel(root, batch)
        if rel not in m["batches"]:
            res.problems.append(f"{rel} is not a locked batch (listed: {', '.join(m['batches']) or 'none'})")
    now = collect(root, [p for p in m["files"] if (root / p).is_file()])
    for p in sorted(set(m["files"]) | set(now)):
        if p not in now:
            res.problems.append(f"missing: {p}")
        elif p not in m["files"]:
            res.problems.append(f"not in manifest (added): {p}")
        elif now[p] != m["files"][p]:
            res.problems.append(f"hash mismatch: {p}")
    if _git(root, "rev-parse", "--git-dir").returncode:
        res.problems.append("not a git repository")
        return res
    dirty = _git(root, "status", "--porcelain", "--", *sorted(m["files"]), str(MANIFEST)).stdout.splitlines()
    res.problems += [f"uncommitted change: {line[3:]}" for line in dirty]
    tag = m["tag"]
    if _git(root, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}").returncode:
        res.problems.append(f"tag {tag} does not exist yet; commit the manifest and tag that commit first")
    else:
        if _git(root, "merge-base", "--is-ancestor", tag, "HEAD").returncode:
            res.problems.append(f"HEAD does not contain tag {tag}")
        at_tag = _git(root, "show", f"{tag}:{MANIFEST}")
        if at_tag.returncode or at_tag.stdout != raw:
            res.problems.append(f"{MANIFEST} differs from the one at tag {tag}")
    return res


def format_result(res: Result) -> str:
    if res.ok:
        return f"prereg: OK (manifest digest {res.digest})"
    return "prereg: FAILED\n" + "\n".join(f"  {p}" for p in res.problems)
