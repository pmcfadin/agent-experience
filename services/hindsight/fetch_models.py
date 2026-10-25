"""Download Hindsight's embedder and reranker at pinned Hugging Face revisions (SPEC §10).

Files land in models/<name>/ and are verified against models.lock.json. With --lock, missing
hashes are filled in (first fetch only); a hash that changes is always an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
LOCK = HERE / "models.lock.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true", help="record hashes that are not yet in the lock file")
    args = ap.parse_args()
    lock = json.loads(LOCK.read_text())
    changed = False
    for name, m in lock["models"].items():
        dest = HERE / "models" / name
        for rel in list(m["files"]):
            path = Path(hf_hub_download(m["repo"], rel, revision=m["revision"], local_dir=dest))
            digest = sha256(path)
            want = m["files"][rel]
            if want and want != digest:
                print(f"{name}/{rel}: sha256 {digest} != locked {want}", file=sys.stderr)
                return 1
            if not want:
                if not args.lock:
                    print(f"{name}/{rel}: no locked hash; rerun with --lock", file=sys.stderr)
                    return 1
                m["files"][rel], changed = digest, True
            print(f"ok {name}/{rel} {digest[:12]}")
    if changed:
        LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
