"""Write a house_lib world's calibration batch files (`calibration.batch_texts`) to experiments/batches.
w1's exp21 / exp24 files were produced the same way; w2's are exp40-calib-<world> and exp41-learn-<world>.
Run from the repository root:

    uv run python analysis/calib_batches.py [WORLD ...]     # default: the six w2 worlds
"""

from __future__ import annotations

import sys
from pathlib import Path

from elh.tasks.base import get_family
from elh.tasks.families.house_lib.calibration import batch_prefixes, batch_texts

W2_WORLDS = [f"L{n}-c{c}-w2" for n in (4, 16, 64) for c in (0, 50)]
OUT = Path("experiments/batches")


def main() -> None:
    for world in sys.argv[1:] or W2_WORLDS:
        for prefix, text in zip(batch_prefixes(world), batch_texts(get_family(f"house_lib/{world}"))):
            (OUT / f"{prefix}-{world}.yaml").write_text(text)
            print(f"wrote {OUT / f'{prefix}-{world}.yaml'}")


if __name__ == "__main__":
    main()
