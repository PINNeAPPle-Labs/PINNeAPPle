"""Confined-cylinder wake dataset of the zoo (``cylinder_wake_lbm``): solver verification and the 11 runs.

    python scripts/zoo/cylinder_data.py verify --out DIR      # channel profile + D = 20/30/40 study -> DIR/cylinder_verify.json
    python scripts/zoo/cylinder_data.py generate --out DIR    # DIR/cylinder_wake_lbm/<split>[-NN].npz + meta.json

Command-line front end of ``pinneapple_simulation.numerical_solvers.cylinder_wake``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinneapple_simulation.numerical_solvers.cylinder_wake import generate, verify  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["verify", "generate"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.command == "verify":
        res = verify()
        (a.out / "cylinder_verify.json").write_text(json.dumps(res, indent=1) + "\n")
    else:
        res = generate(a.out, a.workers)
        res = {k: res[k] for k in ("files", "rows", "runs", "wall_seconds")}
    print(json.dumps(res, indent=1))
