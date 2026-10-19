"""Train the airfoil graph network on the 436-run campaign (zoo model airfoil_cst_rans_gnn).

    python scripts/zoo/train_airfoil_gnn.py RUN_DIR OUT_DIR [--epochs 60]   # ~1.5 h on 4 CPU cores; resumable

RUN_DIR holds the run records of ``apps/aero_optimizer/tools/generate_dataset.py`` (``runs/*.json`` + ``*.npz``).
The split is :func:`pinneapple_design.aero.train.split` (by shape, the held-out shapes of the airfoil surrogates);
the recipe is :func:`pinneapple_design.aero.train.train_gnn` with a fixed epoch budget (the test error is logged,
never used to choose a checkpoint). Writes OUT_DIR/airfoil_gnn.pt, which ``scripts/zoo/build_assets.py`` turns into
the zoo model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinneapple_design.aero.train import load_runs, split, train_gnn  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--mp", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="save the training state here every 5 epochs and resume from it (default OUT/checkpoint)")
    a = ap.parse_args()
    recs = load_runs([str(a.run_dir)])
    tr, te, ref = split(recs)
    print(f"{len(recs)} usable runs: train {len(tr)}, test {len(te)}, reference {len(ref)}", flush=True)
    t0 = time.time()
    model, norm, hist, lay = train_gnn(tr, te, minutes=None, epochs=a.epochs, hidden=a.hidden, mp=a.mp, seed=a.seed,
                                       log=lambda s: print(s, flush=True),
                                       checkpoint_dir=str(a.checkpoint_dir or a.out / "checkpoint"))
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "norm": {k: v.tolist() for k, v in norm.items()},
                "node_in": int(model.head[0].in_features - 2 * a.hidden),
                "hidden": a.hidden, "mp": a.mp, "epochs": a.epochs, "seed": a.seed, "history": hist,
                "train_runs": len(tr), "train_shapes": len({r["shape_id"] for r in tr}),
                "train_seconds": time.time() - t0}, a.out / "airfoil_gnn.pt")
    print(json.dumps({"train_seconds": time.time() - t0, "last": hist[-1]}), flush=True)
