"""Train the black-hole weather forecaster (U-Net of Duarte, Nemmen & Navarro 2022) on simulations from simulate.py.

Two set-ups, as in the paper:

* one-sim: a single run, split in time 70 / 10 / 20 (train / validation / test);
* multi-sim: several runs for training, a withheld run with different physics for testing
  (the paper withholds PL0SS3, alpha = 0.3; here PL0SS0.3).

    python examples/black_hole_weather/train_forecaster.py --runs data/bh --train PL0SS0.1 --out runs/bh_one
    python examples/black_hole_weather/train_forecaster.py --runs data/bh --train PL0SS0.1 PL0ST0.01 PL0.2SS0.1 \\
        --out runs/bh_multi
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import load_run  # noqa: E402

from pinneapple_physics.blackhole.forecast import (  # noqa: E402
    DensityCodec,
    TrainConfig,
    make_blocks,
    train_forecaster,
)


def split_one(n: int):
    a, b = int(0.7 * n), int(0.8 * n)
    return slice(0, a), slice(a, b), slice(b, n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2, help="use every n-th saved frame (10 M -> 20 M)")
    ap.add_argument("--r-cells", type=int, default=96)
    ap.add_argument("--theta-trim", type=int, default=8)
    ap.add_argument("--filters", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--residual", action="store_true", help="predict the change from the last frame")
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)

    runs = {name: load_run(os.path.join(a.runs, name)) for name in a.train}
    rho = {k: v["frames"][::a.stride, 0] for k, v in runs.items()}
    one = len(a.train) == 1
    if one:
        tr, va, _ = split_one(len(rho[a.train[0]]))
        train_frames = {a.train[0]: rho[a.train[0]][tr]}
        val_frames = {a.train[0]: rho[a.train[0]][va]}
    else:                                    # last 10 % of each run for validation
        train_frames = {k: v[:int(0.9 * len(v))] for k, v in rho.items()}
        val_frames = {k: v[int(0.9 * len(v)):] for k, v in rho.items()}
    codec = DensityCodec.fit(np.concatenate(list(train_frames.values())), a.r_cells, a.theta_trim)
    X, Y = zip(*[make_blocks(codec.encode(v)) for v in train_frames.values()], strict=True)
    Xv, Yv = zip(*[make_blocks(codec.encode(v)) for v in val_frames.values()], strict=True)
    X, Y, Xv, Yv = (np.concatenate(z) for z in (X, Y, Xv, Yv))
    print(f"train {X.shape}  val {Xv.shape}  codec {codec}", flush=True)
    cfg = TrainConfig(filters=a.filters, epochs=a.epochs, batch_size=a.batch_size, alpha=a.alpha, lr=a.lr, residual=a.residual,
                      max_minutes=a.max_minutes)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "setup.json"), "w") as f:
        json.dump({"train": a.train, "stride": a.stride, "frame_dt": float(np.diff(runs[a.train[0]]["t"][::a.stride]).mean()),
                   "one_sim": one, "n_train": int(len(X)), "n_val": int(len(Xv))}, f, indent=1)
    train_forecaster(X, Y, Xv, Yv, cfg, codec, a.out, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
