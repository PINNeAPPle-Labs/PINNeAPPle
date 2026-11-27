"""The model family of the black-hole weather forecast, its ensemble, and the verdict (issue #399, item 4).

    python examples/black_hole_weather/family.py --runs runs/bh_data --out runs/bh_family \\
        --train bh_n128_s0 --val bh_n128_s4 --test bh_n128_s5

Trains the three members of the family on one simulated trajectory, evaluates every member, both aggregations and
the baselines on the held-out one through ``skill.evaluate_skill`` (same protocol, same moving-block bootstrap as the
numbers in the docs), and writes:

* ``report.json``: every number, including the paired ensemble-against-best-single comparison;
* ``<out>/members/<name>.pt``: the trained weights (kept out of the repository, like the other checkpoints);
* with ``--assets``: ``docs/assets/blackhole/family_skill.png`` and ``family_forecast.gif``.

The data are frames of the library's own runs (``simulate.py``), so the script needs one train, one validation and one
test trajectory; the frame spacing, the window crop and the codec come from the run itself.
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

from pinneapple_physics.blackhole import family as fam  # noqa: E402
from pinneapple_physics.blackhole.convergence import R_WINDOW, TH_WINDOW, cell_volumes  # noqa: E402
from pinneapple_physics.blackhole.figures import forecast_gif, plot_family  # noqa: E402
from pinneapple_physics.blackhole.forecast import DensityCodec, make_blocks  # noqa: E402


def window(grid) -> tuple[slice, slice]:
    """Rows and columns of the forecast window (r < 126.5 M, theta in [pi/8, 7 pi/8]), as in the rest of the study."""
    r, th = np.asarray(grid["r"]), np.asarray(grid["theta"])
    return slice(0, int((r < R_WINDOW).sum())), slice(int((th < TH_WINDOW).sum()),
                                                      len(th) - int((th < TH_WINDOW).sum()))


def trajectory(folder: str, t_min: float = 800.0) -> dict:
    """Log10 density of one run in the forecast window, frames at the study's spacing: (T, nr, nt) and the volumes."""
    run = load_run(folder)
    t = np.asarray(run["t"], float)
    keep = t >= t_min
    grid = run["grid"]
    ir, ith = window(grid)
    rho = np.asarray(run["frames"], np.float32)[keep][:, 0]
    return {"log10_rho": np.log10(np.clip(rho[:, ir, ith], 1e-30, None)), "t": t[keep], "grid": grid,
            "volume": cell_volumes(grid)[ir, ith]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="folder of simulate.py runs")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", nargs="+", default=list(fam.MEMBERS))
    ap.add_argument("--epochs", type=int, default=0, help="0: the member's own default (see family.DEFAULT_EPOCHS)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0, help="0: the member's own default (the U-Net needs the small one)")
    ap.add_argument("--t-min", type=float, default=800.0, help="drop the initial relaxation before this time")
    ap.add_argument("--blocks", type=int, default=10, help="rollout length in blocks of 5 frames (1000 M at 10)")
    ap.add_argument("--start-every", type=int, default=5)
    ap.add_argument("--dmd-ranks", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--assets", default="", help="folder for the figure and the GIF (e.g. docs/assets/blackhole)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.join(a.out, "members"), exist_ok=True)

    tr = [trajectory(os.path.join(a.runs, n), a.t_min) for n in a.train]
    va = [trajectory(os.path.join(a.runs, n), a.t_min) for n in a.val]
    te = [trajectory(os.path.join(a.runs, n), a.t_min) for n in a.test]
    grid, vol = tr[0]["grid"], tr[0]["volume"]
    frame_dt = float(np.diff(tr[0]["t"]).mean())
    lg = np.concatenate([x["log10_rho"] for x in tr])
    codec = DensityCodec(float(lg.min()), float(lg.max()), lg.shape[1], 0)
    print(f"window {lg.shape[1]} x {lg.shape[2]}, frame spacing {frame_dt:.1f} M, frames "
          f"train {[len(x['t']) for x in tr]} val {[len(x['t']) for x in va]} test {[len(x['t']) for x in te]}",
          flush=True)

    def enc(traj):
        return ((traj["log10_rho"] - codec.log_min) / (codec.log_max - codec.log_min)).astype(np.float32)

    X, Y = zip(*[make_blocks(enc(x), fam.K) for x in tr], strict=True)
    Xv, Yv = zip(*[make_blocks(enc(x), fam.K, stride=4) for x in va], strict=True)
    X, Y, Xv, Yv = (np.concatenate(z) for z in (X, Y, Xv, Yv))
    print(f"blocks: train {X.shape[0]}, val {Xv.shape[0]}", flush=True)

    members, histories = {}, {}
    for name in a.members:
        model = fam.make_member(name)
        print(f"[{name}] {sum(p.numel() for p in model.parameters())} parameters", flush=True)
        lr = a.lr or fam.DEFAULT_LR.get(name, 1e-3)
        epochs = a.epochs or fam.DEFAULT_EPOCHS.get(name, 12)
        print(f"[{name}] {epochs} epochs at lr {lr:g}", flush=True)
        histories[name] = fam.train_member(model, X, Y, Xv, Yv,
                                           fam.MemberTrainConfig(epochs=epochs, batch_size=a.batch_size, lr=lr),
                                           log=lambda s: print(s, flush=True))
        torch.save({"state_dict": model.state_dict()}, os.path.join(a.out, "members", f"{name}.pt"))
        members[name] = model

    report = fam.evaluate_family(members, codec, [x["log10_rho"] for x in tr], [x["log10_rho"] for x in va],
                                 [x["log10_rho"] for x in te], frame_dt=frame_dt, cell_volume=vol, blocks=a.blocks,
                                 start_every=a.start_every, dmd_ranks=a.dmd_ranks, histories=histories)
    report["window"] = {"r_cells": int(lg.shape[1]), "theta_cells": int(lg.shape[2]), "r_max": float(R_WINDOW),
                        "frame_dt": frame_dt, "t_min": a.t_min}
    report["data"] = {"train": a.train, "val": a.val, "test": a.test}
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)

    print("\n" + json.dumps(report["verdict"], indent=1), flush=True)
    for name, m in report["skill"]["models"].items():
        ss = m["vs_persistence"]["skill_score"][-1]
        print(f"  {name:20s} MAE +{report['leads_M'][-1]:.0f} M {report['mean_mae'][name][-1]:.4f}  "
              f"skill {ss[0]:+.3f} [{ss[1]:+.3f}, {ss[2]:+.3f}]  horizon {m['skill_horizon_M']:.0f} M", flush=True)

    if a.assets:
        os.makedirs(a.assets, exist_ok=True)
        plot_family(report, os.path.join(a.assets, "family_skill.png"),
                    title="Black-hole weather: the model family, its ensemble and the best single model")
        make_animation(members, report, codec, grid, te[0], os.path.join(a.assets, "family_forecast.gif"),
                       blocks=a.blocks)
    return report


def make_animation(members: dict, report: dict, codec: DensityCodec, grid: dict, test: dict, path: str, *,
                   blocks: int, n_frames: int = 24) -> str:
    """One rollout from the middle of the test trajectory of the equal-weight ensemble, drawn frame by frame with the
    trust bar of the same model (its anomaly correlation by lead and the lead where it falls below 0.6) and the curve
    of the best single member next to it, so the viewer can see which frames the combination is worth."""
    x = ((test["log10_rho"] - codec.log_min) / (codec.log_max - codec.log_min)).astype(np.float32)
    start = len(x) // 2
    q = fam.forecast_queries(x, blocks, starts=[start])[0]
    truth = x[start + 1:start + 1 + q["blocks"] * fam.K]
    pred = fam.combine([fam.rollout(m, q["x0"], q["blocks"]) for m in members.values()],
                       np.full(len(members), 1.0 / len(members)))
    idx = np.unique(np.linspace(0, len(truth) - 1, n_frames).round().astype(int))
    dt = float(np.diff(test["t"]).mean())
    key = "ensemble-equal"
    return forecast_gif(path, grid, codec, truth[idx], pred[idx], (idx + 1) * dt,
                        acc_by_lead=report["skill"]["models"][key]["acc_by_lead"],
                        horizon_M=report["skill"]["models"][key]["acc_horizon_M"],
                        extra=report["skill"]["models"][report["verdict"]["best_single"]]["acc_by_lead"],
                        extra_name=report["verdict"]["best_single"],
                        title="Black-hole weather: equal-weight ensemble forecast of the accretion flow",
                        model_name="ensemble")


if __name__ == "__main__":
    main()
