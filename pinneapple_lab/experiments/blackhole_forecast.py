"""Black-hole weather forecasting (Duarte, Nemmen & Navarro 2022; #399) as a lab experiment.

Wraps ``examples/black_hole_weather`` (train_forecaster.py and evaluate.py), so it runs from a source checkout of
the repository. The simulations come from ``examples/black_hole_weather/simulate.py`` (or the ``accretion_flow``
experiment) and are passed as ``runs_root``.
"""
from __future__ import annotations

import json
import os
import sys

from ..spec import Experiment, register

_EX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "examples",
                   "black_hole_weather")


def _scripts():
    if not os.path.isdir(_EX):
        raise RuntimeError("bh_forecast needs a source checkout (examples/black_hole_weather not found)")
    if _EX not in sys.path:
        sys.path.insert(0, _EX)
    import evaluate
    import train_forecaster
    return train_forecaster, evaluate


@register
class BlackHoleForecast(Experiment):
    name = "bh_forecast"
    version = "1"
    description = ("U-Net forecast of accretion-flow density (Duarte et al. 2022) trained on simulations, scored "
                   "by lead time against persistence, with the mass check, tendency correlation and optional "
                   "ray-traced GIF. Validation: the forecast must beat persistence on its first block.")
    tags = ["astrophysics", "forecasting", "surrogate", "reproduction"]
    params = {"runs_root": "", "train": "PL0SS0.1", "test": "PL0SS0.1", "split": "test", "stride": 2,
              "filters": 16, "residual": False, "loss": "multi", "lr": 2e-4, "clip": 1.0, "batch_size": 16,
              "epochs": 40, "max_minutes": 1e9, "blocks": 12, "starts": 8, "interstellar": False, "threads": 4}

    def run(self, ctx):
        p = ctx.params
        tf, ev = _scripts()
        model_dir = ctx.path("model")
        eval_dir = ctx.path("eval")
        trains = p["train"].split(",")
        ctx.input("setup", p)
        with ctx.stage("train"):
            args = ["--runs", p["runs_root"], "--train", *trains, "--out", model_dir, "--stride", str(p["stride"]),
                    "--filters", str(p["filters"]), "--epochs", str(p["epochs"]), "--batch-size", str(p["batch_size"]),
                    "--lr", str(p["lr"]), "--loss", p["loss"], "--clip", str(p["clip"]), "--threads", str(p["threads"]),
                    "--max-minutes", str(p["max_minutes"])]
            if p["residual"]:
                args.append("--residual")
            tf.main(args)
        hist = json.load(open(os.path.join(model_dir, "history.json")))
        for i, (a, b) in enumerate(zip(hist["train"], hist["val"], strict=True)):
            ctx.metric("train_loss", a, step=i)
            ctx.metric("val_loss", b, step=i)
        with ctx.stage("evaluate"):
            args = ["--runs", p["runs_root"], "--model", model_dir, "--test", p["test"], "--split", p["split"],
                    "--out", eval_dir, "--blocks", str(p["blocks"]), "--starts", str(p["starts"]), "--mass-projection"]
            if p["interstellar"]:
                args.append("--interstellar")
            ev.main(args)
        s = json.load(open(os.path.join(eval_dir, "summary.json")))
        sc = s["scores"]
        k = int(round(s["block_dt"] / s["frame_dt"]))
        ctx.metric("one_block_mae", s["one_block_mae"])
        ctx.metric("one_block_persistence", s["one_block_persistence"])
        ctx.metric("final_mae", sc["mae"][-1])
        ctx.metric("final_persistence", sc["persistence"][-1])
        ctx.metric("tendency_corr_after_first_block", float(sum(sc["tendency"][k:]) / max(1, len(sc["tendency"][k:]))))
        ctx.metric("horizon_beats_persistence", s["horizon_mean"]["beats_persistence"])
        ctx.metric("rollout_length", s["lead"][-1])
        viol = [v for v in s["mass_check_first_violation"] if v is not None]
        ctx.metric("mass_check_first_violation_median", sorted(viol)[len(viol) // 2] if viol else None)
        ctx.output("summary", s)
        ctx.check("beats_persistence_first_block", value=s["one_block_mae"], max=s["one_block_persistence"],
                  detail="a forecast worse than doing nothing is not useful")
        for f in ("skill.png", "slices.png", "interstellar_forecast.gif", "interstellar_frame.png"):
            fp = os.path.join(eval_dir, f)
            if os.path.exists(fp):
                ctx.figure_file(fp)
