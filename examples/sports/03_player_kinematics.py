"""Player kinematics and load: smoothing with known truth, metabolic power on real tracking with its processing
uncertainty, and 1 s / 2 s trajectory prediction with and without physical limits at equal data.

1. simulated runs (25 and 10 Hz, 5 and 20 cm noise): speed, acceleration and metabolic-power error of raw
   differences, Savitzky-Golay (0.5 s, 1 s) and the Kalman-RTS smoother with its noise fitted by maximum likelihood;
2. Metrica match 1: noise level fitted on the data, plausibility (speed > 12.5 m/s, |a| > 10 m/s^2) and metabolic
   power per player per method;
3. prediction of every outfield player's position 1 s and 2 s ahead (train match 1, test match 2): constant
   velocity, decaying-acceleration physics, boosted trees on the past second, trees on the constant-velocity
   residual, the same projected onto the reachable set; learning curves.

    python examples/sports/03_player_kinematics.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # before numpy / torch / sklearn: small models, no oversubscription

import numpy as np  # noqa: E402

from pinneapple_core.jsonable import to_jsonable  # noqa: E402
from pinneapple_systems.sports.football import data as D  # noqa: E402
from pinneapple_systems.sports.football import kin_study as S  # noqa: E402
from pinneapple_systems.sports.football import plots  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(smoke: bool = False) -> dict:
    res: dict = {"synthetic_smoothing": S.synthetic_smoothing(n=5 if smoke else 20)}
    m1, m2 = D.metrica_match(1), D.metrica_match(2)
    load = S.match_load(m1, max_frames=15000 if smoke else None)
    res["match_load"] = {"noise_fit": load["noise_fit"], "methods": {}}
    for name, r in load["methods"].items():
        mp = np.array(r["mean_mp_per_player"], float)
        res["match_load"]["methods"][name] = {k: v for k, v in r.items() if not isinstance(v, list)}
        res["match_load"]["methods"][name]["mean_mp_team"] = float(np.nanmean(mp))
        res["match_load"]["methods"][name]["mean_mp_per_player"] = mp.tolist()
        res["match_load"]["methods"][name]["median_mp_team"] = float(np.nanmean(r["median_mp_per_player"]))
        res["match_load"]["methods"][name]["gated_mean_mp_team"] = float(np.nanmean(r["gated_mean_mp_per_player"]))
    ref = np.array(load["methods"]["kalman_rts"]["mean_mp_per_player"], float)
    for name, r in load["methods"].items():
        mp = np.array(r["mean_mp_per_player"], float)
        res["match_load"]["methods"][name]["rel_diff_vs_kalman_median"] = float(np.nanmedian(mp / ref - 1))
    res["prediction"] = {}
    sizes = (100, 1000) if smoke else (100, 300, 1000, 3000, 10000, None)
    for H in (1.0, 2.0):
        tr = S.prediction_samples(m1, H, every=5.0 if smoke else 1.0)
        te = S.prediction_samples(m2, H, every=10.0 if smoke else 1.0)
        res["prediction"][f"{H:g}s"] = S.prediction_study(tr, te, H, sizes=sizes, seeds=(0,) if smoke else (0, 1, 2))
    fig = HERE / "figures"
    plots.smoothing_figure(res["synthetic_smoothing"], fig / "kin_smoothing_mp_error.png")
    for H in ("1s", "2s"):
        rows = res["prediction"][H]["rows"]
        flat = {r["model"]: r["fde_mean"] for r in rows if r["n_train"] == 0}
        plots.efficiency_figure([r for r in rows if r["n_train"] > 0], "fde_mean", fig / f"kin_prediction_{H}.png",
                                f"Player position {H} ahead (train match 1, test match 2)", "mean final error [m]",
                                models=["gbm", "gbm_residual_cv", "gbm_residual_cv_projected"], flat=flat)
    out = HERE / "results" / ("player_kinematics_smoke.json" if smoke else "player_kinematics.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(to_jsonable(res, digits=5), indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    r = main(ap.parse_args().smoke)
    print(json.dumps(to_jsonable({k: v for k, v in r["match_load"]["methods"].items()}, digits=3))[:1500])
