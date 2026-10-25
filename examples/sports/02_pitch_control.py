"""Pass outcome on Metrica tracking: physics-based pitch control against data-only classifiers at equal data.

Train on one sample match, test on the other, then swap. Models (``pinneapple_systems.sports.football.pc_study``):
Spearman (2018) pitch control with open-implementation defaults, the same with 5 physical parameters fitted by
maximum likelihood (reaction-time or acceleration-limited intercept), logistic regression and boosted trees on
geometric features, and a hybrid (trees + pitch-control features). Also: learning curves with 25 to all training
passes, the ball travel time of the passes (constant speed against a rolling ball), reliability curves and a
pitch-control map of one pass.

    python examples/sports/02_pitch_control.py [--smoke]
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
from pinneapple_systems.sports.football import pc_study as S  # noqa: E402
from pinneapple_systems.sports.football import pitch_control as PC  # noqa: E402
from pinneapple_systems.sports.football import plots  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(smoke: bool = False) -> dict:
    m1, m2 = D.metrica_match(1), D.metrica_match(2)
    ps = {1: PC.passes_from_metrica(m1), 2: PC.passes_from_metrica(m2)}
    if smoke:
        ps = {k: v.subset(np.random.default_rng(k).choice(len(v), 200, replace=False)) for k, v in ps.items()}
    res: dict = {"n_passes": {k: len(v) for k, v in ps.items()},
                 "completion_rate": {k: float(v.y.mean()) for k, v in ps.items()},
                 "ball_travel": {k: PC.fit_ball_travel(v) for k, v in ps.items()}}
    res["compare_1_to_2"] = S.compare(ps[1], ps[2])
    res["compare_2_to_1"] = S.compare(ps[2], ps[1])
    sizes = (25, 50, 100) if smoke else (25, 50, 100, 200, 400, None)
    res["efficiency_1_to_2"] = S.data_efficiency(ps[1], ps[2], sizes=sizes, seeds=(0, 1) if smoke else (0, 1, 2, 3, 4))

    fig = HERE / "figures"
    c = res["compare_1_to_2"]
    plots.reliability_figure(c["reliability"], fig / "pc_reliability.png",
                             "Pass completion: reliability on match 2 (fitted on match 1)",
                             models=["pc_default", "pc_reaction", "logistic", "gbm", "hybrid"])
    flat = {"pc_default": c["scores"]["pc_default"]["log_loss"]}
    plots.efficiency_figure(res["efficiency_1_to_2"]["rows"], "log_loss", fig / "pc_learning_curve.png",
                            "Pass completion: test log-loss against training passes", "log-loss on match 2",
                            models=["pc_reaction", "logistic", "gbm", "hybrid"], flat=flat)
    # pitch-control map at the kick of an intercepted long pass of match 2
    t2 = ps[2]
    d = np.linalg.norm(t2.target - t2.ball, axis=1)
    i = int(np.argmax(np.where(t2.y == 0, d, 0)))
    p = PC.PCParams(**{k: v for k, v in c["params"]["pc_reaction"].items() if k != "a_max"}, tti="reaction")
    xs, ys, q = PC.pitch_surface(t2.att_pos[i], t2.att_vel[i], t2.def_pos[i], t2.def_vel[i], t2.ball[i], p,
                                 att_gk=t2.att_gk[i], def_gk=t2.def_gk[i])
    plots.pitch_control_figure(xs, ys, q, t2.att_pos[i], t2.att_vel[i], t2.def_pos[i], t2.def_vel[i], t2.ball[i],
                               t2.target[i], fig / "pc_map_intercepted_pass.png",
                               "Fitted pitch control at the kick of an intercepted pass (Metrica match 2)")
    out = HERE / "results" / ("pitch_control_smoke.json" if smoke else "pitch_control.json")
    out.parent.mkdir(exist_ok=True)
    slim = {k: v for k, v in res.items()}
    for k in ("compare_1_to_2", "compare_2_to_1"):
        slim[k] = {kk: vv for kk, vv in res[k].items() if kk != "pred"}
    out.write_text(json.dumps(to_jsonable(slim, digits=5), indent=1))
    return slim


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    r = main(ap.parse_args().smoke)
    for k in ("compare_1_to_2", "compare_2_to_1"):
        print(k, {m: round(s["log_loss"], 4) for m, s in r[k]["scores"].items()})
