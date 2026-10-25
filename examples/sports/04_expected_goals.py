"""Expected goals with and without physics: shot geometry against intercept-time features of the StatsBomb 360
freeze frame (keeper and blockers reaching the ball's path in time).

Train on the 2022 World Cup and Euro 2020, test on Euro 2024 (all shots with a 360 frame, penalties excluded).
The two physical parameters (shot speed, keeper speed) are chosen by grouped cross-validation on the training
shots only. StatsBomb's own xG is shown as an external reference (a proprietary model trained on far more data).

Data: StatsBomb open data (non-commercial use; "Data: StatsBomb").

    python examples/sports/04_expected_goals.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # before numpy / torch / sklearn: small models, no oversubscription

from pinneapple_core.jsonable import to_jsonable  # noqa: E402
from pinneapple_systems.sports.football import data as D  # noqa: E402
from pinneapple_systems.sports.football import plots  # noqa: E402
from pinneapple_systems.sports.football import xg as X  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(smoke: bool = False) -> dict:
    if smoke:
        shots = D.statsbomb_shots({"euro_2020": D.STATSBOMB_360["euro_2020"]}, max_matches=12)
        cut = len(shots) * 2 // 3
        tr, te = shots[:cut], shots[cut:]
    else:
        shots = D.statsbomb_shots()
        tr = [s for s in shots if s["season"] != "euro_2024"]
        te = [s for s in shots if s["season"] == "euro_2024"]
    A, T = X.shots_arrays(tr), X.shots_arrays(te)
    res = X.compare(A, T)
    res["data_efficiency"] = X.data_efficiency(A, T, X.XGParams(**res["params"]),
                                               sizes=(100, None) if smoke else (100, 200, 400, 800, 1600, None))
    res["n_goals"] = {"train": int(A["y"].sum()), "test": int(T["y"].sum())}
    fig = HERE / "figures"
    plots.reliability_figure(res["reliability"], fig / "xg_reliability.png", "xG reliability on Euro 2024",
                             models=["standard_logistic", "standard+physics_logistic",
                                     "standard+freeze_geometry_logistic", "statsbomb_xg_reference"])
    plots.efficiency_figure(res["data_efficiency"]["rows"], "log_loss", fig / "xg_learning_curve.png",
                            "xG: test log-loss against training shots", "log-loss on Euro 2024",
                            models=["standard_logistic", "standard+physics_logistic", "physics_logistic"])
    out = HERE / "results" / ("expected_goals_smoke.json" if smoke else "expected_goals.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(to_jsonable({k: v for k, v in res.items() if k != "pred"}, digits=5), indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    r = main(ap.parse_args().smoke)
    print({m: round(s["log_loss"], 4) for m, s in r["scores"].items()})
