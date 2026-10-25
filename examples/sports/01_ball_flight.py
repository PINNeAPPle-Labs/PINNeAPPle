"""Ball flight: drag, Magnus and gravity from tracked positions; where the inverse problem is posed, and whether the
physics predicts real broadcast-tracking flights better than data-only extrapolation and boosted trees.

1. method check on simulated kicks with known drag and spin (errors, coverage of the 95 % intervals);
2. Cramer-Rao comparison of 3-D and 2-D tracking (2-D: no ball height, as in the Metrica sample data);
3. gravity audit of every aerial flight in the 20 SkillCorner open-data matches (10 Hz, estimated ball height);
4. held-out prediction: first half of each flight observed, the rest predicted; train matches != test matches.

    python examples/sports/01_ball_flight.py            # full (downloads ~1.8 GB of SkillCorner tracking once)
    python examples/sports/01_ball_flight.py --smoke    # 2 matches, few flights
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # before numpy / torch / sklearn: small models, no oversubscription

import numpy as np  # noqa: E402

from pinneapple_core.jsonable import to_jsonable  # noqa: E402
from pinneapple_systems.sports.football import ball as B  # noqa: E402
from pinneapple_systems.sports.football import ball_study as S  # noqa: E402
from pinneapple_systems.sports.football import data as D  # noqa: E402
from pinneapple_systems.sports.football import plots  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(smoke: bool = False, workers: int = 4) -> dict:
    res: dict = {}
    res["synthetic"] = S.synthetic_recovery(10 if smoke else 60, noise=0.2)
    res["synthetic_no_spin_audit"] = S.gravity_audit([f for f, _ in S.synthetic_flights(10 if smoke else 60, seed=3)],
                                                     workers=workers)
    ident = S.identifiability()
    res["identifiability"] = {k: {"std": v["std"], "cond": v["cond"]} for k, v in ident.items()}

    ids = sorted(m["id"] for m in D.skillcorner_matches())
    if smoke:
        ids = ids[:2]
    n_train = max(1, int(round(0.7 * len(ids))))
    train_ids, test_ids = ids[:n_train], ids[n_train:]
    F_train = S.flights_from_skillcorner(train_ids)
    F_test = S.flights_from_skillcorner(test_ids)
    if smoke:
        F_train, F_test = F_train[:30], F_test[:15]
    audit = S.gravity_audit(F_train + F_test, workers=workers)
    res["gravity_audit"] = audit
    ho = S.holdout_prediction(F_train, F_test, workers=workers)
    res["holdout"] = ho
    res["holdout_summary"] = S.summarise(ho)
    res["matches"] = {"train": train_ids, "test": test_ids}

    fig = HERE / "figures"
    plots.gravity_audit_figure(audit, res["synthetic_no_spin_audit"]["g_drag"], fig / "ball_gravity_audit.png")
    plots.holdout_figure(ho, fig / "ball_holdout_landing.png",
                         models=["poly2", "vacuum", "drag", "drag_free_g", "ml", "hybrid"])
    # three long flights of the test matches: observations, the free-g fit and the g = 9.81 drag fit
    longest = sorted(F_test, key=lambda f: -f.duration)[:3]
    show = []
    for i, f in enumerate(longest):
        free = B.fit_flight(f.t, f.xyz, "drag_free_g")
        t_dense = np.linspace(f.t[0], f.t[-1], 60)
        show.append((f.xyz, free.predict(t_dense[1:]), f"flight {i + 1}: fitted g = {free.g:.1f} m/s²"))
    plots.flights_3d(show, fig / "ball_flights_3d.png", "SkillCorner aerial flights and the drag fit with free gravity")
    out = HERE / "results" / ("ball_flight_smoke.json" if smoke else "ball_flight.json")
    out.parent.mkdir(exist_ok=True)
    slim = {k: v for k, v in res.items() if k not in ("gravity_audit", "holdout", "synthetic_no_spin_audit")}
    slim["gravity_audit"] = {k: v for k, v in audit.items() if not isinstance(v, list) or k in ("g_drag_ci95",
                                                                                              "g_vacuum_ci95", "by_speed")}
    slim["synthetic_no_spin_audit"] = {k: res["synthetic_no_spin_audit"][k] for k in ("g_drag_median", "g_drag_ci95",
                                                                                      "n_used")}
    slim["holdout"] = {k: ({kk: vv for kk, vv in v.items() if kk != "landing_errors"} if isinstance(v, dict) else v)
                       for k, v in ho.items()}
    out.write_text(json.dumps(to_jsonable(slim, digits=5), indent=1))
    return slim


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    r = main(a.smoke, a.workers)
    print(json.dumps(to_jsonable({"gravity": r["gravity_audit"], "holdout": r["holdout_summary"]}, digits=4), indent=1))
