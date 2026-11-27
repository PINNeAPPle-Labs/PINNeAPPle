"""Repair of the broadcast ball height (#911): fit a monotone map ``z_true = f(z_feed)`` jointly over every flight
of a training split of matches, with gravity fixed at 9.81, and check it on the held-out matches.

1. the map itself, for a global scale, a pitch-dependent (camera) scale and a monotone spline, with a bootstrap band
   that resamples matches;
2. held-out gravity: the corrected height of a flight the map never saw must give g = 9.81 when gravity is fitted
   freely again (before the correction the same fit gives about 6 m/s^2, the finding of #902);
3. landing point and apex against the uncorrected feed, first half of the flight observed;
4. where the Cramer-Rao bound allows a drag coefficient or a spin to be reported at all;
5. a known-answer test: simulated flights distorted by a known f must give that f back.

Data: the 20 SkillCorner open-data matches (MIT), 10 Hz broadcast tracking with an estimated ball height, downloaded
once into the cache and never committed.

    python examples/sports/06_height_correction.py            # all 20 matches (~1 h with 8 workers)
    python examples/sports/06_height_correction.py --smoke    # 3 matches, a few minutes
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
from pinneapple_systems.sports.football import data as D  # noqa: E402
from pinneapple_systems.sports.football import height_study as H  # noqa: E402
from pinneapple_systems.sports.football import plots  # noqa: E402

HERE = Path(__file__).resolve().parent
KINDS = ("scale", "camera", "spline")
FIT_KW = {"dt": 0.05}                 # at the 10 Hz of the feed: the same fits as dt = 0.02, three times faster
EDGES = np.linspace(0, 16, 33)        # gravity histogram bins, shared by every figure


def main(smoke: bool = False, workers: int = 8, rounds: int = 2) -> dict:
    ids = sorted(m["id"] for m in D.skillcorner_matches())
    if smoke:
        ids = ids[:3]
    flights = []
    for m in ids:
        flights += B.extract_flights(D.skillcorner_ball(m), match_id=m)
    print(f"{len(flights)} aerial flights in {len(ids)} matches", flush=True)
    n_train = max(1, int(round(0.7 * len(ids))))
    train_ids, test_ids = ids[:n_train], ids[n_train:]
    train = [f for f in flights if f.match_id in train_ids]
    test = [f for f in flights if f.match_id in test_ids]
    print(f"train {len(train)} flights ({len(train_ids)} matches), test {len(test)} flights "
          f"({len(test_ids)} matches)", flush=True)

    res = H.height_correction(train, test, kinds=KINDS, rounds=1 if smoke else rounds, workers=workers,
                              n_boot=200, known_n=12 if smoke else 30, seed=0, fit_kw=FIT_KW)
    res["matches"] = {"train": train_ids, "test": test_ids}
    res["settings"] = {"rounds": 1 if smoke else rounds, "max_rmse": 0.6, "fit_dt": FIT_KW["dt"],
                       "cr_cd_max": B.CR_CD_MAX, "cr_cl_max": B.CR_CL_MAX}

    fig = HERE / "figures"
    for kind in KINDS:
        r = res["kinds_result"][kind]
        c = r["curve"]
        plots.height_map_figure([(c["grid"], c["values"], f"{kind} map")], fig / f"ball_height_map_{kind}.png",
                                band=(r["band"]["grid"], r["band"]["lo"], r["band"]["hi"]),
                                title=f"Broadcast ball height: physical height against the feed ({kind})")
    plots.gravity_histogram_figure([(res["before"]["test_hist"], "feed height, unaltered"),
                                    (res["kinds_result"]["spline"]["hist_test"], "corrected height (spline map)")],
                                   fig / "ball_height_gravity.png", edges=EDGES,
                                   title="Gravity fitted freely to held-out flights, before and after the correction")
    ka = res["known_answer"]["spline"]["curve"]
    plots.height_map_figure([(ka["grid"], ka["recovered"], "recovered from the distorted flights")],
                            fig / "ball_height_known_answer.png", truth=(ka["grid"], ka["truth"], "distortion used"),
                            title="Known-answer test of the correction on simulated flights")

    out = HERE / "results" / ("ball_height_smoke.json" if smoke else "ball_height.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(to_jsonable(res, digits=5), indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    r = main(a.smoke, a.workers, a.rounds)
    print(json.dumps(to_jsonable({k: {"g_test": v["g_test"], "prediction": v["prediction"],
                                      "identifiability": v["identifiability"]}
                                  for k, v in r["kinds_result"].items()}, digits=4), indent=1))
