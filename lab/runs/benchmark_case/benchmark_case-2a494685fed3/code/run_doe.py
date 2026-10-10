"""Design of experiments for the OpenRadioss Bumper Beam: 100-point Latin Hypercube over
two linked shell gauges, staged and solved with PINNeAPPle's OpenRadioss bridge.

DV1 = DP1000 gauge -> /PROP/SHELL/2 and /PROP/SHELL/7   (nominal 1.8 mm, +-30%: 1.26..2.34)
DV2 = DP600  gauge -> /PROP/SHELL/1 and /PROP/SHELL/6   (nominal 2.2 mm, +-30%: 1.54..2.86)
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2] / "PINNeAPPle"))

from pinneapple_data.parameter_sampling import sample_parameters  # noqa: E402
from pinneapple_simulation.external_solvers.openradioss import (  # noqa: E402
    OpenRadiossDockerConfig, read_shell_thickness, run_openradioss_cases, stage_case, write_engine_anim,
)

RUN = "Bumper_Beam_AP_meshed"
STARTER = ROOT / "model" / f"{RUN}_0000.rad"
N_RUNS = 100
BOUNDS = {"dv1_dp1000": (1.26, 2.34), "dv2_dp600": (1.54, 2.86)}
ENGINE_TAIL = "/DT/NODA/CST/0\n0.9 0.001\n/PRINT/-100/55\n/TFILE/0\n#            dT_HIS\n0.100000"
ENGINE_HEADER = ("#\n# Engine file derived from the OpenRadioss 'Bumper Beam' example (Altair, CC BY-NC 4.0).\n"
                 "# Change vs. original: H3D output replaced by ANIM displacement output every 1 ms.")


def main(max_parallel: int = 8) -> None:
    deck = STARTER.read_text()
    nominal = {p: read_shell_thickness(deck, p) for p in (1, 2, 6, 7)}
    print("nominal gauges:", nominal)
    doe_dir = ROOT / "runs"
    doe_dir.mkdir(exist_ok=True)
    engine = write_engine_anim(doe_dir / f"{RUN}_0001.rad", RUN, 100.0, 1.0,
                               header=ENGINE_HEADER, tail=ENGINE_TAIL)
    pts = sample_parameters(BOUNDS, N_RUNS, method="lhs", seed=42)
    cases = []
    with open(ROOT / "data" / "design_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["exp", "dv1_dp1000_mm", "dv2_dp600_mm"])
        for i, p in enumerate(pts, start=1):
            d1, d2 = round(p["dv1_dp1000"], 4), round(p["dv2_dp600"], 4)
            w.writerow([f"Exp_{i}", d1, d2])
            cases.append(stage_case(STARTER, engine, doe_dir / f"Exp_{i}", {2: d1, 7: d1, 1: d2, 6: d2}))
    cfg = OpenRadiossDockerConfig(openradioss_dir=str(ROOT / "solver" / "OpenRadioss"))
    t0 = time.time()

    def on_done(case, err):
        status = "OK" if err is None else f"FAILED: {str(err)[:300]}"
        print(f"[{time.time() - t0:7.0f}s] {case.name}: {status}", flush=True)

    # run the write-up's held-out test cases and our validation cases first, then the rest
    first = [5, 15, 16, 19, 30, 33, 37, 83, 88, 96, 3, 12, 24, 41, 50, 58, 67, 74, 91, 99]
    order = sorted(cases, key=lambda c: (int(c.name.split("_")[1]) not in first, int(c.name.split("_")[1])))
    order = [c for c in order if not (c / "done.flag").exists()]
    res = run_openradioss_cases(order, RUN, cfg, max_parallel=max_parallel, on_done=on_done)
    print(f"finished {sum(r is not None for r in res)}/{len(res)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
