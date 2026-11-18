"""The reference case: a NACA 0012 at zero incidence with a spoiler that deploys to 90 degrees.

Re = 4e4, a spoiler of 10 % of the chord hinged where its trailing edge is 20 % of the chord ahead of the airfoil's
trailing edge, angle of attack 0 -- the case of the reference video. The run reports the loads before and after
the spoiler stands up, and saves the surface pressure, the field and the time series.

    python naca0012_spoiler.py [--full] [--out DIR]
"""
from __future__ import annotations

import argparse


def run(smoke: bool = True, out_dir=None) -> dict:
    from pinneapple_simulation.vortex_domains import (
        grid_field,
        naca0012_spoiler_case,
        run_study,
        surface_pressure,
    )

    case = naca0012_spoiler_case(re=40_000.0, panels=180, spoiler_panels=24, t_start=0.4, deploy_time=0.5,
                                 t_end=0.7 if smoke else 1.4)
    result = run_study(case)
    print(result.summary())
    sim = result.simulation
    series = result.series
    before = series["t"] < 0.4
    after = series["t"] > 1.0
    metrics = {
        "cl_before": float(series["fy"][before].mean()),
        "cl_after": float(series["fy"][after].mean()) if after.any() else float("nan"),
        "cd_after": float((series["fx"] + series["fx_friction"])[after].mean()) if after.any() else float("nan"),
        "peak_cd": float((series["fx"] + series["fx_friction"]).max()),
        "domains": float(len(sim.domains)),
        "circulation_shed": float(sim.shed),
        "panels": float(len(sim.panels)),
    }
    mid, cp = surface_pressure(sim)
    field = grid_field(sim, extent=(-0.2, 1.6, -0.5, 0.5), n=64 if smoke else 200)
    metrics["cp_min"] = float(cp.min())
    metrics["vorticity_rms"] = float((field.vorticity ** 2).mean() ** 0.5)
    if out_dir is not None:
        import numpy as np

        result.save(out_dir)
        np.savez(f"{out_dir}/fields.npz", mid=mid, cp=cp, **field.to_dict())
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="the resolution and end time of the reference case")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run(smoke=not args.full, out_dir=args.out)
