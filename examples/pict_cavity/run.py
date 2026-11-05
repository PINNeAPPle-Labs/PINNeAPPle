"""Lid-driven cavity against Ghia et al. (1982), on a machine without CUDA.

The cavity is the benchmark every incompressible solver is asked for: a square box, three still walls, a lid moving at
U = 1, and the published centreline profiles of Ghia, Ghia and Shin (J. Comput. Phys. 48 (1982) 387-411) to compare
with. PINNeAPPle runs it here with the local engine of
:mod:`pinneapple_simulation.external_solvers.pict` -- a vorticity-streamfunction solver in numpy/scipy, so this script
needs no GPU. The same call with ``--engine pict`` runs PICT (multi-block PISO on CUDA) when it is installed, and says
what is missing when it is not; the output always names the engine that produced the numbers.

What is measured, per Reynolds number:

- the rms difference from the Ghia table on the u centreline (x = 0.5) and the v centreline (y = 0.5), in units of
  the lid velocity -- the number the solver is judged on;
- the discrete divergence of the velocity (a streamfunction velocity should be divergence-free to round-off);
- the three closed walls and the lid, i.e. that the boundary conditions that were imposed are the ones in the field.

    python examples/pict_cavity/run.py              # 64x64 at Re = 100 and 128x128 at Re = 1000: ~25 s
    python examples/pict_cavity/run.py --smoke      # 32x32 at Re = 100: a tenth of a second (CI)
    python examples/pict_cavity/run.py --engine pict --re 100 1000    # on a machine with PICT and an NVIDIA GPU
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pinneapple_simulation.external_solvers.pict import CavityCase, PICTRunConfig, run_cavity
from pinneapple_simulation.external_solvers.pict.validation import GHIA, centerline

HERE = os.path.dirname(os.path.abspath(__file__))
# cells per side per Reynolds number: the 1/Re boundary layer needs to be resolved, so Re = 1000 gets a finer grid.
# measured on this repository's CPU engine: 0.1 s at 32^2, 0.7 s at 64^2, 23 s at 128^2.
GRID = {100: 64, 1000: 128}
SMOKE = {"re": (100,), "cells": 32}   # 0.1 s, and coarse enough to see the check fail if it will


def ghia_reference(component: str) -> tuple[np.ndarray, np.ndarray]:
    """The published centreline of Ghia et al. (1982) at Re = 1000, as ``(station, value)``."""
    station = GHIA["y"] if component == "u" else GHIA["x"]
    return np.asarray(station, float), np.asarray(GHIA[component][1000], float)


def main(smoke: bool = False, base: str | None = None, engine: str = "local",
         reynolds: tuple[int, ...] | None = None) -> dict:
    """Run the cavity at each Reynolds number and check it. ``base``: where the figure goes (tests pass a temp dir)."""
    from pinneapple_lab import experiment

    out = os.path.join(base or HERE)
    os.makedirs(out, exist_ok=True)
    reynolds = reynolds or (SMOKE["re"] if smoke else (100, 1000))
    rows, results = [], {}

    with experiment("pict_cavity", title="Lid-driven cavity vs Ghia et al. (1982)",
                    domain="fluid dynamics") as ex:
        for re in reynolds:
            cells = SMOKE["cells"] if smoke else GRID[int(re)]
            case = CavityCase(nx=cells, ny=cells, Re=re, refine=False)
            config = PICTRunConfig(iterations=50) if engine == "pict" else None
            result = run_cavity(case, config, engine=engine)
            results[int(re)] = result
            checks = result.checks()
            row = {"Re": int(re), "cells": cells, "engine": result.engine, "layout": result.layout,
                   "passed": bool(checks["passed"]),
                   "ghia_u_rms": checks.get("ghia_u", {}).get("rms"),
                   "ghia_v_rms": checks.get("ghia_v", {}).get("rms"),
                   "continuity": checks["continuity"]["value"],
                   "walls": checks["walls"]["value"],
                   "lid": checks["lid"]["value"],
                   "wall_time_s": result.wall_time_s}
            rows.append(row)
            if row["ghia_u_rms"] is not None:
                ex.log_metric(f"ghia_u_rms_re{int(re)}", float(row["ghia_u_rms"]), unit="",
                              how=f"rms of u(x=0.5) against Ghia et al. (1982) Table I at Re={int(re)}, "
                                  f"{cells}x{cells} cells, {result.engine} engine")
                ex.log_metric(f"ghia_v_rms_re{int(re)}", float(row["ghia_v_rms"]), unit="",
                              how=f"rms of v(y=0.5) against Ghia et al. (1982) Table II at Re={int(re)}, "
                                  f"{cells}x{cells} cells, {result.engine} engine")
            ex.log_metric(f"continuity_re{int(re)}", float(row["continuity"]), unit="U/L",
                          how="largest discrete du/dx + dv/dy, normalised by the lid velocity over the side")
            ex.check(f"Re={int(re)}: walls and lid are the imposed boundary conditions",
                     passed=bool(row["walls"] < 1e-9 and abs(row["lid"] - 1.0) < 0.1))
            if row["ghia_u_rms"] is not None:
                ex.check(f"Re={int(re)}: centreline within 0.05 of Ghia et al.", passed=row["passed"],
                         detail=json.dumps({k: row[k] for k in ("ghia_u_rms", "ghia_v_rms")}))

        # the figure: computed centreline against the published one, at the finest Reynolds number that ran --
        # from the run already made, not from a second one (a repeat would also hide a nondeterministic difference)
        finest = max(rows, key=lambda r: r["Re"])
        if not smoke:
            _plot(results[finest["Re"]], finest["Re"], out)

    summary = {"engine": rows[0]["engine"] if rows else engine, "rows": rows,
               "passed": all(r["passed"] for r in rows)}
    with open(os.path.join(out, "cavity.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


def _plot(result, reynolds: int, out: str) -> str:
    """Computed u(x = 0.5) and v(y = 0.5) against the published table (matplotlib, no display needed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u = centerline(result.component("u")[-1], direction="vertical")
    v = centerline(result.component("v")[-1], direction="horizontal")
    x0, y0, x1, y1 = result.case_summary["bounds"]
    ys = np.linspace(y0, y1, u.size)
    xs = np.linspace(x0, x1, v.size)
    ref_y, ref_u = ghia_reference("u")
    ref_x, ref_v = ghia_reference("v")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(u, (ys - y0) / (y1 - y0), label=f"PINNeAPPle ({result.engine})")
    axes[0].plot(ref_u, ref_y, "o", markersize=3, label="Ghia et al. (1982)")
    axes[0].set_xlabel("u / U   (x = 0.5)"), axes[0].set_ylabel("y / L"), axes[0].legend()
    axes[1].plot((xs - x0) / (x1 - x0), v, label=f"PINNeAPPle ({result.engine})")
    axes[1].plot(ref_x, ref_v, "o", markersize=3, label="Ghia et al. (1982)")
    axes[1].set_xlabel("x / L"), axes[1].set_ylabel("v / U   (y = 0.5)"), axes[1].legend()
    fig.suptitle(f"Lid-driven cavity, Re = {reynolds}")
    fig.tight_layout()
    path = os.path.join(out, "cavity_profiles.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="one 32x32 run at Re = 100 (a tenth of a second)")
    parser.add_argument("--engine", default="local", choices=("local", "pict", "auto"),
                        help="local (CPU, this repository's engine), pict (needs CUDA) or auto")
    parser.add_argument("--re", type=int, nargs="+", default=None, help="Reynolds numbers to run")
    args = parser.parse_args()
    main(smoke=args.smoke, engine=args.engine, reynolds=tuple(args.re) if args.re else None)
