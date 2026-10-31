"""H-B0, last item: Stokes drag of the Schafer-Turek cylinder (no convection) by the immersed solver at the finest
rung of its C1 ladder against the finest body-fitted Taylor-Hood run (not timed). Writes results/stokes_check.json.

    python examples/cfd_direct_from_stl/stokes_check.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from pinneapple_design.geometry.mesh.generate import generate_mesh  # noqa: E402
from pinneapple_simulation.cfd_direct.cases import SCHAFER_TUREK as c  # noqa: E402
from pinneapple_simulation.cfd_direct.cases import cylinder_polygon  # noqa: E402
from pinneapple_simulation.immersed import flow  # noqa: E402
from pinneapple_simulation.numerical_solvers.taylor_hood import TaylorHood, read_gmsh  # noqa: E402


def main(n_ref: int = 64, n_imm: int = 32) -> dict:
    m = generate_mesh("polygon", vertices=cylinder_polygon(512).tolist(), enclosure=[[0, 0], [c.L, c.H]],
                      h=4 * c.D / n_ref, refine=[{"center": list(c.c), "radius": c.D, "h": c.D / n_ref}],
                      out_dir=tempfile.mkdtemp(), max_elements=5_000_000)
    th = TaylorHood(read_gmsh(m["files"]["msh"]), c.nu)

    def body(x):
        return np.hypot(x[0] - 0.2, x[1] - 0.2) < 0.051
    r = th.solve([(lambda x: x[0] < 1e-9, lambda P: np.stack([c.inflow(P[1]), 0 * P[1]])),
                  (lambda x: (x[1] < 1e-9) | (x[1] > c.H - 1e-9), 0.0), (body, 0.0)], navier_stokes=False)
    cd_ref = c.coefficients(r.force(body))[0]
    fd = flow.FlowDiscretization(flow.PolygonGeometry(cylinder_polygon(512)), c.box, h=c.D / n_imm)
    s = flow.solve_stokes(fd, c.nu, {"x-": flow.Velocity(lambda P: np.column_stack([c.inflow(P[:, 1]), 0 * P[:, 1]])),
                                     "y-": "wall", "y+": "wall", "x+": "outflow"})
    F = s.force()
    cd = c.coefficients(F["variational"])[0]
    out = {"Cd_stokes_reference": cd_ref, "n_reference": n_ref, "Cd_stokes_immersed": cd, "n_immersed": n_imm,
           "Cd_stokes_immersed_surface": c.coefficients(F["surface"])[0], "rel_diff": abs(cd - cd_ref) / cd_ref,
           "criterion": "within 2 %", "passed": abs(cd - cd_ref) / cd_ref <= 0.02}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "stokes_check.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    print(main())
