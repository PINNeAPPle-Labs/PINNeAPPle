"""The 3-D edge-element solver of :mod:`pinneapple_simulation.electromagnetics.fem3d` as a flow node: the verification
of ``examples/electrodynamics/19_edge3d_verification.py`` (sphere in a uniform field under mesh refinement, the
GetDP cross-check on the same meshes, the manufactured solution and the conducting slab). Metric names follow the
example's results JSON."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from ..registry import Param, Port, node


def _gmsh_available():
    try:
        import gmsh  # noqa: F401
    except ImportError:
        return False, "needs the gmsh Python module (pip install gmsh)"
    return True, ""


@node("em.edge3d_verification", "3-D edge elements: sphere, GetDP cross-check, manufactured, slab",
      category="Electromagnetics", outputs=[Port("metrics", "metrics")],
      params=[Param("h_sphere", "list", [0.3, 0.2, 0.14, 0.1, 0.07],
                    help="element sizes in the sphere (radius 1), coarse to fine"),
              Param("h_getdp", "list", [0.3, 0.2, 0.14], help="sizes also solved by GetDP (when installed)"),
              Param("mu_r", "float", 10.0, minimum=1.0)],
      calls="pinneapple_simulation.electromagnetics.fem3d.sphere: SphereCase, mesh_sphere, solve_native, errors, "
            "uniform_dirichlet; fem3d.getdp3d.sphere; fem3d.cases: manufactured_magnetostatic, slab_eddy",
      available=_gmsh_available)
def edge3d_verification(ctx, inputs, params):
    """examples/electrodynamics/19_edge3d_verification.py."""
    from pinneapple_simulation.electromagnetics.fem3d import getdp3d, read_msh22
    from pinneapple_simulation.electromagnetics.fem3d.cases import manufactured_magnetostatic, slab_eddy
    from pinneapple_simulation.electromagnetics.fem3d.nedelec import NU0, EdgeSpace, solve_magnetostatic
    from pinneapple_simulation.electromagnetics.fem3d.sphere import (
        SphereCase,
        errors,
        mesh_sphere,
        solve_native,
        uniform_dirichlet,
    )
    case = SphereCase(a=1.0, R=3.0, mu_r=float(params["mu_r"]))
    tmp = Path(tempfile.mkdtemp(prefix="edge3d_flow_"))
    have_getdp = shutil.which("getdp") is not None
    k = 0
    for i, h in enumerate(map(float, params["h_sphere"])):
        ctx.check_stop()
        msh = mesh_sphere(tmp / f"s{h}.msh", h, case)
        r = solve_native(msh, case)
        for key in ("mean_in", "l2_in", "l2_all"):
            ctx.metric(f"sphere.{i}.{key}", float(r[key]), f"h = {h}, {r['unknowns']} unknowns")
        if have_getdp and h in [float(v) for v in params["h_getdp"]]:
            g = getdp3d.sphere(msh, case)
            m = read_msh22(msh)
            sp = EdgeSpace(m)
            s = solve_magnetostatic(m, np.where(m.mask("sphere"), NU0 / case.mu_r, NU0),
                                    dirichlet=uniform_dirichlet(m, sp, case), space=sp, cg_tol=1e-12)
            e = errors(m, s.B, case)
            ctx.metric(f"getdp.{k}.rel_diff_Bz", abs(e["Bz_mean_in"] - g["Bz_mean_in"]) / abs(g["Bz_mean_in"]))
            ctx.metric(f"getdp.{k}.rel_diff_energy", abs(s.energy - g["energy"]) / abs(g["energy"]))
            k += 1
    for i, n in enumerate((5, 9, 17, 33)):
        ctx.metric(f"manufactured.{i}.error", float(manufactured_magnetostatic(n)["error"]))
    for i, (nz, nxy) in enumerate(((4, 2), (8, 4), (16, 8))):
        r = slab_eddy(nz, nxy=nxy, gauge="tree")
        ctx.metric(f"slab.{i}.loss_error", float(r["loss_error"]))
    shutil.rmtree(tmp, ignore_errors=True)
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="em.edge3d_verification", title="3-D magnetostatics and eddy currents on edge elements, verified",
        domain="electromagnetics", example="electrodynamics",
        keywords={"edge element": 3, "nedelec": 3, "3-d magnetostatics": 3, "3d magnetostatics": 3,
                  "whitney": 2, "sphere in a uniform field": 2},
        requires_all=(("3-d", "3d", "edge", "nedelec", "sphere"),), min_score=3,
        nodes=("em.edge3d_verification",), references=("jin2014fem", "bossavit1998"),
        template="electromagnetics/19_edge3d_verification", quick=True,
        description="examples/electrodynamics/19_edge3d_verification.py"))


_register_capabilities()
