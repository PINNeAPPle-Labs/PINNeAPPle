"""The 3-D edge-element solver of :mod:`pinneapple_simulation.electromagnetics.fem3d` and the 3-D machines of
:mod:`pinneapple_simulation.electromagnetics.machines3d` as flow nodes: the verification of
``examples/electrodynamics/19_edge3d_verification.py`` (sphere in a uniform field under mesh refinement, the GetDP
cross-check on the same meshes, the manufactured solution and the conducting slab), the skew study of
``20_spm_skew_3d.py``, TEAM problem 7 against its measurements (``21_team7.py``) and the end effects against L/D
(``22_spm_end_effects_3d.py``). Metric names follow the examples' results JSON."""
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
    from pinneapple_simulation.electromagnetics.fem3d.cases import (
        manufactured_magnetostatic,
        slab_eddy,
    )
    from pinneapple_simulation.electromagnetics.fem3d.nedelec import (
        NU0,
        EdgeSpace,
        solve_magnetostatic,
    )
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


@node("em.spm_skew_3d", "Skew of a surface-PM machine: 3-D cogging against multi-slice 2-D",
      category="Electromagnetics", outputs=[Port("metrics", "metrics")],
      params=[Param("totals_deg", "list", [0, 5, 8, 10, 15], help="total skews (deg)"),
              Param("kinds", "list", ["step2", "step3", "continuous"]),
              Param("n_theta", "int", 720, minimum=24,
                    help="angular steps; an even number per cogging period (720: 30 per 24-slot pitch)"),
              Param("theta_step_deg", "float", 1.5, minimum=0.1, help="rotor angle step over one cogging period"),
              Param("stack_layers", "int", 6, minimum=1), Param("end_length", "float", 0.02, minimum=0.0),
              Param("end_layers", "int", 1, minimum=0), Param("mu_r_iron", "float", 1000.0, minimum=1.0)],
      calls="pinneapple_simulation.electromagnetics.machines3d: SPM3D (extruded polar mesh, magnet skew, end "
            "regions, Arkkio torque, multislice), skew.skew_study")
def spm_skew_3d(ctx, inputs, params):
    """examples/electrodynamics/20_spm_skew_3d.py: cogging peak-to-peak against total skew for step and continuous
    skew, in 3-D and in multi-slice 2-D on the same mesh."""
    import math

    from pinneapple_simulation.electromagnetics.machine import SPMMachine
    from pinneapple_simulation.electromagnetics.machines3d import SPM3D
    from pinneapple_simulation.electromagnetics.machines3d.skew import skew_study
    from pinneapple_simulation.electromagnetics.materials import LinearMaterial
    n = int(params["n_theta"])
    m = SPMMachine(iron=LinearMaterial(mu_r=float(params["mu_r_iron"])), n_theta=n)
    model = SPM3D(m, n_theta=n, stack_layers=int(params["stack_layers"]), end_length=float(params["end_length"]),
                  end_layers=int(params["end_layers"]), mu_r_iron=float(params["mu_r_iron"]))
    thetas = np.radians(np.arange(0.0, 15.0, float(params["theta_step_deg"])))
    res = skew_study(model, [math.radians(float(v)) for v in params["totals_deg"]], kinds=list(params["kinds"]),
                     thetas=thetas, progress=ctx.log)
    for r in res["rows"]:
        key = f"{r['kind']}.{math.degrees(r['total']):g}deg"
        ctx.metric(f"{key}.pkpk_3d", r["pkpk_3d"], "N m")
        ctx.metric(f"{key}.pkpk_ms", r["pkpk_ms"], "N m")
    return {"metrics": ctx.report["metrics"]}


@node("em.team7", "TEAM problem 7: 3-D eddy currents against the published measurements",
      category="Electromagnetics", outputs=[Port("metrics", "metrics")],
      params=[Param("h_plate", "list", [0.012, 0.009, 0.007, 0.006],
                    help="element sizes in the plate (m), coarse to fine; 0.006 needs about 14 GB"),
              Param("freqs", "list", [50, 200], help="frequencies (Hz): 50 and/or 200, the measured ones")],
      calls="pinneapple_simulation.electromagnetics.fem3d.team7: mesh_team7, run (solve_eddy with the tree gauge "
            "and PARDISO, probe, compare against MEASURED)",
      available=_gmsh_available)
def team7_node(ctx, inputs, params):
    """examples/electrodynamics/21_team7.py: the Joule loss and the rms difference from the measured Bz and Jy
    (relative to the largest measured value of each line) at each mesh level."""
    from pinneapple_simulation.electromagnetics.fem3d import team7

    tmp = Path(tempfile.mkdtemp(prefix="team7_flow_"))
    for i, h in enumerate(map(float, params["h_plate"])):
        ctx.check_stop()
        msh = team7.mesh_team7(tmp / f"t7_{h}.msh", h_plate=h, h_coil=0.02, h_air=0.08, h_layer=0.01)
        r = team7.run(msh, freqs=tuple(int(f) for f in params["freqs"]), rtol=1e-10)
        ctx.metric(f"levels.{i}.unknowns", float(r["unknowns"]), f"h_plate = {h}")
        for f, v in r["f"].items():
            ctx.metric(f"levels.{i}.f.{f}.loss_W", float(v["loss_W"]), "W")
            for k, c in v["compare"].items():
                ctx.metric(f"levels.{i}.f.{f}.{k.replace(' wt=', '.wt')}.rms_rel_peak", c["rms_rel_peak"])
        msh.unlink()
    shutil.rmtree(tmp, ignore_errors=True)
    return {"metrics": ctx.report["metrics"]}


@node("em.spm_end_effects_3d", "End effects of a PM machine against L/D: 3-D flux linkage over 2-D",
      category="Electromagnetics", outputs=[Port("metrics", "metrics")],
      params=[Param("ratios", "list", [0.25, 0.5, 1.0, 2.0], help="stack length over bore diameter"),
              Param("end_length", "float", 0.04, minimum=0.0), Param("end_layers", "int", 4, minimum=1),
              Param("n_theta", "int", 240, minimum=24), Param("layer_dz", "float", 0.015, minimum=0.001)],
      calls="pinneapple_simulation.electromagnetics.machines3d.endeffects: end_effect_study, search_coil_linkage")
def spm_end_effects_3d(ctx, inputs, params):
    """examples/electrodynamics/22_spm_end_effects_3d.py: k = lambda_3D / lambda_2D for a full-pitch search coil
    closing in the stack end faces and 10 mm beyond them."""
    from pinneapple_simulation.electromagnetics.machine import SPMMachine
    from pinneapple_simulation.electromagnetics.machines3d.endeffects import end_effect_study
    from pinneapple_simulation.electromagnetics.materials import LinearMaterial

    res = end_effect_study(SPMMachine(iron=LinearMaterial(mu_r=1000.0)), [float(v) for v in params["ratios"]],
                           end_length=float(params["end_length"]), end_layers=int(params["end_layers"]),
                           layer_dz=float(params["layer_dz"]), n_theta=int(params["n_theta"]),
                           overhangs=(0.0, 0.01), progress=ctx.log)
    for i, r in enumerate(res["rows"]):
        ctx.metric(f"rows.{i}.k_o0mm", r["k_o0mm"], f"L/D = {r['L_over_D']:g}")
        ctx.metric(f"rows.{i}.k_o10mm", r["k_o10mm"], f"L/D = {r['L_over_D']:g}")
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
    register_capability(Capability(
        id="em.spm_skew_3d", title="Skew of a PM machine: 3-D cogging against multi-slice 2-D",
        domain="electromagnetics", example="electrodynamics",
        keywords={"skew": 3, "cogging": 2, "multi-slice": 2, "multislice": 2, "3-d": 1, "3d": 1, "step skew": 3},
        requires_all=(("motor", "machine", "pm", "spm", "rotor", "magnet"),), min_score=4,
        nodes=("em.spm_skew_3d",), references=("jin2014fem", "bossavit1998"),
        template="electromagnetics/20_spm_skew_3d", quick=True,
        description="examples/electrodynamics/20_spm_skew_3d.py"))

    register_capability(Capability(
        id="em.team7", title="TEAM problem 7: 3-D eddy currents against measurements",
        domain="electromagnetics", example="electrodynamics",
        keywords={"team 7": 3, "team7": 3, "team problem": 2, "eddy current": 2, "asymmetrical conductor": 3,
                  "3-d eddy": 3, "benchmark": 1},
        requires_all=(("team", "eddy", "conductor"),), min_score=3,
        nodes=("em.team7",), references=("jin2014fem", "bossavit1998"),
        template="electromagnetics/21_team7", quick=True,
        description="examples/electrodynamics/21_team7.py"))
    register_capability(Capability(
        id="em.spm_end_effects_3d", title="End effects of a PM machine against L/D in 3-D",
        domain="electromagnetics", example="electrodynamics",
        keywords={"end effect": 3, "end effects": 3, "l/d": 3, "end leakage": 3, "stack length": 2, "3-d": 1,
                  "3d": 1},
        requires_all=(("motor", "machine", "pm", "spm", "stack"),), min_score=4,
        nodes=("em.spm_end_effects_3d",), references=("jin2014fem", "bossavit1998"),
        template="electromagnetics/22_spm_end_effects_3d", quick=True,
        description="examples/electrodynamics/22_spm_end_effects_3d.py"))


_register_capabilities()
