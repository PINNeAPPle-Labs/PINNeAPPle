"""The corner as one Studio study: the loop, the 3-D field of the arm, the curves a chassis engineer reads, and the
checks that say whether the answer stands.

Two tools, both executable from the catalog with their defaults:

* :func:`suspension_study` — the whole problem in one call: it searches the design (tube, bushes, two hardpoints) for
  the lightest corner that passes the structural, kinematic and compliance constraints, re-checks the winner with a
  finite-element model of the arm, and writes the files the Studio shows: the 3-D field of von Mises on the deformed
  leg, the CAD of the corner, and four figures (wheels in bump, compliance against bush rate, the search's own
  history, and the field along the leg). It returns the numbers and **its own checks**;
* :func:`suspension_verify` — the same deterministic search, re-checked with a finer mesh: the design must come back
  the same (the search is reproducible), the finite-element peak must agree with beam theory and must not move when the
  mesh is refined.

Every number the report carries is computed here, from the models, and nothing is typed in. The limits of what this
answers are printed with the answer:

* linear elasticity and small displacements; no plasticity, no contact, no weld or heat-affected zone;
* static load cases only — a static margin against yield is **not** a fatigue life, and the corner's own inertia, the
  damper's reaction and the tyre's compliance are outside the model;
* bushings linear and independent, with the compliance modes treated uncoupled;
* one corner of a symmetric axle; no roll stiffness, no steering system, no aero;
* the arm as a uniform tube on a clamped root, with the ball joint's offset as a couple — not the forged or welded
  shape a production arm has.

    from pinneapple_design.suspension.study import suspension_study
    out = suspension_study(out_dir=".")
    out["checks"]                       # what passed, what failed, with the number each one measured
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .compliance import wheel_compliance  # noqa: E402
from .design import CONSTRAINTS, Design, suspension_design  # noqa: E402
from .geometry import corner_cad  # noqa: E402
from .hardpoints import front_corner  # noqa: E402
from .kinematics import kinematic_sweep, pose  # noqa: E402
from .loads import worst_case  # noqa: E402
from .structure import TubeSection, fem_lower_arm  # noqa: E402

__all__ = ["suspension_study", "suspension_verify", "MESH_DENSITY"]

PASS, FAIL = "pass", "fail"

#: mesh of the finite-element check: ``(n_theta, n_radial, n_len)`` for the loop's confirmation and for the re-check
MESH_DENSITY = {"coarse": (24, 2, 22), "fine": (36, 2, 40)}

INK = "#1c1a18"
ACCENT = "#fa7c1f"
MUTED = "#6d6862"


def _spec(corner_mass_kg: float, bump_g: float):
    """The corner, with the corner's sprung mass and the vertical load case the problem states."""
    spec = front_corner()
    from dataclasses import replace
    return replace(spec, corner_mass_kg=float(corner_mass_kg),
                   sources={**spec.sources, "load case": f"bump {bump_g:g} g, scaled from the corner's m g"})


def _cases(spec, bump_g: float) -> dict[str, np.ndarray]:
    from .loads import GRAVITY, LOAD_CASES
    static = spec.corner_mass_kg * GRAVITY
    cases = {k: np.asarray(v, float) * static for k, v in LOAD_CASES.items()}
    cases["bump 3 g"] = np.array([0.0, 0.0, float(bump_g)]) * static       # the problem's own vertical case
    return cases


def _check(code: str, ok: bool, value, detail: str) -> dict:
    return {"code": code, "status": PASS if ok else FAIL, "value": value, "detail": detail}


# ------------------------------------------------------------------------------------------------- figures
def _figure_kinematics(sweep, spec, path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    t = sweep.travel_mm
    ax[0].plot(t, sweep.camber_deg, color=ACCENT, lw=2)
    ax[0].axvline(0, color=MUTED, lw=0.8, ls=":")
    ax[0].set_title("camber against wheel travel", loc="left", fontsize=10, color=INK)
    ax[0].set_xlabel("wheel travel [mm] (bump +)")
    ax[0].set_ylabel("camber [deg] (negative = top inboard)")
    ax[1].plot(t, sweep.toe_deg, color="#0081be", lw=2)
    ax[1].axvline(0, color=MUTED, lw=0.8, ls=":")
    ax[1].set_title("toe against wheel travel (bump steer)", loc="left", fontsize=10, color=INK)
    ax[1].set_xlabel("wheel travel [mm] (bump +)")
    ax[1].set_ylabel("toe [deg] (positive = toe-in)")
    for a in ax:
        a.grid(alpha=0.25)
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _figure_compliance(spec, design: Design, comp: dict, path: str) -> None:
    rates = np.linspace(400.0, 1600.0, 13)
    toe, camber = [], []
    for r in rates:
        c = wheel_compliance(design.apply(spec).with_bushings(r * 1e3))
        toe.append(c["toe_deg_per_kn"])
        camber.append(c["camber_deg_per_kn"])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(rates, toe, color=ACCENT, lw=2, label="toe compliance")
    ax.plot(rates, camber, color="#0081be", lw=2, ls="--", label="camber compliance")
    ax.axhline(CONSTRAINTS["toe_compliance_deg_per_kn"]["limit"], color=MUTED, lw=1, ls=":")
    ax.annotate("limit 0.45 deg/kN", (rates[0], CONSTRAINTS["toe_compliance_deg_per_kn"]["limit"]), fontsize=7,
                color=MUTED, va="bottom")
    ax.plot([design.bushing_n_per_mm], [comp["toe_deg_per_kn"]], "o", color=INK, ms=6)
    ax.annotate(f"design: {design.bushing_n_per_mm:.0f} N/mm", (design.bushing_n_per_mm, comp["toe_deg_per_kn"]),
                textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
    ax.set_xlabel("bush rate [N/mm]")
    ax.set_ylabel("deg per kN of lateral force")
    ax.set_title("elastokinematics: what the bushes let the wheel do", loc="left", fontsize=10, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _figure_search(result, path: str) -> None:
    mass = np.array([h["mass_kg"] for h in result.history])
    viol = np.array([h["worst_violation"] for h in result.history])
    ok = np.array([h["feasible"] for h in result.history])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.scatter(viol[~ok], mass[~ok], s=14, color=MUTED, alpha=0.55, label="candidate: constraint violated")
    ax.scatter(viol[ok], mass[ok], s=18, color=ACCENT, label="candidate: feasible")
    if ok.any():
        ax.axhline(mass[ok].min(), color="#0081be", lw=1, ls=":", label=f"lightest feasible: {mass[ok].min():.2f} kg")
    ax.set_xlabel("worst constraint violation [relative]")
    ax.set_ylabel("corner mass (arms + bushes) [kg]")
    ax.set_title(f"{result.evaluations} candidates, {int(ok.sum())} of them feasible", loc="left", fontsize=10, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _figure_stress(fem: dict, spec, path: str) -> None:
    mesh, res = fem["mesh"], fem["fem"]
    x = mesh.nodes[:, 0] * 1e3
    vm = res.von_mises / 1e6
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    radius = np.hypot(mesh.nodes[:, 1], mesh.nodes[:, 2])
    outer = np.abs(radius - radius.max()) < 1e-3 * radius.max()      # the outer fibre only: where the beam formula is
    band = (x > 0) & outer
    order = np.argsort(x[band])
    ax.plot(x[band][order], vm[band][order], ".", ms=2.4, color=MUTED, alpha=0.7, label="FE nodes, outer fibre")
    L = fem["length_m"]
    stations = np.linspace(0, L, 60) * 1e3
    section = TubeSection(spec.tube_od_m, spec.tube_wall_m)
    moment = fem["root_moment_nm"] * (1.0 - stations / (L * 1e3))
    ax.plot(stations, (moment * section.c_m / section.inertia_m4 + abs(fem["axial_n"]) / section.area_m2) / 1e6,
            color=ACCENT, lw=2, label="beam: N/A + M c / I")
    ax.axhline(spec.yield_pa / spec.safety_factor / 1e6, color="#a23a00", lw=1, ls="--",
               label=f"allowable {spec.yield_pa / spec.safety_factor / 1e6:.0f} MPa (yield / {spec.safety_factor:g})")
    ax.set_xlabel("along the leg from the inner pivots [mm]")
    ax.set_ylabel("von Mises [MPa]")
    ax.set_title(f"lower arm in the {fem.get('case', 'sizing')} case: {fem['elements']} hexahedra", loc="left",
                 fontsize=10, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- 3-D views
def _write_field_view(fem: dict, spec, path_glb: str, path_json: str) -> None:
    """The arm's field in the Studio's viewer: von Mises and displacement on the deformed leg."""
    from pinneapple_data.cae.model import CellBlock, Mesh
    from pinneapple_tools.visualization.studio.fieldview import field_view

    mesh, res = fem["mesh"], fem["fem"]
    cae = Mesh(points=mesh.nodes, blocks=[CellBlock("hexahedron", mesh.elements)])
    fv = field_view(cae, [{"name": "von_mises", "values": res.von_mises, "label": "von Mises", "unit": "Pa"},
                          {"name": "displacement", "values": res.u, "label": "displacement", "unit": "m"}],
                    deform={"displacement": res.u}, title="lower arm, von Mises under the sizing load case")
    with open(path_glb, "wb") as fh:
        fh.write(fv.glb())
    with open(path_json, "w") as fh:
        json.dump(fv.meta, fh)


def _write_cad(spec, path: str) -> None:
    corner_cad(spec).export(path)


# ------------------------------------------------------------------------------------------------- the study
def suspension_study(corner_mass_kg: float = 350.0, bump_g: float = 3.0, toe_target_deg_per_kn: float = 0.45,
                     budget: int = 28, mesh: str = "coarse", out_dir: str = ".") -> dict:
    """Design and size one double-wishbone corner: structural, kinematic and elastokinematic constraints in one loop.

    Typed in words this is "the design and sizing of suspension components, with structural and elastokinematic
    objectives, in a CAD/FEM/multibody loop" — this is that loop, run.

    Parameters
    ----------
    corner_mass_kg : float [kg]
        Sprung mass carried by the corner (the loads scale with it).
    bump_g : float [g]
        The vertical load case's acceleration at the contact patch (a pothole or kerb).
    toe_target_deg_per_kn : float [deg/kN]
        How much the wheel may steer under 1 kN of lateral force.
    budget : int [-]
        Rounds of the pattern search (each round tries every variable both ways).
    mesh : {"coarse", "fine"}
        Density of the finite-element check of the winning arm.
    out_dir : str
        Where the files are written (the Studio's sandbox by default).

    Returns
    -------
    out : dict
        ``design`` (the winning variables), ``metrics`` (named numbers with units), ``checks`` (each one's status,
        value and what it measured), ``files`` and ``limits`` (what the model does not answer).
    """
    os.makedirs(out_dir, exist_ok=True)
    spec = _spec(corner_mass_kg, bump_g)
    result = suspension_design(spec, steps=int(budget))
    design = result.design
    optimum = result.best
    s = optimum["spec"]

    # the fine sweep and the finite-element check of the winner
    sweep = kinematic_sweep(s)
    summary = sweep.summary()
    cases = _cases(s, bump_g)
    sizing = worst_case(s, pose(s), cases=cases)
    n_theta, n_radial, n_len = MESH_DENSITY[mesh]
    fem = fem_lower_arm(s, TubeSection(s.tube_od_m, s.tube_wall_m), sizing, n_theta=n_theta, n_radial=n_radial,
                        n_len=n_len) | {"case": sizing["case"]}
    comp = wheel_compliance(s)

    # files
    files: list[str] = []
    _figure_kinematics(sweep, s, os.path.join(out_dir, "wheels_in_bump.png"))
    _figure_compliance(spec, design, comp, os.path.join(out_dir, "compliance_vs_bush.png"))
    _figure_search(result, os.path.join(out_dir, "design_search.png"))
    _figure_stress(fem, s, os.path.join(out_dir, "arm_stress.png"))
    _write_field_view(fem, s, os.path.join(out_dir, "scene.glb"), os.path.join(out_dir, "scene.json"))
    _write_cad(s, os.path.join(out_dir, "corner.glb"))
    for name in ("wheels_in_bump.png", "compliance_vs_bush.png", "design_search.png", "arm_stress.png",
                 "scene.glb", "scene.json", "corner.glb"):
        files.append(os.path.join(out_dir, name))

    # checks — each one a number the models produced, graded against the constraint it belongs to
    checks = [
        _check("suspension.kinematics_closure", summary["kinematic_closure_residual_m"] < 1e-8,
               summary["kinematic_closure_residual_m"],
               "the closed-loop solve closes over the whole travel sweep (|residual| < 1e-8 m)"),
        _check("suspension.camber_target",
               CONSTRAINTS["camber_gain_deg_per_50mm"]["limit"][0] <= summary["camber_gain_deg_per_50mm"]
               <= CONSTRAINTS["camber_gain_deg_per_50mm"]["limit"][1],
               summary["camber_gain_deg_per_50mm"],
               f"camber gain inside the band {CONSTRAINTS['camber_gain_deg_per_50mm']['limit']} deg per 50 mm"),
        _check("suspension.bump_steer", abs(summary["bump_steer_deg_per_50mm"]) <= 0.10,
               summary["bump_steer_deg_per_50mm"], "|toe change| <= 0.10 deg per 50 mm of bump"),
        _check("suspension.stress", sizing["static_margin"] >= s.safety_factor,
               sizing["sigma_vm_pa"] / 1e6,
               f"outer fibre {sizing['sigma_vm_pa'] / 1e6:.0f} MPa <= {s.yield_pa / s.safety_factor / 1e6:.0f} MPa "
               f"(yield / {s.safety_factor:g}), case: {sizing['case']}"),
        _check("suspension.buckling", sizing["buckling_factor"] >= 1.5, sizing["buckling_factor"],
               "Euler load over the axial force on the compressed member >= 1.5"),
        _check("suspension.fem_agrees", abs(fem["ratio"] - 1.0) <= 0.15, fem["ratio"],
               f"finite elements {fem['sigma_peak_pa'] / 1e6:.0f} MPa against beam theory "
               f"{fem['sigma_beam_pa'] / 1e6:.0f} MPa at the same station"),
        _check("suspension.fem_within_allowable",
               fem["sigma_peak_pa"] <= s.yield_pa / s.safety_factor, fem["sigma_peak_pa"] / 1e6,
               f"finite-element peak {fem['sigma_peak_pa'] / 1e6:.0f} MPa within the allowable"),
        _check("suspension.compliance", comp["toe_deg_per_kn"] <= float(toe_target_deg_per_kn),
               comp["toe_deg_per_kn"], f"toe compliance <= {toe_target_deg_per_kn:g} deg/kN"),
        _check("suspension.mass", optimum["mass_kg"] < result.baseline["mass_kg"], optimum["mass_kg"],
               f"{optimum['mass_kg']:.2f} kg against the nominal corner's {result.baseline['mass_kg']:.2f} kg"),
    ]
    metrics = {
        "mass_kg": optimum["mass_kg"], "baseline_mass_kg": result.baseline["mass_kg"],
        "mass_reduction_pct": 100.0 * (result.baseline["mass_kg"] - optimum["mass_kg"]) / result.baseline["mass_kg"],
        "sigma_mpa": sizing["sigma_vm_pa"] / 1e6, "static_margin": sizing["static_margin"],
        "buckling_factor": sizing["buckling_factor"],
        "fem_peak_mpa": fem["sigma_peak_pa"] / 1e6, "fem_to_beam_ratio": fem["ratio"],
        "fem_elements": float(fem["elements"]),
        "camber_gain_deg_per_50mm": summary["camber_gain_deg_per_50mm"],
        "bump_steer_deg_per_50mm": summary["bump_steer_deg_per_50mm"],
        "scrub_radius_mm": summary["static_scrub_radius_mm"],
        "caster_deg": summary["static_caster_deg"], "kpi_deg": summary["static_kpi_deg"],
        "roll_centre_mm": summary["roll_centre_height_mm"],
        "motion_ratio": summary["motion_ratio"], "wheel_rate_n_per_mm": summary["wheel_rate_n_per_mm"],
        "toe_compliance_deg_per_kn": comp["toe_deg_per_kn"],
        "camber_compliance_deg_per_kn": comp["camber_deg_per_kn"],
        "evaluations": float(result.evaluations),
        **{f"design_{k}": float(v) for k, v in design.as_dict().items()},
    }
    payload = {"design": design.as_dict(), "metrics": metrics, "checks": checks,
               "load_cases": {k: [float(x) for x in v] for k, v in cases.items()},
               "sweep": {k: np.asarray(v).round(6).tolist() for k, v in asdict(sweep).items()},
               "limits": LIMITS}
    with open(os.path.join(out_dir, "suspension_design.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    files.append(os.path.join(out_dir, "suspension_design.json"))
    return {"design": design.as_dict(), "metrics": metrics, "checks": checks, "files": files,
            "sweep_summary": summary, "limits": LIMITS,
            "verdict": "the design meets every constraint" if all(c["status"] == PASS for c in checks)
            else "the design does not meet every constraint (see the checks)"}


LIMITS = [
    "linear elasticity, small displacements: no plasticity, no contact, no weld or heat-affected zone",
    "static load cases only — a margin against yield is not a fatigue life; the corner's inertia and the damper's "
    "reaction are outside the cases",
    "bushings linear and uncoupled; no rate, temperature or friction in the ball joints",
    "one corner of a symmetric axle: no roll stiffness, no steering system, no aerodynamics",
    "the arm is a uniform tube on a clamped root with the ball joint's offset as a couple, not a forged production arm",
]


# ------------------------------------------------------------------------------------------------- verification
def suspension_verify(budget: int = 28, mesh: str = "fine", out_dir: str = ".") -> dict:
    """Re-check the design with a finer mesh and an independent statement of the answer.

    The search is deterministic, so running it again must return the same design — that is the first check. Then the
    arm's finite elements are refined (more rings and stations) and the peak must (a) still agree with beam theory and
    (b) not move between the coarse and the fine mesh: a number that changes with the mesh is a discretisation, not a
    stress.

    Returns
    -------
    out : dict
        ``metrics`` (the refined numbers and their change), ``checks`` (``verify.*``), ``files``.
    """
    os.makedirs(out_dir, exist_ok=True)
    spec = _spec(350.0, 3.0)
    result = suspension_design(spec, steps=int(budget))
    s = result.best["spec"]
    sizing = worst_case(s, pose(s), cases=_cases(s, 3.0))
    n_theta, n_radial, n_len = MESH_DENSITY[mesh]
    fine = fem_lower_arm(s, TubeSection(s.tube_od_m, s.tube_wall_m), sizing, n_theta=n_theta, n_radial=n_radial,
                         n_len=n_len) | {"case": sizing["case"]}
    c_theta, c_radial, c_len = MESH_DENSITY["coarse"]
    coarse = fem_lower_arm(s, TubeSection(s.tube_od_m, s.tube_wall_m), sizing, n_theta=c_theta, n_radial=c_radial,
                           n_len=c_len)
    change = abs(fine["sigma_peak_pa"] - coarse["sigma_peak_pa"]) / max(coarse["sigma_peak_pa"], 1e-12)
    again = suspension_design(spec, steps=int(budget))
    same = again.design.as_dict() == result.design.as_dict()
    sweep = kinematic_sweep(s)
    summary = sweep.summary()
    checks = [
        _check("verify.reproducible", same, int(same),
               "the deterministic search returns the same design when it is run again"),
        _check("verify.mesh_independence", change <= 0.08, change,
               f"the FE peak moves {100 * change:.1f} % between {coarse['elements']} and {fine['elements']} hexahedra"),
        _check("verify.fem_agrees", abs(fine["ratio"] - 1.0) <= 0.12, fine["ratio"],
               f"refined finite elements {fine['sigma_peak_pa'] / 1e6:.0f} MPa against beam theory "
               f"{fine['sigma_beam_pa'] / 1e6:.0f} MPa"),
        _check("verify.closure", summary["kinematic_closure_residual_m"] < 1e-8,
               summary["kinematic_closure_residual_m"], "the linkage closes over the travel sweep"),
    ]
    _figure_stress(fine, s, os.path.join(out_dir, "arm_stress_refined.png"))
    with open(os.path.join(out_dir, "suspension_verify.json"), "w") as fh:
        json.dump({"design": again.design.as_dict(), "checks": checks,
                   "coarse_mpa": coarse["sigma_peak_pa"] / 1e6, "fine_mpa": fine["sigma_peak_pa"] / 1e6}, fh,
                  indent=1)
    metrics = {"fem_peak_mpa": fine["sigma_peak_pa"] / 1e6, "coarse_peak_mpa": coarse["sigma_peak_pa"] / 1e6,
               "peak_change_pct": 100 * change, "fem_to_beam_ratio": fine["ratio"],
               "fem_elements": float(fine["elements"]), "beam_mpa": fine["sigma_beam_pa"] / 1e6,
               "mass_kg": result.best["mass_kg"], "same_design": float(same)}
    return {"metrics": metrics, "checks": checks,
            "files": [os.path.join(out_dir, "arm_stress_refined.png"),
                      os.path.join(out_dir, "suspension_verify.json")],
            "verdict": "the refined model confirms the design" if all(c["status"] == PASS for c in checks)
            else "the refined model does not confirm the design (see the checks)"}
