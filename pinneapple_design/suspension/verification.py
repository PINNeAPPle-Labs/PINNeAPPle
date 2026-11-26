"""What this corner's models are allowed to claim, and the measurement behind each claim.

A benchmark that reports numbers without saying how wrong they are is a brochure. Every case here has an answer that
is known **independently of the code being checked** -- a linkage whose camber cannot change, the same mechanism solved
by two different formulations, a bush limit where the compliance must vanish, a tube whose deflection Timoshenko
already wrote down, and the same leg solved by an external solver -- and it returns what was measured, not a verdict.

    from pinneapple_design.suspension.verification import run_all, summary
    report = run_all()                       # external=True runs CalculiX too; minutes
    for row in summary(report):
        print(row["case"], row["measured"], row["reference"], row["unit"])

The values are used twice: in ``tests/test_suspension_design.py`` (as assertions, with the tolerances stated there) and
in :func:`pinneapple_design.suspension.study.suspension_verify`, which prints them next to the numbers they qualify.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np

from .compliance import wheel_compliance
from .hardpoints import SuspensionSpec, front_corner
from .kinematics import pose, pose_from_angles
from .loads import corner_load_cases, worst_case
from .structure import TubeSection, fem_lower_arm

__all__ = ["kinematics_agreement", "parallelogram", "rigid_bushes", "cantilever", "arm_models", "run_all", "summary",
           "SWEEP_MM"]

#: the travel positions every case is measured at [mm], bump positive
SWEEP_MM = (-50.0, -25.0, 0.0, 25.0, 50.0)


def _secant(spec: SuspensionSpec, half_span_mm: float, what: str) -> float:
    """``d(angle)/d(travel)`` per 50 mm, as the central secant about the design height."""
    a = pose(spec, -half_span_mm * 1e-3)
    b = pose(spec, +half_span_mm * 1e-3)
    da = getattr(b, what) - getattr(a, what)
    return float(da / (2.0 * half_span_mm * 1e-3) * 0.050)


def kinematics_agreement(travels_mm=SWEEP_MM, design: SuspensionSpec | None = None) -> dict[str, object]:
    """One mechanism, two formulations: the pose from the three closure unknowns against the pose from three joint
    angles.

    :func:`pose` solves for the upright's lateral shift and rotation; :func:`pose_from_angles` places each ball joint on
    its own circle and spins the upright about the kingpin. They share the hardpoints and the travel, and nothing else,
    so the difference between them is the kinematics' numerical error -- and the closure residual each reports measures
    the same thing a third way. The closed-form answer is zero difference.
    """
    spec = design or front_corner()
    rows, worst_camber, worst_toe, worst_point = [], 0.0, 0.0, 0.0
    for travel_mm in travels_mm:
        a = pose(spec, float(travel_mm) * 1e-3)
        b = pose_from_angles(spec, float(travel_mm) * 1e-3, seed=(a.theta, a.phi, a.psi))
        d_camber = abs(a.camber_deg - b.camber_deg)
        d_toe = abs(a.toe_deg - b.toe_deg)
        d_point = float(np.abs(a.wheel_centre - b.wheel_centre).max())
        worst_camber = max(worst_camber, d_camber)
        worst_toe = max(worst_toe, d_toe)
        worst_point = max(worst_point, d_point)
        rows.append({"travel_mm": float(travel_mm), "d_camber_deg": d_camber, "d_toe_deg": d_toe,
                     "d_wheel_centre_m": d_point})
    return {"worst_camber_deg": worst_camber, "worst_toe_deg": worst_toe, "worst_wheel_centre_m": worst_point,
            "rows": rows, "reference": 0.0}


def parallelogram(half_span_mm: float = 10.0) -> dict[str, object]:
    """A linkage that cannot gain camber: equal, parallel wishbones, and the tie rod square to the travel.

    Where the two A-arms are equal, parallel and offset by one rigid vector, the upright does not rotate as it travels,
    so the camber is the static camber at every wheel position -- a closed-form zero that no tolerance can be argued
    with. The tie rod is the one part that can spoil it: the upright's translation changes the tie rod's length, and
    the spin that restores it tilts the wheel about an inclined kingpin. Pointing the tie rod perpendicular to that
    translation removes the effect to first order, which is what this case builds; what remains grows with the square
    of the travel, and :func:`run_all` reports it beside the first-order number so the two are not confused.

    The corner is the reference one with its upper arm moved onto the parallelogram and its rack point moved onto the
    perpendicular: a verification scaffold, not a car.
    """
    spec = front_corner()
    offset = spec.uca_front - spec.lca_front
    par = replace(spec, name="parallelogram", uca_front=spec.lca_front + offset,
                  uca_rear=spec.lca_rear + offset, uca_outer=spec.lca_outer + offset)
    travel_dir = np.cross(par.lca_axis, par.lca_outer - par.lca_front)
    travel_dir = travel_dir / np.linalg.norm(travel_dir)
    rod = par.tierod_outer - par.tierod_inner
    length = float(np.linalg.norm(rod))
    direction = rod / length
    square = direction - float(direction @ travel_dir) * travel_dir
    par = replace(par, tierod_inner=par.tierod_outer - length * square / np.linalg.norm(square))
    rod = par.tierod_outer - par.tierod_inner
    full = pose(par, 0.050)
    return {"camber_gain_deg_per_50mm": _secant(par, half_span_mm, "camber_deg"),
            "bump_steer_deg_per_50mm": _secant(par, half_span_mm, "toe_deg"),
            "half_span_mm": float(half_span_mm),
            "camber_at_full_bump_deg": float(full.camber_deg),
            "camber_change_full_bump_deg": float(full.camber_deg - pose(par, 0.0).camber_deg),
            "tie_rod_perpendicular": float(rod / np.linalg.norm(rod) @ travel_dir),
            "tie_rod_length_m": float(np.linalg.norm(rod)),
            "reference": 0.0}


def rigid_bushes(design: SuspensionSpec | None = None) -> dict[str, object]:
    """The limit where the bushes stop being rubber: the compliance must vanish, and it must vanish as ``1 / rate``.

    The first-order model says the wheel's elastic response is proportional to the bush compliance, so multiplying
    every rate by a thousand divides the steer by a thousand. Both statements have a closed-form answer -- zero, and
    the ratio of the rates -- and the case reports them against what the mode expansion returns.
    """
    spec = design or front_corner()
    soft = wheel_compliance(spec)["toe_deg_per_kn"]
    rate = spec.bushing_lca.radial_n_per_m
    stiff = wheel_compliance(spec.with_bushings(1000.0 * rate))["toe_deg_per_kn"]
    return {"toe_deg_per_kn_nominal": float(soft), "toe_deg_per_kn_stiff": float(stiff),
            "rate_ratio": 1000.0, "measured_ratio": float(soft / stiff) if stiff else float("inf"),
            "reference": 0.0}


def cantilever(*, order: int | None = None, size: float | None = None, workdir: str | None = None) -> dict:
    """The hollow tube in CalculiX against Timoshenko's cantilever -- an external solver on a case whose answer is a
    formula.

    The displacement it returns is what the arm's own finite elements are built the same way as, so this is the check
    that the *pipeline* (geometry, mesh, deck, solver, post-processing) is right, not only the formula. Needs ``gmsh``
    and ``ccx``.
    """
    from . import calculix

    kw = {}
    if order is not None:
        kw["order"] = order
    if size is not None:
        kw["size"] = size
    out = calculix.fem_tube_calculix(workdir=workdir or os.path.join(os.getcwd(), "_suspension_tube"), **kw)
    out["reference"] = 0.0
    return out


def arm_models(design: SuspensionSpec | None = None, *, workdir: str | None = None) -> dict[str, object]:
    """The sizing case on one leg, solved three ways: beam theory, this library's hexahedra, and CalculiX.

    The in-house finite elements and the beam formula already agree in :func:`pinneapple_design.suspension.structure.
    fem_lower_arm`; what this adds is a second **program** on the same force and the same root moment. Two
    discretisations that agree with each other and with the closed form is the strongest statement the structural
    model can make about itself, and the disagreement is reported as measured. Needs ``gmsh`` and ``ccx``.
    """
    from . import calculix

    spec = design or front_corner()
    section = TubeSection(spec.tube_od_m, spec.tube_wall_m)
    sizing = worst_case(spec, pose(spec), cases=corner_load_cases(spec))
    hexa = fem_lower_arm(spec, section, sizing)
    work = workdir or os.path.join(os.getcwd(), "_suspension_leg")
    ext = calculix.fem_lower_arm_calculix(spec, section, sizing, workdir=work)
    beam = float(hexa["sigma_beam_pa"])
    peaks = {"beam": beam, "hex": float(hexa["sigma_peak_pa"]), "calculix": float(ext["sigma_peak_pa"])}
    spread = (max(peaks.values()) - min(peaks.values())) / beam
    return {"case": sizing["case"], "sigma_pa": peaks, "spread_rel": float(spread),
            "hex_to_beam": float(peaks["hex"] / beam), "calculix_to_beam": float(peaks["calculix"] / beam),
            "hex_elements": int(hexa["elements"]), "calculix_elements": int(ext["elements"]),
            "leg_length_m": float(sizing["leg_length_m"]), "reference": 0.0}


def run_all(*, external: bool = True, workdir: str | None = None) -> dict[str, object]:
    """Every case, in one report. ``external=False`` skips the two that need CalculiX (and records why)."""
    work = workdir or os.path.join(os.getcwd(), "_suspension_verification")
    os.makedirs(work, exist_ok=True)
    report: dict[str, object] = {"kinematics_agreement": kinematics_agreement(),
                                 "parallelogram": parallelogram(),
                                 "rigid_bushes": rigid_bushes()}
    from . import calculix

    if external and calculix.available():
        report["cantilever"] = cantilever(workdir=os.path.join(work, "tube"))
        report["arm_models"] = arm_models(workdir=os.path.join(work, "leg"))
    else:
        why = "skipped: ccx is not installed (native or Docker)" if external else "skipped: external=False"
        report["cantilever"] = {"skipped": why}
        report["arm_models"] = {"skipped": why}
    return report


def summary(report: dict[str, object]) -> list[dict[str, object]]:
    """The report as one table: what was measured, against what, in which unit, and how it came out."""
    rows: list[dict[str, object]] = []
    k = report.get("kinematics_agreement") or {}
    if k:
        rows.append({"case": "two kinematics formulations", "measured": 1e6 * k["worst_camber_deg"],
                     "reference": 0.0, "unit": "micro-deg of camber", "detail": "constraints vs joint angles"})
    p = report.get("parallelogram") or {}
    if p:
        rows.append({"case": "parallelogram, first order", "measured": p["camber_gain_deg_per_50mm"], "reference": 0.0,
                     "unit": "deg/50 mm", "detail": f"central secant over +/- {p['half_span_mm']:g} mm"})
        rows.append({"case": "parallelogram, full bump", "measured": p["camber_change_full_bump_deg"],
                     "reference": 0.0, "unit": "deg", "detail": "the tie rod's second-order coupling"})
    rb = report.get("rigid_bushes") or {}
    if rb:
        rows.append({"case": "rigid bushes", "measured": rb["toe_deg_per_kn_stiff"], "reference": 0.0,
                     "unit": "deg/kN", "detail": f"rates x{rb['rate_ratio']:g}; measured ratio {rb['measured_ratio']:.1f}"})
    c = report.get("cantilever") or {}
    if c and "skipped" not in c:
        rows.append({"case": "tube in CalculiX vs Timoshenko", "measured": 100.0 * c["error_rel"], "reference": 0.0,
                     "unit": "% of deflection", "detail": f"order {c['order']}, {c['elements']} elements"})
    am = report.get("arm_models") or {}
    if am and "skipped" not in am:
        rows.append({"case": "leg, three models", "measured": 100.0 * am["spread_rel"], "reference": 0.0,
                     "unit": "% between beam, hexahedra and CalculiX",
                     "detail": f"beam {am['sigma_pa']['beam'] / 1e6:.0f} MPa, hex "
                               f"{am['sigma_pa']['hex'] / 1e6:.0f} MPa, CalculiX {am['sigma_pa']['calculix'] / 1e6:.0f} MPa"})
    for key in ("cantilever", "arm_models"):
        skipped = (report.get(key) or {}).get("skipped")
        if skipped:
            rows.append({"case": key, "measured": float("nan"), "reference": 0.0, "unit": "-", "detail": skipped})
    return rows
