"""What this benchmark is allowed to claim, and the measurement behind each claim.

A benchmark that reports numbers without saying how wrong they are is a brochure. Every entry below is a case whose
answer is known independently of the code being checked -- a closed-form beam, the same mechanism solved two ways, a
degenerate linkage whose camber gain must be exactly zero, a limit where the compliant answer must vanish -- and it
returns what was measured, not a verdict.

    report = run_all()            # {"kinematics_agreement": {...}, "tube_cantilever": {...}, ...}

The values are used twice: in the tests (as assertions with tolerances that are stated there) and in the study's
report, where the same numbers are printed next to the results they qualify.
"""
from __future__ import annotations

import os

import numpy as np

from .hardpoints import BASELINE, Corner, corner
from .kinematics import metrics, solve_pose, solve_pose_angles

__all__ = ["kinematics_agreement", "parallelogram", "anticipation", "rigid_limit", "run_all"]

SWEEP_MM = (-50.0, -25.0, 0.0, 25.0, 50.0)


def kinematics_agreement(travels_mm=SWEEP_MM, design=None) -> dict[str, object]:
    """Two solvers, one mechanism: the pose from the five constraint residuals against the pose from three joint
    angles. They are independent formulations, so the difference between them measures the kinematics' numerical
    error -- and the constraint residual of the constraint solver measures it independently of of both."""
    c = corner(design or BASELINE)
    rows, worst = [], 0.0
    for tr in travels_mm:
        a = solve_pose(c, tr / 1e3)
        b = solve_pose_angles(c, tr / 1e3)
        d_cam = abs(a.camber_deg() - b.camber_deg())
        d_toe = abs(a.toe_deg() - b.toe_deg())
        d_pts = float(np.abs(a.points["wheel_centre"] - b.points["wheel_centre"]).max())
        worst = max(worst, d_cam, d_toe)
        rows.append({"travel_mm": float(tr), "d_camber_deg": d_cam, "d_toe_deg": d_toe, "d_point_m": d_pts,
                     "residual_constraints_m": a.residual, "residual_angles_m": b.residual})
    return {"rows": rows, "max_deg": worst,
            "max_residual_m": max(max(r["residual_constraints_m"], r["residual_angles_m"]) for r in rows)}


def _parallelogram() -> Corner:
    """Four equal, parallel links: the upright can only translate, so its orientation cannot change.

    Lower arm: axis along x through the origin, ball joint 400 mm outboard. Upper arm: the same axis 300 mm up, its
    ball joint the same 400 mm outboard, so the coupler (the upright) is vertical and both arms are equal and parallel
    -- the textbook case whose camber gain is exactly zero.

    The tie rod's inner end is put **on the lower arm's axis**, which is what keeps the tie-rod equation out of the
    way instead of over-constraining it: in this linkage every point of the upright travels on a circle about that
    axis, so its distance to a fixed point of the axis is invariant, and the closed form stays exact rather than
    approximately exact. (Pin the inner end anywhere else and the rod fights the linkage, because a tie rod whose
    outer end sits on the lower ball joint would remove the freedom the wishbones already spent.)
    """
    lbj = np.array([0.0, 0.400, 0.0])
    return Corner(
        lca_front=np.array([-0.2, 0.0, 0.0]), lca_rear=np.array([0.2, 0.0, 0.0]),
        lbj=lbj,
        uca_front=np.array([-0.2, 0.0, 0.300]), uca_rear=np.array([0.2, 0.0, 0.300]),
        ubj=np.array([0.0, 0.400, 0.300]),
        tie_inner=np.array([0.0, 0.0, 0.0]), tie_outer=np.array([0.0, 0.500, 0.050]),
        wheel_centre=np.array([0.0, 0.500, 0.300]),
        damper_lower=np.array([0.0, 0.200, 0.0]), damper_upper=np.array([0.0, 0.050, 0.500]),
        lc_od=0.03, lc_wall=0.004, uc_od=0.02, wheel_radius=0.31, spring_rate=100.0, design={},
    )


def parallelogram(travels_mm=SWEEP_MM) -> dict[str, object]:
    """Closed-form check: an equal, parallel wishbone pair translates its upright, so camber and toe cannot change."""
    c = _parallelogram()
    rows, worst_cam, worst_toe, worst_rod = [], 0.0, 0.0, 0.0
    ref_rod = float(np.linalg.norm(c.tie_outer - c.tie_inner))
    for tr in travels_mm:
        p = solve_pose(c, tr / 1e3)
        rod = float(np.linalg.norm(p.points["tie_outer"] - c.tie_inner))
        rows.append({"travel_mm": float(tr), "camber_deg": p.camber_deg(), "toe_deg": p.toe_deg(),
                     "wheel_dy_mm": float(p.wheel_centre[1] - c.wheel_centre[1]) * 1e3,
                     "tie_rod_drift_m": rod - ref_rod})
        worst_cam, worst_toe = max(worst_cam, abs(p.camber_deg())), max(worst_toe, abs(p.toe_deg()))
        worst_rod = max(worst_rod, abs(rod - ref_rod))
    return {"rows": rows, "max_camber_deg": worst_cam, "max_toe_deg": worst_toe, "max_tie_rod_drift_m": worst_rod}


def anticipation(design=None, h_mm: float = 1.0) -> dict[str, object]:
    """The arc is smooth: the camber and toe *rates* measured on the bump side and on the rebound side of the static
    pose must agree to O(h), which they cannot do if the branch or the sign of a definition is wrong."""
    c = corner(design or BASELINE)
    p0 = solve_pose(c, 0.0)
    up, dn = solve_pose(c, h_mm / 1e3, p0.constrain()), solve_pose(c, -h_mm / 1e3, p0.constrain())
    rc = ((up.camber_deg() - p0.camber_deg()) - (p0.camber_deg() - dn.camber_deg())) / 2
    rt = ((up.toe_deg() - p0.toe_deg()) - (p0.toe_deg() - dn.toe_deg())) / 2
    return {"camber_rate_asymmetry_deg_per_mm": abs(rc), "toe_rate_asymmetry_deg_per_mm": abs(rt),
            "h_mm": h_mm}


def rigid_limit(design=None) -> dict[str, object]:
    """The elastokinematic model's limit: infinitely stiff joints must give exactly zero compliance steer, and the
    6 x 6 stiffness must be symmetric with a positive smallest eigenvalue (a body on springs, not a mechanism)."""
    from .elastokinematics import elastokinematics
    c = corner(design or BASELINE)
    soft = {"lower": np.zeros((3, 3)), "upper": np.zeros((3, 3))}
    ek = elastokinematics(c, soft)
    stiff = elastokinematics(c, soft, bush={"wishbone_n_per_m": 1e15, "tie_rod_n_per_m": 1e15})
    return {"steer_deg_per_kN_finite_bush": ek["compliance_steer_deg_per_kN_lateral"],
            "steer_deg_per_kN_rigid": stiff["compliance_steer_deg_per_kN_lateral"],
            "lateral_compliance_mm_per_kN_finite_bush": ek["lateral_compliance_mm_per_kN"],
            "lateral_compliance_mm_per_kN_rigid": stiff["lateral_compliance_mm_per_kN"],
            "min_eigenvalue": ek["body_stiffness_min_eig"]}


def run_all(*, fem: bool = True, workdir: str | None = None, log=print) -> dict[str, object]:
    """Every check, with the FEM ones only when ``fem`` (they need ``ccx`` and a few minutes)."""
    out: dict[str, object] = {}
    log("kinematics: two solvers on the same mechanism")
    out["kinematics_agreement"] = kinematics_agreement()
    log("kinematics: parallelogram linkage (closed form: zero camber gain)")
    out["parallelogram"] = parallelogram()
    log("kinematics: rate symmetry about the static pose")
    out["anticipation"] = anticipation()
    log("elastokinematics: the rigid limit")
    out["rigid_limit"] = rigid_limit()
    if fem:
        from .structure import arm, mesh_convergence, tube_cantilever
        work = workdir or os.path.join(os.getcwd(), "_suspension_verification")
        log("FEM: hollow tube cantilever against Timoshenko (production mesh and reference mesh)")
        out["tube_cantilever_production"] = tube_cantilever(size=0.014, order=1,
                                                           workdir=os.path.join(work, "tube_prod"))
        out["tube_cantilever_reference"] = tube_cantilever(size=0.012, order=2,
                                                          workdir=os.path.join(work, "tube_ref"))
        log("FEM: mesh convergence of the lower wishbone (GCI)")
        out["mesh_convergence"] = mesh_convergence(arm(corner(BASELINE), "lower"),
                                                   os.path.join(work, "arm_gci"))
    return out


def summary(report: dict[str, object]) -> list[dict[str, object]]:
    """The report rows: what was checked, what was measured, against what."""
    rows = []
    k = report.get("kinematics_agreement") or {}
    if k:
        rows.append({"check": "kinematics: two independent solver formulations agree",
                     "measured": f"{k['max_deg']:.2e} deg (constraint residual {k['max_residual_m']:.1e} m)",
                     "limit": "<= 1e-6 deg"})
    p = report.get("parallelogram") or {}
    if p:
        rows.append({"check": "kinematics: parallelogram linkage has zero camber gain (closed form)",
                     "measured": f"max |camber| {p['max_camber_deg']:.2e} deg, |toe| {p['max_toe_deg']:.2e} deg",
                     "limit": "<= 1e-9 deg (machine precision)"})
    a = report.get("anticipation") or {}
    if a:
        rows.append({"check": "kinematics: bump and rebound rates agree at the static pose",
                     "measured": f"{a['camber_rate_asymmetry_deg_per_mm']:.2e} deg/mm",
                     "limit": "O(h^2) of the second derivative, <= 1e-4"})
    r = report.get("rigid_limit") or {}
    if r:
        rows.append({"check": "elastokinematics: infinitely stiff joints give zero compliance",
                     "measured": f"{r['steer_deg_per_kN_rigid']:.2e} deg/kN",
                     "limit": "<= 1e-6 deg/kN"})
    for key, label in (("tube_cantilever_production", "FEM: wishbone tube against Timoshenko, production mesh"),
                       ("tube_cantilever_reference", "FEM: wishbone tube against Timoshenko, reference mesh")):
        t = report.get(key) or {}
        if t:
            rows.append({"check": label,
                         "measured": f"{t['deflection_m'] * 1e3:.4f} mm vs {t['timoshenko_m'] * 1e3:.4f} mm "
                                     f"({t['error_rel'] * 100:+.1f} %)",
                         "limit": "the production mesh is expected to be stiff; the reference mesh <= 1 %"})
    g = report.get("mesh_convergence") or {}
    rows.append({"check": "FEM: grid convergence of the lower wishbone (GCI)",
                 "measured": _gci_text(g), "limit": "reported, not assumed"})
    return rows


def _gci_text(g: dict) -> str:
    try:
        v = g["gci"]["results"]["vm"]
        u = g["gci"]["results"]["u"]
        return (f"stress GCI_fine {v['gci_fine'] * 100:.1f} % (p={v.get('p', float('nan')):.2f}), "
                f"deflection GCI_fine {u['gci_fine'] * 100:.1f} %")
    except Exception:                                                  # pragma: no cover - shape of the helper
        return "not measured"

