"""Closed-loop wing design: the vortex lattice and a friction drag build-up, a learned model only when it pays off,
optimisation of the planform, and the design re-verified (#1181).

Design variables: aspect ratio, taper ratio and tip twist (washout) at a fixed wing area. For each design:

* **Vortex lattice** (:mod:`.vlm`, wing alone): the lift is linear in the angle of attack, so one solve gives the
  cruise and the landing condition: angle for the required lift coefficient, induced drag in the Trefftz plane, span
  efficiency ``e = CL^2 / (pi AR CDi)`` and the spanwise loading.
* **Profile drag**: turbulent flat-plate friction ``Cf = 0.455 / (log10 Re)^2.58 / (1 + 0.144 M^2)^0.65`` on the mean
  chord, the form factor ``FF = (1 + 0.6/(x/c)_m t/c + 100 (t/c)^4) (1.34 M^0.18 cos^0.28 sweep)`` and the wetted area
  ``S (1.977 + 0.52 t/c)`` (Raymer, *Aircraft Design: A Conceptual Approach*, 6th ed., eqs. 12.27, 12.30, 7.9).
* **Root bending moment** of the half wing from the lattice's loading, ``M = q sum cl c y dy`` over the strips at the
  angle that pulls ``n_load`` g at cruise speed (a manoeuvre: the loading shape follows the angle), the structural
  limit.
* **Stall margin**: the largest local ``cl`` at the landing lift coefficient against the section's ``cl_max`` (a
  measured value of the section, given by the caller): the station that stalls first, and whether it is the tip.

The objective is the cruise drag ``q S (CD0 + CDi)``; the limits are the root bending moment, the stall margin and
the span. The design is re-verified with a finer lattice (convergence), against Munk's bound ``e <= 1`` and against
Prandtl's lifting line on the same planform. An OpenFOAM polar of the section is the next fidelity
(:mod:`.verify`); without the solver in the environment it is reported as not run.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import vlm
from .airframe import Airframe
from .geometry import REFERENCE, properties


@dataclass
class WingSpec:
    weight_N: float = 10_500.0             # aircraft weight (light 4-seat aircraft class)
    area_m2: float = 16.2
    cruise_V: float = 62.0                 # m/s
    cruise_rho: float = 1.007              # kg/m^3 (ISA, 2 000 m)
    cruise_T: float = 275.15               # K
    landing_V: float = 31.0                # m/s, approach
    landing_rho: float = 1.225
    section: str = "NACA 2412"
    cl_max: float = 1.6                    # section cl_max (Abbott & von Doenhoff 1959, NACA 2412, Re ~ 6e6)
    stall_margin: float = 0.95             # largest local cl at landing <= margin * cl_max
    n_load: float = 3.8                    # limit load factor (FAR 23 normal category)
    M_root_max: float | None = 60_000.0    # N m, half-wing root bending limit at n_load
    span_max: float | None = 12.0          # m
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "aspect_ratio": (5.0, 12.0), "taper": (0.3, 1.0), "twist": (-6.0, 0.0)})
    assumptions: list[str] = field(default_factory=lambda: [
        "wing alone: the fuselage and tail interference are not in the lattice",
        "inviscid lattice: viscous decambering lowers the real lift slope; profile drag from the friction build-up",
        "section cl_max is the measured two-dimensional value; the local cl check finds the station that stalls first",
        "constant section along the span, no sweep, no dihedral"])


def _profile_drag(spec: WingSpec, af: Airframe) -> dict[str, float]:
    p = properties(REFERENCE[spec.section])
    tc, xm = float(p["thickness"]), float(p["thickness_at"])
    a = math.sqrt(1.4 * 287.05 * spec.cruise_T)
    M = spec.cruise_V / a
    mu = 1.458e-6 * spec.cruise_T ** 1.5 / (spec.cruise_T + 110.4)
    Re = spec.cruise_rho * spec.cruise_V * af.mac / mu
    Cf = 0.455 / math.log10(Re) ** 2.58 / (1 + 0.144 * M * M) ** 0.65
    FF = (1 + 0.6 / xm * tc + 100 * tc ** 4) * (1.34 * M ** 0.18)
    Swet = spec.area_m2 * (1.977 + 0.52 * tc)
    return {"CD0": Cf * FF * Swet / spec.area_m2, "Re": Re, "Cf": Cf, "FF": FF, "mach": M, "t_c": tc}


def evaluate(spec: WingSpec, x: dict[str, float], *, nc: int = 5, ns: int = 14) -> dict[str, Any]:
    af = Airframe(wing_area=spec.area_m2, aspect_ratio=float(x["aspect_ratio"]), taper=float(x["taper"]),
                  twist=float(x["twist"]), incidence=0.0, dihedral=0.0, sweep_le=0.0,
                  section=REFERENCE[spec.section].copy())
    sol = vlm.solve(af, x_ref=af.wing_x, nc=nc, ns_wing=ns, tail=False)
    q = 0.5 * spec.cruise_rho * spec.cruise_V ** 2
    CL = spec.weight_N / (q * spec.area_m2)
    a = (CL - sol.CL[0]) / sol.CL[1]
    c = sol.coefficients(a, 0.0)
    pd = _profile_drag(spec, af)
    # landing
    ql = 0.5 * spec.landing_rho * spec.landing_V ** 2
    CL_l = spec.weight_N / (ql * spec.area_m2)
    a_l = (CL_l - sol.CL[0]) / sol.CL[1]
    cl_l = sol.strip_cl(a_l, 0.0)
    k = int(np.argmax(cl_l))
    # root bending of the half wing pulling n_load g at cruise speed (the loading at that angle, twist included)
    a_n = (spec.n_load * CL - sol.CL[0]) / sol.CL[1]
    cl_n = sol.strip_cl(a_n, 0.0)
    half = sol.L.yc > 0
    M = q * float(np.sum(cl_n[half] * sol.L.chord[half] * sol.L.yc[half] * sol.L.dy[half]))
    CDi = c["CDi"]
    return {"CL": CL, "alpha_deg": math.degrees(a), "CDi": CDi, "CD0": pd["CD0"], "CD": pd["CD0"] + CDi,
            "drag_N": q * spec.area_m2 * (pd["CD0"] + CDi), "e": CL ** 2 / (math.pi * af.aspect_ratio * CDi),
            "M_root_Nm": M, "span_m": af.span, "cl_local_max_landing": float(cl_l[k]),
            "stall_station": float(abs(sol.L.yc[k]) / (af.span / 2)), "CL_landing": CL_l,
            "CL_alpha": float(sol.CL[1]), "profile": pd}


def lifting_line(spec: WingSpec, x: dict[str, float], n: int = 120) -> dict[str, float]:
    """Prandtl's lifting line on the same planform (straight wing, linear twist, section slope 2 pi, the section's
    zero-lift angle from the thin-airfoil value of the lattice): lift slope and span efficiency."""
    AR, lam, tw = float(x["aspect_ratio"]), float(x["taper"]), math.radians(float(x["twist"]))
    th = np.linspace(0, math.pi, n + 2)[1:-1]
    b = 1.0
    S = b * b / AR
    cr = 2 * S / (b * (1 + lam))
    eta = np.abs(np.cos(th))
    c = cr * (1 - (1 - lam) * eta)
    m = np.arange(1, n + 1)
    Smat = np.sin(np.outer(th, m))
    A = Smat * (4 * b / (2 * math.pi * c))[:, None] + Smat * m[None, :] / np.sin(th)[:, None]
    a_unit = np.linalg.solve(A, np.ones(n))
    a_tw = np.linalg.solve(A, tw * eta)
    CLa = math.pi * AR * a_unit[0]
    out = {"CL_alpha": CLa}
    # span efficiency at the cruise CL: An = alpha * a_unit + a_tw
    CL = spec.weight_N / (0.5 * spec.cruise_rho * spec.cruise_V ** 2 * spec.area_m2)
    alpha = (CL / (math.pi * AR) - a_tw[0]) / a_unit[0]
    An = alpha * a_unit + a_tw
    out["e"] = float(1 / (1 + np.sum(m[1:] * (An[1:] / An[0]) ** 2)))
    return out


def _limits(spec: WingSpec, y: dict[str, Any]) -> dict[str, float]:
    g = {"stall": spec.stall_margin * spec.cl_max - y["cl_local_max_landing"]}
    if spec.M_root_max is not None:
        g["root_bending"] = (spec.M_root_max - y["M_root_Nm"]) / spec.M_root_max
    if spec.span_max is not None:
        g["span"] = spec.span_max - y["span_m"]
    return g


def design_wing(spec: WingSpec, *, maxiter: int = 30, popsize: int = 10, seed: int = 0) -> dict[str, Any]:
    from scipy.optimize import NonlinearConstraint, differential_evolution
    names = list(spec.bounds)
    bounds = [spec.bounds[k] for k in names]
    centre = {k: 0.5 * (a + b) for k, (a, b) in zip(names, bounds, strict=True)}
    t0 = time.perf_counter()
    for _ in range(3):
        y0 = evaluate(spec, centre)
    solve_s = (time.perf_counter() - t0) / 3
    n_eval = popsize * len(names) * (maxiter + 1)
    rec: dict[str, Any] = {"recommended": ["none"], "reasoning": "recommendation unavailable"}
    try:
        from pinneapple_analysis.verification.architecture_recommendation import (
            recommend_architecture,
        )
        r = recommend_architecture(n_high_fidelity_simulations=0, needs_parameter_generalization=True,
                                   geometry_varies=True, has_analytical_solver=True, solve_seconds=solve_s,
                                   n_evaluations=n_eval, output_kind="scalar", n_inputs=len(names))
        rec = {"recommended": list(r.recommended), "reasoning": r.reasoning, "decision_path": list(r.decision_path),
               "checks": list(r.checks)}
    except (ImportError, TypeError) as e:
        rec["reasoning"] = f"recommendation unavailable: {e}"
    cache: dict[tuple, dict] = {}

    def run(z):
        key = tuple(np.round(z, 10))
        if key not in cache:
            cache[key] = evaluate(spec, dict(zip(names, z, strict=True)))
        return cache[key]
    g0 = len(_limits(spec, y0))
    res = differential_evolution(lambda z: run(z)["drag_N"] / y0["drag_N"], bounds, seed=seed, maxiter=maxiter,
                                 popsize=popsize, tol=1e-8, polish=False, init="sobol",
                                 constraints=NonlinearConstraint(lambda z: np.array(list(_limits(spec, run(z))
                                                                                          .values())),
                                                                 np.zeros(g0), np.inf))
    x = dict(zip(names, map(float, res.x), strict=True))
    y = evaluate(spec, x)
    fine = evaluate(spec, x, nc=8, ns=28)
    ll = lifting_line(spec, x)
    base = evaluate(spec, {"aspect_ratio": 7.5, "taper": 0.7, "twist": -3.0})        # the library's light aircraft
    g = _limits(spec, y)
    checks = []

    def chk(code, ok, detail, value=None):
        ok = None if ok is None else bool(ok)
        checks.append({"code": code, "status": "pass" if ok is True else ("fail" if ok is False else "inconclusive"),
                       "detail": detail, "value": value})
    for k, v in g.items():
        chk(f"design.{k}", v >= -1e-9, f"margin {v:.4g} at the design, from the lattice", v)
    dcd = abs(fine["CDi"] - y["CDi"]) / y["CDi"]
    chk("model.converged", dcd < 0.01, f"induced drag changes by {dcd:.2%} with a lattice 1.6 x finer chordwise and "
                                       "2 x spanwise", dcd)
    chk("model.munk", y["e"] <= 1.0 + 1e-3, f"span efficiency e = {y['e']:.4f} <= 1 (Munk)", y["e"])
    d_cla = abs(y["CL_alpha"] - ll["CL_alpha"]) / ll["CL_alpha"]
    chk("model.lifting_line", d_cla < 0.1, f"lift slope {y['CL_alpha']:.3f} /rad against Prandtl's lifting line "
                                            f"{ll['CL_alpha']:.3f} /rad ({d_cla:.1%}; the lattice is lower at low "
                                            "aspect ratio, as lifting-surface theory is)", d_cla)
    chk("verify.cfd", None, "an OpenFOAM polar of the section (pinneapple_design.aero.verify) is the next fidelity; "
                            "not run: no OpenFOAM in this environment" if not _has_openfoam() else
        "OpenFOAM is available: run pinneapple_design.aero.verify on the section", None)
    return {"design": x, "outputs": {k: v for k, v in y.items() if k != "profile"}, "profile": y["profile"],
            "baseline": {"design": {"aspect_ratio": 7.5, "taper": 0.7, "twist": -3.0},
                         "drag_N": base["drag_N"], "M_root_Nm": base["M_root_Nm"]},
            "lifting_line": ll, "limits": g, "checks": checks, "recommendation": rec,
            "cost": {"solve_seconds": solve_s, "planned_evaluations": n_eval, "model_solves": len(cache) + 3},
            "assumptions": spec.assumptions}


def _has_openfoam() -> bool:
    import shutil
    return shutil.which("simpleFoam") is not None
