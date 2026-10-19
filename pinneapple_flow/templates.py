"""Equation templates per PDE family and dimension: the PDEs, conditions, coefficients and (when one is known) the
exact solution a planner proposes and a user then edits.

Everything is text in the grammar of :mod:`pinneapple_physics.symbolic_pde.text`, so a template is exactly what the
equations node shows. ``exact`` is an expression in the coordinates, or the name of a library reference
(``"cole_hopf"`` for viscous Burgers, :func:`pinneapple_analysis.validation.pinn.cole_hopf`).

    t = template("heat", dimension=2)
    t["pdes"]          # ['u_t = alpha*lap(u)']
    t["exact"]         # 'exp(-2*alpha*pi**2*t)*sin(pi*x)*sin(pi*y)'
"""
from __future__ import annotations

from typing import Any

FAMILIES = ("heat", "wave", "poisson", "laplace", "helmholtz", "burgers", "advection_diffusion",
            "reaction_diffusion")
SPATIAL = ("x", "y", "z")

DESCRIPTIONS = {
    "heat": "diffusion of heat (or any scalar): u_t = alpha * laplacian(u)",
    "wave": "waves / vibration: u_tt = c^2 * laplacian(u)",
    "poisson": "steady field with a source: laplacian(u) = f",
    "laplace": "steady potential / temperature without sources: laplacian(u) = 0",
    "helmholtz": "time-harmonic waves (acoustics, EM): laplacian(u) + k^2 u = f",
    "burgers": "nonlinear advection with viscosity (shocks): u_t + u u_x = nu u_xx",
    "advection_diffusion": "transport by a flow plus diffusion: u_t + c u_x = D u_xx",
    "reaction_diffusion": "diffusion with logistic growth (Fisher-KPP): u_t = D lap(u) + r u (1 - u)",
}


def _prod_sin(dim: int) -> str:
    return "*".join(f"sin(pi*{c})" for c in SPATIAL[:dim])


def template(family: str, dimension: int = 1) -> dict[str, Any]:
    """PDEs, conditions, coefficients, domain and exact solution for ``family`` in ``dimension`` (1-3) spatial
    dimensions. ValueError for an unknown family or a dimension the family does not have here."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; known: {FAMILIES}")
    d = int(dimension)
    if d not in (1, 2, 3):
        raise ValueError("dimension is 1, 2 or 3")
    ps = _prod_sin(d)
    box = {c: [0.0, 1.0] for c in SPATIAL[:d]}
    zero_bc = [{"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}]
    t: dict[str, Any] = {"family": family, "dimension": d, "fields": ["u"], "bounds": box, "time": None,
                         "description": DESCRIPTIONS[family], "exact": None}
    if family == "heat":
        t.update(pdes=["u_t = alpha*lap(u)"], coefficients={"alpha": 0.1}, time=[0.0, 0.5],
                 conditions=[{"kind": "initial", "on": "t_min", "equation": f"u = {ps}"}, *zero_bc],
                 exact=f"exp(-{d}*alpha*pi**2*t)*{ps}")
    elif family == "wave":
        t.update(pdes=["u_tt = c**2*lap(u)"], coefficients={"c": 1.0}, time=[0.0, 1.0],
                 conditions=[{"kind": "initial", "on": "t_min", "equation": f"u = {ps}"},
                             {"kind": "initial", "on": "t_min", "equation": "u_t = 0"}, *zero_bc],
                 exact=f"cos(c*pi*sqrt({d})*t)*{ps}")
    elif family == "poisson":
        t.update(pdes=[f"lap(u) = -{d}*pi**2*{ps}"], coefficients={}, conditions=list(zero_bc), exact=ps)
    elif family == "laplace":
        if d == 1:
            t.update(pdes=["u_xx = 0"], coefficients={},
                     conditions=[{"kind": "dirichlet", "on": "x_min", "equation": "u = 0"},
                                 {"kind": "dirichlet", "on": "x_max", "equation": "u = 1"}], exact="x")
        else:
            others = [c for c in SPATIAL[:d] if c != "y"]
            top = "*".join(f"sin(pi*{c})" for c in others)
            k = f"pi*sqrt({len(others)})"
            conds = [{"kind": "dirichlet", "on": f, "equation": "u = 0"}
                     for f in [f"{c}_{s}" for c in SPATIAL[:d] for s in ("min", "max")] if f != "y_max"]
            conds.append({"kind": "dirichlet", "on": "y_max", "equation": f"u = {top}"})
            t.update(pdes=["lap(u) = 0"], coefficients={}, conditions=conds,
                     exact=f"{top}*sinh({k}*y)/sinh({k})")
    elif family == "helmholtz":
        t.update(pdes=[f"lap(u) + k**2*u = (k**2 - {d}*pi**2)*{ps}"], coefficients={"k": 2.0},
                 conditions=list(zero_bc), exact=ps)
    elif family == "burgers":
        if d != 1:
            raise ValueError("the Burgers template is 1-D")
        t.update(pdes=["u_t + u*u_x = nu*u_xx"], coefficients={"nu": 0.01 / 3.141592653589793},
                 bounds={"x": [-1.0, 1.0]}, time=[0.0, 1.0],
                 conditions=[{"kind": "initial", "on": "t_min", "equation": "u = -sin(pi*x)"}, *zero_bc],
                 exact="cole_hopf")
    elif family == "advection_diffusion":
        drift = " + ".join(f"c*u_{c}" for c in SPATIAL[:d])
        gauss = "exp(-(" + " + ".join(f"({c} - 0.3)**2" for c in SPATIAL[:d]) + ")/0.01)"
        t.update(pdes=[f"u_t + {drift} = D*lap(u)"], coefficients={"c": 1.0, "D": 0.01}, time=[0.0, 0.4],
                 conditions=[{"kind": "initial", "on": "t_min", "equation": f"u = {gauss}"}, *zero_bc])
    elif family == "reaction_diffusion":
        gauss = "exp(-(" + " + ".join(f"({c} - 0.5)**2" for c in SPATIAL[:d]) + ")/0.01)"
        t.update(pdes=["u_t = D*lap(u) + r*u*(1 - u)"], coefficients={"D": 0.001, "r": 5.0}, time=[0.0, 1.0],
                 conditions=[{"kind": "initial", "on": "t_min", "equation": f"u = {gauss}"}]
                 + [{"kind": "neumann", "on": f"{c}_{s}", "equation": f"u_{c} = 0"} for c in SPATIAL[:d]
                    for s in ("min", "max")])
    return t


def retarget(equations: dict[str, Any], dimension: int) -> dict[str, Any]:
    """The equations of a template family moved to another dimension (the 2-D <-> 3-D switch). PDEs written with
    ``lap()`` keep their text; conditions on ``boundary`` stay; a known template is rebuilt for the new dimension."""
    fam = equations.get("family")
    if fam in FAMILIES:
        try:
            t = template(fam, dimension)
        except ValueError:
            t = None
        if t is not None:
            keep = {k: v for k, v in (equations.get("coefficients") or {}).items() if k in t["coefficients"]}
            t["coefficients"].update(keep)
            return t
    out = dict(equations)
    out["dimension"] = dimension
    return out
