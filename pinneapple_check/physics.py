"""Governing equations of the six physics, written against the derivative providers of
:mod:`pinneapple_check.numerics`, plus the parameter table (names, dimensions, defaults).

Equations (SI, any consistent units):

* ``incompressible-flow``: div u = 0 and du/dt + (u.grad)u + grad(p)/rho - nu lap(u) = 0. ``p`` is kinematic
  (OpenFOAM's incompressible solvers) unless ``rho`` is given. The time term is used when time is known.
* ``heat``: rho_cp dT/dt = k lap(T) + q. ``alpha`` (diffusivity) instead of ``k`` means rho_cp = 1.
* ``advection-diffusion``: dC/dt + c.grad(C) = alpha lap(C) for a scalar transported field (``C``, ``T`` or
  ``phi``); ``c`` is the advecting velocity (a scalar on a 1-D grid, a vector otherwise), ``alpha`` the
  diffusivity, and ``nu`` is accepted for the same number (the name the 1-D test bed uses).
* ``shallow-water``: the depth-averaged shallow-water equations in conservation form,
  ``dh/dt + div(h u) = 0`` and ``d(h u_i)/dt + div(h u_i u + g h^2 / 2 e_i) = 0`` on a horizontal 2-D grid, from
  ``h`` with ``hu, hv`` or ``u, v`` and the gravity ``g``.
* ``elasticity``: div(sigma) + f = 0 with sigma = lambda tr(eps) I + 2 mu eps, from ``E`` and ``nu`` (Poisson
  ratio; plane strain in 2-D). The displacement is ``D`` (OpenFOAM), ``disp``, or ``ux, uy[, uz]``.
* ``conservation``: dq/dt + div F = S for every conserved field ``q`` with its flux field ``F`` declared under
  ``flux = { q = "F" }``.
* ``maxwell``: Maxwell's equations (Gauss, Faraday, Ampere-Maxwell, continuity) and Poynting's theorem for
  field components ``Ex, Ey, Ez, Bx.. or Hx.., Dx.., Jx.., rho`` on a grid (``pinneapple_physics.maxwell_checks``),
  time-domain with a ``t`` axis or phasors (``<name>_re`` / ``<name>_im``) with ``omega``; parameters ``eps_r``,
  ``mu_r``, ``sigma``.
* ``darcy``: -div(a grad u) = f on the unit square with u = 0 on the boundary, cell-centred grid results with the
  coefficient ``a`` (or ``K``) and the solution ``u`` (or ``p``) stored side by side; checked with the finite-volume
  discretisation of the reference solver (``pinneapple_physics.darcy_checks``), not with the generic derivatives
  below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _reuse

__all__ = ["PARAMS", "Equation", "parse_parameters", "equations", "infer_physics", "VEL", "DISP"]

VEL = ("u", "v", "w")
DISP = ("ux", "uy", "uz")

#: name -> (unit of the dimension it must have, default). ``None`` default = required by the equation using it.
PARAMS: dict[str, dict[str, tuple[str, Any]]] = {
    "incompressible-flow": {"nu": ("m^2/s", None), "rho": ("kg/m^3", 1.0)},
    "heat": {"k": ("W/(m*K)", 1.0), "q": ("W/m^3", 0.0), "rho_cp": ("J/(m^3*K)", 1.0), "alpha": ("m^2/s", None)},
    "elasticity": {"E": ("Pa", None), "nu": ("1", None), "f": ("N/m^3", 0.0)},
    "advection-diffusion": {"c": ("m/s", 0.0), "alpha": ("m^2/s", None), "nu": ("m^2/s", None)},
    "shallow-water": {"g": ("m/s^2", 9.80665)},
    "conservation": {"S": ("1", 0.0)},
    "darcy": {"f": ("1", 1.0)},
    "maxwell": {"eps_r": ("1", 1.0), "mu_r": ("1", 1.0), "sigma": ("siemens/m", 0.0), "omega": ("1/s", None)},
    "none": {},
}


@dataclass
class Equation:
    name: str
    residual: np.ndarray            # (n,) or (n, k)
    terms: list[np.ndarray]
    mask: np.ndarray | None
    description: str
    scale: float = 0.0              # characteristic size of the terms (floor of the relative residual)


def parse_parameters(physics: str, given: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Values in SI and a list of (parameter, problem) for wrong dimensions (rule PC504). A value is a number (taken
    as SI) or a string ``"1e-5 m2/s"``; vectors are lists."""
    U = _reuse.units()
    table = PARAMS.get(physics, {})
    out: dict[str, Any] = {}
    problems: list[tuple[str, str]] = []
    alias = {"rhoCp": "rho_cp", "rho-cp": "rho_cp", "poisson": "nu", "young": "E", "source": "q"}
    for k, v in given.items():
        name = alias.get(k, k)
        if isinstance(v, str):
            parts = v.strip().split(None, 1)
            try:
                num = float(parts[0])
            except ValueError:
                problems.append((k, f"{v!r} is not '<number> <unit>'"))
                continue
            if len(parts) == 2 and name in table:
                try:
                    unit = U.parse_unit(parts[1])
                    want = U.parse_unit(table[name][0])
                except U.UnitError as e:
                    problems.append((k, str(e)))
                    continue
                if unit.dim != want.dim:
                    problems.append((k, f"{v!r} has dimension {U.dim_str(unit.dim)}, {name} needs "
                                        f"{U.dim_str(want.dim)} ({table[name][0]})"))
                    continue
                num = float(unit.to_si(num))
            out[name] = num
        elif isinstance(v, (list, tuple)):
            out[name] = np.asarray(v, float)
        else:
            out[name] = float(v)
    for name, (_, default) in table.items():
        out.setdefault(name, default)
    if physics == "heat" and out.get("alpha") is not None and "k" not in given:
        out["k"], out["rho_cp"] = out["alpha"], 1.0
    return out, problems


def infer_physics(names: set[str]) -> str | None:
    """A physics from the field names alone (zero-config): U and p -> incompressible flow."""
    has_u = bool({"U", "u", "velocity"} & names) or {"u", "v"} <= names or {"Ux", "Uy"} <= names
    if has_u and "p" in names and not ({"alpha.water", "T", "rho"} & names):
        return "incompressible-flow"
    return None


# ---------------------------------------------------------------------------------------------- equations
def equations(physics: str, D, p: dict[str, Any], conserved: dict[str, str] | None = None) -> tuple[
        list[Equation], list[str]]:
    """Residuals of ``physics`` on provider ``D``; returns (equations, reasons for those that could not run)."""
    nd = len(D.axes)
    xs = D.axes
    L, tau = D.length, D.duration
    ptp = lambda n: float(np.nanmax(D.val(n)) - np.nanmin(D.val(n))) if D.has(n) else 0.0   # noqa: E731
    eqs: list[Equation] = []
    why: list[str] = []
    if physics == "incompressible-flow":
        vel = list(VEL[:nd])
        if not D.has(*vel):
            return [], [f"no velocity components {', '.join(vel)}"]
        div_terms = [D.d(c, x) for c, x in zip(vel, xs, strict=False)]
        uc = float(np.nanmax(np.sqrt(np.sum([D.val(c) ** 2 for c in vel], axis=0))))
        eqs.append(Equation("continuity", np.sum(div_terms, axis=0), div_terms, D.interior, "div u = 0", uc / L))
        if not D.has("p"):
            why.append("momentum: no pressure field p")
        elif p.get("nu") is None:
            why.append("momentum: no valid kinematic viscosity nu (missing, or rejected by PC504)")
        else:
            rho = float(p.get("rho") or 1.0)
            R, conv, grad, visc, ddt = [], [], [], [], []
            for c, xi in zip(vel, xs, strict=False):
                cv = np.sum([D.val(cj) * D.d(c, xj) for cj, xj in zip(vel, xs, strict=False)], axis=0)
                gp = D.d("p", xi) / rho
                vs = p["nu"] * np.sum([D.dd(c, xj, xj) for xj in xs], axis=0)
                dt = D.dt(c) if D.has_time else np.zeros_like(cv)
                R.append(dt + cv + gp - vs)
                conv.append(cv), grad.append(gp), visc.append(vs), ddt.append(dt)
            st = lambda L: np.stack(L, axis=-1)                     # noqa: E731
            eqs.append(Equation("momentum", st(R), [st(ddt), st(conv), st(grad), st(visc)], D.interior,
                                "du/dt + (u.grad)u + grad p/rho - nu lap u = 0",
                                uc ** 2 / L + p["nu"] * uc / L ** 2 + (uc / tau if D.has_time else 0.0)))
    elif physics == "heat":
        if not D.has("T"):
            return [], ["no temperature field T"]
        lap = p["k"] * np.sum([D.dd("T", x, x) for x in xs], axis=0)
        q = np.broadcast_to(np.asarray(p.get("q") or 0.0, float), lap.shape)
        dt = p["rho_cp"] * D.dt("T") if D.has_time else np.zeros_like(lap)
        sc = p["k"] * ptp("T") / L ** 2 + abs(float(p.get("q") or 0.0)) + \
            (p["rho_cp"] * ptp("T") / tau if D.has_time else 0.0)
        eqs.append(Equation("heat", lap + q - dt, [lap, q, dt], D.interior, "rho_cp dT/dt = k lap T + q", sc))
    elif physics == "advection-diffusion":
        fld = next((n for n in ("C", "T", "phi") if D.has(n)), None)
        if fld is None:
            return [], ["no transported scalar C (or T, phi)"]
        alpha = p.get("alpha") if p.get("alpha") is not None else p.get("nu")
        if alpha is None:
            return [], ["no diffusivity alpha (or nu) for the advection-diffusion equation"]
        c = np.atleast_1d(np.asarray(p.get("c") if p.get("c") is not None else 0.0, float))
        if c.size == 1:
            c = np.full(nd, float(c[0]))
        if c.size != nd:
            return [], [f"the advecting velocity c must have {nd} components on this grid, got {c.size}"]
        adv = [float(c[i]) * D.d(fld, x) for i, x in enumerate(xs)]
        dif = float(alpha) * np.sum([D.dd(fld, x, x) for x in xs], axis=0)
        dt = D.dt(fld) if D.has_time else np.zeros_like(dif)
        sc = float(np.max(np.abs(c))) * ptp(fld) / L + float(alpha) * ptp(fld) / L ** 2 + \
            (ptp(fld) / tau if D.has_time else 0.0)
        eqs.append(Equation("advection-diffusion", dt + np.sum(adv, axis=0) - dif, [dt, *adv, dif], D.interior,
                            "dC/dt + c.grad C = alpha lap C", sc))
    elif physics == "shallow-water":
        if nd != 2:
            return [], [f"the shallow-water equations are checked on a horizontal 2-D grid, this one has {nd} "
                        f"spatial axes"]
        if not D.has("h"):
            return [], ["no water depth h"]
        g = float(p.get("g") if p.get("g") is not None else 9.80665)
        h = D.val("h")
        dt_h = D.dt("h") if D.has_time else np.zeros_like(h)
        dh = [D.d("h", a) for a in xs]
        if D.has("hu", "hv"):
            # conservative variables: h and the momentum h u, h v. The products of two stored fields are
            # differentiated with the product rule, so every derivative is of a field the provider has.
            hu, hv = D.val("hu"), D.val("hv")
            safe = np.where(h > 0, h, np.nan)
            uu, uv = hu / safe, hv / safe
            dhu, dhv = [D.d("hu", a) for a in xs], [D.d("hv", a) for a in xs]
            du = [(dhu[i] * h - hu * dh[i]) / safe ** 2 for i in (0, 1)]      # d(hu/h)/dx, d(hu/h)/dy
            dv = [(dhv[i] * h - hv * dh[i]) / safe ** 2 for i in (0, 1)]
            dt_hu = D.dt("hu") if D.has_time else np.zeros_like(hu)
            dt_hv = D.dt("hv") if D.has_time else np.zeros_like(hv)
            mass_terms = [dt_h, dhu[0], dhu[1], dhv[0], dhv[1]]
            ru = [dt_hu, dhu[0] * uu + hu * du[0], dhu[1] * uv + hu * dv[1], g * h * dh[0]]
            rv = [dt_hv, dhv[0] * uu + hv * du[0], dhv[1] * uv + hv * dv[1], g * h * dh[1]]
        elif D.has("u", "v"):
            # primitive variables: h, u, v, with the mass flux expanded by the product rule
            uu, uv = D.val("u"), D.val("v")
            du, dv = [D.d("u", a) for a in xs], [D.d("v", a) for a in xs]
            dt_u = D.dt("u") if D.has_time else np.zeros_like(uu)
            dt_v = D.dt("v") if D.has_time else np.zeros_like(uv)
            mass_terms = [dt_h, uu * dh[0], h * du[0], uv * dh[1], h * dv[1]]
            ru = [dt_u, uu * du[0], uv * du[1], g * dh[0]]
            rv = [dt_v, uu * dv[0], uv * dv[1], g * dh[1]]
        else:
            return [], ["no velocity: give hu, hv (conservative) or u, v with the depth h"]
        if not (np.all(np.isfinite(uu)) and np.all(np.isfinite(uv))):
            return [], ["shallow-water momentum: the depth h is zero or negative where the velocity is h u / h"]
        hs = max(float(np.nanmax(np.abs(h))), 1e-300)
        us = max(float(np.nanmax(np.hypot(uu, uv))), 1e-300)
        sc_m = hs * us / L + (hs / tau if D.has_time else 0.0)
        sc_u = hs * us * us / L + g * hs ** 2 / L + (hs * us / tau if D.has_time else 0.0)
        eqs.append(Equation("continuity", np.sum(mass_terms, axis=0), mass_terms, D.interior,
                            "dh/dt + div(h u) = 0", sc_m))
        st = lambda L: np.stack(L, axis=-1)                             # noqa: E731
        eqs.append(Equation("momentum", st([np.sum(ru, axis=0), np.sum(rv, axis=0)]), [st(x) for x in (ru, rv)],
                            D.interior, "d(h u_i)/dt + div(h u_i u) + g h grad(h) = 0", sc_u))
    elif physics == "elasticity":
        u = list(DISP[:nd])
        if not D.has(*u):
            return [], [f"no displacement ({', '.join(u)}, or a vector D)"]
        if p.get("E") is None or p.get("nu") is None:
            return [], ["parameters E (Young's modulus) and nu (Poisson ratio) not given"]
        E, nu = float(p["E"]), float(p["nu"])
        lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
        f = np.atleast_1d(np.asarray(p.get("f") if p.get("f") is not None else 0.0, float))
        if f.size == 1 and f[0] == 0.0:
            f = np.zeros(nd)
        if f.size != nd:
            return [], [f"body force f must be a vector of {nd} components, e.g. f = [0, -9.81e4]"]
        R, t1, t2, t3 = [], [], [], []
        for i, (ui, xi) in enumerate(zip(u, xs, strict=False)):
            a = mu * np.sum([D.dd(ui, xj, xj) for xj in xs], axis=0)
            b = (lam + mu) * np.sum([D.dd(uj, xi, xj) for uj, xj in zip(u, xs, strict=False)], axis=0)
            fi = np.full_like(a, f[i])
            R.append(a + b + fi), t1.append(a), t2.append(b), t3.append(fi)
        st = lambda L: np.stack(L, axis=-1)                         # noqa: E731
        sc = E * max(ptp(c) for c in u) / L ** 2 + float(np.abs(f).max())
        eqs.append(Equation("equilibrium", st(R), [st(t1), st(t2), st(t3)], D.interior, "div sigma + f = 0", sc))
    elif physics == "darcy":
        return [], ["darcy is checked on stored cell-centred grid results (fields a and u), not on point samples"]
    elif physics == "maxwell":
        return [], ["maxwell is checked on stored grid results (components Ex..Hz, rho, J), not on point samples"]
    elif physics == "conservation":
        if not conserved:
            return [], ["declare the flux of each conserved field: flux = { q = \"F\" }"]
        if not D.has_time:
            return [], ["a conservation law needs a time axis (t)"]
        for q, F in conserved.items():
            comps = [f"{F}_{a}" for a in "xyz"[:nd]] if nd > 1 or f"{F}_x" in D.comps else [F]
            if not D.has(q, *comps):
                why.append(f"{q}: fields {q} and {', '.join(comps)} not found")
                continue
            dq = D.dt(q)
            div = [D.d(c, x) for c, x in zip(comps, xs, strict=False)]
            S = np.broadcast_to(np.asarray(p.get("S") or 0.0, float), dq.shape)
            sc = ptp(q) / tau + max(float(np.nanmax(np.abs(D.val(c)))) for c in comps) / L + \
                abs(float(p.get("S") or 0.0))
            eqs.append(Equation(f"conservation of {q}", dq + np.sum(div, axis=0) - S, [dq, *div, S], D.interior,
                                f"d{q}/dt + div {F} = S", sc))
    return eqs, why
