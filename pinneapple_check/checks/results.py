"""Checks of a stored result (OpenFOAM case, grid, point cloud, QoI table)."""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import _reuse, balances
from ..config import ResultTarget
from ..fields import FieldData, LoadError, load_result
from ..numerics import GridD, PointD, components, rel_rms
from ..physics import DISP, VEL, equations, infer_physics, parse_parameters
from .common import Ctx, check_units, check_values, fmt, rel_l2

__all__ = ["check_result", "evaluate_expr", "check_balance", "report_equations"]

_SAFE = {k: getattr(np, k) for k in ("sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "sinh", "cosh", "abs",
                                       "arctan", "arctan2", "where", "minimum", "maximum")}


def evaluate_expr(value: Any, coords: dict[str, np.ndarray], n: int) -> np.ndarray:
    """A declared value: a number, a vector (list) or an expression of the coordinates (``"sin(pi*x)"``)."""
    if isinstance(value, str):
        from pinneapple_data.expressions import (
            evaluate,  # interpreted, never eval(): configs are uploaded
        )
        try:
            out = evaluate(value, {"pi": np.pi, "e": np.e, **coords}, _SAFE)
        except Exception as e:
            raise ValueError(f"cannot evaluate {value!r}: {e}") from e
        return np.broadcast_to(np.asarray(out, float), (n,)).copy()
    a = np.asarray(value, float)
    return np.broadcast_to(a, (n,) + a.shape).copy()


def check_result(ctx: Ctx, t: ResultTarget) -> FieldData | None:
    try:
        fd = load_result(t.path, t.kind, t.time, t.fields)
    except (LoadError, Exception) as e:                               # any reader failure is PC001
        ctx.emit("PC001", "read", False, f"cannot read {ctx.cfg.rel(t.path)}: {e}")
        return None
    ctx.emit("PC001", "read", True, f"read {fd.kind} result {ctx.cfg.rel(t.path)}")
    if fd.kind == "qoi":
        _qoi(ctx, t, fd)
        _summary(ctx, t)
        return fd
    names = [n for n in (t.fields or list(fd.fields)) if n in fd.fields]
    missing = [n for n in (t.fields or []) if n not in fd.fields]
    for n in missing:
        ctx.emit("PC001", n, False, f"field {n} declared under `fields` is not in {ctx.cfg.rel(t.path)}")
    where = _where(fd)
    physics = t.physics or infer_physics(set(fd.fields))
    params, problems = parse_parameters(physics or "none", t.parameters)
    for k, msg in problems:
        ctx.emit("PC504", k, False, f"parameter {k}: {msg}")
    for k, v in t.parameters.items():
        if isinstance(v, str) and k not in dict(problems):
            ctx.emit("PC504", k, True, f"parameter {k} = {v}")
    used = _used_fields(physics, fd)
    for n in names:
        loc = _field_loc(ctx, fd, n)
        unit = t.units.get(n) or fd.units.get(n)
        check_values(ctx, n, fd.flat(n), unit, loc, where)
        check_units(ctx, n, fd.units.get(n), t.units.get(n), fd.flat(n), loc, n in used)
    if physics and physics != "none":
        if fd.kind == "fv":
            _physics_fv(ctx, t, fd, physics, params)
        elif fd.kind == "grid":
            _physics_grid(ctx, t, fd, physics, params)
        else:
            _physics_points(ctx, t, fd, physics, params)
    if t.bc:
        _bc(ctx, t, fd)
    if t.ic:
        _ic(ctx, t, fd)
    if t.reference is not None:
        _reference(ctx, t, fd)
    _summary(ctx, t)
    return fd


def _summary(ctx: Ctx, t) -> None:
    substantive = ctx.ran - {"PC001", "PC403", "PC502", "PC504"}
    if t.declared:
        ctx.emit("PC002", "checks", bool(substantive),
                 f"checked by {', '.join(sorted(substantive))}" if substantive else
                 f"no physics, bounds, boundary condition or reference check applies to {t.name}")


def _used_fields(physics, fd) -> set[str]:
    if physics == "incompressible-flow":
        return {n for n in fd.fields if n in ("U", "u", "v", "w", "p", "Ux", "Uy", "Uz")}
    if physics == "heat":
        return {"T"} & set(fd.fields)
    if physics == "elasticity":
        return {n for n in fd.fields if n in ("D", "disp", "displacement", "ux", "uy", "uz")}
    if physics == "darcy":
        return {n for n in fd.fields if n in DARCY_COEF + DARCY_SOL}
    if physics == "maxwell":
        return {n for n in fd.fields if _maxwell_name(n) is not None}
    return set()


def _where(fd: FieldData):
    if fd.kind == "fv":
        C = fd.geom["cell_centres"]
        dirs = fd.geom["dirs"]
        return lambda i: f"cell {i} (" + ", ".join(f"{a}={C[i, k]:.4g}" for k, a in enumerate("xyz") if dirs[k]) + ")"
    if fd.kind == "grid":
        shape = tuple(len(v) for v in fd.axes.values())
        return lambda i: ", ".join(f"{a}={fd.axes[a][j]:.4g}" for a, j in zip(fd.axes, np.unravel_index(i, shape), strict=False))
    if fd.points is not None:
        P = fd.points
        return lambda i: ", ".join(f"{a}={P[i, k]:.4g}" for k, a in enumerate("xyz") if np.ptp(P[:, k]) > 0)
    return None


def _field_loc(ctx: Ctx, fd: FieldData, name: str, patch: str | None = None):
    f = fd.files.get(name)
    if fd.kind == "fv":
        if patch:
            return ctx.file_location(f, patch, after="boundaryField", then="value")
        return ctx.file_location(f, "internalField")
    return ctx.file_location(f)


# ---------------------------------------------------------------------------------------------- shared reporters
def report_equations(ctx: Ctx, eqs, why, where, rule_div: str, rule_res: str, tol_div: float, tol_res: float,
                     loc, subject_prefix: str = "") -> None:
    for eq in eqs:
        if eq.mask is not None and not np.any(eq.mask):
            # every point is on the boundary (e.g. 2 points along an axis): there is nothing to judge, and an empty
            # residual must not be reported as a pass
            ctx.note(f"{subject_prefix}{eq.name}: no interior point where the stencil fits, residual not evaluated")
            continue
        rel = rel_rms(eq.residual, eq.terms, eq.mask, floor=1e-6 * eq.scale)
        r = np.asarray(eq.residual)
        mag = np.linalg.norm(r, axis=-1) if eq.mask is not None and r.ndim == eq.mask.ndim + 1 else np.abs(r)
        m = np.where(eq.mask, mag, -np.inf) if eq.mask is not None else mag
        i = int(np.argmax(np.where(np.isfinite(m), m, -np.inf)))
        rule, tol = (rule_div, tol_div) if eq.name == "continuity" else (rule_res, tol_res)
        what = "divergence" if eq.name == "continuity" else f"{eq.name} residual"
        ok = bool(np.isfinite(rel) and rel <= tol)
        at = f"; largest at {where(i)}" if where and not ok else ""
        ctx.emit(rule, f"{subject_prefix}{eq.name}", ok,
                 f"{what} ({eq.description}): relative RMS {fmt(rel)} {'<=' if ok else '>'} {fmt(tol)}{at}",
                 loc, value=rel, threshold=tol)
    for w in why:
        ctx.note(f"{subject_prefix}{w}")


def check_balance(ctx: Ctx, rule: str, what: str, b: dict[str, Any], tol: float, loc, unit: str = "",
                  closed_scale: float | None = None, subject: str | None = None) -> None:
    net, tot = b["net"], b["total"]
    if closed_scale is not None and tot <= 1e-9 * max(closed_scale, 1e-300):
        rel = abs(net) / max(closed_scale, 1e-300)
        ok = rel <= tol
        ctx.emit(rule, subject or what, ok, f"{what}: closed boundary, net {fmt(net)}{unit} "
                                            f"({fmt(rel)} of the reference flow)", loc, value=rel, threshold=tol)
        return
    rel = abs(net) / tot if tot > 0 else 0.0
    ok = rel <= tol
    parts = ", ".join(f"{k} {fmt(v)}" for k, v in sorted(b.get("patches", {}).items(), key=lambda kv: -abs(kv[1]))[:6]
                      if abs(v) > 1e-12 * max(tot, 1e-300))
    msg = f"{what}: net {fmt(net)}{unit} is {fmt(rel)} of the total {fmt(tot)}{unit} " \
          f"({'<=' if ok else '>'} {fmt(tol)})" + (f"; by boundary: {parts}" if parts and not ok else "")
    ctx.emit(rule, subject or what, ok, msg, loc, value=rel, threshold=tol)


# ---------------------------------------------------------------------------------------------- FV
def _physics_fv(ctx: Ctx, t: ResultTarget, fd: FieldData, physics: str, params: dict[str, Any]) -> None:
    from ..fv import FV, fv_heat_balance, fv_mass_balance, fv_residuals
    fv = FV(fd)
    where = _where(fd)
    if physics == "incompressible-flow":
        b = fv_mass_balance(fv)
        if b is not None:
            src = "phi" if fd.phi is not None else "U"
            check_balance(ctx, "PC101", f"volume flux through the boundary (from {src})", b, t.tol("mass"),
                          _field_loc(ctx, fd, src), closed_scale=b["uref"] * b["area"], subject="mass")
        _walls(ctx, t, fd, fv)
    if physics == "heat":
        b = fv_heat_balance(fv, params)
        if b is not None:
            if fd.previous and "T" in fd.previous["fields"] and fd.time is not None:
                dt = fd.time - fd.previous["time"]
                store = float(np.sum(params["rho_cp"] * (fd.fields["T"] - fd.previous["fields"]["T"]) / dt * fv.V))
                b = {**b, "net": b["net"] - store, "total": b["total"] + abs(store)}
            check_balance(ctx, "PC103", "heat flow through the boundary", b, t.tol("energy"), _field_loc(ctx, fd, "T"),
                          subject="energy")
    res, why = fv_residuals(fv, physics, params)
    for w in why:
        ctx.note(w)
    for name, R, terms, desc in res:
        rel = rel_rms(R, terms, fv.interior)
        mag = np.linalg.norm(R, axis=1) if R.ndim > 1 else np.abs(R)
        i = int(np.argmax(np.where(fv.interior, mag, -np.inf)))
        rule, tol, fld = ("PC102", t.tol("divergence"), "U") if name == "continuity" else ("PC301", t.tol("residual"),
                                                                                              "U" if "U" in fd.fields else "T")
        ok = bool(np.isfinite(rel) and rel <= tol)
        what = "divergence" if name == "continuity" else f"{name} residual"
        ctx.emit(rule, name, ok, f"{what} ({desc}): relative RMS {fmt(rel)} {'<=' if ok else '>'} {fmt(tol)}"
                 + ("" if ok else f"; largest at {where(i)}"), _field_loc(ctx, fd, fld), value=rel, threshold=tol)


def _walls(ctx: Ctx, t: ResultTarget, fd: FieldData, fv) -> None:
    if "U" not in fd.fields or "U" not in fd.boundary:
        return
    U = fd.fields["U"]
    uref = max(float(np.max(np.linalg.norm(U, axis=1))), 1e-300)
    for pt in fd.patches:
        if pt["type"] != "wall" or pt["name"] in t.bc or pt["nFaces"] == 0:
            continue
        sl = fv.patch_slice(pt["name"])
        ub = np.asarray(fd.boundary["U"][sl], float)
        if not np.isfinite(ub).all():
            continue
        m = float(np.max(np.linalg.norm(ub, axis=1))) / uref
        ok = m <= t.tol("wall-velocity")
        ctx.emit("PC203", f"U@{pt['name']}", ok,
                 f"U on wall patch '{pt['name']}': max |U| = {fmt(m * uref)} ({fmt(m)} of the maximum velocity)"
                 + ("" if ok else "; declare it under [result.bc." + pt["name"] + "] if the wall moves"),
                 _field_loc(ctx, fd, "U", pt["name"]), value=m, threshold=t.tol("wall-velocity"))


# ---------------------------------------------------------------------------------------------- grid
def _grid_slices(fd: FieldData):
    """(label, {name: array over the spatial axes}) for every time (one entry without a time axis)."""
    if "t" not in fd.axes:
        return [("", fd.fields)]
    ti = list(fd.axes).index("t")
    out = []
    for k, tv in enumerate(fd.axes["t"]):
        out.append((f" at t={tv:.4g}", {n: np.take(v, k, axis=ti) for n, v in fd.fields.items()}))
    return out


def _physics_grid(ctx: Ctx, t: ResultTarget, fd: FieldData, physics: str, params: dict[str, Any]) -> None:
    if physics == "darcy":
        _darcy(ctx, t, fd, params)
        return
    if physics == "maxwell":
        _maxwell(ctx, t, fd, params)
        return
    nd_all = len(fd.axes)
    comps = components(fd.fields, vector_ndim=nd_all + 1)
    D = GridD(fd.axes, comps)
    where = _where(fd)
    loc = ctx.default_location() if t.declared else ctx.file_location(fd.source)
    eqs, why = equations(physics, D, params, t.flux or None)
    report_equations(ctx, eqs, why, where, "PC102", "PC301", t.tol("divergence"), t.tol("residual"), loc)
    sp = fd.spatial_axes
    coords = [fd.axes[a] for a in sp]
    slices = _grid_slices(fd)
    if physics == "incompressible-flow" and all(n in comps for n in VEL[:len(sp)]):
        worst = None
        for label, f in slices:
            c = components(f, vector_ndim=len(sp) + 1)
            b = balances.grid_mass(coords, sp, [c[n] for n in VEL[:len(sp)]])
            rel = abs(b["net"]) / b["total"] if b["total"] > 0 else 0.0
            if worst is None or rel > worst[0]:
                worst = (rel, label, b)
        check_balance(ctx, "PC101", "volume flux through the boundary" + worst[1], worst[2], t.tol("mass"), loc,
                      closed_scale=worst[2]["uref"] * worst[2]["area"], subject="mass")
    if physics == "heat" and "T" in comps and "t" not in fd.axes:
        b = balances.grid_heat(coords, sp, comps["T"], params["k"], params.get("q") or 0.0)
        check_balance(ctx, "PC103", "heat flow through the boundary", b, t.tol("energy"), loc, subject="energy")
    if physics == "elasticity" and all(n in comps for n in DISP[:len(sp)]) and params.get("E") is not None:
        label, f = slices[-1]
        c = components(f, vector_ndim=len(sp) + 1)
        fb = np.atleast_1d(np.asarray(params.get("f") if params.get("f") is not None else 0.0, float))
        fb = np.zeros(len(sp)) if fb.size == 1 and fb[0] == 0 else fb
        if fb.size == len(sp):
            b = balances.grid_force(coords, sp, [c[n] for n in DISP[:len(sp)]], params["E"], params["nu"], fb)
            check_balance(ctx, "PC104", "resultant force on the body" + label, b, t.tol("momentum"), loc,
                          subject="momentum")
    if physics == "conservation" and "t" in fd.axes:
        _total_drift(ctx, t, fd, coords, sp, loc)


def _total_drift(ctx: Ctx, t: ResultTarget, fd: FieldData, coords, sp, loc) -> None:
    trapz = balances._trapz
    ti = list(fd.axes).index("t")
    tt = fd.axes["t"]
    names = t.conserved or list(t.flux) or []
    for q in names:
        if q not in fd.fields:
            ctx.emit("PC105", q, False, f"conserved field {q} is not in the result", loc)
            continue
        tot = np.array([balances._volume_integral(np.take(fd.fields[q], k, axis=ti), coords) for k in range(len(tt))])
        F = t.flux.get(q)
        out = np.zeros(len(tt))
        if F and F in fd.fields or (F and f"{F}_x" in components(fd.fields, len(fd.axes) + 1)):
            comps = components(fd.fields, vector_ndim=len(fd.axes) + 1)
            fc = [comps[f"{F}_{a}"] if f"{F}_{a}" in comps else comps[F] for a in "xyz"[:len(sp)]]
            for k in range(len(tt)):
                out[k] = balances.grid_mass(coords, sp, [np.take(c, k, axis=ti) for c in fc])["net"]
        cum_out = np.concatenate([[0.0], np.cumsum(0.5 * (out[1:] + out[:-1]) * np.diff(tt))])
        drift = tot - tot[0] + cum_out
        ref = max(abs(tot[0]), float(np.max(np.abs(tot))), 1e-300)
        k = int(np.argmax(np.abs(drift)))
        rel = float(abs(drift[k]) / ref)
        ok = rel <= t.tol("total-drift")
        ctx.emit("PC105", q, ok, f"total of {q}{' (net of its boundary flux ' + F + ')' if F else ''}: "
                                 f"{fmt(tot[0])} at t={tt[0]:.4g}, drift {fmt(rel)} of it "
                                 f"{'<=' if ok else '>'} {fmt(t.tol('total-drift'))}"
                                 + ("" if ok else f" (worst at t={tt[k]:.4g})"), loc, value=rel,
                 threshold=t.tol("total-drift"))
    _ = trapz


# ---------------------------------------------------------------------------------------------- darcy
DARCY_COEF = ("a", "K", "k", "permeability")
DARCY_SOL = ("u", "p", "pressure", "h")


def _darcy(ctx: Ctx, t: ResultTarget, fd: FieldData, params: dict[str, Any]) -> None:
    """-div(a grad u) = f, u = 0 on the boundary of the unit square, on the reference solver's cell-centred
    finite-volume discretisation (``pinneapple_physics.darcy_checks.sample_checks``): certified energy-norm and L2
    error bounds from the residual (PC301), net boundary flux against the source (PC101), boundary condition (PC201),
    maximum principle (PC402) and, with ``envelope``, the coefficient against its training envelope (PC401)."""
    loc = ctx.default_location() if t.declared else ctx.file_location(fd.source)
    coef = next((n for n in DARCY_COEF if n in fd.fields), None)
    sol = next((n for n in (t.fields or []) if n in DARCY_SOL and n in fd.fields), None) or \
        next((n for n in DARCY_SOL if n in fd.fields), None)
    sp = fd.spatial_axes
    if coef is None or sol is None or len(sp) != 2 or "t" in fd.axes:
        ctx.note("darcy: needs a steady 2-D grid with a coefficient field (a or K) and a solution field (u or p)")
        return
    x, y = fd.axes[sp[0]], fd.axes[sp[1]]
    n = len(x)
    centres = (np.arange(n) + 0.5) / n
    if len(y) != n or not (np.allclose(x, centres, atol=1e-6) and np.allclose(y, centres, atol=1e-6)):
        ctx.note(f"darcy: the grid must be the {n} x {n} cell centres of the unit square ((i + 1/2) / n)")
        return
    a, u = np.asarray(fd.fields[coef], float), np.asarray(fd.fields[sol], float)
    if not (np.isfinite(a).all() and np.isfinite(u).all()):
        return                                                   # PC403 already reported
    dc = _reuse.darcy_checks()
    f = float(params.get("f") if params.get("f") is not None else 1.0)
    eq = f"-div({coef} grad {sol}) = {fmt(f)}"
    tol_key = {"energy_bound": "darcy-energy", "l2_bound": "darcy-l2", "flux": "darcy-flux", "bc": "darcy-bc",
               "positivity": "darcy-positivity"}
    env = dc.CoefficientEnvelope.from_check_config(t.envelope) if t.envelope else None
    checks = dc.sample_checks(a, u, f, tolerances={k: t.tol(v) for k, v in tol_key.items()}, envelope=env)
    i = int(np.argmax(np.abs(dc.darcy_residual(u, a, f))))
    for c in checks:
        k, v, lim, ok = c["key"], c["value"], c["limit"], c["passed"]
        rel = "<=" if ok else ">"
        if k in ("energy_bound", "l2_bound"):
            what = "energy-norm" if k == "energy_bound" else "L2"
            ctx.emit("PC301", "darcy" if k == "energy_bound" else "darcy-l2", ok,
                     f"darcy residual ({eq}, finite volumes): certified {what} error bound {fmt(v)} {rel} {fmt(lim)}"
                     + ("" if ok else f"; largest residual at {_where(fd)(i)}"), loc, value=v, threshold=lim)
        elif k == "flux":
            ctx.emit("PC101", "darcy-flux", ok, f"darcy flux through the boundary against the source integral: "
                     f"imbalance {fmt(v)} {rel} {fmt(lim)}", loc, value=v, threshold=lim)
        elif k == "bc":
            ctx.emit("PC201", f"{sol}@boundary", ok, f"{sol} = 0 on the boundary: RMS of the extrapolated boundary "
                     f"value {fmt(v)} of max |{sol}| {rel} {fmt(lim)}", loc, value=v, threshold=lim)
        elif k == "positivity":
            ctx.emit("PC402", sol, ok, f"{sol} >= 0 (maximum principle, f >= 0): most negative value {fmt(v)} of "
                     f"max |{sol}| {rel} {fmt(lim)}", loc, value=v, threshold=lim)
        else:                                                    # coefficient against the training envelope
            rng = f"[{fmt(lim[0])}, {fmt(lim[1])}]" if isinstance(lim, list) else \
                (f">= {fmt(lim)}" if k == "a_min" else f"<= {fmt(lim)}")
            ctx.emit("PC401", f"{coef}-envelope:{k.replace('_', '-')}", ok,
                     f"{coef}: {k.replace('_', ' ')} {fmt(v)} {'inside' if ok else 'outside'} the training envelope "
                     f"{rng}", loc, value=v, threshold=lim)


# ---------------------------------------------------------------------------------------------- maxwell
MAXWELL_COMPONENTS = tuple(f"{v}{c}" for v in "EBHDJ" for c in "xyz") + ("rho",)


def _maxwell_name(n: str) -> str | None:
    base = n[:-3] if n.endswith(("_re", "_im")) else n
    return base if base in MAXWELL_COMPONENTS else None


def _maxwell(ctx: Ctx, t: ResultTarget, fd: FieldData, params: dict[str, Any]) -> None:
    """Maxwell's equations on stored grid fields (``pinneapple_physics.maxwell_checks``): Gauss's laws (PC102),
    Faraday, Ampere-Maxwell and charge continuity (PC301) and Poynting's theorem over the grid box (PC103)."""
    loc = ctx.default_location() if t.declared else ctx.file_location(fd.source)
    fields: dict[str, np.ndarray] = {}
    for n, v in fd.fields.items():
        base = _maxwell_name(n)
        if base is None:
            continue
        v = np.asarray(v)
        if n.endswith("_re"):
            fields[base] = fields.get(base, 0j) + v
        elif n.endswith("_im"):
            fields[base] = fields.get(base, 0j) + 1j * v
        else:
            fields[base] = v
    omega = params.get("omega")
    if not fields:
        ctx.note("maxwell: no field components found (Ex, Ey, Ez, Bx/Hx.., Dx.., Jx.., rho)")
        return
    if any(np.iscomplexobj(v) for v in fields.values()) and omega is None:
        ctx.note("maxwell: complex (phasor) fields need the parameter omega (rad/s)")
        return
    mc = _reuse.maxwell_checks()
    eps = mc.EPS0 * float(params.get("eps_r") or 1.0)
    mu = mc.MU0 * float(params.get("mu_r") or 1.0)
    sigma = float(params.get("sigma") or 0.0)
    tol = {"gauss_electric": t.tol("maxwell-gauss"), "gauss_magnetic": t.tol("maxwell-gauss"),
           "faraday": t.tol("maxwell-faraday"), "ampere_maxwell": t.tol("maxwell-ampere"),
           "continuity": t.tol("maxwell-continuity"), "poynting_balance": t.tol("maxwell-energy")}
    try:
        checks = mc.sample_checks(fd.axes, fields, eps=eps, mu=mu, sigma=sigma, omega=omega, tolerances=tol)
    except ValueError as e:
        ctx.note(f"maxwell: {e}")
        return
    rule = {"gauss_electric": "PC102", "gauss_magnetic": "PC102", "faraday": "PC301", "ampere_maxwell": "PC301",
            "continuity": "PC301", "poynting_balance": "PC103", "finite": "PC403"}
    for c in checks:
        if c["passed"] is None:
            ctx.note(f"maxwell: {c['name']}: {c['detail']}")
            continue
        ok, v, lim = c["passed"], c["value"], c["limit"]
        ctx.emit(rule.get(c["key"], "PC301"), f"maxwell-{c['key'].replace('_', '-')}", ok,
                 f"{c['name']}: {fmt(v)} {'<=' if ok else '>'} {fmt(lim)}; {c['detail']}", loc, value=v,
                 threshold=lim)


# ---------------------------------------------------------------------------------------------- points
def _physics_points(ctx: Ctx, t: ResultTarget, fd: FieldData, physics: str, params: dict[str, Any]) -> None:
    span = np.ptp(fd.points, axis=0)
    axes = [a for k, a in enumerate("xyz") if span[k] > 1e-12 * max(span.max(), 1e-300)]
    comps = components(fd.fields, vector_ndim=2, ncomp=len(axes))
    D = PointD(fd.points, comps, axes)
    loc = ctx.default_location() if t.declared else ctx.file_location(fd.source)
    eqs, why = equations(physics, D, params, t.flux or None)
    report_equations(ctx, eqs, why, _where(fd), "PC102", "PC301", t.tol("divergence"), t.tol("residual"), loc)


# ---------------------------------------------------------------------------------------------- BC / IC
def _face_select(fd: FieldData, key: str):
    """For grids and point clouds: ``"x=0"``, ``"x=min"``, ``"y=max"`` -> a selector of values on that plane."""
    k = key.replace(" ", "")
    if "=" not in k:
        return None
    a, v = k.split("=", 1)
    if fd.kind == "grid":
        if a not in fd.axes:
            return None
        ax = fd.axes[a]
        i = 0 if v == "min" else len(ax) - 1 if v == "max" else int(np.argmin(np.abs(ax - float(v))))
        if v not in ("min", "max") and abs(ax[i] - float(v)) > 1e-9 * max(np.ptp(ax), 1e-300):
            return None
        pos = list(fd.axes).index(a)
        return lambda arr: np.take(arr, i, axis=pos)
    if fd.points is not None and a in "xyz":
        c = fd.points[:, "xyz".index(a)]
        target = c.min() if v == "min" else c.max() if v == "max" else float(v)
        sel = np.abs(c - target) <= 1e-6 * max(np.ptp(c), 1e-300)
        return (lambda arr: np.asarray(arr)[sel]) if sel.any() else None
    return None


def _compare(declared: np.ndarray, actual: np.ndarray, scale_from: np.ndarray) -> tuple[float, float]:
    actual = np.asarray(actual, float)
    declared = np.broadcast_to(declared, actual.shape)
    err = np.abs(actual - declared)
    scale = max(float(np.nanmax(np.abs(scale_from))) if np.size(scale_from) else 0.0,
                float(np.max(np.abs(declared))) if declared.size else 0.0, 1e-300)
    return float(np.nanmax(err)) if err.size else 0.0, scale


def _coords_on(fd: FieldData, sel) -> dict[str, np.ndarray]:
    if fd.kind == "grid":
        mg = np.meshgrid(*[fd.axes[a] for a in fd.axes], indexing="ij")
        return {a: sel(m).ravel() for a, m in zip(fd.axes, mg, strict=False)}
    return {a: sel(fd.points[:, k]) for k, a in enumerate("xyz")}


def _bc(ctx: Ctx, t: ResultTarget, fd: FieldData) -> None:
    tol = t.tol("bc")
    if fd.kind == "fv":
        from ..fv import FV
        fv = FV(fd)
        Cf = fd.geom["face_centres"][fv.ni:]
        for patch, vals in t.bc.items():
            sl = fv.patch_slice(patch)
            for field, val in vals.items():
                loc = _field_loc(ctx, fd, field, patch)
                if sl is None:
                    ctx.emit("PC201", f"{field}@{patch}", False, f"patch '{patch}' is not in the mesh (patches: "
                             f"{', '.join(p['name'] for p in fd.patches)})", ctx.default_location())
                    continue
                if field not in fd.fields:
                    ctx.emit("PC201", f"{field}@{patch}", False, f"field {field} is not in the result", loc)
                    continue
                b = fv.bvals(field, fd.fields[field])[sl]
                xyz = {a: Cf[sl, k] for k, a in enumerate("xyz")}
                try:
                    dv = evaluate_expr(val, xyz, b.shape[0])
                except ValueError as e:
                    ctx.emit("PC201", f"{field}@{patch}", False, str(e), ctx.default_location())
                    continue
                err, scale = _compare(dv, b, fd.fields[field])
                ok = err / scale <= tol
                ctx.emit("PC201", f"{field}@{patch}", ok,
                         f"{field} on patch '{patch}': max deviation from the declared {_show(val)} is {fmt(err)} "
                         f"({fmt(err / scale)} of the field scale {'<=' if ok else '>'} {fmt(tol)})"
                         + ("" if ok else f"; found e.g. {_show(b[0])}"), loc, value=err / scale, threshold=tol)
        return
    loc = ctx.default_location() if t.declared else ctx.file_location(fd.source)
    for key, vals in t.bc.items():
        sel = _face_select(fd, key)
        for field, val in vals.items():
            if sel is None:
                ctx.emit("PC201", f"{field}@{key}", False, f"boundary '{key}' does not match a face of the data "
                                                          f"(use x=min, x=max or x=<value on the grid>)", loc)
                continue
            if field not in fd.fields:
                ctx.emit("PC201", f"{field}@{key}", False, f"field {field} is not in the result", loc)
                continue
            a = sel(fd.fields[field])
            a2 = a.reshape(-1, a.shape[-1]) if np.ndim(val) and a.ndim > 1 else a.ravel()
            dv = evaluate_expr(val, _coords_on(fd, sel), a2.shape[0])
            err, scale = _compare(dv, a2, fd.fields[field])
            ok = err / scale <= tol
            ctx.emit("PC201", f"{field}@{key}", ok, f"{field} on {key}: max deviation from {_show(val)} is "
                     f"{fmt(err)} ({fmt(err / scale)} of the field scale {'<=' if ok else '>'} {fmt(tol)})",
                     loc, value=err / scale, threshold=tol)


def _show(v) -> str:
    if isinstance(v, str):
        return repr(v)
    a = np.asarray(v, float)
    return fmt(float(a)) if a.ndim == 0 else "(" + " ".join(fmt(float(x)) for x in a.ravel()[:3]) + ")"


def _ic(ctx: Ctx, t: ResultTarget, fd: FieldData) -> None:
    tol = t.tol("ic")
    if fd.kind == "fv":
        first = fd.first or {"time": fd.time, "fields": fd.fields, "files": fd.files}
        C = fd.geom["cell_centres"]
        for field, val in t.ic.items():
            v = first["fields"].get(field)
            loc = ctx.file_location(first.get("files", {}).get(field), "internalField")
            if v is None:
                ctx.emit("PC202", field, False, f"field {field} is not written at t={first['time']}", loc)
                continue
            dv = evaluate_expr(val, {a: C[:, k] for k, a in enumerate("xyz")}, v.shape[0])
            err, scale = _compare(dv, v, v)
            ok = err / scale <= tol
            ctx.emit("PC202", field, ok, f"{field} at t={first['time']:.4g}: max deviation from the declared initial "
                     f"value {_show(val)} is {fmt(err / scale)} of the field scale ({'<=' if ok else '>'} {fmt(tol)})",
                     loc, value=err / scale, threshold=tol)
        return
    loc = ctx.default_location() if t.declared else ctx.file_location(fd.source)
    if fd.kind != "grid" or "t" not in fd.axes:
        for field in t.ic:
            ctx.emit("PC202", field, False, "an initial condition needs a time axis t in the data", loc)
        return
    sel = _face_select(fd, "t=min")
    for field, val in t.ic.items():
        if field not in fd.fields:
            ctx.emit("PC202", field, False, f"field {field} is not in the result", loc)
            continue
        a = sel(fd.fields[field]).ravel()
        dv = evaluate_expr(val, _coords_on(fd, sel), a.shape[0])
        err, scale = _compare(dv, a, fd.fields[field])
        ok = err / scale <= tol
        ctx.emit("PC202", field, ok, f"{field} at t={fd.axes['t'][0]:.4g}: max deviation from {_show(val)} is "
                 f"{fmt(err / scale)} of the field scale ({'<=' if ok else '>'} {fmt(tol)})", loc,
                 value=err / scale, threshold=tol)


# ---------------------------------------------------------------------------------------------- reference
def _reference(ctx: Ctx, t: ResultTarget, fd: FieldData) -> None:
    ref = t.reference
    loc = ctx.default_location()
    if not ref.path.exists():
        ctx.emit("PC703", "reference", False, f"the reference {ctx.cfg.rel(ref.path)} does not exist", loc)
        return
    try:
        rd = load_result(ref.path, "qoi" if fd.kind == "qoi" else None, t.time, None)
    except Exception as e:
        ctx.emit("PC703", "reference", False, f"cannot read the reference {ctx.cfg.rel(ref.path)}: {e}", loc)
        return
    tol = ref.rel_l2 if ref.rel_l2 is not None else t.tol("rel-l2")
    names = ref.fields or [n for n in fd.fields if n in rd.fields]
    if not names:
        ctx.emit("PC703", "reference", False, f"no field in common with the reference {ctx.cfg.rel(ref.path)}", loc)
        return
    ctx.emit("PC703", "reference", True, f"reference {ctx.cfg.rel(ref.path)} read ({rd.kind}, fields "
                                         f"{', '.join(n for n in names if n in rd.fields)})", loc)
    for n in names:
        if n not in fd.fields or n not in rd.fields:
            ctx.emit("PC703", n, False, f"{n} is missing from the {'result' if n not in fd.fields else 'reference'}",
                     loc)
            continue
        a, b = fd.flat(n), rd.flat(n)
        how = ""
        if a.shape != b.shape:
            try:
                b = _nearest(rd, fd, n)
                how = " (reference interpolated to the result points, nearest neighbour)"
            except Exception as e:
                ctx.emit("PC703", n, False, f"{n}: result {a.shape} and reference {b.shape} cannot be matched: {e}",
                         loc)
                continue
        e = rel_l2(a, b)
        extra = ""
        if fd.kind == "grid" and rd.kind == "grid" and fd.fields[n].shape == rd.fields[n].shape:
            try:
                fm = _field_metrics(fd.fields[n], rd.fields[n], len(fd.spatial_axes))
                extra = f"; nrmse {fmt(fm.nrmse)}, error of the mean {fmt(fm.crmse)}, boundary {fmt(fm.brmse)}"
            except Exception:
                extra = ""
        ok = e <= tol
        ctx.emit("PC701", n, ok, f"{n}: relative L2 error {fmt(e)} against {ctx.cfg.rel(ref.path)} "
                 f"{'<=' if ok else '>'} {fmt(tol)}{how}{extra if not ok else ''}", _field_loc(ctx, fd, n)
                 if not t.declared else loc, value=e, threshold=tol)
        if fd.kind == "grid" and rd.kind == "grid" and fd.fields[n].shape == rd.fields[n].shape:
            _spectral(ctx, t, fd, n, rd.fields[n], ref)


def _spectral(ctx: Ctx, t: ResultTarget, fd: FieldData, n: str, b: np.ndarray, ref) -> None:
    """PC704: high-wavenumber energy of the field against the reference (grid results of the same shape)."""
    from .. import _reuse
    names = list(fd.axes)
    nd = len(fd.spatial_axes)
    a, b = np.asarray(fd.fields[n], float), np.asarray(b, float)
    if a.ndim > len(names):                                           # components last -> first
        a, b = np.moveaxis(a, -1, 0), np.moveaxis(b, -1, 0)
    if "t" in names:                                                  # time axis -> leading
        it = names.index("t") + (a.ndim - len(names))
        a, b = np.moveaxis(a, it, 0), np.moveaxis(b, it, 0)
    if nd == 0 or min(a.shape[-nd:]) < 8:
        return
    try:
        r = _reuse.spectral().spectral_comparison(a, b, nd, periodic=False)
    except ValueError as e:
        ctx.emit("PC703", n, False, f"{n}: no spectral comparison with the reference ({e})", _field_loc(ctx, fd, n))
        return
    tol = t.tol("hf-bias")
    ok = abs(r.hf_bias) <= tol
    lo, hi = r.bands["high"]
    ctx.emit("PC704", n, ok, f"{n}: high-wavenumber energy {fmt(r.band_ratio['high'])} x the reference's "
             f"(shells {lo}-{hi}, bias {r.hf_bias:+.3g}, tolerance {fmt(tol)}); coherence {fmt(r.band_coherence['high'])}"
             f" against {ctx.cfg.rel(ref.path)}", _field_loc(ctx, fd, n), value=r.hf_bias, threshold=tol)


def _field_metrics(a, b, ndim):
    from .. import _reuse
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim > ndim:                                                 # vector: components first
        a, b = np.moveaxis(a, -1, 0), np.moveaxis(b, -1, 0)
    return _reuse.field_metrics().field_error_metrics(a, b, ndim=ndim)


def _nearest(rd: FieldData, fd: FieldData, n: str) -> np.ndarray:
    from scipy.spatial import cKDTree
    Xr, Xa = rd.coords(), fd.coords()
    if Xr.shape[1] != Xa.shape[1]:
        raise ValueError("different dimensions")
    _, idx = cKDTree(Xr).query(Xa)
    return rd.flat(n)[idx]


def _qoi(ctx: Ctx, t: ResultTarget, fd: FieldData) -> None:
    loc = ctx.default_location() if t.declared else ctx.file_location(fd.source)
    for k, v in fd.qoi.items():
        check_values(ctx, k, np.array([v]), t.units.get(k) or fd.units.get(k), ctx.file_location(fd.source, k))
        if k in t.units and k in fd.units:
            check_units(ctx, k, fd.units[k], t.units[k], None, ctx.file_location(fd.source, k), False)
    if t.reference is None:
        return
    ref = t.reference
    try:
        rd = load_result(ref.path, "qoi")
    except Exception as e:
        ctx.emit("PC703", "reference", False, f"cannot read the reference {ctx.cfg.rel(ref.path)}: {e}", loc)
        return
    names = list(ref.qoi) or list(rd.qoi)
    ctx.emit("PC703", "reference", True, f"reference {ctx.cfg.rel(ref.path)} read ({len(rd.qoi)} quantities)", loc)
    for k in names:
        if k not in rd.qoi or k not in fd.qoi:
            ctx.emit("PC703", k, False, f"{k} is missing from the {'reference' if k not in rd.qoi else 'result'}",
                     loc)
            continue
        tol = ref.qoi.get(k, {"rel": t.tol("qoi")})
        a, b = fd.qoi[k], rd.qoi[k]
        if k in fd.units and k in rd.units and fd.units[k] != rd.units[k]:
            from .. import _reuse
            try:
                b = float(_reuse.units().convert(b, rd.units[k], fd.units[k]))
            except Exception:
                ctx.emit("PC501", k, False, f"{k}: result in {fd.units[k]}, reference in {rd.units[k]}", loc)
                continue
        d = abs(a - b)
        ok_rel = "rel" in tol and d <= tol["rel"] * abs(b)
        ok_abs = "abs" in tol and d <= tol["abs"]
        ok = ok_rel or ok_abs
        lim = " or ".join(([f"rel {fmt(tol['rel'])}"] if "rel" in tol else []) + ([f"abs {fmt(tol['abs'])}"]
                                                                               if "abs" in tol else []))
        ctx.emit("PC702", k, ok, f"{k} = {fmt(a)}, reference {fmt(b)}: difference {fmt(d)} "
                 f"({fmt(d / abs(b)) if b else 'inf'} relative), tolerance {lim}",
                 ctx.file_location(fd.source, k), value=d, threshold=tol)
