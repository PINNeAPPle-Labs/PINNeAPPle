"""Run a drafted model (:class:`pinneapple_problemdesign.formulate.ModelDraft`) as a job (#1246).

:func:`run_draft` executes the system a draft states, on a box domain:

* **classical** (``method="fd"``, or ``"auto"`` when the draft allows it): a scalar diffusion-reaction law
  ``a*T_t = div(K grad T) + S(T, x, t)`` (heat conduction, a battery cell's Bernardi heat, a species diffusing with
  a source) with linear conditions ``alpha*T_n + beta*T + gamma = 0`` on every face (Dirichlet, Neumann, Robin /
  convection) is discretised by second-order finite differences on a node grid (ghost points for the flux
  conditions) and integrated by Crank-Nicolson, with Newton iterations for a source that depends on ``T``
  (``exp(EaR*(1/T - 1/Tref))``, ``-I*T*dUdT``). The coefficients of the derivatives must be constants; the source
  may be any expression of the field, the coordinates and time;
* **PINN** (``method="pinn"``, or ``"auto"`` when no classical path exists): the draft's own compiled loss
  (``pinn_problem`` on ``draft.system()`` with ``draft.scales()``, the code of ``draft.loss_code``) trained by
  ``train_pinn`` and checked by ``validate_pinn`` on points never trained on; when a classical path exists too, the
  PINN is compared with it.

Checks in every result:

* ``run.mms_order``: a manufactured solution of the **same** law and conditions (its source and boundary data
  computed by SymPy) solved on three nested grids; the observed order of the maximum error is stated next to the
  expected one (2 for this scheme, in space and, with the time step refined with the grid, in time);
* ``run.balance``: the balance of the drafted law over the box (``d/dt int a*T dV = int S dV + boundary inflow``),
  each term computed from the solution and the conditions, not from the assembled matrix;
* ``run.assumptions``: every assumption the draft applied or offered, re-stated, with its criterion evaluated on
  the box (``Bi = h*Lc/k``, ``Lc = V/A``) and, for the lumped body, the internal temperature spread the solution
  shows next to the lumped estimate;
* the draft's own checks (dimensions, closure, conditions on every face, constants, compiled residual vs SymPy).

A draft the library cannot discretise is **refused** (``status == "refused"``) with the reason named: an algebraic
relation (nothing to discretise: it is calibrated, ``loss_code``), several coupled fields (the incompressible flow
of ``room_air_species``), an advective or mixed derivative, a nonlinear condition, a missing constant or face.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

METHODS = ("auto", "fd", "pinn")
EXPECTED_ORDER = 2.0
ORDER_TOLERANCE = 0.2            # the observed order must be at least EXPECTED_ORDER - ORDER_TOLERANCE
BALANCE_TOLERANCE = 1e-6         # flux conditions only: the discrete balance closes to the Newton tolerance


class Refused(ValueError):
    """The draft cannot run by the method asked; the message is the reason."""


@dataclass
class DraftRun:
    """A run of a drafted model: job-friendly (:meth:`to_dict` is JSON)."""
    status: str                                  # done | refused
    method: str                                  # fd | pinn | "" (refused)
    title: str = ""
    reason: str = ""                             # why it was refused, or why "auto" chose this method
    coords: list[str] = field(default_factory=list)
    field_name: str = ""
    unit: str = ""
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    axes: dict[str, np.ndarray] = field(default_factory=dict)
    fields: dict[str, np.ndarray] = field(default_factory=dict)       # field at the final time on the grid
    history: dict[str, list[float]] = field(default_factory=dict)     # t, mean, min, max of the field
    metrics: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[dict[str, Any]] = field(default_factory=list)
    mms: dict[str, Any] = field(default_factory=dict)
    balance: dict[str, Any] = field(default_factory=dict)
    next: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "done" and all(c["status"] != "fail" for c in self.checks)

    def to_dict(self, *, include_fields: bool = False) -> dict[str, Any]:
        out = {"status": self.status, "method": self.method, "title": self.title, "reason": self.reason,
               "coords": self.coords, "field": self.field_name, "unit": self.unit,
               "bounds": {k: list(v) for k, v in self.bounds.items()}, "history": self.history,
               "metrics": _jsonable(self.metrics), "checks": _jsonable(self.checks),
               "assumptions": _jsonable(self.assumptions), "mms": _jsonable(self.mms),
               "balance": _jsonable(self.balance), "next": self.next, "passed": self.passed,
               "grid": {k: len(v) for k, v in self.axes.items()}}
        if include_fields:
            out["axes"] = {k: v.tolist() for k, v in self.axes.items()}
            out["fields"] = {k: v.tolist() for k, v in self.fields.items()}
        return out

    def summary(self) -> str:
        if self.status != "done":
            return f"{self.title}: refused: {self.reason}"
        m = self.metrics
        bad = [c["code"] for c in self.checks if c["status"] == "fail"]
        return (f"{self.title} ({self.method}): {self.field_name} in [{m.get('min', float('nan')):.6g}, "
                f"{m.get('max', float('nan')):.6g}] {self.unit} at t = {m.get('t_end', 0):g} s; MMS order "
                f"{self.mms.get('observed_order', float('nan')):.3g} (expected {EXPECTED_ORDER:g}); "
                + ("fail: " + ", ".join(bad) if bad else "every check passes or is inconclusive"))


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


# --------------------------------------------------------------------------------------------- the scalar form
@dataclass
class _Face:
    axis: str
    side: str                       # min | max
    alpha: sp.Expr                  # alpha*T_n + beta*T + gamma = 0 (T_n: derivative along the axis, not outward)
    beta: sp.Expr
    gamma: sp.Expr
    text: str


@dataclass
class _Form:
    """``a*T_t = sum_i K_i*T_ii + S(T, x, t)`` with a linear condition on each face and an initial field."""
    field: str
    coords: list[str]
    spatial: list[str]
    steady: bool
    a: float
    K: dict[str, float]
    S: sp.Expr                      # of Tv, the coordinates and t
    faces: dict[tuple[str, str], _Face]
    initial: sp.Expr | None         # of the coordinates
    Tv: sp.Symbol
    syms: dict[str, sp.Symbol]


def _values(draft) -> dict[sp.Symbol, sp.Float]:
    return {sp.Symbol(k, real=True): sp.Float(c.value) for k, c in draft.coefficients.items() if c.value is not None}


def scalar_form(draft) -> _Form:
    """The draft's system in the form the box finite-difference solver discretises; :class:`Refused` names what
    does not fit."""
    if draft.relation:
        raise Refused(f"{draft.title!r} is an algebraic relation ({draft.relation['equation']}): there is nothing to "
                      "discretise; calibrate its unknowns from measurements (draft.loss_code(), inverse.workflow)")
    if len(draft.fields) != 1:
        raise Refused(f"{len(draft.fields)} coupled fields ({', '.join(draft.fields)}; laws {', '.join(draft.laws)}): "
                      "the library has no discretisation of a drafted coupled system (incompressible flow with "
                      "energy and species) on a box; method='pinn' trains on the draft's own loss")
    s = draft.system()
    if len(s.pdes) != 1:
        raise Refused(f"{len(s.pdes)} equations for one field: the box solver takes one conservation law")
    f = draft.fields and next(iter(draft.fields))
    ctx = s.ctx
    syms = dict(ctx.symbols)
    spatial = list(ctx.spatial)
    steady = ctx.time not in ctx.coords
    T = ctx.functions[f]
    Tv = sp.Symbol("T__value", real=True)
    rep = _values(draft)
    R = sp.expand(s.pdes[0].residual.xreplace(rep).doit())
    placeholders: dict[sp.Derivative, sp.Symbol] = {}
    kinds: dict[sp.Symbol, str] = {}
    for d in R.atoms(sp.Derivative):
        wrt = [(v.name, int(n)) for v, n in d.variable_count]
        if len(wrt) == 1 and wrt[0] == ("t", 1):
            kind = "t"
        elif len(wrt) == 1 and wrt[0][1] == 2 and wrt[0][0] in spatial:
            kind = wrt[0][0]
        else:
            what = "a first spatial derivative (advection)" if len(wrt) == 1 and wrt[0][1] == 1 else \
                "a mixed or higher derivative"
            raise Refused(f"the term {sp.sstr(d)} is {what}: the box finite-difference path discretises "
                          "a*T_t = K*lap(T) + S(T, x, t) only; method='pinn' trains on the draft's loss")
        ph = sp.Symbol(f"D__{kind}", real=True)
        placeholders[d] = ph
        kinds[ph] = kind
    Rs = R.xreplace(placeholders).xreplace({T: Tv})
    if Rs.has(sp.Function) and any(isinstance(a, sp.core.function.AppliedUndef) for a in Rs.atoms(sp.Function)):
        raise Refused("the equation holds a field term the box solver cannot read")
    a = 0.0
    K: dict[str, float] = {}
    for ph, kind in kinds.items():
        c = sp.diff(Rs, ph)
        if c.free_symbols & ({Tv} | set(placeholders.values()) | set(syms.values())):
            raise Refused(f"the coefficient of {'T_t' if kind == 't' else f'{f}_{kind}{kind}'} ({sp.sstr(c)}) depends "
                          "on the field or the position: the box solver takes constant coefficients there")
        if c.free_symbols:
            raise Refused(f"constants to be given: {', '.join(sorted(str(x) for x in c.free_symbols))}")
        if kind == "t":
            a = float(c)
        else:
            K[kind] = -float(c)
    rest = Rs.xreplace({ph: 0 for ph in kinds})
    if sp.simplify(Rs - rest - sum(sp.diff(Rs, ph) * ph for ph in kinds)) != 0:
        raise Refused("the equation is not linear in the field's derivatives")
    S = -rest
    free = S.free_symbols - {Tv} - set(syms.values())
    if free:
        raise Refused(f"constants to be given: {', '.join(sorted(str(x) for x in free))}")
    for x in spatial:
        if K.get(x, 0.0) <= 0.0:
            raise Refused(f"no diffusion along {x} (K = {K.get(x, 0.0):g}): the box solver needs K > 0 on every axis")
    if not steady and a <= 0.0:
        raise Refused(f"the coefficient of the time derivative is {a:g}: a transient law needs a > 0")
    faces: dict[tuple[str, str], _Face] = {}
    initial = None
    for c in s.conditions:
        lhs, rhs = c.lhs.xreplace(rep), c.rhs.xreplace(rep)
        if c.kind == "initial":
            if lhs != T:
                raise Refused(f"initial condition {c.text!r}: the box solver takes '{f} = expression of x'")
            initial = rhs.doit()
            if initial.has(T) or initial.atoms(sp.Derivative) or (initial.free_symbols - set(syms.values())):
                raise Refused(f"initial condition {c.text!r}: constants to be given or a field on the right side")
            continue
        if c.kind == "periodic":
            raise Refused(f"condition {c.text!r}: periodic faces are not in the box solver")
        on = [(x, sd) for x in spatial for sd in ("min", "max")] if c.on == "boundary" else \
            ([(c.on, "min"), (c.on, "max")] if c.on in spatial else [tuple(c.on.split("_", 1))])
        Rc = sp.expand((lhs - rhs).doit())
        for axis, side in on:
            ph = sp.Symbol("D__n", real=True)
            sub = {}
            for d in Rc.atoms(sp.Derivative):
                wrt = [(v.name, int(n)) for v, n in d.variable_count]
                if wrt != [(axis, 1)]:
                    raise Refused(f"condition {c.text!r} on {axis}_{side}: only the derivative along {axis} may "
                                  "appear on that face")
                sub[d] = ph
            Rf = Rc.xreplace(sub).xreplace({T: Tv})
            al, be = sp.sympify(sp.diff(Rf, ph)), sp.sympify(sp.diff(Rf, Tv))
            ga = sp.sympify(Rf.xreplace({ph: 0, Tv: 0}))
            if (al.free_symbols | be.free_symbols) & {Tv, ph} or sp.simplify(Rf - (al * ph + be * Tv + ga)) != 0:
                raise Refused(f"condition {c.text!r} is nonlinear in the field: the box solver takes "
                              "alpha*T_n + beta*T + gamma = 0")
            free = (al.free_symbols | be.free_symbols | ga.free_symbols) - set(syms.values())
            if free:
                raise Refused(f"constants to be given: {', '.join(sorted(str(x) for x in free))}")
            if al == 0 and be == 0:
                raise Refused(f"condition {c.text!r} does not involve the field")
            if (axis, side) in faces:
                raise Refused(f"two conditions on {axis}_{side}: {faces[(axis, side)].text!r} and {c.text!r}")
            faces[(axis, side)] = _Face(axis, side, al, be, ga, c.text)
    missing = [f"{x}_{sd}" for x in spatial for sd in ("min", "max") if (x, sd) not in faces]
    if missing:
        raise Refused(f"no condition on {', '.join(missing)}")
    if not steady and initial is None:
        raise Refused("a transient law needs an initial condition")
    return _Form(f, list(ctx.coords), spatial, steady, a, K, S, faces, initial, Tv, syms)


# --------------------------------------------------------------------------------------------- the solver
class _Grid:
    def __init__(self, bounds: dict[str, tuple[float, float]], spatial: list[str], n: dict[str, int]):
        import scipy.sparse as sps
        self.spatial = spatial
        self.axes = {x: np.linspace(*bounds[x], n[x]) for x in spatial}
        self.h = {x: (bounds[x][1] - bounds[x][0]) / (n[x] - 1) for x in spatial}
        self.shape = tuple(n[x] for x in spatial)
        self.N = int(np.prod(self.shape))
        mesh = np.meshgrid(*[self.axes[x] for x in spatial], indexing="ij")
        self.P = {x: m.ravel() for x, m in zip(spatial, mesh, strict=True)}
        w = np.ones(self.shape)                 # trapezoid weights
        for i, x in enumerate(spatial):
            wi = np.full(n[x], self.h[x])
            wi[0] = wi[-1] = self.h[x] / 2
            w = w * wi.reshape([-1 if j == i else 1 for j in range(len(spatial))])
        self.w = w.ravel()
        idx = np.arange(self.N).reshape(self.shape)
        self.face = {}
        self.face_w = {}
        for i, x in enumerate(spatial):
            for side, k in (("min", 0), ("max", n[x] - 1)):
                sl = [slice(None)] * len(spatial)
                sl[i] = k
                self.face[(x, side)] = idx[tuple(sl)].ravel()
                fw = np.ones([n[y] for y in spatial if y != x]) if len(spatial) > 1 else np.ones(1)
                for j, y in enumerate([y for y in spatial if y != x]):
                    wy = np.full(n[y], self.h[y])
                    wy[0] = wy[-1] = self.h[y] / 2
                    fw = fw * wy.reshape([-1 if jj == j else 1 for jj in range(len(spatial) - 1)])
                self.face_w[(x, side)] = fw.ravel()
        self._eye = [sps.identity(n[x], format="csr") for x in spatial]

    def second_difference(self, x: str):
        """The 1-D second difference along ``x`` with the ghost-point rows (2T1 - 2T0)/h^2 at both ends, on the
        whole grid."""
        import scipy.sparse as sps
        i = self.spatial.index(x)
        m = len(self.axes[x])
        h2 = self.h[x] ** 2
        main = np.full(m, -2.0 / h2)
        up = np.full(m - 1, 1.0 / h2)
        lo = np.full(m - 1, 1.0 / h2)
        up[0] = 2.0 / h2
        lo[-1] = 2.0 / h2
        A = sps.diags([lo, main, up], [-1, 0, 1], format="csr")
        out = None
        for j in range(len(self.spatial)):
            B = A if j == i else self._eye[j]
            out = B if out is None else sps.kron(out, B, format="csr")
        return out


def _lam(expr: sp.Expr, form: _Form):
    args = [form.Tv] + [form.syms[c] for c in form.coords]
    fn = sp.lambdify(args, expr, "numpy")

    def call(T, P, t):
        vals = [T] + [P[c] if c != "t" else t for c in form.coords]
        out = fn(*vals)
        return np.broadcast_to(np.asarray(out, dtype=float), np.shape(T) if np.ndim(T) else
                               np.shape(next(iter(P.values())))).astype(float)
    return call


class _Solver:
    """Crank-Nicolson (or a steady Newton solve) of a :class:`_Form` on a :class:`_Grid`."""

    def __init__(self, form: _Form, grid: _Grid, *, extra_source: sp.Expr | None = None,
                 face_gamma: dict[tuple[str, str], sp.Expr] | None = None, initial: sp.Expr | None = None):
        self.form, self.g = form, grid
        S = form.S + (extra_source if extra_source is not None else 0)
        self.S = _lam(S, form)
        self.dS = _lam(sp.diff(S, form.Tv), form)
        self.faces = {}
        for key, fc in form.faces.items():
            gam = face_gamma[key] if face_gamma and key in face_gamma else fc.gamma
            self.faces[key] = (fc, _lam(fc.alpha, form), _lam(fc.beta, form), _lam(gam, form))
        self.initial = _lam(initial if initial is not None else (form.initial if form.initial is not None
                                                                 else sp.Float(0)), form)
        self.L0 = sum(form.K[x] * grid.second_difference(x) for x in form.spatial)
        self.dirichlet = np.zeros(grid.N, dtype=bool)
        for key, (fc, *_) in self.faces.items():
            if fc.alpha == 0:
                self.dirichlet[grid.face[key]] = True

    def _sub(self, key, t):
        g = self.g
        idx = g.face[key]
        P = {x: g.P[x][idx] for x in g.spatial}
        fc, al, be, ga = self.faces[key]
        z = np.zeros(len(idx))
        return idx, P, al(z, P, t), be(z, P, t), ga(z, P, t)

    def linear(self, t):
        """``(L, c, gD)``: the operator with the flux conditions, its constant part, the Dirichlet values."""
        import scipy.sparse as sps
        g, F = self.g, self.form
        diag = np.zeros(g.N)
        c = np.zeros(g.N)
        gD = np.full(g.N, np.nan)
        for key in self.faces:
            x, side = key
            idx, P, al, be, ga = self._sub(key, t)
            if np.all(al == 0):
                gD[idx] = -ga / be
                continue
            sgn = 1.0 if side == "min" else -1.0
            np.add.at(diag, idx, sgn * F.K[x] * (2.0 / g.h[x]) * be / al)
            np.add.at(c, idx, sgn * F.K[x] * (2.0 / g.h[x]) * ga / al)
        return self.L0 + sps.diags(diag), c, gD

    def G(self, T, t, lin=None):
        L, c, _ = lin or self.linear(t)
        return L @ T + c + self.S(T, self.g.P, t)

    def inflow(self, T, t) -> float:
        """Net inflow through the faces, from the conditions (K*T_n at max minus at min, integrated)."""
        g, F = self.g, self.form
        total = 0.0
        for key in self.faces:
            x, side = key
            idx, P, al, be, ga = self._sub(key, t)
            if np.all(al == 0):                 # Dirichlet: second-order one-sided derivative
                axis = g.spatial.index(x)
                Tg = T.reshape(g.shape)
                m = g.shape[axis]
                p0, p1, p2 = (np.take(Tg, k, axis=axis).ravel() for k in ((0, 1, 2) if side == "min" else
                                                                         (m - 1, m - 2, m - 3)))
                dT = (1.0 if side == "min" else -1.0) * (-3 * p0 + 4 * p1 - p2) / (2 * g.h[x])
            else:
                dT = -(be * T[idx] + ga) / al
            total += (1.0 if side == "max" else -1.0) * F.K[x] * float(np.dot(g.face_w[key], dT))
        return total

    def _newton(self, T, F_of, J_of, mask, gD, key=None, tol=1e-11, maxit=60):
        """Newton with the factorised Jacobian kept between iterations and steps (``key``: the step size and
        weight); it is refactorised when an iteration contracts less than by half."""
        import scipy.sparse as sps
        from scipy.sparse.linalg import splu
        T = T.copy()
        T[mask] = gD[mask]
        cache = self.__dict__.setdefault("_lu", {})

        def factor(T):
            J = J_of(T)
            if mask.any():                       # Dirichlet rows: T - g = 0
                J = sps.diags((~mask).astype(float)) @ J + sps.diags(mask.astype(float))
            cache[key] = splu(sps.csc_matrix(J))
            return cache[key]
        if key in cache:
            lu, fresh = cache[key], False
        else:
            lu, fresh = factor(T), True
        last = math.inf
        for it in range(maxit):
            r = F_of(T)
            r[mask] = T[mask] - gD[mask]
            dT = lu.solve(-r)
            T += dT
            step = float(np.max(np.abs(dT)))
            if step <= tol * max(1.0, float(np.max(np.abs(T)))):
                return T, it + 1
            if step > 0.5 * last and not fresh:  # poor contraction: a fresh Jacobian
                lu, fresh = factor(T), True
            else:
                fresh = False
            last = step
        raise RuntimeError(f"Newton did not converge in {maxit} iterations (last step {step:.3g})")

    def steady(self, T0=None):
        import scipy.sparse as sps
        g = self.g
        lin = self.linear(0.0)
        T = np.zeros(g.N) if T0 is None else T0
        T, its = self._newton(T, lambda T: self.G(T, 0.0, lin),
                              lambda T: lin[0] + sps.diags(self.dS(T, g.P, 0.0)), self.dirichlet, lin[2],
                              key=("steady",))
        Qs = float(np.dot(g.w, self.S(T, g.P, 0.0)))
        B = self.inflow(T, 0.0)
        return T, {"t": [0.0], "newton": its, "source": Qs, "inflow": B, "stored": 0.0}

    def transient(self, t_end: float, n_t: int, theta: float = 0.5, record: int = 50, startup: int = 2):
        """Crank-Nicolson, its first ``startup`` steps replaced by twice as many backward-Euler half steps
        (Rannacher), so the stiff modes of the initial data are damped instead of oscillating."""
        import scipy.sparse as sps
        g, a = self.g, self.form.a
        dt_full = t_end / n_t
        k0 = min(startup, n_t) if theta < 1 else 0
        steps = [(dt_full / 2, 1.0)] * (2 * k0) + [(dt_full, theta)] * (n_t - k0)
        T = self.initial(np.zeros(g.N), g.P, 0.0).copy()
        lin0 = self.linear(0.0)
        T[self.dirichlet] = lin0[2][self.dirichlet]
        E0 = float(np.dot(g.w, a * T))
        hist = {"t": [0.0], "mean": [float(np.dot(g.w, T) / g.w.sum())], "min": [float(T.min())],
                "max": [float(T.max())]}
        Qs = B = 0.0
        S0, B0 = float(np.dot(g.w, self.S(T, g.P, 0.0))), self.inflow(T, 0.0)
        its_max = 0
        every = max(1, len(steps) // record)
        t1 = 0.0
        for n, (dt, th) in enumerate(steps):
            t0, t1 = t1, (t1 + dt if n < len(steps) - 1 else t_end)
            G0 = self.G(T, t0, lin0)
            lin1 = self.linear(t1)
            Tn = T

            def F_of(T, Tn=Tn, G0=G0, lin1=lin1, t1=t1, dt=dt, th=th):
                return a * (T - Tn) / dt - th * self.G(T, t1, lin1) - (1 - th) * G0

            def J_of(T, lin1=lin1, t1=t1, dt=dt, th=th):
                return sps.identity(g.N, format="csr") * (a / dt) - th * (lin1[0] + sps.diags(self.dS(T, g.P, t1)))
            T, its = self._newton(Tn, F_of, J_of, self.dirichlet, lin1[2], key=(dt, th))
            its_max = max(its_max, its)
            S1, B1 = float(np.dot(g.w, self.S(T, g.P, t1))), self.inflow(T, t1)
            Qs += dt * (th * S1 + (1 - th) * S0)
            B += dt * (th * B1 + (1 - th) * B0)
            S0, B0, lin0 = S1, B1, lin1
            if (n + 1) % every == 0 or n == len(steps) - 1:
                hist["t"].append(t1)
                hist["mean"].append(float(np.dot(g.w, T) / g.w.sum()))
                hist["min"].append(float(T.min()))
                hist["max"].append(float(T.max()))
        hist.update(newton=its_max, source=Qs, inflow=B, stored=float(np.dot(g.w, a * T)) - E0)
        return T, hist


# --------------------------------------------------------------------------------------------- checks
def _manufactured(form: _Form, bounds, t_end: float, offset: float, scale: float) -> sp.Expr:
    """A smooth solution that satisfies none of the conditions by itself (so their data are exercised)."""
    s = form.syms
    prod = sp.Integer(1)
    for i, x in enumerate(form.spatial):
        lo, hi = bounds[x]
        prod *= sp.cos(sp.Rational(3, 4) * sp.pi * (s[x] - lo) / (hi - lo) + sp.Rational(2, 5) + i / 7)
    time_factor = sp.exp(-s["t"] / (2 * t_end)) if not form.steady else 1
    return sp.Float(offset) + sp.Float(scale) * (1 + sp.Rational(1, 5) * prod * time_factor)


def mms_order(form: _Form, bounds, *, t_end: float, offset: float, scale: float, levels: tuple[int, ...],
              steps: float = 1.0) -> dict[str, Any]:
    """The manufactured solution of the same law and conditions on nested grids: errors and observed orders.
    A transient law takes ``steps*(n-1)**2`` time steps on the grid of ``n`` nodes (dt refined with h^2), so the
    measured order is the spatial one and the time error (second order) stays below it."""
    Tm = _manufactured(form, bounds, t_end, offset, scale)
    s = form.syms
    f_src = (form.a * sp.diff(Tm, s["t"]) if not form.steady else 0) - sum(
        form.K[x] * sp.diff(Tm, s[x], 2) for x in form.spatial) - form.S.xreplace({form.Tv: Tm})
    gam = {key: -(fc.alpha * sp.diff(Tm, s[fc.axis]) + fc.beta * Tm) for key, fc in form.faces.items()}
    exact = _lam(Tm, form)
    rows = []
    for n in levels:
        g = _Grid(bounds, form.spatial, {x: n for x in form.spatial})
        sol = _Solver(form, g, extra_source=f_src, face_gamma=gam, initial=Tm.xreplace({s["t"]: 0}) if "t" in s
                      else Tm)
        if form.steady:
            T, _ = sol.steady(np.full(g.N, offset + scale))
            tt = 0.0
        else:
            T, _ = sol.transient(t_end, max(4, round(steps * (n - 1) ** 2)))
            tt = t_end
        err = float(np.max(np.abs(T - exact(np.zeros(g.N), g.P, tt))))
        rows.append({"n": n, "h": max(g.h.values()), "max_error": err, "max_error_rel": err / scale})
    for a, b in zip(rows, rows[1:], strict=False):
        b["order"] = math.log(a["max_error"] / b["max_error"]) / math.log(a["h"] / b["h"]) \
            if a["max_error"] > 0 and b["max_error"] > 0 else float("nan")
    obs = rows[-1].get("order", float("nan"))
    return {"solution": sp.sstr(Tm), "levels": rows, "observed_order": obs, "expected_order": EXPECTED_ORDER,
            "time_step": "dt refined with h^2 (the spatial order is measured)" if not form.steady else "steady"}


def _assumption_rows(draft, form: _Form | None, bounds, T=None, grid=None) -> list[dict[str, Any]]:
    """Every assumption of the draft, re-stated, with its criterion on this box (and on the solution)."""
    vals = {k: c.value for k, c in draft.coefficients.items() if c.value is not None}
    spatial = [c for c in draft.coords if c != "t"]
    lengths = [bounds[x][1] - bounds[x][0] for x in spatial]
    V = float(np.prod(lengths)) if lengths else float("nan")
    A = 0.0
    for i in range(len(lengths)):
        A += 2 * float(np.prod([L for j, L in enumerate(lengths) if j != i])) if len(lengths) > 1 else 2.0
    Lc = V / A if A else float("nan")
    out = []
    for a in draft.assumptions:
        row = {"id": a.id, "title": a.title, "applied": a.applied, "drops": a.drops, "criterion": a.criterion,
               "limit": a.limit, "source": a.source, "value": a.value, "holds": a.holds}
        if a.id == "lumped" and "h" in vals and "k" in vals and math.isfinite(Lc):
            bi = vals["h"] * Lc / vals["k"]
            row.update(value=bi, holds=bool(bi < 0.1), evaluated=f"Bi = h*Lc/k = {vals['h']:g}*{Lc:.4g}/{vals['k']:g}"
                                                                    f" = {bi:.3g} on this box (Lc = V/A)")
            if T is not None and grid is not None and "Tamb" in vals:
                rise = float(np.dot(grid.w, T) / grid.w.sum()) - vals["Tamb"]
                spread = float(T.max() - T.min())
                row["solution"] = {"internal_spread": spread, "mean_minus_ambient": rise,
                                   "spread_over_rise": spread / abs(rise) if rise else float("nan")}
                row["evaluated"] += (f"; the solution's internal spread is {spread:.3g} K for a mean "
                                     f"{rise:+.3g} K from ambient")
        out.append(row)
    return out


def _draft_checks(draft) -> list[dict[str, Any]]:
    return [{"code": c["code"], "status": c["status"], "detail": c["detail"]} for c in draft.checks()]


# --------------------------------------------------------------------------------------------- entry point
def _bounds(draft, geometry) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]]]:
    spatial = [c for c in draft.coords if c != "t"]
    notes = []
    if geometry is None:
        geometry = {}
        notes.append({"id": "geometry_default", "title": "No geometry given: the unit box [0, 1] m on every axis",
                      "applied": True, "source": "run_draft default"})
    if isinstance(geometry, (list, tuple)):
        geometry = {x: (0.0, float(L)) for x, L in zip(spatial, geometry, strict=False)}
    if "box" in geometry:
        geometry = {x: (0.0, float(L)) for x, L in zip(spatial, geometry["box"], strict=False)}
    out = {}
    for x in spatial:
        v = geometry.get(x, (0.0, 1.0))
        lo, hi = (0.0, float(v)) if isinstance(v, (int, float)) else (float(v[0]), float(v[1]))
        if not hi > lo:
            raise Refused(f"the box along {x} is empty: {lo} .. {hi}")
        out[x] = (lo, hi)
    notes.append({"id": "box_domain", "title": "Box domain " + " x ".join(f"[{lo:g}, {hi:g}] m" for lo, hi in
                                                                        out.values()),
                  "applied": True, "source": "run_draft: the faces of the box carry the draft's conditions"})
    return out, notes


def _default_levels(dims: int) -> tuple[int, ...]:
    return {1: (11, 21, 41), 2: (9, 17, 33), 3: (5, 9, 17)}[dims]


def run_draft(draft, geometry=None, method: str = "auto", budget: dict[str, Any] | None = None, *,
              mms: bool = True) -> DraftRun:
    """Run ``draft`` (a :class:`~pinneapple_problemdesign.formulate.ModelDraft`) on a box.

    ``geometry``: ``{"x": (lo, hi), ...}`` in metres, ``{"box": [Lx, Ly, Lz]}`` or ``[Lx, Ly, Lz]``; None: the unit
    box (stated). ``method``: ``"auto"`` (classical when the draft fits the box solver, else the PINN), ``"fd"``,
    ``"pinn"``. ``budget``: ``n`` (nodes per axis), ``t_end`` (s; default one diffusion time ``a*L^2/K``), ``n_t``
    (time steps), ``mms_levels`` (nodes per axis of the three MMS grids), ``adam_steps`` / ``lbfgs_steps`` /
    ``n_collocation`` / ``width`` / ``depth`` (PINN), ``n_evaluations`` (how many runs a study needs: the
    ``recommend_architecture`` answer on a surrogate is then attached).

    A draft that cannot run is returned with ``status == "refused"`` and the reason; nothing is solved silently in
    a simplified form."""
    budget = dict(budget or {})
    title = getattr(draft, "title", "draft")
    if method not in METHODS:
        return DraftRun("refused", "", title, reason=f"method {method!r} is not one of {METHODS} (the classical path "
                                                       "here is box finite differences: method='fd')")
    if not draft.relation:
        bad = [c for c in draft.checks() if c["status"] != "pass" and c["code"] != "model.mms"]
        if bad:
            return DraftRun("refused", "", title, reason="the draft does not pass its own checks: " + "; ".join(
                f"{c['code']}: {c['detail']}" for c in bad), checks=_draft_checks(draft))
    form, why_not = None, ""
    try:
        form = scalar_form(draft)
    except Refused as e:
        why_not = str(e)
    if method == "fd" and form is None:
        return DraftRun("refused", "", title, reason=f"no discretisation for this draft: {why_not}")
    if draft.relation:
        return DraftRun("refused", "", title, reason=why_not)
    try:
        bounds, notes = _bounds(draft, geometry)
    except Refused as e:
        return DraftRun("refused", "", title, reason=str(e))
    use = "fd" if (method == "fd" or (method == "auto" and form is not None)) else "pinn"
    reason = ("the draft is a scalar diffusion-reaction law with linear face conditions: box finite differences"
              if use == "fd" and method == "auto" else
              f"no classical discretisation ({why_not}): PINN on the draft's loss" if use == "pinn" and method ==
              "auto" else f"method={method!r} asked")
    t0 = time.perf_counter()
    if use == "fd":
        run = _run_fd(draft, form, bounds, budget, mms=mms)
    else:
        run = _run_pinn(draft, form, bounds, budget)
    run.reason = reason
    run.assumptions = notes + run.assumptions
    run.metrics["seconds"] = time.perf_counter() - t0
    if budget.get("n_evaluations"):
        from pinneapple_analysis.verification.architecture_recommendation import (
            recommend_architecture,
        )
        rec = recommend_architecture(has_analytical_solver=use == "fd", solve_seconds=run.metrics.get("solve_seconds"),
                                     n_evaluations=int(budget["n_evaluations"]),
                                     needs_parameter_generalization=True, output_kind="field")
        run.metrics["surrogate.recommended"] = (rec.recommended or ["none"])[0]
        run.next.append(f"surrogate for {int(budget['n_evaluations'])} evaluations: {rec.reasoning}")
    return run


def _t_end(form: _Form, bounds, budget) -> tuple[float, str]:
    if form.steady:
        return 0.0, ""
    if budget.get("t_end"):
        return float(budget["t_end"]), "given"
    L = max(bounds[x][1] - bounds[x][0] for x in form.spatial)
    return form.a * L ** 2 / max(form.K.values()), "one diffusion time a*L^2/K (no t_end given)"


def _run_fd(draft, form: _Form, bounds, budget, *, mms: bool) -> DraftRun:
    dims = len(form.spatial)
    n = int(budget.get("n", {1: 81, 2: 41, 3: 17}[dims]))
    t_end, t_note = _t_end(form, bounds, budget)
    n_t = int(budget.get("n_t", 200))
    grid = _Grid(bounds, form.spatial, {x: n for x in form.spatial})
    sol = _Solver(form, grid)
    off, sc = draft.scales().get(form.field, (0.0, 1.0))
    ts = time.perf_counter()
    if form.steady:
        T, h = sol.steady(np.full(grid.N, off))
    else:
        T, h = sol.transient(t_end, n_t)
    solve_s = time.perf_counter() - ts
    dt_check = None
    if not form.steady and budget.get("time_check", True):
        T2, _ = sol.transient(t_end, 2 * n_t)       # the same grid with half the time step
        span = max(float(T2.max() - T2.min()), float(np.max(np.abs(T2 - off))), sc)
        dt_check = (float(np.max(np.abs(T - T2))), span)
    unit = draft.fields[form.field]
    run = DraftRun("done", "fd", draft.title, coords=list(form.coords), field_name=form.field, unit=unit,
                   bounds=bounds, axes=dict(grid.axes), fields={form.field: T.reshape(grid.shape)},
                   history={k: h[k] for k in ("t", "mean", "min", "max") if k in h})
    i = int(np.argmax(T))
    run.metrics.update({"t_end": t_end, "max": float(T.max()), "min": float(T.min()),
                        "mean": float(np.dot(grid.w, T) / grid.w.sum()),
                        "argmax": {x: float(grid.P[x][i]) for x in form.spatial}, "nodes": grid.N,
                        "time_steps": 0 if form.steady else n_t, "newton_iterations_max": h["newton"],
                        "solve_seconds": solve_s})
    if t_note:
        run.metrics["t_end_from"] = t_note
    if dt_check is not None:
        d, span = dt_check
        run.metrics["time_step_change_rel"] = d / span
        run.checks.append({"code": "run.time_step", "status": "pass" if d / span <= 1e-3 else "fail",
                           "detail": f"{n_t} vs {2 * n_t} steps: max change {d:.3g} {unit} at t_end, "
                                     f"{d / span:.2e} of the field's range from {off:g} (limit 1e-3; raise n_t)"})
    # balance of the drafted law
    terms = [abs(h["stored"]), abs(h["source"]), abs(h["inflow"])]
    imb = abs(h["stored"] - h["source"] - h["inflow"]) / max(max(terms), 1e-300)
    dirichlet = any(fc.alpha == 0 for fc in form.faces.values())
    tol = 1e-2 if dirichlet else BALANCE_TOLERANCE
    run.balance = {"stored": h["stored"], "source": h["source"], "inflow": h["inflow"], "imbalance_rel": imb,
                   "tolerance": tol, "what": f"int a*{form.field} dV change = int S dV dt + boundary inflow dt "
                                             f"(a = {form.a:g})"}
    run.checks.append({"code": "run.balance", "status": "pass" if imb <= tol else "fail",
                       "detail": f"stored {h['stored']:.6g}, source {h['source']:.6g}, boundary inflow "
                                 f"{h['inflow']:.6g}: relative imbalance {imb:.2e} (limit {tol:g}"
                                 + (", Dirichlet faces: one-sided fluxes)" if dirichlet else ")")})
    # manufactured solution at the expected order
    if mms:
        levels = tuple(budget.get("mms_levels") or _default_levels(dims))
        m = mms_order(form, bounds, t_end=t_end or 1.0, offset=off, scale=sc, levels=levels)
        run.mms = m
        ok = m["observed_order"] >= EXPECTED_ORDER - ORDER_TOLERANCE
        run.checks.append({"code": "run.mms_order", "status": "pass" if ok else "fail",
                           "detail": f"observed order {m['observed_order']:.3f} (expected {EXPECTED_ORDER:g}, at "
                                     f"least {EXPECTED_ORDER - ORDER_TOLERANCE:g}) on grids "
                                     + "/".join(str(r['n']) for r in m["levels"]) + "; max error "
                                     + ", ".join(f"{r['max_error']:.2e}" for r in m["levels"])})
        run.metrics["mms.observed_order"] = m["observed_order"]
    run.checks += _draft_checks(draft)
    run.assumptions = _assumption_rows(draft, form, bounds, T, grid)
    run.checks.append({"code": "run.assumptions", "status": "pass",
                       "detail": "; ".join(f"{a['id']}: {'applied' if a['applied'] else 'not applied'}"
                                           + (f" ({a['evaluated']})" if a.get("evaluated") else "")
                                           for a in run.assumptions) or "the draft applies no assumption"})
    run.metrics["solve_seconds"] = solve_s
    run.next.append("method='pinn' trains on the same draft's loss and is compared with this solution")
    return run


def _run_pinn(draft, form: _Form | None, bounds, budget) -> DraftRun:
    from pinneapple_analysis.validation.pinn import train_pinn, validate_pinn
    from pinneapple_physics.symbolic_pde.pinn import pinn_problem
    s = draft.system()
    b = dict(bounds)
    t_end = 0.0
    if "t" in draft.coords:
        if form is not None:
            t_end, _ = _t_end(form, bounds, budget)
        else:
            t_end = float(budget.get("t_end") or 1.0)
        b["t"] = (0.0, t_end)
    import torch
    torch.manual_seed(int(budget.get("seed", 0)))
    scales = draft.scales()
    problem, model = pinn_problem(s, b, scales=scales, width=int(budget.get("width", 32)),
                                  depth=int(budget.get("depth", 4)))
    ts = time.perf_counter()
    net, info = train_pinn(problem, model, adam_steps=int(budget.get("adam_steps", 2000)),
                           lbfgs_steps=int(budget.get("lbfgs_steps", 200)),
                           n_collocation=int(budget.get("n_collocation", 2000)), seed=int(budget.get("seed", 0)))
    train_s = time.perf_counter() - ts
    rep = validate_pinn(net, problem)
    spatial = [c for c in draft.coords if c != "t"]
    n = int(budget.get("n", {1: 81, 2: 41, 3: 17}[len(spatial)]))
    axes = {x: np.linspace(*bounds[x], n) for x in spatial}
    mesh = np.meshgrid(*axes.values(), indexing="ij")
    cols = [m.ravel() for m in mesh] + ([np.full(mesh[0].size, t_end)] if "t" in draft.coords else [])
    X = torch.tensor(np.stack(cols, 1), dtype=torch.float32)
    with torch.no_grad():
        U = net(X).numpy()
    run = DraftRun("done", "pinn", draft.title, coords=list(draft.coords), field_name=", ".join(draft.fields),
                   unit=", ".join(draft.fields.values()), bounds=bounds, axes=axes,
                   fields={f: U[:, j].reshape(mesh[0].shape) for j, f in enumerate(draft.fields)})
    first = U[:, 0]
    run.metrics.update({"t_end": t_end, "max": float(first.max()), "min": float(first.min()),
                        "mean": float(first.mean()), "train_seconds": train_s, "solve_seconds": train_s,
                        "final_loss": float(info["loss"]) if info.get("loss") is not None else None})
    run.checks += [{"code": c["id"], "status": c["status"], "detail": c["summary"]} for c in rep.checks]
    run.mms = {"observed_order": float("nan"), "expected_order": None,
               "note": "a PINN has no grid order: the compiled residual it trains on is checked against SymPy on "
                       "a manufactured solution (model.mms)"}
    run.checks.append({"code": "run.mms_order", "status": "inconclusive", "detail": run.mms["note"]})
    if form is not None:                         # the classical solution as the reference
        ref = _run_fd(draft, form, bounds, {**budget, "n": n}, mms=False)
        T_ref = ref.fields[form.field].ravel()
        off, sc = scales.get(form.field, (0.0, 1.0))
        span = max(float(T_ref.max() - T_ref.min()), sc)
        err = float(np.max(np.abs(first - T_ref))) / span
        run.metrics["reference.fd_max_error_rel"] = err
        run.checks.append({"code": "run.reference_fd", "status": "pass" if err < 0.05 else "fail",
                           "detail": f"max |PINN - FD| / range = {err:.3g} at t = {t_end:g} s (limit 0.05)"})
        run.balance = ref.balance | {"from": "the FD reference solution"}
    run.checks += _draft_checks(draft)
    run.assumptions = _assumption_rows(draft, form, bounds)
    run.checks.append({"code": "run.assumptions", "status": "pass",
                       "detail": "; ".join(f"{a['id']}: {'applied' if a['applied'] else 'not applied'}"
                                           for a in run.assumptions) or "the draft applies no assumption"})
    run.next.append("draft.loss_code() is the training script of this run, editable")
    return run


__all__ = ["run_draft", "DraftRun", "Refused", "scalar_form", "mms_order", "METHODS", "EXPECTED_ORDER"]
