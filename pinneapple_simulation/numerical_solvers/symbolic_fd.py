"""Finite differences for any parsed equation system on a box: steady (Newton) or method of lines (BDF).

The system comes from :func:`pinneapple_physics.symbolic_pde.text.parse_system`, so the same text a user edits is
what gets solved, in 1-D, 2-D or 3-D:

    from pinneapple_physics.symbolic_pde.text import parse_system
    from pinneapple_simulation.numerical_solvers.symbolic_fd import solve_fd

    sys_ = parse_system(["u_t = alpha*lap(u)"],
                        [{"kind": "initial", "equation": "u = sin(pi*x)"},
                         {"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}],
                        coords=["x", "t"], fields=["u"], coefficients={"alpha": 0.1})
    res = solve_fd(sys_, {"x": (0, 1), "t": (0, 0.5)}, n=81)
    res.fields["u"].shape          # (n_save, 81)

**Scheme.** Second-order central differences inside, second-order one-sided differences at the faces, ``np.roll``
along periodic axes. Time-dependent systems are solved for the highest time derivative of each field (second order
in time adds ``f_t`` as a state) and integrated with SciPy's BDF with the stencil's Jacobian sparsity. Steady systems
are solved by Newton iterations on a coloured finite-difference Jacobian (one step for a linear problem).

**Conditions.** A condition on a face (``x_min``, ``y_max``, ``boundary``) is solved for the face value of its field
after the normal derivative is replaced by its one-sided difference, so Dirichlet ``u = g``, Neumann ``u_x = g`` and
Robin ``a*u + b*u_x = g`` all work (also nonlinear in tangential terms). In time, face nodes are relaxed to that value
at a rate ``relax / (t1 - t0)`` (stiff, handled by BDF). Faces without a condition keep the PDE with one-sided
differences, reported in ``info["unconstrained_faces"]``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef

from pinneapple_physics.symbolic_pde.text import compile_expr, derivative_key

__all__ = ["FDResult", "solve_fd", "FDError"]

class FDError(ValueError):
    """The system cannot be solved by this finite-difference scheme (and why)."""


@dataclass
class FDResult:
    axes: dict[str, np.ndarray]                      # spatial coordinate -> nodes
    fields: dict[str, np.ndarray]                    # field -> (n_t, *grid) or grid
    t: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def final(self, name: str) -> np.ndarray:
        a = self.fields[name]
        return a[-1] if self.t is not None else a

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        """All (space[, t]) points as ``(N, d)`` in the system's coordinate order and the values ``(N, n_fields)``."""
        names = list(self.axes)
        axes = [self.axes[c] for c in names]
        if self.t is not None:
            axes = axes + [self.t]
        mesh = np.meshgrid(*axes, indexing="ij")
        X = np.stack([m.ravel() for m in mesh], 1)
        vals = []
        for a in self.fields.values():
            v = np.moveaxis(a, 0, -1) if self.t is not None else a
            vals.append(v.ravel())
        return X, np.stack(vals, 1)


def _compile(expr: sp.Expr):
    return compile_expr(expr, "numpy")


def _dkey(d: sp.Derivative) -> str:
    return derivative_key(d)


# ----------------------------------------------------------------------------------------------- grid derivatives
class _Grid:
    def __init__(self, spatial, bounds, n, periodic):
        self.names = list(spatial)
        self.periodic = periodic
        self.axes, self.h = {}, {}
        for c in self.names:
            lo, hi = map(float, bounds[c])
            m = int(n[c])
            if m < 5:
                raise FDError(f"at least 5 nodes along {c}")
            if c in periodic:
                self.axes[c] = np.linspace(lo, hi, m, endpoint=False)
                self.h[c] = (hi - lo) / m
            else:
                self.axes[c] = np.linspace(lo, hi, m)
                self.h[c] = (hi - lo) / (m - 1)
        self.shape = tuple(len(self.axes[c]) for c in self.names)
        mesh = np.meshgrid(*[self.axes[c] for c in self.names], indexing="ij")
        self.coords = dict(zip(self.names, mesh, strict=True))

    def _ax(self, c):
        return self.names.index(c) - len(self.names)          # from the end: works with a leading ensemble axis

    def d1(self, a, c):
        ax, h = self._ax(c), self.h[c]
        if c in self.periodic:
            return (np.roll(a, -1, ax) - np.roll(a, 1, ax)) / (2 * h)
        return np.gradient(a, h, axis=ax, edge_order=2)

    def d2(self, a, c):
        ax, h = self._ax(c), self.h[c]
        if c in self.periodic:
            return (np.roll(a, -1, ax) - 2 * a + np.roll(a, 1, ax)) / h ** 2
        a = np.moveaxis(a, ax, 0)
        out = np.empty_like(a)
        out[1:-1] = (a[2:] - 2 * a[1:-1] + a[:-2]) / h ** 2
        out[0] = (2 * a[0] - 5 * a[1] + 4 * a[2] - a[3]) / h ** 2
        out[-1] = (2 * a[-1] - 5 * a[-2] + 4 * a[-3] - a[-4]) / h ** 2
        return np.moveaxis(out, 0, ax)

    def derivative(self, a, wrt: str):
        for c in sorted(set(wrt)):
            k = wrt.count(c)
            while k >= 2:
                a, k = self.d2(a, c), k - 2
            if k:
                a = self.d1(a, c)
        return a

    def face(self, c, side, depth=0):
        idx = [slice(None)] * len(self.names)
        idx[self.names.index(c)] = depth if side == "min" else -1 - depth
        return (Ellipsis, *idx)


# ----------------------------------------------------------------------------------------------- the solver
def _faces_of(on: str, spatial, periodic):
    if on == "boundary":
        return [(c, s) for c in spatial if c not in periodic for s in ("min", "max")]
    if on in spatial:
        return [(on, "min"), (on, "max")]
    c, _, s = on.partition("_")
    if c in spatial and s in ("min", "max"):
        return [(c, s)]
    raise FDError(f"unknown boundary {on!r}")


class FDOperator:
    """A parsed system compiled on a grid: the right-hand sides, the face conditions and the initial state. Arrays
    may carry leading axes (an ensemble); coefficients named in ``keep`` stay symbolic and are read from the
    environment (``extra``), so an ensemble of coefficient values can be stepped at once."""


def compile_system(system, bounds: dict[str, tuple[float, float]], *, n: int | dict[str, int] = 64,
                   coefficients: dict[str, float] | None = None, keep: tuple[str, ...] = ()) -> FDOperator:
    """Compile ``system`` for finite differences (see :func:`solve_fd`, which uses it)."""
    issues = [i for i in system.issues() if "without a value" in i and not coefficients and not keep]
    if issues:
        raise FDError("; ".join(issues))
    ctx = system.ctx
    spatial, tname = ctx.spatial, ctx.time
    if not spatial:
        raise FDError("finite differences need at least one spatial coordinate")
    missing = [c for c in ctx.coords if c not in bounds]
    if missing:
        raise FDError(f"no bounds for {missing}")
    nn = {c: (n[c] if isinstance(n, dict) else n) for c in spatial}
    periodic = set()
    for c in system.conditions:
        if c.kind == "periodic":
            periodic |= {f for f, _ in _faces_of(c.on, spatial, set())}
    g = _Grid(spatial, bounds, nn, periodic)
    fields = list(ctx.fields)
    vals = {k: v for k, v in {**system.coefficients, **(coefficients or {})}.items() if k not in keep}
    rep = {sp.Symbol(k, real=True): sp.Float(v) for k, v in vals.items()}
    rep.update({sp.Symbol(k, real=True): sp.Symbol(k) for k in keep})
    R = [e.residual.xreplace(rep).doit() for e in system.pdes]
    left = sorted({str(s) for r in R for s in r.free_symbols} - set(ctx.coords) - set(keep))
    if left:
        raise FDError(f"coefficient(s) without a value: {left}")
    unsteady = system.time_dependent
    fn = ctx.functions
    tsym = ctx.symbols.get(tname)

    # time order per field and the explicit form f^(k)_t = G(...)
    order = {}
    for f in fields:
        order[f] = max((int(k) for r in R for dv in r.atoms(sp.Derivative) if dv.expr == fn[f]
                        for s, k in dv.variable_count if s == tsym), default=0) if unsteady else 0
    aux = {f: sp.Symbol(f"{f}__t") for f in fields if order[f] == 2}
    if unsteady:
        if any(order[f] == 0 for f in fields):
            raise FDError("every field needs a time derivative in a time-dependent system (DAEs are not supported)")
        if any(o > 2 for o in order.values()):
            raise FDError("time derivatives above second order are not supported")
        tops = [sp.Derivative(fn[f], *([tsym] * order[f])) for f in fields]
        heads = {d: sp.Symbol(f"__head{i}") for i, d in enumerate(tops)}
        Rs = [r.xreplace(heads) for r in R]
        sol = sp.solve(Rs, list(heads.values()), dict=True)
        if len(sol) != 1:
            raise FDError("cannot solve the system for its time derivatives (it must be linear in them)")
        G = {f: sol[0][heads[tops[i]]] for i, f in enumerate(fields)}
        G = {f: e.xreplace({sp.Derivative(fn[h], tsym): aux[h] for h in aux}) for f, e in G.items()}
        bad = [f for f, e in G.items() if any(tsym in dv.free_symbols and any(s == tsym for s, _ in
                                                                                dv.variable_count)
                                           for dv in e.atoms(sp.Derivative))]
        if bad:
            raise FDError(f"mixed space-time derivatives are not supported ({bad})")
        exprs = {f: G[f] for f in fields}
    else:
        if len(R) != len(fields):
            raise FDError("a steady system needs one PDE per field")
        exprs = {f: R[i] for i, f in enumerate(fields)}

    def plain(e):
        return e.xreplace({fn[f]: sp.Symbol(f) for f in fields}).xreplace(
            {s: sp.Symbol(s.name) for s in e.free_symbols if s.name in ctx.coords})

    dkeys: dict[str, tuple[str, str]] = {}

    def collect(e):
        for dv in e.atoms(sp.Derivative):
            if isinstance(dv.expr, AppliedUndef):
                wrt = "".join(str(s) * int(k) for s, k in dv.variable_count)
                if tname in wrt and unsteady:
                    raise FDError(f"time derivative {dv} where only spatial ones are allowed")
                dkeys[_dkey(dv)] = (dv.expr.func.__name__, wrt)

    for e in exprs.values():
        collect(e)
    compiled = {f: _compile(_plainder(exprs[f], fn, fields, ctx)) for f in fields}

    # ---- conditions on faces: solve for the face value
    face_rules: list[tuple[str, str, str, Any]] = []        # (field, coord, side, compiled target)
    covered: set[tuple[str, str]] = set()
    conds = sorted([c for c in system.conditions if c.kind not in ("initial", "periodic")],
                   key=lambda c: 0 if c.kind != "dirichlet" else 1)      # Dirichlet last: it wins at corners
    u0s, u1s, u2s = sp.Symbol("__u0"), sp.Symbol("__u1"), sp.Symbol("__u2")
    for c in conds:
        C = c.residual.xreplace(rep).doit()
        used = [f for f in fields if C.has(fn[f])]
        if not used:
            raise FDError(f"condition {c.text!r} involves no field")
        f = used[0]
        for (ax, side) in _faces_of(c.on, spatial, periodic):
            h = g.h[ax]
            xs = ctx.symbols[ax]
            dn = (-3 * u0s + 4 * u1s - u2s) / (2 * h) if side == "min" else (3 * u0s - 4 * u1s + u2s) / (2 * h)
            e = C.xreplace({sp.Derivative(fn[f], xs): dn})
            if any(isinstance(dv.expr, AppliedUndef) and dv.expr == fn[f] and any(s == xs for s, _ in
                                                                                 dv.variable_count)
                   for dv in e.atoms(sp.Derivative)):
                raise FDError(f"condition {c.text!r}: only first normal derivatives are supported on faces")
            e = e.xreplace({fn[f]: u0s})
            s = sp.solve(e, u0s, dict=True)
            if len(s) != 1:
                raise FDError(f"condition {c.text!r} does not determine {f} on {ax}_{side} uniquely")
            target = s[0][u0s]
            collect(target)
            face_rules.append((f, ax, side, _compile(_plainder(target, fn, fields, ctx))))
            covered.add((ax, side))
    unconstrained = [f"{a}_{s}" for a in spatial if a not in periodic for s in ("min", "max")
                     if (a, s) not in covered]

    def environment(U: dict[str, np.ndarray], t: float | None, extra: dict[str, Any] | None = None):
        env: dict[str, Any] = dict(g.coords)
        env.update(extra or {})
        if t is not None:
            env[tname] = t
        env.update(U)
        for k, (f, wrt) in dkeys.items():
            env[k] = g.derivative(U[f], wrt)
        return env

    def face_targets(U, env):
        out = []
        for f, ax, side, tgt in face_rules:
            sl = g.face(ax, side)
            fenv = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim >= len(g.shape) else v)
                    for k, v in env.items()}
            fenv["__u1"] = U[f][g.face(ax, side, 1)]
            fenv["__u2"] = U[f][g.face(ax, side, 2)]
            out.append((f, sl, np.broadcast_to(tgt(fenv), U[f][sl].shape)))
        return out

    op = FDOperator()
    op.system, op.g, op.fields, op.order, op.aux, op.compiled = system, g, fields, order if unsteady else {}, aux \
        if unsteady else {}, compiled
    op.unsteady, op.tname, op.rep, op.fn, op.ctx = unsteady, tname, rep, fn, ctx
    op.environment, op.face_targets, op.unconstrained, op.periodic = environment, face_targets, unconstrained, \
        periodic
    op.keep = tuple(keep)
    return op


def initial_state(op: FDOperator, t0: float) -> dict[tuple[str, bool], np.ndarray]:
    """Initial values (and ``f_t`` for second order) of every field on the grid, faces set by their conditions."""
    g, fn, fields, ctx = op.g, op.fn, op.fields, op.ctx
    names = [(f, False) for f in fields] + [(f, True) for f in fields if op.order.get(f) == 2]
    init = {nm: np.zeros(g.shape) for nm in names}
    env0 = dict(g.coords)
    env0[op.tname] = t0
    for c in [c for c in op.system.conditions if c.kind == "initial"]:
        lhs = c.lhs
        if lhs in fn.values():
            key = (lhs.func.__name__, False)
        elif isinstance(lhs, sp.Derivative) and lhs.expr in fn.values() and [str(s) for s, _ in
                                                                             lhs.variable_count] == [op.tname]:
            key = (lhs.expr.func.__name__, True)
        else:
            raise FDError(f"initial condition {c.text!r}: write it as 'u = ...' or 'u_t = ...'")
        val = _compile(_plainder(c.rhs.xreplace(op.rep), fn, fields, ctx))(env0)
        init[key] = np.broadcast_to(np.asarray(val, float), g.shape).copy()
    return init


def solve_fd(system, bounds: dict[str, tuple[float, float]], *, n: int | dict[str, int] = 64,
             n_save: int = 41, coefficients: dict[str, float] | None = None, relax: float = 2e4,
             rtol: float = 1e-6, atol: float = 1e-8, max_unknowns: int = 250_000, newton_tol: float = 1e-9,
             max_newton: int = 25) -> FDResult:
    """Solve ``system`` on the box ``bounds`` (one interval per coordinate, ``t`` included when time-dependent)
    with ``n`` nodes per spatial axis. Returns :class:`FDResult`; FDError when the system is outside the scheme."""
    t0 = time.perf_counter()
    op = compile_system(system, bounds, n=n, coefficients=coefficients)
    g, fields, order, aux, compiled = op.g, op.fields, op.order, op.aux, op.compiled
    unsteady, tname, environment, face_targets = op.unsteady, op.tname, op.environment, op.face_targets
    spatial, periodic, unconstrained = op.ctx.spatial, op.periodic, op.unconstrained
    n_nodes = int(np.prod(g.shape))
    nst = sum(order[f] for f in fields) if unsteady else len(fields)
    if n_nodes * max(nst, 1) > max_unknowns:
        raise FDError(f"{n_nodes * nst} unknowns > max_unknowns={max_unknowns}: use fewer nodes")
    info: dict[str, Any] = {"scheme": "central FD, one-sided at faces", "grid": dict(zip(spatial, g.shape,
                                                                                        strict=True)),
                            "spacing": g.h, "periodic": sorted(periodic), "unconstrained_faces": unconstrained,
                            "n_unknowns": n_nodes * max(nst, 1)}

    if not unsteady:
        U, it, res = _newton(compiled, fields, g, environment, face_targets, newton_tol, max_newton)
        info.update(method="newton", iterations=it, residual_inf=res, seconds=time.perf_counter() - t0)
        return FDResult(axes=dict(g.axes), fields=U, info=info)

    # ---- method of lines
    from scipy.integrate import solve_ivp
    T0, T1 = map(float, bounds[tname])
    names = [(f, False) for f in fields] + [(f, True) for f in fields if order[f] == 2]
    init = initial_state(op, T0)
    K = relax / (T1 - T0)
    sizes = [n_nodes] * len(names)

    def unpack(y):
        return {nm: y[i * n_nodes:(i + 1) * n_nodes].reshape(g.shape) for i, nm in enumerate(names)}

    def rhs(t, y):
        S = unpack(y)
        U = {f: S[(f, False)] for f in fields}
        env = environment(U, t)
        for f in aux:
            env[aux[f].name] = S[(f, True)]
        out = {}
        for f in fields:
            gval = np.broadcast_to(compiled[f](env), g.shape)
            if order[f] == 2:
                out[(f, False)] = S[(f, True)].copy()
                out[(f, True)] = np.array(gval, float)
            else:
                out[(f, False)] = np.array(gval, float)
        for f, sl, tgt in face_targets(U, env):
            out[(f, False)][sl] = K * (tgt - U[f][sl])
            if order[f] == 2:
                out[(f, True)][sl] = -K * S[(f, True)][sl]
        return np.concatenate([out[nm].ravel() for nm in names])

    y0 = np.concatenate([init[nm].ravel() for nm in names])
    # boundary nodes start on their condition
    S0 = unpack(y0.copy())
    U0 = {f: S0[(f, False)] for f in fields}
    env = environment(U0, T0)
    for f in aux:
        env[aux[f].name] = S0[(f, True)]
    for f, sl, tgt in face_targets(U0, env):
        U0[f][sl] = tgt
    y0 = np.concatenate([(U0[f] if not a else S0[(f, a)]).ravel() for f, a in names])
    sparsity = _sparsity(g.shape, len(names))
    t_eval = np.linspace(T0, T1, n_save)
    sol = solve_ivp(rhs, (T0, T1), y0, method="BDF", t_eval=t_eval, rtol=rtol, atol=atol, jac_sparsity=sparsity)
    if not sol.success:
        raise FDError(f"time integration failed: {sol.message}")
    out = {}
    for i, (f, a) in enumerate(names):
        if not a:
            out[f] = sol.y[i * n_nodes:(i + 1) * n_nodes].T.reshape((len(sol.t), *g.shape))
    info.update(method="method of lines + BDF", nfev=int(sol.nfev), njev=int(sol.njev), relax_rate=K,
                seconds=time.perf_counter() - t0)
    del sizes
    return FDResult(axes=dict(g.axes), fields=out, t=sol.t, info=info)


def _plainder(expr, fn, fields, ctx):
    """Fields as plain symbols, derivatives as their keys, coordinates as plain symbols (for :func:`_compile`)."""
    rep = {}
    for dv in expr.atoms(sp.Derivative):
        if isinstance(dv.expr, AppliedUndef):
            rep[dv] = sp.Symbol(_dkey(dv))
    e = expr.xreplace(rep)
    e = e.xreplace({fn[f]: sp.Symbol(f) for f in fields})
    return e.xreplace({s: sp.Symbol(s.name) for s in e.free_symbols if s.name in ctx.coords})


def _offsets(d: int):
    offs = set(product((-1, 0, 1), repeat=d))
    for a in range(d):
        for k in (-3, -2, 2, 3):
            o = [0] * d
            o[a] = k
            offs.add(tuple(o))
    return sorted(offs)


def _sparsity(shape, n_blocks):
    from scipy import sparse
    n = int(np.prod(shape))
    idx = np.arange(n).reshape(shape)
    rows, cols = [], []
    for off in _offsets(len(shape)):
        src = [slice(max(0, -o), s - max(0, o)) for o, s in zip(off, shape, strict=True)]
        dst = [slice(max(0, o), s - max(0, -o)) for o, s in zip(off, shape, strict=True)]
        rows.append(idx[tuple(src)].ravel())
        cols.append(idx[tuple(dst)].ravel())
    r, c = np.concatenate(rows), np.concatenate(cols)
    # periodic wrap-around: add the rolled neighbours too (cheap superset)
    blocks = sparse.coo_matrix((np.ones_like(r, dtype=np.int8), (r, c)), shape=(n, n)).tocsr()
    for a in range(len(shape)):
        for o in (-1, 1):
            rolled = np.roll(idx, o, axis=a).ravel()
            blocks = blocks + sparse.coo_matrix((np.ones(n, dtype=np.int8), (idx.ravel(), rolled)),
                                                shape=(n, n)).tocsr()
    blocks.data[:] = 1
    return sparse.kron(np.ones((n_blocks, n_blocks), dtype=np.int8), blocks, format="csr")


def _newton(compiled, fields, g, environment, face_targets, tol, max_it):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    n = int(np.prod(g.shape))
    nf = len(fields)

    def F(y):
        U = {f: y[i * n:(i + 1) * n].reshape(g.shape) for i, f in enumerate(fields)}
        env = environment(U, None)
        out = {f: np.array(np.broadcast_to(compiled[f](env), g.shape), float) for f in fields}
        for f, sl, tgt in face_targets(U, env):
            out[f][sl] = U[f][sl] - tgt
        return np.concatenate([out[f].ravel() for f in fields])

    # colouring: nodes 7 apart along each axis never share a row of the stencil (offsets up to 3)
    shape = g.shape
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    colour = np.zeros(shape, dtype=np.int64)
    for a, gr in enumerate(grids):
        colour = colour + (gr % 7) * 7 ** a
    colour = colour.ravel()
    pattern = _sparsity(shape, nf).tocsc()
    y = np.zeros(n * nf)
    res = float("inf")
    it = 0
    for it in range(1, max_it + 1):
        F0 = F(y)
        res = float(np.max(np.abs(F0)))
        scale = max(1.0, float(np.max(np.abs(y))))
        if res < tol * scale and it > 1:
            break
        eps = 1e-7 * scale
        data, ri, ci = [], [], []
        for b in range(nf):
            for col in np.unique(colour):
                members = np.nonzero(colour == col)[0] + b * n
                yp = y.copy()
                yp[members] += eps
                dF = (F(yp) - F0) / eps
                for j in members:
                    rows = pattern.indices[pattern.indptr[j]:pattern.indptr[j + 1]]
                    data.append(dF[rows])
                    ri.append(rows)
                    ci.append(np.full(len(rows), j))
        J = sparse.csr_matrix((np.concatenate(data), (np.concatenate(ri), np.concatenate(ci))), shape=(n * nf,
                                                                                                     n * nf))
        step = spsolve(J.tocsc(), -F0)
        if not np.all(np.isfinite(step)):
            raise FDError("Newton step is not finite (singular Jacobian: missing boundary conditions?)")
        y = y + step
        if float(np.max(np.abs(step))) < tol * max(1.0, float(np.max(np.abs(y)))):
            res = float(np.max(np.abs(F(y))))
            break
    if not math.isfinite(res):
        raise FDError("Newton iterations diverged")
    return {f: y[i * n:(i + 1) * n].reshape(g.shape) for i, f in enumerate(fields)}, it, res
