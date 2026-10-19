"""A parsed equation system (:mod:`.text`) as a :class:`~pinneapple_analysis.validation.pinn.PINNProblem`.

The residual is evaluated by autograd from the same SymPy expressions the finite-difference solver uses, so a PINN
and a classical solution of one edited system are comparable point for point. Dirichlet and initial conditions
become the problem's ``boundary`` / ``initial`` samplers (checked by ``validate_pinn``); conditions with derivatives
(Neumann, Robin, ``u_t = ...`` at ``t = 0``) become ``constraints``. Coefficients listed in ``unknowns`` are trainable
scalars (an inverse PINN, with ``train_pinn(..., data=(X, Y))``).

    P, model = pinn_problem(system, {"x": (0, 1), "t": (0, 1)})
    net, info = train_pinn(P, model, adam_steps=2000, lbfgs_steps=200)
    report = validate_pinn(net, P)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef

from .text import EquationSystem, compile_expr, derivative_key, derivative_keys

__all__ = ["pinn_problem", "PINNWithParameters"]


def _torch():
    import torch
    return torch


class PINNWithParameters:  # placeholder replaced by the torch module class on first use
    pass


def _module_class():
    torch = _torch()

    class _PINNWithParameters(torch.nn.Module):
        """A network and the trainable coefficients of an inverse problem (``.coefficients[name]``)."""

        def __init__(self, net, unknowns: dict[str, float]):
            super().__init__()
            self.net = net
            self.coefficients = torch.nn.ParameterDict(
                {k: torch.nn.Parameter(torch.tensor(float(v))) for k, v in unknowns.items()})

        def forward(self, X):
            return self.net(X)

        def estimates(self) -> dict[str, float]:
            return {k: float(v.detach()) for k, v in self.coefficients.items()}

    return _PINNWithParameters


def _derivatives(u, X, keys: dict[str, tuple[str, str]], fields: list[str], coords: list[str]):
    torch = _torch()
    cache: dict[tuple[str, str], Any] = {}
    out = {}
    for key, (f, wrt) in keys.items():
        col = u[:, fields.index(f):fields.index(f) + 1]
        done = ""
        for ch in wrt:
            nxt = done + ch
            if (f, nxt) not in cache:
                g = torch.autograd.grad(col.sum(), X, create_graph=True)[0]
                cache[(f, nxt)] = g[:, coords.index(ch):coords.index(ch) + 1]
            col, done = cache[(f, nxt)], nxt
        out[key] = col
    return out


def _plain(expr, system: EquationSystem):
    fn = system.ctx.functions
    e = expr.xreplace({dv: sp.Symbol(derivative_key(dv)) for dv in expr.atoms(sp.Derivative)
                       if isinstance(dv.expr, AppliedUndef)})
    e = e.xreplace({fn[f]: sp.Symbol(f) for f in system.fields})
    return e.xreplace({s: sp.Symbol(s.name) for s in e.free_symbols if s.name in system.coords})


def _sampler_face(bounds, coords, axis, side):
    lo = np.array([bounds[c][0] for c in coords], float)
    hi = np.array([bounds[c][1] for c in coords], float)
    k = coords.index(axis)

    def sample(n, rng):
        X = lo + (hi - lo) * rng.random((n, len(coords)))
        X[:, k] = lo[k] if side == "min" else hi[k]
        return X
    return sample


def _faces(on: str, spatial: list[str]):
    if on == "boundary":
        return [(c, s) for c in spatial for s in ("min", "max")]
    if on in spatial:
        return [(on, "min"), (on, "max")]
    c, _, s = on.partition("_")
    return [(c, s)]


def _union(samplers):
    def sample(n, rng):
        k = max(1, n // len(samplers))
        parts = [s(k, rng) for s in samplers]
        return tuple(np.concatenate(p) for p in zip(*parts, strict=True)) if isinstance(parts[0], tuple) \
            else np.concatenate(parts)
    return sample


def pinn_problem(system: EquationSystem, bounds: dict[str, tuple[float, float]], *,
                 coefficients: dict[str, float] | None = None, unknowns: dict[str, float] | None = None,
                 width: int = 32, depth: int = 4, exact=None, name: str = "flow", grid: tuple[int, ...] | None = None):
    """``(PINNProblem, model)`` for ``system`` on the box ``bounds``. ``unknowns``: coefficient -> initial guess,
    trained with the network (read back with ``model.estimates()``). ``exact``: optional ``X -> (N, n_fields)``."""
    torch = _torch()
    from pinneapple_analysis.validation.pinn import PINNProblem, mlp

    coords, fields = list(system.coords), list(system.fields)
    spatial, tname = system.ctx.spatial, system.ctx.time
    unknowns = dict(unknowns or {})
    vals = {**system.coefficients, **(coefficients or {})}
    for k in unknowns:
        vals.pop(k, None)
    rep = {sp.Symbol(k, real=True): sp.Float(v) for k, v in vals.items()}
    missing = sorted({str(s) for e in system.pdes + system.conditions for s in e.residual.xreplace(rep).free_symbols}
                     - set(coords) - set(unknowns))
    if missing:
        raise ValueError(f"coefficient(s) without a value: {missing}")
    model = _module_class()(mlp(len(coords), len(fields), width, depth), unknowns)

    residuals = [e.residual.xreplace(rep).doit() for e in system.pdes]
    keys: dict[str, tuple[str, str]] = {}
    for r in residuals:
        keys.update(derivative_keys(r))
    comp = [compile_expr(_plain(r, system), "torch") for r in residuals]

    def env_of(u, X, extra_keys=None):
        env: dict[str, Any] = {c: X[:, i:i + 1] for i, c in enumerate(coords)}
        env.update({f: u[:, j:j + 1] for j, f in enumerate(fields)})
        env.update(_derivatives(u, X, extra_keys if extra_keys is not None else keys, fields, coords))
        env.update(dict(model.coefficients.items()))
        return env

    def residual(u, X):
        env = env_of(u, X)
        out = [c(env) for c in comp]
        out = [o if torch.is_tensor(o) else torch.full((len(X), 1), float(o)) for o in out]
        return torch.cat([o.reshape(len(X), 1) for o in out], 1)

    boundary, initial, constraints = [], [], []
    fn = system.ctx.functions
    for c in system.conditions:
        if c.kind == "periodic":
            continue
        lhs, rhs = c.lhs.xreplace(rep), c.rhs.xreplace(rep)
        if c.kind == "initial":
            if tname not in coords:
                continue
            smp = _sampler_face(bounds, coords, tname, "min")
        else:
            smp = _union([_sampler_face(bounds, coords, a, s) for a, s in _faces(c.on, spatial)])
        if lhs in fn.values() and not rhs.has(*fn.values()) and not rhs.atoms(sp.Derivative):
            j = fields.index(lhs.func.__name__)
            target = compile_expr(_plain(rhs, system), "numpy")

            def pair(n, rng, smp=smp, j=j, target=target):
                X = smp(n, rng)
                T = np.full((len(X), len(fields)), np.nan)
                env = {cc: X[:, i] for i, cc in enumerate(coords)}
                T[:, j] = np.broadcast_to(target(env), len(X))
                return X, T
            (initial if c.kind == "initial" else boundary).append(pair)
        else:
            res = (lhs - rhs).doit()
            ck = derivative_keys(res)
            cf = compile_expr(_plain(res, system), "torch")

            def cres(u, X, cf=cf, ck=ck):
                v = cf(env_of(u, X, ck))
                return v if torch.is_tensor(v) else torch.full((len(X), 1), float(v))
            constraints.append((c.text, smp, cres))
    P = PINNProblem(
        name=name, bounds=[tuple(map(float, bounds[c])) for c in coords], residual=residual,
        boundary=_union(boundary) if boundary else None, initial=_union(initial) if initial else None,
        exact=exact, time_axis=coords.index(tname) if tname in coords else None,
        grid=tuple(grid) if grid else tuple(64 if len(coords) <= 2 else 16 for _ in coords),
        outputs=tuple(fields), constraints=tuple(constraints))
    return P, model


def has_field(expr: sp.Expr) -> bool:
    return bool(expr.atoms(AppliedUndef))
