"""Legendre transforms between potentials, with a convexity check.

``legendre(Phi, swap={"x": "y"})`` replaces the natural variable ``x`` by its conjugate ``y = s dPhi/dx`` and
returns ``Psi(y, ...) = Phi - x dPhi/dx`` evaluated at the ``x`` that solves ``s dPhi/dx(x, ...) = y``:

* Helmholtz to Gibbs: ``legendre(A, swap={"V": "p"}, conjugate_sign=-1)`` gives ``G(T, p) = A + p V``;
* Lagrangian to Hamiltonian: ``legendre(L, swap={"qd": "p"}, negate=True)`` gives ``H(q, p) = p qd - L``.

The transform is one-to-one only where ``Phi`` is strictly convex (or strictly concave) in the swapped variables.
:func:`legendre` checks that on a grid (``bounds=``) and reports where it fails: for a van der Waals fluid below its
critical temperature the Helmholtz energy is not convex in ``V`` between the two spinodals, the region where the
homogeneous fluid is unstable; the report gives that interval at each temperature of the grid.

Numerically, ``x`` is found by Newton's method (bracketed when bounds are given), without a graph; then two Newton
steps are repeated inside the autodiff graph from that root. Each step doubles the order to which ``x(y, ...)`` is
exact, and ``Psi`` is stationary in ``x``, so derivatives of the transformed potential are exact to order seven at a
converged root. When the potential has a SymPy expression and ``s dPhi/dx = y`` has one solution, the symbolic
transform is used (``method="auto"``).
"""
from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .core import DTYPE, Potential, Var

__all__ = ["ConvexityReport", "LegendreResult", "convexity", "legendre"]


@dataclass
class ConvexityReport:
    """Convexity of a potential in some of its variables, on a grid.

    ``eig_min`` / ``eig_max``: extreme eigenvalues of the Hessian block of ``variables`` at each grid point (grid
    shape, ``indexing="ij"`` in the order of ``axes``); ``stable``: where that block is definite with the expected
    sign (``expect="convex"``: positive definite)."""

    variables: tuple[str, ...]
    expect: str
    axes: dict[str, np.ndarray]
    eig_min: np.ndarray
    eig_max: np.ndarray
    stable: np.ndarray
    finite: np.ndarray = field(default=None)

    @property
    def ok(self) -> bool:
        """True when the block is definite (with the expected sign) at every finite grid point."""
        return bool(self.stable[self.finite].all())

    @property
    def fraction_stable(self) -> float:
        return float(self.stable[self.finite].mean()) if self.finite.any() else float("nan")

    def intervals(self, along: str) -> list[tuple[dict[str, float], list[tuple[float, float]]]]:
        """For each combination of the other grid axes, the intervals of ``along`` where the check fails (from the
        first to the last failing grid value of each run)."""
        names = list(self.axes)
        ia = names.index(along)
        others = [n for n in names if n != along]
        out = []
        for combo in itertools.product(*[range(len(self.axes[n])) for n in others]):
            idx = list(combo)
            idx.insert(ia, slice(None))
            bad = ~self.stable[tuple(idx)] & self.finite[tuple(idx)]
            ax = self.axes[along]
            runs, start = [], None
            for i, b in enumerate(bad):
                if b and start is None:
                    start = i
                if (not b or i == len(bad) - 1) and start is not None:
                    end = i if b else i - 1
                    runs.append((float(ax[start]), float(ax[end])))
                    start = None
            out.append(({n: float(self.axes[n][c]) for n, c in zip(others, combo, strict=False)}, runs))
        return out

    def __str__(self) -> str:
        s = (f"{self.expect} in {', '.join(self.variables)}: "
             f"{'yes' if self.ok else 'no'} ({100 * self.fraction_stable:.1f} % of {int(self.finite.sum())} grid "
             f"points)")
        return s


def convexity(potential: Potential, variables, bounds: Mapping[str, tuple[float, float]], n_grid: int = 41,
              expect: str = "convex", fixed: Mapping[str, float] | None = None) -> ConvexityReport:
    """Check that ``potential`` is convex (``expect="convex"``) or concave in ``variables`` on a grid.

    ``bounds`` gives ``(lo, hi)`` for every variable on the grid; ``fixed`` gives values for variables kept
    constant. Points where the potential is not finite (outside its domain) are excluded."""
    if expect not in ("convex", "concave"):
        raise ValueError("expect must be 'convex' or 'concave'")
    variables = tuple(potential.key(variables if not isinstance(variables, str) else (variables,)))
    fixed = dict(fixed or {})
    axes = {k: np.linspace(*bounds[k], n_grid) for k in potential.variables if k in bounds}
    missing = [k for k in potential.variables if k not in axes and k not in fixed]
    if missing:
        raise ValueError(f"give bounds or fixed values for {missing}")
    mesh = np.meshgrid(*axes.values(), indexing="ij")
    point = {k: m for k, m in zip(axes, mesh, strict=False)}
    point.update({k: np.full(mesh[0].shape, float(v)) for k, v in fixed.items()})
    z, bshape = potential._prepare(point)
    d, _ = potential.derivatives(z, 2)
    k = len(variables)
    H = torch.empty(z.shape[0], k, k, dtype=DTYPE)
    for i, j in itertools.product(range(k), range(k)):
        H[:, i, j] = d[potential.key((variables[i], variables[j]))].reshape(-1)
    finite = torch.isfinite(H).all(dim=(1, 2)) & torch.isfinite(d[()])
    Hs = torch.where(finite[:, None, None], H, torch.eye(k, dtype=DTYPE))
    eig = torch.linalg.eigvalsh(Hs)
    emin, emax = eig[:, 0].numpy().reshape(bshape), eig[:, -1].numpy().reshape(bshape)
    fin = finite.numpy().reshape(bshape)
    stable = (emin > 0) if expect == "convex" else (emax < 0)
    return ConvexityReport(variables, expect, axes, emin, emax, stable & fin, fin)


@dataclass
class LegendreResult:
    """A transformed potential and the convexity check of the original one."""

    potential: Potential
    convexity: ConvexityReport | None
    method: str
    swap: dict[str, str]
    note: str = ""
    _residual: Callable | None = None

    def residual(self, **point) -> np.ndarray:
        """``|s dPhi/dx - y|`` at the root found for ``point`` (numerical transform only; 0 for symbolic)."""
        if self._residual is None:
            return np.zeros(np.broadcast_shapes(*[np.shape(v) for v in point.values()]))
        return self._residual(point)

    def __str__(self) -> str:
        c = str(self.convexity) if self.convexity is not None else "convexity not checked (no bounds)"
        return (f"{self.potential.name}({', '.join(self.potential.variables)}) by Legendre transform "
                f"({self.method}); {c}" + (f"; {self.note}" if self.note else ""))


def _signs(swap, conjugate_sign) -> dict[str, float]:
    if isinstance(conjugate_sign, Mapping):
        return {x: float(conjugate_sign.get(x, 1.0)) for x in swap}
    return {x: float(conjugate_sign) for x in swap}


def _symbolic(potential, swap, signs, negate):
    """Symbolic transform when s dPhi/dx = y has exactly one solution; else (None, reason)."""
    import sympy as sp

    syms = dict(zip(potential.variables, potential.symbols, strict=False))
    xs = [syms[x] for x in swap]
    ys = [sp.Symbol(y, real=True) for y in swap.values()]
    eqs = [signs[x] * sp.diff(potential.expr, syms[x]) - y for x, y in zip(swap, ys, strict=False)]
    try:
        linear = all(sp.Poly(e, *xs).total_degree() <= 1 for e in eqs)
    except sp.PolynomialError:
        linear = False
    try:
        if linear:  # a quadratic potential in the swapped variables (a Lagrangian): one linear solve
            sols = [dict(zip(xs, s, strict=False)) for s in sp.linsolve(eqs, xs)]
        else:
            sols = sp.solve(eqs, xs, dict=True)
    except (NotImplementedError, ValueError) as exc:  # pragma: no cover - sympy limits
        return None, f"symbolic solve failed ({exc})"
    if len(sols) != 1:
        return None, f"{len(sols)} symbolic roots of the conjugate equation; numerical transform used"
    sol = sols[0]
    psi = potential.expr - sum(syms[x] * y / signs[x] for x, y in zip(swap, ys, strict=False))
    psi = psi.subs(sol)  # no simplify: it can take minutes on float-coefficient trigonometric expressions
    if negate:
        psi = -psi
    new_syms = [ys[list(swap).index(k)] if k in swap else s for k, s in syms.items()]
    return (psi, new_syms), ""


def legendre(potential: Potential, swap: Mapping[str, str], *, conjugate_sign: float | Mapping[str, float] = 1.0,
             negate: bool = False, name: str | None = None, unit: str | None = None,
             bounds: Mapping[str, tuple[float, float]] | None = None, n_grid: int = 41, expect: str = "convex",
             guess: Mapping[str, Any] | None = None, method: str = "auto", names: Mapping | None = None,
             responses: Mapping | None = None, relation_names: Mapping | None = None, tol: float = 1e-12,
             max_iter: int = 100, warn: bool = True) -> LegendreResult:
    """Legendre transform of ``potential`` in the variables of ``swap`` (``{old: new}``).

    Parameters
    ----------
    conjugate_sign:
        ``s`` in ``y = s dPhi/dx`` (``-1`` for ``p = -dA/dV``), one value or one per swapped variable.
    negate:
        return ``-(Phi - x dPhi/dx)`` (the Hamiltonian from a Lagrangian).
    bounds:
        ``{variable: (lo, hi)}`` of the original variables: the grid of the convexity check, and the bracket of the
        Newton solve for the swapped variables.
    guess:
        starting value of each swapped (old) variable: a number, or ``fn(**new_point) -> array``. Defaults to the
        middle of its bounds.
    method:
        ``"auto"`` (symbolic when possible), ``"numeric"`` or ``"symbolic"``.
    """
    swap = dict(swap)
    for x in swap:
        if x not in potential.variables:
            raise KeyError(f"{x!r} is not a variable of {potential.name}")
        if potential.variables[x].shape:
            raise NotImplementedError("Legendre transforms of tensor variables are not supported; swap components")
    signs = _signs(swap, conjugate_sign)
    new_vars: dict[str, Var] = {}
    for k, v in potential.variables.items():
        if k in swap:
            nm = potential.names.get((k,))
            new_vars[swap[k]] = Var(unit=nm.unit if nm and nm.unit else potential.unit_of((k,)))
        else:
            new_vars[k] = v
    name = name or f"L[{potential.name}]"
    unit = potential.unit if unit is None else unit

    conv = None
    if bounds is not None and all(k in bounds for k in potential.variables):
        conv = convexity(potential, tuple(swap), bounds, n_grid=n_grid, expect=expect)
        if warn and not conv.ok:
            warnings.warn(f"{potential.name} is not {expect} in {tuple(swap)} on the whole grid "
                          f"({100 * conv.fraction_stable:.1f} % stable): the transform is not one-to-one there",
                          stacklevel=2)

    kw = dict(name=name, unit=unit, names=names, responses=responses, relation_names=relation_names,
              description=f"Legendre transform of {potential.name} in {', '.join(swap)}")
    note = ""
    if method in ("auto", "symbolic") and potential.expr is not None:
        sym, note = _symbolic(potential, swap, signs, negate)
        if sym is not None:
            psi, new_syms = sym
            p = Potential.from_sympy(psi, {s: new_vars[str(s)].unit for s in new_syms}, **kw)
            return LegendreResult(p, conv, "symbolic", swap, note)
    if method == "symbolic":
        raise ValueError(f"no symbolic transform: {note or 'the potential has no SymPy expression'}")

    old = list(potential.variables)
    ix = [potential._offsets[x] for x in swap]
    xnames, ynames = list(swap), list(swap.values())
    s_vec = torch.tensor([signs[x] for x in xnames], dtype=DTYPE)
    guess = dict(guess or {})
    for x in xnames:
        if x not in guess:
            if bounds is None or x not in bounds:
                raise ValueError(f"give a guess or bounds for {x!r}")
            guess[x] = 0.5 * (bounds[x][0] + bounds[x][1])
    lo = torch.tensor([bounds[x][0] if bounds and x in bounds else -np.inf for x in xnames], dtype=DTYPE)
    hi = torch.tensor([bounds[x][1] if bounds and x in bounds else np.inf for x in xnames], dtype=DTYPE)

    def build_z(x: torch.Tensor, newx: Mapping[str, torch.Tensor]) -> torch.Tensor:
        cols = []
        for k in old:
            if k in swap:
                cols.append(x[:, xnames.index(k)][:, None])
            else:
                v = potential.variables[k]
                cols.append(newx[k].reshape(-1, v.size))
        return torch.cat(cols, dim=1)

    def g_J(z: torch.Tensor, create_graph: bool):
        flat = potential.flat_derivatives(z, 2, create_graph=create_graph)
        g = torch.stack([flat[(i,)] for i in ix], dim=1) * s_vec
        k = len(ix)
        J = torch.empty(z.shape[0], k, k, dtype=DTYPE)
        for a, b in itertools.product(range(k), range(k)):
            J[:, a, b] = flat[tuple(sorted((ix[a], ix[b])))] * s_vec[a]
        return flat[()], g, J

    def solve_root(newx: Mapping[str, torch.Tensor]) -> torch.Tensor:
        n = next(iter(newx.values())).shape[0]
        y = torch.stack([newx[yn].detach().reshape(n) for yn in ynames], dim=1)
        nd = {k: v.detach() for k, v in newx.items()}
        cols = []
        for xn in xnames:
            gv = guess[xn]
            gv = gv(**{k: v.numpy() for k, v in nd.items()}) if callable(gv) else gv
            cols.append(torch.as_tensor(np.broadcast_to(np.asarray(gv, float), (n,)).copy(), dtype=DTYPE))
        x = torch.stack(cols, dim=1)
        a, b = lo.expand(n, -1).clone(), hi.expand(n, -1).clone()
        bracketed = len(xnames) == 1 and bool(torch.isfinite(lo).all() and torch.isfinite(hi).all())
        with torch.enable_grad():
            for _ in range(max_iter):
                _, g, J = g_J(build_z(x, nd).requires_grad_(True), False)
                r = (g - y).detach()
                J = J.detach()
                if bracketed:
                    # g is monotone where Phi is convex: keep a bracket [a, b] around the root
                    inc = J[:, 0, 0] > 0
                    lower = torch.where(inc, r[:, 0] < 0, r[:, 0] > 0)
                    a[:, 0] = torch.where(lower, torch.maximum(a[:, 0], x[:, 0]), a[:, 0])
                    b[:, 0] = torch.where(~lower, torch.minimum(b[:, 0], x[:, 0]), b[:, 0])
                step = torch.linalg.solve(J, r[..., None])[..., 0]
                xn = x - step
                if bracketed:
                    out = (xn <= a) | (xn >= b) | ~torch.isfinite(xn)
                    xn = torch.where(out, 0.5 * (a + b), xn)
                else:
                    xn = torch.where(torch.isfinite(xn), xn, x)
                    xn = torch.minimum(torch.maximum(xn, lo), hi)
                done = (xn - x).abs().max() <= tol * (1.0 + x.abs().max())
                x = xn
                if done:
                    break
        return x.detach()

    def fn(**newx):
        n = next(iter(newx.values())).shape[0]
        x = solve_root(newx)
        y = torch.stack([newx[yn].reshape(n) for yn in ynames], dim=1)
        with torch.enable_grad():
            for _ in range(2):
                z = build_z(x, newx)
                if not z.requires_grad:
                    z = z.requires_grad_(True)
                _, g, J = g_J(z, True)
                x = x - torch.linalg.solve(J, (g - y)[..., None])[..., 0]
            z = build_z(x, newx)
            phi = potential.call_flat(z)
            psi = phi - (x * y / s_vec).sum(dim=1)
        return -psi if negate else psi

    p = Potential(fn, new_vars, **kw)

    def residual(point):
        z, bshape = p._prepare(point)
        newx = p.unflatten(z)
        x = solve_root(newx)
        with torch.enable_grad():
            _, g, _ = g_J(build_z(x, {k: v for k, v in newx.items()}).requires_grad_(True), False)
        y = torch.stack([newx[yn].reshape(-1) for yn in ynames], dim=1)
        r = (g.detach() - y).abs().max(dim=1).values.numpy()
        return r.reshape(bshape) if bshape else r.reshape(())

    return LegendreResult(p, conv, "numeric", swap, note, residual)
