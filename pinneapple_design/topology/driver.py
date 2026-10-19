"""Optimization loop: design variables -> (symmetry) -> density filter -> Heaviside projection -> problem.

    from pinneapple_design.topology import Compliance2D, DensityFilter, optimize
    prob = Compliance2D.mbb(60, 20)
    res = optimize(prob, volfrac=0.5, filt=DensityFilter.grid(60, 20, r_min=2.4), optimizer="oc")
    res.x_phys  # physical densities, res.history["objective"]

The volume constraint ``sum(v x_phys) <= volfrac sum(v)`` acts on the physical (projected) densities.
Extra constraints ``g(x_phys) <= 0`` (with gradients) can be given to MMA as callables. With
projection, ``beta`` is doubled every ``beta_every`` iterations up to ``beta_max`` (continuation).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .filters import DensityFilter, SymmetryMap, heaviside, heaviside_derivative
from .optimizers import MMA, oc_update

__all__ = ["TopOptResult", "optimize", "finite_difference_check", "DesignChain"]


class DesignChain:
    """``x_phys = H_beta(W (S x))`` and its adjoint (chain rule)."""

    def __init__(self, n_el: int, filt: DensityFilter | None = None, symmetry: SymmetryMap | None = None,
                 beta: float = 0.0, eta: float = 0.5):
        self.sym = symmetry or SymmetryMap.identity(n_el)
        self.filt = filt
        self.beta, self.eta = beta, eta

    @property
    def n_var(self):
        return self.sym.n_var

    def forward(self, x):
        xe = self.sym(x)
        self._xt = self.filt(xe) if self.filt is not None else xe
        return heaviside(self._xt, self.beta, self.eta)

    def backward(self, g):
        g = g * heaviside_derivative(self._xt, self.beta, self.eta)
        if self.filt is not None:
            g = self.filt.backward(g)
        return self.sym.backward(g)


@dataclass
class TopOptResult:
    x: np.ndarray
    x_phys: np.ndarray
    objective: float
    history: dict = field(default_factory=dict)


def optimize(problem, volfrac: float, filt: DensityFilter | None = None, optimizer: str = "oc",
             max_iter: int = 200, tol: float = 0.01, x0: np.ndarray | None = None,
             symmetry: SymmetryMap | None = None, beta0: float = 0.0, beta_max: float = 0.0,
             beta_every: int = 40, eta: float = 0.5, move: float = 0.2,
             constraints: Sequence[Callable] = (), objective_scale: float | None = None,
             callback: Callable | None = None, min_iter: int = 0) -> TopOptResult:
    """Run density-based topology optimization of ``problem`` (see module docstring).

    ``optimizer``: ``"oc"`` (volume constraint only, non-increasing objectives such as compliance) or
    ``"mma"``. Stops when the largest design change is below ``tol`` (after the last ``beta`` step) or
    after ``max_iter`` iterations.
    """
    chain = DesignChain(problem.n, filt, symmetry, beta0, eta)
    v = np.asarray(problem.volumes, float)
    Vtot = v.sum()
    x = np.full(chain.n_var, volfrac) if x0 is None else np.asarray(x0, float).copy()
    hist = {"objective": [], "volume": [], "change": [], "beta": []}
    mma = MMA(chain.n_var, 1 + len(constraints), move=move) if optimizer == "mma" else None
    scale = objective_scale
    f = np.nan
    for it in range(max_iter):
        xp = chain.forward(x)
        f, df = problem.evaluate(xp)
        if scale is None:
            scale = 1.0 / max(abs(f), 1e-300)
        dfx = chain.backward(df)
        vol = float(v @ xp) / Vtot
        dvx = chain.backward(v / Vtot)
        if mma is None:
            xn = oc_update(x, dfx, dvx, volfrac, lambda z: float(v @ chain.forward(z)) / Vtot, move=move)
            chain.forward(x)  # restore the cached filtered field of x
        else:
            gs, dgs = [vol / volfrac - 1.0], [dvx / volfrac]
            for con in constraints:
                gc, dgc = con(xp)
                gs.append(gc)
                dgs.append(chain.backward(dgc))
            xn = mma.update(x, scale * f, scale * dfx, np.array(gs), np.array(dgs))
        change = float(np.max(np.abs(xn - x)))
        hist["objective"].append(float(f))
        hist["volume"].append(vol)
        hist["change"].append(change)
        hist["beta"].append(chain.beta)
        if callback is not None:
            callback(it, xp, f)
        x = xn
        if beta_max > 0 and chain.beta < beta_max and (it + 1) % beta_every == 0:
            chain.beta = max(1.0, 2.0 * chain.beta)
            continue
        if it + 1 >= min_iter and change < tol and (beta_max <= 0 or chain.beta >= beta_max):
            break
    xp = chain.forward(x)
    f, _ = problem.evaluate(xp)
    return TopOptResult(x, xp, float(f), hist)


def finite_difference_check(fun: Callable[[np.ndarray], tuple], x: np.ndarray, idx: Sequence[int],
                            h: float = 1e-6) -> np.ndarray:
    """Relative errors ``|g_adj - g_fd| / max(|g_fd|, tiny)`` of the gradient of ``fun`` (returning
    ``(value, gradient)``) at the entries ``idx``, central differences with step ``h``."""
    _, g = fun(x)
    errs = []
    for i in idx:
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd = (fun(xp)[0] - fun(xm)[0]) / (2 * h)
        errs.append(abs(g[i] - fd) / max(abs(fd), 1e-12 * max(1.0, np.max(np.abs(g)))))
    return np.array(errs)
