"""Morris elementary effects: a cheap screening of which inputs matter, ``r (d + 1)`` evaluations.

Each of ``r`` trajectories starts at a random point of a ``levels``-level grid of the unit hypercube and moves one
input at a time by ``delta = levels / (2 (levels - 1))``, in random order (Morris 1991, Technometrics 33:161). The
elementary effect of input ``i`` is the output change per unit of its *normalised* range (the unit hypercube, mapped
to each distribution through its inverse CDF; for an unbounded distribution the unit interval covers the
0.5 %-99.5 % quantiles). Per input: ``mu`` (mean effect), ``mu_star`` (mean absolute effect, Campolongo et al. 2007:
the importance measure) and ``sigma`` (spread: nonlinearity or interactions), with a bootstrap interval on
``mu_star`` over trajectories.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from pinneapple_analysis.uncertainty.distributions import as_space

EPS_UNBOUNDED = 0.005


def _to_values(space, U: np.ndarray) -> np.ndarray:
    cols = []
    for i, p in enumerate(space):
        u = U[:, i] if p.bounded else EPS_UNBOUNDED + (1 - 2 * EPS_UNBOUNDED) * U[:, i]
        cols.append(p.ppf(u))
    return np.column_stack(cols)


@dataclass
class MorrisResult:
    names: list[str]
    outputs: list[str]
    r: int
    levels: int
    delta: float
    n_evals: int
    mu: np.ndarray          # (m, d)
    mu_star: np.ndarray     # (m, d)
    sigma: np.ndarray       # (m, d)
    mu_star_ci: np.ndarray  # (2, m, d)
    level: float = 0.95
    effects: np.ndarray = field(default=None, repr=False)   # (r, d, m)

    def ranking(self, output: int = 0) -> list[str]:
        return [self.names[i] for i in np.argsort(-self.mu_star[output])]

    def to_dict(self) -> dict:
        return {"method": "Morris elementary effects (one-at-a-time trajectories)", "names": self.names,
                "outputs": self.outputs, "r": self.r, "levels": self.levels, "delta": self.delta,
                "n_evals": self.n_evals, "mu": self.mu.tolist(), "mu_star": self.mu_star.tolist(),
                "sigma": self.sigma.tolist(), "mu_star_ci": self.mu_star_ci.tolist(), "level": self.level}


def morris_design(d: int, r: int = 20, levels: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-hypercube trajectories ``(r (d + 1), d)``, the input moved at each step ``(r, d)`` and its sign."""
    if levels < 2 or levels % 2:
        raise ValueError("levels must be an even number >= 2")
    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    U = np.empty((r, d + 1, d))
    order = np.empty((r, d), dtype=int)
    sign = np.empty((r, d))
    for t in range(r):
        x = rng.choice(grid, size=d)
        U[t, 0] = x
        perm = rng.permutation(d)
        order[t] = perm
        for s, i in enumerate(perm):
            step = delta if x[i] + delta <= 1 + 1e-12 else -delta
            x = x.copy()
            x[i] += step
            U[t, s + 1] = x
            sign[t, s] = np.sign(step)
    return U.reshape(-1, d), order, sign


def morris(fn, space, r: int = 20, levels: int = 4, seed: int = 0, n_boot: int = 500, level: float = 0.95,
           outputs: Sequence[str] | None = None) -> MorrisResult:
    """Morris screening of ``fn`` over ``space`` with ``r`` trajectories (cost ``r (d + 1)`` evaluations)."""
    sp = as_space(space)
    d = len(sp)
    U, order, sign = morris_design(d, r, levels, seed)
    delta = levels / (2.0 * (levels - 1))
    Y = np.asarray(fn(_to_values(sp, U)), dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if not np.all(np.isfinite(Y)):
        raise ValueError("the model returned NaN or infinite values on the Morris design")
    m = Y.shape[1]
    Y = Y.reshape(r, d + 1, m)
    EE = np.empty((r, d, m))
    for t in range(r):
        dy = np.diff(Y[t], axis=0)                                  # (d, m), step s moves input order[t, s]
        EE[t, order[t]] = dy / (sign[t][:, None] * delta)
    mu = EE.mean(axis=0).T
    mu_star = np.abs(EE).mean(axis=0).T
    sigma = (EE.std(axis=0, ddof=1) if r > 1 else np.zeros((d, m))).T
    rng = np.random.default_rng(seed + 1)
    boots = np.abs(EE)[rng.integers(0, r, size=(n_boot, r))].mean(axis=1)      # (n_boot, d, m)
    a = (1 - level) / 2
    ci = np.moveaxis(np.quantile(boots, [a, 1 - a], axis=0), -1, 1)          # (2, m, d)
    return MorrisResult(names=[p.name for p in sp], outputs=list(outputs or [f"y{j}" for j in range(m)]), r=r,
                        levels=levels, delta=delta, n_evals=r * (d + 1), mu=mu, mu_star=mu_star, sigma=sigma,
                        mu_star_ci=ci, level=level, effects=EE)


__all__ = ["MorrisResult", "morris", "morris_design"]
