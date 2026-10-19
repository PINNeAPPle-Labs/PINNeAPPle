"""Residual-to-error calibration (#699): turn the PDE residual of a prediction, which needs no true answer, into an
error bound an engineer can use ("the error is below X with 90 % confidence").

Two routes, both reported:

**Theory (linear elliptic problems).** For a discrete problem ``A u = f`` with ``A`` symmetric positive definite and
``A >= c L`` for a reference operator ``L`` (both SPD), the error of any candidate ``u`` against the discrete solution
``u_h`` obeys

    ||u_h - u||_A  <=  ||r||_{L^-1} / sqrt(c),     r = f - A u,

because ``A^-1 <= L^-1 / c``. For Darcy and steady heat conduction, ``-div(k grad u) = f``, in the cell-centred
finite volumes of ``operator_benchmarks.darcy_2d`` (:func:`darcy_problem`), ``L`` is the same operator with ``k = 1``
and ``c = min k`` (every face coefficient is a harmonic mean, never below ``min k``). The bound is guaranteed and
needs no data; it can be loose when the coefficient contrast is large.

**Calibration (any problem).** :class:`ResidualErrorCalibrator` learns the map residual -> error on a calibration set
with split conformal prediction (the finite-sample quantile of ``uncertainty.conformal``): the score is
``error / g(residual)`` with ``g`` the identity (``mode="ratio"``) or a power law fitted on a separate set
(``mode="power"``), and the bound of a new query is ``q * g(residual)``. Coverage is guaranteed for queries
exchangeable with the calibration set; under an input shift it is measured and reported, not assumed
(:meth:`ResidualErrorCalibrator.evaluate`).

The residual is evaluated with the discretisation of the reference solver, so the reference solution has a residual
of round-off size, not a truncation-error floor (a finite-difference residual of a finite-volume solution would not
be zero).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["LinearSPDProblem", "darcy_problem", "heat_conduction_problem", "energy_error_bound",
           "ResidualErrorCalibrator", "CoverageReport", "coverage_study", "ResidualLoss"]


@dataclass
class LinearSPDProblem:
    """``A u = f`` with ``A`` SPD and ``A - c L`` positive semi-definite. ``weight`` is the quadrature weight of the
    inner products (the cell area ``h^2`` on a uniform grid), so norms approximate their continuous counterparts."""
    A: Any
    f: np.ndarray
    L: Any
    c: float
    weight: float = 1.0
    shape: tuple[int, ...] | None = None
    _L_solve: Any = field(default=None, repr=False)

    def _vec(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(u, float).reshape(-1)

    def residual(self, u: np.ndarray) -> np.ndarray:
        """``f - A u`` (same shape as ``u``)."""
        r = self.f - self.A @ self._vec(u)
        return r.reshape(np.shape(u))

    def energy_norm(self, v: np.ndarray) -> float:
        v = self._vec(v)
        return math.sqrt(max(self.weight * float(v @ (self.A @ v)), 0.0))

    def dual_norm(self, r: np.ndarray) -> float:
        """``||r||_{L^-1} = sqrt(w r . L^-1 r)``."""
        if self._L_solve is None:
            self._L_solve = spla.factorized(self.L.tocsc())
        r = self._vec(r)
        return math.sqrt(max(self.weight * float(r @ self._L_solve(r)), 0.0))

    def solve(self) -> np.ndarray:
        u = spla.spsolve(self.A.tocsc(), self.f)
        return u.reshape(self.shape) if self.shape else u


_L_CACHE: dict[int, tuple[Any, Any]] = {}


def darcy_problem(a: np.ndarray, f: float | np.ndarray = 1.0) -> LinearSPDProblem:
    """``-div(a grad u) = f`` on the unit square, ``u = 0`` on the boundary, in the discretisation of
    ``operator_benchmarks.darcy_2d`` (``a``, ``f`` at the ``n x n`` cell centres)."""
    from pinneapple_simulation.numerical_solvers.operator_benchmarks import darcy_matrix
    a = np.asarray(a, float)
    n = a.shape[0]
    if n not in _L_CACHE:
        L = darcy_matrix(np.ones((n, n)))
        _L_CACHE[n] = (L, spla.factorized(L.tocsc()))
    L, Ls = _L_CACHE[n]
    fv = np.broadcast_to(np.asarray(f, float), (n, n)).reshape(-1).copy()
    return LinearSPDProblem(darcy_matrix(a), fv, L, float(a.min()), 1.0 / n ** 2, (n, n), Ls)


def heat_conduction_problem(k: np.ndarray, q: float | np.ndarray) -> LinearSPDProblem:
    """Steady conduction ``-div(k grad T) = q`` with ``T = 0`` on the boundary of the unit square: the same operator
    as Darcy, with conductivity ``k`` and heat source ``q``."""
    return darcy_problem(k, q)


def energy_error_bound(problem: LinearSPDProblem, u: np.ndarray) -> dict[str, float]:
    """Guaranteed bound on ``||u_h - u||_A`` from the residual alone, absolute and relative to ``||u||_A``."""
    b = problem.dual_norm(problem.residual(u)) / math.sqrt(problem.c)
    e = problem.energy_norm(u)
    return {"bound": b, "relative": b / e if e > 0 else math.inf, "energy_norm": e}


def _quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample split-conformal quantile (the ``ceil((n + 1)(1 - alpha))``-th order statistic), from
    ``uncertainty.conformal``."""
    import torch

    from .conformal import _conformal_quantile
    return _conformal_quantile(torch.as_tensor(np.asarray(scores, float)), alpha)


@dataclass
class CoverageReport:
    label: str
    coverage: float
    target: float
    n: int
    holds: bool                      # coverage within sampling error of the target, or above it
    tolerance: float                 # two binomial standard deviations at the target
    median_bound: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __str__(self) -> str:
        return (f"{self.label}: {100 * self.coverage:.1f} % of {self.n} queries within the bound (target "
                f"{100 * self.target:.0f} %, +- {100 * self.tolerance:.1f} %) -> {'holds' if self.holds else 'BROKEN'}"
                + (f"; {self.note}" if self.note else ""))


class ResidualLoss:
    """A differentiable physics loss for training: ``mean_i ||f_i - A_i u_i||^2 / ||f_i||^2`` over a fixed set of
    problems (e.g. :func:`darcy_problem` of each training input), in PyTorch with sparse matrices. Call
    ``loss(u, idx)`` with predictions ``u`` (``(B, ...)``, any shape with ``n`` values per sample) for the problems
    ``idx`` (a list or tensor of indices into the set)."""

    def __init__(self, problems: list[LinearSPDProblem]):
        import torch
        self.A, self.f = [], []
        for P in problems:
            C = sp.coo_matrix(P.A)
            idx = torch.tensor(np.vstack([C.row, C.col]), dtype=torch.long)
            self.A.append(torch.sparse_coo_tensor(idx, torch.tensor(C.data, dtype=torch.float32), C.shape,
                                                   check_invariants=False).coalesce())
            self.f.append(torch.tensor(P.f, dtype=torch.float32))

    def __call__(self, u, idx):
        import torch
        u = u.reshape(len(u), -1)
        terms = []
        for k, i in enumerate(idx.tolist() if hasattr(idx, "tolist") else idx):
            r = self.f[i] - torch.sparse.mm(self.A[i], u[k:k + 1].T.to(torch.float32)).squeeze(1)
            terms.append((r ** 2).sum() / (self.f[i] ** 2).sum())
        return torch.stack(terms).mean()


class ResidualErrorCalibrator:
    """Split-conformal map from a residual measure to an error bound.

    ``fit`` (``mode="power"`` only) fits ``log err = b0 + b1 log res`` on its own set; ``calibrate`` computes the
    conformal quantile of ``err / g(res)``; ``bound(res)`` is ``q * g(res)``."""

    def __init__(self, alpha: float = 0.1, mode: str = "ratio"):
        if mode not in ("ratio", "power"):
            raise ValueError("mode must be 'ratio' or 'power'")
        self.alpha, self.mode = float(alpha), mode
        self.coef: tuple[float, float] = (0.0, 1.0)
        self.q: float | None = None
        self.n_cal = 0

    def g(self, res: np.ndarray) -> np.ndarray:
        r = np.maximum(np.asarray(res, float), 1e-300)
        return np.exp(self.coef[0]) * r ** self.coef[1]

    def fit(self, res: np.ndarray, err: np.ndarray) -> ResidualErrorCalibrator:
        if self.mode == "power":
            x, y = np.log(np.maximum(res, 1e-300)), np.log(np.maximum(err, 1e-300))
            b1, b0 = np.polyfit(x, y, 1)
            self.coef = (float(b0), float(b1))
        return self

    def calibrate(self, res: np.ndarray, err: np.ndarray) -> ResidualErrorCalibrator:
        scores = np.asarray(err, float) / self.g(res)
        self.q, self.n_cal = _quantile(scores, self.alpha), len(scores)
        return self

    def bound(self, res: np.ndarray) -> np.ndarray:
        if self.q is None:
            raise RuntimeError("call calibrate() first")
        return self.q * self.g(res)

    def evaluate(self, res: np.ndarray, err: np.ndarray, label: str = "held-out") -> CoverageReport:
        """Observed coverage on queries with known error. ``holds`` when it is at least the target minus two standard
        deviations (binomial, over the test and the calibration sizes); below that the report says the guarantee is
        broken (e.g. by an input shift)."""
        b = self.bound(res)
        err = np.asarray(err, float)
        cov = float(np.mean(err <= b))
        target = 1.0 - self.alpha
        # two standard deviations of the observed coverage: test sampling plus the calibration quantile's own spread
        tol = 2.0 * math.sqrt(target * (1 - target) * (1 / max(len(err), 1) + 1 / max(self.n_cal, 1)))
        holds = cov >= target - tol
        note = "" if holds else "coverage below the target: these queries are not exchangeable with the " \
                                "calibration set (input shift); do not use the bound for them"
        return CoverageReport(label, cov, target, len(err), holds, tol, float(np.median(b)), note)

    def to_check(self, res: float, limit: float | None = None):
        """A trust-report check "error bound without ground truth" for one query (supports when the bound is below
        ``limit``; without a limit the bound is reported and the check is not run)."""
        from pinneapple_analysis.trust.trust_report import CONTRADICTS, NOT_RUN, SUPPORTS, Check
        b = float(self.bound(np.asarray([res]))[0])
        status = NOT_RUN if limit is None else SUPPORTS if b <= limit else CONTRADICTS
        return Check("error bound without ground truth", status, value=b, limit=limit,
                     source="pinneapple_analysis.uncertainty.residual_bound",
                     reason=f"{100 * (1 - self.alpha):.0f} % split-conformal bound from the residual "
                            f"(n_cal = {self.n_cal}, mode {self.mode})")


def coverage_study(res: np.ndarray, err: np.ndarray, *, alpha: float = 0.1, n_cal: int | None = None,
                   n_splits: int = 200, seed: int = 0) -> dict[str, float]:
    """Coverage of the ``mode="ratio"`` bound over ``n_splits`` random calibration / test splits of one pool of
    exchangeable queries (``n_cal`` calibration queries, default half). One split's coverage fluctuates by about
    ``sqrt(alpha (1 - alpha) / n)``; the mean over splits is what the conformal guarantee is about."""
    res, err = np.asarray(res, float), np.asarray(err, float)
    n = len(res)
    n_cal = n_cal or n // 2
    rng = np.random.default_rng(seed)
    cov = []
    for _ in range(n_splits):
        p = rng.permutation(n)
        c = ResidualErrorCalibrator(alpha).calibrate(res[p[:n_cal]], err[p[:n_cal]])
        cov.append(float(np.mean(err[p[n_cal:]] <= c.bound(res[p[n_cal:]]))))
    cov = np.asarray(cov)
    return {"mean": float(cov.mean()), "std": float(cov.std()), "min": float(cov.min()), "max": float(cov.max()),
            "n_cal": n_cal, "n_test": n - n_cal, "n_splits": n_splits}
