"""X-TFC for data-driven parameter discovery: the fields and the unknown parameters of a differential system are
solved together by iterative least squares on stacked data and physics losses.

Method of E. Schiassi, A. D'Ambrosio, M. De Florio, R. Furfaro, F. Curti, "Physics-Informed Extreme Theory of
Functional Connections Applied to Data-Driven Parameters Discovery of Epidemiological Compartmental Models",
arXiv:2008.05554 (2020), Sec. 2:

- each field is a TFC constrained expression around an ELM (``xtfc_general``), so its constraints hold exactly;
- the loss vector stacks the data misfits ``u(x_d) - y_d`` and the residuals of the system, unweighted;
- the unknowns ``[xi, lambda]`` (output weights of every field, then the parameters) are updated by Gauss-Newton,
  ``Xi_{k+1} = Xi_k - (J^T J)^-1 J^T L``, until ``||L||`` is below a tolerance or stops decreasing (eq. 9).

The residual reads the parameters as tensors with one value per point, so their Jacobian columns come from one
autograd pass, the same way the field columns do.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from .xtfc_general import XTFCProblem, _affine, _Basis, _projection_terms

__all__ = ["InverseXTFCResult", "solve_xtfc_inverse"]


@dataclass
class InverseXTFCResult:
    params: dict[str, float]
    beta: dict[str, np.ndarray]
    iterations: int
    stop_reason: str
    loss_rms: list[float]
    seconds: float
    params_init: dict[str, float]
    _problem: XTFCProblem = field(repr=False)
    _basis: _Basis = field(repr=False)
    _terms: dict = field(repr=False)

    def predict(self, X, field_name: str, alpha=None) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        a = alpha or (0,) * self._problem.ndim
        A, c = _affine(self._problem, self._basis, self._terms, X, field_name, tuple(a))
        return A @ self.beta[field_name] + c


def solve_xtfc_inverse(problem: XTFCProblem, points, data: Mapping[str, tuple[np.ndarray, np.ndarray]],
                       residual: Callable[[torch.Tensor, dict, dict], torch.Tensor | Sequence[torch.Tensor]],
                       unknowns: Sequence[str], *, n_neurons: int, activation: str = "tanh",
                       weight_range: tuple[float, float] = (-1.0, 1.0), seed: int = 0, map_to_unit: bool = False,
                       tol: float = 1e-12, max_iter: int = 200,
                       params0: Mapping[str, float] | None = None) -> InverseXTFCResult:
    """Fit the fields of ``problem`` and the ``unknowns`` to ``data`` (field -> ``(X, y)``) under ``residual``.

    ``residual(X, D, P)``: ``D[(field, alpha)]`` as in :class:`XTFCProblem`, ``P[name]`` the parameter repeated at
    every point (torch, (N,)). Start: without ``params0``, every field is fitted to its data alone and the
    parameters by one linear least-squares step on the residuals at that fit (exact when the residual is linear
    in the parameters, as for compartmental models). Stop: RMS of the loss vector ``<= tol`` ("converged"), an
    increase of it (the step is undone, "increase", eq. 9 of the paper), or ``max_iter``.
    """
    t_start = time.perf_counter()
    X = np.atleast_2d(np.asarray(points, dtype=np.float64))
    rng = np.random.default_rng(seed)
    lo, hi = weight_range
    W = rng.uniform(lo, hi, size=(n_neurons, problem.ndim))
    b = rng.uniform(lo, hi, size=n_neurons)
    if map_to_unit:
        x0, x1 = np.array(problem.bounds, dtype=np.float64).T
        scale = 2.0 / (x1 - x0)
        b = b - W @ (scale * x0 + 1.0)
        W = W * scale
    basis = _Basis(W, b, activation)
    names = list(problem.fields)
    terms = {f: _projection_terms(problem.ndim, problem.fields[f]) for f in names}
    affine = {(f, tuple(a)): _affine(problem, basis, terms, X, f, tuple(a)) for f, a in problem.derivatives}
    zero = (0,) * problem.ndim
    data_rows = []
    for f, (Xd, yd) in data.items():
        A, c = _affine(problem, basis, terms, np.atleast_2d(np.asarray(Xd, dtype=np.float64)), f, zero)
        data_rows.append((names.index(f), A, c, np.asarray(yd, dtype=np.float64).ravel()))
    nb, npar = n_neurons * len(names), len(unknowns)
    Xt = torch.tensor(X, dtype=torch.float64)

    def losses(beta, params, jac=True):
        D = {k: torch.tensor(A @ beta[k[0]] + c, dtype=torch.float64, requires_grad=jac) for k, (A, c) in affine.items()}
        Pt = {p: torch.full((X.shape[0],), float(v), dtype=torch.float64, requires_grad=jac) for p, v in params.items()}
        R = residual(Xt, D, Pt)
        R = [R] if isinstance(R, torch.Tensor) else list(R)
        rows_J, rows_r = [], []
        for j, A, c, y in data_rows:
            Jd = np.zeros((A.shape[0], nb + npar))
            Jd[:, j * n_neurons:(j + 1) * n_neurons] = A
            rows_J.append(Jd)
            rows_r.append(A @ beta[names[j]] + c - y)
        for r in R:
            Jr = np.zeros((X.shape[0], nb + npar))
            if jac:
                ins = list(D.values()) + [Pt[p] for p in unknowns]
                grads = torch.autograd.grad(r.sum(), ins, retain_graph=True, allow_unused=True)
                for (key, (A, _)), g in zip(affine.items(), grads[:len(D)], strict=True):
                    if g is not None:
                        j = names.index(key[0])
                        Jr[:, j * n_neurons:(j + 1) * n_neurons] += g.detach().numpy()[:, None] * A
                for k, g in enumerate(grads[len(D):]):
                    if g is not None:
                        Jr[:, nb + k] = g.detach().numpy()
            rows_J.append(Jr)
            rows_r.append(r.detach().numpy())
        return np.vstack(rows_J), np.concatenate(rows_r)

    beta = {f: np.zeros(n_neurons) for f in names}
    for j, A, c, y in data_rows:
        beta[names[j]] = np.linalg.lstsq(A, y - c, rcond=None)[0]
    params = {p: 0.0 for p in unknowns}
    if params0 is not None:
        params.update({p: float(v) for p, v in params0.items()})
    else:
        J, res = losses(beta, params)
        rows = slice(sum(A.shape[0] for _, A, _, _ in data_rows), None)
        dp = np.linalg.lstsq(J[rows, nb:], -res[rows], rcond=None)[0]
        params = {p: params[p] + float(dp[k]) for k, p in enumerate(unknowns)}

    params_init = dict(params)
    J, res = losses(beta, params)
    history = [float(np.sqrt(np.mean(res * res)))]
    reason, it = "max_iter", 0
    for it in range(1, max_iter + 1):  # noqa: B007 (the count is reported)
        if history[-1] <= tol:
            reason, it = "converged", it - 1
            break
        delta = np.linalg.lstsq(J, -res, rcond=None)[0]
        new_beta = {f: beta[f] + delta[j * n_neurons:(j + 1) * n_neurons] for j, f in enumerate(names)}
        new_params = {p: params[p] + float(delta[nb + k]) for k, p in enumerate(unknowns)}
        J2, res2 = losses(new_beta, new_params)
        rms = float(np.sqrt(np.mean(res2 * res2)))
        if not math.isfinite(rms) or rms > history[-1]:
            reason = "increase"
            break
        beta, params, J, res = new_beta, new_params, J2, res2
        history.append(rms)
    return InverseXTFCResult(params, beta, it, reason, history, time.perf_counter() - t_start, params_init, problem,
                             basis, terms)
