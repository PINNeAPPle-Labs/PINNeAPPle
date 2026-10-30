"""General X-TFC on a box: any number of dimensions and fields, value and derivative constraints on faces,
linear or non-linear residuals.

The existing ``xtfc_ivp`` / ``xtfc_pde`` solvers each cover a fixed set of named problems. This module is the
method of Schiassi et al. for an arbitrary differential problem on a hyper-rectangle:

- the solution of each field is a TFC constrained expression ``u = P[g]`` whose free function ``g`` is an ELM:
  ``g(x) = sigma(W x + b)^T beta`` with ``W``, ``b`` drawn once and frozen (Huang et al. 2006);
- ``P`` is built dimension by dimension (multivariate TFC, Mortari & Leake 2019). For each dimension ``i`` with
  constraints ``C_k u = d^{m_k} u / dx_i^{m_k} (x_i = a_k)``, ``P_i[e] = e + sum_k s_k(x_i) (c_k - C_k e)`` with
  switching functions ``s_k`` on the monomial support ``{1, x, ..., x^{K-1}}`` (``C_k s_j = delta_kj``). The
  composition of the ``P_i`` equals the tensor form ``g + v^T M(c - g) v`` of the paper for compatible data;
- the constraint data come from any smooth function ``h`` that satisfies them (``h`` is only read on the
  constrained faces): ``u = P_0[g] + h - P_0[h]``, so every constraint holds exactly for any ``beta``;
- ``u`` and all its derivatives are affine in ``beta``; the residual is solved by Gauss-Newton with
  ``numpy.linalg.lstsq`` at each step (one step is the exact solve of a linear problem).

Derivatives of the ELM features are analytic (``d^n sigma / dz^n`` as a polynomial in ``sigma``), so no
automatic differentiation goes through the basis; only the user's ``h`` and residual go through torch autograd.

References: E. Schiassi, C. Leake, M. De Florio, H. Johnston, R. Furfaro, D. Mortari, arXiv:2005.10632 (2020),
Neurocomputing 457 (2021) 334-356; D. Mortari, C. Leake, "The Multivariate Theory of Connections", Mathematics
7 (2019) 296; G.-B. Huang, Q.-Y. Zhu, C.-K. Siew, Neurocomputing 70 (2006) 489-501.
"""
from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.polynomial import polynomial as P

__all__ = ["Constraint", "XTFCProblem", "XTFCResult", "solve_xtfc", "activation_derivative",
           "switching_functions"]

Alpha = tuple[int, ...]

# d^n/dz^n of the activation as a polynomial in the activation value a = act(z): p_{n+1}(a) = p_n'(a) * a'(a).
_ACT = {
    "tanh": (np.tanh, np.array([1.0, 0.0, -1.0])),                       # a' = 1 - a^2
    "logistic": (lambda z: 0.5 * (1.0 + np.tanh(0.5 * z)), np.array([0.0, 1.0, -1.0])),   # a' = a - a^2
}
_ACT["sigmoid"] = _ACT["logistic"]


def _act_polys(name: str, n: int) -> list[np.ndarray]:
    _, dpoly = _ACT[name]
    polys = [np.array([0.0, 1.0])]
    for _ in range(n):
        polys.append(P.polymul(P.polyder(polys[-1]), dpoly))
    return polys


def activation_derivative(name: str, z: np.ndarray, n: int) -> np.ndarray:
    """``d^n act / dz^n`` at ``z`` (``name`` in ``tanh``, ``logistic``, ``arctan``; ``arctan`` up to n = 3)."""
    if name == "arctan":
        q = 1.0 / (1.0 + z * z)
        derivs = (np.arctan(z), q, -2.0 * z * q * q, (6.0 * z * z - 2.0) * q ** 3)
        if n >= len(derivs):
            raise ValueError("arctan derivatives are implemented up to order 3")
        return derivs[n]
    if name not in _ACT:
        raise ValueError(f"activation must be one of {sorted(_ACT)}, got {name!r}")
    a = _ACT[name][0](z)
    return P.polyval(a, _act_polys(name, n)[n])


@dataclass(frozen=True)
class Constraint:
    """``d^order u / dx_dim^order`` at ``x_dim = at`` is prescribed (by the problem's constraint function)."""
    dim: int
    at: float
    order: int = 0


def switching_functions(constraints: Sequence[Constraint]) -> np.ndarray:
    """Coefficients (monomial basis, row k = s_k) of the switching functions of constraints on one dimension."""
    K = len(constraints)
    C = np.zeros((K, K))
    for k, c in enumerate(constraints):
        for j in range(K):
            C[k, j] = P.polyval(c.at, P.polyder(np.eye(K)[j], c.order)) if j >= c.order else 0.0
    if abs(np.linalg.det(C)) < 1e-12:
        raise ValueError(f"constraints {constraints} do not define switching functions on the monomial support")
    return np.linalg.inv(C).T          # S[k] = coefficients of s_k


# A term of P_0[.]: (sign, per-dimension entry); entry None = free, or (at, g_order, switch_coeffs) = fixed.
_Entry = tuple[float, int, np.ndarray] | None


def _projection_terms(ndim: int, constraints: Sequence[Constraint]) -> list[tuple[float, tuple[_Entry, ...]]]:
    terms: list[tuple[float, tuple[_Entry, ...]]] = [(1.0, (None,) * ndim)]
    for d in range(ndim):
        cs = [c for c in constraints if c.dim == d]
        if not cs:
            continue
        S = switching_functions(cs)
        new = list(terms)
        for k, c in enumerate(cs):
            for sign, ent in terms:
                e = list(ent)
                e[d] = (c.at, c.order, S[k])
                new.append((-sign, tuple(e)))
        terms = new
    return terms


@dataclass
class XTFCProblem:
    """A differential problem on ``bounds`` for X-TFC.

    ``fields``: field name -> its constraints. ``constraint_fn``: field name -> torch callable ``h(X) -> (N,)``
    that satisfies that field's constraints (e.g. the exact solution, or any smooth extension of the data).
    ``residual(X, D)``: ``D[(field, alpha)]`` is the derivative ``alpha`` (a multi-index) of ``field`` at ``X``
    (torch, shape (N,)); returns one residual tensor (N,) or a list of them, zero at the solution. ``derivatives``
    lists every ``(field, alpha)`` the residual reads.
    """
    bounds: Sequence[tuple[float, float]]
    fields: Mapping[str, Sequence[Constraint]]
    constraint_fn: Mapping[str, Callable[[torch.Tensor], torch.Tensor]]
    residual: Callable[[torch.Tensor, dict], torch.Tensor | Sequence[torch.Tensor]]
    derivatives: Sequence[tuple[str, Alpha]]
    linear: bool = False

    @property
    def ndim(self) -> int:
        return len(self.bounds)


def _torch_derivative(h: Callable, X: np.ndarray, alpha: Alpha) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float64, requires_grad=sum(alpha) > 0)
    out = h(Xt).reshape(-1)
    for d, n in enumerate(alpha):
        for _ in range(n):
            out = torch.autograd.grad(out.sum(), Xt, create_graph=True)[0][:, d]
    return out.detach().numpy()


class _Basis:
    def __init__(self, W: np.ndarray, b: np.ndarray, activation: str):
        self.W, self.b, self.activation = W, b, activation

    def derivative(self, X: np.ndarray, alpha: Alpha) -> np.ndarray:
        z = X @ self.W.T + self.b
        scale = np.prod([self.W[:, d] ** n for d, n in enumerate(alpha)], axis=0)
        return activation_derivative(self.activation, z, sum(alpha)) * scale


def _apply(terms, X: np.ndarray, alpha: Alpha, deriv: Callable[[np.ndarray, Alpha], np.ndarray],
           skip_identity: bool = False) -> np.ndarray:
    """``D^alpha P_0[f]`` at ``X`` (``deriv(Xs, a)`` = ``D^a f`` at shifted points ``Xs``)."""
    total = None
    for sign, ent in terms:
        if skip_identity and all(e is None for e in ent):
            continue
        Xs = X.copy()
        g_alpha = list(alpha)
        coef = np.full(X.shape[0], sign)
        for d, e in enumerate(ent):
            if e is None:
                continue
            at, order, s = e
            Xs[:, d] = at
            g_alpha[d] = order
            coef = coef * P.polyval(X[:, d], P.polyder(s, alpha[d]) if alpha[d] else s)
        if not np.any(coef):
            continue
        val = deriv(Xs, tuple(g_alpha))
        val = coef[:, None] * val if val.ndim == 2 else coef * val
        total = val if total is None else total + val
    if total is None:
        return np.zeros((X.shape[0],))
    return total


@dataclass
class XTFCResult:
    beta: dict[str, np.ndarray]
    iterations: int
    converged: bool
    step_norms: list[float]
    lstsq_seconds: float
    total_seconds: float
    n_neurons: int
    _problem: XTFCProblem = field(repr=False)
    _basis: _Basis = field(repr=False)
    _terms: dict = field(repr=False)

    def predict(self, X, field_name: str | None = None, alpha: Alpha | None = None) -> np.ndarray:
        """Field (or its derivative ``alpha``) at ``X`` (N, ndim)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        f = field_name or next(iter(self._problem.fields))
        a = alpha or (0,) * self._problem.ndim
        A, c = _affine(self._problem, self._basis, self._terms, X, f, a)
        return A @ self.beta[f] + c

    def residual(self, X) -> list[np.ndarray]:
        """The problem's residual(s) at ``X`` (fresh points: a physics check, not the training loss)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        D = {(f, tuple(a)): torch.tensor(self.predict(X, f, tuple(a)), dtype=torch.float64)
             for f, a in self._problem.derivatives}
        with torch.no_grad():
            R = self._problem.residual(torch.tensor(X, dtype=torch.float64), D)
        return [r.numpy() for r in ([R] if isinstance(R, torch.Tensor) else R)]

    def constraint_error(self, n: int = 200, seed: int = 0) -> float:
        """Max |D^order u - D^order h| over ``n`` random points on every constrained face, all fields."""
        p = self._problem
        lo, hi = np.array(p.bounds, dtype=np.float64).T
        rng = np.random.default_rng(seed)
        worst = 0.0
        for f, cs in p.fields.items():
            for c in cs:
                X = lo + (hi - lo) * rng.random((n, p.ndim))
                X[:, c.dim] = c.at
                a = tuple(c.order if d == c.dim else 0 for d in range(p.ndim))
                ref = _torch_derivative(p.constraint_fn[f], X, a)
                worst = max(worst, float(np.max(np.abs(self.predict(X, f, a) - ref))))
        return worst


def _affine(problem: XTFCProblem, basis: _Basis, terms: dict, X: np.ndarray, f: str, alpha: Alpha):
    """``D^alpha u_f = A beta_f + c`` at ``X``."""
    A = _apply(terms[f], X, alpha, basis.derivative)
    h = problem.constraint_fn[f]
    c = -_apply(terms[f], X, alpha, lambda Xs, a: _torch_derivative(h, Xs, a), skip_identity=True)
    return A, c


def solve_xtfc(problem: XTFCProblem, points, *, n_neurons: int, activation: str = "tanh",
               weight_range: tuple[float, float] = (-1.0, 1.0), seed: int = 0, max_iter: int | None = None,
               tol: float = 1e-14, beta0: float = 0.0, map_to_unit: bool = False,
               constraints: str = "expression") -> XTFCResult:
    """Solve ``problem`` on the training ``points`` (N, ndim) by X-TFC.

    Input weights and biases are drawn ``unif(weight_range)`` with ``numpy.random.default_rng(seed)``. Linear
    problems take one least-squares solve; non-linear ones Gauss-Newton from ``beta = beta0`` until the max-norm
    of the step is ``<= tol`` or ``max_iter`` steps (default 1 for linear, 20 otherwise). ``map_to_unit`` feeds
    the ELM each coordinate mapped affinely onto [-1, 1] (the drawn weights then act on the mapped inputs).

    ``constraints="expression"`` is X-TFC (constraints exact through the constrained expression). ``"rows"`` is the
    plain physics-informed ELM baseline (Dwivedi & Srinivasan, arXiv:1907.03507): ``u = g``, and each constraint
    becomes least-squares rows at the training points that lie on its face, with weight 1.
    """
    if constraints not in ("expression", "rows"):
        raise ValueError(f"constraints must be 'expression' or 'rows', got {constraints!r}")
    t_start = time.perf_counter()
    X = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if X.shape[1] != problem.ndim:
        raise ValueError(f"points have {X.shape[1]} columns, the problem has {problem.ndim} dimensions")
    rng = np.random.default_rng(seed)
    lo, hi = weight_range
    W = rng.uniform(lo, hi, size=(n_neurons, problem.ndim))
    b = rng.uniform(lo, hi, size=n_neurons)
    if map_to_unit:      # w . (2 (x - x0) / (x1 - x0) - 1) + b, written as W' x + b'
        x0, x1 = np.array(problem.bounds, dtype=np.float64).T
        scale = 2.0 / (x1 - x0)
        b = b - W @ (scale * x0 + 1.0)
        W = W * scale
    basis = _Basis(W, b, activation)
    names = list(problem.fields)
    rows_mode = constraints == "rows"
    terms = {f: _projection_terms(problem.ndim, [] if rows_mode else problem.fields[f]) for f in names}
    con_rows = []                     # (field index, A on the face points, target derivative of h there)
    if rows_mode:
        for j, f in enumerate(names):
            for c in problem.fields[f]:
                sel = np.isclose(X[:, c.dim], c.at, rtol=0.0, atol=1e-12)
                if not np.any(sel):
                    raise ValueError(f"no training point lies on the face of {c}")
                a = tuple(c.order if d == c.dim else 0 for d in range(problem.ndim))
                con_rows.append((j, basis.derivative(X[sel], a), _torch_derivative(problem.constraint_fn[f], X[sel], a)))
    affine = {(f, a): _affine(problem, basis, terms, X, f, tuple(a)) for f, a in problem.derivatives}
    beta = {f: np.full(n_neurons, beta0) for f in names}
    max_iter = max_iter if max_iter is not None else (1 if problem.linear else 20)
    Xt = torch.tensor(X, dtype=torch.float64)
    steps: list[float] = []
    t_lstsq = 0.0
    converged = False
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 (the count is reported)
        D = {key: torch.tensor(A @ beta[key[0]] + c, dtype=torch.float64, requires_grad=True)
             for key, (A, c) in affine.items()}
        R = problem.residual(Xt, D)
        R = [R] if isinstance(R, torch.Tensor) else list(R)
        rows_J, rows_r = [], []
        for r in R:
            grads = torch.autograd.grad(r.sum(), list(D.values()), retain_graph=True, allow_unused=True)
            J = np.zeros((X.shape[0], n_neurons * len(names)))
            for (key, (A, _)), g in zip(affine.items(), grads, strict=True):
                if g is None:
                    continue
                j = names.index(key[0])
                J[:, j * n_neurons:(j + 1) * n_neurons] += g.detach().numpy()[:, None] * A
            rows_J.append(J)
            rows_r.append(r.detach().numpy())
        for j, A_c, h_c in con_rows:
            J = np.zeros((A_c.shape[0], n_neurons * len(names)))
            J[:, j * n_neurons:(j + 1) * n_neurons] = A_c
            rows_J.append(J)
            rows_r.append(A_c @ beta[names[j]] - h_c)
        J = np.vstack(rows_J)
        res = np.concatenate(rows_r)
        t0 = time.perf_counter()
        delta = np.linalg.lstsq(J, -res, rcond=None)[0]
        t_lstsq += time.perf_counter() - t0
        for j, f in enumerate(names):
            beta[f] = beta[f] + delta[j * n_neurons:(j + 1) * n_neurons]
        step = float(np.max(np.abs(delta)))
        steps.append(step)
        if problem.linear or step <= tol:
            converged = True
            break
        if not np.isfinite(step):
            break
    return XTFCResult(beta, it, converged, steps, t_lstsq, time.perf_counter() - t_start, n_neurons,
                      problem, basis, terms)


def grid(bounds: Sequence[tuple[float, float]], n: int | Sequence[int], kind: str = "uniform") -> np.ndarray:
    """Tensor grid (N, ndim) of ``n`` points per dimension, ``uniform`` or ``cgl`` (Chebyshev-Gauss-Lobatto),
    boundary included."""
    ns = [n] * len(bounds) if isinstance(n, int) else list(n)
    axes = []
    for (lo, hi), m in zip(bounds, ns, strict=True):
        if kind == "uniform":
            s = np.linspace(0.0, 1.0, m)
        elif kind == "cgl":
            s = 0.5 * (1.0 - np.cos(np.pi * np.arange(m) / (m - 1)))
        else:
            raise ValueError(f"kind must be 'uniform' or 'cgl', got {kind!r}")
        axes.append(lo + (hi - lo) * s)
    return np.array(list(itertools.product(*axes)))


__all__.append("grid")
