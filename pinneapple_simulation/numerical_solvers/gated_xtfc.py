"""Gated X-TFC: X-TFC with Gaussian RBF features whose centers and widths follow a soft domain split, for 1D
second-order linear boundary value problems with sharp layers (forward and inverse).

Method of V. Dwivedi, E. Schiassi, M. Sigovan, B. Sixou, "Gated X-TFC: Soft Domain Decomposition for Forward and
Inverse Problems in Sharp-Gradient PDEs", arXiv:2510.01039 (2025), Sec. 3:

- trial ``u = g + sum_i c_i psi_i`` with ``g = (1 - x) u_L + x u_R`` and
  ``psi_i(x) = phi(z_i(x)) - (1 - x) phi(z_i(0)) - x phi(z_i(1))``, ``phi(z) = exp(-z^2)``,
  ``z_i = m_i x + b_i``, ``m_i = 1 / (sqrt(2) sigma_i)``, ``b_i = -m_i alpha_i`` (Dirichlet data exact);
- splits ``x_s`` cut [0, 1] into blocks; each block gets evenly spaced centers and collocation points
  ("endpoint grids") and a nominal width ``k * spacing``; widths are blended across each split by a logistic gate
  of scale ``eps_b = max(eps_scale * min(adjacent spacings), 5 nu)``;
- forward: ridge least squares on the strong-form residual, the gate parameters chosen by the residual energy on
  an independent validation grid (nested bounded Brent search, as MATLAB ``fminbnd``);
- inverse: data and physics rows whitened and stacked, the coefficient prior precision set by evidence
  maximisation, and the (parameter, gate) triple chosen by maximising the evidence.

The operator is ``a(x) u'' + b(x) u' + c(x) u = f(x)`` on [0, 1]; :func:`convection_diffusion` and
:func:`twin_boundary_layer` are the paper's two 1D cases.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = ["LinearBVP1D", "convection_diffusion", "twin_boundary_layer", "GatedLayout", "gated_layout",
           "GatedSolution", "solve_forward", "validation_energy", "search_gates_nested", "log_evidence",
           "InverseFit", "fit_inverse", "search_gates_bo"]

Fn = Callable[[np.ndarray], np.ndarray]


def _const(v: float) -> Fn:
    return lambda x: np.full_like(np.asarray(x, dtype=np.float64), v)


@dataclass(frozen=True)
class LinearBVP1D:
    """``a u'' + b u' + c u = f`` on [0, 1], ``u(0) = u_left``, ``u(1) = u_right``."""
    a: Fn
    b: Fn
    c: Fn
    f: Fn
    u_left: float
    u_right: float
    nu: float
    exact: Fn | None = None


def convection_diffusion(nu: float) -> LinearBVP1D:
    """``u' - nu u'' = 0``, ``u(0) = 0``, ``u(1) = 1``; ``u = (e^{x/nu} - 1)/(e^{1/nu} - 1)`` (paper eq. 1-2)."""
    def exact(x):
        x = np.asarray(x, dtype=np.float64)
        return np.expm1(x / nu) / np.expm1(1.0 / nu) if 1.0 / nu < 700 else np.exp((x - 1.0) / nu) * (x > 0)
    return LinearBVP1D(_const(-nu), _const(1.0), _const(0.0), _const(0.0), 0.0, 1.0, nu, exact)


def twin_boundary_layer(nu: float) -> LinearBVP1D:
    """``2(2x - 1) u' - nu u'' + 4u = 0``, ``u(0) = u(1) = 1``; ``u = exp(-2x(1 - x)/nu)`` (paper Sec. 4.4)."""
    return LinearBVP1D(_const(-nu), lambda x: 2 * (2 * np.asarray(x) - 1), _const(4.0), _const(0.0), 1.0, 1.0, nu,
                       lambda x: np.exp(-2 * np.asarray(x) * (1 - np.asarray(x)) / nu))


@dataclass(frozen=True)
class GatedLayout:
    centers: np.ndarray
    widths: np.ndarray
    collocation: np.ndarray
    splits: tuple[float, ...]
    eps_b: tuple[float, ...]


def _endpoint_grid(lo: float, hi: float, n: int) -> np.ndarray:
    return lo + (np.arange(n) / n) * (hi - lo)


def gated_layout(splits: Sequence[float], eps_scales: Sequence[float], nu: float, n_centers: int,
                 n_colloc: int | None = None, k: float = 1.5) -> GatedLayout:
    """Centers, widths and collocation points of the soft split (per block ``n_centers`` centers and ``n_colloc``
    collocation points). With one split this is the paper's two-block layout; with several, each split blends
    the widths of its two neighbouring blocks by its own logistic gate."""
    edges = [0.0, *[float(s) for s in splits], 1.0]
    if any(b <= a for a, b in zip(edges[:-1], edges[1:], strict=True)):
        raise ValueError(f"splits must be increasing inside (0, 1), got {splits}")
    if len(eps_scales) != len(splits):
        raise ValueError("one eps_scale per split")
    n_colloc = n_colloc or n_centers
    d = [(b - a) / n_centers for a, b in zip(edges[:-1], edges[1:], strict=True)]
    centers = np.concatenate([_endpoint_grid(a, b, n_centers) for a, b in zip(edges[:-1], edges[1:], strict=True)])
    colloc = np.concatenate([_endpoint_grid(a, b, n_colloc) for a, b in zip(edges[:-1], edges[1:], strict=True)])
    sig = [k * dj for dj in d]
    widths = np.full_like(centers, sig[0])
    eps_b = []
    for j, (xs, es) in enumerate(zip(splits, eps_scales, strict=True)):
        eb = max(es * min(d[j], d[j + 1]), 5.0 * nu)
        eps_b.append(eb)
        s = 0.5 * (1.0 + np.tanh(0.5 * (centers - xs) / eb))          # logistic, overflow-free
        widths = widths + s * (sig[j + 1] - sig[j])
    return GatedLayout(centers, widths, colloc, tuple(float(s) for s in splits), tuple(eps_b))


def _features(x: np.ndarray, lay: GatedLayout):
    """psi, psi', psi'' of every feature at x: (N, M) each."""
    m = 1.0 / (math.sqrt(2.0) * lay.widths)
    b = -m * lay.centers
    z = np.outer(x, m) + b
    phi = np.exp(-z * z)
    phi0, phi1 = np.exp(-b * b), np.exp(-(m + b) ** 2)
    x = x[:, None]
    psi = phi - (1 - x) * phi0 - x * phi1
    dpsi = m * (-2 * z * phi) + phi0 - phi1
    d2psi = m * m * (4 * z * z - 2) * phi
    return psi, dpsi, d2psi


def _operator_rows(p: LinearBVP1D, x: np.ndarray, lay: GatedLayout):
    """R (N, M) and r0 (N,) with residual = R c + r0."""
    psi, dpsi, d2psi = _features(x, lay)
    a, bb, cc = p.a(x)[:, None], p.b(x)[:, None], p.c(x)[:, None]
    R = a * d2psi + bb * dpsi + cc * psi
    g = (1 - x) * p.u_left + x * p.u_right
    r0 = p.b(x) * (p.u_right - p.u_left) + p.c(x) * g - p.f(x)
    return R, r0


@dataclass
class GatedSolution:
    problem: LinearBVP1D
    layout: GatedLayout
    coef: np.ndarray
    solve_seconds: float

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        psi, _, _ = _features(x, self.layout)
        return (1 - x) * self.problem.u_left + x * self.problem.u_right + psi @ self.coef

    def residual(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        R, r0 = _operator_rows(self.problem, x, self.layout)
        return R @ self.coef + r0


def solve_forward(problem: LinearBVP1D, layout: GatedLayout, lam: float = 1e-12) -> GatedSolution:
    """``argmin (1/N) ||R c + r0||^2 + lam ||c||^2`` at the layout's collocation points (stacked least squares)."""
    t0 = time.perf_counter()
    R, r0 = _operator_rows(problem, layout.collocation, layout)
    n, M = R.shape
    A = np.vstack([R / math.sqrt(n), math.sqrt(lam) * np.eye(M)])
    rhs = np.concatenate([-r0 / math.sqrt(n), np.zeros(M)])
    coef = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return GatedSolution(problem, layout, coef, time.perf_counter() - t0)


def validation_energy(sol: GatedSolution, x_val) -> float:
    """Mean squared strong-form residual on the validation points."""
    r = sol.residual(x_val)
    return float(np.mean(r * r)) if np.all(np.isfinite(r)) else math.inf


def search_gates_nested(problem: LinearBVP1D, x_val, *, n_centers: int, n_colloc: int | None = None,
                        k: float = 1.5, lam: float = 1e-12, xs_bounds=(0.80, 0.999), eps_bounds=(10.0, 100.0),
                        xs_tol: float = 1e-4, eps_tol: float = 1e-1) -> dict:
    """One split: outer bounded Brent search over ``x_s``, inner over ``eps_scale``, minimising the validation
    residual energy (the paper's nested ``fminbnd``). Returns the best solution and the search record."""
    t0 = time.perf_counter()
    n_eval = [0]
    best: dict = {"J": math.inf}

    def J(xs, es):
        n_eval[0] += 1
        sol = solve_forward(problem, gated_layout([xs], [es], problem.nu, n_centers, n_colloc, k), lam)
        val = validation_energy(sol, x_val)
        if val < best["J"]:
            best.update(J=val, sol=sol, xs=xs, eps_scale=es)
        return math.log10(val) if val > 0 else -300.0

    def inner(xs):
        r = minimize_scalar(lambda es: J(xs, es), bounds=eps_bounds, method="bounded",
                            options={"xatol": eps_tol})
        return r.fun

    minimize_scalar(inner, bounds=xs_bounds, method="bounded", options={"xatol": xs_tol})
    return {**best, "n_evaluations": n_eval[0], "seconds": time.perf_counter() - t0}


def log_evidence(Phi: np.ndarray, y: np.ndarray, *, eta0: float = 1e-7, eta_bounds=(1e-12, 1e-2),
                 max_iter: int = 50, damping: float = 0.5) -> dict:
    """Bayesian linear model ``y = Phi c + e``, ``e ~ N(0, I)`` (whitened), ``c ~ N(0, eta^-1 I)``: the prior
    precision ``eta`` by the evidence fixed point ``gamma = sum k/(k + eta)``, ``eta <- gamma / ||m||^2``
    (damped in log space, clipped), then the log-evidence, posterior mean and the eigen-decomposition."""
    N, M = Phi.shape
    kappa, V = np.linalg.eigh(Phi.T @ Phi)
    kappa = np.clip(kappa, 0.0, None)
    Vty = V.T @ (Phi.T @ y)
    eta = eta0
    lo, hi = eta_bounds
    for _ in range(max_iter):
        mN_v = Vty / (kappa + eta)
        gamma = float(np.sum(kappa / (kappa + eta)))
        target = min(max(gamma / max(float(mN_v @ mN_v), 1e-300), lo), hi)
        new = math.exp((1 - damping) * math.log(eta) + damping * math.log(target))
        if abs(math.log(new) - math.log(eta)) < 1e-6:
            eta = new
            break
        eta = new
    mN = V @ (Vty / (kappa + eta))
    resid = y - Phi @ mN
    logdetA = float(np.sum(np.log(kappa + eta)))
    ev = 0.5 * M * math.log(eta) - 0.5 * (float(resid @ resid) + eta * float(mN @ mN)) - 0.5 * logdetA \
        - 0.5 * N * math.log(2 * math.pi)
    return {"log_evidence": ev, "eta": eta, "mean": mN, "V": V, "kappa": kappa}


@dataclass
class InverseFit:
    nu: float
    splits: tuple[float, ...]
    eps_scales: tuple[float, ...]
    log_evidence: float
    eta: float
    solution: GatedSolution
    history: list
    seconds: float

    def predictive_std(self, x) -> np.ndarray:
        """Model standard deviation ``sqrt(h^T A^-1 h)`` of the posterior mean (whitened units)."""
        psi, _, _ = _features(np.asarray(x, dtype=np.float64), self.solution.layout)
        V, k = self._post["V"], self._post["kappa"]
        P = psi @ V
        return np.sqrt(np.sum(P * P / (k + self.eta), axis=1))


def fit_inverse(make_problem: Callable[[float], LinearBVP1D], x_data, y_data, *, sigma: float, beta_pde: float,
                n_centers: int, n_colloc: int, k: float = 1.5, log10_nu_bounds=(-3.0, 1.0),
                xs_bounds=(0.85, 0.995), eps_bounds=(10.0, 100.0), n_eval: int = 30, n_init: int = 5,
                seed: int = 0) -> InverseFit:
    """Infer ``nu`` (and the gate) by maximising the evidence of the stacked whitened data + physics system over
    ``(log10 nu, x_s, eps_scale)`` with the library's Bayesian optimiser (GP + expected improvement)."""
    from pinneapple_design.design_optimizer.optimizer import (
        BayesianDesignOptimizer,
        DesignOptimizerConfig,
    )

    t0 = time.perf_counter()
    x_data = np.asarray(x_data, dtype=np.float64)
    y_data = np.asarray(y_data, dtype=np.float64)
    bounds = np.array([log10_nu_bounds, xs_bounds, eps_bounds], dtype=np.float64)
    opt = BayesianDesignOptimizer(DesignOptimizerConfig(method="bayesian", n_initial_random=n_init), seed=seed)
    unit = np.c_[np.zeros(3), np.ones(3)]
    history = []
    best = None
    for _ in range(n_eval):
        u = opt.propose(unit)
        lnu, xs, es = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
        nu = 10.0 ** lnu
        p = make_problem(nu)
        lay = gated_layout([xs], [es], nu, n_centers, n_colloc, k)
        psi, _, _ = _features(x_data, lay)
        g = (1 - x_data) * p.u_left + x_data * p.u_right
        R, r0 = _operator_rows(p, lay.collocation, lay)
        Phi = np.vstack([psi / sigma, math.sqrt(beta_pde) * R])
        y = np.concatenate([(y_data - g) / sigma, -math.sqrt(beta_pde) * r0])
        post = log_evidence(Phi, y)
        ev = post["log_evidence"] if math.isfinite(post["log_evidence"]) else -1e300
        history.append({"nu": nu, "xs": float(xs), "eps_scale": float(es), "log_evidence": ev, "eta": post["eta"]})
        opt.update(np.asarray(u, dtype=np.float64), -ev)
        if best is None or ev > best[0]:
            best = (ev, nu, xs, es, post, GatedSolution(p, lay, post["mean"], 0.0))
    ev, nu, xs, es, post, sol = best
    fit = InverseFit(nu, (float(xs),), (float(es),), ev, post["eta"], sol, history, time.perf_counter() - t0)
    fit._post = post
    return fit


def search_gates_bo(problem: LinearBVP1D, x_val, *, split_bounds: Sequence[tuple[float, float]],
                    eps_bounds=(10.0, 100.0), n_centers: int, n_colloc: int | None = None, k: float = 1.5,
                    lam: float = 1e-12, n_eval: int = 60, n_init: int = 5, seed: int = 0) -> dict:
    """Several splits: minimise the validation residual energy (log10) over all splits and their eps_scales with
    the library's Bayesian optimiser (the paper's choice for the multi-split twin boundary layer)."""
    from pinneapple_design.design_optimizer.optimizer import (
        BayesianDesignOptimizer,
        DesignOptimizerConfig,
    )

    t0 = time.perf_counter()
    ns = len(split_bounds)
    bounds = np.array(list(split_bounds) + [eps_bounds] * ns, dtype=np.float64)
    opt = BayesianDesignOptimizer(DesignOptimizerConfig(method="bayesian", n_initial_random=n_init), seed=seed)
    unit = np.c_[np.zeros(2 * ns), np.ones(2 * ns)]
    best: dict = {"J": math.inf}
    history = []
    for _ in range(n_eval):
        u = opt.propose(unit)
        v = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
        splits, eps = sorted(v[:ns]), v[ns:]
        sol = solve_forward(problem, gated_layout(splits, eps, problem.nu, n_centers, n_colloc, k), lam)
        val = validation_energy(sol, x_val)
        history.append({"splits": [float(s) for s in splits], "eps_scales": [float(e) for e in eps], "J": val})
        opt.update(np.asarray(u, dtype=np.float64), math.log10(val) if 0 < val < math.inf else 300.0)
        if val < best["J"]:
            best.update(J=val, sol=sol, splits=[float(s) for s in splits], eps_scales=[float(e) for e in eps])
    return {**best, "history": history, "n_evaluations": n_eval, "seconds": time.perf_counter() - t0}
