r"""Fisher information, Cramér-Rao bounds and experimental design.

The Fisher information of a parameter vector :math:`\theta` is :math:`I(\theta) = E[\nabla\log p\,\nabla\log p^T]`.
Under the regularity conditions (a likelihood differentiable in :math:`\theta`, whose support does not depend on
:math:`\theta`, and a nonsingular :math:`I`), every **unbiased** estimator satisfies the Cramér-Rao bound
:math:`\operatorname{Cov}(\hat\theta) \succeq I(\theta)^{-1}`, and for a function :math:`g(\theta)` (a prediction,
a first-passage time) :math:`\operatorname{Var}(\hat g) \ge \nabla g^T I^{-1} \nabla g`. This module keeps three
quantities apart and labels each:

* the **theoretical bound** (:class:`CramerRaoBound`): a property of the model and the design, not of any estimator;
* the **empirical performance** of an estimator (:func:`estimator_performance`): least-squares fits of simulated
  data, whose spread may exceed the bound (finite samples, non-linearity) or fall below it (bias, bounds);
* the **Bayesian** (Laplace) posterior covariance :math:`(I + P)^{-1}` with a prior precision :math:`P`, which is
  not a Cramér-Rao bound (a prior makes the estimator biased).

How the information is computed (:class:`FisherInformation`):

* ``gaussian``: :math:`I = J^T \Sigma^{-1} J` for independent Gaussian noise, with :math:`J = \partial y/\partial\theta`
  from :func:`pinneapple_analysis.inverse_problems.forward.jacobian` (the model's own sensitivities, e.g. the forward
  sensitivity equations of :class:`~pinneapple_analysis.inverse_problems.ode_model.ODEForwardModel`, else PyTorch
  autograd through ``predict_torch``, else central differences); with a parameter-dependent noise level
  :math:`\sigma_i(\theta)` the term :math:`2\,\partial\sigma_i\,\partial\sigma_i^T / \sigma_i^2` is added;
* ``poisson``: :math:`I = J^T \operatorname{diag}(1/\lambda) J` for counts with mean :math:`\lambda(\theta)`;
* ``from_loglik``: the expectation of the score outer product, by Monte Carlo over simulated data, for any
  likelihood; the score from central differences, PyTorch or JAX (optional backends, detected at call time);
* ``linear_gaussian``: :math:`X^T X / \sigma^2`, exact.

Design (:class:`ExperimentalDesign`, :class:`ObservationScheduler`): choose measurements from candidates to optimise a
criterion of the total information (with the information already in hand and an optional prior): D (maximise
:math:`\log\det I`), A (minimise :math:`\operatorname{tr} I^{-1}`), E (maximise :math:`\lambda_{\min}`), c (minimise
the bound on a target :math:`g`), Bayesian D (D with the prior precision). Greedy selection followed by exchange
passes (Fedorov 1972); replicates allowed. :class:`InformationGain` scores one more measurement;
:class:`SamplingStrategy` compares whole schedules; :class:`ParameterIdentifiability` gives the local verdict from the
information matrix and delegates the full analysis (profile likelihood, posterior) to the existing
:func:`~pinneapple_analysis.inverse_problems.identifiability.analyze_identifiability`.

Limits: everything is local, at one parameter point (the design is optimal for the nominal parameters: a locally
optimal design); the bound is attained only asymptotically or for linear-Gaussian models; a rank-deficient
information matrix means non-identifiable directions at that point, not a proof of global non-identifiability.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..inverse_problems.forward import ForwardModel, Observations, Parameter, ParameterSet, jacobian
from ._backends import require, to_numpy
from ._contract import Assumption, Diagnostic, DynamicsResult, MethodCard, make_rng, reproducibility

__all__ = [
    "FisherInformationMatrix", "FisherInformation", "CramerRaoBound", "ParameterIdentifiability",
    "SensitivityAnalyzer", "ExperimentalDesign", "ObservationScheduler", "InformationGain", "SamplingStrategy",
    "estimator_performance", "REGULARITY", "CARD_FISHER", "CARD_DESIGN",
]

REGULARITY = [
    Assumption("differentiable likelihood", "log p(y | theta) is differentiable in theta at the analysis point"),
    Assumption("fixed support", "the support of p(y | theta) does not depend on theta"),
    Assumption("unbiased estimator", "the bound applies to unbiased estimators; regularised, bounded or Bayesian "
               "estimators can have a smaller variance"),
    Assumption("correct model", "the model and its noise distribution generate the data"),
]


# =========================================================================== information matrix
@dataclass
class FisherInformationMatrix:
    """``F`` (``p x p``) in the physical parameters ``names`` at ``theta``. ``how`` says how the derivatives were
    obtained, ``kind`` which likelihood (``gaussian``, ``poisson``, ``score-mc``, ``linear``), ``n_obs`` how many
    measurements it contains. ``mc_se`` holds the Monte Carlo standard error of each entry when ``F`` was
    estimated by sampling."""
    F: np.ndarray
    names: list[str]
    theta: dict[str, float]
    kind: str
    how: str = ""
    n_obs: int = 0
    mc_se: np.ndarray | None = None
    prior_precision: np.ndarray | None = None

    def __post_init__(self):
        self.F = np.asarray(self.F, dtype=np.float64)
        if self.F.shape != (len(self.names), len(self.names)):
            raise ValueError(f"F has shape {self.F.shape} for {len(self.names)} parameters")
        self.F = 0.5 * (self.F + self.F.T)

    @property
    def p(self) -> int:
        return len(self.names)

    @property
    def total(self) -> np.ndarray:
        """``F`` plus the prior precision when one is attached."""
        return self.F if self.prior_precision is None else self.F + self.prior_precision

    def eig(self) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(self.total)

    def rank(self, rtol: float = 1e-10) -> int:
        w = np.linalg.eigvalsh(self.total)
        return int(np.sum(w > rtol * max(float(w.max(initial=0.0)), 1e-300)))

    @property
    def condition(self) -> float:
        w = np.linalg.eigvalsh(self.total)
        return float(w.max() / w.min()) if w.min() > 0 else math.inf

    def null_directions(self, rtol: float = 1e-8) -> list[dict[str, float]]:
        """Parameter combinations (unit eigenvectors) that the data do not inform at this point."""
        w, V = self.eig()
        top = max(float(w.max(initial=0.0)), 1e-300)
        out = []
        for k in np.nonzero(w <= rtol * top)[0]:
            v = V[:, k]
            v = v * np.sign(v[np.argmax(np.abs(v))])
            out.append({n: float(c) for n, c in zip(self.names, v, strict=True) if abs(c) > 1e-6})
        return out

    def criteria(self, target_gradient: np.ndarray | None = None) -> dict[str, float]:
        """D (``log det F``), A (``tr F^-1``), E (smallest eigenvalue), condition number and, with a target
        gradient, c (``g^T F^-1 g``). Singular matrices give ``-inf`` / ``inf``."""
        w = np.linalg.eigvalsh(self.total)
        sing = w.min() <= 1e-12 * max(float(w.max(initial=0.0)), 1e-300)
        out = {"D": -math.inf if sing else float(np.sum(np.log(w))),
               "A": math.inf if sing else float(np.sum(1.0 / w)), "E": float(w.min()), "condition": self.condition}
        if target_gradient is not None:
            out["c"] = _c_variance(self.total, np.asarray(target_gradient, dtype=float))
        return out

    def __add__(self, other: FisherInformationMatrix) -> FisherInformationMatrix:
        """Information of two independent experiments on the same parameters adds."""
        if other.names != self.names:
            raise ValueError(f"parameters differ: {self.names} vs {other.names}")
        return FisherInformationMatrix(self.F + other.F, self.names, self.theta, f"{self.kind}+{other.kind}",
                                       self.how, self.n_obs + other.n_obs, prior_precision=self.prior_precision)

    def with_prior(self, precision) -> FisherInformationMatrix:
        """The same information with a Gaussian prior precision (``p x p`` or one value per parameter)."""
        P = np.asarray(precision, dtype=float)
        P = np.diag(P) if P.ndim == 1 else P
        return FisherInformationMatrix(self.F, self.names, self.theta, self.kind, self.how, self.n_obs, self.mc_se, P)

    def crlb(self, target: Callable[[Mapping[str, float]], float] | Mapping[str, Callable] | None = None,
             rel_step: float = 1e-5) -> CramerRaoBound:
        return CramerRaoBound.from_fim(self, target, rel_step=rel_step)


@dataclass
class CramerRaoBound:
    """Lower bound on the covariance of unbiased estimators: ``cov = F^-1``, ``std`` per parameter, ``correlation``
    and, for targets ``g``, ``target_std`` from ``grad g^T F^-1 grad g``. With a prior attached this is the Laplace
    posterior covariance instead (``bayesian = True``), which is not a Cramér-Rao bound."""
    names: list[str]
    theta: dict[str, float]
    cov: np.ndarray
    std: dict[str, float]
    correlation: np.ndarray
    singular: bool
    target_std: dict[str, float] = field(default_factory=dict)
    target_value: dict[str, float] = field(default_factory=dict)
    target_gradient: dict[str, np.ndarray] = field(default_factory=dict)
    bayesian: bool = False

    @property
    def kind(self) -> str:
        return "Laplace posterior covariance (with prior)" if self.bayesian else \
            "Cramér-Rao lower bound for unbiased estimators"

    def relative_std(self) -> dict[str, float]:
        return {n: (s / abs(self.theta[n]) if self.theta[n] != 0 else math.inf) for n, s in self.std.items()}

    @classmethod
    def from_fim(cls, fim: FisherInformationMatrix, target=None, rel_step: float = 1e-5) -> CramerRaoBound:
        Ft = fim.total
        w = np.linalg.eigvalsh(Ft)
        singular = bool(w.min() <= 1e-12 * max(float(w.max(initial=0.0)), 1e-300))
        cov = np.full_like(Ft, np.inf) if singular else np.linalg.inv(Ft)
        if singular:                                   # finite only along informed directions: report inf per param
            std = {n: math.inf for n in fim.names}
            corr = np.full_like(Ft, np.nan)
        else:
            sd = np.sqrt(np.diag(cov))
            std = dict(zip(fim.names, sd.tolist(), strict=True))
            corr = cov / np.outer(sd, sd)
        out = cls(fim.names, dict(fim.theta), cov, std, corr, singular, bayesian=fim.prior_precision is not None)
        if target is not None:
            targets = target if isinstance(target, Mapping) else {"target": target}
            for name, g in targets.items():
                val, grad = _gradient(g, fim.theta, fim.names, rel_step)
                out.target_value[name] = val
                out.target_gradient[name] = grad
                var = _c_variance(Ft, grad) if singular else float(grad @ cov @ grad)
                out.target_std[name] = float(math.sqrt(max(var, 0.0))) if np.isfinite(var) else math.inf
        return out


def _c_variance(F: np.ndarray, g: np.ndarray, rtol: float = 1e-9) -> float:
    """``g^T F^+ g`` when ``g`` lies in the range of ``F`` (the target is estimable even if ``F`` is singular: a
    c-optimal design is often singular), else ``inf``."""
    w, V = np.linalg.eigh(F)
    keep = w > rtol * max(float(w.max(initial=0.0)), 1e-300)
    coef = V.T @ g
    if np.linalg.norm(coef[~keep]) > 1e-6 * max(np.linalg.norm(g), 1e-300):
        return math.inf
    return float(np.sum(coef[keep] ** 2 / w[keep]))


def _gradient(g: Callable[[Mapping[str, float]], float], theta: Mapping[str, float], names: Sequence[str],
              rel_step: float) -> tuple[float, np.ndarray]:
    """Value and central-difference gradient of a scalar target in the physical parameters."""
    base = dict(theta)
    val = float(g(base))
    grad = np.zeros(len(names))
    for i, n in enumerate(names):
        h = rel_step * max(abs(base[n]), 1e-8)
        up, dn = dict(base), dict(base)
        up[n] += h
        dn[n] -= h
        grad[i] = (float(g(up)) - float(g(dn))) / (2 * h)
    return val, grad


# =========================================================================== computing the information
def _params(model: ForwardModel, parameters, theta) -> ParameterSet:
    ps = list(parameters) if parameters is not None else model.default_parameters()
    if theta is not None:
        ps = [Parameter(p.name, float(theta.get(p.name, p.value)), p.lower, p.upper, p.unit, p.log, p.fixed,
                        p.description, p.truth) for p in ps]
    return ParameterSet(ps)


def _jac_theta(model: ForwardModel, ps: ParameterSet, obs: Observations, method: str
               ) -> tuple[np.ndarray, np.ndarray, str]:
    """Predictions and ``d pred / d theta`` (physical parameters) at the nominal point."""
    z = ps.z0()
    pred, Jz, how = jacobian(model, ps, z, obs, method=method)
    theta = ps.theta(z)
    dtdz = np.array([theta[p.name] if p.log else p.scale for p in ps.free])
    return pred, Jz / dtdz[None, :], how


class FisherInformation:
    """Constructors of :class:`FisherInformationMatrix`; see the module docstring."""

    @staticmethod
    def gaussian(model: ForwardModel, obs: Observations, parameters: Sequence[Parameter] | None = None, *,
                 theta: Mapping[str, float] | None = None, method: str = "auto",
                 sigma_model: Callable[[Mapping[str, float], Observations], np.ndarray] | None = None,
                 rel_step: float = 1e-5) -> FisherInformationMatrix:
        """``J^T diag(1/sigma^2) J`` at ``theta`` (default: the parameters' values). ``sigma_model(theta, obs)``
        makes the noise level depend on the parameters (e.g. a relative error) and adds its information."""
        ps = _params(model, parameters, theta)
        names = ps.names
        th = ps.theta()
        pred, J, how = _jac_theta(model, ps, obs, method)
        sig = obs.sigma if sigma_model is None else np.asarray(sigma_model(th, obs), dtype=float)
        Jw = J / sig[:, None]
        F = Jw.T @ Jw
        kind = "gaussian"
        if sigma_model is not None:
            dS = np.zeros_like(J)
            for i, n in enumerate(names):
                h = rel_step * max(abs(th[n]), 1e-8)
                up, dn = dict(th), dict(th)
                up[n] += h
                dn[n] -= h
                dS[:, i] = (np.asarray(sigma_model(up, obs)) - np.asarray(sigma_model(dn, obs))) / (2 * h)
            dSw = dS / sig[:, None]
            F = F + 2.0 * dSw.T @ dSw
            kind = "gaussian (parameter-dependent noise)"
        return FisherInformationMatrix(F, names, th, kind, how, obs.n)

    @staticmethod
    def poisson(model: ForwardModel, obs: Observations, parameters: Sequence[Parameter] | None = None, *,
                theta: Mapping[str, float] | None = None, method: str = "auto") -> FisherInformationMatrix:
        """Counts ``y_i ~ Poisson(lambda_i(theta))`` with ``lambda`` the model's prediction: ``J^T diag(1/lambda) J``.
        ``obs.sigma`` is ignored."""
        ps = _params(model, parameters, theta)
        lam, J, how = _jac_theta(model, ps, obs, method)
        if np.any(lam <= 0):
            raise ValueError("Poisson rates must be positive at the analysis point")
        return FisherInformationMatrix(J.T @ (J / lam[:, None]), ps.names, ps.theta(), "poisson", how, obs.n)

    @staticmethod
    def linear_gaussian(X, sigma: float | np.ndarray, names: Sequence[str] | None = None,
                        theta: Mapping[str, float] | None = None) -> FisherInformationMatrix:
        """``y = X beta + e``, ``e ~ N(0, sigma^2)``: ``X^T X / sigma^2`` (exact, independent of ``beta``)."""
        X = np.asarray(X, dtype=float)
        s = np.broadcast_to(np.asarray(sigma, dtype=float), (X.shape[0],))
        names = list(names or [f"b{i}" for i in range(X.shape[1])])
        Xw = X / s[:, None]
        return FisherInformationMatrix(Xw.T @ Xw, names, dict(theta or {n: 0.0 for n in names}), "linear", "exact",
                                       X.shape[0])

    @staticmethod
    def from_loglik(loglik: Callable, theta: Mapping[str, float], simulate: Callable[[Mapping[str, float],
                    np.random.Generator], Any], *, n_mc: int = 2000, seed: int | None = 0, backend: str = "fd",
                    rel_step: float = 1e-5) -> FisherInformationMatrix:
        """``E[s s^T]`` by Monte Carlo: ``simulate(theta, rng)`` draws a data set, ``loglik(theta_vector, y)`` is its
        log-likelihood. The score ``s`` is computed by ``backend``: ``"fd"`` (central differences, NumPy),
        ``"torch"`` (``loglik`` written with PyTorch operations on a tensor) or ``"jax"`` (with ``jax.numpy``).
        The Monte Carlo standard error of each entry is returned in ``mc_se``."""
        names = list(theta)
        th = np.array([float(theta[n]) for n in names])
        rng, seed = make_rng(seed)
        score = _score_function(loglik, backend, rel_step)
        S = np.empty((n_mc, th.size))
        for k in range(n_mc):
            S[k] = score(th, simulate(dict(theta), rng))
        outer = S[:, :, None] * S[:, None, :]
        F = outer.mean(axis=0)
        se = outer.std(axis=0, ddof=1) / math.sqrt(n_mc)
        return FisherInformationMatrix(F, names, dict(theta), "score-mc", f"score by {backend}, {n_mc} samples",
                                       0, mc_se=se)


def _score_function(loglik: Callable, backend: str, rel_step: float) -> Callable[[np.ndarray, Any], np.ndarray]:
    if backend == "fd":
        def score(th, y):
            g = np.zeros_like(th)
            for i in range(th.size):
                h = rel_step * max(abs(th[i]), 1e-8)
                up, dn = th.copy(), th.copy()
                up[i] += h
                dn[i] -= h
                g[i] = (loglik(up, y) - loglik(dn, y)) / (2 * h)
            return g
        return score
    if backend == "torch":
        torch = require("torch", "the PyTorch score")

        def score(th, y):
            t = torch.tensor(th, dtype=torch.float64, requires_grad=True)
            yy = torch.as_tensor(np.asarray(y), dtype=torch.float64)
            (g,) = torch.autograd.grad(loglik(t, yy), t)
            return to_numpy(g)
        return score
    if backend == "jax":
        jax = require("jax", "the JAX score")
        jax.config.update("jax_enable_x64", True)
        grad = jax.grad(loglik)

        def score(th, y):
            return to_numpy(grad(jax.numpy.asarray(th), jax.numpy.asarray(np.asarray(y))))
        return score
    raise ValueError("backend must be 'fd', 'torch' or 'jax'")


# =========================================================================== identifiability and sensitivity
CARD_FISHER = MethodCard(
    "fisher.identifiability", "experimental_design",
    inputs={"model": "ForwardModel (predict, optional sensitivities / predict_torch)",
            "observations": "measurement points with noise levels (values not needed)",
            "parameters": "Parameter list (analysis point = their values)"},
    assumptions=REGULARITY,
    config={"rtol": "relative eigenvalue threshold for the rank", "practical_threshold": "largest acceptable "
            "relative Cramér-Rao standard deviation"},
    outputs={"rank": "numerical rank of F", "null_directions": "uninformed parameter combinations",
             "crlb": "Cramér-Rao standard deviation per parameter", "status": "identifiable / practically / "
             "structurally non-identifiable per parameter"},
    failure_modes=["F singular: some combination is structurally non-identifiable at this point",
                   "local only: another parameter point can behave differently",
                   "the bound is not attained by non-linear estimators with few data"],
    references=["Rao 1945, Bull. Calcutta Math. Soc. 37:81", "Cramér 1946, Mathematical Methods of Statistics",
                "Raue et al. 2009, Bioinformatics 25:1923 (structural vs practical)"])


class ParameterIdentifiability:
    """Local identifiability verdict from the information matrix (:meth:`local`), or the full analysis with profile
    likelihood and posterior delegated to the existing
    :func:`~pinneapple_analysis.inverse_problems.identifiability.analyze_identifiability` (:meth:`full`)."""

    card = CARD_FISHER

    def __init__(self, model: ForwardModel, obs: Observations, parameters: Sequence[Parameter] | None = None, *,
                 method: str = "auto"):
        self.model, self.obs, self.method = model, obs, method
        self.parameters = list(parameters) if parameters is not None else model.default_parameters()

    def local(self, *, rtol: float = 1e-8, practical_threshold: float = 0.5) -> DynamicsResult:
        t0 = time.perf_counter()
        fim = FisherInformation.gaussian(self.model, self.obs, self.parameters, method=self.method)
        crlb = fim.crlb()
        rank = fim.rank(rtol)
        nulls = fim.null_directions(rtol)
        in_null = {n for d in nulls for n in d}
        rel = crlb.relative_std() if not crlb.singular else {}
        status = {}
        for n in fim.names:
            if n in in_null:
                status[n] = "structurally non-identifiable"
            elif crlb.singular:
                status[n] = "identifiable"
            else:
                status[n] = "identifiable" if rel[n] <= practical_threshold else "practically non-identifiable"
        diags = [Diagnostic("rank", rank, "ok" if rank == fim.p else "fail", threshold=fim.p),
                 Diagnostic("condition", fim.condition, "ok" if fim.condition < 1e8 else "warn", threshold=1e8)]
        cfg = {"rtol": rtol, "practical_threshold": practical_threshold, "derivatives": fim.how}
        rep = reproducibility(CARD_FISHER.name, cfg, [fim.F], wall_time_s=time.perf_counter() - t0)
        return DynamicsResult(CARD_FISHER.name, {"rank": rank, "status": status, "null_directions": nulls},
                              {"crlb_std": crlb.std, "crlb_relative_std": rel, "kind": crlb.kind},
                              list(REGULARITY), cfg, diags, {},
                              limits=["local, at the analysis point", "a structural verdict here is numerical "
                                      "(eigenvalue threshold), not a symbolic proof"], reproducibility=rep,
                              details={"fim": fim, "crlb": crlb})

    def full(self, **kwargs) -> DynamicsResult:
        """``analyze_identifiability`` (Fisher, sensitivities, profile likelihood, posterior, D-optimal suggestions)
        wrapped in the contract; keyword arguments are passed through."""
        from ..inverse_problems.identifiability import analyze_identifiability

        t0 = time.perf_counter()
        rep_ = analyze_identifiability(self.model, self.obs, self.parameters, **kwargs)
        status = {p["name"]: p["status"] for p in rep_.parameters}
        diags = [Diagnostic("rank", rep_.rank, "ok" if rep_.rank == rep_.n_parameters else "fail",
                            threshold=rep_.n_parameters),
                 Diagnostic("condition", rep_.condition_number, "info")]
        cfg = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}
        rep = reproducibility("fisher.identifiability_full", cfg, [], wall_time_s=time.perf_counter() - t0)
        return DynamicsResult("fisher.identifiability_full", {"status": status, "summary": rep_.summary,
                                                              "rank": rep_.rank},
                              {"kind": "Cramér-Rao, profile likelihood and posterior (see details)"},
                              list(REGULARITY), cfg, diags, {}, limits=list(rep_.notes), reproducibility=rep,
                              details={"report": rep_})


class SensitivityAnalyzer:
    """Local sensitivities of the predictions: raw ``dy/dtheta``, relative ``(theta/y) dy/dtheta`` and noise-scaled
    ``(1/sigma) dy/dtheta`` (the rows of the information matrix), and the sensitivity of a target ``g``. Global
    (Sobol, Morris) analysis lives in :mod:`pinneapple_analysis.sensitivity`."""

    def __init__(self, model: ForwardModel, parameters: Sequence[Parameter] | None = None, *, method: str = "auto"):
        self.model, self.method = model, method
        self.parameters = list(parameters) if parameters is not None else model.default_parameters()

    def local(self, obs: Observations) -> dict[str, Any]:
        ps = _params(self.model, self.parameters, None)
        pred, J, how = _jac_theta(self.model, ps, obs, self.method)
        th = np.array([ps.theta()[n] for n in ps.names])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = J * th[None, :] / pred[:, None]
        return {"names": ps.names, "prediction": pred, "dy_dtheta": J, "relative": rel,
                "noise_scaled": J / obs.sigma[:, None], "derivatives": how,
                "ranking": sorted(ps.names, key=lambda n: -float(np.linalg.norm(J[:, ps.names.index(n)] /
                                                                                obs.sigma)))}

    def target(self, g: Callable[[Mapping[str, float]], float], rel_step: float = 1e-5) -> dict[str, float]:
        ps = _params(self.model, self.parameters, None)
        val, grad = _gradient(g, ps.theta(), ps.names, rel_step)
        th = ps.theta()
        return {n: float(gr * th[n] / val) if val else math.nan for n, gr in zip(ps.names, grad, strict=True)}


# =========================================================================== experimental design
CARD_DESIGN = MethodCard(
    "fisher.experimental_design", "experimental_design",
    inputs={"model": "ForwardModel", "candidates": "candidate measurements (Observations with sigma, no values)",
            "base": "measurements already planned or made (optional)", "prior_precision": "optional"},
    assumptions=REGULARITY[:2] + [REGULARITY[3], Assumption("local design", "the nominal parameters are close to "
                                                           "the true ones (the design is locally optimal)")],
    config={"criterion": "D, A, E, c (target), bayes-D", "n": "number of measurements to choose",
            "replicates": "allow choosing a candidate more than once", "exchange": "Fedorov exchange passes"},
    outputs={"chosen": "indices (and counts) of the chosen candidates", "criterion": "value of the criterion",
             "crlb": "bounds of the resulting design"},
    failure_modes=["candidates that cannot make F nonsingular (the parameters are not identifiable from any "
                   "of them)", "strongly non-linear models far from the nominal point",
                   "greedy + exchange finds a local optimum of a combinatorial problem (not certified global)"],
    references=["Fedorov 1972, Theory of Optimal Experiments", "Pukelsheim 1993, Optimal Design of Experiments",
                "Box & Lucas 1959, Biometrika 46:77 (exponential decay designs)"])


class ExperimentalDesign:
    """Choose ``n`` measurements from ``candidates`` to optimise a criterion of the total information
    ``F_base + sum of chosen + prior``. See the module docstring."""

    card = CARD_DESIGN

    def __init__(self, model: ForwardModel, candidates: Observations, parameters: Sequence[Parameter] | None = None,
                 *, theta: Mapping[str, float] | None = None, base: Observations | None = None,
                 prior_precision=None, method: str = "auto"):
        self.model, self.candidates, self.method = model, candidates, method
        self.ps = _params(model, parameters, theta)
        self.names = self.ps.names
        self.theta = self.ps.theta()
        _, Jc, self.how = _jac_theta(model, self.ps, candidates, method)
        self.Jw = Jc / candidates.sigma[:, None]
        p = len(self.names)
        self.F0 = np.zeros((p, p))
        if base is not None:
            self.F0 = FisherInformation.gaussian(model, base, list(self.ps.free), method=method).F
        P = None if prior_precision is None else np.asarray(prior_precision, float)
        self.P = np.zeros((p, p)) if P is None else (np.diag(P) if P.ndim == 1 else P)
        self.has_prior = prior_precision is not None

    def _value(self, F: np.ndarray, criterion: str, g: np.ndarray | None) -> float:
        """Larger is better for every criterion (A and c are negated)."""
        Ft = F + self.P
        w = np.linalg.eigvalsh(Ft)
        if criterion == "E":
            return float(w.min())
        if criterion == "c":
            v = _c_variance(Ft, g)
            return -v if np.isfinite(v) else -math.inf
        tiny = 1e-12 * max(float(w.max(initial=0.0)), 1e-300)
        if w.min() <= tiny:
            return -math.inf
        if criterion in ("D", "bayes-D"):
            return float(np.sum(np.log(w)))
        if criterion == "A":
            return -float(np.sum(1.0 / w))
        raise ValueError("criterion must be D, A, E, c or bayes-D")

    def _seeded(self, F: np.ndarray, criterion: str, g) -> float:
        """Criterion with a vanishing ridge, so a greedy step from a singular start can still rank candidates."""
        v = self._value(F, criterion, g)
        if np.isfinite(v):
            return v
        scale = max(float(np.trace(self.Jw.T @ self.Jw)) / max(len(self.names), 1), 1e-300)
        return self._value(F + 1e-9 * scale * np.eye(len(self.names)), criterion, g) - 1e6

    def select(self, n: int, criterion: str = "D", *, target: Callable[[Mapping[str, float]], float] | None = None,
               replicates: bool = True, exchange: bool = True, max_passes: int = 20) -> DynamicsResult:
        t0 = time.perf_counter()
        if criterion == "bayes-D" and not self.has_prior:
            raise ValueError("bayes-D needs prior_precision")
        if criterion == "c" and target is None:
            raise ValueError("criterion 'c' needs a target")
        g = None
        if criterion == "c":
            _, g = _gradient(target, self.theta, self.names, 1e-5)
        m = self.Jw.shape[0]
        if not replicates and n > m:
            raise ValueError(f"{n} distinct measurements requested from {m} candidates")
        outer = np.einsum("ij,ik->ijk", self.Jw, self.Jw)
        chosen: list[int] = []
        F = self.F0.copy()
        for _ in range(n):
            best, best_v = None, -math.inf
            for k in range(m):
                if not replicates and k in chosen:
                    continue
                v = self._seeded(F + outer[k], criterion, g)
                if v > best_v:
                    best, best_v = k, v
            chosen.append(best)
            F = F + outer[best]
        history = [self._value(F, criterion, g)]
        passes = 0
        if exchange:
            improved = True
            while improved and passes < max_passes:
                improved = False
                passes += 1
                for i in range(len(chosen)):
                    F_wo = F - outer[chosen[i]]
                    cur = self._value(F, criterion, g)
                    for k in range(m):
                        if k == chosen[i] or (not replicates and k in chosen):
                            continue
                        v = self._value(F_wo + outer[k], criterion, g)
                        if v > cur + 1e-12 * max(abs(cur), 1.0):
                            chosen[i], F, cur = k, F_wo + outer[k], v
                            F_wo = F - outer[chosen[i]]
                            improved = True
                history.append(self._value(F, criterion, g))
        fim = FisherInformationMatrix(F, self.names, self.theta, "gaussian", self.how, n,
                                      prior_precision=self.P if self.has_prior else None)
        crlb = fim.crlb(target)
        value = self._value(F, criterion, g)
        counts: dict[int, int] = {}
        for k in chosen:
            counts[k] = counts.get(k, 0) + 1
        failures = [] if np.isfinite(value) else ["the chosen measurements leave F singular: the parameters cannot "
                                                  "all be identified from these candidates"]
        diags = [Diagnostic("criterion_value", value, "ok" if np.isfinite(value) else "fail"),
                 Diagnostic("exchange_passes", passes, "info"), Diagnostic("condition", fim.condition, "info")]
        cfg = {"criterion": criterion, "n": n, "replicates": replicates, "exchange": exchange,
               "derivatives": self.how, "theta": self.theta}
        rep = reproducibility(CARD_DESIGN.name, cfg, [self.Jw, self.F0, self.P], wall_time_s=time.perf_counter() - t0)
        inputs = {k: np.asarray(v)[sorted(counts)].tolist() for k, v in self.candidates.inputs.items()}
        est = {"chosen": sorted(counts), "counts": counts, "inputs": inputs,
               "criterion": criterion, "value": value, "criteria": fim.criteria(g)}
        return DynamicsResult(CARD_DESIGN.name, est,
                              {"crlb_std": crlb.std, "target_std": crlb.target_std, "kind": crlb.kind},
                              list(CARD_DESIGN.assumptions), cfg, diags, {"history": history},
                              applicable=not failures, failures=failures,
                              limits=["locally optimal at the nominal parameters",
                                      "greedy + exchange: a local optimum, not a certified global one"],
                              reproducibility=rep, details={"fim": fim, "crlb": crlb})


class ObservationScheduler:
    """When to measure: candidate times on a grid over ``t_range`` for one observable, then
    :class:`ExperimentalDesign`. ``inputs`` holds the other input columns (constant), ``sigma`` the noise level
    (a number or a function of time)."""

    def __init__(self, model: ForwardModel, t_range: tuple[float, float], *, n_grid: int = 201, time_input: str = "t",
                 observable: str | None = None, sigma: float | Callable[[np.ndarray], np.ndarray] = 1.0,
                 inputs: Mapping[str, float] | None = None, parameters: Sequence[Parameter] | None = None,
                 log_grid: bool = False, method: str = "auto"):
        lo, hi = map(float, t_range)
        t = np.geomspace(lo, hi, n_grid) if log_grid else np.linspace(lo, hi, n_grid)
        self.times = t
        cols = {time_input: t, **{k: np.full(n_grid, float(v)) for k, v in (inputs or {}).items()}}
        sig = sigma(t) if callable(sigma) else np.full(n_grid, float(sigma))
        obsv = None if observable is None else np.array([observable] * n_grid, dtype=object)
        self.candidates = Observations(cols, sig, observable=obsv)
        self.model, self.parameters, self.method, self.time_input = model, parameters, method, time_input

    def schedule(self, n: int, criterion: str = "D", *, base: Observations | None = None, prior_precision=None,
                 target=None, replicates: bool = True) -> DynamicsResult:
        d = ExperimentalDesign(self.model, self.candidates, self.parameters, base=base,
                               prior_precision=prior_precision, method=self.method)
        r = d.select(n, criterion, target=target, replicates=replicates)
        r.estimate["times"] = [float(self.times[k]) for k in r.estimate["chosen"] for _ in
                               range(r.estimate["counts"][k])]
        return r

    def next_best(self, base: Observations, criterion: str = "D", target=None) -> dict[str, Any]:
        """The single most informative next measurement time given ``base``."""
        r = self.schedule(1, criterion, base=base, target=target)
        return {"time": r.estimate["times"][0], "value": r.estimate["value"], "result": r}


class InformationGain:
    """Value of more measurements over a current information matrix ``fim``: the gain in ``log det`` (D), the
    expected information gain (mutual information) of a linear-Gaussian posterior
    ``0.5 log det((F + F_new + P) / (F + P))`` (requires a prior or a nonsingular ``F``), and the reduction of the
    bound on a target."""

    def __init__(self, fim: FisherInformationMatrix, model: ForwardModel, parameters=None, method: str = "auto"):
        self.fim, self.model, self.method = fim, model, method
        self.parameters = parameters

    def of(self, new: Observations, target=None) -> dict[str, Any]:
        add = FisherInformation.gaussian(self.model, new, self.parameters, theta=self.fim.theta, method=self.method)
        before = self.fim
        after = FisherInformationMatrix(before.F + add.F, before.names, before.theta, before.kind, before.how,
                                        before.n_obs + add.n_obs, prior_precision=before.prior_precision)
        b, a = before.criteria(), after.criteria()
        out = {"delta_logdet": a["D"] - b["D"] if np.isfinite(b["D"]) else (math.inf if np.isfinite(a["D"]) else
                                                                           math.nan),
               "expected_information_gain_nats": 0.5 * (a["D"] - b["D"]) if np.isfinite(b["D"]) else math.nan,
               "crlb_before": before.crlb().std, "crlb_after": after.crlb().std}
        if target is not None:
            out["target_std_before"] = before.crlb(target).target_std["target"]
            out["target_std_after"] = after.crlb(target).target_std["target"]
        return out


class SamplingStrategy:
    """Named schedules and their comparison. Builders return :class:`Observations` with ``n`` rows over
    ``[t0, t1]``: :meth:`uniform`, :meth:`logspaced`, :meth:`random`, :meth:`endpoints`. :meth:`compare` reports the
    criteria and bounds of each schedule (same model, same noise)."""

    @staticmethod
    def _obs(t, sigma, time_input, inputs, observable):
        t = np.asarray(t, dtype=float)
        cols = {time_input: t, **{k: np.full(t.size, float(v)) for k, v in (inputs or {}).items()}}
        obsv = None if observable is None else np.array([observable] * t.size, dtype=object)
        return Observations(cols, sigma, observable=obsv)

    @classmethod
    def uniform(cls, t0, t1, n, sigma=1.0, time_input="t", inputs=None, observable=None):
        return cls._obs(np.linspace(t0, t1, n), sigma, time_input, inputs, observable)

    @classmethod
    def logspaced(cls, t0, t1, n, sigma=1.0, time_input="t", inputs=None, observable=None):
        if t0 <= 0:
            raise ValueError("a log-spaced schedule needs t0 > 0")
        return cls._obs(np.geomspace(t0, t1, n), sigma, time_input, inputs, observable)

    @classmethod
    def random(cls, t0, t1, n, sigma=1.0, seed=0, time_input="t", inputs=None, observable=None):
        rng = np.random.default_rng(seed)
        return cls._obs(np.sort(rng.uniform(t0, t1, n)), sigma, time_input, inputs, observable)

    @classmethod
    def endpoints(cls, t0, t1, n, sigma=1.0, time_input="t", inputs=None, observable=None):
        half = n // 2
        return cls._obs(np.r_[np.full(n - half, t0), np.full(half, t1)], sigma, time_input, inputs, observable)

    @staticmethod
    def compare(model: ForwardModel, strategies: Mapping[str, Observations], parameters=None, *, target=None,
                method: str = "auto") -> DynamicsResult:
        rows = {}
        for name, obs in strategies.items():
            fim = FisherInformation.gaussian(model, obs, parameters, method=method)
            g = None
            if target is not None:
                _, g = _gradient(target, fim.theta, fim.names, 1e-5)
            c = fim.crlb(target)
            rows[name] = {"n": obs.n, "rank": fim.rank(), **fim.criteria(g), "crlb_std": c.std,
                          "target_std": c.target_std.get("target")}
        ranked = sorted(rows, key=lambda k: -rows[k]["D"])
        return DynamicsResult("fisher.sampling_strategies", {"strategies": rows, "ranking_by_D": ranked},
                              {"kind": "Cramér-Rao lower bounds (theoretical)"}, list(REGULARITY),
                              {"strategies": list(strategies)},
                              limits=["equal noise model across strategies; local at the nominal parameters"])


# =========================================================================== empirical performance
def estimator_performance(model: ForwardModel, obs: Observations, parameters: Sequence[Parameter] | None = None, *,
                          n_rep: int = 200, seed: int | None = 0, method: str = "auto") -> DynamicsResult:
    """Empirical performance of the least-squares (Gaussian maximum-likelihood) estimator against the Cramér-Rao
    bound: ``n_rep`` data sets simulated at the nominal parameters with the noise of ``obs``, each fitted by
    :func:`~pinneapple_analysis.inverse_problems.estimation.least_squares_fit`. Reports the bias, the empirical
    standard deviation and the efficiency ``crlb_var / empirical_var`` per parameter (1: the bound is attained)."""
    from ..inverse_problems.estimation import least_squares_fit

    t0 = time.perf_counter()
    ps = _params(model, parameters, None)
    th0 = ps.theta()
    fim = FisherInformation.gaussian(model, obs, list(ps.free), method=method)
    crlb = fim.crlb()
    pred = model.predict(th0, obs.inputs, obs.observable)
    rng, seed = make_rng(seed)
    est = np.empty((n_rep, len(ps.names)))
    fails = 0
    for r in range(n_rep):
        y = pred + obs.sigma * rng.standard_normal(obs.n)
        fit = least_squares_fit(model, ps, obs.with_values(y))
        if not fit.success:
            fails += 1
        est[r] = [fit.theta[n] for n in ps.names]
    mean = est.mean(axis=0)
    sd = est.std(axis=0, ddof=1)
    rows = {}
    for i, n in enumerate(ps.names):
        b = crlb.std[n]
        rows[n] = {"truth": th0[n], "mean": float(mean[i]), "bias": float(mean[i] - th0[n]),
                   "bias_se": float(sd[i] / math.sqrt(n_rep)), "empirical_std": float(sd[i]), "crlb_std": b,
                   "efficiency": float(b ** 2 / sd[i] ** 2) if sd[i] > 0 and np.isfinite(b) else math.nan,
                   "efficiency_ci95": _efficiency_ci(b, sd[i], n_rep)}
    diags = [Diagnostic("failed_fits", fails, "ok" if fails == 0 else "warn")]
    cfg = {"n_rep": n_rep, "estimator": "least squares (Gaussian MLE)", "derivatives": fim.how}
    rep = reproducibility("fisher.estimator_performance", cfg, [np.asarray(pred)], seed=seed,
                          wall_time_s=time.perf_counter() - t0)
    return DynamicsResult("fisher.estimator_performance", rows,
                          {"kind": "empirical, over simulated data sets; compare with crlb_std (theoretical)"},
                          list(REGULARITY), cfg, diags, {"n_rep": n_rep},
                          limits=["simulated at the nominal parameters with the stated noise only"],
                          reproducibility=rep, details={"estimates": est, "crlb": crlb})


def _efficiency_ci(bound: float, sd: float, n: int) -> tuple[float, float]:
    """95 % interval of ``bound^2 / var`` from the chi-square law of a Gaussian sample variance."""
    from scipy import stats

    if not (sd > 0 and np.isfinite(bound)):
        return (math.nan, math.nan)
    lo_v = (n - 1) * sd ** 2 / stats.chi2.ppf(0.975, n - 1)
    hi_v = (n - 1) * sd ** 2 / stats.chi2.ppf(0.025, n - 1)
    return (float(bound ** 2 / hi_v), float(bound ** 2 / lo_v))
