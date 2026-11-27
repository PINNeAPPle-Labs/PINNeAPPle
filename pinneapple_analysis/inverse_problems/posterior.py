"""The posterior itself: a Hamiltonian sampler, its convergence diagnostics, and the coverage of its intervals (#105).

:func:`recover` gives the *linearised* interval, which is the right answer when the model is nearly linear and a
misleading one when it is not. This module samples the posterior instead, so the interval is whatever the posterior
says — asymmetric, bounded by the parameter's own bounds, and able to be multimodal, where a Gaussian around the
optimum never is.

    >>> from pinneapple_analysis.inverse_problems import get_preset
    >>> from pinneapple_analysis.inverse_problems.posterior import hmc
    >>> b = get_preset("cooling_h").build()                                        # doctest: +SKIP
    >>> chain = hmc(b["model"], ParameterSet(b["parameters"]), b["observations"])   # doctest: +SKIP
    >>> chain.summary()["ess"], chain.rhat                                          # doctest: +SKIP

What it is: Hamiltonian Monte Carlo in the scaled parameters ``z``, with the gradient taken from the exact Jacobian the
inverse-problem models already provide (autograd, the model's own sensitivities, or central differences), a diagonal
mass matrix from the Fisher information, and the step size tuned by dual averaging during warmup (Nesterov 2009;
Hoffman & Gelman 2014). Bounds are handled by rejecting a proposal that leaves them, which is exact but depends on the
posterior not sitting on a bound — :class:`PosteriorChain` reports how often that happened.

What it answers, and why the diagnostics are not optional: a chain that has not mixed produces a confident interval
around wherever it happened to be. Split-:math:`\\hat R` (Gelman et al. 2013), the effective sample size from the
initial-positive-sequence autocorrelation (Geyer 1992) and the acceptance rate are reported with every chain, and
:func:`posterior_coverage` is how the interval is checked against a truth that is known — the same measurement the
linearised interval goes through in :mod:`~pinneapple_analysis.inverse_problems.recovery`.

SBI is not here: see the note at the end of the module documentation.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .estimation import Prior
from .forward import ForwardModel, Observations, ParameterSet, jacobian
from .recovery import CoverageReport, ParameterInterval, measure_coverage

__all__ = ["PosteriorChain", "ess", "hmc", "posterior_coverage", "rhat"]


def _log_posterior(model: ForwardModel, params: ParameterSet, obs: Observations, prior: Prior | None,
                   z: np.ndarray) -> tuple[float, np.ndarray]:
    """Log posterior and its gradient in ``z``: ``-0.5 chi2`` plus the prior, ``-inf`` outside the bounds."""
    lo, hi = params.bounds()
    if np.any(z < lo) or np.any(z > hi):
        return -math.inf, np.zeros_like(z)
    pred, J, _ = jacobian(model, params, z, obs)
    if not np.all(np.isfinite(pred)):
        return -math.inf, np.zeros_like(z)
    r = (pred - obs.values) / obs.sigma
    lp = -0.5 * float(r @ r)
    g = -(J.T @ (r / obs.sigma))                      # d(-0.5 chi2)/dz, chi2 = sum(r^2)
    if prior is not None:
        rp = prior.residuals(z)
        lp -= 0.5 * float(rp @ rp)
        g = g - prior.jac(len(z)).T @ rp
    return lp, g


def _hamiltonian(z: np.ndarray, p: np.ndarray, logp: float, inv_mass: np.ndarray) -> float:
    return -logp + 0.5 * float(p @ (inv_mass @ p))


def _trajectory(z: np.ndarray, eps: float, n_steps: int, evaluate, inv_mass: np.ndarray, chol_mass: np.ndarray,
                rng) -> tuple:
    """A reversible, volume-preserving leapfrog trajectory and the Metropolis decision on it.

    ``evaluate(z) -> (log posterior, gradient)`` is called once per position; the total energy on the trajectory is the
    potential at the end plus the kinetic energy of the final momentum, which is the momentum after the closing half
    kick — the reason the half kick is not the one at the start of the next step. ``inv_mass`` is the posterior
    covariance, so a proposal moves as far in a direction the data constrain weakly as in one it constrains strongly,
    and ``chol_mass`` draws the momentum with the matching (inverse) covariance.
    """
    lp0, g0 = evaluate(z)
    p0 = chol_mass @ rng.standard_normal(len(z))
    h0 = _hamiltonian(z, p0, lp0, inv_mass)
    z1, p1 = z.copy(), p0.copy()
    p1 = p1 + 0.5 * eps * g0
    g = g0
    lp1 = lp0
    for i in range(n_steps):
        z1 = z1 + eps * (inv_mass @ p1)
        lp1, g = evaluate(z1)
        if not np.isfinite(lp1):
            return z1, -math.inf
        if i < n_steps - 1:
            p1 = p1 + eps * g
    p1 = p1 + 0.5 * eps * g
    return z1, _hamiltonian(z1, p1, lp1, inv_mass) - h0


@dataclass
class PosteriorChain:
    """Samples of the posterior in ``z`` and in physical units, with the diagnostics that say whether to trust them."""

    names: list[str]
    chains_z: list[np.ndarray]                      # one (n_samples, p) array per chain
    chains_theta: list[np.ndarray]
    logp: list[np.ndarray]
    acceptance: float
    step_size: float
    n_leapfrog: int
    n_evaluations: int
    seconds: float
    bounded_fraction: float = 0.0                   # share of proposals rejected for leaving the bounds
    mass: np.ndarray = field(default_factory=lambda: np.ones((1, 1)))
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_chains(self) -> int:
        return len(self.chains_z)

    def _samples(self, physical: bool = True) -> np.ndarray:
        arr = self.chains_theta if physical else self.chains_z
        return np.concatenate(arr, axis=0)

    def summary(self, level: float = 0.95) -> dict[str, Any]:
        """Posterior mean, median, standard deviation, the central ``level`` interval and the correlation matrix."""
        s = self._samples()
        a = (1 - level) / 2
        corr = np.corrcoef(s.T) if s.shape[1] > 1 else np.ones((1, 1))
        ess_z = [ess(c) for c in self.chains_z]
        return {"names": self.names, "mean": s.mean(axis=0).tolist(), "median": np.median(s, axis=0).tolist(),
                "std": s.std(axis=0, ddof=1).tolist(),
                "lo": np.quantile(s, a, axis=0).tolist(), "hi": np.quantile(s, 1 - a, axis=0).tolist(),
                "level": level, "correlation": np.nan_to_num(corr).round(4).tolist(),
                "ess": [float(np.min(e)) for e in ess_z], "rhat": self.diagnostics.get("rhat"),
                "acceptance": round(self.acceptance, 3), "n_samples": int(s.shape[0]),
                "n_chains": self.n_chains, "n_evaluations": self.n_evaluations, "seconds": round(self.seconds, 2)}

    @property
    def rhat(self) -> float:
        """The worst split-:math:`\\hat R` over the parameters: above 1.01 the chains disagree with each other."""
        vals = self.diagnostics.get("rhat") or []
        return float(max(vals)) if vals else float("nan")

    def intervals(self, level: float = 0.95) -> list[ParameterInterval]:
        """The credible intervals in the shape :func:`measure_coverage` reads."""
        s = self.summary(level)
        out = []
        for k, name in enumerate(self.names):
            lo, hi, med = float(s["lo"][k]), float(s["hi"][k]), float(s["median"][k])
            scale = abs(med) if med else 1.0
            out.append(ParameterInterval(name, med, min(lo, hi), max(lo, hi),
                                         float(s["std"][k]) / scale, level))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"names": self.names, "summary": self.summary(), "acceptance": self.acceptance,
                "step_size": self.step_size, "n_leapfrog": self.n_leapfrog, "n_evaluations": self.n_evaluations,
                "bounded_fraction": self.bounded_fraction, "seconds": round(self.seconds, 2),
                "diagnostics": self.diagnostics}

    def to_markdown(self, level: float = 0.95) -> str:
        s = self.summary(level)
        rows = ["| parameter | median | credible interval | std |", "|---|---|---|---|"]
        for k, name in enumerate(self.names):
            rows.append(f"| `{name}` | {s['median'][k]:.6g} | [{s['lo'][k]:.6g}, {s['hi'][k]:.6g}] | "
                        f"{s['std'][k]:.3g} |")
        head = (f"**posterior, {self.n_chains} chain(s), {s['n_samples']} samples, {level:.0%} interval** — "
                f"acceptance {self.acceptance:.2f}, worst split-R̂ {self.rhat:.3f}, "
                f"min ESS {min(s['ess']):.0f}")
        return head + "\n\n" + "\n".join(rows)


def ess(x: np.ndarray) -> list[float]:
    """Effective sample size per column, from the initial positive sequence of the autocorrelation (Geyer 1992)."""
    a = np.asarray(x, float)
    if a.ndim == 1:
        a = a[:, None]
    n, p = a.shape
    out = []
    for j in range(p):
        col = a[:, j]
        s = col.std()
        if n < 4 or s == 0:
            out.append(float(n))
            continue
        centred = col - col.mean()
        size = 1 << int(math.ceil(math.log2(2 * n)))
        f = np.fft.rfft(centred, size)
        acf = np.fft.irfft(f * np.conjugate(f), size)[:n].real
        acf = acf / acf[0]
        total = 0.0
        for k in range(0, n - 1, 2):                 # pair the autocorrelations, as Geyer's estimator does
            pair = acf[k] + acf[k + 1] if k + 1 < n else acf[k]
            if pair <= 0:
                break
            total += pair
        out.append(float(min(n, n / max(2 * total - 1.0, 1e-12))))
    return out


def rhat(chains: Sequence[np.ndarray]) -> list[float]:
    """Split-:math:`\\hat R` per parameter (Gelman et al. 2013): each chain split in half, then the variance ratio."""
    arr = [np.asarray(c, float) for c in chains]
    if len(arr) < 2:
        return [float("nan")] * (arr[0].shape[1] if arr else 0)
    split = []
    for c in arr:
        half = len(c) // 2
        split += [c[:half], c[half:2 * half]]
    m, n = len(split), split[0].shape[0]
    if n < 2:
        return [float("nan")] * split[0].shape[1]
    out = []
    for j in range(split[0].shape[1]):
        means = np.array([s[:, j].mean() for s in split])
        vars_ = np.array([s[:, j].var(ddof=1) for s in split])
        W = vars_.mean()
        B = n * means.var(ddof=1)
        var_hat = (n - 1) / n * W + B / n
        out.append(float(math.sqrt(max(var_hat / W, 0.0))) if W > 0 else float("inf"))
    return out


def hmc(model: ForwardModel, params: ParameterSet, obs: Observations, *, prior: Prior | None = None,
        n_samples: int = 1000, n_warmup: int = 500, n_chains: int = 4, n_leapfrog: int = 10,
        step_size: float | None = None, target_accept: float = 0.8, max_step: float = 5.0,
        seed: int = 0, z0: np.ndarray | None = None, time_budget: float | None = None) -> PosteriorChain:
    """Sample the posterior of the free parameters with Hamiltonian Monte Carlo.

    ``n_warmup`` iterations per chain tune the step size by dual averaging (target: ``target_accept``); the mass
    matrix is the Fisher information at the starting point, which is the quadratic approximation of the posterior and
    makes the sampler close to isotropic even when two parameters differ by orders of magnitude. ``n_chains``
    independent chains are what makes split-:math:`\\hat R` meaningful; one chain leaves it undefined and the summary
    says so.
    """
    if not obs.has_values:
        raise ValueError("the observations have no measured values to sample from")
    if n_samples < 1 or n_warmup < 1 or n_chains < 1:
        raise ValueError("n_samples, n_warmup and n_chains all have to be at least 1")
    if n_leapfrog < 1:
        raise ValueError("n_leapfrog has to be at least 1")
    t0 = time.time()
    n_eval0 = getattr(model, "_n_eval", 0)
    p = len(params)
    lo, hi = params.bounds()
    z_start = params.clip(params.z0() if z0 is None else np.asarray(z0, float).copy())
    rng = np.random.default_rng(seed)

    # the mass matrix: the posterior precision at the start (dense, so the trade-off between two parameters is
    # preconditioned as well as their difference in scale), which is the quadratic approximation of the posterior
    _, J0, _ = jacobian(model, params, z_start, obs)
    precision = (J0 / obs.sigma[:, None]).T @ (J0 / obs.sigma[:, None])
    if prior is not None:
        precision = precision + prior.precision(p)
    mass = 0.5 * (precision + precision.T) + 1e-10 * np.eye(p)
    inv_mass = np.linalg.pinv(mass, hermitian=True)
    try:
        chol_mass = np.linalg.cholesky(mass)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(mass)
        chol_mass = V @ np.diag(np.sqrt(np.maximum(w, 1e-12)))
    eps0 = step_size if step_size is not None else 0.5

    chains_z, chains_theta, chains_lp = [], [], []
    bounded = 0
    total = 0
    last_eps, total_acc = eps0, 0.0
    eval_budget_hit = False

    cache: dict[str, Any] = {}

    def evaluate(zz: np.ndarray) -> tuple[float, np.ndarray]:
        key = zz.tobytes()
        if cache.get("key") != key:
            cache.update(key=key, value=_log_posterior(model, params, obs, prior, zz))
        return cache["value"]

    for c in range(n_chains):
        z = params.clip(z_start + (1e-3 * rng.standard_normal(p) if c else 0.0))
        lp, _ = evaluate(z)
        eps = eps0
        mu = math.log(10 * eps)
        log_eps, log_eps_bar, h_bar = math.log(eps), 0.0, 0.0
        acc_hist = 0
        samples, thetas, lps = [], [], []
        for it in range(n_warmup + n_samples):
            if time_budget is not None and time.time() - t0 > time_budget:
                eval_budget_hit = True
                break
            z_new, dH = _trajectory(z, eps, int(rng.integers(max(1, n_leapfrog // 2), n_leapfrog + 1)), evaluate,
                                    inv_mass, chol_mass, rng)
            total += 1
            if np.any(z_new < lo) or np.any(z_new > hi):
                bounded += 1
                accept = False
            else:
                accept = bool(np.isfinite(dH) and math.log(max(rng.random(), 1e-300)) < -dH)
            if accept:
                z = z_new
                lp, _ = evaluate(z)
                acc_hist += 1
            if it < n_warmup:                                  # dual averaging (Hoffman & Gelman 2014)
                eta = 1.0 / (it + 10.0)
                h_bar = (1 - eta) * h_bar + eta * (target_accept - (1.0 if accept else 0.0))
                log_eps = mu - math.sqrt(it + 1) / 0.05 * h_bar
                log_eps = float(np.clip(log_eps, math.log(1e-8), math.log(max_step)))
                eps = math.exp(log_eps)
                w = (it + 1) ** -0.75
                log_eps_bar = w * log_eps + (1 - w) * log_eps_bar
            else:
                samples.append(z.copy())
                thetas.append([params.free[i].from_z(float(z[i])) for i in range(p)])
                lps.append(lp)
        last_eps = math.exp(log_eps_bar) if n_warmup else math.exp(log_eps)
        total_acc += acc_hist / max(1, n_warmup + n_samples)
        if samples:
            chains_z.append(np.array(samples))
            chains_theta.append(np.array(thetas))
            chains_lp.append(np.array(lps))

    if not chains_z:
        raise RuntimeError("no samples were drawn: the chain never started (check the parameter bounds and the data)")
    diag = {"rhat": rhat(chains_z) if len(chains_z) > 1 else [],
            "ess": [float(np.min(ess(c))) for c in chains_z],
            "n_chains": len(chains_z), "stopped_early": eval_budget_hit}
    return PosteriorChain(params.names, chains_z, chains_theta, chains_lp,
                          acceptance=total_acc / max(1, n_chains), step_size=last_eps, n_leapfrog=n_leapfrog,
                          n_evaluations=getattr(model, "_n_eval", 0) - n_eval0, seconds=time.time() - t0,
                          bounded_fraction=bounded / max(1, total), mass=mass, diagnostics=diag)


def posterior_coverage(case: Callable[[np.random.Generator], tuple[Any, ParameterSet, Observations]], *,
                       n: int = 40, level: float = 0.95, seed: int = 0, tolerance: float = 0.1,
                       **hmc_kwargs: Any) -> CoverageReport:
    """How often the posterior's credible interval contains the truth, over ``n`` synthetic experiments.

    The measurement is the one the linearised interval goes through
    (:func:`~pinneapple_analysis.inverse_problems.recovery.measure_coverage`): the same case, the same count, so the
    two intervals can be compared on equal terms. It is slower — a sampler per repetition, not one Jacobian — so the
    default ``n`` is smaller.
    """
    return measure_coverage(case, lambda model, params, obs: hmc(model, params, obs, seed=seed, **hmc_kwargs)
                            .intervals(level), n=n, level=level, seed=seed, tolerance=tolerance)
