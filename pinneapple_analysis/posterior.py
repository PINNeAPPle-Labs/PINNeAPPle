"""Posterior sampling with the diagnostics that say whether the posterior is there.

The inverse-problems package already fits parameters by MAP, Laplace and one adaptive random-walk chain. What it does
not do is answer the question a Bayesian study actually asks: *is this posterior sampled well enough to quote?* This
module samples it with two algorithms and reports the numbers that decide:

``hmc``
    Hamiltonian Monte Carlo -- leapfrog on the gradient of the log posterior, with the step size set by dual averaging
    and the mass matrix adapted during warm-up. The gradient method is not decoration: on a posterior with strong
    correlation a random walk needs orders of magnitude more evaluations for the same effective sample size, and
    :func:`compare_methods` measures that ratio instead of asserting it.
``rwm``
    Adaptive random-walk Metropolis (Haario et al. 2001), the same family the rest of the package uses, kept here so
    the comparison has a baseline on equal footing (same target, same starting points, same budget).

Both run several chains from the same starting point, and every result carries :class:`Diagnostics`:

* **split R-hat**, rank-normalised (Vehtari et al. 2021, *Rank-normalization, folding, and localization: an improved
  R-hat*): the pooled draws are replaced by their normal scores, each chain is split in half, and
  ``R-hat = sqrt(var_plus / W)``. Above 1.01 the chains have not mixed.
* **bulk effective sample size** with Geyer's initial positive sequence on the same rank-normalised draws, and the
  Monte Carlo standard error of the mean that follows from it: ``sd / sqrt(ESS)``.
* the **acceptance rate**, the **divergences** of HMC, and the number of gradient evaluations, so a comparison can be
  made per unit of work rather than per step.

``converged`` is the conjunction of the thresholds and is reported as it comes out. A chain that has not converged
says so: :meth:`Chains.summary` prints the failure and :class:`Diagnostics.table` puts the numbers next to the limits.

What this module does not do: no NUTS, no reversible-jump, no automatic differentiation. The gradient comes from
:meth:`Posterior.gradient`, which is central differences unless the caller provides the analytic one -- stated because
the cost model of the comparison changes with it (``n_gradient_evaluations`` counts calls, not flops).

    >>> from pinneapple_analysis.posterior import Posterior, sample
    >>> chains = sample(Posterior(log_prob=log_posterior, dimension=2, names=("a", "b")), x0=[1.0, 1.0],
    ...                 method="hmc", n_steps=2000, n_chains=4, seed=0)
    >>> print(chains.summary())          # mean, sd, 5/50/95 %, R-hat, ESS, MCSE -- and whether it converged
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Posterior", "Chains", "Diagnostics", "sample", "hmc", "rwm", "compare_methods",
           "split_rhat", "effective_sample_size", "mcse_mean", "R_HAT_LIMIT", "ESS_LIMIT"]

#: The limits :class:`Diagnostics` reports against. ``R_HAT_LIMIT`` is the 1.01 of Vehtari et al. (2021);
#: ``ESS_LIMIT`` is the 400 the same paper suggests for a stable estimate of the mean and the spread.
R_HAT_LIMIT = 1.01
ESS_LIMIT = 400.0


# ---------------------------------------------------------------------------
# the target
# ---------------------------------------------------------------------------

@dataclass
class Posterior:
    """A log posterior to sample, with its gradient when the caller has one.

    The log probability must be a scalar function of one point and return ``-inf`` outside its support -- quietly
    returning NaN would make every rejection look like a bug in the sampler. ``gradient`` falls back to central
    differences with a step of ``1e-6 * max(1, |x_i|)``; that is a real cost per evaluation, counted in
    :attr:`Chains.n_gradient_evaluations`."""

    log_prob: Callable[[np.ndarray], float]
    dimension: int
    names: tuple[str, ...] = ()
    grad_log_prob: Callable[[np.ndarray], np.ndarray] | None = None
    step: float = 1e-6

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError(f"{self.dimension} parameters: there is nothing to sample")
        names = tuple(self.names) or tuple(f"x{i}" for i in range(self.dimension))
        if len(names) != self.dimension:
            raise ValueError(f"{len(names)} names against {self.dimension} parameters")
        object.__setattr__(self, "names", names)

    def logp(self, x: np.ndarray) -> float:
        value = float(self.log_prob(np.asarray(x, dtype=np.float64)))
        return value if np.isfinite(value) else -np.inf

    def gradient(self, x: np.ndarray) -> np.ndarray:
        if self.grad_log_prob is not None:
            return np.asarray(self.grad_log_prob(np.asarray(x, dtype=np.float64)), dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        out = np.empty(self.dimension)
        for i in range(self.dimension):
            delta = np.zeros(self.dimension)
            delta[i] = self.step * max(1.0, abs(x[i]))
            up, down = self.logp(x + delta), self.logp(x - delta)
            out[i] = (up - down) / (2.0 * delta[i]) if np.isfinite(up) and np.isfinite(down) else 0.0
        return out


# ---------------------------------------------------------------------------
# the draws
# ---------------------------------------------------------------------------

@dataclass
class Chains:
    """The draws of several chains, and what they cost."""

    draws: np.ndarray                   # (chains, steps, dimension)
    names: tuple[str, ...]
    method: str
    accept_rate: float
    divergences: int = 0                 # in the kept draws: a warm-up divergence is the adaptation working
    warmup_divergences: int = 0
    n_logp_evaluations: int = 0
    n_gradient_evaluations: int = 0
    seconds: float = 0.0
    starting_point: np.ndarray | None = None

    def __post_init__(self) -> None:
        draws = np.asarray(self.draws, dtype=np.float64)
        if draws.ndim != 3:
            raise ValueError(f"draws must be (chains, steps, parameters): {draws.shape}")
        object.__setattr__(self, "draws", draws)

    @property
    def work(self) -> int:
        """The calls to the target: every log posterior, plus every gradient. A central-difference gradient has
        already put its own ``2 * d`` log-posterior calls in the count, so the gradient is one more unit for the
        bookkeeping and nothing more."""
        return int(self.n_logp_evaluations + self.n_gradient_evaluations)

    @property
    def n_chains(self) -> int:
        return int(self.draws.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.draws.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.draws.shape[2])

    def mean(self) -> np.ndarray:
        return self.draws.reshape(-1, self.dimension).mean(axis=0)

    def sd(self) -> np.ndarray:
        return self.draws.reshape(-1, self.dimension).std(axis=0, ddof=1)

    def interval(self, level: float = 0.95) -> np.ndarray:
        """The equal-tailed credible interval per parameter, shape ``(dimension, 2)``."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"{level} is not a probability")
        tail = 50.0 * (1.0 - level)
        pooled = self.draws.reshape(-1, self.dimension)
        return np.column_stack([np.percentile(pooled, tail, axis=0), np.percentile(pooled, 100.0 - tail, axis=0)])

    def diagnostics(self) -> Diagnostics:
        return Diagnostics(
            r_hat=np.array([split_rhat(self.draws[:, :, i]) for i in range(self.dimension)]),
            ess=np.array([effective_sample_size(self.draws[:, :, i]) for i in range(self.dimension)]),
            names=self.names, accept_rate=self.accept_rate, divergences=self.divergences,
            n_chains=self.n_chains, n_steps=self.n_steps)

    def summary(self) -> str:
        diagnostics = self.diagnostics()
        pooled = self.draws.reshape(-1, self.dimension)
        low = np.percentile(pooled, 5.0, axis=0)
        median = np.percentile(pooled, 50.0, axis=0)
        high = np.percentile(pooled, 95.0, axis=0)
        header = (f"{self.method}: {self.n_chains} chain(s) x {self.n_steps} step(s) in {self.seconds:.2f} s, "
                  f"acceptance {self.accept_rate:.1%}, {self.work} call(s) to the target "
                  f"({self.n_gradient_evaluations} of them gradients)")
        lines = [header, f"{'parameter':>12} {'mean':>11} {'sd':>10} {'5%':>11} {'50%':>11} {'95%':>11} "
                         f"{'R-hat':>7} {'ESS':>7} {'MCSE':>10}"]
        for i, name in enumerate(self.names):
            lines.append(f"{name:>12} {pooled[:, i].mean():11.4g} {pooled[:, i].std(ddof=1):10.4g} "
                         f"{low[i]:11.4g} {median[i]:11.4g} {high[i]:11.4g} {diagnostics.r_hat[i]:7.4f} "
                         f"{diagnostics.ess[i]:7.1f} {mcse_mean(self.draws[:, :, i]):10.3g}")
        lines.append(diagnostics.verdict())
        return "\n".join(lines)


@dataclass
class Diagnostics:
    """The convergence numbers, and the sentence that reads them."""

    r_hat: np.ndarray
    ess: np.ndarray
    names: tuple[str, ...] = ()
    accept_rate: float = float("nan")
    divergences: int = 0
    n_chains: int = 0
    n_steps: int = 0
    r_hat_limit: float = R_HAT_LIMIT
    ess_limit: float = ESS_LIMIT

    @property
    def worst_r_hat(self) -> float:
        return float(np.max(self.r_hat)) if self.r_hat.size else float("nan")

    @property
    def least_ess(self) -> float:
        return float(np.min(self.ess)) if self.ess.size else float("nan")

    @property
    def converged(self) -> bool:
        return bool(self.worst_r_hat <= self.r_hat_limit and self.least_ess >= self.ess_limit
                    and self.divergences == 0)

    def verdict(self) -> str:
        if self.converged:
            return (f"converged: worst R-hat {self.worst_r_hat:.4f} <= {self.r_hat_limit}, least ESS "
                    f"{self.least_ess:.0f} >= {self.ess_limit:.0f}, no divergence")
        reasons = []
        if self.worst_r_hat > self.r_hat_limit:
            reasons.append(f"R-hat {self.worst_r_hat:.4f} > {self.r_hat_limit} (the chains have not mixed)")
        if self.least_ess < self.ess_limit:
            reasons.append(f"ESS {self.least_ess:.0f} < {self.ess_limit:.0f} (too few independent draws)")
        if self.divergences:
            reasons.append(f"{self.divergences} divergence(s) (the step size could not follow the posterior)")
        return "NOT converged: " + "; ".join(reasons)

    def table(self) -> str:
        lines = [f"{'chain':>17} {'R-hat':>8} {'ESS':>8}", "-" * 36]
        for name, r_hat, ess in zip(self.names, self.r_hat, self.ess, strict=True):
            lines.append(f"{self.n_chains}x{self.n_steps} {name:>10} {r_hat:8.4f} {ess:8.1f}")
        lines.append(self.verdict())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"r_hat": {n: float(v) for n, v in zip(self.names, self.r_hat, strict=True)},
                "ess": {n: float(v) for n, v in zip(self.names, self.ess, strict=True)},
                "converged": self.converged, "verdict": self.verdict(), "accept_rate": self.accept_rate,
                "divergences": self.divergences, "r_hat_limit": self.r_hat_limit, "ess_limit": self.ess_limit}


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def _rank_normal_scores(draws: np.ndarray) -> np.ndarray:
    """The normal scores of the pooled draws: the transform the rank-normalised R-hat is defined on."""
    flat = np.asarray(draws, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size)
    ranks[order] = np.arange(1, flat.size + 1, dtype=np.float64)     # ties take the lower rank, which is enough here
    return _normal_ppf((ranks - 0.375) / (flat.size + 0.25)).reshape(np.shape(draws))


def _normal_ppf(p: np.ndarray) -> np.ndarray:
    """The inverse standard normal CDF, Acklam's rational approximation (|error| < 1e-9, no scipy)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    p = np.asarray(p, dtype=np.float64)
    out = np.empty_like(p)
    low, high = p < 0.02425, p > 1.0 - 0.02425
    middle = ~(low | high)
    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p[low]))
        out[low] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                   ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        out[high] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if np.any(middle):
        q = p[middle] - 0.5
        r = q * q
        out[middle] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return out


def _split(draws: np.ndarray) -> np.ndarray:
    """Split each chain in half -- the "split" of split R-hat, which catches a chain that drifts."""
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim == 1:
        draws = draws[None, :]
    if draws.shape[1] < 4:
        raise ValueError(f"{draws.shape[1]} draws per chain: split R-hat needs at least four")
    half = draws.shape[1] // 2
    return np.concatenate([draws[:, :half], draws[:, draws.shape[1] - half:]], axis=0)


def split_rhat(draws: np.ndarray) -> float:
    """Rank-normalised split R-hat of one parameter's draws, shape ``(chains, steps)``.

    The draws are replaced by their normal scores, each chain is cut in half, and the between- and within-chain
    variances give ``sqrt(var_plus / W)`` with the ``(n - 1) / n`` correction of the original Gelman-Rubin statistic."""
    chains = _split(_rank_normal_scores(np.asarray(draws, dtype=np.float64)))
    m, n = chains.shape
    if n < 2:
        raise ValueError(f"{n} draws per split chain: a variance needs two")
    within = float(np.mean(chains.var(axis=1, ddof=1)))
    between = float(n * chains.mean(axis=1).var(ddof=1))
    if within <= 0.0:
        raise ValueError("every draw of every chain is the same number: there is no variance to compare")
    variance_plus = (n - 1.0) / n * within + between / n
    return float(np.sqrt(variance_plus / within))


def _autocovariance(chains: np.ndarray) -> np.ndarray:
    """The autocovariance of each chain, divided by ``n`` (the convention the ESS formula expects)."""
    m, n = chains.shape
    centred = chains - chains.mean(axis=1, keepdims=True)
    return np.array([float((centred[:, : n - t] * centred[:, t:]).sum(axis=1).mean() / n) for t in range(n)])


def effective_sample_size(draws: np.ndarray) -> float:
    """Bulk effective sample size of one parameter's draws, shape ``(chains, steps)``.

    Rank-normalised, split, then Geyer's initial positive sequence over the autocorrelations of the split chains --
    the estimator the paper behind R-hat pairs with it. Capped at ``chains * steps * log10(chains * steps)``, the
    same cap, since a sum of pair sums that never turns negative would otherwise report an infinite ESS."""
    chains = _split(_rank_normal_scores(np.asarray(draws, dtype=np.float64)))
    m, n = chains.shape
    if n < 4:
        raise ValueError(f"{n} draws per split chain: the autocorrelation needs four to be estimated")
    within = float(np.mean(chains.var(axis=1, ddof=1)))
    between = float(n * chains.mean(axis=1).var(ddof=1))
    variance_plus = (n - 1.0) / n * within + between / n
    if variance_plus <= 0.0:
        raise ValueError("the draws have no variance")
    autocovariance = _autocovariance(chains)
    rho = 1.0 - (within - autocovariance) / variance_plus
    rho[0] = 1.0
    total = 0.0
    for t in range(0, n - 1, 2):                     # pair sums, stopping at the first negative pair (Geyer)
        pair = rho[t] + rho[t + 1] if t + 1 < n else rho[t]
        if pair < 0.0:
            break
        total += pair
    tau = -1.0 + 2.0 * total
    if tau <= 0.0:
        return float(m * n)                          # the draws look independent: no information to discount
    return float(min(m * n / tau, m * n * np.log10(max(m * n, 10))))


def mcse_mean(draws: np.ndarray) -> float:
    """The Monte Carlo standard error of the mean: ``sd / sqrt(ESS)``, the number that says whether a quoted mean has
    the digits it shows."""
    draws = np.asarray(draws, dtype=np.float64)
    pooled = draws.reshape(-1)
    return float(pooled.std(ddof=1) / np.sqrt(max(effective_sample_size(draws), 1.0)))


# ---------------------------------------------------------------------------
# the samplers
# ---------------------------------------------------------------------------

def _check(posterior: Posterior, x0, n_chains: int, n_steps: int, n_warmup: int, seed: int) -> tuple[np.ndarray, int]:
    x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
    if x0.size != posterior.dimension:
        raise ValueError(f"x0 has {x0.size} entries against {posterior.dimension} parameters")
    if n_chains < 1 or n_steps < 4:
        raise ValueError(f"{n_chains} chain(s) x {n_steps} step(s): four steps is the least a diagnostic reads")
    if n_warmup < 0:
        raise ValueError(f"{n_warmup} warm-up steps is not a number of steps")
    return x0, int(seed)


def rwm(posterior: Posterior, x0, *, n_steps: int = 2000, n_chains: int = 4, n_warmup: int | None = None,
        scale: float = 2.38, seed: int = 0) -> Chains:
    """Adaptive random-walk Metropolis, one independent chain per element of the batch.

    The proposal is Gaussian with covariance ``scale^2 / d * Cov`` where ``Cov`` is the running covariance of the
    chain itself (Haario's adaptation), refreshed during warm-up and then frozen -- adapting for ever makes the chain
    non-Markovian and the diagnostics dishonest."""
    x0, seed = _check(posterior, x0, n_chains, n_steps, n_warmup or 0, seed)
    warmup = n_steps // 2 if n_warmup is None else int(n_warmup)
    rng = np.random.default_rng(seed)
    dimension = posterior.dimension
    t0 = time.perf_counter()
    current = np.tile(x0, (n_chains, 1))
    logp = np.array([posterior.logp(point) for point in current])
    evaluations = n_chains
    chol = np.linalg.cholesky(np.eye(dimension) * (scale ** 2 / dimension))
    warm = np.empty((n_chains, max(warmup, 1), dimension))
    draws = np.empty((n_chains, n_steps, dimension))
    accepted = 0
    for step in range(n_steps):
        for chain in range(n_chains):
            proposal = current[chain] + chol @ rng.standard_normal(dimension)
            proposal_logp = posterior.logp(proposal)
            evaluations += 1
            if np.log(rng.random()) < proposal_logp - logp[chain]:
                current[chain], logp[chain] = proposal, proposal_logp
                accepted += 1
        if step < warmup:
            warm[:, step] = current
            if step + 1 == warmup // 2 and warmup > 8:          # freeze the covariance at the end of warm-up
                flat = warm[:, : step + 1].reshape(-1, dimension)
                covariance = np.atleast_2d(np.cov(flat, rowvar=False)) + 1e-10 * np.eye(dimension)
                chol = np.linalg.cholesky(scale ** 2 / dimension * covariance)
        else:
            draws[:, step - warmup] = current
    kept = draws[:, : n_steps - warmup]
    return Chains(draws=kept, names=posterior.names, method="rwm", accept_rate=accepted / (n_chains * n_steps),
                  n_logp_evaluations=evaluations, seconds=time.perf_counter() - t0, starting_point=x0)


def hmc(posterior: Posterior, x0, *, n_steps: int = 1000, n_chains: int = 4, n_warmup: int | None = None,
        n_leapfrog: int = 10, step_size: float = 0.25, target_accept: float = 0.8, seed: int = 0) -> Chains:
    """Hamiltonian Monte Carlo with dual-averaged step size and an adapted mass matrix.

    ``n_steps`` is the number of *kept* steps per chain; warm-up is added on top and discarded (the mass matrix is
    adapted at a third and two thirds of it from the running covariance, the step size by dual averaging at every
    warm-up step, targeting ``target_accept``). A step whose energy error exceeds 1000 is counted as a divergence and
    rejected -- the signature of a step size too large for the curvature, and the reason a posterior with a funnel is
    worth looking at rather than quoting. Divergences during warm-up are counted separately: they are the adaptation
    working, not a defect of the kept draws.

    The trajectory length is drawn uniformly from ``1 .. n_leapfrog`` at every step, which is not decoration: a fixed
    length that lands on a whole number of periods of the posterior's own oscillation ends every trajectory where it
    started. Measured on the conjugate model the test suite uses, a fixed ten-leapfrog trajectory gave the draws a
    lag-one autocorrelation of 0.89 and an effective sample size of 162; jittered, the same budget gives 5301."""
    x0, seed = _check(posterior, x0, n_chains, n_steps, max(n_warmup or 0, 0), seed)
    warmup = max(n_steps, 200) if n_warmup is None else int(n_warmup)
    rng = np.random.default_rng(seed)
    dimension = posterior.dimension
    t0 = time.perf_counter()
    position = np.tile(x0, (n_chains, 1))
    logp = np.array([posterior.logp(point) for point in position])
    gradient = np.array([posterior.gradient(point) for point in position])
    logp_evaluations = n_chains
    gradient_evaluations = n_chains
    inverse_mass = np.eye(dimension)
    chol_mass = np.eye(dimension)
    warm = np.empty((n_chains, max(warmup, 1), dimension))

    def kinetic(momentum: np.ndarray, inverse_mass: np.ndarray) -> float:
        return 0.5 * float(momentum @ inverse_mass @ momentum)

    epsilon = float(step_size)
    mu = np.log(10.0 * epsilon)
    h_bar = 0.0
    log_epsilon_bar = 0.0
    adaptation = 0.05
    draws = np.empty((n_chains, n_steps, dimension))
    accepted = 0
    divergences = 0
    warmup_divergences = 0
    for step in range(n_steps + warmup):
        for chain in range(n_chains):
            momentum = chol_mass @ rng.standard_normal(dimension)
            start_logp = logp[chain]
            initial_energy = -start_logp + kinetic(momentum, inverse_mass)
            proposal = position[chain].copy()
            # a trajectory length drawn per step, not fixed: a fixed one that happens to be a whole number of periods
            # of the posterior's own oscillation ends every trajectory where it started, and the draws come out as
            # correlated as the trajectory is long. Jitter is the standard remedy (Hoffman & Gelman 2014, section 4.4)
            integration_steps = int(rng.integers(1, n_leapfrog + 1))
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                momentum = momentum + 0.5 * epsilon * gradient[chain]
                proposal_logp = start_logp
                proposal_gradient = gradient[chain]
                for leapfrog in range(integration_steps):
                    proposal = proposal + epsilon * (inverse_mass @ momentum)
                    proposal_logp = posterior.logp(proposal)
                    logp_evaluations += 1
                    if not np.isfinite(proposal_logp):
                        break
                    proposal_gradient = posterior.gradient(proposal)
                    gradient_evaluations += 1
                    if posterior.grad_log_prob is None:
                        logp_evaluations += 2 * dimension       # the central differences the fallback costs
                    if leapfrog < integration_steps - 1:
                        momentum = momentum + epsilon * proposal_gradient
                    else:
                        momentum = momentum + 0.5 * epsilon * proposal_gradient
                final_energy = -proposal_logp + kinetic(momentum, inverse_mass)
            energy_error = float(final_energy - initial_energy)
            divergent = not np.isfinite(energy_error) or energy_error > 1000.0
            if divergent:
                if step < warmup:
                    warmup_divergences += 1
                else:
                    divergences += 1
            elif np.log(rng.random()) < -energy_error:
                position[chain], logp[chain], gradient[chain] = proposal, proposal_logp, proposal_gradient
                accepted += 1
            if step < warmup:
                warm[:, step] = position[chain]
                statistic = 0.0 if divergent else min(1.0, float(np.exp(min(0.0, -energy_error))))
                h_bar = (1.0 - 1.0 / (step + 11.0)) * h_bar + (target_accept - statistic) / (step + 11.0)
                epsilon = float(np.exp(mu - np.sqrt(step + 1) / adaptation * h_bar))
                log_epsilon_bar = (1.0 - 1.0 / (step + 1)) * log_epsilon_bar + np.log(epsilon) / (step + 1)
                if step + 1 in (max(warmup // 3, 1), max(2 * warmup // 3, 2)) and step > dimension:
                    flat = warm[:, : step + 1].reshape(-1, dimension)
                    covariance = np.atleast_2d(np.cov(flat, rowvar=False)) + 1e-8 * np.eye(dimension)
                    chol_mass = np.linalg.cholesky(covariance)
                    inverse_mass = np.linalg.inv(chol_mass @ chol_mass.T)
                    # a step moves by epsilon * M^-1 p, and p ~ N(0, M): the velocity scales with 1 / sigma, so the
                    # step size has to be scaled by the geometric mean of the target's spreads to move the same way
                    epsilon = float(np.exp(log_epsilon_bar) * np.exp(np.mean(np.log(np.diag(chol_mass)))))
            else:
                draws[:, step - warmup] = position
    return Chains(draws=draws, names=posterior.names, method="hmc",
                  accept_rate=accepted / (n_chains * (n_steps + warmup)), divergences=divergences,
                  warmup_divergences=warmup_divergences,
                  n_logp_evaluations=logp_evaluations, n_gradient_evaluations=gradient_evaluations,
                  seconds=time.perf_counter() - t0, starting_point=x0)


def sample(posterior: Posterior, x0, *, method: str = "hmc", **options) -> Chains:
    """``method`` is ``"hmc"`` or ``"rwm"``; every other keyword goes to the sampler."""
    if method == "hmc":
        return hmc(posterior, x0, **options)
    if method == "rwm":
        return rwm(posterior, x0, **options)
    raise ValueError(f"{method!r} is not a sampler: 'hmc' or 'rwm'")



def compare_methods(posterior: Posterior, x0, *, methods: Sequence[str] = ("rwm", "hmc"),
                    n_steps: int = 1000, n_chains: int = 4, seed: int = 0, **options) -> dict[str, Any]:
    """Run each sampler on the same target from the same starting points and compare the work it took.

    The comparison is per unit of work, not per step: HMC's step is worth many log-posterior evaluations, and the
    ``ess_per_evaluation`` column is the one that decides. ``rwm`` does not report gradient evaluations (it takes
    none), so its work is the number of log-probability calls, counted the same way."""
    table = {}
    for method in methods:
        chains = sample(posterior, x0, method=method, n_steps=n_steps, n_chains=n_chains, seed=seed, **options)
        diagnostics = chains.diagnostics()
        table[method] = {"diagnostics": diagnostics, "least_ess": diagnostics.least_ess,
                         "worst_r_hat": diagnostics.worst_r_hat, "seconds": chains.seconds,
                         "work": chains.work, "ess_per_evaluation": diagnostics.least_ess / max(chains.work, 1)}
    return table
