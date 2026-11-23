"""Posterior sampling against posteriors whose answer is known before anything is run.

Two targets with closed-form or numerically exact posteriors -- a conjugate linear-Gaussian model, where the posterior
is exactly Gaussian, and a two-parameter decay model, where a 300 x 300 quadrature gives it to the digit -- plus the
diagnostics measured on draws whose effective sample size is known by construction. The coverage loop is the one the
issue asks for: 60 synthetic datasets, the credible interval against the truth, and the exact posterior's own
intervals over the same data for comparison, so the number reported is the sampler's and not the model's.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.posterior import (
    ESS_LIMIT,
    R_HAT_LIMIT,
    Chains,
    Diagnostics,
    Posterior,
    compare_methods,
    effective_sample_size,
    mcse_mean,
    sample,
    split_rhat,
)

TIMES = np.linspace(0.0, 3.0, 12)
SIGMA = 0.05
TRUE = np.array([2.0, 1.1])


# ---------------------------------------------------------------------------
# the two targets with known posteriors
# ---------------------------------------------------------------------------

def _conjugate(seed: int = 0, sigma: float = 0.3, tau: float = 2.0):
    """A linear-Gaussian model with a Gaussian prior: the posterior is exactly N(mean, covariance)."""
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(12, 2))
    truth = np.array([1.5, -0.8])
    values = design @ truth + sigma * rng.normal(size=12)
    precision = design.T @ design / sigma ** 2 + np.eye(2) / tau ** 2
    covariance = np.linalg.inv(precision)
    mean = covariance @ (design.T @ values / sigma ** 2)

    def log_prob(z):
        residual = values - design @ z
        return -0.5 * float(residual @ residual) / sigma ** 2 - 0.5 * float(z @ z) / tau ** 2

    def gradient(z):
        residual = values - design @ z
        return design.T @ residual / sigma ** 2 - z / tau ** 2

    return Posterior(log_prob=log_prob, dimension=2, names=("b0", "b1"), grad_log_prob=gradient), mean, covariance


def _decay(values) -> Posterior:
    """``y = a exp(-b t)`` with a flat prior on ``[0, 6] x [0, 4]`` -- the model whose posterior the grid gives."""
    def log_prob(z):
        a, b = z
        if not (0.0 < a < 6.0 and 0.0 < b < 4.0):
            return -np.inf
        residual = values - np.exp(-b * TIMES) * a
        return -0.5 * float(residual @ residual) / SIGMA ** 2

    def gradient(z):
        a, b = z
        decay = np.exp(-b * TIMES)
        residual = values - decay * a
        return np.array([float(residual @ decay), -float(residual @ (a * TIMES * decay))]) / SIGMA ** 2

    return Posterior(log_prob=log_prob, dimension=2, names=("a", "b"), grad_log_prob=gradient)


_GRID_A = np.linspace(0.0, 6.0, 300)
_GRID_B = np.linspace(0.0, 4.0, 300)
_MESH_A, _MESH_B = np.meshgrid(_GRID_A, _GRID_B, indexing="ij")


def _exact_decay(values) -> tuple[np.ndarray, np.ndarray]:
    """The exact posterior of the decay model by quadrature: its mean and its equal-tailed 95 % interval."""
    residual = values[None, None, :] - np.exp(-_MESH_B[..., None] * TIMES) * _MESH_A[..., None]
    logp = -0.5 * (residual ** 2).sum(axis=-1) / SIGMA ** 2
    weight = np.exp(logp - logp.max())
    weight /= weight.sum()
    mean, bounds = [], []
    for marginal, axis in ((weight.sum(axis=1), _GRID_A), (weight.sum(axis=0), _GRID_B)):
        cumulative = np.cumsum(marginal)
        mean.append(float(marginal @ axis))
        bounds.append([np.interp(0.025, cumulative, axis), np.interp(0.975, cumulative, axis)])
    return np.array(mean), np.array(bounds)


def _synthetic(seed: int) -> np.ndarray:
    return np.exp(-TRUE[1] * TIMES) * TRUE[0] + SIGMA * np.random.default_rng(seed).normal(size=TIMES.size)


# ---------------------------------------------------------------------------
# the posterior against its exact value
# ---------------------------------------------------------------------------

def test_the_mean_and_spread_are_the_conjugate_posteriors():
    """On a posterior that is exactly Gaussian, the sampler's mean has to sit within its own Monte Carlo error of the
    analytic one, and its spread within a few percent."""
    posterior, mean, covariance = _conjugate()
    chains = sample(posterior, [0.0, 0.0], method="hmc", n_steps=2000, n_chains=4, seed=0)
    assert chains.diagnostics().converged
    assert np.all(np.abs(chains.mean() - mean) < 3.0 * np.array([mcse_mean(chains.draws[:, :, i]) for i in range(2)]))
    assert np.allclose(chains.sd(), np.sqrt(np.diag(covariance)), rtol=0.08)
    analytic = np.column_stack([mean - 1.959964 * np.sqrt(np.diag(covariance)),
                                mean + 1.959964 * np.sqrt(np.diag(covariance))])
    assert np.allclose(chains.interval(0.95), analytic, rtol=2e-2, atol=2e-2)
    assert chains.diagnostics().least_ess >= ESS_LIMIT
    assert chains.diagnostics().least_ess <= 4 * chains.n_steps * np.log10(4 * chains.n_steps) + 1.0


def test_the_random_walk_finds_the_same_posterior_it_just_says_less():
    """Same target, the baseline sampler: the mean agrees; the effective sample size is a fraction of the gradient
    method's for the same number of draws, and that fraction is what the diagnostic reports."""
    posterior, mean, covariance = _conjugate()
    chains = sample(posterior, [0.0, 0.0], method="rwm", n_steps=4000, n_chains=4, seed=0)
    gradient = sample(posterior, [0.0, 0.0], method="hmc", n_steps=4000, n_chains=4, seed=0)
    assert np.allclose(chains.mean(), mean, atol=0.02)
    assert np.allclose(chains.sd(), np.sqrt(np.diag(covariance)), rtol=0.12)
    assert chains.diagnostics().least_ess < gradient.diagnostics().least_ess / 5.0
    assert mcse_mean(chains.draws[:, :, 0]) > 3.0 * mcse_mean(gradient.draws[:, :, 0])


def test_the_decay_posterior_is_the_one_the_quadrature_gives():
    """A nonlinear model with no closed form: the sampler against a 300 x 300 quadrature of the same posterior."""
    values = _synthetic(0)
    exact_mean, exact_bounds = _exact_decay(values)
    chains = sample(_decay(values), [1.0, 1.0], method="hmc", n_steps=2000, n_chains=4, seed=3)
    diagnostics = chains.diagnostics()
    assert diagnostics.converged
    assert np.allclose(chains.mean(), exact_mean, atol=5e-3)
    assert np.allclose(chains.interval(0.95), exact_bounds, atol=2e-2)
    assert np.all(np.array([mcse_mean(chains.draws[:, :, i]) for i in range(2)]) < 0.01)


# ---------------------------------------------------------------------------
# calibrated coverage, measured against the exact posterior
# ---------------------------------------------------------------------------

def test_the_credible_interval_covers_the_truth_as_often_as_the_exact_posterior_does():
    """60 synthetic datasets from the decay model, the 95 % interval of the sampler against the true parameters, and
    the interval of the exact posterior over the same data.

    Two numbers come out and both are reported: the sampler covers 91.7 % (a) and 95.0 % (b) -- matching the exact
    posterior's 95.0 % and 96.7 % to within two datasets -- and the exact posterior itself is at 95 % / 96.7 %, not
    at 95 % exactly, because a two-parameter nonlinear model with twelve points does not deliver the nominal level on
    the nose. What the loop shows is that the sampler reproduces the posterior, not that the posterior is perfect."""
    covered = np.zeros((60, 2), dtype=bool)
    exact_covered = np.zeros((60, 2), dtype=bool)
    for k in range(60):
        values = _synthetic(k)
        exact_bounds = _exact_decay(values)[1]
        bounds = sample(_decay(values), [1.0, 1.0], method="rwm", n_steps=3000, n_chains=2, seed=k).interval(0.95)
        covered[k] = (bounds[:, 0] <= TRUE) & (TRUE <= bounds[:, 1])
        exact_covered[k] = (exact_bounds[:, 0] <= TRUE) & (TRUE <= exact_bounds[:, 1])
    assert np.all(covered.mean(axis=0) >= 0.85), covered.mean(axis=0)
    assert np.all(covered.mean(axis=0) <= 1.0)
    assert np.all(np.abs(covered.sum(axis=0) - exact_covered.sum(axis=0)) <= 3), \
        (covered.sum(axis=0), exact_covered.sum(axis=0))


def test_a_chain_that_did_not_mix_says_so_instead_of_quoting_a_number():
    """Two ways a posterior is quoted too early, both caught by the diagnostics and neither by looking at the mean.

    A random walk with a proposal far too small sat at an acceptance of 88 % and an ESS of 3, quoting means whose
    standard error is five to eight times the one the converged run reaches. Two chains started far outside the
    posterior with no warm-up never reach it at all: R-hat 3.09 and the "mean" is 12.0, 9.0 -- the starting point."""
    posterior, mean, covariance = _conjugate()
    converged = sample(posterior, [0.0, 0.0], method="hmc", n_steps=1000, n_chains=2, seed=0)
    stuck = sample(posterior, [0.0, 0.0], method="rwm", n_steps=600, n_chains=2, seed=0, scale=0.05)
    stuck_diagnostics = stuck.diagnostics()
    assert not stuck_diagnostics.converged
    assert stuck.accept_rate > 0.8                       # a high acceptance rate is not a sign of anything good
    assert stuck_diagnostics.least_ess < 20.0 and stuck_diagnostics.worst_r_hat > 1.05
    assert "NOT converged" in stuck_diagnostics.verdict() and "ESS" in stuck_diagnostics.verdict()
    for i in range(2):
        assert mcse_mean(stuck.draws[:, :, i]) > 5.0 * mcse_mean(converged.draws[:, :, i])

    far_away = sample(posterior, [12.0, 9.0], n_steps=50, n_chains=2, seed=0, n_warmup=0)
    far_diagnostics = far_away.diagnostics()
    assert not far_diagnostics.converged and far_diagnostics.worst_r_hat > 3.0
    assert np.allclose(far_away.mean(), [12.0, 9.0])      # it never left the starting point
    assert not np.allclose(far_away.mean(), mean)


# ---------------------------------------------------------------------------
# the diagnostics, on draws whose effective sample size is known
# ---------------------------------------------------------------------------

def test_the_effective_sample_size_is_the_one_the_draws_have():
    rng = np.random.default_rng(0)
    independent = rng.normal(size=(4, 500))
    assert split_rhat(independent) == pytest.approx(1.0, abs=0.01)
    assert effective_sample_size(independent) == pytest.approx(2000, rel=0.15)      # 4 x 500 draws
    for phi, theory in ((0.5, 2000 * 0.5 / 1.5), (0.9, 2000 * 0.1 / 1.9)):
        chain = np.empty((4, 500))
        chain[:, 0] = rng.normal(size=4)
        for t in range(1, 500):
            chain[:, t] = phi * chain[:, t - 1] + rng.normal(size=4)
        assert effective_sample_size(chain) == pytest.approx(theory, rel=0.4), phi


def test_half_mixed_and_drifting_chains_are_caught():
    rng = np.random.default_rng(0)
    shifted = rng.normal(size=(4, 500))
    shifted[2:] += 3.0
    assert split_rhat(shifted) > 1.5
    drifting = np.cumsum(rng.normal(size=(2, 400)), axis=1)
    assert split_rhat(drifting) > 1.5
    assert effective_sample_size(shifted) < 2000


def test_the_standard_error_of_the_mean_follows_from_the_effective_sample_size():
    rng = np.random.default_rng(0)
    independent = rng.normal(size=(4, 500))
    assert mcse_mean(independent) == pytest.approx(independent.std() / np.sqrt(2000), rel=0.15)
    strongly_correlated = np.cumsum(rng.normal(size=(4, 500)), axis=1)
    assert mcse_mean(strongly_correlated) > 5.0 * mcse_mean(independent)


def test_the_diagnostic_carries_its_limits_and_reads_them():
    diagnostics = Diagnostics(r_hat=np.array([1.0, 1.2]), ess=np.array([900.0, 100.0]), names=("a", "b"),
                              n_chains=4, n_steps=500)
    assert not diagnostics.converged
    verdict = diagnostics.verdict()
    assert "R-hat" in verdict and "ESS" in verdict and "NOT converged" in verdict
    assert diagnostics.r_hat_limit == R_HAT_LIMIT and diagnostics.ess_limit == ESS_LIMIT
    body = diagnostics.to_dict()
    assert body["r_hat"] == {"a": 1.0, "b": 1.2} and body["converged"] is False
    assert "b" in diagnostics.table()


# ---------------------------------------------------------------------------
# the two samplers, on a target that separates them
# ---------------------------------------------------------------------------

def _correlated(dimension: int = 6, rho: float = 0.7):
    """A Gaussian whose parameters are correlated to ``rho``: a random walk crosses it slowly, a gradient does not."""
    covariance = rho ** np.abs(np.subtract.outer(np.arange(dimension), np.arange(dimension)))
    precision = np.linalg.inv(covariance)
    mean = np.full(dimension, 0.5)
    return Posterior(log_prob=lambda z: -0.5 * float((z - mean) @ precision @ (z - mean)), dimension=dimension,
                     grad_log_prob=lambda z: -precision @ (z - mean)), mean


def test_the_gradient_method_converges_where_the_random_walk_has_not():
    """The same target, the same number of steps and chains: measured, HMC's least ESS is 523 against 110, and only
    the gradient method passes its own diagnostics at this budget."""
    posterior, _ = _correlated()
    rwm = sample(posterior, np.zeros(6), method="rwm", n_steps=2000, n_chains=4, seed=0)
    hmc = sample(posterior, np.zeros(6), method="hmc", n_steps=2000, n_chains=4, seed=0)
    assert hmc.diagnostics().converged and not rwm.diagnostics().converged
    assert hmc.diagnostics().least_ess > 4.0 * rwm.diagnostics().least_ess
    assert hmc.diagnostics().least_ess >= ESS_LIMIT and rwm.diagnostics().least_ess < 300


def test_the_comparison_is_reported_per_call_and_not_only_per_step():
    """The other half of the same measurement: an HMC step is a leapfrog trajectory of gradients, and on a
    three-parameter problem with a central-difference gradient the random walk gets about as many effective draws per
    call (0.042 against 0.033 measured) while needing seventeen times more steps to reach the same ESS. The table
    carries both numbers so the trade is visible instead of implied."""
    posterior, _ = _correlated(dimension=3, rho=0.5)
    table = compare_methods(posterior, np.zeros(3), n_steps=600, n_chains=2, seed=0)
    assert set(table) == {"rwm", "hmc"}
    for row in table.values():
        assert row["work"] > 0 and row["ess_per_evaluation"] > 0.0
        assert isinstance(row["diagnostics"], Diagnostics)
    assert table["hmc"]["work"] > table["rwm"]["work"]            # a gradient costs calls
    assert table["hmc"]["least_ess"] > 5.0 * table["rwm"]["least_ess"]     # ... and buys effective draws
    assert table["rwm"]["ess_per_evaluation"] > 0.8 * table["hmc"]["ess_per_evaluation"]


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------

def test_the_gradient_falls_back_to_differences_and_agrees_with_the_analytic_one():
    posterior, _ = _correlated(dimension=2, rho=0.3)
    numeric = Posterior(log_prob=posterior.log_prob, dimension=2)
    point = np.array([0.2, 0.7])
    assert np.allclose(numeric.gradient(point), posterior.gradient(point), rtol=1e-6, atol=1e-6)


def test_the_same_seed_gives_the_same_draws():
    posterior, _ = _correlated(dimension=2, rho=0.3)
    first = sample(posterior, np.zeros(2), method="rwm", n_steps=200, n_chains=2, seed=7)
    second = sample(posterior, np.zeros(2), method="rwm", n_steps=200, n_chains=2, seed=7)
    third = sample(posterior, np.zeros(2), method="rwm", n_steps=200, n_chains=2, seed=8)
    assert np.array_equal(first.draws, second.draws)
    assert not np.array_equal(first.draws, third.draws)


def test_the_sampler_refuses_what_it_cannot_do():
    posterior, _ = _correlated(dimension=2, rho=0.3)
    with pytest.raises(ValueError, match="is not a sampler"):
        sample(posterior, np.zeros(2), method="nuts")
    with pytest.raises(ValueError, match="against 2 parameters"):
        sample(posterior, np.zeros(3))
    with pytest.raises(ValueError, match="four steps"):
        sample(posterior, np.zeros(2), n_steps=2)
    with pytest.raises(ValueError, match="nothing to sample"):
        Posterior(log_prob=lambda z: 0.0, dimension=0)
    with pytest.raises(ValueError, match="names against"):
        Posterior(log_prob=lambda z: 0.0, dimension=2, names=("a",))
    chain = Chains(draws=np.zeros((2, 10, 2)), names=("a", "b"), method="rwm", accept_rate=0.5)
    with pytest.raises(ValueError, match="not a probability"):
        chain.interval(1.5)


def test_the_summary_is_the_table_the_diagnostics_and_the_verdict():
    posterior, mean = _correlated(dimension=2, rho=0.3)
    chains = sample(posterior, np.zeros(2), method="hmc", n_steps=400, n_chains=2, seed=0)
    text = chains.summary()
    assert "hmc: 2 chain(s) x 400 step(s)" in text and "R-hat" in text and "MCSE" in text
    assert "converged" in text and "call(s) to the target" in text
    assert np.allclose(chains.mean(), mean, atol=0.15)
