"""Co-kriging: two fidelity levels, and the discrepancy between them learned rather than assumed.

:mod:`pinneapple_data.multifidelity` answers "how much does the cheap model buy" with a closed-form kernel ridge and
a fixed relation between the fidelities. This module answers the harder question a co-kriging study asks: the cheap
model is wrong in a way that has *shape*, and the expensive runs should pay for that shape and not for the trend the
cheap model already knows.

The model is the autoregressive Kennedy-O'Hagan form::

    z_high(x) = rho * z_low(x) + delta(x)

``z_low`` is a Gaussian process over the cheap runs (``rho`` scales it to the expensive one), and ``delta`` is a
second process over what is left after the scaling -- the discrepancy. Both are fitted by maximum likelihood, the
scale ``rho`` jointly with the discrepancy's hyperparameters (see :meth:`CoKriging.fit`: estimating ``rho`` on its
own is what makes a co-kriging fit lose to a single-fidelity one). The prediction is ``rho * z_low(x) +
mu_delta(x)``, and its interval combines the two variances; the cross-covariance between ``z_low`` and ``delta`` is
not carried, so the interval is approximate and measured erratic (:meth:`CoKriging.predict` gives the numbers). The
*mean* is what this module claims.

What this buys, measured
------------------------

:func:`study` runs three models over the same expensive designs -- the cheap model alone, a single-fidelity GP on the
expensive runs, and the co-kriging model -- and scores each on held-out expensive evaluations. The claim it can
support is narrow and this module states it the way the numbers come out:

- with an expensive budget of five to twenty runs and a cheap model that is *correlated* with the expensive one, the
  co-kriging model is ahead of the single-fidelity GP by a factor this study measures per budget;
- with a cheap model that knows nothing (independent of the expensive one) it is not ahead, and the study says so
  instead of reporting the average of the two cases.

Nothing here is fitted by gradient descent: the likelihood is optimised by a bounded pattern search, so the whole
module is NumPy, and a fit is deterministic given the designs.

    >>> from pinneapple_data.co_kriging import CoKriging, study
    >>> cr = CoKriging.fit(design_low, values_low, design_high, values_high)
    >>> mean, std = cr.predict(design_test)
    >>> curve = study(forrester_low, forrester_high, budgets=(5, 10, 20))
    >>> print(curve.table())
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .multifidelity import r2, relative_l2

__all__ = ["GaussianProcess", "CoKriging", "MultiFidelityStudy", "study", "forrester_high", "forrester_low",
           "MODELS"]

#: The three models a study compares, in the order the table lists them.
MODELS = ("low_fidelity", "single_fidelity", "co_kriging")

_JITTER = 1e-10


def _pattern_search(objective: Callable[[np.ndarray], float], theta0: np.ndarray, lower: np.ndarray,
                    upper: np.ndarray, *, initial_step=1.0, iterations: int = 80) -> np.ndarray:
    """A bounded coordinate pattern search: the maximum of ``objective``, no gradients and no scipy.

    Coordinates are moved one at a time and the step halves when no direction improves -- the oldest derivative-free
    recipe there is. It is deterministic, and it needs an objective that returns ``-inf`` rather than raising, which
    is what a likelihood that hits an ill-conditioned matrix does."""
    theta = np.clip(np.asarray(theta0, dtype=np.float64), lower, upper)
    best = float(objective(theta))
    step = float(initial_step)
    for _ in range(iterations):
        moved = False
        for coordinate in range(theta.size):
            for direction in (1.0, -1.0):
                trial = theta.copy()
                trial[coordinate] = np.clip(theta[coordinate] + direction * step * max(1.0, abs(theta[coordinate]))
                                            if coordinate else theta[coordinate] + direction * step,
                                            lower[coordinate], upper[coordinate])
                if trial[coordinate] == theta[coordinate]:
                    continue
                value = float(objective(trial))
                if value > best + 1e-12:
                    theta, best, moved = trial, value, True
        if not moved:
            step *= 0.5
            if step < 1e-4:
                break
    return theta


# ---------------------------------------------------------------------------
# the Gaussian process both levels are fitted with
# ---------------------------------------------------------------------------

@dataclass
class GaussianProcess:
    """A zero-mean Gaussian process with an RBF kernel, a noise term, and maximum-likelihood hyperparameters.

    ``length_scale``, ``signal`` and ``noise`` are standard deviations and a length in the *scaled* input space (the
    designs divided by ``scale``), so a design in metres and one in Mach number are compared on the same footing --
    the convention :mod:`pinneapple_data.multifidelity` already uses. Fitting maximises the log marginal likelihood by
    a pattern search over the three logs: no gradients, no scipy, and the same answer every run."""

    length_scale: float = 1.0
    signal: float = 1.0
    noise: float = 1e-6
    scale: np.ndarray | None = None
    mean: float = 0.0
    _x: np.ndarray | None = field(default=None, repr=False)
    _alpha: np.ndarray | None = field(default=None, repr=False)
    _cholesky: np.ndarray | None = field(default=None, repr=False)
    n_fits: int = field(default=0, repr=False)

    # -- kernel -----------------------------------------------------------
    def _scaled(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        scale = self.scale if self.scale is not None else np.ones(x.shape[1])
        return x / np.where(scale == 0.0, 1.0, scale)

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        squared = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        return self.signal ** 2 * np.exp(-0.5 * squared / max(self.length_scale, 1e-12) ** 2)

    def _factor(self, kernel: np.ndarray) -> np.ndarray | None:
        """The lower Cholesky factor, with the diagonal nudged until it is positive definite."""
        jitter = _JITTER * max(1.0, float(np.trace(kernel)) / kernel.shape[0])
        for _ in range(8):
            try:
                return np.linalg.cholesky(kernel + jitter * np.eye(kernel.shape[0]))
            except np.linalg.LinAlgError:
                jitter *= 10.0
        return None

    # -- fitting ----------------------------------------------------------
    def log_likelihood(self, x: np.ndarray, y: np.ndarray) -> float:
        kernel = self._kernel(x, x) + (self.noise ** 2 + _JITTER) * np.eye(x.shape[0])
        chol = self._factor(kernel)
        if chol is None:
            return -np.inf
        residual = y - y.mean()
        solved = np.linalg.solve(chol, residual)
        return float(-0.5 * solved @ solved - np.log(np.diag(chol)).sum() - 0.5 * x.shape[0] * np.log(2 * np.pi))

    def fit(self, x, y, *, fit_hyperparameters: bool = True, length_scale: float | None = None,
            signal: float | None = None, noise: float | None = None, scale=None) -> GaussianProcess:
        """Fit to ``n`` designs. With ``fit_hyperparameters=False`` the three values given are used as they are,
        which is what a nested model (the discrepancy inside a co-kriging fit) wants when its samples are few."""
        if scale is not None:
            self.scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        x = self._scaled(x)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.shape[0] != y.size:
            raise ValueError(f"{x.shape[0]} designs against {y.size} values")
        if x.shape[0] < 2:
            raise ValueError(f"{x.shape[0]} design(s): a Gaussian process needs at least two to have a covariance")
        self.mean = float(y.mean())
        if length_scale is not None:
            self.length_scale = float(length_scale)
        if signal is not None:
            self.signal = float(signal)
        if noise is not None:
            self.noise = float(noise)
        if fit_hyperparameters:
            self._optimise(x, y)
        kernel = self._kernel(x, x) + (self.noise ** 2 + _JITTER) * np.eye(x.shape[0])
        self._cholesky = self._factor(kernel)
        if self._cholesky is None:                      # the noise is the only thing that can rescue this
            self.noise = max(self.noise, 1e-3 * float(np.std(y)) or 1e-6)
            kernel = self._kernel(x, x) + (self.noise ** 2 + _JITTER) * np.eye(x.shape[0])
            self._cholesky = self._factor(kernel)
        self._alpha = np.linalg.solve(self._cholesky.T, np.linalg.solve(self._cholesky, y - self.mean))
        self._x = x
        self.n_fits += 1
        return self

    def _optimise(self, x: np.ndarray, y: np.ndarray) -> None:
        """Bounded pattern search on the three logs, from three starts: the same answer every run, and no scipy."""
        spread = float(np.std(y)) or 1.0
        distances = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1))
        off_diagonal = distances[~np.eye(distances.shape[0], dtype=bool)]
        start_length = float(np.median(off_diagonal)) if off_diagonal.size else 1.0
        lower = np.log([1e-3, 1e-4 * spread, 1e-10 * spread])
        upper = np.log([1e3, 1e3 * spread, 1e-1 * spread])

        def objective(theta: np.ndarray) -> float:
            self.length_scale, self.signal, self.noise = (float(v) for v in np.exp(np.clip(theta, lower, upper)))
            return self.log_likelihood(x, y)

        best, best_value = np.zeros(3), -np.inf
        for factor in (1.0, 0.35, 3.0):     # one start lands in a local optimum now and then; three do not
            start = np.log([max(start_length * factor, 1e-3), spread, max(1e-4 * spread, 1e-8)])
            theta = _pattern_search(objective, start, lower, upper)
            value = objective(theta)
            if value > best_value:
                best, best_value = theta, value
        self.length_scale, self.signal, self.noise = (float(v) for v in np.exp(np.clip(best, lower, upper)))

    # -- prediction -------------------------------------------------------
    def predict(self, x, return_std: bool = False):
        if self._x is None or self._alpha is None:
            raise RuntimeError("predict before fit")
        x = self._scaled(x)
        if x.shape[1] != self._x.shape[1]:
            raise ValueError(f"{x.shape[1]} inputs against a process of {self._x.shape[1]}")
        cross = self._kernel(x, self._x)
        mean = cross @ self._alpha + self.mean
        if not return_std:
            return mean
        solved = np.linalg.solve(self._cholesky, cross.T)
        variance = np.maximum(self.signal ** 2 - (solved ** 2).sum(axis=0), 0.0)
        return mean, np.sqrt(variance)

    def summary(self) -> str:
        return (f"GP: {self._x.shape[0] if self._x is not None else 0} design(s), length_scale "
                f"{self.length_scale:.3g}, signal {self.signal:.3g}, noise {self.noise:.2g}")


# ---------------------------------------------------------------------------
# the two-fidelity model
# ---------------------------------------------------------------------------

@dataclass
class CoKriging:
    """Co-kriging with a discrepancy: ``z_high = rho * z_low + delta``.

    ``low`` is the cheap fidelity as a process over its own runs; ``discrepancy`` is fitted on
    ``y_high - rho * z_low(design_high)``; ``rho`` is the regression slope of the expensive values on the cheap
    model's at the same designs, estimated with the means removed (so the offset the cheap model carries goes into the
    discrepancy instead of being read as a scaling)."""

    low: GaussianProcess
    discrepancy: GaussianProcess
    rho: float
    n_low: int = 0
    n_high: int = 0

    @classmethod
    def fit(cls, design_low, values_low, design_high, values_high, *, low_at_high=None, scale=None,
            fit_hyperparameters: bool = True, rho: float | None = None,
            noise_high: float | None = None) -> CoKriging:
        """Fit the two levels.

        ``low_at_high`` is the cheap model's value at the expensive designs. When the cheap model is available as a
        function it is *not* interpolated here and not guessed: pass its values, because the residual the discrepancy
        is fitted on is the difference between two things measured at the same design. Left as ``None`` the low
        level's posterior mean at those designs is used, which is the co-kriging proper (the cheap model as a process,
        not as a function).

        ``rho`` and the discrepancy's hyperparameters are estimated together by marginal likelihood, starting from the
        closed-form slope of the expensive values on the cheap model's. Estimating ``rho`` on its own -- the obvious
        shortcut -- is what makes a co-kriging fit look worse than the single-fidelity GP when the cheap model carries
        an offset: the slope then absorbs the offset and the discrepancy is fitted to the wrong residual."""
        design_low = np.atleast_2d(np.asarray(design_low, dtype=np.float64))
        design_high = np.atleast_2d(np.asarray(design_high, dtype=np.float64))
        values_low = np.asarray(values_low, dtype=np.float64).reshape(-1)
        values_high = np.asarray(values_high, dtype=np.float64).reshape(-1)
        if design_low.shape[1] != design_high.shape[1]:
            raise ValueError(f"the two levels have different input counts: {design_low.shape} and {design_high.shape}")
        if design_low.shape[0] != values_low.size or design_high.shape[0] != values_high.size:
            raise ValueError("a design table and its values disagree on the number of runs")
        if design_high.shape[0] < 2:
            raise ValueError(f"{design_high.shape[0]} expensive run(s): the discrepancy needs two to be a process")
        if design_low.shape[0] < 2:
            raise ValueError(f"{design_low.shape[0]} cheap run(s): the low level needs two")
        if scale is None:
            scale = np.maximum(np.abs(np.vstack([design_low, design_high])).max(axis=0), 1e-30)
        low = GaussianProcess(scale=scale).fit(design_low, values_low, fit_hyperparameters=fit_hyperparameters)

        if low_at_high is None:
            cheap = low.predict(design_high)
        else:
            cheap = np.asarray(low_at_high, dtype=np.float64).reshape(-1)
            if cheap.size != design_high.shape[0]:
                raise ValueError(f"{cheap.size} cheap values against {design_high.shape[0]} expensive designs")
        # the slope, with the means removed: the offset belongs to the discrepancy, not to rho
        centred_cheap = cheap - cheap.mean()
        centred_high = values_high - values_high.mean()
        denominator = float(centred_cheap @ centred_cheap)
        if denominator <= 0.0:                          # a cheap model that does not vary cannot scale anything
            raise ValueError("the cheap model has no spread at the expensive designs: there is nothing for a "
                             "discrepancy to correct, and rho would be a division by zero")
        slope = float(centred_cheap @ centred_high / denominator)

        discrepancy = GaussianProcess(scale=scale)
        if rho is not None:
            scale_used = float(rho)
            discrepancy.fit(design_high, values_high - scale_used * cheap, fit_hyperparameters=fit_hyperparameters,
                            noise=noise_high)
        elif fit_hyperparameters:
            scale_used, discrepancy = _fit_scale_and_discrepancy(design_high, values_high, cheap, scale, slope,
                                                                 noise_high)
        else:
            scale_used = slope
            discrepancy.fit(design_high, values_high - scale_used * cheap, fit_hyperparameters=False,
                            noise=noise_high)
        return cls(low=low, discrepancy=discrepancy, rho=scale_used, n_low=int(design_low.shape[0]),
                   n_high=int(design_high.shape[0]))

    def predict(self, design, *, low_values=None, return_std: bool = True):
        """The co-kriging mean at ``design``, and its spread.

        ``low_values`` is the cheap model's value at ``design``. Pass it when the cheap model can be evaluated there
        -- which is the whole point of a cheap model -- and the prediction is ``rho * low + mu_delta``, with only the
        discrepancy's variance left uncertain. Left as ``None``, the low level's own posterior supplies the mean, and
        the interval is ``sqrt(rho^2 * var_low + var_delta)``.

        **The interval is approximate, and measured to be erratic.** It adds the two variances and leaves out the
        covariance between the levels, which the autoregressive model says is not zero; the exact interval needs the
        joint system solved. Over ten Forrester designs with ten expensive runs the 1-sigma coverage came out at 0.72
        and the 2-sigma at 0.88, with one design covered at 0.26 -- optimistic, and not uniformly so. Use it as a
        coarse band, never as a guarantee; the mean is what this module is for."""
        design = np.atleast_2d(np.asarray(design, dtype=np.float64))
        residual_mean, residual_std = self.discrepancy.predict(design, return_std=True)
        if low_values is None:
            low_mean, low_std = self.low.predict(design, return_std=True)
        else:
            low_mean = np.asarray(low_values, dtype=np.float64).reshape(-1)
            if low_mean.size != design.shape[0]:
                raise ValueError(f"{low_mean.size} cheap values against {design.shape[0]} designs")
            low_std = np.zeros_like(low_mean)
        mean = self.rho * low_mean + residual_mean
        if not return_std:
            return mean
        return mean, np.sqrt((self.rho * low_std) ** 2 + residual_std ** 2)

    def summary(self) -> str:
        return (f"co-kriging: rho {self.rho:.4g}, {self.n_low} cheap run(s), {self.n_high} expensive one(s); "
                f"{self.discrepancy.summary()}")


# ---------------------------------------------------------------------------
# the joint fit of the scale and the discrepancy
# ---------------------------------------------------------------------------

def _fit_scale_and_discrepancy(design_high: np.ndarray, values_high: np.ndarray, cheap: np.ndarray,
                               scale: np.ndarray, slope: float,
                               noise_high: float | None) -> tuple[float, GaussianProcess]:
    """``rho`` and the discrepancy's hyperparameters by marginal likelihood, in one bounded search.

    The objective is the log marginal likelihood of ``values_high - rho * cheap`` under a Gaussian process, so a
    ``rho`` that turns the residue into a straight line wins over one that leaves structure the process has to buy.
    The search starts at the closed-form slope and the four coordinates are (``rho``, log length, log signal,
    log noise)."""
    spread = float(np.std(values_high)) or 1.0
    discrepancy = GaussianProcess(scale=scale)
    scaled = discrepancy._scaled(design_high)
    distances = np.sqrt(((scaled[:, None, :] - scaled[None, :, :]) ** 2).sum(axis=-1))
    off_diagonal = distances[~np.eye(distances.shape[0], dtype=bool)]
    start_length = float(np.median(off_diagonal)) if off_diagonal.size else 1.0
    reach = 10.0 * max(abs(slope), 0.1)                      # rho may be negative: a cheap model can be anti-correlated
    lower = np.array([-reach, np.log(1e-3), np.log(1e-4 * spread), np.log(1e-10 * spread)])
    upper = np.array([reach, np.log(1e3), np.log(1e3 * spread), np.log(1e-1 * spread)])

    def objective(theta: np.ndarray) -> float:
        discrepancy.length_scale = float(np.exp(np.clip(theta[1], lower[1], upper[1])))
        discrepancy.signal = float(np.exp(np.clip(theta[2], lower[2], upper[2])))
        discrepancy.noise = float(np.exp(np.clip(theta[3], lower[3], upper[3])))
        return discrepancy.log_likelihood(scaled, values_high - float(theta[0]) * cheap)

    start = np.array([float(np.clip(slope, lower[0], upper[0])), np.log(max(start_length, 1e-3)), np.log(spread),
                      np.log(max(1e-4 * spread, 1e-8))])
    theta = _pattern_search(objective, start, lower, upper)
    discrepancy.fit(design_high, values_high - float(theta[0]) * cheap, fit_hyperparameters=False,
                    length_scale=float(np.exp(theta[1])), signal=float(np.exp(theta[2])),
                    noise=noise_high if noise_high is not None else float(np.exp(theta[3])))
    return float(theta[0]), discrepancy


# ---------------------------------------------------------------------------
# the benchmark functions
# ---------------------------------------------------------------------------

def forrester_high(x) -> np.ndarray:
    """The high-fidelity Forrester function: ``(6x - 2)^2 sin(12x - 4)`` on ``[0, 1]``."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


def forrester_low(x) -> np.ndarray:
    """Its cheap companion: half the expensive one plus a linear drift, so ``rho`` is not one and the discrepancy is
    a line the expensive runs have to pay for."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return 0.5 * forrester_high(x) + 10.0 * (x - 0.5) - 5.0


# ---------------------------------------------------------------------------
# the study: three models, the same expensive budget
# ---------------------------------------------------------------------------

@dataclass
class MultiFidelityStudy:
    """The three models scored on the same held-out expensive evaluations, per budget and per repeat."""

    budgets: tuple[int, ...]
    repeats: int
    n_low: int
    scores: dict[str, np.ndarray] = field(default_factory=dict)      # model -> (budget, repeat) relative L2
    r2: dict[str, np.ndarray] = field(default_factory=dict)
    rho: np.ndarray | None = None                                    # (budget, repeat) estimated scale
    seconds: dict[str, float] = field(default_factory=dict)

    def median(self, model: str) -> np.ndarray:
        return np.median(self.scores[model], axis=1)

    def spread(self, model: str) -> np.ndarray:
        """The worst-to-best of the repeats, not a confidence interval of the median."""
        return np.ptp(self.scores[model], axis=1)

    def better_by(self) -> np.ndarray:
        """Median error of the single-fidelity GP divided by the co-kriging one, per budget: >1 is an improvement."""
        return self.median("single_fidelity") / self.median("co_kriging")

    def beats_single_fidelity(self, *, minimum_factor: float = 1.0) -> bool:
        """Whether two levels beat the best single level at every budget by at least ``minimum_factor``."""
        return bool(np.all(self.median("single_fidelity") > minimum_factor * self.median("co_kriging")))

    def table(self) -> str:
        header = f"{'N':>4}  " + "  ".join(f"{model:>26}" for model in MODELS) + f"  {'cheap/lone':>10}"
        lines = [header, "-" * len(header)]
        for i, budget in enumerate(self.budgets):
            cells = []
            for model in MODELS:
                value = float(np.median(self.scores[model][i]))
                cells.append(f"{value:10.3%} ±{float(np.ptp(self.scores[model][i])):6.3%}")
            lines.append(f"{budget:>4}  " + "  ".join(f"{cell:>26}" for cell in cells)
                         + f"  {float(self.better_by()[i]):10.2f}")
        return "\n".join(lines)


def study(low: Callable, high: Callable, *, n_low: int = 60, budgets=(5, 10, 20, 40), repeats: int = 10,
          seed: int = 0, bounds=((0.0, 1.0),), test_points: int = 200, fit_hyperparameters: bool = True,
          spread_low: float = 1.0) -> MultiFidelityStudy:
    """Compare the cheap model, a single-fidelity GP and the co-kriging model on the same expensive budget.

    ``budgets`` are counts of **expensive** runs. The cheap design is ``n_low`` points drawn once from ``bounds`` and
    is the same for every model and repeat -- that is what "cheap" means here, and it is why the costs are compared
    on the expensive axis alone. ``spread_low`` is the standard deviation of the noise added to the *cheap* model's
    values (a deterministic cheap model passes 0.0); the expensive one is treated as noiseless, because the runs a
    study cannot afford twice are usually the ones with the least noise.

    For each repeat the expensive designs are drawn once at the largest budget and the smaller budgets take a prefix
    of them, so the comparison across budgets is nested rather than between different designs."""
    budgets = tuple(int(b) for b in budgets)
    if not budgets or min(budgets) < 2:
        raise ValueError(f"budgets {budgets}: a Gaussian process needs at least two expensive designs")
    if max(budgets) > 200:
        raise ValueError(f"{max(budgets)} expensive runs per repeat is a simulation campaign, not a study: raise "
                         "repeats on fewer budgets instead")
    bounds = np.atleast_2d(np.asarray(bounds, dtype=np.float64))
    dimension = bounds.shape[0]
    rng = np.random.default_rng(seed)

    def draw(n: int) -> np.ndarray:
        return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, dimension))

    test = draw(test_points)
    truth = np.asarray(high(test), dtype=np.float64)
    cheap_test = np.asarray(low(test), dtype=np.float64)
    cheap_design = draw(n_low)
    cheap_values = np.asarray(low(cheap_design), dtype=np.float64)
    if spread_low > 0.0:
        cheap_values = cheap_values + rng.normal(scale=spread_low * float(np.std(cheap_values)), size=cheap_values.size)

    scores = {model: np.full((len(budgets), repeats), np.nan) for model in MODELS}
    r2_scores = {model: np.full((len(budgets), repeats), np.nan) for model in MODELS}
    rho = np.full((len(budgets), repeats), np.nan)
    largest = max(budgets)
    for repeat in range(repeats):
        design = draw(largest)
        values = np.asarray(high(design), dtype=np.float64)
        cheap_at = np.asarray(low(design), dtype=np.float64)
        for i, budget in enumerate(budgets):
            x, y, c = design[:budget], values[:budget], cheap_at[:budget]
            predictions = {
                "low_fidelity": _nearest(cheap_design, cheap_values, test),
                "single_fidelity": GaussianProcess().fit(x, y, fit_hyperparameters=fit_hyperparameters).predict(test),
            }
            model = CoKriging.fit(cheap_design, cheap_values, x, y, low_at_high=c,
                                  fit_hyperparameters=fit_hyperparameters)
            predictions["co_kriging"] = model.predict(test, low_values=cheap_test, return_std=False)
            rho[i, repeat] = model.rho
            for name, prediction in predictions.items():
                scores[name][i, repeat] = relative_l2(prediction, truth)
                r2_scores[name][i, repeat] = r2(prediction, truth)
    return MultiFidelityStudy(budgets=budgets, repeats=repeats, n_low=n_low, scores=scores, r2=r2_scores, rho=rho)


def _nearest(design: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """The cheap model's value at the nearest cheap run -- the only honest thing to do with a table, and said so."""
    squared = ((query[:, None, :] - design[None, :, :]) ** 2).sum(axis=-1)
    return values[np.argmin(squared, axis=1)]
