"""Generalized extreme value (GEV) and generalized Pareto (GPD) models fitted by maximum likelihood, with
covariates in the GEV location and log-scale, return levels, delta-method and profile-likelihood intervals.

Parametrisation and sign convention follow Coles (2001), *An Introduction to Statistical Modeling of Extreme
Values*: the GEV distribution function is ``G(z) = exp{-[1 + xi (z - mu) / sigma]^(-1/xi)}`` and the GPD
survival function of an excess ``y`` over the threshold is ``(1 + xi y / sigma)^(-1/xi)``. A positive shape
``xi`` is a heavy (Frechet) tail, a negative one a bounded (Weibull) tail; ``xi = 0`` is the Gumbel / exponential
limit. (SciPy's ``genextreme`` uses ``c = -xi``.)

Non-stationary GEV: ``mu_i = b0 + sum_j b_j x_ij`` and ``log sigma_i = g0 + sum_j g_j x_ij`` on named covariates;
the shape is constant. Covariates are centred and scaled inside the optimiser and the coefficients are reported in
the covariates' own units.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize, stats

__all__ = ["GEVFit", "GPDFit", "fit_gev", "fit_gpd", "gev_nll", "gpd_nll", "gev_return_level", "gpd_return_level",
           "gev_sample", "gpd_sample", "profile_interval", "likelihood_ratio_test", "anderson_darling"]

_XI0 = 1e-6          # |xi| below this is the Gumbel / exponential limit


# ---------------------------------------------------------------------------
# Densities
# ---------------------------------------------------------------------------

def gev_nll(z, mu, sigma, xi) -> float:
    """Negative log-likelihood of GEV observations ``z`` (``mu`` and ``sigma`` may be arrays)."""
    z, mu, sigma = np.asarray(z, float), np.asarray(mu, float), np.asarray(sigma, float)
    if np.any(sigma <= 0) or not np.isfinite(xi):
        return np.inf
    y = (z - mu) / sigma
    if abs(xi) < _XI0:
        return float(np.sum(np.log(sigma) + y + np.exp(-y)))
    t = 1.0 + xi * y
    if np.any(t <= 0):
        return np.inf
    lt = np.log(t)
    return float(np.sum(np.log(sigma) + (1.0 + 1.0 / xi) * lt + np.exp(-lt / xi)))


def gpd_nll(y, sigma, xi) -> float:
    """Negative log-likelihood of GPD excesses ``y > 0``."""
    y = np.asarray(y, float)
    if sigma <= 0 or not np.isfinite(xi):
        return np.inf
    if abs(xi) < _XI0:
        return float(len(y) * np.log(sigma) + np.sum(y) / sigma)
    t = 1.0 + xi * y / sigma
    if np.any(t <= 0):
        return np.inf
    return float(len(y) * np.log(sigma) + (1.0 + 1.0 / xi) * np.sum(np.log(t)))


def _yp(p):
    return -np.log1p(-np.asarray(p, float))


def gev_return_level(T, mu, sigma, xi):
    """Level exceeded with probability ``1/T`` per block: ``mu - sigma/xi [1 - y_p^-xi]``, ``y_p = -log(1 - 1/T)``."""
    yp = _yp(1.0 / np.asarray(T, float))
    if abs(xi) < _XI0:
        return mu - sigma * np.log(yp)
    return mu - sigma / xi * (1.0 - yp ** (-xi))


def gpd_return_level(m, threshold, sigma, xi, rate):
    """Level exceeded once every ``m`` observations: ``u + sigma/xi [(m zeta)^xi - 1]`` with ``zeta`` the
    probability that an observation is a (cluster) exceedance."""
    mz = np.asarray(m, float) * rate
    if abs(xi) < _XI0:
        return threshold + sigma * np.log(mz)
    return threshold + sigma / xi * (mz ** xi - 1.0)


def gev_sample(n, mu, sigma, xi, rng: np.random.Generator):
    """GEV draws by inversion (``mu`` and ``sigma`` may be arrays of length ``n``)."""
    u = rng.uniform(size=n)
    return _gev_ppf(u, mu, sigma, xi)


def _gev_ppf(q, mu, sigma, xi):
    yq = -np.log(np.asarray(q, float))
    if abs(xi) < _XI0:
        return mu - sigma * np.log(yq)
    return mu + sigma / xi * (yq ** (-xi) - 1.0)


def _gev_cdf(z, mu, sigma, xi):
    y = (np.asarray(z, float) - mu) / sigma
    if abs(xi) < _XI0:
        return np.exp(-np.exp(-y))
    t = np.maximum(1.0 + xi * y, 0.0)
    with np.errstate(divide="ignore"):
        return np.where(t > 0, np.exp(-t ** (-1.0 / xi)), 0.0 if xi > 0 else 1.0)


def gpd_sample(n, sigma, xi, rng: np.random.Generator):
    u = rng.uniform(size=n)
    if abs(xi) < _XI0:
        return -sigma * np.log(u)
    return sigma / xi * (u ** (-xi) - 1.0)


# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------

def _minimise(f: Callable[[np.ndarray], float], starts: Sequence[np.ndarray], *, first_only: bool = False
              ) -> tuple[np.ndarray, float, bool]:
    """Nelder-Mead from each start, polished by a restart from the best point; the best finite optimum.
    ``first_only``: stop at the first start with a finite likelihood (a warm-started profile point)."""
    best = None
    opts = dict(xatol=1e-9, fatol=1e-11, maxiter=40000, maxfev=40000)
    for x0 in starts:
        x0 = np.asarray(x0, float)
        if not np.isfinite(f(x0)):
            continue
        r = optimize.minimize(f, x0, method="Nelder-Mead", options=opts)
        if not first_only:
            r = optimize.minimize(f, r.x, method="Nelder-Mead", options=opts)
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
        if first_only and best is not None:
            break
    if best is None:
        raise ValueError("no start value gives a finite likelihood")
    return best.x, float(best.fun), bool(best.success)


def _profile_min(f, starts) -> tuple[np.ndarray | None, float]:
    """:func:`_minimise` for a profile point: +inf (outside the support) instead of an error."""
    try:
        x, val, _ = _minimise(f, starts, first_only=True)
    except ValueError:
        return None, np.inf
    return x, val


def _hessian(f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    k = len(x)
    h = 1e-4 * np.maximum(1.0, np.abs(x))
    H = np.empty((k, k))
    f0 = f(x)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def _covariance(f, x) -> np.ndarray:
    H = _hessian(f, x)
    try:
        C = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full((len(x), len(x)), np.nan)
    if not np.all(np.isfinite(C)) or np.any(np.diag(C) <= 0):
        return np.full((len(x), len(x)), np.nan)
    return C


def profile_interval(dev: Callable[[float], float], center: float, scale: float, *, level: float = 0.95,
                     max_steps: int = 60) -> tuple[float, float]:
    """Interval ``{v : dev(v) <= chi2_1(level)}`` of a profile deviance ``dev(v) = 2 (nll_profile(v) - nll_min)``.

    Steps outward from ``center`` by ``scale / 2`` until the deviance exceeds the cut, then solves with Brent's
    method. A side that never crosses within ``max_steps`` steps is returned as +-inf (the likelihood is flat
    there, as for the upper return level of a heavy tail with few data).
    """
    cut = stats.chi2.ppf(level, 1)
    g = lambda v: dev(v) - cut  # noqa: E731
    step = scale / 2.0 if np.isfinite(scale) and scale > 0 else max(abs(center), 1.0) * 0.05
    out = []
    for sgn in (-1.0, 1.0):
        a, b = center, center + sgn * step
        root = sgn * np.inf
        for _ in range(max_steps):
            gb = g(b)
            if np.isfinite(gb) and gb > 0:
                lo, hi = (b, a) if sgn < 0 else (a, b)
                root = optimize.brentq(g, lo, hi, xtol=1e-10 * max(1.0, abs(center)) + 1e-12)
                break
            if not np.isfinite(gb):          # outside the support: halve back toward the centre
                b = a + (b - a) / 2.0
                if abs(b - a) < 1e-12:
                    break
                continue
            a, b = b, b + sgn * step
            step *= 1.3
        out.append(float(root))
    return out[0], out[1]


def likelihood_ratio_test(nll_small: float, nll_big: float, df: int) -> dict[str, float]:
    """Deviance ``D = 2 (nll_small - nll_big)`` against chi-square with ``df`` degrees of freedom."""
    D = 2.0 * (nll_small - nll_big)
    return {"deviance": float(D), "df": int(df), "p_value": float(stats.chi2.sf(max(D, 0.0), df))}


def anderson_darling(u) -> float:
    """Anderson-Darling statistic ``A^2`` of probability-integral-transformed values ``u`` in (0, 1)."""
    u = np.clip(np.sort(np.asarray(u, float)), 1e-12, 1 - 1e-12)
    n = len(u)
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(u) + np.log1p(-u[::-1]))))


# ---------------------------------------------------------------------------
# GEV
# ---------------------------------------------------------------------------

@dataclass
class GEVFit:
    """A fitted (possibly non-stationary) GEV. ``theta`` = [location coefficients, log-scale coefficients, xi] on
    the internally standardised covariates; use the methods, which take covariates in their own units."""
    theta: np.ndarray
    cov: np.ndarray
    nll: float
    n: int
    mu_covariates: tuple[str, ...]
    sigma_covariates: tuple[str, ...]
    center: dict[str, float]
    scale: dict[str, float]
    converged: bool
    data: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    covariates: dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    # -- structure ----------------------------------------------------------------
    @property
    def k_mu(self) -> int:
        return 1 + len(self.mu_covariates)

    @property
    def k_sigma(self) -> int:
        return 1 + len(self.sigma_covariates)

    @property
    def n_params(self) -> int:
        return self.k_mu + self.k_sigma + 1

    @property
    def xi(self) -> float:
        return float(self.theta[-1])

    @property
    def stationary(self) -> bool:
        return not self.mu_covariates and not self.sigma_covariates

    def _design(self, names, x: Mapping[str, Any] | None, n: int | None = None) -> np.ndarray:
        cols = []
        for nm in names:
            if x is None or nm not in x:
                raise ValueError(f"covariate {nm!r} needed")
            cols.append((np.asarray(x[nm], float) - self.center[nm]) / self.scale[nm])
        if not cols:
            return np.ones((1 if n is None else n, 1))
        cols = np.broadcast_arrays(*cols)
        return np.column_stack([np.ones(cols[0].size)] + [c.ravel() for c in cols])

    def mu(self, x: Mapping[str, Any] | None = None, theta=None) -> np.ndarray:
        th = self.theta if theta is None else theta
        return self._design(self.mu_covariates, x) @ th[: self.k_mu]

    def sigma(self, x: Mapping[str, Any] | None = None, theta=None) -> np.ndarray:
        th = self.theta if theta is None else theta
        return np.exp(self._design(self.sigma_covariates, x) @ th[self.k_mu: self.k_mu + self.k_sigma])

    def coefficients(self) -> dict[str, float]:
        """Location and scale coefficients in the covariates' own units (``mu = mu0 + sum b_j x_j`` and
        ``log sigma = log_sigma0 + sum g_j x_j``), and the shape."""
        out: dict[str, float] = {}
        b = self.theta[: self.k_mu]
        g = self.theta[self.k_mu: self.k_mu + self.k_sigma]
        out["mu0"] = float(b[0] - sum(b[j + 1] * self.center[c] / self.scale[c] for j, c in enumerate(self.mu_covariates)))
        for j, c in enumerate(self.mu_covariates):
            out[f"mu_{c}"] = float(b[j + 1] / self.scale[c])
        ls0 = float(g[0] - sum(g[j + 1] * self.center[c] / self.scale[c] for j, c in enumerate(self.sigma_covariates)))
        if self.sigma_covariates:
            out["log_sigma0"] = ls0
            for j, c in enumerate(self.sigma_covariates):
                out[f"log_sigma_{c}"] = float(g[j + 1] / self.scale[c])
        else:
            out["sigma"] = float(np.exp(ls0))
        out["xi"] = self.xi
        return out

    def standard_errors(self) -> dict[str, float]:
        """Delta-method standard errors of :meth:`coefficients` from the observed information."""
        names = list(self.coefficients())
        f = lambda th: np.array(list(_coef_of(self, th).values()))  # noqa: E731
        J = _jacobian(f, self.theta)
        V = J @ self.cov @ J.T
        return {nm: float(np.sqrt(V[i, i])) if V[i, i] > 0 else float("nan") for i, nm in enumerate(names)}

    # -- results ------------------------------------------------------------------
    def return_level(self, T, x: Mapping[str, Any] | None = None, theta=None):
        th = self.theta if theta is None else theta
        return gev_return_level(T, float(self.mu(x, th)[0]), float(self.sigma(x, th)[0]), float(th[-1]))

    def return_level_se(self, T, x: Mapping[str, Any] | None = None) -> float:
        g = _jacobian(lambda th: np.atleast_1d(self.return_level(T, x, th)), self.theta)[0]
        v = float(g @ self.cov @ g)
        return float(np.sqrt(v)) if v > 0 else float("nan")

    def exceedance_probability(self, z, x: Mapping[str, Any] | None = None, theta=None) -> float:
        th = self.theta if theta is None else theta
        return float(1.0 - _gev_cdf(z, float(self.mu(x, th)[0]), float(self.sigma(x, th)[0]), float(th[-1])))

    def cdf_of_data(self) -> np.ndarray:
        """``G_i(z_i)`` of every observation under its own covariates (the PIT used by the diagnostics)."""
        mu = self._design(self.mu_covariates, self.covariates, self.n) @ self.theta[: self.k_mu]
        sg = np.exp(self._design(self.sigma_covariates, self.covariates, self.n)
                    @ self.theta[self.k_mu: self.k_mu + self.k_sigma])
        return np.array([_gev_cdf(z, m, s, self.xi) for z, m, s in zip(self.data, mu, sg)], float)

    def gumbel_residuals(self) -> np.ndarray:
        """Observations standardised to the standard Gumbel scale (Coles 2001, §6.2.3): the QQ plot of these
        against ``-log(-log(i / (n + 1)))`` checks a non-stationary fit the same way as a stationary one."""
        return -np.log(-np.log(np.clip(self.cdf_of_data(), 1e-300, 1 - 1e-16)))

    def profile_return_level(self, T, x: Mapping[str, Any] | None = None, level: float = 0.95
                             ) -> tuple[float, float]:
        """Profile-likelihood interval of the ``T``-block return level at covariates ``x``: the location
        intercept is eliminated through ``mu(x) = z + sigma(x)/xi [1 - y_p^-xi]`` and every other parameter is
        maximised at each fixed ``z``."""
        p = 1.0 / float(T)
        yp = float(_yp(p))
        xm = self._design(self.mu_covariates, x)[0]
        xs = self._design(self.sigma_covariates, x)[0]
        nll = _gev_objective(self)

        def theta_of(z, phi):
            b_rest, g, xi = phi[: self.k_mu - 1], phi[self.k_mu - 1: self.k_mu - 1 + self.k_sigma], phi[-1]
            s = np.exp(xs @ g)
            mux = z + (s * np.log(yp) if abs(xi) < _XI0 else s / xi * (1.0 - yp ** (-xi)))
            b0 = mux - xm[1:] @ b_rest
            return np.concatenate([[b0], b_rest, g, [xi]])

        state = {"phi": np.delete(self.theta, 0)}

        def prof(z):
            f = lambda phi: nll(theta_of(z, phi))  # noqa: E731
            phi, val = _profile_min(f, [state["phi"], np.delete(self.theta, 0), self._wide_start()[1:]])
            if phi is not None:
                state["phi"] = phi
            return val

        z_hat = float(self.return_level(T, x))
        dev = lambda z: 2.0 * (prof(z) - self.nll)  # noqa: E731
        se = self.return_level_se(T, x)
        lo, _ = profile_interval(dev, z_hat, se, level=level)
        state["phi"] = np.delete(self.theta, 0)
        _, hi = profile_interval(dev, z_hat, se, level=level)
        return lo, hi

    def _wide_start(self) -> np.ndarray:
        """Stationary start with three times the moment scale: inside the support for any moderate shape."""
        th = np.zeros(self.n_params)
        s0 = np.sqrt(6.0) * np.std(self.data) / np.pi
        th[0], th[self.k_mu], th[-1] = np.mean(self.data), np.log(3 * s0), self.xi
        return th

    def profile_shape(self, level: float = 0.95) -> tuple[float, float]:
        nll = _gev_objective(self)
        state = {"phi": self.theta[:-1].copy()}

        def prof(xi):
            phi, val = _profile_min(lambda ph: nll(np.concatenate([ph, [xi]])),
                                    [state["phi"], self.theta[:-1], self._wide_start()[:-1]])
            if phi is not None:
                state["phi"] = phi
            return val

        se = float(np.sqrt(self.cov[-1, -1])) if np.isfinite(self.cov[-1, -1]) else 0.1
        dev = lambda v: 2.0 * (prof(v) - self.nll)  # noqa: E731
        lo, _ = profile_interval(dev, self.xi, se, level=level)
        state["phi"] = self.theta[:-1].copy()
        _, hi = profile_interval(dev, self.xi, se, level=level)
        return lo, hi


def _coef_of(fit: GEVFit, theta) -> dict[str, float]:
    tmp = GEVFit(np.asarray(theta, float), fit.cov, fit.nll, fit.n, fit.mu_covariates, fit.sigma_covariates,
                 fit.center, fit.scale, fit.converged)
    return tmp.coefficients()


def _jacobian(f, x) -> np.ndarray:
    x = np.asarray(x, float)
    f0 = np.atleast_1d(f(x))
    J = np.empty((len(f0), len(x)))
    for i in range(len(x)):
        h = 1e-6 * max(1.0, abs(x[i]))
        e = np.zeros(len(x))
        e[i] = h
        J[:, i] = (np.atleast_1d(f(x + e)) - np.atleast_1d(f(x - e))) / (2 * h)
    return J


def _gev_objective(fit: GEVFit) -> Callable[[np.ndarray], float]:
    Xm = fit._design(fit.mu_covariates, fit.covariates, fit.n)
    Xs = fit._design(fit.sigma_covariates, fit.covariates, fit.n)
    z = fit.data
    km, ks = fit.k_mu, fit.k_sigma

    def f(th):
        th = np.asarray(th, float)
        return gev_nll(z, Xm @ th[:km], np.exp(Xs @ th[km: km + ks]), th[-1])
    return f


def fit_gev(z, covariates: Mapping[str, Any] | None = None, *, mu: Sequence[str] = (), sigma: Sequence[str] = (),
            start: GEVFit | None = None) -> GEVFit:
    """Maximum-likelihood GEV fit of block maxima ``z``, with ``mu`` / ``sigma`` naming the covariates (from
    ``covariates``) of the location and of the log-scale.

    Starts: the Gumbel method of moments with ``xi`` in {-0.2, 0.05, 0.3}, plus ``start`` (a nested fit whose
    parameters seed the larger model with zero slopes), so a larger model is never fitted worse than the model it
    contains.
    """
    z = np.asarray(z, float).ravel()
    if len(z) < 5 or not np.all(np.isfinite(z)):
        raise ValueError("a GEV fit needs at least 5 finite block maxima")
    covariates = {k: np.asarray(v, float).ravel() for k, v in (covariates or {}).items()}
    names = tuple(dict.fromkeys(list(mu) + list(sigma)))
    for nm in names:
        if nm not in covariates or len(covariates[nm]) != len(z):
            raise ValueError(f"covariate {nm!r} must have one value per block maximum")
    center = {nm: float(np.mean(covariates[nm])) for nm in names}
    scale = {nm: float(np.std(covariates[nm]) or 1.0) for nm in names}
    fit = GEVFit(np.zeros(len(mu) + len(sigma) + 3), np.zeros((1, 1)), np.inf, len(z), tuple(mu), tuple(sigma),
                 center, scale, False, z, {nm: covariates[nm] for nm in names})
    f = _gev_objective(fit)
    s0 = np.sqrt(6.0) * np.std(z) / np.pi
    m0 = np.mean(z) - 0.5772 * s0
    starts = []
    for xi0 in (-0.2, 0.05, 0.3):
        th = np.zeros(fit.n_params)
        th[0], th[fit.k_mu], th[-1] = m0, np.log(s0), xi0
        starts.append(th)
    if start is not None:
        th = np.zeros(fit.n_params)
        th[0], th[fit.k_mu], th[-1] = start.theta[0], start.theta[start.k_mu], start.xi
        for j, c in enumerate(fit.mu_covariates):
            if c in start.mu_covariates:
                th[1 + j] = start.theta[1 + start.mu_covariates.index(c)]
        for j, c in enumerate(fit.sigma_covariates):
            if c in start.sigma_covariates:
                th[fit.k_mu + 1 + j] = start.theta[start.k_mu + 1 + start.sigma_covariates.index(c)]
        starts.insert(0, th)
    theta, val, ok = _minimise(f, starts)
    fit.theta, fit.nll, fit.converged = theta, val, ok
    fit.cov = _covariance(f, theta)
    return fit


# ---------------------------------------------------------------------------
# GPD
# ---------------------------------------------------------------------------

@dataclass
class GPDFit:
    """A GPD fitted to the excesses over ``threshold``; ``rate`` is the probability that an observation is a
    (cluster) exceedance and ``rate_var`` its binomial variance."""
    sigma: float
    xi: float
    cov: np.ndarray
    nll: float
    threshold: float
    n_exceed: int
    n_obs: int
    rate: float
    converged: bool
    excess: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    @property
    def rate_var(self) -> float:
        return self.rate * (1 - self.rate) / max(self.n_obs, 1)

    @property
    def modified_scale(self) -> float:
        """``sigma* = sigma - xi u``: constant above a valid threshold (the threshold-stability plot)."""
        return self.sigma - self.xi * self.threshold

    def return_level(self, m, sigma=None, xi=None, rate=None):
        return gpd_return_level(m, self.threshold, self.sigma if sigma is None else sigma,
                                self.xi if xi is None else xi, self.rate if rate is None else rate)

    def return_level_se(self, m) -> float:
        """Delta method with the rate's binomial variance (Coles 2001, §4.3.3)."""
        th = np.array([self.rate, self.sigma, self.xi])
        V = np.zeros((3, 3))
        V[0, 0] = self.rate_var
        V[1:, 1:] = self.cov
        g = _jacobian(lambda t: np.atleast_1d(self.return_level(m, t[1], t[2], t[0])), th)[0]
        v = float(g @ V @ g)
        return float(np.sqrt(v)) if v > 0 else float("nan")

    def profile_return_level(self, m, level: float = 0.95) -> tuple[float, float]:
        """Profile-likelihood interval of the ``m``-observation return level (the rate held at its estimate):
        ``sigma = (x_m - u) xi / ((m zeta)^xi - 1)`` and ``xi`` is maximised at each fixed ``x_m``."""
        mz = float(m) * self.rate
        u = self.threshold

        def sig(x, xi):
            return (x - u) / np.log(mz) if abs(xi) < _XI0 else (x - u) * xi / (mz ** xi - 1.0)

        state = {"xi": self.xi}

        def prof(x):
            f = lambda v: gpd_nll(self.excess, sig(x, v[0]), v[0]) if sig(x, v[0]) > 0 else np.inf  # noqa: E731
            v, val = _profile_min(f, [np.array([state["xi"]]), np.array([self.xi]), np.array([0.0])])
            if v is not None:
                state["xi"] = float(v[0])
            return val

        x_hat = float(self.return_level(m))
        dev = lambda x: 2.0 * (prof(x) - self.nll)  # noqa: E731
        se = self.return_level_se(m)
        lo, _ = profile_interval(dev, x_hat, se, level=level)
        state["xi"] = self.xi
        _, hi = profile_interval(dev, x_hat, se, level=level)
        return lo, hi

    def profile_shape(self, level: float = 0.95) -> tuple[float, float]:
        state = {"ls": np.log(self.sigma)}

        def prof(xi):
            wide = np.log(3 * np.max(self.excess) * max(abs(xi), 1.0) + 3 * self.sigma)
            v, val = _profile_min(lambda p: gpd_nll(self.excess, np.exp(p[0]), xi),
                                  [np.array([state["ls"]]), np.array([wide])])
            if v is not None:
                state["ls"] = float(v[0])
            return val

        se = float(np.sqrt(self.cov[1, 1])) if np.isfinite(self.cov[1, 1]) else 0.1
        dev = lambda v: 2.0 * (prof(v) - self.nll)  # noqa: E731
        lo, _ = profile_interval(dev, self.xi, se, level=level)
        state["ls"] = np.log(self.sigma)
        _, hi = profile_interval(dev, self.xi, se, level=level)
        return lo, hi

    def cdf_of_data(self) -> np.ndarray:
        y = self.excess
        if abs(self.xi) < _XI0:
            return 1.0 - np.exp(-y / self.sigma)
        return 1.0 - np.maximum(1.0 + self.xi * y / self.sigma, 0.0) ** (-1.0 / self.xi)


def fit_gpd(x, threshold: float, *, n_obs: int | None = None, already_excesses: bool = False) -> GPDFit:
    """Maximum-likelihood GPD fit of the values of ``x`` above ``threshold`` (or of excesses when
    ``already_excesses``). ``n_obs`` is the number of observations the exceedances came from (the denominator of
    the rate); by default ``len(x)``."""
    x = np.asarray(x, float).ravel()
    y = x if already_excesses else x[x > threshold] - threshold
    if len(y) < 5:
        raise ValueError(f"only {len(y)} exceedances of {threshold:g}: a GPD fit needs at least 5")
    n_obs = len(x) if n_obs is None else int(n_obs)
    f = lambda p: gpd_nll(y, np.exp(p[0]), p[1])  # noqa: E731
    s0 = float(np.mean(y))
    p, val, ok = _minimise(f, [np.array([np.log(s0), xi0]) for xi0 in (-0.1, 0.1, 0.3)])
    sigma, xi = float(np.exp(p[0])), float(p[1])
    g = lambda q: gpd_nll(y, q[0], q[1])  # noqa: E731
    cov = _covariance(g, np.array([sigma, xi]))
    return GPDFit(sigma, xi, cov, val, float(threshold), len(y), n_obs, len(y) / n_obs, ok, y)
