"""Classical pieces of the standard time-series workflow that need only NumPy and SciPy: ARMA(p, q) by conditional
least squares with order selection by an information criterion and an ARIMA forecaster, Holt-Winters additive
exponential smoothing, the augmented Dickey-Fuller and KPSS stationarity tests, the Hampel outlier filter and split
conformal intervals.

``ARMAForecaster`` and ``HoltWintersForecaster`` follow the :class:`.forecast_lab.base.Forecaster` protocol, so they
run in the same backtest as the baselines and the learned models.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, signal
from scipy.special import expit

from .forecast_lab.base import Forecaster, train_end

__all__ = ["ARMAFit", "fit_arma", "select_arma", "ARMAForecaster", "HoltWintersForecaster", "adf_test", "kpss_test",
           "hampel", "conformal_quantile", "arma_sample"]


# ---------------------------------------------------------------------------
# ARMA
# ---------------------------------------------------------------------------

def _roots_ok(poly_coefs) -> bool:
    """All roots of 1 + c1 z + c2 z^2 + ... outside the unit circle."""
    c = np.asarray(poly_coefs, float)
    if len(c) == 0 or not np.any(c):
        return True
    r = np.roots(np.concatenate([c[::-1], [1.0]]))
    return bool(np.all(np.abs(r) > 1.0 + 1e-6))


def _residuals(y, mu, phi, theta):
    """e_t from phi(B)(y_t - mu) = theta(B) e_t with zero pre-sample values (conditional sum of squares)."""
    return signal.lfilter(np.concatenate([[1.0], -np.asarray(phi)]), np.concatenate([[1.0], np.asarray(theta)]),
                          np.asarray(y, float) - mu)


@dataclass
class ARMAFit:
    p: int
    q: int
    mu: float
    phi: np.ndarray
    theta: np.ndarray
    sigma2: float
    loglik: float
    n: int
    aic: float
    bic: float

    def to_dict(self) -> dict[str, Any]:
        return {"order": [self.p, self.q], "mu": self.mu, "phi": self.phi.tolist(), "theta": self.theta.tolist(),
                "sigma2": self.sigma2, "loglik": self.loglik, "aic": self.aic, "bic": self.bic, "n": self.n}


def fit_arma(y, p: int, q: int) -> ARMAFit:
    """ARMA(p, q) with a mean, by conditional least squares (Hannan-Rissanen start, then Levenberg-Marquardt on the
    residuals of the first ``max(p, q)`` samples dropped); stationarity and invertibility are enforced by a
    penalty. The Gaussian conditional log-likelihood gives AIC and BIC with ``p + q + 2`` parameters."""
    y = np.asarray(y, float).ravel()
    n = len(y)
    m = max(p, q)
    if n < 3 * (p + q + 2) + m:
        raise ValueError(f"series of {n} samples too short for ARMA({p}, {q})")
    mu0 = float(y.mean())
    yc = y - mu0
    # Hannan-Rissanen: a long AR gives the innovations, then OLS on lags of y and of the innovations
    L = min(max(10, 2 * (p + q)), n // 5)
    X = np.column_stack([yc[L - k - 1: n - k - 1] for k in range(L)])
    a = np.linalg.lstsq(X, yc[L:], rcond=None)[0]
    e = np.zeros(n)
    e[L:] = yc[L:] - X @ a
    s = L + m
    cols = [yc[s - i: n - i] for i in range(1, p + 1)] + [e[s - j: n - j] for j in range(1, q + 1)]
    if cols:
        b = np.linalg.lstsq(np.column_stack(cols), yc[s:], rcond=None)[0]
        x0 = np.concatenate([[0.0], b])
    else:
        x0 = np.zeros(1)
    x0[1: p + 1] = np.clip(x0[1: p + 1], -0.95, 0.95) if p else x0[1: p + 1]

    def res(v):
        mu, phi, th = mu0 + v[0], v[1: p + 1], v[p + 1:]
        r = _residuals(y, mu, phi, th)[m:]
        if not (_roots_ok(-phi) and _roots_ok(th)) or not np.all(np.isfinite(r)):
            return np.full(len(r), 1e6)
        return r

    if not (_roots_ok(-x0[1: p + 1]) and _roots_ok(x0[p + 1:])):
        x0[1:] = 0.0
    starts = [x0, np.zeros_like(x0)]
    if p and q:                                   # the AR part alone, then the MA part alone
        a_only = np.zeros_like(x0)
        a_only[1: p + 1] = x0[1: p + 1]
        starts.append(a_only)
    best = None
    for st in starts:
        sol = optimize.least_squares(res, st, method="lm", xtol=1e-12, ftol=1e-12, max_nfev=5000)
        if best is None or sol.cost < best.cost:
            best = sol
    v = best.x
    mu, phi, th = mu0 + v[0], v[1: p + 1], v[p + 1:]
    r = _residuals(y, mu, phi, th)[m:]
    ne = len(r)
    s2 = float(r @ r / ne)
    ll = -0.5 * ne * (np.log(2 * np.pi * s2) + 1.0)
    k = p + q + 2
    return ARMAFit(p, q, float(mu), np.asarray(phi), np.asarray(th), s2, float(ll), ne, float(-2 * ll + 2 * k),
                   float(-2 * ll + k * np.log(ne)))


def select_arma(y, max_p: int = 4, max_q: int = 3, criterion: str = "bic") -> dict[str, Any]:
    """Every ARMA(p, q) with ``p <= max_p``, ``q <= max_q`` fitted on the same samples (the first
    ``max(max_p, max_q)`` are conditioned on for all), and the order with the lowest criterion."""
    y = np.asarray(y, float).ravel()
    m = max(max_p, max_q)
    table = []
    fits = {}
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            f = fit_arma(y, p, q)
        except (ValueError, np.linalg.LinAlgError):
            continue
        # put every candidate on the same sample count: re-evaluate the likelihood on y[m:]
        r = _residuals(y, f.mu, f.phi, f.theta)[m:]
        s2 = float(r @ r / len(r))
        ll = -0.5 * len(r) * (np.log(2 * np.pi * s2) + 1.0)
        k = p + q + 2
        aic, bic = -2 * ll + 2 * k, -2 * ll + k * np.log(len(r))
        table.append({"p": p, "q": q, "aic": float(aic), "bic": float(bic), "sigma2": s2})
        fits[(p, q)] = f
    if not table:
        raise ValueError("no ARMA order could be fitted")
    best = min(table, key=lambda t: t[criterion])
    return {"order": (best["p"], best["q"]), "criterion": criterion, "fit": fits[(best["p"], best["q"])],
            "table": table}


def arma_sample(n: int, phi: Sequence[float], theta: Sequence[float], *, mu: float = 0.0, sigma: float = 1.0,
                burn: int = 500, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    e = sigma * rng.standard_normal(n + burn)
    y = signal.lfilter(np.concatenate([[1.0], theta]), np.concatenate([[1.0], -np.asarray(phi, float)]), e)
    return mu + y[burn:]


class ARMAForecaster(Forecaster):
    """ARIMA(p, d, q): the order of the (``d`` times differenced) target selected by ``criterion`` on the
    training segment, then forecasts from every origin with the parameters fixed. The innovations come from the
    causal filter over the data up to each origin, so a forecast uses no later sample. The Gaussian spread is the
    validation RMSE per horizon (the base class)."""
    name, family = "arima", "statistical"

    def __init__(self, max_p: int = 4, max_q: int = 3, d: int = 0, criterion: str = "bic"):
        super().__init__()
        self.max_p, self.max_q, self.d, self.criterion = int(max_p), int(max_q), int(d), criterion

    def _z(self, y):
        return np.diff(y, n=self.d) if self.d else y

    def _fit(self, data, train_origins, val_origins):
        seg = data.y[: train_end(train_origins, self.horizons)]
        sel = select_arma(self._z(seg), self.max_p, self.max_q, self.criterion)
        self.fit_ = sel["fit"]
        self.selection_ = sel
        self.info = {"order": [self.fit_.p, self.d, self.fit_.q], "criterion": self.criterion,
                     "phi": self.fit_.phi.tolist(), "theta": self.fit_.theta.tolist()}

    def _predict_mean(self, data, origins):
        f = self.fit_
        y = data.y
        z = self._z(y)
        off = self.d                                  # z[i] corresponds to y[i + d]
        e = _residuals(z, f.mu, f.phi, f.theta)
        o = np.asarray(origins) - off
        H = int(self.horizons.max())
        p, q = f.p, f.q
        zh = np.stack([z[np.maximum(o - k, 0)] for k in range(max(p, 1))], axis=1)[:, :p] - f.mu
        eh = np.stack([e[np.maximum(o - k, 0)] for k in range(max(q, 1))], axis=1)[:, :q]
        path = np.empty((len(o), H))
        for h in range(H):
            nxt = (zh @ f.phi if p else 0.0) + (eh @ f.theta if q else 0.0)
            nxt = np.broadcast_to(nxt, (len(o),)).astype(float)
            path[:, h] = nxt + f.mu
            if p:
                zh = np.concatenate([nxt[:, None], zh[:, :-1]], axis=1)
            if q:
                eh = np.concatenate([np.zeros((len(o), 1)), eh[:, :-1]], axis=1)
        if self.d == 1:
            path = y[np.asarray(origins)][:, None] + np.cumsum(path, axis=1)
        elif self.d > 1:
            raise NotImplementedError("d > 1")
        return path[:, self.horizons - 1]

    def n_params(self):
        return self.fit_.p + self.fit_.q + 2


class HoltWintersForecaster(Forecaster):
    """Additive Holt-Winters (ETS(A,A,A), or ETS(A,A,N) without a period): level, trend and seasonal smoothing
    weights fitted by least squares of the one-step errors on the training segment; forecasts from every origin
    by running the filter over the data up to it."""
    name, family = "ets", "statistical"

    def __init__(self, period: int = 0, damped: bool = True):
        super().__init__()
        self.period, self.damped = int(period), bool(damped)

    def _run(self, y, a, b, g, phi):
        m = self.period
        n = len(y)
        lev = np.empty(n)
        tr = np.empty(n)
        if m > 1:
            l0 = y[:m].mean()
            s = list(y[:m] - l0)
        else:
            l0, s = y[0], []
        l_, t_ = l0, 0.0
        err = np.empty(n)
        seas = np.empty(n)
        for i in range(n):
            si = s[i % m] if m > 1 else 0.0
            pred = l_ + phi * t_ + si
            err[i] = y[i] - pred
            ln = a * (y[i] - si) + (1 - a) * (l_ + phi * t_)
            tn = b * (ln - l_) + (1 - b) * phi * t_
            if m > 1:
                s[i % m] = g * (y[i] - ln) + (1 - g) * si
            seas[i] = si
            l_, t_ = ln, tn
            lev[i], tr[i] = l_, t_
        return lev, tr, err, s

    def _fit(self, data, train_origins, val_origins):
        seg = data.y[: train_end(train_origins, self.horizons)]
        m = self.period
        skip = max(m, 2)

        def obj(v):
            a, b, g, ph = expit(v)
            ph = 0.8 + 0.2 * ph if self.damped else 1.0
            return np.sum(self._run(seg, a, b, g if m > 1 else 0.0, ph)[2][skip:] ** 2)

        r = optimize.minimize(obj, np.array([0.0, -2.0, -2.0, 2.0]), method="Nelder-Mead",
                              options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8})
        a, b, g, ph = expit(r.x)
        self.par_ = (float(a), float(b), float(g if m > 1 else 0.0), float(0.8 + 0.2 * ph if self.damped else 1.0))
        self.info = {"alpha": self.par_[0], "beta": self.par_[1], "gamma": self.par_[2], "phi": self.par_[3],
                     "period": m}

    def _predict_mean(self, data, origins):
        a, b, g, ph = self.par_
        m = self.period
        y = data.y
        hi = int(np.max(origins)) + 1
        n = hi
        lev = np.empty(n)
        tr = np.empty(n)
        seas_hist = np.zeros((n, max(m, 1)))
        if m > 1:
            l0 = y[:m].mean()
            s = list(y[:m] - l0)
        else:
            l0, s = y[0], []
        l_, t_ = l0, 0.0
        for i in range(n):
            si = s[i % m] if m > 1 else 0.0
            ln = a * (y[i] - si) + (1 - a) * (l_ + ph * t_)
            tn = b * (ln - l_) + (1 - b) * ph * t_
            if m > 1:
                s[i % m] = g * (y[i] - ln) + (1 - g) * si
                seas_hist[i] = s
            l_, t_ = ln, tn
            lev[i], tr[i] = l_, t_
        o = np.asarray(origins)
        out = np.empty((len(o), len(self.horizons)))
        for j, h in enumerate(self.horizons):
            damp = np.sum(ph ** np.arange(1, h + 1))
            sv = seas_hist[o, (o + h) % m] if m > 1 else 0.0
            out[:, j] = lev[o] + damp * tr[o] + sv
        return out

    def n_params(self):
        return 4 + max(self.period, 0)


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

# MacKinnon (2010) response surface for the ADF statistic with a constant: c(n) = b_inf + b1 / n + b2 / n^2
_ADF_C = {"1%": (-3.43035, -6.5393, -16.786), "5%": (-2.86154, -2.8903, -4.234), "10%": (-2.56677, -1.5384, -2.809)}


def adf_test(y, max_lag: int | None = None) -> dict[str, Any]:
    """Augmented Dickey-Fuller test with a constant: ``dy_t = a + b y_{t-1} + sum g_i dy_{t-i} + e``; the lag
    count minimises AIC up to ``12 (n/100)^(1/4)``. H0: a unit root (b = 0). The critical values are MacKinnon's
    (2010) finite-sample response surface."""
    y = np.asarray(y, float).ravel()
    n = len(y)
    if max_lag is None:
        max_lag = int(np.floor(12 * (n / 100.0) ** 0.25))
    dy = np.diff(y)
    best = None
    for k in range(max_lag + 1):
        # the same sample for every k: start at max_lag
        t0 = max_lag
        Y = dy[t0:]
        X = [np.ones(len(Y)), y[t0: n - 1]] + [dy[t0 - i: len(dy) - i] for i in range(1, k + 1)]
        X = np.column_stack(X)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        r = Y - X @ beta
        s2 = r @ r / (len(Y) - X.shape[1])
        aic = len(Y) * np.log(r @ r / len(Y)) + 2 * X.shape[1]
        if best is None or aic < best[0]:
            cov = s2 * np.linalg.inv(X.T @ X)
            best = (aic, k, float(beta[1] / np.sqrt(cov[1, 1])), len(Y))
    _, k, _, _ = best
    # the chosen lag count refitted on every sample it allows (as statsmodels' adfuller does after autolag)
    Y = dy[k:]
    X = np.column_stack([np.ones(len(Y)), y[k: n - 1]] + [dy[k - i: len(dy) - i] for i in range(1, k + 1)])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r = Y - X @ beta
    cov = (r @ r / (len(Y) - X.shape[1])) * np.linalg.inv(X.T @ X)
    stat, nobs = float(beta[1] / np.sqrt(cov[1, 1])), len(Y)
    crit = {lv: b0 + b1 / nobs + b2 / nobs ** 2 for lv, (b0, b1, b2) in _ADF_C.items()}
    return {"statistic": stat, "lags": k, "n": nobs, "critical": crit, "unit_root_rejected_5pct": stat < crit["5%"]}


_KPSS_C = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}


def kpss_test(y, lags: int | None = None) -> dict[str, Any]:
    """KPSS test of level stationarity (H0: stationary): ``eta = sum S_t^2 / (n^2 s^2)`` with the Newey-West
    long-run variance over ``lags = 12 (n/100)^(1/4)`` lags; Kwiatkowski et al. (1992) critical values."""
    y = np.asarray(y, float).ravel()
    n = len(y)
    e = y - y.mean()
    if lags is None:
        lags = int(np.ceil(12 * (n / 100.0) ** 0.25))
    lags = min(lags, n - 1)
    s2 = e @ e / n
    for k in range(1, lags + 1):
        s2 += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    S = np.cumsum(e)
    eta = float(S @ S / (n * n * s2))
    return {"statistic": eta, "lags": lags, "critical": dict(_KPSS_C), "stationarity_rejected_5pct": eta > _KPSS_C["5%"]}


# ---------------------------------------------------------------------------
# Outliers and conformal intervals
# ---------------------------------------------------------------------------

def hampel(y, half_window: int = 12, n_sigmas: float = 5.0) -> dict[str, Any]:
    """Hampel filter: a sample more than ``n_sigmas`` robust standard deviations (1.4826 MAD) from the median of
    its centred window is an outlier, replaced by that median in ``cleaned``."""
    y = np.asarray(y, float).ravel()
    n = len(y)
    k = int(half_window)
    pad = np.pad(y, k, mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1)
    med = np.median(win, axis=1)
    mad = 1.4826 * np.median(np.abs(win - med[:, None]), axis=1)
    out = np.abs(y - med) > n_sigmas * np.maximum(mad, 1e-12)
    cleaned = np.where(out, med, y)
    return {"index": np.nonzero(out)[0].tolist(), "count": int(out.sum()), "cleaned": cleaned, "n": n,
            "rule": f"|y - median| > {n_sigmas:g} x 1.4826 MAD over a {2 * k + 1}-sample window"}


def conformal_quantile(abs_residuals, alpha: float) -> float:
    """Split-conformal quantile: the ``ceil((n + 1)(1 - alpha))``-th smallest absolute residual, which gives
    coverage of at least ``1 - alpha`` for an exchangeable new residual."""
    r = np.sort(np.asarray(abs_residuals, float)[np.isfinite(abs_residuals)])
    n = len(r)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(r[min(k, n) - 1]) if k <= n else float("inf")
