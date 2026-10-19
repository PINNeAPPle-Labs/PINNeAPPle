"""Is there anything to predict? Measures of predictability to compute before fitting any forecaster.

* :func:`fit_ar` -- least-squares AR(p) with the order chosen by AIC or BIC, and its one-step R^2.
* :func:`permutation_entropy` -- Bandt & Pompe (2002), Phys. Rev. Lett. 88:174102; 1 for an i.i.d. series.
* :func:`sample_entropy` -- Richman & Moorman (2000), Am. J. Physiol. 278:H2039.
* :func:`rosenstein_lyapunov` -- largest Lyapunov exponent from a scalar series (Rosenstein, Collins & De Luca
  1993, Physica D 65:117). It returns a positive slope for noise too; read it only next to the surrogate test.
* :func:`iaaft_surrogate`, :func:`phase_randomized_surrogate`, :func:`shuffle_surrogate` -- surrogate data
  (Theiler et al. 1992, Physica D 58:77; Schreiber & Schmitz 1996, Phys. Rev. Lett. 77:635).
* :func:`nonlinear_prediction_error`, :func:`time_reversal_asymmetry` -- discriminating statistics.
* :func:`surrogate_test` -- rank test of a statistic against surrogates: does the series have more structure than
  an i.i.d. process (shuffle) or a linear Gaussian process seen through a static monotone transform (IAAFT)?
* :func:`predictability_report` -- all of the above for one series, with the persistence and climatology errors.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from math import factorial

import numpy as np

__all__ = [
    "ARFit", "fit_ar", "permutation_entropy", "sample_entropy", "delay_embedding", "first_acf_zero",
    "rosenstein_lyapunov", "shuffle_surrogate", "phase_randomized_surrogate", "iaaft_surrogate",
    "nonlinear_prediction_error", "time_reversal_asymmetry", "SurrogateTest", "surrogate_test",
    "baseline_errors", "predictability_report",
]


# ---------------------------------------------------------------------------
# Linear structure
# ---------------------------------------------------------------------------

@dataclass
class ARFit:
    order: int
    intercept: float
    coef: np.ndarray
    sigma: float
    r2_one_step: float
    criterion: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["coef"] = [float(c) for c in self.coef]
        d["criterion"] = {int(k): float(v) for k, v in self.criterion.items()}
        return d


def _ar_design(x: np.ndarray, p: int, p_max: int):
    n = len(x)
    X = np.column_stack([np.ones(n - p_max)] + [x[p_max - k:n - k] for k in range(1, p + 1)])
    return X, x[p_max:]


def fit_ar(x, max_order: int = 20, criterion: str = "aic") -> ARFit:
    """Least-squares AR(p), p = 0..max_order, all fitted on the same sample (the first ``max_order`` values are
    dropped for every p) so the information criteria are comparable."""
    x = np.asarray(x, float).ravel()
    n_eff = len(x) - max_order
    crit, fits = {}, {}
    for p in range(0, max_order + 1):
        X, y = _ar_design(x, p, max_order)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        s2 = float(res @ res) / n_eff
        pen = 2 * (p + 1) if criterion == "aic" else np.log(n_eff) * (p + 1)
        crit[p] = n_eff * np.log(max(s2, 1e-300)) + pen
        fits[p] = (beta, s2, res, y)
    p = min(crit, key=crit.get)
    beta, s2, res, y = fits[p]
    r2 = 1.0 - float(res @ res) / float(((y - y.mean()) ** 2).sum())
    dof = max(n_eff - p - 1, 1)
    return ARFit(p, float(beta[0]), np.asarray(beta[1:]), float(np.sqrt(s2 * n_eff / dof)), r2, crit)


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------

def permutation_entropy(x, order: int = 5, delay: int = 1, normalize: bool = True) -> float:
    """Shannon entropy of the ordinal patterns of length ``order`` (ties broken by position)."""
    x = np.asarray(x, float).ravel()
    n = len(x) - (order - 1) * delay
    if n <= 0:
        raise ValueError("series too short for this order and delay")
    emb = np.column_stack([x[i * delay:i * delay + n] for i in range(order)])
    ranks = np.argsort(emb, axis=1, kind="stable")
    codes = (ranks * (order ** np.arange(order))).sum(axis=1)
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    h = float(-(p * np.log(p)).sum())
    return h / np.log(factorial(order)) if normalize else h


def sample_entropy(x, m: int = 2, r: float | None = None) -> float:
    """-log(A / B): B = template pairs of length m within tolerance r (Chebyshev), A = the same for m + 1; self
    matches excluded and the same N - m templates used for both lengths. ``r`` defaults to 0.2 std(x)."""
    from scipy.spatial import cKDTree
    x = np.asarray(x, float).ravel()
    r = 0.2 * x.std() if r is None else float(r)
    n = len(x) - m

    def pairs(mm):
        emb = np.column_stack([x[i:i + n] for i in range(mm)])
        tree = cKDTree(emb)
        return (tree.count_neighbors(tree, r, p=np.inf) - n) / 2.0

    B, A = pairs(m), pairs(m + 1)
    if A <= 0 or B <= 0:
        return float("inf")
    return float(-np.log(A / B))


# ---------------------------------------------------------------------------
# Chaos
# ---------------------------------------------------------------------------

def delay_embedding(x, dim: int, delay: int) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    n = len(x) - (dim - 1) * delay
    return np.column_stack([x[i * delay:i * delay + n] for i in range(dim)])


def first_acf_zero(x, max_lag: int | None = None) -> int:
    """First lag at which the autocorrelation drops below 1/e (a common embedding-delay rule)."""
    x = np.asarray(x, float).ravel()
    x = x - x.mean()
    max_lag = max_lag or len(x) // 4
    f = np.fft.rfft(x, 2 * len(x))
    acf = np.fft.irfft(f * np.conj(f))[:max_lag + 1]
    acf = acf / acf[0]
    below = np.nonzero(acf < np.exp(-1))[0]
    return int(below[0]) if len(below) else int(max_lag)


def rosenstein_lyapunov(x, *, dim: int = 5, delay: int | None = None, dt: float = 1.0,
                        min_tsep: int | None = None, n_steps: int = 40, fit: tuple = (5, 30),
                        max_points: int = 3000) -> dict[str, object]:
    """Largest Lyapunov exponent per unit time from a scalar series.

    Each embedded point is paired with its nearest neighbour at least ``min_tsep`` samples away (default: the
    mean period from the power spectrum); the mean log separation over the next ``n_steps`` samples is fitted by
    a straight line on ``fit`` = (first, last) step. Returns the exponent, the divergence curve and the choices.
    """
    from scipy.spatial import cKDTree
    x = np.asarray(x, float).ravel()
    delay = first_acf_zero(x) if delay is None else int(delay)
    if min_tsep is None:
        f = np.fft.rfft(x - x.mean())
        freqs = np.fft.rfftfreq(len(x))
        p = np.abs(f[1:]) ** 2
        min_tsep = int(np.ceil(1.0 / max((p * freqs[1:]).sum() / p.sum(), 1e-12)))
    emb = delay_embedding(x, dim, delay)
    n = len(emb) - n_steps
    if n < 50:
        raise ValueError("series too short for this embedding")
    idx = np.arange(n) if n <= max_points else np.linspace(0, n - 1, max_points).astype(int)
    tree = cKDTree(emb[:n])
    k = min(n, 2 * min_tsep + 20)
    dist, nb = tree.query(emb[idx], k=k)
    nn = np.full(len(idx), -1)
    for row in range(len(idx)):
        ok = np.nonzero((np.abs(nb[row] - idx[row]) > min_tsep) & (dist[row] > 0))[0]
        if len(ok):
            nn[row] = nb[row, ok[0]]
    keep = nn >= 0
    i0, j0 = idx[keep], nn[keep]
    curve = np.empty(n_steps)
    for s in range(n_steps):
        d = np.linalg.norm(emb[i0 + s] - emb[j0 + s], axis=1)
        curve[s] = np.mean(np.log(d[d > 0]))
    a, b = fit
    slope = np.polyfit(np.arange(a, b + 1) * dt, curve[a:b + 1], 1)[0]
    return {"lyapunov": float(slope), "curve": curve, "dim": dim, "delay": delay, "min_tsep": int(min_tsep),
            "fit": list(fit), "dt": dt, "n_pairs": int(keep.sum())}


# ---------------------------------------------------------------------------
# Surrogates
# ---------------------------------------------------------------------------

def shuffle_surrogate(x, rng: np.random.Generator) -> np.ndarray:
    """Random permutation: keeps the amplitude distribution, destroys all temporal structure."""
    return rng.permutation(np.asarray(x, float).ravel())


def phase_randomized_surrogate(x, rng: np.random.Generator) -> np.ndarray:
    """Fourier-transform surrogate: same power spectrum, random phases (a linear Gaussian process)."""
    x = np.asarray(x, float).ravel()
    f = np.fft.rfft(x - x.mean())
    ph = np.exp(2j * np.pi * rng.random(len(f)))
    ph[0] = 1.0
    if len(x) % 2 == 0:
        ph[-1] = 1.0
    return np.fft.irfft(f * ph, len(x)) + x.mean()


def iaaft_surrogate(x, rng: np.random.Generator, n_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier transform surrogate (Schreiber & Schmitz 1996): the same amplitude
    distribution as ``x`` exactly and nearly the same power spectrum."""
    x = np.asarray(x, float).ravel()
    sorted_x = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    s = rng.permutation(x)
    prev = None
    for _ in range(int(n_iter)):
        f = np.fft.rfft(s)
        s = np.fft.irfft(amp * np.exp(1j * np.angle(f)), len(x))
        ranks = np.argsort(np.argsort(s))
        s = sorted_x[ranks]
        if prev is not None and np.array_equal(ranks, prev):
            break
        prev = ranks
    return s


_SURROGATES: dict[str, Callable] = {
    "shuffle": shuffle_surrogate, "phase": phase_randomized_surrogate, "iaaft": iaaft_surrogate,
}


def nonlinear_prediction_error(x, *, dim: int = 4, delay: int = 1, horizon: int = 1, k: int = 10,
                               train_frac: float = 0.5) -> float:
    """Out-of-sample error of a local-constant (nearest-neighbour) predictor, divided by std(x).

    The library is the first ``train_frac`` of the delay vectors, the predictions are made for the rest: each test
    vector is predicted as the mean future of its ``k`` nearest library vectors (Sugihara & May 1990, Nature
    344:734; Theiler et al. 1992). ~1 when nothing is predictable, < 1 when it is.
    """
    from scipy.spatial import cKDTree
    x = np.asarray(x, float).ravel()
    emb = delay_embedding(x, dim, delay)
    last = (dim - 1) * delay
    n = len(emb) - horizon
    fut = x[last + horizon:last + horizon + n]
    emb = emb[:n]
    n_tr = int(train_frac * n)
    tree = cKDTree(emb[:n_tr - horizon])
    _, nb = tree.query(emb[n_tr:], k=k)
    pred = fut[nb].mean(axis=1)
    return float(np.sqrt(np.mean((pred - fut[n_tr:]) ** 2)) / x.std())


def time_reversal_asymmetry(x, lag: int = 1) -> float:
    """Normalised third moment of the lag-``lag`` increments; 0 in expectation for any time-reversible process
    (every linear Gaussian process is)."""
    x = np.asarray(x, float).ravel()
    d = x[lag:] - x[:-lag]
    return float(np.mean(d ** 3) / np.mean(d ** 2) ** 1.5)


@dataclass
class SurrogateTest:
    statistic: str
    kind: str
    observed: float
    surrogates: list[float]
    p_value: float
    alternative: str
    reject_at_0_05: bool

    def to_dict(self) -> dict:
        return asdict(self)


def surrogate_test(x, statistic: Callable[[np.ndarray], float], *, kind: str = "iaaft", n_surrogates: int = 99,
                   alternative: str = "less", seed: int = 0, name: str | None = None) -> SurrogateTest:
    """Rank test of ``statistic(x)`` against ``n_surrogates`` surrogates of the given ``kind``.

    ``alternative="less"``: the original is more predictable (smaller statistic) than the surrogates, p = (1 +
    #{s_i <= s_0}) / (n + 1). ``"two-sided"`` compares |s - median(s_i)|. With 99 surrogates the smallest
    attainable p is 0.01.
    """
    rng = np.random.default_rng(seed)
    gen = _SURROGATES[kind]
    s0 = float(statistic(np.asarray(x, float)))
    sur = [float(statistic(gen(x, rng))) for _ in range(int(n_surrogates))]
    arr = np.asarray(sur)
    if alternative == "less":
        p = (1 + np.sum(arr <= s0)) / (len(arr) + 1)
    elif alternative == "greater":
        p = (1 + np.sum(arr >= s0)) / (len(arr) + 1)
    else:
        med = np.median(arr)
        p = (1 + np.sum(np.abs(arr - med) >= abs(s0 - med))) / (len(arr) + 1)
    return SurrogateTest(name or getattr(statistic, "__name__", "statistic"), kind, s0, sur, float(p), alternative,
                         bool(p <= 0.05))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def baseline_errors(x, horizons, train_frac: float = 0.8) -> dict[str, dict[int, float]]:
    """RMSE of persistence (x_{t+h} = x_t) and climatology (x_{t+h} = mean of the first ``train_frac``) on the last
    1 - ``train_frac`` of the series, per horizon, both divided by std(x)."""
    x = np.asarray(x, float).ravel()
    n_tr = int(train_frac * len(x))
    mu, sd = x[:n_tr].mean(), x.std()
    out: dict[str, dict[int, float]] = {"persistence": {}, "climatology": {}}
    for h in horizons:
        t = np.arange(n_tr, len(x) - h)
        out["persistence"][int(h)] = float(np.sqrt(np.mean((x[t + h] - x[t]) ** 2)) / sd)
        out["climatology"][int(h)] = float(np.sqrt(np.mean((x[t + h] - mu) ** 2)) / sd)
    return out


def predictability_report(x, *, horizons=(1, 2, 4, 8, 16, 32), dt: float = 1.0, max_ar_order: int = 20,
                          pe_order: int = 5, embed_dim: int = 4, n_surrogates: int = 99, seed: int = 0,
                          lyapunov: bool = True, lyapunov_kwargs: dict | None = None,
                          difference_if_unit_root: bool = True) -> dict[str, object]:
    """Predictability of a scalar series before any modelling.

    The surrogate tests use the nonlinear prediction error at horizon 1 (one-sided: is the series more predictable
    by nearest neighbours than its surrogates?) and the time-reversal asymmetry (two-sided). When the AR(1)
    coefficient exceeds 0.99 and ``difference_if_unit_root`` is set, the tests and entropies run on the first
    differences, because Fourier surrogates of a non-stationary series are not a valid null (Schreiber & Schmitz
    2000, Physica D 142:346); ``tested_on`` says which.
    """
    x = np.asarray(x, float).ravel()
    ar1 = fit_ar(x, max_order=1)
    phi1 = float(ar1.coef[0]) if ar1.order == 1 else float(np.corrcoef(x[:-1], x[1:])[0, 1])
    z = np.diff(x) if (difference_if_unit_root and phi1 > 0.99) else x
    tested_on = "differences" if z is not x else "levels"
    ar = fit_ar(z, max_order=max_ar_order)
    delay = first_acf_zero(z)
    tau_npe = max(1, min(delay, 10))
    npe = lambda s: nonlinear_prediction_error(s, dim=embed_dim, delay=tau_npe)  # noqa: E731
    npe.__name__ = "nonlinear_prediction_error"
    tests = {}
    for kind in ("shuffle", "iaaft"):
        tests[f"npe_{kind}"] = surrogate_test(z, npe, kind=kind, n_surrogates=n_surrogates, seed=seed,
                                              name="nonlinear_prediction_error").to_dict()
        tests[f"tra_{kind}"] = surrogate_test(z, time_reversal_asymmetry, kind=kind, n_surrogates=n_surrogates,
                                              seed=seed + 1, alternative="two-sided").to_dict()
    out: dict[str, object] = {
        "n": int(len(x)), "tested_on": tested_on, "ar1_coefficient": phi1,
        "ar": ar.to_dict(), "permutation_entropy": permutation_entropy(z, order=pe_order),
        "sample_entropy": sample_entropy(z), "acf_delay": int(delay), "npe_delay": int(tau_npe),
        "baselines_rmse_over_std": baseline_errors(x, horizons), "surrogate_tests": tests,
    }
    if lyapunov:
        kw = {"dim": embed_dim, "delay": tau_npe, "dt": dt, **(lyapunov_kwargs or {})}
        ly = rosenstein_lyapunov(z, **kw)
        ly["curve"] = [float(c) for c in ly["curve"]]
        out["rosenstein"] = ly
    return out
