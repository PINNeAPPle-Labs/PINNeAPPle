"""Solar-cycle envelope from the daily sunspot number: (E)EMD, the IMF nearest 11 years, its Hilbert amplitude, and
a fit of that amplitude by an extreme learning machine (ELM) with the end-point equality f(0) = f(1) built into the
basis (a constrained expression of the Theory of Functional Connections), then extended past the data.

Data. :func:`fetch_silso_daily` downloads SILSO's daily total sunspot number V2.0 (``SN_d_tot_V2.0.csv``,
Royal Observatory of Belgium, Brussels; licence CC BY-NC 4.0: cite "SILSO data/image, Royal Observatory of
Belgium, Brussels" and do not use it commercially). Nothing from SILSO is shipped here; :func:`synthetic_daily`
makes a series of the same shape for tests and offline demos.

Method.
1. Min-max scale to [0, 1] on a normalised time axis t in [0, 1].
2. Decompose with EMD or EEMD (``PyEMD`` from the ``EMD-signal`` package, Apache-2.0, when installed; otherwise
   :mod:`pinneapple_simulation.numerical_solvers.eemd`, whose IMFs plus the returned residue also sum to the
   signal). IMFs run from high to low frequency, and which index lands near 11 years depends on the series length
   and ``max_imf``: on a 2005-2025 SILSO slice the source app measured IMF 0 at ~3.6 days and the ~10-year
   component last. :func:`closest_to_schwabe_cycle` therefore picks the IMF whose measured period (median Hilbert
   instantaneous frequency) is nearest 11 years and warns when even that one is more than 40 % off.
3. Hilbert amplitude of that IMF (``scipy.signal.hilbert``).
4. :func:`periodic_elm_fit`: with g(x) = H(x) xi (H: random tanh features), f(x) = g(x) + x (g(0) - g(1))
   satisfies f(0) = f(1) for every xi (Mortari, "The Theory of Connections", 2017), so the fit is ordinary least
   squares on Hc(x) = H(x) + x (H(0) - H(1)). Only the two end values are tied (no derivative continuity);
   ``backend="tfc"`` uses the ``tfc`` package's ``ELMTanh`` basis instead of the NumPy one.

Moved here from the PINNeAPPle-apps ``solar_cycle_forecasting`` product (HelioTFC), which reconstructed this from
an older script whose ``tfc.decorators.constrained_expression`` / ``tfc.utils.elm_fit`` calls do not exist in
``tfc`` 1.4.0.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["SILSO_DAILY_URL", "SILSO_COLUMNS", "SILSO_CREDIT", "SCHWABE_CYCLE_DAYS", "fetch_silso_daily",
           "parse_silso_csv", "synthetic_daily", "preprocess", "decompose", "imf_statistics",
           "closest_to_schwabe_cycle", "hilbert_analysis", "PeriodicFit", "periodic_elm_fit", "run_pipeline"]

SILSO_DAILY_URL = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
SILSO_COLUMNS = ["Year", "Month", "Day", "DecDate", "Sunspot", "Std", "Obs", "Def"]
SILSO_CREDIT = "SILSO data, Royal Observatory of Belgium, Brussels (CC BY-NC 4.0)"
SCHWABE_CYCLE_DAYS = 365.25 * 11.0


def parse_silso_csv(source):
    """A SILSO daily CSV (path, file-like or bytes; ';'-separated, no header, -1 = missing) to a DataFrame with a
    DatetimeIndex and the non-missing ``Sunspot`` values."""
    import pandas as pd

    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    df = pd.read_csv(source, sep=";", header=None, names=SILSO_COLUMNS)
    df = df[df["Sunspot"] >= 0].copy()
    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    return df.set_index("Date").sort_index()


def fetch_silso_daily(url: str = SILSO_DAILY_URL, timeout: float = 30.0):
    """Download and parse the current SILSO daily series (network; raises on failure, never substitutes data)."""
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return parse_silso_csv(resp.content)


def synthetic_daily(start: str = "2005-01-01", end: str = "2025-06-30", *, period_years: float = 11.0,
                    peak: float = 180.0, seed: int = 0):
    """A SILSO-shaped synthetic daily series: a sin^2 cycle of ``period_years`` (minima 3.9 years after ``start``,
    as 2008.9 is after 2005), ~27-day rotational modulation and multiplicative noise, clipped at 0. Not solar
    data; for tests and offline demos."""
    import pandas as pd

    dates = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    t = (dates - dates[0]).days.to_numpy(dtype=float)
    cycle = np.sin(np.pi * (t / 365.25 - 3.9) / period_years) ** 2
    rot = 1.0 + 0.25 * np.sin(2 * np.pi * t / 27.0)
    ssn = np.clip(peak * cycle * rot * (1.0 + 0.3 * rng.standard_normal(t.size)) + 3.0 * rng.standard_normal(t.size),
                  0.0, None).round(0)
    df = pd.DataFrame({"Year": dates.year, "Month": dates.month, "Day": dates.day,
                       "DecDate": dates.year + (dates.dayofyear - 0.5) / 365.25, "Sunspot": ssn,
                       "Std": np.full(t.size, -1.0), "Obs": np.zeros(t.size, dtype=int), "Def": np.ones(t.size, dtype=int)},
                      index=pd.DatetimeIndex(dates, name="Date"))
    return df


def preprocess(df) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """``(t, y_scaled, (y_min, y_max))``: t = linspace(0, 1, N), y min-max scaled to [0, 1]."""
    y = df["Sunspot"].to_numpy(dtype=float)
    lo, hi = float(y.min()), float(y.max())
    return np.linspace(0.0, 1.0, y.size), (y - lo) / ((hi - lo) or 1.0), (lo, hi)


@dataclass
class Decomposition:
    imfs: np.ndarray          # (n_imf, N); rows sum to the input (the last row is the residue)
    method: str
    backend: str
    trials: int | None = None
    noise_width: float | None = None


def _pyemd():
    try:
        import PyEMD  # noqa: F401
        return True
    except ImportError:
        return False


def decompose(y: np.ndarray, method: str = "emd", *, max_imf: int = 6, trials: int = 16, noise_width: float = 0.05,
              seed: int = 0, backend: str = "auto") -> Decomposition:
    """EMD (deterministic) or EEMD (noise-assisted ensemble of ``trials`` runs) of a 1-D series."""
    y = np.asarray(y, dtype=np.float64)
    if method not in ("emd", "eemd"):
        raise ValueError(f"method must be 'emd' or 'eemd', not {method!r}")
    if backend == "auto":
        backend = "pyemd" if _pyemd() else "pinneapple"
    if backend == "pyemd":
        if method == "emd":
            from PyEMD import EMD

            return Decomposition(EMD().emd(y, max_imf=max_imf), "emd", "pyemd")
        from PyEMD import EEMD

        np.random.seed(seed)   # PyEMD's EEMD noise comes from NumPy's global generator
        return Decomposition(EEMD(trials=trials, noise_width=noise_width).eemd(y, max_imf=max_imf), "eemd", "pyemd",
                             trials, noise_width)
    if backend != "pinneapple":
        raise ValueError(f"backend must be 'auto', 'pyemd' or 'pinneapple', not {backend!r}")
    from pinneapple_simulation.numerical_solvers.eemd import eemd_1d, emd_1d

    if method == "emd":
        imfs = np.asarray(emd_1d(y, max_imfs=max_imf))
    else:
        imfs = eemd_1d(y, n_ensembles=trials, noise_std=noise_width, max_imfs=max_imf, seed=seed)
    imfs = np.vstack([imfs, y - imfs.sum(axis=0)])          # the residue, so the rows sum to y
    return Decomposition(imfs, method, "pinneapple", trials if method == "eemd" else None,
                         noise_width if method == "eemd" else None)


def imf_statistics(imfs: np.ndarray, dt_days: float = 1.0) -> list[dict[str, Any]]:
    """Per IMF: dominant period (days and years, from the median positive Hilbert instantaneous frequency), mean
    instantaneous amplitude and share of the variance of the summed signal."""
    from scipy.signal import hilbert

    total = float(np.var(imfs.sum(axis=0), ddof=1)) if imfs.shape[1] > 1 else float("nan")
    out = []
    for i, imf in enumerate(imfs):
        a = hilbert(imf)
        dphi = np.diff(np.unwrap(np.angle(a))) / (2 * np.pi)
        pos = dphi[dphi > 1e-9]
        period = float(np.median(1.0 / pos) * dt_days) if pos.size else float("nan")
        out.append({"imf_index": i, "dominant_period_days": period,
                    "dominant_period_years": period / 365.25 if np.isfinite(period) else None,
                    "mean_instantaneous_amplitude": float(np.abs(a).mean()),
                    "variance_share": float(np.var(imf, ddof=1) / total) if total > 0 else float("nan")})
    return out


def closest_to_schwabe_cycle(stats: list[dict]) -> tuple[dict | None, str | None]:
    """The IMF whose period is nearest 11 years, and a warning when it is more than 40 % off."""
    finite = [s for s in stats if s["dominant_period_days"] is not None and np.isfinite(s["dominant_period_days"])]
    if not finite:
        return None, "no IMF has a measurable dominant period"
    best = min(finite, key=lambda s: abs(s["dominant_period_days"] - SCHWABE_CYCLE_DAYS))
    rel = abs(best["dominant_period_days"] - SCHWABE_CYCLE_DAYS) / SCHWABE_CYCLE_DAYS
    warning = None
    if rel > 0.4:
        warning = (f"the closest IMF ({best['imf_index']}) has a period of {best['dominant_period_days'] / 365.25:.1f} "
                   f"years, {rel:.0%} off 11 years: this series length / max_imf does not resolve the cycle; try "
                   "another max_imf or a series a few cycles long")
    return best, warning


@dataclass
class HilbertResult:
    amplitude: np.ndarray
    phase: np.ndarray
    instantaneous_frequency: np.ndarray   # d(phase)/dt on the normalised axis


def hilbert_analysis(imf: np.ndarray, t: np.ndarray) -> HilbertResult:
    """Instantaneous amplitude, unwrapped phase and frequency of one IMF."""
    from scipy.signal import hilbert

    a = hilbert(np.asarray(imf, dtype=np.float64))
    phase = np.unwrap(np.angle(a))
    return HilbertResult(np.abs(a), phase, np.gradient(phase, t))


@dataclass
class PeriodicFit:
    rmse: float
    fitted: np.ndarray
    forecast_t: np.ndarray
    forecast: np.ndarray
    hidden_units: int
    backend: str
    endpoint_error: float      # |f(0) - f(1)|, zero up to round-off by construction


def _numpy_elm_basis(hidden_units: int, seed: int):
    rng = np.random.default_rng(seed)
    w = rng.uniform(-10.0, 10.0, hidden_units)
    b = rng.uniform(-10.0, 10.0, hidden_units)
    return lambda x: np.tanh(np.asarray(x, dtype=np.float64)[:, None] * w + b)


def _tfc_elm_basis(n: int, hidden_units: int, seed: int):
    import tfc

    np.random.seed(seed)      # tfc's ELM bases draw their weights from NumPy's global generator
    u = tfc.utfc(n, [0], hidden_units, basis="ELMTanh", x0=0.0, xf=1.0)
    return lambda x: np.asarray(u.H(np.asarray(x, dtype=np.float64)))


def periodic_elm_fit(t: np.ndarray, y: np.ndarray, *, hidden_units: int = 50, forecast_horizon_frac: float = 1.0,
                     n_forecast_points: int | None = None, seed: int = 0, backend: str = "numpy") -> PeriodicFit:
    """Least-squares ELM fit of ``y(t)`` on t in [0, 1] with f(0) = f(1) exact, extended to [1, 1 + horizon]."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if backend == "numpy":
        H = _numpy_elm_basis(hidden_units, seed)
    elif backend == "tfc":
        H = _tfc_elm_basis(t.size, hidden_units, seed)
    else:
        raise ValueError(f"backend must be 'numpy' or 'tfc', not {backend!r}")
    h0, h1 = H(np.array([0.0])), H(np.array([1.0]))

    def Hc(x):
        x = np.asarray(x, dtype=np.float64)
        return H(x) + x[:, None] * (h0 - h1)

    xi, *_ = np.linalg.lstsq(Hc(t), y, rcond=None)
    fitted = Hc(t) @ xi
    ft = np.linspace(1.0, 1.0 + forecast_horizon_frac, n_forecast_points or t.size)
    ends = Hc(np.array([0.0, 1.0])) @ xi
    return PeriodicFit(float(np.sqrt(np.mean((y - fitted) ** 2))), fitted, ft, Hc(ft) @ xi, hidden_units, backend,
                       float(abs(ends[0] - ends[1])))


def run_pipeline(df, *, decomposition_method: str = "emd", max_imf: int = 6, eemd_trials: int = 16,
                 eemd_noise_width: float = 0.05, imf_index: int | str = "auto", hidden_units: int = 50,
                 forecast_horizon_frac: float = 1.0, seed: int = 0, decomposition_backend: str = "auto",
                 fit_backend: str = "numpy") -> dict[str, Any]:
    """Steps 1-4 of the module docstring on a daily DataFrame (``Sunspot`` column, DatetimeIndex)."""
    t, y, (lo, hi) = preprocess(df)
    dec = decompose(y, decomposition_method, max_imf=max_imf, trials=eemd_trials, noise_width=eemd_noise_width,
                    seed=seed, backend=decomposition_backend)
    dt_days = float((df.index[-1] - df.index[0]).days) / max(len(df) - 1, 1)
    stats = imf_statistics(dec.imfs, dt_days=dt_days)
    warning = None
    if imf_index == "auto":
        best, warning = closest_to_schwabe_cycle(stats)
        if best is None:
            raise ValueError("no IMF has a measurable period; use a longer series")
        imf_index = best["imf_index"]
    elif not 0 <= int(imf_index) < dec.imfs.shape[0]:
        raise ValueError(f"imf_index={imf_index} out of range: {dec.imfs.shape[0]} IMFs (0..{dec.imfs.shape[0] - 1})")
    imf_index = int(imf_index)
    hil = hilbert_analysis(dec.imfs[imf_index], t)
    fit = periodic_elm_fit(t, hil.amplitude, hidden_units=hidden_units, forecast_horizon_frac=forecast_horizon_frac,
                           seed=seed, backend=fit_backend)
    return {"n_samples": int(len(df)), "date_range": [str(df.index[0].date()), str(df.index[-1].date())],
            "decomposition_method": dec.method, "decomposition_backend": dec.backend, "n_imfs": int(dec.imfs.shape[0]),
            "imf_index_used": imf_index, "imf_statistics": stats, "auto_select_warning": warning,
            "rmse": fit.rmse, "endpoint_constraint_error": fit.endpoint_error, "hidden_units": hidden_units,
            "fit_backend": fit.backend, "t": t, "amplitude_envelope": hil.amplitude,
            "instantaneous_frequency": hil.instantaneous_frequency, "fitted_envelope": fit.fitted,
            "forecast_t": fit.forecast_t, "forecast_amplitude": fit.forecast, "scale_min": lo, "scale_max": hi}
