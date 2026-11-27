"""Point and interval metrics for power prediction.

Plain numpy, with the definitions written next to the number so a table cannot
be read the wrong way:

* ``rmse`` / ``mae`` / ``bias`` -- in kW, over every sample in the test window,
  including the zeros below cut-in. Zeros dominate a SCADA test set, so an RMSE
  on the raw sample is flattered by them; the energy-weighted numbers below are
  the ones that matter for production.
* ``energy_weighted_rmse`` -- RMSE weighted by the measured power of the sample,
  i.e. "how wrong is the model where the energy actually is".
* ``energy_error`` -- total predicted energy minus measured energy over the
  window, as a fraction of measured energy. This is the number an owner cares
  about, and it can be near zero while every single sample is wrong in opposite
  directions, so it never travels alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PointMetrics",
    "point_metrics",
    "per_bin_error",
    "calibration_report",
    "conformal_interval",
]


@dataclass(frozen=True)
class PointMetrics:
    n: int
    rmse_kw: float
    mae_kw: float
    bias_kw: float
    r2: float
    energy_weighted_rmse_kw: float
    energy_error_fraction: float
    measured_energy_mwh: float
    predicted_energy_mwh: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "rmse_kw": self.rmse_kw,
            "mae_kw": self.mae_kw,
            "bias_kw": self.bias_kw,
            "r2": self.r2,
            "energy_weighted_rmse_kw": self.energy_weighted_rmse_kw,
            "energy_error_fraction": self.energy_error_fraction,
            "measured_energy_mwh": self.measured_energy_mwh,
            "predicted_energy_mwh": self.predicted_energy_mwh,
        }


def point_metrics(measured, predicted, *, sample_minutes: float = 10.0) -> PointMetrics:
    """All scalar metrics for one test window."""
    y = np.asarray(measured, dtype=float)
    f = np.asarray(predicted, dtype=float)
    ok = np.isfinite(y) & np.isfinite(f)
    y, f = y[ok], f[ok]
    if y.size == 0:
        raise ValueError("no finite pairs to score")
    err = f - y
    weights = np.maximum(y, 0.0)
    w_sum = weights.sum()
    ew = float(np.sqrt((weights * err**2).sum() / w_sum)) if w_sum > 0 else 0.0
    ss_tot = float(((y - y.mean()) ** 2).sum())
    hours = sample_minutes / 60.0
    return PointMetrics(
        n=int(y.size),
        rmse_kw=float(np.sqrt(np.mean(err**2))),
        mae_kw=float(np.mean(np.abs(err))),
        bias_kw=float(np.mean(err)),
        r2=float(1.0 - ((err**2).sum() / ss_tot)) if ss_tot > 0 else 0.0,
        energy_weighted_rmse_kw=ew,
        energy_error_fraction=float(err.sum() / y.sum()) if y.sum() > 0 else 0.0,
        measured_energy_mwh=float(y.sum() * hours / 1000.0),
        predicted_energy_mwh=float(f.sum() * hours / 1000.0),
    )


def per_bin_error(wind_speed, measured, predicted, *, bin_width: float = 1.0) -> list[dict]:
    """Error as a function of wind speed -- where a curve is wrong matters."""
    u = np.asarray(wind_speed, dtype=float)
    y = np.asarray(measured, dtype=float)
    f = np.asarray(predicted, dtype=float)
    ok = np.isfinite(u) & np.isfinite(y) & np.isfinite(f)
    u, y, f = u[ok], y[ok], f[ok]
    edges = np.arange(0.0, 30.0 + bin_width, bin_width)
    idx = np.digitize(u, edges) - 1
    rows = []
    for k in range(edges.size - 1):
        sel = idx == k
        if sel.sum() < 10:
            continue
        rows.append({
            "speed_lo": float(edges[k]),
            "speed_hi": float(edges[k + 1]),
            "n": int(sel.sum()),
            "measured_kw": float(y[sel].mean()),
            "predicted_kw": float(f[sel].mean()),
            "bias_kw": float((f[sel] - y[sel]).mean()),
            "mae_kw": float(np.abs(f[sel] - y[sel]).mean()),
        })
    return rows


def conformal_interval(
    residuals,
    *,
    level: float = 0.9,
) -> float:
    """Split-conformal half-width from a *calibration* set of residuals.

    Returns the ``level`` empirical quantile of ``|residual|``. Applied to a
    later window it gives a marginal coverage guarantee under exchangeability --
    which a year-later test window does not have. The report states the coverage
    measured on the test window rather than claiming the guarantee.
    """
    r = np.abs(np.asarray(residuals, dtype=float))
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("no residuals in the calibration set")
    return float(np.quantile(r, level))


def calibration_report(residuals, half_width: float) -> dict:
    """Empirical coverage of a fixed interval on a set of residuals."""
    r = np.abs(np.asarray(residuals, dtype=float))
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"n": 0, "coverage": None, "target": None}
    return {
        "n": int(r.size),
        "coverage": float((r <= half_width).mean()),
        "mean_abs_residual_kw": float(r.mean()),
        "p90_abs_residual_kw": float(np.quantile(r, 0.9)),
    }
