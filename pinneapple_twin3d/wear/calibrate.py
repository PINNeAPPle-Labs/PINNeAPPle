"""Calibrate the optimizer's assumptions from the plant's own records.

What the data can and cannot tell:

=====================  ===========================================  =========================================
parameter              estimated from                               needs
=====================  ===========================================  =========================================
setup_hours,           stop log: ``duration = setup + per_zone * n``  >= 3 stops, >= 2 different zone counts
hours_per_zone
cost_per_hour,         stop log with a cost: ``cost = c_h*duration     >= 4 stops with cost
cost_per_zone          + c_z*n_zones``
recovery_frac          the dataset itself: size of the wear drop      >= 20 repaired cells
                       at each detected repair / usable thickness
breach_cost_per_x,     **business inputs** (what a heat beyond the    nothing: never inferred
safety_margin, ...     limit costs, how much margin you want)
=====================  ===========================================  =========================================

Every parameter is reported with its source (``calibrated`` / ``default`` / ``business input``), the sample size and
the standard error where one exists; a fit that gives a negative time or cost is rejected and the default is kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

import numpy as np

from .model import WearDataset
from .optimize import OptimizerConfig

BUSINESS = ("breach_cost_per_x", "safety_margin", "horizon", "heats_per_day", "fixed_interval")


@dataclass(frozen=True)
class StopRecord:
    zones: Sequence[str]
    duration_h: float
    cost: Optional[float] = None
    x: Optional[float] = None  # when it happened (information only)


@dataclass
class ParamReport:
    name: str
    value: float
    source: str  # calibrated | default | business input
    n: int = 0
    se: Optional[float] = None
    note: str = ""


@dataclass
class Calibration:
    cfg: OptimizerConfig
    report: List[ParamReport] = field(default_factory=list)


def _ols(X: np.ndarray, y: np.ndarray):
    """Least squares with standard errors. Returns (beta, se, dof) with se=None when dof < 1."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(y) - X.shape[1]
    if dof < 1:
        return beta, None, dof
    res = y - X @ beta
    cov = (res @ res / dof) * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.maximum(np.diag(cov), 0.0)), dof


def estimate_recovery(ds: WearDataset, repair_drop: float = 0.08, min_cells: int = 20):
    """Median fraction of the usable thickness restored per repaired cell, and the number of cells used."""
    fr = []
    for name in ds.zone_names:
        z = ds.spec.zone(name)
        w = ds.wear[name]
        drop = w[:-1] - w[1:]
        m = np.isfinite(drop) & (drop > repair_drop * z.usable)
        fr.extend((drop[m] / z.usable).tolist())
    return (float(np.median(fr)) if len(fr) >= min_cells else None), len(fr)


def calibrate(stops: Sequence[StopRecord] = (), ds: Optional[WearDataset] = None,
              base: Optional[OptimizerConfig] = None) -> Calibration:
    base = base or OptimizerConfig()
    out = {}
    rep: List[ParamReport] = []

    def keep(name, why):
        rep.append(ParamReport(name, getattr(base, name), "default", note=why))

    # --- stop durations
    stops = [s for s in stops if np.isfinite(s.duration_h) and s.duration_h > 0 and len(s.zones) > 0]
    nz = np.array([len(s.zones) for s in stops], float)
    if len(stops) >= 3 and len(set(nz)) >= 2:
        beta, se, dof = _ols(np.column_stack([np.ones(len(stops)), nz]), np.array([s.duration_h for s in stops]))
        if beta[0] >= 0 and beta[1] >= 0:
            out.update(setup_hours=float(beta[0]), hours_per_zone=float(beta[1]))
            rep.append(ParamReport("setup_hours", float(beta[0]), "calibrated", len(stops), None if se is None else float(se[0])))
            rep.append(ParamReport("hours_per_zone", float(beta[1]), "calibrated", len(stops), None if se is None else float(se[1])))
        else:
            why = f"fit gave setup={beta[0]:.2f} h, per-zone={beta[1]:.2f} h (negative): rejected, default kept"
            keep("setup_hours", why); keep("hours_per_zone", why)
    else:
        why = f"needs >= 3 stops with >= 2 different zone counts (got {len(stops)} stops, {len(set(nz))} counts)"
        keep("setup_hours", why); keep("hours_per_zone", why)

    # --- costs
    costed = [s for s in stops if s.cost is not None and np.isfinite(s.cost) and s.cost >= 0]
    if len(costed) >= 4:
        X = np.column_stack([[s.duration_h for s in costed], [len(s.zones) for s in costed]])
        beta, se, dof = _ols(X, np.array([s.cost for s in costed], float))
        if np.all(beta >= 0):
            out.update(cost_per_hour=float(beta[0]), cost_per_zone=float(beta[1]))
            rep.append(ParamReport("cost_per_hour", float(beta[0]), "calibrated", len(costed), None if se is None else float(se[0])))
            rep.append(ParamReport("cost_per_zone", float(beta[1]), "calibrated", len(costed), None if se is None else float(se[1])))
        else:
            why = "cost fit gave a negative coefficient: rejected, default kept"
            keep("cost_per_hour", why); keep("cost_per_zone", why)
    else:
        why = f"needs >= 4 stops with a cost (got {len(costed)}); costs stay in relative units"
        keep("cost_per_hour", why); keep("cost_per_zone", why)

    # --- recovery per gunning, from the data
    if ds is not None:
        rec, n = estimate_recovery(ds)
        if rec is not None:
            out["recovery_frac"] = float(min(max(rec, 0.05), 1.0))
            rep.append(ParamReport("recovery_frac", out["recovery_frac"], "calibrated", n, note="median wear drop / usable thickness at detected repairs"))
        else:
            keep("recovery_frac", f"only {n} repaired cells in the dataset (needs 20)")
    else:
        keep("recovery_frac", "no dataset given")

    for name in BUSINESS:
        rep.append(ParamReport(name, getattr(base, name), "business input", note="not inferable from plant records"))
    order = ["setup_hours", "hours_per_zone", "cost_per_hour", "cost_per_zone", "recovery_frac", *BUSINESS]
    rep.sort(key=lambda r: order.index(r.name))
    return Calibration(replace(base, **out), rep)
