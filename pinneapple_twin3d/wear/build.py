"""WearDataset -> Scene (3D viewer) + hub contract + forecast/optimizer report."""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from ..scene import Scene
from .forecast import backtest, fit_zone, forecast, forecast_history, status
from .geometry import cell_mesh, cell_to_vertex
from .model import SYNTHETIC, WearDataset
from .optimize import OptimizerConfig, explain, optimize

GROUP_COLOR = {"wall": (0.62, 0.66, 0.72), "floor": (0.55, 0.58, 0.64), "leg": (0.6, 0.64, 0.7)}


def _ffill_time(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaN along axis 0 (the viewer cannot colour NaN); never-read cells become 0."""
    out = a.copy()
    for t in range(1, len(out)):
        m = ~np.isfinite(out[t])
        out[t][m] = out[t - 1][m]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def remaining_life_fields(ds: WearDataset, cap: float, **kw) -> dict:
    """Per-cell remaining life at every reading, capped at ``cap`` (no trend -> cap; reading 0 -> cap)."""
    out = {}
    for n in ds.zone_names:
        arr = np.full(ds.wear[n].shape, cap)
        for s in range(1, len(ds.x)):
            r = fit_zone(ds, n, s, **kw).remaining
            arr[s] = np.where(np.isfinite(r), np.minimum(r, cap), cap)
        out[n] = arr
    return out


def scene_from_wear(ds: WearDataset, *, life_cap: Optional[float] = None, **fit_kw) -> Scene:
    spec = ds.spec
    title = spec.title + ("  [SYNTHETIC DATA]" if ds.origin == SYNTHETIC else "")
    sc = Scene(title, length_unit="m", times=ds.x, time_unit=spec.x_name,
               source=ds.note or f"wear dataset ({ds.origin})")
    cap = life_cap or float(2 * (ds.x[-1] - ds.x[0]))
    life = remaining_life_fields(ds, cap, **fit_kw)
    for name in ds.zone_names:
        z = spec.zone(name)
        v, f = cell_mesh(z)
        sc.add_part(name, v, f, group=z.group or "wall", color=GROUP_COLOR.get(z.group, (0.6, 0.64, 0.7)))
        w = _ffill_time(ds.wear[name])
        sc.add_field(name, "wear_depth", cell_to_vertex(w), unit="mm")
        sc.add_field(name, "consumed_fraction", cell_to_vertex(w / z.usable), unit="")
        sc.add_field(name, "residual_thickness", cell_to_vertex(z.e0 - w), unit="mm")
        sc.add_field(name, "remaining_life", cell_to_vertex(life[name]), unit=spec.x_name)
    return sc


def report(ds: WearDataset, cfg: Optional[OptimizerConfig] = None, **fit_kw) -> dict:
    """Observed / predicted / recommended, kept apart on purpose."""
    cfg = cfg or OptimizerConfig()
    fc = forecast(ds, **fit_kw)
    res = optimize(fc, cfg)
    s = len(ds.x) - 1
    worst = max((f.worst_fraction for f in fc.values() if np.isfinite(f.worst_fraction)), default=float("nan"))
    inf = lambda v: None if not np.isfinite(v) else float(v)
    pol = lambda p: dict(id=p.id, name=p.name, stops=p.stops, downtime_hours=p.downtime_hours, breach=p.breach,
                         cost=p.cost, schedule=p.schedule)
    return {
        "title": ds.spec.title, "data_origin": ds.origin, "note": ds.note,
        "observed": {"last_x": float(ds.x[s]), "worst_fraction": inf(worst), "status": status(worst) if np.isfinite(worst) else "n/a",
                     "zones": {n: {"worst_fraction": inf(f.worst_fraction), "cells_over_limit": f.n_over} for n, f in fc.items()}},
        "predicted": {"method": "linear least squares on residual thickness (trailing window, repair-aware)",
                      "zones": {n: {"min_remaining": inf(f.min_remaining), "rate_mm_per_x": f.rate, "confidence": f.confidence,
                                    "r2": inf(f.r2), "critical_cell": f.critical_cell} for n, f in fc.items()},
                      "history": [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in h.items()}
                                  for h in forecast_history(ds, **fit_kw)],
                      "backtest": backtest(ds, **fit_kw)},
        "recommended": None if res.recommended is None else {
            "policy": pol(res.recommended), "baseline": pol(res.baseline), "alternatives": [pol(p) for p in res.policies],
            "explanation": explain(res), "assumptions": cfg.__dict__ | {"windows": list(cfg.windows)}, "notes": res.notes,
            "disclaimer": "Decision support only; nothing is sent to equipment."},
    }


def export_wear_twin(ds: WearDataset, folder: str, *, cfg: Optional[OptimizerConfig] = None, usd: bool = False, **fit_kw) -> str:
    """Write the 3D viewer scene, ``wear_dataset.json`` (hub contract) and ``report.json`` to ``folder``."""
    sc = scene_from_wear(ds, **fit_kw)
    path = sc.export(folder, usd=usd)
    with open(os.path.join(folder, "wear_dataset.json"), "w") as f:
        json.dump(ds.to_contract(), f, separators=(",", ":"))
    with open(os.path.join(folder, "report.json"), "w") as f:
        json.dump(report(ds, cfg, **fit_kw), f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return path
