"""Scientific summary utilities: repeated seeds, uncertainty and energy balance.

All metrics are derived from actual model predictions. No fabricated runs.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def seed_summary(rows):
    """Aggregate only finite observed test RMSE values; sample std if n>1."""
    groups={}
    for row in rows:
        if row.get("seed") is None or "rmse_c" not in row:
            continue
        groups.setdefault(row["model"],[]).append(float(row["rmse_c"]))
    return {name:{"n":len(values),"rmse_mean_c":float(np.mean(values)),
                  "rmse_std_c":float(np.std(values,ddof=1)) if len(values)>1 else None,
                  "rmse_min_c":float(np.min(values)),
                  "rmse_max_c":float(np.max(values))}
            for name,values in groups.items()}


def ensemble_interval(predictions, coverage=.90):
    """Empirical across-seed quantiles; not a calibrated predictive interval."""
    p=np.asarray(predictions,dtype=float)
    if p.ndim!=2 or p.shape[0]<2:
        raise ValueError("At least two seed predictions required")
    alpha=(1-coverage)/2
    return np.quantile(p,alpha,axis=0),np.quantile(p,1-alpha,axis=0)


def energy_balance(temperature_c, ambient_c, heat_w, dt_seconds, *,
                   capacity_j_per_k, conductance_w_per_k,
                   heating_efficiency, bias_w=0):
    """Forward finite-difference energy residual in watts.

    C * (T[k+1]-T[k])/dt - H*(Tamb[k]-T[k]) - eta*Q[k] - bias.
    Not a conservation certificate without trustworthy physical parameters.
    """
    t=np.asarray(temperature_c,dtype=float)
    a=np.asarray(ambient_c,dtype=float)
    q=np.asarray(heat_w,dtype=float)
    if len(t)<2 or len(a)!=len(t) or len(q)!=len(t) or dt_seconds<=0:
        raise ValueError("Aligned arrays with at least two samples required")
    residual=(capacity_j_per_k*np.diff(t)/dt_seconds
              -conductance_w_per_k*(a[:-1]-t[:-1])
              -heating_efficiency*q[:-1]-bias_w)
    return {"residual_rmse_w":float(np.sqrt(np.mean(residual**2))),
            "residual_mean_w":float(np.mean(residual)),
            "residual_max_abs_w":float(np.max(np.abs(residual)))}


def write_summary(rows, predictions, outdir):
    outdir=Path(outdir)
    summary=seed_summary(rows)
    (outdir/"seed_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    names=sorted(set(k.rsplit("_seed",1)[0] for k in predictions if "_seed" in k))
    intervals={}
    for name in names:
        members=[v for k,v in predictions.items() if k.startswith(name+"_seed")]
        if len(members)<2:
            continue
        lo,hi=ensemble_interval(members)
        intervals[name+"_p05"]=lo
        intervals[name+"_p95"]=hi
    if intervals:
        import pandas as pd
        pd.DataFrame(intervals).to_csv(outdir/"ensemble_intervals.csv",index=False)
    return summary
