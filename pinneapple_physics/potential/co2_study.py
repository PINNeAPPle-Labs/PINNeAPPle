"""Measured study: a potential-based surrogate against a direct multi-output surrogate on CO2 near its critical point.

Data: CoolProp's Helmholtz-energy equation of state for CO2 (HEOS backend, Span & Wagner, J. Phys. Chem. Ref. Data
25 (1996) 1509-1596), the same backend ``pinneapple_systems.process_components.real_gas_eos`` wraps. Six specific
properties at given ``(T, rho)``: ``p, s, u, c_v, c_p, w``.

Protocol (same for every model):

* training region ``T`` in [305, 335] K (just above ``T_c`` = 304.13 K), ``rho`` in [100, 800] kg/m3, where ``c_p``
  peaks along the Widom line; ``n_per_prop`` random points per property, each point carrying one property (mixed
  data);
* the same data, loss (:func:`~pinneapple_physics.potential.surrogate.fit`: per-property standardised squared error,
  log for the heat capacities), optimiser (Adam, cosine decay), number of steps, width and depth for every model;
* scores on a held-out grid inside the training region (``interp``) and on two extrapolation regions: hotter
  (``T`` in [335, 365] K) and denser (``rho`` in [800, 950] kg/m3); normalised RMS error per property (RMS error over
  the standard deviation of the property in the training region), the thermodynamic identities of
  :func:`~pinneapple_physics.potential.thermo.identities_T_rho` checked by
  :func:`~pinneapple_physics.potential.relations.consistency_check`, and the share of points with ``(dp/drho)_T > 0``
  and ``c_v > 0`` (thermodynamic stability);
* several seeds; intervals are 95 % Student-t intervals over seeds, and the comparison is paired by seed.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from .relations import consistency_check
from .surrogate import (
    PROPERTIES,
    DirectSurrogate,
    MixedData,
    PotentialSurrogate,
    Scaling,
    fit,
    n_parameters,
)

__all__ = ["T_CRIT", "RHO_CRIT", "REGIONS", "MODELS", "co2_properties", "make_mixed", "region_grid", "default_grids",
           "run_seed", "run_study", "summarize", "interval"]

T_CRIT, RHO_CRIT = 304.1282, 467.6  # K, kg/m3 (Span & Wagner 1996)

#: name -> ((T_lo, T_hi), (rho_lo, rho_hi))
REGIONS = {
    "train": ((305.0, 335.0), (100.0, 800.0)),
    "hotter": ((335.0, 365.0), (100.0, 800.0)),
    "denser": ((305.0, 335.0), (800.0, 950.0)),
}

MODELS = ("direct", "potential", "potential_icnn")


def co2_properties(T, rho) -> dict[str, np.ndarray]:
    """``p`` (Pa), ``s`` (J/(kg K)), ``u`` (J/kg), ``cv``, ``cp`` (J/(kg K)), ``w`` (m/s) of CO2 from CoolProp's
    HEOS backend at temperatures ``T`` (K) and densities ``rho`` (kg/m3)."""
    import CoolProp.CoolProp as CP

    AS = CP.AbstractState("HEOS", "CO2")
    T, rho = np.broadcast_arrays(np.asarray(T, float), np.asarray(rho, float))
    out = {k: np.empty(T.shape) for k in PROPERTIES}
    for idx in np.ndindex(T.shape):
        AS.update(CP.DmassT_INPUTS, float(rho[idx]), float(T[idx]))
        out["p"][idx], out["s"][idx], out["u"][idx] = AS.p(), AS.smass(), AS.umass()
        out["cv"][idx], out["cp"][idx], out["w"][idx] = AS.cvmass(), AS.cpmass(), AS.speed_sound()
    return out


def make_mixed(n_per_prop: int = 200, seed: int = 0, props: Sequence[str] = PROPERTIES,
               region: str = "train") -> MixedData:
    """Random points in ``region``, ``n_per_prop`` per property, each carrying that one property."""
    rng = np.random.default_rng(seed)
    (t0, t1), (r0, r1) = REGIONS[region]
    T, R, P, Y = [], [], [], []
    for p in props:
        t = rng.uniform(t0, t1, n_per_prop)
        r = rng.uniform(r0, r1, n_per_prop)
        T.append(t)
        R.append(r)
        P += [p] * n_per_prop
        Y.append(co2_properties(t, r)[p])
    return MixedData(np.concatenate(T), np.concatenate(R), np.array(P, dtype=object), np.concatenate(Y))


def region_grid(region: str, n: int = 21) -> dict[str, np.ndarray]:
    """A regular ``n x n`` grid of ``region`` (flattened) with all properties."""
    (t0, t1), (r0, r1) = REGIONS[region]
    TT, RR = np.meshgrid(np.linspace(t0, t1, n), np.linspace(r0, r1, n), indexing="ij")
    T, R = TT.reshape(-1), RR.reshape(-1)
    return {"T": T, "rho": R, **co2_properties(T, R)}


def build(kind: str, scaling: Scaling, hidden: int, layers: int):
    if kind == "direct":
        return DirectSurrogate(scaling, hidden, layers)
    if kind == "potential":
        return PotentialSurrogate(scaling, hidden, layers)
    if kind == "potential_icnn":
        return PotentialSurrogate(scaling, hidden, layers, convex=True)
    raise KeyError(kind)


def _scores(model, grid: dict, sd: dict[str, float]) -> dict:
    pred = model.predict(grid["T"], grid["rho"])
    nrmse = {p: float(np.sqrt(np.mean((pred[p] - grid[p]) ** 2)) / sd[p]) for p in PROPERTIES}
    mare = {p: float(np.median(np.abs(pred[p] / grid[p] - 1))) for p in ("p", "cv", "cp", "w")}
    T, r = grid["T"], grid["rho"]
    h = 1e-3 * r
    dp = (model.predict(T, r + h)["p"] - model.predict(T, r - h)["p"]) / (2 * h)
    stable = float(np.mean((dp > 0) & (pred["cv"] > 0)))

    def values(pt):
        return model.predict(pt["T"], pt["rho"])

    cons = consistency_check(values, "helmholtz_T_rho", at={"T": T, "rho": r}, tol=1e-3, step=1e-4)
    return {"nrmse": nrmse, "mare": mare, "stable_fraction": stable,
            "identities": {row["name"]: row["rel_p95"] for row in cons.rows},
            "identities_median": {row["name"]: row["rel_median"] for row in cons.rows}}


def default_grids(n: int = 21) -> dict[str, dict[str, np.ndarray]]:
    """Held-out grid inside the training region and the two extrapolation grids."""
    return {"interp": region_grid("train", n), "hotter": region_grid("hotter", n), "denser": region_grid("denser", n)}


def run_seed(seed: int, *, n_per_prop: int = 200, epochs: int = 3000, hidden: int = 32, layers: int = 3,
             lr: float = 1e-3, props: Sequence[str] = PROPERTIES, models: Sequence[str] = MODELS,
             grids: dict | None = None, log_every: int = 0) -> dict:
    """Train every model of ``models`` on the same mixed data (drawn with ``seed``; ``props``: the properties
    the training samples carry) and score them on every grid (all six properties)."""
    data = make_mixed(n_per_prop, seed=1000 + seed, props=props)
    scaling = Scaling.from_data(data)
    grids = grids or default_grids()
    sd = {p: float(np.std(grids["interp"][p])) for p in PROPERTIES}
    out = {"seed": seed, "n_train": len(data), "counts": data.counts(), "models": {}}
    for kind in models:
        torch.manual_seed(seed)
        model = build(kind, scaling, hidden, layers)
        h = fit(model, data, epochs=epochs, lr=lr, seed=seed, log_every=log_every)
        res = {"parameters": n_parameters(model), "seconds": h["seconds"], "final_loss": h["loss"][-1],
               "loss_curve": h["loss"][:: max(1, epochs // 100)]}
        for g, grid in grids.items():
            res[g] = _scores(model, grid, sd)
        out["models"][kind] = res
    return out


def interval(x: Sequence[float], level: float = 0.95) -> dict[str, float]:
    """Mean and Student-t interval over seeds."""
    from scipy import stats

    x = np.asarray(x, float)
    m = float(x.mean())
    if len(x) < 2:
        return {"mean": m, "lo": float("nan"), "hi": float("nan"), "n": len(x)}
    hw = float(stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return {"mean": m, "lo": m - hw, "hi": m + hw, "n": len(x)}


def summarize(runs: Sequence[dict], baseline: str = "direct") -> dict:
    """Intervals over seeds for every model, region and metric, and paired differences against ``baseline``
    (log10 ratio of normalised errors: negative means the model beats the baseline)."""
    models = list(runs[0]["models"])
    regions = [k for k in runs[0]["models"][models[0]] if isinstance(runs[0]["models"][models[0]][k], dict)]
    out = {"seeds": [r["seed"] for r in runs], "models": {}, "paired": {}}
    for m in models:
        rm = {"parameters": runs[0]["models"][m]["parameters"],
              "seconds": interval([r["models"][m]["seconds"] for r in runs])}
        for g in regions:
            sc = [r["models"][m][g] for r in runs]
            rm[g] = {
                "nrmse": {p: interval([s["nrmse"][p] for s in sc]) for p in PROPERTIES},
                "nrmse_mean": interval([np.mean(list(s["nrmse"].values())) for s in sc]),
                "mare": {p: interval([s["mare"][p] for s in sc]) for p in sc[0]["mare"]},
                "stable_fraction": interval([s["stable_fraction"] for s in sc]),
                "identities": {k: interval([s["identities"][k] for s in sc]) for k in sc[0]["identities"]},
            }
        out["models"][m] = rm
    for m in models:
        if m == baseline:
            continue
        pm = {}
        for g in regions:
            pm[g] = {p: interval([math.log10(r["models"][m][g]["nrmse"][p] / r["models"][baseline][g]["nrmse"][p])
                                  for r in runs]) for p in PROPERTIES}
            pm[g]["mean"] = interval([math.log10(np.mean(list(r["models"][m][g]["nrmse"].values()))
                                                 / np.mean(list(r["models"][baseline][g]["nrmse"].values())))
                                      for r in runs])
        out["paired"][m] = pm
    return out


def run_study(seeds: Sequence[int] = range(5), grids: dict | None = None, **kw) -> dict:
    """All seeds, then :func:`summarize`. Returns ``{"config": ..., "runs": [...], "summary": ...}``."""
    grids = grids or default_grids()
    runs = [run_seed(s, grids=grids, **kw) for s in seeds]
    cfg = {"regions": REGIONS, "grid_points": {g: int(len(v["T"])) for g, v in grids.items()}, **kw,
           "seeds": list(seeds), "data": "CoolProp HEOS CO2 (Span & Wagner 1996)"}
    return {"config": cfg, "runs": runs, "summary": summarize(runs)}
