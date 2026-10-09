"""Scores of the reference forecasts on the 2020 test cases (00 UTC every 3 days, leads to 10 days): persistence,
climatology and the published IFS HRES, Pangu-Weather, Keisler (GNN) and NeuralGCM forecasts (WeatherBench2).
Writes ``baselines_2020.json`` next to the store."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pinneapple_physics.weather.data import BASELINES_64x32, Era5Store, climatology
from pinneapple_physics.weather.evaluate import (
    PublishedForecasts,
    climatology_forecast,
    persistence,
    score_forecasts,
)

STEPS = 40


def test_inits(store: Era5Store, year: int = 2020, every_days: int = 3) -> np.ndarray:
    idx = store.years(year, year)
    idx = idx[(store.times[idx].astype("datetime64[h]").astype(int) % 24 == 0)]
    idx = idx[::every_days]
    return idx[idx + STEPS < len(store.times)]


def main(store_path: str):
    store = Era5Store(store_path)
    clim = climatology(store)
    inits = test_inits(store)
    pub = PublishedForecasts(store)
    fcs = {"persistence": persistence(store), "climatology": climatology_forecast(store, clim)}
    fcs.update({n: pub(n) for n in BASELINES_64x32})
    res = score_forecasts(store, clim, fcs, inits, STEPS)
    out = {n: {v: {k: np.nanmean(a, 0).tolist() for k, a in d.items()} | {"n": int(np.isfinite(d["rmse"][:, -1]).sum())}
               for v, d in r.items()} for n, r in res.items()}
    out["_meta"] = {"inits": [str(t) for t in store.times[inits]], "leads_h": (np.arange(STEPS + 1) * 6).tolist()}
    Path(store_path, "baselines_2020.json").write_text(json.dumps(out))
    for n in res:
        z = out[n]["z500"]
        print(f"{n:16s} z500 RMSE day3 {z['rmse'][12]:.0f} day5 {z['rmse'][20]:.0f} | ACC day5 {z['acc'][20]:.3f}"
              f" day7 {z['acc'][28]:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
