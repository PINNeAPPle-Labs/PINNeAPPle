"""One ensemble of IFS HRES, Pangu-Weather and the PINNeAPPle weather model on 2020: the skill of the ensemble next to
each model alone, the weights it learns per region and season, and the physics residual that flags where each model
leaves its domain (#348).

The three models are wrapped by :func:`pinneapple_physics.pretrained.from_pretrained_expert`, so each arrives with its
source, its licence and a cost estimate in the provenance, and all three are scored on the same cases with the same
arithmetic. The ensemble is :class:`pinneapple_physics.ensemble.PhysicsEnsemble` in ``combine`` mode, one per
(region, season) cell, with the label-free checks of :mod:`pinneapple_physics.domain_checks` as
``prediction_checks``:

* **advection** -- the part of the 6-hour change of ``z500`` that the model's *own* 500 hPa wind cannot explain;
* **energy** -- the drift per day of the area-weighted dry static energy ``cp T700 + g z700``, in units of the field's
  own spatial spread.

Both are read against the analysis itself: a model is out of its domain on a case when it does worse than twice the
level ERA5 scores on the same cases (the discretisation floor -- measured here, not declared, and printed in
``calibration``).

Cases are the 00 UTC initialisations of 2020 every ``--every-days`` days, each carried to +240 h. Regions are latitude
bands (the grid is 64 x 32, 5.625 degrees) and seasons are DJF/MAM/JJA/SON. The weights are learned online in
chronological order inside each cell, from the cases already seen only, so the error reported for a case is the one the
ensemble made *before* seeing that case's truth.

    python examples/weather_forecasting/pretrained_ensemble.py wx/era5 \\
        examples/weather_forecasting/checkpoints/phase1_step15000_fp16.pt --out wx/pretrained --every-days 15

Needs a network the first time (the published archives are read over HTTPS, one read per model and case, cached under
``--out/_fields`` for the reruns), the ERA5 2020 store and the phase-1 checkpoint named above. Writes
``pretrained_ensemble_2020.json`` under ``--out`` and prints the tables.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_physics.domain_checks import (
    advection_check,
    advection_residual,
    as_prediction_check,
    dry_static_energy,
    energy_check,
    global_drift,
)
from pinneapple_physics.ensemble import Expert, PhysicsEnsemble
from pinneapple_physics.pretrained import (
    TorchCheckpointAdapter,
    WeatherBench2Adapter,
    from_pretrained_expert,
    licence_report,
)
from pinneapple_physics.weather.data import Era5Store
from pinneapple_physics.weather.evaluate import WB2_CHECKS

STEPS = 40                                   # 6-hour steps scored: +6 h to +240 h
HOURS = 6.0
#: What the ensemble scores and what the checks need, in one read per model and case: the tracer, the wind that should
#: carry it, and the two fields of a dry static energy at one level (700 hPa: the published IFS HRES archive has
#: leads with no geopotential at 850 hPa, see ``WB2_CHECKS``).
NEEDED = ("z500", "u500", "v500", "t700", "z700")
PUBLISHED = {"ifs-hres": "IFS HRES", "pangu-weather": "Pangu-Weather"}
HOME = "pinneapple-weather"
REGIONS = {"tropics": (-20.0, 20.0), "midlatitudes N": (30.0, 70.0), "midlatitudes S": (-70.0, -30.0),
           "Arctic": (70.0, 90.0)}
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


# ---------------------------------------------------------------------------------------------------- the cases

def test_inits(store: Era5Store, year: int = 2020, every_days: int = 15) -> np.ndarray:
    """00 UTC initialisations of ``year`` every ``every_days`` days that have a previous state (the model trained here
    starts from two) and ``STEPS * 6`` hours of truth after them."""
    idx = store.years(year, year)
    idx = idx[store.times[idx].astype("datetime64[h]").astype(int) % 24 == 0]
    idx = idx[::every_days]
    return idx[(idx >= 1) & (idx + STEPS < len(store.times))]


def season_of(time) -> str:
    month = int(np.datetime64(time, "ns").astype("datetime64[M]").astype(int) % 12) + 1
    return next(s for s, months in SEASONS.items() if month in months)


def band(store: Era5Store, box: tuple[float, float]) -> tuple[slice, np.ndarray, np.ndarray]:
    """Row slice of a latitude band (the grid runs north to south), and the latitudes and longitudes it keeps."""
    lo, hi = box
    rows = np.nonzero((store.lat >= lo) & (store.lat <= hi))[0]
    sl = slice(int(rows.min()), int(rows.max()) + 1)
    return sl, store.lat[sl], store.lon


# ------------------------------------------------------------------------------------------------ the experts

def build_experts(store: Era5Store, checkpoint: str) -> dict[str, Expert]:
    """The three models as experts, each with its licence and its cost estimate in the provenance.

    Keyed by the expert's own name -- the archive adapters carry the published model's display name (``"IFS HRES"``,
    the key ``BASELINES_64x32`` uses), which is what the ensemble, the weights and :func:`licence_report` call it."""
    built = [from_pretrained_expert(name, WeatherBench2Adapter(store, wb, variables=WB2_CHECKS), lazy=True)
             for name, wb in PUBLISHED.items()]
    built.append(from_pretrained_expert(HOME, TorchCheckpointAdapter(store, checkpoint)))
    return {e.name: e for e in built}


def fetch(expert: Expert, i0: int, cache: Path) -> dict[str, np.ndarray]:
    """Every channel of :data:`NEEDED` for one initialisation, in one call, cached on disk: an archive is read once per
    case and model (about 20 s over HTTPS) and a rerun of this script is offline. The number of leads is part of the
    cache key, and a file that holds fewer is read again: a cache from a short smoke run must not shorten a long one."""
    f = cache / f"{expert.name}_{i0:05d}_{STEPS:03d}.npz"
    if f.exists():
        with np.load(f) as z:
            if all(z[v].shape[0] == STEPS + 1 for v in NEEDED):
                return {v: z[v] for v in NEEDED}
    stack = np.asarray(expert.predict({"i0": int(i0), "steps": STEPS, "variable": list(NEEDED)}), float)
    if stack.shape[1] != STEPS + 1:
        raise RuntimeError(f"{expert.name}: asked for {STEPS} steps, got {stack.shape[1] - 1}")
    fields = {v: stack[k] for k, v in enumerate(NEEDED)}
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **fields)
    return fields


def analysis_fields(store: Era5Store, i0: int) -> dict[str, np.ndarray]:
    """The analysis (ERA5) over the same window, in physical units: the truth of the scores."""
    out = {}
    for v in NEEDED:
        c = store.channels.index(v)
        out[v] = store.denorm(np.asarray(store.state[i0:i0 + STEPS + 1, c], np.float32), v)
    return out


def cell_fields(fields_by_case: dict[int, dict[str, np.ndarray]], sl: slice, drop_analysis: bool = True):
    """One model's fields over a region, leads 6 h to +240 h (lead 0 is the analysis every model starts from, so
    leaving it in would score IFS HRES, which has none, against a model that is handed it)."""
    out = {}
    for i0, f in fields_by_case.items():
        out[i0] = {v: (f[v][1:] if drop_analysis else f[v])[:, sl, :] for v in f}
    return out


# ------------------------------------------------------------------------------------------------- the checks

def cell_checks(fields: dict[int, dict[str, np.ndarray]], lat: np.ndarray, lon: np.ndarray, adv_tol: float,
                energy_tol: float) -> tuple[dict[str, Any], Any]:
    """The two checks of one model over one region: the ``ValidityCheck`` objects (for their measured values) and the
    single ``(name, prediction, query)`` callable ``PhysicsEnsemble(prediction_checks=...)`` takes, the worst of the
    two scores deciding and the reason naming both."""
    def wind_of(q):
        f = fields[int(q["i0"])]
        return f["u500"], f["v500"]

    def energy_of(_prediction, q):
        f = fields[int(q["i0"])]
        return dry_static_energy(f["t700"], f["z700"])

    checks = {"advection": advection_check(wind_of, adv_tol, lat, lon, hours=HOURS),
              "energy": energy_check(energy_tol, lat, hours=HOURS, field_fn=energy_of)}
    scored = {k: as_prediction_check(c) for k, c in checks.items()}

    def fn(prediction, query):
        got = {k: scored[k](prediction, query) for k in scored}
        score, reason = min(got.items(), key=lambda kv: kv[1][0])[1]
        return score, "; ".join(got[k][1] for k in got)

    return checks, fn


def calibrate(analysis: dict[int, dict[str, np.ndarray]], lat: np.ndarray, lon: np.ndarray) -> dict[str, float]:
    """The level the analysis itself scores on these cases: the floor both checks are read against. Twice it is the
    tolerance, so a model is flagged where it is worse than the analysis and the discretisation put together."""
    adv, ene = [], []
    for f in analysis.values():
        adv.append(advection_residual(f["z500"][1:], f["u500"][1:], f["v500"][1:], HOURS, lat, lon))
        ene.append(global_drift(dry_static_energy(f["t700"][1:], f["z700"][1:]), lat, HOURS))
    return {"analysis_advection_residual": float(np.median(adv)),
            "analysis_energy_drift": float(np.median(ene)),
            "advection_tolerance": float(2.0 * np.median(adv)),
            "energy_tolerance": float(max(2.0 * np.median(ene), 0.01))}


# ---------------------------------------------------------------------------------------------------- a cell

def run_cell(experts: dict[str, Expert], names: list[str], cases: list[int], truth: dict[int, dict[str, np.ndarray]],
             model: dict[str, dict[int, dict[str, np.ndarray]]], lat: np.ndarray, lon: np.ndarray,
             cal: dict[str, float]) -> dict[str, Any]:
    """One (region, season) cell: the three models and their ensemble on its cases, in chronological order."""
    raw, callables = {}, {}
    for n in names:
        raw[n], callables[n] = cell_checks(model[n], lat, lon, cal["advection_tolerance"], cal["energy_tolerance"])

    def prediction_checks(name, prediction, query):
        return callables[name](prediction, query)

    members = [Expert(e.name, lambda q, n=e.name: model[n][int(q["i0"])]["z500"], e.cost, e.kind, e.provenance)
               for e in (experts[n] for n in names)]
    # the area weights of one cell, as ``relative_l2`` and the checks want them: a prediction here is the whole window,
    # ``(steps, lat, lon)`` (lead 0 is the analysis every model was handed, see ``cell_fields``)
    weight = np.cos(np.deg2rad(lat))[None, :, None]
    ens = PhysicsEnsemble(members, mode="combine", point_weights=weight, prediction_checks=prediction_checks,
                          check_threshold=0.5, min_weight=0.05)
    rows = []
    day5 = min(19, STEPS - 1)                        # lead 0 is dropped: index 19 is +120 h at 6-hour steps
    area = weight[0, :, 0][:, None]                  # cos(latitude) per row, for the RMSE of one lead
    for i0 in cases:
        q = {"i0": int(i0), "steps": STEPS}
        p = ens.predict(q)
        info = ens.update(q, truth[i0]["z500"])
        alive = [n for n in names if n not in p["failed_checks"]]
        share_total = sum(float(p["weights"][n]) for n in alive) or 1.0
        # every model that was run is reported, including the ones a check took out of the combination for this case:
        # the demo is the skill of the ensemble *next to each model alone*, so no model may go missing from the table
        rows.append({
            "i0": int(i0),
            "expert_errors": {n: float(info["errors"][k]) for k, n in enumerate(names)},
            "ensemble_error": float(info["ensemble_error"]),
            "expert_rmse_day5": {n: float(np.sqrt(np.mean((model[n][i0]["z500"][day5] - truth[i0]["z500"][day5]) ** 2
                                                          * area)))
                                 for n in names},
            "ensemble_rmse_day5": float(np.sqrt(np.mean((p["prediction"][day5] - truth[i0]["z500"][day5]) ** 2
                                                        * area))),
            "weights": {n: float(p["weights"][n]) for n in names},
            # the share of the *combination* this case: a model a check took out counts zero here, while its learned
            # weight above is kept for when the case comes back to its domain
            "share": {n: (float(p["weights"][n]) / share_total if n not in p["failed_checks"] else 0.0) for n in names},
            "residuals": {n: {k: float(raw[n][k](model[n][i0]["z500"], q).value) for k in raw[n]} for n in names},
            "failed": [n for n in p["failed_checks"]],
            "reasons": {n: p["checks"][n]["reason"] for n in p["failed_checks"] if n in p["checks"]},
            "active": p["active"],
        })
    return {"rows": rows}


def summarise(names: list[str], cell: dict[str, Any]) -> dict[str, Any]:
    rows = cell["rows"]

    def mean_over(key: str, n: str) -> float:
        vals = [r[key][n] for r in rows if n in r.get(key, {})]
        return float(np.mean(vals)) if vals else float("nan")

    def median_of(check: str, n: str) -> float:
        vals = [r["residuals"][n][check] for r in rows if n in r["residuals"]]
        return float(np.median(vals)) if vals else float("nan")

    out = {"n_cases": len(rows),
           "error": {n: mean_over("expert_errors", n) for n in names},
           "weights": {n: mean_over("weights", n) for n in names},
           "share": {n: mean_over("share", n) for n in names},
           "out_of_domain": {n: float(np.mean([n in r["failed"] for r in rows])) for n in names},
           "advection_residual": {n: median_of("advection", n) for n in names},
           "energy_drift": {n: median_of("energy", n) for n in names},
           # the first sentence a check gave for taking this model out of the case, kept verbatim: it is the evidence
           "failure_reason": {n: next((r["reasons"][n] for r in rows if n in r.get("reasons", {})), "") for n in names},
           "rmse_day5": {n: mean_over("expert_rmse_day5", n) for n in names}}
    out["ensemble"] = float(np.mean([r["ensemble_error"] for r in rows]))
    out["ensemble_rmse_day5"] = float(np.mean([r["ensemble_rmse_day5"] for r in rows]))
    best = min(out["error"], key=lambda n: out["error"][n])
    out["best_single"] = {"name": best, "error": out["error"][best]}
    out["ensemble_beats_best_single"] = bool(out["ensemble"] < out["error"][best])
    out["ensemble_wins"] = float(np.mean([r["ensemble_error"] < min(r["expert_errors"].values()) for r in rows]))
    return out


# ---------------------------------------------------------------------------------------------------- the run

def main(argv: list[str] | None = None) -> dict[str, Any]:
    global STEPS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("store", help="ERA5 2020 store written by pinneapple_physics.weather.data.download")
    ap.add_argument("checkpoint", help="phase-1 checkpoint of pinneapple_physics.weather.train")
    ap.add_argument("--out", default="wx/pretrained", help="where the JSON, the field cache and the tables go")
    ap.add_argument("--every-days", type=int, default=15, help="spacing of the 00 UTC initialisations of 2020")
    ap.add_argument("--regions", default="", help="comma-separated subset of the regions (default: all four)")
    ap.add_argument("--steps", type=int, default=STEPS, help="6-hour steps scored after the analysis (default 40)")
    args = ap.parse_args(argv)
    STEPS = int(args.steps)

    t_start = time.perf_counter()
    out = Path(args.out)
    cache = out / "_fields"
    store = Era5Store(args.store)
    inits = [int(i) for i in test_inits(store, 2020, args.every_days)]
    experts = build_experts(store, args.checkpoint)
    names = list(experts)
    regions = {k: v for k, v in REGIONS.items() if not args.regions or k in args.regions.split(",")}
    print(f"{len(inits)} initialisations of 2020, {store.times[inits[0]]} to {store.times[inits[-1]]}, "
          f"+{STEPS * 6} h, {len(regions)} regions x {len(SEASONS)} seasons", flush=True)

    truth, model = {}, {n: {} for n in names}
    for k, i0 in enumerate(inits):
        t0 = time.perf_counter()
        truth[i0] = analysis_fields(store, i0)
        for n in names:
            model[n][i0] = fetch(experts[n], i0, cache)
        print(f"  case {k + 1:2d}/{len(inits)} {str(np.datetime64(store.times[i0], 'h'))} "
              f"fetched in {time.perf_counter() - t0:.1f} s", flush=True)

    result = {"steps": STEPS, "hours": HOURS, "every_days": args.every_days,
              "inits": [str(np.datetime64(store.times[i], "h")) for i in inits],
              "licences": licence_report([experts[n] for n in names])["licences"],
              "calibration": {}, "cells": {}, "models": {}}
    p = out / "pretrained_ensemble_2020.json"

    def save():
        out.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=1))

    for region, box in regions.items():
        sl, lat, lon = band(store, box)
        cal = calibrate({i: {v: a[v][:, sl, :] for v in a} for i, a in truth.items()}, lat, lon)
        result["calibration"][region] = cal
        print(f"{region}: analysis advection {cal['analysis_advection_residual']:.3f} "
              f"energy {cal['analysis_energy_drift']:.4f} spreads/day "
              f"-> tolerances {cal['advection_tolerance']:.3f} / {cal['energy_tolerance']:.3f}", flush=True)
        truth_r = cell_fields(truth, sl)
        model_r = {n: cell_fields(model[n], sl) for n in names}
        for season in SEASONS:
            cases = [i for i in inits if season_of(store.times[i]) == season]
            if not cases:
                continue
            got = summarise(names, run_cell(experts, names, cases, truth_r, model_r, lat, lon, cal))
            got["inits"] = [str(np.datetime64(store.times[c], "h")) for c in cases]
            result["cells"][f"{region}/{season}"] = got
            save()                                    # a long run that dies keeps every cell already measured
            print(f"  {region:15s} {season} n={got['n_cases']:2d} z500 rel-L2 ensemble {got['ensemble']:.3f} | "
                  + " ".join(f"{n} {got['error'][n]:.3f}" for n in names)
                  + " | weights " + " ".join(f"{n.split('-')[0]} {got['weights'][n]:.2f}" for n in names)
                  + " | out " + " ".join(f"{n.split('-')[0]} {100 * got['out_of_domain'][n]:.0f}%" for n in names),
                  flush=True)

    cells = list(result["cells"].values())
    result["models"] = {
        "ensemble": float(np.mean([c["ensemble"] for c in cells])),
        "ensemble_rmse_day5": float(np.mean([c["ensemble_rmse_day5"] for c in cells])),
        **{n: float(np.mean([c["error"][n] for c in cells])) for n in names},
        "rmse_day5": {n: float(np.mean([c["rmse_day5"][n] for c in cells])) for n in names},
        "weights": {n: float(np.mean([c["weights"][n] for c in cells])) for n in names},
        "share": {n: float(np.mean([c["share"][n] for c in cells])) for n in names},
        "out_of_domain": {n: float(np.mean([c["out_of_domain"][n] for c in cells])) for n in names},
        "advection_residual": {n: float(np.mean([c["advection_residual"][n] for c in cells])) for n in names},
        "energy_drift": {n: float(np.mean([c["energy_drift"][n] for c in cells])) for n in names},
        "ensemble_wins": float(np.mean([c["ensemble_wins"] for c in cells])),
        "cells_where_ensemble_beats_best_single": int(sum(c["ensemble_beats_best_single"] for c in cells)),
        "n_cells": len(cells),
    }
    result["seconds"] = round(time.perf_counter() - t_start, 1)
    save()
    m = result["models"]
    t = f"z500 relative L2 over leads 6 h to {STEPS * 6} h, {m['n_cells']} cells ({len(inits)} cases)"
    print(f"\n{t}\n" + "-" * len(t))
    print("  ensemble             " + "   ".join(f"{n:>19s}" for n in names))
    print(f"  {m['ensemble']:.3f} (ensemble)      " + "   ".join(f"{m[n]:>19.3f}" for n in names))
    print("  day-5 RMSE (m2/s2)   " + "   ".join(f"{m['rmse_day5'][n]:>19.0f}" for n in names))
    print("  weight               " + "   ".join(f"{m['weights'][n]:>19.2f}" for n in names))
    print("  share of combination " + "   ".join(f"{m['share'][n]:>19.2f}" for n in names))
    print("  out of domain        " + "   ".join(f"{100 * m['out_of_domain'][n]:>18.0f}%" for n in names))
    print("  advection residual   " + "   ".join(f"{m['advection_residual'][n]:>19.3f}" for n in names))
    print("  energy drift /day    " + "   ".join(f"{m['energy_drift'][n]:>19.4f}" for n in names))
    print(f"\n  the ensemble beats the best single model on {100 * m['ensemble_wins']:.0f} % of the cases and in "
          f"{m['cells_where_ensemble_beats_best_single']}/{m['n_cells']} cells")
    print(f"  wrote {p} in {result['seconds']} s")
    return result


if __name__ == "__main__":
    main()
