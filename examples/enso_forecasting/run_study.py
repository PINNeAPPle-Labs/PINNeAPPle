"""El Nino (ENSO) prediction: data-driven, physical, physics-informed and hybrid models under one honest protocol.

Reproduces docs/guides/enso_forecasting.md. Every model lives in ``pinneapple_systems.time_series.enso`` and
``forecast_lab``; this script only wires the calls together:

1. the data set (OISST v2.1, WOD thermocline casts, ERA5 stress; cached with the SHA-256 of every source file);
2. the solver checks (Kelvin / Rossby speeds, convergence, Gill vs closed form, oscillators vs eigenvalues);
3. the zoo, one process per model, merged (purged walk-forward tuning, untouched test period 2012-2024, DM tests
   with Holm correction against persistence and the best baseline);
4. ENSO scores (ACC, target-month skill, categorical), case-study plumes (1997/98 refitted before the event,
   2015/16, 2023/24), spring-barrier hindcasts, the recharge-oscillator phase portrait, the shallow-water PINN;
5. tables, figures and the Blender clip of 1997/98.

    python examples/enso_forecasting/run_study.py --jobs 4           # full study (a few hours on 4 cores)
    python examples/enso_forecasting/run_study.py --smoke            # tiny sizes, a few minutes (data still needed)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")                # one thread per worker process

HERE = Path(__file__).resolve().parent
_DS = {}


def _dataset():
    from pinneapple_systems.time_series.enso import dataset as DS
    if "ds" not in _DS:
        manifests = {k: str(p) for k in ("oisst", "wod", "era5", "ncei_ohc")
                     if (p := HERE / "results" / "manifests" / f"{k}.json.gz").exists()}
        _DS["ds"] = DS.build_dataset(manifests=manifests, log=lambda *_: None)
        _DS["data"] = DS.to_series(_DS["ds"])
    return _DS["ds"], _DS["data"]


def _work(job):
    import torch
    torch.set_num_threads(1)
    from pinneapple_core.jsonable import to_jsonable
    from pinneapple_systems.time_series.enso import pipeline, study
    kind, name, kw, out = job
    ds, data = _dataset()
    prot = study.protocol(ds.months)
    if kind == "model":
        res = study.run_model(name, data, prot, quick=kw.get("quick", False), budget=kw.get("budget", 4),
                              log=lambda s: print(s, flush=True))
        Path(out).write_bytes(pickle.dumps(res))
    elif kind == "event":
        ev = {n: study.event_hindcast(n, data, "1997-01", study.CASE_ORIGINS["1997/98"], quick=kw.get("quick", False))
              for n in name}
        Path(out).write_bytes(pickle.dumps(ev))
    else:
        if kind == "checks":
            res = pipeline.solver_checks()
        elif kind == "pinn_sw":
            res = pipeline.shallow_water_pinn_study(ds, epochs=kw.get("epochs", 6000))
        elif kind == "spring":
            from pinneapple_systems.time_series.enso import forecasters as F
            from pinneapple_systems.time_series.forecast_lab import ARForecaster, Persistence
            makers = {"persistence": Persistence, "ar": ARForecaster,
                      "ro": lambda: F.RechargeOscillatorForecaster(stochastic=False),
                      "ro_seasonal_multiplicative": lambda: F.RechargeOscillatorForecaster(
                          seasonal=("R", "F1", "F2", "eps"), multiplicative=True, n_members=100),
                      "lim": lambda: F.LIMForecaster(n_eofs=10, extra_cols=("h_west", "h_east"))}
            res = pipeline.spring_barrier_hindcasts(data, prot, makers)
        elif kind == "phase":
            res = pipeline.phase_portrait(data, prot)
        Path(out).write_text(json.dumps(to_jsonable(res, digits=5)))
    return out


def _render(res_dir: Path, fig_dir: Path, smoke: bool):
    from pinneapple_systems.time_series.enso import data as D
    from pinneapple_systems.time_series.enso import pipeline, render
    ds, _ = _dataset()
    oisst = D.oisst_monthly("1981-09", ds.months[-1], log=lambda *_: None)
    casts = D.wod_d20_casts(range(1981, 2025), ("mrb", "xbt", "pfl"), log=lambda *_: None)
    frames = pipeline.slab_frames(ds, oisst, casts, "1997-01", "1997-03" if smoke else "1998-12")
    tmp = res_dir / "_frames"
    paths = render.render_frames(frames, str(tmp), interpolate=1 if smoke else 2, samples=8 if smoke else 24,
                                 size=(480, 270) if smoke else (960, 540))
    render.encode_video(paths, str(fig_dir / "enso_1997_98_slab.mp4"), fps=6)
    peak = [i for i, f in enumerate(frames) if f["month"] == "1997-12"]
    k = (peak[0] * (1 if smoke else 2)) if peak else len(paths) // 2
    shutil.copy(paths[min(k, len(paths) - 1)], fig_dir / "enso_1997_12_slab.png")
    shutil.copy(paths[0], fig_dir / "enso_1997_01_slab.png")
    shutil.rmtree(tmp)


def run(smoke: bool = False, jobs: int = 1, models=None, out_dir: Path = HERE, render_clip: bool = True):
    from pinneapple_core.jsonable import to_jsonable
    from pinneapple_systems.time_series.enso import pipeline, plots, study
    from pinneapple_systems.time_series.forecast_lab import plots as fl_plots
    from pinneapple_systems.time_series.forecast_lab.study import merge_zoo

    res_dir = out_dir / ("results_smoke" if smoke else "results")
    fig_dir = out_dir / ("figures_smoke" if smoke else "figures")
    parts_dir = res_dir / "_parts"
    for d in (res_dir, fig_dir, parts_dir, res_dir / "manifests"):
        d.mkdir(parents=True, exist_ok=True)
    ds, data = _dataset()
    # data summary and the manifests of every source file
    from pinneapple_systems.time_series.enso.data import manifests
    for k, src in manifests().items():
        (res_dir / "manifests" / f"{k}.json.gz").write_bytes(gzip.compress(src.read_bytes(), mtime=0))
    summary = {"months": [ds.months[0], ds.months[-1]], "base": ds.base, "checks": ds.checks, "sources": ds.sources,
               "indices": {k: v for k, v in ds.indices.items()}, "columns": data.meta["columns"],
               "grid": {"dlon": ds.grid.dlon, "dlat": ds.grid.dlat, "lat_max": ds.grid.lat_max,
                        "lon0": ds.grid.lon0, "lon1": ds.grid.lon1},
               "protocol": study.protocol(ds.months).describe(len(data))}
    (res_dir / "data.json").write_text(json.dumps(to_jsonable(summary, digits=4)))

    names = models or [s.name for s in study.zoo()]
    kw = {"quick": True, "budget": 1} if smoke else {}
    heavy = ["im_enkf", "cnn_pretrained", "pinn_extrapolation", "im_nudged", "cnn"]   # start these first
    names = sorted(names, key=lambda n: (n not in heavy, heavy.index(n) if n in heavy else 0))
    todo = [("checks", "", {}, str(res_dir / "solver_checks.json")),
            ("pinn_sw", "", {"epochs": 300 if smoke else 6000}, str(res_dir / "shallow_water_pinn.json"))]
    todo += [("model", n, kw, str(parts_dir / f"{n}.pkl")) for n in names]
    todo += [("spring", "", {}, str(res_dir / "spring_barrier.json")), ("phase", "", {}, str(res_dir / "phase.json"))]
    todo = [j for j in todo if not Path(j[3]).exists()]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            for out in ex.map(_work, todo):
                print("wrote", out, flush=True)
    else:
        for j in todo:
            print("wrote", _work(j), flush=True)

    parts = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(parts_dir.glob("*.pkl"))}
    merged = merge_zoo(list(parts.values()))
    reps = merged["family_representative"]
    ev_path = parts_dir / "events_1997.pkl"
    if not ev_path.exists():
        _work(("event", sorted({reps[f] for f in reps if f != "baseline"} | {"persistence"}), kw, str(ev_path)))
    events = {"1997/98": pickle.loads(ev_path.read_bytes())}
    moy = data.meta["month_of_year"]
    scores = pipeline.enso_scores(merged, moy)
    panels = pipeline.case_panels(parts, merged, data, events=events)
    vs_clim = pipeline.skill_against(merged, "climatology")
    out = {"zoo": merged, "enso_scores": scores, "case_panels": panels, "skill_vs_climatology": vs_clim,
           "event_1997_98": {n: {"train_last_label": v["train_last_label"], "data_end": v["data_end"],
                                 "origins": v["origins"],
                                 "median": v["forecast"].quantiles([0.5])[0].tolist()} for n, v in events["1997/98"].items()},
           "model_info": {n: p.get("_model_info") for n, p in parts.items()}}
    for k in ("solver_checks", "shallow_water_pinn", "spring_barrier", "phase"):
        f = res_dir / f"{k}.json"
        if f.exists():
            out[k] = json.loads(f.read_text())
    out = to_jsonable(out, digits=4, drop_private=True, unknown="repr")
    (res_dir / "study.json").write_text(json.dumps(out))
    tables = pipeline.markdown_tables(out)
    (res_dir / "tables.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in tables.items()) + "\n")

    # figures
    import numpy as np
    H = merged["protocol"]["horizons"]
    curves = {}
    for n, m in out["zoo"]["models"].items():
        if "test" not in m or not (n in reps.values() or n in ("persistence", "climatology", "arima")):
            continue
        sv = out["skill_vs_climatology"]["models"][n]
        curves[n] = {"family": m["family"], "horizons": H, "acc": out["enso_scores"][n]["acc"],
                     "rmse_skill_vs_climatology": sv["skill_rmse"],
                     "crps_skill_vs_climatology": m["test"].get("crps_skill_vs_climatology")}
        if "significant" in sv:
            curves[n]["rmse_skill_vs_climatology_significant"] = sv["significant"]
    plots.plot_skill_vs_lead(curves, str(fig_dir / "skill_vs_lead.png"),
                             metrics=("acc", "rmse_skill_vs_climatology", "crps_skill_vs_climatology"),
                             labels={"rmse_skill_vs_climatology": "RMSE skill vs climatology (test)"})
    fam_order = ["physical", "hybrid", "data", "data+physics"]
    plots.plot_plumes([p for p in panels], str(fig_dir / "plumes.png"), ncols=3)
    sb = {}
    for n, v in out.get("spring_barrier", {}).items():
        sb[f"{n} (CV hindcast)"] = {**v["by_target_month"], "period": f"{v['first'][:4]}-{v['last'][:4]}"}
    for fam in fam_order:
        n = reps.get(fam)
        if n and n in out["enso_scores"]:
            sb[f"{n} (test)"] = {**out["enso_scores"][n]["by_target_month"], "period": "2012-2022 origins"}
    if sb:
        plots.plot_spring_barrier(sb, str(fig_dir / "spring_barrier.png"), ncols=4)
    if "phase" in out:
        ph = out["phase"]
        cols = {"ro": "#1baf7a", "ro_seasonal_multiplicative": "#6250d6"}
        traj = {"observed (3-month running mean)": {"h": ph["observed"]["h_smooth"], "T": ph["observed"]["T_smooth"],
                                                    "color": "#0b0b0b", "lw": 0.7, "alpha": 0.6, "left": True}}
        for n, d in ph["models"].items():
            for k, (on, pth) in enumerate(d["onset_paths"].items()):
                traj[f"{n}: 48-month mean path" if k == 0 else f"{n} {on}"] = {
                    **pth, "color": cols[n], "lw": 1.8, "left": True, "label": k == 0}
        for n, d in ph["models"].items():
            traj[f"{n}: one stochastic run"] = {**d["stochastic"], "lw": 0.7, "alpha": 0.8, "left": False,
                                                "color": cols[n]}
        traj["observed"] = {"h": ph["observed"]["h"], "T": ph["observed"]["T"], "color": "#0b0b0b",
                            "lw": 0.6, "alpha": 0.6, "left": False}
        plots.plot_phase_portrait(traj, str(fig_dir / "phase_portrait.png"))
    if "shallow_water_pinn" in out:
        hv = out["shallow_water_pinn"]["hovmoller"]
        lon = np.asarray(hv["lon"])
        hov = [{"title": "D20 anomaly, observed (TAO/XBT/Argo, 2S-2N)", "lon": lon, "t": hv["t"],
                "z": np.asarray(hv["h_obs"], float), "vmax": 50, "units": "m"},
               {"title": "h, 2-D shallow-water model forced by ERA5 (scaled)", "lon": np.asarray(hv["lon_model"]),
                "t": hv["t"], "z": np.asarray(hv["h_model2d_full"], float), "vmax": 50, "units": "m"},
               {"title": "D20 anomaly, equatorial-wave PINN", "lon": lon, "t": hv["t"],
                "z": np.asarray(hv["h_pinn"], float), "vmax": 50, "units": "m"},
               {"title": "SST anomaly, observed (OISST, 2S-2N)", "lon": lon, "t": hv["t"],
                "z": np.asarray(hv["T_obs"], float), "vmax": 4, "units": "K"},
               {"title": "SST anomaly, PINN (learned closure)", "lon": lon, "t": hv["t"],
                "z": np.asarray(hv["T_pinn"], float), "vmax": 4, "units": "K"}]
        plots.plot_hovmoller(hov, str(fig_dir / "hovmoller_1996_1999.png"), ncols=5)
    if "solver_checks" in out:
        sc = out["solver_checks"]
        plots.plot_validation(sc["kelvin"], sc["gill"], sc["kelvin_convergence"], sc["recharge_oscillator"],
                              str(fig_dir / "solver_checks.png"))
    pit = []
    for fam, n in reps.items():
        pooled = out["zoo"]["models"][n]["test"]["pit"]["pooled"]
        if pooled and fam != "baseline":
            pit.append({"title": n, "family": fam, **pooled})
    if pit:
        fl_plots.plot_pit(pit, str(fig_dir / "pit.png"), ncols=4)
    if render_clip:
        try:
            _render(res_dir, fig_dir, smoke)
        except ImportError as e:                         # bpy not installed: the clip is skipped, not faked
            print("render skipped:", e)
    return {"models": len(out["zoo"]["models"]), "dir": str(res_dir)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1)))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    print(run(smoke=a.smoke, jobs=a.jobs, models=a.models, render_clip=not a.no_render))
