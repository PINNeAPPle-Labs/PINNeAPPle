"""Visual pack of the near-random forecasting study: four Blender Cycles renders and two 2-D figures.

Needs the case results of ``run_study.py`` (``results/cases``) and, for the renders, the ``bpy`` module
(``pip install bpy``) plus ffmpeg for the clip. Every number drawn is written to ``results/visuals.json``.

    python examples/near_random_forecasting/render_visuals.py            # all (about 10 min on 4 cores)
    python examples/near_random_forecasting/render_visuals.py --no-3d    # 2-D figures only
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LANDSCAPE = ["lorenz63_noise0.01", "lorenz63_noise0.3", "mackey_glass_noise0.01", "mackey_glass_noise0.3",
             "lorenz96_noise0.1", "double_well", "stochastic_oscillator", "white_noise", "usd_eur_log_returns",
             "sunspots_yearly", "youtube_us_log_views", "youtube_us_entertainment_slots"]
CALENDAR = ["youtube_us_log_views", "youtube_us_entertainment_slots", "usd_eur_log_returns"]


def run(three_d: bool = True, out_dir: Path = HERE, quick: bool = False):
    import torch

    from pinneapple_core.jsonable import to_jsonable
    from pinneapple_systems.time_series import synthetic as syn
    from pinneapple_systems.time_series.forecast_lab import GBMForecaster, pipeline, plots, renders
    from pinneapple_systems.time_series.forecast_lab.presets import case
    torch.set_num_threads(4)

    res = {n: json.loads((out_dir / "results" / "cases" / f"{n}.json").read_text()) for n in LANDSCAPE + CALENDAR}
    fig = out_dir / "figures" / "visuals"
    fig.mkdir(parents=True, exist_ok=True)
    meta: dict = {}

    # 5. memorise vs learn (refits the overfit and the regularised MLP of the overfitting study)
    mem = [pipeline.memorise_vs_learn(n, window=150) for n in ("double_well", "lorenz63_noise0.1")]
    plots.plot_memorise_vs_learn(mem, str(fig / "memorise_vs_learn.png"))
    meta["memorise_vs_learn"] = [{"case": m["case"], "horizon": m["horizon"],
                                  "variants": {k: {"rmse": v["rmse"], "n_params": v["n_params"]}
                                               for k, v in m["variants"].items()}} for m in mem]

    # 6. surprise calendar
    recs = [{**res[n]["surprise"], "title": res[n]["title"]} for n in CALENDAR if res[n].get("surprise")]
    plots.plot_surprise_calendar(recs, str(fig / "surprise_calendar.png"))
    meta["surprise_calendar"] = [{"case": n, "model": r["model"], "baseline": r["baseline"], "horizon": r["horizon"],
                                  "n_days": len(r["dates"]), "first": r["dates"][0], "last": r["dates"][-1],
                                  "fraction_model_closer": float(np.mean(np.asarray(r["abs_error_model"])
                                                                         < np.asarray(r["abs_error_baseline"])))}
                                 for n, r in zip([n for n in CALENDAR if res[n].get("surprise")], recs, strict=True)]

    if three_d:
        from pinneapple_blender.cycles import available, frames_to_video
        if not available():
            raise RuntimeError("the 3-D renders need the bpy module (pip install bpy)")
        spp = 16 if quick else 96

        # 1. predictability landscape
        short = {"lorenz63_noise0.01": "Lorenz-63, noise 0.01", "lorenz63_noise0.3": "Lorenz-63, noise 0.3",
                 "mackey_glass_noise0.01": "Mackey-Glass, noise 0.01", "mackey_glass_noise0.3": "Mackey-Glass, noise 0.3",
                 "lorenz96_noise0.1": "Lorenz-96, noise 0.1", "double_well": "double well",
                 "stochastic_oscillator": "noise-driven oscillator", "white_noise": "white noise",
                 "usd_eur_log_returns": "USD/EUR returns", "sunspots_yearly": "sunspots",
                 "youtube_us_log_views": "YouTube: log views", "youtube_us_entertainment_slots": "YouTube: Entertainment"}
        tiles = renders.landscape_tiles([res[n] for n in LANDSCAPE], titles=short)
        renders.render_landscape(tiles, str(fig / "landscape.png"), samples=spp)
        meta["landscape"] = tiles

        # 2. forecast cones: GBM refitted at every horizon 1..48, Lorenz-63 (noise 0.1) next to white noise
        lanes = []
        for n, title in (("lorenz63_noise0.1", "Lorenz-63 (noise 0.1)"), ("white_noise", "white noise")):
            c = case(n)
            d = renders.dense_forecast(c.data, c.protocol, lambda: GBMForecaster(n_lags=8, max_iter=200,
                                                                                 max_leaf_nodes=15), 48)
            lanes.append({"title": title, "dense": d, "origin_pos": len(d["origins"]) // 2,
                          "std": float(np.std(c.data.y))})
            meta.setdefault("cones", []).append({"case": n, "model": d["model"], "max_horizon": 48,
                                                 "n_test_origins": len(d["origins"]),
                                                 "coverage50_mean": float(np.mean(d["coverage50"])),
                                                 "coverage90_mean": float(np.mean(d["coverage90"])),
                                                 "origin": d["origins"][len(d["origins"]) // 2]})
        renders.render_cones(lanes, str(fig / "forecast_cone.png"), samples=spp, resolution=(1280, 540))
        with tempfile.TemporaryDirectory() as tmp:
            small = renders.render_cones(lanes, str(Path(tmp) / "still_small.png"), samples=max(8, spp // 3),
                                         frames_dir=str(Path(tmp) / "clip"), n_frames=12 if quick else 60,
                                         resolution=(640, 270))
            frames_to_video(small["frames"], str(fig / "forecast_cone.mp4"), fps=20)

        # 3. attractor vs surrogates
        s = syn.lorenz63(n=6000, noise=0.01, lyapunov_steps=10)
        views = renders.embedding_views(s.obs[:, 0], n_points=3000)
        renders.render_attractors(views, str(fig / "attractor_vs_surrogates.png"), samples=spp)
        meta["attractor"] = {"delay": views["delay"], "npe": views["npe"]}

        # 4. ensemble divergence: 50 members from one state perturbed by 1 % of the attractor std
        panels = []
        lam = res["lorenz63_noise0.01"]["truth"]["lyapunov"]
        step63 = syn.ode_stepper(syn.lorenz63_rhs, 0.02, sub=2)
        x0 = s.state[3000]
        fib = renders.ensemble_fibres(step63, x0, n_steps=int(5 / lam / 0.02), scale=s.state.std(axis=0),
                                      components=(0, 2))
        t = np.arange(fib["trajectories"].shape[0]) * 0.02
        panels.append({"title": "Lorenz-63 (x, z), 5 Lyapunov times", "fibres": fib, "times": t,
                       "lt_time": 1 / lam, "family": "physical"})
        meta.setdefault("ensembles", []).append({"case": "lorenz63", "spread_at_1LT": float(
            np.interp(1 / lam, t, fib["spread"])), "spread_at_end": float(fib["spread"][-1]),
            "spread_at_start": float(fib["spread"][0])})

        mg = syn.mackey_glass(n=600, noise=0.0, lyapunov_time=2000.0)
        lam_mg = res["mackey_glass_noise0.01"]["truth"]["lyapunov"]
        m = int(round(17.0 / 0.1))
        hist = np.interp(np.arange(-m, 1) * 0.1, np.arange(-6, 1) * 3.0, mg.state[300 - 6:301, 0])
        fib = renders.ensemble_fibres(syn.mackey_glass_stepper(0.1, 10), hist, n_steps=int(5 / lam_mg), scale=np.full(m + 1, mg.state.std()),
                                      components=(m, 0))
        t = np.arange(fib["trajectories"].shape[0]) * 1.0
        panels.append({"title": "Mackey-Glass (x(t), x(t-17)), 5 Lyapunov times", "fibres": fib, "times": t,
                       "lt_time": 1 / lam_mg, "family": "data+physics"})
        meta["ensembles"].append({"case": "mackey_glass", "spread_at_1LT": float(np.interp(1 / lam_mg, t,
                                                                                           fib["spread"])),
                                  "spread_at_end": float(fib["spread"][-1]),
                                  "spread_at_start": float(fib["spread"][0])})

        osc = syn.stochastic_oscillator(n=4000, noise=0.0)
        step_osc = syn.linear_sde_stepper(*syn.oscillator_transition(1.0, 0.05, 0.3, 0.25), seed=1)  # shared forcing
        fib = renders.ensemble_fibres(step_osc, osc.state[2000], n_steps=150, scale=osc.state.std(axis=0))
        t = np.arange(fib["trajectories"].shape[0]) * 0.25
        panels.append({"title": "noise-driven oscillator (x, v), same forcing, 6 periods (no Lyapunov time: not chaotic)", "fibres": fib,
                       "times": t, "lt_time": None, "family": "hybrid"})
        meta["ensembles"].append({"case": "stochastic_oscillator", "spread_at_end": float(fib["spread"][-1]),
                                  "spread_at_start": float(fib["spread"][0])})
        renders.render_ensembles(panels, str(fig / "ensemble_divergence.png"), samples=spp)

    path = out_dir / "results" / "visuals.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(to_jsonable(meta, digits=4))
    path.write_text(json.dumps(old, indent=1))
    return {"figures": sorted(p.name for p in fig.iterdir()), "json": str(path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-3d", action="store_true")
    ap.add_argument("--quick", action="store_true", help="few samples and frames (preview)")
    a = ap.parse_args()
    print(run(three_d=not a.no_3d, quick=a.quick))
