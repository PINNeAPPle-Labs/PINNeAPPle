"""Forecasting series that are close to random, and not fooling yourself while doing it.

Reproduces docs/guides/near_random_forecasting.md: for every case in
``pinneapple_systems.time_series.forecast_lab.presets.CASES`` it measures predictability before modelling, runs the
model zoo (baselines, data-driven, data + physics, physical, hybrid) under the purged walk-forward protocol with an
untouched test period, trains surrogate twins, and runs the anti-overfitting study on three cases. Results go to
``results/`` (one JSON per case, ``overfitting.json``, ``tables.md``) and figures to ``figures/``.

    python examples/near_random_forecasting/run_study.py --jobs 4          # full study (about 1-2 h on 4 cores)
    python examples/near_random_forecasting/run_study.py --smoke           # tiny sizes, a few minutes
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERFIT_CASES = ["lorenz63_noise0.1", "double_well", "usd_eur_log_returns"]


def _work(job):
    import torch
    torch.set_num_threads(1)
    from pinneapple_systems.time_series.forecast_lab import pipeline
    kind, name, kw, out = job
    if kind == "case":
        res = pipeline.run_case(name, **kw)
    elif kind == "lyapunov":
        res = pipeline.lyapunov_study(**kw)
    else:
        res = pipeline.run_overfitting(name, **kw)
    Path(out).write_text(json.dumps(res))
    return out


def run(smoke: bool = False, jobs: int = 1, cases=None, out_dir: Path = HERE, force: bool = False):
    from pinneapple_systems.time_series.forecast_lab import pipeline, plots
    from pinneapple_systems.time_series.forecast_lab.presets import CASES

    res_dir, fig_dir = out_dir / ("results_smoke" if smoke else "results"), out_dir / ("figures_smoke" if smoke else "figures")
    (res_dir / "cases").mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = cases or (["lorenz63_noise0.1", "double_well", "white_noise", "sunspots_yearly"] if smoke else CASES)
    kw = {"n": 1500, "budget": 1, "n_surrogates": 19} if smoke else {}
    ov_kw = {"n": 1500} if smoke else {}
    todo = [("case", n, kw, str(res_dir / "cases" / f"{n}.json")) for n in names]
    todo = [("overfit", n, ov_kw, str(res_dir / f"overfitting_{n}.json"))
             for n in (OVERFIT_CASES if not smoke else ["lorenz63_noise0.1"]) if cases is None or n in cases] + todo
    lyap_kw = {"total_times": (2000.0,), "seeds": (0,)} if smoke else {}
    todo = [("lyapunov", "mackey_glass", lyap_kw, str(res_dir / "lyapunov_mackey_glass.json"))] + todo
    todo = [j for j in todo if not Path(j[3]).exists() or (force and j[0] == "case")]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:          # longest jobs (the overfitting studies) were queued first
            for out in ex.map(_work, todo):
                print("wrote", out, flush=True)
    else:
        for j in todo:
            print("wrote", _work(j))

    results = [json.loads((res_dir / "cases" / f"{n}.json").read_text()) for n in CASES
               if (res_dir / "cases" / f"{n}.json").exists()]
    overfit = [json.loads(p.read_text()) for p in sorted(res_dir.glob("overfitting_*.json"))]
    tables = pipeline.markdown_tables(results, overfit)
    (res_dir / "tables.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in tables.items()) + "\n")

    lyap = {r["zoo"]["series"]: r["truth"]["lyapunov"] for r in results if r["truth"].get("lyapunov")}
    titles = {r["zoo"]["series"]: r["title"] for r in results}
    plots.plot_skill_vs_horizon([r["zoo"] for r in results], str(fig_dir / "skill_vs_horizon.png"),
                                lyapunov=lyap, titles=titles, reference="best_baseline")
    plots.plot_skill_vs_horizon([r["zoo"] for r in results], str(fig_dir / "skill_vs_persistence.png"),
                                lyapunov=lyap, titles=titles, reference="persistence")
    plots.plot_surrogate_tests({r["title"]: r["predictability"] for r in results}, str(fig_dir / "surrogate_tests.png"))
    pit = []
    for r in results:
        z = r["zoo"]
        for fam, name in z.get("family_representative", {}).items():
            pooled = z["models"][name]["test"]["pit"]["pooled"]
            if pooled and fam != "baseline":
                pit.append({"title": f"{r['title']}: {name}", "family": fam, **pooled})
    plots.plot_pit(pit, str(fig_dir / "pit.png"), ncols=6)
    fans = [r["fan_panels"][i] for r in results if r["case"] in (
        "lorenz63_noise0.1", "mackey_glass_noise0.1", "double_well", "usd_eur_log_returns", "sunspots_yearly")
        for i in range(0, len(r["fan_panels"]), 2)]                          # one origin per family
    if fans:
        plots.plot_fan_charts(fans, str(fig_dir / "fan_charts.png"), ncols=3)
    if overfit:
        plots.plot_overfitting(overfit, str(fig_dir / "overfitting.png"))
    return {"results": len(results), "overfit": len(overfit), "dir": str(res_dir)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1)))
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--force", action="store_true", help="rerun the named cases even if their results exist")
    a = ap.parse_args()
    print(run(smoke=a.smoke, jobs=a.jobs, cases=a.cases, force=a.force))
