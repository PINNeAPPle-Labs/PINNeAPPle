"""Figures, report and experiment manifest of the CO2 surrogate study, from ``results/*.json`` of 01_run_study.py.

Every number written here is read from those results (measured); the verdict per region follows from the paired
intervals: the potential is called more (less) accurate only when the whole 95 % interval of the paired log10 error
ratio is below (above) zero.

    python examples/potential_co2/02_report.py [--results results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pinneapple_lab import experiment
from pinneapple_physics.potential import co2_study as cs
from pinneapple_physics.potential import plots

HERE = Path(__file__).resolve().parent
CONFIG_LABELS = {"all_properties": "all six properties, 3000 steps",
                 "without_cp": "no c_p in training, 3000 steps",
                 "all_properties_3x": "all six properties, 9000 steps"}
REGIONS = ("interp", "hotter", "denser")


def _iv(d, digits=3):
    return f"{d['mean']:.{digits}g} [{d['lo']:.{digits}g}, {d['hi']:.{digits}g}]"


def verdict(paired: dict) -> str:
    if paired["hi"] < 0:
        return "potential more accurate"
    if paired["lo"] > 0:
        return "direct more accurate"
    return "no significant difference"


def report_md(results: dict[str, dict]) -> str:
    L = ["# CO2 near the critical point: potential-based against direct surrogates", "",
         "Data: CoolProp HEOS (Span & Wagner 1996) for CO2. Training region T 305-335 K, rho 100-800 kg/m3; "
         "extrapolation: T 335-365 K (hotter) and rho 800-950 kg/m3 (denser). Each training sample carries one "
         "property. Same data, loss, optimiser, steps, width (32) and depth (3) for every model; 5 seeds; intervals "
         "are 95 % Student-t over seeds. Error = RMS error / standard deviation of the property on the held-out grid "
         "of the training region (21 x 21 points per region).", ""]
    for name, res in results.items():
        s = res["summary"]
        cfg = res["config"]
        L += [f"## {CONFIG_LABELS.get(name, name)}", "",
              f"{cfg['n_per_prop']} samples per training property ({', '.join(cfg['props'])}), {cfg['epochs']} Adam "
              "steps.", "",
              "| model | parameters | training s / seed | error, training region | error, hotter | error, denser "
              "| stable share, hotter |",
              "|---|---|---|---|---|---|---|"]
        for m, v in s["models"].items():
            L.append(f"| {plots.MODEL_LABELS.get(m, m)} | {v['parameters']} | {v['seconds']['mean']:.0f} | "
                     + " | ".join(_iv(v[g]["nrmse_mean"]) for g in REGIONS)
                     + f" | {_iv(v['hotter']['stable_fraction'])} |")
        L += ["", "Per property, training region (mean [95 % interval]):", "",
              "| model | " + " | ".join(cs.PROPERTIES) + " |", "|---" * (len(cs.PROPERTIES) + 1) + "|"]
        for m, v in s["models"].items():
            L.append(f"| {plots.MODEL_LABELS.get(m, m)} | "
                     + " | ".join(_iv(v["interp"]["nrmse"][p], 2) for p in cs.PROPERTIES) + " |")
        L += ["", "Paired against the direct surrogate (log10 error ratio, mean of the six properties; < 0: the "
              "potential is more accurate):", "", "| model | " + " | ".join(REGIONS) + " |",
              "|---" * (len(REGIONS) + 1) + "|"]
        for m, pv in s["paired"].items():
            L.append(f"| {plots.MODEL_LABELS.get(m, m)} | "
                     + " | ".join(f"{_iv(pv[g]['mean'])}: {verdict(pv[g]['mean'])}" for g in REGIONS) + " |")
        L += ["", "Thermodynamic identities, held-out grid of the training region (relative violation, 95th "
              "percentile, mean over seeds; finite differences of the predictions, step 1e-4):", "",
              "| identity | " + " | ".join(plots.MODEL_LABELS.get(m, m) for m in s["models"]) + " |",
              "|---" * (len(s["models"]) + 1) + "|"]
        names = list(next(iter(s["models"].values()))["interp"]["identities"])
        for n in names:
            L.append(f"| {n} | " + " | ".join(f"{v['interp']['identities'][n]['mean']:.2g}"
                                                for v in s["models"].values()) + " |")
        L.append("")
    L += ["## Not measured", "",
          "- Two-phase states and the subcritical region near the saturation line (the data are supercritical).",
          "- Other fluids, noisy data, other widths or depths, other optimisers.",
          "- Wall-clock at equal time: the budget is equal in steps; training seconds are reported (single thread, "
          "three studies running at once on four cores).", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    rdir = HERE / a.results
    results = {n: json.loads((rdir / f"{n}.json").read_text()) for n in CONFIG_LABELS if (rdir / f"{n}.json").exists()}
    if not results:
        raise SystemExit(f"no results in {rdir}: run 01_run_study.py first")
    fig_dir = HERE / "figures"
    with experiment("examples/potential_co2", title="Potential-based against direct surrogates for CO2 near Tc",
                    description="A network for the Helmholtz energy (every property a derivative of it; MLP and "
                                "input-convex variants) against one network output per property, on the same mixed "
                                "CoolProp data of supercritical CO2 near the critical point, at equal data and "
                                "steps: accuracy inside the training region and in two extrapolation regions, "
                                "thermodynamic-identity violations, stability; 5 seeds with 95 % intervals.",
                    domain="thermodynamics", methods=["neural surrogate", "input-convex network",
                                                       "automatic differentiation"],
                    modules=["pinneapple_physics.potential.surrogate", "pinneapple_physics.potential.co2_study",
                             "pinneapple_physics.potential.relations", "pinneapple_physics.potential.thermo",
                             "pinneapple_physics.potential.plots"],
                    params={n: {k: r["config"][k] for k in ("n_per_prop", "epochs", "props", "seeds")}
                            for n, r in results.items()},
                    references=["Span & Wagner, J. Phys. Chem. Ref. Data 25 (1996) 1509-1596",
                                "Amos, Xu & Kolter, Input Convex Neural Networks, ICML 2017"],
                    out_dir=str(HERE)) as ex:
        for name, res in results.items():
            s = res["summary"]
            plots.errors_figure(s, CONFIG_LABELS.get(name, name), path=fig_dir / f"errors_{name}.png")
            plots.identities_figure(s, path=fig_dir / f"identities_{name}.png")
            run0 = res["runs"][0]
            ref = {f"{T:g}": cs.co2_properties(np.full(cs.ISO_RHO.size, T), cs.ISO_RHO) for T in cs.ISO_T}
            plots.isotherms_figure(run0, ref, cs.ISO_RHO, title=f"{CONFIG_LABELS.get(name, name)}, seed 0",
                                   path=fig_dir / f"isotherms_{name}.png")
            for m, v in s["models"].items():
                for g in REGIONS:
                    d = v[g]["nrmse_mean"]
                    ex.log_metric(f"{name}_{m}_{g}_error", round(d["mean"], 5), "",
                                  f"RMS error / std of the property, mean of p, s, u, c_v, c_p, w; mean over "
                                  f"{d['n']} seeds, 95 % interval [{d['lo']:.4g}, {d['hi']:.4g}]")
                ex.log_metric(f"{name}_{m}_maxwell_p95", float(f"{v['interp']['identities']['Maxwell relation (Helmholtz)']['mean']:.4g}"),
                              "", "relative violation of -rho^2 (ds/drho)_T = (dp/dT)_rho, 95th percentile on the "
                                  "held-out grid, mean over seeds")
            for m, pv in s["paired"].items():
                for g in REGIONS:
                    d = pv[g]["mean"]
                    ex.log_metric(f"{name}_{m}_vs_direct_{g}_log10_ratio", round(d["mean"], 5), "",
                                  f"paired by seed, 95 % interval [{d['lo']:.4g}, {d['hi']:.4g}]: "
                                  f"{verdict(d)}")
        main_s = results.get("all_properties", next(iter(results.values())))["summary"]["models"]
        pot_max = main_s["potential"]["interp"]["identities"]["Maxwell relation (Helmholtz)"]["mean"]
        ex.check("potential_maxwell_relation_by_construction", value=pot_max, max=1e-4, kind="physics",
                 label="Maxwell relation of the potential surrogate (finite differences)",
                 detail="the potential surrogate's p and s come from one function: the violation is the "
                        "finite-difference error")
        d_max = main_s["direct"]["interp"]["identities"]["Maxwell relation (Helmholtz)"]["mean"]
        ex.check("direct_violates_maxwell_relation", value=d_max, min=1e-3, kind="sanity",
                 label="the direct surrogate is not consistent by construction",
                 detail="relative violation of the Maxwell relation, 95th percentile")
        main_p = results.get("all_properties", next(iter(results.values())))["summary"]["paired"]["potential"]
        ex.check("potential_more_accurate_in_training_region", main_p["interp"]["mean"]["hi"] < 0, kind="comparison",
                 label="potential surrogate beats the direct one inside the training region (all properties)",
                 detail=f"paired log10 error ratio {_iv(main_p['interp']['mean'])}")
        summaries = {CONFIG_LABELS.get(n, n): r["summary"] for n, r in results.items()}
        plots.paired_figure(summaries, "potential", path=fig_dir / "paired_potential.png")
        plots.paired_figure(summaries, "potential_icnn", path=fig_dir / "paired_potential_icnn.png")
        md = report_md(results)
        (rdir / "report.md").write_text(md)
        ex.add_file(f"{a.results}/report.md")
        for n in results:
            ex.add_file(f"{a.results}/{n}.json")
    print(md)


if __name__ == "__main__":
    main()
