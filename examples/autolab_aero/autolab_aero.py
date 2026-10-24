"""The AI Aerodynamicist (PINNeAPPle Autonomous Physics Lab, stage 1: the scientific cycle without an agent).

Mission: "Find a lower-drag aerodynamic design at 70 m/s. Preserve the required downforce, stay within the geometry
constraints, and validate the final design using CFD." on a parametric open-wheel racing car (not any team's car).

    python examples/autolab_aero/autolab_aero.py                 # qualification + campaign (hours on 4 cores)
    python examples/autolab_aero/autolab_aero.py --budget 20 --no-qualify

1. Qualify the solver on the baseline: the same design on the "medium", "campaign" and "fine" meshes.
2. Run the lab (pinneapple_autolab.AutonomousLab): baseline -> space-filling CFD -> one CFD at a time chosen by
   what the surrogates still do not know -> final verification CFD. Every number in the outputs comes from the
   ledger of OpenFOAM runs made by this script on this machine.
3. Write figures, the report, the ledger and the snapshot into ``_out/`` (small files only; the OpenFOAM cases are
   deleted after their forces and surface fields are extracted; the work folder keeps the flow scenes).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import pinneapple_lab
from pinneapple_autolab import AutonomousLab
from pinneapple_autolab import report as R
from pinneapple_autolab.domains.racecar import (
    BASELINE,
    RaceCarAero,
    cycles_renders,
    flagship_mission,
    render_run,
)
from pinneapple_autolab.lab import showcase
from pinneapple_autolab.qualification import solver_qualification

HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = ["drag_N", "downforce_N", "balance_front"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=22)
    ap.add_argument("--reserve", type=int, default=3)
    ap.add_argument("--doe", type=int, default=10)
    ap.add_argument("--mesh", default="campaign")
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--work", default=os.path.join(HERE, "_work"))
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    ap.add_argument("--no-qualify", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-only", action="store_true", help="rebuild figures and report from _out/*.json")
    ap.add_argument("--dashboard", action="store_true",
                    help="build the four-area dashboard (replay of the session) in _work/dashboard, with screenshots "
                         "and a replay video in _out/figures (needs playwright and ffmpeg)")
    ap.add_argument("--confirm-fine", action="store_true",
                    help="also simulate the recommended design on the fine mesh (outside the budget)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    figs = os.path.join(a.out, "figures")
    with pinneapple_lab.experiment(
            "examples/autolab_aero", title="The AI Aerodynamicist (Autonomous Physics Lab)",
            description="Lower drag at 70 m/s keeping the downforce and the aero balance: OpenFOAM 3-D on a parametric "
                        "racing car, surrogates compared on held-out CFD, NSGA-II on the surrogate, every candidate "
                        "triaged (favourable / uncertain / out of domain), the next CFD chosen by what is unknown, "
                        "the recommendation verified by CFD.",
            domain="external aerodynamics",
            methods=["OpenFOAM simpleFoam k-omega SST", "snappyHexMesh", "Gaussian process", "MLP ensemble",
                     "quadratic response surface", "POD + GP surface field", "NSGA-II", "active learning"],
            modules=["pinneapple_autolab", "pinneapple_simulation.numerical_solvers.external_flow",
                     "pinneapple_design.geometry.vehicles3d", "pinneapple_design.design_optimizer",
                     "pinneapple_analysis.trust"],
            params={k: v for k, v in vars(a).items() if k not in ("work", "out")}, out_dir=HERE, register=False,
            links={"script": "examples/autolab_aero/autolab_aero.py", "report": "examples/autolab_aero/_out/report.md"}
    ) as ex:
        sim = RaceCarAero(resolution=a.mesh, iterations=a.iterations)
        qual = None
        qpath = os.path.join(a.out, "qualification.json")
        if not (a.no_qualify or a.report_only):
            t0 = time.time()
            qual = solver_qualification(sim, BASELINE, {"medium": {"resolution": "medium"}, a.mesh: {},
                                                        "fine": {"resolution": "fine"}},
                                        os.path.join(a.work, "qualification"), reference="fine", campaign=a.mesh,
                                        outputs=OUTS)
            json.dump(qual, open(qpath, "w"), indent=1, default=float)
            print(f"qualification: {time.time() - t0:.0f} s; campaign vs fine:",
                  {k: f"{100 * v:+.1f} %" for k, v in qual["campaign_vs_reference"].items()})
        elif os.path.exists(qpath):
            qual = json.load(open(qpath))

        mission = flagship_mission(a.budget, a.reserve)
        if a.report_only:
            summary = json.load(open(os.path.join(a.out, "summary.json")))
            snap = json.load(open(os.path.join(a.out, "snapshot.json")))
        else:
            lab = AutonomousLab(mission, sim, a.work, critical_region="diffuser", field_tol=0.05, seed=a.seed,
                                listeners=[lambda e: print(f"[{e['t']:8.1f} s] {e['stage']:9s} {e['message'][:160]}")])
            summary = lab.run(n_doe=a.doe)
            snap = lab.snapshot()
            json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
            json.dump(snap, open(os.path.join(a.out, "snapshot.json"), "w"), default=float)
            lab.ledger.save(os.path.join(a.out, "ledger.json"))
        fine = None
        fpath = os.path.join(a.out, "fine_check.json")
        if a.confirm_fine and summary.get("best") and qual and qual["variants"].get("fine", {}).get("ok"):
            fine = solver_qualification(sim, summary["best"]["params"], {"fine": {"resolution": "fine"}},
                                        os.path.join(a.work, "fine_check"), reference="fine", campaign="fine",
                                        outputs=OUTS)
            json.dump(fine, open(fpath, "w"), indent=1, default=float)
        elif os.path.exists(fpath):
            fine = json.load(open(fpath))

        # ------------------------------------------------------------------ figures
        F = {}
        F["pareto"] = R.fig_pareto(snap, os.path.join(figs, "pareto.png"))
        F["models"] = R.fig_models(snap, os.path.join(figs, "models.png"), OUTS,
                                   {"drag_N": "drag [N]", "downforce_N": "downforce [N]",
                                    "balance_front": "front share"})
        F["learning"] = R.fig_learning(snap, os.path.join(figs, "learning.png"), OUTS, "drag_N", mission.acceptance)
        sc = summary["showcase"] = showcase(summary["decisions"], summary["baseline"]["outputs"])
        json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
        if sc:
            F["abc"] = R.fig_trust_cards([sc[k] for k in sorted(sc)], os.path.join(figs, "trust_abc.png"), OUTS)
        held = [d for d in summary["decisions"] if d.get("run") and os.path.exists(
            os.path.join(a.work, "fields", f"{d['run']}_pred.npz"))]
        if held:
            d = held[-1]
            ref = np.load(os.path.join(a.work, "fields", f"{d['run']}.npz"))["cp"]
            pr = np.load(os.path.join(a.work, "fields", f"{d['run']}_pred.npz"))["cp"]
            F["cp"] = R.fig_field_triptych(sim.field_points(d["params"]), ref, pr,
                                           os.path.join(figs, "cp_numerical_prediction_error.png"),
                                           title=f"Surface Cp, {d['run']} (the surrogate had not seen this CFD)")
        for tag, run in (("baseline", summary["baseline"]), ("best", summary["best"])):
            npz = os.path.join(a.work, "cases", f"{run['id']}.flow.npz") if run else ""
            if os.path.exists(npz):
                for k, v in render_run(run["params"], npz, os.path.join(figs, tag),
                                       f"{tag}: {run['id']} (CFD)").items():
                    F[f"{tag}_{k}"] = ex.add_file(v)
                if a.dashboard:                                  # Blender Cycles, when bpy is installed
                    for k, v in cycles_renders(run["params"], npz, os.path.join(figs, tag),
                                               f"{'original' if tag == 'baseline' else 'recommended'} {run['id']} (CFD)").items():
                        F[f"{tag}_{k}"] = ex.add_file(v)
        extra = ""
        if fine and summary.get("improvement"):
            def fine_noise(q):            # iteration std of the fine run, from its ledger (older files lack it inline)
                v = q["variants"]["fine"]
                if v.get("iteration_std"):
                    return v["iteration_std"]
                led_q = json.load(open(q["ledger"])) if os.path.exists(q["ledger"]) else {"runs": []}
                return next((r["meta"].get("iteration_std", {}) for r in led_q["runs"] if r["id"] == "qual-fine"), {})
            bf, wf = qual["variants"]["fine"]["outputs"], fine["variants"]["fine"]["outputs"]
            nb, nw = fine_noise(qual), fine_noise(fine)
            b0, w0 = summary["baseline"]["outputs"], summary["best"]["outputs"]
            extra = ("## The recommendation on the fine mesh (outside the budget)\n\nBoth designs on the fine mesh "
                     f"({fine['variants']['fine']['cells']:,} cells for the recommendation). Noise = standard deviation of "
                     "the output over the averaged iterations of each fine run.\n\n| output | baseline (fine) | "
                     "recommended (fine) | change (fine) | change / noise | change (campaign mesh) |\n"
                     "|---|---|---|---|---|---|\n" + "\n".join(
                         f"| {k} | {bf[k]:.4g} ± {nb.get(k, float('nan')):.2g} | {wf[k]:.4g} ± {nw.get(k, float('nan')):.2g} | "
                         f"{100 * (wf[k] - bf[k]) / abs(bf[k]):+.2f} % | "
                         f"{(wf[k] - bf[k]) / max((nb.get(k, 0) ** 2 + nw.get(k, 0) ** 2) ** 0.5, 1e-12):+.1f} | "
                         f"{100 * (w0[k] - b0[k]) / abs(b0[k]):+.2f} % |" for k in OUTS))
            lim_b = summary["mission"]["constraints"][1]
            extra += (f"\n\nOn the fine mesh the balance limit ({lim_b['text']}) of the fine baseline is "
                      f"{bf['balance_front'] * lim_b['lower_frac_of_baseline']:.4g} to "
                      f"{bf['balance_front'] * lim_b['upper_frac_of_baseline']:.4g}; the recommendation's "
                      f"{wf['balance_front']:.4g} is " + ("inside it" if bf['balance_front'] * lim_b['lower_frac_of_baseline']
                                                          <= wf['balance_front'] <= bf['balance_front'] * lim_b['upper_frac_of_baseline']
                                                          else "outside it") +
                      ". The campaign mesh moves the baseline's balance by "
                      f"{100 * qual['campaign_vs_reference'].get('balance_front', 0):+.1f} % against the fine mesh, so the "
                      "balance limit is the least resolved of the three outputs.")
        led = json.load(open(os.path.join(a.out, "ledger.json")))
        noise = {r["id"]: r["meta"].get("iteration_std", {}) for r in led["runs"] if r["ok"]}
        lim = {c["name"]: c for c in summary["mission"].get("constraints_absolute", [])}
        rows = []
        for d in summary["decisions"]:
            for k, c in lim.items():
                v, sd = d["reference"].get(k), noise.get(d["run"], {}).get(k)
                if v is None or not sd:
                    continue
                m = min(v - c["lower"] if c["lower"] is not None else float("inf"),
                        c["upper"] - v if c["upper"] is not None else float("inf"))
                rows.append(f"| {d['run']} | {k} | {m:+.4g} | {sd:.3g} | {m / sd:+.1f} | {d['label_after']} |")
        extra += ("\n\n## Limit margins against the solver's own iteration noise\n\nmargin = distance to the limit in "
                  "the CFD (negative: violated); noise = standard deviation of the output over the averaged "
                  "iterations of that run. A verdict with |margin / noise| < 2 is within the solver's noise.\n\n"
                  "| run | output | margin | noise | margin / noise | verdict |\n|---|---|---|---|---|---|\n"
                  + "\n".join(rows))
        if os.path.exists(os.path.join(a.work, "events.jsonl")):          # the session's event stream, for replays
            import shutil
            shutil.copy(os.path.join(a.work, "events.jsonl"), os.path.join(a.out, "events.jsonl"))
        held_run = held[-1]["run"] if held else None
        json.dump({"cp": held_run}, open(os.path.join(figs, "figures.json"), "w"))   # the run each figure shows
        if a.dashboard:
            from pinneapple_autolab.dashboard import capture, record_replay
            from pinneapple_autolab.domains.racecar import build_dashboard
            page = build_dashboard(a.work, os.path.join(a.work, "dashboard"), summary, results_dir=a.out,
                                   figures={"cp": F.get("cp"), **{k: v for k, v in F.items() if "cycles" in k}},
                                   figure_runs={"cp": held_run} if held_run else None, renders=True)
            F["dashboard"] = ex.add_file(capture(page, os.path.join(figs, "dashboard.png")))
            R.shrink_png(F["dashboard"])
            ex.add_file(record_replay(page, os.path.join(figs, "dashboard_replay.mp4")))
        R.write_report(summary, os.path.join(a.out, "report.md"), F, qual, extra)

        # ------------------------------------------------------------------ evidence for the lab manifest
        b = summary["budget"]
        ex.log_metric("cfd_runs", b["used_runs"], "runs", "OpenFOAM runs charged to the mission budget")
        ex.log_metric("cfd_wall", b["solver_wall_s"], "s", "summed wall time of the mission's OpenFOAM runs, 4 cores")
        ex.log_metric("failed_runs", b["failed_runs"], "runs", "runs that did not finish")
        if qual:
            for k, v in qual["campaign_vs_reference"].items():
                ex.log_metric(f"mesh_{k}_vs_fine", 100 * v, "%", f"baseline: {a.mesh} mesh vs fine mesh")
        if summary.get("improvement"):
            imp = summary["improvement"]
            ex.log_metric("drag_change", imp["drag_N"]["change_pct"], "%", "best simulated feasible vs baseline (CFD)")
            ex.log_metric("drag_best", imp["drag_N"]["best"], "N", "CFD of the recommended design")
            ex.log_metric("downforce_best", imp["downforce_N"]["best"], "N", "CFD of the recommended design")
            ex.check("recommendation_meets_downforce", value=imp["downforce_N"]["best"],
                     min=imp["downforce_N"]["baseline"], kind="physical", label="downforce >= baseline (CFD)")
            ex.check("recommendation_lowers_drag", passed=imp["drag_N"]["change"] < 0, kind="physical",
                     label="drag below the baseline's (CFD)")
            if fine:
                bf, wf = qual["variants"]["fine"]["outputs"], fine["variants"]["fine"]["outputs"]
                ex.log_metric("drag_change_fine", 100 * (wf["drag_N"] - bf["drag_N"]) / bf["drag_N"], "%",
                              "recommended vs baseline, both on the fine mesh")
                ex.check("fine_mesh_confirms_lower_drag", passed=wf["drag_N"] < bf["drag_N"], kind="reference",
                         label="the drag reduction holds on the fine mesh")
                ex.check("fine_mesh_confirms_downforce", passed=wf["downforce_N"] >= bf["downforce_N"],
                         kind="reference", label="the downforce limit holds on the fine mesh (margin within the "
                                                 "iteration noise of the fine runs)")
                lb = summary["mission"]["constraints"][1]
                ex.check("fine_mesh_balance_limit", value=wf["balance_front"],
                         min=bf["balance_front"] * lb["lower_frac_of_baseline"],
                         max=bf["balance_front"] * lb["upper_frac_of_baseline"], kind="reference",
                         label="the aero-balance limit holds on the fine mesh")
        for k, v in summary["board"]["table"].items():
            fam = summary["board"]["chosen"][k]
            ex.log_metric(f"cv_rel_error_{k}", 100 * v[fam]["rel_mae"], "%",
                          f"{fam}, {v[fam]['n']} CFD runs, 5-fold cross-validation")
        for d in summary["decisions"]:
            for k, e in d["rel_error"].items():
                if k in mission.acceptance and d["verification"]:
                    ex.check(f"{d['run']}_{k}_prediction", value=e, max=mission.acceptance[k], kind="reference",
                             label=f"{d['run']}: surrogate {k} within {100 * mission.acceptance[k]:.0f} % of CFD")
        print(summary["conclusion"])


if __name__ == "__main__":
    main()
