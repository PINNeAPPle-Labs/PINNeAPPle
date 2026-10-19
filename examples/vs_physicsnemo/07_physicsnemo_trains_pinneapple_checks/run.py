"""PhysicsNeMo trains, PINNeAPPle says when to trust: FNO and MeshGraphNet on 2-D Darcy, then the trust layer.

    python run.py data                       # shifted test sets (reference solver), ~3 min on 4 cores
    python run.py train --model fno --budget-s 3600
    python run.py train --model meshgraphnet --budget-s 3600
    python run.py evaluate                   # checks, trust cards, results.json, figure, GIF, `pinneapple check` files
    python run.py datapipe                   # the same checks on a batch of PhysicsNeMo's Darcy2D datapipe
    python run.py report                     # Veriphysics Decision Record + Evidence Report (PDF) per model

Both models are PhysicsNeMo's own (``physicsnemo.models.fno.FNO``, ``physicsnemo.models.meshgraphnet.MeshGraphNet``),
trained on ``darcy_2d_li2021`` (1000 train / 200 test, 128 x 128) with the same wall-clock budget
(``ComputeBudget``). PINNeAPPle trains nothing: it checks every prediction against the physics
(``pinneapple_physics.darcy_checks``) without the reference solution, and gives a deploy decision per model and
regime. Everything here is composition of library calls; outputs go to ``examples/_out/physicsnemo_darcy``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import pinneapple as pp
from pinneapple_analysis.cost.equal_budget import ComputeBudget, hardware_record, time_per_query
from pinneapple_neural.physicsnemo_bridge.field_operators import (
    GridFieldPredictor,
    physicsnemo_fno,
    physicsnemo_meshgraphnet,
    train_grid_operator,
)
from pinneapple_physics import darcy_checks as dc
from pinneapple_simulation.numerical_solvers.operator_datasets import DARCY, darcy_set
from pinneapple_tools.visualization.trust_panels import Panel, Row, trust_panels, trust_panels_gif

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "_out" / "physicsnemo_darcy"
CHECK = HERE / "check"
#: Regimes: the dataset's test split, and two shifts of the input distribution solved with the same reference solver.
SHIFTS = {"contrast": dict(high=20.0, low=1.0, seed=7701, label="contrast a in {1, 20}"),
          "length_scale": dict(tau=9.0, seed=7702, label="3x shorter length scale (tau 9)")}
MODELS = {"fno": dict(build=lambda: physicsnemo_fno(128, latent_channels=32, num_fno_layers=4, num_fno_modes=12,
                                                    decoder_layer_size=64), batch_size=8, lr=1e-3),
          "meshgraphnet": dict(build=lambda: physicsnemo_meshgraphnet(128, processor_size=15, hidden_dim=32),
                               batch_size=2, lr=1e-3)}
LABEL = {"fno": "FNO", "meshgraphnet": "MeshGraphNet"}
N_CAL = 100           # first half of the test split: calibration inputs of the drift monitor (labels not used)


def load_regimes(n_shift: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Test split halves: ``calibration`` (inputs only are used, to calibrate the drift limits) and
    ``in_distribution``; then the shifted sets."""
    test = pp.datasets.load("darcy_2d_li2021", split="test")
    a, u = test["a"].astype(np.float64), test["u"].astype(np.float64)
    reg = {"calibration": {"a": a[:N_CAL], "u": u[:N_CAL]}, "in_distribution": {"a": a[N_CAL:], "u": u[N_CAL:]}}
    for k in SHIFTS:
        with np.load(OUT / f"shift_{k}.npz") as z:
            reg[k] = {"a": z["a"][:n_shift], "u": z["u"][:n_shift]}
    return reg


def cmd_data(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for k, s in SHIFTS.items():
        t0 = time.perf_counter()
        a, u = darcy_set(args.n_shift, s["seed"], tau=s.get("tau"), high=s.get("high"), low=s.get("low"),
                         workers=args.workers)
        np.savez_compressed(OUT / f"shift_{k}.npz", a=a, u=u)
        print(f"{k}: {len(a)} samples in {time.perf_counter() - t0:.0f} s")


def cmd_train(args) -> None:
    train = pp.datasets.load("darcy_2d_li2021", split="train")
    spec = MODELS[args.model]
    model = spec["build"]()
    budget = ComputeBudget(wall_s=args.budget_s)
    t0 = time.perf_counter()
    res = train_grid_operator(model, train["a"], train["u"], budget, batch_size=spec["batch_size"], lr=spec["lr"],
                              threads=args.threads)
    model.save(OUT / "models" / args.model)
    rec = {"model": args.model, "parameters": model.n_parameters(), "config": model.config,
           "budget": budget.to_dict(), "elapsed_s": time.perf_counter() - t0, "hardware": hardware_record(),
           **res.to_dict()}
    rec.pop("history", None)
    (OUT / "models" / f"{args.model}_training.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"{args.model}: {res.run.steps} steps, {res.epochs:.2f} epochs, final loss {res.final_loss:.4f}")


def _predictions(models, regimes, recompute: bool) -> dict:
    cache = OUT / "predictions.npz"
    if cache.exists() and not recompute:
        with np.load(cache) as z:
            return {tuple(k.split("__")): z[k] for k in z.files}
    preds = {(r, m): model.predict(data["a"]) for r, data in regimes.items() for m, model in models.items()}
    np.savez_compressed(cache, **{f"{r}__{m}": v for (r, m), v in preds.items()})
    return preds


def cmd_evaluate(args) -> None:
    train = pp.datasets.load("darcy_2d_li2021", split="train")
    envelope = dc.CoefficientEnvelope.fit(train["a"])
    regimes = load_regimes(args.n_shift)
    models = {m: GridFieldPredictor.load(OUT / "models" / m) for m in MODELS}
    preds = _predictions(models, regimes, args.recompute)
    results: dict = {"envelope": envelope.to_dict(), "profiles": {k: {"use": p.use, "limits": dict(p.limits),
                                                                       "reasons": dict(p.reasons)}
                                                                   for k, p in dc.PROFILES.items()},
                     "regimes": {}, "models": {}}
    calibrated = {}
    for m, model in models.items():
        a0 = regimes["in_distribution"]["a"][:8]
        tq = time_per_query(lambda model=model, a0=a0: model.predict(a0), repeats=3, warmup=1, n_per_call=len(a0))
        train_rec = json.loads((OUT / "models" / f"{m}_training.json").read_text())
        calibrated[m] = dc.calibrate_tolerances(regimes["calibration"]["a"], preds[("calibration", m)], DARCY["f"])
        results["models"][m] = {"parameters": model.n_parameters(), "inference_s_per_sample": tq,
                                "drift_limits": calibrated[m],
                                "training": {k: train_rec[k] for k in ("steps", "epochs", "wall_s", "final_loss",
                                                                        "stopped_by")}}
    for r, data in regimes.items():
        if r == "calibration":
            continue
        out = results["regimes"][r] = {"n": len(data["a"])}
        out["reference_solution"] = dc.profile_verdicts(data["a"], data["u"], DARCY["f"])
        for m in models:
            u = preds[(r, m)]
            prof = dc.profile_verdicts(data["a"], u, DARCY["f"])
            ref = dc.darcy_checks(data["a"], u, DARCY["f"], reference=data["u"], envelope=envelope)
            drift = dc.darcy_checks(data["a"], u, DARCY["f"], tolerances=calibrated[m], physics_kind="drift")
            verdict = {**{f"physics:{k}": v["decision"] for k, v in prof.items()},
                       "drift": dc.trust_report(drift).decisao,
                       "inputs": dc.trust_report(ref, kinds=("applicability",)).decisao}
            err = ref[[c.kind for c in ref].index("reference")].values
            est = dc.regional_errors(u, data["a"], DARCY["f"])
            masks = dc.region_masks(u.shape[-1])
            true = {k: np.linalg.norm(((data["u"] - u) * mk).reshape(len(u), -1), axis=1)
                    / np.linalg.norm((data["u"] * mk).reshape(len(u), -1), axis=1) for k, mk in masks.items()}
            uses = [dc.use_verdicts(data["a"][i], u[i], DARCY["f"], envelope=envelope) for i in range(len(u))]
            regions = {"estimated_median": {k: float(np.median(v)) for k, v in est.items()},
                       "true_median": {k: float(np.median(v)) for k, v in true.items()},
                       "use_ok_fraction": {k: float(np.mean([x[k]["ok"] for x in uses])) for k in uses[0]}}
            out[m] = {"verdicts": verdict, "profiles": prof, "regions": regions,
                      "rel_l2_median": float(np.median(err)),
                      "rel_l2_p90": float(np.quantile(err, 0.9)),
                      "checks": [c.to_dict() for c in ref + drift]}
            print(f"[{r:15s}] {LABEL[m]:12s} rel L2 median {out[m]['rel_l2_median']:.4f}  "
                  + "  ".join(f"{k} {v['decision']} ({v['ratio']:.2f}x, worst {v['worst']})" for k, v in prof.items())
                  + f"  drift {verdict['drift']}  inputs {verdict['inputs']}")
            print(f"      regions: estimated {regions['estimated_median']}, true {regions['true_median']}, "
                  f"uses OK {regions['use_ok_fraction']}")
    exported = _export_for_check(regimes, preds, calibrated, envelope)
    results["exported"] = exported
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    _figures(regimes, preds, results, exported)


def _exported_samples(regimes, envelope) -> dict[str, list[int]]:
    """Two samples per regime: the first two of the in-distribution and contrast sets; for the length-scale set the
    first one outside the training envelope of a, then the first one inside it."""
    out = {}
    for r, data in regimes.items():
        if r == "calibration":
            continue
        if r == "length_scale":
            inside = envelope.contains(data["a"])
            out[r] = [int(np.flatnonzero(~inside)[0]), int(np.flatnonzero(inside)[0])]
        else:
            out[r] = [0, 1]
    return out


def _export_for_check(regimes, preds, calibrated, envelope) -> dict:
    """The exported predictions as grid results for `pinneapple check`, and its configuration: per target, the
    tolerances are its model's validated limits (``calibrate_tolerances``) and the envelope of ``a`` is the
    training envelope. Returns, per exported file, the verdict of ``sample_checks`` with the configuration as
    written (what the CI computes)."""
    try:
        import tomllib
    except ModuleNotFoundError:                           # Python 3.10
        import tomli as tomllib
    idx = _exported_samples(regimes, envelope)
    head = ["# `pinneapple check` on predictions of the two PhysicsNeMo models (written by ../run.py evaluate).",
            "# One predicted field u per file, for one coefficient a on the 128 x 128 cell centres of the unit square.",
            "# physics = \"darcy\": -div(a grad u) = 1, u = 0 on the boundary, checked with the reference solver's",
            "# finite volumes. tolerances: the model's own 99th percentiles on held-out in-distribution inputs (no",
            "# labels); bounds and envelope of a: the training coefficients. Exported samples: the first two of the",
            "# test and contrast sets; for the length-scale set the first outside the envelope (_0), then the first",
            "# inside it (_1).", ""]
    entries, files = [], {}
    for (r, m), u in preds.items():
        if r == "calibration":
            continue
        for j, i in enumerate(idx[r]):
            rel = f"predictions/{r}/{m}_{j}.npz"
            dc.save_darcy_npz(CHECK / rel, regimes[r]["a"][i], u[i])
            entries.append(dc.check_config_entry(f"{r}/{m}_{j}", rel, f=DARCY["f"], tolerances=calibrated[m],
                                                 envelope=envelope))
            files[f"{r}/{m}_{j}"] = (r, m, i, rel)
    text = "\n".join(head) + "\n".join(entries)
    (CHECK / "pinneapple-check.toml").write_text(text)
    cfg = {t["name"]: t for t in tomllib.loads(text)["result"]}
    out = {}
    for name, (r, m, i, rel) in files.items():
        t = cfg[name]
        tol = {k: t["tolerances"][v] for k, v in dc.CHECK_TOLERANCE_KEYS.items()}
        with np.load(CHECK / rel) as z:                      # the float32 file the CI reads
            checks = dc.sample_checks(z["a"].astype(float), z["u"].astype(float), t["parameters"]["f"],
                                      tolerances=tol, envelope=dc.CoefficientEnvelope.from_check_config(t["envelope"]))
        out[name] = {"regime": r, "model": m, "sample": i, "verdict": dc.sample_verdict(checks),
                     "failed": [c["key"] for c in checks if not c["passed"]]}
    (CHECK / "sample_verdicts.json").write_text(json.dumps(out, indent=1) + "\n")
    # the rules `pinneapple check` breaks on the predictions that must fail (checked by the darcy-trust CI job)
    from pinneapple_check import run as check_run
    rep = check_run([CHECK / "predictions"], config=CHECK / "pinneapple-check.toml")
    got: dict[str, set[str]] = {}
    for v in rep.violations:
        got.setdefault(v.target, set()).add(v.rule)
    (CHECK / "expected.json").write_text(json.dumps({k: sorted(v) for k, v in sorted(got.items())}, indent=1) + "\n")
    return out


def _figures(regimes, preds, results, exported) -> None:
    """Where each model works and why: per regime, the predictions, their error map estimated from the residual
    alone (dashed square: the interior, 8 cells from the walls), and per prediction which uses it can serve."""
    train = pp.datasets.load("darcy_2d_li2021", split="train")
    envelope = dc.CoefficientEnvelope.fit(train["a"])

    def short(use, v):
        if use == "inputs":
            return "inputs: OK, as in training" if v["ok"] else "inputs: NO, outside training"
        k = max(v["limits"], key=lambda k: v["values"][k] / v["limits"][k])
        name = {"interior": "error", "wall": "wall error", "flux": "flux"}[k]
        sign = "<=" if v["values"][k] <= v["limits"][k] else ">"
        label = {"interior field": "interior field", "walls and fluxes": "walls & fluxes"}[use]
        return f"{label}: {'OK' if v['ok'] else 'NO'}, {name} ~{100 * v['values'][k]:.0f}% {sign} " \
               f"{100 * v['limits'][k]:g}%"

    def rows(j):
        out = []
        for r, data in regimes.items():
            if r == "calibration":
                continue
            ex = {m: exported[f"{r}/{m}_{j}"] for m in MODELS}
            i = ex[next(iter(MODELS))]["sample"]
            a = data["a"][i]
            panels = [Panel("reference solver (for reading)", data["u"][i])]
            errs = []
            for m in MODELS:
                u = preds[(r, m)][i]
                err = np.linalg.norm(u - data["u"][i]) / np.linalg.norm(data["u"][i])
                uses = dc.use_verdicts(a, u, DARCY["f"], envelope=envelope)
                panels.append(Panel(f"PhysicsNeMo {LABEL[m]}", u, note=f"true error {100 * err:.1f}%",
                                    verdicts=[short(k, uses[k]) for k in ("interior field", "walls and fluxes",
                                                                          "inputs")]
                                    + [f"CI, pinneapple check: {ex[m]['verdict']}"]))
                e = np.abs(dc.estimated_error(u, a, DARCY["f"])) / np.max(np.abs(u)) * 100
                errs.append(Panel(f"{LABEL[m]}: error of u", e, kind="error", inset=8))
                errs.append(Panel(f"{LABEL[m]}: error of the flux", 100 * dc.estimated_flux_error(u, a, DARCY["f"]),
                                  kind="error", inset=8))
            label = {"in_distribution": "test set (as trained)"}.get(r, SHIFTS.get(r, {}).get("label", r))
            out.append(Row(f"{label}, sample {i}", panels + errs))
        return out

    ind = results["regimes"]["in_distribution"]
    rf, rm = ind["fno"]["regions"]["estimated_median"], ind["meshgraphnet"]["regions"]["estimated_median"]
    title = "PhysicsNeMo trains, PINNeAPPle says where to trust: 2-D Darcy, -div(a grad u) = 1"
    footer = (f"Orange maps: where each model errs, estimated from the PDE residual alone, no reference: the error of u "
              f"(% of max u) and of the flux a grad u (% of the largest flux); dashed: interior, 8 cells from the "
              f"walls.\nUses: interior field OK if the estimated interior error <= 5 %; walls & fluxes OK if the "
              f"wall-band error <= 5 % and the flux balance <= 2 %.\n"
              f"Why, on the test set: the FNO gets u right away from the walls (estimated median interior error "
              f"{100 * rf['interior']:.1f} %) but not in the 2 cells next to them ({100 * rf['wall']:.0f} %), and its "
              f"flux errs at the walls and along the jumps of a: trust its interior values, not its wall gradients or "
              f"boundary fluxes.\nThe MeshGraphNet errs everywhere (interior {100 * rm['interior']:.0f} %, "
              f"walls {100 * rm['wall']:.0f} %; its flux errs along every jump of a): 3.7 epochs in the same CPU "
              f"hour and a 15-cell receptive field. "
              f"Contrast {{1, 20}}: inputs outside training, errors everywhere. CI badge: `pinneapple check` on this "
              f"prediction.")
    kw = dict(title=title, footer=footer, error_max=15.0, error_label="estimated error, % of max", cell=2.6)
    fig = trust_panels(rows(0), **kw)
    fig.savefig(OUT / "physicsnemo_darcy_trust.png", facecolor=fig.get_facecolor())
    trust_panels_gif([rows(j) for j in range(2)], OUT / "physicsnemo_darcy_trust.gif", dpi=80, duration_ms=2500, **kw)
    print("figure:", OUT / "physicsnemo_darcy_trust.png", "GIF:", OUT / "physicsnemo_darcy_trust.gif")


def cmd_report(args) -> None:
    """A Veriphysics Decision Record and Evidence Report per model: the scope of use over the test set (which uses
    the model serves, where it fails and why), the shifted regimes as notes, and the figure."""
    from pinneapple_veriphysics.decision import DecisionRecord
    from pinneapple_veriphysics.evidence_report import render_evidence_report_pdf
    from pinneapple_veriphysics.scope import scope_from_batch
    train = pp.datasets.load("darcy_2d_li2021", split="train")
    envelope = dc.CoefficientEnvelope.fit(train["a"])
    regimes = load_regimes()
    models = {m: GridFieldPredictor.load(OUT / "models" / m) for m in MODELS}
    preds = _predictions(models, regimes, False)
    res = json.loads((OUT / "results.json").read_text())
    evidence = {"interior field": "estimated", "walls and fluxes": "estimated", "inputs": "input check"}
    validation = ("Estimated from the PDE residual alone (no reference). Validated on this case against the "
                  "reference solver: the use verdicts agree with those from the true error on 94-100 % of samples in "
                  "every regime, every disagreement conservative; the estimate over-estimates the error 1.4-2.7x.")
    for m in MODELS:
        scopes = {}
        for r, data in regimes.items():
            if r == "calibration":
                continue
            verdicts = [dc.use_verdicts(data["a"][i], preds[(r, m)][i], DARCY["f"], envelope=envelope)
                        for i in range(len(data["a"]))]
            label = {"in_distribution": "test set"}.get(r, SHIFTS.get(r, {}).get("label", r))
            scopes[r] = scope_from_batch(f"PhysicsNeMo {LABEL[m]}, {label}", verdicts, evidence=evidence)
        main = scopes["in_distribution"]
        main.figure = str(OUT / "physicsnemo_darcy_trust.png")
        main.notes = [validation] + [f"{s.subject}: {s.summary()} ("
                                     + ", ".join(f"{u.use} {100 * u.fraction_ok:.0f} %" for u in s.uses)
                                     + " of samples OK)" for r, s in scopes.items() if r != "in_distribution"]
        tr = res["models"][m]["training"]
        prof = res["regimes"]["in_distribution"][m]["profiles"]
        rec = DecisionRecord(
            run_id=f"physicsnemo-darcy-{m}",
            problem_description="2-D Darcy, -div(a grad u) = 1, u = 0 on the unit-square boundary; zoo dataset "
                                "darcy_2d_li2021 (128 x 128); prediction by a PhysicsNeMo model",
            recommended_strategy=f"PhysicsNeMo {LABEL[m]}: {main.summary()}",
            reasoning=f"Trained with PhysicsNeMo for {tr['wall_s'] / 60:.0f} min on 4 CPU threads ({tr['epochs']:.1f} "
                      f"epochs). Certified profiles on the test set: fields {prof['fields']['decision']} "
                      f"({prof['fields']['ratio']:.1f}x limit), fluxes {prof['fluxes']['decision']} "
                      f"({prof['fluxes']['ratio']:.1f}x limit).",
            trust_score=None, trust_coverage=0.0, trustworthy=None,
            next_best_action=("For the uses listed under 'not for', use the reference finite-volume solver "
                              "(0.07 s per case at 128 x 128 on this CPU) or retrain with a flux-aware loss; check "
                              "new inputs against the training envelope (pinneapple check, PC401)."),
            evidence={"results": str(OUT / "results.json")},
            scope_of_use=main.to_dict())
        (OUT / f"decision_record_{m}.txt").write_text(rec.render() + "\n")
        (OUT / f"evidence_report_{m}.pdf").write_bytes(render_evidence_report_pdf(rec))
        print(rec.render().split("\n\nPer-check")[0])
        print("->", OUT / f"evidence_report_{m}.pdf")


def cmd_datapipe(args) -> None:
    from physicsnemo import __version__ as pnemo_version
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
    pipe = Darcy2D(resolution=args.resolution, batch_size=args.batch, device="cpu", max_iterations=args.iterations,
                   nr_multigrids=args.multigrids)
    t0 = time.perf_counter()
    it = iter(pipe)
    ks, us, fouriers = [], [], []
    for _ in range(args.batches):                           # each batch is drawn with a new seed
        batch = next(it)
        ks.append(batch["permeability"][:, 0].numpy().copy())
        us.append(batch["darcy"][:, 0].numpy().copy())
        fouriers.append(pipe.rand_fourier.numpy().copy())   # (4, batch, n_freq, n_freq) amplitudes of this batch
    k, u = np.concatenate(ks), np.concatenate(us)
    # residual of the documented equation; for comparison, of the stencil with grad k . grad u divided by 2 dx
    # (the Jacobi kernel of this version, NVIDIA/physicsnemo#2036)
    report = dc.check_nodal_darcy_batch(k, u, f=1.0, dx=pipe.dx,
                                        alt_cross_denominators={"grad k . grad u over 2 dx": 2 * pipe.dx})
    per_sample = np.concatenate([fo.transpose(1, 0, 2, 3).reshape(args.batch, -1) for fo in fouriers])
    report["fourier_amplitudes"] = {
        "distinct_values_per_sample": [int(np.unique(row).size) for row in per_sample],
        "distinct_samples": len({row.tobytes() for row in per_sample}),
        "values_per_sample": int(per_sample.shape[1])}
    report.update({"physicsnemo_version": pnemo_version, "resolution": args.resolution, "batch": args.batch,
                   "batches": args.batches,
                   "generation_s": time.perf_counter() - t0})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datapipe_check.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--n-shift", type=int, default=100)
    d.add_argument("--workers", type=int, default=4)
    t = sub.add_parser("train")
    t.add_argument("--model", choices=list(MODELS), required=True)
    t.add_argument("--budget-s", type=float, default=3600.0)
    t.add_argument("--threads", type=int, default=4)
    e = sub.add_parser("evaluate")
    e.add_argument("--n-shift", type=int, default=None)
    e.add_argument("--recompute", action="store_true", help="predict again instead of reading predictions.npz")
    p = sub.add_parser("datapipe")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--multigrids", type=int, default=4)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"data": cmd_data, "train": cmd_train, "evaluate": cmd_evaluate, "datapipe": cmd_datapipe,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
