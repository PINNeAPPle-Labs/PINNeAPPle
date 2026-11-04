"""The whole study as one command: data, surrogate, optimisation, verification, figures and a report.

The workflow is what makes the pieces a product rather than a notebook:

``build-dataset``  sample the design space, label it with the reference model, store the set with its provenance
``train``          fit the ensemble, measure it on held-out designs, write the weights and the model card
``optimize``       NSGA-II on the surrogate alone; the front, the best designs and designs per second
``verify``         re-evaluate designs of that front with the reference model and report the error it finds
``check-coupling`` measure what the one-step magnet-temperature correction costs against a re-solved field
``all``            all of the above into one folder, with the figures and ``report.md``

Run it as ``python -m pinneapple_simulation.electromagnetics.machines.motor3d.workflow <command> [options]`` or call
:func:`run_workflow` from Python. Every step writes its own artefact (``dataset/``, ``model/``, ``optimization.json``,
``verification.json``, ``figures/``) so a later step can be re-run without repeating the expensive one, and the report
is rebuilt from the artefacts on disk rather than from memory.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import MotorDataset, build_dataset
from .design import BOUNDS, NAMES, VARIABLES, Motor3DDesign
from .model import OUTPUTS, PROFILES, SCALARS, Motor3DModel
from .optimize import benchmark, design_problem, optimize, verify_front
from .surrogate import SurrogateConfig, MotorSurrogate3D, load_pretrained

__all__ = ["run_workflow", "check_temperature_coupling", "figures", "write_report", "main"]


# ----------------------------------------------------------------------------------------------------- coupling
def check_temperature_coupling(n: int = 6, seed: int = 0, n_gap: int = 192) -> dict[str, Any]:
    """What the one-step magnet-temperature correction costs.

    The reference model solves the field once (at ``magnet_T_guess``) and then corrects torque, loss and margin for
    the magnet temperature it computed (``fea_iterations=1``). With ``fea_iterations=2`` the field is re-solved at
    that temperature. The two are compared here on ``n`` designs drawn from the box, so the shortcut has a number.
    """
    rng = np.random.default_rng(seed)
    one, two = Motor3DModel(n_gap=n_gap, fea_iterations=1), Motor3DModel(n_gap=n_gap, fea_iterations=2)
    rows = []
    X = []
    while len(X) < n:
        x = BOUNDS[:, 0] + (BOUNDS[:, 1] - BOUNDS[:, 0]) * rng.random(len(NAMES))
        if not Motor3DDesign.from_vector(x).sector_errors():
            X.append(x)
    for x in X:
        d = Motor3DDesign.from_vector(x)
        a, b = one.solve(d), two.solve(d)
        if not (a.get("valid") and b.get("valid")):
            continue
        rows.append({k: {"one_step": float(np.max(a[k])), "resolved": float(np.max(b[k])),
                         "rel": float(abs(np.max(a[k]) - np.max(b[k])) / max(abs(np.max(b[k])), 1e-9))}
                     for k in ("torque_Nm", "T_magnet_C", "iron_loss_W", "copper_loss_W", "demag_ratio")})
    worst = {k: max(r[k]["rel"] for r in rows) if rows else float("nan")
             for k in ("torque_Nm", "T_magnet_C", "iron_loss_W", "copper_loss_W", "demag_ratio")}
    return {"n": len(rows), "rows": rows, "worst_relative_error": worst,
            "description": "field solved once and corrected for Br(T) against the field re-solved at the converged "
                           "magnet temperature"}


# ----------------------------------------------------------------------------------------------------- figures
def figures(out_dir: str | Path, opt: dict[str, Any], ver: dict[str, Any], bench: dict[str, Any],
            surrogate: MotorSurrogate3D, data: MotorDataset | None = None) -> list[str]:
    """The figures of the report, from the artefacts: Pareto front, thermal trade-off, axial profiles, error and
    speed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    front = opt.get("front") or []
    E = opt["result"].evaluations
    ok = E.feasible & ~E.relaxed

    def scatter(ax, kx, ky, xl, yl, title):
        ax.scatter(E.Y[kx][~ok], E.Y[ky][~ok], s=6, c="0.75", label="infeasible")
        ax.scatter(E.Y[kx][ok], E.Y[ky][ok], s=6, c="C0", alpha=0.5, label="feasible")
        idx = opt["result"].pareto()
        if len(idx):
            ax.scatter(E.Y[kx][idx], E.Y[ky][idx], s=22, c="C3", marker="*", label="Pareto front")
        ax.set(xlabel=xl, ylabel=yl, title=title)
        ax.legend(fontsize=7, loc="best")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    scatter(ax[0], "torque_density_Nm_kg", "efficiency", "torque density [N m/kg]", "efficiency [-]",
            "Torque density against efficiency")
    scatter(ax[1], "torque_density_Nm_kg", "magnet_mass_kg", "torque density [N m/kg]", "magnet mass [kg]",
            "Torque density against magnet mass")
    scatter(ax[2], "torque_density_Nm_kg", "T_winding_C", "torque density [N m/kg]", "winding temperature [degC]",
            "What the temperature limit costs")
    ax[2].axhline(180.0, color="k", ls=":", lw=1, label="limit 180 degC")
    fig.tight_layout()
    p = out / "pareto.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    # axial profiles: the baseline against the best torque-density design, surrogate and reference
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    z = np.linspace(-0.5, 0.5, 200)
    for label, d, c in (("baseline (Prius 2004)", Motor3DDesign.baseline(), "0.4"),
                        ("best torque density", Motor3DDesign.from_vector(E.X[opt["result"].best(0)]), "C3")
                        if opt["result"].best(0) is not None else (None, None, None)):
        if d is None:
            continue
        mean, _ = surrogate.predict_one(d)
        prof_w = surrogate.predict(d.as_vector()[None, :])["T_winding_profile_C"][0][0]
        prof_t = surrogate.predict(d.as_vector()[None, :])["torque_slice_Nm"][0][0]
        K = len(prof_w)
        zz = np.linspace(-0.5, 0.5, K)
        ax[0].plot(zz, prof_w, "o-", color=c, label=label)
        ax[1].plot(zz, prof_t, "o-", color=c, label=label)
        ax[2].plot(zz, surrogate.predict(d.as_vector()[None, :])["T_magnet_profile_C"][0][0], "o-", color=c,
                   label=label)
    for a, t, yl in ((ax[0], "Winding temperature along the stack", "T [degC]"),
                     (ax[1], "Torque contributed by each axial segment", "T [N m]"),
                     (ax[2], "Magnet temperature along the stack", "T [degC]")):
        a.set(xlabel="axial position z / L [-]", ylabel=yl, title=t)
        a.legend(fontsize=7)
    fig.tight_layout()
    p = out / "axial_profiles.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    # surrogate error on the verified front
    if ver.get("errors"):
        names = [n for n in ver["errors"] if n in SCALARS]
        vals = [100 * ver["errors"][n]["mean_rel"] for n in names]
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.bar(range(len(names)), vals, color="C0")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set(ylabel="mean relative error [%]",
               title="Surrogate against the reference model, on the verified Pareto front")
        fig.tight_layout()
        p = out / "verification_error.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(str(p))

    # held-out parity of the surrogate, one panel per family
    if data is not None:
        okd = np.asarray(data.valid, bool)
        X = data.X[okd]
        pred = surrogate.predict(X)
        shows = [("torque_Nm", "torque [N m]"), ("efficiency", "efficiency [-]"), ("T_winding_C", "T winding [degC]"),
                 ("copper_loss_W", "copper loss [W]")]
        fig, ax = plt.subplots(1, len(shows), figsize=(13, 3.4))
        for a, (k, lab) in zip(ax, shows, strict=True):
            t = np.asarray(data.Y[k], float)[okd]
            g = np.asarray(pred[k][0], float)
            a.scatter(t, g, s=5, alpha=0.5)
            lo, hi = float(min(t.min(), g.min())), float(max(t.max(), g.max()))
            a.plot([lo, hi], [lo, hi], "k--", lw=1)
            a.set(xlabel=f"reference {lab}", ylabel=f"surrogate {lab}", title=k)
        fig.tight_layout()
        p = out / "parity.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(str(p))

    # speed
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(["reference\n(FEA + thermal)", "surrogate\n(ensemble)"],
           [bench["reference_per_design"], bench["surrogate_per_design"]], color=["C1", "C0"])
    ax.set_yscale("log")
    ax.set(ylabel="seconds per design (log scale)",
           title=f"{bench['speedup']:.0f}x fewer seconds per design")
    fig.tight_layout()
    p = out / "speed.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))
    return paths


# ------------------------------------------------------------------------------------------------------ report
def write_report(folder: str | Path) -> Path:
    """Rebuild ``report.md`` from the artefacts in ``folder`` (nothing is taken from memory)."""
    folder = Path(folder)
    card = json.loads((folder / "model" / "card.json").read_text())
    opt = json.loads((folder / "optimization.json").read_text())
    ver = json.loads((folder / "verification.json").read_text())
    bench = json.loads((folder / "benchmark.json").read_text())
    data_meta = json.loads((folder / "dataset" / "dataset.json").read_text())

    L: list[str] = []
    L.append("# 3-D multiphysics optimisation of an interior-PM motor on a pretrained surrogate\n")
    L.append("Every number below was produced by the code in "
             "`pinneapple_simulation.electromagnetics.machines.motor3d` on this machine; the artefacts it came from "
             "are in this folder.\n")
    L.append("## What was run\n")
    L.append("| step | result |\n|---|---|\n"
             f"| reference model | {data_meta['model_description']} |\n"
             f"| training set | {data_meta['n_cases']} designs drawn by {data_meta['method']} (seed "
             f"{data_meta['seed']}), {data_meta['n_valid']} valid, rejection rate "
             f"{100 * data_meta['rejection_rate']:.1f} % |\n"
             f"| cost of the set | {data_meta['seconds']} s wall, {data_meta['seconds_per_design']} s per design, "
             f"{data_meta['fea_solves']} FEA operating points |\n"
             f"| surrogate | {len(card['ensemble_members'])} networks, {card['n_train']} train / {card['n_val']} "
             f"held out, {card['seconds']} s to train |\n"
             f"| optimisation | NSGA-II, {opt['budget']} designs on the surrogate, {opt['wall_seconds']:.1f} s, "
             f"{opt['n_reference_solves']} FEA calls |\n"
             f"| verification | {ver['n_checked']} designs of the front re-evaluated with the reference model |\n")
    L.append("\n## Held-out error of the surrogate\n")
    L.append("| output | unit | MAE | R2 | mean rel. | 95th rel. |\n|---|---|---|---|---|---|\n")
    const = card.get("constant_outputs", {})
    for k in SCALARS:
        e = card["heldout"].get(k)
        if e is None:
            L.append(f"| {k} | constant ({const.get(k, 0.0):.3g}) | - | - | - | - |\n")
            continue
        L.append(f"| {k} | {e.get('unit', '')} | {e['mae']:.4g} | {e['r2']:.4f} | {100 * e['mean_rel']:.2f} % | "
                 f"{100 * e['p95_rel']:.2f} % |\n")
    L.append(f"\nMean R2 over the outputs that vary: **{card['mean_r2']:.4f}**; the output it explains least is "
             f"**{card['worst_output']}** (R2 {card['worst_r2']:.3f}). "
             + (f"Predicted as constants, because they do not move in this design space at these speeds: "
                f"{', '.join(sorted(const))}.\n" if const else ""))
    L.append("\n## The front\n")
    L.append(f"- designs evaluated: **{opt['n_evaluations']}** at **{opt['designs_per_second']:.0f} designs/s**\n")
    if opt.get("best_torque_density"):
        b = opt["best_torque_density"]
        L.append(f"- best torque density design: {b['torque_density_Nm_kg']:.2f} N m/kg "
                 f"({b['torque_Nm']:.1f} N m, magnet mass {b['magnet_mass_kg']:.2f} kg, "
                 f"T_winding {b['T_winding_C']:.0f} degC, ripple {b['ripple']:.3f})\n")
    if opt.get("best_efficiency"):
        b = opt["best_efficiency"]
        L.append(f"- best efficiency design: {b['efficiency']:.4f} "
                 f"({b['torque_Nm']:.1f} N m, T_winding {b['T_winding_C']:.0f} degC)\n")
    L.append(f"- designs of the front: {len(opt.get('front') or [])}\n")
    L.append("\n## Verification against the reference model\n")
    L.append(f"{ver['still_feasible']} of {ver['n_checked']} designs of the front are still feasible when the "
             f"reference model evaluates them under the same limits; Spearman rank agreement on torque density is "
             f"{ver['rank_agreement']:.3f}.\n")
    L.append("\n| output | mean rel. error | max rel. error |\n|---|---|---|\n")
    for k in SCALARS:
        e = ver["errors"][k]
        L.append(f"| {k} | {100 * e['mean_rel']:.2f} % | {100 * e['max_rel']:.2f} % |\n")
    L.append(f"\n## Speed\n\nThe reference model spends {bench['reference_per_design']:.3f} s per design and the "
             f"surrogate {1e6 * bench['surrogate_per_design']:.1f} us: **{bench['speedup']:.0f}x**. The whole front "
             f"above therefore cost {bench['reference_per_design'] * opt['n_evaluations'] / 3600:.1f} h of reference "
             f"computing that was not spent -- it was spent once, on the {data_meta['n_cases']} designs of the "
             f"training set.\n")
    figures_dir = folder / "figures"
    if figures_dir.exists():
        for p in sorted(figures_dir.glob("*.png")):
            L.append(f"\n![{p.stem}](figures/{p.name})\n")
    L.append("\n## What the study does not model\n")
    L.append("- the field is 2-D per axial segment: the axial flux paths (end leakage, axial saturation) enter through "
             "the end-winding resistance and the thermal network, not through a 3-D field solve;\n"
             "- the coolant is a boundary conductance that follows the flow as flow^0.8, not a conjugate heat "
             "transfer solution: the heat-transfer coefficient is an input, as in the rest of the library's machine "
             "model;\n"
             "- the structural side (bridge stress at overspeed) is not evaluated;\n"
             "- the surrogate is only valid inside the box of its card, and its ensemble spread is a proxy for "
             "uncertainty, not a calibrated interval.\n")
    path = folder / "report.md"
    path.write_text("".join(L))
    return path


# ----------------------------------------------------------------------------------------------------- workflow
def run_workflow(out_dir: str | Path, *, n_train: int = 600, seed: int = 0, workers: int = 4, budget: int = 2000,
                 n_gap: int = 192, n_positions: int = 12, reuse: bool = True,
                 progress=None) -> dict[str, Any]:
    """data -> surrogate -> optimisation -> verification -> figures -> report, into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    data_dir, model_dir = out / "dataset", out / "model"
    if reuse and (data_dir / "dataset.npz").exists():
        data = MotorDataset.load(data_dir)
    else:
        data = build_dataset(n_train, seed=seed, model=Motor3DModel(n_gap=n_gap, n_positions=n_positions),
                             workers=workers, extra=[Motor3DDesign.baseline()], progress=progress)
        data.save(data_dir)
    if progress:
        progress(f"dataset: {len(data)} designs, {int(data.valid.sum())} valid")

    if reuse and (model_dir / "weights.pt").exists():
        surrogate = MotorSurrogate3D.load(model_dir)
    else:
        surrogate = MotorSurrogate3D(SurrogateConfig(seed=seed)).fit(data, progress=None)
        surrogate.save(model_dir, version=f"workflow-{seed}")
    if progress:
        progress(f"surrogate: mean held-out relative error {100 * surrogate.card_['mean_rel_overall']:.2f} %")

    problem = design_problem(surrogate=surrogate)
    opt = optimize(problem, budget=budget, seed=seed)
    (out / "optimization.json").write_text(json.dumps(_jsonable(opt), indent=2))
    if progress:
        progress(f"optimisation: {opt['n_evaluations']} designs, {len(opt['front'])} on the front")

    ver = verify_front(opt, n=8)
    (out / "verification.json").write_text(json.dumps(_jsonable(ver), indent=2))
    bench = benchmark(surrogate, n=32, seed=seed)
    (out / "benchmark.json").write_text(json.dumps(_jsonable(bench), indent=2))

    figs = figures(out, opt, ver, bench, surrogate, data)
    write_report(out)
    return {"folder": str(out), "figures": figs, "seconds": time.perf_counter() - t0,
            "dataset": data.meta, "card": surrogate.card_, "optimization": _jsonable(opt),
            "verification": _jsonable(ver), "benchmark": bench}


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if k != "result"}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return str(o)


# ---------------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="motor3d", description="3-D multiphysics motor study on a pretrained surrogate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-dataset", help="sample the design space and label it with the reference model")
    b.add_argument("--n", type=int, default=600)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--n-gap", type=int, default=192)
    b.add_argument("--out", default="motor3d_run/dataset")

    t = sub.add_parser("train", help="fit the ensemble on a stored dataset")
    t.add_argument("--data", default="motor3d_run/dataset")
    t.add_argument("--out", default="motor3d_run/model")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--ensemble", type=int, default=5)

    o = sub.add_parser("optimize", help="NSGA-II on the surrogate alone")
    o.add_argument("--model", default="", help="model folder (default: the pretrained one)")
    o.add_argument("--budget", type=int, default=2000)
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--out", default="motor3d_run/optimization.json")

    v = sub.add_parser("verify", help="re-evaluate designs of the front with the reference model")
    v.add_argument("--optimization", default="motor3d_run/optimization.json")
    v.add_argument("--n", type=int, default=8)

    c = sub.add_parser("check-coupling", help="cost of the one-step magnet-temperature correction")
    c.add_argument("--n", type=int, default=4)
    c.add_argument("--seed", type=int, default=0)

    a = sub.add_parser("all", help="the whole study into one folder")
    a.add_argument("--out", default="motor3d_run")
    a.add_argument("--n", type=int, default=600)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--budget", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--fresh", action="store_true", help="recompute the dataset and the surrogate")

    args = ap.parse_args(argv)
    if args.cmd == "build-dataset":
        ds = build_dataset(args.n, seed=args.seed, model=Motor3DModel(n_gap=args.n_gap), workers=args.workers,
                           extra=[Motor3DDesign.baseline()])
        print(ds.save(args.out), f"{len(ds)} designs, {int(ds.valid.sum())} valid, "
                                 f"{ds.meta['seconds_per_design']} s per design")
    elif args.cmd == "train":
        ds = MotorDataset.load(args.data)
        s = MotorSurrogate3D(SurrogateConfig(seed=args.seed, ensemble=args.ensemble)).fit(ds)
        print(s.save(args.out), f"held-out mean relative error {100 * s.card_['mean_rel_overall']:.2f} %")
    elif args.cmd == "optimize":
        s = load_pretrained() if not args.model else MotorSurrogate3D.load(args.model)
        opt = optimize(design_problem(surrogate=s), budget=args.budget, seed=args.seed)
        Path(args.out).write_text(json.dumps(_jsonable(opt), indent=2))
        print(f"{Path(args.out)}: {opt['n_evaluations']} designs, {len(opt['front'])} on the front, "
              f"{opt['designs_per_second']:.0f} designs/s, {opt['n_reference_solves']} FEA calls")
    elif args.cmd == "verify":
        opt = json.loads(Path(args.optimization).read_text())
        ver = verify_front(opt, n=args.n)
        out = Path(args.optimization).with_name("verification.json")
        out.write_text(json.dumps(_jsonable(ver), indent=2))
        print(f"{out}: {ver['n_checked']} designs re-evaluated, {ver['still_feasible']} still feasible, "
              f"rank agreement {ver.get('rank_agreement', float('nan')):.3f}")
    elif args.cmd == "check-coupling":
        r = check_temperature_coupling(args.n, args.seed)
        print(json.dumps(r["worst_relative_error"], indent=2))
    else:
        out = run_workflow(args.out, n_train=args.n, workers=args.workers, budget=args.budget, seed=args.seed,
                           reuse=not args.fresh, progress=print)
        print(Path(out["folder"]) / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
