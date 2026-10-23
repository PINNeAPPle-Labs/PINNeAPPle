"""End-to-end studies of a printed end-fire antenna, as library calls (the example and the Flow Studio call these).

* :func:`limits_report` -- the end-fire lengths a target gain needs across a band and the feasibility frequency of
  given lengths (no solver);
* :func:`open_study` -- the open AVA model of :mod:`.ava` in EMerge: DOE, coarse and fine runs (one frequency per
  process), checks of every row, multi-fidelity bias, LODO of four surrogates, NSGA-II on the GP, verification of
  the best predicted design, active learning, figures, ``summary.json`` and CSV tables;
* :func:`package_study` -- an external package (results table + CLI script, :mod:`.package`): checks, limits, LODO,
  figures, and a re-run of one of its designs compared with the table.

Each returns a JSON-able summary of measured values only.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import bounds, plots, runner
from . import design as dz
from . import surrogates as sg
from .ava import AVAParams, layout
from .checks import antenna_checks
from .package import load_package

__all__ = ["TARGETS", "limits_report", "check_summary", "validate_targets", "study_figure", "open_study",
           "package_study"]

TARGETS = ("Greal_bore_dBi", "D_peak_dBi", "S11_dB")


def limits_report(target_dbi: float = 10.0, f_min_ghz: float = 2.0, f_max_ghz: float = 4.0, n: int = 5,
                  lengths_mm: Sequence[float] = (150.0, 206.0, 260.0)) -> dict[str, Any]:
    """Shortest end-fire length for ``target_dbi`` at ``n`` frequencies of the band (Hansen-Woodyard and ordinary)
    and the lowest frequency each of ``lengths_mm`` can reach it."""
    fs = [f_min_ghz + i * (f_max_ghz - f_min_ghz) / max(n - 1, 1) for i in range(n)]
    return {"target_dbi": target_dbi,
            "min_length_hw_mm": {f"{f:g} GHz": float(bounds.min_length(target_dbi, f * 1e9) * 1e3) for f in fs},
            "min_length_ordinary_mm": {f"{f:g} GHz": float(bounds.min_length(target_dbi, f * 1e9, "ordinary") * 1e3)
                                       for f in fs},
            "feasibility_GHz": {f"{L:g} mm": float(bounds.feasibility_frequency(target_dbi, L * 1e-3) / 1e9)
                                for L in lengths_mm}}


def check_summary(table: runner.ResultTable, *, n_solved: int | None = None, interpolated: bool = False,
                  margin_db: float = 0.5) -> dict[str, Any]:
    """Counts of check outcomes and decisions over every row, and how often each check failed."""
    out: dict[str, Any] = {"rows": len(table), "supports": 0, "contradicts": 0, "not_run": 0, "decisions": {},
                           "failed": {}}
    for r in table.rows:
        rep = antenna_checks(r, n_solved=n_solved, interpolated=interpolated, margin_db=margin_db)
        out["decisions"][rep.decisao] = out["decisions"].get(rep.decisao, 0) + 1
        for c in rep.checks:
            out[c.status] += 1
            if c.status == "contradicts":
                out["failed"][c.name] = out["failed"].get(c.name, 0) + 1
    return out


def validate_targets(table: runner.ResultTable, targets: Sequence[str] = TARGETS) -> dict[str, sg.ValidationResult]:
    """Leave-one-design-out of the default surrogates for each target present in ``table``."""
    return {t: sg.lodo(table, t) for t in targets if t in table.columns and len(table.tags) >= 3}


def study_figure(table: runner.ResultTable, vals: Mapping[str, sg.ValidationResult], path: str, *,
                 title: str = "", target_dbi: float = 10.0, column: str = "D_peak_dBi") -> str:
    """Directivity against ``L/lambda`` with the end-fire limits and the target, and the LODO RMSE per model."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    plots.directivity_vs_length(table, ax[0], column=column, target_dbi=target_dbi)
    ax[0].set_title(f"{title}: {column} vs end-fire limits".strip(": "), fontsize=9)
    if vals:
        plots.lodo_rmse_bars(vals, ax[1])
    else:
        ax[1].text(0.5, 0.5, "fewer than 3 designs: no LODO", ha="center")
        ax[1].set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _dump(summary: dict[str, Any], out: str) -> None:
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)


def open_study(out: str, *, freqs: Sequence[float] = (2.0, 2.5, 3.0, 3.5, 4.0), n_designs: int = 16,
               n_fine: int = 4, coarse_res: float = 0.25, fine_res: float = 0.2, al_rounds: int = 2,
               al_per_round: int = 2, budget: int = 800, timeout: float = 3600.0, seed: int = 0,
               target_dbi: float = 10.0, ranges: Mapping[str, tuple[float, float]] | None = None,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    """The open-model study (needs EMerge). Writes tables, figures and ``summary.json`` to ``out``."""
    os.makedirs(out, exist_ok=True)
    freqs = [float(f) for f in freqs]
    summary: dict[str, Any] = {"limits": limits_report(target_dbi, min(freqs), max(freqs), len(freqs))}
    try:
        import emerge  # noqa: F401
    except Exception as e:  # noqa: BLE001 - EMerge is optional
        summary["emerge"] = f"not available ({type(e).__name__}: {e})"
        log(summary["emerge"] + ": only the limits were computed")
        _dump(summary, out)
        return summary

    t0 = time.perf_counter()
    designs, info = dz.doe(n_designs, ranges, method="lhs", seed=seed)
    summary["doe"] = info | {"n": len(designs)}
    log(f"DOE: {len(designs)} buildable designs from {info['drawn']} drawn ({info['invalid']} unbuildable)")
    sim = runner.ava_simulator(os.path.join(out, "runs"), timeout=timeout, log=log)
    tags = [f"d{i:02d}" for i in range(len(designs))]
    coarse = sim.run([d | {"res": coarse_res} for d in designs], freqs, tags=tags)
    fine = sim.run([d | {"res": fine_res} for d in designs[:n_fine]], freqs, tags=tags[:n_fine])
    runner.write_table(coarse, os.path.join(out, "results_coarse.csv"))
    runner.write_table(fine, os.path.join(out, "results_fine.csv"))
    summary["runs_doe"] = sim.summary()
    summary["checks_coarse"] = check_summary(coarse)
    summary["checks_fine"] = check_summary(fine)
    summary["multi_fidelity"] = {t: dz.multi_fidelity(coarse, fine, t) for t in TARGETS}
    vals = validate_targets(coarse)
    summary["lodo"] = {k: v.to_dict() for k, v in vals.items()}
    log("checks (coarse): " + json.dumps(summary["checks_coarse"]))
    log("LODO RMSE: " + json.dumps({k: v.rmse for k, v in vals.items()}))

    models = {t: sg.GPModel().fit(coarse, t) for t in ("Greal_bore_dBi", "S11_dB")}
    opt_ranges = dz.training_box(designs, ranges=ranges)
    opt = dz.optimize(models, freqs, ranges=opt_ranges, fixed={"res": coarse_res}, budget=budget, seed=seed)
    summary["optimization"] = opt.to_dict()
    log(f"NSGA-II: {opt.n_evals} surrogate evaluations in {opt.wall_s:.1f} s, {len(opt.designs)} on the front")
    if opt.designs:
        best = opt.designs[0]
        lodo_g = vals["Greal_bore_dBi"].rmse.get("gp") if "Greal_bore_dBi" in vals else None
        card = dz.prediction_card(models["Greal_bore_dBi"], best, freqs[0], coarse, lodo_rmse=lodo_g)
        ver = sim.run([best], freqs, tags=["best"])
        pred = {f"{f:g}": [float(v[0]) for v in models["Greal_bore_dBi"].predict([best], f)] for f in freqs}
        summary["best"] = {"design": best, "card": card.to_dict(), "predicted_Greal_mean_std": pred,
                           "simulated": [{k: r[k] for k in runner.RESULT_COLUMNS if k in r} for r in ver.rows],
                           "checks": check_summary(ver)}
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.5, 6))
        plots.layout_plot(layout(AVAParams.from_dict({k: v for k, v in best.items()
                                                      if k in AVAParams.__dataclass_fields__})), ax)
        ax.set_title("best predicted design", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "best_layout.png"), dpi=110)
        plt.close(fig)

    def simulate(ds, r):
        return sim.run(ds, freqs, tags=[f"al{r}_{i}" for i in range(len(ds))])

    table, hist = dz.active_learning(simulate, coarse, freqs, rounds=al_rounds, per_round=al_per_round,
                                     ranges=ranges, fixed={"res": coarse_res}, seed=seed)
    summary["active_learning"] = [h.__dict__ for h in hist]
    runner.write_table(table, os.path.join(out, "results_all.csv"))
    study_figure(table, vals, os.path.join(out, "directivity_and_lodo.png"), title="open AVA (EMerge)",
                 target_dbi=target_dbi)
    summary["runs_total"] = sim.summary()
    summary["wall_s"] = round(time.perf_counter() - t0, 1)
    _dump(summary, out)
    return summary


def package_study(path: str, out: str, *, rerun: str | None = None, rerun_freqs: int = 1,
                  interpolated: bool = False, timeout: float = 3600.0, target_dbi: float = 10.0,
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    """Reproduce an external package: checks of every row, limits, LODO, figures, and a re-run of design
    ``rerun`` (default: the first; ``"none"`` to skip) at its first ``rerun_freqs`` frequencies."""
    os.makedirs(out, exist_ok=True)
    pkg = load_package(path)
    t = pkg.table
    fs = [float(f) for f in t.freqs]
    summary: dict[str, Any] = {"package": pkg.describe(),
                               "limits": limits_report(target_dbi, min(fs), max(fs), len(fs))}
    summary["checks"] = check_summary(t, n_solved=len(fs), interpolated=interpolated)
    vals = validate_targets(t)
    summary["lodo"] = {k: v.to_dict() for k, v in vals.items()}
    study_figure(t, vals, os.path.join(out, "directivity_and_lodo.png"), title="package", target_dbi=target_dbi)
    if rerun != "none" and pkg.script:
        tag = rerun or t.tags[0]
        freqs = [float(f) for f in t.where(lambda r: str(r["tag"]) == tag).col("f_GHz")][:rerun_freqs]
        sim = pkg.simulator(os.path.join(out, "rerun"), timeout=timeout, log=log)
        new = sim.run([t.design(tag)], freqs, tags=[tag])
        diff = []
        for r in new.rows:
            old = next(o for o in t.rows if str(o["tag"]) == tag and abs(float(o["f_GHz"]) - float(r["f_GHz"])) < 1e-6)
            diff.append({"f_GHz": float(r["f_GHz"]),
                         **{f"delta_{c}": float(r[c]) - float(old[c]) for c in runner.RESULT_COLUMNS
                            if c != "f_GHz" and c in r and c in old}})
        summary["rerun"] = {"tag": tag, "runs": [x.to_dict() for x in sim.records], "rerun_minus_table": diff}
    _dump(summary, out)
    return summary
