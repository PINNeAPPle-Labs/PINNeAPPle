"""Deliverables of the case: the results table, the figures and the written report.

Every number printed here comes from the benchmark's own measurement or from the reference's own
checks — except the two operating points of the third-party post, which are labelled as the
author's on the line where they appear and are never mixed into a measured column.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .plant import T0_K
from .twin import TwinRun

FAMILY_TITLE = {"baseline": "baseline (no fault)", "noise": "sensor noise", "cooling": "loss of cooling capacity",
                "margin": "safety margin to the alarm"}


def _save(fig, path: str) -> str:
    """Write the figure, creating its folder: the deliverables go under ``_out/figures``."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _arrays(run: TwinRun) -> dict[str, np.ndarray]:
    return {"t_min": run.t / 60.0, "truth_c": run.reference.temperature_c,
            "twin_c": run.estimate[:, 2] - T0_K, "measured_c": run.reference.readings[:, 0],
            "truth_x": run.reference.conversion,
            "twin_x": 1.0 - run.estimate[:, 0] / run.reference.reactor.c_feed,
            "open_c": run.open_loop[:, 2] - T0_K, "phi": run.estimate[:, 3],
            "phi_lo": run.estimate[:, 3] - 2.0 * run.standard_deviation[:, 3],
            "phi_hi": run.estimate[:, 3] + 2.0 * run.standard_deviation[:, 3],
            "forecast_peak_c": run.forecast_peak_c, "margin_k": run.forecast_margin_k}


def figure_case(run: TwinRun, path: str, *, title: str = "") -> str:
    """One case: the plant, the instrument, the twin, the forecast that decides the alarm, and phi."""
    import matplotlib.pyplot as plt

    a = _arrays(run)
    sc = run.reference.scenario
    R = run.reference.reactor
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.6), sharex=True, constrained_layout=True,
                           gridspec_kw={"height_ratios": [2.4, 1.6, 1.4]})
    ax[0].plot(a["t_min"], a["measured_c"], ".", ms=2, color="#9ca3af", label="RTD reading (one per sample)")
    ax[0].plot(a["t_min"], a["truth_c"], lw=1.6, color="#111827", label="reference (true bulk)")
    ax[0].plot(a["t_min"], a["twin_c"], lw=1.2, color="#0081be", label="twin estimate")
    ax[0].plot(a["t_min"], a["open_c"], lw=0.9, ls="--", color="#a23a00", label="lumped model, no assimilation")
    ax[0].axhline(R.t_alarm_c, color="#dc2626", lw=1.0, ls="-", label=f"alarm {R.t_alarm_c:g} °C")
    ax[0].axhline(R.t_trip_c, color="#7f1d1d", lw=1.0, ls=":", label=f"trip {R.t_trip_c:g} °C")
    alarm = run.alarm
    if alarm.any():
        ax[0].axvline(a["t_min"][alarm][0], color="#dc2626", lw=1.4, alpha=0.7)
        ax[0].annotate(f"twin alarms {a['t_min'][alarm][0]:.0f} min", (a["t_min"][alarm][0], R.t_alarm_c),
                       textcoords="offset points", xytext=(6, 8), fontsize=8, color="#dc2626")
    truth = run.reference.alarms
    if truth.any():
        ax[0].axvline(a["t_min"][truth][0], color="#111827", lw=1.0, ls="--", alpha=0.7)
        ax[0].annotate(f"plant crosses {a['t_min'][truth][0]:.0f} min", (a["t_min"][truth][0], R.t_trip_c),
                       textcoords="offset points", xytext=(6, -14), fontsize=8)
    if sc.onset_s is not None and sc.onset_s >= 0:
        ax[0].axvline(sc.onset_s / 60.0, color="#6b7280", lw=0.8, ls=":", label="perturbation starts")
    ax[0].set_ylabel("bulk temperature [°C]")
    ax[0].legend(fontsize=7, ncol=4, loc="upper left")
    ax[0].set_title(title or f"{sc.name}: {sc.kind}, sensor noise {sc.noise}. "
                              f"State error {np.sqrt(np.mean((a['twin_c'] - a['truth_c']) ** 2)):.2f} K rms",
                    fontsize=9)
    ax[1].plot(a["t_min"], a["truth_x"], lw=1.5, color="#111827", label="reference conversion")
    ax[1].plot(a["t_min"], a["twin_x"], lw=1.1, color="#0081be", label="twin conversion")
    ax[1].set_ylabel("conversion of A [-]")
    ax[1].legend(fontsize=7, loc="best")
    ax[2].plot(a["t_min"], a["phi"], lw=1.2, color="#059669", label="twin's estimated cooling fraction φ")
    ax[2].fill_between(a["t_min"], a["phi_lo"], a["phi_hi"], color="#059669", alpha=0.15, label="±2σ")
    true_phi = np.array([run.reference.effective_phi(i) for i in range(len(run.t))])
    ax[2].plot(a["t_min"], true_phi, lw=1.0, ls="--", color="#111827", label="true cooling fraction")
    ax[2].set_ylim(min(0.5, float(a["phi"].min()) - 0.05), 1.08)
    ax[2].set_ylabel("cooling fraction\n(twin / truth)")
    ax[2].set_xlabel("time [min]")
    ax[2].legend(fontsize=7, ncol=3, loc="lower left")
    return _save(fig, path)


def figure_author(comp: dict[str, Any], path: str) -> str:
    """The post's two operating points against what this model does with them, in the (T, x) plane."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ag, un = comp["agitated"], comp["unagitated"]
    ax.plot(ag["author_c"], 100 * ag["author_conversion"], "o", ms=11, mfc="none", mec="#111827", mew=1.6,
            label="the post: agitated 48 °C, 75 %")
    ax.plot(un["author_c"], 100 * un["author_conversion"], "s", ms=11, mfc="none", mec="#111827", mew=1.6,
            label="the post: unagitated 75 °C, 20 %")
    ax.plot(ag["our_c"], 100 * ag["our_conversion"], "o", ms=7, color="#0081be",
            label=f"this model, agitated: {ag['our_c']:.1f} °C, {100 * ag['our_conversion']:.1f} %")
    ax.plot(un["our_c"], 100 * un["our_conversion"], "s", ms=7, color="#a23a00",
            label=f"this model, impeller stopped: {un['our_c']:.1f} °C, {100 * un['our_conversion']:.1f} %")
    ax.set_xlabel("bulk temperature [°C]")
    ax.set_ylabel("conversion of A [%]")
    ax.set_title("The post's two points and the same tank, well mixed, with the impeller on and off", fontsize=9)
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(alpha=0.25)
    return _save(fig, path)


def figure_results(bench: dict[str, Any], path: str) -> str:
    """The results table as a figure: the three columns, family by family."""
    import matplotlib.pyplot as plt

    fams = [k for k in ("baseline", "noise", "cooling", "margin") if k in bench["families"]]
    if not fams:
        raise ValueError("no measured family to plot")
    labels = [FAMILY_TITLE.get(k, k) for k in fams]
    x = np.arange(len(fams))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    latency = [bench["families"][k]["latency_median_ms"] for k in fams]
    error = [bench["families"][k]["temperature_error_rms_k"] for k in fams]
    sensor = [bench["families"][k]["sensor_temperature_error_rms_k"] for k in fams]
    openloop = [bench["families"][k]["open_loop_temperature_error_rms_k"] for k in fams]
    w = 0.27

    ax[0].bar(x - w, sensor, w, color="#9ca3af", label="RTD alone")
    ax[0].bar(x, openloop, w, color="#a23a00", label="model, no assimilation")
    ax[0].bar(x + w, error, w, color="#0081be", label="twin")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("temperature error, rms [K]")
    ax[0].set_title("state error against the reference", fontsize=9)
    ax[0].set_xticks(x, labels, rotation=12, fontsize=8)
    ax[0].legend(fontsize=7)

    ax[1].bar(x, latency, 0.5, color="#059669")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("one twin sample [ms]")
    ax[1].set_title("latency (median; the 30 s cycle is the budget)", fontsize=9)
    ax[1].set_xticks(x, labels, rotation=12, fontsize=8)

    leads, names, colors = [], [], []
    for r in bench["rows"]:
        if r["lead_s"] is None:
            continue
        leads.append(r["lead_s"] / 60.0)
        names.append(r["name"])
        colors.append("#0081be" if r["lead_s"] > 0 else "#dc2626")
    if leads:
        ax[2].barh(np.arange(len(leads)), leads, color=colors)
        ax[2].axvline(0, color="k", lw=0.8)
        ax[2].set_yticks(np.arange(len(leads)), names, fontsize=8)
        ax[2].set_xlabel("lead of the twin's alarm over the plant [min]")
        ax[2].set_title("alarm detection (positive: early)", fontsize=9)
    else:
        ax[2].text(0.5, 0.5, "no case crossed the alarm", ha="center", va="center")
        ax[2].set_axis_off()
    return _save(fig, path)


def results_json(bench: dict[str, Any]) -> dict[str, Any]:
    """The machine-readable results: the tables, the protocol, the reference's checks, what is missing."""
    return {
        "schema": "pinneapple.stirred_reactor_benchmark/1",
        "protocol": bench["protocol"],
        "results": bench["families"],
        "overall": bench["overall"],
        "cases": [{k: v for k, v in r.items() if k != "truth"} for r in bench["rows"]],
        "timings": bench["seconds"],
        "reference_trust": bench["reference_trust"],
        "author_comparison": bench["author_comparison"],
        "not_measurable": bench["not_measurable"],
    }


def write_report(bench: dict[str, Any], out_dir: str, *, figures: list[str] | None = None) -> str:
    """The written report: the table, the reference's credentials, the comparison and the gaps."""
    from .benchmark import case_table, results_table

    os.makedirs(out_dir, exist_ok=True)
    trust, comp = bench["reference_trust"], bench["author_comparison"]
    p = bench["protocol"]
    gap_points = comp["unagitated"]["our_conversion"] - comp["unagitated"]["author_conversion"]
    gap_kelvin = comp["unagitated"]["author_c"] - comp["unagitated"]["our_c"]
    lines = [
        "# Stirred exothermic reactor: a digital-twin benchmark case",
        "",
        f"Protocol: {p['cases']} controlled cases in {len(p['families'])} perturbation families, "
        f"{p['hours']:g} h each at a {p['sample_period_s']:g} s sample period, alarm at {p['alarm_c']:g} °C and "
        f"interlock at {p['trip_c']:g} °C. Seed {p['seed']}.",
        "",
        "## The results table",
        "",
        results_table(bench),
        "",
        "Columns: **state error** is the twin's rms error against the reference, next to the instrument it had to "
        "beat (the RTD alone) and the same lumped model with no assimilation; **latency** is one twin sample, filter "
        "plus the forecast that decides the alarm, and its share of the 30 s cycle; **alarm detection** counts the "
        "cases where the plant crossed the alarm, the ones the twin caught, the false alarms and the misses, with "
        "the median lead.",
        "",
        "## Case by case",
        "",
        case_table(bench),
        "",
        "## What the reference is, and why it is trusted",
        "",
        "The reference is **not** the post's two-dimensional CFD. It is the well-mixed model of "
        "`pinneapple_systems/digital_twin/stirred_reactor/plant.py`, integrated with a stiff solver at "
        f"`rtol = {trust['rtol']:g}` and sampled where the twin is sampled. Three checks, run by this benchmark:",
        "",
        f"- **two routes to the stationary point**: the algebraic stationary solve gives "
        f"{trust['steady_algebraic_c']:.4f} °C; starting the transient 15 K away and marching {trust['march_hours']:g} h "
        f"gives {trust['steady_marched_c']:.4f} °C — {1e3 * trust['steady_agreement_k']:.2f} mK apart.",
        f"- **tolerance convergence**: the same controlled run at `rtol = 1e-10` and at `rtol = {trust['rtol']:g}` "
        f"differs by {trust['tolerance_difference_k']:.1e} K over the whole history.",
        f"- **the first law**: over a case with a tube isolated and the impeller stopped, the heat generated "
        f"({trust['energy_generated_mj']:.0f} MJ) closes against the sensible change plus what the coolant and the "
        f"wall removed to {trust['energy_relative_residual']:.1e} relative.",
        "",
        "## The post's two numbers, measured",
        "",
        f"The reaction thermodynamics is derived from the post's own pair — dH = "
        f"{comp['thermo']['delta_h_j_mol'] / 1e3:.1f} kJ/mol and dS = {comp['thermo']['delta_s_j_mol_k']:.0f} "
        f"J/(mol K) — so agreeing with the agitated point is arithmetic, not evidence. The impeller-off point is a "
        f"prediction of the same tank:",
        "",
        "| | agitated | impeller stopped |",
        "|---|---|---|",
        f"| the post | {comp['agitated']['author_c']:.0f} °C, {100 * comp['agitated']['author_conversion']:.0f} % | "
        f"{comp['unagitated']['author_c']:.0f} °C, {100 * comp['unagitated']['author_conversion']:.0f} % |",
        f"| this model | {comp['agitated']['our_c']:.1f} °C, {100 * comp['agitated']['our_conversion']:.1f} % | "
        f"{comp['unagitated']['our_c']:.1f} °C, {100 * comp['unagitated']['our_conversion']:.1f} % |",
        f"| cooling conductance | {comp['agitated']['ua_w_k']:.0f} W/K | {comp['unagitated']['ua_w_k']:.0f} W/K |",
        "",
        f"To sit at {comp['unagitated']['author_c']:.0f} °C with the impeller stopped, the same well-mixed tank needs "
        f"a feed at {comp['what_the_pair_would_take']['feed_c']:.1f} °C (its conversion there would be "
        f"{100 * comp['what_the_pair_would_take']['conversion_at_75c']:.1f} %), against the "
        f"{40:.0f} °C feed this case uses. Read as two energy-balance points of one feed, the pair is not reachable: "
        f"{100 * gap_points:.0f} points of conversion at {gap_kelvin:.0f} K hotter is a transport limitation, and a "
        "well-mixed model cannot carry one.",
        "",
        "## What is not measured, and why",
        "",
    ]
    for item in bench["not_measurable"]:
        lines.append(f"- **{item['item']}**: {item['why']}")
    sec = bench["seconds"]
    lines += [
        "",
        "## Cost",
        "",
        f"{p['cases']} cases took {sec['total']:.0f} s of wall clock on the machine that ran them; the twin's own "
        f"compute over {sec['modelled'] / 3600:.1f} h of modelled operation was {sec['twin']:.1f} s, a real-time "
        f"factor of {sec['real_time_factor']:.2e}. The twin is not faster than this 0-D reference — it does not have "
        "to be, it runs on the sample it is given — and no speed-up against the post's 2-D model is claimed, because "
        "that model is not run here.",
        "",
    ]
    if figures:
        lines += ["## Figures", ""]
        lines += [f"![{os.path.basename(f)}]({os.path.relpath(os.path.abspath(f), os.path.abspath(out_dir))})"
                  for f in figures]
        lines.append("")
    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def write_json(bench: dict[str, Any], path: str) -> str:
    """The results as JSON, with nothing in it that was not measured."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results_json(bench), fh, indent=1, default=str)
    return path
