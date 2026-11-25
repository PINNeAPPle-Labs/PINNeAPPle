"""The protocol, the three perturbations, and the results table (state error, latency, alarm).

The protocol
------------
One plant (:mod:`~.plant`) and one twin (:mod:`~.twin`) per case, run over a five-hour operating
history of slow operator moves inside the normal operating box. Every case is a controlled
experiment: the truth is integrated by the reference at ``rtol = 1e-10``, the twin sees only the
noisy readings and the four instrumented inputs, and both are compared at every sample.

Three perturbation families, the ones issue #179 asks for:

``noise``
    **Sensor noise.** The same healthy operating history at three instrument classes: a class A
    RTD with an in-line analyser (0.2 K, 10 mol/m3), a degraded installation (1.0 K, 40 mol/m3)
    and a bad one (3.0 K, 120 mol/m3). The plant never approaches the alarm: everything the twin
    does here is either tracking or a false alarm.

``cooling``
    **Loss of cooling capacity.** A cooling tube is isolated at 1.5 h (a valve closes over two
    minutes, the header keeps feeding the other four), the impeller trips (the drive coasts down,
    and with it the tank-side film coefficient falls to natural convection), or both. This is the
    perturbation the third-party post is about: agitation is a *cooling* variable.

``margin``
    **The safety margin against the temperature alarm.** The coolant inlet ramps over the run,
    as a fouled cooling tower or a hot afternoon does, so the stationary temperature walks
    towards the alarm setpoint at 56 °C from 12, 8 and 3 K of margin, and in one case past it.
    The twin's own margin report — ``t_trip`` minus the highest temperature its forecast reaches
    — is scored against the margin the plant actually had left.

The three columns
-----------------
``state error``
    the twin's error against the reference, in kelvin (bulk temperature) and in conversion
    points, with the two things it has to beat next to it: the raw instrument (the uncertainty
    the operator has without a twin) and the same lumped model with no assimilation at all (what
    the twin would carry if it only believed its own equations).
``latency``
    the wall time of one twin sample, filter plus the forecast that decides the alarm, and the
    fraction of the sample period it takes. Measured in-process with :func:`time.perf_counter`
    on the machine that ran the benchmark, so it is this machine's number and is quoted as such.
``alarm detection``
    per case: did the truth cross the alarm setpoint, did the twin say so, how long before (a
    positive lead is early), where were the false alarms, and what margin the twin reported.

Nothing here is a published number. Every figure in the table is measured by this run, and the
one thing that is *not* measured — the two-dimensional field with the immersed-boundary impeller
in the post — is named in :func:`not_measurable` with the reason.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .plant import (
    AUTHOR,
    NOISE_LEVELS,
    NORMAL_BOX,
    Faults,
    Reactor,
    Scenario,
    coolant_for_margin,
    inputs_of,
    run_reference,
    thermo_from_pair,
    unagitated_reading,
)
from .twin import run_twin

__all__ = ["FAMILIES", "make_cases", "run_benchmark", "author_comparison", "reference_trust",
           "not_measurable", "results_table", "case_table"]

FAMILIES = ("baseline", "noise", "cooling", "margin")

#: the run every case shares: five hours at a 30 s sample period, the fault at 1.5 h
HOURS = 5.0
DT_S = 30.0
ONSET_S = 5400.0

#: what the post did not report and this case does not attempt: the 2-D field itself
NOT_MEASURABLE = [
    {"item": "the 2-D field with an immersed-boundary impeller",
     "why": "the post's case is a two-dimensional Python/Numba CFD model; running one here would cost hours per "
            "operating point and would not be the same code, so it is not run and no number from it is quoted as "
            "if it had been. What is run is the well-mixed reduction in plant.py, and the difference it makes is "
            "the reason the unagitated point cannot be a bulk temperature."},
    {"item": "a spatial hotspot",
     "why": "a 0-D model carries one temperature for the whole tank. The post's 75 °C hotspot is a local maximum in "
            "an unmixed zone; here the bulk is what there is, and the numbers say so (59 °C with the impeller "
            "stopped, not 75 °C)."},
    {"item": "a conversion limited by mixing",
     "why": "the well-mixed assumption removes by construction the mechanism that would stall conversion at a low "
            "value while the bulk is hot. If the post's 20 % is a segregation or bypass limit, no 0-D model can "
            "reproduce it, and the case does not pretend to."},
    {"item": "the post's own numbers as an independent measurement",
     "why": "the reaction thermodynamics is derived from the two numbers themselves (thermo_from_pair), so "
            "reproducing the agitated operating point is arithmetic, not evidence."},
]


def make_cases(seed: int = 0, *, hours: float = HOURS, dt_s: float = DT_S, onset_s: float = ONSET_S) -> list[Scenario]:
    """The protocol as a list of controlled cases. Seeded, so the history is the same every run."""
    base = Reactor()
    steady = inputs_of(base)
    cases = [
        Scenario("baseline_a", "baseline", 1001 + seed, hours=hours, dt_s=dt_s, onset_s=onset_s, noise="typical"),
        Scenario("baseline_b", "baseline", 1002 + seed, hours=hours, dt_s=dt_s, onset_s=onset_s, noise="typical"),
    ]
    for i, (name, level) in enumerate((("noise_tight", "tight"), ("noise_typical", "typical"),
                                       ("noise_poor", "poor"))):
        cases.append(Scenario(name, "noise", 1100 + seed + i, hours=hours, dt_s=dt_s, onset_s=onset_s, noise=level))
    cases += [
        Scenario("tube_isolated", "cooling", 1201 + seed, hours=hours, dt_s=dt_s, onset_s=onset_s,
                 faults=Faults(tubes_lost=(2,)),
                 note="tube 3, the largest of the five, is valved off"),
        Scenario("impeller_trip", "cooling", 1202 + seed, hours=hours, dt_s=dt_s, onset_s=onset_s,
                 faults=Faults(impeller_tripped=True),
                 note="the drive coasts down: the film coefficient falls to natural convection"),
        Scenario("tube_and_trip", "cooling", 1203 + seed, hours=hours, dt_s=dt_s, onset_s=onset_s,
                 faults=Faults(tubes_lost=(2, 4), impeller_tripped=True),
                 note="the tube failure and the trip together"),
    ]
    for name, start, end in (("margin_wide", 12.0, 8.0), ("margin_close", 8.0, 2.0), ("margin_cross", 5.0, -3.0)):
        cases.append(Scenario(name, "margin", 1300 + seed + int(abs(start)),
                              hours=hours, dt_s=dt_s, onset_s=onset_s,
                              coolant_start_c=coolant_for_margin(base, steady, start),
                              coolant_end_c=coolant_for_margin(base, steady, end),
                              margin_k=(start, end),
                              note=f"coolant ramps so the stationary margin to the alarm goes {start:g} -> {end:g} K"))
    return cases


# ----------------------------------------------------------------------------- the reference's own checks
def reference_trust(reactor: Reactor | None = None, *, rtol: float = 1e-12) -> dict[str, Any]:
    """Why the reference is the reference: three checks, all of them computed here.

    *Two routes to the stationary point.* :meth:`Reactor.steady_state` solves the stationary
    equations algebraically; :func:`~.plant.run_reference` marches the transient to the same
    point. They agree to a few microkelvin, which is the only statement that the algebra and the
    dynamics are the same model.
    *Tolerance convergence.* The same controlled run integrated at ``rtol = 1e-10`` and at
    ``rtol = 1e-12`` differs by less than a microkelvin over the whole history.
    *The first law.* The heat the reaction released, integrated over the run, balances the
    sensible heat plus what the coolant and the wall took out, to the trapezoidal rule's accuracy.
    """
    R = reactor or Reactor()
    steady = inputs_of(R)
    algebraic = R.steady_state(steady)
    # the same point, reached by marching the transient with the inputs held
    sc = Scenario("march", "baseline", 77, hours=16.0, dt_s=120.0, onset_s=1e9, hold=True,
                  charge_offset_c=-15.0)
    marched = run_reference(R, sc)
    t_marched = float(marched.temperature_c[-1])
    # tolerance convergence
    fine = Scenario("tol", "baseline", 78, hours=2.0, dt_s=60.0, onset_s=1e9, hold=True,
                    charge_offset_c=8.0)
    a = run_reference(R, fine, rtol=1e-10)
    b = run_reference(R, fine, rtol=rtol)
    # the first law on a case with a fault in it
    case = Scenario("balance", "cooling", 79, hours=2.0, dt_s=30.0, onset_s=1800.0,
                    faults=Faults(tubes_lost=(1,), impeller_tripped=True))
    run = run_reference(R, case)
    eb = run.energy_balance()
    sensible = R.rho * R.cp * R.volume_m3 * float(run.y_true[-1, 2] - run.y_true[0, 2])
    lhs = sensible + eb["cooling_j"] + eb["wall_j"] - eb["feed_j"]
    return {
        "steady_algebraic_c": algebraic["t_c"],
        "steady_marched_c": t_marched,
        "steady_agreement_k": abs(algebraic["t_c"] - t_marched),
        "steady_roots": int(algebraic["n_roots"]),
        "tolerance_difference_k": float(np.abs(a.y_true[:, 2] - b.y_true[:, 2]).max()),
        "rtol": rtol,
        "energy_relative_residual": float(abs(eb["generated_j"] - lhs) / abs(eb["generated_j"])),
        "energy_generated_mj": eb["generated_j"] / 1e6,
        "march_hours": sc.hours,
        "balance_case": case.name,
    }


# ----------------------------------------------------------------------------- the author's numbers
def author_comparison(reactor: Reactor | None = None) -> dict[str, Any]:
    """Our model against the post's two operating points, measured, with what the pair implies.

    The agitated point is where the case is defined to operate (the sizing puts it there), so its
    agreement is arithmetic. The **unagitated** point is not: the model is asked what the same
    tank, the same feed and the same five tubes do with the impeller stopped, and the answer is
    compared with the post's 75 °C and 20 %.
    """
    R = reactor or Reactor()
    inputs = inputs_of(R)
    agitated = R.steady_state(inputs)
    stopped = unagitated_reading(R, inputs)
    return {
        "author": AUTHOR,
        "thermo": {"delta_h_j_mol": R.delta_h, "delta_s_j_mol_k": R.delta_s,
                   "from_pair": thermo_from_pair(AUTHOR["agitated"]["temperature_c"],
                                                 AUTHOR["agitated"]["conversion"],
                                                 AUTHOR["unagitated"]["temperature_c"],
                                                 AUTHOR["unagitated"]["conversion"])},
        "agitated": {"our_c": agitated["t_c"], "our_conversion": agitated["conversion"],
                     "author_c": AUTHOR["agitated"]["temperature_c"],
                     "author_conversion": AUTHOR["agitated"]["conversion"],
                     "delta_c": agitated["t_c"] - AUTHOR["agitated"]["temperature_c"],
                     "delta_conversion": agitated["conversion"] - AUTHOR["agitated"]["conversion"],
                     "ua_w_k": stopped["ua_w_k_agitated"]},
        "unagitated": {"our_c": stopped["t_c"], "our_conversion": stopped["conversion"],
                       "author_c": AUTHOR["unagitated"]["temperature_c"],
                       "author_conversion": AUTHOR["unagitated"]["conversion"],
                       "delta_c": stopped["t_c"] - AUTHOR["unagitated"]["temperature_c"],
                       "delta_conversion": stopped["conversion"] - AUTHOR["unagitated"]["conversion"],
                       "ua_w_k": stopped["ua_w_k_stopped"]},
        "what_the_pair_would_take": {"feed_c": stopped["feed_for_75c_c"],
                                     "conversion_at_75c": stopped["conversion_at_75c"]},
    }


def not_measurable() -> list[dict[str, str]]:
    """What this case does not measure, and why, in the house's own format."""
    return [dict(item) for item in NOT_MEASURABLE]


# ----------------------------------------------------------------------------- the benchmark
def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One family's row of the results table: medians over its cases and the alarm arithmetic."""
    def med(key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.median(values)) if values else None

    crossed = [r for r in rows if r["truth_alarmed"]]
    detected = [r for r in crossed if r["detected"]]
    leads = [r["lead_s"] for r in detected if r["lead_s"] is not None]
    return {
        "cases": len(rows),
        "temperature_error_rms_k": med("temperature_error_rms_k"),
        "temperature_error_max_k": max(r["temperature_error_max_k"] for r in rows),
        "conversion_error_rms_pp": med("conversion_error_rms_pp"),
        "open_loop_temperature_error_rms_k": med("open_loop_temperature_error_rms_k"),
        "sensor_temperature_error_rms_k": med("sensor_temperature_error_rms_k"),
        "phi_error_end": med("phi_error_end"),
        "latency_median_ms": med("latency_median_ms"),
        "latency_p95_ms": max(r["latency_p95_ms"] for r in rows),
        "share_of_the_sample_period": med("share_of_the_sample_period"),
        "cases_that_crossed_the_alarm": len(crossed),
        "detected": len(detected),
        "missed": sum(r["missed"] for r in rows),
        "false_alarms": sum(r["false_alarm"] for r in rows),
        "lead_s_median": float(np.median(leads)) if leads else None,
        "lead_s_min": float(min(leads)) if leads else None,
        "margin_reported_end_k": med("margin_reported_end_k"),
        "margin_truth_worst_k": med("margin_truth_worst_k"),
    }


def run_benchmark(*, cases: list[Scenario] | None = None, reactor: Reactor | None = None,
                  seed: int = 0, log=None) -> dict[str, Any]:
    """Run the protocol and return the measured table plus the evidence around it."""
    import time as _time

    R = reactor or Reactor()
    scs = cases if cases is not None else make_cases(seed)
    t0 = _time.perf_counter()
    rows: list[dict[str, Any]] = []
    runs = {}
    for sc in scs:
        if log:
            log(f"{sc.name}: {sc.kind} ({sc.noise})")
        run = run_twin(R, sc)
        row = run.summary()
        row["phi_error_end"] = row["phi_end"] - row["phi_truth_end"]
        row["share_of_the_sample_period"] = (row["latency_median_ms"] / 1e3) / row["sample_period_s"] * 100.0
        row["note"] = sc.note
        row["truth"] = sc.truth()
        rows.append(row)
        runs[sc.name] = run
    families = {k: _summarise([r for r in rows if r["kind"] == k]) for k in FAMILIES if any(r["kind"] == k for r in rows)}
    total = _summarise(rows)
    seconds = _time.perf_counter() - t0
    modelled_s = sum(float(run.t[-1] - run.t[0]) for run in runs.values())
    twin_s = sum(float(run.filter_seconds.sum() + run.forecast_seconds.sum()) for run in runs.values())
    return {
        "rows": rows, "runs": runs, "families": families, "overall": total,
        "protocol": {"cases": len(scs), "families": list(families),
                     "sample_period_s": float(np.median([c.dt_s for c in scs])),
                     "hours": float(np.median([c.hours for c in scs])), "onset_s": ONSET_S, "seed": seed,
                     "alarm_c": R.t_alarm_c, "trip_c": R.t_trip_c,
                     "noise_levels": {k: list(v) for k, v in NOISE_LEVELS.items()},
                     "normal_box": {k: list(v) for k, v in NORMAL_BOX.items()}},
        "seconds": {"total": seconds, "modelled": modelled_s, "twin": twin_s,
                    "real_time_factor": (twin_s / modelled_s) if modelled_s else None},
        "reference_trust": reference_trust(R),
        "author_comparison": author_comparison(R),
        "not_measurable": not_measurable(),
    }


# ----------------------------------------------------------------------------- the table
def results_table(bench: dict[str, Any]) -> str:
    """The acceptance's results table as markdown, straight from the measured rows."""
    lines = [
        "| family | cases | state error: T rms [K] | conv. rms [pp] | open loop [K] | sensor [K] | "
        "latency [ms/sample] | % of the 30 s cycle | alarm: crossed / detected / false / missed | "
        "median lead [min] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = [k for k in FAMILIES if k in bench["families"]]
    for key in order + ["overall"]:
        f = bench["families"].get(key) if key != "overall" else bench["overall"]
        if not f:
            continue
        lead = "-" if f["lead_s_median"] is None else f"{f['lead_s_median'] / 60:.0f}"
        lines.append(
            f"| {key} | {f['cases']} | {f['temperature_error_rms_k']:.2f} | {f['conversion_error_rms_pp']:.2f} | "
            f"{f['open_loop_temperature_error_rms_k']:.2f} | {f['sensor_temperature_error_rms_k']:.2f} | "
            f"{f['latency_median_ms']:.1f} | {f['share_of_the_sample_period']:.3f} % | "
            f"{f['cases_that_crossed_the_alarm']} / {f['detected']} / {f['false_alarms']} / {f['missed']} | {lead} |")
    return "\n".join(lines)


def case_table(bench: dict[str, Any]) -> str:
    """The per-case table: every row of the protocol with its three columns."""
    lines = ["| case | perturbation | T error [K] | conv. error [pp] | phi (twin / truth) | latency [ms] | "
             "truth alarm at [min] | twin alarm at [min] | lead [min] | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in bench["rows"]:
        truth_at = "-" if r["truth_alarm_at_s"] is None else f"{r['truth_alarm_at_s'] / 60:.0f}"
        twin_at = "-" if r["twin_alarm_at_s"] is None else f"{r['twin_alarm_at_s'] / 60:.0f}"
        lead = "-" if r["lead_s"] is None else f"{r['lead_s'] / 60:+.0f}"
        if r["false_alarm"]:
            verdict = "false alarm"
        elif r["missed"]:
            verdict = "MISSED"
        elif r["detected"]:
            verdict = "detected"
        else:
            verdict = "quiet (the plant never crossed)"
        lines.append(f"| {r['name']} | {r['kind']} / {r['noise']} | {r['temperature_error_rms_k']:.2f} | "
                     f"{r['conversion_error_rms_pp']:.2f} | {r['phi_end']:.3f} / {r['phi_truth_end']:.3f} | "
                     f"{r['latency_median_ms']:.1f} | {truth_at} | {twin_at} | {lead} | {verdict} |")
    return "\n".join(lines)
