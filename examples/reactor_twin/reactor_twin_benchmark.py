"""Stirred exothermic reactor: a digital-twin benchmark case (issue #179).

An exothermic reactor with a three-blade impeller and five cooling tubes, reduced from the
third-party two-dimensional model the issue cites to a well-mixed model this repository runs in
seconds, and put through the protocol the acceptance asks for: sensor noise, a failed cooling tube
and the safety margin against a temperature alarm, with a results table of **state error**,
**latency** and **alarm detection**.

Run: ``python examples/reactor_twin/reactor_twin_benchmark.py [--smoke] [--out DIR]``
Outputs in ``examples/reactor_twin/_out/``.
"""
from __future__ import annotations

import argparse
import os

import pinneapple_lab
from pinneapple_systems.digital_twin.stirred_reactor import (
    Reactor,
    author_comparison,
    case_table,
    inputs_of,
    make_cases,
    reference_trust,
    results_table,
    run_benchmark,
)
from pinneapple_systems.digital_twin.stirred_reactor.report import (
    figure_author,
    figure_case,
    figure_results,
    write_json,
    write_report,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    ap.add_argument("--smoke", action="store_true", help="a short, partial protocol: four cases, one hour")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with pinneapple_lab.experiment(
            "examples/reactor_twin", title="Stirred exothermic reactor: digital-twin benchmark",
            description="A well-mixed exothermic reactor with a three-blade impeller and five cooling tubes, reduced "
                        "from the two-dimensional case a third-party post describes, and the digital twin that "
                        "follows it through three perturbations (sensor noise, a failed cooling tube, the safety "
                        "margin against a temperature alarm), measured as state error, latency and alarm detection.",
            domain="digital twins", methods=["stirred-tank energy and mass balance", "Arrhenius kinetics",
                                             "van't Hoff equilibrium", "effectiveness-NTU cooling",
                                             "agitated-vessel film correlation", "extended Kalman filter",
                                             "predictive alarm"],
            modules=["pinneapple_systems.digital_twin.stirred_reactor"], out_dir=HERE, register=False,
            links={"guide": "docs/guides/stirred_reactor_twin.md",
                   "script": "examples/reactor_twin/reactor_twin_benchmark.py"}) as ex:

        trust = reference_trust()
        print("The reference: the well-mixed model of pinneapple_systems.digital_twin.stirred_reactor.plant")
        print(f"  two routes to the stationary point: algebraic {trust['steady_algebraic_c']:.4f} °C, "
              f"marched {trust['steady_marched_c']:.4f} °C ({1e3 * trust['steady_agreement_k']:.2f} mK apart)")
        print(f"  tolerance: rtol 1e-10 against 1e-12 differ by {trust['tolerance_difference_k']:.1e} K")
        print(f"  first law over a fault run: {trust['energy_generated_mj']:.0f} MJ generated, residual "
              f"{trust['energy_relative_residual']:.1e} relative")
        ex.log_metric("reference_stationary_agreement", 1e3 * trust["steady_agreement_k"], "mK",
                      "algebraic stationary solve against a 16 h marched transient")
        ex.log_metric("reference_tolerance_difference", trust["tolerance_difference_k"], "K",
                      "rtol 1e-10 against rtol 1e-12 over a two-hour controlled run")
        ex.log_metric("reference_energy_residual", trust["energy_relative_residual"], "relative",
                      "first law over a run with a tube isolated and the impeller stopped")
        ex.check("the reference's two routes agree", value=trust["steady_agreement_k"], max=1e-2, kind="verification",
                 label="algebraic stationary point vs marched transient [K]")
        ex.check("the reference's tolerance is converged", value=trust["tolerance_difference_k"], max=1e-5,
                 kind="verification", label="rtol 1e-10 vs 1e-12 [K]")
        ex.check("the reference closes the first law", value=trust["energy_relative_residual"], max=1e-2,
                 kind="verification", label="energy residual over a fault run [relative]")

        comp = author_comparison()
        ag, un = comp["agitated"], comp["unagitated"]
        print("\nThe post's two numbers, and this model (the post's figures are the author's, not verified here):")
        print(f"  agitated          the post: {ag['author_c']:.0f} °C, {100 * ag['author_conversion']:.0f} %   "
              f"this model: {ag['our_c']:.1f} °C, {100 * ag['our_conversion']:.1f} %")
        print(f"  impeller stopped  the post: {un['author_c']:.0f} °C, {100 * un['author_conversion']:.0f} %   "
              f"this model: {un['our_c']:.1f} °C, {100 * un['our_conversion']:.1f} %")
        print(f"  the post's thermodynamics, read off its two points: dH = {comp['thermo']['delta_h_j_mol'] / 1e3:.1f} "
              f"kJ/mol, dS = {comp['thermo']['delta_s_j_mol_k']:.0f} J/(mol K) -> reproducing the agitated point is "
              "arithmetic, not evidence")
        print(f"  to sit at {un['author_c']:.0f} °C with the impeller stopped, this tank needs a feed at "
              f"{comp['what_the_pair_would_take']['feed_c']:.1f} °C, against the {inputs_of(Reactor()).t_feed_c:.0f} °C "
              "feed used here")
        ex.log_metric("unagitated_temperature", un["our_c"], "°C",
                      "stationary bulk temperature with the impeller stopped, same feed and tubes")
        ex.log_metric("unagitated_conversion", un["our_conversion"], "fraction",
                      "stationary conversion with the impeller stopped")
        ex.log_metric("feed_needed_for_the_post_point", comp["what_the_pair_would_take"]["feed_c"], "°C",
                      "feed temperature a well-mixed tank needs to sit at the post's 75 °C with 20 % conversion")

        cases = None
        if a.smoke:
            cases = [c for c in make_cases(a.seed, hours=1.0, dt_s=60.0) if c.name in
                     ("baseline_a", "noise_poor", "tube_isolated", "margin_cross")]
        bench = run_benchmark(cases=cases, seed=a.seed, log=lambda s: None)

        print("\n" + results_table(bench))
        print("\n" + case_table(bench))
        overall = bench["overall"]
        fps = bench["families"].get("cooling")
        ex.log_metric("state_error_temperature_rms", overall["temperature_error_rms_k"], "K",
                      f"{overall['cases']} cases: twin against the reference at every sample")
        ex.log_metric("state_error_conversion_rms", overall["conversion_error_rms_pp"], "pp",
                      "twin conversion against the reference, percentage points")
        ex.log_metric("sensor_error_for_scale", overall["sensor_temperature_error_rms_k"], "K",
                      "the RTD alone on the same cases: what there is without a twin")
        ex.log_metric("open_loop_error_for_scale", fps["open_loop_temperature_error_rms_k"] if fps else 0.0, "K",
                      "the same lumped model with no assimilation, cooling family")
        ex.log_metric("latency_median", overall["latency_median_ms"], "ms",
                      "one twin sample: Kalman predict+update plus the forecast that decides the alarm")
        ex.log_metric("latency_share_of_the_cycle", overall["share_of_the_sample_period"], "%",
                      f"of the {bench['protocol']['sample_period_s']:g} s sample period")
        ex.log_metric("alarm_cases_that_crossed", overall["cases_that_crossed_the_alarm"], "cases",
                      "the plant reached the alarm setpoint")
        ex.log_metric("alarm_detected", overall["detected"], "cases", "and the twin said so")
        ex.log_metric("alarm_false_alarms", overall["false_alarms"], "cases", "alarms where the plant never crossed")
        ex.log_metric("alarm_missed", overall["missed"], "cases", "crossings the twin never reported")
        if overall["lead_s_median"] is not None:
            ex.log_metric("alarm_lead_median", overall["lead_s_median"] / 60.0, "min",
                          "how early the twin's alarm was, median over the detected crossings")
        ex.check("false alarms", passed=overall["false_alarms"] == 0, kind="physics",
                 detail=f"{overall['false_alarms']} false alarms in {overall['cases']} cases")
        ex.check("no crossing missed", passed=overall["missed"] == 0, kind="physics",
                 detail=f"{overall['detected']}/{overall['cases_that_crossed_the_alarm']} detected")
        ex.check("twin beats the instrument", value=overall["temperature_error_rms_k"],
                 max=overall["sensor_temperature_error_rms_k"], kind="physics",
                 label="state error against the RTD alone [K]")
        ex.check("one sample fits the cycle", value=overall["share_of_the_sample_period"], max=100.0, kind="physics",
                 label="twin compute as a share of the sample period [%]")

        figures = [figure_author(comp, os.path.join(a.out, "figures", "post_points_vs_model.png")),
                   figure_results(bench, os.path.join(a.out, "figures", "results.png"))]
        for name in ("impeller_trip", "margin_cross"):
            run = bench["runs"].get(name)
            if run is not None:
                figures.append(figure_case(run, os.path.join(a.out, "figures", f"{name}.png")))
        for f in figures:
            ex.add_file(f)
        report = write_report(bench, a.out, figures=figures)
        ex.add_file(report)
        ex.add_file(write_json(bench, os.path.join(a.out, "benchmark.json")))
        print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
