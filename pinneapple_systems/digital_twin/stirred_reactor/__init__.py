"""Stirred exothermic reactor as a digital-twin benchmark case (issue #179).

A continuous stirred tank with a reversible exothermic reaction, a three-blade impeller and five
cooling tubes, reduced from the third-party two-dimensional model the issue cites to a well-mixed
model this machine runs in seconds. Three perturbations — sensor noise, a failed cooling tube (and
a stopped impeller, which is the same loss of cooling by another route) and the safety margin
against a temperature alarm — and a results table with the three columns the acceptance asks for:
**state error**, **latency** and **alarm detection**.

    from pinneapple_systems.digital_twin.stirred_reactor import run_benchmark, results_table
    bench = run_benchmark()
    print(results_table(bench))

Modules: :mod:`.plant` (the reactor and its reference simulator), :mod:`.twin` (the lumped fast
model, the Kalman filter and the alarm), :mod:`.benchmark` (the protocol, the three perturbation
families and the tables), :mod:`.report` (figures and the written report).
"""
from .benchmark import (
    FAMILIES,
    NOT_MEASURABLE,
    author_comparison,
    case_table,
    make_cases,
    not_measurable,
    reference_trust,
    results_table,
    run_benchmark,
)
from .plant import (
    AUTHOR,
    NOISE_LEVELS,
    NORMAL_BOX,
    SENSORS,
    UNITS,
    Faults,
    Inputs,
    Reactor,
    ReferenceRun,
    Scenario,
    coolant_for_margin,
    inputs_of,
    operating_history,
    readings,
    run_reference,
    thermo_from_pair,
    unagitated_reading,
)
from .twin import LumpedModel, LumpedTwin, OpenLoop, TwinRun, alarm_scores, run_twin

__all__ = ["AUTHOR", "FAMILIES", "NOT_MEASURABLE", "NOISE_LEVELS", "NORMAL_BOX", "SENSORS", "UNITS", "Faults",
           "Inputs", "LumpedModel", "LumpedTwin", "OpenLoop", "Reactor", "ReferenceRun", "Scenario", "TwinRun",
           "alarm_scores", "author_comparison", "case_table", "coolant_for_margin", "inputs_of", "make_cases",
           "not_measurable", "operating_history", "readings", "reference_trust", "results_table", "run_benchmark",
           "run_reference", "run_twin", "thermo_from_pair", "unagitated_reading"]
