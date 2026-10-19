"""Fault injection: measure how often each validation check catches known model and data defects (#697).

A "pass" from a check carries no stated power until the check has been run against defects of known size. This
package injects a library of controlled faults into correct cases of problems with a verified reference solver and
records every check's response::

    from pinneapple_analysis.validation.fault_injection import run_fault_injection
    rep = run_fault_injection("darcy_2d", n_cases=50)
    print(rep.to_markdown())            # detection matrix, false alarms, minimal detectable magnitudes
    assert rep.passed                   # every fault caught at its documented magnitude, <= 1 % false alarms

Modules: :mod:`.problems` (Darcy 2-D and Burgers 1-D cases from ``operator_benchmarks``), :mod:`.checks` (the
checks under test, as ``physics_audit.monitor.ValidityCheck``), :mod:`.faults` (the fault library), :mod:`.matrix`
(the detection matrix and its report). The adversarial arena (:mod:`pinneapple_analysis.validation.arena`) is
built on these.
"""
from __future__ import annotations

from .checks import check_library, dataset_checks, default_checks, flags, score
from .faults import DATA_FAULTS, MODEL_FAULTS, Fault, data_faults, get_fault, model_faults
from .matrix import DetectionMatrix, FaultInjectionReport, auc, run_fault_injection
from .problems import (
           PROBLEMS,
           BenchmarkProblem,
           BurgersProblem,
           Case,
           DarcyProblem,
           Dataset,
           get_problem,
)

__all__ = ["run_fault_injection", "DetectionMatrix", "FaultInjectionReport", "auc", "Fault", "MODEL_FAULTS",
           "DATA_FAULTS", "model_faults", "data_faults", "get_fault", "check_library", "default_checks",
           "dataset_checks", "score", "flags", "Case", "Dataset", "BenchmarkProblem", "DarcyProblem",
           "BurgersProblem", "PROBLEMS", "get_problem"]
