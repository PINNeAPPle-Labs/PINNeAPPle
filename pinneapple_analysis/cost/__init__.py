"""Computational-cost validation per operation (time, memory, FLOPs, scaling, budgets, regressions).

>>> from pinneapple_analysis.cost import measure, scaling_study, Budget
>>> rec = measure(lambda: sum(range(10**6)), repeats=5)
>>> study = scaling_study(lambda n: (lambda a=list(range(n)): sorted(a)), [10**4, 10**5, 10**6],
...                       declared="n log n")
>>> print(study.summary())
>>> Budget(max_time_s=2.0, max_time_exponent=1.3).validate(study)

Equal-budget method comparisons (PINN vs neural operators vs graph networks): ``equal_budget`` --
``ComputeBudget``, ``train_within_budget``, ``count_flops``, ``equal_budget_search``, ``AmortizedCost``.

Peak memory, measured per allocator (issue #54): ``memory`` -- ``measure_peak``, ``PeakMemory``,
``traced_peak``, ``torch_peak``, ``cuda_peak``, ``process_peak``, ``rss_peak``; ``rusage_unit_bytes`` and
``to_bytes`` say how the kernel's ``ru_maxrss`` (KiB on Linux, bytes on macOS) becomes a number of bytes.

From a measurement to the machine to ask for (issue #131): ``resources`` -- ``classify`` (the resource kind, from
the measured work rate and footprint), ``recommend_cores`` (cores from a measured speedup curve),
``plan``/``JobProfile`` (cores, memory per run and the machine it was measured on) and ``JobProfileTable`` (the
table generated from a ``CostLedger``, written with ``measured_on``).

See ``profiler`` for exactly what is and is not measured.
"""
from .equal_budget import (
    AmortizedCost,
    BudgetRun,
    ComputeBudget,
    SearchResult,
    SearchTrial,
    cosine_lr,
    count_flops,
    equal_budget_search,
    hardware_record,
    relative_l2,
    set_lr,
    time_per_query,
    train_within_budget,
)
from .memory import (
    PeakMemory,
    cuda_peak,
    measure_peak,
    process_peak,
    rss_peak,
    rusage_unit_bytes,
    to_bytes,
    torch_peak,
    traced_peak,
)
from .profiler import (
    Budget,
    BudgetExceeded,
    CostLedger,
    CostRecord,
    Regression,
    ScalingFit,
    ScalingStudy,
    expected_exponent,
    measure,
    scaling_study,
)
from .resources import (
    RESOURCE_KINDS,
    JobProfile,
    JobProfileTable,
    MachineRecord,
    classify,
    plan,
    recommend_cores,
)
from .torch_ops import pinn_operation_costs, profile_ops

__all__ = [
    "Budget", "BudgetExceeded", "CostLedger", "CostRecord", "Regression", "ScalingFit", "ScalingStudy",
    "expected_exponent", "measure", "scaling_study", "pinn_operation_costs", "profile_ops",
    "PeakMemory", "cuda_peak", "measure_peak", "process_peak", "rss_peak", "rusage_unit_bytes", "to_bytes",
    "torch_peak", "traced_peak",
    "AmortizedCost", "BudgetRun", "ComputeBudget", "SearchResult", "SearchTrial", "cosine_lr", "count_flops",
    "equal_budget_search", "hardware_record", "relative_l2", "set_lr", "time_per_query", "train_within_budget",
    "RESOURCE_KINDS", "JobProfile", "JobProfileTable", "MachineRecord", "classify", "plan", "recommend_cores",
]
