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
``traced_peak``, ``torch_peak``, ``cuda_peak``, ``process_peak``, ``rss_peak``.

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
from .torch_ops import pinn_operation_costs, profile_ops

__all__ = [
    "Budget", "BudgetExceeded", "CostLedger", "CostRecord", "Regression", "ScalingFit", "ScalingStudy",
    "expected_exponent", "measure", "scaling_study", "pinn_operation_costs", "profile_ops",
    "PeakMemory", "cuda_peak", "measure_peak", "process_peak", "rss_peak", "torch_peak", "traced_peak",
    "AmortizedCost", "BudgetRun", "ComputeBudget", "SearchResult", "SearchTrial", "cosine_lr", "count_flops",
    "equal_budget_search", "hardware_record", "relative_l2", "set_lr", "time_per_query", "train_within_budget",
]
