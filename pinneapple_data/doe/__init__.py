"""Design of experiments for simulation campaigns: designs, coverage, redundancy, priorities, campaign plans.

Built on :mod:`pinneapple_data.contracts` (a ``ParameterSpace`` in, a ``StudyResult`` and a ``DatasetSpec``
skeleton out) and ``scipy.stats.qmc``. The case files per solver (OpenFOAM, CalculiX, Python) are written by
:mod:`pinneapple_simulation.external_solvers.campaign_cases`.
"""
from .campaign import CampaignPlan, plan_campaign
from .coverage import coverage, prioritize, redundancy
from .designs import (
    METHODS,
    Design,
    design,
    fractional_factorial_unit,
    greedy_maximin,
    min_distance,
)

__all__ = ["METHODS", "Design", "design", "fractional_factorial_unit", "greedy_maximin", "min_distance", "coverage",
           "redundancy", "prioritize", "CampaignPlan", "plan_campaign"]
