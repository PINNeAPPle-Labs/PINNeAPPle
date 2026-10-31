"""An end-to-end surrogate study on any reference simulator, one library call per stage.

=================  ===========================================================================================
``campaign``       DOE with space-filling metrics (``plan_doe``), resumable runs with a cost ledger
                   (``run_campaign``, ``load_records``, ``cost_table``)
``mesh_study``     grid convergence on a mesh family: Richardson, GCI, the mesh a tolerance allows
``dataset``        scalars + fields + provenance as a ``DatasetSpec`` with a held-out region (``StudyDataset``)
``benchmark``      surrogate families at equal data and budget, cross-conformal 90 % intervals, physics checks
                   on the predictions, a written selection rule (``benchmark``, ``Board``)
``extras``         Sobol indices, NSGA-II with an uncertainty constraint, uncertainty-driven acquisition
``twin``           a digital-twin page (sliders, 3-D field, band, trust warning, verify button)
``report``         the evidence report (HTML) of a study
=================  ===========================================================================================

The flagship case is the Ahmed body in OpenFOAM (:mod:`pinneapple_autolab.domains.ahmed`,
``examples/end_to_end/ahmed_body``). Inverse problems use
:mod:`pinneapple_analysis.inverse_problems.amortized` (MDN) and :mod:`.estimation` (least squares, Metropolis).
"""
from __future__ import annotations

import importlib as _importlib

__all__ = ["campaign", "mesh_study", "dataset", "benchmark", "extras", "twin", "report"]


def __getattr__(name: str):
    if name in __all__:
        return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
