"""Surrogate models from simulation results: select, compare at equal budget, measure on unseen data, package.

>>> from pinneapple_analysis import surrogate
>>> data = surrogate.examples.airfoil_rans()                 # OpenFOAM RANS campaign, Cl and Cd
>>> study = surrogate.run_study(data, budget=5.0, seeds=2)   # every applicable method, same splits and budget
>>> study.best, study.ranking("test"), study.trust["decision"]
>>> study.model.predict({"T0": 0.15, ..., "alpha_deg": 4.0})  # outputs + per-row OOD assessment
>>> study.model.save("model_dir")

Modules: ``data`` (tables and arrays -> ``SurrogateData``), ``splits`` (held-out and distribution-shift splits),
``methods`` (ridge, gp, gbt, mlp, fno, deeponet, pinn and when each applies), ``pde`` (declared equations for the
PINN), ``limits`` (validity bounds, OOD gate, error concentration), ``model`` (``SurrogateModel``), ``study``
(``run_study``), ``examples`` (datasets that need no download), ``contracts`` (``DatasetSpec``, ``ModelRef``,
``StudyResult`` of ``pinneapple_data.contracts``; imported on use). Needs ``pip install "pinneapple[surrogate]"``
(scikit-learn; Parquet, HDF5 and ONNX support).
"""
from . import examples
from .data import (
           SurrogateData,
           field_columns,
           from_arrays,
           from_snapshots,
           from_table,
           grid_coords,
           inspect_table,
           read_table,
           snapshot_values,
)
from .limits import OODGate, limitations
from .methods import CANDIDATES, applicable, candidate_ids
from .model import MODEL_KIND, SurrogateModel
from .pde import PDE_TYPES, parse_pde
from .splits import Splits, make_splits
from .study import DEFAULT_TARGETS, StudyResult, evaluate, run_study

__all__ = ["SurrogateData", "read_table", "from_table", "from_arrays", "field_columns", "inspect_table", "grid_coords",
           "from_snapshots", "snapshot_values", "Splits", "make_splits",
           "CANDIDATES", "applicable", "candidate_ids", "PDE_TYPES", "parse_pde", "OODGate", "limitations",
           "SurrogateModel", "MODEL_KIND", "run_study", "StudyResult", "evaluate", "DEFAULT_TARGETS", "examples"]
