"""Periodic structures (infinite arrays, FSS) in unit-cell Floquet runs, and the campaigns built on them.

* :mod:`.tcda` -- a generic tightly coupled dipole array cell for EMerge (Floquet port, periodic boundaries with the
  scan phase): active reflection over frequency and scan angle; a CLI the runner starts;
* :mod:`.benchmark` -- DOE campaigns of cells, the (frequency x scan) dataset, an FNO from the cell parameters to
  the whole map with held-out error, and the equal-budget comparison of FNO + optimiser against Bayesian
  optimisation on the solver.

Guide: docs/guides/unit_cell_benchmark.md.
"""
from . import benchmark, tcda
from .tcda import InvalidCell, TCDACell

__all__ = ["benchmark", "tcda", "InvalidCell", "TCDACell"]
