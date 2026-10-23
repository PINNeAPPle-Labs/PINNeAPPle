"""Printed end-fire antennas (antipodal Vivaldi) with an external FEM solver: limits, checks, runner, surrogates.

* :mod:`.bounds` -- end-fire directivity limits (Hansen-Woodyard ``7.28 L/lambda``, ordinary ``4 L/lambda``),
  broadside aperture gain, the shortest length for a target gain and the feasibility frontier;
* :mod:`.checks` -- consistency checks of a result row as a ``TrustReport`` (realized gain, ``e = G/D``,
  ``e <= 1``, peak >= boresight, ``D`` below the end-fire limit, interpolated S11, far-field power closure);
* :mod:`.ava` -- an open parametric antipodal Vivaldi model for EMerge (``pip install emerge``) and its CLI;
* :mod:`.runner` -- one process per design and frequency with timeout, peak memory and the SuperLU fallback;
  the (design, frequency) table reader and writer;
* :mod:`.surrogates` -- frequency mean, HW + offset, linear physical features, Gaussian process; leave-one-design-
  out and held-out validation;
* :mod:`.design` -- DOE (LHS / Sobol), NSGA-II on the surrogates, active learning verified by the solver,
  multi-fidelity bias correction, a trust card per prediction;
* :mod:`.package` -- read an external study package (table + script) from a local folder and re-run a design;
* :mod:`.plots` -- directivity against ``L/lambda`` with the limits, LODO RMSE per model, layout.

Guide: docs/guides/antenna_design.md. Example: examples/use_cases/antennas/vivaldi_2to4ghz.py.
"""
from . import bounds, checks
from .ava import AVAParams, InvalidGeometry, layout, microstrip_width
from .bounds import endfire_directivity_dbi, feasibility_frequency, min_length
from .checks import antenna_checks
from .runner import CLISimulator, ResultTable, ava_simulator, read_table, write_table

__all__ = ["bounds", "checks", "AVAParams", "InvalidGeometry", "layout", "microstrip_width",
           "endfire_directivity_dbi", "feasibility_frequency", "min_length", "antenna_checks", "CLISimulator",
           "ResultTable", "ava_simulator", "read_table", "write_table"]
