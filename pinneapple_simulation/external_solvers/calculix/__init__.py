"""CalculiX (implicit structural FEM) bridge: decks, runs, results, and a drop-in virtual ccx.

CalculiX is to implicit problems what OpenRadioss (``..openradioss``) is to explicit ones:
the engineer's deck-based FEM workflow. This package reads and writes ``.inp`` decks, runs
``ccx`` (native or the bundled Debian-based Docker image), reads ``.frd``/``.dat`` results, and
provides ``virtual_ccx``: the same command line and file formats with the fields coming from a
PINN/surrogate, behind a validity-envelope guard that falls back to the real solver.

Validated: a C3D20R cantilever (1 x 0.1 x 0.1 m, steel, 1 kN tip load) gives a tip deflection
within 1 % of Timoshenko beam theory (tests/test_calculix_bridge.py).
"""
from .frd import FrdResults, read_dat_displacements, read_frd, write_frd
from .inp import CalculixModel, cantilever_hex_mesh, expand_target, read_inp, write_inp
from .runner import ccx_backend, docker_available, run_ccx
from .virtual_ccx import OutOfEnvelope, check_envelope, predict_to_frd

__all__ = [
    "CalculixModel", "cantilever_hex_mesh", "expand_target", "read_inp", "write_inp",
    "FrdResults", "read_frd", "write_frd", "read_dat_displacements",
    "ccx_backend", "docker_available", "run_ccx",
    "OutOfEnvelope", "check_envelope", "predict_to_frd",
]
