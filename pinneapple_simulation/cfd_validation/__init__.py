"""Classical CFD validation cases with their published numbers, run against a real solver (issues #272, #265, #294, #494).

Every learned model and every surrogate in this repository is scored against a reference; this package is where the
reference itself is checked. It holds four things and keeps them apart:

* the published tables (:mod:`~pinneapple_simulation.cfd_validation.reference`), transcribed as data with the table
  they came from, and never re-derived from a run;
* the cases that can be compared with them (:mod:`~pinneapple_simulation.cfd_validation.cavity`,
  :mod:`~pinneapple_simulation.cfd_validation.step`), which write their own OpenFOAM dictionaries and read the answer
  back as cell-centred numbers;
* the convergence arithmetic that turns three refined runs into an observed order and a Richardson-extrapolated value
  (:func:`observed_order`, over :func:`~pinneapple_analysis.verification.convergence.richardson_extrapolate`, which
  the repository already had for exactly this);
* the one place the solver is looked for (:mod:`~pinneapple_simulation.cfd_validation.foam`), so a machine without
  OpenFOAM skips the runs and still gets the tables.

    import pinneapple as pp

    out = pp.cfd.validation.cavity_study(100, "runs/cavity")   # three meshes, errors against Ghia (1982)
    out["errors"][64]["u"]["l2_relative"]                       # the number to quote, at 64 x 64

Without OpenFOAM (:func:`~pinneapple_simulation.cfd_validation.foam.available`) the studies raise
:class:`OpenFOAMMissing`, which the tests turn into a skip and a caller reads as "this machine cannot run the
validation" rather than as a wrong answer. The tables in :mod:`~pinneapple_simulation.cfd_validation.reference` need no
solver at all.
"""
from __future__ import annotations

from . import cavity, foam, reference, step
from .cavity import CavityRun, profile_error, sample
from .cavity import study as cavity_study
from .cavity import write_case as write_cavity_case
from .reference import (
    ARMALY_REATTACHMENT,
    DEV_VAHL_DAVIS_NU,
    GHIA_RE,
    GHIA_U,
    GHIA_V,
    PROFILES,
    Profile,
    nu_for_grashof,
    profile,
)
from .step import StepRun, reattachment_length
from .step import study as step_study
from .step import write_case as write_step_case

__all__ = [
    # where OpenFOAM is, and how a case runs
    "foam", "available", "OpenFOAMMissing",
    # the published tables
    "reference", "Profile", "GHIA_U", "GHIA_V", "GHIA_RE", "DEV_VAHL_DAVIS_NU", "ARMALY_REATTACHMENT", "PROFILES",
    "profile", "nu_for_grashof",
    # the lid-driven cavity
    "cavity", "cavity_study", "write_cavity_case", "sample", "profile_error", "CavityRun",
    # the backward-facing step
    "step", "step_study", "write_step_case", "reattachment_length", "StepRun",
    # convergence
    "observed_order",
]


class OpenFOAMMissing(RuntimeError):
    """No OpenFOAM on this machine: the case cannot be run, and that is not the same as the case being wrong."""


def available(*needs: str) -> bool:
    """Whether an OpenFOAM installation (with ``needs``, e.g. ``"simpleFoam"``) is there to run the cases."""
    return foam.available(*needs)


def require(*needs: str) -> str:
    """The bashrc, or :class:`OpenFOAMMissing` with the one thing a caller can do about it."""
    if not available(*needs):
        raise OpenFOAMMissing("no OpenFOAM installation found"
                             f"{' (needs ' + ', '.join(needs) + ')' if needs else ''}: "
                             "install OpenFOAM or set FOAM_BASHRC to its etc/bashrc")
    return foam.bashrc() or ""


def observed_order(errors: list[float], ratio: float = 2.0) -> list[float]:
    """The observed order of accuracy from the error on successive refinements: ``log(e_i/e_{i+1}) / log(r)``.

    The error against a published table should fall by ``r**p`` per refinement, where ``p`` is the order of the
    scheme; two errors give one estimate and three give two, and their agreement is what makes the pair believable.
    A zero error gives ``inf`` (nothing left to resolve) and a growing one gives a negative order, both reported as
    they are: an error that does not fall is the finding, not something to filter.
    """
    out = []
    for a, b in zip(errors, errors[1:], strict=False):
        if b <= 0:
            out.append(float("inf") if a > 0 else float("nan"))
        elif a <= 0:
            out.append(float("-inf"))
        else:
            import math
            out.append(math.log(a / b) / math.log(ratio))
    return out
