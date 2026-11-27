"""The known numerical limits of the OpenFOAM connectors, each with the case that demonstrates it (issue #132).

Three solvers hit real limits in earlier case campaigns — ``buoyantSimpleFoam``, ``buoyantPimpleFoam`` and free-surface
flow with k-omega SST — and "it failed once" is not something a connector can act on. This module records, per
solver, **the decision** (accepted and documented, worked around, or unsupported), **what was measured**, and **the
minimal case that demonstrates it**, so the connector's rules (:mod:`pinneapple_data.preflight.openfoam`) cite
evidence instead of folklore.

Every entry below was measured on OpenFOAM v2606 (macOS arm64, native build) on 2026-11-24, with the cases of
``tests/test_openfoam_limits.py``: ``buoyant_simple_cavity`` (a 20x20x1 closed cavity, hot left wall at 350 K, cold
right wall at 300 K, k-omega SST, 300 SIMPLE iterations, 0.57 s of solver time), ``buoyant_pimple_room`` (the
shipped ``hotRoom`` tutorial cut to endTime 20 s, ten PIMPLE steps, 2.1 s), and ``dambreak_sst`` (the shipped
``interFoam/RAS/damBreak`` case switched from kEpsilon to kOmegaSST, 71 steps to 0.2 s, 1.05 s). The verbatim log line
of each failure is in ``detail``.

The three decisions
-------------------
``buoyantSimpleFoam`` — **accepted, with two hard requirements the connector enforces**: ``constant/g`` and
``pRefCell``/``pRefPoint``. Both missing cases are a fatal stop at start-up, not a divergence, so a case either has
them or never runs. Relaxation is a quality issue, not a fatal one, and is documented as such.

``buoyantPimpleFoam`` — **accepted, with the start-up flux correction required**: the transient compressible solver
needs a ``rho`` linear-solver entry (fatal without it) and ``correctPhi yes`` with a ``pcorr`` solver. Without the
correction the run still completes, silently, with a first-step local continuity error 12 times larger (measured
1.6e-3 against 1.3e-4); that is the kind of wrong answer a campaign must not accept, so the connector warns.

``interIsoFoam`` + k-omega SST — **supported, and run for real**: the free-surface case that had never been run
completed its 71 steps with the interface bounded (min -7.7e-5, max 1.0005 on the water fraction) and a maximum
Courant number of 1.02. The cost is three dictionary entries that kEpsilon did not need (``wallDist`` in fvSchemes,
``div(rhoPhi,omega)``, and an ``omega``/``omegaFinal`` solver entry), each of which is a fatal stop if missing — so the
connector checks them instead of leaving a model switch to discover them at run time. The known limit that remains
is documented rather than solved: turbulence in the gas phase has no physical scale to calibrate against, so the
fields it produces in air are numerical, and the decision is to accept that and say so.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["Limit", "LIMITS", "DECISIONS", "limits_for", "decision", "unsupported", "as_dicts"]

ACCEPTED, WORKED_AROUND, UNSUPPORTED = "accepted", "worked around", "unsupported"


@dataclass(frozen=True)
class Limit:
    """One measured limit, the case that shows it, and what the connector does about it."""
    solver: str
    name: str
    statement: str                    # what the limit is, in one sentence a reviewer can act on
    decision: str                     # accepted | worked around | unsupported
    detail: str                       # the measured evidence, verbatim where it is a message
    minimal_case: str                 # the case in tests/test_openfoam_limits.py that demonstrates it
    rule: str = ""                    # the preflight rule that enforces or reports it ("" = documentation only)
    fix: str = ""                     # what a user changes, when something can be changed
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LIMITS: tuple[Limit, ...] = (
    # ------------------------------------------------------------------ buoyantSimpleFoam
    Limit(
        solver="buoyantSimpleFoam", name="gravity", decision=ACCEPTED,
        statement="The solver refuses to start without constant/g: a buoyant case without a gravity vector has no "
                  "buoyancy, so this is a hard requirement rather than a suggestion.",
        detail="FOAM FATAL ERROR: cannot find file \".../constant/g\" (measured, v2606, after 0.9 s)",
        minimal_case="buoyant_simple_cavity without constant/g",
        rule="gravity_missing",
        fix="Write constant/g with dimensions [0 1 -2 0 0 0 0] and value (0 -9.81 0).",
        tags=("fatal", "start-up"),
    ),
    Limit(
        solver="buoyantSimpleFoam", name="pressure reference", decision=ACCEPTED,
        statement="Without pRefCell or pRefPoint the pressure is defined up to a constant and the solver stops at "
                  "start-up; a closed buoyant cavity needs the reference the same way the incompressible solvers do.",
        detail="FOAM FATAL IO ERROR: Unable to set reference cell for field p / Please supply either pRefCell or "
               "pRefPoint (measured, v2606)",
        minimal_case="buoyant_simple_cavity without pRefCell/pRefValue",
        rule="pref",
        fix="Add pRefCell 0; pRefValue 0; to the SIMPLE block of system/fvSolution, or fix p on an outlet.",
        tags=("fatal", "start-up"),
    ),
    Limit(
        solver="buoyantSimpleFoam", name="relaxation", decision=ACCEPTED,
        statement="No under-relaxation does not crash this case, it stalls it: the p_rgh initial residual stays at "
                  "0.39 after 300 iterations where 0.7 relaxation reaches 0.11 — a run that must be read as "
                  "not converged, not as converged.",
        detail="p_rgh initial residual at step 300: 0.388 with factors 1.0 against 0.110 with 0.7 (measured, v2606)",
        minimal_case="buoyant_simple_cavity with relaxationFactors 1.0",
        rule="relax_high",
        fix="Keep p_rgh/U/T/k/omega at 0.7 (or lower); check the residualControl verdict, not the exit code.",
        tags=("quality", "residual"),
    ),
    Limit(
        solver="buoyantSimpleFoam", name="thermophysical package", decision=WORKED_AROUND,
        statement="v2606 selects heRhoThermo for this solver: a case written for hePsiThermo (the older tutorials) "
                  "stops at start-up, and so does the alphat wall function name, which must carry the compressible:: "
                  "prefix here.",
        detail="FOAM FATAL ERROR: ... getThermoOrDie: hePsiThermo is not in the table for rhoThermo; and later "
               "\"Unknown patchField type alphatWallFunction for patch type wall\" (measured, v2606)",
        minimal_case="buoyant_simple_cavity (the dictionaries the connector writes)",
        fix="Use type heRhoThermo and compressible::alphatWallFunction; keep constant/turbulenceProperties as the "
            "file name (v2606 reads it, not momentumTransport).",
        tags=("setup", "version"),
    ),
    # ------------------------------------------------------------------ buoyantPimpleFoam
    Limit(
        solver="buoyantPimpleFoam", name="rho linear solver", decision=ACCEPTED,
        statement="The transient compressible solver solves a density equation, so fvSolution needs a rho entry; "
                  "without it the run stops before the first time step.",
        detail="FOAM FATAL IO ERROR: Entry 'rho' not found in dictionary \"system/fvSolution/solvers\" (measured, v2606)",
        minimal_case="buoyant_pimple_room without the rho solver entry",
        rule="rho_solver",
        fix="Add rho { solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0; } to system/fvSolution/solvers.",
        tags=("fatal", "start-up"),
    ),
    Limit(
        solver="buoyantPimpleFoam", name="start-up flux correction", decision=ACCEPTED,
        statement="Without correctPhi/pcorr the run completes and reports nothing, but the first step's local "
                  "continuity error is 12 times larger: the initial mass imbalance is never corrected, and every "
                  "result inherits it.",
        detail="time step continuity errors : sum local = 0.000129275 (with correctPhi yes) against 0.00161142 "
               "(correctPhi no), same case, same 10 steps (measured, v2606)",
        minimal_case="buoyant_pimple_room with correctPhi no and no pcorr solver",
        rule="correct_phi",
        fix="Keep correctPhi yes in the PIMPLE block and a pcorr solver in fvSolution.",
        tags=("silent", "mass conservation"),
    ),
    Limit(
        solver="buoyantPimpleFoam", name="cold-start pressure field", decision=ACCEPTED,
        statement="A transient buoyant case started from a pressure field that is not in hydrostatic balance "
                  "(p = p_rgh = 1e5, U = 0, Co ~ 1) dies in the first PIMPLE iteration with a floating-point "
                  "exception, which the solver reports as a signal rather than as a dictionary error.",
        detail="Illegal instruction (FOAM_SIGFPE) in the first PIMPLE iteration of a hand-built cavity; the same "
               "fields from the hotRoom tutorial, which sets hRef and a hydrostatic p_rgh, run (measured, v2606)",
        minimal_case="buoyant_pimple_room started from uniform p_rgh (see the test's notes)",
        rule="",
        fix="Set hRef and an initial p_rgh consistent with gravity (or run a potential/potentialFoam step first).",
        tags=("fatal", "initial condition"),
    ),
    # ------------------------------------------------------------------ free surface with k-omega SST
    Limit(
        solver="interIsoFoam", name="k-omega SST runs on a free surface", decision=ACCEPTED,
        statement="The free-surface case with k-omega SST runs: 71 steps to endTime 0.2 s, water fraction between "
                  "-7.7e-5 and 1.0005, maximum Courant 1.02. It is supported, not merely untested.",
        detail="ExecutionTime = 1.05 s; Phase-1 volume fraction = 0.130194; min(alpha.water) = -7.7e-05, "
               "max = 1.0005; Courant max 1.02 (measured, v2606, interFoam/RAS/damBreak switched to kOmegaSST)",
        minimal_case="dambreak_sst",
        tags=("supported", "measured"),
    ),
    Limit(
        solver="interIsoFoam", name="wall distance", decision=ACCEPTED,
        statement="k-omega SST needs a wall distance and kEpsilon does not, so switching the model on an existing "
                  "free-surface case stops it at start-up until fvSchemes declares wallDist.",
        detail="FOAM FATAL IO ERROR: Entry 'method' not found in dictionary \"system/fvSchemes/wallDist\" (measured, "
               "v2606, switching interFoam/RAS/damBreak from kEpsilon to kOmegaSST)",
        minimal_case="dambreak_sst before wallDist is added",
        rule="wall_dist",
        fix="Add wallDist { method meshWave; } to system/fvSchemes.",
        tags=("fatal", "start-up"),
    ),
    Limit(
        solver="interIsoFoam", name="turbulence convection and solver entries", decision=ACCEPTED,
        statement="The k-omega fields need their own convection scheme and linear solver even when the case already "
                  "has entries for k and epsilon: with 'default none' and no omega entry the solver stops at the "
                  "first turbulence solve.",
        detail="FOAM FATAL IO ERROR: Entry 'div(rhoPhi,omega)' not found in dictionary ...divSchemes; then Entry "
               "'omegaFinal' not found in dictionary ...solvers (measured, v2606)",
        minimal_case="dambreak_sst before div(rhoPhi,omega) and the omega solver are added",
        rule="omega_schemes",
        fix="Add div(rhoPhi,omega) next to div(rhoPhi,k) and extend the turbulence solver regex to (U|k|omega).*",
        tags=("fatal", "start-up"),
    ),
    Limit(
        solver="interIsoFoam", name="setFields after restore", decision=ACCEPTED,
        statement="The volume fraction is not part of 0.orig: a case whose 0/ was restored after setFields runs to "
                  "the end with alpha = 0 everywhere and reports no error. The run is silent and wrong, which is "
                  "why the campaign records the phase fraction and not only the exit code.",
        detail="23 steps to endTime with \"Phase-1 volume fraction = 0  Min(alpha.water) = 0  Max(alpha.water) = 0\" "
               "(measured, v2606); after setFields the same case reports 0.130194 (measured)",
        minimal_case="dambreak_sst with 0/ restored from 0.orig after setFields",
        rule="",
        fix="Run setFields after restoring 0/, and gate the run on the reported phase fraction being non-zero.",
        tags=("silent", "wrong answer"),
    ),
    Limit(
        solver="interIsoFoam", name="turbulence in the gas phase", decision=UNSUPPORTED,
        statement="Turbulence in the air side of a free-surface flow has no scale to calibrate against: the k and "
                  "omega fields there are numerical, and the decision is to accept them as such rather than to "
                  "pretend they are validated physics.",
        detail="The run above is stable; what is not established is the physical meaning of k/omega in the gas, "
               "which no measurement in this library's campaigns covers.",
        minimal_case="dambreak_sst (the run it produces is the evidence that the case is stable, not that the gas "
                     "turbulence is right)",
        tags=("accuracy", "model scope"),
    ),
)

#: The decision per solver, in one word, as the issue asks for.
DECISIONS: dict[str, str] = {
    "buoyantSimpleFoam": ACCEPTED,
    "buoyantPimpleFoam": ACCEPTED,
    "interIsoFoam": ACCEPTED,
}


def limits_for(solver: str) -> tuple[Limit, ...]:
    """Every recorded limit of one solver (an unknown solver gives an empty tuple, never an error)."""
    return tuple(lim for lim in LIMITS if lim.solver == solver)


def decision(solver: str) -> str:
    """The recorded decision for a solver: ``accepted``, ``worked around``, ``unsupported`` or ``""``."""
    return DECISIONS.get(solver, "")


def unsupported() -> tuple[Limit, ...]:
    """The limits that were decided against: parts of a solver that this library does not claim to model."""
    return tuple(lim for lim in LIMITS if lim.decision == UNSUPPORTED)


def as_dicts(solver: str | None = None) -> list[dict[str, Any]]:
    """The registry as JSON-ready dicts (``solver=None`` gives every entry)."""
    return [lim.to_dict() for lim in (LIMITS if solver is None else limits_for(solver))]
