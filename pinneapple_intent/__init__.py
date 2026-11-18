"""PINNeAPPle Intelligence: a sentence in, a verified and replayable physics study out.

    import pinneapple as pp
    run = pp.study("a 12 mm steel wall, k = 200 W/mK, 300 degC inside, air at 25 degC outside, h = 12 W/m2K")
    print(run.summary())            # cold_face_temperature = 572.95 K (APPROVED); 12/12 checks passed
    run.files["report"]             # the Markdown report; contract.json and run.json sit next to it

    pp.study("a heat sink for a 40 W chip")     # -> the contract, status needs_clarification, with its questions

Three modes over one machinery:

* ``simulation`` — analytic, numerical and refinement rungs: a problem described in words becomes a solved and
  checked case.
* ``physics_ai`` — the same ladder plus a surrogate trained on the solver and scored on designs held out of its fit.
* ``science`` — recognised and refused with what it would need: the literature stage is not built yet.

The stages, each a typed object:

===========================  ===============================================================================
:mod:`.contract`             ``StudyContract``: the typed, versioned description of the study, with the provenance
                             of every value (``stated`` / ``answered`` / ``defaulted`` with its source /
                             ``missing``). Serialisable, addressable by SHA-256, reconstructible without an agent.
:mod:`.family`               ``StudyFamily``: the physics knowledge — slots, equations, assumptions, limitations,
                             references and the fidelity ladder. ``families()`` lists the installed ones.
:mod:`.compile`              ``compile_study(text)``: the Engineering Intent Compiler. Deterministic first, an LLM
                             optional behind ``IntentLLM``, and a refusal with what is missing when nothing fits.
:mod:`.execute`              ``run_study(contract)`` / ``study(text)``: walks the ladder, turns the checklines into a
                             ``TrustReport`` decision, writes ``contract.json``, ``run.json`` and ``report.md``.
:mod:`.replay`               ``replay(run_dir)``: rebuilds the run from its files and reports every number that did
                             not come back the same.

The rule the whole capability is built on: the model plans and proposes, the deterministic compiler and the solvers
check and execute. No value is invented, no result is reported without the checks that make it a result, and every
run can be rebuilt from its contract alone.
"""
from __future__ import annotations

from .compile import IntentLLM, compile_study
from .contract import (
    EXECUTABLE_MODES,
    MODE_PHYSICS_AI,
    MODE_SCIENCE,
    MODE_SIMULATION,
    MODES,
    NEEDS_CLARIFICATION,
    OUT_OF_SCOPE,
    READY,
    RUN_SCHEMA_ID,
    SCHEMA_ID,
    Provenance,
    Quantity,
    Question,
    Rung,
    Slot,
    StudyContract,
)
from .execute import StudyRun, run_study, study
from .family import (
    ANALYTIC,
    CONVERGENCE,
    NUMERICAL,
    SURROGATE,
    Measurement,
    StudyFamily,
    check,
    get_family,
    register,
    select_family,
)
from .family import (
    families as list_families,
)
from .replay import ReplayResult, replay

__all__ = [
    # entry points
    "study", "compile_study", "run_study", "replay",
    # contract
    "StudyContract", "StudyRun", "ReplayResult", "Provenance", "Rung", "Measurement", "StudyFamily",
    "Slot", "Quantity", "Question", "IntentLLM", "check",
    "SCHEMA_ID", "RUN_SCHEMA_ID", "MODES", "EXECUTABLE_MODES",
    "MODE_SIMULATION", "MODE_PHYSICS_AI", "MODE_SCIENCE",
    "READY", "NEEDS_CLARIFICATION", "OUT_OF_SCOPE",
    "ANALYTIC", "NUMERICAL", "CONVERGENCE", "SURROGATE",
    # registries
    "list_families", "get_family", "register", "select_family",
]
