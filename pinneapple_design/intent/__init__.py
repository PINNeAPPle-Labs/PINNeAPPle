"""Design from intent: a plain-language goal in, a buildable and verified design package out.

    import pinneapple as pp
    pkg = pp.design("I want a toy car that does a loop", out_dir="loop_design")
    print(pkg.summary())                      # design, verdict, probability, trust decision
    pp.design("a heat sink for a 40 W chip in still air")   # -> NeedsInput: asks for the temperature limit

Stages (each a typed object): :func:`formulate` (text -> :class:`RequirementSpec`, values with units and provenance),
:func:`plan` (-> :class:`PhysicsPlan` from a registered :class:`ProblemTemplate`), :func:`optimise` (closed-form
screening, optimisation on the reduced model with the design-study core), :func:`robustness` (Monte Carlo),
:func:`verify` (full solver), :func:`assess` (trust report and verdict), :class:`DesignPackage` (files and report).
Requests outside every template return an :class:`OutOfScope` that says what would be needed.
"""
from __future__ import annotations

from . import materials
from .api import NeedsInput, design
from .package import DesignPackage, make_geometry
from .parse import (
    CallLLM,
    IntentLLM,
    OutOfScope,
    find_quantities,
    formulate,
    get_template,
    register,
    select_template,
)
from .parse import templates as list_templates
from .pipeline import (
    Assessment,
    PhysicsPlan,
    Robustness,
    Verification,
    assess,
    optimise,
    plan,
    robustness,
    verify,
)
from .spec import (
    ANSWERED,
    DEFAULTED,
    MISSING,
    STATED,
    ObjectiveChoice,
    Quantity,
    Question,
    RequirementSpec,
    Slot,
    SpecError,
)
from .template import (
    CLOSED_FORM,
    FULL,
    REDUCED,
    DesignVariable,
    FailureMode,
    Fidelity,
    FullResult,
    ProblemTemplate,
    Uncertain,
)

__all__ = [
    "design", "formulate", "plan", "optimise", "robustness", "verify", "assess", "register", "list_templates",
    "get_template", "select_template", "find_quantities", "materials",
    "RequirementSpec", "Quantity", "Slot", "Question", "ObjectiveChoice", "SpecError", "STATED", "ANSWERED",
    "DEFAULTED", "MISSING", "ProblemTemplate", "Fidelity", "FailureMode", "DesignVariable", "Uncertain",
    "FullResult", "CLOSED_FORM", "REDUCED", "FULL", "PhysicsPlan", "Robustness", "Verification", "Assessment",
    "DesignPackage", "NeedsInput", "OutOfScope", "IntentLLM", "CallLLM", "make_geometry",
]
