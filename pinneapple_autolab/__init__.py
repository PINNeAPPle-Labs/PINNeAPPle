"""pinneapple_autolab -- the Autonomous Physics Lab: an AI that manages evidence to make engineering decisions.

Two layers, shared by every mission:

* the agent loop (:mod:`.core`): a mission in plain language -> a plan of controlled library calls -> evidence
  (``Tool``, ``Toolbox``, ``Step``, ``TemplatePlanner``, ``LLMPlanner``, ``execute``); the physics of each mission
  lives in its domain package (e.g. :mod:`pinneapple_design.thermal.board_cooling`);
* the scientific cycle on a costly reference simulator (:mod:`.lab`): plan -> build (geometry and boundary-condition
  checks) -> simulate -> learn (surrogate families compared on held-out data, with uncertainty) -> optimise (NSGA-II
  on the surrogate) -> assess every candidate (validated domain, probability of meeting the limits, expected field
  error where the decision is made) -> choose the next simulation by what is still unknown -> verify -> deliver (a
  simulated design, its trust report and the ledger).

    from pinneapple_autolab import AutonomousLab
    from pinneapple_autolab.domains.racecar import RaceCarAero, flagship_mission

    lab = AutonomousLab(flagship_mission(), RaceCarAero(), "runs/aero", critical_region="diffuser")
    summary = lab.run()

Only the ledger's runs are evidence; predictions are labelled as predictions until a simulation confirms them.
"""
from .core import (
    Evidence,
    LLMPlanner,
    Plan,
    Planner,
    Run,
    Step,
    TemplatePlanner,
    Tool,
    Toolbox,
    execute,
)
from .domain import DomainVerdict, ValidatedDomain
from .lab import AutonomousLab
from .ledger import BudgetExhaustedError, Ledger, RunRecord, environment
from .mission import Budget, Limit, Mission, UnresolvedLimitError
from .simulator import FunctionSimulator, SimOutcome, Simulator
from .surrogates import (
    FAMILIES,
    FieldSurrogate,
    GPModel,
    MLPEnsemble,
    ModelBoard,
    QuadraticRSM,
    compare_models,
    cross_validate,
    field_errors,
)
from .trust import (
    CandidateAssessment,
    NextExperiment,
    assess_candidates,
    choose_next,
    verification_checks,
)

__all__ = [
    "Tool", "Toolbox", "Step", "Plan", "Planner", "TemplatePlanner", "LLMPlanner", "Evidence", "Run", "execute",
    "AutonomousLab", "Budget", "BudgetExhaustedError", "CandidateAssessment", "DomainVerdict", "FAMILIES",
    "FieldSurrogate", "FunctionSimulator", "GPModel", "Ledger", "Limit", "MLPEnsemble", "Mission", "ModelBoard",
    "NextExperiment", "QuadraticRSM", "RunRecord", "SimOutcome", "Simulator", "UnresolvedLimitError",
    "ValidatedDomain", "assess_candidates", "choose_next", "compare_models", "cross_validate", "environment",
    "field_errors", "verification_checks",
]
