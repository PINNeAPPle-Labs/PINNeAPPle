"""pinneapple_decision -- probabilistic decision layer over the Physics AI pipeline.

The LLM (or the offline rule backend) never predicts a physical result. It
decides *which experiment to run next* (model, training strategy, validation),
every option passes hard constraints first, every executed result goes through
a verifier, and only verified results become evidence for the next decision.

Decisions are read as distributions over a closed option set (no text
generation), debiased without labels at level L0 (AnyJev-like; see README).
See ``README.md`` in this package.
"""
from .backends import LogProbLLMBackend, RuleBasedBackend, ScoringBackend
from .constraints import (
    NAMED_CONSTRAINTS,
    Constraint,
    ConstraintViolationError,
    NoFeasibleOptionError,
    apply_constraints,
)
from .decider import PhysicsDecider
from .loop import (
    CallableExecutor,
    DecisionLoop,
    EvidenceStore,
    Executor,
    PhysicsDecisionEngine,
    ValidationRequiredError,
    choice_for_objective,
)
from .schema import (
    Decision,
    DecisionLevel,
    DecisionState,
    Evidence,
    ExecutionResult,
    OptionInfo,
    PhysicsChoice,
    Verification,
    problem_from_spec,
)
from .selectors import ModelSelector, TrainingStrategySelector, ValidationStrategySelector
from .verifiers import ThresholdVerifier, VeriPhysicsVerifier, Verifier

__all__ = [
    "CallableExecutor",
    "Constraint",
    "ConstraintViolationError",
    "Decision",
    "DecisionLevel",
    "DecisionLoop",
    "DecisionState",
    "Evidence",
    "EvidenceStore",
    "ExecutionResult",
    "Executor",
    "LogProbLLMBackend",
    "ModelSelector",
    "NAMED_CONSTRAINTS",
    "NoFeasibleOptionError",
    "OptionInfo",
    "PhysicsChoice",
    "PhysicsDecider",
    "PhysicsDecisionEngine",
    "RuleBasedBackend",
    "ScoringBackend",
    "ThresholdVerifier",
    "TrainingStrategySelector",
    "ValidationRequiredError",
    "ValidationStrategySelector",
    "VeriPhysicsVerifier",
    "Verification",
    "Verifier",
    "apply_constraints",
    "choice_for_objective",
    "problem_from_spec",
]
