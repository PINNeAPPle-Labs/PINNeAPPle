"""Module-level API used by ``import pinneapple as pp; pp.decide(...)``.

``pinneapple`` exposes :func:`decide`, :func:`execute`, :func:`verify`,
:func:`run_tree` and :func:`decision_engine` lazily (``pinneapple.__getattr__``):
``import pinneapple`` does not import this package; the first access to
``pp.decide`` does, and this package does not import torch.

All calls share one default :class:`~pinneapple_decision.loop.PhysicsDecisionEngine`
so that ``decide -> execute -> verify -> decide`` works as a loop. Configure it
(executor, verifier, backend, evidence file) with :func:`decision_engine`, or
pass ``engine=`` to any call.
"""
from __future__ import annotations

from typing import Any, Optional

from .loop import PhysicsDecisionEngine
from .schema import Decision, ExecutionResult, PhysicsChoice, Verification

_ENGINE: Optional[PhysicsDecisionEngine] = None


def decision_engine(executor: Any = None, verifier: Any = None, decider: Any = None, store: Any = None,
                    *, reset: bool = False) -> PhysicsDecisionEngine:
    """Return the default engine. Passing any argument (or ``reset=True``) replaces it
    with a new engine built from those arguments (evidence is not carried over)."""
    global _ENGINE
    if _ENGINE is None or reset or any(x is not None for x in (executor, verifier, decider, store)):
        _ENGINE = PhysicsDecisionEngine(executor=executor, verifier=verifier, decider=decider, store=store)
    return _ENGINE


def decide(problem: Any = None, objective: Optional[str] = None, *, state: Any = None,
           choice: Optional[PhysicsChoice] = None, engine: Optional[PhysicsDecisionEngine] = None) -> Decision:
    """Decide which experiment to run next. ``problem`` may be a ``ProblemSpec`` or a
    free-form dict (see ``adapter.py``). Facts that could not be determined are listed in
    ``decision.diagnostics["problem_facts"]["unknown"]``."""
    return (engine or decision_engine()).decide(problem, objective, state=state, choice=choice)


def execute(decision: Decision, problem: Any = None, *,
            engine: Optional[PhysicsDecisionEngine] = None) -> ExecutionResult:
    """Execute a decision issued by the same engine (constraints are re-checked first)."""
    return (engine or decision_engine()).execute(decision, problem)


def verify(result: ExecutionResult, *, engine: Optional[PhysicsDecisionEngine] = None) -> Verification:
    """Verify a result; only verified results become evidence for the next decision."""
    return (engine or decision_engine()).verify(result)


def run_tree(problem: Any, tree: Any = None, *, max_experiments: int = 3,
             engine: Optional[PhysicsDecisionEngine] = None):
    """Walk the physics-AI decision tree (``tree.physics_ai_tree()`` by default)."""
    return (engine or decision_engine()).run_tree(problem, tree, max_experiments=max_experiments)


__all__ = ["decide", "decision_engine", "execute", "run_tree", "verify"]
