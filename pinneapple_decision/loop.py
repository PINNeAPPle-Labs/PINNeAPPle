"""OBSERVE -> UNDERSTAND -> DECIDE -> EXECUTE -> VALIDATE -> LEARN.

Guarantees enforced here (not just documented):

* only decisions issued by this loop can be executed;
* the selected option is re-checked against the choice's constraints right
  before execution (:class:`ConstraintViolationError` otherwise);
* a result becomes evidence for the next decision only after verification:
  deciding on a state whose ``previous_result`` has no ``verification`` raises
  :class:`ValidationRequiredError` when the previous decision required it
  (``requires_validation=True`` is the default for every decision).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

from .constraints import ConstraintViolationError, apply_constraints
from .decider import PhysicsDecider
from .schema import Decision, DecisionState, Evidence, ExecutionResult, PhysicsChoice, Verification
from .selectors import ModelSelector, TrainingStrategySelector, ValidationStrategySelector
from .verifiers import ThresholdVerifier, Verifier


class ValidationRequiredError(RuntimeError):
    """A result was offered as evidence before it was verified."""


@runtime_checkable
class Executor(Protocol):
    def execute(self, decision: Decision, problem: Dict[str, Any]) -> ExecutionResult:
        ...


class CallableExecutor:
    """Wrap ``fn(decision, problem) -> ExecutionResult | dict``."""

    def __init__(self, fn: Callable[[Decision, Dict[str, Any]], Union[ExecutionResult, Dict[str, Any]]]):
        self.fn = fn

    def execute(self, decision: Decision, problem: Dict[str, Any]) -> ExecutionResult:
        out = self.fn(decision, problem)
        return out if isinstance(out, ExecutionResult) else ExecutionResult(**out)


class EvidenceStore:
    """In-memory evidence log, optionally persisted as JSON."""

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self.path = Path(path) if path else None
        self.records: List[Evidence] = []
        if self.path and self.path.exists():
            self.records = [Evidence.from_dict(d) for d in json.loads(self.path.read_text())]

    def add(self, ev: Evidence) -> Evidence:
        self.records.append(ev)
        self._flush()
        return ev

    def find(self, decision_id: str) -> Optional[Evidence]:
        return next((e for e in reversed(self.records) if e.decision.id == decision_id), None)

    def last(self) -> Optional[Evidence]:
        return self.records[-1] if self.records else None

    def history(self) -> Tuple[Evidence, ...]:
        return tuple(self.records)

    def _flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([e.to_dict() for e in self.records], indent=2, default=str))

    save = _flush


class DecisionLoop:
    def __init__(self, executor: Optional[Executor] = None, verifier: Optional[Verifier] = None,
                 decider: Optional[PhysicsDecider] = None, store: Optional[EvidenceStore] = None):
        self.executor = executor
        self.verifier = verifier or ThresholdVerifier()
        self.decider = decider or PhysicsDecider()
        self.store = store or EvidenceStore()
        self._issued: Dict[str, Tuple[PhysicsChoice, DecisionState]] = {}

    # -- DECIDE -------------------------------------------------------------
    def _state(self, problem_or_state: Any) -> DecisionState:
        state = DecisionState.coerce(problem_or_state)
        if state.previous_result is not None and state.verification is None:
            prev = self.store.find(state.previous_result.decision_id or "")
            if prev is not None and prev.verification is not None:
                state.verification = prev.verification
            elif prev is None or prev.decision.requires_validation:
                raise ValidationRequiredError(
                    "previous_result has not been verified; call verify(result) before using it as evidence")
        if not state.history:
            state.history = self.store.history()
        return state

    def decide_choice(self, problem_or_state: Any, choice: PhysicsChoice) -> Decision:
        state = self._state(problem_or_state)
        decision = self.decider.decide(state, choice)
        self._issued[decision.id] = (choice, state)
        return decision

    # -- EXECUTE ------------------------------------------------------------
    def execute(self, decision: Decision, problem: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        if self.executor is None:
            raise RuntimeError("no executor configured")
        if decision.id not in self._issued:
            raise ConstraintViolationError("decision was not issued by this loop; refusing to execute it")
        choice, state = self._issued[decision.id]
        if problem is not None:
            state = DecisionState.from_problem(problem, previous_result=state.previous_result,
                                               verification=state.verification, history=state.history)
        feasible, excluded = apply_constraints(choice, state)
        if decision.selected not in feasible:
            raise ConstraintViolationError(
                f"{decision.selected!r} no longer satisfies the constraints: {excluded.get(decision.selected)}")
        result = self.executor.execute(decision, dict(state.problem))
        result.decision_id = decision.id
        ev = self.store.find(decision.id)
        if ev is not None and ev.result is None:  # decision already logged (e.g. a tree step)
            ev.result = result
            self.store.save()
        else:
            self.store.add(Evidence(decision=decision, result=result))
        return result

    # -- VALIDATE / LEARN ---------------------------------------------------
    def verify(self, result: ExecutionResult) -> Verification:
        ev = self.store.find(result.decision_id or "")
        if ev is None:
            raise ValueError("result does not belong to a decision executed by this loop")
        problem = self._issued[ev.decision.id][1].problem if ev.decision.id in self._issued else None
        verification = self.verifier.verify(result, problem)
        ev.verification = verification
        self.store.save()
        return verification

    def step(self, problem: Dict[str, Any], choice: PhysicsChoice) -> Evidence:
        """One full cycle; the last verified evidence feeds this decision."""
        last = self.store.last()
        state = DecisionState.from_problem(
            problem,
            previous_result=last.result if last else None,
            verification=last.verification if last else None,
            history=self.store.history(),
        )
        decision = self.decide_choice(state, choice)
        result = self.execute(decision)
        self.verify(result)
        return self.store.find(decision.id)

    def run(self, problem: Dict[str, Any], choice: PhysicsChoice, *, max_steps: int = 3,
            stop_when_passed: bool = True) -> List[Evidence]:
        out = []
        for _ in range(max_steps):
            ev = self.step(problem, choice)
            out.append(ev)
            if stop_when_passed and ev.verification is not None and ev.verification.passed:
                break
        return out

    def run_tree(self, problem: Any, tree: Any = None, *, max_experiments: int = 3):
        """Walk a declarative decision tree (default: ``tree.physics_ai_tree()``).
        Factual nodes are answered by problem facts when known; see ``tree.py``."""
        from .tree import run_tree

        return run_tree(self, problem, tree, max_experiments=max_experiments)


_OBJECTIVES = (
    (("valid", "verif"), ValidationStrategySelector),
    (("train", "strategy", "next experiment", "improve", "sampling"), TrainingStrategySelector),
    (("model", "architecture", "surrogate"), ModelSelector),
)


def choice_for_objective(objective: Optional[str], has_previous_result: bool = False) -> PhysicsChoice:
    text = (objective or "").lower()
    for keys, selector in _OBJECTIVES:
        if any(k in text for k in keys):
            return selector.choice()
    return (TrainingStrategySelector if has_previous_result else ModelSelector).choice()


class PhysicsDecisionEngine(DecisionLoop):
    """High-level facade: ``decide(problem, objective=...)`` / ``decide(state={...})``,
    ``execute(decision)``, ``verify(result)``."""

    def decide(self, problem: Any = None, objective: Optional[str] = None, *, state: Any = None,
               choice: Optional[PhysicsChoice] = None) -> Decision:
        if state is None and problem is None:
            raise ValueError("pass a problem or a state")
        st = self._state(state if state is not None else problem)
        if objective and "objective" not in st.problem:
            st.problem = {**st.problem, "objective": objective}
        choice = choice or choice_for_objective(objective, st.previous_result is not None)
        decision = self.decider.decide(st, choice)
        self._issued[decision.id] = (choice, st)
        return decision


__all__ = [
    "CallableExecutor",
    "DecisionLoop",
    "EvidenceStore",
    "Executor",
    "PhysicsDecisionEngine",
    "ValidationRequiredError",
    "choice_for_objective",
]
