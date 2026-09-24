"""Arena as a decision laboratory: the decision engine picks which models to train.

Default Arena mode trains every configured model and ranks them. Decision mode
(``Arena.run_decision(budget=...)`` or ``decision: {enabled: true}`` in the
config) instead runs the ``pinneapple_decision`` loop over the configured
models::

    DECIDE   which configured model to train next  (PhysicsChoice "arena_model";
             options = model names; constraints: geometry support, budget,
             not already trained in this session)
    EXECUTE  Arena._train_one + Arena._evaluate_one  (ArenaExecutor)
    VALIDATE ThresholdVerifier on rel_l2 (mean relative L2 over fields) and, for
             PINN-family models, residual (final PDE residual); any Verifier works
    LEARN    EvidenceStore; the next decision reads the verification tags and
             penalizes the family of a model that failed

and stops when a result passes verification (``stop_when_passed``) or after
``budget`` trained models. Models that were never chosen are not trained.

The engine decides *which experiment to run*; the metrics come from the Arena's
own training and evaluation against the problem's reference data. A metric the
Arena did not produce (e.g. ``residual`` for a supervised model) is a check that
did not run, not a pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pinneapple_decision import (
    Constraint,
    DecisionLoop,
    DecisionState,
    Evidence,
    EvidenceStore,
    ExecutionResult,
    NoFeasibleOptionError,
    OptionInfo,
    PhysicsChoice,
    PhysicsDecider,
    RuleBasedBackend,
    ThresholdVerifier,
)
from pinneapple_decision.selectors import SELECTORS, ModelSelector

from .model_factory import _GRAPH_TYPES, is_inverse_model, is_pinn_model

CHOICE_NAME = "arena_model"
QUESTION = "Which configured Arena model should be trained and evaluated next?"
_FAMILY_PENALTY = -1.0


# --------------------------------------------------------------------------- families

def catalog_family(model_type: str) -> Optional[str]:
    """Map an Arena model type to an ``ARCHITECTURE_CATALOG`` key, or ``None``."""
    from .config import ModelConfig

    t = model_type.lower().strip()
    cfg = ModelConfig(name=t, type=t)
    if is_inverse_model(cfg):
        return "inverse_pinn"
    if t == "xpinn":
        return "xpinn"
    if t in ("pino", "physics_informed_neural_operator"):
        return "pino"
    if t in ("fno", "fno2d", "fourier", "fourier_neural_operator"):
        return "fno"
    if t in ("deeponet", "multiscale_deeponet"):
        return "deeponet"
    if t in _GRAPH_TYPES:
        return "mesh_graph_net"
    if is_pinn_model(cfg):
        return "pinn"
    return None


def _option_info(arena: Any) -> Tuple[Dict[str, OptionInfo], Dict[str, Optional[str]]]:
    catalog = ModelSelector.option_info()
    infos: Dict[str, OptionInfo] = {}
    families: Dict[str, Optional[str]] = {}
    for m in arena.cfg.models:
        fam = catalog_family(m.type)
        families[m.name] = fam
        impl = f"pinneapple_arena.Arena._train_one (type={m.type})"
        if fam and fam in catalog:
            c = catalog[fam]
            infos[m.name] = OptionInfo(
                name=m.name, description=f"{m.type} ({fam}): {c.description}", implementation=impl,
                representations=c.representations, supports_varying_geometry=c.supports_varying_geometry,
                relative_cost=c.relative_cost, source=c.source)
        else:
            infos[m.name] = OptionInfo(name=m.name, description=f"{m.type} (no catalog family)",
                                       implementation=impl, notes="no ARCHITECTURE_CATALOG entry")
    return infos, families


# --------------------------------------------------------------------------- choice, constraint, rules

def _already_trained(info: OptionInfo, state: DecisionState, choice: PhysicsChoice) -> Optional[str]:
    for ev in state.history:
        if ev.decision.choice == choice.name and ev.decision.selected == info.name and ev.result is not None:
            return f"not_run_before: {info.name} was already trained (its result is in the evidence store)"
    return None


NOT_RUN_BEFORE = Constraint("not_run_before", _already_trained)


def arena_choice(arena: Any) -> Tuple[PhysicsChoice, Dict[str, Optional[str]]]:
    infos, families = _option_info(arena)
    names = [m.name for m in arena.cfg.models]
    if len(names) < 2:
        raise ValueError("decision mode needs at least 2 configured models to choose from")
    choice = PhysicsChoice(CHOICE_NAME, QUESTION, names,
                           ["must_support_geometry", "within_budget", NOT_RUN_BEFORE], infos)
    return choice, families


def arena_model_rules(families: Dict[str, Optional[str]]):
    """Rules for the ``arena_model`` choice: the ``ModelSelector`` score of each
    option's catalog family, minus a penalty for families that already failed."""

    def rules(state: DecisionState, options: Sequence[str]):
        fams = sorted({families.get(o) for o in options if families.get(o)})
        fam_scores: Dict[str, float] = {}
        why: List[str] = []
        if fams:
            fam_scores, why = ModelSelector.rules(state, fams)
        scores = {o: fam_scores.get(families.get(o) or "", 0.0) for o in options}
        for o in options:
            if not families.get(o):
                why.append(f"{o}: no catalog family -> score 0")
        failed = [ev for ev in state.history
                  if ev.decision.choice == CHOICE_NAME and ev.verification is not None and not ev.verification.passed]
        for ev in failed:
            fam = families.get(ev.decision.selected)
            for o in options:
                if fam and families.get(o) == fam:
                    scores[o] += _FAMILY_PENALTY
                    why.append(f"{o} is in family {fam!r}, like {ev.decision.selected} which failed verification "
                               f"({', '.join(ev.verification.tags) or 'no tags'}) -> {_FAMILY_PENALTY:+}")
        return scores, why

    return rules


def default_decider(families: Dict[str, Optional[str]]) -> PhysicsDecider:
    rules = {k: s.rules for k, s in SELECTORS.items()}
    rules[CHOICE_NAME] = arena_model_rules(families)
    return PhysicsDecider(RuleBasedBackend(rules))


# --------------------------------------------------------------------------- problem facts

def arena_problem_facts(arena: Any, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Facts the Arena actually knows about its problem. Nothing else is filled in:
    the adapter lists the rest as unknown. ``overrides`` (e.g. a representation or
    Reynolds number) are given facts and win."""
    p: Dict[str, Any] = {}
    pc = arena.cfg.problem
    if arena.cfg.dataset is not None and arena.cfg.dataset.dataset_id:
        p["description"] = f"pinneapple_data dataset {arena.cfg.dataset.dataset_id}"
        p["has_reference_data"] = True
    else:
        prob = arena._problem
        if prob is not None:
            p["description"] = prob.description or prob.name
            if prob.domain:
                p["domain_context"] = prob.domain
            try:
                pts = [np.array([0.5])] * int(prob.input_dim)
                has_analytic = prob.analytical(*pts, **pc.params) is not None
                p["has_analytical_solution"] = has_analytic
            except Exception:
                pass  # left unknown
            # Arena always evaluates against the problem's reference data (Y_eval) and
            # trains data-driven models on supervised samples of it.
            p["has_reference_data"] = True
            p["has_reference_solution"] = True
        for key in ("re", "reynolds", "Re"):
            if key in pc.params:
                p["reynolds"] = pc.params[key]
                break
    p.update(overrides or {})
    return p


# --------------------------------------------------------------------------- executor

class ArenaExecutor:
    """Executor: train + evaluate the chosen configured model with the Arena's own code."""

    def __init__(self, arena: Any):
        self.arena = arena

    def execute(self, decision: Any, problem: Dict[str, Any]) -> ExecutionResult:
        from .arena import _accuracy_score

        arena = self.arena
        name = decision.selected
        try:
            mcfg = arena._mcfg_by_name(name)
            print(f"\n[Arena/decision] Training  {name}  (type={mcfg.type}, P={decision.probability:.2f})")
            tres = arena._train_one(mcfg)
            eres = arena._evaluate_one(tres)
        except Exception as e:  # a failed run is evidence too, never a pass
            return ExecutionResult("failed", diagnostics=["execution_error"],
                                   artifacts={"arena_model": name, "error": repr(e)})
        arena._train_results.append(tres)
        arena._eval_results.append(eres)

        metrics: Dict[str, float] = {"train_time": float(tres.train_time)}
        rel = _accuracy_score(eres, arena._data["field_names"])
        if not math.isnan(rel):
            metrics["rel_l2"] = rel
        if tres.physics_residual is not None:
            metrics["residual"] = float(tres.physics_residual)
        for k, v in eres["metrics"].items():
            metrics[k] = float(v)
        return ExecutionResult("ok", metrics=metrics, artifacts={"arena_model": name, "model_type": mcfg.type})


# --------------------------------------------------------------------------- loop

@dataclass
class ArenaDecisionReport:
    budget: int
    stop_reason: str  # "verified" | "budget_exhausted" | "no_feasible_option"
    ran: List[str]
    not_trained: List[str]
    accepted: Optional[str]
    ranking: List[Tuple[str, float]]
    evidence: List[Evidence]
    facts: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Decision mode: {len(self.ran)}/{self.budget} models trained, stop: {self.stop_reason}"]
        for ev in self.evidence:
            v = ev.verification
            lines.append(f"  {ev.decision.selected:<22} P={ev.decision.probability:.2f}  "
                         f"passed={None if v is None else v.passed}  checks={None if v is None else v.checks}")
        lines.append(f"  accepted: {self.accepted}")
        lines.append(f"  not trained: {self.not_trained}")
        return "\n".join(lines)


def run_decision_mode(arena: Any, *, budget: int = 3, decider: Optional[PhysicsDecider] = None,
                      verifier: Any = None, store: Optional[EvidenceStore] = None,
                      problem: Optional[Dict[str, Any]] = None, stop_when_passed: bool = True,
                      thresholds: Optional[Dict[str, float]] = None,
                      evidence_path: Optional[str] = None) -> ArenaDecisionReport:
    """Run the decision loop over ``arena``'s configured models (see module docstring)."""
    from .arena import rank_by_accuracy

    if budget < 1:
        raise ValueError("budget must be >= 1")
    if arena._data is None:
        arena._prepare_data()
    arena._train_results, arena._eval_results = [], []

    choice, families = arena_choice(arena)
    loop = DecisionLoop(executor=ArenaExecutor(arena),
                        verifier=verifier or ThresholdVerifier(thresholds),
                        decider=decider or default_decider(families),
                        store=store or EvidenceStore(evidence_path))
    facts_problem = arena_problem_facts(arena, problem)

    run_ids: List[str] = []
    last: Optional[Evidence] = None
    stop_reason = "budget_exhausted"
    facts: Dict[str, Any] = {}
    for i in range(budget):
        state = DecisionState.from_problem(
            facts_problem,
            previous_result=last.result if last else None,
            verification=last.verification if last else None,
            history=loop.store.history(),
        )
        facts = state.facts
        try:
            decision = loop.decide_choice(state, choice)
        except NoFeasibleOptionError:
            stop_reason = "no_feasible_option"
            break
        decision.diagnostics["arena"] = {"experiment": i + 1, "budget": budget}
        result = loop.execute(decision)
        loop.verify(result)
        last = loop.store.find(decision.id)
        run_ids.append(decision.id)
        if stop_when_passed and last.verification is not None and last.verification.passed:
            stop_reason = "verified"
            break
    loop.store.save()

    evidence = [loop.store.find(i) for i in run_ids]
    ran = [ev.decision.selected for ev in evidence]
    accepted = next((ev.decision.selected for ev in evidence if ev.verification and ev.verification.passed), None)
    ranking = rank_by_accuracy(arena._train_results, arena._eval_results, arena._data["field_names"])
    report = ArenaDecisionReport(budget, stop_reason, ran, [m.name for m in arena.cfg.models if m.name not in ran],
                                 accepted, ranking, evidence, facts)
    print("\n" + report.summary())
    return report


__all__ = [
    "ArenaDecisionReport",
    "ArenaExecutor",
    "NOT_RUN_BEFORE",
    "arena_choice",
    "arena_model_rules",
    "arena_problem_facts",
    "catalog_family",
    "default_decider",
    "run_decision_mode",
]
