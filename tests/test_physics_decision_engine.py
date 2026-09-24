"""Tests for pinneapple_decision (no network, no real LLM)."""
from __future__ import annotations

import json
import math
import random

import pytest

from pinneapple_decision import (
    CallableExecutor,
    ConstraintViolationError,
    Decision,
    DecisionLevel,
    DecisionLoop,
    EvidenceStore,
    ExecutionResult,
    LogProbLLMBackend,
    ModelSelector,
    NoFeasibleOptionError,
    PhysicsChoice,
    PhysicsDecider,
    PhysicsDecisionEngine,
    ThresholdVerifier,
    TrainingStrategySelector,
    ValidationRequiredError,
    ValidationStrategySelector,
    VeriPhysicsVerifier,
)
from pinneapple_decision.debias import rotations

AIRFOIL = {
    "description": "2D airfoil, Re=1e5, parametric surrogate over angle of attack",
    "geometry": "airfoil",
    "reynolds": 1e5,
    "representation": "unstructured_mesh",
    "n_simulations": 200,
    "needs_parameter_generalization": True,
}


class BiasedFakeLLM:
    """Fake log-prob backend: true preference + position bias + label prior.

    score(option at position i) = truth[option] + position_bias[i] + label_bias[option]
    On a neutral state the truth term is 0, so only the biases remain.
    """

    name = "fake-biased"
    has_label_prior = True

    def __init__(self, truth, position_bias=None, label_bias=None):
        self.truth = truth
        self.position_bias = position_bias or {}
        self.label_bias = label_bias or {}
        self.calls = 0

    def score(self, state, choice, options):
        self.calls += 1
        neutral = not state.problem or state.problem == {"description": "N/A"}
        return [
            (0.0 if neutral else self.truth[o]) + self.position_bias.get(i, 0.0) + self.label_bias.get(o, 0.0)
            for i, o in enumerate(options)
        ]

    def rationale(self, state, choice, options):
        return ["fake"]


def _toy_choice(options=("pinn", "fno", "deeponet")):
    return PhysicsChoice("toy", "Which model?", list(options))


# ---------------------------------------------------------------- distribution

@pytest.mark.parametrize("selector,problem", [
    (ModelSelector, AIRFOIL),
    (TrainingStrategySelector, {"time_dependent": True}),
    (ValidationStrategySelector, {"has_reference_solution": True, "conserved_quantities": ["mass"]}),
])
def test_distribution_sums_to_one(selector, problem):
    d = PhysicsDecider().decide(problem, selector.choice())
    assert math.isclose(sum(d.probabilities.values()), 1.0, abs_tol=1e-9)
    assert all(0.0 <= p <= 1.0 for p in d.probabilities.values())
    assert d.probability == d.probabilities[d.selected] == max(d.probabilities.values())


def test_l0_defaults_no_calibrated_confidence():
    d = PhysicsDecider().decide(AIRFOIL, ModelSelector.choice())
    assert d.level is DecisionLevel.L0
    assert d.requires_validation is True
    assert d.calibrated_confidence is None
    assert not hasattr(d, "confidence")
    with pytest.raises(ValueError, match="only defined at level L1"):
        Decision(choice="x", question="q", selected="a", probabilities={"a": 1.0},
                 level="L0", calibrated_confidence=0.9)


# ---------------------------------------------------------------- order invariance

def test_permutation_does_not_change_choice_deterministic_backend():
    base = ModelSelector.choice()
    ref = PhysicsDecider().decide(AIRFOIL, base)
    rng = random.Random(0)
    for _ in range(5):
        opts = list(base.options)
        rng.shuffle(opts)
        shuffled = PhysicsChoice(base.name, base.question, opts, base.constraints, base.option_info)
        d = PhysicsDecider().decide(AIRFOIL, shuffled)
        assert d.selected == ref.selected == "deeponet"
        for o, p in ref.probabilities.items():
            assert math.isclose(d.probabilities[o], p, rel_tol=1e-9)


def test_rotations_cover_every_position():
    rots = rotations(["a", "b", "c", "d"])
    assert len(rots) == 4
    for pos in range(4):
        assert {r[pos] for r in rots} == {"a", "b", "c", "d"}


# ---------------------------------------------------------------- debias

def test_l0_removes_position_bias():
    truth = {"pinn": 0.0, "fno": 1.0, "deeponet": 0.5}
    fake = BiasedFakeLLM(truth, position_bias={0: 3.0})  # strong "first option" bias
    raw = PhysicsDecider(fake, debias=False)
    l0 = PhysicsDecider(fake, prior="none")
    for order in (["pinn", "fno", "deeponet"], ["deeponet", "pinn", "fno"]):
        choice = _toy_choice(order)
        assert raw.decide({"x": 1}, choice).selected == order[0]  # raw follows position
        d = l0.decide({"x": 1}, choice)
        assert d.selected == "fno"
        assert d.level is DecisionLevel.L0
        # additive position bias is removed exactly by the geometric combination
        expected = {o: math.exp(v) for o, v in truth.items()}
        z = sum(expected.values())
        for o in truth:
            assert math.isclose(d.probabilities[o], expected[o] / z, rel_tol=1e-9)


def test_l0_removes_label_prior_with_content_free_context():
    truth = {"pinn": 0.0, "fno": 1.0, "deeponet": 0.5}
    fake = BiasedFakeLLM(truth, position_bias={0: 2.0}, label_bias={"pinn": 2.5})
    no_prior = PhysicsDecider(fake, prior="none").decide({"x": 1}, _toy_choice())
    assert no_prior.selected == "pinn"  # label prior wins without correction
    d = PhysicsDecider(fake, prior="content_free").decide({"x": 1}, _toy_choice())
    assert d.selected == "fno"
    assert d.diagnostics["prior_method"] == "content_free"


def test_batch_prior_waits_for_min_n():
    fake = BiasedFakeLLM({"pinn": 0.0, "fno": 1.0, "deeponet": 0.5}, label_bias={"pinn": 2.5})
    decider = PhysicsDecider(fake, prior="batch", min_prior_n=3)
    first = decider.decide({"x": 1}, _toy_choice())
    assert first.diagnostics["prior_method"].startswith("none")
    for _ in range(3):
        d = decider.decide({"x": 1}, _toy_choice())
    assert d.diagnostics["prior_method"] == "batch"


def test_llm_backend_prompt_and_scores_without_generation():
    seen = {}

    def logprob_fn(prompt, labels):
        seen["prompt"], seen["labels"] = prompt, list(labels)
        return [-1.0 - i for i in range(len(labels))]

    backend = LogProbLLMBackend(logprob_fn)
    d = PhysicsDecider(backend).decide(AIRFOIL, ModelSelector.choice())
    assert seen["labels"][:3] == ["A", "B", "C"]
    assert "Answer:" in seen["prompt"]
    shown = [ln.split(". ", 1)[1].split(" --")[0] for ln in seen["prompt"].split("Options:\n")[1].splitlines()
             if len(ln) > 3 and ln[1:3] == ". "]
    assert "fno" not in shown and "pino" not in shown  # excluded before scoring
    assert "deeponet" in shown
    assert math.isclose(sum(d.probabilities.values()), 1.0)


# ---------------------------------------------------------------- constraints

def test_constraint_removes_option_with_reason():
    d = PhysicsDecider().decide(AIRFOIL, ModelSelector.choice())
    assert "fno" in d.excluded and "pino" in d.excluded
    assert "structured_grid" in d.excluded["fno"] and "unstructured_mesh" in d.excluded["fno"]
    assert "fno" not in d.probabilities


def test_requires_implementation_excludes_unimplemented():
    d = PhysicsDecider().decide({}, TrainingStrategySelector.choice())
    assert d.excluded["curriculum"].startswith("requires_implementation")
    v = PhysicsDecider().decide({}, ValidationStrategySelector.choice())
    assert "cross_validation" in v.excluded


def test_no_feasible_option_raises_with_reasons():
    choice = PhysicsChoice("c", "q", ["a", "b"], ["requires_implementation"])
    with pytest.raises(NoFeasibleOptionError) as exc:
        PhysicsDecider().decide({}, choice)
    assert set(exc.value.excluded) == {"a", "b"}


def test_execute_rechecks_constraints():
    loop = DecisionLoop(executor=CallableExecutor(lambda d, p: {"status": "ok"}))
    choice = ModelSelector.choice()
    d = loop.decide_choice({**AIRFOIL, "representation": "structured_grid"}, choice)
    assert d.selected == "fno"
    with pytest.raises(ConstraintViolationError):
        loop.execute(d, problem=AIRFOIL)  # geometry became an unstructured mesh


def test_foreign_decision_is_not_executed():
    loop = DecisionLoop(executor=CallableExecutor(lambda d, p: {"status": "ok"}))
    d = PhysicsDecider().decide(AIRFOIL, ModelSelector.choice())
    with pytest.raises(ConstraintViolationError):
        loop.execute(d)


# ---------------------------------------------------------------- loop

def _pinn_executor(decision, problem):
    # adaptive sampling fixes the localized residual; anything else leaves it there.
    if decision.selected == "adaptive_sampling":
        return ExecutionResult("ok", metrics={"residual": 5e-4, "rel_l2": 0.02})
    return ExecutionResult("ok", metrics={"residual": 3e-2, "rel_l2": 0.2}, diagnostics=["residual_localized"])


def test_loop_uses_verification_in_next_decision(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.json")
    loop = DecisionLoop(CallableExecutor(_pinn_executor), ThresholdVerifier(), store=store)
    problem = {"description": "PINN #17, 2D lid-driven cavity", "reynolds": 100}
    choice = TrainingStrategySelector.choice()

    first = loop.step(problem, choice)
    assert first.decision.selected == "resampling"  # no evidence yet: cheap baseline
    assert first.verification.passed is False
    assert "residual_localized" in first.verification.tags

    second = loop.step(problem, choice)
    assert second.decision.selected == "adaptive_sampling"  # driven by the verification tags
    assert any("residual concentrated" in r for r in second.decision.rationale)
    assert any("failed verification" in r for r in second.decision.rationale)
    assert second.verification.passed is True

    saved = json.loads((tmp_path / "evidence.json").read_text())
    assert [e["decision"]["selected"] for e in saved] == ["resampling", "adaptive_sampling"]
    assert EvidenceStore(tmp_path / "evidence.json").records[1].verification.passed is True


def test_run_stops_when_verified():
    loop = DecisionLoop(CallableExecutor(_pinn_executor))
    evs = loop.run({"reynolds": 100}, TrainingStrategySelector.choice(), max_steps=5)
    assert len(evs) == 2 and evs[-1].verification.passed


def test_high_level_api_requires_verification_before_next_decision():
    engine = PhysicsDecisionEngine(executor=CallableExecutor(_pinn_executor))
    problem = {"description": "PINN #17", "reynolds": 100}
    d1 = engine.decide(problem, objective="next training experiment")
    assert d1.choice == "training_strategy"
    result = engine.execute(d1)
    with pytest.raises(ValidationRequiredError):
        engine.decide(state={"problem": problem, "previous_result": result})
    verification = engine.verify(result)
    d2 = engine.decide(state={"problem": problem, "previous_result": result, "verification": verification})
    assert d2.selected == "adaptive_sampling"


def test_objective_routing():
    engine = PhysicsDecisionEngine()
    assert engine.decide(AIRFOIL, objective="choose physics model").choice == "model"
    assert engine.decide(AIRFOIL, objective="validate result").choice == "validation_strategy"


# ---------------------------------------------------------------- verifiers

def test_threshold_verifier_missing_metric_is_not_a_pass():
    v = ThresholdVerifier({"residual": 1e-3}).verify(ExecutionResult("ok", metrics={}))
    assert v.passed is False and v.checks == {}


def test_veriphysics_verifier_reads_decision_record_duck_typed():
    class Rec:  # same attributes as veriphysics.orchestrator.decision.DecisionRecord
        trust_score, trust_coverage, trustworthy = 96.0, 0.5, True

    ok = VeriPhysicsVerifier().verify(ExecutionResult("ok", artifacts={"decision_record": Rec()}))
    assert ok.passed and ok.source == "veriphysics.DecisionRecord"
    Rec.trust_score = 40.0
    bad = VeriPhysicsVerifier().verify(ExecutionResult("ok", artifacts={"decision_record": Rec()}))
    assert not bad.passed and "low_trust" in bad.tags
    fb = VeriPhysicsVerifier().verify(ExecutionResult("ok", metrics={"residual": 1e-5}))
    assert fb.source.startswith("fallback:threshold")


# ---------------------------------------------------------------- catalog honesty

@pytest.mark.parametrize("selector", [TrainingStrategySelector, ValidationStrategySelector])
def test_declared_implementations_exist(selector):
    import importlib

    for opt, info in selector.choice().option_info.items():
        if not info.implementation:
            continue
        path = info.implementation.split(" ")[0]
        parts = path.split(".")
        for i in range(len(parts) - 1, 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:i]))
            except ImportError:
                continue
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            break
        else:
            pytest.fail(f"{opt}: cannot import {path}")


def test_model_registry_keys_exist():
    from pinneapple_neural.architectures.registry import ModelRegistry

    names = set(ModelRegistry.list()) if hasattr(ModelRegistry, "list") else None
    if names is None:
        pytest.skip("ModelRegistry has no list()")
    for opt, info in ModelSelector.choice().option_info.items():
        assert info.implementation.split(":")[1] in names, opt
