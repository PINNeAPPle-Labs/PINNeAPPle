"""The three MVP decisions: model, training strategy, validation strategy.

Each selector provides:

* ``choice(...)`` -> a :class:`PhysicsChoice` with option metadata
  (``OptionInfo``: real PINNeAPPle implementation path, supported geometry
  representations, coarse relative cost, literature source);
* ``rules(state, options)`` -> ``({option: score}, rationale)`` used by the
  deterministic :class:`~pinneapple_decision.backends.RuleBasedBackend`.
  Scores are logits: only differences matter; they are a function of the
  problem and the evidence only, never of the order the options are listed in.

Reuse, not duplication: the model rules delegate to the existing
``pinneapple_analysis.verification.architecture_recommendation.recommend_architecture``
(its catalog supplies names, registry keys and sources). ``method_selection``
in ``pinneapple_problemdesign`` picks *classical numerical solvers* (LBM/FVM),
a different axis, so it is not wrapped here.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .schema import DecisionState, OptionInfo, PhysicsChoice

Rules = Callable[[DecisionState, Sequence[str]], Tuple[Dict[str, float], List[str]]]

STRUCTURED = ("structured_grid",)
MESH_FREE = ("analytic", "point_cloud", "structured_grid", "unstructured_mesh")

# Geometry/cost metadata is the one thing ARCHITECTURE_CATALOG does not carry.
# Representations follow each catalog entry's own "when_to_use"/"weaknesses" text.
_MODEL_EXTRA = {
    "pinn": dict(representations=MESH_FREE, relative_cost=1),
    "xpinn": dict(representations=MESH_FREE, relative_cost=2),
    "fno": dict(representations=STRUCTURED, relative_cost=2),
    "deeponet": dict(representations=("point_cloud", "structured_grid", "unstructured_mesh"), relative_cost=2),
    "pino": dict(representations=STRUCTURED, relative_cost=3),
    "mesh_graph_net": dict(representations=("unstructured_mesh", "point_cloud"),
                           supports_varying_geometry=True, relative_cost=3),
    "inverse_pinn": dict(representations=MESH_FREE, relative_cost=2),
}

_FAILED_PENALTY = -2.0

_ARCH_MODULE = "pinneapple_analysis.verification.architecture_recommendation"


def _arch():
    """``architecture_recommendation`` without importing ``pinneapple_analysis/__init__``.

    That module is pure stdlib, but its package ``__init__`` imports torch (UQ,
    validation). To keep the decision layer torch-free (``pp.decide`` must not
    load torch), the file is loaded on its own when the package is not already
    imported. If that fails for any reason, the normal import is used.
    """
    import sys

    if _ARCH_MODULE in sys.modules:
        return sys.modules[_ARCH_MODULE]
    private = "pinneapple_decision._architecture_recommendation"
    if private in sys.modules:
        return sys.modules[private]
    try:
        import importlib.util
        from pathlib import Path

        pkg = importlib.util.find_spec("pinneapple_analysis")
        path = Path(list(pkg.submodule_search_locations)[0]) / "verification" / "architecture_recommendation.py"
        spec = importlib.util.spec_from_file_location(private, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[private] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        sys.modules.pop(private, None)
        import importlib

        return importlib.import_module(_ARCH_MODULE)


def _penalize_failed(state: DecisionState, choice: str, scores: Dict[str, float], why: List[str]) -> None:
    for opt in state.failed_options(choice):
        if opt in scores:
            scores[opt] += _FAILED_PENALTY
            why.append(f"{opt} already ran for {choice!r} and failed verification -> {_FAILED_PENALTY:+}")


# --------------------------------------------------------------------------- model

class ModelSelector:
    name = "model"
    question = "Which physics-AI model family should be trained next for this problem?"
    default_constraints = ("must_support_geometry", "within_budget")

    @staticmethod
    def option_info() -> Dict[str, OptionInfo]:
        ARCHITECTURE_CATALOG = _arch().ARCHITECTURE_CATALOG
        out = {}
        for key, c in ARCHITECTURE_CATALOG.items():
            extra = _MODEL_EXTRA.get(key, {})
            out[key] = OptionInfo(
                name=key,
                description=f"{c.name}: {c.when_to_use}",
                implementation=f"pinneapple_neural.architectures.registry.ModelRegistry:{c.registry_key}",
                source=c.source,
                **extra,
            )
        return out

    @classmethod
    def choice(cls, options: Optional[Sequence[str]] = None, constraints: Optional[Sequence] = None) -> PhysicsChoice:
        info = cls.option_info()
        opts = list(options or info)
        return PhysicsChoice(cls.name, cls.question, opts,
                             list(cls.default_constraints if constraints is None else constraints),
                             {o: info[o] for o in opts if o in info})

    @staticmethod
    def rules(state: DecisionState, options: Sequence[str]) -> Tuple[Dict[str, float], List[str]]:
        recommend_architecture = _arch().recommend_architecture
        p = state.problem
        rec = recommend_architecture(
            n_high_fidelity_simulations=int(p.get("n_simulations", 0)),
            has_lots_of_data=p.get("has_lots_of_data"),
            has_analytical_solver=bool(p.get("has_solver", False)),
            needs_parameter_generalization=bool(p.get("needs_parameter_generalization", False)),
            geometry_varies=bool(p.get("geometry_varies", False)),
            is_inverse_problem=bool(p.get("is_inverse", False)),
        )
        scores = {o: 0.0 for o in options}
        why = [f"recommend_architecture: {step}" for step in rec.decision_path]
        defaulted = [k for k in ("n_simulations", "has_solver", "needs_parameter_generalization",
                                 "geometry_varies", "is_inverse") if k not in p]
        if defaulted:
            why.append(f"not known for this problem: {defaulted}; recommend_architecture was called with its "
                       "defaults for them (0 simulations / False). That is the rule's default, not a fact "
                       "about the problem: provide them to change the recommendation")
        for rank, key in enumerate(rec.recommended):
            if key in scores:
                scores[key] += 3.0 if rank == 0 else max(2.0 - 0.25 * (rank - 1), 1.0)
        re_ = p.get("reynolds")
        if re_ is not None and float(re_) >= 1e4:
            for k in ("pinn", "xpinn"):
                if k in scores:
                    scores[k] -= 0.5
            why.append(f"Re={float(re_):.3g} >= 1e4: thin boundary layers are a documented PINN failure "
                       "mode (Krishnapriyan et al., 2021) -> pinn/xpinn -0.5")
        _penalize_failed(state, ModelSelector.name, scores, why)
        return scores, why


# --------------------------------------------------------------------------- training

_TRAINING_INFO = {
    "uniform_sampling": OptionInfo(
        "uniform_sampling", "Fixed uniform collocation points (baseline).",
        implementation="pinneapple_data.collocation.CollocationSampler", relative_cost=1,
        source="Raissi et al. (2019)"),
    "resampling": OptionInfo(
        "resampling", "Uniform random collocation redrawn periodically (non-adaptive).",
        implementation="pinneapple_neural.trainer.collocation.AdaptiveCollocationSampler "
                       "(AdaptiveCollocationConfig(residual_frac=0.0))", relative_cost=1,
        source="Wu et al. (2023), CMAME"),
    "adaptive_sampling": OptionInfo(
        "adaptive_sampling", "Residual-based adaptive collocation (RAR-D): more points where the residual is large.",
        implementation="pinneapple_neural.trainer.collocation.AdaptiveCollocationSampler", relative_cost=2,
        source="Lu et al. (2021) SIAM Rev.; Wu et al. (2023) CMAME"),
    "causal_training": OptionInfo(
        "causal_training", "Temporal causal weighting of the residual (earlier times first).",
        implementation="pinneapple_neural.trainer.causal.CausalPINNTrainer", relative_cost=2,
        source="Wang, Sankaran & Perdikaris (2024), CMAME"),
    "curriculum": OptionInfo(
        "curriculum", "Curriculum on a PDE coefficient (easy -> hard, e.g. low -> high Re).",
        implementation=None, relative_cost=2,
        source="Krishnapriyan et al. (2021), NeurIPS",
        notes="no coefficient-curriculum trainer in PINNeAPPle; TimeMarchingTrainer is a time curriculum only"),
    "loss_weighting": OptionInfo(
        "loss_weighting", "Adaptive loss-term weights (GradNorm / SoftAdapt / ReLoBRaLo).",
        implementation="pinneapple_neural.trainer.weight_scheduler.GradNormBalancer", relative_cost=1,
        source="Wang, Teng & Perdikaris (2021), SIAM J. Sci. Comput."),
}


class TrainingStrategySelector:
    name = "training_strategy"
    question = "Which training strategy should the next physics-AI experiment use?"
    default_constraints = ("requires_implementation", "within_budget")

    @classmethod
    def choice(cls, options: Optional[Sequence[str]] = None, constraints: Optional[Sequence] = None) -> PhysicsChoice:
        opts = list(options or _TRAINING_INFO)
        return PhysicsChoice(cls.name, cls.question, opts,
                             list(cls.default_constraints if constraints is None else constraints),
                             {o: _TRAINING_INFO[o] for o in opts if o in _TRAINING_INFO})

    @staticmethod
    def rules(state: DecisionState, options: Sequence[str]) -> Tuple[Dict[str, float], List[str]]:
        p, tags = state.problem, set(state.tags())
        base = {"uniform_sampling": 0.5, "resampling": 0.8, "adaptive_sampling": 0.5,
                "causal_training": 0.0, "curriculum": 0.0, "loss_weighting": 0.3}
        scores = {o: base.get(o, 0.0) for o in options}
        why = ["baseline: periodic random resampling is at least as good as fixed uniform points at the same "
               "cost (Wu et al., 2023)"]

        def bump(opt: str, v: float, reason: str) -> None:
            if opt in scores:
                scores[opt] += v
                why.append(f"{reason} -> {opt} {v:+}")

        if p.get("time_dependent"):
            bump("causal_training", 1.0, "time-dependent problem")
        if tags & {"late_time_error", "causality_violation"}:
            bump("causal_training", 2.5, "evidence: error grows with time")
        if tags & {"residual_localized", "sharp_gradient"}:
            bump("adaptive_sampling", 2.5, "evidence: residual concentrated in a small region")
        if p.get("sharp_features") or (p.get("reynolds") is not None and float(p["reynolds"]) >= 1e3):
            bump("adaptive_sampling", 1.0, "problem has sharp features / high Re")
        if tags & {"loss_imbalance", "bc_violation"}:
            bump("loss_weighting", 2.5, "evidence: loss terms imbalanced / BCs violated")
        if p.get("multiscale") or p.get("stiff"):
            bump("curriculum", 1.5, "multiscale / stiff problem")
        _penalize_failed(state, TrainingStrategySelector.name, scores, why)
        return scores, why


# --------------------------------------------------------------------------- validation

_VALIDATION_INFO = {
    "physics_residual": OptionInfo(
        "physics_residual", "PDE residual + dimensional checks on the trained model.",
        implementation="pinneapple_llm.guardrail.PhysicsGuardrail", relative_cost=1),
    "reference_solution": OptionInfo(
        "reference_solution", "Compare against an analytical / benchmark / solver reference.",
        implementation="pinneapple_data.physics_case.PhysicsCase.validate_against_benchmark", relative_cost=1),
    "conservation": OptionInfo(
        "conservation", "Check conservation of mass / momentum / energy.",
        implementation="pinneapple_analysis.validation.conservation.ConservationCheck", relative_cost=1),
    "uncertainty_quantification": OptionInfo(
        "uncertainty_quantification", "Epistemic/aleatoric UQ (MC dropout, ensembles, conformal).",
        implementation="pinneapple_analysis.uncertainty.uq_predict", relative_cost=2),
    "cross_validation": OptionInfo(
        "cross_validation", "K-fold cross-validation over simulations.",
        implementation=None, relative_cost=3,
        notes="only train/val splits exist (pinneapple_neural.trainer.splits); no k-fold loop"),
}


class ValidationStrategySelector:
    name = "validation_strategy"
    question = "Which validation should the next result go through?"
    default_constraints = ("requires_implementation",)

    @classmethod
    def choice(cls, options: Optional[Sequence[str]] = None, constraints: Optional[Sequence] = None) -> PhysicsChoice:
        opts = list(options or _VALIDATION_INFO)
        return PhysicsChoice(cls.name, cls.question, opts,
                             list(cls.default_constraints if constraints is None else constraints),
                             {o: _VALIDATION_INFO[o] for o in opts if o in _VALIDATION_INFO})

    @staticmethod
    def rules(state: DecisionState, options: Sequence[str]) -> Tuple[Dict[str, float], List[str]]:
        p = state.problem
        scores = {o: 0.0 for o in options}
        why: List[str] = []

        def bump(opt: str, v: float, reason: str) -> None:
            if opt in scores:
                scores[opt] += v
                why.append(f"{reason} -> {opt} {v:+}")

        bump("physics_residual", 1.0, "always applicable and cheap")
        if p.get("has_reference_solution"):
            bump("reference_solution", 2.5, "a reference solution exists")
        else:
            bump("reference_solution", -2.0, "no reference solution")
        if p.get("conserved_quantities"):
            bump("conservation", 2.0, f"conserved quantities declared: {p['conserved_quantities']}")
        if p.get("noisy_data") or "decision" in str(p.get("objective", "")).lower():
            bump("uncertainty_quantification", 1.5, "noisy data or a downstream engineering decision")
        if int(p.get("n_simulations", 0)) >= 20:
            bump("cross_validation", 1.0, ">= 20 simulations available")
        # Diversify evidence: a check that already passed adds less than an independent one.
        for ev in state.history:
            if (ev.decision.choice == ValidationStrategySelector.name and ev.verification is not None
                    and ev.verification.passed and ev.decision.selected in scores):
                bump(ev.decision.selected, -1.0, "already passed; prefer independent evidence")
        return scores, why


# --------------------------------------------------------------------------- factual questions

class FactQuestion:
    """A yes/no question whose answer is a *fact* about the problem (decision tree).

    The tree reads the answer from the problem when the fact is known
    (``DecisionLevel.FACT``). These rules only run when it is **not** known: the
    decider then has to choose under uncertainty, and the tree marks the step as
    ``resolved_by="decider"``. With no evidence either way the scores are equal
    (probability 0.5 each, ties broken alphabetically); ``hints`` add small,
    stated preferences. A rule answer is never written back as a fact.
    """

    options = ("yes", "no")

    def __init__(self, name: str, question: str, fact: str,
                 hints: Optional[Callable[[DecisionState], List[Tuple[str, float, str]]]] = None):
        self.name, self.question, self.fact, self.hints = name, question, fact, hints

    def choice(self, options: Optional[Sequence[str]] = None, constraints: Optional[Sequence] = None) -> PhysicsChoice:
        return PhysicsChoice(self.name, self.question, list(options or self.options), list(constraints or ()))

    def rules(self, state: DecisionState, options: Sequence[str]) -> Tuple[Dict[str, float], List[str]]:
        scores = {o: 0.0 for o in options}
        why = [f"fact {self.fact!r} is unknown for this problem: this is a decision under uncertainty, "
               "not a fact"]
        for opt, v, reason in (self.hints(state) if self.hints else []):
            if opt in scores:
                scores[opt] += v
                why.append(f"{reason} -> {opt} {v:+}")
        if len(why) == 1:
            why.append("no rule evidence either way: equal scores (tie broken alphabetically)")
        return scores, why


def _analytical_hints(state: DecisionState) -> List[Tuple[str, float, str]]:
    if state.problem.get("geometry_complexity") == "complex":
        return [("no", 1.0, "closed-form solutions are known for canonical domains, not for body-fitted "
                             "geometries (airfoil, vehicle, CAD)")]
    return []


ANALYTICAL_SOLUTION = FactQuestion(
    "has_analytical_solution", "Is there an analytical (closed-form) solution for this problem?",
    "has_analytical_solution", _analytical_hints)
REFERENCE_DATA = FactQuestion(
    "has_reference_data", "Is there a reference simulation (or measured data) to train a surrogate on?",
    "has_reference_data")

SELECTORS = {s.name: s for s in (ModelSelector, TrainingStrategySelector, ValidationStrategySelector,
                                 ANALYTICAL_SOLUTION, REFERENCE_DATA)}

__all__ = ["ANALYTICAL_SOLUTION", "FactQuestion", "ModelSelector", "REFERENCE_DATA", "SELECTORS",
           "TrainingStrategySelector", "ValidationStrategySelector"]
