# pinneapple_decision — PhysicsDecisionEngine (MVP)

A probabilistic decision layer over the PINNeAPPle Physics AI pipeline.

The LLM **does not decide the physical result**. It decides **which physics
experiment to run next**. Every decision goes through hard constraints
(physics/geometry, resources, whether PINNeAPPle actually implements the
option), every executed result goes through a verifier (VeriPhysics when
available), and only verified results become evidence for the next decision:

```
OBSERVE -> UNDERSTAND -> DECIDE -> EXECUTE -> VALIDATE -> LEARN
problem    state        Decision  executor   verifier   EvidenceStore
```

Decisions are read as a **distribution over a closed set of options**, with no
text generation and no answer parsing. This is inspired by
[AnyJev](https://github.com/nokia-applied-research/AnyJev) (Nokia Applied
Research). See "Relation to AnyJev" below for what was taken and what was not.

## Why a separate package

`pinneapple_problemdesign` elicits a `ProblemSpec` from conversation and its
`method_selection` picks classical numerical solvers (LBM/FVM).
`pinneapple_analysis.verification` holds pre-training recommendations and
post-hoc checks. This layer sits across both and the executor/verifier, so it
lives in its own package. It is also light: importing it does not import torch.
Only `ModelSelector` loads `architecture_recommendation`, and it does so lazily.

## Quick example

```python
from pinneapple_decision import PhysicsDecisionEngine, CallableExecutor

airfoil = {
    "description": "2D airfoil, Re=1e5, surrogate over angle of attack",
    "reynolds": 1e5,
    "representation": "unstructured_mesh",   # checked by must_support_geometry
    "n_simulations": 200,
    "needs_parameter_generalization": True,
}

engine = PhysicsDecisionEngine(executor=CallableExecutor(run_experiment))  # your function
d = engine.decide(airfoil, objective="choose physics model")
d.selected        # 'deeponet'
d.probabilities   # {'deeponet': 0.70, 'mesh_graph_net': 0.09, 'inverse_pinn': 0.09, 'pinn': 0.06, 'xpinn': 0.06}
d.excluded        # {'fno': "must_support_geometry: fno supports ['structured_grid'], problem representation is 'unstructured_mesh'", 'pino': ...}
d.level           # DecisionLevel.L0
d.requires_validation      # True
d.calibrated_confidence    # None: there is no calibration at L0

# PINN #17 did not converge: what is the next experiment?
result = engine.execute(d)             # constraints are re-checked here
verification = engine.verify(result)   # ThresholdVerifier or VeriPhysicsVerifier
nxt = engine.decide(state={"problem": airfoil, "previous_result": result,
                           "verification": verification},
                    objective="next training experiment")
# e.g. result.diagnostics == ["residual_localized"] -> nxt.selected == "adaptive_sampling"
```

If you call `decide(state=...)` with a `previous_result` that has not been
verified, it raises `ValidationRequiredError`. `DecisionLoop.step()` / `run()`
run the whole cycle and stop when a verification passes.

## API

| Object | Role |
|---|---|
| `PhysicsChoice(name, question, options, constraints, option_info)` | a typed question with a closed option set |
| `PhysicsDecider(backend).decide(problem_or_state, choice)` | constraints, scoring, L0, then a `Decision` |
| `Decision(selected, probabilities, level, requires_validation=True, calibrated_confidence=None, excluded, rationale, diagnostics)` | `probability` is the probability of `selected`. It is **not** a confidence. |
| `ModelSelector`, `TrainingStrategySelector`, `ValidationStrategySelector` | the 3 MVP decisions: `.choice()` and `.rules()` |
| `ScoringBackend` (Protocol), `RuleBasedBackend`, `LogProbLLMBackend` | scores per option, in the order the options are shown |
| `DecisionLoop(executor, verifier, decider, store)` / `PhysicsDecisionEngine` | the loop and the high-level facade |
| `Executor`, `Verifier` (Protocols), `CallableExecutor`, `ThresholdVerifier`, `VeriPhysicsVerifier` | execution and verification |
| `EvidenceStore(path=None)` | evidence log in memory, optionally JSON on disk |

**Constraints** (`constraints.py`): `must_support_geometry`,
`requires_implementation`, `within_budget`, `not_failed_before`, or any
`Constraint(name, check)`. They run before scoring, so an excluded option is
never shown to the LLM, and again before execution. Every exclusion is kept
with its reason in `decision.excluded`.

**Options and where they are implemented:**

- Model: the keys of `architecture_recommendation.ARCHITECTURE_CATALOG`
  (pinn, xpinn, fno, deeponet, mesh_graph_net, pino, inverse_pinn). Rules
  delegate to `recommend_architecture()`.
- Training: uniform_sampling (`pinneapple_data.collocation.CollocationSampler`),
  resampling / adaptive_sampling (`AdaptiveCollocationSampler`), causal_training
  (`CausalPINNTrainer`), loss_weighting (`GradNormBalancer`), and curriculum,
  which has **no implementation** and is excluded with that reason.
- Validation: physics_residual (`PhysicsGuardrail`), reference_solution
  (`PhysicsCase.validate_against_benchmark`), conservation (`ConservationCheck`),
  uncertainty_quantification (`uq_predict`), and cross_validation, which has
  **no k-fold loop** and is excluded.

A test checks that every implementation path listed here exists.

## What L0 implements (exactly)

`debias.py`. No labels are used.

1. **Cyclic-shift marginalization (position bias).** With K options, the backend
   scores K rotations of the list, so each option appears once in every
   position. Each rotation goes through log-softmax. The default combination is
   `geometric`: average the log-probabilities per option, then softmax. If the
   position bias is additive in logit space, this removes it exactly. `mean`,
   the arithmetic mean of probabilities, is also available.
   `max_permutations` caps the number of rotations.
2. **Prior correction (label prior).** `p ∝ p / prior**strength`:
   - `content_free` (default): the prior is the rotation-averaged distribution
     on *neutral* states (`{}` and `{"description": "N/A"}`). This is the idea of
     Zhao et al. (2021), contextual calibration.
   - `batch`: the running mean of distributions on real states for the same
     choice and option set, applied only after `min_prior_n` states. This is the
     idea of Zhou et al. (2024), batch calibration.
   - `none`.

   Backends with `has_label_prior = False` skip step 2, and the diagnostics
   say so. The deterministic `RuleBasedBackend` is one of them: for it an empty
   problem is not a neutral input, and dividing by it would remove real
   knowledge.

`decision.diagnostics` records `n_rotations`, `rotation_argmaxes`,
`order_flip_rate`, `prior_method` and `prior`.

**What L0 does not do:** it does not calibrate. A distribution can still be
overconfident after L0. That is why the field is `probability`, and why
`calibrated_confidence` only exists at L1. `Decision` refuses to be built with
a `calibrated_confidence` at any other level.

## Backends

- `RuleBasedBackend` (default, offline, deterministic): it reads the problem and
  the evidence (verification tags, options that already failed). It is
  order-invariant by construction.
- `LogProbLLMBackend(logprob_fn)`: the options are listed with letters A, B, …
  and the score of each option is the log-probability of its letter as the next
  token after `Answer:`. `from_transformers(model_id)` builds it on a local
  Hugging Face model. This is an optional dependency, and you must check the
  model's own license. The existing `pinneapple_llm._dispatch.call_llm`
  returns text only (Anthropic / OpenAI chat / Ollama), so it is not used here.
  An OpenAI `top_logprobs` or Ollama backend can be added later behind the same
  `logprob_fn`.

## Verifiers

- `ThresholdVerifier({"residual": 1e-3, "rel_l2": 5e-2})`: a metric that was not
  reported is a check that did not run, not a pass. If no check ran, the result
  does not pass.
- `VeriPhysicsVerifier(min_trust=70, min_coverage=0.5)`: it reads a VeriPhysics
  `DecisionRecord` (`trust_score`, `trust_coverage`, `trustworthy`) from
  `result.artifacts["decision_record"]`, or a PINNeAPPle `PhysicsConfidenceScore`
  from `result.artifacts["physics_confidence"]`. It is duck-typed and does not
  import `veriphysics`. Without either artifact it falls back to the threshold
  verifier, and `source` says so.

## Limitations and what is missing

- **L1 is not implemented.** Calibration (temperature scaling, and later
  conformal abstention) needs labelled outcomes: pairs of (decision, did the
  verified experiment succeed). `EvidenceStore` already records these pairs, but
  nothing is fitted yet. AnyJev also has an L2 (a closed-form head on hidden
  states). It is out of scope here.
- The rule scores are small, explicit heuristics with sources. They are not
  validated performance predictions. The model rules inherit the limits of
  `recommend_architecture` (documented-scope match, no accuracy claim).
- `relative_cost` is a coarse 1–3 scale. It is not a GPU-hour estimate.
- There is no executor in this package. Wiring it to `pinneapple_arena`
  (train the chosen model, then `physics_aware_rank`) is the next step.
- Integration with PINNeAPPle-CFD E7 (the experiment loop of the SME CFD
  product) has not been done yet.
- The LLM backend has only been tested with fake log-prob functions. No real
  model is run in CI.

## Relation to AnyJev (source and license)

- AnyJev: <https://github.com/nokia-applied-research/AnyJev>, **Apache License 2.0**
  (checked on 2026-09-23, commit `a59a69e`). Authors: Jiamu Zhang, Tianze Yang,
  Yucheng Shi and Liang Wu (Nokia / Tencent Hunyuan).
- **No AnyJev code was copied or vendored**, and AnyJev is not a dependency.
  This package reimplements the *ideas* described in AnyJev's README and
  `docs/levels.md`: typed decisions read from next-token distributions, levels
  carried on each decision, cyclic-shift marginalization with geometric
  combination, contextual/batch prior correction, and L1 = calibration on
  labels. It does so with independent code and a physics-specific API
  (`PhysicsChoice`, constraints, verification loop).
- Underlying methods: Zhao et al. (2021) *Calibrate Before Use* (contextual
  calibration); Zhou et al. (2024) *Batch Calibration*; Zheng et al. (2024)
  *Large Language Models Are Not Robust Multiple Choice Selectors* (permutation
  debiasing).
