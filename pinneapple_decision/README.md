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

## From a problem as you write it: the adapter and `pp.decide`

The rules and constraints read canonical keys (`representation`, `reynolds`,
`n_simulations`, ...). `adapter.adapt_problem` maps a real
`pinneapple_problemdesign.ProblemSpec` or a free-form dict with common synonyms
to those keys, and every decision goes through it (`DecisionState.coerce`).
Each fact is **given** (the input had it, under its name or a synonym such as
`Re`, `steady`/`transient`, `mesh`, `dim`, `target`, `n_sims`), **inferred**
(by a written rule from another field, e.g. a NACA/airfoil/vehicle/CAD
geometry name -> body-fitted -> `unstructured_mesh`; the rule text is kept), or
**unknown** (left out of the problem, never filled with a default). Keys the
adapter does not recognize are kept for the LLM and listed as `unrecognized`.
The report is in `decision.diagnostics["problem_facts"]`.

The decision layer is also exposed on the main package, lazily
(`import pinneapple` does not import it; `pinneapple_decision` does not import
torch, so the first `pp.decide` does not load it either):

```python
import pinneapple as pp

d = pp.decide({"name": "airfoil_flow", "reynolds": 1e5, "steady": True,
               "geometry": "naca0012", "target": "pressure_field"},
              objective="minimize_prediction_error")
d.excluded        # {'fno': "must_support_geometry: fno supports ['structured_grid'], problem representation is 'unstructured_mesh'", 'pino': ...}
d.selected        # 'pinn'
d.probabilities   # {'pinn': 0.62, 'xpinn': 0.23, 'deeponet': 0.05, 'mesh_graph_net': 0.05, 'inverse_pinn': 0.05}
facts = d.diagnostics["problem_facts"]
facts["inferred"]["representation"]  # {'value': 'unstructured_mesh', 'rule': "geometry 'naca0012' names a NACA airfoil: ..."}
facts["given"]["time_dependent"]     # {'value': False, 'from': 'steady'}
sorted(facts["unknown"])  # ['dimension', 'geometry_varies', 'has_analytical_solution', 'has_reference_data',
                          #  'has_solver', 'is_inverse', 'n_simulations', 'needs_parameter_generalization']
```

Why PINN here and DeepONet in the quick example: this dict says nothing about
data, so `n_simulations` is unknown. `recommend_architecture` is then called
with its defaults (0 simulations, no parametric family) and the rationale says
so ("not known for this problem: [...]; ... That is the rule's default, not a
fact about the problem"). The Re >= 1e4 penalty on PINN still applies. Add
`"n_simulations": 200, "parametric": True` and the same call selects
`deeponet` (0.70). The adapter never assumes those values for you.

`pp.execute(decision)`, `pp.verify(result)` and `pp.run_tree(problem)` use the
same default engine; configure it with
`pp.decision_engine(executor=..., verifier=..., decider=..., store=...)`.

### PINN #17 did not converge: what is the next experiment?

```python
pp.decision_engine(executor=CallableExecutor(train_pinn_17))   # your training function
d = pp.decide({"description": "PINN #17, 2D lid-driven cavity", "reynolds": 100},
              objective="next training experiment")
d.selected                       # 'resampling' (no evidence yet: cheapest non-trivial baseline)
result = pp.execute(d)           # e.g. metrics residual=3e-2, diagnostics=["residual_localized"]
v = pp.verify(result)            # v.passed False, v.tags ['error_high', 'residual_high', 'residual_localized']
nxt = pp.decide(state={"problem": {...}, "previous_result": result, "verification": v},
                objective="next training experiment")
nxt.selected                     # 'adaptive_sampling' (the residual is localized; resampling already failed)
```

## Decision tree

`tree.py` holds a declarative tree that `DecisionLoop.run_tree(problem)` walks
(`pp.run_tree`, `PhysicsDecisionEngine.run_tree`). The default,
`physics_ai_tree()`:

```
has_analytical_solution? --yes--> analytical_baseline (execute) -> check_baseline (validate)
        |                                                   passed -> accept
        no                                                  failed -> has_reference_data?
        v
has_reference_data? --yes--> surrogate_model  (deeponet | fno | mesh_graph_net | pino) --+
        |                                                                                  |
        no -----------> physics_model    (pinn | xpinn | pino | inverse_pinn)  ------------+--> train (execute)
                                                                                                |
                                                   validate (verifier) <------------------------+
                                                   passed -> accept
                                                   failed -> next_experiment (training strategy) -> train
```

Nodes: `ChoiceNode` (a `PhysicsChoice`, edges per option), `ExecuteNode`,
`ValidateNode` (edge by `verification.passed`), `TerminalNode`. You can build
your own `DecisionTree(name, root, nodes)`.

**What is decided by a fact and what goes to the decider:**

| Node | Answered by | When |
|---|---|---|
| `has_analytical_solution` | the problem fact `has_analytical_solution` | when it is a boolean in the (adapted) problem |
| `has_reference_data` | the fact `has_reference_data` (given, or inferred from `n_simulations` / `ProblemSpec.data.sources`) | same |
| either of the two above | the decider (rules or LLM), marked `resolved_by="decider"`, `fact_status="unknown"` | when the fact is unknown. The rule backend then gives equal scores, plus one stated hint (a body-fitted geometry makes "no analytical solution" more likely). The answer is never written back as a fact |
| `surrogate_model`, `physics_model` | the decider, after constraints | always (this is a real choice) |
| `next_experiment` | the decider, reading the verification tags of the failed run | after a failed validation |

A fact answer is a `Decision` with `level=DecisionLevel.FACT`,
`backend="fact:<key>"`, `requires_validation=False` and probability 1.0 on the
fact: nothing was scored. Every step is logged in the `EvidenceStore` with
`decision.diagnostics["tree"]` (`node`, `resolved_by`, `fact`, `fact_status`),
and execution and verification fill the same record, so the store holds the
path. The executor gets the plan built along the path in
`problem["experiment"]` (`approach`, `model`, `training_strategy`).
`max_experiments` caps the executions. When it is reached the walk stops with
`outcome="budget_exhausted"`, and the next decision is already in the log.

```python
run = pp.run_tree({"description": "PINN #17", "reynolds": 100,
                   "has_analytical_solution": False, "has_reference_data": False},
                  max_experiments=3)
run.outcome             # 'accepted'
[s.node for s in run.path]
# ['has_analytical_solution', 'has_reference_data', 'physics_model', 'train', 'validate',
#  'next_experiment', 'train', 'validate', 'accept']
run.decided_by_fact()   # ['has_analytical_solution', 'has_reference_data']
```

## The Arena as a decision laboratory

`pinneapple_arena` normally trains every configured model and ranks them. In
decision mode it asks this engine which configured model to train first, trains
and evaluates it with its own code, verifies the result, records the evidence,
and asks again, until a result passes or the budget is spent. Models the engine
never chose are not trained.

```python
from pinneapple_arena import Arena, ArenaConfig

arena = Arena(ArenaConfig.from_yaml("benchmark.yaml"))
report = arena.run_decision(budget=2, thresholds={"rel_l2": 0.05, "residual": 1e-3},
                            evidence_path="outputs/decision_evidence.json")
report.ran           # e.g. ['PINN', 'DeepONet']
report.not_trained   # ['MGN']
report.stop_reason   # 'verified' | 'budget_exhausted' | 'no_feasible_option'
report.accepted      # the first model whose result passed, or None
```

or in the config: `decision: {enabled: true, budget: 2, thresholds: {...}}`,
then `Arena(cfg).run()` (UQ, inverse, figures and the summary then cover only
the trained models).

- **Choice**: `arena_model`, whose options are the configured model *names*.
  Each maps to its `ARCHITECTURE_CATALOG` family (`catalog_family`:
  vanilla_pinn/siren/... -> pinn, fno2d -> fno, meshgraphnet -> mesh_graph_net,
  ...) for geometry support and cost. Constraints: `must_support_geometry`,
  `within_budget`, `not_run_before` (a model already trained is not offered
  again).
- **Rules**: the `ModelSelector` score of the option's family, minus 1 for
  every model of the same family that already failed verification.
- **Problem facts**: only what the Arena knows. `has_analytical_solution` comes
  from `ArenaProblem.analytical` (a real call). `has_reference_data` and
  `has_reference_solution` are set because the Arena evaluates against the
  problem's reference data. The Reynolds number is read from `re` in the
  problem params. Everything else is unknown unless you pass
  `problem={"representation": ..., ...}`.
- **Execute**: `Arena._train_one` + `Arena._evaluate_one` (`ArenaExecutor`).
  Metrics: `rel_l2` (the mean relative L2 over the fields, the same number
  Arena ranks by), `residual` (the final PDE residual, PINN family only), and
  the per-field `L2_*`, `rel_*`, `Linf_*`. A training error gives
  `status="failed"`, which is never a pass.
- **Validate**: `ThresholdVerifier` by default, or any `Verifier`
  (`VeriPhysicsVerifier`...). A supervised model has no `residual`, so that check
  did not run: it is absent from `checks`, not passed.
- **Cost**: each step is a real training run. The tests use `poisson_2d` with
  5-epoch, 8-unit models (a few seconds on CPU).

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
| `adapt_problem(problem) -> AdaptedProblem` | `ProblemSpec` / free dict -> canonical problem + `given` / `inferred` / `unknown` / `unrecognized` |
| `decide(problem=None, objective=None, *, state=None, choice=None, engine=None) -> Decision` | module-level, also `pp.decide` |
| `execute(decision, problem=None, *, engine=None) -> ExecutionResult`, `verify(result, *, engine=None) -> Verification` | also `pp.execute`, `pp.verify` |
| `run_tree(problem, tree=None, *, max_experiments=3, engine=None) -> TreeRun` | also `pp.run_tree`; `DecisionLoop.run_tree(problem, tree=None, *, max_experiments=3)` |
| `decision_engine(executor=None, verifier=None, decider=None, store=None, *, reset=False)` | the shared default engine (`pp.decision_engine`) |
| `DecisionTree`, `ChoiceNode`, `ExecuteNode`, `ValidateNode`, `TerminalNode`, `physics_ai_tree()` | declarative tree |

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
- There is no generic executor in this package. The Arena executor
  (`pinneapple_arena.decision_mode.ArenaExecutor`) is the one real executor so
  far. It chooses among the models configured in the Arena, not among training
  strategies: the tree's `next_experiment` needs an executor that can apply a
  training strategy (yours, through `problem["experiment"]`).
- The adapter recognizes a fixed list of synonyms and geometry names
  (`adapter.py`). Anything else is unknown, not guessed. Free text such as
  "200 OpenFOAM runs" is not parsed into `n_simulations`.
- `import pinneapple` itself already imports torch on `main` (eager re-exports
  of the trainer, UQ, etc., which predate this package). The decision layer does
  not add to that: it is lazy and torch-free. `ModelSelector` loads
  `architecture_recommendation.py` on its own so that `pinneapple_analysis/__init__`
  (which imports torch) is not run.
- The fact-question rules (`has_analytical_solution`, `has_reference_data`)
  have almost no evidence to work with when the fact is unknown. Supplying the
  fact is always better than letting the decider answer.
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

## Extra model families (surrogates that are not ModelRegistry networks)

`pinneapple_analysis.verification.architecture_recommendation.EXTRA_CATALOG` adds four families that the
`ModelSelector` can now score. They are only recommended when the problem supplies the fact that justifies
them; with those facts absent the recommendation is exactly the previous one.

| Family | Recommended when | Implementation | Notes |
|---|---|---|---|
| `kpi_regressor` | `target_kind="kpi"` (scalar KPIs over a parameter family; works with tens of samples) | `sklearn.gaussian_process.GaussianProcessRegressor` | external (scikit-learn); geometry variation must be described by the parameters |
| `pod_rom` | field target + `fixed_topology=True` + many samples + parametric, geometry fixed | `pinneapple_neural.architectures.rom.pod.POD` | excluded by `must_support_geometry` when the geometry varies |
| `point_cloud_operator` | field target + varying geometry + many samples | Noether bridge (Transolver) | `license="research_only"`: excluded by the new `commercial_use_allowed` constraint when `PINNEAPPLE_COMMERCIAL_MODE=1` |
| `hybrid_surrogate_physics` | `has_physics_postprocessor=True` (target derived from the flow by an established physics model, e.g. erosion) | none packaged | composition of a field surrogate with the user's physics post-model; declared, not faked |

New adapter facts: `fixed_topology` (synonyms `same_mesh`, `fixed_mesh`) and `has_physics_postprocessor`.
