# Uncertainty and sensitivity studies on any model

Two questions every surrogate (or solver) raises before anyone uses its numbers: *how far can I trust this
prediction?* and *which parameters actually control it?* PINNeAPPle answers both on any model that maps an
`(n, d)` array of inputs to `(n, m)` outputs: a `pp.zoo` model, an ONNX graph, a solver wrapper or a closed form.

```python
import pinneapple as pp
from pinneapple_analysis.study_models import StudyModel

model = StudyModel.from_zoo("heatsink_forced_mlp", outputs=["r_base_k_w"])   # also from_onnx(bytes), builtin("ishigami")
X_cal, y_cal = model.reference("cal")                                         # the model's own calibration split
X_test, y_test = model.reference("test")                                      # held out: never used to calibrate

study = pp.conformal_study(y_cal[:, 0], model.predict(X_cal)[:, 0], y_test[:, 0], model.predict(X_test)[:, 0],
                           alpha=0.1, scale_cal=model.predict(X_cal)[:, 0], scale_test=model.predict(X_test)[:, 0])
print(study.coverage, study.coverage_ci)            # measured on the 900 test designs, with its Wilson interval
```

## Uncertainty (`pp.uncertainty`)

| Question | API |
|---|---|
| Interval with a coverage guarantee, measured on held-out data | `conformal_study(y_cal, pred_cal, y_test, pred_test, alpha, scale_cal=, scale_test=, group_by=)` |
| Uncertain inputs with distributions | `distributions.as_space([...])`: uniform, normal, truncnormal, lognormal (median, gsd), triangular |
| Propagate them through the model | `propagate_space(fn, space, n, method="mc" / "lhs" / "sobol", limits=)` (`pp.propagate_uncertainty`) |
| Cheap moments and Sobol indices for a smooth model with few inputs | `polynomial_chaos(fn, space, degree)` with its leave-one-out Q2 |
| Is this input inside the data the model learned from? | `ApplicabilityDomain.fit(X_ref, names, bounds).assess(X)`: inside / sparse / outside |
| Which cases need the high-fidelity solver? | `route_cases(status, pred, lower, upper, tolerance=, limits=)` + `routing_summary` |

Conformal intervals are valid on average over data like the calibration data. Two reports keep that honest: the
**reliability** list (measured coverage at several nominal levels) and the **per-group coverage** (`group_by=` an
input or the prediction), which shows where a constant-width interval under-covers. Normalised scores (a scale per
point: the prediction itself for a relative error, an ensemble spread) make the width follow the error.

The applicability gate uses the declared validity range (a model card's) and the k-nearest-neighbour distance in
standardised inputs: a point is **sparse** when its 5th neighbour is farther than for 99 % of the reference points
themselves (a hole in the data inside the range). The routing rule sends a case to the high-fidelity solver when it
is outside or sparse, when its interval is wider than the tolerance the decision needs, or when its interval
contains a decision limit.

## Sensitivity (`pp.sensitivity`)

| Question | API |
|---|---|
| Share of output variance per input, with confidence intervals | `sobol_indices(fn, space, n, second_order=True)` (`pp.sobol_indices`) |
| Same, from runs done elsewhere (OpenFOAM, CalculiX, a campaign) | `saltelli_design(space, n, seed=)` → run → `sobol_from_runs(space, n, run_ids, Y, seed=)` |
| From an existing dataset, no designed experiment | `first_order_given_data(X, y)` |
| Cheap screening of many inputs | `morris(fn, space, r)`: mu*, sigma, ``r (d + 1)`` evaluations |
| Shape of each effect, interactions | `effect_curves(fn, space)`: partial dependence + ICE |
| Local gradient | `local_derivatives(fn, x0, torch_fn=)`: autodiff for torch models, central differences otherwise |
| Which inputs can be fixed? | `freeze_recommendation(indices, threshold)` and the direct check `verify_freeze(fn, space, frozen)` |

The estimators are Saltelli (2010) for first order, Jansen (1999) for total and Saltelli (2002) for second order,
with percentile-bootstrap intervals. `tests/test_sensitivity.py` checks them against the exact indices of the
Ishigami and Sobol' G functions (`pinneapple_analysis.sensitivity.testfunctions`). An input is recommended for
freezing only when the **upper** bound of its total index is below the threshold for every output; an estimate
below the threshold with a wider interval is reported as *inconclusive* (more samples needed), never as freezable.

## Apps

- [Engineering Sensitivity Lab](sensitivity_lab.md) (sensitivity.pinneapple.org): Sobol, Morris, effect curves,
  derivatives and the freeze recommendation for a model, solver runs or a dataset.
- [Engineering Uncertainty Lab](uncertainty_lab.md) (uncertainty.pinneapple.org): intervals, applicability alerts,
  the routing rule and propagation for a model and its data.
