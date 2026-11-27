# The trust of a prediction, as one object

The library already measures trust in several places, each in its own vocabulary: `PhysicsEnsemble` carries the
residual lookahead and the check on the readings, `pinneapple_physics.weather.evaluate` measures the useful and the
no-skill horizon, `uncertainty.calibration` measures coverage, `pinneapple_decision.verifiers` reports pass or fail
per check, and `validation.rollout` measures when a rollout diverges. A consumer — a digital twin, an app, a signed
report — had to know all five to answer one question.

`pp.trust(model, data, problem)` answers it once. It returns a `Trust`
(`pinneapple_analysis.trust.record`, schema `pinneapple.trust/1`): the record a prediction carries and a consumer
reads, with `to_json()`, `from_json()`, `badge()`, `badge_text()` and `badge_reason()`.

```python
import numpy as np
import pinneapple as pp
from pinneapple_analysis.trust.adapters import calibration_uncertainty, from_ensemble, weather_horizon
from pinneapple_analysis.trust.trust_gate import TrustGate
from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable

def residual(prediction, query):                       # the residual this ensemble already evaluates
    return float(abs(np.mean(prediction) - query["u"]) / abs(query["u"]))

experts = [from_callable("fno", lambda q: np.full(8, q["u"]), kind="operator"),
           from_callable("gnn", lambda q: np.full(8, 1.001 * q["u"]), kind="graph")]
ensemble = PhysicsEnsemble(experts, residual_fn=residual, residual_lookahead=1.0,
                           residual_baseline={"fno": 0.05, "gnn": 0.05},   # each expert's in-domain residual level
                           measurement_check=lambda q: False)             # the readings passed their check

t = pp.trust(
    ensemble, {"u": 1.0},
    gate=TrustGate().fit_from_bounds({"u": (0.0, 2.0)}),                   # the training region, as a box
    domain_points=np.array([[1.0]]),                                       # the point to score against it
    horizon=weather_horizon({"useful_hours": 120.0, "no_skill_hours": 240.0}, requested_lead=48.0),
)
t = t.with_(uncertainty=calibration_uncertainty(np.zeros(50), np.zeros(50), np.full(50, 10.0), level=0.9))

t.badge()        # 'good'
t.badge_reason() # 'good: all 6 gates measured and passing'  (the 6: domain, horizon, residual:fno/pde,
                 #  residual:gnn/pde, measurement, calibration)
print(t.to_json())
```

## Every field is a number with its source, or an explicit absence

The rule the issue asks for is enforced in the constructor, not in a comment. Each field is a `TrustValue`:

* a **measured** value carries `value`, `unit`, `source` (where the number came from) and, when it is judged
  against one, a `reference` **with its own** `reference_source`;
* a **`not_measured`** value carries the `reason` — and no value.

A measured value without a `source`, a `not_measured` carrying a value, and a reference without its own source are
all refused at construction. A field nobody measured is never written as `0`.

```json
{
  "schema": "pinneapple.trust/1",
  "kind": "ensemble",
  "badge": "good",
  "in_domain": {
    "status": "measured",
    "value": 1.0,
    "unit": "[-]",
    "source": "pinneapple_analysis.trust.trust_gate.TrustGate.ood_p_values",
    "method": "domain_score 1 >= 1: every query point inside the 99 % region of the training distribution"
  },
  "horizon": {
    "unit": "h",
    "useful":   {"status": "measured", "value": 120.0, "source": "pinneapple_physics.weather.evaluate.horizons"},
    "no_skill": {"status": "measured", "value": 240.0, "source": "pinneapple_physics.weather.evaluate.horizons"},
    "requested": {"status": "measured", "value": 48.0, "source": "the lead this prediction is asked to cover"},
    "covers":   {"status": "measured", "value": 1.0}
  },
  "residuals": [{"law": "pde", "expert": "fno", "scope": "whole field",
                 "value": {"status": "measured", "value": 0.0, "reference": 0.05,
                           "reference_source": "pinneapple_physics.ensemble.residual_baseline: the median residual of this expert on queries from its own domain"}}],
  "provenance": {"model": "PhysicsEnsemble(fno, gnn)", "model_fingerprint": "0c753465…",
                 "data_fingerprint": "7987413b…", "library_version": "pinneapple 0.8.0", "created": "…"}
}
```

The residual is judged against the model's **own** in-domain level, not against a fixed tolerance: that is what
makes "0.001" a pass for one expert and a failure for another.

## The badge is a decision, not a score

There is no weighted average, because two models can be "fading" for opposite reasons and the average hides both.
The rule, in order:

| badge | when |
|---|---|
| `no_skill` | a measured gate proves the prediction carries no information: the input is outside the training distribution (`in_domain` = 0), or the requested lead is past the measured no-skill lead |
| `fading` | nothing proves no skill, but at least one measured gate fails (a residual above its in-domain level, a suspect reading, coverage below the nominal level) **or** a gate was not measured |
| `good` | every gate the object carries was measured and passed |

A missing check never counts as a pass. `badge_reason()` names each gate that decided it, its number and its
source, so the sentence is auditable:

```
fading: domain unmeasured — no TrustGate was given and the problem declares no domain_bounds, so the input was
never tested against the training distribution; horizon unmeasured — ...; calibration unmeasured — ...
```

"Fading" because a field was not measured is not a failed prediction: it is the object refusing to claim more than
it measured.

## Adapters read what is already measured

None of them recompute anything: each one reads the number from the piece that produces it, and records that piece
as the `source`.

| adapter | reads |
|---|---|
| `from_ensemble` (the body of `pp.trust` for an ensemble) | `PhysicsEnsemble.predict`: residuals per expert, spread, conformal interval, `measurement_check` |
| `domain_from_gate` | `trust_gate.TrustGate.ood_p_values` on the points you pass |
| `weather_horizon` | `pinneapple_physics.weather.evaluate.horizons` (useful and no-skill lead) |
| `rollout_horizon` | `validation.rollout.divergence_time` |
| `calibration_uncertainty` | `uncertainty.calibration.CalibrationMetrics` (coverage, width, ECE) |
| `verification_measurement` | `pinneapple_decision.verifiers`: each named check as a 1/0 with its own source |

## Three kinds of prediction, one record

`pp.trust` reads three kinds of model, and the three records below are the issue's "done when": a weather forecast, an
online ensemble and a CFD surrogate, emitting **the same `Trust` JSON**. Not three look-alikes — every one of them ends
on the same call, `pinneapple_analysis.trust.adapters.from_ensemble`, with `kind` saying which of the three the
prediction is, and the document each writes is `Trust.to_json` and nothing else.

| producer | the prediction | `kind` | what measures it |
|---|---|---|---|
| `pinneapple_analysis.trust.forecast_skill.weather_trust` | a testbed forecast, twelve initialisations of twenty 6-hour leads | `"weather"` | `weather.evaluate.horizons` (ACC 0.6, climatology RMSE) |
| `pinneapple_analysis.trust.producers.ensemble_trust` | two experts combined online, one exact and one calibrated near `u = 1` | `"ensemble"` | the ensemble's residual lookahead and `residual_baseline` |
| `pinneapple_analysis.trust.producers.cfd_surrogate_trust` | a POD surrogate over steady Navier-Stokes (Kovasznay), asked at a Reynolds number it was fitted for and at one it was not | `"surrogate"` | the momentum residual of the field it emits |

```bash
python scripts/trust_three_kinds.py                      # writes results/trust_three_kinds/{weather,ensemble,surrogate}.json
```

The three files in `results/trust_three_kinds/` are the complete documents, validated against the schema as they are
written. Here is each of them cut to the same depth — every key shown is a key of all three:

```json
{"schema": "pinneapple.trust/1", "kind": "weather", "badge": "fading",
 "in_domain": {"status": "measured", "value": 1.0,
               "source": "pinneapple_analysis.trust.trust_gate.TrustGate.ood_p_values"},
 "horizon": {"useful": "the anomaly correlation 0.6 was never crossed within the leads the forecast was scored on, …",
             "requested": 48.0},
 "residuals": [{"expert": "persistence",             "value": 1.0,       "reference": 1.0},
               {"expert": "semi-lagrangian-linear",  "value": 0.0301800, "reference": 0.0289540},
               {"expert": "semi-lagrangian-cubic",   "value": 0.0264648, "reference": 0.0251716}],
 "uncertainty": {"level": 0.9, "coverage": 0.9950242, "width": 0.0390698},
 "provenance": {"model": "PhysicsEnsemble(persistence, semi-lagrangian-linear, semi-lagrangian-cubic)",
                "model_fingerprint": "f9cfdc71f7fb…"}}
```

```json
{"schema": "pinneapple.trust/1", "kind": "ensemble", "badge": "fading",
 "in_domain": {"status": "measured", "value": 1.0,
               "source": "pinneapple_analysis.trust.trust_gate.TrustGate.ood_p_values"},
 "horizon": {"useful": "an ensemble does not measure a horizon: pass one from weather.evaluate.horizons …",
             "requested": null},
 "residuals": [{"expert": "fno", "value": 0.0, "reference": 0.0},
               {"expert": "gnn", "value": 0.0, "reference": 0.0081250}],
 "uncertainty": {"level": 0.9, "coverage": 1.0, "width": 0.0008705},
 "provenance": {"model": "PhysicsEnsemble(fno, gnn)", "model_fingerprint": "f33b06fa19a2…"}}
```

```json
{"schema": "pinneapple.trust/1", "kind": "surrogate", "badge": "fading",
 "in_domain": {"status": "measured", "value": 1.0,
               "source": "pinneapple_analysis.trust.trust_gate.TrustGate.ood_p_values"},
 "horizon": {"useful": "an ensemble does not measure a horizon: pass one from weather.evaluate.horizons …",
             "requested": null},
 "residuals": [{"expert": "pod-3", "value": 0.0188121, "reference": 0.0108835},
               {"expert": "pod-6", "value": 0.0188182, "reference": 0.0108819}],
 "uncertainty": {"level": 0.9, "coverage": 0.3184317, "width": 0.0000210},
 "provenance": {"model": "PhysicsEnsemble(pod-3, pod-6)", "model_fingerprint": "86593ccb7f4d…"}}
```

The same keys, and the same sentence of `badge_reason`, come out of all three: "fading", because a field nobody
measured is not a pass. What differs is only the case's own inputs (`provenance.inputs`: `i0`/`steps`, `u`, `re`/`n`)
and the length of the per-expert lists — three experts for the weather record, two for each of the others.

### The numbers, and where each one was measured

| record | residual | its in-domain level | coverage (nominal 0.9) | badge |
|---|---|---|---|---|
| weather | 0.030180 (linear), 0.026465 (cubic), 1.0 (persistence) | 0.028954, 0.025172, 1.0 | 0.995 | `fading` |
| ensemble | 0.0 (`fno`), 0.0 (`gnn`) | 0.0, 0.008125 | 1.0 | `fading` |
| CFD surrogate, Re = 70 | 0.018812 (`pod-3`), 0.018818 (`pod-6`) | 0.010884, 0.010882 | 0.318 | `fading` |
| CFD surrogate, Re = 200 | 0.184552, 0.184545 | 0.010884, 0.010882 | — | `no_skill` |

* **weather** — the residual is `domain_checks.advection_residual` on the forecast (the part of the change the wind
  does not explain) and the level is its median over the eleven initialisations the scored case was **held out of**.
  The case is the hardest of the twelve, so it sits 4 % above the level and the gate fails honestly. The horizon is
  `not_measured` because the forecast never crosses either threshold inside the five days scored (the anomaly
  correlation stays at or above 0.9915 over the twenty leads, and the RMSE reaches 0.0222 where climatology's is
  0.4054) — a fact about this testbed, recorded as the reason it has no number. The coverage is the ensemble's own
  conformal interval, 99.5 % of the points inside a 90 % envelope.
* **ensemble** — `gnn` is exact at `u = 1` by construction (it is off by a quadratic in the distance from 1) and its
  median residual over the in-domain cases `0.25 … 2.0` is 0.008125, so the gate passes with the number that says how
  much better this case is than the typical one.
* **CFD surrogate** — `navier_stokes_residual` is the momentum equation on the field the surrogate emits,
  `||u u_x + v u_y + p_x - lap(u)/Re||` divided by the norm of the terms that must balance, in the interior of a
  48 × 48 grid; on the **exact** solution the same operator leaves 0.0107, which is the grid's discretisation floor.
  The level is the median of the same residual over the fitted Reynolds numbers `20 … 120`, and the record is asked
  about `Re = 70`: an interpolated parameter, where the surrogate is 1.73× the floor. At `Re = 200`, outside the
  fitted envelope, it is 17× and the domain gate reports `no_skill`. The interval is the disagreement between the
  `pod-3` and `pod-6` truncations — 2e-5 wide — and it covers 31.8 % of the field against a nominal 90 %: the two
  truncations agree with each other far more than either agrees with the flow, and the calibration gate fails rather
  than being widened until it passed.

### What makes it the same record, and what checks it

`tests/pinneapple_analysis/test_trust_three_kinds.py` holds the sentence the issue is done by:

* the three documents validate against **one** JSON Schema, `pinneapple_analysis.trust.record.TRUST_SCHEMA`
  (`pinneapple.trust/1`) — the schema has a `oneOf` for the two forms of a field, measured and not, and nothing else;
* they agree **key for key**: the test walks the three dictionaries and reports every path that exists in one and not
  in the others, with the two documented exceptions (the case's own inputs, and the per-expert list lengths);
* they come out of the **same adapter**: every residual of all three carries
  `pinneapple_physics.ensemble.PhysicsEnsemble.predict (residual_fn on this query)` as its source, and every
  provenance the ensemble adapter's fingerprint method — the signature `from_ensemble` leaves;
* the numbers above are asserted, so a change that moves one of them fails here and not in a figure.

A document that lost a field, carried a fourth badge or gave a measured value a reason is refused by
`Trust.validate_document`, which takes the dict, the JSON text or the path to the file.

## What it does not cover yet

`pp.trust` reads three kinds of model, and for each of them the record says what it could not measure instead of
leaving a zero in its place. For everything else it **refuses** with `pinneapple_core.errors.NotSupported` and a `fix`
saying what to pass instead, rather than return a half-empty object that would read like a measurement:

* a **PINN** (`pp.solve(..., method="pinn")`) is covered by `from_solution`: the PDE residual is re-evaluated on points
  drawn in the problem's box, with the same `pointwise_residual` the solver trains on — measured, never read back from
  the training loss — and the SHA-256 of the trained weights goes into the provenance. What it cannot measure it
  declares: no horizon (a steady field has none), no measurement (a network reads no instrument) and no interval (a
  deterministic network emits none). And no **in-domain level** for that residual: a PINN does not measure one on
  itself, so without `residual_reference=` and its source the residual gate is `unmeasured` and the badge is
  `fading` — the record refusing to claim more than it measured;
* a **grid operator** (FNO, UNO, a graph network) is refused by name: it predicts a field, not a value at an arbitrary
  point, so its residual cannot be sampled the way a PINN's is. The field it emits is scored the way the CFD
  surrogate's is — `producers.cfd_surrogate_trust` is that path;
* a **bare surrogate** (a `torch.nn.Module` or a callable) is refused for the same reason: wrap it in a
  `PhysicsEnsemble` with the reference solver as the second expert and it is covered by the first kind;
* the **consumers** read `Trust` where they draw something: the weather `earth2_gif` takes a `trust=` and draws its
  lead bar and colours from it, and `twin3d` attaches the record to the scene it renders.

