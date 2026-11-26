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

## What it does not cover yet

`pp.trust` covers the model kind that already carries the residual lookahead and the measurement check, and — being
duck-typed — any surrogate that predicts with the same contract. For everything else it **refuses** with
`pinneapple_core.errors.NotSupported` and a `fix` saying what to pass instead, rather than return a half-empty
object that would read like a measurement:

* a **PINN** or a **neural operator**: their trust needs the residual sampled on collocation points (not the
  training loss), the boundary-condition error on the geometry, and a horizon from a rollout;
* a **bare surrogate** (a `torch.nn.Module` or a callable): wrap it in a `PhysicsEnsemble` with the reference
  solver as the second expert and it is covered;
* the **consumers** (`twin3d`, the weather `earth2_gif`, certificates): they do not read `Trust` yet.

Those are the open slices of #345; the record, the schema and the adapters are the part that does not change when
they land.
