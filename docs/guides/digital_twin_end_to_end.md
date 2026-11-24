# The twin end to end, on real sensors

`pinneapple_systems.digital_twin.end_to_end` runs the standard twin workflow
(`pinneapple_systems.digital_twin.workflow.digital_twin_workflow`) over readings from where they actually come from,
and then asks the question a commissioned twin keeps having to answer: **are its residuals still the residuals it was
validated on?** That second part is `pinneapple_analysis.trust.drift` — the drift monitor, with its thresholds
calibrated on a second held-out split and its `RetrainDecision` written down.

The rule this guide (and the code) follows: **no sensor is invented.** Every record carries its `origin`, and a
synthetic record says it is synthetic wherever it appears.

## Where the readings come from

| `source` | what it is | where it comes from |
|---|---|---|
| `"usgs"` | A real gauge, discharge every 5–15 min | [USGS Water Services instantaneous values API](https://waterservices.usgs.gov/nwis/iv/), site `01646500` (Potomac River near Washington, DC), public domain, no key, **downloaded at run time and never stored in this repository** |
| `"synthetic_river"` | A baseflow recession on the same grid as the gauge | `pinneapple_systems.digital_twin.recession.synthetic_recession`; the origin says `synthetic` and "not a measurement" |
| `"simulated"` | The library's simulated plant: process noise, a known parameter drift, an injected sensor spike | `pinneapple_systems.digital_twin.workflow.simulate_plant`; the origin says `simulated` |
| `"record"` | Arrays you pass, with your own `origin` | a CSV replay, an MQTT / OPC-UA / Modbus / Kafka / REST adapter (`pinneapple_systems.digital_twin.io.stream`), or a saved fixture |

```python
from pinneapple_systems.digital_twin.end_to_end import end_to_end_twin, usgs_record

r = end_to_end_twin(source="usgs", out_dir="out/river")     # a real gauge, live
r["origin"]["site_name"], r["origin"]["url"]                # where the readings came from
r["summary"], r["drift"]["verdict"], r["drift"]["reason"]    # what the twin saw, and whether it aged

record = usgs_record()                                      # the readings alone, with their provenance
r = end_to_end_twin(source="record", record=record)         # the same twin over saved arrays
```

## The two twin cases

**The heated channel** (`heated_channel`) is the workflow's two-sensor case: an upstream node taking the heater power
and a downstream node losing heat to ambient, coupled by a wall conductance `H` that fouls. Two outputs mean a
reading can be lost, frozen or biased on one sensor and the other still reports — the case a twin benchmark has to
score, and the case the benchmark in `pinneapple_systems.digital_twin.benchmark` sweeps.

**The river** is a linear reservoir (`linear_reservoir`), the baseflow-recession model of hydrograph analysis,
`Q' = Q_base + (Q - Q_base) exp(-k dt)`, whose slow parameter is the recession constant `k`. The gauge measures no
inflow, so `Q_base` is zero: the twin extrapolates the recession, and a rise of the river is a departure it flags
rather than an inflow it pretends to know.

### Commissioning is identification, not assumption

`river_model` identifies the model on the record's first half, and every number it uses is measured there:

- `k` from `recession.fit_recession` (log-linear least squares on the opening window);
- the reading noise from the **resolution of the feed** — the smallest step between two distinct reported values, over
  `sqrt(12)` (a value reported on a grid of that step has that standard deviation). The Potomac record reports on a
  30 ft3/s grid;
- the process noise from the **one-step residual** of the fitted recession, `Q_k - a Q_{k-1}`, which is what the model
  cannot explain between two samples.

The twin then runs unassisted over the rest of the record, which is what the checks and the drift verdict judge. On a
synthetic record whose generator knows its own reading noise, that noise is subtracted from the one-step residual
first (independent sample to sample); on a quantised feed the two cannot be separated, and the twin carries a
slightly conservative budget — which its NIS check reports rather than hides.

## Alarms, and a reading that did not arrive

The workflow's alarms are unchanged: anomaly gating on the normalised innovation, a two-sided Page CUSUM with its
in-control average run length, and recalibration through the inverse workflow when the chart fires. What the
end-to-end runner adds is what a real feed does to them:

- a reading that did not arrive is carried as a **gap**, never filled in: the members keep forecasting, nothing is
  assimilated, the drift chart does not move (a gap is not evidence of a changed process), and a recalibration window
  that contains a gap is **refused** rather than fitted. The result lists the gaps and a `twin.missing_readings` check
  that its state estimate stayed finite across them.
- a missing *input* is held at its last value, and the first input being missing is an error rather than a default.

## The drift verdict: is the twin still fitted?

A twin tracks, so its states and its predictions are *supposed* to leave the validated range as the plant moves — a
river recedes, a channel heats up. Monitoring those would alarm on every transient. What must not move is the
distribution of what it failed to predict, so `twin_drift_check` runs the drift monitor in **innovation space**: the
monitored model is `ResidualModel`, which predicts the next normalised innovation as zero with unit spread (what a
consistent filter produces by construction), and each deployment window is one `DriftReport`:

| check | what a change would mean |
|---|---|
| `input_drift` | the innovation distribution moved: a bias the filter cannot absorb, or a variance change |
| `error_drift` | `\|z\|` moved: the twin's errors are no longer the validated ones (the health signal) |
| `envelope` | the joint gate of the last innovations: the residual structure changed |
| `prediction_drift`, `uncertainty_drift` | zero by construction here (the monitored model predicts a constant), reported rather than dropped |

The thresholds come from `DriftMonitor.calibrate` on a *second* held-out part of the record, so "drift" means "further
from the validation baseline than 99 % of genuinely in-distribution windows were"; the decision per window is the
`RetrainDecision` (STABLE / REVIEW / RETRAIN) with its reason.

## What the runs measure

`python examples/digital_twin_end_to_end/twin_on_real_sensors.py` runs both cases and writes, per case, the
workflow's `dashboard.html` and the provenance page `twin_end_to_end.html`.

On the simulated stream (synthetic plant, 300 readings, dt = 1 s, 200 members) every truth-scored check passes: mean
NIS inside its chi-square band before the drift and after the recalibration, NEES over 20 independent plants 97 % of
the steps inside the 95 % band, the drift of `H` detected two steps after it starts, the inverse workflow recovering
`H = 27.76` (95 % [24.69, 31.22]) against the true 28, the injected spike flagged, and the 60-step forecast at 88 %
coverage.

On the real gauge the honest numbers are different, and they are reported as they come out:

- 4 readings of 2005 did not arrive and were carried as gaps (`twin.missing_readings` passes);
- mean NIS 0.86, below its band: an ensemble filter's spread carries both the identified process noise and the
  parameter's own uncertainty, so on a record whose step-to-step scatter is almost entirely the feed's own, the twin
  is 15 % conservative. A conservative twin does not miss a departure; it is also less informative, and saying so is
  the point of the check;
- the drift verdict is RETRAIN on the recording of 2026-11-24, on `error_drift` and `envelope`: the last fifth of the
  record has a residual scale of 1.42 where the first four fifths have 0.76, i.e. the day's operation changed the
  river's small-scale behaviour. The verdict is a statement about that recording, not about the gauge in general.

Nothing on the real record is scored against a truth: the truth-scored checks are reported `inconclusive` with the
reason, which is the honest answer when the truth does not exist.

## What this does not do

- No sensor is simulated and passed off as real. If a record does not exist, the source says so and the issue says so.
- No reading is interpolated, and no gap is filled in.
- The reservoir twin's noise budget is additive and constant; a record whose reading noise scales with the flow (a
  recession with a proportional noise) is flagged by the drift monitor — correctly, as a change in residual structure,
  even though the plant did nothing unusual. Proportional noise is not modelled.
- The drift check measures distribution shift, not its cause: a river that rises and a gauge that drifted both look
  like drift. The workflow's anomaly flags are what separate a single spike from a level change.
