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

## The benchmark: six things besides the error (#57)

`pinneapple_systems.digital_twin.benchmark.twin_benchmark` measures the twin the way a surrogate is measured — on
one case, with the protocol stated and a table at the end — but on the six things a twin is actually judged by. It
defaults to the heated channel (two sensors, a fouling wall conductance) because sensor failure needs more than one
output to be interesting; any model can be passed.

The protocol, in the result's own words: a simulated plant at stated true parameters (a twin is never commissioned
exactly), fixed seeds, the sensor noise and the decision limits stated, `quick` sizes of 200 readings and 80 members
(full: 600 and 200). Every number below is scored against the truth the plant kept.

| section | what it measures | how the number is made |
|---|---|---|
| state error | RMSE of the outlet, the parameter's error, NIS, NEES | against the plant's own truth, and against *persistence* — estimate the next reading with the previous one, the baseline a filter has to beat |
| latency | the per-reading cost of forecast + innovation + analysis, its p95, the share of readings that overrun the data period; and one recalibration against the window it collects | the workflow's own hard constraint, `T_inference <= T_data_input` (`realtime_pipeline`) |
| robustness to noise | the plant's readings at 0.25x .. 8x the `sensor_std` the twin was told | the commission that actually goes wrong: a twin told the wrong sensor noise |
| robustness to sensor failure | *dropout*, *outage*, *stuck*, *bias* injected into a simulated record | RMSE against the healthy run, whether the estimate stayed finite, and whether the twin recorded the loss (a gap) or detected the lie (an alarm) |
| anomaly detection | spikes at known steps | detection rate, delay, false alarms per 100 readings, over several seeds |
| remaining useful life | a wear ramp to a stated loss of the parameter | the health indicator, the observed end of life, and the RUL predicted at checkpoints — or a measured statement that it cannot be predicted |
| decision | the action the twin's estimate would drive | safe / warn / stop by the outlet against the case's limits, compared with the truth's actions and priced by a stated cost matrix (a missed stop is 100, a false alarm 1) |

Measured on the heated channel, 400 readings at dt = 1 s with 120 members, seed 0, plant `H = 42.0` against a
nominal 40, both sensors at sigma 0.15 degC:

<table>
| section | measurement | value | verdict |
|---|---|---|---|
| state error | RMSE of T2 | 0.04128 | pass |
| state error | RMSE of persistence (the baseline to beat) | 27.26 | — |
| state error | mean NIS against its band [1.8, 2.21] | 1.99 | pass |
| state error | H: estimate 41.94 against the plant's 42 | 0.14% | — |
| state error | NEES check / forecast coverage | fail / pass | — |
| latency | per reading (mean / p95 / max) | 0.17 / 0.24 / 0.52 ms | — |
| latency | readings over the 1000 ms data period | 0.0% | pass |
| latency | one recalibration (inverse workflow) | 3.44 s against a 50.0 s window | pass |
| noise | plant readings at 0.25x the sigma the twin was told: RMSE 0.02738 (0.66x), mean NIS 0.174 | outside the band | — |
| noise | plant readings at 0.5x the sigma the twin was told: RMSE 0.03118 (0.76x), mean NIS 0.537 | outside the band | — |
| noise | plant readings at 1x the sigma the twin was told: RMSE 0.04128 (1.00x), mean NIS 1.99 | consistent | — |
| noise | plant readings at 2x the sigma the twin was told: RMSE 0.0649 (1.57x), mean NIS 7.81 | outside the band | — |
| noise | plant readings at 4x the sigma the twin was told: RMSE 0.129 (3.12x), mean NIS 31.1 | outside the band | — |
| noise | plant readings at 8x the sigma the twin was told: RMSE 0.2227 (5.39x), mean NIS 124 | outside the band | — |
| sensor failure | dropout: RMSE 0.04095 (0.99x the healthy run), finite state True, alarmed False | — |
| sensor failure | outage: RMSE 0.04658 (1.13x the healthy run), finite state True, alarmed False | — |
| sensor failure | stuck: RMSE 6.341 (153.59x the healthy run), finite state True, alarmed True after 2 readings (anomaly) | — |
| sensor failure | bias: RMSE 0.8452 (20.47x the healthy run), finite state True, alarmed True after 0 readings (anomaly) | — |
| anomaly | spikes detected | 48/50 (96%), delay 0 readings, 0.00 false alarms per 100 readings | pass |
| RUL | not measurable on this case | the indicator has no usable trend at any checkpoint (R^2 0.01 to 0.13, against the 0.7 the prognostics module calls reliable): a smoothed residual does not degrade in a straight line while the filter follows the wear, so a linear extrapolation of it predicts nothing. The end of life the record shows is at reading 741, of 2380. | not_measurable |
| decision | actions agreeing with the truth's | 100.0% of 400 readings | pass |
| decision | cost: twin 0, never act 7000, perfect 0 | 0 per reading | — |
| decision | limits warn/stop 119.1/153 | the 60th and 85th percentile of the true outlet over the record (the exercise needs all three action classes to occur) | — |
</table>

Read that table as the case's honest report card:

- the twin beats persistence by two orders of magnitude (0.041 against 27 degC) and recovers the plant's parameter to
  0.14 %, with NIS inside its band at the nominal sensor noise;
- the online path costs 0.17 ms per reading against a 1 s data period, and one recalibration 3.4 s against the 50 s
  window it collects: the recovery path is the expensive one and it fits;
- the readings being 8x noisier than the twin was told costs a factor 5.4 in RMSE and drives NIS to 124: a twin
  commissioned with the wrong sensor noise does not fail quietly, it says so;
- losing a whole thermocouple (outage) costs 13 % of RMSE because the other one anchors the state; a thermocouple that
  *lies* (stuck, biased) is caught within 2 readings but does real damage (20x .. 150x) while it lasts;
- 48 of 50 injected spikes found, no false alarms;
- the NEES check fails at these sizes — the ensemble is slightly under-dispersed over 400 readings — and the table
  says so rather than reporting the sections that passed;
- RUL is **not measurable on this case**, and the reason is measured, not asserted: the smoothed residual never reaches
  twice its healthy level, because the filter *follows* the wear by adapting the very parameter that is wearing (the
  twin's `H` walks down with the plant's instead of lagging it by a growing margin), and the trend it does show has an
  R^2 of 0.01 to 0.13 against the 0.7 ``prognostics`` calls reliable. A twin that adapts is not a wear monitor; a wear
  monitor needs a wear state of its own. The section says so, and reports the end of life the record does show
  (reading 741 of 2380) rather than a number that would have looked like a prediction;
- the decision section: the twin's estimate drives the same actions as the truth on every reading, where never acting
  would cost 7000.

`python examples/digital_twin_end_to_end/twin_on_real_sensors.py --only benchmark` prints the table and writes
`_out/benchmark.md`.
