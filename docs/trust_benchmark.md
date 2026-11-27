# The trust benchmark: four failure kinds, one protocol, one leaderboard (#349)

A leaderboard of in-domain accuracy hides the one failure that matters in deployment: *where* a model stops
being valid. This suite injects the four failures a deployed physics model meets -- a hidden regime change,
a sensor fault, a slow drift, an out-of-domain reading -- with the reading at which each one starts written
down, and measures what the model and its monitor did about it.

Every table on this page is generated from
`benchmarks/trust/results.json` by `python scripts/trust_benchmark.py --docs`. Do not edit it by hand;
`--check` refuses a page that drifted from the record, and a record whose order is not the one the
measurements give.

## The protocol

* **the plant**: the base temperature of a fin array,
  `pinneapple_physics.closed_form.fin_array_conduction` (fin efficiency `tanh(m Lc) / (m Lc)`, adiabatic-tip
  corrected length), with a declared fan curve mapping rpm to the convection coefficient. The plant's hidden
  parameters -- the fan's derating, how many fins the array still has -- are not among the inputs.
* **the inputs**: `power_w, ambient_c, fan_rpm`, in the box power_w 10-70, ambient_c 15-35, fan_rpm 900-2400. That box is what the candidates were
  fitted on, and what the monitor's envelope was fitted on.
* **the data**: 240 training, 200 validation and
  400 calibration readings, drawn once (split seed 0); every
  candidate sees the same three splits.
* **the streams**: 100 clean readings then 100 under the injected
  failure, per case, seed 7. The change is at reading 100 of the
  stream, for every case, and the case table below says what changed.
* **the monitor**: `pinneapple_analysis.trust.drift.DriftMonitor` -- input, prediction, uncertainty, error
  and envelope checks -- on a trailing window of
  20 readings, its limits calibrated at the 99% quantile of
  200 genuinely in-distribution windows. The monitor is the protocol's, not the
  candidate's: the candidate changes what the monitor is given, never what the monitor checks.
* **the alarm**: DriftReport.has_alarm: any check of pinneapple_analysis.trust.drift that contradicted.
* **the labels**: the measured response, as a backtest: the monitor sees the response sensor of the same window.
* **the error**: absolute error / std of the in-domain training responses, with a tolerance of 0.25 for the exit.
* **the ranking**: `(missed, error_after, detect_delay, false_alarms_per_1000, id)` -- the four questions a trust benchmark asks, in this order: did the monitor ever notice (a model that never notices is the failure this suite exists to find, so a missed case outranks every other column); was it wrong after the change (the truth column -- a model that notices in four readings and is fifteen times further from the plant is not the more trustworthy one); how many readings did it take to notice (the headline metric of the issue); how often does it cry wolf (a monitor that alarms on everything detects everything, and this is the column that says so). The error column is compared with the resolution below, and so is the false-alarm column, so that a kind where every model misses the change by the same amount is decided by the delay instead of by the last digit.
* **the resolution**: a difference under 5% in the error column, or under 10% in the false-alarm column, is not a difference: the order falls through to the next column.
* **the placement**: the mean of a candidate's rank over the four kinds, lowest first; ties go to the smaller mean error after the change, then to the candidate id.

## The leaderboard

| rank | candidate | mean rank | regime change | sensor fault | slow drift | out of domain | worst kind | error after the change |
|---|---|---|---|---|---|---|---|
| 1 | gp | 1.00 | #1 | #1 | #1 | #1 | the same rank on every kind | 0.356 |
| 2 | ridge | 2.00 | #2 | #2 | #2 | #2 | the same rank on every kind | 0.446 |
| 3 | ridge + constant band | 3.00 | #3 | #3 | #3 | #3 | the same rank on every kind | 0.446 |
| 4 | gbt | 4.00 | #4 | #4 | #4 | #4 | the same rank on every kind | 0.832 |

The mean rank is the average of the candidate's rank on each of the four failure kinds. A candidate that is
excellent on one kind and blind to another cannot hide the second half of that sentence: the *worst kind*
column names it, and the per-kind tables below carry every column the ranking was computed from.

## hidden regime change

The cases: `fan_derate`; `fins_blocked`.

| rank | candidate | detected | readings to detect | false alarms / 1000 | error before | error after | readings to the error exit | coverage before | coverage after | checks that fired |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gp | 2/2 | 4.5 | 216.0 | 2.181e-05 | 0.835 | 0 | not measured | not measured | error_drift (209), input_drift (21), prediction_drift (12) |
| 2 | ridge | 2/2 | 5.5 | 166.7 | 0.036 | 0.834 | 0 | not measured | not measured | error_drift (200), input_drift (21), prediction_drift (18) |
| 3 | ridge + constant band | 2/2 | 5.5 | 166.7 | 0.036 | 0.834 | 0 | 0.96 | 0.01 | error_drift (200), input_drift (21), prediction_drift (18) |
| 4 | gbt | 2/2 | 6.5 | 86.4 | 0.118 | 0.833 | 0 | not measured | not measured | error_drift (187), input_drift (21), prediction_drift (8) |

## sensor fault

The cases: `power_spike`; `fan_stuck`; `ambient_bias`; `thermocouple_drift`.

| rank | candidate | detected | readings to detect | false alarms / 1000 | error before | error after | readings to the error exit | coverage before | coverage after | checks that fired |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gp | 4/4 | 4.0 | 77.2 | 2.121e-05 | 0.228 | 50.0 | not measured | not measured | envelope (20), error_drift (317), input_drift (111), prediction_drift (23) |
| 2 | ridge | 4/4 | 8.8 | 55.6 | 0.036 | 0.245 | 13.2 | not measured | not measured | envelope (20), error_drift (290), input_drift (111), prediction_drift (43) |
| 3 | ridge + constant band | 4/4 | 8.8 | 55.6 | 0.036 | 0.245 | 13.2 | 0.96 | 0.50 | envelope (20), error_drift (290), input_drift (111), prediction_drift (43) |
| 4 | gbt | 4/4 | 14.8 | 55.6 | 0.119 | 0.271 | 0.8 | not measured | not measured | envelope (20), error_drift (220), input_drift (111), prediction_drift (12) |

## slow drift

The cases: `dust_ramp`; `slow_fouling`.

| rank | candidate | detected | readings to detect | false alarms / 1000 | error before | error after | readings to the error exit | coverage before | coverage after | checks that fired |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gp | 2/2 | 6.5 | 98.8 | 1.421e-05 | 0.238 | 54.5 | not measured | not measured | error_drift (187), prediction_drift (16) |
| 2 | ridge | 2/2 | 20.5 | 104.9 | 0.039 | 0.244 | 36.5 | not measured | not measured | error_drift (158), prediction_drift (17) |
| 3 | ridge + constant band | 2/2 | 20.5 | 104.9 | 0.039 | 0.244 | 36.5 | 0.92 | 0.33 | error_drift (158), prediction_drift (17) |
| 4 | gbt | 1/2 | 67.0 | 12.3 | 0.126 | 0.279 | 11.0 | not measured | not measured | error_drift (66), prediction_drift (2) |

## out of domain

The cases: `fan_below_range`; `power_above_range`.

| rank | candidate | detected | readings to detect | false alarms / 1000 | error before | error after | readings to the error exit | coverage before | coverage after | checks that fired |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gp | 2/2 | 0 | 98.8 | 1.679e-05 | 0.125 | 50.0 | not measured | not measured | envelope (189), error_drift (205), input_drift (190), prediction_drift (190) |
| 2 | ridge | 2/2 | 0 | 49.4 | 0.036 | 0.461 | 2.5 | not measured | not measured | envelope (189), error_drift (190), input_drift (190), prediction_drift (191) |
| 3 | ridge + constant band | 2/2 | 0 | 49.4 | 0.036 | 0.461 | 2.5 | 0.96 | 0.17 | envelope (189), error_drift (190), input_drift (190), prediction_drift (191) |
| 4 | gbt | 2/2 | 0 | 0 | 0.130 | 1.944 | 0 | not measured | not measured | envelope (189), error_drift (187), input_drift (190), prediction_drift (158) |

## The injected streams and their ground truth

| case | kind | the change | ground truth |
|---|---|---|---|
| `fan_derate` | regime change | the fan derates: eta_fan 1.00 -> 0.70: the same inputs, 30 % less convection, invisible in the inputs | the failure begins at reading 100 of 200 |
| `fins_blocked` | regime change | fins are blocked: n_fins 12 -> 8: a third of the array stops convecting, and the inputs are unchanged | the failure begins at reading 100 of 200 |
| `power_spike` | sensor fault | spikes on the power meter: 1 reading in 5 reads 40 % high, from the change onwards | the failure begins at reading 100 of 200 |
| `fan_stuck` | sensor fault | the fan tachometer sticks: the rpm input freezes at its value on the change | the failure begins at reading 100 of 200 |
| `ambient_bias` | sensor fault | a bias on the ambient probe: the ambient input reads 3 K high from the change onwards | the failure begins at reading 100 of 200 |
| `thermocouple_drift` | sensor fault | the thermocouple ramps: the response sensor ramps 0.4 K per reading: the plant and the model are fine, the labels are not | the failure begins at reading 100 of 200 |
| `dust_ramp` | slow drift | dust on the fan: eta_fan falls 1.00 -> 0.68 linearly over the tail | the failure begins at reading 100 of 200 |
| `slow_fouling` | slow drift | slow fouling: eta_fan falls 1.00 -> 0.90 over the tail: a ramp that may not cross the tolerance at all | the failure begins at reading 100 of 200 |
| `fan_below_range` | out of domain | the fan below its range: rpm drawn from 480-700, under the 900 the candidates were fitted on | the failure begins at reading 100 of 200 |
| `power_above_range` | out of domain | the power above its range: power drawn from 90-140 W, above the 70 the candidates were fitted on | the failure begins at reading 100 of 200 |

The rows above are measured against the plant, not against the model: the response `y` of every
stream is the closed form's, whatever the sensors read. That is what makes the truth column a truth
column -- in `thermocouple_drift` the model is right and the sensor is wrong, and the record says so.

## The candidates

* **ridge** (ridge): fitted on 240 in-domain samples, 11 parameters, one shot
* **ridge_band** (ridge): ridge with a fixed +-2 sigma band of 0.892 K: the standard deviation of its validation residuals, not a calibrated interval -- reports an uncertainty
* **gp** (gp): fitted on 240 in-domain samples, 960 parameters, one shot
* **gbt** (gbt): fitted on 240 in-domain samples, 2260 parameters, one shot

## Reproducing it

```bash
python scripts/trust_benchmark.py --run     # fit the candidates, run the suite, write the record and this page
python scripts/trust_benchmark.py --docs    # regenerate this page from the record (no fits)
python scripts/trust_benchmark.py --check   # the gate: declaration, record and page agree
```

The candidates are one-shot fits (ridge, Gaussian process, gradient-boosted trees) on one BLAS thread,
so the record reproduces: no time-based budget is involved and the same declaration draws the same
streams. Measured: two consecutive runs of the declared protocol produced a byte-identical record,
apart from the timestamp and the wall time. To place a model of your own on the board, fit it on
`in_domain_data()` and pass it to `run_suite` as a `Candidate` -- a callable `predict(X) -> y`,
or `-> (mean, std)`; nothing else is asked of it.
