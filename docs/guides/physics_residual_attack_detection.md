# Detect false data injection with physics residuals

An attacker who rewrites a sensor stream in a plant keeps every value plausible on its own and close to its own
history. That is exactly what a purely data-driven detector learns, so it is what such a detector checks. What the
attacker cannot keep without a model of the plant is the physics that ties several sensors together: a tank level
must integrate the measured in- and outflows, a valve passes `q = sqrt(a + b * dh)` for the head across it.

`pinneapple_security.cps` turns those laws into residual channels, identifies their constants from trusted data
(a grey-box twin: no hand-tuned coefficients), and raises alarms with a per-channel CUSUM calibrated to a
false-alarm budget. It is a defensive tool: it reads sensor data and reports which balance broke.

## The three-tank example

```text
reservoir --pump Fp--> T1 --valve F12--> T2 --valve F23--> T3 --demand Fd--> consumers
```

```python
from pinneapple_security.cps import FlowLaw, MassBalance, PhysicsResidualDetector, episode_metrics
from pinneapple_security.cps_bench import inject, simulate_three_tank

train = simulate_three_tank(7, seed=0)            # dict of sensor columns, one sample per minute
calib = simulate_three_tank(7, seed=1)            # clean data the detector does not fit on
test = simulate_three_tank(10, seed=2)

det = PhysicsResidualDetector(components=[
    MassBalance("h1", ["Fp"], ["F12"], dt=60, window=1),   # area identified by least squares
    MassBalance("h2", ["F12"], ["F23"], dt=60, window=1),
    MassBalance("h3", ["F23"], ["Fd"], dt=60, window=1),
    FlowLaw("F12", "h1", "h2", midpoint=True),             # q^2 = a + b (h1 - h2), a and b identified
    FlowLaw("F23", "h2", "h3", midpoint=True),
]).fit(train)
det.calibrate(calib, false_alarms_per_day=1.0)

attacked = inject(test, "bias", [(3000, 3240)])            # F12 meter reads 4 % high for four hours
out = det.detect(attacked)
out["first_alarm"], out["channels"][:3]                    # time of the first alarm, balances blamed
episode_metrics(out["alarms"], [(3000, 3240)], len(test["h1"]))
```

Your own plant needs only the column names and the laws that hold between them. Any object with `name`, `fit(D)`
and `residual(D) -> array` is a component, so an energy balance of a heat exchanger, a pump curve or a
state-estimation residual of a power network plugs in the same way.

### Choices that matter

* `window=1` for storage balances when the flow meters report averages over the sample interval: the residual is
  then white noise, which is what the CUSUM assumes. Longer windows reduce level noise but make the residual
  autocorrelated, and the calibrated threshold grows.
* `midpoint=True` for flow laws with averaged flows and levels read at the start of the interval. Without it the
  law is evaluated at the wrong head and the residual carries a bias that changes with the operating point.
* Calibrate on clean data the detector was not fitted on, at least as long as the attack episodes you care about.
  `calibrate` bisects the threshold to meet the budget exactly on that trace.

## Benchmark: physics against data-driven detectors

`python examples/security/02_physics_residual_fdi.py --seeds 0 1 2` fits four detectors on a trusted week,
calibrates each to one false alarm per day on another clean week, then injects six attacks in 20 four-hour episodes
of a ten-day test trace. The same CUSUM alarm rule is used for every detector, so the comparison measures the
residuals, not the alarm logic. Baselines: a linear autoregressive twin of all sensors (three lags, ridge), PCA of
sensors and increments (Q statistic), and an isolation forest.

Mean of three seeds. AUC is episode-level (largest statistic in each attack episode against every clean window of the
same length); "det" is the fraction of episodes detected, "delay" the median delay in minutes, "loc" the fraction of
detections whose blamed channel contains the attacked sensor.

**Nominal operating regime** (false alarms per day on clean test data: physics 0.63, AR twin 0.63, PCA 0.80,
isolation forest 0.50)

| attack | physics AUC / det / delay / loc | AR twin | PCA | isolation forest |
|---|---|---|---|---|
| bias (F12 +4 %) | 1.000 / 1.00 / 8 / 0.98 | 1.000 / 1.00 / 16 / 0.03 | 0.490 / 0.10 / 122 | 0.497 / 0.03 / 152 |
| ramp (h2, 0 → 8 cm) | 1.000 / 1.00 / 108 / 0.93 | 1.000 / 1.00 / 82 / 0.57 | 0.497 / 0.10 / 119 | 0.504 / 0.05 / 148 |
| stuck (h3) | 1.000 / 0.98 / 9 / 0.93 | 1.000 / 0.98 / 10 / 0.76 | 0.815 / 0.67 / 89 | 0.368 / 0.03 / 202 |
| delayed (F23, 24 h old) | 1.000 / 0.97 / 1 / 1.00 | 0.994 / 0.90 / 3 / 1.00 | 0.877 / 0.77 / 51 | 0.592 / 0.10 / 148 |
| scaling (Fd −8 %) | 1.000 / 1.00 / 4 / 0.90 | 1.000 / 1.00 / 6 / 0.07 | 0.632 / 0.25 / 137 | 0.499 / 0.08 / 144 |
| coordinated replay | 0.986 / 0.97 / 0 / – | 0.990 / 0.97 / 0 / – | 0.496 / 0.13 / 96 | 0.463 / 0.05 / 94 |

**Shifted operating regime** (25 % more demand, pump band 1.4–2.2 m instead of 1.0–2.5 m; nothing refitted). False
alarms per day on clean test data: physics **0.47**, AR twin **84.4**, PCA **32.0**, isolation forest 1.07.

| attack | physics AUC / det / delay / loc | AR twin AUC | PCA AUC | isolation forest AUC |
|---|---|---|---|---|
| bias | 1.000 / 1.00 / 6 / 0.98 | 0.670 | 0.562 | 0.454 |
| ramp | 1.000 / 1.00 / 128 / 0.95 | 0.733 | 0.560 | 0.439 |
| stuck | 1.000 / 0.90 / 16 / 0.71 | 0.697 | 0.689 | 0.339 |
| delayed | 1.000 / 0.83 / 5 / 0.77 | 0.552 | 0.662 | 0.521 |
| scaling | 1.000 / 1.00 / 4 / 0.70 | 0.669 | 0.611 | 0.431 |
| coordinated replay | 0.901 / 0.83 / 0 / – | 0.538 | 0.551 | 0.480 |

What the numbers say:

* In the regime it was trained on, a good data-driven twin (the AR model) detects as well as the physics: both
  reach AUC 1.0. Physics is faster on the bias (8 vs 16 min) and slower on the slow level ramp (108 vs 82 min: a
  level drift barely moves a storage balance until the valve laws see it), and it points at the right sensor far
  more often (0.98 vs 0.03 for the bias, 0.90 vs 0.07 for metering fraud).
* When the plant moves to an operating point the data never covered, the data-driven detectors flood the operator
  (84 and 32 false alarms per day) while the physics residuals stay calibrated (0.47 per day) and keep AUC 1.0.
  This is the practical argument for physics: plants change set-points, seasons and demand far more often than
  anyone refits a detector.
* The isolation forest and PCA, which only look at the joint distribution of the readings, miss most attacks that
  keep values in range.

## Public data: BATADAL (C-Town)

`pinneapple_security.batadal` runs the same detectors and the same alarm rule on BATADAL (Taormina et al., J. Water
Resour. Plann. Manage. 144(8), 2018): hourly SCADA data of the C-Town water network with 7 tanks, 11 pumps, a
flow-control valve and 12 pressure sensors. It is a public benchmark produced by an independent hydraulic simulator
(EPANET), not a recording from a physical plant. The BATADAL site licenses it under CC BY 4.0 (its footer also says
"All rights reserved"), so the data are not in the repository: `batadal.fetch()` downloads three files (about 4 MB)
to `$PINNEAPPLE_BATADAL_CACHE` (default `~/.cache/pinneapple/batadal`) and checks their SHA-256.

```python
from pinneapple_security import batadal

batadal.fetch()                              # once; nothing is downloaded at import
result = batadal.run_benchmark(false_alarms_per_day=0.25)
```

What is scored, and what is not:

* Dataset 03 is one attack-free year. The first half fits each detector, the next quarter calibrates its threshold
  to the false-alarm budget, and the last quarter is clean data used for neither (the false-alarm rate to trust).
* Dataset 04 has five labelled attack episodes (42, 60, 37, 7 and 73 hours). Every other row is labelled `-999`,
  which means unlabelled, not clean: the organisers say undiscovered attacks remain. Alarms outside the labelled
  episodes are therefore an upper bound on false alarms, not a measurement of them.
* **The competition's own score is not reproduced.** The test dataset has no labels in the file and its attack times
  are not on the site, and the rules document with the exact formulas could not be read. `ttd_score` and
  `classification_score` follow the published description (the overall score is the mean of a time-to-detection and
  a classification score, the latter the mean of TPR and TNR); `ttd_score`'s normalisation by the attack length and
  the 12-hour `hold` that turns alarms into hourly flags are this module's choices. Do not compare these numbers
  with the competition ranking.
* Which component was attacked is not in the labels, so localisation (the blamed channel) is not evaluated.

The physics detector has seven tank balances (`TankBalance`) and the head curves of the two pumps that have pressure
sensors at both ends and run in the attack-free year (`PumpLaw` for PU2 and PU10). Each balance identifies, on the
attack-free half, a coefficient per flow that the EPANET topology allows to touch that tank and a demand-by-hour-of-day
profile for the demand nobody measures. On the calibration quarter, a balance explains 90 % of the hourly level
changes of T1, T2 and T3 but only 49, 73, 37 and 54 % for T4 to T7 (demand dominates there), and the two pump laws
fit to 0.03 and 0.01 L/s. Baselines: the three of the simulated benchmark, reading all 31 measured columns.

The parameters were fixed before dataset 04's labels were read: budget 0.25 false alarms per day, 6 hours of
grace after an episode, a 12-hour hold. The conservative score `S_floor` was added after the first run, because the
AR twin's score looked excellent for a reason a clean year cannot show (below).

Budget 0.25 false alarms per day (alarms on the clean held-out quarter: physics 0.20, AR twin 0.26, PCA 0.18,
isolation forest 0.27 per day):

| detector | detected | delay per episode (h) | alarms per day outside the attacks | S_TTD | S | S_floor |
|---|---|---|---|---|---|---|
| physics | 4 / 5 | 0, 4, 7, missed, 55 | 0.33 | 0.598 | 0.667 | 0.664 |
| AR twin | 5 / 5 | 0, 0, 0, 0, 0 | **19.5** | 1.000 | 0.972 | 0.750 |
| PCA | 5 / 5 | 1, 0, 1, 1, 0 | 1.52 | 0.961 | 0.945 | 0.853 |
| isolation forest | 5 / 5 | 18, 27, 18, 4, 4 | 0.32 | 0.602 | 0.683 | 0.680 |

`S` takes the true-negative rate from the clean held-out quarter. `S_floor` takes it from dataset 04's own
unlabelled hours, assuming all of them clean, which makes it a lower bound. The same budget at 0.1 and 1.0 per day
(detected / alarms per day outside the attacks / S_floor):

| budget per day | physics | AR twin | PCA | isolation forest |
|---|---|---|---|---|
| 0.1 | 3 / 5, 0.21, 0.578 | 5 / 5, 19.06, 0.750 | 5 / 5, 1.03, 0.802 | 4 / 5, 0.08, 0.630 |
| 1.0 | 4 / 5, 1.22, 0.721 | 5 / 5, 20.51, 0.750 | 5 / 5, 3.37, 0.797 | 5 / 5, 1.11, 0.699 |

What the numbers say:

* **The physics residuals did not beat PCA on this data.** At every budget they raised fewer alarms outside the
  labelled attacks than PCA (0.21 against 1.03 at 0.1 a day, 0.33 against 1.52 at 0.25, 1.22 against 3.37 at 1.0),
  but they detected 3 or 4 of the 5 attacks, missed the 7-hour attack at the main budget and took 55 hours on the
  last one, so their conservative score (0.58 to 0.72) is below PCA's (0.80 to 0.85).
* **The AR twin catches every attack because it is in alarm almost all the time**: 19 to 21 alarms a day outside the
  attacks at every budget. At the main budget its flags cover 99.9 % of dataset 04's unlabelled hours (3,436 alarms
  in 4,177 hours), against 45 % for PCA and 11 % for the physics residuals. Dataset 03 is from 2014 and dataset 04
  from 2016. This is the failure the simulated benchmark shows under a regime shift (84 alarms a day there); the
  physics residuals do not show it (0.33).
* With five labelled attacks, any single number is coarse: one missed 7-hour attack moves the detection rate by 20
  points.

## Limits

* Coordinated replay of every sensor is physically consistent. Both residual detectors catch it here only because
  the splice from the live to the replayed recording is a discontinuity (delay 0). An attacker who aligns the
  splice, or one who knows the plant model and injects consistent data (a stealthy attack in the sense of Liu, Ning
  and Reiter 2009), is invisible to any residual test. Pair the detector with `anomaly.replay_score`, with signed
  data paths (`pinneapple_security.signing`) and with physical watermarking where the control loop allows it.
* The three-tank benchmark is simulated (seeded, reproducible). SWaT and WADI need a signed request; BATADAL is
  public and has an adapter (above), but it is itself simulated, its labels are partial and its test labels are not
  public, so no result here is a measurement on a physical plant.
* Attacks rewrite the reported readings only; the controller here acts on the true T1 level, so no attack changes
  the physical trajectory.

## References

* E. S. Page, "Continuous inspection schemes", Biometrika 41 (1954).
* Y. Liu, P. Ning, M. K. Reiter, "False data injection attacks against state estimation in electric power grids",
  ACM CCS 2009.
* A. A. Cárdenas et al., "Attacks against process control systems: risk assessment, detection, and response",
  ASIACCS 2011.
* D. I. Urbina et al., "Limiting the impact of stealthy attacks on industrial control systems", ACM CCS 2016.
* J. Giraldo et al., "A survey of physics-based attack detection in cyber-physical systems", ACM Computing Surveys
  51(4) (2018).
* R. Taormina et al., "The battle of the attack detection algorithms (BATADAL)", J. Water Resour. Plann. Manage.
  144(8) (2018).
