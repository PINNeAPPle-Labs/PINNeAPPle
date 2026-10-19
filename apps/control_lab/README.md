# Physics-Based Control Lab

**Use physical models to compute better control actions, and compare them fairly with what runs today.**

Day 29 of the PINNeAPPle app series (control.pinneapple.org, port 8112).

> **Safety.** This version runs in **simulation or recommendation mode only**. It has no command path to real
> equipment: it never writes to a controller, PLC or DCS. This is stated in the UI, the API description, every
> `StudyResult`, and here.

**The pain.** Most process loops run on PID gains tuned years ago, or on on/off rules. Both react to the error after
it appears, ignore actuator limits until they saturate, and know nothing about the constraint 2 °C away. The physics
that would let a controller plan ahead (mass and energy balances) is already known. It just isn't used.

| You give it | You get |
|---|---|
| A dynamic model (the heated tank's balances, or a reduced-order surrogate as a `ModelRef`), the estimated state (e.g. from the Digital Twin Runtime), objectives (set-point schedule, energy price), constraints (level band, maximum temperature) and actuator limits (range and rate) | Reference trajectories, the proposed control actions and the planned trajectory over the horizon, predicted performance, and a closed-loop comparison against PID and rule-based baselines on the same simulated plant: tracking error, constraint violations, energy, actuator travel, solve time. The study comes as a `StudyResult` and a trajectories CSV. |

## Methods

Everything is a library call (`pinneapple_systems.operations.control`, tests in
`tests/test_operations_anomaly_control.py`):

- **Linear MPC:** successive linearisation of the balances at every step, with zero-order-hold discretisation
  (matrix exponential). The condensed QP is solved with SLSQP, using an analytic gradient, hard actuator and rate
  limits, and soft output constraints.
- **Nonlinear MPC:** the full balances over the horizon, as multiple shooting with RK4, solved by CasADi and IPOPT.
  Without CasADi, the fallback is single shooting with SciPy SLSQP and batched finite-difference gradients.
- **Surrogate as the model:** the same nonlinear MPC with a controlled operator-inference ROM as the model. The ROM
  is exported to CasADi, and its `ModelRef` comes from the Surrogate Studio or the twin's series flow.
- **State estimate:** the MPCs read the state from the twin runtime's ensemble Kalman filter, which also estimates
  the unaccounted outflow and the heater efficiency. That is what makes them offset-free under unmeasured
  disturbances.
- **Baselines:** two PID loops (level → pump, temperature → heater) with feed-forward bias, saturation and
  anti-windup, from the extended `PIDController`, plus hysteresis rules. All controllers run on the same plant with
  the same noise realisation and disturbances.

## Default study (Try with example, about 15 s)

Setpoint steps run T 25 → 29 °C and h 1.2 → 1.0 m. A 20 % demand increase hits at 20 min and colder inlet water
(15 → 11 °C) at 40 min. The constraints are T ≤ 31 °C and 0.6 ≤ h ≤ 1.6 m, with a 600 kW heater and 80 kW per step.

Results of one run (reproduced by the app on every request):

| Controller | IAE T [K·s] | IAE h [m·s] | Constraint violations | Heater travel [kW] | Solve [ms/step] |
|---|---|---|---|---|---|
| PID | 1191 | 30.8 | none | 1313 | — |
| Rule-based | 3517 | 139.1 | none | 31200 | — |
| Linear MPC | 255.6 | 21.6 | none | 1964 | 22 |
| Nonlinear MPC (CasADi) | 255.1 | 21.4 | none | 1957 | 48 |
| Nonlinear MPC on the surrogate | 314.3 | 79.9 | none | 1891 | 15 |

The MPCs cut the temperature tracking error by 4.7× against PID. They move the heater more than PID does, a
trade-off that is set with the move weight. The surrogate MPC tracks the level worse because the ROM has no
unaccounted-outflow input, so it cannot cancel the demand change. The comparison shows this instead of hiding it.

## Recommendation mode

`POST /api/recommend` takes the current estimate and the current readings. The estimate can be a whole Digital Twin
Runtime state (`GET twin.pinneapple.org/api/assets/tank-1/state`). The answer holds the proposed next pump flow and
heater power, the planned actions and predicted trajectory over the horizon, and the predicted performance. A person
decides; nothing is sent anywhere.

## Integrations

- **REST API** at `/docs`:
  - `POST /api/compare` returns a job; `GET /api/jobs/{id}` gives the metrics, trajectories and plans.
  - `GET /api/jobs/{id}/files/control.study.json` and `/trajectories.csv` are the downloads.
  - `POST /api/recommend` returns a recommendation.
  - `POST /api/models` registers a surrogate as a `ModelRef` plus its model file (SHA-256 checked; its step must
    equal the controller's `dt`).
  - The shared runtime adds jobs, metering (`studies`, `scenarios`, `recommendations`) and API keys.
- **The other apps:** the estimated state comes from the Digital Twin Runtime (day 20), surrogates from the
  Surrogate Studio (day 14) as `ModelRef`, and the `StudyResult` goes to the Evidence Report (day 30).

## Limits

- The comparison runs on a simulated plant whose physics the MPC knows, up to estimated parameters. On a real
  plant, model mismatch is larger, so read the ranking as potential, not a guaranteed gain.
- The baselines are reasonable hand-tuned values, not optimal tunings. Your own gains can be entered through the
  API.
- Output constraints are soft (heavily penalised); actuator and rate limits are hard.
- There is one plant model in this version (the heated tank), plus a ROM of it as the alternative model.
- There is no command path to equipment, by design. Deploying to a controller is a separate, certified step.

Monetization: assets, scenarios and control studies.
