# Digital Twin Runtime

**Put a validated physical model or surrogate to work next to operational data, and know when to stop trusting it.**

Day 20 of the PINNeAPPle app series (twin.pinneapple.org, port 8103).

**The pain.** A model gets validated once, in a report. Then the plant changes: a valve wears, a sensor drifts, the
inlet water gets colder than anything in the training data. Nobody re-runs the model against the live signals, so
nobody notices when it stops being right. The opposite happens too: an alarm that the model could have explained
(a leak of 0.5 L/s) goes unnoticed until the level is low.

This first version is deliberately small and honest. It handles **one model type** at a time: the lumped heated-tank
model, or a reduced-order surrogate given as a `ModelRef`. Each asset has **one live data source**. It does not
claim universal industrial compatibility.

| You give it | You get |
|---|---|
| A validated model (the lumped mass + energy model with your parameters, or a surrogate `ModelRef` with its held-out metrics), live sensor data (MQTT, REST, or OPC UA, which is experimental) or a recorded table (CSV / Parquet / JSON), and the update frequency (the model step; readings may come slower) | The estimated state with its spread (ensemble Kalman filter), estimated parameters (an unaccounted outflow, the heater efficiency), forecasts with 5-95 % bands, alerts, model-validity checks and revalidation, the full history in a database, a state endpoint for other systems, and a `StudyResult` contract of the run |

## Integrations

- **REST API** at `/docs`:
  - `POST /api/live/start` starts a twin for an asset.
  - `POST /api/assets/{asset}/readings` pushes readings.
  - `GET /api/assets/{asset}/state` is the endpoint other systems read: estimate, spread, status, active alerts,
    validity.
  - `GET /api/assets/{asset}/history?table=estimates|readings|alerts|events` returns the stored history.
  - `GET /api/assets/{asset}/study` returns the run as a `StudyResult`.
  - `POST /api/replay` runs a twin over a recorded table, and `POST /api/series` runs the series flow.
  - The shared runtime adds `/api/jobs`, `/api/uploads`, `/api/usage` and `/api/keys` (`X-API-Key`).
- **MQTT** (implemented): JSON readings on `pinneapple/<asset>/sensors`. The app embeds a small MQTT 3.1.1 broker,
  `pinneapple_systems.digital_twin.io.mqtt_broker`, which supports QoS 0/1, retained messages, wildcards and an
  optional login. The demo plant publishes to it and the twin subscribes with `MQTTStream`. Expose the embedded
  broker with `TWIN_MQTT_HOST=0.0.0.0` and a published port 1883, or use your own broker (Mosquitto, EMQX, HiveMQ)
  with `TWIN_MQTT_URL=mqtt://user:pass@host:1883`.
- **Database:** SQLite by default (`<data dir>/twin.sqlite`). Postgres works when `TWIN_DB_URL=postgresql://...`
  and `psycopg[binary]` is installed in the image. There is one table each for readings, estimates, alerts and
  events (validity checks, revalidations).
- **OPC UA (experimental):** `TWIN_OPCUA_URL` and `TWIN_OPCUA_NODES` (`{"h": "ns=2;s=Level", ...}`) poll a server
  through `OPCUAStream`. This needs `asyncua` in the image, which is not installed by default. It is not tested
  against a real server in CI.
- **The other apps:** models arrive as `ModelRef` (`POST /api/models` takes the ModelRef and the model file and
  checks its SHA-256). Runs leave as `StudyResult`, which the Evidence Report (day 30) can read. The Control Lab
  (day 29) reads the twin's state as its estimated state.

## Methods

Everything is a library call (`pinneapple_systems.operations`, tests in `tests/test_operations_plant_twin.py`):

- **Models:** `LumpedTankModel` is the tank's mass and energy balances, integrated with RK4. It can carry parameters
  in the state. `SurrogateModel` wraps a `ControlledOpInf`, which is operator inference with inputs (Peherstorfer &
  Willcox 2016). That model is fitted on whole campaign runs, with other runs held out for its metrics.
- **Assimilation:** the stochastic EnKF of `pinneapple_analysis.state_estimation` (Evensen 2009), with a vectorised
  forecast step.
- **Forecast:** the ensemble is propagated with the inputs held at the mean of the last readings. Each member gets
  an offset drawn from the spread of those readings, so the band includes the uncertainty of the input.
- **Alerts:** rules with hysteresis:
  - estimated state outside its limits;
  - the forecast band reaching a limit;
  - a parameter beyond 3σ of its limit;
  - the normalised innovation squared above its 99.9 % χ² quantile for 3 readings, meaning the data disagree with
    the model.
- **Validity:** `pp.assess_drift` runs on the last 90 one-step predictions every 30 readings. It looks at bias
  (only if it is also practically relevant), coverage of the predictive band, residual trend, and inputs outside
  the model's training box. A `review` or `retrain` decision raises *revalidation required*. Revalidation replays
  the recorded inputs open loop from an earlier estimate. It passes when the RMSE is within 3× the model's
  validation RMSE.

## Series flow (Campaign → Surrogate → Uncertainty → Optimizer → Twin)

The *Series flow* tab chains the series on this asset through `pinneapple_data.contracts`
(`pinneapple_systems.operations.series`):

1. The Campaign Planner's `plan_campaign` (Latin hypercube with measured coverage) over a `ParameterSpace` of operating
   inputs, run as 12 campaigns (`DatasetSpec`, Parquet).
2. A controlled operator-inference ROM with 3 runs held out (`ModelRef` with held-out metrics).
3. The Uncertainty Lab's `uncertainty_study`: 90 % split-conformal intervals calibrated on 2 held-out runs, with
   their coverage measured on a third.
4. Bayesian optimisation (`pinneapple_design`) of the operating point, polished with SLSQP, then checked against
   the closed-form physics optimum.
5. The ROM deployed in this twin at the optimum. When the inlet water drops below the training range, the twin
   requests revalidation.

The Campaign Planner (day 13) and the Uncertainty Lab (day 15) are on main, so their steps call the same library
functions those apps use (`pinneapple_data.doe.plan_campaign`,
`pinneapple_analysis.uncertainty.study.uncertainty_study`). The Surrogate Studio (day 14) and the Optimizer (day 17)
are not on main yet, so those steps run as library stand-ins. The result says which is which, step by step.

## Try with example (offline, seconds)

- **Live twin → Start** runs the built-in plant over MQTT at 60× speed. Inject a leak: the twin estimates its flow
  (0.57-0.59 L/s for a 0.6 L/s leak in our runs) and raises a parameter alert.
- **Recorded data → Try with example** runs two hours of the simulated tank with a 0.5 L/s leak in the second hour
  (`examples/tank_day_with_leak.csv`). The run gives one leak alert, a passing revalidation and a `StudyResult`.
- **Series flow** runs the chain above in about 10 seconds.

## Limits

- One model type (lumped tank or controlled-OpInf ROM) and one data source per asset. FMUs and other plant models
  are not supported yet.
- Alerts are screening rules, not a safety or protection system. The twin sends no command to equipment.
- The EnKF assumes Gaussian noise. Parameters drift as random walks, and their step sizes are tuning choices.
- Live twins live in the server's memory (one worker). A restart stops them; the history stays in the database.
- OPC UA is experimental and not part of the default image.

Monetization: per monitored asset, data volume (readings) and inferences (assimilations and forecasts).
