# Physics Anomaly Detector

**Detect abnormal behaviour with data *and* physical knowledge: which balance broke, by how much, and which sensors
are involved.**

Day 28 of the PINNeAPPle app series (anomaly.pinneapple.org, port 8111).

**The pain.** Plant historians are full of alarms nobody trusts. A threshold on one sensor fires at every set-point
change and misses a slow leak, and a neural detector says "anomaly score 0.93" without saying why. Engineers already
know the physics: what goes into a tank comes out or accumulates, and energy too. A detector should use that
knowledge and explain its findings in those terms.

| You give it | You get |
|---|---|
| Sensor time series (CSV, Parquet or JSON with a time column `t`), the operating variables, the end of a window you know is normal, conservation balances written as formulas over your columns (optional), a physical reference model (optional: the lumped tank, or a reduced model as a `ModelRef`), and a label column (optional) | Suspicious events with severity, the variables involved, the residual series of every detector, and explanations from verifiable rules ("mass balance broken by −0.39 L/s, 5.8 % of the inflow, over 2440 s: something leaves, or an inflow is over-read"). With labels, you also get point and event precision/recall, false alarms per hour and detection delays. The study is saved as a `StudyResult` contract. |

## Methods (not only neural networks)

Everything is a library call (`pinneapple_systems.operations.anomaly`, tests in
`tests/test_operations_anomaly_control.py`):

- **Statistical:** robust z-scores (median / MAD) of each value and of its rate of change, against the normal
  window.
- **Time-series model:** an AR(3) model of each signal's increments, fitted on the normal window. Its one-step
  residuals feed a two-sided CUSUM (Page 1954) with per-sample clipping, so a single set-point step cannot trip it
  but a slow drift does.
- **Conservation balances:** `d(accumulation)/dt = inflow − outflow`, written as formulas over your columns and
  constants. They are evaluated by a safe expression parser, never Python `eval`. The residual is taken over a
  trailing window: `[Acc(t) − Acc(t−W)]/W − mean(in − out)`. A broken balance names the law, its units, its size
  relative to the inflow term and the direction.
- **Physical-model residuals:** a reference model predicts each state one step ahead from the measured state and
  inputs, and the windowed prediction error is the residual.

Each detector's spread is calibrated on the normal window only. A sample is flagged above `threshold` × that spread.
Flags closer than 6 samples merge into events, and events shorter than 3 samples are dropped. Severity comes from
the peak score / threshold ratio: below 2 is low, below 4 medium, below 8 high, and critical above that.

## Benchmark

The *Benchmark* tab scores each detector family alone and combined on labelled runs. The labels are exact because
the faults are injected into the heated-tank simulation. Each run has two fault-free hours (the calibration window),
then six faults: a leak (6 % of the inflow), a 10 % heater loss, a +0.5 °C temperature-sensor bias, a +6 cm/h
level-sensor drift, a stuck flow meter, and a drain valve 8 % more closed. Seeds 101-103 were not used while the
detectors were developed. The numbers are recomputed on every run.

Results on seeds 101-103:

| Detector | Point precision | Point recall | Faults found | False alarms /h |
|---|---|---|---|---|
| Statistical | 64.6 % | 1.2 % | 0 % | 0 |
| AR + CUSUM | 55.2 % | 13.8 % | 50 % | 0.18 |
| Balances | 100 % | 24.5 % | 83 % | 0 |
| Model residuals | 100 % | 51.1 % | 100 % | 0 |
| Combined | 80.0 % | 57.1 % | 100 % | 0.18 |

The balances do not see the drain-valve fault. The outflow is measured, so the physics still holds; only the model
residual catches it. Point recall is low by construction: windowed detectors lag the fault start, and a sensor drift
is only visible once it is large. Event recall ("faults found") and delay are the operational figures.

The public BATADAL water-network adapter (PR #731) is **not merged**, so this benchmark is synthetic. To score the
detector on your own data, upload a history with a label column.

## Integrations

- **REST API** at `/docs`:
  - `POST /api/analyze` takes an `upload_id` from `POST /api/uploads`, or `example: true`, and returns a job.
  - `GET /api/jobs/{id}` returns the events, channels and score.
  - `GET /api/jobs/{id}/files/events.csv` and `/anomaly.study.json` are the downloads.
  - `POST /api/benchmark` runs the benchmark, and `GET /api/columns/{upload_id}` returns the columns of an upload.
  - `POST /api/models` registers a reference model as a `ModelRef` plus its model file (SHA-256 checked).
  - The shared runtime adds jobs, uploads, metering (`analyses`, `rows`, `benchmarks`) and API keys.
- **Files:** CSV, Parquet and JSON tables, read with `pinneapple_data.contracts.read_table`; contract sidecars are
  honoured.
- **The other apps:** a reduced model from the Surrogate Studio or the Digital Twin Runtime's series flow can be the
  reference model, and the `StudyResult` feeds the Evidence Report (day 30).

## Try with example (offline, seconds)

**Try with example** analyses nine and a half hours of the heated tank with the six labelled faults
(`examples/tank_six_faults.csv`). It finds all six faults with no false-alarm event. The leak is quantified as
−0.39 L/s in the mass balance (true: 0.40 L/s), and the heater loss as −27 kW in the energy balance (true: 10 % of
about 280 kW).

## Limits

- Detectors learn "normal" from the declared window. A fault already present there becomes normal.
- A broken balance says that something enters or leaves, or that a meter is wrong. It does not say which one.
  Sensor-fault isolation is future work.
- The benchmark is synthetic until a public labelled set (BATADAL, PR #731) is available.
- The statistical and time-series channels can fire at large operating changes (0.1-0.6 false alarms per hour in our
  runs). Base decisions on the balance and model events.

Monetization: monitored assets, data volume and analysis frequency.
