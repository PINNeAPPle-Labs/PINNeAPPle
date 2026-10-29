"""10 — Wrap a trained model as a live digital twin.

A digital twin is a trained surrogate plus a running loop: sensor data comes in,
the field is re-evaluated, observations are compared with predictions and
anomalies are flagged. `MockStream` needs no broker; production streams are
`MQTTStream`, `KafkaStream`, `OPCUAStream`, `ModbusStream`, `HTTPPollStream`.

Run from the repository root:

    python examples/quickstart/10_digital_twin.py
"""
from __future__ import annotations

import time

import numpy as np

import pinneapple as pp
from pinneapple_systems.digital_twin import build_digital_twin
from pinneapple_systems.digital_twin.io import MockStream

# --- 1. A trained surrogate (the model that drives the twin) ------------
sol = pp.solve("burgers_1d", method="pinn", epochs=400)
print("surrogate:", type(sol.model).__name__)

# --- 2. Wrap it as a twin ------------------------------------------------
twin = build_digital_twin(
    sol.model,
    field_names=["u"],
    coord_names=["x", "t"],
    update_interval=0.1,          # seconds between field updates
    anomaly_z_threshold=3.0,      # z-score above which an observation is flagged
)
twin.set_domain_coords({
    "x": np.tile(np.linspace(-1.0, 1.0, 40), 20),
    "t": np.repeat(np.linspace(0.0, 1.0, 20), 40),
})

# --- 3. A sensor stream (a fault appears after 1.5 s) -------------------
def sensor(t: float) -> dict:
    value = float(sol.predict([[0.0, min(t / 3.0, 1.0)]])[0, 0])
    return {"u": value + (5.0 if t > 1.5 else 0.0)}


twin.add_stream(MockStream("probe", ["u"], sensor, coords={"x": 0.0, "t": 0.0}))

# --- 4. Run it -----------------------------------------------------------
with twin:
    time.sleep(3.0)

print(f"field mean      : {np.mean(twin.state.fields['u']):.4f}")
print(f"history records : {len(twin.get_history_df())}")
events = twin.anomaly_monitor.recent_events()
print(f"anomalies       : {len(events)}")
for event in events[-3:]:
    print(f"  {event.field_name}: observed {event.observed:.3f}, predicted {event.predicted:.3f}, "
          f"score {event.score:.2f} ({event.detector})")
