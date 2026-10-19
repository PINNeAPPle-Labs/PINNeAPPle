"""Run a :class:`~pinneapple_systems.operations.twin.TwinRuntime` as a service: on a live stream (MQTT, REST pushes,
an experimental OPC UA poller) or over a historical table.

``LiveTwin`` wires the pieces the twin app needs and nothing else:

* a source -- the built-in simulated heated tank publishing JSON readings to an MQTT topic (through the embedded
  broker or an external one), REST pushes (:meth:`LiveTwin.push`), or an OPC UA server polled by
  :class:`~pinneapple_systems.digital_twin.io.stream.OPCUAStream` (experimental, needs ``asyncua``);
* a consumer thread that subscribes with :class:`~pinneapple_systems.digital_twin.io.stream.MQTTStream` and feeds
  every reading to the twin, which writes its history to a :class:`~.twin.TwinStore`.

Simulated time runs ``speed`` times faster than the wall clock so a demo shows an hour of plant in a minute.
:func:`replay` runs the same twin over a recorded table (CSV / Parquet / JSON) in one go.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Sequence

import numpy as np

from pinneapple_systems.digital_twin.io.mqtt_broker import MQTTBroker
from pinneapple_systems.digital_twin.io.stream import MQTTStream, OPCUAStream

from .plant import FAULT_KINDS, Fault, TankParams, TankSimulator, demo_scenario
from .twin import AlertRules, LumpedTankModel, SurrogateModel, TwinRuntime, TwinStore

READING_FIELDS = ["h", "T", "q_in", "q_out", "Q", "T_in", "T_amb"]
TWIN_SIGMA_OBS = {"h": 0.003, "T": 0.05}
TWIN_SIGMA_PROCESS = {"h": 2e-4, "T": 0.01, "q_leak": 2e-5, "eta": 2e-3}


def make_twin(model, *, asset: str = "tank-1", store: TwinStore | None = None, rules: AlertRules | None = None,
              sigma_obs: dict | None = None, sigma_process: dict | None = None, seed: int = 0) -> TwinRuntime:
    """A twin of the heated tank with the noise levels of its sensors (override for another asset)."""
    sp = dict(TWIN_SIGMA_PROCESS)
    if isinstance(model, SurrogateModel):
        sp.update({"h": 3e-4, "T": 0.02})
    sp.update(sigma_process or {})
    return TwinRuntime(model, ["h", "T"], sigma_obs or TWIN_SIGMA_OBS, sp, rules=rules, asset=asset, store=store,
                       seed=seed)


class LiveTwin:
    """One asset's live twin. ``source``: ``"simulated"`` (built-in plant over MQTT), ``"mqtt"`` (readings from an
    external publisher on ``topic``), ``"rest"`` (only :meth:`push`) or ``"opcua"`` (experimental)."""

    def __init__(self, twin: TwinRuntime, *, source: str = "simulated", topic: str | None = None,
                 broker_host: str = "127.0.0.1", broker_port: int = 0, embedded_broker: bool = True,
                 mqtt_user: str | None = None, mqtt_password: str | None = None, speed: float = 60.0,
                 dt: float = 10.0, seed: int = 0, opcua_url: str | None = None,
                 opcua_nodes: dict[str, str] | None = None):
        if source not in ("simulated", "mqtt", "rest", "opcua"):
            raise ValueError(f"unknown source {source!r}")
        self.twin = twin
        self.source = source
        self.topic = topic or f"pinneapple/{twin.asset}/sensors"
        self.speed, self.dt = float(speed), float(dt)
        self.seed = seed
        self.broker: MQTTBroker | None = None
        self._embedded = embedded_broker
        self._host, self._port = broker_host, int(broker_port)
        self._user, self._password = mqtt_user, mqtt_password
        self._opcua = (opcua_url, opcua_nodes or {})
        self._q: queue.Queue = queue.Queue()
        self._stream = None
        self._threads: list[threading.Thread] = []
        self._running = threading.Event()
        self._lock = threading.Lock()
        self.sim: TankSimulator | None = None
        self.published = 0
        self.errors: list[str] = []
        self.started_at: float | None = None

    # ------------------------------------------------------------------------------------------------ lifecycle
    def start(self) -> LiveTwin:
        self._running.set()
        self.started_at = time.time()
        if self.source in ("simulated", "mqtt"):
            if self._embedded:
                self.broker = MQTTBroker(self._host, self._port, self._user, self._password).start()
                self._port = self.broker.port
            self._stream = MQTTStream(self._host, self.topic, self.twin.asset, READING_FIELDS, port=self._port,
                                      username=self._user, password=self._password)
            self._stream.start(self._q)
        elif self.source == "opcua":
            url, nodes = self._opcua
            if not url or not nodes:
                raise ValueError("OPC UA needs a server URL and a {field: node id} map")
            self._stream = OPCUAStream(url, nodes, self.twin.asset, READING_FIELDS, poll_interval=self.dt / self.speed)
            self._stream.start(self._q)
        self._spawn(self._consume, "twin-consumer")
        if self.source == "simulated":
            self._spawn(self._simulate, "plant-simulator")
        return self

    def _spawn(self, target, name):
        th = threading.Thread(target=target, name=name, daemon=True)
        th.start()
        self._threads.append(th)

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
        for th in self._threads:
            th.join(timeout=5.0)
        if self.broker is not None:
            self.broker.stop()
        self._threads.clear()

    @property
    def running(self) -> bool:
        return self._running.is_set()

    # ------------------------------------------------------------------------------------------------ sources
    def _simulate(self) -> None:
        import paho.mqtt.client as mqtt

        sc = demo_scenario(10 ** 6, self.dt, seed=self.seed)
        self.sim = TankSimulator(TankParams(), sc.x0, seed=self.seed)
        cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"plant-{self.twin.asset}")
        if self._user:
            cli.username_pw_set(self._user, self._password)
        time.sleep(0.2)                                        # let the subscriber connect first
        cli.connect(self._host, self._port)
        cli.loop_start()
        try:
            while self._running.is_set():
                t0 = time.perf_counter()
                with self._lock:
                    t = self.sim.t
                    u, d = (sc.q_in(t), sc.Q(t)), (sc.T_in(t), sc.T_amb(t))
                    reading = self.sim.read(u, d)
                    self.sim.step(u, d, self.dt)
                cli.publish(self.topic, json.dumps({"timestamp": t, **reading}), qos=0)
                self.published += 1
                time.sleep(max(0.0, self.dt / self.speed - (time.perf_counter() - t0)))
        finally:
            cli.loop_stop()
            cli.disconnect()

    def _consume(self) -> None:
        while self._running.is_set():
            try:
                obs = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.push({"t": obs.timestamp, **obs.values})
            except (ValueError, KeyError) as e:
                self.errors.append(f"{type(e).__name__}: {e}")
                del self.errors[:-20]

    def push(self, reading: dict) -> dict:
        """Assimilate one reading (REST ingestion, or called by the stream consumer)."""
        with self._lock:
            return self.twin.ingest(reading)

    # ------------------------------------------------------------------------------------------------ control
    def inject_fault(self, kind: str, magnitude: float = 0.0, target: str = "", duration: float = 1e9) -> dict:
        """Start a fault in the simulated plant now (demo only)."""
        if self.sim is None:
            raise RuntimeError("faults can only be injected into the built-in simulated plant")
        if kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault {kind!r}; one of {FAULT_KINDS}")
        with self._lock:
            f = Fault(kind, self.sim.t, self.sim.t + duration, magnitude, target)
            self.sim.faults.append(f)
        return f.to_dict()

    def clear_faults(self) -> int:
        if self.sim is None:
            return 0
        with self._lock:
            n = len(self.sim.faults)
            self.sim.faults.clear()
        return n

    def snapshot(self, n_history: int = 360, forecast_steps: int = 60) -> dict:
        """State, recent history (thinned), forecast and stream statistics, JSON-ready."""
        with self._lock:
            snap = self.twin.snapshot()
            hist = list(self.twin.history)[-n_history:]
            fc = self.twin.forecast(forecast_steps) if self.twin.t is not None else None
            faults = [f.to_dict() for f in self.sim.faults] if self.sim is not None else []
            truth = None
            if self.sim is not None:
                truth = {"t": self.sim.t, "h": float(self.sim.x[0]), "T": float(self.sim.x[1])}
        return {**snap, "source": self.source, "topic": self.topic, "broker_port": self._port if self.broker else None,
                "published": self.published, "running": self.running, "speed": self.speed, "faults": faults,
                "plant_truth": truth, "errors": self.errors[-5:], "history": _thin(hist), "forecast": fc,
                "alerts": [a.to_dict() for a in self.twin.alerts[-50:]]}


def _thin(hist: Sequence[dict]) -> dict:
    if not hist:
        return {}
    keys = list(hist[-1]["state"])
    out = {"t": [h["t"] for h in hist], "status": [h["status"] for h in hist]}
    for k in keys:
        out[k] = [h["state"].get(k) for h in hist]
        out[f"{k}_sd"] = [h["state_sd"].get(k) for h in hist]
    for k in hist[-1]["observed"]:
        out[f"obs_{k}"] = [h["observed"].get(k) for h in hist]
    out["nis"] = [h.get("nis") for h in hist]
    return out


def replay(table, model, *, asset: str = "tank-1", store: TwinStore | None = None,
           rules: AlertRules | None = None, forecast_steps: int = 60, revalidate: bool = True,
           progress=None) -> tuple[TwinRuntime, dict]:
    """Run a twin over a recorded table (DataFrame or dict of columns with ``t`` and the model's inputs and
    observed states). Returns the twin and a JSON-ready result with the history, alerts, final forecast and, when
    ``revalidate``, an open-loop re-score of the model on the last part of the record."""
    cols = {str(k): np.asarray(v) for k, v in (table.items() if isinstance(table, dict) else table.items())}
    need = ["t", "h", "T"] + list(model.input_names)
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"the table lacks columns {missing}; it needs {need}")
    tw = make_twin(model, asset=asset, store=store, rules=rules)
    n = len(cols["t"])
    for i in range(n):
        tw.ingest({k: float(cols[k][i]) for k in need})
        if progress and i % 200 == 0:
            progress(0.05 + 0.85 * i / n, f"assimilated {i}/{n} readings")
    reval = tw.revalidate(n_last=min(180, n)) if revalidate and n >= 10 else None
    hist = list(tw.history)
    return tw, {"n": n, "history": _thin(hist), "alerts": [a.to_dict() for a in tw.alerts],
                "forecast": tw.forecast(forecast_steps), "snapshot": tw.snapshot(), "revalidation": reval,
                "validity": tw.last_drift}


def demo_record(duration: float = 7200.0, dt: float = 10.0, seed: int = 4,
                faults: Sequence[Fault] | None = None) -> dict:
    """A recorded run of the simulated tank with a 0.5 L/s leak over its second half (the twin app's offline
    example)."""
    from .plant import simulate

    faults = [Fault("leak", duration / 2, duration + dt, 5e-4)] if faults is None else list(faults)
    return simulate(demo_scenario(duration, dt, seed=seed), faults=faults, seed=seed)


def default_model(kind: str = "lumped", dt: float = 10.0):
    if kind != "lumped":
        raise ValueError("only the built-in lumped model has a default; load a ModelRef for a surrogate")
    return LumpedTankModel(dt=dt)
