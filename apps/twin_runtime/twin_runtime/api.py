"""Digital Twin Runtime (day 20): put a validated physical model or surrogate to work next to live data.

One model type at a time (the heated-tank lumped model, or a reduced model given as a ModelRef) and one live data
source per asset (the built-in simulated plant over MQTT, an external MQTT topic, REST pushes, or OPC UA,
experimental). Every computation is a library call (``pinneapple_systems.operations``); this module is routes,
uploads and the UI.

Run:  uvicorn twin_runtime.api:app --port 8103   (from apps/twin_runtime; one worker: live twins live in memory)

Environment (all optional):
  TWIN_USER / TWIN_PASSWORD   HTTP Basic login on everything except /health
  TWIN_MAX_JOBS               replays / series flows running at once (default 2)
  TWIN_MAX_ASSETS             live twins at once (default 4)
  TWIN_DB_URL                 history database: SQLite path or postgresql://... (default <data dir>/twin.sqlite)
  TWIN_MQTT_HOST/PORT         embedded broker address (default 127.0.0.1 and a free port; 0.0.0.0:1883 to accept
                              external publishers when the port is published)
  TWIN_MQTT_URL               mqtt://user:pass@host:port of an external broker instead of the embedded one
  TWIN_OPCUA_URL, TWIN_OPCUA_NODES   experimental OPC UA source: server URL and a JSON {field: node id} map
  TWIN_FREE_*                 monthly free units (see install_runtime)
"""
from __future__ import annotations

import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Body, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_core.jsonable import to_jsonable
from pinneapple_data.contracts import ModelRef, read_table
from pinneapple_systems.operations.contracts import load_model, twin_study
from pinneapple_systems.operations.live import LiveTwin, make_twin, replay
from pinneapple_systems.operations.plant import FAULT_KINDS, SENSOR_NAMES
from pinneapple_systems.operations.series import run_series_flow
from pinneapple_systems.operations.twin import LumpedTankModel, TwinStore

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
HERE = Path(__file__).parent
STATIC = HERE / "static"
EXAMPLES = HERE.parent / "examples"
APPS_DIR = HERE.parent.parent
MAX_ASSETS = int(os.environ.get("TWIN_MAX_ASSETS") or 4)

LIVE: dict[str, LiveTwin] = {}


@asynccontextmanager
async def lifespan(_app):
    yield
    for lt in list(LIVE.values()):                         # stop live twins (threads, embedded brokers)
        lt.stop()


app = FastAPI(title="Digital Twin Runtime", version=VERSION, lifespan=lifespan,
              description="Run a validated lumped model or reduced-order surrogate next to live data: EnKF state "
                          "estimation, forecasts with bands, rule-based alerts, drift monitoring and revalidation. "
                          "Readings arrive by MQTT, REST or (experimental) OPC UA; other systems read the state at "
                          "/api/assets/{asset}/state. Every number comes from the run itself.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="TWIN")
rt = install_runtime(app, prefix="TWIN", name="twin", max_jobs=2,
                     units={"replays": 200, "series_flows": 50, "readings": 500000, "live_assets": 30},
                     upload_suffixes={".csv", ".parquet", ".json"})
DB_URL = os.environ.get("TWIN_DB_URL") or str(rt.root / "twin.sqlite")
STORE = TwinStore(DB_URL)
MODELS = rt.root / "models"
MODELS.mkdir(parents=True, exist_ok=True)
OPTIONAL_BODY = Body(default={})
REQUIRED_BODY = Body(...)
LOCK = threading.Lock()


def scope() -> dict:
    return {"title": "Scope & validation", "noun": "limitation", "items_title": "What this version does not do",
            "validated": [
                "EnKF on the heated tank: leak estimated within 15 % of the truth (0.589 vs 0.600 L/s seen), heater "
                "efficiency within 0.03 (0.749 vs 0.75 seen), no alert on fault-free held-out runs "
                "(tests/test_operations_plant_twin.py)",
                "Forecast 5-95 % band contains the true level over 60 steps (10 min), input uncertainty included",
                "A drain-valve fault on the surrogate twin raises the consistency and revalidation alerts; the "
                "open-loop revalidation then fails, as it should",
                "Series flow: the optimised operating point is within 5 % of the closed-form physics optimum; "
                "revalidation is requested only after the inlet water leaves the surrogate's training range",
                "Embedded MQTT broker tested with paho-mqtt: QoS 0/1, wildcards, retained messages, login",
            ],
            "band": "Numbers on screen come from the running twin: estimates, bands, alerts and checks are recomputed "
                    "from the data that arrived.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can miss a problem", "check": "Know this"},
            "items": [
                {"topic": "One model type", "effect": "check",
                 "detail": "The heated-tank lumped model (mass + energy balance, optionally with leak and heater "
                           "efficiency in the state) or a reduced model (operator inference with inputs) given as a "
                           "ModelRef. Not a universal plant library.",
                 "today": "Bring your asset as a ROM fitted in the Surrogate Studio, or as the lumped model with "
                          "your parameters.", "planned": "FMU (FMI 2.0) co-simulation models."},
                {"topic": "One live source per asset", "effect": "check",
                 "detail": "Simulated plant over MQTT, an external MQTT topic, REST pushes, or OPC UA polling. OPC UA "
                           "is experimental and needs asyncua in the image.",
                 "today": "Use MQTT or REST for anything that matters.", "planned": "OPC UA subscriptions, Modbus."},
                {"topic": "Alerts are screening", "effect": "optimistic",
                 "detail": "Rules on limits, forecasts, estimated parameters and filter consistency. Not a safety "
                           "system and not certified for protection functions.",
                 "today": "Keep the plant's own alarms and interlocks.", "planned": "Alert routing (e-mail, webhooks)."},
                {"topic": "Gaussian assumptions", "effect": "check",
                 "detail": "The EnKF assumes Gaussian noise and tracks parameters as random walks; the validity "
                           "decision (assess_drift) is a screening heuristic.",
                 "today": "Read the reasons of a revalidation alert, then revalidate.",
                 "planned": "Particle filter for strongly non-Gaussian states."},
                {"topic": "Live twins live in memory", "effect": "check",
                 "detail": "A restart stops the live twins; the history stays in the database.",
                 "today": "Restart the twin from the UI or the API.", "planned": "Automatic resume."},
            ]}


# --------------------------------------------------------------------------------------------------- models

def _example_rom_ref() -> ModelRef:
    return ModelRef.load(EXAMPLES / "tank_rom.json.modelref.json")


def _model(model: str, model_id: str | None):
    """``lumped`` | ``rom`` (the example surrogate) | ``uploaded`` (``model_id`` from /api/models)."""
    if model == "lumped":
        return LumpedTankModel(dt=10.0)
    if model == "rom":
        return load_model(_example_rom_ref(), EXAMPLES)
    if model == "uploaded":
        d = MODELS / str(model_id or "")
        if not model_id or not d.is_dir() or "/" in model_id or ".." in model_id:
            raise HTTPException(404, "unknown model_id; upload a ModelRef first (POST /api/models)")
        ref = ModelRef.load(d / "modelref.json")
        return load_model(ref, d)
    raise HTTPException(422, "model is lumped, rom or uploaded")


@app.post("/api/models")
async def upload_model(request: Request, modelref: UploadFile, model_file: UploadFile):
    """Register a surrogate: its ``ModelRef`` (JSON) and the model file it points to (a controlled-OpInf JSON,
    e.g. from the Surrogate Studio or the series flow). The SHA-256 in the ModelRef is checked."""
    try:
        ref = ModelRef.model_validate(json.loads(await modelref.read()))
    except Exception as e:
        raise HTTPException(422, f"not a valid ModelRef: {e}") from e
    data = await model_file.read()
    if len(data) > 5 * 2**20:
        raise HTTPException(413, "model files up to 5 MB")
    import uuid

    mid = uuid.uuid4().hex[:12]
    d = MODELS / mid
    d.mkdir()
    (d / Path(ref.uri).name).write_bytes(data)
    ref = ref.model_copy(update={"uri": Path(ref.uri).name})
    ref.save(d / "modelref.json")
    try:
        m = load_model(ref, d)
    except (ValueError, KeyError, FileNotFoundError) as e:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
        raise HTTPException(422, str(e)) from e
    return {"model_id": mid, "kind": m.kind, "states": m.state_names, "inputs": m.input_names,
            "metrics": ref.metrics, "metrics_split": ref.metrics_split}


# --------------------------------------------------------------------------------------------------- live

def _mqtt_settings() -> dict:
    url = os.environ.get("TWIN_MQTT_URL")
    if url:
        u = urlparse(url)
        return {"embedded_broker": False, "broker_host": u.hostname or "127.0.0.1", "broker_port": u.port or 1883,
                "mqtt_user": u.username, "mqtt_password": u.password}
    return {"embedded_broker": True, "broker_host": os.environ.get("TWIN_MQTT_HOST") or "127.0.0.1",
            "broker_port": int(os.environ.get("TWIN_MQTT_PORT") or 0)}


def _live(asset: str) -> LiveTwin:
    lt = LIVE.get(asset)
    if lt is None:
        raise HTTPException(404, f"no live twin for asset {asset!r}")
    return lt


@app.post("/api/live/start")
def live_start(request: Request, body: dict = OPTIONAL_BODY):
    """Start a live twin. ``asset`` (default tank-1), ``model`` lumped | rom | uploaded (+ ``model_id``),
    ``source`` simulated | mqtt | rest | opcua, ``speed`` (simulated seconds per wall second, simulated source)."""
    asset = str(body.get("asset") or "tank-1")
    if not asset.replace("-", "").replace("_", "").isalnum() or len(asset) > 40:
        raise HTTPException(422, "asset ids are letters, digits, - and _ (40 at most)")
    source = body.get("source") or "simulated"
    speed = float(body.get("speed") or 60.0)
    if not 1.0 <= speed <= 600.0:
        raise HTTPException(422, "speed between 1 and 600")
    model = _model(body.get("model") or "lumped", body.get("model_id"))
    with LOCK:
        if asset in LIVE:
            LIVE.pop(asset).stop()
        if len(LIVE) >= MAX_ASSETS:
            raise HTTPException(429, f"{MAX_ASSETS} live twins at most on this server; stop one first")
        rt.consume(request, "live_assets", 1, ref=asset)
        kw = _mqtt_settings() if source in ("simulated", "mqtt") else {}
        if source == "opcua":
            url, nodes = os.environ.get("TWIN_OPCUA_URL"), os.environ.get("TWIN_OPCUA_NODES")
            if not url or not nodes:
                raise HTTPException(422, "OPC UA (experimental) needs TWIN_OPCUA_URL and TWIN_OPCUA_NODES on the server")
            try:
                import asyncua  # noqa: F401
            except ImportError as e:
                raise HTTPException(422, "OPC UA (experimental) needs asyncua installed in the image") from e
            kw = {"opcua_url": url, "opcua_nodes": json.loads(nodes)}
        try:
            twin = make_twin(model, asset=asset, store=STORE)
            lt = LiveTwin(twin, source=source, speed=speed, dt=model.dt, seed=int(body.get("seed") or 0),
                          **kw).start()
        except (ValueError, OSError, RuntimeError) as e:
            raise HTTPException(422, f"could not start: {e}") from e
        LIVE[asset] = lt
    return {"asset": asset, "source": source, "model": model.kind, "topic": lt.topic,
            "broker_port": lt.broker.port if lt.broker else None}


@app.post("/api/live/{asset}/stop")
def live_stop(asset: str):
    with LOCK:
        lt = LIVE.pop(asset, None)
    if lt is None:
        raise HTTPException(404, f"no live twin for asset {asset!r}")
    lt.stop()
    return {"stopped": asset, "n_updates": lt.twin.n_updates}


@app.post("/api/live/{asset}/fault")
def live_fault(asset: str, body: dict = REQUIRED_BODY):
    """Inject a fault in the simulated plant (demo only): ``kind`` in leak, heater, valve, sensor_bias,
    sensor_drift, stuck_sensor; ``magnitude``; ``target`` sensor for sensor faults."""
    lt = _live(asset)
    if body.get("kind") not in FAULT_KINDS:
        raise HTTPException(422, f"kind is one of {FAULT_KINDS}")
    try:
        return lt.inject_fault(body["kind"], float(body.get("magnitude") or 0.0), str(body.get("target") or ""))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(422, str(e)) from e


@app.post("/api/live/{asset}/clear")
def live_clear(asset: str):
    return {"cleared_faults": _live(asset).clear_faults()}


@app.post("/api/live/{asset}/revalidate")
def live_revalidate(asset: str, n_last: int = 180):
    lt = _live(asset)
    try:
        with lt._lock:
            return lt.twin.revalidate(n_last=n_last)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/api/assets")
def assets():
    return [{"asset": a, "source": lt.source, "model": lt.twin.model.kind, "status": lt.twin.status,
             "n_updates": lt.twin.n_updates, "running": lt.running} for a, lt in LIVE.items()]


@app.get("/api/assets/{asset}/live")
def asset_live(asset: str, n: int = 360, forecast_steps: int = 60):
    """State, recent history, forecast and alerts of a live twin (what the UI draws)."""
    lt = _live(asset)
    d = lt.snapshot(n_history=min(n, 2000), forecast_steps=min(forecast_steps, 360))
    d.update(limits=lt.twin.rules.limits, nis_threshold=lt.twin.nis_threshold)
    return to_jsonable(d)


@app.get("/api/assets/{asset}/state")
def asset_state(asset: str):
    """The endpoint for other systems: current estimate with its spread, status, active alerts, validity."""
    lt = _live(asset)
    with lt._lock:
        return to_jsonable(lt.twin.snapshot())


@app.get("/api/assets/{asset}/history")
def asset_history(asset: str, table: str = "estimates", since: float = float("-inf"), limit: int = 5000):
    """Stored history (``estimates``, ``readings``, ``alerts`` or ``events``) from the database."""
    if table not in TwinStore.TABLES:
        raise HTTPException(422, f"table is one of {TwinStore.TABLES}")
    return STORE.read(table, asset, since=since, limit=min(limit, 50000))


@app.post("/api/assets/{asset}/readings")
def push_readings(request: Request, asset: str, body: dict | list = REQUIRED_BODY):
    """Push one reading or a list (``t`` [s] and the model's inputs and observed states) to a live twin started
    with ``source`` = ``rest`` (or any live twin)."""
    lt = _live(asset)
    rows = body if isinstance(body, list) else [body]
    if len(rows) > 10000:
        raise HTTPException(413, "10,000 readings per request at most")
    rt.consume(request, "readings", len(rows), ref=asset)
    out = None
    try:
        for r in rows:
            out = lt.push({k: float(v) for k, v in r.items()})
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(422, str(e)) from e
    return to_jsonable({"accepted": len(rows), "last": out})


@app.get("/api/assets/{asset}/study")
def asset_study(asset: str):
    """The live twin's run as a ``StudyResult`` contract."""
    lt = _live(asset)
    with lt._lock:
        st = twin_study(lt.twin, app="twin")
    return Response(st.to_json(), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{asset}.twin.study.json"'})


# --------------------------------------------------------------------------------------------------- jobs

def _replay_job(params: dict, ctx) -> dict:
    ctx.progress(0.02, "reading the table")
    if params.get("example"):
        df, _ = read_table(EXAMPLES / "tank_day_with_leak.csv")
    else:
        df, _ = read_table(params["path"])
    model = _model(params.get("model") or "lumped", params.get("model_id"))
    tw, res = replay(df, model, asset=params.get("asset") or "replay", progress=ctx.progress)
    st = twin_study(tw, app="twin", title=f"Replay of {params.get('name') or 'the example record'}")
    (ctx.workdir / "twin.study.json").write_text(st.to_json())
    ctx.progress(1.0, "done")
    res["study"] = {"checks": [c.model_dump() for c in st.checks], "passed": st.passed, "metrics": st.metrics}
    res["model"] = model.kind
    res.update(limits=tw.rules.limits, nis_threshold=tw.nis_threshold)
    res["truth"] = {k: df[k].tolist() for k in ("true_q_leak",) if k in df}
    return to_jsonable(res)


def _series_job(params: dict, ctx) -> dict:
    out = run_series_flow(ctx.workdir, seed=int(params.get("seed") or 0), dt=10.0, apps_dir=APPS_DIR,
                          progress=ctx.progress)
    out["files"] = sorted(p.name for p in ctx.workdir.iterdir() if p.is_file() and p.name != "output.log")
    return to_jsonable(out)


rt.jobs.register("replay", _replay_job)
rt.jobs.register("series", _series_job)


@app.post("/api/replay")
def api_replay(request: Request, body: dict = OPTIONAL_BODY):
    """Run the twin over a recorded table: ``upload_id`` (from POST /api/uploads; CSV, Parquet or JSON with t, h, T,
    q_in, Q, T_in, T_amb) or ``example: true``; ``model`` lumped | rom | uploaded (+ ``model_id``). Returns a job."""
    params = {"model": body.get("model") or "lumped", "model_id": body.get("model_id"), "asset": "replay"}
    if body.get("example"):
        params["example"] = True
    else:
        path = rt.upload_path(request, str(body.get("upload_id") or ""))
        params.update(path=str(path), name=path.name)
    _model(params["model"], params["model_id"])               # fail fast on a bad model choice
    return rt.submit(request, "replay", params, units={"replays": 1})


@app.post("/api/series")
def api_series(request: Request, body: dict = OPTIONAL_BODY):
    """Run the series flow campaign -> surrogate -> uncertainty -> optimizer -> twin (about 10 s). Returns a job."""
    return rt.submit(request, "series", {"seed": int(body.get("seed") or 0)}, units={"series_flows": 1})


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(request: Request, job_id: str, name: str):
    """A contract file written by a replay or series job (StudyResult, ModelRef, DatasetSpec, Parquet table)."""
    rt.job_for(request, job_id)
    if "/" in name or name.startswith(".") or name == "output.log":
        raise HTTPException(404, "no such file")
    p = rt.jobs.workdir / job_id / name
    if not p.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(p, filename=name)


# --------------------------------------------------------------------------------------------------- meta

@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    ref = _example_rom_ref()
    return to_jsonable({"version": VERSION, "fault_kinds": FAULT_KINDS, "sensors": SENSOR_NAMES,
                        "max_assets": MAX_ASSETS, "db": "postgres" if DB_URL.startswith("postgres") else "sqlite",
                        "mqtt": "external" if os.environ.get("TWIN_MQTT_URL") else "embedded",
                        "opcua": bool(os.environ.get("TWIN_OPCUA_URL")),
                        "example_rom": {"name": ref.name, "metrics": ref.metrics, "inputs": [v.model_dump()
                                                                                             for v in ref.inputs]},
                        "examples": ["tank_day_with_leak.csv", "tank_rom.json", "tank_rom.json.modelref.json"],
                        "scope": scope()})


@app.get("/api/example-file/{name}")
def example_file(name: str):
    if name not in ("tank_day_with_leak.csv", "tank_rom.json", "tank_rom.json.modelref.json"):
        raise HTTPException(404, "no such file")
    return FileResponse(EXAMPLES / name, filename=name)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
