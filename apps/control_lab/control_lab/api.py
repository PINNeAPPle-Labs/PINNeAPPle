"""Physics-Based Control Lab (day 29): compute better control actions from physical models, in simulation or as
recommendations only.

Model predictive control (linear, and nonlinear with CasADi/IPOPT or SciPy), with a reduced-order surrogate as an
alternative model, compared with PID and rule-based baselines on the same simulated plant. There is no command path
to real equipment: the lab simulates, and recommends actions for a person to review. Every computation is a library
call (``pinneapple_systems.operations.control``); this module is routes and the UI.

Run:  uvicorn control_lab.api:app --port 8112   (from apps/control_lab)

Environment (all optional):
  CONTROL_USER / CONTROL_PASSWORD   HTTP Basic login on everything except /health
  CONTROL_MAX_JOBS                  comparisons running at once (default 2)
  CONTROL_FREE_*                    monthly free units (see install_runtime)
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pinneapple_core.jsonable import to_jsonable
from pinneapple_data.contracts import ModelRef
from pinneapple_systems.operations.contracts import control_study, load_model
from pinneapple_systems.operations.control import (
    MODE,
    ControlScenario,
    ControlSpec,
    compare_controllers,
    recommend,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
HERE = Path(__file__).parent
STATIC = HERE / "static"
EXAMPLES = HERE.parent / "examples"
CONTROLLERS = {"pid": "PID (two loops, anti-windup)", "rules": "Rule-based (hysteresis)",
               "lmpc": "Linear MPC (successive linearisation, QP)", "nmpc": "Nonlinear MPC (physics model)",
               "smpc": "Nonlinear MPC on the reduced-order surrogate"}
OPTIONAL_BODY = Body(default={})

try:
    import casadi  # noqa: F401
    CASADI = True
except ImportError:  # pragma: no cover
    CASADI = False

app = FastAPI(title="Physics-Based Control Lab", version=VERSION,
              description="Model predictive control (linear and nonlinear) from physical models and surrogates, "
                          "compared with PID and rule-based baselines on the same simulated plant: reference "
                          "trajectories, proposed actions, predicted and simulated performance. " + MODE + ".")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="CONTROL")
rt = install_runtime(app, prefix="CONTROL", name="control", max_jobs=2,
                     units={"studies": 100, "scenarios": 400, "recommendations": 2000},
                     upload_suffixes={".json"})
MODELS = rt.root / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def scope() -> dict:
    return {"title": "Scope & validation", "noun": "limitation", "items_title": "What the lab assumes",
            "validated": [
                "On the same plant, scenario and noise, linear MPC cuts the temperature tracking error (IAE) below "
                "0.6x PID's with no constraint violation and no failed solve (tests/test_operations_anomaly_control.py)",
                "Actuator limits and rate limits hold at every step of every controller",
                "CasADi/IPOPT and SciPy nonlinear MPC agree on the proposed action",
                "Every metric on screen is recomputed from the closed-loop simulation of this request",
            ],
            "band": MODE + ". The lab simulates and recommends; it never writes to a controller, PLC or DCS.",
            "tags": {"conservative": "Safe by default", "optimistic": "Better than reality", "check": "Know this"},
            "items": [
                {"topic": "Simulated plant", "effect": "optimistic",
                 "detail": "The comparison runs on the heated-tank model with sensor noise and unmeasured "
                           "disturbances. The MPC's model is the same physics with estimated parameters, so the "
                           "model mismatch is smaller than on a real plant.",
                 "today": "Treat the ranking as the potential of each approach, not a guaranteed gain.",
                 "planned": "Plant-model mismatch scenarios and calibration from historian data."},
                {"topic": "Baselines are tuned, not optimal", "effect": "check",
                 "detail": "PID gains and the rule bands are reasonable hand-tuned values, editable in the API.",
                 "today": "Enter your own gains and bands to compare against what runs today.",
                 "planned": "Automatic PID tuning (IMC) as a third baseline."},
                {"topic": "Soft constraints", "effect": "check",
                 "detail": "Output limits are soft (heavily penalised) so that infeasible disturbances degrade "
                           "gracefully; actuator and rate limits are hard.",
                 "today": "Check the violation columns; zero means the limits held.",
                 "planned": "Hard output constraints with feasibility recovery."},
                {"topic": "No command path", "effect": "conservative",
                 "detail": "There is no interface to real equipment in this version, on purpose.",
                 "today": "Use the recommendation and the plan as decision support.",
                 "planned": "None in this app; deployment to a controller is a separate, certified step."},
            ]}


def _surrogate(choice: str | None, model_id: str | None):
    if not choice or choice == "none":
        return None
    if choice == "example":
        return load_model(ModelRef.load(EXAMPLES / "tank_rom.json.modelref.json"), EXAMPLES)
    if choice == "uploaded":
        d = MODELS / str(model_id or "")
        if not model_id or "/" in model_id or ".." in model_id or not d.is_dir():
            raise HTTPException(404, "unknown model_id; upload a ModelRef first (POST /api/models)")
        return load_model(ModelRef.load(d / "modelref.json"), d)
    raise HTTPException(422, "surrogate is none, example or uploaded")


@app.post("/api/models")
async def upload_model(modelref: UploadFile, model_file: UploadFile):
    """Register a surrogate (controlled-OpInf JSON) with its ``ModelRef``; its step must equal the controller's
    ``dt`` (20 s by default)."""
    try:
        ref = ModelRef.model_validate(json.loads(await modelref.read()))
    except Exception as e:
        raise HTTPException(422, f"not a valid ModelRef: {e}") from e
    data = await model_file.read()
    if len(data) > 5 * 2**20:
        raise HTTPException(413, "model files up to 5 MB")
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
    return {"model_id": mid, "kind": m.kind, "dt": m.dt, "states": m.state_names, "inputs": m.input_names}


def _parse(body: dict):
    try:
        spec = ControlSpec.from_dict(body.get("spec") or {})
        scenario = ControlScenario.from_dict(body.get("scenario") or {})
    except (TypeError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    if not 300 <= scenario.duration <= 4 * 3600:
        raise HTTPException(422, "duration between 300 s and 4 h")
    if spec.horizon > 40:
        raise HTTPException(422, "horizon up to 40 steps")
    return spec, scenario


def _compare_job(params: dict, ctx) -> dict:
    spec = ControlSpec.from_dict(params["spec"])
    scenario = ControlScenario.from_dict(params["scenario"])
    sur = _surrogate(params.get("surrogate"), params.get("model_id"))
    names = params["controllers"]
    runs = {}
    for i, name in enumerate(names):
        ctx.progress(0.05 + 0.9 * i / len(names), f"closed loop: {CONTROLLERS[name]}")
        runs.update(compare_controllers(spec, scenario, controllers=(name,), surrogate=sur)["runs"])
    out = {"mode": MODE, "spec": spec.to_dict(), "scenario": scenario.to_dict(), "runs": runs,
           "casadi_available": CASADI}
    st = control_study(out, title="Controller comparison on the heated tank")
    (ctx.workdir / "control.study.json").write_text(st.to_json())
    rows = []
    for name, r in runs.items():
        tr = r["trajectory"]
        for k in range(len(tr["t"])):
            rows.append(",".join([name] + [f"{tr[c][k]:.6g}" for c in ("t", "h", "T", "q_in", "Q", "h_ref", "T_ref",
                                                                     "T_in")]))
    (ctx.workdir / "trajectories.csv").write_text("controller,t,h,T,q_in,Q,h_ref,T_ref,T_in\n" + "\n".join(rows) + "\n")
    out["checks"] = [c.model_dump() for c in st.checks]
    ctx.progress(1.0, "done")
    return to_jsonable(out)


rt.jobs.register("compare", _compare_job)


@app.post("/api/compare")
def api_compare(request: Request, body: dict = OPTIONAL_BODY):
    """Closed-loop comparison on the simulated plant. ``spec`` (ControlSpec fields: setpoints, h_bounds, T_max,
    actuator bounds and rates, weights, dt, horizon), ``scenario`` (duration, T_in_steps, faults/disturbances,
    noise, seed), ``controllers`` (pid, rules, lmpc, nmpc, smpc), ``surrogate`` (none | example | uploaded +
    ``model_id``, for smpc). Returns a job."""
    spec, scenario = _parse(body)
    names = list(body.get("controllers") or ["pid", "rules", "lmpc", "nmpc"])
    bad = set(names) - set(CONTROLLERS)
    if bad or not names:
        raise HTTPException(422, f"controllers from {sorted(CONTROLLERS)}")
    if "smpc" in names:
        sur = _surrogate(body.get("surrogate") or "example", body.get("model_id"))
        if abs(sur.dt - spec.dt) > 1e-9:
            raise HTTPException(422, f"the surrogate steps {sur.dt} s; set spec.dt to it")
    params = {"spec": spec.to_dict(), "scenario": scenario.to_dict(), "controllers": names,
              "surrogate": body.get("surrogate") or ("example" if "smpc" in names else "none"),
              "model_id": body.get("model_id")}
    return rt.submit(request, "compare", params, units={"studies": 1, "scenarios": len(names)})


@app.post("/api/recommend")
def api_recommend(request: Request, body: dict = OPTIONAL_BODY):
    """Proposed next actions and the predicted trajectory from the current state, for a person to review.
    ``estimate`` {h, T, q_leak?, eta?} (e.g. the Digital Twin Runtime's ``/api/assets/{asset}/state`` ``state``),
    ``reading`` {q_in, Q, T_in, T_amb}, ``spec``, ``t`` (time in the set-point schedule), ``method`` lmpc | nmpc."""
    est, reading = body.get("estimate") or {}, body.get("reading") or {}
    if "state" in est and isinstance(est["state"], dict):           # a whole twin snapshot
        est = est["state"]
    missing = [k for k in ("h", "T") if k not in est] + [k for k in ("q_in", "Q", "T_in", "T_amb") if k not in reading]
    if missing:
        raise HTTPException(422, f"missing {missing}")
    try:
        spec = ControlSpec.from_dict(body.get("spec") or {})
        est = {k: float(v) for k, v in est.items() if k in ("h", "T", "q_leak", "eta")}
        reading = {k: float(v) for k, v in reading.items() if k in ("q_in", "Q", "T_in", "T_amb")}
    except (TypeError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    if not 0.05 < est["h"] < 3.0:
        raise HTTPException(422, "h outside the model's range (0.05-3 m)")
    rt.consume(request, "recommendations", 1)
    method = body.get("method") if body.get("method") in ("lmpc", "nmpc") else "nmpc"
    rec = recommend(est, reading, spec, t=float(body.get("t") or 0.0), method=method)
    rec["reference"] = {"t": rec["plan"]["plan_t"],
                        "h_ref": [spec.reference(t)[0] for t in rec["plan"]["plan_t"]],
                        "T_ref": [spec.reference(t)[1] for t in rec["plan"]["plan_t"]]}
    rec["constraints"] = {"h_bounds": spec.h_bounds, "T_max": spec.T_max}
    return to_jsonable(rec)


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(request: Request, job_id: str, name: str):
    rt.job_for(request, job_id)
    if name not in ("control.study.json", "trajectories.csv"):
        raise HTTPException(404, "no such file")
    p = rt.jobs.workdir / job_id / name
    if not p.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(p, filename=name)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    ref = ModelRef.load(EXAMPLES / "tank_rom.json.modelref.json")
    return to_jsonable({"version": VERSION, "mode": MODE, "controllers": CONTROLLERS, "casadi_available": CASADI,
                        "spec": ControlSpec().to_dict(), "scenario": ControlScenario().to_dict(),
                        "example_surrogate": {"name": ref.name, "metrics": ref.metrics},
                        "scope": scope(), "units_note": "h [m], T [degC], q_in [m3/s], Q [W], time [s]",
                        "nominal": {"h": 1.2, "T": 25.0, "q_in": float(np.round(0.006 * np.sqrt(1.2), 6))}})


@app.get("/api/example-file/{name}")
def example_file(name: str):
    if name not in ("tank_rom.json", "tank_rom.json.modelref.json"):
        raise HTTPException(404, "no such file")
    return FileResponse(EXAMPLES / name, filename=name)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
