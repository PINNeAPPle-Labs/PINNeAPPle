"""Ahmed Twin: the surrogate digital twin of the Ahmed body, with a button that runs a verifying CFD.

The page (sliders for slant, clearance and speed; the predicted Cp on the body in 3-D; CD and CL with their
calibrated 90 % band; the out-of-trust-region warning) is written by
``pinneapple_autolab.domains.ahmed_study.twin`` from the study in ``examples/end_to_end/ahmed_body`` at start-up.
``POST /api/verify`` queues an OpenFOAM run of the slider point (``AhmedAero`` through ``run_campaign``, one at a
time, about 5 minutes on 4 cores); ``GET /api/verify/{id}`` reports it against the prediction.

Run:  uvicorn ahmed_twin.api:app --port 8171   (from apps/ahmed_twin; OpenFOAM needed only for /api/verify)

Environment (all optional):
  AHMED_STUDY   the study folder (default: examples/end_to_end/ahmed_body)
  AHMED_PROCS   OpenFOAM processes per verifying run (default 4)
  AHMED_USER / AHMED_PASSWORD (or AHMED_PASSWORD_HASH)   HTTP Basic login on everything except /health (appkit)
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pinneapple_autolab.domains import ahmed as A
from pinneapple_autolab.domains import ahmed_study as S
from pinneapple_autolab.study import campaign as C

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STUDY = os.environ.get("AHMED_STUDY") or os.path.join(ROOT, "examples", "end_to_end", "ahmed_body")
PAGE = os.path.join(STUDY, "_runs", "twin_app")
VERIFY = os.path.join(STUDY, "_runs", "verify")
PROCS = int(os.environ.get("AHMED_PROCS") or 4)

app = FastAPI(title="Ahmed Twin", version="0.1.0",
              description="Surrogate digital twin of the Ahmed body (OpenFOAM campaign) with verifying CFD runs.")
install(app, prefix="AHMED")
S.twin(S.load_records(os.path.join(STUDY, "_runs", "cfd"), ok_only=True), os.path.join(STUDY, "results"), PAGE,
       api="api")
_jobs: dict[str, dict] = {}
_queue: queue.Queue = queue.Queue()


def _worker() -> None:
    while True:
        jid = _queue.get()
        job = _jobs[jid]
        job["status"] = "running"
        try:
            rec = C.run_campaign(A.AhmedAero(procs=PROCS), [{"id": jid, **job["params"]}], VERIFY,
                                 purpose="verification", reason="twin: verify the slider point",
                                 predicted={jid: job["predicted"]}, log=lambda s: None)[0]
            job["record"] = rec.to_dict()
            job["status"] = "done" if rec.ok else "failed"
            if rec.ok:
                lines = [f"{k}: CFD {rec.outputs[k]:.4f}, predicted {m:.4f} +- {h:.4f} "
                         f"({'inside' if abs(rec.outputs[k] - m) <= h else 'OUTSIDE'} the 90 % band)"
                         for k, (m, h) in job["predicted"].items() if k in rec.outputs]
                trust = A.run_trust_report(rec.meta).decisao
                job["text"] = "\n".join(lines + [f"CFD checks: {trust}; {rec.cost_s / 60:.1f} min"])
            else:
                job["text"] = f"CFD failed: {rec.message}"
        except Exception as e:  # noqa: BLE001 - reported to the page
            job["status"], job["text"] = "failed", f"{type(e).__name__}: {e}"
        _queue.task_done()


threading.Thread(target=_worker, name="ahmed-verify", daemon=True).start()


class VerifyBody(BaseModel):
    params: dict[str, float]
    predicted: dict[str, list[float]] = {}


@app.post("/api/verify")
def verify(body: VerifyBody):
    issues = A.AhmedAero().check(body.params)
    if issues:
        raise HTTPException(422, "; ".join(issues))
    jid = "twin_" + time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    _jobs[jid] = {"status": "queued", "params": body.params, "predicted": body.predicted,
                  "text": f"queued ({_queue.qsize()} ahead)"}
    _queue.put(jid)
    return {"id": jid}


@app.get("/api/verify/{jid}")
def verify_status(jid: str):
    if jid not in _jobs:
        raise HTTPException(404, "unknown job")
    j = _jobs[jid]
    return {"status": j["status"], "text": j.get("text") or j["status"]}


@app.get("/")
def index():
    return RedirectResponse("twin/index.html")


app.mount("/twin", StaticFiles(directory=PAGE), name="twin")
