"""Predictive Twin: a river digital twin on a live USGS gauge. Baseflow recession calibrated on the 72 h before the
last 24 h, forecast through the last 24 h and checked against what the river actually did.

Glue only: the twin is ``pinneapple_systems.digital_twin.recession.run_twin`` and the data client is
``pinneapple_systems.digital_twin.io.usgs`` (public-domain USGS data, downloaded at run time, never stored).

Run:  uvicorn predictive_twin.api:app --port 8184   (from apps/predictive_twin)

Environment (all optional): PTWIN_USER / PTWIN_PASSWORD (or PTWIN_PASSWORD_HASH) for an HTTP Basic login on
everything except /health; PTWIN_SITE_ID the USGS gauge (default 01646500, Potomac River near Washington, DC);
PTWIN_OFFLINE=1 serves a labelled synthetic record instead of USGS (demos without network); the rest in
apps/_shared/README.md.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pinneapple_systems.digital_twin import recession
from pinneapple_systems.digital_twin.io import usgs

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

VERSION = "1.0.0"
STATIC = Path(__file__).resolve().parent / "static"
SITE_ID = os.environ.get("PTWIN_SITE_ID", "01646500")
PERIOD = "P5D"

app = FastAPI(title="Predictive Twin", version=VERSION,
              description="Baseflow-recession digital twin of a live USGS river gauge: calibrate, forecast, flag a "
                          "sustained 3-sigma departure. GET /api/scenario fetches and fits.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="PTWIN")


def _offline() -> bool:
    return (os.environ.get("PTWIN_OFFLINE") or "").lower() in ("1", "true", "yes")


def _series():
    if _offline():
        q = recession.synthetic_recession(q0=2000.0, k_per_hour=0.0035, bump_at_hour=86.0, bump=0.06, seed=1)
        h = recession.synthetic_recession(q0=4.5, k_per_hour=0.0015, noise=0.002, seed=2)
        now = datetime.now(timezone.utc)
        return ({"site_id": "synthetic", "site_name": "SYNTHETIC record (PTWIN_OFFLINE=1), not a real gauge",
                 "unit": "ft3/s", "fetched_at": now, "from_cache": False, "usgs_url": ""},
                q["timestamps"], q["values"], h["timestamps"], h["values"])
    s = usgs.fetch_iv(SITE_ID, usgs.DISCHARGE, PERIOD)
    try:
        st = usgs.fetch_iv(SITE_ID, usgs.GAGE_HEIGHT, PERIOD)
        st_t, st_v = st.timestamps, st.values
    except usgs.USGSDataError:
        st_t, st_v = None, None              # not every gauge reports gage height: no water-level view then
    return ({"site_id": s.site_id, "site_name": s.site_name, "unit": s.unit, "fetched_at": s.fetched_at,
             "from_cache": s.from_cache, "usgs_url": usgs.monitoring_url(s.site_id)}, s.timestamps, s.values,
            st_t, st_v)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/scenario")
def scenario():
    """Fetch the trailing 5 days (cached 90 s) and run the twin; 502 with the reason when USGS fails."""
    try:
        meta, t, q, st_t, st_v = _series()
        r = recession.run_twin(t, q, stage_timestamps=st_t, stage_values=st_v)
    except usgs.USGSDataError as exc:
        raise HTTPException(502, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(502, f"USGS site {SITE_ID}: {exc}") from exc
    cal = r["calibration"]
    return {
        "synthetic": _offline(),
        "water_level": r["water_level"],
        "gauge": {"site_id": meta["site_id"], "site_name": meta["site_name"], "usgs_url": meta["usgs_url"],
                  "parameter": f"{usgs.DISCHARGE} - discharge, cubic feet per second", "unit": meta["unit"],
                  "latest_reading_time": r["latest"]["time"], "latest_discharge_cfs": r["latest"]["value"],
                  "fetched_at": meta["fetched_at"].isoformat(), "server_time": r["server_time"],
                  "from_cache": meta["from_cache"]},
        "timeseries": {"hours": r["timeseries"]["hours"], "timestamps": r["timeseries"]["timestamps"],
                       "discharge_cfs": r["timeseries"]["values"]},
        "calibration": {**cal, "q0_cfs": cal["q0"], "residual_mean_cfs": cal["residual_mean"],
                        "residual_std_cfs": cal["residual_std"]},
        "forecast": {"hours": r["forecast"]["hours"], "discharge_cfs": r["forecast"]["values"]},
        "comparison": {"start_hour": r["comparison"]["start_hour"], "start_time": r["comparison"]["start_time"],
                       "hours": r["comparison"]["hours"], "actual_cfs": r["comparison"]["actual"],
                       "predicted_cfs": r["comparison"]["predicted"], "residuals_cfs": r["comparison"]["residuals"],
                       "n_sigma_series": r["comparison"]["n_sigma_series"]},
        "detection": {**r["detection"], "upper_threshold_cfs": r["detection"]["upper_threshold"],
                      "lower_threshold_cfs": r["detection"]["lower_threshold"]},
        "verdict": r["verdict"],
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
