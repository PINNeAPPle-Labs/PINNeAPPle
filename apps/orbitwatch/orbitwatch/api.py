"""OrbitWatch: change between dated satellite images of one place (per-band reflectance change and, with a real
near-infrared band, NDVI change), as a self-contained HTML report.

* ``POST /v1/detect_changes_upload``: two or more images you supply (PNG / JPEG, RGB; no NIR, so no NDVI).
* ``POST /v1/detect_changes``: Sentinel-2 surface reflectance fetched from Google Earth Engine around a point
  (B4, B3, B2 and the real NIR band B8). Needs Earth Engine credentials in the environment (below); without
  them it answers 503 with the setup steps and never invents imagery.
* ``GET /v1/threshold_reference``: the default thresholds and scales.

The maths and the report are ``pinneapple_perception.change_detection``; the fetch is
``pinneapple_perception.sentinel2``.

Run:  uvicorn orbitwatch.api:app --port 8150   (from apps/orbitwatch)

Environment (all optional; the rest in apps/_shared/README.md):
  ORBITWATCH_USER / ORBITWATCH_PASSWORD(_HASH)   login on everything except /health
  ORBITWATCH_FREE_REPORTS                        free reports per caller and month (default 100)
  EE_PROJECT_ID                                  Google Cloud project registered for Earth Engine
  EE_SERVICE_ACCOUNT_EMAIL, EE_SERVICE_ACCOUNT_KEY_FILE   a service account and the path of its JSON key (kept
                                                 outside the repository and the image, e.g. a mounted secret)
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pinneapple_perception import change_detection as det

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"

app = FastAPI(title="OrbitWatch", version=VERSION,
              description="Satellite image change detection (band reflectance and NDVI) with an HTML report.")
install(app, prefix="ORBITWATCH")
rt = install_runtime(app, prefix="ORBITWATCH", name="orbitwatch", units={"reports": 100})


class DetectRequest(BaseModel):
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-90, le=90)
    buffer_m: float = Field(5000.0, gt=0, le=50000)
    start_date: str = "2024-01-01"
    end_date: str = "2025-07-27"
    max_cloud_pct: float = Field(10.0, gt=0, le=100)
    n_images: int = Field(5, ge=2, le=20)
    scale: int = Field(10, ge=10, le=100)
    band_diff_threshold: float = Field(det.DEFAULT_BAND_DIFF_THRESHOLD, gt=0)
    ndvi_threshold: float = Field(det.DEFAULT_NDVI_THRESHOLD, gt=0)
    reflectance_scale: float = Field(det.DEFAULT_REFLECTANCE_SCALE, gt=0)


def _report(arrays, dates, nirs, *, band, ndvi, scale, lon, lat, buffer_m) -> str:
    pairs, pngs = [], []
    for i in range(len(arrays) - 1):
        a, b = arrays[i], arrays[i + 1]
        na, nb = (nirs[i], nirs[i + 1]) if nirs else (None, None)
        pairs.append(det.analyze_pair(a, b, dates[i], dates[i + 1], nir_a=na, nir_b=nb, band_diff_threshold=band,
                                      ndvi_threshold=ndvi, reflectance_scale=scale))
        va = det.compute_ndvi(a[..., 0], na) if na is not None else None
        vb = det.compute_ndvi(b[..., 0], nb) if nb is not None else None
        pngs.append(det.render_comparison_figure(a, b, dates[i], dates[i + 1], ndvi_a=va, ndvi_b=vb))
    return det.render_html_report(pairs, pngs, lon=lon, lat=lat, buffer_m=buffer_m, has_ndvi=bool(nirs))


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/v1/threshold_reference")
def threshold_reference():
    return {"default_band_diff_threshold_reflectance": det.DEFAULT_BAND_DIFF_THRESHOLD,
            "default_ndvi_threshold": det.DEFAULT_NDVI_THRESHOLD,
            "default_reflectance_scale": det.DEFAULT_REFLECTANCE_SCALE,
            "preview_stretch": det.LEGACY_VISUALIZATION_NORMALIZER,
            "note": "Sentinel-2 SR stores DN = 10000 x reflectance: the threshold uses 10000. 3000 is only the "
                    "contrast stretch of the RGB previews. Uploaded 8-bit images default to a 255 scale."}


@app.post("/v1/detect_changes", response_class=HTMLResponse)
def detect_changes(request: Request, req: DetectRequest):
    from pinneapple_perception.sentinel2 import EarthEngineUnavailable, fetch_sentinel2_sequence

    rt.consume(request, "reports")
    try:
        frames = fetch_sentinel2_sequence(lon=req.lon, lat=req.lat, buffer_m=req.buffer_m, start_date=req.start_date,
                                          end_date=req.end_date, max_cloud_pct=req.max_cloud_pct,
                                          n_images=req.n_images, scale=req.scale)
    except ImportError as e:
        raise HTTPException(503, f"earthengine-api / geemap are not installed on this server: {e}") from None
    except EarthEngineUnavailable as e:
        raise HTTPException(503, str(e)) from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return HTMLResponse(_report([f.rgb for f in frames], [f.date for f in frames], [f.nir for f in frames],
                                band=req.band_diff_threshold, ndvi=req.ndvi_threshold, scale=req.reflectance_scale,
                                lon=req.lon, lat=req.lat, buffer_m=req.buffer_m))


@app.post("/v1/detect_changes_upload", response_class=HTMLResponse)
async def detect_changes_upload(request: Request, images: list[UploadFile] = File(...), dates: list[str] = Form(...),  # noqa: B008
                                band_diff_threshold: float = Form(det.DEFAULT_BAND_DIFF_THRESHOLD, gt=0),
                                ndvi_threshold: float = Form(det.DEFAULT_NDVI_THRESHOLD, gt=0),
                                reflectance_scale: float = Form(255.0, gt=0)):
    from PIL import Image

    if len(images) != len(dates):
        raise HTTPException(422, f"{len(images)} images and {len(dates)} dates: give one date per image")
    if len(images) < 2:
        raise HTTPException(422, "need at least 2 images")
    rt.consume(request, "reports")
    arrays = []
    for f in images:
        try:
            arr = np.asarray(Image.open(io.BytesIO(await f.read())).convert("RGB"), dtype=np.float64)
        except Exception as e:  # noqa: BLE001 -- any decoder error
            raise HTTPException(422, f"{f.filename!r} is not a readable image: {e}") from None
        if arrays and arr.shape != arrays[0].shape:
            raise HTTPException(422, f"{f.filename!r} is {arr.shape}, the first image is {arrays[0].shape}")
        arrays.append(arr)
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    return HTMLResponse(_report([arrays[i] for i in order], [dates[i] for i in order], None,
                                band=band_diff_threshold, ndvi=ndvi_threshold, scale=reflectance_scale,
                                lon=0.0, lat=0.0, buffer_m=0.0))
