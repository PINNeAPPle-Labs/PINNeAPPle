"""FlowTrace: upload a video or an image sequence of a seeded flow, get a cross-correlation PIV velocity field back.

The PIV is ``pinneapple_perception.video_piv`` (FFT-normalised cross-correlation, three-point parabolic sub-pixel
peak); frame decoding, the grid size, the summary and the parameter advice are ``pinneapple_perception.piv_tools``.
The only vectors dropped are windows whose search region would cross the image border; there is no
correlation-quality (signal-to-noise) rejection, and motion blur degrades the result.

Run:  uvicorn flowtrace.api:app --port 8145   (from apps/flowtrace; video uploads need ffmpeg on PATH)

Environment (all optional; the rest in apps/_shared/README.md):
  FLOWTRACE_USER / FLOWTRACE_PASSWORD(_HASH)   login on everything except /health
  FLOWTRACE_FREE_ANALYSES                      free analyses per caller and month (default 100)
  FLOWTRACE_UPLOAD_MAX_MB                      largest upload (default 200)
  FLOWTRACE_MAX_FRAMES                         most frames analysed per request (default 200)
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from pinneapple_perception import piv_tools as pt
from pinneapple_perception.video_piv import piv_velocity_sequence

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
MAX_MB = float(os.environ.get("FLOWTRACE_UPLOAD_MAX_MB") or 200)
MAX_FRAMES = int(os.environ.get("FLOWTRACE_MAX_FRAMES") or 200)

app = FastAPI(title="FlowTrace", version=VERSION, description="Camera-agnostic cross-correlation PIV.")
install(app, prefix="FLOWTRACE", max_body_mb=MAX_MB + 10)
rt = install_runtime(app, prefix="FLOWTRACE", name="flowtrace", units={"analyses": 100}, upload_max_mb=MAX_MB)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/v1/parameters_guide")
def parameters_guide(approx_displacement_px: float = Query(..., gt=0), image_width_px: int = Query(..., gt=0),
                     image_height_px: int = Query(..., gt=0)):
    return pt.piv_parameter_guide(approx_displacement_px, image_width_px, image_height_px)


@app.post("/v1/analyze_sequence")
async def analyze_sequence(request: Request, video: UploadFile | None = File(None),  # noqa: B008
                           frames: list[UploadFile] | None = File(None), window_size: int = Form(32, ge=8, le=512),  # noqa: B008
                           search_margin: int = Form(16, ge=1, le=256), step: int | None = Form(None, ge=1),
                           dt: float = Form(1.0, gt=0), units_per_pixel: float = Form(1.0, gt=0),
                           output: Literal["npz", "json"] = Form("npz")):
    if (video is None) == (not frames):
        raise HTTPException(422, "upload either 'video' (one file) or 'frames' (image files), not both")
    rt.consume(request, "analyses")
    try:
        if video is not None:
            with tempfile.NamedTemporaryFile(suffix=Path(video.filename or "v.mp4").suffix or ".mp4") as tmp:
                tmp.write(await video.read())
                tmp.flush()
                arr = pt.decode_video(tmp.name)
        else:
            arr = pt.decode_image_frames([(f.filename or f"{i:06d}", await f.read()) for i, f in enumerate(frames)])
    except pt.FFmpegMissing as e:
        raise HTTPException(503, str(e)) from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    arr = arr[:MAX_FRAMES]
    try:
        seq = piv_velocity_sequence(arr, dt=dt, window_size=window_size, search_margin=search_margin, step=step,
                                    units_per_pixel=units_per_pixel)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if seq["u"].size == 0:
        raise HTTPException(422, f"no interrogation window fits a {arr.shape[1]}x{arr.shape[2]} frame with "
                                 f"window_size={window_size}, search_margin={search_margin}: see /v1/parameters_guide")
    summary = pt.summarize_piv(seq, arr.shape, window_size, search_margin, step)
    summary.update({"source_frames": int(arr.shape[0]), "resolution": {"height": int(arr.shape[1]),
                                                                       "width": int(arr.shape[2])},
                    "parameters": {"window_size": window_size, "search_margin": search_margin,
                                   "step": step or window_size, "dt": dt, "units_per_pixel": units_per_pixel}})
    buf = io.BytesIO()
    np.savez(buf, **{k: seq[k] for k in ("x", "y", "t", "u", "v")})
    if output == "json":
        return {"summary": summary, "npz_base64": base64.b64encode(buf.getvalue()).decode("ascii")}
    return Response(buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": 'attachment; filename="velocity_field.npz"',
                             "X-Summary": json.dumps(summary)})
