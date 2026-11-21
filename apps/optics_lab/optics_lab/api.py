"""Optics Lab: a Studio page for the lithography imaging of ``pinneapple_physics.lithography``.

The app is glue only: it takes the optical system (wavelength, NA, illumination), calls the library's Abbe imaging,
its metrics (CD, NILS, edge placement) and its process window, and serves the aerial image at several defocus planes
as a 3-D view. Every number on screen comes with its check: the clear-mask gain, the cutoff frequency against
``NA(1 + sigma) / lambda``, and the Abbe sum against the Hopkins/SOCS kernels.

Run:  uvicorn optics_lab.api:app --port 8166   (from apps/optics_lab, apps/_shared on the path)

Environment (all optional):
  OPTICS_USER / OPTICS_PASSWORD (or OPTICS_PASSWORD_HASH)   HTTP Basic login on everything except /health (appkit)
"""
from __future__ import annotations

import os
import secrets
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pinneapple_physics.lithography.design import clip_lines_and_elbow
from pinneapple_physics.lithography.imaging import (
    Optics,
    Source,
    aerial_image,
    aerial_image_socs,
    resolution_limit,
    socs_kernels,
)
from pinneapple_physics.lithography.metrics import cd_along, nils, process_window
from pinneapple_tools.visualization.studio.fieldview import colour_range
from pinneapple_tools.visualization.studio.scene import Scene, Surface

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

STATIC = os.path.join(os.path.dirname(__file__), "static")
# one torch thread and one computation at a time: the imaging is a few small FFTs, and torch inside FastAPI's
# threadpool deadlocks on the second concurrent call (seen here: the first answer arrives, the next never does)
torch.set_num_threads(1)
_LOCK = threading.Lock()
MAX_VIEWS = 8
N = 128                     # imaging grid of the clip
L = 1024.0                  # field size, nm
Z_STEP_UM = 0.35            # the planes are spaced for the view, not to scale (the focus range is 0.12 um)

app = FastAPI(title="Optics Lab", version="0.1.0",
              description="Aerial image, CD, NILS and process window of a lithographic clip through "
                          "pinneapple_physics.lithography, with the closed-form checks beside every number.")
install(app, prefix="OPTICS")

_VIEWS: OrderedDict[str, tuple[bytes, dict]] = OrderedDict()

PRESETS = {
    "ArF imersão": {"wavelength_nm": 193.0, "na": 1.35, "sigma": 0.7, "shape": "conventional"},
    "ArF seco": {"wavelength_nm": 193.0, "na": 0.93, "sigma": 0.7, "shape": "annular"},
    "KrF": {"wavelength_nm": 248.0, "na": 0.80, "sigma": 0.6, "shape": "conventional"},
    "EUV": {"wavelength_nm": 13.5, "na": 0.33, "sigma": 0.5, "shape": "conventional"},
}


class ImageBody(BaseModel):
    wavelength_nm: float = Field(193.0, gt=1.0, le=1000.0)
    na: float = Field(1.35, gt=0.01, le=1.5)
    sigma: float = Field(0.7, ge=0.0, le=1.0)
    shape: str = "conventional"
    threshold: float = Field(0.3, gt=0.0, lt=1.0)
    focus_nm: list[float] = [-60.0, -30.0, 0.0, 30.0, 60.0]
    window: bool = True


def _planes(mask: np.ndarray, images: list[np.ndarray], focus: list[float]) -> Scene:
    """The aerial image at each defocus, as a surface in its own z plane (µm in the view)."""
    sc = Scene(title="aerial image through focus", axes="z_up")
    half = L / 2.0 / 1000.0                       # µm
    n = mask.shape[0]
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    for k, (img, z) in enumerate(zip(images, focus, strict=True)):
        V = np.stack([X.ravel(), Y.ravel(), np.full(X.size, (z / 1000.0) + k * Z_STEP_UM)], axis=1)
        i = np.arange(n)
        ii, jj = np.meshgrid(i[:-1], i[:-1], indexing="ij")
        a = ii * n + jj
        F = np.stack([a.ravel(), (a + n).ravel(), (a + n + 1).ravel()], axis=1)
        F2 = np.stack([a.ravel(), (a + n + 1).ravel(), (a + 1).ravel()], axis=1)
        F = np.concatenate([F, F2]).astype(np.int64)
        sc.surfaces.append(Surface(f"focus {z:+.0f} nm", V, F, material="glass",
                                   fields={"f0": np.asarray(img, float).ravel()}))
    sc.labels = {"f0": "aerial image (a.u.)"}
    return sc


def _meta(sc: Scene, notes: list[str]) -> dict:
    vals = np.concatenate([s.fields["f0"] for s in sc.surfaces])
    rng = colour_range(vals)
    lo, hi = sc.bounds()
    return {
        "schema": "pinneapple-field-view/v1", "title": sc.title, "axes": "z_up",
        "fields": [{"key": "f0", "name": "intensity", "label": "aerial image", "unit": "", "kind": "value",
                    "group": "intensity", "where": "point", "note": "", **rng}],
        "deform": {}, "markers": [],
        "bounds": {"min": np.round(lo, 6).tolist(), "max": np.round(hi, 6).tolist(),
                   "size": float(np.max(hi - lo))},
        "faces": int(sum(len(s.faces) for s in sc.surfaces)), "faces_original": None, "decimated": False,
        "max_faces": None, "points": int(sum(len(s.vertices) for s in sc.surfaces)), "cells": 0,
        "notes": notes, "labels": sc.labels,
    }


def _source(body: ImageBody) -> Source:
    if body.shape == "coherent":
        return Source("coherent")
    if body.shape == "annular":
        return Source("annular", sigma_out=max(body.sigma, 0.2), sigma_in=0.5 * max(body.sigma, 0.2))
    return Source("conventional", sigma_out=max(body.sigma, 0.05))


@app.get("/api/meta")
def meta():
    return {"presets": PRESETS, "default": ImageBody().model_dump(), "grid": {"N": N, "L_nm": L},
            "clip": "three 64 nm lines at 192 nm pitch plus an elbow, dark tone",
            "z_step_um": Z_STEP_UM,
            "note": "the defocus planes are spaced for the view; the physical range is the focus list in nm"}


@app.post("/api/image")
def compute(body: ImageBody):
    with _LOCK:
        return _compute(body)


def _compute(body: ImageBody):
    t0 = time.time()
    src = _source(body)
    mask = (~clip_lines_and_elbow(N, L)).astype(float)      # o clipe é de tom escuro: 1 = claro
    optics0 = Optics(wavelength=body.wavelength_nm, NA=body.na)
    images = [aerial_image(mask, L, Optics(body.wavelength_nm, body.na, z), src).numpy() for z in body.focus_nm]
    ref = images[int(np.argmin(np.abs(np.array(body.focus_nm))))]

    target = mask < 0.5                   # o que se quer imprimir: as linhas escuras do clipe
    cd = cd_along(ref, L, body.threshold)
    cd_target = 64.0
    k1 = resolution_limit(optics0, body.sigma) * body.na / body.wavelength_nm
    pmin = resolution_limit(optics0, body.sigma)

    # checks: the three the library's own tests make, repeated here at the numbers on screen
    clear = aerial_image(np.ones((64, 64)), 500.0, optics0, src).numpy()
    pmin_check = []
    for f, modulated in ((0.95, False), (1.10, True)):
        m = (np.mod(np.arange(N) * (f * pmin) / N, f * pmin) < 0.5 * f * pmin).astype(float)
        img = aerial_image(np.tile(m[:, None], (1, N)), f * pmin, optics0, src).numpy()
        pmin_check.append({"pitch_nm": round(f * pmin, 1), "modulated": bool(np.ptp(img) > 1e-3),
                           "ok": bool(np.ptp(img) > 1e-3) == modulated})
    # Abbe contra Hopkins na grade 64², com todos os kernels: é a comparação exata que o teste da biblioteca faz
    m64 = torch.as_tensor(mask[::2, ::2])
    i64 = aerial_image(m64, L / 2.0, optics0, src)
    socs = socs_kernels(64, L / 2.0, optics0, src)
    abbe_vs_socs = float((i64 - aerial_image_socs(m64, socs)).abs().max())

    window = None
    if body.window:
        focus = [-60.0, -30.0, 0.0, 30.0, 60.0]
        doses = [0.9, 0.95, 1.0, 1.05, 1.1]

        def image_at(z):
            # process_window hands this to printed(), which expects the torch tensor aerial_image returns
            return aerial_image(mask, L, Optics(body.wavelength_nm, body.na, z), src)

        window = process_window(image_at, target, L, focus=focus, doses=doses, threshold=body.threshold,
                                diffusion=0.0, tone="dark", epe_tol=10.0)

    sc = _planes(mask, images, body.focus_nm)
    token = secrets.token_urlsafe(16)
    _VIEWS[token] = (sc.to_glb(), _meta(sc, [
        f"{body.focus_nm[0]:+.0f} a {body.focus_nm[-1]:+.0f} nm de defoco, planos espaçados para a vista",
        f"grade {N}x{N} sobre {L:.0f} nm · limiar {body.threshold:.2f}"]))
    while len(_VIEWS) > MAX_VIEWS:
        _VIEWS.popitem(last=False)

    return {
        "input": body.model_dump(),
        "source_points": int(len(src.points()[0])),
        "metrics": {
            "cd_nm": round(cd, 2), "cd_mask_nm": cd_target,
            "cd_bias_nm": round(cd - cd_target, 2),
            "nils": round(nils(ref, L, body.threshold), 3),
            "resolution_limit_nm": round(pmin, 2),
            "k1": round(k1, 4),
            "k1_clip": round(0.5 * 192.0 * body.na / body.wavelength_nm, 4),
            "dof_nm": None if window is None else round(window["dof_nm"], 1),
            "exposure_latitude": None if window is None else round(window["exposure_latitude"], 3),
            "best_focus_nm": None if window is None else round(window["best_focus"], 1),
            "epe_best_nm": None if window is None else round(float(np.min(window["epe"])), 2),
        },
        "checks": {
            "clear_mask_gain": {"value": float(np.abs(clear - 1).max()), "ok": bool(np.abs(clear - 1).max() < 1e-9),
                                "text": "máscara toda clara tem de imagear em 1,0"},
            "modulation_around_the_limit": pmin_check,
            "abbe_vs_socs": {"value": abbe_vs_socs, "ok": abbe_vs_socs < 1e-6,
                             "text": "soma de Abbe contra os kernels de Hopkins (SOCS)"},
            "energy": {"value": float(np.abs(ref.mean() - mask.mean()).max()),
                       "ok": bool(abs(ref.mean() - mask.mean()) < 0.25),
                       "text": "média da imagem perto da transmissão média da máscara"},
        },
        "window": window,
        "view": token,
        "seconds": round(time.time() - t0, 2),
    }


@app.get("/api/view/{token}.glb")
def view_glb(token: str):
    if token not in _VIEWS:
        raise HTTPException(404, "view expired")
    return Response(_VIEWS[token][0], media_type="model/gltf-binary")


@app.get("/api/view/{token}")
def view_meta(token: str):
    if token not in _VIEWS:
        raise HTTPException(404, "view expired")
    return _VIEWS[token][1]


@app.get("/health")
def health():
    return {"ok": True, "views": len(_VIEWS)}


@app.get("/api/status")
def status():
    return {"views": len(_VIEWS), "grid": {"N": N, "L_nm": L}, "presets": list(PRESETS)}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
