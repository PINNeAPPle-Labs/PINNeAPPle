"""3-D renders of the tropical Pacific during an ENSO event with the library's Blender Cycles renderer
(:func:`pinneapple_tools.visualization.studio.render`).

:func:`slab_scene` builds a cut-away slab of the basin: the sea surface coloured by SST from 2S to 20N, a vertical
wall along 2S from the surface down to the thermocline coloured by the SST of that row, and the 20 deg C isotherm
(thermocline) surface over the whole width below, coloured by its depth anomaly on a diverging scale with its own
colour bar -- so the warm-water layer's thickness, the thermocline tilt and the eastward shift of warm water during
El Nino are visible together. Depths are exaggerated (``z_per_100m``).
:func:`render_frames` renders a sequence of months (optionally interpolated in time) with a date label, and
:func:`encode_video` writes an H.264 MP4 with ffmpeg.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence

import numpy as np

__all__ = ["slab_scene", "render_frames", "encode_video"]


def _grid_mesh(X, Y, Z):
    """Vertices and triangles of a structured (ny, nx) surface."""
    ny, nx = X.shape
    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    i = np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)[None, :]
    a, b, c, d = i, i + 1, i + nx, i + nx + 1
    F = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    return V, F


def slab_scene(lon: np.ndarray, lat: np.ndarray, sst: np.ndarray, d20: np.ndarray, *,
               d20_anom: np.ndarray | None = None, band_deg: float = 2.0, z_per_100m: float = 1.0,
               x_per_deg: float = 0.06, y_per_deg: float = 0.12, floor_m: float = 300.0, title: str = ""):
    """Scene of SST [deg C] (ny, nx) and thermocline depth ``d20`` [m, positive down] (ny, nx) on the same
    (lat, lon) grid (lon 0-360). The sea surface north of ``band_deg`` N and the equatorial band (``band_deg`` S
    to N) carry ``sst``; the band's southern wall drops from the surface to the thermocline and carries the SST of
    that row; ``d20_anom`` [m] (optional) is the thermocline surface's field ``dz``. NaN SST (land) is drawn at the
    field's minimum; NaN depth takes its row mean."""
    from pinneapple_tools.visualization.studio.scene import Scene, Surface
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    x = (lon - lon[0]) * x_per_deg
    y = lat * y_per_deg
    X, Y = np.meshgrid(x, y)
    s = np.where(np.isfinite(sst), sst, np.nanmin(sst))
    d = np.array(d20, float)
    rows = np.nanmean(d, axis=1, keepdims=True)
    d = np.where(np.isfinite(d), d, rows)
    zs = -d / 100.0 * z_per_100m
    north = lat >= -band_deg                                   # the band and everything north of it
    V, F = _grid_mesh(X[north], Y[north], np.zeros_like(X[north]))
    surfs = [Surface("sea_surface", V, F, "grey", {"sst": s[north].ravel()})]
    V, F = _grid_mesh(X, Y, zs)
    fields = {} if d20_anom is None else {"dz": np.nan_to_num(np.asarray(d20_anom, float)).ravel()}
    surfs.append(Surface("thermocline", V, F, "grey", fields))
    j0 = int(np.nonzero(north)[0][0])                          # southern edge of the band
    nz = 12
    frac = np.linspace(0, 1, nz)[:, None]
    Xc = np.repeat(x[None], nz, 0)
    Zc = frac * zs[j0][None]
    V, F = _grid_mesh(Xc, np.full_like(Xc, y[j0]), Zc)
    row = np.asarray(sst, float)[j0]
    ok = np.isfinite(row)
    row = np.interp(x, x[ok], row[ok]) if ok.any() else s[j0]   # land cells on the wall: neighbouring ocean
    surfs.append(Surface("equator_curtain", V, F, "grey", {"sst": np.repeat(row[None], nz, 0).ravel()}))
    # a fixed floor so that the camera framing does not change from frame to frame
    zf = -floor_m / 100.0 * z_per_100m
    xb, yb = np.array([x[0], x[-1]]), np.array([y[0], y[-1]])
    Xb, Yb = np.meshgrid(xb, yb)
    V, F = _grid_mesh(Xb, Yb, np.full_like(Xb, zf))
    surfs.append(Surface("floor", V, F, "concrete"))
    return Scene(surfs, axes="z_up", title=title, labels={"sst": "SST [deg C]"})


def render_frames(frames: Sequence[dict], out_dir: str, *, sst_range: tuple[float, float] = (22.0, 31.0),
                  size: tuple[int, int] = (960, 540), samples: int = 24, view=(-0.4, -1.0, 0.75),
                  interpolate: int = 1, dz_range: float = 40.0, **scene_kw) -> list[str]:
    """Render ``frames`` (dicts with ``lon``, ``lat``, ``sst``, ``d20``, optionally ``d20_anom``, and a ``label``)
    to PNG files in
    ``out_dir``; ``interpolate`` > 1 inserts linearly interpolated frames between consecutive months. Needs
    ``bpy`` (``pip install bpy``) or a run inside Blender."""
    from PIL import Image, ImageDraw, ImageFont

    from pinneapple_tools.visualization.studio import render
    os.makedirs(out_dir, exist_ok=True)
    seq = []
    for a, b in zip(frames[:-1], frames[1:], strict=False):
        for k in range(interpolate):
            w = k / interpolate
            item = {"lon": a["lon"], "lat": a["lat"], "sst": (1 - w) * a["sst"] + w * b["sst"],
                    "d20": (1 - w) * a["d20"] + w * b["d20"], "label": a["label"] if w < 0.5 else b["label"]}
            if a.get("d20_anom") is not None:
                item["d20_anom"] = (1 - w) * a["d20_anom"] + w * b["d20_anom"]
            seq.append(item)
    seq.append(frames[-1])
    paths = []
    for i, f in enumerate(seq):
        p = os.path.join(out_dir, f"frame_{i:04d}.png")
        sc = slab_scene(f["lon"], f["lat"], f["sst"], f["d20"], d20_anom=f.get("d20_anom"), **scene_kw)
        sf = None
        if f.get("d20_anom") is not None:
            sf = {"thermocline": ("dz", -dz_range, dz_range, "rdbu", "20 C isotherm depth anomaly [m] (red = deeper)")}
        render(sc, p, field="sst", field_range=sst_range, view=view, samples=samples, size=size,
               title="SST [deg C]", lo_txt="cooler", hi_txt="warm pool", distance=0.92, surface_fields=sf)
        im = Image.open(p).convert("RGB")
        dr = ImageDraw.Draw(im, "RGBA")
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, size[0] // 32))
            small = ImageFont.truetype("DejaVuSans.ttf", max(9, size[0] // 64))
        except OSError:
            font = small = ImageFont.load_default()
        dr.rounded_rectangle([12, 10, 12 + size[0] * 0.70, 12 + size[1] * 0.13], radius=8, fill=(255, 255, 255, 215))
        dr.text((22, 14), f["label"], fill=(15, 23, 42), font=font)
        dr.text((22, 14 + size[0] // 28), "top: SST (2S-20N); wall at 2S: SST down to the thermocline;"
                " lower surface: 20 C isotherm coloured by its depth anomaly",
                fill=(71, 85, 105), font=small)
        im.save(p)
        paths.append(p)
    return paths


def encode_video(paths: Sequence[str], out: str, fps: int = 8, crf: int = 30) -> str:
    """H.264 MP4 (yuv420p, even frame size) from PNG frames with ffmpeg."""
    d = os.path.dirname(paths[0])
    pattern = os.path.join(d, "frame_%04d.png")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern, "-vf",
           "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), out]
    subprocess.run(cmd, check=True)
    return out
