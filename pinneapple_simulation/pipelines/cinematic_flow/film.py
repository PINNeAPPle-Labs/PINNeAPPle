"""The orchestrator: solve once, render, assemble, and keep the record next to the film.

Every film leaves a folder with the frames, the ``.mp4``, a still, and ``ficha.json`` — the resolution, cells,
Reynolds number and step count the picture came from. A film without that record is a picture of nothing.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np

from .cases import Case, build
from .solve import plan_tunnel, run_tunnel

__all__ = ["make_film", "render_still"]

# per-case camera: azimuth in degrees around the body, starting and ending, elevation, distance in radii
CAMERA = {
    "carro": dict(az0=-58.0, az1=-22.0, elev0=11.0, elev1=17.0, dist0=3.15, dist1=2.75, lens=45.0, fstop=2.4),
    "asa": dict(az0=-42.0, az1=18.0, elev0=15.0, elev1=11.0, dist0=3.1, dist1=2.7, lens=48.0, fstop=2.4),
    "foguete": dict(az0=-50.0, az1=-12.0, elev0=10.0, elev1=15.0, dist0=3.2, dist1=2.8, lens=46.0, fstop=2.4),
}

# "that look": a deep navy body, a blue-white smoke, and a dark studio with two strip lights
STYLE = {
    "paint": [0.014, 0.030, 0.080],
    "tint": [0.45, 0.70, 1.0],
    "emission": 0.9,
    "noise_scale": 6.5,
    "fiber": 0.5,
    # what reads as smoke: the vorticity cut, and how much of the speed deficit is added to it
    "smoke": {"percentile": 99.5, "floor": 0.20, "power": 1.15, "vort_weight": 0.95,
              "deficit_weight": 0.45, "gain": 0.9},
}


def _write_bodies(case: Case, folder: str) -> list:
    import trimesh

    os.makedirs(folder, exist_ok=True)
    parts, kinds = case.render_parts
    out = []
    for name, (V, F) in parts.items():
        mesh = trimesh.Trimesh(vertices=np.asarray(V, float), faces=np.asarray(F, int), process=True)
        mesh.fix_normals()
        path = os.path.join(folder, f"{name}.ply")
        mesh.export(path)
        out.append({"name": name, "path": path, "material": kinds.get(name, "paint")})
    return out


def _frame(case: Case) -> dict:
    lo, hi = case.bounds
    center = 0.5 * (lo + hi)
    radius = float(np.linalg.norm(hi - lo)) * 0.5
    return {"center": [float(v) for v in center], "radius": radius}


def _blender_binary() -> str:
    for name in ("blender", "/Applications/Blender.app/Contents/MacOS/Blender"):
        if shutil.which(name) or os.path.exists(name):
            return shutil.which(name) or name
    raise FileNotFoundError("Blender não encontrado: instale-o para renderizar os filmes")


def _run_blender(cfg: dict, timeout: int = 6 * 3600) -> None:
    cfg_path = os.path.join(cfg["out_dir"], "cena.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    script = os.path.join(os.path.dirname(__file__), "scene.py")
    r = subprocess.run([_blender_binary(), "--background", "--python", script, "--", cfg_path],
                       capture_output=True, text=True, timeout=timeout)
    log = os.path.join(cfg["out_dir"], "blender.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write(r.stdout[-20000:] + "\n--- stderr ---\n" + r.stderr[-8000:])
    if "FRAMES_DONE" not in r.stdout and "STILL" not in r.stdout:
        raise RuntimeError(f"o Blender não terminou o render (log em {log})")


def _assemble(frames_dir: str, out_mp4: str, fps: int) -> str | None:
    ff = shutil.which("ffmpeg")
    if ff is None:
        return None
    subprocess.run([ff, "-y", "-framerate", str(fps), "-i", os.path.join(frames_dir, "frame_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-movflags", "+faststart",
                    out_mp4], capture_output=True, text=True, check=False)
    return out_mp4 if os.path.exists(out_mp4) else None


def make_film(case: str | Case = "carro", *, out_dir: str, cells_per_length: int = 140, steps: int = 2400,
              snapshots: int = 10, stride: int = 2, device: str = "mps", frames: int = 144, fps: int = 24,
              res_x: int = 1920, res_y: int = 1080, samples: int = 64, camera: dict | None = None,
              style: dict | None = None, force_solve: bool = False) -> dict:
    """Solve ``case`` if it is not solved yet, render it, and write the film plus its record."""
    case = build(case) if isinstance(case, str) else case
    os.makedirs(out_dir, exist_ok=True)
    tunnel = plan_tunnel(case, cells_per_length=cells_per_length, steps=steps)
    meta_path = os.path.join(out_dir, "meta.json")
    if force_solve or not os.path.exists(meta_path):
        print(f"[{case.name}] malha {tunnel.shape} ({tunnel.cells / 1e6:.1f} M células), {steps} passos", flush=True)
        meta = run_tunnel(case, tunnel, device=device, snapshots=snapshots, stride=stride, out_dir=out_dir)
    else:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[{case.name}] solve reaproveitado de {meta_path}", flush=True)

    bodies = _write_bodies(case, os.path.join(out_dir, "corpo"))
    cfg = {
        "out_dir": out_dir,
        "bodies": bodies,
        "snapshots": meta["snapshots"],
        "voxel_size": tunnel.dx * meta["stride"],
        "origin": list(tunnel.origin),
        "ground": case.ground,
        "frame": _frame(case),
        "camera": {**CAMERA.get(case.name, CAMERA["carro"]), **(camera or {})},
        "style": {**STYLE, **(style or {})},
        "render": {"res_x": res_x, "res_y": res_y, "samples": samples, "frames": frames, "fps": fps},
    }
    _run_blender(cfg)
    mp4 = _assemble(os.path.join(out_dir, "frames"), os.path.join(out_dir, f"{case.name}.mp4"), fps)
    ficha = {
        **meta["report"],
        "render": {"res": [res_x, res_y], "frames": frames, "fps": fps, "samples": samples,
                   "engine": "EEVEE", "segundos": frames / fps},
        "video": mp4,
        "fontes": {"campo": "WindTunnelLBM (D3Q19, Smagorinsky LES)",
                   "volume": "densidade = vorticidade + déficit de velocidade",
                   "motor": "Blender EEVEE + OpenVDB"},
    }
    with open(os.path.join(out_dir, "ficha.json"), "w", encoding="utf-8") as f:
        json.dump(ficha, f, indent=2, ensure_ascii=False)
    print(f"[{case.name}] vídeo: {mp4}", flush=True)
    return ficha


def render_still(case: str | Case = "carro", *, out_dir: str, frame: int = 1, **kw) -> str:
    """One frame only — the way to look at a look before spending the night on it."""
    kw.setdefault("frames", max(frame, 2))
    case = build(case) if isinstance(case, str) else case
    os.makedirs(out_dir, exist_ok=True)
    tunnel = plan_tunnel(case, cells_per_length=kw.get("cells_per_length", 140), steps=kw.get("steps", 2400))
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    bodies = _write_bodies(case, os.path.join(out_dir, "corpo"))
    cfg = {
        "out_dir": out_dir, "bodies": bodies, "snapshots": meta["snapshots"],
        "voxel_size": tunnel.dx * meta["stride"], "origin": list(tunnel.origin), "ground": case.ground,
        "frame": _frame(case),
        "camera": {**CAMERA.get(case.name, CAMERA["carro"]), **(kw.get("camera") or {})},
        "style": {**STYLE, **(kw.get("style") or {})},
        "render": {"res_x": kw.get("res_x", 1920), "res_y": kw.get("res_y", 1080),
                   "samples": kw.get("samples", 48), "frames": kw["frames"], "fps": 24, "only_frame": frame},
    }
    _run_blender(cfg)
    return os.path.join(out_dir, "frames", f"frame_{frame - 1:04d}.png")
