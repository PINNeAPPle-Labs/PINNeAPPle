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
    "carro": dict(az0=-60.0, az1=-34.0, elev0=9.0, elev1=13.0, dist0=2.95, dist1=2.75, lens=40.0, fstop=5.0),
    # the wing is a chord along x and a span along y. A camera on the -y side looks down the span and sees a tip;
    # one on the +x side looks along the chord. Slightly downstream of broadside is the shot that keeps the span
    # across the frame and brings the wake toward the lens.
    "asa": dict(az0=-56.0, az1=-38.0, elev0=30.0, elev1=24.0, dist0=3.60, dist1=3.30, lens=48.0, fstop=4.5),
    "foguete": dict(az0=-58.0, az1=-26.0, elev0=8.0, elev1=13.0, dist0=2.90, dist1=2.60, lens=48.0, fstop=5.0),
}

# "that look": a deep navy body, a blue-white smoke, and a dark studio with two strip lights
STYLE = {
    "paint": [0.014, 0.030, 0.080],
    "tint": [0.28, 0.55, 0.95],
    "hot": [0.92, 0.97, 1.0],
    "emission": 2.0,
    "noise_scale": 26.0,
    "fiber": 0.6,
    "fiber_floor": 0.0,
    # what reads as smoke: the Q-criterion cut, and how much of the speed deficit is added to it. The emission
    # is what makes it glow, so the field is kept sparse and bright instead of wide and dense — a dense field
    # integrates along the ray into a white slab that hides the body.
    "smoke": {"percentile": 99.0, "floor": 0.45, "power": 1.3, "vort_weight": 1.0,
              "deficit_weight": 0.0, "gain": 1.0, "resample": 2, "taper": 0.14},
    "light": {"strip": 45.0, "area": 420.0},
    "paint_rough": 0.30,
    "coat_rough": 0.10,
    "motion_blur": False,
    # measured on this machine at 1920x1080: these give the same picture as tile 2 / 256 samples for a third of
    # the time (23 s a frame against 57 s), because the field is smooth and the volume is what costs the frame
    "volumetric_samples": 160,
    "volumetric_tile": "4",
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


CAPTION_FONTS = ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _caption_png(text: str, sub: str, path: str, size: tuple[int, int]) -> str | None:
    """The short caption, drawn once as a transparent overlay.

    It is drawn here rather than burned by the renderer so that the text never goes through the camera's focus
    and its own motion blur — a caption that blurs with the subject reads as a mistake. ``ffmpeg``'s drawtext is
    not in every build, so the picture is an image and the assembly is an overlay.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    font_path = next((f for f in CAPTION_FONTS if os.path.exists(f)), None)
    if font_path is None:
        return None
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    big = ImageFont.truetype(font_path, max(14, int(h * 0.032)))
    small = ImageFont.truetype(font_path, max(11, int(h * 0.0155)))
    x, y = int(w * 0.065), int(h * 0.795)
    for dx, dy, title_col, sub_col in ((2, 3, (0, 0, 0, 150), (0, 0, 0, 140)),
                                       (0, 0, (255, 255, 255, 232), (176, 206, 240, 210))):
        draw.text((x + dx, y + dy), text, font=big, fill=title_col)
        draw.text((x + dx, y + int(h * 0.052) + dy), sub, font=small, fill=sub_col)
    img.save(path)
    return path


def _sci(v: float, digits: int = 1) -> str:
    """``9,2×10⁶`` — the caption is read, not parsed, so it gets a number a reader formats."""
    if v <= 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10.0 ** e
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m:.{digits}f}×10{str(e).translate(sup)}".replace(".", ",")


def _assemble(frames_dir: str, out_mp4: str, fps: int, caption: str | None = None) -> str | None:
    ff = shutil.which("ffmpeg")
    if ff is None:
        return None
    base = [ff, "-y", "-framerate", str(fps), "-start_number", "1",
            "-i", os.path.join(frames_dir, "frame_%04d.png")]
    tail = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-movflags", "+faststart", out_mp4]
    if caption and os.path.exists(caption):
        # the overlay is a single image, so it has to be looped: fed once, it exists only at t=0, the fade holds
        # it transparent there, and the caption never appears — the film renders and silently loses its text.
        # Looped, it never ends on its own, so the overlay is told to end the output when the frames do.
        cmd = base + ["-loop", "1", "-i", caption, "-filter_complex",
                      "[1:v]format=rgba,fade=t=in:st=0.5:d=0.9:alpha=1[c];"
                      "[0:v][c]overlay=0:0:format=auto:eof_action=endall,format=yuv420p"] + tail
    else:
        cmd = base + tail
    subprocess.run(cmd, capture_output=True, text=True, check=False)
    return out_mp4 if os.path.exists(out_mp4) else None


def make_film(case: str | Case = "carro", *, out_dir: str, cells_per_length: int = 140, steps: int = 2400,
              snapshots: int = 10, stride: int = 1, device: str = "mps", frames: int = 144, fps: int = 24,
              res_x: int = 1920, res_y: int = 1080, samples: int = 32, camera: dict | None = None,
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
    st = {**STYLE, **(style or {})}
    cfg = {
        "out_dir": out_dir,
        "bodies": bodies,
        "snapshots": meta["snapshots"],
        "voxel_size": float(meta["tunnel"]["dx"]) * meta["stride"],
        "origin": list(meta["tunnel"]["origin"]),
        "shape": list(meta["shape_sub"]),
        "ground": case.ground,
        "frame": _frame(case),
        "camera": {**CAMERA.get(case.name, CAMERA["carro"]), **(camera or {})},
        "style": st,
        "render": {"res_x": res_x, "res_y": res_y, "samples": samples, "frames": frames, "fps": fps,
                   "volumetric_samples": st["volumetric_samples"], "volumetric_tile": st["volumetric_tile"],
                   "motion_blur": st.get("motion_blur", True)},
    }
    _run_blender(cfg)
    rep = meta["report"]
    nx, ny, nz = rep["malha"]
    sub = (f"{nx}×{ny}×{nz} células · Re {_sci(rep['re_fisico'])} · Q-criterion · WindTunnelLBM")
    cap = _caption_png(case.caption, sub, os.path.join(out_dir, "legenda.png"), (res_x, res_y))
    mp4 = _assemble(os.path.join(out_dir, "frames"), os.path.join(out_dir, f"{case.name}.mp4"), fps, cap)
    ficha = {
        **meta["report"],
        "render": {"res": [res_x, res_y], "frames": frames, "fps": fps, "samples": samples,
                   "engine": "EEVEE", "segundos": frames / fps},
        # the film's name relative to the folder that holds it, not an absolute path: the record travels with the
        # film into a repository, and the machine it was rendered on is nobody's business
        "video": os.path.basename(mp4) if mp4 else None,
        "fontes": {"campo": "WindTunnelLBM (D3Q19, Smagorinsky LES)",
                   "volume": "densidade = critério Q (rotação menos deformação)",
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
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    # the voxel size and the origin come from the run that is on disk, not from a fresh plan: a still of a
    # 120-cell solve must not be placed as if it were the 140-cell one
    bodies = _write_bodies(case, os.path.join(out_dir, "corpo"))
    st = {**STYLE, **(kw.get("style") or {})}
    cfg = {
        "out_dir": out_dir, "bodies": bodies, "snapshots": meta["snapshots"],
        "voxel_size": float(meta["tunnel"]["dx"]) * meta["stride"],
        "origin": list(meta["tunnel"]["origin"]), "ground": case.ground,
        "shape": list(meta["shape_sub"]),
        "frame": _frame(case),
        "camera": {**CAMERA.get(case.name, CAMERA["carro"]), **(kw.get("camera") or {})},
        "style": st,
        "render": {"res_x": kw.get("res_x", 1920), "res_y": kw.get("res_y", 1080),
                   "samples": kw.get("samples", 48), "frames": kw["frames"], "fps": 24, "only_frame": frame,
                   "volumetric_samples": st["volumetric_samples"],
                   "volumetric_tile": st["volumetric_tile"],
                   "motion_blur": st.get("motion_blur", True)},
    }
    _run_blender(cfg)
    return os.path.join(out_dir, "frames", f"frame_{frame:04d}.png")
