"""From a sentence to a verified 3-D aircraft aerodynamics surrogate, shown where VeriPhysics rejects it and where it
accepts it (#1438).

The demo starts from a plain-text request: no geometry, no data. :func:`parse_request` reads the planform and the
angle of attack from the text (Portuguese or English) and says which values it did not find. The same request then
goes through three rounds of :func:`pinneapple_design.aero.aircraft_surrogate.aircraft_surrogate_study`:

1. a quick model, few designs and a short training: VeriPhysics grades it and rejects it;
2. the full dataset over the library's design space: the model is approved, but the requested design is outside the
   space it was trained and validated on, so the request is rejected by the applicability gate;
3. the design space extended to cover the request, with new solver data: the model is approved and the request is
   accepted; its error is then measured against the solver.

An acceptance map (aspect ratio x sweep) shows, cell by cell, where each model accepts or rejects a request, with the
solver's error for every cell. When Blender's Python module (``bpy``) is installed, Cycles renders the aircraft with
the pressure jump on its lifting surfaces and the vortex-lattice streamlines, tip vortices included; the
streamlines of a surrogate's render come from the circulation that surrogate predicts. Everything is put together in
one MP4 with ffmpeg.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
import unicodedata
from typing import Any

import numpy as np

from .aircraft_surrogate import (
    AMBER,
    BG,
    BOUNDS,
    DESIGN,
    FG,
    GRID,
    MUTED,
    NC,
    NS_TAIL,
    NS_WING,
    RED,
    TEAL,
    _airframe,
    _case,
    _ffmpeg,
    _gamma,
    _study,
)

__all__ = ["aircraft_surrogate_demo", "parse_request", "extend_bounds", "request_verdict", "acceptance_map"]

# the cheaper renders of cinematic="fast": resolution scale, samples per pixel of the stills and of the camera move,
# frames of the camera move
FAST_RENDER = {"scale": 0.6, "spp_still": 12, "spp_orbit": 6, "orbit_frames": 12}
DEFAULT_REQUEST = {"wing_area": 16.0, "aspect_ratio": 8.0, "taper": 0.7, "sweep_le": 4.0, "twist": -2.0, "alpha": 4.0}
# the values the vortex lattice and the airframe model can take (not the surrogate's design space)
PHYSICAL = {"wing_area": (6.0, 40.0), "aspect_ratio": (4.0, 16.0), "taper": (0.3, 1.0), "sweep_le": (0.0, 30.0),
            "twist": (-6.0, 2.0), "alpha": (-4.0, 10.0)}
LABEL = {"wing_area": "wing area (m²)", "aspect_ratio": "aspect ratio", "taper": "taper ratio",
         "sweep_le": "leading-edge sweep (°)", "twist": "tip twist (°)", "alpha": "angle of attack (°)"}
_KEYS = {
    "wing_area": r"area (?:da |de |of the )?(?:asa|alar|wing)|wing area|area alar|superficie alar|\bS ?=",
    "aspect_ratio": r"alongamento|razao de aspecto|aspect ratio|\bAR\b",
    "taper": r"afilamento|taper(?: ratio)?|razao de afilamento",
    "sweep_le": r"enflechamento|flecha|sweep(?:back)?",
    "twist": r"torcao|twist|washout",
    "alpha": r"angulo de ataque|angle of attack|\baoa\b|\balpha\b|\bα",
}
_NUM = r"([-−–]?\s?\d+(?:[.,]\d+)?)"
_PT = ("asa", "aeronave", "aviao", "angulo", "quero", "preciso", "enflechamento", "alongamento", "treinar", "modelo")


def _fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def parse_request(text: str) -> dict[str, Any]:
    """Read the design from a plain-text request.

    Each design value is the first number after one of its names (PT or EN: "alongamento 12", "aspect ratio of 12",
    "enflechamento de 15°", "ângulo de ataque 4 graus"); a decimal comma is read as a point. A value the text does not
    give is taken from :data:`DEFAULT_REQUEST` and listed in ``defaulted``; a value outside what the lattice and the
    airframe model can take (:data:`PHYSICAL`) is clipped and listed in ``clipped``. "Washout of 3°" is a tip twist
    of -3°.

    Returns
    -------
    dict
        ``design`` (the six values), ``found`` / ``defaulted`` / ``clipped`` (names), ``lang`` ("pt" or "en").
    """
    t = _fold(text)
    design, found, defaulted, clipped = {}, [], [], []
    for k in DESIGN:
        m = re.search(rf"(?:{_KEYS[k]})[^\d\-−–]{{0,28}}?{_NUM}", t)
        if m is None:
            design[k] = DEFAULT_REQUEST[k]
            defaulted.append(k)
            continue
        v = float(m.group(1).replace(" ", "").replace(",", ".").replace("−", "-").replace("–", "-"))
        if k == "twist" and "washout" in m.group(0) and v > 0:
            v = -v
        lo, hi = PHYSICAL[k]
        if not lo <= v <= hi:
            clipped.append(k)
            v = min(max(v, lo), hi)
        design[k] = v
        found.append(k)
    lang = "pt" if sum(w in t for w in _PT) >= 2 else "en"
    return {"design": design, "found": found, "defaulted": defaulted, "clipped": clipped, "lang": lang}


def extend_bounds(bounds: dict[str, tuple[float, float]], design: dict[str, float], margin: float = 0.15
                  ) -> dict[str, tuple[float, float]]:
    """The design space grown to contain ``design``: each value outside its range moves that end past the value by
    ``margin`` of the range's original width (clipped to :data:`PHYSICAL`)."""
    out = {}
    for k in DESIGN:
        lo, hi = bounds[k]
        w = hi - lo
        v = design[k]
        if v > hi:
            hi = v + margin * w
        if v < lo:
            lo = v - margin * w
        out[k] = (max(lo, PHYSICAL[k][0]), min(hi, PHYSICAL[k][1]))
    return out


def request_verdict(model_verdict: str, flagged: bool) -> str:
    """A request is answered only by an approved model, and only inside the space that model was validated on:
    REJECT when the model is rejected or the request is flagged outside its domain, else the model's verdict."""
    if model_verdict == "REJECT" or flagged:
        return "REJECT"
    return model_verdict


def acceptance_map(models: dict[str, Any], design: dict[str, float], n: int = 8,
                   ar: tuple[float, float] | None = None, sweep: tuple[float, float] | None = None) -> dict[str, Any]:
    """For an ``n`` x ``n`` grid of aspect ratio x sweep (the other values from ``design``), each model's
    applicability flag and its field error against the solver.

    Returns
    -------
    dict
        ``ar``, ``sweep`` (grid axes), ``error`` and ``accepted`` per model name (n x n arrays, [sweep, ar]),
        ``solver_seconds``.
    """
    ar = ar or (6.0, max(14.0, design["aspect_ratio"] + 1.5))
    sweep = sweep or (0.0, max(20.0, design["sweep_le"] + 3.0))
    A = np.linspace(*ar, n)
    S = np.linspace(*sweep, n)
    err = {k: np.zeros((n, n)) for k in models}
    acc = {k: np.zeros((n, n), bool) for k in models}
    t = time.perf_counter()
    for i, s in enumerate(S):
        for j, a in enumerate(A):
            row = dict(design, aspect_ratio=float(a), sweep_le=float(s))
            c = _case({k: row[k] for k in DESIGN if k != "alpha"})
            truth = c.dcp(row["alpha"])
            x = np.array([row[k] for k in DESIGN])
            for name, m in models.items():
                p = m.predict(x[None])[0][0]
                err[name][i, j] = float(np.linalg.norm(p - truth) / max(np.linalg.norm(truth), 1e-300))
                acc[name][i, j] = not m.ood(x)["ood"]
    return {"ar": A, "sweep": S, "error": err, "accepted": acc, "solver_seconds": time.perf_counter() - t}


# ------------------------------------------------------------------------------------------------- cinematic
def _cycles_scene(row: dict[str, float], dcp: np.ndarray, gamma: np.ndarray, alpha_deg: float, res, spp: int):
    """A Cycles scene of the aircraft: the pressure jump as vertex colours on the wing and tail, the body in dark
    gloss, streamlines from the circulation ``gamma`` (an upstream rake and rings around the tips)."""
    import matplotlib.pyplot as plt

    from pinneapple_blender.cycles import CyclesScene

    from .vlm import solve, streamlines

    af = _airframe(row)
    sol = solve(af, af.mac_le_x + 0.25 * af.mac, nc=NC, ns_wing=NS_WING, ns_tail=NS_TAIL)
    L = sol.L
    parts = [p for p in af.build("high") if p.aero]
    cmap = plt.get_cmap("turbo")
    sc = CyclesScene(resolution=res, samples=spp, background=(0.006, 0.008, 0.014), ambient=0.25)
    body = sc.material((0.10, 0.11, 0.13), roughness=0.28)
    glass = sc.material((0.02, 0.03, 0.05), roughness=0.05)
    vcol = sc.material(roughness=0.32, vertex_colors="col")
    for p in parts:
        if p.name.startswith(("wing", "htail")):
            V = p.vertices
            near = np.argmin(np.linalg.norm(V[:, None, :2] - L.C[None, :, :2], axis=2)
                             + 50.0 * (np.abs(V[:, None, 2] - L.C[None, :, 2]) > 0.6), axis=1)
            sc.add_mesh(V, p.faces, material=vcol, colors=cmap(np.clip(dcp[near] / 1.6, 0, 1))[:, :3], name=p.name)
        else:
            sc.add_mesh(p.vertices, p.faces, material=glass if "glass" in p.name else body, name=p.name)
    a = math.radians(alpha_deg)
    G = np.zeros((3, len(gamma)))
    G[0] = gamma
    flow = type("Circulation", (), {"L": L, "G": G})
    b2 = af.span / 2
    x0, z0 = af.wing_x - 1.6, af.wing_z_root()
    seeds = np.array([[x0, y, z] for y in np.linspace(-0.95 * b2, 0.95 * b2, 11) for z in (z0 - 0.18, z0 + 0.10)])
    tips = []
    for sgn in (-1, 1):
        k = int(np.argmax(L.C[:, 1] * sgn * (L.surf == 0)))
        xt, zt = L.C[k, 0] + 0.35 * L.chord[L.strip[k]], L.C[k, 2]
        tips += [[xt, sgn * b2 + 0.11 * math.cos(th), zt + 0.11 * math.sin(th)]
                 for th in np.linspace(0, 2 * math.pi, 7)[:-1]]
    SL = np.concatenate([streamlines(flow, a, 0.0, seeds, steps=230, h=0.06),
                         streamlines(flow, a, 0.0, np.array(tips), steps=230, h=0.04, core=0.05)])
    for k, line in enumerate(SL):
        tip = k >= len(seeds)
        col = (1.0, 0.62, 0.18) if tip else (0.18, 0.85, 1.0) if abs(line[0, 1]) < 0.7 * b2 else (0.62, 0.45, 1.0)
        sc.add_tube(line, radius=0.011 if tip else 0.007, color=col, emission=2.2 if tip else 1.4, resolution=2)
    P = np.concatenate([p.vertices for p in parts])
    lo, hi = P.min(0), P.max(0)
    sc.add_plane(z=lo[2] - 0.9, size=(60, 60), center=(4, 0), color=(0.015, 0.018, 0.024), alpha=1.0, roughness=0.65)
    sc.light("AREA", energy=260, location=(1, -6, 6), rotation=(45, 0, -15), size=7)
    sc.light("AREA", energy=420, location=(12, 3, 2.5), rotation=(80, 0, 100), size=3)
    sc.light("AREA", energy=90, location=(-4, 4, 3), rotation=(60, 0, -130), size=5)
    _filmic(sc)
    return sc, lo, hi


def _filmic(sc) -> None:
    """AgX tone mapping and a fog-glow bloom (both skipped where this Blender lacks them)."""
    try:
        sc.scene.view_settings.view_transform = "AgX"
        sc.scene.view_settings.look = "AgX - Punchy"
    except (TypeError, AttributeError):
        pass
    try:
        bpy, scn = sc.bpy, sc.scene
        if not hasattr(scn, "compositing_node_group"):
            return
        ng = bpy.data.node_groups.new("comp", "CompositorNodeTree")
        scn.compositing_node_group = ng
        rl = ng.nodes.new("CompositorNodeRLayers")
        gl = ng.nodes.new("CompositorNodeGlare")
        out = ng.nodes.new("NodeGroupOutput")
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        try:
            gl.glare_type = "FOG_GLOW"
        except (TypeError, AttributeError):
            if "Type" in gl.inputs:
                gl.inputs["Type"].default_value = "Fog Glow"
        for k, v in (("Threshold", 0.6), ("Strength", 0.6), ("Size", 0.7)):
            if k in gl.inputs:
                gl.inputs[k].default_value = v
        ng.links.new(rl.outputs["Image"], gl.inputs["Image"])
        ng.links.new(gl.outputs["Image"], out.inputs[0])
    except Exception:  # noqa: BLE001 - the bloom is cosmetic
        pass


def _render_views(row: dict[str, float], dcp: np.ndarray, gamma: np.ndarray, alpha_deg: float, res, spp: int,
                  views: list) -> list[str]:
    """Build the Cycles scene once and render it from each ``(direction, margin, path)`` of ``views``."""
    sc, lo, hi = _cycles_scene(row, dcp, gamma, alpha_deg, res, spp)
    for direction, margin, path in views:
        sc.fit_camera(lo, hi, direction=direction, ortho=False, lens=36, margin=margin)
        sc.render(path)
    return [v[2] for v in views]


def _in_child(fn, *args):
    """Run ``fn(*args)`` in a fresh (spawned) process: Blender keeps global state and can crash on teardown, and
    neither may reach the caller's process (a job runner reads a crash at exit as a failed run)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fn, *args).result()


def _orbit_dir(k: int, n: int) -> tuple[float, float, float]:
    th = math.radians(-138.0 + 80.0 * k / max(n - 1, 1))          # from the front-left round to the rear-left
    return (math.cos(th), math.sin(th), 0.55 - 0.15 * k / max(n - 1, 1))


def cinematic_available() -> bool:
    """Whether Blender's Python module is installed. Found, not imported: Blender is only ever imported in the child
    process that renders (:func:`_in_child`); importing it in the caller, often a worker thread, can crash it."""
    import importlib.util

    try:
        return importlib.util.find_spec("bpy") is not None
    except (ImportError, ValueError):
        return False


# ------------------------------------------------------------------------------------------------- cards
_T = {
    "en": {"title": "PINNeAPPle · from a sentence to a verified surrogate", "request": "THE REQUEST",
           "read": "what PINNeAPPle read from it", "default": "not in the text, default", "round": "ROUND",
           "model": "VeriPhysics grade of the model", "req": "your request", "inside": "inside the validated domain",
           "outside": "outside the validated domain", "map": "Where each model accepts a request",
           "r1": "a quick model: 8 designs, a short training", "r2": "the full dataset over the library's design space",
           "r3": "design space extended to your request, new solver data",
           "truth": "vortex lattice (the solver)", "pred": "surrogate", "end": "measured in this run",
           "acc": "accepted", "rej": "rejected", "err": "error vs solver", "cbar": "field error vs solver (relative L2)",
           "caption": "ΔCp on the lifting surfaces · streamlines from this circulation · tip vortices",
           "nogeo": "no geometry, no data: the airframe is built from these numbers, the training data comes from "
                    "the vortex-lattice solver",
           "rnd": "round", "model_v": "model", "req_v": "request", "req_err": "request error vs solver",
           "held": "held-out field error", "designs": "designs", "ep": "epochs"},
    "pt": {"title": "PINNeAPPle · de uma frase a um surrogate verificado", "request": "O PEDIDO",
           "read": "o que o PINNeAPPle leu do texto", "default": "não está no texto, padrão", "round": "RODADA",
           "model": "nota VeriPhysics do modelo", "req": "seu pedido", "inside": "dentro do domínio validado",
           "outside": "fora do domínio validado", "map": "Onde cada modelo aceita um pedido",
           "r1": "um modelo rápido: 8 projetos, treino curto", "r2": "o conjunto completo no espaço de projeto da biblioteca",
           "r3": "espaço de projeto estendido até o seu pedido, novos dados do solver",
           "truth": "vortex lattice (o solver)", "pred": "surrogate", "end": "medido nesta execução",
           "acc": "aceito", "rej": "rejeitado", "err": "erro vs solver", "cbar": "erro do campo vs solver (L2 relativo)",
           "caption": "ΔCp nas superfícies sustentadoras · linhas de corrente desta circulação · vórtices de ponta",
           "nogeo": "sem geometria, sem dados: a aeronave é montada com estes números, os dados de treino vêm do "
                    "solver vortex lattice",
           "rnd": "rodada", "model_v": "modelo", "req_v": "pedido", "req_err": "erro do pedido vs solver",
           "held": "erro do campo em projetos não vistos", "designs": "projetos", "ep": "épocas"},
}
_COL = {"APPROVED": TEAL, "REVIEW": AMBER, "REJECT": RED}


def _card_fig(W):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(W, W * 9 / 16), dpi=100, facecolor=BG)
    return fig


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(" ".join(text.split()), width) or [""]


def _write_frames(frames_dir: str, start: int, draw, n: int) -> int:
    """``draw(fig, i)`` for i < n into frame PNGs numbered from ``start``; returns the next number."""
    import matplotlib.pyplot as plt
    for i in range(n):
        fig = draw(i)
        fig.savefig(os.path.join(frames_dir, f"f{start + i:05d}.png"), facecolor=BG)
        plt.close(fig)
    return start + n


def _encode(exe: str, pattern_dir: str, fps: int, out: str, size=(1280, 720)) -> None:
    subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(pattern_dir, "f%05d.png"),
                    "-vf", f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,"
                           f"pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2:color=0x0b0f17,fps=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-movflags", "+faststart", out], check=True)


# ------------------------------------------------------------------------------------------------- the demo
def aircraft_surrogate_demo(request: str = "", out_dir: str = "aircraft_surrogate_demo", rounds: int = 3,
                            n_train: tuple[int, int, int] = (8, 48, 64), epochs: tuple[int, int, int] = (150, 1500, 1500),
                            n_test: int = 12, n_alpha: int = 6, frames: int = 50, fps: int = 10,
                            width_px: int = 1280, map_n: int = 8, cinematic: str = "auto", orbit_frames: int = 28,
                            render_scale: float = 1.0, seed: int = 0) -> dict[str, Any]:
    """From a plain-text request to a 3-D aircraft aerodynamics surrogate graded by VeriPhysics, shown where the
    grade rejects it and where it accepts it.

    Parameters
    ----------
    request : str
        The problem in words (Portuguese or English), e.g. "a light aircraft with a slender wing: aspect ratio 12,
        sweep 15°, at 4° angle of attack". No geometry: the airframe is the library's light aircraft with the
        planform the text gives (:func:`parse_request`).
    out_dir : str
        Folder for the video, the renders, the figures and the 3-D field view.
    rounds : int
        1 to 3: the quick model, the full model on the library's design space, the design space extended to cover
        the request.
    n_train : tuple of int
        Training geometries of each round.
    epochs : tuple of int
        Training epochs of each round.
    n_test : int
        Held-out geometries of each round, never seen in training.
    n_alpha : int
        Angles of attack per geometry.
    frames : int
        Snapshots in each round's training video.
    fps : int [1/s]
        Frame rate of the training videos.
    width_px : int [px]
        Width of the videos and figures.
    map_n : int
        Cells per side of the acceptance map (each solved by the lattice).
    cinematic : str
        "auto" (Cycles renders when ``bpy`` is installed), "on", "fast" or "off". "fast" is the same renders at a
        lower cost (:data:`FAST_RENDER`: at most 0.6 of the resolution, fewer samples, at most 12 orbit frames), for a
        run that has to fit a short job limit.
    orbit_frames : int
        Frames of the Cycles camera move.
    render_scale : float
        Scale of the Cycles resolution (1 is 1280 x 720 for the stills, 960 x 540 for the camera move).
    seed : int
        Seed of every round's designs and networks.

    Returns
    -------
    dict
        ``request`` (the parsed design), ``rounds`` (each round's verdicts, checks and numbers), ``map`` (acceptance
        per model), ``checks`` (the demo's own, measured), ``outputs``, ``files`` (in ``out_dir``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    req = parse_request(request or "")
    design, lang = req["design"], req["lang"]
    T = _T[lang]
    W = width_px / 100
    rounds = max(1, min(int(rounds), 3))
    files: list[str] = []

    # the rounds ----------------------------------------------------------------------------------------------
    plans = [("r1", dict(BOUNDS), n_train[0], epochs[0]), ("r2", dict(BOUNDS), n_train[1], epochs[1]),
             ("r3", extend_bounds(BOUNDS, design), n_train[2], epochs[2])][:rounds]
    R, models = [], {}
    for k, (key, B, ntr, ep) in enumerate(plans, 1):
        d = os.path.join(out_dir, f"round{k}")
        res, model, show, show_a = _study(d, ntr, n_test, n_alpha, ep, frames, fps, width_px, seed, B, design, design,
                                          headline=f"{T['round']} {k} · {T[key]}")
        o = res["outputs"]
        rv = request_verdict(res["decision"]["verdict"], o["target_flagged"])
        R.append({"round": k, "plan": T[key], "bounds": B, "n_train": ntr, "epochs": ep,
                  "model_verdict": res["decision"]["verdict"], "model_reasons": res["decision"]["reasons"],
                  "request_flagged": o["target_flagged"], "request_verdict": rv,
                  "request_field_error": o["target_field_error"], "field_error_held_out": o["field_error_held_out"],
                  "checks": res["checks"], "outputs": o, "files": [f"round{k}/{f}" for f in res["files"]],
                  "seconds": o["total_seconds"]})
        models[f"round {k}"] = model
        for f in res["files"]:
            if f.endswith((".png", ".glb", ".json")) and (k == len(plans) or f == "aircraft_surrogate_poster.png"):
                dst = f"round{k}_{f}"
                shutil.copy(os.path.join(d, f), os.path.join(out_dir, dst))
                files.append(dst)

    # the acceptance map -------------------------------------------------------------------------------------
    use = {k: m for k, m in models.items() if k != "round 1"} or models
    amap = acceptance_map(use, design, n=map_n)
    tol = 0.05
    fig, axs = plt.subplots(1, len(use), figsize=(W, W * 0.42), facecolor=BG, squeeze=False)
    norm = mcolors.LogNorm(vmin=1e-3, vmax=1.0)
    for ax, name in zip(axs[0], use, strict=True):
        E, Acc = amap["error"][name], amap["accepted"][name]
        ax.set_facecolor(BG)
        im = ax.pcolormesh(amap["ar"], amap["sweep"], np.clip(E, 1e-3, 1.0), cmap="magma_r", norm=norm,
                           shading="nearest")
        for i in range(len(amap["sweep"])):
            for j in range(len(amap["ar"])):
                ax.text(amap["ar"][j], amap["sweep"][i], "✔" if Acc[i, j] else "✘",
                        color=TEAL if Acc[i, j] else RED, ha="center", va="center", fontsize=11, weight="bold")
        ax.scatter([design["aspect_ratio"]], [design["sweep_le"]], marker="*", s=420, color=AMBER, edgecolor=BG,
                   zorder=5)
        ax.set_title(f"{T['rnd']} {name.split()[-1]}: ✔ {T['acc']}  ✘ {T['rej']}  ★ {T['req']}", color=FG,
                     fontsize=12)
        ax.set_xlabel(LABEL["aspect_ratio"], color=MUTED)
        ax.set_ylabel(LABEL["sweep_le"], color=MUTED)
        ax.tick_params(colors=MUTED)
        for s in ax.spines.values():
            s.set_color(GRID)
    cb = fig.colorbar(im, ax=axs[0].tolist(), fraction=0.03, pad=0.02)
    cb.set_label(T["cbar"], color=FG)
    cb.ax.tick_params(colors=MUTED)
    fig.suptitle(T["map"], color=FG, fontsize=15)
    fig.savefig(os.path.join(out_dir, "acceptance_map.png"), dpi=100, facecolor=BG)
    plt.close(fig)
    files.append("acceptance_map.png")

    # the demo's own checks -----------------------------------------------------------------------------------
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None):
        checks.append({"code": code, "status": "pass" if ok else "fail", "detail": detail, "value": value})

    if rounds >= 2:
        chk("demo.grade_discriminates", R[0]["model_verdict"] == "REJECT" and R[1]["model_verdict"] == "APPROVED",
            f"round 1 ({R[0]['n_train']} designs, {R[0]['epochs']} epochs) graded {R[0]['model_verdict']} with "
            f"held-out field error {R[0]['field_error_held_out']:.3f}; round 2 ({R[1]['n_train']} designs) graded "
            f"{R[1]['model_verdict']} with {R[1]['field_error_held_out']:.3f}")
        outside = any(not (BOUNDS[k][0] <= design[k] <= BOUNDS[k][1]) for k in DESIGN)
        if outside:
            chk("demo.request_gate", R[1]["request_verdict"] == "REJECT",
                f"the request is outside the round-2 design space; round 2 {'flags' if R[1]['request_flagged'] else 'does NOT flag'} "
                f"it (its error against the solver: {R[1]['request_field_error']:.3f})", R[1]["request_field_error"])
    if rounds >= 3:
        chk("demo.extension_accepted", R[2]["request_verdict"] == "APPROVED",
            f"round 3 grades the model {R[2]['model_verdict']} and the request {R[2]['request_verdict']}; error of "
            f"the request against the solver {R[2]['request_field_error']:.4f}", R[2]["request_field_error"])
    for name in use:
        E, Acc = amap["error"][name], amap["accepted"][name]
        worst = float(E[Acc].max()) if Acc.any() else float("nan")
        rej = float(np.median(E[~Acc])) if (~Acc).any() else float("nan")
        chk(f"demo.accepted_cells_accurate[{name}]", bool(Acc.any()) and worst <= tol,
            f"{int(Acc.sum())} of {Acc.size} map cells accepted; worst error of an accepted cell against the solver "
            f"{worst:.4f} (allowed {tol:g}); median error of the rejected cells {rej:.3f}", worst)

    # cinematic renders -----------------------------------------------------------------------------------
    spp_still, spp_orbit = 24, 12
    if cinematic == "fast":
        render_scale = min(render_scale, FAST_RENDER["scale"])
        orbit_frames = min(orbit_frames, FAST_RENDER["orbit_frames"])
        spp_still, spp_orbit = FAST_RENDER["spp_still"], FAST_RENDER["spp_orbit"]
    want = cinematic in ("on", "fast") or (cinematic == "auto" and cinematic_available())
    stills, orbit_dir, render_s = [], None, 0.0
    if want:
        t = time.perf_counter()
        try:
            c = _case({k: design[k] for k in DESIGN if k != "alpha"})
            truth = c.dcp(design["alpha"])
            x = np.array([[design[k] for k in DESIGN]])
            sel = [("truth", truth, T["truth"], None)]
            for k in (1, len(R)) if len(R) > 1 else (1,):
                p = models[f"round {k}"].predict(x)[0][0]
                sel.insert(len(sel) - 1, (f"round{k}", p, f"{T['round']} {k} · {T['pred']}", R[k - 1]))
            rs = (int(1280 * render_scale), int(720 * render_scale))
            for name, dcp, label, rr in sel:
                raw = os.path.join(out_dir, f"_{name}.png")
                _in_child(_render_views, design, dcp, _gamma(c.L, dcp), design["alpha"], rs, spp_still,
                          [((-0.9, -0.85, 0.55), 0.92, raw)])
                img = plt.imread(raw)
                fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
                fig.figimage(img, 0, 0)
                fig.text(0.025, 0.93, label, color=FG, fontsize=20 * render_scale, weight="bold")
                if rr is not None:
                    err = rr["request_field_error"]
                    fig.text(0.025, 0.875, f"{T['err']} {err:.3f}", color=MUTED, fontsize=14 * render_scale)
                    v = rr["request_verdict"]
                    fig.text(0.975, 0.91, v, color=_COL[v], fontsize=34 * render_scale, weight="bold", ha="right",
                             bbox={"boxstyle": "round,pad=0.3", "fc": "none", "ec": _COL[v], "lw": 3})
                fig.text(0.025, 0.04, T["caption"], color=MUTED, fontsize=12 * render_scale)
                fn = f"cinematic_{name}.png"
                fig.savefig(os.path.join(out_dir, fn), dpi=100)
                plt.close(fig)
                os.remove(raw)
                stills.append(fn)
                files.append(fn)
            # the camera move, on the accepted model's field (the last round)
            best = sel[-2] if len(sel) > 1 else sel[0]
            ors = (int(960 * render_scale), int(540 * render_scale))
            orbit_dir = os.path.join(out_dir, "_orbit")
            os.makedirs(orbit_dir, exist_ok=True)
            _in_child(_render_views, design, best[1], _gamma(c.L, best[1]), design["alpha"], ors, spp_orbit,
                      [(_orbit_dir(k, orbit_frames), 0.95, os.path.join(orbit_dir, f"f{k:05d}.png"))
                       for k in range(orbit_frames)])
        except Exception as e:  # noqa: BLE001 - the renders are an extra; the demo stands without them
            checks.append({"code": "view.cinematic", "status": "inconclusive",
                           "detail": f"Cycles renders not made: {e}", "value": None})
            orbit_dir = None
        render_s = time.perf_counter() - t

    # the composed video ------------------------------------------------------------------------------------
    exe = _ffmpeg()
    video_s = 0.0
    if exe:
        t = time.perf_counter()
        seg_dir = os.path.join(out_dir, "_segments")
        os.makedirs(seg_dir, exist_ok=True)
        segs = []

        def seg(draw, n, rate=10):
            fd = os.path.join(seg_dir, f"_s{len(segs)}")
            os.makedirs(fd, exist_ok=True)
            _write_frames(fd, 0, draw, n)
            out = os.path.join(seg_dir, f"s{len(segs):02d}.mp4")
            _encode(exe, fd, rate, out)
            shutil.rmtree(fd, ignore_errors=True)
            segs.append(out)

        text = " ".join((request or "").split())

        def title(i, n_type=40):
            fig = _card_fig(W)
            fig.text(0.05, 0.88, T["title"], color=FG, fontsize=24, weight="bold")
            fig.text(0.05, 0.78, T["request"], color=AMBER, fontsize=14, weight="bold")
            shown = text[: int(len(text) * min(1.0, (i + 1) / n_type))]
            for li, line in enumerate(_wrap(shown + ("▌" if i < n_type else ""), 72)):
                fig.text(0.05, 0.72 - 0.055 * li, line, color=FG, fontsize=18, family="serif", style="italic")
            if i >= n_type:
                y = 0.36
                fig.text(0.05, y + 0.05, T["read"], color=TEAL, fontsize=14, weight="bold")
                for j, k in enumerate(DESIGN):
                    mark = T["default"] if k in req["defaulted"] else ""
                    fig.text(0.05 + 0.31 * (j % 3), y - 0.06 - 0.07 * (j // 3),
                             f"{LABEL[k]}: {design[k]:g}" + (f"  ({mark})" if mark else ""),
                             color=MUTED if mark else FG, fontsize=14, family="monospace")
                fig.text(0.05, 0.08, T["nogeo"], color=MUTED, fontsize=12)
            return fig

        seg(title, 40 + 35)
        for rr in R:
            mp4 = os.path.join(out_dir, f"round{rr['round']}", "aircraft_surrogate.mp4")
            if os.path.exists(mp4):
                out = os.path.join(seg_dir, f"s{len(segs):02d}.mp4")
                subprocess.run([exe, "-y", "-loglevel", "error", "-i", mp4, "-vf",
                                "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", out], check=True)
                segs.append(out)

            def verdict(i, rr=rr):
                fig = _card_fig(W)
                fig.text(0.05, 0.9, f"{T['round']} {rr['round']} · {rr['plan']}", color=FG, fontsize=20,
                         weight="bold")
                fig.text(0.05, 0.8, T["model"], color=MUTED, fontsize=14)
                fig.text(0.05, 0.68, rr["model_verdict"], color=_COL[rr["model_verdict"]], fontsize=44, weight="bold")
                for j, c in enumerate(rr["checks"][:8]):
                    mark, col = {"pass": ("✔", TEAL), "fail": ("✘", RED)}.get(c["status"], ("?", AMBER))
                    v = c.get("value")
                    vs = f"{v:.4g}" if isinstance(v, (int, float)) and v is not None and math.isfinite(v) else ""
                    fig.text(0.05, 0.58 - 0.055 * j, f"{mark}  {c['code']:<24s} {vs}", color=col, fontsize=13,
                             family="monospace")
                fig.text(0.58, 0.8, T["req"], color=MUTED, fontsize=14)
                fig.text(0.58, 0.68, rr["request_verdict"], color=_COL[rr["request_verdict"]], fontsize=44,
                         weight="bold")
                where = T["outside"] if rr["request_flagged"] else T["inside"]
                fig.text(0.58, 0.6, where, color=FG, fontsize=15)
                b = rr["bounds"]
                fig.text(0.58, 0.53, f"AR {b['aspect_ratio'][0]:g}–{b['aspect_ratio'][1]:g} · sweep "
                                     f"{b['sweep_le'][0]:g}–{b['sweep_le'][1]:g}°", color=MUTED, fontsize=13,
                         family="monospace")
                fig.text(0.58, 0.46, f"{T['req_v']}: AR {design['aspect_ratio']:g} · sweep {design['sweep_le']:g}°",
                         color=AMBER, fontsize=13, family="monospace")
                fig.text(0.58, 0.36, f"{T['held']} {rr['field_error_held_out']:.4f}", color=FG, fontsize=13,
                         family="monospace")
                fig.text(0.58, 0.31, f"{T['req_err']} {rr['request_field_error']:.4f}", color=FG,
                         fontsize=13, family="monospace")
                fig.text(0.58, 0.2, f"{rr['n_train']} {T['designs']} · {rr['epochs']} {T['ep']} · {rr['seconds']:.0f} s",
                         color=MUTED, fontsize=12)
                return fig

            seg(verdict, 45)

        def still(path, n=30):
            img = plt.imread(path)

            def draw(i):
                fig = _card_fig(W)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(img)
                ax.set_axis_off()
                return fig
            seg(draw, n)

        still(os.path.join(out_dir, "acceptance_map.png"), 45)
        for s in stills:
            still(os.path.join(out_dir, s), 30)
        if orbit_dir:
            out = os.path.join(seg_dir, f"s{len(segs):02d}.mp4")
            _encode(exe, orbit_dir, 10, out)
            segs.append(out)

        def end(i):
            fig = _card_fig(W)
            fig.text(0.05, 0.85, T["end"], color=MUTED, fontsize=16)
            y = 0.72
            for rr in R:
                fig.text(0.05, y, f"{T['round']} {rr['round']}", color=FG, fontsize=18, weight="bold")
                fig.text(0.2, y, f"{T['model_v']} {rr['model_verdict']:<9s} {T['req_v']} {rr['request_verdict']:<9s} "
                                 f"{T['err']} {rr['request_field_error']:.4f}", color=_COL[rr["request_verdict"]],
                         fontsize=16, family="monospace")
                y -= 0.09
            if R:
                o = R[-1]["outputs"]
                fig.text(0.05, y - 0.04, f"surrogate {o['surrogate_ms_per_inference']:.2f} ms per inference · solver "
                                         f"{o['solver_seconds_per_design'] * 1e3:.0f} ms per design · "
                                         f"{o['speedup_vs_solver']:.0f}x", color=FG, fontsize=14)
            fig.text(0.05, 0.08, "PINNeAPPle · pinneapple_design.aero.aircraft_surrogate_demo", color=MUTED,
                     fontsize=12)
            return fig

        seg(end, 40)
        lst = os.path.join(seg_dir, "list.txt")
        with open(lst, "w") as fh:
            fh.writelines(f"file '{os.path.abspath(s)}'\n" for s in segs)
        subprocess.run([exe, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                        "-movflags", "+faststart", os.path.join(out_dir, "aircraft_surrogate_demo.mp4")], check=True)
        files.insert(0, "aircraft_surrogate_demo.mp4")
        shutil.rmtree(seg_dir, ignore_errors=True)
        video_s = time.perf_counter() - t
    if orbit_dir:
        shutil.rmtree(orbit_dir, ignore_errors=True)

    last = R[-1]
    outputs = {"request_verdict": last["request_verdict"], "model_verdict": last["model_verdict"],
               "request_field_error": last["request_field_error"],
               "rounds": len(R), "map_cells": int(map_n * map_n), "cinematic": bool(stills),
               "render_seconds": render_s, "video_seconds": video_s, "map_solver_seconds": amap["solver_seconds"],
               "total_seconds": time.perf_counter() - t0}
    for rr in R:
        k = rr["round"]
        outputs[f"round{k}.model_verdict"] = rr["model_verdict"]
        outputs[f"round{k}.request_verdict"] = rr["request_verdict"]
        outputs[f"round{k}.field_error_held_out"] = rr["field_error_held_out"]
        outputs[f"round{k}.request_field_error"] = rr["request_field_error"]
    m = {name: {"accepted": amap["accepted"][name].tolist(), "error": amap["error"][name].tolist()} for name in use}
    return {"request": req, "rounds": [{k: v for k, v in rr.items() if k != "outputs"} for rr in R],
            "map": {"ar": amap["ar"].tolist(), "sweep": amap["sweep"].tolist(), "models": m},
            "checks": checks + last["checks"], "outputs": outputs, "files": files,
            "assumptions": ["the airframe is the library's light aircraft with the planform read from the text",
                            "vortex lattice of wing + horizontal tail: inviscid, small angles, no fuselage load",
                            "a request is answered only by an approved model and only inside the design space the "
                            "model was trained and validated on"]}
