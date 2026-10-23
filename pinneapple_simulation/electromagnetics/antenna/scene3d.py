"""3-D views of an antipodal Vivaldi from its solved fields, on the generic studio scene of PINNeAPPle.

Everything is built on :mod:`pinneapple_tools.visualization.studio` (``Scene`` / ``Surface`` / ``Animation``, glTF
and USD export, the three.js viewer, Blender Cycles with the jet colour bar); nothing here draws pixels itself.

* :func:`antenna_surfaces` -- the substrate (translucent laminate) and the copper of both faces (flares, slots,
  director, feed line) of :func:`.ava.layout`, 35 um thick, in millimetres, end-fire along +y;
* :func:`read_farfield` / :func:`balloon` -- the far field ``ff_<f>GHz.npz`` saved by :mod:`.ava` as a radiation
  balloon centred on the board: radius ``R * (G - floor) / (G_max - floor)``, coloured by the gain in dBi
  (``G = 4 pi U / P_acc``); the directivity of every vertex is kept too (``D = 4 pi U / P_rad``), so the balloon's
  maximum is the table's ``D_peak_dBi`` by construction (:func:`farfield_metrics` checks it);
* :func:`read_nearfield` -- the complex E on the two cut planes ``nf_<f>GHz.npz`` (substrate mid-plane and the
  vertical plane through the end-fire axis), as coloured surfaces: ``|E|`` in dB below the plane's 99.5th
  percentile (the singular feed gap would set the scale otherwise; clipped at 0 dB), or the instantaneous
  ``Re(E_x e^{j wt})`` over the same percentile of ``|E_x|`` (clipped to +-1);
* :func:`frequency_scene` -- antenna + balloon + cuts over the solved frequencies (one frame per frequency, stepped,
  never blended): the balloon changes shape with :meth:`Animation.add_surface_shape`;
* :func:`phase_scene` -- the cuts over one period (wt from 0 to 360 degrees) at one frequency;
* :func:`build_3d` -- every file: glTF and USD per frequency, the self-contained viewers, Blender stills (beauty and
  balloon) and the GIF / MP4 of the sweep and of the phase.
"""
from __future__ import annotations

import copy
import glob
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ava import ETA0, AVAParams, layout

__all__ = ["MATERIALS", "antenna_surfaces", "FarField", "read_farfield", "farfield_metrics", "balloon",
           "read_nearfield", "plane_surface", "frequency_scene", "phase_scene", "build_3d", "solved_files"]

#: copper and a translucent laminate (sRGB colours; alpha / transmission for glTF, USD and Cycles)
MATERIALS = {
    "copper": {"color": [0.85, 0.50, 0.32], "metallic": 1.0, "roughness": 0.3},
    "laminate": {"color": [0.86, 0.80, 0.55], "metallic": 0.0, "roughness": 0.35, "alpha": 0.45,
                 "transmission": 0.6, "double_sided": True},
}
COPPER_MM = 0.035


def antenna_surfaces(params: AVAParams | Mapping[str, Any], *, copper_mm: float = COPPER_MM):
    """Substrate and copper of the design as studio ``Surface`` objects (mm, group ``"antenna"``)."""
    from pinneapple_design.geometry_library.kit import box, extrude
    from pinneapple_tools.visualization.studio import Surface

    p = params if isinstance(params, AVAParams) else AVAParams.from_dict(
        {k: v for k, v in params.items() if k in AVAParams.__dataclass_fields__})
    lay = layout(p)
    out = [Surface("substrate", *box((p.W, p.Lb, p.h), (0.0, p.Lb / 2, p.h / 2)), "laminate", group="antenna")]
    out.append(Surface("copper_top", *extrude(lay.top, p.h, p.h + copper_mm), "copper", group="antenna"))
    out.append(Surface("copper_bottom", *extrude(lay.bottom, -copper_mm, 0.0), "copper", group="antenna"))
    if lay.director is not None:
        out.append(Surface("director", *extrude(lay.director, p.h, p.h + copper_mm), "copper", group="antenna"))
    return out


# --------------------------------------------------------------------------------------------- far field
@dataclass
class FarField:
    f_ghz: float
    theta: np.ndarray            # (n_phi, n_theta) radians
    phi: np.ndarray
    U: np.ndarray                # radiation intensity, W/sr for 1 W incident
    P_rad: float                 # integral of U over the sphere
    P_acc: float | None          # accepted power (None for files written before it was saved)

    @property
    def directivity_dbi(self) -> np.ndarray:
        return 10 * np.log10(np.maximum(4 * np.pi * self.U / self.P_rad, 1e-30))

    @property
    def gain_dbi(self) -> np.ndarray:
        """IEEE gain ``4 pi U / P_acc``; the directivity when the accepted power is unknown."""
        if not self.P_acc:
            return self.directivity_dbi
        return 10 * np.log10(np.maximum(4 * np.pi * self.U / self.P_acc, 1e-30))


def read_farfield(path: str) -> FarField:
    """A far-field file of :mod:`.ava` (``theta``, ``phi`` 1-D; ``E_theta``, ``E_phi`` on the (phi, theta) grid)."""
    d = np.load(path)
    th, ph = d["theta"], d["phi"]
    T, P = np.meshgrid(th, ph)
    U = (np.abs(d["E_theta"]) ** 2 + np.abs(d["E_phi"]) ** 2) / (2 * ETA0)
    prad = float(d["P_rad"]) if "P_rad" in d.files else \
        float(np.trapezoid(np.trapezoid(U * np.sin(T), th, axis=1), ph))
    pacc = float(d["P_acc"]) if "P_acc" in d.files else None
    return FarField(float(d["f_GHz"]), T, P, U, prad, pacc)


def farfield_metrics(ff: FarField) -> dict[str, float]:
    """Peak and boresight (+y) directivity and gain of the far-field grid, in dBi."""
    D, G = ff.directivity_dbi, ff.gain_dbi
    j = np.argmin(np.abs(ff.theta - np.pi / 2) + np.abs(ff.phi - np.pi / 2))
    return {"f_GHz": ff.f_ghz, "D_peak_dBi": float(D.max()), "G_peak_dBi": float(G.max()),
            "D_bore_grid_dBi": float(D.flat[j]), "G_bore_grid_dBi": float(G.flat[j])}


def balloon(ff: FarField, centre: Sequence[float], radius: float, *, g_max: float | None = None,
            dyn_db: float = 30.0, quantity: str = "gain") -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Radiation balloon (vertices, faces, per-vertex fields ``gain_dBi`` and ``directivity_dBi``).

    The radius of direction (theta, phi) is ``radius * clip((X - (g_max - dyn_db)) / dyn_db, 0.01, 1)`` with X the
    gain (``quantity="gain"``) or the directivity in dBi, so balloons of several frequencies drawn with one
    ``g_max`` share a scale."""
    X = ff.gain_dbi if quantity == "gain" else ff.directivity_dbi
    top = float(X.max()) if g_max is None else float(g_max)
    r = radius * np.clip((X - (top - dyn_db)) / dyn_db, 0.01, 1.0)
    d = np.stack([np.sin(ff.theta) * np.cos(ff.phi), np.sin(ff.theta) * np.sin(ff.phi), np.cos(ff.theta)], -1)
    V = (np.asarray(centre, float) + r[..., None] * d).reshape(-1, 3)
    n_phi, n_th = ff.theta.shape
    i, j = np.meshgrid(np.arange(n_phi - 1), np.arange(n_th - 1), indexing="ij")
    a, b, c, e = i * n_th + j, (i + 1) * n_th + j, (i + 1) * n_th + j + 1, i * n_th + j + 1
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, e], -1).reshape(-1, 3)])
    return V, F, {"gain_dBi": ff.gain_dbi.ravel(), "directivity_dBi": ff.directivity_dbi.ravel()}


# --------------------------------------------------------------------------------------------- near field
def read_nearfield(path: str) -> dict[str, dict[str, np.ndarray]]:
    """``{plane: {"points": (nv, nu, 3) mm, "E": (3, nv, nu) complex}}`` of a near-field file of :mod:`.ava`."""
    d = np.load(path)
    out = {}
    for name in sorted({k.rsplit("_", 1)[0] for k in d.files if k.endswith("_E")}):
        nv, nu = (int(x) for x in d[f"{name}_shape"])
        a, b = np.linspace(0, 1, nu), np.linspace(0, 1, nv)
        P = d[f"{name}_origin"] + a[None, :, None] * d[f"{name}_u"] + b[:, None, None] * d[f"{name}_v"]
        out[name] = {"points": P, "E": d[f"{name}_E"].astype(complex)}
    return out


def plane_surface(points: np.ndarray, name: str, group: str = "cuts"):
    """A studio ``Surface`` of a structured (nv, nu, 3) grid of points."""
    from pinneapple_tools.visualization.studio import Surface
    nv, nu = points.shape[:2]
    i, j = np.meshgrid(np.arange(nv - 1), np.arange(nu - 1), indexing="ij")
    a, b, c, e = i * nu + j, i * nu + j + 1, (i + 1) * nu + j + 1, (i + 1) * nu + j
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, e], -1).reshape(-1, 3)])
    return Surface(name, points.reshape(-1, 3).astype(float), F, "plastic", group=group)


REF_PERCENTILE = 99.5   # field reference of a plane: the singular feed gap would otherwise set the whole scale


def _e_db(E: np.ndarray) -> np.ndarray:
    """``20 log10(|E| / |E|_ref)`` clipped at 0 dB, ``|E|_ref`` the plane's 99.5th percentile."""
    m = np.sqrt(np.nansum(np.abs(E) ** 2, axis=0))
    ref = float(np.nanpercentile(m, REF_PERCENTILE)) or 1.0
    return np.minimum(20 * np.log10(np.maximum(m / ref, 1e-6)), 0.0)


# --------------------------------------------------------------------------------------------- scenes
def solved_files(run_dir: str) -> dict[float, dict[str, str]]:
    """``{f_GHz: {"ff": path, "nf": path?}}`` of every far-field file under ``run_dir`` (any depth)."""
    out: dict[float, dict[str, str]] = {}
    for p in glob.glob(os.path.join(run_dir, "**", "ff_*GHz.npz"), recursive=True):
        f = float(re.search(r"ff_([0-9.]+)GHz", os.path.basename(p)).group(1))
        out.setdefault(f, {})["ff"] = p
        nf = os.path.join(os.path.dirname(p), f"nf_{f:g}GHz.npz")
        if os.path.exists(nf):
            out[f]["nf"] = nf
    return dict(sorted(out.items()))


def _balloon_geometry(p: AVAParams) -> tuple[np.ndarray, float]:
    return np.array([0.0, p.Lb / 2, p.h / 2]), 0.6 * p.Lb


def frequency_scene(params: AVAParams | Mapping[str, Any], files: Mapping[float, Mapping[str, str]], *,
                    rows: Sequence[Mapping[str, Any]] = (), dyn_db: float = 30.0, title: str = "") -> tuple[Any, dict]:
    """Antenna, gain balloon and |E| cuts over the solved frequencies of ``files`` (:func:`solved_files`).

    Returns ``(scene, info)``; ``info`` holds the balloon metrics per frequency (:func:`farfield_metrics`) and the
    common gain scale. ``rows`` (the table rows of this design) feed the head-up values (S11, realized gain)."""
    from pinneapple_tools.visualization.studio import Scene
    from pinneapple_tools.visualization.studio.animation import Animation

    p = params if isinstance(params, AVAParams) else AVAParams.from_dict(
        {k: v for k, v in params.items() if k in AVAParams.__dataclass_fields__})
    freqs = sorted(files)
    ffs = [read_farfield(files[f]["ff"]) for f in freqs]
    g_max = max(float(ff.gain_dbi.max()) for ff in ffs)
    centre, R = _balloon_geometry(p)
    balls = [balloon(ff, centre, R, g_max=g_max, dyn_db=dyn_db) for ff in ffs]
    surfaces = antenna_surfaces(p)
    from pinneapple_tools.visualization.studio import Surface
    # animated surfaces carry no static field: the viewer opens on the materials, the animation colours them
    surfaces.append(Surface("balloon", balls[0][0], balls[0][1], "plastic", group="balloon"))
    nfs = [read_nearfield(files[f]["nf"]) if "nf" in files[f] else None for f in freqs]
    planes = sorted(nfs[0]) if all(n is not None for n in nfs) else []
    for name in planes:
        surfaces.append(plane_surface(nfs[0][name]["points"], f"cut_{name}"))
    sc = Scene(surfaces, axes="z_up", title=title or "Antipodal Vivaldi: gain balloon and |E| cuts",
               materials=dict(MATERIALS),
               labels={"gain_dBi": "gain [dBi]", "directivity_dBi": "directivity [dBi]", "E_dB": "|E| [dB re 99.5th pct]"})
    an = Animation(np.asarray(freqs, float), time_label="f", time_unit="GHz", interpolate=False)
    an.add_surface_shape("balloon", np.stack([b[0] for b in balls]))
    an.add_surface_field("balloon", np.stack([b[2]["gain_dBi"] for b in balls]), label="gain [dBi]",
                         color_range=(g_max - dyn_db, g_max))
    for name in planes:
        an.add_surface_field(f"cut_{name}", np.stack([_e_db(n[name]["E"]).ravel() for n in nfs]),
                             label="|E| [dB re 99.5th pct]", color_range=(-40.0, 0.0))
    mets = [farfield_metrics(ff) for ff in ffs]
    an.add_hud("Dpk", [m["D_peak_dBi"] for m in mets], "D peak", "dBi", "{:.2f}")
    an.add_hud("Gpk", [m["G_peak_dBi"] for m in mets], "G peak", "dBi", "{:.2f}")
    by_f = {round(float(r["f_GHz"]), 6): r for r in rows}
    for key, col, lab, unit in (("Gr", "Greal_bore_dBi", "G realized, +y", "dBi"), ("S11", "S11_dB", "S11", "dB")):
        vals = [float(by_f.get(round(f, 6), {}).get(col, np.nan)) for f in freqs]
        if np.isfinite(vals).any():
            an.add_hud(key, vals, lab, unit, "{:.2f}")
    sc.animation = an
    return sc, {"freqs": freqs, "g_max_dBi": g_max, "dyn_db": dyn_db, "balloon": mets, "planes": planes}


def phase_scene(params: AVAParams | Mapping[str, Any], nf_path: str, *, n_phase: int = 24, component: int = 0,
                title: str = ""):
    """The cut planes at one frequency over one period: ``Re(E_c e^{j wt})`` over the 99.5th percentile of
    ``|E_c|`` (clipped to +-1) on jet, wt in degrees
    (``component`` 0 = x, the polarisation of the AVA aperture field)."""
    from pinneapple_tools.visualization.studio import Scene
    from pinneapple_tools.visualization.studio.animation import Animation

    p = params if isinstance(params, AVAParams) else AVAParams.from_dict(
        {k: v for k, v in params.items() if k in AVAParams.__dataclass_fields__})
    nf = read_nearfield(nf_path)
    f = float(np.load(nf_path)["f_GHz"])
    wt = np.arange(n_phase) * 360.0 / n_phase
    surfaces = antenna_surfaces(p)
    an = Animation(wt, time_label="ωt", time_unit="°", time_decimals=0)
    comp = "xyz"[component]
    for name in sorted(nf):
        Ec = nf[name]["E"][component]
        amp = float(np.nanpercentile(np.abs(Ec), REF_PERCENTILE)) or 1.0
        frames = np.clip(np.stack([np.real(Ec * np.exp(1j * np.deg2rad(a))).ravel() / amp for a in wt]), -1, 1)
        surfaces.append(plane_surface(nf[name]["points"], f"cut_{name}"))
        an.add_surface_field(f"cut_{name}", frames, label=f"Re(E{comp} e^jωt) / |E{comp}| 99.5th pct",
                             color_range=(-1.0, 1.0))
    an.add_hud("f", np.full(n_phase, f), "f", "GHz", "{:.2f}")
    sc = Scene(surfaces, axes="z_up", title=title or f"Antipodal Vivaldi: E{comp} over one period at {f:g} GHz",
               materials=dict(MATERIALS))
    sc.animation = an
    return sc


# --------------------------------------------------------------------------------------------- everything
def build_3d(params: AVAParams | Mapping[str, Any], run_dir: str, out_dir: str, *,
             rows: Sequence[Mapping[str, Any]] = (), name: str = "design", render: bool = True,
             size: tuple[int, int] = (960, 540), samples: int = 32, phase_freq: float | None = None,
             log=print) -> dict[str, Any]:
    """Every 3-D output of one design from the files of its runs (``run_dir``) into ``out_dir``:

    * ``<name>_<f>GHz.glb`` / ``.usda``: antenna, balloon (fields gain_dBi, directivity_dBi) and cuts (E_dB) at
      each solved frequency, ``<name>_antenna.glb`` the antenna alone;
    * ``<name>_viewer.html`` (sweep: frequency slider, layers antenna / balloon / cuts) and
      ``<name>_phase_viewer.html`` (wt slider), self-contained;
    * with ``render`` and Blender (``pip install bpy``): ``<name>_beauty.png``, ``<name>_balloon.png``,
      ``<name>_sweep.gif`` / ``.mp4`` and ``<name>_phase.gif`` / ``.mp4``.

    Returns the file paths, the balloon metrics per frequency and what was skipped."""
    from pinneapple_tools.visualization.studio import Scene, web_viewer

    os.makedirs(out_dir, exist_ok=True)
    files = solved_files(run_dir)
    if not files:
        raise FileNotFoundError(f"no far-field files ff_*GHz.npz under {run_dir}")
    sc, info = frequency_scene(params, files, rows=rows, title=f"{name}: gain balloon and |E| cuts")
    out: dict[str, Any] = {"info": info, "files": {}, "skipped": []}
    W = out["files"]
    ant = Scene(antenna_surfaces(params), axes="z_up", title=f"{name} antenna", materials=dict(MATERIALS))
    W["antenna_glb"] = ant.save(os.path.join(out_dir, f"{name}_antenna.glb"))
    an = sc.animation
    for k, f in enumerate(info["freqs"]):          # one static scene per frequency (frame k)
        s1 = Scene([copy.copy(s) for s in sc.surfaces], axes="z_up", title=f"{name} at {f:g} GHz",
                   materials=dict(MATERIALS), labels=sc.labels)
        for s in s1.surfaces:
            s.fields = dict(s.fields)
            if s.name in an.surface_shapes:
                s.vertices = an.surface_shapes[s.name]["vertices"][k]
            if s.name in an.surface_fields:
                key = "gain_dBi" if s.name == "balloon" else "E_dB"
                s.fields[key] = an.surface_fields[s.name]["values"][k]
        W[f"glb_{f:g}"] = s1.save(os.path.join(out_dir, f"{name}_{f:g}GHz.glb"))
        W[f"usda_{f:g}"] = s1.save(os.path.join(out_dir, f"{name}_{f:g}GHz.usda"))
    W["viewer"] = web_viewer(sc, out_dir, single_file=True, filename=f"{name}_viewer.html")
    nf_fs = [f for f in info["freqs"] if "nf" in files[f]]
    ph = None
    if nf_fs:
        fp = phase_freq if phase_freq in nf_fs else nf_fs[len(nf_fs) // 2]
        ph = phase_scene(params, files[fp]["nf"], title=f"{name}: Ex over one period at {fp:g} GHz")
        W["phase_viewer"] = web_viewer(ph, out_dir, single_file=True, filename=f"{name}_phase_viewer.html")
        out["phase_freq"] = fp
    else:
        out["skipped"].append("near field: no nf_*GHz.npz (run the design with nf_step > 0)")
    if not render:
        return out
    try:
        import bpy  # noqa: F401
    except ImportError:
        out["skipped"].append("Blender renders: bpy is not installed (pip install bpy)")
        return out
    from pinneapple_tools.visualization.studio import blender
    from pinneapple_tools.visualization.studio.animation import Animation
    t0 = time.perf_counter()
    W["beauty_png"] = blender.render(ant, os.path.join(out_dir, f"{name}_beauty.png"), view="iso", samples=samples,
                                     size=size, background="studio", title=f"{name}: copper on a translucent laminate")
    k = len(info["freqs"]) // 2
    still = Scene([s for s in sc.surfaces if not s.name.startswith("cut_")], axes="z_up", materials=dict(MATERIALS),
                  labels=sc.labels)
    still.animation = Animation(an.times, time_label="f", time_unit="GHz", interpolate=False)
    still.animation.surface_shapes = {"balloon": an.surface_shapes["balloon"]}
    still.animation.surface_fields = {"balloon": an.surface_fields["balloon"]}
    still.animation.hud = list(an.hud)
    W["balloon_png"] = blender.render_still(still, os.path.join(out_dir, f"{name}_balloon.png"), k, size=size,
                                            samples=samples, camera="fixed", background="studio")
    r = blender.render_animation(sc, os.path.join(out_dir, f"{name}_sweep"), size=size, samples=max(8, samples // 2),
                                 camera="fixed", background="studio", fps=1, gif=True, mp4=True, name=f"{name}_sweep")
    W["sweep_gif"], W["sweep_mp4"] = r["gif"], r["mp4"]
    if ph is not None:
        r = blender.render_animation(ph, os.path.join(out_dir, f"{name}_phase"), size=size,
                                     samples=max(8, samples // 2), camera="fixed", background="studio", fps=12,
                                     gif=True, mp4=True, name=f"{name}_phase")
        W["phase_gif"], W["phase_mp4"] = r["gif"], r["mp4"]
    out["render_s"] = round(time.perf_counter() - t0, 1)
    log(f"{name}: Blender renders in {out['render_s']:.0f} s")
    return out
