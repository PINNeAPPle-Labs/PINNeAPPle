"""Planetary rover nodes: vehicle, soil and site presets, the multibody traverse, the wheel-contact models (Wong-Reece
and the model zoo's PINN) and the animated dashboard of ``pinneapple_simulation.vehicle`` (the flow form of
``examples/studio/07_lunar_rover_dashboard.py``)."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import pickle
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact

G_MOON = 1.62
SCENARIOS = ["crater_site", "flat_site"]
FIELDS = ["normal_stress", "shear_stress", "sinkage", "none"]


def _vehicle_available() -> tuple[bool, str]:
    if importlib.util.find_spec("pinneapple_simulation.vehicle") is None:
        return False, "needs pinneapple_simulation.vehicle"
    return True, ""


def _dashboard_available() -> tuple[bool, str]:
    for mod in ("pyvista", "imageio"):
        if importlib.util.find_spec(mod) is None:
            return False, f"needs {mod} (No module named '{mod}')"
    if not os.environ.get("DISPLAY") and not shutil.which("Xvfb"):
        return False, "needs a display for pyvista (set DISPLAY, or install Xvfb)"
    return True, ""


def vehicle_presets() -> dict[str, Any]:
    """Vehicle presets of the library: the functions of ``pinneapple_simulation.vehicle`` that build a VehicleSpec."""
    from pinneapple_simulation import vehicle
    out = {}
    for name in dir(vehicle):
        fn = getattr(vehicle, name)
        if callable(fn) and not name.startswith("_") and \
                str(getattr(fn, "__annotations__", {}).get("return", "")) in ("VehicleSpec", "'VehicleSpec'"):
            out[name] = fn
    return out


def _preset_choices() -> list[str]:
    try:
        return sorted(vehicle_presets()) or ["viper"]
    except Exception:  # noqa: BLE001 - the catalog must load without the simulation package
        return ["viper"]


def _soil_choices() -> list[str]:
    try:
        from pinneapple_simulation.terramechanics import list_soils
        return list(list_soils())
    except Exception:  # noqa: BLE001
        return ["GRC-1"]


SOIL_KEYS = ("c", "phi_deg", "K", "k_c", "k_phi", "n0", "n1", "rho")


# -- building the library objects from the port values -------------------------------------------------------------
def build_spec(v: dict[str, Any]):
    from pinneapple_simulation.terramechanics import Wheel
    presets = vehicle_presets()
    if v.get("preset") not in presets:
        raise StepFailed(f"unknown vehicle preset {v.get('preset')!r} (have {sorted(presets)})")
    spec = presets[v["preset"]](total_mass=float(v["total_mass"]))
    w = spec.wheel
    over = {k: float(v[k]) for k in ("radius", "width") if v.get(k)}
    if over:
        from dataclasses import replace
        wheel = replace(w, **over)
        spec = presets[v["preset"]](wheel=wheel, total_mass=float(v["total_mass"]))
    assert isinstance(spec.wheel, Wheel)
    return spec


def build_soil(v: dict[str, Any]):
    from dataclasses import replace

    from pinneapple_simulation.terramechanics import get_soil
    soil = get_soil(v["preset"])
    over = {k: float(v[k]) for k in SOIL_KEYS if v.get(k) is not None}
    return replace(soil, **over) if over else soil


def build_site(v: dict[str, Any]):
    from pinneapple_simulation.vehicle import scenarios
    if v["scenario"] not in SCENARIOS:
        raise StepFailed(f"unknown scenario {v['scenario']!r}")
    return getattr(scenarios, v["scenario"])(**v.get("args", {}))


def load_run(v: Any) -> dict[str, Any]:
    """The traverse a ``vehicle.simulate`` node wrote (its checksum is verified before it is read)."""
    if not (isinstance(v, dict) and v.get("kind") == "vehicle_run" and v.get("sha256")):
        raise StepFailed("connect the run of a vehicle.simulate node")
    p = Path(v["$artifact"])
    if p.name != "vehicle_run.pkl" or not p.is_file():
        raise StepFailed("the vehicle run artifact is missing (re-run the simulation)")
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != v["sha256"]:
        raise StepFailed("the vehicle run artifact changed after it was written; re-run the simulation")
    return pickle.loads(data)                      # noqa: S301 - written by vehicle.simulate in this run, checksum verified


_DISPLAY_LOCK = threading.Lock()


@contextlib.contextmanager
def virtual_display(size=(1920, 1080)):
    """A display for pyvista: the current one, or an Xvfb started for the duration of the block."""
    if os.environ.get("DISPLAY"):
        yield os.environ["DISPLAY"]
        return
    with _DISPLAY_LOCK:
        num = 90 + os.getpid() % 400
        proc = subprocess.Popen(["Xvfb", f":{num}", "-screen", "0", f"{size[0]}x{size[1]}x24", "-nolisten", "tcp"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.environ["DISPLAY"] = f":{num}"
        try:
            time.sleep(1.0)
            if proc.poll() is not None:
                raise StepFailed("Xvfb did not start")
            yield os.environ["DISPLAY"]
        finally:
            os.environ.pop("DISPLAY", None)
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(5)


# -- nodes ---------------------------------------------------------------------------------------------------------
@node("vehicle.spec", "Rover (vehicle preset)", category="Problem", tags=("section:problem", "rover", "viper"),
      outputs=[Port("vehicle", "vehicle"), Port("metrics", "metrics", required=False)],
      params=[Param("preset", "enum", "viper", choices=_preset_choices(), help="a vehicle of pinneapple_simulation.vehicle"),
              Param("total_mass", "float", 450.0, minimum=1.0, help="kg"),
              Param("wheel_radius", "float", 0.0, minimum=0.0, help="m; 0 keeps the preset's wheel"),
              Param("wheel_width", "float", 0.0, minimum=0.0, help="m; 0 keeps the preset's wheel")],
      calls="pinneapple_simulation.vehicle.viper (vehicle presets) + build_vehicle", available=_vehicle_available)
def vehicle_spec(ctx, inputs, params):
    """A rover from the vehicle library (NASA VIPER: four corner modules, wheel geometry from public NASA figures)
    with its mass and wheel size editable."""
    v = {"preset": params["preset"], "total_mass": params["total_mass"],
         "radius": params["wheel_radius"] or None, "width": params["wheel_width"] or None}
    spec = build_spec(v)
    w = spec.wheel
    ctx.metric("total_mass_kg", float(spec.total_mass))
    ctx.metric("wheel_radius_m", float(w.radius))
    ctx.metric("wheel_width_m", float(w.width))
    ctx.metric("wheels", len(spec.corners))
    ctx.summary(f"{spec.name}: {spec.total_mass:.0f} kg, {len(spec.corners)} wheels R {w.radius:.3f} m, "
                f"b {w.width:.3f} m")
    return {"vehicle": v, "metrics": dict(ctx.report["metrics"])}


@node("terramechanics.soil", "Soil (terramechanics)", category="Problem", tags=("section:problem", "regolith", "bekker"),
      outputs=[Port("soil", "soil")],
      params=[Param("preset", "enum", "GRC-1", choices=_soil_choices(), help="soil of the terramechanics database"),
              *[Param(k, "float", None, help="override the preset (blank keeps it)", advanced=True) for k in SOIL_KEYS]],
      calls="pinneapple_simulation.terramechanics.get_soil", available=_vehicle_available)
def soil(ctx, inputs, params):
    """A soil of the terramechanics database (Bekker / Wong-Reece parameters with their sources); any parameter can
    be overridden."""
    v = {"preset": params["preset"], **{k: params.get(k) for k in SOIL_KEYS if params.get(k) is not None}}
    s = build_soil(v)
    ctx.table(f"soil {s.name}", ["parameter", "value"], [[k, getattr(s, k)] for k in SOIL_KEYS])
    ctx.summary(f"{s.name}: c {s.c:g} Pa, phi {s.phi_deg:g} deg, k_c {s.k_c:g}, k_phi {s.k_phi:g}, n {s.n0:g}")
    return {"soil": v}


@node("scenario.site", "Terrain scenario", category="Problem", tags=("section:problem", "crater", "terrain"),
      outputs=[Port("site", "site")],
      params=[Param("scenario", "enum", "crater_site", choices=SCENARIOS),
              Param("depth_ratio", "float", 0.08, minimum=0.0, maximum=0.5, help="crater depth / diameter (crater_site)"),
              Param("scenery_craters", "int", 60, minimum=0, maximum=400, help="background craters (crater_site)"),
              Param("scenery_rocks", "int", 5, minimum=0, maximum=100, help="background rocks (crater_site)"),
              Param("slope_deg", "float", 0.0, minimum=-30.0, maximum=30.0, help="ground slope (flat_site)"),
              Param("cell", "float", 0.01, minimum=0.005, maximum=0.1, help="terrain cell [m]", advanced=True),
              Param("seed", "int", 5, help="scenery seed", advanced=True)],
      calls="pinneapple_simulation.vehicle.scenarios.crater_site / flat_site", available=_vehicle_available)
def site(ctx, inputs, params):
    """A deformable terrain: a crater with its rim and a rock on the path, scenery craters and rocks out to the lunar
    horizon (crater_site), or a plain slope (flat_site)."""
    if params["scenario"] == "crater_site":
        args = {"depth_ratio": params["depth_ratio"], "scenery_craters": params["scenery_craters"],
                "scenery_rocks": params["scenery_rocks"], "cell": params["cell"], "seed": params["seed"]}
    else:
        args = {"slope_deg": params["slope_deg"], "cell": params["cell"], "seed": params["seed"]}
    defaults = {"crater_site": {"seed": 5}, "flat_site": {"seed": 2}}[params["scenario"]]
    args = {k: v for k, v in args.items() if not (k == "seed" and v == defaults["seed"])}
    t = build_site({"scenario": params["scenario"], "args": args})
    ctx.summary(f"{params['scenario']}: {t.h.shape[1]} x {t.h.shape[0]} cells" if hasattr(t, "h") else params["scenario"])
    return {"site": {"scenario": params["scenario"], "args": args}}


@node("vehicle.simulate", "Rover traverse (multibody + terramechanics)", category="Solvers",
      tags=("section:solvers", "rover", "multibody", "slip", "sinkage"),
      inputs=[Port("vehicle", "vehicle"), Port("soil", "soil"), Port("site", "site")],
      outputs=[Port("run", "vehicle_run"), Port("metrics", "metrics")],
      params=[Param("duration", "float", 45.0, minimum=1.0, maximum=600.0, help="s (12 s for a quick run)"),
              Param("speed", "float", 0.2, minimum=0.01, maximum=2.0, help="commanded speed [m/s]"),
              Param("gravity", "float", G_MOON, minimum=0.1, maximum=30.0, help="m/s^2", advanced=True),
              Param("dt", "float", 0.002, minimum=1e-4, maximum=0.01, advanced=True),
              Param("output_dt", "float", 0.05, minimum=0.005, maximum=1.0, advanced=True)],
      calls="pinneapple_simulation.vehicle.VehicleSimulator(build_vehicle(spec), site, soil).run + VehicleRun.summary",
      available=_vehicle_available)
def simulate(ctx, inputs, params):
    """Drive the rover across the site: multibody dynamics with Wong-Reece wheels (slip-sinkage, multipass, ruts
    stamped into the deformable terrain); the run's summary() as metrics."""
    from pinneapple_simulation.vehicle import VehicleSimulator, build_vehicle
    spec, sl, terrain = build_spec(inputs["vehicle"]), build_soil(inputs["soil"]), build_site(inputs["site"])
    t0 = time.perf_counter()
    sim = VehicleSimulator(build_vehicle(spec, gravity=params["gravity"]), terrain, sl, g=params["gravity"])
    run = sim.run(params["duration"], speed=params["speed"], dt=params["dt"], output_dt=params["output_dt"],
                  progress=lambda *_: ctx.check_stop())
    secs = time.perf_counter() - t0
    summ = run.summary()
    for k, v in summ.items():
        ctx.metric(k, float(v), "pinneapple_simulation.vehicle.VehicleRun.summary()")
    ctx.metric("sim_seconds", round(secs, 1), "wall time of VehicleSimulator.run on this machine")
    p = Path(ctx.workdir) / "vehicle_run.pkl"
    data = pickle.dumps({"run": run, "site": terrain, "spec": spec, "soil": sl, "inputs": inputs, "params": params},
                        protocol=pickle.HIGHEST_PROTOCOL)
    p.write_bytes(data)
    out = {**artifact(p, "vehicle_run"), "sha256": hashlib.sha256(data).hexdigest()}
    ctx.table("run summary", ["quantity", "value"], [[k, round(float(v), 4)] for k, v in summ.items()])
    ctx.summary(f"{summ['distance_m']:.2f} m in {summ['duration_s']:.0f} s, mean slip {summ['mean_slip']:.3f}, "
                f"max sinkage {summ['max_sinkage_mm']:.1f} mm, {summ['energy_per_metre_J_m']:.0f} J/m "
                f"({secs:.0f} s)")
    return {"run": out, "metrics": {k: float(v) for k, v in summ.items()}}


@node("terramechanics.wheel_model", "Wheel-soil contact (Wong-Reece)", category="Solvers",
      tags=("section:solvers", "wheel", "drawbar pull", "terramechanics"),
      inputs=[Port("vehicle", "vehicle"), Port("soil", "soil")],
      outputs=[Port("model", "model"), Port("metrics", "metrics", required=False)],
      params=[Param("bulldozing", "enum", "lateral", choices=["lateral", "none"], advanced=True),
              Param("slip_sinkage", "enum", "drive", choices=["drive", "none"], advanced=True),
              Param("n_gl", "int", 24, minimum=4, maximum=96, advanced=True)],
      calls="pinneapple_simulation.terramechanics.WongReeceModel + steady_state", available=_vehicle_available)
def wheel_model(ctx, inputs, params):
    """The Wong-Reece wheel model of the run (stress distribution under a rigid grousered wheel); its single-wheel
    testbed curves at the static wheel load."""
    from pinneapple_simulation.terramechanics import WongReeceModel
    spec, sl = build_spec(inputs["vehicle"]), build_soil(inputs["soil"])
    m = WongReeceModel(sl, spec.wheel, bulldozing=params["bulldozing"], slip_sinkage=params["slip_sinkage"],
                       n_gl=params["n_gl"])
    load = spec.total_mass * G_MOON / max(len(spec.corners), 1)
    rows = []
    for s in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7):
        r = m.steady_state(load, s)
        rows.append([s, *(round(float(r.get(k, np.nan)), 4) for k in ("sinkage", "drawbar_pull", "torque"))])
    ctx.table(f"testbed at the static load {load:.0f} N", ["slip", "sinkage [m]", "drawbar pull [N]", "torque [N m]"],
              rows)
    ctx.metric("static_wheel_load_N", round(load, 2))
    ctx.metric("drawbar_pull_at_slip_0.2_N", rows[2][2], "WongReeceModel.steady_state(load, 0.2)")
    ctx.summary(f"Wong-Reece on {sl.name}: drawbar pull {rows[2][2]:.1f} N at 20 % slip, static load {load:.0f} N")
    v = {"kind": "wong_reece", "vehicle": inputs["vehicle"], "soil": inputs["soil"],
         "physics": m.physics_config(), "options": dict(params)}
    return {"model": v, "metrics": dict(ctx.report["metrics"])}


def build_wheel_model(v: dict[str, Any]):
    from pinneapple_simulation.terramechanics import WongReeceModel
    o = v.get("options", {})
    return WongReeceModel(build_soil(v["soil"]), build_spec(v["vehicle"]).wheel, bulldozing=o.get("bulldozing", "lateral"),
                          slip_sinkage=o.get("slip_sinkage", "drive"), n_gl=int(o.get("n_gl", 24)))


@node("terramechanics.zoo_surrogate", "Wheel-contact PINN (model zoo)", category="Learning",
      tags=("section:physics_ai", "pinn", "zoo", "wheel", "surrogate"),
      inputs=[Port("model", "model", help="the Wong-Reece model it must reproduce (physics check)")],
      outputs=[Port("surrogate", "model"), Port("metrics", "metrics")],
      params=[Param("zoo_name", "str", "rover_contact_grc1_viper", help="model zoo entry")],
      calls="pinneapple_hub.zoo.load + pinneapple_simulation.terramechanics.surrogate.ZooContactSurrogate "
            "(check_physics)", available=_vehicle_available)
def zoo_surrogate(ctx, inputs, params):
    """The pre-trained wheel-contact PINN of the model zoo (downloaded once, checksum-verified), with its held-out
    errors from the model card and a check that it was trained on the same physics as the run's wheel model."""
    from pinneapple_hub import zoo
    from pinneapple_simulation.terramechanics.surrogate import ZooContactSurrogate
    if not (isinstance(inputs["model"], dict) and inputs["model"].get("kind") == "wong_reece"):
        raise StepFailed("connect a terramechanics.wheel_model node")
    sur = ZooContactSurrogate(zoo.load(params["zoo_name"]))
    diff = sur.check_physics(build_wheel_model(inputs["model"]), warn=False)
    errs = sur.meta.get("test_errors") or {}
    rows = []
    for comp, e in errs.items():
        for k, val in e.items():
            ctx.metric(f"heldout_{comp}_{k}", float(val), "held-out error from the zoo model card")
            rows.append([comp, k, float(val)])
    ctx.table(f"{sur.meta.get('name')}: held-out errors (model card)", ["output", "error", "value"], rows)
    if diff:
        ctx.warn(f"trained on different physics than the run's wheel model: {'; '.join(diff)}")
    ok = not diff
    ctx.metric("same_physics", ok, "ZooContactSurrogate.check_physics(WongReeceModel) is empty")
    ctx.summary(f"{sur.meta.get('name')}: same physics as the run {'yes' if ok else 'NO (' + '; '.join(diff) + ')'}")
    checks = [{"name": "zoo_pinn_same_physics", "passed": ok, "kind": "physics", "value": "; ".join(diff) or "same",
               "detail": "ZooContactSurrogate.check_physics against the run's Wong-Reece model"}]
    return {"surrogate": {"kind": "zoo_contact", "zoo": params["zoo_name"], "name": sur.meta.get("name")},
            "metrics": {**{f"heldout_{c}_{k}": float(v) for c, e in errs.items() for k, v in e.items()},
                        "same_physics": ok, "checks": checks}}


@node("vehicle.dashboard", "Rover dashboard (animation + charts)", category="Visualization",
      tags=("section:viz", "dashboard", "animation", "video", "rover"),
      inputs=[Port("run", "vehicle_run"), Port("model", "model", required=False, help="the run's wheel model"),
              Port("surrogate", "model", required=False, help="the zoo PINN, compared with the wheel model")],
      outputs=[Port("report", "report"), Port("figure", "figure"), Port("metrics", "metrics")],
      params=[Param("frames", "int", 150, minimum=4, maximum=900, help="animation frames (24 for a quick run)"),
              Param("width", "int", 1280, minimum=320, maximum=3840),
              Param("height", "int", 720, minimum=240, maximum=2160),
              Param("field", "enum", "normal_stress", choices=FIELDS, help="what colours the ruts"),
              Param("clim_low", "float", 5.0, minimum=0.0, maximum=100.0, help="colour range: low percentile"),
              Param("clim_high", "float", 95.0, minimum=0.0, maximum=100.0, help="colour range: high percentile"),
              Param("field_alpha", "float", 0.45, minimum=0.0, maximum=1.0, advanced=True),
              Param("field_glow", "float", 0.05, minimum=0.0, maximum=1.0, advanced=True),
              Param("field_saturation", "float", 0.6, minimum=0.0, maximum=1.0, advanced=True),
              Param("sun_azimuth_deg", "float", 160.0, advanced=True),
              Param("title", "str", "NASA VIPER crossing a lunar crater"),
              Param("subtitle", "str", "Multibody dynamics, Wong-Reece wheels with slip-sinkage on GRC-1 (lunar g); "
                    "3-D ruts coloured by normal stress; zoo PINN surrogate; low Sun (12 deg)"),
              Param("footer", "str", "PINNeAPPle: pinneapple_simulation.vehicle, .terramechanics; "
                    "pinneapple_tools.visualization.dashboard. VIPER geometry from public NASA figures (estimates "
                    "listed in VIPER_SOURCES).", advanced=True),
              Param("video", "enum", "mp4+gif", choices=["none", "mp4", "gif", "mp4+gif"]),
              Param("fps", "int", 12, minimum=1, maximum=60, advanced=True),
              Param("gif_every", "int", 2, minimum=1, maximum=10, help="every n-th frame in the GIF (1 for quick)",
                    advanced=True),
              Param("horizon_check", "bool", True, help="check that no edge of the ground mesh shows in any frame")],
      calls="pinneapple_simulation.vehicle.dashboard.render_frames + vehicle_dashboard (to_html / to_png / "
            "to_video); pinneapple_simulation.vehicle.viz.frame_view + follow_camera; "
            "pinneapple_tools.visualization.studio.planetary.border_visibility",
      available=_dashboard_available)
def dashboard(ctx, inputs, params):
    """The animation-centred dashboard: the planetary render of the traverse (ruts coloured by the field, low Sun,
    lunar horizon) with charts synced to its time cursor (testbed curves, sinkage and load per wheel, speed and
    slip, the zoo PINN against the run's Wong-Reece model), KPI tiles and the parameters; interactive HTML, PNG,
    MP4 / GIF and one rendered frame."""
    from PIL import Image

    from pinneapple_simulation.vehicle import dashboard as vdash
    from pinneapple_simulation.vehicle import viz
    from pinneapple_tools.visualization.studio.planetary import border_visibility
    d = load_run(inputs["run"])
    run, terrain = d["run"], d["site"]
    wr = build_wheel_model(inputs["model"]) if isinstance(inputs.get("model"), dict) and \
        inputs["model"].get("kind") == "wong_reece" else None
    sur = None
    if isinstance(inputs.get("surrogate"), dict) and inputs["surrogate"].get("kind") == "zoo_contact":
        from pinneapple_hub import zoo
        from pinneapple_simulation.terramechanics.surrogate import ZooContactSurrogate
        sur = ZooContactSurrogate(zoo.load(inputs["surrogate"]["zoo"]))
    size = (params["width"], params["height"])
    field = None if params["field"] == "none" else params["field"]
    look = dict(field_alpha=params["field_alpha"], field_glow=params["field_glow"],
                field_saturation=params["field_saturation"])
    out = Path(ctx.workdir)
    t0 = time.perf_counter()
    checks = []
    with virtual_display((max(1920, size[0]), max(1080, size[1]))):
        frames, ks, clim = vdash.render_frames(
            run, terrain, n=params["frames"], size=size, field=field,
            clim_percentiles=(params["clim_low"], params["clim_high"]), sun_azimuth_deg=params["sun_azimuth_deg"],
            progress=lambda j, n: (ctx.check_stop(), ctx.log(f"frame {j}/{n}") if j % 25 == 0 else None), **look)
        ctx.metric("render_seconds", round(time.perf_counter() - t0, 1), "wall time of render_frames on this machine")
        if params["horizon_check"]:
            seen = []
            for k in ks[:: max(1, len(ks) // 10)]:
                view = viz.frame_view(run, terrain, int(k), field=field or "normal_stress", clim=clim,
                                      sun_azimuth_deg=params["sun_azimuth_deg"], half_window=(3.2, 3.4), **look)
                vis = border_visibility(view, viz.follow_camera(run, int(k)), size)
                seen.append([int(k), int(vis["visible"])])
            bad = [k for k, v in seen if v]
            ctx.table("ground border visibility (sampled frames)", ["sample", "visible border pixels"], seen)
            checks.append({"name": "horizon_no_ground_border", "passed": not bad, "kind": "visual",
                           "value": f"{len(seen) - len(bad)}/{len(seen)} frames clean",
                           "detail": "planetary.border_visibility of the ground mesh in every 10th of the frames"})
            ctx.metric("border_frames_visible", len(bad), "frames where an edge of the ground mesh shows")
        frame_png = out / "rover-frame.png"
        Image.fromarray(frames[len(frames) * 2 // 3]).save(frame_png)
        db = vdash.vehicle_dashboard(
            run, frames, ks, model=wr, surrogate=sur, clim=clim, scene3d=True, terrain=terrain,
            field_label="Peak normal stress in the ruts [kPa]" if params["field"] == "normal_stress" else params["field"],
            field_saturation=params["field_saturation"], title=params["title"], subtitle=params["subtitle"],
            footer=params["footer"])
        html, png = out / "rover-dashboard.html", out / "rover-dashboard.png"
        db.to_html(str(html), max_width=1024, quality=30)
        db.to_png(str(png), k=int(len(ks) * 0.62))
        mp4 = gif = None
        if params["video"] != "none":
            mp4 = out / "rover-dashboard.mp4" if "mp4" in params["video"] else None
            gif = out / "rover-dashboard.gif" if "gif" in params["video"] else None
            db.to_video(gif=str(gif) if gif else None, mp4=str(mp4) if mp4 else None, fps=params["fps"],
                        size=(1920, 1080), gif_size=(1280, 720), gif_font_scale=1.0, gif_every=params["gif_every"])
    ctx.metric("dashboard_seconds", round(time.perf_counter() - t0, 1), "render, dashboard and videos on this machine")
    ctx.metric("frames", len(frames))
    ctx.metric("clim_kPa", f"{clim[0]:.1f} to {clim[1]:.1f}" if clim is not None else "")
    files = {"html": html, "png": png, "frame": frame_png, "mp4": mp4, "gif": gif}
    for k, p in files.items():
        if p is not None and p.is_file():
            ctx.metric(f"{k}_MB", round(p.stat().st_size / 1e6, 2), "file size")
    ctx.view("html", {"path": str(html.resolve()), "title": params["title"],
                      "links": {k: str(p.resolve()) for k, p in files.items() if p is not None and p.is_file()}})
    if mp4 is not None and mp4.is_file():
        ctx.view("video", {"path": str(mp4.resolve()), "poster": str(png.resolve()), "caption": params["title"]})
    fig = {**artifact(png, "png")}
    ctx.report["figures"].append({"path": str(png.resolve()), "name": "dashboard", "caption": params["title"]})
    ok = all(c["passed"] for c in checks)
    ctx.summary(f"dashboard: {len(frames)} frames, " + (f"horizon check {'passed' if ok else 'FAILED'}, "
                                                          if checks else "") + ", ".join(
        k for k, p in files.items() if p is not None and p.is_file()))
    return {"report": artifact(html, "html"), "figure": fig,
            "metrics": {"frames": len(frames), "checks": checks}}
