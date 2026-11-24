"""A repeatable visual benchmark: the same scenes, cameras and lights every run, so a change to the renderer can be
judged instead of described.

    python -m pinneapple_tools.visualization.studio.benchmark --out provas/ --quality preview

Each case pairs a model from the geometry library with the materials, the HDRI and the camera a figure of it would
use. A run writes the PNGs and a JSON record per case with the time, the device and four statistics of the picture:
mean luminance, its spread, the fraction of the frame the subject covers and the edge energy. Those catch what a
regression actually looks like -- a render that went black, blew out, lost its micro-detail or stopped showing the
subject -- without the brittleness of comparing pixels to a golden image across GPUs, drivers and Blender versions.

Pass two records to :func:`compare` (or `--baseline`) to see the deltas; the numbers are the point, not a pass/fail.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

#: The configuration the gate compares in: one size, one sample count, one preset, so the record of today can be
#: compared with the record of last month. Small enough to run on a CPU in CI in a few minutes.
GATE = {"quality": "preview", "size": [960, 540], "samples": 32}

#: How far a statistic may move before the gate calls it a regression. Absolute: the four statistics are already in
#: their own units (luminance 0-255, coverage 0-1, edge energy a mean gradient).
TOLERANCES: Dict[str, float] = {"mean": 8.0, "std": 4.0, "coverage": 0.06, "edge_energy": 0.12}

#: Where the committed baseline lives, relative to the repository root.
BASELINE = os.path.join("data", "benchmark", "baseline.json")

#: name -> (geometry in ``data/geometries``, its materials, background and camera). The values are the ones a figure
#: of that part would use, so the benchmark measures the look the product actually ships.
CASES: Dict[str, dict] = {
    "impeller_bronze": {
        "geometry": "centrifugal_impeller",
        "materials": {"cast_bronze": {"color": [0.72, 0.45, 0.24], "metallic": 1.0, "roughness": 0.34,
                                      "shader": "cast_metal"},
                      "cast_bronze_blade": {"color": [0.66, 0.40, 0.21], "metallic": 1.0, "roughness": 0.42,
                                            "shader": "cast_metal"}},
        "render": {"background": "hdri:machine_shop_02", "hdri_strength": 2.0, "hdri_rotation": 120.0,
                   "floor": True, "camera": "product"},
    },
    "heatsink_aluminium": {
        "geometry": "pin_fin_heatsink",
        "materials": {"heater_copper": {"color": [0.85, 0.50, 0.32], "metallic": 1.0, "roughness": 0.3,
                                        "shader": "cast_metal"},
                      "heatsink_al": {"color": [0.80, 0.81, 0.83], "metallic": 1.0, "roughness": 0.26,
                                      "shader": "brushed_metal"}},
        "render": {"background": "hdri:studio_small_09", "hdri_strength": 1.2, "hdri_rotation": 60.0,
                   "floor": True, "camera": "hero"},
    },
    "airliner_library": {
        "geometry": "airliner_a320",
        "materials": {"paint": {"color": [0.93, 0.94, 0.95], "metallic": 0.0, "roughness": 0.45,
                                "shader": "painted_metal", "wear": 0.06},
                      "livery": {"color": [0.72, 0.10, 0.12], "metallic": 0.0, "roughness": 0.45,
                                 "shader": "painted_metal", "wear": 0.05},
                      "livery2": {"color": [0.10, 0.20, 0.52], "metallic": 0.0, "roughness": 0.45,
                                  "shader": "painted_metal", "wear": 0.05},
                      "belly": {"color": [0.80, 0.81, 0.83], "metallic": 1.0, "roughness": 0.3},
                      "wingmetal": {"color": [0.72, 0.73, 0.74], "metallic": 1.0, "roughness": 0.22,
                                    "shader": "brushed_metal"},
                      "nacelle": {"color": [0.80, 0.82, 0.84], "metallic": 1.0, "roughness": 0.08,
                                  "shader": "brushed_metal"},
                      "exhaust": {"color": [0.42, 0.43, 0.44], "metallic": 0.9, "roughness": 0.45,
                                  "shader": "oxidized_metal"},
                      "fan": {"color": [0.35, 0.36, 0.38], "metallic": 1.0, "roughness": 0.22},
                      "window": {"color": [0.02, 0.035, 0.05], "metallic": 0.0, "roughness": 0.03,
                                 "transmission": 0.9, "clearcoat": 1.0}},
        "render": {"background": "hdri:empty_warehouse_01", "hdri_strength": 1.2, "hdri_rotation": 30.0,
                   "floor": True, "camera": "documentary"},
    },
}


def repository_root() -> str:
    """The checkout the geometry library and the HDRIs live in."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def build(name: str):
    """The scene of one case, with its materials: ready for :func:`~.blender.render_headless`."""
    from .scene import Scene

    case = CASES[name]
    path = os.path.join(repository_root(), "data", "geometries", case["geometry"],
                        f"{case['geometry']}.light.glb")
    sc = Scene.from_file(path)
    sc.materials = {k: dict(v) for k, v in case["materials"].items()}
    sc.title = name
    return sc


def picture_stats(path: str) -> dict:
    """What the picture looks like, as numbers: mean luminance and its spread, the share of the frame the subject
    covers (pixels that differ from the top-left corner by more than a hair) and the edge energy (mean gradient, which
    falls when micro-detail disappears)."""
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(path).convert("RGB"), float)
    lum = a @ np.array([0.2126, 0.7152, 0.0722])
    bg = lum[0, 0]
    coverage = float(np.mean(np.abs(lum - bg) > 6.0))
    gy, gx = np.gradient(lum)
    return {"mean": round(float(lum.mean()), 2), "std": round(float(lum.std()), 2),
            "coverage": round(coverage, 4), "edge_energy": round(float(np.hypot(gx, gy).mean()), 3),
            "size": [int(a.shape[1]), int(a.shape[0])]}


def run(out_dir: str, *, scenes: Optional[list] = None, quality: str = "preview", device: str = "auto",
        samples: Optional[int] = None, size: Optional[tuple] = None, gate: bool = False,
        log=None) -> dict:
    """Render every case into ``out_dir`` and return the record: ``{case: {seconds, device, picture..., png}}``.

    ``gate=True`` renders the canonical :data:`GATE` configuration instead of the arguments, so the numbers are the
    ones :data:`BASELINE` was recorded with -- a comparison has to be an apples-to-apples render, not whatever preset
    the caller happened to use."""
    from .blender import render_headless

    if gate:
        quality, size, samples = GATE["quality"], tuple(GATE["size"]), GATE["samples"]
    say = log or print
    os.makedirs(out_dir, exist_ok=True)
    record: dict = {}
    for name in (scenes or list(CASES)):
        sc = build(name)
        case_render = dict(CASES[name]["render"])
        t0 = time.time()
        png = os.path.join(os.path.abspath(out_dir), f"{name}.png")
        render_headless(sc, png, quality=quality, device=device, colorbar_on=False,
                        samples=samples, size=size, **case_render)
        secs = time.time() - t0
        stats = picture_stats(png)
        record[name] = {"seconds": round(secs, 1), "device": device, "quality": quality, "png": png, **stats}
        say(f"{name}: {secs:.1f} s, mean {stats['mean']}, coverage {stats['coverage']}, "
            f"edge {stats['edge_energy']}")
    with open(os.path.join(os.path.abspath(out_dir), "benchmark.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    return record


def baseline_of(record: dict, *, reason: str, blender: Optional[str] = None) -> dict:
    """What a record keeps when it is committed as the reference: the picture statistics and the configuration they
    were measured with, plus a ``_meta`` line saying when, with what and **why** -- a gate whose reference has no date
    and no reason is a number nobody can judge.

    ``reason`` is required on purpose: a baseline moves because somebody decided the pictures should look different,
    and that decision is the one piece of the record a machine cannot supply.

    The time and the absolute path stay out: one is the machine's, the other is a folder that does not exist anywhere
    else."""
    reason = (reason or "").strip()
    if not reason:
        raise ValueError("a baseline records why the pictures moved: pass reason= (the command line takes --why)")
    out = {name: {k: v for k, v in case.items() if k in ("quality", "size", "mean", "std", "coverage",
                                                         "edge_energy")}
           for name, case in record.items()}
    out["_meta"] = {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "reason": reason, "gate": dict(GATE), "tolerances": dict(TOLERANCES),
                    "blender": blender if blender is not None else _blender_version()}
    return out


def _blender_version() -> str:
    """The renderer's version, so a baseline records what drew it (``"unknown"`` when there is no binary here)."""
    from .blender import blender_executable

    exe = blender_executable()
    if not exe:
        return "unknown"
    try:
        import subprocess
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=120).stdout
        return (out.splitlines() or ["unknown"])[0].strip() or "unknown"
    except Exception:                      # noqa: BLE001 -- a version string must never break a recording
        return "unknown"


def cases_of(record: dict) -> List[str]:
    """The case names in a record or a baseline: everything but the bookkeeping keys, which start with ``_``."""
    return [name for name in record if not name.startswith("_")]


def gate(current: dict, baseline: dict, *, tolerances: Optional[dict] = None) -> List[str]:
    """The differences between a record and the committed baseline that are too large to be noise.

    Absolute tolerances, chosen to pass a CPU render on another machine and to fail a regression: a picture that went
    black, blew out, lost its micro-detail or stopped showing the subject moves these by far more than they allow.
    The time is not compared -- it is the machine's, not the renderer's."""
    tol = dict(TOLERANCES if tolerances is None else tolerances)
    problems: List[str] = []
    for name in sorted(CASES):
        if name not in current:
            problems.append(f"{name}: not rendered")
            continue
        if name not in baseline:
            problems.append(f"{name}: no baseline to compare against")
            continue
        cur, base = current[name], baseline[name]
        for key, allowed in tol.items():
            if key not in cur or key not in base:
                continue
            delta = abs(float(cur[key]) - float(base[key]))
            if delta > allowed:
                problems.append(f"{name}: {key} moved {delta:+.3f} ({base[key]} -> {cur[key]}), "
                                f"allowed {allowed:+.3f}")
    return problems


def compare(current: dict, baseline: dict) -> dict:
    """The deltas of one record against another, per case and per statistic (``current - baseline``)."""
    out: dict = {}
    for name, cur in current.items():
        base = baseline.get(name)
        if not base:
            continue
        out[name] = {k: round(float(cur[k]) - float(base[k]), 3) for k in ("mean", "std", "coverage", "edge_energy",
                                                                          "seconds") if k in cur and k in base}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render the reference scenes and record how they look and how long they "
                                             "take.")
    ap.add_argument("--out", default="benchmark", help="where the PNGs and benchmark.json go")
    ap.add_argument("--scenes", nargs="*", default=None, help=f"any of: {', '.join(CASES)}")
    ap.add_argument("--quality", default="preview", choices=["preview", "production", "cinematic"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--baseline", default=None, help="a previous benchmark.json to compare against")
    ap.add_argument("--gate", action="store_true",
                    help=f"render the canonical configuration ({GATE['size'][0]}x{GATE['size'][1]}, "
                         f"{GATE['samples']} samples) and fail when a statistic leaves its tolerance")
    ap.add_argument("--record-baseline", action="store_true",
                    help=f"write the run to {BASELINE} in the shape the gate compares against (needs --why)")
    ap.add_argument("--why", default="", help="why the pictures are meant to move; goes into the baseline itself")
    a = ap.parse_args(argv)
    if a.record_baseline and not a.why.strip():
        ap.error("--record-baseline needs --why: a new baseline is a decision, and the record keeps the reason")
    rec = run(a.out, scenes=a.scenes, quality=a.quality, device=a.device, gate=a.gate)
    if a.record_baseline:
        path = os.path.join(repository_root(), BASELINE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(baseline_of(rec, reason=a.why), fh, indent=1, sort_keys=True)
        print(f"wrote {path}: {a.why.strip()}")
    if a.baseline:
        with open(a.baseline, encoding="utf-8") as fh:
            base = json.load(fh)
        meta = base.get("_meta", {})
        if meta:
            say = f"baseline: {meta.get('recorded_at', '?')}, {meta.get('blender', '?')}"
            print(f"{say} -- {meta.get('reason', 'no reason recorded')}")
        print(json.dumps(compare(rec, base), indent=1))
        if a.gate:
            problems = gate(rec, base)
            for p in problems:
                print(f"regression: {p}")
            if problems:
                print("If the pictures were meant to move, record a new baseline on purpose and say why: "
                      "--record-baseline --why \"...\"")
                return 1
            print(f"gate ok: every statistic within {TOLERANCES}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
