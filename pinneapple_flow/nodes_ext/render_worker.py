"""Blender rendering in its own process, for :mod:`pinneapple_flow.nodes_ext.render`: ``bpy`` is never imported by
the flow engine or the Flow Studio server (a ``bpy`` wheel built against another numpy can crash the process that
imports it, and Cycles holds the interpreter for minutes).

    python -m pinneapple_flow.nodes_ext.render_worker scenes.pkl options.json out_dir

``scenes.pkl`` is the ``anim_scene`` artifact of a scene node; ``options.json`` holds the node's parameters and each
scene's render preset; the clips are listed in ``out_dir/clips.json``. ``--check`` only checks that bpy imports.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np


def check() -> int:
    import bpy

    from pinneapple_tools.visualization.studio.blender import _check_bpy
    _check_bpy(bpy)
    return 0


def render(scenes_pkl: str, options_json: str, out_dir: str) -> int:
    from pinneapple_tools.visualization.studio.blender import render_animation, render_still
    with open(scenes_pkl, "rb") as f:
        scenes = pickle.load(f)  # noqa: S301 - the artifact a scene node of this run wrote
    o = json.loads(Path(options_json).read_text())
    out = Path(out_dir)
    size = (int(o["width"]), int(o["height"]))
    clips = []
    for e in scenes:
        key, sc0 = e["key"], e["scene"]
        if o["only"] and key not in o["only"]:
            continue
        opts = o["presets"][key]
        print(f"render {key}: {opts}", flush=True)
        r = render_animation(sc0, str(out / f"frames_{key}"), frames=int(o["frames"]), size=size,
                             samples=int(o["samples"]), fps=int(o["fps"]), mp4=True, name=key,
                             log=lambda m: print(m, flush=True), **opts)
        frames = list(r.get("frames") or [])
        # the scene was resampled to `frames` frames: the marked time (35 ft screen, gust peak) on that clock
        times = np.asarray(sc0.animation.times, float)
        tk = np.linspace(times[0], times[-1], len(frames)) if frames else times
        mark = e.get("t_mark")
        k = int(np.argmin(np.abs(tk - mark))) if mark is not None else int(len(tk) * 2 // 3)
        still = None
        if o["still"]:
            kk = int(np.argmin(np.abs(times - tk[min(k, len(tk) - 1)])))
            still = render_still(sc0, str(out / f"{key}_still.png"), kk,
                                 size=(int(o["still_width"]), int(o["still_height"])),
                                 samples=int(o["still_samples"]), **opts)
        clips.append({"key": key, "mp4": r.get("mp4"), "still": still,
                      "mark_frame": frames[min(k, len(frames) - 1)] if frames else None, "title": e.get("title")})
    (out / "clips.json").write_text(json.dumps(clips, indent=1, default=str))
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--check"]:
        return check()
    return render(*argv[:3])


if __name__ == "__main__":
    import os
    code = main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(code)                      # bpy can crash while tearing down
