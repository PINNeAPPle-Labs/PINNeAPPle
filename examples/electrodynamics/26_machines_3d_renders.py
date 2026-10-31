"""Studio renders of the flux density in 3-D machines: stills, cut views and turntables (Blender Cycles).

Three machines, each solved once at no load with the edge-element solvers of ``machines3d``:

* ``afpm_single`` -- the Axial-Flux Motor Designer's default design (12 slots, 8 poles, single-sided) on its grid,
  rotor at the tooth flux-linkage angle (a magnet centred on a tooth);
* ``afpm_double`` -- the same machine with two rotors around a yokeless stator;
* ``spm_skew`` -- the radial-flux SPM of ``20_spm_skew_3d.py`` with a 10 deg step skew in six axial steps and
  20 mm of end region at each end (360 steps round the axis instead of 720: one solve, for the picture).

For each: the iron and the magnets coloured by ``|B|`` (``machines3d.renders``: every face carries the ``|B|`` of
the tetrahedron it bounds, no averaging), a full still and a cut still (a 90 deg sector removed towards the camera)
at 1920 x 1080, the cut scene as glTF for the web viewer; a 3840 x 2160 still of ``afpm_single``'s cut view; an
8 s turntable of the cut view of ``afpm_single`` and ``spm_skew`` (1280 x 720, 24 fps). Colours run from the
minimum ``|B|`` on the faces shown to their 99.5th percentile; the colour bar states the maximum when it is above.
Every range, size and render time goes to ``results/26_machines_3d_renders.json``; assets already rendered are
kept (the script resumes).

    pip install bpy        # Blender as a Python module (5.2 here)
    python examples/electrodynamics/26_machines_3d_renders.py        # about 1 h on 4 cores
"""
from __future__ import annotations

import json
import math
import os
import platform
import resource
import time
from pathlib import Path

OUT = Path(__file__).parent / "results" / "26_renders"
LOG = Path(__file__).parent / "results" / "26_machines_3d_renders.json"
CUT = (183.0, 273.0)          # the sector removed: centred on the azimuth the "iso" camera looks from (228 deg)
CLIP = 99.5
STILL = (1920, 1080)
HERO = (3840, 2160)
TURNTABLE = {"size": (1280, 720), "seconds": 8.0, "fps": 24.0, "samples": 16, "lens": 50}
SAMPLES = 64


def solve(key: str):
    """(mesh, labels, B, facts) of one machine at no load."""
    from pinneapple_simulation.electromagnetics.machines3d import renders as R

    t0 = time.perf_counter()
    if key.startswith("afpm"):
        from pinneapple_simulation.electromagnetics.machines3d.afpm import AFPM3D
        from pinneapple_simulation.electromagnetics.machines3d.afpm_design import (
            grid_for,
            machine_from,
        )

        m = machine_from({"topology": "single" if key == "afpm_single" else "double_rotor"})
        model = AFPM3D(m, air_r=0.015, air_z=0.015, **grid_for(m))
        theta = math.pi / m.slots
        sol = model.solve(theta)
        nu, Br = model.materials(theta)
        facts = {"machine": f"AFPM, {m.slots} slots, {m.poles} poles, {m.topology}", "theta_deg": math.degrees(theta)}
    else:
        from pinneapple_simulation.electromagnetics.machine import SPMMachine
        from pinneapple_simulation.electromagnetics.machines3d import SPM3D
        from pinneapple_simulation.electromagnetics.machines3d.spm import step_skew
        from pinneapple_simulation.electromagnetics.materials import LinearMaterial

        m = SPMMachine(iron=LinearMaterial(mu_r=1000.0), n_theta=360)
        model = SPM3D(m, n_theta=360, stack_layers=6, end_length=0.02, end_layers=1)
        skew = step_skew(math.radians(10.0), 6, m.length)
        sol = model.solve(0.0, skew=skew)
        nu, Br = model.materials(0.0, skew=skew)
        facts = {"machine": f"SPM, {m.slots} slots, {2 * m.pole_pairs} poles, 10 deg step skew in 6 steps",
                 "theta_deg": 0.0}
    facts.update(tets=int(len(model.mesh.tets)), solve_s=round(time.perf_counter() - t0, 1))
    labels = R.tet_labels(nu, Br)
    mesh, B = model.mesh, sol.B
    if hasattr(model, "release"):
        model.release()
    return mesh, labels, B, facts


def main() -> dict:
    import bpy

    from pinneapple_simulation.electromagnetics.machines3d import renders as R
    from pinneapple_tools.visualization.studio import blender

    OUT.mkdir(parents=True, exist_ok=True)
    log = json.loads(LOG.read_text()) if LOG.exists() else {}
    log["hardware"] = {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(),
                       "python": platform.python_version(), "blender": bpy.app.version_string,
                       "engine": "Cycles, CPU, denoised"}
    log.setdefault("machines", {})

    def timed(rec: dict, name: str, path: Path, fn) -> None:
        if path.exists() and name in rec.get("assets", {}):
            return
        t = time.perf_counter()
        fn()
        rec.setdefault("assets", {})[name] = {"file": str(path.relative_to(LOG.parent)),
                                              "seconds": round(time.perf_counter() - t, 1),
                                              "bytes": path.stat().st_size}
        LOG.write_text(json.dumps(log, indent=1))
        print(f"{name}: {rec['assets'][name]['seconds']:.0f} s", flush=True)

    for key in ("afpm_single", "afpm_double", "spm_skew"):
        rec = log["machines"].setdefault(key, {})
        mesh, labels, B, facts = solve(key)
        rec.update(facts)
        print(f"{key}: {facts['tets']} tetrahedra, solved in {facts['solve_s']} s", flush=True)
        for view, cut in (("full", None), ("cut", CUT)):
            sc, info = R.machine_scene(mesh, labels, B, cut=cut, clip=CLIP, title=facts["machine"])
            rec[view] = {"B_min_T": info["B_min"], "B_max_T": info["B_max"], "colour_range_T": list(info["range"]),
                         "faces_above_range": info["above"], "faces": info["faces"], "cut_deg": cut,
                         "colour_bar": sc.labels["B"]}
            kw = {"field": "B", "field_range": info["range"], "view": "iso", "background": "studio"}
            p = OUT / f"{key}_{view}.jpg"
            timed(rec, f"{view}_still", p, lambda p=p, sc=sc, kw=kw: blender.render(sc, str(p), size=STILL,
                                                                                   samples=SAMPLES, **kw))
            if view == "cut":
                p = OUT / f"{key}_cut.glb"
                timed(rec, "cut_glb", p, lambda p=p, sc=sc: sc.save(str(p)))
                if key == "afpm_single":
                    p = OUT / f"{key}_cut_4k.jpg"
                    timed(rec, "cut_still_4k", p, lambda p=p, sc=sc, kw=kw: blender.render(
                        sc, str(p), size=HERO, samples=SAMPLES, **kw))
                if key in ("afpm_single", "spm_skew"):
                    tt = TURNTABLE
                    R.turntable(sc, seconds=tt["seconds"], fps=tt["fps"])
                    p = OUT / f"{key}_turntable.mp4"

                    def movie(sc=sc, info=info, key=key, tt=tt):
                        r = blender.render_animation(sc, str(OUT / f"{key}_turntable_frames"), size=tt["size"],
                                                     samples=tt["samples"], camera="fixed", lens=tt["lens"],
                                                     background="studio", field="B", field_range=info["range"],
                                                     hud=False, fps=tt["fps"], mp4=True, gif=False,
                                                     name=f"{key}_turntable")
                        os.replace(r["mp4"], OUT / f"{key}_turntable.mp4")
                        for f in r["frames"]:
                            os.remove(f)
                        os.rmdir(OUT / f"{key}_turntable_frames")
                    timed(rec, "turntable_mp4", p, movie)
                    rec["turntable"] = {**tt, "frames": int(round(tt["seconds"] * tt["fps"]))}
        del mesh, labels, B
    log["peak_memory_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
    log["date"] = time.strftime("%Y-%m-%d")
    LOG.write_text(json.dumps(log, indent=1))
    return log


if __name__ == "__main__":
    main()
