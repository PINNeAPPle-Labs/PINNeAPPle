"""PINNeAPPle Autonomous Physics Lab, demo 4: the rover that learns the planet.

A mission in plain language runs the lab loop of :mod:`pinneapple_simulation.rover.autolab` on a synthetic site whose
true soils are known: route on the orbital map -> multibody VIPER drive on the true terrain -> the first route stalls
in soft sand the orbiter called compact regolith -> the soil is learned from the wheels (sinkage, slip, torque;
iterated EKF) -> replanned route with the learned risk -> drive -> trust report (including the error of the learned
soil against the known truth). Then figures, Blender Cycles renders and the traverse video.

Writes ``results/``: report.md, trust_report.md, summary.json (every number), figures (PNG), renders (JPG),
traverse.mp4 / traverse.gif. ``_out/mission.pkl`` keeps the runs so ``--render-only`` can redo the renders.

Run: ``python examples/autolab_rover/rover_learns_planet.py [--mission "..."] [--no-render] [--render-only]
[--frames 120] [--samples 24]``. A full run takes about 10 min (mission) + 15 min (renders) on 4 CPU cores.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import pinneapple_lab
from pinneapple_simulation.rover.autolab import RoverMission, demo_site
from pinneapple_simulation.rover.autolab.report import run_info, write_report

HERE = os.path.dirname(os.path.abspath(__file__))
MISSION = ("Leve o rover até (37, 15) sem atolar, escorregamento máximo de 40%, minimizando a energia.")


def renders(site, res, out, frames: int, samples: int) -> dict:
    from PIL import Image

    from pinneapple_simulation.rover.autolab import render
    paths = {}
    if not render.available():
        print("bpy not installed: no Cycles renders (pip install bpy)")
        return paths
    t = time.perf_counter()
    for name, fn, kw in (("render_overview", render.render_overview,
                          dict(runs=res.runs, routes=res.routes, planner=res.planners[-1], samples=samples * 2)),
                         ("render_stall", render.render_hero, dict(runs=res.runs, samples=samples * 4))):
        png = os.path.join(HERE, "_out", f"{name}.png")
        fn(site, out_png=png, **kw)
        jpg = os.path.join(out, f"{name}.jpg")
        Image.open(png).convert("RGB").save(jpg, quality=88)
        paths[name] = jpg
        print(f"  {name}: {time.perf_counter() - t:.0f} s")
    _, mp4, gif = render.render_traverse(site, res.runs, os.path.join(HERE, "_out", "frames"), samples=samples,
                                         max_frames=frames,
                                         progress=lambda i, n: print(f"  frame {i + 1}/{n}", end="\r"))
    for src, name in ((mp4, "traverse.mp4"), (gif, "traverse.gif")):
        dst = os.path.join(out, name)
        os.replace(src, dst)
        paths[name] = dst
    paths["render_seconds"] = time.perf_counter() - t
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default=MISSION)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--render-only", action="store_true", help="reuse _out/mission.pkl")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--samples", type=int, default=24, help="Cycles samples of the video frames")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(HERE, "_out"), exist_ok=True)
    pkl = os.path.join(HERE, "_out", "mission.pkl")
    site = demo_site()
    with pinneapple_lab.experiment(
            "examples/autolab_rover", title="Autonomous Physics Lab: the rover that learns the planet",
            description="Mission in plain language -> route on the orbital map -> multibody VIPER drive -> stall in "
                        "soft sand mapped as regolith -> soil learned from the wheels (iterated EKF) -> replanned "
                        "route -> trust report against the physics and the known truth of a synthetic site.",
            domain="vehicle dynamics",
            methods=["multibody dynamics", "Wong-Reece terramechanics", "iterated EKF", "risk-aware Dijkstra",
                     "Blender Cycles"],
            modules=["pinneapple_simulation.rover.autolab", "pinneapple_simulation.vehicle",
                     "pinneapple_simulation.terramechanics"],
            params={"mission": a.mission}, out_dir=HERE, register=False,
            links={"guide": "docs/guides/rover_learns_the_planet.md",
                   "script": "examples/autolab_rover/rover_learns_planet.py"}) as ex:
        t0 = time.perf_counter()
        if a.render_only:
            with open(pkl, "rb") as f:
                m, res = pickle.load(f)
        else:
            m = RoverMission(site, a.mission, progress=lambda st, tool, s: print(
                f"[{time.perf_counter() - t0:6.0f} s] {st:>8} / {tool}: "
                + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()
                            if k in ("label", "ended", "distance_m", "energy_J", "max_slip", "max_risk",
                                     "predicted_energy_J", "decision", "rows"))))
            res = m.run()
            for r in res.runs:
                r.meta.pop("terrain", None)
            m.progress = None
            with open(pkl, "wb") as f:
                pickle.dump((m, res), f)
        mission_s = time.perf_counter() - t0
        info = run_info()
        nums = write_report(m, res, a.out, info=info)
        print(open(os.path.join(a.out, "report.md")).read())
        for r in nums["routes"]:
            mm = r["measured_multibody"]
            ex.log_metric(f"route_{r['name']}_distance", mm["distance_m"], "m", "multibody run on the true terrain")
            ex.log_metric(f"route_{r['name']}_energy", mm["energy_J"], "J", "sum of motor torque x wheel speed")
            ex.log_metric(f"route_{r['name']}_max_slip", mm["max_slip_2s_front"], "-",
                          "2-s mean front-wheel slip, maximum over the run")
            ex.log_metric(f"route_{r['name']}_max_sinkage", mm["max_sinkage_mm"], "mm", "max wheel sinkage")
            ex.log_metric(f"route_{r['name']}_learned_risk", r["predicted_with_learned_belief"]["max_risk"], "-",
                          "largest share of learned-belief samples above the slip limit along the route")
        for p, v in (nums.get("recovery") or {}).items():
            ex.log_metric(f"learned_{p}_error", v["error"], "", "learned minus true (synthetic site)")
            ex.log_metric(f"learned_{p}_z", v["z"], "sd", "error in posterior standard deviations")
        if not a.render_only:
            ex.log_metric("mission_wall_time", mission_s, "s", f"RoverMission.run on {info.get('cpu', '?')}")
        for c in res.report.checks:
            if c.status != "not_run":
                ex.check(f"trust: {c.name}", c.status == "supports", kind="verification",
                         detail=f"value {c.value}, limit {c.limit}; {c.source}")
        ex.check("first route stalls, final route reaches the goal",
                 res.runs[0].meta["ended"].startswith("stalled") and res.runs[-1].meta["ended"] == "goal",
                 kind="sanity", detail=" -> ".join(r.meta["ended"] for r in res.runs))
        if not a.no_render:
            rp = renders(site, res, a.out, a.frames, a.samples)
            if "render_seconds" in rp:
                ex.log_metric("render_wall_time", rp["render_seconds"], "s", "Blender Cycles (CPU) stills + frames")
            for k, p in rp.items():
                if k != "render_seconds":
                    ex.add_file(p)
        for f in sorted(os.listdir(a.out)):
            if f.endswith((".png", ".md", ".json")):
                ex.add_file(os.path.join(a.out, f))


if __name__ == "__main__":
    main()
