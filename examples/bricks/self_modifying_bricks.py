"""Self-modifying interlocking-brick builds: simulate, judge by physics, redesign, repeat.

1. Check the physics against closed forms (harmonic stacking, tipping, stud cantilever, tower push, beam reactions,
   voxel FEA against Timoshenko, Newton replays) -> a trust report.
2. Three goals: the longest cantilever past a table edge with 8 parts, a bridge over a 96 mm gap carrying 1 N with
   the fewest parts, the tallest tower on a 6 x 6 stud footprint that survives a 10 mN push at the top.
3. Two search strategies at the same budget of physics evaluations and the same seeds: NSGA-II evolution and a
   Gaussian-process-surrogate-guided search; best score per seed, mean, spread.
4. The best builds: sensitivity to the joint model, glTF / STL / LDraw files, building instructions.
5. Visuals: the evolution in the three.js viewer (generation by generation, rejected candidates replayed by Newton,
   joints that let go in red, stud utilisation as a colour map), GIF / MP4, Blender Cycles renders and a turntable.

Run: ``python examples/bricks/self_modifying_bricks.py [--full] [--no-blender]``; outputs in ``examples/bricks/_out``.
Serve the viewer with ``python -m http.server -d examples/bricks/_out/viewer_cantilever``.
"""
from __future__ import annotations

import argparse
import json
import os

import pinneapple as pp
from pinneapple_design.bricks import (
    BridgeGoal,
    CantileverGoal,
    TowerGoal,
    compare,
    dynamics,
    ldraw,
    mesh,
    reference_cantilever,
    report,
    statics,
    validation,
    visuals,
)

HERE = os.path.dirname(os.path.abspath(__file__))
GOALS = {
    "cantilever": lambda: CantileverGoal(max_bricks=8),
    "bridge": lambda: BridgeGoal(gap=12, load_n=1.0),
    "tower": lambda: TowerGoal(max_bricks=16, push_n=0.01, base=6),
}
METRIC = {"cantilever": "reach past the edge [mm]", "bridge": "parts (fewer is better)", "tower": "height [mm]"}
# the bridge score is minus the part count: show the count, fewest at the top
DISPLAY = {"cantilever": {"unit": " [mm]"}, "bridge": {"scale": -1, "invert": True, "unit": " [parts]"},
           "tower": {"unit": " [mm]"}}


def run(smoke: bool = True, out_dir: str | None = None, blender: bool | None = None) -> dict:
    out = out_dir or os.path.join(HERE, "_out")
    os.makedirs(out, exist_ok=True)
    budget, seeds = (80, (0, 1)) if smoke else (400, (0, 1, 2, 3, 4))
    blender = visuals.blender_available() if blender is None else blender
    files = {}

    # 1. physics checks first: nothing below is reported without them
    val = validation.validate(fea=True)
    print(val["report"].to_markdown())

    # 2-3. the loop, both strategies, same budget and seeds
    comps, runs, best, sens = {}, {}, {}, {}
    for name, make in GOALS.items():
        res = compare(make, budget=budget, seeds=seeds, progress=None)
        comps[name], runs[name] = res["summary"], res["runs"]
        cands = [(r.best, s, r) for s, rs in res["runs"].items() for r in rs if r.best is not None]
        if not cands:
            continue
        b, strat, run_ = max(cands, key=lambda t: (t[0].ev.score, -t[0].ev.n_bricks))
        goal = make()
        best[name] = {"strategy": strat, "seed": run_.seed, "score": b.ev.score, "n_bricks": b.ev.n_bricks,
                      "explanation": b.ev.explanation, "metrics": b.ev.metrics, "build": b.asm.to_dict(),
                      "run": run_, "asm": b.asm, "goal": goal}
        sens[name] = report.sensitivity(goal, b.asm)
        for k, s in comps[name].items():
            print(f"{name:10s} {k:9s} median best {s['best_median']:.1f} (feasible {s['success_rate']:.0%}, "
                  f"range {s['best_min']:.1f} to {s['best_max']:.1f}) over seeds {s['seeds']}")
    files["comparison chart"] = report.plot_comparison(comps, os.path.join(out, "comparison.png"), ylabels=METRIC,
                                                       display=DISPLAY)

    # statics against Newton on the searched candidates, and the two known sources of disagreement, measured
    checks, notes = {}, []
    for name, b in best.items():
        hist = [c for r in runs[name]["evolution"] + runs[name]["surrogate"] for c in r.history]
        checks[name] = dynamics.cross_check(b["goal"], hist, max_n=8 if smoke else 20)
        mech = [c for c in hist if c.ev.status == "mechanism"][: 10 if smoke else 40]
        tors = [statics.single_stud_torsion(c.asm, b["goal"].loads(c.asm), b["goal"].joint) for c in mech]
        flips = sum(t["status_with_torsion"] == "stable" for t in tors)
        notes.append(f"{name}: {flips} of {len(tors)} statics mechanisms stand once single-stud joints get a little "
                     "torsion capacity (a lone stud is a pivot in the statics; Newton welds every intact joint).")
    if "tower" in best:
        a, goal = best["tower"]["asm"], best["tower"]["goal"]
        creep = dynamics.friction_creep(a, goal.loads(a))
        notes.append("XPBD friction creep under the sustained 10 mN push on the best tower (statics: friction holds): "
                     + ", ".join(f"{c['max_displacement_mm']:.2f} mm at dt = {c['dt']:g} s" for c in creep)
                     + f" over {creep[0]['duration_s']} s; it shrinks with dt, so it is a solver artefact.")
    hn = [r for r in val["harmonic_newton"] if r["factor"] == 1.0]
    if hn:
        notes.append("Harmonic stack at exactly the closed-form limit (neutral equilibrium): Newton "
                     f"{'keeps it standing' if hn[0]['newton_stands'] else 'tips it'} "
                     f"(top block drop {hn[0]['newton_top_drop_mm']:.1f} mm); at 0.99x it stands and at 1.01x it falls.")

    # 4. best builds as files and instructions
    for name, b in best.items():
        a = b["asm"]
        a.name = f"best {name}"
        files[f"{name} glTF"] = mesh.write_gltf(a, os.path.join(out, f"best_{name}.glb"))
        files[f"{name} STL"] = mesh.write_stl(a, os.path.join(out, f"best_{name}.stl"))
        files[f"{name} LDraw"] = ldraw.write_ldr(a, os.path.join(out, f"best_{name}.ldr"))
        files[f"{name} instructions"] = visuals.write_instructions(
            a, os.path.join(out, f"instructions_{name}"), title=f"Best {name}: building instructions")

    # 5. visuals: viewer + GIF/MP4 per goal (the cantilever in full length), Blender for the best builds
    for name, b in best.items():
        r, goal = b["run"], b["goal"]
        n = len(r.generations)
        k = 6 if smoke else 10
        gens = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
        frames = visuals.evolution_timeline(r, goal, generations=gens, hold=3, frames_per_replay=12 if smoke else 18)
        sc = visuals.timeline_scene(frames, title=f"Self-modifying {name}", source="pinneapple_design.bricks")
        files[f"{name} viewer"] = os.path.dirname(sc.export(os.path.join(out, f"viewer_{name}")))
        imgs = visuals.render_frames(frames, size=(640, 360) if smoke else (960, 540))
        files[f"{name} GIF"] = visuals.write_gif(os.path.join(out, f"evolution_{name}.gif"), imgs, fps=8)
        files[f"{name} MP4"] = visuals.write_mp4(os.path.join(out, f"evolution_{name}.mp4"), imgs, fps=8)
        if blender:
            a = b["asm"]
            u = visuals.utilisation(a, goal.loads(a), goal.joint)
            size = (960, 540) if smoke else (1600, 900)
            files[f"{name} Cycles render"] = visuals.render_blender(
                a, os.path.join(out, f"best_{name}_cycles.png"), samples=32 if smoke else 128, size=size)
            files[f"{name} Cycles load path"] = visuals.render_blender(
                a, os.path.join(out, f"best_{name}_load_path.png"), values=u, samples=32 if smoke else 96, size=size)
    if blender and "cantilever" in best:
        files["turntable GIF"] = os.path.join(out, "turntable_cantilever.gif")
        files["turntable MP4"] = os.path.join(out, "turntable_cantilever.mp4")
        visuals.turntable_blender(best["cantilever"]["asm"], os.path.join(out, "turntable_frames"),
                                  frames=8 if smoke else 36, samples=16 if smoke else 48,
                                  size=(640, 360) if smoke else (960, 540),
                                  gif=files["turntable GIF"], mp4=files["turntable MP4"])

    refs = {"cantilever": reference_cantilever(8)}
    files["report"] = report.write_report(os.path.join(out, "report.md"), validation=val, comparisons=comps,
                                          best=best, sensitivities=sens, references=refs, files=files,
                                          cross_checks=checks, notes=notes, display=DISPLAY)
    summary = {"validation": val["rows"], "harmonic_newton": val["harmonic_newton"],
               "trust_decision": val["report"].decisao, "budget": budget, "seeds": list(seeds),
               "comparison": {g: {s: {k: v for k, v in d.items() if k != "curves"} for s, d in c.items()}
                              for g, c in comps.items()},
               "best": {g: {k: v for k, v in b.items() if k not in ("run", "asm", "goal")} for g, b in best.items()},
               "sensitivity": sens, "references": refs, "files": {k: os.path.relpath(v, out) for k, v in files.items()},
               "cross_checks": checks, "notes": notes}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(pp.to_jsonable(summary), f, indent=1)
    print(f"report: {files['report']}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full", action="store_true", help="5 seeds, 400 evaluations per run, full-size renders")
    ap.add_argument("--no-blender", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(smoke=not args.full, out_dir=args.out, blender=False if args.no_blender else None)
