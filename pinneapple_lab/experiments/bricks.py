"""Self-modifying interlocking-brick builds (``pinneapple_design.bricks``): a build is checked by a static equilibrium
analysis with stud clutch, mutated, and checked again, until it meets a goal.

* ``cantilever`` -- the longest stable reach past a table edge with at most 8 parts.
* ``bridge``     -- the fewest parts that span a 96 mm gap and carry 1 N at mid-span.
* ``tower``      -- the tallest tower on a 6 x 6 stud footprint that survives a 10 mN push at the top.

Every run first checks the physics against closed forms (harmonic stacking, tipping, stud cantilever, tower push,
beam reactions; Newton replays when installed), then runs NSGA-II evolution and a GP-surrogate-guided search with
the same budget and seeds, and stores the best build (LDraw, glTF), the evolution GIF and the comparison chart.

    python -m pinneapple_lab sweep bricks -g case=cantilever,bridge,tower
"""
from __future__ import annotations

import importlib.util

from ..spec import Experiment, register

CASES = {
    "cantilever": "Longest stable cantilever past a table edge with at most 8 parts (gravity only).",
    "bridge": "Fewest parts spanning a 96 mm gap between two tables and carrying 1 N at mid-span.",
    "tower": "Tallest tower on a 6 x 6 stud footprint that survives a 10 mN horizontal push at its top.",
}


def make_goal(case: str):
    from pinneapple_design.bricks import BridgeGoal, CantileverGoal, TowerGoal
    return {"cantilever": lambda: CantileverGoal(max_bricks=8), "bridge": lambda: BridgeGoal(gap=12, load_n=1.0),
            "tower": lambda: TowerGoal(max_bricks=16, push_n=0.01, base=6)}[case]


@register
class BricksExperiment(Experiment):
    name = "bricks"
    version = "1"
    description = ("Self-modifying interlocking-brick builds: rigid-body statics with stud clutch judge every "
                   "candidate, an evolutionary and a surrogate-guided search redesign it at equal evaluation budget.")
    tags = ["design", "optimisation", "rigid-body", "statics", "newton"]
    references = ["R. Liu, K. Deng, Z. Wang, C. Liu, StableLego: stability analysis of block stacking assembly, "
                  "IEEE RA-L 9(11) 9383-9390 (2024): 0.98 N clutch per stud, brick masses",
                  "P. B. Johnson, Leaning tower of lire, Am. J. Phys. 23, 240 (1955); M. Paterson, U. Zwick, "
                  "Overhang, Amer. Math. Monthly 116(1) 19-44 (2009)",
                  "K. Deb et al., NSGA-II, IEEE TEC 6(2) 182-197 (2002)"]
    limitations = ["rigid parts, most favourable force distribution (lower-bound theorem): optimistic for brittle "
                   "clutch and compliant joints",
                   "clutch 0.98 N per stud with no published uncertainty; horizontal stud forces unbounded unless "
                   "shear_n is set; a single stud carries no torsion",
                   "Newton replays release failed joints from the start (no breakable joints)"]
    case_param = "case"
    case_descriptions = CASES
    params = {"case": "cantilever", "budget": 200, "seeds": 3, "newton": True, "gif": True}
    space = {"case": list(CASES)}

    def run(self, ctx):
        from pinneapple_design.bricks import compare, ldraw, mesh, report, validation, visuals
        p = ctx.params
        newton = bool(p["newton"]) and importlib.util.find_spec("newton") is not None
        with ctx.stage("closed-form checks"):
            val = validation.validate(newton=newton, fea=True)
        for r in val["rows"]:
            ctx.check(r["check"], bool(r["passed"]), value=r["rel_error"], detail=f"tolerance {r['tolerance']}",
                      kind="reference")
        make = make_goal(p["case"])
        seeds = tuple(range(int(p["seeds"])))
        with ctx.stage("search"):
            res = compare(make, budget=int(p["budget"]), seeds=seeds)
        for s, summ in res["summary"].items():
            ctx.metric(f"{s}_best_mean", summ["best_mean"])
            ctx.metric(f"{s}_best_std", summ["best_std"])
            ctx.metric(f"{s}_success_rate", summ["success_rate"])
        ctx.check("a_strategy_found_a_feasible_build",
                  any(summ["success_rate"] > 0 for summ in res["summary"].values()), kind="sanity")
        ctx.figure_file(report.plot_comparison({p["case"]: res["summary"]}, ctx.path("outputs", "comparison.png")))
        cands = [(r.best, r) for rs in res["runs"].values() for r in rs if r.best is not None]
        if not cands:
            return
        best, run = max(cands, key=lambda t: (t[0].ev.score, -t[0].ev.n_bricks))
        goal = make()
        ctx.metric("best_score", best.ev.score)
        ctx.metric("best_parts", best.ev.n_bricks)
        ctx.output("best_build", best.asm.to_dict())
        ctx.output("best_explanation", {"text": best.ev.explanation, "metrics": best.ev.metrics})
        ctx.check("best_build_passes_statics", bool(best.ev.feasible), detail=best.ev.explanation, kind="physics")
        ldraw.write_ldr(best.asm, ctx.path("outputs", "best.ldr"))
        mesh.write_gltf(best.asm, ctx.path("outputs", "best.glb"))
        if p["gif"]:
            n = len(run.generations)
            gens = sorted({round(i * (n - 1) / 5) for i in range(6)})
            frames = visuals.evolution_timeline(run, goal, generations=gens, hold=2, replay=newton)
            imgs = visuals.render_frames(frames, size=(640, 360))
            ctx.figure_file(visuals.write_gif(ctx.path("outputs", "evolution.gif"), imgs, fps=8))
        ctx.metric("evaluations_per_run", int(p["budget"]))
        ctx.metric("seeds", len(seeds))
