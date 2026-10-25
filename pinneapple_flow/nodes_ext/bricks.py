"""Self-modifying interlocking-brick builds (``examples/bricks/self_modifying_bricks``): the closed-form and Newton
checks that gate the loop (:func:`pinneapple_design.bricks.validation.validate`) and the two search strategies at
the same evaluation budget and seeds (:func:`pinneapple_design.bricks.compare`) for the three goals."""
from __future__ import annotations

from ..registry import Param, Port, node

_GOALS = ("cantilever", "bridge", "tower")


def _goal(name):
    from pinneapple_design.bricks import BridgeGoal, CantileverGoal, TowerGoal
    return {"cantilever": lambda: CantileverGoal(max_bricks=8),
            "bridge": lambda: BridgeGoal(gap=12, load_n=1.0),
            "tower": lambda: TowerGoal(max_bricks=16, push_n=0.01, base=6)}[name]


@node("bricks.validate", "Brick physics against closed forms and Newton (trust report)", category="Design",
      outputs=[Port("metrics", "metrics")], params=[Param("fea", "bool", True, help="voxel FEA plate beam check")],
      calls="pinneapple_design.bricks.validation.validate (harmonic stacking, tipping, stud cantilever, tower push, "
            "beam reactions, voxel FEA vs Timoshenko, Newton replays)")
def validate(ctx, inputs, params):
    """Every check's simulated value (``check.<i>``, in the library's order, the check named in the note), whether it
    passed, and the trust decision."""
    from pinneapple_design.bricks import validation
    val = validation.validate(fea=bool(params["fea"]))
    for i, r in enumerate(val["rows"]):
        ctx.metric(f"check.{i}", float(r["simulated"]), f"{r['check']} (reference {r['reference']:.6g} {r['unit']})")
    ctx.metric("checks_passed", sum(bool(r["passed"]) for r in val["rows"]))
    ctx.metric("checks", len(val["rows"]))
    ctx.metric("harmonic_newton_agree", sum(bool(h["agree"]) for h in val["harmonic_newton"]))
    ctx.metric("trust_decision", str(val["report"].decisao))
    ctx.table("checks", ["check", "simulated", "reference", "rel. error", "tolerance", "passed"],
              [[r["check"], r["simulated"], r["reference"], r["rel_error"], r["tolerance"], r["passed"]]
               for r in val["rows"]])
    return {"metrics": ctx.report["metrics"]}


@node("bricks.search", "Self-modifying brick search: NSGA-II vs GP surrogate", category="Design",
      outputs=[Port("metrics", "metrics")],
      params=[Param("goal", "enum", "cantilever", choices=list(_GOALS),
                    help="cantilever reach (8 parts), bridge over 96 mm (fewest parts), tower on 6 x 6 (height)"),
              Param("budget", "int", 400, minimum=10, help="physics evaluations per run"),
              Param("seeds", "list", [0, 1, 2, 3, 4])],
      calls="pinneapple_design.bricks.compare (evolve, surrogate_search) with CantileverGoal / BridgeGoal / TowerGoal")
def search(ctx, inputs, params):
    """Both strategies on every seed at the same budget: best score per seed, median (a seed with no feasible build
    counts as the worst), mean, spread, success rate; and the best build found."""
    from pinneapple_design.bricks import compare
    name = str(params["goal"])
    seeds = tuple(int(s) for s in params["seeds"])
    res = compare(_goal(name), budget=int(params["budget"]), seeds=seeds, progress=None)
    for strat, s in res["summary"].items():
        for k in ("best_median", "best_mean", "best_min", "best_max", "success_rate"):
            if s.get(k) is not None:
                ctx.metric(f"{name}.{strat}.{k}", float(s[k]))
        for sd, v in zip(s["seeds"], s["per_seed"], strict=True):
            if v is not None:
                ctx.metric(f"{name}.{strat}.seed{sd}", float(v), "best score of this seed")
    cands = [(r.best, st, r) for st, rs in res["runs"].items() for r in rs if r.best is not None]
    if cands:
        b, st, r = max(cands, key=lambda t: (t[0].ev.score, -t[0].ev.n_bricks))
        ctx.metric(f"{name}.best.score", float(b.ev.score), f"{st}, seed {r.seed}: {b.ev.explanation}")
        ctx.metric(f"{name}.best.n_bricks", int(b.ev.n_bricks))
    else:
        ctx.warn("no feasible build found")
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="bricks.self_modifying", title="Self-modifying brick builds judged by physics", domain="design",
        example="bricks",
        keywords={"brick": 2, "bricks": 2, "lego": 3, "interlocking": 2, "self-modifying": 3, "cantilever": 1,
                  "tower": 1, "bridge": 1},
        requires_all=(("brick", "bricks", "lego", "interlocking"),), min_score=3,
        nodes=("bricks.validate", "bricks.search", "examples.compare"), references=("deb2002nsga2", "rasmussen2006gp"),
        template="bricks/self_modifying_bricks", quick=True, description="examples/bricks/self_modifying_bricks.py"))


_register_capabilities()
