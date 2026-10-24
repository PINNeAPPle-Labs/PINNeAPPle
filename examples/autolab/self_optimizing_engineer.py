"""The self-optimizing engineer (PINNeAPPle Autonomous Physics Lab, demo 2: thermal).

A board with an overheating SoC, a junction limit, the room for a heat sink and a fan budget, stated in plain
language. The agent plans and runs, through controlled library tools only: read the request -> solve the bare board
-> try still air -> reference solves -> learn the temperature field (measured on held-out designs, more solves until
good enough) -> lightest heat sink + fan speed under the limits on the surrogate, confirmed on the reference ->
independent fine-grid verification + trust report -> figures, 3-D scene, viewer, Blender renders and the report.

Run: ``python examples/autolab/self_optimizing_engineer.py [--n 64] [--no-render] [--planner llm] [--text "..."]``
(``--smoke``: 24 solves, no renders). Outputs in ``examples/autolab/_out/self_optimizing_engineer/``.
"""
from __future__ import annotations

import argparse
import os

import pinneapple_lab
from pinneapple_design.thermal.board_cooling import DEFAULT_TEXT, run_mission

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--n", type=int, default=64, help="initial reference solves")
    ap.add_argument("--planner", choices=("template", "llm"), default="template")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "_out", "self_optimizing_engineer"))
    a = ap.parse_args()
    n, render = (24, False) if a.smoke else (a.n, not a.no_render)
    with pinneapple_lab.experiment(
            "examples/autolab", title="Autonomous Physics Lab",
            description="Missions in plain language planned and run through controlled library tools, ending in a "
                        "verified decision with its evidence: the self-optimizing engineer (board cooling).",
            domain="design and geometry", methods=["3-D board FV", "heat-sink FV + correlations", "POD-GPR",
                                                   "differential evolution"],
            modules=["pinneapple_autolab", "pinneapple_design.thermal.board_cooling"], out_dir=HERE, register=False,
            links={"guide": "docs/guides/autonomous_physics_lab.md",
                   "script": "examples/autolab/self_optimizing_engineer.py"}) as ex:
        run = run_mission(a.text, out_dir=a.out, planner=a.planner, n=n, render=render, workers=a.workers)
        st = run.state
        m = st["read_mission"]["mission"]
        v, h = st["verify"], st["learn"]["holdout"]
        ex.log_metric("baseline_tj_soc", st["baseline"]["tj_hot_c"], "°C", "bare board, still air, reference solve")
        ex.log_metric("verified_tj_soc_pessimistic", v["tj_fine_pessimistic"][m.hot], "°C",
                      "fine-grid coupled solve, pessimistic cooling")
        ex.log_metric("holdout_tj_max_abs_error", h["surrogate"]["tj_max_abs_k"], "K",
                      f"surrogate vs reference on {h['n_test']} unseen designs")
        ex.log_metric("holdout_map_rmse", h["surrogate"]["map_rmse_k"], "K", "board-top map on unseen designs")
        ex.log_metric("reference_solves", st["learn"]["n_solves"], "solves", "training + hold-out")
        ex.log_metric("heat_sink_mass", v["fine"]["mass_g"], "g", "closed form of the verified design")
        ex.check("every part within its junction limit (pessimistic, fine grid)",
                 all(v["tj_fine_pessimistic"][c] <= lim for c, lim in m.limits().items()), kind="physics")
        ex.check("trust report decision", v["decision"] == "APPROVED", detail=f"{v['decision']} "
                 f"{v['report'].motivos}", kind="verification")
        for k, p in st["deliver"]["files"].items():
            if k in ("maps", "evolution", "holdout", "report"):
                ex.add_file(p)
        print(f"\n{v['decision']}: report {st['deliver']['report']}")


if __name__ == "__main__":
    main()
