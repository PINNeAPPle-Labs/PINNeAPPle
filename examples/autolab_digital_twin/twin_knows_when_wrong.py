"""The digital twin that knows when it is wrong (PINNeAPPle Autonomous Physics Lab, demo 3).

A pumped, heated water loop (pump, heater, elbow, T junction, two branches with valves) runs normally while a
fast model tracks its ten sensors. Then something changes: a partial obstruction (the process), a drifting or noisy
sensor, or the plant is driven outside the envelope the model was trained on. The twin tells the three apart with
physics residuals (mass and energy balance, Darcy-Weisbach / Colebrook, pump and valve laws), a Kalman filter and its
domain of validity, says what it needs next, asks the physics for a new simulation where its model is out of its
domain, and measures on controlled cases how often it names the right cause (confusion matrix).

Run: ``python examples/autolab_digital_twin/twin_knows_when_wrong.py [--n-per-kind 20] [--no-render]``
(``--smoke``: 3 cases per kind, no renders). Outputs in ``examples/autolab_digital_twin/_out/``.
"""
from __future__ import annotations

import argparse
import os

import pinneapple_lab
from pinneapple_systems.digital_twin.pipe_loop import DEFAULT_TEXT, run_mission

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--n-per-kind", type=int, default=20)
    ap.add_argument("--planner", choices=("template", "llm"), default="template")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    a = ap.parse_args()
    n, render = (3, False) if a.smoke else (a.n_per_kind, not a.no_render)
    with pinneapple_lab.experiment(
            "examples/autolab_digital_twin", title="Digital twin that knows when it is wrong",
            description="A pipe-loop twin that tells a changed process from a wrong sensor from a model outside its "
                        "domain, with physics residuals, a Kalman filter and its domain of validity; confusion "
                        "matrix on controlled cases.",
            domain="system dynamics", methods=["Darcy-Weisbach / Colebrook network", "POD-GPR",
                                                         "structured residuals", "EKF"],
            modules=["pinneapple_autolab", "pinneapple_systems.digital_twin.pipe_loop"], out_dir=HERE, register=False,
            links={"guide": "docs/guides/autonomous_physics_lab.md",
                   "script": "examples/autolab_digital_twin/twin_knows_when_wrong.py"}) as ex:
        run = run_mission(a.text, out_dir=a.out, planner=a.planner, render=render, n_per_kind=n, workers=a.workers)
        st = run.state
        b = st["benchmark"]["benchmark"]
        ex.log_metric("cause_accuracy", b["accuracy"], "fraction", f"{b['n_cases']} controlled cases")
        ex.log_metric("cases_with_false_alarm", b["cases_with_false_alarm"], "cases",
                      "alarm before the fault, or any alarm on a healthy case")
        ex.log_metric("fast_model_holdout_max", st["build_twin"]["holdout_max_over_sigma"], "sensor sigmas",
                      f"{st['build_twin']['n_test']} held-out operating points")
        ex.check("showcase events diagnosed right", st["monitor"]["correct"] == st["monitor"]["n"], kind="physics")
        ex.check("trust report decision", st["trust"]["decision"] == "APPROVED",
                 detail=f"{st['trust']['decision']} {st['trust']['report'].motivos}", kind="verification")
        for k, p in st["deliver"]["files"].items():
            if k.startswith(("control_room", "confusion", "request")) or k == "report":
                ex.add_file(p)
        print(f"\n{st['trust']['decision']}: report {st['deliver']['report']}")


if __name__ == "__main__":
    main()
