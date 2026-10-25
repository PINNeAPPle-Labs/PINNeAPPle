"""Cyber-physical security (``examples/security/02_physics_residual_fdi``): physics residuals against false data
injection on the three-tank water network, against three data-driven detectors
(:func:`pinneapple_security.cps_bench.run_benchmark`)."""
from __future__ import annotations

import numpy as np

from ..registry import Param, Port, node


@node("security.fdi_benchmark", "False-data-injection benchmark: physics residuals vs data-driven detectors",
      category="Security", outputs=[Port("metrics", "metrics")],
      params=[Param("seeds", "list", [0, 1, 2], help="one benchmark per seed; metrics are the mean over seeds"),
              Param("episodes", "int", 20, minimum=1), Param("test_days", "float", 10.0)],
      calls="pinneapple_security.cps_bench.run_benchmark (+ ATTACKS)")
def fdi_benchmark(ctx, inputs, params):
    """Per detector, regime and attack: episode AUC, detection rate, median delay (minutes, median over the seeds'
    medians) and localisation, averaged over the seeds as the example script prints them; false alarms per day on
    clean data. The isolation forest needs scikit-learn (skipped by the library without it)."""
    from pinneapple_security.cps_bench import ATTACKS, run_benchmark
    seeds = [int(s) for s in params["seeds"]]
    runs = []
    for s in seeds:
        ctx.check_stop()
        runs.append(run_benchmark(seed=s, episodes=int(params["episodes"]), test_days=float(params["test_days"])))
        ctx.log(f"seed {s} done")
    dets = list(runs[0]["results"])
    if "iforest" not in dets:
        ctx.warn("no isolation forest: scikit-learn is not installed")
    rows = []
    for regime in ("nominal", "shifted"):
        for d in dets:
            ctx.metric(f"{regime}.{d}.false_alarms_per_day",
                       float(np.mean([r["results"][d][regime]["false_alarms_per_day_clean"] for r in runs])))
        for kind in ATTACKS:
            for d in dets:
                xs = [r["results"][d][regime][kind] for r in runs]
                auc = float(np.mean([x["auc"] for x in xs]))
                det = float(np.mean([x["detection_rate"] for x in xs]))
                delays = [x["median_delay_min"] for x in xs if x["median_delay_min"] is not None]
                locs = [x["localisation"] for x in xs if x["localisation"] is not None]
                k = f"{regime}.{d}.{kind}"
                ctx.metric(f"{k}.auc", auc)
                ctx.metric(f"{k}.detection_rate", det)
                if delays:
                    ctx.metric(f"{k}.delay_min", float(np.median(delays)))
                if locs:
                    ctx.metric(f"{k}.localisation", float(np.mean(locs)))
                rows.append([regime, kind, d, round(auc, 3), round(det, 2),
                             round(float(np.median(delays))) if delays else "-",
                             round(float(np.mean(locs)), 2) if locs else "-"])
    ctx.table(f"mean over seeds {seeds}", ["regime", "attack", "detector", "AUC", "det", "delay min", "loc"], rows)
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="security.fdi_benchmark", title="Detect false data injection with physics residuals", domain="security",
        example="security",
        keywords={"false data injection": 4, "fdi": 3, "attack detection": 3, "cyber-physical": 3, "cyberattack": 3,
                  "anomaly detection": 2, "water network": 2, "three-tank": 3, "scada": 2, "intrusion": 2},
        min_score=3, nodes=("security.fdi_benchmark", "examples.compare"),
        uncited="the benchmark's own protocol (docs/guides/physics_residual_attack_detection.md)",
        template="security/02_physics_residual_fdi", quick=True,
        description="examples/security/02_physics_residual_fdi.py"))


_register_capabilities()
