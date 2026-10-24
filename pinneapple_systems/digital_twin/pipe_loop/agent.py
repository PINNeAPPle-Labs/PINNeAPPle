"""The digital twin that knows when it is wrong: a monitoring mission -> plan -> controlled tools -> evidence.

    run = run_mission(DEFAULT_TEXT, out_dir="out")         # deterministic planner; planner="llm" to let a model plan

Tools: build the twin's fast model (hold-out measured), calibrate the alarms on healthy operation, monitor three
showcase events (a partial obstruction, a drifting flow meter, the plant driven outside the envelope), ask for a new
reference simulation where the model is outside its domain and extend the model with it, run the controlled
benchmark (confusion matrix), write the trust report and the deliverables.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from pinneapple_autolab import LLMPlanner, Run, Step, TemplatePlanner, Tool, Toolbox, execute

from .benchmark import run_benchmark
from .diagnose import Diagnoser, Surrogate
from .plant import (
    INPUTS,
    NORMAL_BOX,
    SENSORS,
    SIGMA,
    Loop,
    Scenario,
    readings,
    run_plant,
    steady_state,
)

DEFAULT_TEXT = ("Acompanhe a linha de água quente (bomba, aquecedor, cotovelo, junção em T, ramais B e C) com o gêmeo "
                "digital. Diga quando a previsão deixar de ser confiável e por quê: o processo mudou, um sensor está "
                "errado ou o modelo saiu do domínio. Mostre a evidência, peça nova simulação quando precisar e meça "
                "quantas vezes você acerta a causa em casos controlados.")

SHOWCASE = [Scenario("obstruction_B", "process", 4242, onset_s=1500.0, target="B", magnitude=6.0),
            Scenario("drift_Q_B", "sensor", 4243, onset_s=1500.0, target="Q_B", magnitude=2.4, mode="drift"),
            Scenario("overspeed", "domain", 4244, onset_s=1500.0, target="n", magnitude=1.15),
            Scenario("healthy", "normal", 4245)]


def _loop(state) -> Loop:
    return state["read_mission"]["loop"]


def t_read_mission(state, text: str) -> dict[str, Any]:
    low = text.lower()
    asks = {k: any(w in low for w in ws) for k, ws in {
        "process": ("processo", "process", "obstru"), "sensor": ("sensor",),
        "domain": ("domínio", "dominio", "domain", "regime"), "benchmark": ("controlad", "controlled", "matriz",
                                                                            "confusion", "quantas vezes")}.items()}
    return {"loop": Loop(), "text": text, "asks": asks, "n_sensors": len(SENSORS), "n_inputs": len(INPUTS),
            "hypotheses": "normal / process / sensor / domain"}


def t_build_twin(state, n_train: int = 120, n_test: int = 40, seed: int = 0) -> dict[str, Any]:
    from scipy.stats import qmc
    loop = _loop(state)
    sur = Surrogate(n_train=n_train, seed=seed).fit(loop)
    U = sur.lo + qmc.LatinHypercube(d=len(INPUTS), seed=seed + 99).random(n_test) * (sur.hi - sur.lo)
    Y = np.array([readings(steady_state(loop, dict(zip(INPUTS, u, strict=True)))) for u in U])
    P, S = sur.predict(U)
    err = np.abs(P - Y)
    return {"surrogate": sur, "holdout_max_abs": dict(zip(SENSORS, err.max(0).tolist(), strict=True)),
            "holdout_max_over_sigma": float((err / SIGMA).max()), "holdout_median_over_sigma":
            float(np.median(err / SIGMA)), "coverage_2std": float(np.mean(err <= 2 * S + 1e-12)),
            "n_train": n_train, "n_test": n_test}


def t_calibrate(state, n_runs: int = 12) -> dict[str, Any]:
    loop = _loop(state)
    diag = Diagnoser(loop, state["build_twin"]["surrogate"])
    cal = [run_plant(loop, Scenario(f"cal_{i}", "normal", 900_000 + i)) for i in range(n_runs)]
    diag.calibrate(cal)
    return {"diagnoser": diag, "n_runs": n_runs, **{f"threshold_{k}": v for k, v in diag.thresholds.items()}}


def t_monitor(state) -> dict[str, Any]:
    diag: Diagnoser = state["calibrate"]["diagnoser"]
    out = {}
    for sc in SHOWCASE:
        run = run_plant(diag.loop, sc)
        out[sc.name] = {"run": run, "diagnosis": diag.diagnose(run)}
    verdicts = {k: f"{v['diagnosis']['kind']}:{v['diagnosis'].get('target', '')}" for k, v in out.items()}
    return {"cases": out, "verdicts": verdicts, "correct": sum(
        v["diagnosis"]["kind"] == v["run"]["scenario"].kind and v["diagnosis"].get("target", "") ==
        v["run"]["scenario"].target for v in out.values()), "n": len(out)}


def t_request_simulation(state, n_new: int = 40, widen: float = 0.25) -> dict[str, Any]:
    """For every showcase case diagnosed as outside the domain: run the reference at the operating point, compare
    it with the readings (does the plant still obey the physics?), extend the fast model's envelope by ``widen`` of
    the box beyond the input that left it, add ``n_new`` reference solves there, refit, and monitor again."""
    from scipy.stats import qmc
    loop = _loop(state)
    out = {}
    for name, c in state["monitor"]["cases"].items():
        d = c["diagnosis"]
        if d["kind"] != "domain" or d["target"] not in INPUTS:
            continue
        run = c["run"]
        u = {k: float(run["u"][k][-1]) for k in INPUTS}
        ref = readings(steady_state(loop, u))
        refs = np.array([readings(steady_state(loop, {k: float(run["u"][k][i]) for k in INPUTS}))
                         for i in range(len(run["t"]) - 30, len(run["t"]))])     # reference at each of the last 30
        z_ref = (run["y"][-30:] - refs).mean(0) / (SIGMA / np.sqrt(30))
        sur0: Surrogate = state["build_twin"]["surrogate"]
        p0, s0 = sur0.predict(np.array([[u[k] for k in INPUTS]]))
        lo, hi = sur0.lo.copy(), sur0.hi.copy()
        jx = INPUTS.index(d["target"])
        w = hi[jx] - lo[jx]
        if u[d["target"]] > hi[jx]:
            hi[jx] = max(u[d["target"]], hi[jx]) + widen * w
        else:
            lo[jx] = min(u[d["target"]], lo[jx]) - widen * w
        Unew = lo + qmc.LatinHypercube(d=len(INPUTS), seed=7).random(n_new) * (hi - lo)
        Ynew = np.array([readings(steady_state(loop, dict(zip(INPUTS, x, strict=True)))) for x in Unew])
        from pinneapple_neural.architectures.rom.parametric import ParametricPOD
        sur1 = Surrogate(n_train=sur0.n_train + n_new, seed=sur0.seed)
        U = np.vstack([sur0.U_, Unew])
        Y = np.vstack([sur0.Y_, Ynew])
        sur1._scale, sur1._mean = Y.std(0) + 1e-12, Y.mean(0)
        sur1.model = ParametricPOD(r=len(SENSORS), energy=None, regressor="gpr").fit(U, (Y - sur1._mean) / sur1._scale)
        sur1.U_, sur1.Y_, sur1.lo, sur1.hi = U, Y, lo, hi
        diag1 = Diagnoser(loop, sur1, thresholds=dict(state["calibrate"]["diagnoser"].thresholds))
        d1 = diag1.diagnose(run)
        p1, _ = sur1.predict(np.array([[u[k] for k in INPUTS]]))
        out[name] = {"u": u, "reference": ref, "z_reference_vs_readings": z_ref, "max_abs_z_reference": float(
            np.abs(z_ref).max()), "old_prediction": p0[0], "old_std": s0[0], "new_prediction": p1[0],
            "old_error_over_sigma": float(np.abs((p0[0] - ref) / SIGMA).max()),
            "new_error_over_sigma": float(np.abs((p1[0] - ref) / SIGMA).max()),
            "envelope_before": dict(zip(INPUTS, zip(sur0.lo.tolist(), sur0.hi.tolist(), strict=True), strict=True)),
            "envelope_after": dict(zip(INPUTS, zip(lo.tolist(), hi.tolist(), strict=True), strict=True)),
            "diagnosis_after": d1}
    first = next(iter(out.values()), None)
    return {"cases": out, "n": len(out),
            "physics_consistent": bool(first and first["max_abs_z_reference"] < 4.0),
            "old_error_over_sigma": None if first is None else first["old_error_over_sigma"],
            "new_error_over_sigma": None if first is None else first["new_error_over_sigma"],
            "after": None if first is None else first["diagnosis_after"]["kind"]}


def t_benchmark(state, n_per_kind: int = 20, n_calibration: int = 12, workers: int = 4) -> dict[str, Any]:
    b = run_benchmark(_loop(state), n_per_kind=n_per_kind, n_calibration=n_calibration, workers=workers)
    return {"benchmark": b, "accuracy": b["accuracy"], "n_cases": b["n_cases"],
            "cases_with_false_alarm": b["cases_with_false_alarm"]}


def t_trust(state) -> dict[str, Any]:
    from .report import trust_report
    rep = trust_report(state)
    return {"report": rep, "decision": rep.decisao}


def t_deliver(state, out_dir: str = "out", render: bool = True, samples: int = 16) -> dict[str, Any]:
    from .report import deliver
    return deliver(state, out_dir, render=render, samples=samples)


def _summary(keys):
    return lambda out: {k: out[k] for k in keys if k in out}


def toolbox() -> Toolbox:
    T = Tool
    return Toolbox("pipe_twin", [
        T("read_mission", t_read_mission, "Read the monitoring request.", {"text": "the request"},
          _summary(["n_sensors", "n_inputs", "hypotheses"])),
        T("build_twin", t_build_twin, "Fit the fast model of the readings on the operating envelope; hold-out.",
          {"n_train": "reference solves", "n_test": "held-out solves", "seed": "seed"},
          _summary(["holdout_max_over_sigma", "holdout_median_over_sigma", "coverage_2std", "n_train"])),
        T("calibrate", t_calibrate, "Alarm thresholds from healthy runs.", {"n_runs": "healthy runs"},
          _summary(["n_runs", "threshold_shift", "threshold_spread", "threshold_model"])),
        T("monitor", t_monitor, "Monitor the showcase events and diagnose each.", {}, _summary(["correct", "n"])),
        T("request_simulation", t_request_simulation, "Reference simulation where the model is out of its domain; "
          "extend the model with it and monitor again.", {"n_new": "new reference solves", "widen": "envelope margin"},
          _summary(["n", "physics_consistent", "old_error_over_sigma", "new_error_over_sigma", "after"])),
        T("benchmark", t_benchmark, "Controlled cases with known answers: confusion matrix.",
          {"n_per_kind": "cases per kind", "n_calibration": "healthy calibration runs", "workers": "processes"},
          _summary(["accuracy", "n_cases", "cases_with_false_alarm"])),
        T("trust", t_trust, "Trust report of the twin.", {}, _summary(["decision"])),
        T("deliver", t_deliver, "Figures, 3-D scene, viewer, renders and the report.",
          {"out_dir": "output folder", "render": "Blender renders", "samples": "Cycles samples"},
          _summary(["report", "n_files"])),
    ])


def template_plan(text: str, *, out_dir: str = "out", render: bool = True, n_per_kind: int = 20,
                  workers: int = 4) -> TemplatePlanner:
    return TemplatePlanner([
        Step("read_mission", {"text": text}, "what to watch and what to answer"),
        Step("build_twin", {"n_train": 120, "n_test": 40}, "a fast model, measured before it is trusted"),
        Step("calibrate", {"n_runs": 12}, "alarm thresholds from healthy operation, not guessed"),
        Step("monitor", {}, "three events and a healthy hour"),
        Step("request_simulation", {"n_new": 40}, "where the model is out of its domain, ask the physics"),
        Step("benchmark", {"n_per_kind": n_per_kind, "workers": workers}, "how often the cause is right"),
        Step("trust", {}, "can the twin be trusted, and where"),
        Step("deliver", {"out_dir": out_dir, "render": render}, "evidence a person can check"),
    ])


REQUIRED = ("build_twin", "calibrate", "monitor", "benchmark", "trust")


def run_mission(text: str = DEFAULT_TEXT, *, out_dir: str = "out", planner: str = "template", render: bool = True,
                n_per_kind: int = 20, workers: int = 4, llm: Callable[[str, str], str] | None = None,
                provider: str = "anthropic", log: Callable[[str], None] | None = print) -> Run:
    box = toolbox()
    tpl = template_plan(text, out_dir=out_dir, render=render, n_per_kind=n_per_kind, workers=workers)
    p = (LLMPlanner(tpl, provider=provider, llm=llm, required=REQUIRED) if planner == "llm" else tpl).plan(text, box)
    return execute(p, box, log=log)


__all__ = ["run_mission", "toolbox", "template_plan", "DEFAULT_TEXT", "SHOWCASE", "NORMAL_BOX"]
