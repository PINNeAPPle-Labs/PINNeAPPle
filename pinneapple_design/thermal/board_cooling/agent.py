"""The self-optimizing engineer: a cooling request in plain language -> plan -> controlled tools -> evidence.

    run = run_mission(DEFAULT_TEXT, out_dir="out")           # deterministic planner, no API key needed
    run = run_mission(text, out_dir="out", planner="llm")    # a language model orders the same tools

The agent can only call the tools of :func:`toolbox` (library calls with declared arguments); every call is logged
with its arguments, a summary of its result and its wall time (:mod:`pinneapple_autolab.core`).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from pinneapple_autolab import LLMPlanner, Run, Step, TemplatePlanner, Tool, Toolbox, execute

from .learn import CoolingSurrogate, holdout
from .mission import CoolingDesign, parse_mission
from .optimize import design_loop
from .simulate import simulate, simulate_many
from .verify import verify


def _mission(state):
    return state["read_mission"]["mission"]


def t_read_mission(state, text: str) -> dict[str, Any]:
    m = parse_mission(text)
    hot = m.hot_component()
    return {"mission": m, "hot": m.hot, "power_w": hot.power_w, "tj_limit_c": m.tj_limit_c,
            "t_ambient_c": m.t_ambient_c, "envelope_mm": f"{m.max_width_mm:g} x {m.max_depth_mm:g} x "
            f"{m.max_height_mm:g}", "fan_power_budget_w": m.fan_power_budget_w,
            "from_text": sum(v != "default" for v in m.provenance.values()), "provenance": m.provenance}


def t_baseline(state) -> dict[str, Any]:
    m = _mission(state)
    r = simulate(m, None, fidelity="train")
    return {"result": r, "tj_hot_c": r["tj_c"][m.hot], "board_max_c": r["board_max_c"],
            "over_limit_k": r["tj_c"][m.hot] - m.tj_limit_c, "seconds_solve": r["seconds"]}


def t_screen_natural(state, gaps_mm: tuple[float, ...] = (4.0, 6.0, 8.0, 11.0)) -> dict[str, Any]:
    """Still air: the largest heat sink the envelope allows, for a few fin gaps (the optimum gap of a natural-
    convection sink is several mm). If even the best one misses the limit, forced air is needed."""
    m = _mission(state)
    W, D, H = m.max_width_mm, m.max_depth_mm, m.max_height_mm
    rows = []
    for g in gaps_mm:
        t = 1.5
        n = max(2, int((W + g) // (t + g)))
        d = CoolingDesign(W, D, 5.0, n, t, H, 0.0, m.material)
        r = simulate(m, d, fidelity="train")
        rows.append({"gap_mm": d.fin_gap_mm, "n_fins": n, "result": r, "tj_hot_c": r["tj_c"][m.hot],
                     "tj_hot_pessimistic_c": r["tj_pessimistic_c"][m.hot]})
    best = min(rows, key=lambda r: r["tj_hot_pessimistic_c"])
    return {"rows": rows, "best": best, "tj_hot_c": best["tj_hot_c"],
            "tj_hot_pessimistic_c": best["tj_hot_pessimistic_c"],
            "feasible": bool(max(best["result"]["tj_pessimistic_c"][c] - v for c, v in m.limits().items()) <= 0)}


def t_sample(state, n: int = 64, seed: int = 0, workers: int = 4) -> dict[str, Any]:
    m = _mission(state)
    X = m.space.sample(n, np.random.default_rng(seed))
    data = simulate_many(m, X, fidelity="train", workers=workers)
    return {"data": data, "n": int(n), "median_seconds": float(np.median(data["seconds"])),
            "max_energy_balance": float(data["energy_balance"].max()), "all_converged": bool(data["converged"].all())}


def t_learn(state, test_fraction: float = 0.2, seed: int = 0, max_error_k: float = 3.0, min_coverage: float = 0.9,
            add: int = 32, max_n: int = 320, workers: int = 4) -> dict[str, Any]:
    """Fit and measure on held-out designs; while the hold-out misses ``max_error_k`` or ``min_coverage``, run
    ``add`` more reference solves (new Latin-hypercube points) and measure again, up to ``max_n`` solves. The
    learning curve (hold-out metrics per sample size) is part of the evidence."""
    m = _mission(state)
    data = state["sample"]["data"]
    curve = []
    k = 0
    while True:
        h = holdout(data, m.t_ambient_c, test_fraction=test_fraction, seed=seed)
        s = h["surrogate"]
        curve.append({"n": int(len(data["X"])), "tj_mae_k": s["tj_mae_k"], "tj_max_abs_k": s["tj_max_abs_k"],
                      "map_rmse_k": s["map_rmse_k"], "coverage_2std": h["coverage_2std"]})
        ok = s["tj_max_abs_k"] <= max_error_k and h["coverage_2std"] >= min_coverage
        if ok or len(data["X"]) + add > max_n:
            break
        k += 1
        X = m.space.sample(add, np.random.default_rng(1000 + seed + k))
        more = simulate_many(m, X, fidelity="train", workers=workers)
        data = {key: np.concatenate([data[key], more[key]]) for key in data}
    state["sample"]["data"] = data
    state["sample"]["n"] = int(len(data["X"]))
    return {"holdout": h, "curve": curve, "accepted": bool(ok), "n_solves": int(len(data["X"])),
            "tj_mae_k": s["tj_mae_k"], "tj_max_abs_k": s["tj_max_abs_k"], "map_rmse_k": s["map_rmse_k"],
            "coverage_2std": h["coverage_2std"], "nearest_tj_mae_k": h["nearest_snapshot"]["tj_mae_k"],
            "speedup": h["speedup"]}


def t_optimize(state, rounds: int = 5, k_sigma: float = 2.0, seed: int = 0) -> dict[str, Any]:
    m = _mission(state)
    data = {k: np.array(v, copy=True) for k, v in state["sample"]["data"].items()}
    loop = design_loop(m, data, rounds=rounds, k_sigma=k_sigma, seed=seed)
    last = loop["rounds"][-1]
    return {"loop": loop, "data": data, "confirmed": loop["confirmed"],
            "rounds": len(loop["rounds"]), "mass_g": last["mass_g"], "fan_power_w": last["fan_power_w"],
            "tj_hot_pessimistic_ref_c": last["ref_tj_pessimistic_c"][m.hot],
            "max_abs_error_k": last["max_abs_error_k"]}


def t_verify(state) -> dict[str, Any]:
    m = _mission(state)
    o = state["optimize"]
    loop = o["loop"]
    last = loop["rounds"][-1]
    d = CoolingDesign(**{k: last["design"][k] for k in CoolingDesign.__dataclass_fields__})
    sur: CoolingSurrogate = loop["surrogate"]
    pred = {k: v[0] for k, v in sur.predict(d.vector()[None]).items()}
    coarse = simulate(m, d, fidelity="train")
    data, h = o["data"], state["learn"]["holdout"]
    te = np.array(h["test_index"])
    n0 = len(state["sample"]["data"]["X"])                          # sampled designs, before the loop added any
    tr = np.setdiff1d(np.arange(n0), te)
    v = verify(m, d, coarse=coarse, prediction=pred, holdout=h, train_X=data["X"][tr], test_X=data["X"][te])
    return {**v, "design": d, "coarse": coarse, "prediction": pred, "tj_hot_fine_pessimistic_c":
            v["tj_fine_pessimistic"][m.hot], "decision_str": v["decision"]}


def t_deliver(state, out_dir: str = "out", render: bool = True, samples: int = 24) -> dict[str, Any]:
    from .report import deliver
    return deliver(state, out_dir, render=render, samples=samples)


def _summary(keys):
    return lambda out: {k: out[k] for k in keys if k in out}


def toolbox() -> Toolbox:
    T = Tool
    return Toolbox("board_cooling", [
        T("read_mission", t_read_mission, "Read the cooling request: part power, junction limit, air temperature, "
          "heat-sink envelope, fan budget (numbers from the text, defaults otherwise).", {"text": "the request"},
          _summary(["hot", "power_w", "tj_limit_c", "t_ambient_c", "envelope_mm", "fan_power_budget_w", "from_text"])),
        T("baseline", t_baseline, "Reference solve of the bare board in still air.", {},
          _summary(["tj_hot_c", "board_max_c", "over_limit_k", "seconds_solve"])),
        T("screen_natural", t_screen_natural, "Still air with the largest heat sink the envelope allows.",
          {"gaps_mm": "fin gaps to try [mm]"}, _summary(["tj_hot_c", "tj_hot_pessimistic_c", "feasible"])),
        T("sample", t_sample, "Reference solves of a Latin-hypercube sample of forced-air designs.",
          {"n": "number of designs", "seed": "random seed", "workers": "parallel processes"},
          _summary(["n", "median_seconds", "max_energy_balance", "all_converged"])),
        T("learn", t_learn, "Fit the field + junction-temperature surrogate, measure it on held-out designs and "
          "add reference solves until the hold-out is accepted.",
          {"test_fraction": "fraction held out", "seed": "split seed", "max_error_k": "accepted max |error| [K]",
           "min_coverage": "accepted +-2 std coverage", "add": "solves added per refinement",
           "max_n": "solve budget", "workers": "parallel processes"},
          _summary(["accepted", "n_solves", "tj_mae_k", "tj_max_abs_k", "map_rmse_k", "coverage_2std", "nearest_tj_mae_k", "speedup"])),
        T("optimize", t_optimize, "Minimum-mass design on the surrogate under the limits; verify each optimum on "
          "the reference and refit until confirmed.", {"rounds": "max rounds", "k_sigma": "std margin",
                                                        "seed": "seed"},
          _summary(["confirmed", "rounds", "mass_g", "fan_power_w", "tj_hot_pessimistic_ref_c", "max_abs_error_k"])),
        T("verify", t_verify, "Independent fine-grid solve of the chosen design and the trust report.", {},
          _summary(["decision_str", "tj_hot_fine_pessimistic_c", "grid_max_dtj_k"])),
        T("deliver", t_deliver, "Figures, 3-D scene, viewer, renders and the report.",
          {"out_dir": "output folder", "render": "Blender renders", "samples": "Cycles samples"},
          _summary(["report", "n_files"])),
    ])


def template_plan(text: str, *, n: int = 64, out_dir: str = "out", render: bool = True, workers: int = 4
                  ) -> TemplatePlanner:
    return TemplatePlanner([
        Step("read_mission", {"text": text}, "turn the request into numbers with their source"),
        Step("baseline", {}, "how hot is it today"),
        Step("screen_natural", {}, "is a passive heat sink enough"),
        Step("sample", {"n": n, "seed": 0, "workers": workers}, "reference solves to learn from"),
        Step("learn", {"test_fraction": 0.2, "seed": 0, "workers": workers},
             "fast approximation, measured on unseen designs, more solves until it is good enough"),
        Step("optimize", {"rounds": 5, "k_sigma": 2.0, "seed": 0}, "search on the surrogate, confirm on physics"),
        Step("verify", {}, "independent finer solve and trust report"),
        Step("deliver", {"out_dir": out_dir, "render": render}, "evidence a person can check"),
    ])


REQUIRED = ("read_mission", "sample", "learn", "optimize", "verify")


def run_mission(text: str, *, out_dir: str = "out", planner: str = "template", n: int = 64, render: bool = True,
                workers: int = 4, llm: Callable[[str, str], str] | None = None, provider: str = "anthropic",
                log: Callable[[str], None] | None = print) -> Run:
    """Plan and run the cooling study. ``planner``: "template" (deterministic) or "llm" (falls back to the
    template, with the reason in ``run.plan.notes``, when no model answers or its plan is not valid)."""
    box = toolbox()
    tpl = template_plan(text, n=n, out_dir=out_dir, render=render, workers=workers)
    p = (LLMPlanner(tpl, provider=provider, llm=llm, required=REQUIRED) if planner == "llm" else tpl).plan(text, box)
    return execute(p, box, log=log)
