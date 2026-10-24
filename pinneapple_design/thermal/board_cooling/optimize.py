"""Constrained search on the surrogate, verification on the reference, and the loop between them.

:func:`optimize`: lightest heat sink (mass from the closed form, exact) whose every part meets its junction limit
with pessimistic cooling at ``mean + k std`` of the surrogate, whose fan air power stays inside the budget and whose
fins keep the minimum gap; differential evolution with the fin count as an integer (``scipy.optimize``).

:func:`design_loop`: fit -> optimise -> run the reference on the optimum -> add it to the snapshots -> refit, until
the reference confirms the optimum (it meets every limit and lies inside the surrogate's +-2 std band) or the
rounds run out. Every round is kept (prediction, reference, error), so the evolution of the maximum temperature is a
measured history, not a story.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .learn import CoolingSurrogate
from .mission import CoolingDesign, CoolingMission
from .simulate import response, simulate, sink_mass_and_air


def _air(mission: CoolingMission, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mass and fan air power of each design; a geometry the fins do not fit gets a prohibitive value."""
    m, p = [], []
    for x in X:
        d = CoolingDesign.from_vector(x, mission.material)
        if d.fin_gap_mm <= 0 or d.fin_height_mm <= 0:
            m.append(1e4)
            p.append(1e3 * mission.fan_power_budget_w)
            continue
        a = sink_mass_and_air(mission, d)
        m.append(a["mass_g"])
        p.append(a["fan_power_w"])
    return np.array(m), np.array(p)


def constraint_violation(mission: CoolingMission, X: np.ndarray, tj_upper: np.ndarray, fan_w: np.ndarray
                         ) -> np.ndarray:
    """Sum of the positive parts of every constraint, each scaled to a comparable size (1 = 1 K, 10 % of the fan
    budget, 0.1 mm of fin gap). ``tj_upper``: (n, n_parts) pessimistic junction temperatures to hold."""
    lim = np.array([mission.limits()[c.name] for c in mission.components])
    v = np.clip(tj_upper - lim, 0, None).sum(1)
    v += np.clip(fan_w / mission.fan_power_budget_w - 1, 0, None) * 10
    sp = mission.space
    v += np.array([max(-sp.gap_margin(x), 0.0) * 10 + max(sp.min_fin_height_mm - (x[5] - x[2]), 0.0) for x in X])
    return v


def optimize(mission: CoolingMission, sur: CoolingSurrogate, *, k_sigma: float = 2.0, seed: int = 0,
             maxiter: int = 120, popsize: int = 24) -> dict[str, Any]:
    """Minimum-mass design on the surrogate. Returns the design, its prediction and the per-generation history of
    the best member (mass, predicted pessimistic hot-part temperature, violation)."""
    from scipy.optimize import differential_evolution
    sp = mission.space
    n = len(mission.components)
    hot = [c.name for c in mission.components].index(mission.hot)
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    def objective(Xt):                                   # vectorized: (7, S)
        X = Xt.T.copy()
        X[:, 3] = np.round(X[:, 3])
        p = sur.predict(X, maps=False)
        upper = p["tj"][:, n:] + k_sigma * p["tj_std"][:, n:]
        mass, fan = _air(mission, X)
        return mass + 1e3 * constraint_violation(mission, X, upper, fan)

    def callback(intermediate_result):
        x = np.asarray(intermediate_result.x)
        p = sur.predict(x[None], maps=False)
        mass, fan = _air(mission, x[None])
        history.append({"generation": len(history) + 1, "mass_g": float(mass[0]),
                        "tj_hot_pessimistic_pred_c": float(p["tj"][0, n + hot]),
                        "tj_hot_pred_c": float(p["tj"][0, hot]), "fan_power_w": float(fan[0]),
                        "objective": float(intermediate_result.fun)})

    integrality = np.zeros(len(sp.lo), bool)
    integrality[3] = True
    res = differential_evolution(objective, list(zip(sp.lo, sp.hi, strict=True)), seed=seed, maxiter=maxiter,
                                 popsize=popsize, tol=1e-7, polish=False, integrality=integrality, vectorized=True,
                                 updating="deferred", callback=callback)
    x = np.asarray(res.x, float)
    x[3] = round(x[3])
    p = sur.predict(x[None])
    mass, fan = _air(mission, x[None])
    upper = p["tj"][:, n:] + k_sigma * p["tj_std"][:, n:]
    return {"x": x, "design": CoolingDesign.from_vector(x, mission.material),
            "pred": {k: v[0] for k, v in p.items()}, "mass_g": float(mass[0]), "fan_power_w": float(fan[0]),
            "violation": float(constraint_violation(mission, x[None], upper, fan)[0]),
            "history": history, "seconds": time.perf_counter() - t0, "n_evaluations": int(res.nfev)}


def design_loop(mission: CoolingMission, data: dict[str, np.ndarray], *, rounds: int = 5, k_sigma: float = 2.0,
                seed: int = 0, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Learn -> optimise -> verify on the reference -> add the verified point, until confirmed. ``data``: the
    output of :func:`.simulate.simulate_many` (extended in place with the verified points)."""
    names = [c.name for c in mission.components]
    n = len(names)
    lim = np.array([mission.limits()[c] for c in names])
    out_rounds: list[dict[str, Any]] = []
    sur = None
    confirmed = False
    for k in range(rounds):
        sur = CoolingSurrogate(mission.t_ambient_c)
        sur.fit(data["X"], data["Y"], data["maps"])
        opt = optimize(mission, sur, k_sigma=k_sigma, seed=seed + k)
        ref = simulate(mission, opt["design"], fidelity="train")
        y = response(mission, ref)
        err = opt["pred"]["tj"] - y
        inside = bool(np.all(np.abs(err) <= 2 * opt["pred"]["tj_std"] + 0.05))
        meets = bool(np.all(y[n:] <= lim))
        rnd = {"round": k + 1, "n_snapshots": int(len(data["X"])), "design": opt["design"].to_dict(),
               "mass_g": opt["mass_g"], "fan_power_w": opt["fan_power_w"],
               "pred_tj_c": dict(zip(names, opt["pred"]["tj"][:n].tolist(), strict=True)),
               "pred_tj_pessimistic_c": dict(zip(names, opt["pred"]["tj"][n:].tolist(), strict=True)),
               "pred_std_k": dict(zip(names, opt["pred"]["tj_std"][n:].tolist(), strict=True)),
               "ref_tj_c": ref["tj_c"], "ref_tj_pessimistic_c": ref["tj_pessimistic_c"],
               "max_abs_error_k": float(np.abs(err).max()), "inside_2std": inside, "meets_limits": meets,
               "pred_map_c": opt["pred"]["map"], "ref_map_c": ref["map_c"],
               "optimizer_history": opt["history"], "optimizer_seconds": opt["seconds"],
               "optimizer_evaluations": opt["n_evaluations"], "reference_seconds": ref["seconds"],
               "fit_seconds": sur.fit_seconds}
        out_rounds.append(rnd)
        if log:
            log(f"round {k + 1}: {opt['mass_g']:.1f} g, {mission.hot} predicted {rnd['pred_tj_pessimistic_c'][mission.hot]:.2f} "
                f"/ reference {ref['tj_pessimistic_c'][mission.hot]:.2f} °C (pessimistic), max error "
                f"{rnd['max_abs_error_k']:.2f} K, {'meets' if meets else 'VIOLATES'} limits")
        for key, val in (("X", opt["x"][None]), ("Y", y[None]), ("maps", ref["map_c"][None]),
                         ("seconds", np.array([ref["seconds"]])),
                         ("energy_balance", np.array([ref["energy_balance_board"]])),
                         ("converged", np.array([ref["coupling_converged"]]))):
            data[key] = np.concatenate([data[key], val])
        if inside and meets:
            confirmed = True
            break
    return {"rounds": out_rounds, "confirmed": confirmed, "surrogate": sur,
            "best": out_rounds[-1]["design"] if out_rounds else None}
