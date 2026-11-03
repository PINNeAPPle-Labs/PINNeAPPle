"""Closed-loop reactor design: the reactor model, a learned model only when it pays off, optimisation, and the optimum
re-verified with the model (#1179).

1. **Measure** what one solve of the reactor model costs.
2. **Ask** :func:`~pinneapple_analysis.verification.architecture_recommendation.recommend_architecture` whether a
   surrogate pays off for the evaluations the optimiser will make (``none`` when the model is cheap enough; a
   Gaussian process for a few scalar outputs otherwise), with the checks the chosen model must pass.
3. **Optimise** the design variables (volume, coolant temperature, heat-transfer coefficient, inlet temperature) for
   the smallest volume subject to the conversion, selectivity and hot-spot limits, with differential evolution
   on the model directly or on the surrogate. With a surrogate, the design is re-solved by the model; while it
   misses a limit there, the solve joins the training set and the loop repeats (active learning).
4. **Verify** the final design with the model: the limits, the integration re-run ten times tighter, the Levenspiel
   area against the volume (PFR), the element balance through the stoichiometry (the change of the flows lies in
   the span of the reactions), and safety: Semenov's number and the parametric-sensitivity margin of a cooled PFR.

Every number in the result is computed by the run; the surrogate never grades itself.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .kinetics import Feed, Network
from .models import Bed, Cooling, cstr_series, pfr
from .sizing import adiabatic_rise, levenspiel, parametric_sensitivity, semenov_number, yield_selectivity

VARIABLES = ("V", "Tc", "Ua", "UA", "T0")


@dataclass
class DesignSpec:
    network: Network
    feed: Feed
    key: str                                       # the reactant whose conversion is required
    X_target: float
    bounds: dict[str, tuple[float, float]]         # design variables (subset of VARIABLES) -> (low, high)
    reactor: str = "pfr"                           # "pfr" (a packed bed when ``bed`` is given) or "cstr"
    cooling: Cooling = field(default_factory=Cooling)
    bed: Bed | None = None
    desired: str | None = None
    S_min: float | None = None                     # minimum selectivity to ``desired``
    T_limit: float | None = None                   # maximum temperature anywhere in the reactor, K
    margin_min: float | None = None                # minimum runaway margin (K of coolant temperature), cooled PFR
    tanks: int = 1                                 # CSTRs in series (equal volumes; ``V`` is the total)
    assumptions: list[str] = field(default_factory=list)
    source: str = ""

    def __post_init__(self):
        bad = set(self.bounds) - set(VARIABLES)
        if bad or "V" not in self.bounds:
            raise ValueError(f"bounds name V and any of {VARIABLES[1:]}; got {sorted(self.bounds)}")
        if self.reactor not in ("pfr", "cstr"):
            raise ValueError("reactor is 'pfr' or 'cstr'")


def evaluate(spec: DesignSpec, x: dict[str, float], *, rtol: float = 1e-8) -> dict[str, Any]:
    """The reactor model at design ``x``: conversion, selectivity and yield, peak temperature, outlet."""
    cool = replace(spec.cooling, **{k: float(x[k]) for k in ("Tc", "Ua", "UA") if k in x})
    feed = replace(spec.feed, T=float(x["T0"])) if "T0" in x else spec.feed
    V = float(x["V"])
    if spec.reactor == "pfr":
        prof = pfr(spec.network, feed, V, cool, spec.bed, rtol=rtol, atol=1e-12 * max(sum(feed.F.values()), 1e-12))
        out = prof.outlet()
        F_out, T_max = out["F"], out["T_max"]
        extra = {"P_out": out["P"], "pressure_lost": prof.info.get("pressure_lost", False)}
    else:
        train = cstr_series(spec.network, feed, [V / spec.tanks] * spec.tanks, cool)
        last = train[-1]["chosen"]
        F_out, T_max = last.F, max(t["chosen"].T for t in train)
        extra = {"tank_T": [t["chosen"].T for t in train], "tank_stable": [t["chosen"].stable for t in train],
                 "tank_states": [len(t["states"]) for t in train]}
    X = 1.0 - F_out[spec.key] / feed.F[spec.key]
    r = {"X": X, "T_max": T_max, "F_out": F_out, **extra}
    if spec.desired:
        r.update(yield_selectivity(feed.F, F_out, spec.key, spec.desired))
    return r


def _limits(spec: DesignSpec, y: dict[str, Any]) -> dict[str, float]:
    """Constraint values, feasible when >= 0."""
    g = {"conversion": y["X"] - spec.X_target}
    if spec.S_min is not None:
        g["selectivity"] = y["selectivity"] - spec.S_min
    if spec.T_limit is not None:
        g["hot_spot"] = spec.T_limit - y["T_max"]
    return g


def _optimise(objective_and_limits, names, bounds, seed, maxiter, popsize):
    from scipy.optimize import NonlinearConstraint, differential_evolution
    cache: dict[tuple, tuple[float, np.ndarray]] = {}

    def run(z):
        key = tuple(np.round(z, 12))
        if key not in cache:
            cache[key] = objective_and_limits(dict(zip(names, z, strict=True)))
        return cache[key]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    n_g = len(run((lo + hi) / 2)[1])
    res = differential_evolution(lambda z: run(z)[0], bounds, seed=seed, maxiter=maxiter, popsize=popsize,
                                 tol=1e-7, polish=False, init="sobol",
                                 constraints=NonlinearConstraint(lambda z: run(z)[1], np.zeros(n_g), np.inf))
    return dict(zip(names, map(float, res.x), strict=True)), len(cache)


def design_reactor(spec: DesignSpec, *, n_doe: int = 40, maxiter: int = 60, popsize: int = 12, seed: int = 0,
                   force_surrogate: bool = False, active_rounds: int = 6, sensitivity: bool = True) -> dict[str, Any]:
    """Run the closed loop and return a JSON-ready record (see the module docstring)."""
    names = list(spec.bounds)
    bounds = [tuple(map(float, spec.bounds[k])) for k in names]
    centre = {k: 0.5 * (a + b) for k, (a, b) in zip(names, bounds, strict=True)}

    # 1. the cost of one solve, measured
    t0 = time.perf_counter()
    for _ in range(3):
        evaluate(spec, centre)
    solve_s = (time.perf_counter() - t0) / 3
    n_g = 1 + (spec.S_min is not None) + (spec.T_limit is not None)
    n_eval = n_doe + popsize * len(names) * (maxiter + 1)

    # 2. which model, if any
    rec: dict[str, Any] = {"recommended": ["none"], "reasoning": "recommendation unavailable", "checks": []}
    try:
        from pinneapple_analysis.verification.architecture_recommendation import recommend_architecture
        try:
            r = recommend_architecture(n_high_fidelity_simulations=n_doe, needs_parameter_generalization=True,
                                       geometry_varies=False, has_analytical_solver=True, solve_seconds=solve_s,
                                       n_evaluations=n_eval, output_kind="scalar", n_inputs=len(names))
            rec = {"recommended": list(r.recommended), "reasoning": r.reasoning,
                   "decision_path": list(r.decision_path), "checks": list(getattr(r, "checks", []) or [])}
        except TypeError:
            rec["reasoning"] = "the installed recommendation predates the cost inputs"
    except ImportError:
        pass
    use_surrogate = force_surrogate or rec["recommended"][0] != "none"

    def scale_V(x):
        lo, hi = spec.bounds["V"]
        return (x["V"] - lo) / (hi - lo)

    history: list[dict[str, Any]] = []
    surrogate_report: dict[str, Any] | None = None
    if not use_surrogate:
        def f(x):
            y = evaluate(spec, x)
            return scale_V(x), np.array(list(_limits(spec, y).values()))
        x_opt, n_solves = _optimise(f, names, bounds, seed, maxiter, popsize)
        n_model_solves = n_solves
    else:
        from scipy.stats import qmc
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        Z = qmc.scale(qmc.Sobol(len(names), seed=seed).random(n_doe), lo, hi)
        rows = [evaluate(spec, dict(zip(names, z, strict=True))) for z in Z]
        outs = ["X"] + (["selectivity"] if spec.S_min is not None else []) + (["T_max"] if spec.T_limit else [])
        Zs, Ys = list(Z), {o: [r[o] for r in rows] for o in outs}
        n_model_solves = n_doe + 3

        def gp():
            k = ConstantKernel() * Matern(length_scale=[0.3] * len(names), nu=2.5) + WhiteKernel(1e-8, (1e-12, 1e-2))
            return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=seed)

        def fit():
            U = (np.array(Zs) - lo) / (hi - lo)
            with warnings.catch_warnings():          # hyperparameters at a bound are reported by the held-out check
                warnings.simplefilter("ignore")
                return {o: gp().fit(U, np.array(Ys[o])) for o in outs}

        # held-out error of a model trained on 80 % of the design of experiments (before any active point)
        cut = int(0.8 * n_doe)
        U_all = (Z - lo) / (hi - lo)
        held = {}
        for o in outs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = gp().fit(U_all[:cut], np.array(Ys[o][:cut]))
            mu, sd = g.predict(U_all[cut:], return_std=True)
            truth = np.array(Ys[o][cut:])
            scale = max(float(np.ptp(Ys[o])), 1e-12)
            held[o] = {"rel_rmse": float(np.sqrt(np.mean((mu - truth) ** 2)) / scale),
                       "coverage_95": float(np.mean(np.abs(mu - truth) <= 1.96 * np.maximum(sd, 1e-15)))}
        surrogate_report = {"family": "kpi_regressor (Gaussian process, Matern 5/2)", "n_train": n_doe,
                            "held_out": held}
        x_opt = None
        for rnd in range(active_rounds):
            gps = fit()

            def f(x, gps=gps):
                # limits on the pessimistic side of the 95 % band: a design the model may miss is not taken
                u = (np.array([x[k] for k in names]) - lo) / (hi - lo)
                pred = {}
                for o in outs:
                    mu, sd = gps[o].predict(u[None], return_std=True)
                    pred[o] = float(mu[0]) + (1.96 if o == "T_max" else -1.96) * float(sd[0])
                y = {"X": pred["X"], "T_max": pred.get("T_max", 0.0), "selectivity": pred.get("selectivity", 1.0)}
                return scale_V(x), np.array(list(_limits(spec, y).values()))
            x_opt, _ = _optimise(f, names, bounds, seed + rnd, maxiter, popsize)
            y = evaluate(spec, x_opt)
            n_model_solves += 1
            g = _limits(spec, y)
            history.append({"round": rnd, "x": x_opt, "limits_on_model": g})
            if min(g.values()) >= -1e-6:
                break
            Zs.append(np.array([x_opt[k] for k in names]))
            for o in outs:
                Ys[o].append(y[o])
        surrogate_report["active_rounds"] = len(history)

    # 4. verification with the model
    y = evaluate(spec, x_opt)
    y_tight = evaluate(spec, x_opt, rtol=1e-10)
    g = _limits(spec, y)
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None):
        checks.append({"code": code, "status": "pass" if ok is True else ("fail" if ok is False else "inconclusive"),
                       "detail": detail, "value": value})
    for name, v in g.items():
        chk(f"design.{name}", v >= -1e-6, f"margin {v:.4g} at the design, computed by the model", v)
    if spec.reactor == "pfr":
        dX = abs(y["X"] - y_tight["X"])
        chk("model.converged", dX < 1e-6, f"conversion changes by {dX:.1e} with a 100x tighter tolerance", dX)
    else:
        worst = _cstr_balance_residual(spec, x_opt)
        chk("model.balances", worst < 1e-8, f"mole and energy balances of every tank closed to {worst:.1e} "
                                            "(relative to the feed and its heat flow)", worst)
    F0 = spec.feed.vector(spec.network)
    dF = np.array([y["F_out"][s] for s in spec.network.species]) - F0
    nu = spec.network.nu
    xi, *_ = np.linalg.lstsq(nu, dF, rcond=None)
    resid = float(np.abs(nu @ xi - dF).max() / max(float(F0.sum()), 1e-30))
    chk("model.stoichiometry", resid < 1e-8, f"the flow change lies in the span of the reactions (residual "
                                             f"{resid:.1e} of the feed)", resid)
    cool = replace(spec.cooling, **{k: float(x_opt[k]) for k in ("Tc", "Ua", "UA") if k in x_opt})
    feed = replace(spec.feed, T=float(x_opt["T0"])) if "T0" in x_opt else spec.feed
    safety: dict[str, Any] = {"adiabatic_rise_K": adiabatic_rise(spec.network, feed, spec.key)}
    if spec.reactor == "pfr":
        lv = levenspiel(spec.network, feed, spec.key, x_opt["V"], cool, spec.bed, n=801)
        if spec.bed is None and np.all(np.isfinite(lv["FA0_over_rate"])):
            err = abs(lv["area"] - x_opt["V"]) / x_opt["V"]
            chk("model.levenspiel", err < 1e-2, f"area under F_A0/(-r_A) vs X is {lv['area']:.5g} m^3 for a volume "
                                                f"of {x_opt['V']:.5g} m^3 ({err:.1e})", err)
        if cool.Ua > 0:
            sem = semenov_number(spec.network, feed, cool)
            safety["semenov"] = sem
            chk("safety.semenov", True if not sem["runaway"] else None,
                f"psi = {sem['psi']:.3g} against 1/e" + ("" if not sem["runaway"] else
                                                       ": above the zeroth-order bound, which is conservative for "
                                                       "a reaction that consumes its reactant; the sensitivity "
                                                       "margin decides"), sem["psi"])
            if sensitivity:
                ps = parametric_sensitivity(spec.network, feed, x_opt["V"], cool, spec.bed,
                                            inlet_follows="T0" not in x_opt)
                safety["sensitivity"] = {"Tc_critical": ps["Tc_critical"], "margin_K": ps["margin"],
                                         "s_at_operation": ps["s_at_operation"]}
                if spec.margin_min is not None:
                    m = ps["margin"]
                    chk("safety.runaway_margin", None if m is None else m >= spec.margin_min,
                        "no sensitivity peak inside the scan" if m is None else
                        f"coolant {m:.1f} K below the runaway boundary (Morbidelli-Varma), required "
                        f"{spec.margin_min:g} K", m)
    else:
        stable = all(y["tank_stable"])
        chk("safety.stable_state", stable, f"every tank on a stable steady state (slope condition); steady states "
                                           f"per tank: {y['tank_states']}", stable)
    if surrogate_report:
        for o, h in surrogate_report["held_out"].items():
            chk(f"surrogate.held_out_error.{o}", h["rel_rmse"] <= 0.05,
                f"relative RMSE {h['rel_rmse']:.2%} of the range on {n_doe - int(0.8 * n_doe)} held-out solves",
                h["rel_rmse"])
    return {"design": x_opt, "outputs": {k: v for k, v in y.items() if k != "F_out"}, "outlet": y["F_out"],
            "limits": g, "checks": checks, "safety": safety, "recommendation": rec,
            "cost": {"solve_seconds": solve_s, "planned_evaluations": n_eval, "model_solves": n_model_solves},
            "surrogate": surrogate_report, "history": history, "assumptions": spec.assumptions,
            "source": spec.source}


def _cstr_balance_residual(spec: DesignSpec, x: dict[str, float]) -> float:
    """Largest relative residual of the mole and energy balances over the tanks of the design."""
    from .kinetics import concentrations
    cool = replace(spec.cooling, **{k: float(x[k]) for k in ("Tc", "Ua", "UA") if k in x})
    f = replace(spec.feed, T=float(x["T0"])) if "T0" in x else spec.feed
    Vt = float(x["V"]) / spec.tanks
    net, worst = spec.network, 0.0
    for t in cstr_series(net, f, [Vt] * spec.tanks, cool):
        st = t["chosen"]
        F0, F = f.vector(net), np.array([st.F[s] for s in net.species])
        c, p = concentrations(F, st.T, f.P, f)
        r = net.rates(c, p, st.T)
        mole = np.abs(F0 - F + Vt * (net.nu @ r)).max() / max(float(F0.sum()), 1e-30)
        UA = cool.UA if cool.UA is not None else cool.Ua * Vt
        q_flow = float(F0 @ net.cp_vec)
        energy = abs(q_flow * (f.T - st.T) + UA * (cool.Tc - st.T) + Vt * float(r @ -net.dH_at(st.T)))
        worst = max(worst, float(mole), energy / max(q_flow * max(abs(st.T - f.T), 1.0), 1e-30))
        f = Feed(dict(st.F), st.T, f.phase, f.v, f.P)
    return worst


def _json(v):
    if isinstance(v, dict):
        return {k: _json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def to_json(record: dict[str, Any]) -> dict[str, Any]:
    return _json(record)
