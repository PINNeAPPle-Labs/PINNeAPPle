"""Whole-aircraft search: NSGA-II over airfoil + planform + wing position, evaluated with the vortex lattice, the CFD
section surrogate and the component build-up (aircraft3d). Same objectives as the section search: top speed up,
CO2 per 100 km down, stall speed down; requirements from Requirements3D; trust from the surrogate ensemble spread.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pinneapple_design.design_optimizer.optimizer import DesignOptimizerConfig, EvolutionaryDesignOptimizer
from pinneapple_design.design_optimizer.pareto import compute_pareto_front

from .aircraft import Aircraft, Polar
from .aircraft3d import BOUNDS3D, PLAN, Aircraft3D, Requirements3D, airframe_from, baseline_x
from .geometry import REFERENCE, properties, valid
from .optimize import ALPHAS, Engine


class Engine3D:
    def __init__(self, engine: Engine):
        self.e = engine
        co = engine.mlp_coefficients(REFERENCE["NACA 0012"][None])
        self.tail_polar = Polar(ALPHAS, co["cl"][0], co["cd"][0], co["cm"][0])
        self._base_w = None

    def base_weights(self, ac: Aircraft) -> Dict[str, float]:
        x = baseline_x()
        co = self.e.mlp_coefficients(x[None, :6])
        a = Aircraft3D(x, Polar(ALPHAS, co["cl"][0], co["cd"][0], co["cm"][0]), self.tail_polar, ac)
        return a.weights

    def evaluate(self, X: np.ndarray, ac: Aircraft, req: Requirements3D, fine: bool = False,
                 base_w: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        X = np.atleast_2d(X)
        base_w = base_w or self.base_weights(ac)
        co = self.e.mlp_coefficients(X[:, :6])
        lat = dict(nc=6, ns_wing=24, ns_tail=8) if fine else dict(nc=4, ns_wing=12, ns_tail=5)
        out = []
        for k, x in enumerate(X):
            ok, why = valid(x[:6])
            rec: Dict[str, Any] = {"x": x.tolist(), "props": properties(x[:6]), "plan": dict(zip(PLAN, x[6:12].tolist()))}
            if not ok:
                rec.update(feasible=False, valid=False, violations=[f"invalid geometry: {why}"], penalty=100.0,
                           vmax=float("nan"), co2_100km=float("nan"), v_stall=float("nan"), trust="n/a")
                out.append(rec)
                continue
            wp = Polar(ALPHAS, co["cl"][k], co["cd"][k], co["cm"][k])
            r = Aircraft3D(x, wp, self.tail_polar, ac, req, base_w, **lat).evaluate()
            ia = int(np.clip(np.searchsorted(ALPHAS, r["cruise_alpha"]), 0, len(ALPHAS) - 1)) if r["cruise_alpha"] == r["cruise_alpha"] else 4
            rel_cd = float(co["cd_std"][k][ia] / co["cd"][k][ia])
            cl_sd = float(co["cl_std"][k][int(np.argmax(co["cl"][k]))])
            trust = "high" if rel_cd < 0.03 and cl_sd < 0.04 else "medium" if rel_cd < 0.06 and cl_sd < 0.08 else "low"
            if r["extrapolated"]:
                trust = "low"                                       # cruise outside the sampled polar
            # cl max not reached by 14 deg: the 14-deg value is a lower bound, so the stall speed errs on the safe side
            rec.update(r, valid=True, trust=trust, unc={"cd_rel": rel_cd, "clmax_sd": cl_sd},
                       polar={"alpha": ALPHAS.tolist(), "cl": co["cl"][k].tolist(), "cd": co["cd"][k].tolist(),
                              "cm": co["cm"][k].tolist(), "cl_sd": co["cl_std"][k].tolist(), "cd_sd": co["cd_std"][k].tolist()})
            out.append(rec)
        return out

    def search(self, ac: Aircraft = Aircraft(), req: Requirements3D = Requirements3D(), population: int = 48,
               generations: int = 40, seed: int = 0, bounds: Optional[np.ndarray] = None,
               progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        bounds = BOUNDS3D if bounds is None else bounds
        cfg = DesignOptimizerConfig(population_size=population, mutation_std=0.0, crossover_rate=0.9)
        opt = EvolutionaryDesignOptimizer(cfg, seed=seed, multi_objective=True)
        span = bounds[:, 1] - bounds[:, 0]
        rng = np.random.default_rng(seed)
        base_w = self.base_weights(ac)
        recs: List[Dict[str, Any]] = []
        t0 = time.time()
        for gen in range(generations):
            xs = opt.ask(bounds, population)
            if gen == 0:                                            # seed the baseline into the first population
                xs = [np.clip(baseline_x(), bounds[:, 0], bounds[:, 1])] + list(xs[1:])
            else:
                xs = [np.clip(x + rng.normal(0, 0.06, len(x)) * span * (rng.random(len(x)) < 0.3), bounds[:, 0], bounds[:, 1]) for x in xs]
            rs = self.evaluate(np.array(xs), ac, req, base_w=base_w)
            objs, pens = [], []
            for r in rs:
                bad = not (r.get("vmax", float("nan")) == r.get("vmax", float("nan")) and r.get("v_stall", float("nan")) == r.get("v_stall", float("nan")))
                objs.append([1e3 if bad else -r["vmax"], 1e3 if bad else r["co2_100km"], 1e3 if bad else r["v_stall"]])
                pens.append(float(r.get("penalty", 0.0)) + (0.5 if r.get("trust") == "low" else 0.0))
                r["generation"] = gen
            opt.tell(xs, [o[0] for o in objs], obj_vecs=objs, penalties=pens)
            recs += rs
            if progress:
                progress(gen + 1, generations)
        ok = [i for i, r in enumerate(recs) if r.get("feasible") and r.get("trust") != "low"]
        front = []
        if ok:
            F = np.array([[-recs[i]["vmax"], recs[i]["co2_100km"], recs[i]["v_stall"]] for i in ok])
            front = sorted([ok[i] for i in np.where(compute_pareto_front(F))[0]], key=lambda i: recs[i]["vmax"])
        return {"designs": recs, "pareto": front, "seconds": time.time() - t0, "evaluations": len(recs), "base_weights": base_w}


def summarize3d(r: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("vmax_kt", "co2_100km", "v_stall_kt", "cruise_ld", "fuel_l_100km", "feasible", "trust", "violations",
            "generation", "penalty", "static_margin", "stall_station", "span", "mass")
    s = {k: r.get(k) for k in keys}
    s["t"] = r["props"]["thickness"]
    s["x"] = r["x"]
    return s
