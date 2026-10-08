"""Airliner search: NSGA-II over the wing section (from the CFD surrogate), planform, cruise Mach and wing position.

Objectives: CO2 per passenger-km (down), cruise Mach (up), approach speed Vref (down). Requirements from
RequirementsAL; trust from the surrogate ensemble spread where the wing sections fly in cruise.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pinneapple_design.design_optimizer.optimizer import DesignOptimizerConfig, EvolutionaryDesignOptimizer
from pinneapple_design.design_optimizer.pareto import compute_pareto_front

from .aircraft import Polar
from .airliner3d import BOUNDS_AL, PLAN, AirlinerModel, Mission, RequirementsAL, baseline_x_al
from .geometry import REFERENCE, properties, valid
from .optimize import ALPHAS, Engine


class EngineAL:
    def __init__(self, engine: Engine):
        self.e = engine
        co = engine.mlp_coefficients(REFERENCE["NACA 0012"][None])
        self.tail_polar = Polar(ALPHAS, co["cl"][0], co["cd"][0], co["cm"][0])

    def evaluate(self, X: np.ndarray, mission: Mission, req: RequirementsAL, fine: bool = False) -> List[Dict[str, Any]]:
        X = np.atleast_2d(X)
        co = self.e.mlp_coefficients(X[:, :6])
        lat = dict(nc=6, ns_wing=24, ns_tail=8) if fine else dict(nc=4, ns_wing=12, ns_tail=5)
        out = []
        for k, x in enumerate(X):
            ok, why = valid(x[:6])
            rec: Dict[str, Any] = {"x": x.tolist(), "props": properties(x[:6]), "plan": dict(zip(PLAN, x[6:13].tolist()))}
            if not ok:
                rec.update(feasible=False, valid=False, violations=[f"invalid geometry: {why}"], penalty=100.0,
                           co2_pkm=float("nan"), mach=float(x[11]), vref_kt=float("nan"), trust="n/a")
                out.append(rec)
                continue
            wp = Polar(ALPHAS, co["cl"][k], co["cd"][k], co["cm"][k])
            try:
                r = AirlinerModel(x, wp, self.tail_polar, mission, req, **lat).evaluate()
            except (np.linalg.LinAlgError, ValueError, OverflowError) as e:
                rec.update(feasible=False, valid=False, violations=[f"no solution: {e}"], penalty=50.0,
                           co2_pkm=float("nan"), mach=float(x[11]), vref_kt=float("nan"), trust="n/a")
                out.append(rec)
                continue
            # trust: ensemble spread of the section drag at the angle the sections fly in cruise (~ cl / cl_alpha)
            cl_sec = float(np.median(r["loading"]["cl"]))
            a_sec = float(np.interp(cl_sec, co["cl"][k], ALPHAS))
            rel_cd = float(np.interp(a_sec, ALPHAS, co["cd_std"][k] / co["cd"][k]))
            cl_sd = float(co["cl_std"][k][int(np.argmax(co["cl"][k]))])
            trust = "high" if rel_cd < 0.03 and cl_sd < 0.04 else "medium" if rel_cd < 0.06 and cl_sd < 0.08 else "low"
            rec.update(r, valid=True, trust=trust, unc={"cd_rel": rel_cd, "clmax_sd": cl_sd},
                       polar={"alpha": ALPHAS.tolist(), "cl": co["cl"][k].tolist(), "cd": co["cd"][k].tolist(),
                              "cl_sd": co["cl_std"][k].tolist(), "cd_sd": co["cd_std"][k].tolist()})
            out.append(rec)
        return out

    def search(self, mission: Mission = Mission(), req: RequirementsAL = RequirementsAL(), population: int = 48,
               generations: int = 40, seed: int = 0, progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        bounds = BOUNDS_AL
        cfg = DesignOptimizerConfig(population_size=population, mutation_std=0.0, crossover_rate=0.9)
        opt = EvolutionaryDesignOptimizer(cfg, seed=seed, multi_objective=True)
        span = bounds[:, 1] - bounds[:, 0]
        rng = np.random.default_rng(seed)
        recs: List[Dict[str, Any]] = []
        t0 = time.time()
        for gen in range(generations):
            xs = opt.ask(bounds, population)
            if gen == 0:
                xs = [baseline_x_al()] + list(xs[1:])
            else:
                xs = [np.clip(x + rng.normal(0, 0.06, len(x)) * span * (rng.random(len(x)) < 0.3), bounds[:, 0], bounds[:, 1]) for x in xs]
            rs = self.evaluate(np.array(xs), mission, req)
            objs, pens = [], []
            for r in rs:
                bad = not (r.get("co2_pkm") == r.get("co2_pkm") and r.get("vref_kt") == r.get("vref_kt"))
                objs.append([1e3 if bad else r["co2_pkm"], 1e3 if bad else -r["mach"], 1e3 if bad else r["vref_kt"]])
                pens.append(float(r.get("penalty", 0.0)) + (0.5 if r.get("trust") == "low" else 0.0))
                r["generation"] = gen
            opt.tell(xs, [o[0] for o in objs], obj_vecs=objs, penalties=pens)
            recs += rs
            if progress:
                progress(gen + 1, generations)
        ok = [i for i, r in enumerate(recs) if r.get("feasible") and r.get("trust") != "low"]
        front = []
        if ok:
            F = np.array([[recs[i]["co2_pkm"], -recs[i]["mach"], recs[i]["vref_kt"]] for i in ok])
            front = sorted([ok[i] for i in np.where(compute_pareto_front(F))[0]], key=lambda i: recs[i]["mach"])
        return {"designs": recs, "pareto": front, "seconds": time.time() - t0, "evaluations": len(recs)}


def summarize_al(r: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("co2_pkm", "mach", "speed_kt", "vref_kt", "cruise_ld", "mtow", "feasible", "trust", "violations",
            "generation", "penalty", "static_margin", "stall_station", "span", "fuel_kg")
    s = {k: r.get(k) for k in keys}
    s["t"] = r["props"]["thickness"]
    s["x"] = r["x"]
    return s
