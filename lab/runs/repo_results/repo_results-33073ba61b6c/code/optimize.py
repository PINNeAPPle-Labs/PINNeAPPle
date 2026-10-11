"""Design search: NSGA-II over airfoil shape + wing area on the surrogate, with feasibility and trust per design.

Three objectives: top speed (up), CO2 per 100 km (down) and stall speed (down: shorter runways, gentler landings).
Speed and CO2 both come from low drag; what they fight is the low-speed side, so the front is speed vs stall speed.

Each candidate gets a section polar from the MLP ensemble (fast, with a spread), aircraft performance from it, the
requirement checks, and a trust level. The graph network adds the flow field and a second, independent set of
coefficients for the designs you look at; OpenFOAM is the last word (see apps/aero_optimizer/tools/verify.py).
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from pinneapple_design.design_optimizer.optimizer import DesignOptimizerConfig, EvolutionaryDesignOptimizer
from pinneapple_design.design_optimizer.pareto import compute_pareto_front

from .aircraft import Aircraft, Polar, Requirements, evaluate
from .geometry import BOUNDS, PARAMS, REFERENCE, outline, properties, valid
from .surrogate import (A_HI, A_LO, COEFS, AeroGNN, Layout, MLPEnsemble, coef_from_targets, edge_features,
                        node_features)

ALPHAS = np.arange(A_LO, A_HI + 0.01, 1.0)
WING_MASS_PER_M2 = 6.0           # kg per m2 of wing (light aircraft), applied to the change of wing area
AREA_BOUNDS = (10.0, 20.0)


class Engine:
    """Trained surrogates + the aircraft evaluation."""

    def __init__(self, bundle: Dict[str, Any]):
        self.meta = bundle.get("meta", {})
        self.mlp = MLPEnsemble(bundle["mlp_k"], bundle.get("mlp_width", 96))
        self.mlp.load_state_dict(bundle["mlp_state"])
        self.mlp.eval()
        self.mlp_norm = bundle["mlp_norm"]                       # mean/std of the targets
        self.gnn = None
        if bundle.get("gnn_state") is not None:
            self.layout = Layout()
            self.gnn = AeroGNN(bundle["gnn_node_in"], hidden=bundle.get("gnn_hidden", 48), mp=bundle.get("gnn_mp", 8))
            self.gnn.load_state_dict(bundle["gnn_state"])
            self.gnn.eval()
            self.gnn_norm = bundle["gnn_norm"]

    @classmethod
    def load(cls, path: str) -> "Engine":
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    # ------------------------------------------------------------------ coefficients
    @torch.no_grad()
    def mlp_coefficients(self, shapes: np.ndarray, alphas: np.ndarray = ALPHAS):
        """(n, len(alphas)) arrays: mean Cl, Cd, Cm and their ensemble std."""
        S = np.atleast_2d(shapes)
        n, m = len(S), len(alphas)
        x = MLPEnsemble.inputs(np.repeat(S, m, 0), np.tile(alphas, n))
        y = self.mlp(torch.from_numpy(x)).numpy() * self.mlp_norm["std"] + self.mlp_norm["mean"]   # (K, n*m, 3)
        cl, cd, cm = coef_from_targets(y)
        r = lambda v: v.reshape(len(v), n, m)                     # noqa: E731
        cl, cd, cm = r(cl), r(cd), r(cm)
        return {"cl": cl.mean(0), "cd": cd.mean(0), "cm": cm.mean(0),
                "cl_std": cl.std(0), "cd_std": cd.std(0), "cm_std": cm.std(0)}

    @torch.no_grad()
    def gnn_predict(self, shape: np.ndarray, alphas: List[float]) -> Dict[str, Any]:
        """Fields on the graph nodes and coefficients from the graph network, one row per angle."""
        if self.gnn is None:
            return {}
        lay = self.layout
        pos = lay.positions(shape)
        x = np.stack([node_features(lay, shape, a, pos) for a in alphas])
        e = np.repeat(edge_features(lay, pos)[None], len(alphas), 0)
        wall = torch.arange(len(lay.i_sel))
        y, c = self.gnn(torch.from_numpy(x), torch.from_numpy(lay.edge_index), torch.from_numpy(e), wall)
        nf = self.gnn_norm
        y = y.numpy() * nf["field_std"] + nf["field_mean"]
        c = c.numpy() * nf["coef_std"] + nf["coef_mean"]
        cl, cd, cm = coef_from_targets(c)
        return {"pos": pos, "fields": y, "cl": cl, "cd": cd, "cm": cm, "shape2": lay.shape2}

    # ------------------------------------------------------------------ designs
    def evaluate(self, X: np.ndarray, aircraft: Aircraft, req: Requirements) -> List[Dict[str, Any]]:
        X = np.atleast_2d(X)
        co = self.mlp_coefficients(X[:, :6])
        out = []
        for k, x in enumerate(X):
            shape, area = x[:6], float(x[6])
            ok, why = valid(shape)
            props = properties(shape)
            rec: Dict[str, Any] = {"x": x.tolist(), "shape": shape.tolist(), "wing_area": area, "props": props}
            if not ok:
                rec.update(feasible=False, valid=False, violations=[f"invalid geometry: {why}"], penalty=100.0,
                           vmax=float("nan"), co2_100km=float("nan"), trust="n/a")
                out.append(rec)
                continue
            ac = dataclasses.replace(aircraft, wing_area=area, mass=aircraft.mass + WING_MASS_PER_M2 * (area - aircraft.wing_area))
            pol = Polar(ALPHAS, co["cl"][k], co["cd"][k], co["cm"][k])
            perf = evaluate(props, pol, ac, req)
            # trust: ensemble spread at the operating points, and whether the polar is inside the sampled range
            ia = int(np.clip(np.searchsorted(ALPHAS, perf["cruise_alpha"]) if perf["cruise_alpha"] == perf["cruise_alpha"] else 4, 0, len(ALPHAS) - 1))
            rel_cd = float(co["cd_std"][k][ia] / co["cd"][k][ia])
            kmax = int(np.argmax(co["cl"][k]))
            cl_sd = float(co["cl_std"][k][kmax])
            trust = "high" if rel_cd < 0.03 and cl_sd < 0.04 else "medium" if rel_cd < 0.06 and cl_sd < 0.08 else "low"
            if perf["extrapolated"] or perf["clmax_at_edge"]:
                trust = "low"
            rec.update(perf, valid=True, mass=ac.mass, trust=trust, unc={"cd_rel": rel_cd, "clmax_sd": cl_sd},
                       polar={"alpha": ALPHAS.tolist(), "cl": co["cl"][k].tolist(), "cd": co["cd"][k].tolist(),
                              "cm": co["cm"][k].tolist(), "cl_sd": co["cl_std"][k].tolist(), "cd_sd": co["cd_std"][k].tolist()})
            out.append(rec)
        return out

    def search(self, aircraft: Aircraft = Aircraft(), req: Requirements = Requirements(), population: int = 64,
               generations: int = 50, seed: int = 0, area_bounds=AREA_BOUNDS,
               progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        bounds = np.vstack([BOUNDS, [area_bounds]])
        cfg = DesignOptimizerConfig(population_size=population, mutation_std=0.0, crossover_rate=0.9)
        opt = EvolutionaryDesignOptimizer(cfg, seed=seed, multi_objective=True)
        span = bounds[:, 1] - bounds[:, 0]
        rng = np.random.default_rng(seed)
        all_recs: List[Dict[str, Any]] = []
        t0 = time.time()
        for gen in range(generations):
            xs = opt.ask(bounds, population)
            if gen:                                               # per-variable mutation scaled to the bounds
                xs = [np.clip(x + rng.normal(0, 0.06, len(x)) * span * (rng.random(len(x)) < 0.4), bounds[:, 0], bounds[:, 1]) for x in xs]
            recs = self.evaluate(np.array(xs), aircraft, req)
            objs, pens = [], []
            for r in recs:
                bad = not (r.get("vmax", float("nan")) == r.get("vmax", float("nan")))
                objs.append([1e3 if bad else -r["vmax"], 1e3 if bad else r["co2_100km"], 1e3 if bad else r["v_stall"]])
                pens.append(float(r.get("penalty", 0.0)) + (0.5 if r.get("trust") == "low" else 0.0))
                r["generation"] = gen
            opt.tell(xs, [o[0] for o in objs], obj_vecs=objs, penalties=pens)
            all_recs += recs
            if progress:
                progress(gen + 1, generations)
        # Pareto set among feasible, trusted designs
        ok = [i for i, r in enumerate(all_recs) if r.get("feasible") and r.get("trust") != "low"]
        front = []
        if ok:
            F = np.array([[-all_recs[i]["vmax"], all_recs[i]["co2_100km"], all_recs[i]["v_stall"]] for i in ok])
            mask = compute_pareto_front(F)
            front = sorted([ok[i] for i in np.where(mask)[0]], key=lambda i: all_recs[i]["vmax"])
        return {"designs": all_recs, "pareto": front, "seconds": time.time() - t0, "evaluations": len(all_recs)}


def baseline(engine: Engine, aircraft: Aircraft, req: Requirements, name: str = "NACA 2412") -> Dict[str, Any]:
    x = np.r_[REFERENCE[name], aircraft.wing_area]
    return engine.evaluate(x[None], aircraft, req)[0]


def summarize(rec: Dict[str, Any]) -> Dict[str, Any]:
    """The compact form sent to the browser for the cloud of designs."""
    keys = ("vmax_kt", "co2_100km", "v_stall_kt", "cruise_ld", "fuel_l_100km", "feasible", "trust", "violations",
            "wing_area", "generation", "penalty")
    s = {k: rec.get(k) for k in keys}
    s["t"] = rec["props"]["thickness"]
    s["camber"] = rec["props"]["camber"]
    return s
