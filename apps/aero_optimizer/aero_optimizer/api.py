"""Aircraft Design Optimizer: web API + single-page UI.

OpenFOAM runs -> surrogates (MeshGraphNet on the CFD grid + MLP ensemble) -> NSGA-II over airfoil shape and wing area
for a light aircraft -> feasible / infeasible designs, the Pareto front of top speed vs CO2, and OpenFOAM verification.

Run:  uvicorn aero_optimizer.api:app --port 8092   (from apps/aero_optimizer)

Environment (all optional):
  ADO_USER / ADO_PASSWORD   HTTP Basic login on everything except /health
  ADO_MAX_HEAVY             concurrent optimizations per worker (default 2)
  ADO_MODEL                 surrogate bundle (default model/surrogate.pt)
"""
from __future__ import annotations

import dataclasses
import io
import json
import math
import os
import sys
import tempfile
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pinneapple_design.aero.aircraft import KT, Aircraft, Requirements
from pinneapple_design.aero.case import write_case
from pinneapple_design.aero.geometry import BOUNDS, LABELS, PARAMS, REFERENCE, outline, properties
from pinneapple_design.aero.mesh import GridSpec
from pinneapple_design.aero.optimize import AREA_BOUNDS, Engine, baseline, summarize
from pinneapple_design.aero import vlm as _vlm
from pinneapple_design.aero.aircraft3d import (BOUNDS3D, LABELS3D, PLAN, Requirements3D, airframe_from,
                                               baseline_x)
from pinneapple_design.aero.airframe import to_glb, to_stl, to_usda
from pinneapple_design.aero.optimize3d import Engine3D, summarize3d
from pinneapple_design.aero.airliner3d import (BOUNDS_AL, LABELS_AL, PLAN as PLAN_AL, Mission, RequirementsAL,
                                               airliner_from, baseline_x_al)
from pinneapple_design.aero.optimize_airliner import EngineAL, summarize_al

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

torch.set_num_threads(int(os.environ.get("ADO_THREADS", "1")))
VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
MODEL_DIR = os.path.join(HERE, "..", "model")
ENGINE = Engine.load(os.environ.get("ADO_MODEL", os.path.join(MODEL_DIR, "surrogate.pt")))
ENGINE3D = Engine3D(ENGINE)
ENGINE_AL = EngineAL(ENGINE)


def _json(name: str, default=None):
    p = os.path.join(MODEL_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else default


METRICS = _json("surrogate.metrics.json", {})
VERIFY = _json("verification.json", {"rounds": []})
VERIFY3D = _json("verification3d.json", {"designs": []})
DEFAULT3D = _json("default_search3d.json", None)
DATASET = _json("dataset.json", {})

app = FastAPI(title="Aircraft Design Optimizer", version=VERSION,
              description="Optimize a light aircraft's wing section and wing area for top speed and CO2 per 100 km "
                          "under stall, structure and trim requirements, on surrogates trained on OpenFOAM runs.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="ADO")
_HEAVY = BusyLimiter("ADO_MAX_HEAVY")


class AircraftIn(BaseModel):
    mass: float = Field(1100, ge=300, le=5000)
    wing_area: float = Field(16.2, ge=6, le=40)
    aspect_ratio: float = Field(7.5, ge=4, le=14)
    oswald: float = Field(0.8, ge=0.5, le=1.0)
    cd0_rest: float = Field(0.020, ge=0.005, le=0.06)
    power_kw: float = Field(120, ge=30, le=600)
    prop_efficiency: float = Field(0.8, ge=0.5, le=0.92)
    bsfc: float = Field(0.30, ge=0.15, le=0.5)
    cruise_speed: float = Field(55, ge=25, le=120)
    cruise_altitude: float = Field(2000, ge=0, le=6000)


class RequirementsIn(BaseModel):
    max_stall_speed_kt: float = Field(61, ge=35, le=120)
    min_thickness: float = Field(0.11, ge=0.04, le=0.2)
    max_abs_cm: float = Field(0.10, ge=0.01, le=0.3)
    max_cruise_cl_ratio: float = Field(0.70, ge=0.3, le=0.95)


class OptimizeIn(BaseModel):
    aircraft: AircraftIn = AircraftIn()
    requirements: RequirementsIn = RequirementsIn()
    area_min: float = Field(AREA_BOUNDS[0], ge=6, le=40)
    area_max: float = Field(AREA_BOUNDS[1], ge=6, le=40)
    population: int = Field(80, ge=16, le=160)
    generations: int = Field(60, ge=5, le=150)
    seed: int = 0


class DesignIn(BaseModel):
    x: List[float]
    aircraft: AircraftIn = AircraftIn()
    requirements: RequirementsIn = RequirementsIn()
    fields: bool = True


def _ac(a: AircraftIn) -> Aircraft:
    return Aircraft(**a.model_dump())


def _req(r: RequirementsIn) -> Requirements:
    return Requirements(max_stall_speed=r.max_stall_speed_kt * KT, min_thickness=r.min_thickness,
                        max_abs_cm=r.max_abs_cm, max_cruise_cl_ratio=r.max_cruise_cl_ratio)


def _clean(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 6)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return _clean(o.item())
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _detail(rec: Dict[str, Any]) -> Dict[str, Any]:
    d = {k: rec.get(k) for k in ("x", "shape", "wing_area", "props", "feasible", "valid", "violations", "trust", "unc",
                                 "vmax_kt", "co2_100km", "fuel_l_100km", "v_stall_kt", "cruise_ld", "cruise_cl",
                                 "cruise_alpha", "cruise_cm", "clmax", "mass", "checks", "polar", "penalty")}
    d["outline"] = outline(np.array(rec["shape"]))
    return d


def _picks(designs, front):
    if not front:
        return {}
    v = np.array([designs[i]["vmax_kt"] for i in front])
    c = np.array([designs[i]["co2_100km"] for i in front])
    s = np.array([designs[i]["v_stall_kt"] for i in front])
    n = lambda a, up: (a - a.min()) / (np.ptp(a) or 1) if up else (a.max() - a) / (np.ptp(a) or 1)  # noqa: E731
    # balanced: closest to the ideal point (fastest and lowest stall at once), CO2 follows speed
    knee = front[int(np.argmin((1 - n(v, True)) ** 2 + (1 - n(s, False)) ** 2))]
    return {"fastest": front[int(np.argmax(v))], "greenest": front[int(np.argmin(c))],
            "safest": front[int(np.argmin(s))], "balanced": knee}


def scope() -> dict:
    m = METRICS
    val = []
    if m.get("mlp_test"):
        val.append(f"Surrogate (MLP ensemble) on {m['counts']['test_shapes']} airfoils it never saw, all angles: "
                   f"drag within {m['mlp_test']['Cd']['mape']:.1f} % on average, lift within {m['mlp_test']['Cl']['mae']:.3f}")
    if m.get("gnn_test"):
        val.append(f"MeshGraphNet on the same unseen airfoils: drag within {m['gnn_test']['Cd']['mape']:.1f} %, "
                   f"lift within {m['gnn_test']['Cl']['mae']:.3f}, plus the pressure and velocity fields")
    val += DATASET.get("validation", [])
    for r in VERIFY.get("rounds", []):
        if r.get("summary"):
            val.append(r["summary"])
    return {"validated": val, "title": "Scope & validation", "noun": "simplification",
            "items_title": "What the numbers include",
            "band": "2D sections from RANS (k-ω SST, fully turbulent, Re 4 million); aircraft numbers from a "
                    "first-order performance model. Ranking designs is reliable; absolute values are estimates.",
            "tags": {"conservative": "Penalises the design", "optimistic": "Can flatter the design", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Fully turbulent flow", "effect": "conservative",
                 "detail": "No laminar run: real smooth wings have 20-40 % less section drag at cruise, and "
                           "laminar-flow shapes would gain more than shown here.",
                 "today": "Compare designs with each other, not with wind-tunnel drag of laminar sections.",
                 "planned": "Transition model (k-ω SST γ-Reθ) runs for the final candidates."},
                {"topic": "2D section, one Reynolds number", "effect": "check",
                 "detail": "The wing is the section polar plus elliptic induced drag (Oswald factor); stall uses "
                           "0.9 × section cl_max, at Re 4×10⁶ for every speed.",
                 "today": "Treat stall speed as ±2 kt; use the Oswald factor of your planform.",
                 "planned": "3D wing runs (lifting line → OpenFOAM 3D) for the chosen design."},
                {"topic": "Stall near and past cl_max", "effect": "optimistic",
                 "detail": "Steady RANS around stall is the least accurate region (runs flagged unsteady are "
                           "averaged); cl_max between sampled angles is interpolated.",
                 "today": "Keep a margin on the stall speed requirement.", "planned": "URANS for the stall points."},
                {"topic": "Wing mass with area", "effect": "check",
                 "detail": "6 kg per m² of wing change; no structural sizing, no fuel-volume check beyond thickness.",
                 "today": "Set the minimum thickness from your spar design.", "planned": "Spar sizing per design."},
                {"topic": "Surrogate trust", "effect": "check",
                 "detail": "Five networks vote; when they disagree (drag spread > 6 % or cl_max spread > 0.08) the "
                           "design is marked low-trust and kept out of the front.",
                 "today": "Verify the chosen design with OpenFOAM (download the ready case).",
                 "planned": "Live OpenFOAM verification and retraining from the app."},
            ]}


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "gnn": ENGINE.gnn is not None}


@app.get("/api/meta")
def meta():
    refs = {k: {"x": np.r_[v, 16.2].tolist(), "outline": outline(v), "props": properties(v)} for k, v in REFERENCE.items()}
    return _clean({"params": [{"name": p, "label": LABELS[p], "lo": b[0], "hi": b[1]} for p, b in zip(PARAMS, BOUNDS)],
                   "area_bounds": AREA_BOUNDS, "aircraft": dataclasses.asdict(Aircraft()),
                   "requirements": {"max_stall_speed_kt": 61, "min_thickness": 0.11, "max_abs_cm": 0.10,
                                    "max_cruise_cl_ratio": 0.70},
                   "references": refs, "metrics": {k: v for k, v in METRICS.items() if not k.endswith(("points", "history"))},
                   "dataset": {k: v for k, v in DATASET.items() if k != "points"}, "verification": VERIFY,
                   "scope": scope(), "gnn": ENGINE.gnn is not None, "meta": ENGINE.meta})


@app.get("/api/surrogate")
def surrogate():
    """Parity data (true vs predicted on unseen airfoils) for both surrogates, and the training history."""
    return _clean({k: v for k, v in METRICS.items()})


@app.get("/api/dataset")
def dataset():
    return _clean(DATASET)


@app.post("/api/optimize")
def optimize(body: OptimizeIn):
    if body.area_max <= body.area_min:
        raise HTTPException(422, "area_max must be larger than area_min")
    ac, req = _ac(body.aircraft), _req(body.requirements)
    with _HEAVY:
        res = ENGINE.search(ac, req, population=body.population, generations=body.generations, seed=body.seed,
                            area_bounds=(body.area_min, body.area_max))
        base = baseline(ENGINE, ac, req)
    D = res["designs"]
    front = res["pareto"]
    picks = _picks(D, front)
    keep = set(front) | set(picks.values())
    reasons: Dict[str, int] = {}
    for r in D:
        if not r.get("feasible"):
            for v in r.get("violations") or ["no solution"]:
                reasons[v] = reasons.get(v, 0) + 1
        elif r.get("trust") == "low":
            reasons["low surrogate trust"] = reasons.get("low surrogate trust", 0) + 1
    return _clean({"designs": [summarize(r) for r in D], "pareto": front, "picks": picks,
                   "details": {str(i): _detail(D[i]) for i in keep}, "baseline": _detail(base),
                   "evaluations": res["evaluations"], "seconds": res["seconds"],
                   "feasible": sum(1 for r in D if r.get("feasible")), "reasons": reasons,
                   "aircraft": dataclasses.asdict(ac), "requirements": req.as_dict()})


@app.post("/api/design")
def design(body: DesignIn):
    if len(body.x) != 7:
        raise HTTPException(422, "x needs 7 numbers: T0 T1 T2 M0 M1 M2 wing_area")
    x = np.array(body.x, float)
    if (x[:6] < BOUNDS[:, 0] - 1e-9).any() or (x[:6] > BOUNDS[:, 1] + 1e-9).any():
        raise HTTPException(422, "shape weights outside the trained range " + json.dumps(BOUNDS.tolist()))
    ac, req = _ac(body.aircraft), _req(body.requirements)
    rec = ENGINE.evaluate(x[None], ac, req)[0]
    out = _detail(rec)
    if body.fields and rec.get("valid") and ENGINE.gnn is not None:
        a_cruise = rec.get("cruise_alpha")
        a_cruise = float(np.clip(a_cruise if a_cruise == a_cruise and a_cruise is not None else 2.0, -2, 14))
        k = int(np.argmax(rec["polar"]["cl"]))
        a_high = float(min(rec["polar"]["alpha"][k], 12.0))
        g = ENGINE.gnn_predict(x[:6], [a_cruise, a_high])
        nj, ni = g["shape2"]
        F = g["fields"]
        gs, isel = ENGINE.layout.grid, ENGINE.layout.i_sel
        side = ["t" if i < gs.n_te else "l" if i < gs.n_te + gs.n_surf else "u" for i in isel]
        out["flow"] = {"nj": nj, "ni": ni, "side": side, "x": g["pos"][:, 0].round(5).tolist(), "y": g["pos"][:, 1].round(5).tolist(),
                       "cases": [{"alpha": a, "label": lab, "p": F[i, :, 0].round(4).tolist(),
                                  "speed": np.hypot(F[i, :, 1], F[i, :, 2]).round(4).tolist(),
                                  "nut": F[i, :, 3].round(3).tolist(),
                                  "cl": float(g["cl"][i]), "cd": float(g["cd"][i]), "cm": float(g["cm"][i]),
                                  "cp_wall": (F[i, :ni, 0] / 0.5).round(4).tolist()}
                                 for i, (a, lab) in enumerate(((a_cruise, "cruise"), (a_high, "near cl_max")))]}
        # second opinion: the graph network's coefficients vs the MLP ensemble at the same angles
        co = ENGINE.mlp_coefficients(x[None, :6], np.array([a_cruise, a_high]))
        out["agreement"] = [{"alpha": a, "gnn": {"cl": float(g["cl"][i]), "cd": float(g["cd"][i])},
                             "mlp": {"cl": float(co["cl"][0][i]), "cd": float(co["cd"][0][i])},
                             "cd_diff_pct": float(100 * (g["cd"][i] - co["cd"][0][i]) / co["cd"][0][i])}
                            for i, a in enumerate((a_cruise, a_high))]
    return _clean(out)


COEF_SCRIPT = r'''"""Lift, drag and pitching moment from this case's last time directory (numpy only).
Run after simpleFoam:  python coefficients.py
"""
import glob, json, math, os, re
import numpy as np

info = json.load(open("grid.json"))
ni, nj, nu, alpha = info["ni"], info["nj"], info["nu"], math.radians(info["alpha"])

def internal(path):
    t = open(path).read()
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", t)
    body = t[m.end(): t.index("\n)\n", m.end())]
    a = np.array(body.replace("(", " ").replace(")", " ").split(), float)
    return a.reshape(int(m.group(2)), -1)

pts = open("constant/polyMesh/points").read()
pts = np.array(pts[pts.index("(", pts.index("\n(")) + 1: pts.rindex(")")].replace("(", " ").replace(")", " ").split(), float).reshape(-1, 3)
P = pts[: len(pts) // 2, :2].reshape(nj + 1, ni, 2)
last = max((d for d in os.listdir(".") if re.fullmatch(r"[0-9.]+", d) and float(d) > 0), key=float)
p = internal(os.path.join(last, "p"))[:ni, 0]
U = internal(os.path.join(last, "U"))[:ni, :2]
w, w1 = P[0], np.roll(P[0], -1, 0)
e = w1 - w
L = np.linalg.norm(e, axis=1)
n = np.stack([e[:, 1], -e[:, 0]], 1) / L[:, None]                 # into the body
c = 0.5 * (w + w1)
q = np.concatenate([P, P[:, :1]], 1)
cc = 0.25 * (q[0, :-1] + q[0, 1:] + q[1, :-1] + q[1, 1:])
d = np.abs(((cc - c) * n).sum(1))
ut = U - (U * n).sum(1, keepdims=True) * n
f = (p[:, None] * n + nu * ut / d[:, None]) * L[:, None]
F = f.sum(0)
drag, lift = np.array([math.cos(alpha), math.sin(alpha)]), np.array([-math.sin(alpha), math.cos(alpha)])
r = c - [0.25, 0]
Mz = (r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]).sum()
print(f"time {last}:  Cl = {F @ lift / 0.5:.4f}   Cd = {F @ drag / 0.5:.5f}   Cm(c/4, nose-up +) = {-Mz / 0.5:.4f}")
'''


@app.get("/api/openfoam-case")
def openfoam_case(x: str = Query(..., description="7 comma-separated numbers: T0,T1,T2,M0,M1,M2,wing_area"),
                  alpha: float = Query(4.0, ge=-5, le=20)):
    """A ready OpenFOAM case (mesh + setup) for this airfoil at this angle, to verify a design yourself."""
    try:
        v = np.array([float(s) for s in x.split(",")])[:6]
    except ValueError:
        raise HTTPException(422, "x: comma-separated numbers")
    if len(v) != 6 or (v < BOUNDS[:, 0] - 1e-9).any() or (v > BOUNDS[:, 1] + 1e-9).any():
        raise HTTPException(422, "x: 6 shape weights inside the trained range (+ wing area)")
    g = GridSpec()
    re_ = float(ENGINE.meta.get("reynolds", 4e6))
    with tempfile.TemporaryDirectory() as tmp:
        case = os.path.join(tmp, "case")
        write_case(case, v, alpha, reynolds=re_, iterations=1200, grid=g)
        with open(os.path.join(case, "grid.json"), "w") as f:
            json.dump({"ni": g.ni, "nj": g.nj, "nu": 1 / re_, "alpha": alpha, "shape": v.tolist()}, f)
        with open(os.path.join(case, "coefficients.py"), "w") as f:
            f.write(COEF_SCRIPT)
        with open(os.path.join(case, "Allrun"), "w") as f:
            f.write("#!/bin/sh\n# OpenFOAM v1912 or newer (ESI); about 1 minute on one core\nsimpleFoam > log.simpleFoam 2>&1\n"
                    "python3 coefficients.py\n")
        os.chmod(os.path.join(case, "Allrun"), 0o755)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(case):
                for fn in files:
                    full = os.path.join(root, fn)
                    z.write(full, os.path.join(f"airfoil_a{alpha:g}", os.path.relpath(full, case)))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="airfoil_a{alpha:g}_openfoam.zip"'})




# ====================================================================== the airliner (3D)
class MissionIn(BaseModel):
    passengers: int = Field(180, ge=50, le=400)
    range_km: float = Field(4800, ge=500, le=12000)
    cruise_altitude: float = Field(10668, ge=6000, le=13000)
    tsfc: float = Field(1.61e-5, ge=0.8e-5, le=2.5e-5)


class RequirementsALIn(BaseModel):
    max_span: float = Field(36.0, ge=20, le=80)
    max_vref_kt: float = Field(145, ge=100, le=180)
    buffet_margin: float = Field(1.30, ge=1.0, le=1.6)
    fuel_volume_margin: float = Field(1.0, ge=0.5, le=2.0)
    min_static_margin: float = Field(0.05, ge=-0.1, le=0.4)
    max_static_margin: float = Field(0.35, ge=0.05, le=0.7)
    max_stall_station: float = Field(0.80, ge=0.1, le=1.0)
    max_tail_incidence: float = Field(4.0, ge=0.5, le=15)
    min_thickness: float = Field(0.09, ge=0.04, le=0.2)


class Optimize3DIn(BaseModel):
    mission: MissionIn = MissionIn()
    requirements: RequirementsALIn = RequirementsALIn()
    population: int = Field(48, ge=16, le=96)
    generations: int = Field(40, ge=5, le=80)
    seed: int = 0


class AircraftDesignIn(BaseModel):
    x: List[float]
    mission: MissionIn = MissionIn()
    requirements: RequirementsALIn = RequirementsALIn()


def _mission(m: MissionIn) -> Mission:
    return Mission(**m.model_dump())


def _reqal(r: RequirementsALIn) -> RequirementsAL:
    return RequirementsAL(**r.model_dump())


def _x13(x) -> np.ndarray:
    v = np.asarray(x, float)
    if v.shape != (13,):
        raise HTTPException(422, "x needs 13 numbers: T0 T1 T2 M0 M1 M2 wing_area aspect_ratio taper sweep_qc twist cruise_mach wing_x")
    if (v < BOUNDS_AL[:, 0] - 1e-6).any() or (v > BOUNDS_AL[:, 1] + 1e-6).any():
        raise HTTPException(422, "x outside the design space " + json.dumps(BOUNDS_AL.tolist()))
    return v


def _verified(x: np.ndarray):
    for d in VERIFY3D.get("designs", []):
        if len(d["x"]) == len(x) and np.allclose(np.asarray(d["x"]), x, atol=1e-4):
            return d
    return None


def _detail_al(r: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("x", "props", "plan", "feasible", "valid", "violations", "trust", "unc", "co2_pkm", "mach", "speed_kt",
            "vref_kt", "fuel_kg", "fuel_l_100pkm", "mtow", "oew", "wing_mass", "tail_mass", "cruise_cl", "cruise_cd",
            "cruise_ld", "cruise_alpha", "tail_incidence", "oswald", "mdd", "clmax_clean", "clmax_land", "stall_station",
            "static_margin", "x_cg", "x_np", "span", "fuel_volume_m3", "fuel_ratio", "drag_breakdown", "checks", "loading",
            "polar", "penalty")
    d = {k: r.get(k) for k in keys}
    if r.get("valid"):
        d["outline"] = outline(np.array(r["x"][:6]))
        d["summary"] = airliner_from(np.array(r["x"])).summary()
    return d


def _picks_al(D, front):
    if not front:
        return {}
    c = np.array([D[i]["co2_pkm"] for i in front])
    m = np.array([D[i]["mach"] for i in front])
    v = np.array([D[i]["vref_kt"] for i in front])
    n = lambda a, up: (a - a.min()) / (np.ptp(a) or 1) if up else (a.max() - a) / (np.ptp(a) or 1)  # noqa: E731
    return {"greenest": front[int(np.argmin(c))], "fastest": front[int(np.argmax(m))], "shortfield": front[int(np.argmin(v))],
            "balanced": front[int(np.argmin((1 - n(c, False)) ** 2 + (1 - n(m, True)) ** 2 + 0.5 * (1 - n(v, False)) ** 2))]}


def run_search3d(body: Optimize3DIn, progress=None) -> Dict[str, Any]:
    mi, req = _mission(body.mission), _reqal(body.requirements)
    res = ENGINE_AL.search(mi, req, population=body.population, generations=body.generations, seed=body.seed,
                           progress=progress)
    D, front = res["designs"], res["pareto"]
    picks = _picks_al(D, front)
    keep = set(picks.values())
    fine = {i: ENGINE_AL.evaluate(np.array(D[i]["x"])[None], mi, req, fine=True)[0] for i in keep}
    base = ENGINE_AL.evaluate(baseline_x_al()[None], mi, req, fine=True)[0]
    reasons: Dict[str, int] = {}
    for r in D:
        if not r.get("feasible"):
            for v in r.get("violations") or ["no solution"]:
                reasons[v] = reasons.get(v, 0) + 1
        elif r.get("trust") == "low":
            reasons["low surrogate trust"] = reasons.get("low surrogate trust", 0) + 1
    return _clean({"designs": [summarize_al(r) for r in D], "pareto": front, "picks": picks,
                   "details": {str(i): _detail_al(fine[i]) for i in keep}, "baseline": _detail_al(base),
                   "evaluations": res["evaluations"], "seconds": res["seconds"],
                   "feasible": sum(1 for r in D if r.get("feasible")), "reasons": reasons,
                   "mission": dataclasses.asdict(mi), "requirements": req.as_dict()})


JOBS: Dict[str, Dict[str, Any]] = {}
_POOL = None


@app.post("/api/optimize3d")
def optimize3d(body: Optimize3DIn):
    """Start an airliner search; poll /api/job/{id}. The default inputs return the stored result at once."""
    import concurrent.futures
    import uuid
    global _POOL
    def _norm(o):
        return json.dumps(_clean(json.loads(json.dumps(o))), sort_keys=True, default=lambda v: float(f"{v:.9g}"))
    rnd = lambda d: {k: (rnd(v) if isinstance(v, dict) else round(v, 12) if isinstance(v, float) else v) for k, v in d.items()}  # noqa: E731
    if DEFAULT3D and rnd(body.model_dump()) == rnd(Optimize3DIn().model_dump()):
        return {"job": "default", "status": "done", "result": DEFAULT3D}
    if sum(1 for j in JOBS.values() if j["status"] == "running") >= int(os.environ.get("ADO_MAX_HEAVY", "2")):
        raise HTTPException(429, "Server busy with other searches -- please retry in a minute.")
    _POOL = _POOL or concurrent.futures.ThreadPoolExecutor(int(os.environ.get("ADO_MAX_HEAVY", "2")))
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {"status": "running", "done": 0, "total": body.generations}

    def prog(k, n):
        JOBS[jid].update(done=k, total=n)

    def work():
        try:
            JOBS[jid].update(status="done", result=run_search3d(body, prog))
        except Exception as e:                                          # noqa: BLE001
            JOBS[jid].update(status="error", error=str(e))
    _POOL.submit(work)
    for k in [k for k, j in JOBS.items() if j["status"] != "running"][:-20]:
        JOBS.pop(k, None)
    return {"job": jid, "status": "running"}


@app.get("/api/job/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "unknown job")
    return j


@app.post("/api/aircraft")
def aircraft(body: AircraftDesignIn):
    """One airliner: fine-lattice cruise and low-speed numbers, checks, span loading, potential-flow streamlines at
    cruise, and the OpenFOAM 3D fields when this design was run."""
    x = _x13(body.x)
    mi, req = _mission(body.mission), _reqal(body.requirements)
    r = ENGINE_AL.evaluate(x[None], mi, req, fine=True)[0]
    out = _detail_al(r)
    if not r.get("valid"):
        return _clean(out)
    af = airliner_from(x)
    sol = _vlm.solve(af, r["x_cg"], nc=4, ns_wing=12, ns_tail=5)
    a, it = math.radians(r["cruise_alpha"]), math.radians(r["tail_incidence"])
    b2, zw = af.span / 2, af.wing_z_root()
    ys = np.r_[np.linspace(2.5, b2 - 2.0, 8), np.linspace(b2 - 1.4, b2 + 0.9, 9)]
    seeds = np.c_[np.array([af._wing_point(min(1, y / b2), 0)[0] - 4.0 for y in ys]), ys,
                  np.array([af._wing_point(min(1, y / b2), 0)[2] - 0.35 for y in ys])]
    S = _vlm.streamlines(sol, a, it, seeds, steps=230, h=0.25, core=0.25)
    lines = [L[::2].round(3).tolist() for L in S]
    out["lines_vlm"] = lines + [[[p[0], -p[1], p[2]] for p in L] for L in lines]
    out["ground_z"] = af.ground_z()
    v = _verified(x)
    if v:
        out["openfoam"] = {k: v[k] for k in ("id", "label", "alpha", "speed", "cells", "CL", "CD", "CD_pressure",
                                             "CD_friction", "model", "render") if k in v}
        out["lines_cfd"] = v.get("lines", [])
    return _clean(out)


def _parts_for(x: np.ndarray, detail="high"):
    return airliner_from(x).build(detail)


@app.get("/api/aircraft.glb")
def aircraft_glb(x: str, cp: bool = False):
    v = _x13([float(s) for s in x.split(",")])
    parts = _parts_for(v)
    scalars = None
    ver = _verified(v)
    if cp and ver and os.path.exists(os.path.join(MODEL_DIR, "cfd3d", f"{ver['id']}_cp.npz")):
        z = np.load(os.path.join(MODEL_DIR, "cfd3d", f"{ver['id']}_cp.npz"))
        from pinneapple_design.aero.case3d import surface_scalars
        scalars = surface_scalars(airliner_from(v), parts, {"xyz": z["xyz"], "cp": z["cp"].astype(np.float32)})
    return Response(to_glb(parts, scalars), media_type="model/gltf-binary",
                    headers={"Content-Disposition": 'inline; filename="airliner.glb"', "Cache-Control": "max-age=3600"})


@app.get("/api/aircraft.usda")
def aircraft_usda(x: str):
    v = _x13([float(s) for s in x.split(",")])
    return Response(to_usda(_parts_for(v), "airliner"), media_type="text/plain",
                    headers={"Content-Disposition": 'attachment; filename="airliner.usda"'})


@app.get("/api/aircraft.stl")
def aircraft_stl(x: str):
    v = _x13([float(s) for s in x.split(",")])
    parts = [p for p in _parts_for(v, "cfd") if p.aero]
    return Response(to_stl(parts, "airliner"), media_type="model/stl",
                    headers={"Content-Disposition": 'attachment; filename="airliner_clean.stl"'})


@app.get("/api/meta3d")
def meta3d():
    return _clean({"plan": [{"name": p, "label": LABELS_AL[p], "lo": b[0], "hi": b[1]} for p, b in zip(PLAN_AL, BOUNDS_AL[6:])],
                   "requirements": RequirementsALIn().model_dump(), "mission": MissionIn().model_dump(),
                   "baseline_x": baseline_x_al(), "verification": VERIFY3D,
                   "renders": sorted(os.listdir(os.path.join(MODEL_DIR, "renders")))
                   if os.path.isdir(os.path.join(MODEL_DIR, "renders")) else []})


app.mount("/renders", StaticFiles(directory=os.path.join(MODEL_DIR, "renders"), check_dir=False), name="renders")

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
