"""Measured demonstration of the physics query engine on the chip package (track 2).

:func:`query_demo` builds an engine over a package, asks thousands of point, region and interface queries under the
package's own power map, under maps that are combinations of solved ones, under an unseen map and with loose
accuracy targets, and compares every answer with the fine finite-volume reference of the same conditions. It reports
latency per query (by method and kind), the method the planner picked, the error against the reference and whether
the rigorous intervals contain it. Run it through :func:`pinneapple_condense.benchmark.isolated` for peak memory.
"""
from __future__ import annotations

import time

import numpy as np

from . import fv
from .package import hotspot_power, uniform_package
from .query import package_engine
from .substructure import reference_observables


def _stats(xs):
    a = np.asarray(xs, float)
    if not len(a):
        return {}
    return {"n": int(len(a)), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "mean": float(a.mean())}


def query_demo(nb: int = 48, m: int = 12, n_points: int = 2000, n_interior: int = 100, seed: int = 0,
               reference: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    pkg = uniform_package((nb, nb), seed=seed)
    st = pkg.stack
    n = st.n
    t0 = time.perf_counter()
    eng, model = package_engine(pkg, m=m, reference_cells=0)        # the reference is the judge, not a method
    build = time.perf_counter() - t0
    P0 = pkg.power
    P1 = hotspot_power(st, pkg.shape, P0.sum(), seed=seed + 11, hotspots=2)
    P2 = hotspot_power(st, pkg.shape, P0.sum(), seed=seed + 12, hotspots=4, width=0.05)
    Pc = 0.5 * P0 + 0.3 * P1 + 0.2 * P2                               # in the span of solved maps
    Pn = hotspot_power(st, pkg.shape, 1.3 * P0.sum(), seed=seed + 13, hotspots=1, width=0.04)   # unseen
    refs = {}
    ref_s = []
    if reference:
        for name, P in (("P0", P0), ("Pc", Pc), ("Pn", Pn)):
            t1 = time.perf_counter()
            T, info = fv.solve_package(pkg, P)
            ref_s.append(time.perf_counter() - t1)
            refs[name] = (T, reference_observables(pkg, T))
    out = {"blocks": nb * nb, "cells": pkg.n_cells, "m": m, "build_s": build, "timings": dict(model.timings),
           "reference_solve_s": ref_s, "phases": {}}
    pts = rng.uniform(0, 1, (n_points, 2))
    ipts = rng.uniform(0, 1, (n_interior, 3))

    def truth_point(name, u, v, w=None):
        T = refs[name][0]
        ix, iy = min(int(u * nb * n), nb * n - 1), min(int(v * nb * n), nb * n - 1)
        if w is None:
            return float(T[ix, iy, st.active_k])
        die = st.layer_slices()["die"]
        return float(T[ix, iy, die.start + min(int(w * (die.stop - die.start)), die.stop - die.start - 1)])

    def phase(label, name, P, queries, accuracy=None, rigorous=False):
        rows = []
        for qty, where, truth in queries:
            a = eng.query(qty, where, conditions={"power": P}, accuracy=accuracy, rigorous=rigorous)
            r = {"method": a.method, "kind": a.kind, "seconds": a.seconds, "value": a.value,
                 "uncertainty": a.uncertainty, "rigorous": a.rigorous}
            if reference and truth is not None:
                tv = truth()
                r["error"] = abs(a.value - tv)
                if a.interval is not None:
                    r["inside"] = bool(a.interval[0] - 1e-9 <= tv <= a.interval[1] + 1e-9)
                r["within_uncertainty"] = bool(r["error"] <= a.uncertainty + 1e-9)
            rows.append(r)
        methods = sorted({r["method"] for r in rows})
        summ = {"queries": len(rows), "accuracy_target": accuracy, "rigorous_requested": rigorous}
        for mth in methods:
            rr = [r for r in rows if r["method"] == mth]
            s = {"count": len(rr), "kind": sorted({r["kind"] for r in rr}), "latency_s": _stats([r["seconds"] for r in rr]),
                 "uncertainty": _stats([r["uncertainty"] for r in rr])}
            if reference:
                s["error_K_or_W"] = _stats([r["error"] for r in rr if "error" in r])
                ins = [r["inside"] for r in rr if "inside" in r]
                if ins:
                    s["interval_contains_truth"] = f"{sum(ins)}/{len(ins)}"
                wu = [r["within_uncertainty"] for r in rr if "within_uncertainty" in r]
                s["error_within_uncertainty"] = f"{sum(wu)}/{len(wu)}"
            summ[mth] = s
        out["phases"][label] = summ

    def pq(name):
        return [("T", f"die@{u:.6f},{v:.6f}", (lambda u=u, v=v: truth_point(name, u, v))) for u, v in pts]

    def iq(name):
        return [("T", f"die@{u:.6f},{v:.6f},{w:.6f}", (lambda u=u, v=v, w=w: truth_point(name, u, v, w)))
                for u, v, w in ipts]

    def rq(name):
        o = refs[name][1] if reference else None
        return [("Tj_max", "region:die", (lambda: float(o["Tj"].max())) if o else None),
                ("q", "interface:die/tim", (lambda: float(o["q_tim"].sum())) if o else None),
                ("q", "lid:top", (lambda: float(o["q_top"].sum())) if o else None),
                ("q", f"ubump[{nb // 2},{nb // 3}]", (lambda: float(o["q_ubump"][nb // 2, nb // 3])) if o else None)]

    # 1. the package's own map: solved at build (calibration), every query is a read
    phase("own map: junction points", "P0", P0, pq("P0"))
    phase("own map: interior points (block rebuilt on demand)", "P0", P0, iq("P0"))
    phase("own map: regions and interfaces", "P0", P0, rq("P0"))
    # 2. two more maps get solved (the user asked for them), then a combination of the three: superposition
    for P in (P1, P2):
        eng.query("Tj_max", "region:die", conditions={"power": P}, accuracy=1e-9)
    phase("combination of solved maps: junction points", "Pc", Pc, pq("Pc")[: n_points // 4])
    phase("combination of solved maps: regions and interfaces", "Pc", Pc, rq("Pc"))
    # 3. an unseen map, a guaranteed interval asked for: the monotone bound answers without solving
    phase("unseen map, rigorous interval: junction points", "Pn", Pn, pq("Pn")[: n_points // 4], rigorous=True)
    phase("unseen map, rigorous interval: peak junction", "Pn", Pn, rq("Pn")[:1], rigorous=True)
    # 4. the same unseen map, tight target: one condensed solve, then reads
    phase("unseen map, accuracy 0.5 K: junction points", "Pn", Pn, pq("Pn"), accuracy=0.5)
    phase("unseen map, accuracy 0.5 K: regions and interfaces", "Pn", Pn, rq("Pn"), accuracy=0.5)
    out["log_methods"] = {k: sum(1 for e in eng.log if e["method"] == k) for k in {e["method"] for e in eng.log}}
    out["queries_total"] = len(eng.log)
    return out
