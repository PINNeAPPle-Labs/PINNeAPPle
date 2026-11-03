"""The standard terramechanics workflow (#1199): soil parameters -> single-wheel performance -> wheel design ->
re-verification, each step with its checks.

1. **Soil.** A database soil (:mod:`.soils`, every parameter with its source), or the soil identified from plate
   (bevameter) and shear-box tests through the inverse workflow
   (:func:`pinneapple_analysis.inverse_problems.workflow.inverse_workflow`): Bekker ``p = (k_c / b + k_phi) z^n``
   from plates of at least two widths (one width cannot separate ``k_c`` from ``k_phi``, and the identifiability
   step says so), Mohr-Coulomb with Janosi-Hanamoto ``tau = (c + sigma tan phi)(1 - exp(-j / K))`` from the shear
   curves. Without measurements, ``identify="synthetic"`` runs the same identification on tests generated from the
   database soil with the stated noise and checks that the intervals hold the soil that made them.
2. **Single-wheel performance** (:meth:`.wong_reece.WongReeceModel.steady_state`): sinkage from the load balance,
   drawbar pull, driving torque and DP/W against slip, with the Wong-Reece stress distribution (Bekker normal
   stress, Janosi-Hanamoto shear, the maximum-stress angle ``theta_m = (a0 + a1 s) theta_f``, slip-sinkage
   ``n = n0 + n1 s``). Checks: the vertical load balance, the quadrature converged (24 against 96 nodes) and DP rising
   with slip over the operating range.
3. **Wheel design**: rim radius, width and grouser height for a required drawbar pull at a slip limit, a sinkage
   limit and an optional torque limit, at minimum wheel mass (a rim, side discs and grousers of one material:
   :func:`wheel_mass`) or minimum torque at the required pull: a grid over the box, then SLSQP from the best
   feasible point.
4. **Re-verification**: the design re-run with the full model (96 quadrature nodes per arc piece, 80 bisection
   steps), its margins restated, the driving slip at the required pull found, and
   :func:`~pinneapple_analysis.verification.architecture_recommendation.recommend_architecture` asked whether a
   contact surrogate pays off; when it does, the zoo PINN is used only if it was trained on the same physics
   (:func:`.wong_reece.physics_mismatch`), otherwise the step says why it was not run.

**Verification against published tests**: :mod:`.testbed` runs the model on the bundled single-wheel points of
Ding et al. (2011) (smooth and grousered wheel, 80 N, planetary soil simulant, the cited soil), and the errors are
checked against the band stated in ``PUBLISHED_BAND``. The Wong & Reece (1967) test data are not bundled (the
original tables could not be read from the build environment); the Ding et al. points are the published
single-wheel case used here.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .soils import Soil, get_soil
from .wheel import Grousers, Wheel
from .wong_reece import WongReeceModel

__all__ = ["bekker_pressure", "janosi_shear", "synthetic_soil_tests", "identify_soil", "wheel_curves", "wheel_mass",
           "design_wheel", "terramechanics_workflow", "PUBLISHED_BAND", "DEFAULT_WHEEL", "DEFAULT_BOUNDS"]

#: the band the published single-wheel tests must be reproduced within, as RMSE relative to the scale named in
#: :func:`.testbed.validate` (drawbar pull and torque: the load and load x radius; sinkage: the largest measured).
#: Set from the measured errors of the cited soil (no calibration) on Ding et al. (2011), rounded up; the
#: calibrated slip-sinkage soil (:func:`.testbed.calibrate_holdout`) is reported next to it.
PUBLISHED_BAND = {"drawbar_pull_N": 0.10, "sinkage_m": 0.35, "torque_Nm": 0.10}
DEFAULT_WHEEL = {"radius": 0.25, "width": 0.20, "grouser_count": 24, "grouser_height": 0.01}
DEFAULT_BOUNDS = {"radius": [0.10, 0.40], "width": [0.08, 0.40], "grouser_height": [0.0, 0.03]}
DEFAULT_MATERIAL = {"density": 2700.0, "rim_thickness": 0.002, "side_thickness": 0.002, "side_fill": 0.3,
                    "grouser_thickness": 0.004, "hub_mass": 0.3}


# ---------------------------------------------------------------- 1. soil
def bekker_pressure(width: float, sinkage: float, k_c: float, k_phi: float, n: float) -> float:
    """Bekker plate pressure [Pa] ``(k_c / b + k_phi) z^n`` of a plate of width ``b`` [m] at sinkage ``z`` [m]."""
    return (k_c / width + k_phi) * max(sinkage, 0.0) ** n


def janosi_shear(normal_stress: float, displacement: float, c: float, phi_deg: float, K: float) -> float:
    """Janosi-Hanamoto shear stress [Pa] ``(c + sigma tan phi)(1 - exp(-j / K))`` at displacement ``j`` [m]."""
    return (c + normal_stress * math.tan(math.radians(phi_deg))) * (1.0 - math.exp(-max(displacement, 0.0) / K))


def synthetic_soil_tests(soil: Soil, widths: Sequence[float] = (0.03, 0.06, 0.12), z_max: float = 0.04,
                         normal_stresses: Sequence[float] = (5e3, 10e3, 20e3), j_max: float = 0.06, n_points: int = 8,
                         noise: float = 0.02, seed: int = 0) -> dict[str, Any]:
    """Plate and shear-box tests of ``soil`` with a relative Gaussian noise ``noise`` (the measurements an
    identification is checked on): ``{"plates": {"width", "sinkage", "pressure", "sigma"}, "shear": {...}}``."""
    rng = np.random.default_rng(seed)
    z = np.linspace(z_max / n_points, z_max, n_points)
    W, Z = np.meshgrid(np.asarray(widths, float), z, indexing="ij")
    p = np.array([bekker_pressure(b, zz, soil.k_c, soil.k_phi, soil.n0) for b, zz in zip(W.ravel(), Z.ravel(),
                                                                                         strict=True)])
    j = np.linspace(j_max / n_points, j_max, n_points)
    S, J = np.meshgrid(np.asarray(normal_stresses, float), j, indexing="ij")
    t = np.array([janosi_shear(s, jj, soil.c, soil.phi_deg, soil.K) for s, jj in zip(S.ravel(), J.ravel(),
                                                                                     strict=True)])
    sp, st = noise * p, noise * t
    return {"plates": {"width": W.ravel().tolist(), "sinkage": Z.ravel().tolist(),
                       "pressure": (p + sp * rng.standard_normal(p.size)).tolist(), "sigma": sp.tolist()},
            "shear": {"normal_stress": S.ravel().tolist(), "displacement": J.ravel().tolist(),
                      "shear_stress": (t + st * rng.standard_normal(t.size)).tolist(), "sigma": st.tolist()},
            "noise": noise, "truth": {"k_c": soil.k_c, "k_phi": soil.k_phi, "n0": soil.n0, "c": soil.c,
                                      "phi_deg": soil.phi_deg, "K": soil.K}}


def identify_soil(plates: Mapping[str, Any] | None = None, shear: Mapping[str, Any] | None = None, *,
                  base: Soil | str = "Ding2011-TYII", bootstrap: int = 20, n_recovery: int = 8,
                  seed: int = 0) -> dict[str, Any]:
    """Bekker (``k_c``, ``k_phi``, ``n``) from plate tests and Mohr-Coulomb / Janosi (``c``, ``phi``, ``K``) from
    shear-box tests through :func:`~pinneapple_analysis.inverse_problems.workflow.inverse_workflow` (identifiability
    first, then estimation with intervals, held-out rows and a synthetic recovery). The parameters not measured
    keep the ``base`` soil's values. Returns ``{"soil", "pressure", "shear", "checks"}``."""
    from pinneapple_analysis.inverse_problems.workflow import inverse_workflow
    base = get_soil(base) if isinstance(base, str) else base
    out: dict[str, Any] = {"checks": []}
    new: dict[str, float] = {}
    if plates:
        r = inverse_workflow(
            tool=bekker_pressure, output="pressure", measurements={
                "inputs": {"width": plates["width"], "sinkage": plates["sinkage"]}, "values": plates["pressure"],
                "sigma": plates.get("sigma", 1.0)},
            parameters={"k_c": {"lower": 1.0, "upper": 1e6, "initial": max(base.k_c, 100.0), "unit": "N/m^(n+1)"},
                        "k_phi": {"lower": 1e3, "upper": 1e8, "initial": base.k_phi, "unit": "N/m^(n+2)"},
                        "n": {"lower": 0.2, "upper": 2.0, "initial": base.n0}},
            bootstrap=bootstrap, n_recovery=n_recovery, seed=seed)
        out["pressure"] = _strip(r)
        out["checks"] += [{**c, "code": "soil.pressure." + c["code"].split(".", 1)[1]} for c in r["checks"]]
        if r.get("estimated"):
            new.update(k_c=r["estimate"]["k_c"], k_phi=r["estimate"]["k_phi"], n0=r["estimate"]["n"])
    if shear:
        r = inverse_workflow(
            tool=janosi_shear, output="shear", measurements={
                "inputs": {"normal_stress": shear["normal_stress"], "displacement": shear["displacement"]},
                "values": shear["shear_stress"], "sigma": shear.get("sigma", 1.0)},
            parameters={"c": {"lower": 1.0, "upper": 2e4, "initial": max(base.c, 50.0), "unit": "Pa"},
                        "phi_deg": {"lower": 10.0, "upper": 60.0, "initial": base.phi_deg, "unit": "deg"},
                        "K": {"lower": 1e-4, "upper": 0.1, "initial": base.K, "unit": "m"}},
            bootstrap=bootstrap, n_recovery=n_recovery, seed=seed)
        out["shear"] = _strip(r)
        out["checks"] += [{**c, "code": "soil.shear." + c["code"].split(".", 1)[1]} for c in r["checks"]]
        if r.get("estimated"):
            new.update(c=r["estimate"]["c"], phi_deg=r["estimate"]["phi_deg"], K=r["estimate"]["K"])
    soil = base
    if new:
        from dataclasses import replace
        src = dict(base.sources)
        for k in new:
            src[k] = "identified: inverse_workflow on the given plate / shear tests"
        ky = {"Ky": new["K"]} if "K" in new and base.Ky == base.K else {}
        soil = replace(base, name=f"{base.name} (identified)", description=f"{base.name} with parameters identified "
                       "from tests", sources=src, **new, **ky)
    out["soil"] = soil
    out["identified"] = sorted(new)
    return out


def _z_two_sided(alpha: float) -> float:
    """``z`` with ``P(|Z| > z) = alpha`` for a standard normal (bisection on erfc)."""
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if math.erfc(mid / math.sqrt(2)) > alpha else (lo, mid)
    return 0.5 * (lo + hi)


def _strip(r: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("estimated", "estimate", "intervals", "reduced_chi2", "recovery", "n_measurements", "identifiability")
    d = {k: r.get(k) for k in keys if k in r}
    if "identifiability" in d:
        d["identifiability"] = {k: v for k, v in d["identifiability"].items() if k != "suggestions"}
    return d


# ---------------------------------------------------------------- 2. single wheel
def _wheel(radius: float, width: float, grouser_count: int = 0, grouser_height: float = 0.0) -> Wheel:
    g = Grousers(count=int(grouser_count), height=float(grouser_height)) if grouser_height > 0 and grouser_count \
        else Grousers()
    return Wheel(radius=float(radius), width=float(width), grousers=g)


def wheel_curves(soil: Soil, wheel: Wheel, load: float, g: float, slips: Sequence[float], *,
                 n_gl: int = 24) -> dict[str, list[float]]:
    """Sinkage [m], drawbar pull [N], torque [N m], DP/W and the load the stresses carry [N] against slip."""
    m = WongReeceModel(soil, wheel, g=g, n_gl=n_gl)
    s = np.asarray(slips, float)
    r = m.steady_state(np.full(s.shape, float(load)), s)
    return {"slip": s.tolist(), "sinkage_m": np.asarray(r["sinkage"]).tolist(),
            "drawbar_pull_N": np.asarray(r["drawbar_pull"]).tolist(), "torque_Nm": np.asarray(r["torque"]).tolist(),
            "dp_over_w": np.asarray(r["dp_over_w"]).tolist(), "load_carried_N": np.asarray(r["load"]).tolist()}


# ---------------------------------------------------------------- 3. design
def wheel_mass(radius: float, width: float, grouser_height: float = 0.0, grouser_count: int = 0,
               **material: float) -> float:
    """Mass [kg] of a rigid wheel: a cylindrical rim ``2 pi R b t_rim``, two side discs ``pi R^2 t_side`` filled to
    ``side_fill`` (spokes), ``grouser_count`` lugs ``h b t_g`` and a hub, all of density ``density``
    (:data:`DEFAULT_MATERIAL`: aluminium, 2 mm rim and sides, 30 % filled sides, 4 mm lugs, 0.3 kg hub; a stated
    assumption, change it with keywords)."""
    p = {**DEFAULT_MATERIAL, **material}
    rho = p["density"]
    rim = 2 * math.pi * radius * width * p["rim_thickness"]
    sides = 2 * math.pi * radius ** 2 * p["side_thickness"] * p["side_fill"]
    lugs = grouser_count * grouser_height * width * p["grouser_thickness"]
    return float(rho * (rim + sides + lugs) + p["hub_mass"])


def _evaluate(soil, g, load, x, count, max_slip, n_gl):
    whl = _wheel(x["radius"], x["width"], count, x["grouser_height"])
    m = WongReeceModel(soil, whl, g=g, n_gl=n_gl)
    r = m.steady_state(float(load), float(max_slip))
    return {"drawbar_pull_N": float(r["drawbar_pull"]), "sinkage_m": float(r["sinkage"]),
            "torque_Nm": float(r["torque"]), "model": m}


def _slip_for_pull(model: WongReeceModel, load: float, pull: float, s_hi: float, iters: int = 40) -> float:
    """Driving slip at which the drawbar pull equals ``pull`` (bisection on [0, s_hi]); nan if not reached."""
    f = lambda s: float(model.steady_state(load, s)["drawbar_pull"]) - pull  # noqa: E731
    lo, hi = 0.0, float(s_hi)
    if f(hi) < -1e-4 * abs(pull):                         # the design sits on the limit: allow round-off
        return float("nan")
    if f(hi) < 0:
        return hi
        return float("nan")
    if f(lo) >= 0:
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def design_wheel(soil: Soil, *, load: float, g: float, required_pull: float, max_slip: float = 0.3,
                 max_sinkage: float = 0.05, max_torque: float | None = None, grouser_count: int = 24,
                 bounds: Mapping[str, Sequence[float]] | None = None, objective: str = "mass",
                 grid: int = 7, n_gl: int = 16, material: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Rim radius, width and grouser height with ``DP(max_slip) >= required_pull``, sinkage at ``max_slip`` below
    ``max_sinkage`` and (when given) torque below ``max_torque``, at minimum ``objective`` ("mass" or "torque", the
    torque at ``max_slip``). A ``grid``-per-variable search of the box, then SLSQP from the best feasible point."""
    from scipy.optimize import minimize
    bounds = {k: [float(a), float(b)] for k, (a, b) in (bounds or DEFAULT_BOUNDS).items()}
    names = ["radius", "width", "grouser_height"]
    mat = dict(material or {})
    calls = [0]
    cache: dict[tuple, dict] = {}

    def ev(x):
        key = tuple(round(float(v), 9) for v in x)
        if key not in cache:
            calls[0] += 1
            cache[key] = _evaluate(soil, g, load, dict(zip(names, key, strict=True)), grouser_count, max_slip, n_gl)
        return cache[key]

    def obj(x):
        if objective == "torque":
            return ev(x)["torque_Nm"]
        return wheel_mass(x[0], x[1], x[2], grouser_count, **mat)

    def cons(x):
        e = ev(x)
        c = [e["drawbar_pull_N"] / required_pull - 1.0, 1.0 - e["sinkage_m"] / max_sinkage]
        if max_torque:
            c.append(1.0 - e["torque_Nm"] / max_torque)
        return np.array(c)

    t0 = time.perf_counter()
    axes = [np.linspace(*bounds[k], grid if bounds[k][1] > bounds[k][0] else 1) for k in names]
    pts = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    feas = [(obj(p), p) for p in pts if np.all(cons(p) >= 0)]
    grid_calls = calls[0]
    if not feas:
        return {"feasible": False, "calls": calls[0], "seconds": time.perf_counter() - t0, "grid_points": len(pts),
                "reason": "no point of the grid meets the pull, sinkage and torque limits: widen the bounds, relax "
                          "a limit or lower the required pull"}
    f0, x0 = min(feas, key=lambda t: t[0])
    lo = np.array([bounds[k][0] for k in names])
    span = np.array([max(bounds[k][1] - bounds[k][0], 1e-12) for k in names])
    res = minimize(lambda u: obj(lo + u * span) / max(abs(f0), 1e-12), (x0 - lo) / span, method="SLSQP",
                   bounds=[(0.0, 1.0)] * 3, constraints=[{"type": "ineq", "fun": lambda u: cons(lo + u * span)}],
                   options={"maxiter": 60, "ftol": 1e-7, "eps": 1e-4})
    x = lo + np.clip(res.x, 0, 1) * span
    if not (np.all(cons(x) >= -1e-4) and obj(x) <= f0 * (1 + 1e-9)):
        x, refined = x0, False                    # SLSQP left the feasible set or did not improve: keep the grid point
    else:
        refined = True
    e = ev(x)
    design = dict(zip(names, map(float, x), strict=True))
    return {"feasible": True, "design": design, "grouser_count": int(grouser_count),
            "mass_kg": wheel_mass(*x, grouser_count, **mat), "objective": objective, "objective_value": float(obj(x)),
            "grid_best": {"design": dict(zip(names, map(float, x0), strict=True)), "objective_value": float(f0)},
            "refined": refined, "slsqp": {"success": bool(res.success), "message": str(res.message),
                                          "iterations": int(res.nit)},
            "at_max_slip": {k: e[k] for k in ("drawbar_pull_N", "sinkage_m", "torque_Nm")},
            "constraints": cons(x).tolist(), "calls": calls[0], "grid_calls": grid_calls,
            "grid_points": len(pts), "n_feasible_grid": len(feas), "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------- 4. the workflow
def _published(datasets: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    from . import testbed
    rows = []
    for ds, kind in datasets:
        d = testbed.load_dataset(ds, kind)
        rep = testbed.validate(d)
        cal = testbed.calibrate_holdout(d)
        rel = {k: float(v["rmse_rel_scale"]) for k, v in rep.errors.items()}
        after = {k: (float(v["rmse_rel_scale"]) if isinstance(v, Mapping) and "rmse_rel_scale" in v else None)
                 for k, v in (cal.get("test_after") or {}).items()}
        rows.append({"dataset": d.label, "source": d.source, "soil": d.soil.name, "load_N": d.load,
                     "n_points": int(len(d.slip)), "rmse_rel": rel, "calibrated_holdout_rmse_rel": after,
                     "calibrated": {k: float(v) for k, v in (cal.get("params") or {}).items()}})
    return rows


def terramechanics_workflow(*, soil: str = "Ding2011-TYII", identify: str = "synthetic",
                            plates: Mapping[str, Any] | None = None, shear: Mapping[str, Any] | None = None,
                            load: float = 80.0, g: float = 9.81, wheel: Mapping[str, float] | None = None,
                            slips: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
                            required_pull: float | None = None, max_slip: float = 0.3, max_sinkage: float = 0.03,
                            max_torque: float | None = None, bounds: Mapping[str, Sequence[float]] | None = None,
                            objective: str = "mass", grid: int = 7, material: Mapping[str, float] | None = None,
                            published: Sequence[tuple[str, str]] = (("ding2011", "smooth"), ("ding2011", "grousered")),
                            noise: float = 0.02, n_evaluations: int | None = None, seed: int = 0) -> dict[str, Any]:
    """Soil -> single-wheel curves -> wheel design -> re-verification -> published tests (see the module docstring).

    ``identify``: "none" (the database soil), "data" (``plates`` and/or ``shear`` measurements) or "synthetic"
    (tests generated from the database soil with relative noise ``noise``, then identified). ``wheel``: the wheel the
    curves are computed for (``radius``, ``width``, ``grouser_count``, ``grouser_height``). ``required_pull``: the
    drawbar pull [N] the designed wheel must give at ``max_slip`` (default: what the given wheel gives at 20 % slip).
    ``published``: (dataset, wheel) pairs of :mod:`.testbed` the model is checked on."""
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None, status=None):
        st = status or ("pass" if ok is True else "fail" if ok is False else "inconclusive")
        checks.append({"code": code, "status": st, "detail": detail, "value": value})

    base = get_soil(soil)
    out: dict[str, Any] = {"inputs": {"soil": soil, "identify": identify, "load_N": load, "g": g}}
    # 1. soil
    if identify == "none":
        sl = base
        out["soil_source"] = "database"
    else:
        if identify == "synthetic":
            tests = synthetic_soil_tests(base, noise=noise, seed=seed)
            plates, shear = tests["plates"], tests["shear"]
        elif identify != "data":
            raise ValueError("identify is 'none', 'data' or 'synthetic'")
        if not (plates or shear):
            raise ValueError("identify='data' needs plate and/or shear measurements")
        idr = identify_soil(plates, shear, base=base, seed=seed)
        checks += idr["checks"]
        sl = idr["soil"]
        out["identification"] = {k: idr.get(k) for k in ("pressure", "shear", "identified")}
        out["soil_source"] = "identified from " + ("synthetic tests of the database soil" if identify == "synthetic"
                                                  else "the given tests")
        if identify == "synthetic":
            truth = tests["truth"]
            names = {"k_c": ("pressure", "k_c"), "k_phi": ("pressure", "k_phi"), "n0": ("pressure", "n"),
                     "c": ("shear", "c"), "phi_deg": ("shear", "phi_deg"), "K": ("shear", "K")}
            # joint test: every parameter within the Bonferroni-corrected band of its (log-space) interval, so
            # the check fails by chance 5 % of the time for all six together, not 26 %
            zc = _z_two_sided(0.05 / len(names))
            zs, miss = {}, []
            for k, (grp, pk) in names.items():
                g_ = out["identification"].get(grp) or {}
                iv, est = (g_.get("intervals") or {}).get(pk), (g_.get("estimate") or {}).get(pk)
                if not iv or est is None or min(iv[0], iv[1], est, truth[k]) <= 0:
                    miss.append(k)
                    continue
                se = (math.log(iv[1]) - math.log(iv[0])) / (2 * 1.959964)
                zs[k] = abs(math.log(est) - math.log(truth[k])) / max(se, 1e-300)
                if zs[k] > zc:
                    miss.append(k)
            chk("soil.truth_recovered", not miss, "identified against the database values that made the tests: |z| "
                + ", ".join(f"{k} {v:.2f}" for k, v in zs.items()) + f" (<= {zc:.2f}, 95 % jointly)" +
                (f"; outside: {miss}" if miss else ""), zs)
    out["soil"] = {k: float(getattr(sl, k)) for k in ("c", "phi_deg", "K", "k_c", "k_phi", "n0", "n1", "rho")}
    out["soil"]["name"] = sl.name
    out["soil_provenance"] = [[k, float(v), str(s)] for k, v, s in sl.provenance()]
    # 2. single wheel
    w = {**DEFAULT_WHEEL, **dict(wheel or {})}
    whl = _wheel(w["radius"], w["width"], int(w["grouser_count"]), w["grouser_height"])
    t0 = time.perf_counter()
    cur = wheel_curves(sl, whl, load, g, slips)
    solve_s = (time.perf_counter() - t0) / max(len(slips), 1)
    fine = wheel_curves(sl, whl, load, g, slips, n_gl=96)
    out["wheel"] = {**w, "mass_kg": wheel_mass(w["radius"], w["width"], w["grouser_height"],
                                               int(w["grouser_count"]), **dict(material or {}))}
    out["curves"] = cur
    bal = max(abs(a - load) / load for a in cur["load_carried_N"])
    chk("wheel.load_balance", bal <= 1e-3, f"the stresses carry the load within {bal:.2e} of it (<= 1e-3)", bal)
    dq = max(abs(a - b) for a, b in zip(cur["drawbar_pull_N"], fine["drawbar_pull_N"], strict=True)) / load
    chk("wheel.quadrature", dq <= 5e-3, f"drawbar pull with 24 against 96 quadrature nodes differs by {dq:.2e} of the "
        "load (<= 5e-3)", dq)
    s_arr, dp = np.asarray(cur["slip"]), np.asarray(cur["drawbar_pull_N"])
    op = s_arr <= max(max_slip, 0.3) + 1e-12
    rising = bool(np.all(np.diff(dp[op]) > 0)) if op.sum() > 1 else None
    chk("wheel.pull_rises_with_slip", rising, "drawbar pull rises with slip up to the slip limit" if rising else
        "drawbar pull does not rise with slip over the operating range", None)
    tq = np.asarray(cur["torque_Nm"])[s_arr > 0]
    chk("wheel.driving_torque_positive", bool(np.all(tq > 0)), "the driving torque is positive at every driving slip",
        float(tq.min()) if tq.size else None)
    # 3. design
    req = float(required_pull) if required_pull else float(np.interp(0.2, s_arr, dp))
    des = design_wheel(sl, load=load, g=g, required_pull=req, max_slip=max_slip, max_sinkage=max_sinkage,
                       max_torque=max_torque, grouser_count=int(w["grouser_count"]), bounds=bounds,
                       objective=objective, grid=grid, material=material)
    out["requirement"] = {"drawbar_pull_N": req, "max_slip": max_slip, "max_sinkage_m": max_sinkage,
                          "max_torque_Nm": max_torque, "objective": objective}
    out["design"] = des
    if not des["feasible"]:
        chk("design.feasible", False, des["reason"])
    else:
        chk("design.feasible", True, f"{des['n_feasible_grid']} of {des['grid_points']} grid points meet every limit; "
            f"SLSQP {'refined the best one' if des['refined'] else 'kept the grid point'}")
        # 4. re-verification with the full model
        x = des["design"]
        full = WongReeceModel(sl, _wheel(x["radius"], x["width"], int(w["grouser_count"]), x["grouser_height"]), g=g,
                              n_gl=96)
        z = full.sinkage_for_load(load, max_slip, iters=80)
        r = full.forces(max_slip, z)
        s_req = _slip_for_pull(full, load, req, max_slip)
        tq_req = float(full.steady_state(load, s_req)["torque"]) if math.isfinite(s_req) else float("nan")
        ver = {"drawbar_pull_N": float(r.Fx), "sinkage_m": float(z), "torque_Nm": float(r.My),
               "load_carried_N": float(r.Fz), "slip_at_required_pull": s_req, "torque_at_required_pull_Nm": tq_req,
               "pull_margin": float(r.Fx) / req - 1.0, "sinkage_margin": 1.0 - float(z) / max_sinkage}
        out["verified"] = ver
        dsearch = abs(ver["drawbar_pull_N"] - des["at_max_slip"]["drawbar_pull_N"]) / req
        chk("verify.full_model_agrees", dsearch <= 0.01, f"the full model (96 nodes, 80 bisections) changes the drawbar "
            f"pull of the design by {dsearch:.2%} of the requirement (<= 1 %)", dsearch)
        chk("verify.pull", ver["pull_margin"] >= -1e-3, f"drawbar pull {ver['drawbar_pull_N']:.2f} N at slip "
            f"{max_slip:g} against the required {req:.2f} N (margin {ver['pull_margin']:+.2%})", ver["pull_margin"])
        chk("verify.sinkage", ver["sinkage_margin"] >= -1e-3, f"sinkage {1e3 * z:.2f} mm against the "
            f"{1e3 * max_sinkage:.1f} mm limit", ver["sinkage_margin"])
        if max_torque:
            chk("verify.torque", ver["torque_Nm"] <= max_torque * (1 + 1e-3), f"torque {ver['torque_Nm']:.2f} N m "
                f"against {max_torque:g} N m", ver["torque_Nm"])
        chk("verify.operating_slip", math.isfinite(s_req) and s_req <= max_slip, f"the required pull is reached at "
            f"slip {s_req:.3f} (limit {max_slip:g}), torque {tq_req:.2f} N m there", s_req)
        active = des["constraints"][0] <= 0.02 if objective == "mass" else None
        chk("design.pull_limit_active", active, "the pull limit is active at the minimum-mass design (a lighter "
            "wheel gives less pull, so a minimum that is not on the limit is not converged)" if active else
            f"pull limit slack {des['constraints'][0]:+.2%} at the design", des["constraints"][0],
            status=None if objective == "mass" else "info")
        bb = {k: [float(a), float(b)] for k, (a, b) in (bounds or DEFAULT_BOUNDS).items()}
        on = [f"{k} at its {'lower' if abs(v - bb[k][0]) <= 1e-6 * max(bb[k][1] - bb[k][0], 1e-12) else 'upper'} "
              f"bound {v:g}" for k, v in x.items() if k in bb and min(abs(v - bb[k][0]), abs(v - bb[k][1]))
              <= 1e-6 * max(bb[k][1] - bb[k][0], 1e-12)]
        chk("design.bounds", None, "; ".join(on) + ": the optimum would move past it (widen the box if that is "
            "allowed)" if on else "every design variable inside its bounds", on, status="info")
        # surrogate: only when it pays off, and only one trained on this physics
        rec = {"recommended": ["none"], "reasoning": "recommendation unavailable"}
        try:
            from pinneapple_analysis.verification.architecture_recommendation import recommend_architecture
            n_ev = int(n_evaluations or des["calls"])
            rr = recommend_architecture(n_high_fidelity_simulations=0, needs_parameter_generalization=True,
                                        has_analytical_solver=True, solve_seconds=solve_s, n_evaluations=n_ev,
                                        output_kind="scalar", n_inputs=3)
            rec = {"recommended": list(rr.recommended), "reasoning": rr.reasoning, "n_evaluations": n_ev,
                   "solve_seconds": solve_s}
        except (ImportError, TypeError) as e:
            rec["reasoning"] = f"recommendation unavailable: {e}"
        out["architecture"] = rec
        if rec["recommended"][0] == "none":
            chk("verify.surrogate", None, f"recommend_architecture: none ({rec['reasoning'][:160]}): the full model "
                "is called directly", status="info")
        else:
            chk("verify.surrogate", None, "recommend_architecture asks for a surrogate; the zoo wheel-contact PINN is "
                "trained for one wheel and soil (VIPER on GRC-1), so it is not used for this design: train one with "
                "terramechanics.surrogate.train_contact_surrogate on the design's WongReeceModel", status="not run")
    # published single-wheel tests
    if published:
        pub = _published(published)
        out["published"] = pub
        for row in pub:
            bad = {k: v for k, v in row["rmse_rel"].items() if v > PUBLISHED_BAND.get(k, 0.1)}
            chk(f"published.{row['dataset'].replace(' ', '_')}", not bad, f"{row['source']}, cited soil "
                f"{row['soil']}: RMSE " + ", ".join(f"{k} {v:.1%}" for k, v in row["rmse_rel"].items()) +
                " (band " + ", ".join(f"{k} {PUBLISHED_BAND[k]:.0%}" for k in row["rmse_rel"]) + ")",
                row["rmse_rel"])
    out["checks"] = checks
    bad = [c["code"] for c in checks if c["status"] == "fail"]
    out["verdict"] = "fail" if bad else "pass"
    out["failed"] = bad
    return out
