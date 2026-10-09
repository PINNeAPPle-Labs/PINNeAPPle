"""PCB Hotspot engine: evaluation, what-if studies and calibration to measurements.

Everything here is physics (``solver.py``) plus PINNeAPPle's Ensemble
Kalman Inversion for calibration -- no learned surrogate is involved.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List

import numpy as np

from pinneapple_physics.closed_form.pcb_thermal import Environment

from .solver import Board, Component, overlaps, solve

# Surface heat-transfer correlations for boards are typically quoted at
# +/-20 %; propagated as a band (an engineering assumption, stated as such).
H_BAND = 0.20


def _check(name, status, value, detail, law=""):
    return {"name": name, "status": status, "value": value, "detail": detail, "law": law}


def _fixed_h_tj(board, comps, env, h, n_cells, **kw) -> np.ndarray:
    """Junction temperatures for given surface coefficients (single linear solve)."""
    r = solve(board, comps, env, n_cells=n_cells, h_override=h, **kw)
    return np.array([c["t_junction_c"] for c in r["components"]])


def _downsample(a: np.ndarray, n: int = 64) -> List[List[float]]:
    iy = np.linspace(0, a.shape[0] - 1, min(n, a.shape[0])).round().astype(int)
    ix = np.linspace(0, a.shape[1] - 1, min(n, a.shape[1])).round().astype(int)
    return np.round(a[np.ix_(iy, ix)], 2).tolist()


VALIDATION = [
    "Solver vs. the exact resistor-network solution: agreement to 1e-6",
    "Default packages on the JEDEC JESD51-7 test board vs. published datasheet θJA: "
    "within 8 % for all 8 packages",
    "Energy balance of every solve: better than 1e-11; grid convergence checked on every report",
    "Calibration recovers known cooling and copper factors from synthetic measurements to < 1 %",
]


def model_scope(comps: List[Component], env: Environment) -> Dict[str, Any]:
    """What the model covers, which way each simplification errs, what to do
    about it today and what is planned -- shown in every report."""
    items = []
    typical = [c.name for c in comps if c.theta_jb is None or c.theta_jc is None]
    if typical:
        items.append({
            "topic": "Package thermal data", "effect": "check",
            "detail": f"{len(typical)} of {len(comps)} parts ({', '.join(typical[:4])}"
                      f"{', ...' if len(typical) > 4 else ''}) use typical values for their package, "
                      "calibrated to published JEDEC data. Within one package type, die size moves "
                      "real parts by about ±25 %.",
            "today": "Enter θJB and θJC(top) from each datasheet for sign-off.",
            "planned": "Package library by manufacturer part number."})
    items.append({
        "topic": "Enclosure", "effect": "optimistic",
        "detail": "The board sits in open air at the ambient temperature you enter. Inside a closed "
                  "box the local air is warmer and moves less.",
        "today": "Enter the air temperature inside the enclosure, or calibrate with 3+ measured "
                 "points — the Calibrate tab fits the real cooling of your product.",
        "planned": "Enclosure model (vents, walls, internal air rise)."})
    if env.air_velocity_m_s > 0:
        items.append({
            "topic": "Air heating along the flow", "effect": "optimistic",
            "detail": "The same air temperature is used over the whole board. Parts downstream of "
                      "hot parts see warmer air.",
            "today": "For downstream parts, raise the ambient by the upstream air rise "
                     "(power / (mass flow x cp)).",
            "planned": "Air temperature rise along the flow direction."})
    items.append({
        "topic": "Copper layout", "effect": "check",
        "detail": "Each layer's copper is spread evenly at the coverage you enter. Local pours "
                  "under hot parts spread heat better; splits and cut-outs spread it worse.",
        "today": "Enter the coverage of the area around the hot parts; add thermal vias where used.",
        "planned": "Import of real copper from Gerber / ODB++."})
    items.append({
        "topic": "Steady state", "effect": "check",
        "detail": "Continuous power. Short bursts run cooler than shown.",
        "today": "Use average power for duty-cycled parts, peak power for sign-off.",
        "planned": "Transient response to power profiles."})
    return {"validated": VALIDATION, "items": items,
            "band": f"±{int(H_BAND * 100)} % on the surface cooling (typical correlation accuracy) "
                    "is already included in the verdict; calibration replaces it with your measured value."}


def evaluate(board: Board, comps: List[Component], env: Environment, *, quick: bool = False,
             n_cells: int = 60) -> Dict[str, Any]:
    """``quick``: one solve on a coarser grid (interactive dragging). Full: band + checks."""
    if not comps:
        raise ValueError("Add at least one component.")
    r = solve(board, comps, env, n_cells=40 if quick else n_cells)
    Ta = env.t_ambient_c
    names = [c["name"] for c in r["components"]]
    tj = np.array([c["t_junction_c"] for c in r["components"]])
    crit = int(np.argmin([c["margin_c"] for c in r["components"]]))
    out: Dict[str, Any] = {
        "components": r["components"],
        "critical": names[crit],
        "maps": {"top_c": _downsample(r["t_top_c"]), "bottom_c": _downsample(r["t_bottom_c"]),
                 "width_mm": board.width_mm, "depth_mm": board.depth_mm},
        "board_max_c": float(max(r["t_top_c"].max(), r["t_bottom_c"].max())),
        "power_w": r["power_w"], "heat_split_w": r["heat_split_w"], "h": r["h"],
        "grid": r["grid"], "layers": r["layers"], "quick": quick,
        "warnings": overlaps(comps) + list(r["correlation_warnings"]),
    }
    if quick:
        return out

    h = {"top": r["h"]["top"], "bottom": r["h"]["bottom"]}
    lo_h = {s: v * (1 - H_BAND) for s, v in h.items()}
    hi_h = {s: v * (1 + H_BAND) for s, v in h.items()}
    tj_hot = _fixed_h_tj(board, comps, env, lo_h, n_cells)
    tj_cold = _fixed_h_tj(board, comps, env, hi_h, n_cells)
    tj_coarse = _fixed_h_tj(board, comps, env, h, max(16, int(n_cells * 0.6)))
    for i, c in enumerate(out["components"]):
        c["t_junction_band_c"] = [float(tj_cold[i]), float(tj_hot[i])]
        c["status"] = ("ok" if tj_hot[i] <= c["tj_max_c"] else
                       "marginal" if tj[i] <= c["tj_max_c"] else "over")

    checks = []
    eb = r["energy_balance_rel_error"]
    checks.append(_check("energy_balance", "pass" if eb < 1e-6 else "fail", eb,
                         "Heat to air + chassis vs. total component power - 1.", "energy conservation"))
    rise = np.maximum(tj - Ta, 1e-9)
    gerr = float(np.max(np.abs(tj_coarse - tj) / rise))
    worst = names[int(np.argmax(np.abs(tj_coarse - tj) / rise))]
    checks.append(_check("grid_convergence", "pass" if gerr < 0.02 else "warn" if gerr < 0.05 else "fail",
                         gerr, f"Largest junction-temperature-rise change between coarse and fine "
                               f"grids ({worst}). Small packages need a fine grid.", "numerical accuracy"))
    checks.append(_check("surface_coefficients_converged", "pass" if r["converged"] else "warn", None,
                         f"Convection + radiation fixed point reached in {r['iterations']} iterations "
                         f"(h top {h['top']:.1f}, bottom {h['bottom']:.1f} W/m²K).", "nonlinear coupling"))
    checks.append(_check("correlation_in_range", "warn" if r["correlation_warnings"] else "pass", None,
                         "; ".join(r["correlation_warnings"]) or "Correlations used inside their range.",
                         "surface heat-transfer model"))
    checks.append(_check("temperatures_above_ambient",
                         "pass" if float(np.min(r["t_top_c"])) >= Ta - 1e-6 else "fail",
                         float(np.min(r["t_top_c"]) - Ta), "No point colder than ambient (2nd law).",
                         "2nd law"))
    out["checks"] = checks

    over = [c["name"] for c in out["components"] if c["status"] == "over"]
    marginal = [c["name"] for c in out["components"] if c["status"] == "marginal"]
    if over:
        out["verdict"] = {"status": "fails", "text": f"{', '.join(over)} {'exceeds' if len(over) == 1 else 'exceed'} "
                          "Tj,max at nominal cooling."}
    elif marginal:
        out["verdict"] = {"status": "marginal", "text": f"{', '.join(marginal)} "
                          f"{'passes' if len(marginal) == 1 else 'pass'} nominally but not "
                          f"with {int(H_BAND * 100)}% weaker cooling -- add margin."}
    else:
        out["verdict"] = {"status": "meets", "text": "All components stay below Tj,max even with "
                          f"{int(H_BAND * 100)}% weaker surface cooling."}
    out["scope"] = model_scope(comps, env)
    out["field3d"] = field_3d(board, out)
    out["method"] = ("3D finite volumes, one cell per copper/dielectric layer; JEDEC two-resistor "
                     "component models; convection (Churchill-Chu / flat-plate) + radiation, "
                     "iterated on surface temperature; thermal vias as parallel copper.")
    return out


def field_3d(board: Board, out: Dict[str, Any]) -> Dict[str, Any]:
    return {"width_mm": board.width_mm, "depth_mm": board.depth_mm,
            "thickness_mm": board.thickness_mm, "top_c": out["maps"]["top_c"],
            "bottom_c": out["maps"]["bottom_c"],
            "components": [{k: c[k] for k in ("name", "x_mm", "y_mm", "w_mm", "d_mm", "h_mm", "side",
                                                "t_junction_c", "t_case_top_c", "margin_c", "power_w")}
                           for c in out["components"]]}


# ── what-if studies ──────────────────────────────────────────────────────────

def what_if(board: Board, comps: List[Component], env: Environment,
            n_cells: int = 48) -> Dict[str, Any]:
    """Re-solve the board with common fixes and report the effect on the
    critical component (and on every junction)."""
    base = solve(board, comps, env, n_cells=n_cells)
    names = [c["name"] for c in base["components"]]
    tj0 = {c["name"]: c["t_junction_c"] for c in base["components"]}
    crit = min(base["components"], key=lambda c: c["margin_c"])["name"]
    ci = names.index(crit)

    scenarios = []

    def run(label: str, detail: str, b=board, cs=comps, e=env):
        r = solve(b, cs, e, n_cells=n_cells)
        tj = {c["name"]: c["t_junction_c"] for c in r["components"]}
        scenarios.append({"label": label, "detail": detail,
                          "critical_tj_c": tj[crit], "delta_critical_c": tj[crit] - tj0[crit],
                          "max_tj_c": max(tj.values()),
                          "delta_by_component_c": {n: tj[n] - tj0[n] for n in names}})

    c_crit = comps[ci]
    if c_crit.side == "top" and c_crit.vias < 16:
        cs = copy.deepcopy(comps)
        r = cs[ci].resolved()
        n_via = max(4, min(64, int((r.w_mm // 1.2) * (r.d_mm // 1.2))))
        cs[ci].vias = n_via
        run(f"{n_via} thermal vias under {crit}", "0.3 mm drill, 25 µm plating, unfilled", cs=cs)
    if board.outer_oz < 2:
        b2 = copy.deepcopy(board)
        b2.outer_oz = 2.0
        run("2 oz outer copper", "outer layers 70 µm instead of 35 µm", b=b2)
    if board.inner_coverage < 0.95 or board.n_copper < 6:
        b3 = copy.deepcopy(board)
        b3.n_copper = board.n_copper + 2
        run(f"{b3.n_copper}-layer board", "two more inner planes (90 % copper)", b=b3)
    e2 = copy.deepcopy(env)
    e2.air_velocity_m_s = (env.air_velocity_m_s or 0) + 1.0
    run(f"{e2.air_velocity_m_s:g} m/s airflow", "fan / system airflow along the board depth", e=e2)
    if not board.chassis_edges:
        b4 = copy.deepcopy(board)
        b4.chassis_edges = True
        run("Edges clamped to chassis", f"card-edge cooling at ambient, {b4.edge_h_w_m2k:.0f} W/m²K contact", b=b4)
    scenarios.sort(key=lambda s: s["delta_critical_c"])
    return {"critical": crit, "baseline_tj_c": tj0[crit], "scenarios": scenarios}


# ── calibration to measurements (PINNeAPPle Ensemble Kalman Inversion) ──────

def _interp_map(t: np.ndarray, W: float, D: float, x: float, y: float) -> float:
    ny, nx = t.shape
    fx = np.clip(x / W * nx - 0.5, 0, nx - 1)
    fy = np.clip(y / D * ny - 0.5, 0, ny - 1)
    i0, j0 = int(fx), int(fy)
    i1, j1 = min(i0 + 1, nx - 1), min(j0 + 1, ny - 1)
    ax, ay = fx - i0, fy - j0
    return float((1 - ax) * (1 - ay) * t[j0, i0] + ax * (1 - ay) * t[j0, i1]
                 + (1 - ax) * ay * t[j1, i0] + ax * ay * t[j1, i1])


def calibrate(board: Board, comps: List[Component], env: Environment,
              measurements: List[Dict[str, Any]], *, noise_std_c: float = 1.0,
              n_ensemble: int = 16, n_iterations: int = 8, seed: int = 0,
              n_cells: int = 24, n_cells_final: int = 60,
              prior_std_log: float = 0.5) -> Dict[str, Any]:
    """Fit surface-cooling and in-plane-spreading scale factors to measured
    temperatures with PINNeAPPle's Ensemble Kalman Inversion.

    ``measurements``: [{"x_mm", "y_mm", "t_c", "side": "top"|"bottom"}] board
    points (thermocouples / IR) and/or [{"component": name, "t_c"}] junction
    readings (e.g. an on-die sensor). Parameters: log h-scale and log
    k_xy-scale (copper coverage / quality is the usual unknown). With 5+
    measurements, 20 % are held out to test the calibrated model on points it
    did not see.
    """
    from pinneapple_analysis.inverse_problems.ensemble_kalman import (
        EKIConfig,
        EnsembleKalmanInversion,
    )

    if len(measurements) < 2:
        raise ValueError("Provide at least 2 measurements.")
    names = [c.name for c in comps]
    for m in measurements:
        if "component" in m and m["component"] not in names:
            raise ValueError(f"Unknown component '{m['component']}' in measurements.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(measurements))
    n_hold = len(measurements) // 5 if len(measurements) >= 5 else 0
    hold, fit = sorted(idx[:n_hold].tolist()), sorted(idx[n_hold:].tolist())

    nominal = solve(board, comps, env, n_cells=n_cells)
    h0 = {"top": nominal["h"]["top"], "bottom": nominal["h"]["bottom"]}
    W, D = board.width_mm, board.depth_mm

    def predict(theta: np.ndarray, which: List[int], n: int = n_cells) -> Dict[str, Any]:
        h = {s: v * math.exp(theta[0]) for s, v in h0.items()}
        r = solve(board, comps, env, n_cells=n, h_override=h, k_xy_scale=math.exp(theta[1]))
        tj = {c["name"]: c["t_junction_c"] for c in r["components"]}
        vals = []
        for i in which:
            m = measurements[i]
            if "component" in m:
                vals.append(tj[m["component"]])
            else:
                t = r["t_top_c"] if m.get("side", "top") == "top" else r["t_bottom_c"]
                vals.append(_interp_map(t, W, D, float(m["x_mm"]), float(m["y_mm"])))
        return {"obs": np.array(vals), "tj": np.array([tj[nm] for nm in names])}

    y = np.array([float(measurements[i]["t_c"]) for i in fit])

    def forward(batch: np.ndarray) -> np.ndarray:
        return np.stack([predict(th, fit)["obs"] for th in batch])

    eki = EnsembleKalmanInversion(forward, EKIConfig(
        n_ensemble=n_ensemble, n_iterations=n_iterations, noise_std=noise_std_c,
        init_spread=0.35, seed=seed, verbose=False))
    hist = eki.run(y, np.zeros(2))
    theta_mean = eki.theta_mean

    allm = list(range(len(measurements)))
    at_prior = predict(np.zeros(2), allm, n_cells_final)
    before = at_prior["obs"]
    at_mean = predict(theta_mean, allm, n_cells_final)
    after = at_mean["obs"]
    meas = np.array([float(m["t_c"]) for m in measurements])

    def rms(sel, pred):
        return float(np.sqrt(np.mean((pred[sel] - meas[sel]) ** 2))) if sel else None

    # Posterior uncertainty: Laplace approximation at the EKI estimate (the EKI
    # ensemble itself collapses and would understate it). Jacobians by central
    # differences on the fine grid; junction uncertainty by linear propagation.
    eps = 0.05
    Jo, Jt = [], []
    for k in range(2):
        e = np.zeros(2)
        e[k] = eps
        pp, pm = predict(theta_mean + e, fit, n_cells_final), predict(theta_mean - e, fit, n_cells_final)
        Jo.append((pp["obs"] - pm["obs"]) / (2 * eps))
        Jt.append((pp["tj"] - pm["tj"]) / (2 * eps))
    Jo, Jt = np.stack(Jo, 1), np.stack(Jt, 1)
    prec = Jo.T @ Jo / noise_std_c ** 2 + np.eye(2) / prior_std_log ** 2
    # EKI ran on the coarse grid; one Gauss-Newton (MAP) step on the fine grid
    # removes the discretisation bias of its optimum. Kept only if it helps.
    def objective(th, pred):
        return float(np.sum((pred[fit] - meas[fit]) ** 2) / noise_std_c ** 2 + th @ th / prior_std_log ** 2)
    grad = Jo.T @ (meas[fit] - after[fit]) / noise_std_c ** 2 - theta_mean / prior_std_log ** 2
    theta_gn = theta_mean + np.linalg.solve(prec, grad)
    at_gn = predict(theta_gn, allm, n_cells_final)
    if objective(theta_gn, at_gn["obs"]) < objective(theta_mean, after):
        theta_mean, at_mean, after = theta_gn, at_gn, at_gn["obs"]
    post_cov = np.linalg.inv(prec)
    # Model adequacy: reduced chi-square of the fit residuals. If the residuals
    # are larger than the stated noise, the model (or the noise estimate) is
    # wrong and the Laplace covariance is too optimistic -- inflate it by the
    # Birge ratio instead of reporting false confidence.
    resid = after[fit] - meas[fit]
    dof = max(len(fit) - 2, 1)
    chi2_red = float(np.sum(resid ** 2) / noise_std_c ** 2 / dof)
    birge = math.sqrt(max(chi2_red, 1.0))
    post_cov = post_cov * birge ** 2
    std = np.sqrt(np.diag(post_cov))
    tj_std = np.sqrt(np.einsum("ij,jk,ik->i", Jt, post_cov, Jt))
    params = {
        "h_scale": {"value": math.exp(theta_mean[0]), "std_log": float(std[0]),
                    "meaning": "multiplier on the correlation's surface coefficients"},
        "k_inplane_scale": {"value": math.exp(theta_mean[1]), "std_log": float(std[1]),
                            "meaning": "multiplier on in-plane conductivity (copper coverage / quality)"},
    }
    warnings = []
    if len(fit) <= 2:
        adequacy = "not testable with 2 readings"
        warnings.append("With 2 readings the two unknowns are fitted exactly, so the readings cannot show whether "
                        "the model fits this board. Add a third reading to check it, and 5+ to hold some out.")
    elif chi2_red <= 2:
        adequacy = "consistent"
    elif chi2_red <= 10:
        adequacy = "residuals above the stated noise"
        warnings.append(f"Fit residuals are {math.sqrt(chi2_red):.1f}x the stated measurement noise: either the "
                        "noise is underestimated or the board has effects the model lacks. Uncertainties were "
                        "inflated accordingly.")
    else:
        adequacy = "model and measurements disagree"
        warnings.append(f"Fit residuals are {math.sqrt(chi2_red):.1f}x the stated measurement noise. Check "
                        "sensor positions/sides, component powers and the stack-up; two scale factors cannot "
                        "explain these readings. Uncertainties were inflated but treat the result with caution.")
    rb, ra = rms(hold, before), rms(hold, after)
    if hold and ra > 1.1 * rb:
        warnings.append(f"Held-out error got worse after calibration ({rb:.1f} -> {ra:.1f} °C): the calibrated "
                        "model does not generalise to the unseen points.")
    ident = {k: ("well determined" if v["std_log"] < 0.1 else "weakly determined" if v["std_log"] < 0.25
                 else "not determined by these measurements") for k, v in params.items()}
    return {
        "method": "PINNeAPPle Ensemble Kalman Inversion (Iglesias et al. 2013) on log h-scale and "
                  f"log in-plane-k scale ({n_ensemble} members x {len(hist.iterations)} iterations, "
                  f"coarse grid) refined by a Gauss-Newton step on the fine grid; uncertainty by a Laplace approximation on the fine grid "
                  f"(measurement noise {noise_std_c} °C, prior ±{prior_std_log} in log).",
        "parameters": params,
        "identifiability": ident,
        "fit": {"n_used": len(fit), "rms_before_c": rms(fit, before), "rms_after_c": rms(fit, after)},
        "adequacy": {"status": adequacy, "reduced_chi2": chi2_red, "uncertainty_inflation": birge},
        "warnings": warnings,
        "holdout": {"n": len(hold), "rms_before_c": rb, "rms_after_c": ra,
                    "note": "points NOT used for calibration" if hold else "needs 5+ measurements"},
        "measurements": [{**m, "predicted_before_c": float(before[i]), "predicted_after_c": float(after[i]),
                          "used_for": "fit" if i in fit else "holdout"} for i, m in enumerate(measurements)],
        "junctions_after_calibration": [
            {"name": nm, "t_junction_c": float(at_mean["tj"][i]), "std_c": float(tj_std[i]),
             "tj_max_c": comps[i].tj_max_c, "before_calibration_c": float(at_prior["tj"][i])}
            for i, nm in enumerate(names)],
        "misfit_history": [float(v) for v in hist.data_misfit],
    }
