"""Surrogate quality report: the "can I trust this model?" section of a run.

Every section is computed from held-out data or from points the model never
trained on, and every number states what it was measured against:

* ``error``          -- relative/absolute error vs the reference solution on a
                        held-out test split (never on training points).
* ``convergence``    -- did the loss actually settle, diverge or stall.
* ``generalization`` -- train vs held-out error, and training vs unseen-point
                        PDE residual.
* ``variables``      -- inputs, outputs, predicted ranges vs expected ranges,
                        and input sensitivity.
* ``weights``        -- loss-term weights, each term's share of the final
                        loss, and network parameter health.
* ``uncertainty``    -- snapshot-ensemble spread and, when reference data
                        exists, split-conformal intervals with measured
                        coverage.
* ``physics_checks`` -- governing-equation residual, boundary/initial
                        conditions, mass conservation, energy balance and
                        physical bounds, each with pass/warn/fail.

A check that cannot be computed says so (``status="n/a"`` with a reason); it
is never reported as passing.
"""
from __future__ import annotations

import math
from typing import Any
from collections.abc import Sequence

import numpy as np
import torch

from .physics_adapter import (
    PhysicsAdapter, all_faces, characteristic_length, sample_faces, spatial_coords, unwrap,
)

PASS, WARN, FAIL, NA = "pass", "warn", "fail", "n/a"

_INCOMPRESSIBLE_KINDS = (
    "navier_stokes_incompressible", "incompressible_navier_stokes_2d",
    "incompressible_navier_stokes_rotating_frame", "incompressible_navier_stokes_energy_2d",
    "incompressible_navier_stokes_energy_3d", "navier_stokes_energy_2d", "stokes", "brinkman",
)
# Source-free diffusion: the divergence theorem gives zero net boundary flux.
_SOURCE_FREE_DIFFUSION = ("laplace", "poisson", "heat_equation_steady",
                          "heat_equation_steady_multilayer")
_NON_NEGATIVE_FIELDS = ("rho", "h", "C", "k", "omega", "nu_t", "S", "I", "R")


def _f(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _grade(value: float | None, pass_below: float, warn_below: float) -> str:
    if value is None:
        return NA
    if value < pass_below:
        return PASS
    if value < warn_below:
        return WARN
    return FAIL


def _predict(model, X: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return unwrap(model(X)).detach().cpu().numpy()


def _rel_l2(pred: np.ndarray, ref: np.ndarray) -> float | None:
    denom = np.linalg.norm(ref)
    if denom == 0:
        return None
    return _f(np.linalg.norm(pred - ref) / denom)


# ── sections ─────────────────────────────────────────────────────────────────

def error_section(model, fields, X_test, Y_test, reference_source) -> dict[str, Any]:
    if X_test is None or len(X_test) == 0:
        return {"available": False,
                "reason": "No reference solution for this run (physics-only training); "
                          "accuracy cannot be measured -- see physics_checks."}
    pred = _predict(model, X_test)
    per_field = {}
    for i, f in enumerate(fields):
        p, r = pred[:, i], Y_test[:, i]
        rel = _rel_l2(p, r)
        per_field[f] = {
            "rel_l2_pct": None if rel is None else 100 * rel,
            "rmse": _f(np.sqrt(np.mean((p - r) ** 2))),
            "mae": _f(np.mean(np.abs(p - r))),
            "max_abs_error": _f(np.max(np.abs(p - r))),
            "reference_range": [_f(r.min()), _f(r.max())],
        }
    rel = _rel_l2(pred, Y_test)
    ss_res = float(np.sum((pred - Y_test) ** 2))
    ss_tot = float(np.sum((Y_test - Y_test.mean(axis=0)) ** 2))
    return {
        "available": True,
        "rel_l2_pct": None if rel is None else 100 * rel,
        "r2": _f(1 - ss_res / ss_tot) if ss_tot > 0 else None,
        "per_field": per_field,
        "n_test_points": int(len(X_test)),
        "measured_on": "held-out test split (never used for training)",
        "reference_source": reference_source,
    }


def convergence_section(loss_history: Sequence[float]) -> dict[str, Any]:
    h = np.asarray(loss_history, dtype=float)
    if len(h) == 0:
        return {"status": "not_run", "message": "No training epochs recorded."}
    if not np.all(np.isfinite(h)):
        first_bad = int(np.argmax(~np.isfinite(h)))
        return {"status": "diverged", "epochs": int(len(h)),
                "message": f"Loss became non-finite at epoch {first_bad}."}
    k = max(1, len(h) // 20)
    initial, final = float(np.mean(h[:k])), float(np.mean(h[-k:]))
    reduction = initial / final if final > 0 else float("inf")
    win = max(4, len(h) // 10)
    tail = h[-win:] if len(h) >= win else h
    half = max(1, len(tail) // 2)
    a, b = float(np.mean(tail[:half])), float(np.mean(tail[half:]))
    tail_change = abs(b - a) / a if a > 0 else 0.0
    if final > initial:
        status, msg = "diverged", "Final loss is higher than the initial loss."
    elif reduction >= 10 and tail_change < 0.05:
        status, msg = "converged", "Loss dropped by at least 10x and has flattened out."
    elif tail_change >= 0.05 and b < a:
        status, msg = "still_improving", "Loss is still falling at the end -- more epochs should help."
    elif tail_change >= 0.05:
        status, msg = "oscillating", ("Loss is noisy or rising at the end of training -- lower the "
                                      "learning rate or add a decay schedule.")
    else:
        status, msg = "stalled", "Loss flattened out before dropping 10x -- check weights/learning rate."
    return {
        "status": status,
        "message": msg,
        "epochs": int(len(h)),
        "initial_loss": _f(initial),
        "final_loss": _f(final),
        "reduction_factor": _f(reduction),
        "orders_of_magnitude": _f(math.log10(reduction)) if reduction > 0 and math.isfinite(reduction) else None,
        "tail_relative_change_pct": _f(100 * tail_change),
    }


def generalization_section(model, X_train, Y_train, X_test, Y_test,
                           res_train: float | None, res_unseen: float | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ratios = []
    if X_test is not None and len(X_test) and X_train is not None and len(X_train):
        tr = _rel_l2(_predict(model, X_train), Y_train)
        te = _rel_l2(_predict(model, X_test), Y_test)
        gap = te / tr if tr and te is not None and tr > 0 else None
        out["data"] = {"rel_l2_train_pct": None if tr is None else 100 * tr,
                       "rel_l2_test_pct": None if te is None else 100 * te,
                       "test_over_train": _f(gap)}
        if gap is not None:
            ratios.append(gap)
    if res_train is not None and res_unseen is not None and res_train > 0:
        r = res_unseen / res_train
        out["physics"] = {"residual_rms_training_points": res_train,
                          "residual_rms_unseen_points": res_unseen,
                          "unseen_over_training": _f(r)}
        ratios.append(r)
    if not ratios:
        out.update(status=NA, message="Nothing to compare (no held-out data and no physics residual).")
        return out
    worst = max(ratios)
    if worst <= 1.5:
        out.update(status="good", message="Error on unseen points is close to the training error.")
    elif worst <= 3:
        out.update(status="moderate", message="Noticeably worse on unseen points -- mild overfitting.")
    else:
        out.update(status="poor", message="Much worse on unseen points -- the model memorised its "
                                          "training points; add collocation points or regularise.")
    out["worst_ratio"] = _f(worst)
    return out


def variables_section(model, adapter: PhysicsAdapter, X_eval: torch.Tensor,
                      supervised: bool) -> dict[str, Any]:
    pred = _predict(model, X_eval)
    inputs = [{"name": c, "min": adapter.bounds[c][0], "max": adapter.bounds[c][1],
               "role": "time" if c == "t" else "space"} for c in adapter.coords]
    ranges = getattr(adapter, "field_ranges", {}) or {}
    outputs = []
    for i, f in enumerate(adapter.fields):
        col = pred[:, i]
        row = {"name": f, "pred_min": _f(col.min()), "pred_max": _f(col.max()),
               "pred_mean": _f(col.mean()), "pred_std": _f(col.std()),
               "trained_with": "physics + reference data" if supervised else "physics only"}
        if f in ranges:
            lo, hi = map(float, ranges[f])
            row["expected_range"] = [lo, hi]
            row["pct_outside_expected_range"] = _f(100 * np.mean((col < lo) | (col > hi)))
        outputs.append(row)

    # Sensitivity: mean |d(output)/d(input)| x input range, as % per output.
    X = X_eval[: min(len(X_eval), 1024)].clone().requires_grad_(True)
    Y = unwrap(model(X))
    spans = torch.tensor([adapter.bounds[c][1] - adapter.bounds[c][0] for c in adapter.coords],
                         dtype=X.dtype, device=X.device)
    sens = {}
    for i, f in enumerate(adapter.fields):
        g = torch.autograd.grad(Y[:, i].sum(), X, retain_graph=True)[0]
        s = (g.abs().mean(dim=0) * spans).detach().cpu().numpy()
        tot = float(s.sum())
        sens[f] = {c: _f(100 * v / tot) if tot > 0 else None for c, v in zip(adapter.coords, s, strict=True)}
    return {"inputs": inputs, "outputs": outputs, "input_sensitivity_pct": sens}


def weights_section(model, adapter: PhysicsAdapter, final_raw: dict[str, float],
                    w_data: float, data_loss: float | None) -> dict[str, Any]:
    weighted = adapter.weighted(final_raw)
    if data_loss is not None:
        weighted["data_reference"] = w_data * data_loss
    total = sum(v for v in weighted.values() if v is not None and math.isfinite(v))
    terms = []
    for k, v in sorted(weighted.items(), key=lambda kv: -kv[1]):
        w = adapter.w_pde if k == "pde" else (
            w_data if k == "data_reference" else adapter.conditions[k].weight)
        terms.append({"term": k, "weight": _f(w), "raw_loss": _f(final_raw.get(k, data_loss)),
                      "weighted_loss": _f(v), "share_pct": _f(100 * v / total) if total > 0 else None})
    dominant = terms[0] if terms else None
    imbalance = bool(dominant and dominant["share_pct"] and dominant["share_pct"] > 90 and len(terms) > 1)

    params = [p.detach().float().cpu().reshape(-1) for p in model.parameters()]
    flat = torch.cat(params) if params else torch.zeros(0)
    return {
        "loss_terms": terms,
        "dominant_term": dominant["term"] if dominant else None,
        "imbalanced": imbalance,
        "message": (f"'{dominant['term']}' is over 90% of the final loss; the other terms are "
                    "barely being optimised -- rebalance the weights.") if imbalance else
                   "No single loss term dominates.",
        "network": {
            "n_params": int(flat.numel()),
            "l2_norm": _f(flat.norm()) if flat.numel() else None,
            "max_abs": _f(flat.abs().max()) if flat.numel() else None,
            "non_finite": int((~torch.isfinite(flat)).sum()) if flat.numel() else 0,
        },
    }


def uncertainty_section(model, snapshots: list[dict[str, torch.Tensor]], fields,
                        X_eval: torch.Tensor, X_cal, Y_cal, X_test, Y_test,
                        coverage: float = 0.9) -> dict[str, Any]:
    out: dict[str, Any] = {}
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    preds = []
    try:
        for sd in snapshots:
            model.load_state_dict(sd)
            preds.append(_predict(model, X_eval))
    finally:
        model.load_state_dict(base_state)
    if len(preds) >= 2:
        P = np.stack(preds)                       # (K, N, F)
        std = P.std(axis=0)
        scale = np.maximum(P.mean(axis=0).std(axis=0), 1e-12)
        out["snapshot_ensemble"] = {
            "method": f"spread of {len(preds)} weight snapshots from the last 20% of training",
            "note": "Epistemic proxy: shows where the fit is still moving, not a calibrated interval.",
            "per_field": {f: {"mean_std": _f(std[:, i].mean()), "max_std": _f(std[:, i].max()),
                              "mean_std_pct_of_field_spread": _f(100 * std[:, i].mean() / scale[i])}
                          for i, f in enumerate(fields)},
        }
    else:
        out["snapshot_ensemble"] = {"status": NA, "reason": "Too few epochs to collect snapshots."}

    if X_cal is not None and len(X_cal) >= 10 and X_test is not None and len(X_test):
        pc, pt = _predict(model, X_cal), _predict(model, X_test)
        n = len(X_cal)
        level = min(1.0, math.ceil((n + 1) * coverage) / n)
        per_field = {}
        for i, f in enumerate(fields):
            q = float(np.quantile(np.abs(pc[:, i] - Y_cal[:, i]), level))
            cov = float(np.mean(np.abs(pt[:, i] - Y_test[:, i]) <= q))
            rng = float(Y_test[:, i].max() - Y_test[:, i].min()) or 1.0
            per_field[f] = {"interval_half_width": _f(q), "half_width_pct_of_range": _f(100 * q / rng),
                            "empirical_coverage_pct": _f(100 * cov)}
        out["conformal"] = {
            "method": f"split conformal, target coverage {int(coverage * 100)}%",
            "n_calibration": int(n), "n_test": int(len(X_test)), "per_field": per_field,
        }
    else:
        out["conformal"] = {"status": NA,
                            "reason": "Needs reference data (calibration split) -- physics-only run."}
    return out


def _divergence_check(model, adapter, X: torch.Tensor) -> list[dict[str, Any]]:
    sp = spatial_coords(adapter.coords)
    vel = ["u", "v", "w"][: len(sp)]
    if not all(v in adapter.fields for v in vel):
        return [{"name": "mass_conservation", "status": NA,
                 "detail": f"Velocity fields {vel} not found in {adapter.fields}."}]
    checks = []
    Xg = X[: min(len(X), 2048)].clone().requires_grad_(True)
    Y = unwrap(model(Xg))
    div = torch.zeros(len(Xg), device=X.device)
    mag = torch.zeros(len(Xg), device=X.device)
    for v, c in zip(vel, sp, strict=True):
        g = torch.autograd.grad(Y[:, adapter.fields.index(v)].sum(), Xg, retain_graph=True)[0]
        d = g[:, adapter.coords.index(c)]
        div, mag = div + d, mag + d.abs()
    ratio = _f(torch.sqrt(torch.mean(div ** 2)) / (torch.sqrt(torch.mean(mag ** 2)) + 1e-12))
    checks.append({
        "name": "mass_conservation_local", "law": "div(u) = 0 (incompressibility)",
        "value": ratio, "threshold": {"pass": 0.05, "warn": 0.2}, "status": _grade(ratio, 0.05, 0.2),
        "detail": "RMS of div(u) relative to the RMS size of its individual terms, on unseen points.",
    })

    rng = np.random.default_rng(12345)
    Xb, Nb = sample_faces(adapter.bounds, adapter.coords, all_faces(adapter.coords), 4000, rng)
    Xb_t = torch.as_tensor(Xb, device=X.device)
    Yb = _predict(model, Xb_t)
    un = sum(Yb[:, adapter.fields.index(v)] * Nb[:, adapter.coords.index(c)] for v, c in zip(vel, sp, strict=True))
    net, gross = float(np.mean(un)), float(np.mean(np.abs(un)))
    if gross < 1e-8:
        checks.append({"name": "mass_conservation_global", "status": NA,
                       "detail": "No flow crosses the boundary (closed domain); net flux is trivially zero."})
    else:
        r = _f(abs(net) / gross)
        checks.append({
            "name": "mass_conservation_global", "law": "net mass flux through the boundary = 0",
            "value": r, "threshold": {"pass": 0.05, "warn": 0.2}, "status": _grade(r, 0.05, 0.2),
            "detail": "|inflow - outflow| / (inflow + outflow) through the box faces.",
        })
    return checks


def _energy_balance_check(model, adapter) -> dict[str, Any]:
    f = adapter.fields[0]
    rng = np.random.default_rng(54321)
    Xb, Nb = sample_faces(adapter.bounds, adapter.coords, all_faces(adapter.coords), 4000, rng)
    X = torch.as_tensor(Xb).requires_grad_(True)
    Y = unwrap(model(X))[:, adapter.fields.index(f)]
    g = torch.autograd.grad(Y.sum(), X)[0].detach().numpy()
    flux = np.sum(g * Nb, axis=1)
    net, gross = float(np.mean(flux)), float(np.mean(np.abs(flux)))
    if gross < 1e-8:
        return {"name": "energy_balance", "status": NA,
                "detail": "No flux crosses the boundary; the balance is trivially satisfied."}
    r = _f(abs(net) / gross)
    return {
        "name": "energy_balance", "law": "net boundary flux = 0 (no internal source)",
        "value": r, "threshold": {"pass": 0.05, "warn": 0.2}, "status": _grade(r, 0.05, 0.2),
        "detail": f"|net outward flux of grad({f})| / total |flux| (divergence theorem).",
    }


# Dimensionless-number parameters: they divide terms (1/Re, ...), so they
# must not inflate the coefficient scale below.
_DIMENSIONLESS = {"re", "ma", "pr", "ra", "pe", "gr", "st", "fr", "we", "alpha_deg"}


def field_scales(adapter: PhysicsAdapter, pred: np.ndarray) -> np.ndarray:
    """Per-field magnitude: max of the predicted spread, the declared range
    and the RMS of the Dirichlet/initial values imposed on that field."""
    out = np.std(pred, axis=0) if pred.size else np.zeros(len(adapter.fields))
    ranges = getattr(adapter, "field_ranges", {}) or {}
    for i, f in enumerate(adapter.fields):
        if f in ranges:
            lo, hi = map(float, ranges[f])
            out[i] = max(out[i], hi - lo)
        for info in adapter.conditions.values():
            if info.kind in ("dirichlet", "initial") and f in info.fields:
                out[i] = max(out[i], info.target_rms)
    return out


def residual_scale(model, adapter: PhysicsAdapter, X: torch.Tensor, pred: np.ndarray) -> float:
    """Typical size of the PDE's individual terms, to normalise its residual.

    ``c * max(G1, G2)``: G1/G2 are the RMS first and (unmixed) second
    derivatives of the predicted fields (individual derivatives do not
    cancel even when the equation is satisfied, unlike the residual), floored
    by field_scale/L and field_scale/L^2; ``c`` is the largest dimensional
    coefficient of the PDE (e.g. conductivity k), at least 1.
    """
    Xg = X[: min(len(X), 512)].clone().requires_grad_(True)
    Y = unwrap(model(Xg))
    g1, g2 = 0.0, 0.0
    for i in range(Y.shape[1]):
        g = torch.autograd.grad(Y[:, i].sum(), Xg, create_graph=True)[0]
        for j in range(Xg.shape[1]):
            g1 = max(g1, float(torch.sqrt(torch.mean(g[:, j] ** 2))))
            h = torch.autograd.grad(g[:, j].sum(), Xg, retain_graph=True)[0][:, j]
            g2 = max(g2, float(torch.sqrt(torch.mean(h ** 2))))
    L = max(characteristic_length(adapter.bounds, adapter.coords), 1e-12)
    U = float(np.max(field_scales(adapter, pred))) if pred.size else 0.0
    g1, g2 = max(g1, U / L), max(g2, U / L ** 2)
    coeffs = [abs(float(v)) for k, v in (adapter.pde_params or {}).items()
              if isinstance(v, (int, float)) and k.lower() not in _DIMENSIONLESS]
    c = max([1.0] + coeffs)
    return c * max(g1, g2)


def physics_checks_section(model, adapter: PhysicsAdapter, X_eval: torch.Tensor,
                           raw_unseen: dict[str, float], raw_initial: dict[str, float] | None,
                           pred_eval: np.ndarray) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    # 1. Governing equation, graded against the size of its own terms.
    res = raw_unseen.get("pde")
    res0 = (raw_initial or {}).get("pde")
    rms = _f(math.sqrt(res)) if res is not None else None
    if res is None:
        checks.append({"name": "governing_equation", "status": NA,
                       "detail": "No governing equation for this problem."})
    else:
        S = residual_scale(model, adapter, X_eval, pred_eval)
        rel = _f(rms / S) if S > 0 else None
        checks.append({
            "name": "governing_equation", "law": f"PDE residual ({adapter.pde_kind}) ~ 0",
            "value": rel, "rms_residual": rms, "term_scale": _f(S),
            "reduction_vs_untrained": _f(res0 / res) if res and res0 else None,
            "threshold": {"pass": 0.05, "warn": 0.2}, "status": _grade(rel, 0.05, 0.2),
            "detail": "RMS PDE residual on points not used in training divided by the typical "
                      "size of the equation's terms (coefficient x derivative scale).",
        })

    # 2. Boundary / initial conditions.
    L = characteristic_length(adapter.bounds, adapter.coords)
    fscale = field_scales(adapter, pred_eval)
    for key, info in adapter.conditions.items():
        mse = raw_unseen.get(key)
        if mse is None:
            continue
        idx = [adapter.fields.index(f) for f in info.fields if f in adapter.fields]
        scale = float(np.max(fscale[idx])) if idx else info.target_rms
        scale = max(scale, info.target_rms, 1e-8)
        if info.kind == "neumann":
            scale = scale / max(L, 1e-12)
        rel = _f(math.sqrt(mse) / scale)
        checks.append({
            "name": f"{'initial' if info.kind == 'initial' else 'boundary'}:{info.name}",
            "law": f"{info.kind} condition on {', '.join(info.fields)}",
            "value": rel, "rms_error": _f(math.sqrt(mse)),
            "threshold": {"pass": 0.05, "warn": 0.2}, "status": _grade(rel, 0.05, 0.2),
            "detail": f"RMS error on fresh points / field scale. Points: {info.how}.",
        })
    for u in adapter.unresolved:
        checks.append({"name": f"boundary:{u['name']}", "law": f"{u['kind']} condition",
                       "status": FAIL, "detail": f"NOT APPLIED during training: {u['reason']}."})

    # 3. Conservation laws.
    if adapter.pde_kind in _INCOMPRESSIBLE_KINDS:
        checks.extend(_divergence_check(model, adapter, X_eval))
    if (adapter.pde_kind in _SOURCE_FREE_DIFFUSION and len(adapter.fields) == 1
            and "t" not in adapter.coords):
        checks.append(_energy_balance_check(model, adapter))

    # 4. Physical bounds.
    ranges = getattr(adapter, "field_ranges", {}) or {}
    for i, f in enumerate(adapter.fields):
        col = pred_eval[:, i]
        if f in _NON_NEGATIVE_FIELDS:
            frac = _f(100 * np.mean(col < 0))
            checks.append({"name": f"non_negative:{f}", "law": f"{f} >= 0", "value": frac,
                           "threshold": {"pass": 0.0, "warn": 1.0},
                           "status": PASS if frac == 0 else WARN if frac < 1 else FAIL,
                           "detail": "% of unseen points with a negative value."})
        if f in ranges:
            lo, hi = map(float, ranges[f])
            margin = 0.1 * (hi - lo)
            frac = _f(100 * np.mean((col < lo - margin) | (col > hi + margin)))
            checks.append({"name": f"expected_range:{f}", "law": f"{lo} <= {f} <= {hi} (+10% margin)",
                           "value": frac, "threshold": {"pass": 1.0, "warn": 5.0},
                           "status": _grade(frac, 1.0, 5.0),
                           "detail": "% of unseen points outside the preset's declared field range."})
    return checks


def verdict(report: dict[str, Any]) -> dict[str, Any]:
    reasons_fail, reasons_warn = [], []
    conv = report["convergence"].get("status")
    if conv in ("diverged", "not_run"):
        reasons_fail.append(f"training {conv}")
    elif conv in ("stalled", "still_improving", "oscillating"):
        reasons_warn.append(f"training {conv.replace('_', ' ')}")
    err = report["error"]
    if err.get("available"):
        pct = err.get("rel_l2_pct")
        if pct is None or pct > 20:
            reasons_fail.append(f"test error {pct:.1f}%" if pct is not None else "test error undefined")
        elif pct > 5:
            reasons_warn.append(f"test error {pct:.1f}%")
    else:
        reasons_warn.append("no reference data: accuracy not measured")
    gen = report["generalization"].get("status")
    if gen == "poor":
        reasons_fail.append("poor generalization")
    elif gen == "moderate":
        reasons_warn.append("moderate generalization gap")
    for c in report["physics_checks"]:
        if c["status"] == FAIL:
            reasons_fail.append(f"physics check failed: {c['name']}")
        elif c["status"] == WARN:
            reasons_warn.append(f"physics check warning: {c['name']}")
    if reasons_fail:
        return {"status": "not_trustworthy", "reasons": reasons_fail + reasons_warn}
    if reasons_warn:
        return {"status": "use_with_caution", "reasons": reasons_warn}
    return {"status": "trustworthy", "reasons": ["all checks passed"]}


def build_report(*, model, adapter: PhysicsAdapter, loss_history, final_raw_train,
                 raw_initial, raw_unseen, res_train, X_eval, split, snapshots,
                 reference_source, w_data, data_loss, warnings) -> dict[str, Any]:
    """Assemble all sections. ``split`` holds numpy/tensor train/cal/test arrays."""
    fields = adapter.fields
    was_training = model.training
    model.eval()
    try:
        X_te, Y_te = split.get("X_test"), split.get("Y_test")
        pred_eval = _predict(model, X_eval)
        res_unseen = _f(math.sqrt(raw_unseen["pde"])) if "pde" in raw_unseen else None
        report: dict[str, Any] = {}
        sections = [
            ("error", lambda: error_section(model, fields, X_te, Y_te, reference_source)),
            ("convergence", lambda: convergence_section(loss_history)),
            ("generalization", lambda: generalization_section(
                model, split.get("X_train"), split.get("Y_train"), X_te, Y_te, res_train, res_unseen)),
            ("variables", lambda: variables_section(model, adapter, X_eval, X_te is not None)),
            ("weights", lambda: weights_section(model, adapter, final_raw_train, w_data, data_loss)),
            ("uncertainty", lambda: uncertainty_section(
                model, snapshots, fields, X_eval, split.get("X_cal"), split.get("Y_cal"), X_te, Y_te)),
            ("physics_checks", lambda: physics_checks_section(
                model, adapter, X_eval, raw_unseen, raw_initial, pred_eval)),
        ]
        for name, fn in sections:
            try:
                report[name] = fn()
            except Exception as e:  # report the failure, never hide it
                report[name] = ({"status": NA, "error": f"{type(e).__name__}: {e}"}
                                if name != "physics_checks" else
                                [{"name": "physics_checks", "status": NA,
                                  "detail": f"could not be computed: {type(e).__name__}: {e}"}])
        report["warnings"] = list(warnings)
        report["verdict"] = verdict(report)
        return report
    finally:
        model.train(was_training)
