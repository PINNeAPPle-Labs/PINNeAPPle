"""Trust report for the heat-sink surrogates.

Same sections as the PINNeAPPle app's surrogate report (error, convergence,
generalization, variables, weights, uncertainty, physics checks, verdict),
measured on designs the model never saw (held-out test split).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from pinneapple_app.backend.core.surrogate_report import convergence_section

READABLE = {
    "log_W": "base width", "log_D": "base depth (flow length)", "log_tb": "base thickness",
    "log_gap": "fin gap", "log_t": "fin thickness", "log_H": "fin height", "log_N": "fin count",
    "log_k": "material conductivity", "log_fw": "source width / base width",
    "log_fd": "source depth / base depth", "T_amb": "ambient temperature",
    "log_V": "air velocity", "log_Q": "power",
}

# Physical laws the learned response must respect: (feature, output index,
# required sign of d(output)/d(feature), statement).
LAWS = {
    "forced": [
        ("log_V", 0, -1, "More airflow never raises the thermal resistance."),
        ("log_k", 0, -1, "A more conductive material never raises the thermal resistance."),
        ("log_fw", 0, -1, "A wider heat source spreads less (lower resistance)."),
        ("log_V", 1, +1, "Pressure drop grows with air velocity."),
    ],
    "natural": [
        ("log_Q", 0, -1, "In natural convection more power drives more buoyant flow, "
                         "so the resistance falls as power rises."),
        ("log_k", 0, -1, "A more conductive material never raises the thermal resistance."),
        ("log_fw", 0, -1, "A wider heat source spreads less (lower resistance)."),
    ],
}


def _f(x) -> float:
    return float(x)


def _grad(sur, Z: torch.Tensor, out_idx: int) -> np.ndarray:
    """d(log output)/d(raw feature) at standardized inputs Z."""
    Z = Z.clone().requires_grad_(True)
    y = sur.model(Z)
    y = y.y if hasattr(y, "y") else y
    g = torch.autograd.grad(y[:, out_idx].sum(), Z)[0].detach().numpy()
    return g * (sur.y_std[out_idx] / sur.x_std)[None, :]


def _convergence(history: Dict[str, List[float]]) -> Dict:
    """Adam stage (noisy, fixed learning rate) + L-BFGS polish (deterministic):
    converged when the last L-BFGS round no longer improves the loss."""
    adam = convergence_section(history["adam"])
    rounds = [float(x) for x in history["lbfgs_rounds"]]
    out = {"adam_stage": adam, "lbfgs_round_losses": rounds}
    if len(rounds) >= 2 and all(map(np.isfinite, rounds)):
        last_gain = (rounds[-2] - rounds[-1]) / max(rounds[-2], 1e-30)
        total = history["adam"][0] / max(rounds[-1], 1e-30)
        out["reduction_factor"] = float(total)
        out["last_round_improvement_pct"] = float(100 * last_gain)
        if last_gain < 0.02 and total >= 10:
            out.update(status="converged", message="L-BFGS polish no longer improves the loss "
                       f"(last round {100 * last_gain:.2f}%); total reduction {total:.0f}x.")
        else:
            out.update(status="still_improving", message="The loss was still falling in the last "
                       "L-BFGS round -- more rounds could help.")
    else:
        out.update(status=adam["status"], message=adam.get("message", ""),
                   reduction_factor=adam.get("reduction_factor"))
    return out


def surrogate_report(sur, *, X, Y, split, history, train_seconds, n_rows) -> Dict:
    from .surrogate import COMMON_RANGES, FEATURES, MODE_RANGES, TARGETS

    mode = sur.mode
    names = FEATURES[mode]
    tr, te, cal = split["train"], split["test"], split["cal"]
    P_te, P_tr = sur.predict_log(X[te]), sur.predict_log(X[tr])
    rel_te = np.exp(P_te - Y[te]) - 1.0            # relative error of the physical output
    rel_tr = np.exp(P_tr - Y[tr]) - 1.0

    error = {"n_test_designs": int(len(te)), "measured_on": "held-out designs (never trained on)",
             "per_output": {}}
    for j, t in enumerate(TARGETS[mode]):
        e = np.abs(rel_te[:, j]) * 100
        label = {"log_R": "thermal resistance (base to air)", "log_dP": "pressure drop"}[t]
        error["per_output"][label] = {
            "mean_abs_pct": _f(e.mean()), "p95_abs_pct": _f(np.percentile(e, 95)),
            "max_abs_pct": _f(e.max()), "pct_within_5pct": _f(np.mean(e <= 5) * 100),
        }
    main_err = error["per_output"]["thermal resistance (base to air)"]

    gap = float(np.mean(np.abs(rel_te[:, 0])) / max(np.mean(np.abs(rel_tr[:, 0])), 1e-12))
    generalization = {
        "train_mean_abs_pct": _f(np.mean(np.abs(rel_tr[:, 0])) * 100),
        "test_mean_abs_pct": main_err["mean_abs_pct"],
        "test_over_train": gap,
        "status": "good" if gap <= 1.5 else "moderate" if gap <= 3 else "poor",
        "valid_design_space": {**COMMON_RANGES, **MODE_RANGES[mode]},
        "note": "Outside this design space the surrogate is not used: the sizer only samples "
                "inside it, and every proposal is re-verified by the physics engine.",
    }

    Z = torch.as_tensor((X[te] - sur.x_mean) / sur.x_std, dtype=torch.float32)
    G = _grad(sur, Z, 0)
    elasticity = {READABLE[n]: _f(np.median(G[:, i])) for i, n in enumerate(names)}
    ranked = sorted(elasticity.items(), key=lambda kv: -abs(kv[1]))
    variables = {
        "inputs": [READABLE[n] for n in names],
        "outputs": [{"log_R": "thermal resistance", "log_dP": "pressure drop"}[t] for t in TARGETS[mode]],
        "elasticity_of_resistance": dict(ranked),
        "elasticity_note": "Median % change of the thermal resistance per +1% of each input "
                           "(per +1 °C for ambient temperature). Negative = helps cooling.",
    }

    params = torch.cat([p.detach().reshape(-1) for p in sur.model.parameters()])
    mse_out = np.mean((P_te - Y[te]) ** 2, axis=0) / sur.y_std ** 2
    weights = {
        "loss_terms": [{"term": t, "weight": 1.0, "test_mse_standardized": _f(v),
                        "share_pct": _f(100 * v / mse_out.sum())}
                       for t, v in zip(TARGETS[mode], mse_out, strict=True)],
        "network": {"architecture": "PINNeAPPle ModelRegistry 'bench_mlp' (4x128, SiLU)",
                    "n_params": int(params.numel()), "l2_norm": _f(params.norm()),
                    "max_abs": _f(params.abs().max()),
                    "non_finite": int((~torch.isfinite(params)).sum())},
        "training": {"optimizer": "Adam (PINNeAPPle Trainer, best-val checkpoint) + L-BFGS polish",
                     "n_physics_samples": int(n_rows), "split": {k: int(len(v)) for k, v in split.items()},
                     "train_seconds": _f(train_seconds)},
    }

    uq = {"method": "split conformal on log thermal resistance (calibration split)",
          "n_calibration": int(len(cal)), "levels": {}}
    for cov, q in sur.conformal_log_q.items():
        covered = np.mean(np.abs(P_te[:, 0] - Y[te, 0]) <= q) * 100
        uq["levels"][cov] = {"multiplicative_factor": _f(np.exp(q)),
                             "empirical_test_coverage_pct": _f(covered),
                             "reading": f"true resistance <= predicted x {np.exp(q):.3f} "
                                        f"for {cov}% of designs"}
    uq["use"] = "The sizer screens with the 95% upper bound, then verifies with physics."

    checks: List[Dict] = []
    for feat, j, sign, law in LAWS[mode]:
        Gj = G if j == 0 else _grad(sur, Z, j)
        g = sign * Gj[:, names.index(feat)]
        strict = float(np.mean(g >= 0) * 100)
        # A wrong-sign slope only matters if doubling the input would move the
        # output by more than the model's own mean error (else it is noise).
        tol = np.mean(np.abs(rel_te[:, j])) / np.log(2) if feat != "T_amb" else 0.0
        ok = float(np.mean(g >= -tol) * 100)
        checks.append({"name": f"monotonic:{READABLE[feat]}", "law": law, "value": ok,
                       "strict_pct": strict,
                       "status": "pass" if ok >= 99 else "warn" if ok >= 95 else "fail",
                       "detail": f"{ok:.1f}% of held-out designs have the physically required sign "
                                 f"beyond the model's noise level; {strict:.1f}% with no tolerance "
                                 f"at all (median effect {np.median(Gj[:, names.index(feat)]):+.3f})."})
    checks.append({"name": "positive_resistance", "status": "pass", "value": None,
                   "law": "R > 0 (2nd law)", "detail": "Guaranteed by construction (log output)."})
    cov95 = uq["levels"].get("95", {}).get("empirical_test_coverage_pct", 0)
    checks.append({"name": "uncertainty_calibrated", "law": "95% interval covers 95% of designs",
                   "value": cov95, "status": "pass" if cov95 >= 93 else "warn" if cov95 >= 88 else "fail",
                   "detail": "Empirical coverage of the conformal 95% bound on held-out designs."})

    report = {
        "mode": mode,
        "error": error,
        "convergence": _convergence(history),
        "generalization": generalization,
        "variables": variables,
        "weights": weights,
        "uncertainty": uq,
        "physics_checks": checks,
    }
    reasons_fail, reasons_warn = [], []
    if main_err["p95_abs_pct"] > 10:
        reasons_fail.append(f"p95 error {main_err['p95_abs_pct']:.1f}% on held-out designs")
    elif main_err["p95_abs_pct"] > 5:
        reasons_warn.append(f"p95 error {main_err['p95_abs_pct']:.1f}% on held-out designs")
    if report["convergence"]["status"] in ("diverged", "not_run"):
        reasons_fail.append("training did not converge")
    elif report["convergence"]["status"] != "converged":
        reasons_warn.append(f"training {report['convergence']['status'].replace('_', ' ')}")
    if generalization["status"] == "poor":
        reasons_fail.append("poor generalization")
    for c in checks:
        (reasons_fail if c["status"] == "fail" else reasons_warn if c["status"] == "warn"
         else []).append(f"{c['name']}: {c['status']}")
    report["verdict"] = {
        "status": "not_trustworthy" if reasons_fail else "use_with_caution" if reasons_warn
        else "trustworthy",
        "reasons": (reasons_fail + reasons_warn) or [
            f"mean error {main_err['mean_abs_pct']:.2f}%, p95 {main_err['p95_abs_pct']:.2f}% on "
            f"{len(te)} held-out designs; all physics checks pass"],
        "role": "Screening only: recommended designs are always re-verified by the physics engine.",
    }
    return report
