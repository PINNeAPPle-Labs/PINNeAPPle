"""The physics checks the operator has to pass before any number it produces is reported.

PINNeAPPle's rule is that a field is never reported without its checks, so this module is what stands between the
surrogate and the dashboard. Three layers, each answering a different question:

``compare`` -- is the prediction the solver's answer, at the nodes and on the grid? Relative L2 per field, the
**peak** of von Mises (the number an engineer reads: the stress concentration at the flare, where a surrogate that
looks fine on L2 is usually wrong), the displacement at the load point, and the displacement at the clamped face
which has to be zero whatever the model says.

``field_trust`` -- does the field survive PINNeAPPle's own checks? ``pinneapple_analysis.trust.field_card.trust_card``
on the predicted von Mises against the CalculiX reference: its checks, its deploy decision and its tier, unchanged
and not re-implemented here.

``linearity`` -- is the assumption that lets the dashboard's load slider be exact actually true? The step is linear
elastic, so ``u(F) = F / F_ref * u(F_ref)``; that is a claim about ``ccx`` and it is measured, not assumed, by
solving the nominal design at two loads and comparing.

A check that cannot run is reported as not run, with its reason. Nothing here widens a tolerance or drops a failure.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .design import REFERENCE_LOAD_N, DesignSpace
from .grid import sample_field

__all__ = ["compare", "field_trust", "linearity", "report"]

DEFAULTS = {"rel_l2_displacement": 0.10, "rel_peak_vm": 0.20, "clamped_displacement_fraction": 1e-3}


def compare(surrogate, samples: dict[str, np.ndarray], index: int | np.ndarray | None = None, *,
            tolerances: dict[str, float] | None = None) -> dict[str, Any]:
    """Score the surrogate against the CalculiX samples it was not trained on.

    ``index`` selects which samples to score (default: all of them). Returns the per-sample numbers, their summary,
    and the list of checks that failed -- empty when the operator is within :data:`DEFAULTS`.
    """
    tol = {**DEFAULTS, **dict(tolerances or {})}
    idx = np.arange(len(samples["occupancy"])) if index is None else np.atleast_1d(index)
    grid = surrogate.grid
    rows = []
    for i in idx:
        i = int(i)
        occ = np.asarray(samples["occupancy"][i]).astype(bool)
        feats = surrogate.features_of({"occupancy": occ})
        pred = surrogate.predict_fields(feats)
        ref_u = np.stack([samples["u"][i][..., k] for k in range(3)], axis=-1)
        ref_vm = samples["vm"][i]
        part = occ

        def rel(p, t, mask):
            a, b = np.asarray(p)[mask], np.asarray(t)[mask]
            den = np.linalg.norm(b)
            return float(np.linalg.norm(a - b) / den) if den > 0 else float("nan")

        pred_u = np.stack([pred["ux"], pred["uy"], pred["uz"]], axis=-1)
        peak_pred = float(np.percentile(pred["vm"][part], 99.5)) if part.any() else float("nan")
        peak_ref = float(np.percentile(ref_vm[part], 99.5)) if part.any() else float("nan")
        nodes = np.asarray(samples["nodes"][i]) if "nodes" in samples else None
        clamped = None
        tip = None
        if nodes is not None:
            zmin, zmax = float(nodes[:, 2].min()), float(nodes[:, 2].max())
            t = 1e-6 * max(1.0, zmax)
            base = nodes[:, 2] <= zmin + t
            top = nodes[:, 2] >= zmax - t
            u_nodes = np.stack([sample_field(pred[c], grid, nodes) for c in ("ux", "uy", "uz")], axis=-1)
            u_true = np.asarray(samples["u_nodes"][i])
            clamped = float(np.abs(u_nodes[base]).max() / max(np.abs(u_true).max(), 1e-30))
            tip = {"predicted_m": float(np.abs(u_nodes[top]).mean()),
                   "solver_m": float(np.abs(u_true[top]).mean())}
        rows.append({
            "index": i,
            "design": {k: float(v) for k, v in zip(("t", "dc", "r", "h"), samples["design"][i], strict=False)},
            "rel_l2_displacement": rel(pred_u, ref_u, part),
            "rel_l2_vm": rel(pred["vm"], ref_vm, part),
            "peak_vm_predicted_pa": peak_pred, "peak_vm_solver_pa": peak_ref,
            "rel_peak_vm": abs(peak_pred - peak_ref) / peak_ref if peak_ref else float("nan"),
            "clamped_displacement_fraction": clamped,
            "tip_displacement": tip,
        })
    summary = {}
    for key in ("rel_l2_displacement", "rel_l2_vm", "rel_peak_vm", "clamped_displacement_fraction"):
        vals = np.array([r[key] for r in rows if r[key] is not None], float)
        vals = vals[np.isfinite(vals)]
        summary[key] = {"median": float(np.median(vals)) if len(vals) else float("nan"),
                        "p90": float(np.percentile(vals, 90)) if len(vals) else float("nan"),
                        "max": float(vals.max()) if len(vals) else float("nan")}
    failed = []
    if summary["rel_l2_displacement"]["p90"] > tol["rel_l2_displacement"]:
        failed.append(f"displacement p90 {summary['rel_l2_displacement']['p90']:.3f} > "
                      f"{tol['rel_l2_displacement']:.3f}")
    if summary["rel_peak_vm"]["p90"] > tol["rel_peak_vm"]:
        failed.append(f"peak von Mises p90 {summary['rel_peak_vm']['p90']:.3f} > {tol['rel_peak_vm']:.3f}")
    if summary["clamped_displacement_fraction"]["max"] > tol["clamped_displacement_fraction"]:
        failed.append(f"clamped face moves {summary['clamped_displacement_fraction']['max']:.2e} of the peak "
                      f"(limit {tol['clamped_displacement_fraction']:.0e})")
    return {"n": len(rows), "rows": rows, "summary": summary, "tolerances": tol, "failed": failed}


def field_trust(surrogate, samples: dict[str, np.ndarray], index: int, *, name: str = "von Mises (surrogate)",
                limitations=()) -> dict[str, Any]:
    """PINNeAPPle's field trust card for one prediction against its CalculiX reference.

    The card is the library's own (``pinneapple_analysis.trust.field_card.trust_card``); this only hands it the
    grid, the predicted field and the reference, so the deploy decision is not something this pipeline invents.
    """
    from pinneapple_analysis.trust.field_card import trust_card
    i = int(index)
    occ = np.asarray(samples["occupancy"][i]).astype(bool)
    pred = surrogate.predict_fields(surrogate.features_of({"occupancy": occ}))
    axes = surrogate.grid.axes()
    grid = {"x": axes[0], "y": axes[1], "z": axes[2]}
    card = trust_card(grid, pred["vm"], name=f"{name}, sample {i}", is_ml=True,
                      reference=np.asarray(samples["vm"][i]),
                      limitations=list(limitations) or ["trained on the pedestal family only",
                                                        "linear elastic, small strain, one load direction"],
                      train_bounds={})
    return {"decision": str(getattr(card, "decision", "") or ""),
            "tier": (getattr(card, "stages", {}) or {}).get("tier"),
            "reasons": [str(r) for r in (getattr(card, "reasons", []) or [])],
            "checks": [{"name": c.name, "passed": c.passed, "critical": c.critical, "value": c.value,
                        "limit": getattr(c, "limit", getattr(c, "threshold", None)),
                        "detail": getattr(c, "detail", "")} for c in getattr(card, "checks", [])],
            "card": card}


def linearity(space: DesignSpace, design: dict[str, float], workdir: str, *,
              loads=(REFERENCE_LOAD_N, 2.5 * REFERENCE_LOAD_N), element_size: float = 0.024,
              threads: int = 2) -> dict[str, Any]:
    """Solve one design at two loads and measure how far the fields are from scaling exactly.

    This is the check behind the dashboard's load slider: the surrogate is trained at one load and the slider
    multiplies. If the step were not linear the slider would be a lie, so it is measured with two ``ccx`` runs.
    """
    from .design import run_design
    out = []
    for k, F in enumerate(loads):
        r = run_design(design, f"{workdir}/lin_{k}", element_size=element_size, threads=threads)
        out.append({"load_n": float(F), "u": r["u"], "vm": r["von_mises"]})
    u0, u1 = out[0]["u"], out[1]["u"]
    ratio = float(loads[1]) / float(loads[0])
    u_err = float(np.linalg.norm(u1 - ratio * u0) / max(np.linalg.norm(u1), 1e-30))
    vm_err = float(np.linalg.norm(out[1]["vm"] - ratio * out[0]["vm"]) / max(np.linalg.norm(out[1]["vm"]), 1e-30))
    return {"loads_n": [float(x) for x in loads], "ratio": ratio,
            "displacement_scaling_error": u_err, "von_mises_scaling_error": vm_err,
            "linear": bool(u_err < 1e-6 and vm_err < 1e-6),
            "note": "linear elastic: u and von Mises scale exactly with the load, so the dashboard's slider is exact"}


def report(surrogate, samples: dict[str, np.ndarray], index: int | np.ndarray | None = None,
           **kwargs) -> dict[str, Any]:
    """``compare`` plus one trust card per scored sample, as written to ``validation.json``."""
    cmp = compare(surrogate, samples, index, **kwargs)
    cards = []
    worst = max(cmp["rows"], key=lambda r: r["rel_peak_vm"]) if cmp["rows"] else None
    if worst is not None:
        card = field_trust(surrogate, samples, worst["index"])
        cards.append({k: v for k, v in card.items() if k != "card"})
    return {"comparison": cmp, "trust_cards": cards,
            "worst_sample": worst["index"] if worst else None}
