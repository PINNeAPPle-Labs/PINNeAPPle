"""Closed-loop optical proximity correction: imaging model, a learned operator only when it pays off, inverse
lithography, and the mask re-verified by the exact model over the process window (#1180).

1. **Exact model**: SOCS kernels at each focus of the process window (Hopkins, equal to Abbe when every kernel is
   kept; ``kernel_energy`` truncates them for speed and the final verification uses the full Abbe sum).
2. **Measure** the cost of one exact image; the evaluations are ``clips x ILT steps x corners`` (``clips``: how
   many layout clips this correction will run on).
3. **Ask** :func:`~pinneapple_analysis.verification.architecture_recommendation.recommend_architecture` (a grid
   field): ``none`` when the exact model is cheap enough; otherwise a neural operator, trained on exact images and
   checked (held-out error, finer resolution, printed-pattern EPE) before it is used.
4. **ILT** through the exact kernels at every corner, or through the operator at best focus.
5. **Verify** the binarized mask with the full Abbe model: EPE at nominal against the uncorrected mask, the EPE
   over focus x dose (depth of focus, exposure latitude), the mask's smallest feature against the mask rule, and
   Abbe against the kernels on that mask.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .ilt import optimise_mask
from .imaging import Optics, Source, aerial_image, aerial_image_socs, socs_kernels
from .metrics import process_window


@dataclass
class LithoSpec:
    target: np.ndarray                     # (N, N) bool: where resist must remain (dark tone) or clear (clear tone)
    L: float                               # cell size, nm (wafer)
    optics: Optics = field(default_factory=Optics)
    source: Source = field(default_factory=lambda: Source("annular", 0.9, 0.6, n=6))
    threshold: float = 0.3
    diffusion: float = 5.0                 # nm, acid diffusion length
    tone: str = "dark"
    focus: tuple[float, ...] = (-50.0, 0.0, 50.0)
    doses: tuple[float, ...] = (0.95, 1.0, 1.05)
    epe_tol: float = 2.0                   # nm, mean EPE allowed over the process window
    min_feature_nm: float = 16.0           # mask rule: smallest feature or space (wafer scale)
    clips: int = 1
    assumptions: list[str] = field(default_factory=lambda: [
        "scalar imaging: polarization and mask 3-D topography not modelled (vector effects lower the contrast at "
        "NA > 0.7)", "threshold resist with Gaussian acid diffusion (no development kinetics)",
        "periodic cell: the clip repeats (its edges see the neighbouring copies)"])


def min_feature(mask: np.ndarray, L: float) -> float:
    """Smallest run of equal pixels along any row or column, nm (features and spaces of the mask)."""
    m = np.asarray(mask, bool)
    px = L / m.shape[0]
    best = m.shape[0]
    for a in (m, m.T):
        for row in a:
            if row.all() or (~row).all():
                continue
            k = int(np.argmax(row != row[0]))                  # start at a transition (periodic rows)
            r = np.roll(row, -k)
            edges = np.flatnonzero(np.diff(np.concatenate([r, r[:1]]).astype(int)) != 0)
            runs = np.diff(np.concatenate([[-1], edges]))
            best = min(best, int(runs.min()))
    return float(best * px)


def correct_mask(spec: LithoSpec, *, steps: int = 150, lr: float = 0.5, force_surrogate: bool = False,
                 kernel_energy: float = 0.995, operator: dict | None = None, seed: int = 0) -> dict[str, Any]:
    N = spec.target.shape[0]
    corners_f = sorted(set(spec.focus))
    t0 = time.perf_counter()
    socs = {z: socs_kernels(N, spec.L, replace(spec.optics, defocus=z), spec.source, energy=kernel_energy)
            for z in corners_f}
    kernel_s = time.perf_counter() - t0
    no_opc = np.asarray(spec.target, bool) if spec.tone == "clear" else ~np.asarray(spec.target, bool)
    t1 = time.perf_counter()
    for _ in range(3):
        aerial_image_socs(no_opc.astype(float), socs[0.0])
    solve_s = (time.perf_counter() - t1) / 3
    corners = [(z, d) for z in corners_f for d in spec.doses]          # every corner of the process window
    n_eval = spec.clips * steps * len(corners)

    rec: dict[str, Any] = {"recommended": ["none"], "reasoning": "recommendation unavailable", "checks": []}
    try:
        from pinneapple_analysis.verification.architecture_recommendation import (
            recommend_architecture,
        )
        r = recommend_architecture(n_high_fidelity_simulations=0, needs_parameter_generalization=True,
                                   geometry_varies=True, has_analytical_solver=True, solve_seconds=solve_s,
                                   n_evaluations=n_eval, output_kind="grid_field")
        rec = {"recommended": list(r.recommended), "reasoning": r.reasoning, "decision_path": list(r.decision_path),
               "checks": list(r.checks)}
    except (ImportError, TypeError) as e:
        rec["reasoning"] = f"recommendation unavailable: {e}"
    use_op = force_surrogate or rec["recommended"][0] != "none"
    op_report = None
    if use_op:
        from .surrogate import operator_imager, train_operator
        if operator is None:
            fine = socs_kernels(2 * N, spec.L, spec.optics, spec.source, energy=kernel_energy)
            operator = train_operator(socs[0.0], spec.L, N, threshold=spec.threshold, socs_fine=fine, seed=seed)
        op_report = {k: v for k, v in operator.items() if k != "model"}
        imager = operator_imager(operator["model"])
        ilt_corners = [(0.0, d) for d in spec.doses]          # the operator knows best focus only
    else:
        def imager(m, z):
            return aerial_image_socs(m, socs[z])
        ilt_corners = corners
    t2 = time.perf_counter()
    res = optimise_mask(spec.target, spec.L, imager, corners=ilt_corners, threshold=spec.threshold,
                        diffusion=spec.diffusion, tone=spec.tone, steps=steps, lr=lr,
                        filter_nm=spec.min_feature_nm / 2, min_feature_nm=spec.min_feature_nm)
    ilt_s = time.perf_counter() - t2
    mask = res["mask"]

    # verification: the full Abbe sum, never the operator
    def abbe(m, z):
        return aerial_image(m.astype(float), spec.L, replace(spec.optics, defocus=z), spec.source)

    def pw(m):
        return process_window(lambda z: abbe(m, z), spec.target, spec.L, focus=list(corners_f),
                              doses=list(spec.doses), threshold=spec.threshold, diffusion=spec.diffusion,
                              tone=spec.tone, epe_tol=spec.epe_tol)
    pw_opc, pw_raw = pw(mask), pw(no_opc)
    jn = list(spec.doses).index(1.0) if 1.0 in spec.doses else 0
    i0 = corners_f.index(0.0) if 0.0 in corners_f else 0
    epe_opc, epe_raw = pw_opc["epe"][i0][jn], pw_raw["epe"][i0][jn]
    worst = float(np.max(pw_opc["epe"]))
    mf = min_feature(mask, spec.L)
    d_model = float((abbe(mask, 0.0) - aerial_image_socs(mask.astype(float), socs[0.0])).abs().max())
    checks = []

    def chk(code, ok, detail, value=None):
        ok = None if ok is None else bool(ok)
        checks.append({"code": code, "status": "pass" if ok is True else ("fail" if ok is False else "inconclusive"),
                       "detail": detail, "value": value})
    chk("litho.epe_nominal", epe_opc <= spec.epe_tol, f"mean EPE {epe_opc:.2f} nm at best focus and dose (uncorrected "
                                                       f"mask: {epe_raw:.2f} nm), full Abbe image", epe_opc)
    chk("litho.process_window", worst <= spec.epe_tol, f"worst mean EPE {worst:.2f} nm over focus "
                                                       f"{list(corners_f)} nm x dose {list(spec.doses)} "
                                                       f"(uncorrected: {float(np.max(pw_raw['epe'])):.2f} nm)", worst)
    chk("litho.improves", epe_opc < epe_raw, "the corrected mask prints closer to the target than the target "
                                              "itself used as the mask", epe_raw - epe_opc)
    chk("mask.min_feature", mf >= spec.min_feature_nm, f"smallest mask feature or space {mf:.1f} nm, rule "
                                                        f"{spec.min_feature_nm:g} nm", mf)
    chk("model.kernels_vs_abbe", d_model < 0.02, f"largest intensity difference between the {kernel_energy:.3f}-"
                                                  f"energy kernels and the full Abbe sum on the mask: {d_model:.2e}",
        d_model)
    if op_report:
        chk("surrogate.held_out_error", op_report["held_out_rel_l2"] <= 0.05,
            f"relative L2 {op_report['held_out_rel_l2']:.2%} on {op_report['n_test']} held-out clips", None)
        if "fine_rel_l2" in op_report:
            chk("surrogate.resolution_invariance", op_report["fine_rel_l2"] <= 0.1,
                f"relative L2 {op_report['fine_rel_l2']:.2%} at twice the sampling", None)
        chk("surrogate.printed_epe", op_report["held_out_epe_nm"] <= spec.epe_tol,
            f"EPE between patterns printed from predicted and exact images: {op_report['held_out_epe_nm']:.2f} nm",
            None)
    return {"mask": mask, "mask_continuous": res["mask_continuous"], "checks": checks,
            "process_window": {"corrected": pw_opc, "uncorrected": pw_raw},
            "epe_nm": {"corrected": epe_opc, "uncorrected": epe_raw}, "min_feature_nm": mf,
            "recommendation": rec, "surrogate": op_report,
            "cost": {"image_seconds": solve_s, "kernel_seconds": kernel_s, "ilt_seconds": ilt_s,
                     "planned_evaluations": n_eval, "kernels": {str(z): int(s["weights"].shape[0])
                                                                 for z, s in socs.items()}},
            "loss_history": res["loss_history"], "assumptions": spec.assumptions}


def clip_lines_and_elbow(N: int = 128, L: float = 1024.0) -> np.ndarray:
    """A test clip (dark tone): three 64 nm lines at 192 nm pitch with line ends, and an elbow."""
    px = L / N
    t = np.zeros((N, N), bool)
    w = int(round(64 / px))
    for k in range(3):
        x0 = int(round((256 + k * 192) / px))
        t[x0:x0 + w, int(256 / px):int(768 / px)] = True
    t[int(840 / px):int(840 / px) + w, int(200 / px):int(600 / px)] = True
    t[int(840 / px):int(1000 / px), int(536 / px):int(536 / px) + w] = True
    return t
