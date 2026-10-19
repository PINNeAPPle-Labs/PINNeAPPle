"""PINNeAPPle's validation stack applied to a benchmark's prepared artifacts, as recorded checks.

Wrappers (each returns ``Check`` objects; nothing is skipped silently, an item that does not apply says why):

* ``physics_audit_check``: ``pinneapple_physics.physics_audit`` (ModelProfile + audit_assumptions).
* ``envelope_checks``: applicability domain with ``physics_audit.Envelope``: test inputs inside the training
  envelope, OOD inputs outside it.
* ``gci_check``: three-grid convergence with ``pinneapple_design.fitmesh.gci`` (observed order, GCI, asymptotic range).
* ``mesh_quality_check``: ``pinneapple_data.cae.mesh_report`` on a mesh.
* ``preflight_check``: ``pinneapple_data.preflight`` on a solver case (OpenFOAM or CalculiX).
* ``yplus_check``: first-cell height against ``verification.mesh_intelligence.estimate_required_first_cell_height``.
* ``trust_card``: the six-stage trust card of ``pinneapple_lab.curation.trust_stages`` from the benchmark's checks.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .registry import Check

__all__ = ["physics_audit_check", "envelope_checks", "gci_check", "mesh_quality_check", "preflight_check",
           "yplus_check", "trust_card", "field_validation", "field_features"]


def physics_audit_check(subject: str, name: str, family: str, equations: Sequence[str], assumptions: Sequence,
                        features: Mapping[str, float]) -> Check:
    from pinneapple_physics.physics_audit import ModelProfile, audit_assumptions
    prof = ModelProfile(name, family, tuple(equations), tuple(assumptions))
    a = audit_assumptions(prof, dict(features))
    ok = None if a.status == "UNKNOWN" else a.status in ("VALID", "MARGINAL")
    return Check(f"physics audit: {name}", "validation", subject, ok, value=a.status, severity="warning",
                 detail="; ".join(a.lines()[:4]))


def envelope_checks(subject: str, train: Sequence[Mapping[str, float]], test: Sequence[Mapping[str, float]],
                    ood: Sequence[Mapping[str, float]], features: Sequence[str] | None = None) -> list[Check]:
    from pinneapple_physics.physics_audit import Envelope
    env = Envelope.from_samples(list(train), features)
    inside = [env.check(f)["inside"] for f in test]
    out = [Check("applicability: test inputs inside the training envelope", "validation", subject,
                 float(np.mean(inside)) >= 0.9 if inside else None, value=f"{int(np.sum(inside))}/{len(inside)}",
                 threshold=">= 90 %", severity="warning",
                 detail=f"physics_audit.Envelope over {', '.join(env.sample_features)}")]
    if ood:
        outside = [not env.check(f)["inside"] for f in ood]
        out.append(Check("applicability: OOD inputs outside the training envelope", "validation", subject,
                         float(np.mean(outside)) >= 0.5, value=f"{int(np.sum(outside))}/{len(outside)}",
                         threshold=">= 50 %", severity="warning",
                         detail="an 'OOD' set that sits inside the envelope tests interpolation, not extrapolation"))
    return out


def gci_check(subject: str, quantity: str, values_fine_to_coarse: Sequence[float], sizes_fine_to_coarse: Sequence[float],
              max_gci: float = 0.05) -> Check:
    from pinneapple_design.fitmesh import gci
    v = np.asarray(values_fine_to_coarse, float)
    spread = float(np.ptp(v) / max(np.abs(v).max(), 1e-300))
    name = f"grid convergence (GCI) of {quantity}"
    vals = f"values (fine to coarse) {[f'{x:.8g}' for x in v]}"
    if spread < 1e-3:                                   # the three grids agree: GCI is noise (and may oscillate)
        return Check(name, "validation", subject, True, value=f"spread {spread:.1e} across the 3 grids",
                     threshold="spread < 1e-3 or GCI <= " + f"{max_gci:g}", detail=f"converged; {vals}")
    g = gci(v, sizes_fine_to_coarse)
    gf = g.get("gci_fine")
    p = g.get("p")
    if g["status"] == "warn":                           # oscillatory convergence with a real spread
        return Check(name, "validation", subject, False, value=f"oscillatory, spread {spread:.1e}", severity="warning",
                     detail=f"fitmesh.gci: {g.get('note', '')}; {vals}")
    ok = g["status"] != "fail" and (gf is None or gf <= max_gci)
    return Check(name, "validation", subject, ok,
                 value=f"p = {p:.2f}, GCI_fine = {gf:.2e}" if p is not None and gf is not None else g.get("note", g["status"]),
                 threshold=f"GCI <= {max_gci:g}", detail=f"fitmesh.gci; {vals}; {g.get('note', '')}")


def mesh_quality_check(subject: str, mesh, target: str = "cfd") -> Check:
    from pinneapple_data.cae import mesh_report
    r = mesh_report(mesh, target=target)
    checks = r.get("checks", [])
    bad = [c for c in checks if str(c.get("verdict", "")).lower() in ("fail", "bad", "error")]
    warn = [c for c in checks if str(c.get("verdict", "")).lower() in ("warn", "warning", "marginal")]
    return Check("mesh quality (pinneapple_data.cae.mesh_report)", "validation", subject, not bad,
                 value={"fail": len(bad), "warn": len(warn), "checks": len(checks)}, severity="error",
                 detail="; ".join(f"{c.get('name', c.get('key', '?'))}: {c.get('verdict')}" for c in (bad + warn)[:5]))


def preflight_check(subject: str, case_dir: Path) -> Check:
    from pinneapple_data.preflight import preflight
    case_dir = Path(case_dir)
    fs = {str(p.relative_to(case_dir)): p.read_bytes() for p in case_dir.rglob("*") if p.is_file()}
    rep = preflight(fs)
    return Check("simulation preflight (pinneapple_data.preflight)", "validation", subject, rep["status"] != "FAIL",
                 value=rep["status"], severity="error", detail=f"counts {rep.get('counts')}; " + "; ".join(
                     f"{f.get('status')}: {f.get('title')}" for f in rep.get("findings", [])
                     if f.get("status") in ("FAIL", "WARNING"))[:400])


def yplus_check(subject: str, first_cell: float, length: float, velocity: float, nu: float,
                target: float = 1.0) -> Check:
    """First-cell y+ from the flat-plate estimate (``length``: the reference length, e.g. the chord)."""
    from pinneapple_analysis.verification.mesh_intelligence import (
        estimate_required_first_cell_height,
    )
    est = estimate_required_first_cell_height(reynolds_length=velocity * length / nu, velocity=velocity,
                                              kinematic_viscosity=nu, target_yplus=target)
    need = float(est.required_first_cell_height)
    yplus = target * first_cell / need if need > 0 else math.nan
    return Check("first cell y+ (mesh_intelligence estimate)", "validation", subject, yplus <= 1.5 * target,
                 value=round(yplus, 3), threshold=f"<= {1.5 * target:g}", severity="warning",
                 detail=f"first cell {first_cell:.2e} vs {need:.2e} needed for y+ = {target}")


# ---------------------------------------------------------------------- field problems
def field_features(x: np.ndarray) -> dict[str, float]:
    """Scalar descriptors of an input field: amplitude and spectral centroid (for the applicability envelope)."""
    a = np.asarray(x, float)
    spec = np.abs(np.fft.rfftn(a - a.mean())) ** 2
    k = np.sqrt(sum(g ** 2 for g in np.meshgrid(*[np.fft.fftfreq(n, 1 / n) if i < a.ndim - 1 else
                                                  np.fft.rfftfreq(n, 1 / n) for i, n in enumerate(a.shape)],
                                                 indexing="ij")))
    return {"amplitude": float(a.std()), "spectral_centroid": float((k * spec).sum() / max(spec.sum(), 1e-30))}


_features = field_features


def field_validation(problem, out: Path, grid_study=None) -> list[Check]:
    z = np.load(Path(out) / "data.npz")
    f = {s: [_features(x) for x in z[f"{s}_x"]] for s in ("train", "test", "ood")}
    C = envelope_checks(problem.id, f["train"], f["test"], f["ood"], ["amplitude", "spectral_centroid"])
    if grid_study is not None:
        C.append(grid_study())
    return C


# ---------------------------------------------------------------------- trust card
def trust_card(out_dir: Path, checks: Sequence[Check], tier: str, limitations: Sequence[str]) -> dict[str, Any]:
    """The six questions of pinneapple_lab's trust card, answered from the benchmark's own checks."""
    from pinneapple_lab.curation import trust_stages
    def kind(c):
        n = c.name.lower()
        if "residual" in n or "physics audit" in n or "conserv" in n:
            return "physics"
        if "applicability" in n or "ood" in n or "spread" in n or "determinism" in n:
            return "uncertainty"
        if c.stage == "problem" or "exact" in n or "order" in n:
            return "reference"
        if c.stage in ("data", "validation"):
            return "sanity"
        if c.stage == "architecture":
            return "generalization"
        return "baseline"
    cs = [{"name": c.name, "passed": bool(c.passed), "detail": str(c.detail)[:120], "kind": kind(c)}
          for c in checks if c.passed is not None]
    return trust_stages(str(out_dir), cs, {}, {}, tier, {}, {}, True, list(limitations))
