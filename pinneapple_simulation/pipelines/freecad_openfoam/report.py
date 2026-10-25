"""Validation of a study: the same case on successively finer meshes, Richardson extrapolation (Roache's GCI,
:func:`pinneapple_analysis.verification.convergence.richardson_extrapolate`), the error of every mesh and of the
extrapolated value against the reference, a table and figures.

    from pinneapple_simulation.pipelines.freecad_openfoam import studies
    from pinneapple_simulation.pipelines.freecad_openfoam.report import mesh_study
    ms = mesh_study(studies.heated_pipe, [12, 16, 24], "work/heated_pipe", "Nu")
    print(ms["extrapolated"], ms["extrapolated_error"])
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import PipelineSpec, run_pipeline

__all__ = ["mesh_study", "mesh_study_figure", "sweep_figure", "validation_markdown"]


def mesh_study(factory: Callable[..., PipelineSpec], resolutions: Sequence[float], folder: str | Path, reference: str,
               inputs: dict[str, float] | None = None, h_of: Callable[[float], float] | None = None,
               **kw) -> dict[str, Any]:
    """Run ``factory(resolution, **kw)`` for each resolution (coarse to fine) in ``folder/r<resolution>`` (resumed
    when already run) and extrapolate the measured value of reference ``reference``. ``h_of``: the cell size of
    a resolution (default 1 / resolution: cells across a length; for snappy surface levels pass ``lambda l: 2**-l``).
    """
    folder = Path(folder)
    h_of = h_of or (lambda r: 1.0 / r)
    rows = []
    for r in resolutions:
        spec = factory(r, **kw)
        rec = run_pipeline(spec, folder / f"r{r:g}", inputs)
        mj = folder / f"r{r:g}" / "post" / "measured.json"
        m = json.loads(mj.read_text()) if mj.is_file() else {"measured": {}, "references": {}}
        ref = m["references"].get(reference, {})
        wall = sum((s.ended or 0) - (s.started or 0) for s in rec.steps)
        rows.append({"resolution": r, "h": h_of(r), "state": rec.state, "cells": m["measured"].get("cells"),
                     "measured": ref.get("measured"), "reference": ref.get("reference"), "error": ref.get("error"),
                     "tol": ref.get("tol"), "source": ref.get("source", ""), "wall_s": wall, "folder": str(folder / f"r{r:g}"),
                     "iterations": m["measured"].get("iterations")})
    out: dict[str, Any] = {"study": spec.id, "title": spec.title, "quantity": reference, "rows": rows,
                           "inputs": inputs or spec.defaults()}
    vals = [r["measured"] for r in rows]
    if len(rows) >= 3 and all(v is not None for v in vals[-3:]):
        from pinneapple_analysis.verification.convergence import richardson_extrapolate
        c, m_, f = rows[-3:]
        try:
            res = richardson_extrapolate(c["measured"], m_["measured"], f["measured"],
                                         (m_["h"] / f["h"], c["h"] / m_["h"]))
            ext, ref = float(res.extrapolated_value), rows[-1]["reference"]
            if not res.observed_order > 0:                  # not converging monotonically: no extrapolation
                out["richardson_note"] = (f"observed order {res.observed_order:.2f} <= 0: the three meshes are not in "
                                          "the asymptotic range (changes below the discretisation noise)")
                return out
            out.update(extrapolated=ext, observed_order=float(res.observed_order), gci_fine=float(res.gci_fine),
                       asymptotic=bool(res.is_asymptotic), extrapolated_error=abs(ext - ref) / abs(ref),
                       citation=res.citation)
        except ValueError as exc:                         # equal values on two meshes: no order to estimate
            out["richardson_note"] = str(exc)
    return out


def mesh_study_figure(studies: Sequence[dict[str, Any]], path: str | Path, labels: Sequence[str] | None = None) -> Path:
    """One panel per mesh study: measured value against cell size, the reference and its tolerance band, the
    Richardson-extrapolated value at h = 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(studies), figsize=(4.6 * len(studies), 3.8), dpi=120, squeeze=False)
    for k, (ax, st) in enumerate(zip(axes[0], studies, strict=True)):
        rows = [r for r in st["rows"] if r["measured"] is not None]
        h = np.array([r["h"] for r in rows]) / rows[0]["h"]
        v = np.array([r["measured"] for r in rows])
        ref, tol = rows[-1]["reference"], rows[-1]["tol"] or 0.05
        ax.axhspan(ref * (1 - tol), ref * (1 + tol), color="#2ca02c", alpha=0.12, label=f"reference ± {100 * tol:g} %")
        ax.axhline(ref, color="#2ca02c", lw=1.4, label=rows[-1]["source"][:40])
        ax.plot(h, v, "o-", color="#1f77b4", label="measured")
        for hh, vv, r in zip(h, v, rows, strict=True):
            ax.annotate(f"{r['cells'] / 1e3:.0f}k cells", (hh, vv), textcoords="offset points", xytext=(4, 6), fontsize=7)
        if "extrapolated" in st:
            ax.plot([0], [st["extrapolated"]], "D", color="#d62728",
                    label=f"Richardson {st['extrapolated']:.4g} ({100 * st['extrapolated_error']:.1f} %)")
        ax.set_xlim(left=-0.05)
        ax.set_xlabel("cell size / coarsest cell size")
        ax.set_ylabel(st["quantity"])
        ax.set_title((labels[k] if labels else st["title"])[:60], fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def sweep_figure(rows: Sequence[dict[str, Any]], inputs: Sequence[str], output: str, study, path: str | Path,
                 unit: str = "") -> Path:
    """The sweep (``output`` over the first two inputs) and the surrogate's predictions against the solver on the
    training and the held-out cases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use = [r for r in rows if r.get(output) is not None and r.get("state") in ("passed", "rejected")]
    X = np.array([[r[k] for k in inputs] for r in use], float)
    y = np.array([r[output] for r in use], float)
    pred = np.asarray(study.model.predict({k: X[:, j].tolist() for j, k in enumerate(inputs)})["outputs"][output],
                      float)
    test = set(np.asarray(study.splits.test).tolist())
    is_test = np.array([i in test for i in range(len(use))])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=120)
    sc = a.scatter(X[:, 0], X[:, 1] if X.shape[1] > 1 else y, c=y, cmap="viridis", s=60,
                   edgecolor=np.where(is_test, "r", "k"), linewidth=np.where(is_test, 2.0, 0.6))
    fig.colorbar(sc, ax=a, label=f"{output} {unit}")
    a.set_xlabel(inputs[0])
    a.set_ylabel(inputs[1] if X.shape[1] > 1 else output)
    a.set_title(f"{len(use)} OpenFOAM cases (red edge: held out)", fontsize=9)
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    b.plot([lo, hi], [lo, hi], "k--", lw=1)
    b.scatter(y[~is_test], pred[~is_test], s=30, label="training / validation", color="#1f77b4")
    b.scatter(y[is_test], pred[is_test], s=50, label="held out", color="#d62728", marker="s")
    rank = study.ranking("test")
    best = next((r for r in rank if r["method"] == study.best), rank[0] if rank else None)
    b.set_title(f"surrogate {study.best}: held-out nRMSE {best['error']:.3g}" if best else "surrogate", fontsize=9)
    b.set_xlabel(f"OpenFOAM {output} {unit}")
    b.set_ylabel(f"surrogate {output} {unit}")
    b.legend(fontsize=8)
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def validation_markdown(studies: Sequence[dict[str, Any]]) -> str:
    """The measured validation table (one row per mesh, then the extrapolated value)."""
    out = ["| study | quantity | mesh | cells | measured | reference | error | source |", "|---|---|---|---|---|---|---|---|"]
    for st in studies:
        for r in st["rows"]:
            if r["measured"] is None:
                out.append(f"| {st['study']} | {st['quantity']} | {r['resolution']:g} | - | {r['state']} | | | |")
                continue
            out.append(f"| {st['study']} | {st['quantity']} | {r['resolution']:g} | {r['cells']} | {r['measured']:.4f} | "
                       f"{r['reference']:.4f} | {100 * r['error']:.2f} % | {r['source']} |")
        if "extrapolated" in st:
            out.append(f"| {st['study']} | {st['quantity']} | Richardson (p = {st['observed_order']:.2f}, GCI "
                       f"{100 * st['gci_fine']:.1f} %) | - | {st['extrapolated']:.4f} | {st['rows'][-1]['reference']:.4f} | "
                       f"{100 * st['extrapolated_error']:.2f} % | {st['rows'][-1]['source']} |")
        elif "richardson_note" in st:
            out.append(f"| {st['study']} | {st['quantity']} | Richardson: not applied | | | | | {st['richardson_note']} |")
    return "\n".join(out) + "\n"
