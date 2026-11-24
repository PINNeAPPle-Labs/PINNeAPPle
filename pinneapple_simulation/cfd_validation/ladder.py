"""The low-fidelity ladder: what each rung costs, and where it stops ranking designs correctly (#294).

A design search cannot afford the fine solver for every sample, so it uses the cheapest thing that *ranks* the designs
the way the fine solver would. That is a claim about a specific family of designs and a specific quantity, and it is
either measured or it is folklore. This module measures it: one external-flow problem, several rungs of fidelity, and
a table of wall time, cells, error in ``CL`` and ``CD`` against the finest rung, and the Spearman rank correlation of
the designs — which is the column that decides whether a rung is usable for optimisation, since an error that is the
same for every design changes nothing about which one wins.

The family here
---------------
A NACA 0012 section at several angles of attack, which every rung in the repository can attempt: the thin-airfoil
rung is the closed-form answer (``CL = 2 pi sin(alpha - alpha0)``, no cells, microseconds), and the RANS rungs are
:func:`pinneapple_design.aero.case.run_case` at three :class:`~pinneapple_design.aero.mesh.GridSpec` resolutions, so
the "levels" are the same case at fewer cells rather than a different case.

Which rungs are *not* in the measured table, and why
---------------------------------------------------
The vortex lattice (:mod:`pinneapple_design.aero.vlm`) is a lifting-surface method for a finite wing: its ``CL`` is a
3-D wing's, and comparing it with a 2-D section's would put the aspect ratio's effect in the error column. The LBM
solver (:mod:`pinneapple_simulation.numerical_solvers.lbm`) is a low-Reynolds method: at the Re = 4e6 this family is
solved at, its collision model and resolution are not the same physical regime, so a number would say more about the
comparison than about the solver. Both are rungs of the ladder in the sense the issue means, and neither is on *this*
family; the table says so instead of adding a column that would mislead.

Calibration
-----------
:func:`calibrate` takes a few designs measured at the fine level and fits the offset (and the slope, if there are
enough) of a coarse rung against them, then reports the ranking with and without the correction: a bias that is the
same for every design cannot change a ranking, and a bias that varies with the design can.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import foam

__all__ = ["Rung", "design", "thin_airfoil", "rans_airfoil", "ladder", "spearman", "calibrate", "LEVELS",
           "MULTIFIDELITY"]

#: The RANS levels, cheapest first: the resolution of the O-grid, in the same case.
LEVELS = (
    {"name": "rans-coarse", "n_surf": 40, "nj": 32, "radius": 50.0, "first": 1e-5},
    {"name": "rans-medium", "n_surf": 56, "nj": 45, "radius": 80.0, "first": 6e-6},
    {"name": "rans-fine", "n_surf": 80, "nj": 64, "radius": 100.0, "first": 4e-6},
)

#: What the multi-fidelity workflow (#35) needs to treat the ladder as data levels: the cost of each rung and the
#: quantity it produces, so a sampler can decide which rung to launch for a given design.
MULTIFIDELITY = {"quantity": "CL", "levels": [{"name": "thin-airfoil", "cost": 0},
                                              {"name": "rans-coarse", "cost": 20},
                                              {"name": "rans-medium", "cost": 38},
                                              {"name": "rans-fine", "cost": 80}],
                 "note": "wall seconds per design, rounded, from the measured table in examples/cfd/classical_cases"}


@dataclass(frozen=True)
class Rung:
    """One fidelity level: what it is, what it costs to ask, and what it produces."""

    name: str
    kind: str                    # "analytic" | "rans" | "vlm" | "lbm"
    produces: tuple[str, ...]    # ("CL",) or ("CL", "CD")
    levels: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "produces": list(self.produces), "levels": dict(self.levels)}


@dataclass(frozen=True)
class design:
    """One point of the family: a section and an angle of attack."""

    section: str = "0012"
    alpha: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"section": self.section, "alpha": self.alpha}


def thin_airfoil(name: str = "0012", alpha: float = 0.0) -> dict[str, float]:
    """The closed-form rung: ``CL = 2 pi sin(alpha - alpha0)`` and no ``CD`` (inviscid theory has none).

    ``alpha0`` is the zero-lift angle, which for the symmetric 4-digit sections of this family is zero. The lift slope
    is the thin-airfoil one, which is exact only in the limit of a thin section at a small angle: the rung exists to be
    the cheapest possible one in a cost-accuracy table, and its error is the price of that.
    """
    del name
    a = np.radians(float(alpha))
    return {"CL": float(2.0 * np.pi * np.sin(a)), "cells": 0, "seconds": 0.0}


def rans_airfoil(name: str, alpha: float, level: dict[str, Any], root: str, *, reynolds: float = 4e6,
                 iterations: int = 3000, timeout: float = 3600.0, reuse: bool = False) -> dict[str, Any]:
    """One RANS rung: the same case at the resolution of ``level``, returning ``CL``, ``CD`` and what it cost.

    Runs with the OpenFOAM :func:`~pinneapple_simulation.cfd_validation.foam.bashrc` finds, not with the fixed path
    the aero module defaults to, so the ladder works wherever the case builder works.
    """
    from pinneapple_design.aero import case as aero_case
    from pinneapple_design.aero.geometry import naca4
    from pinneapple_design.aero.mesh import GridSpec

    found = foam.bashrc()
    if found is None:
        raise RuntimeError("no OpenFOAM installation found (set FOAM_BASHRC)")
    aero_case.FOAM_BASHRC = found
    shape = naca4(name)
    grid = GridSpec(n_surf=int(level["n_surf"]), n_te=2, nj=int(level["nj"]),
                    radius=float(level["radius"]), first=float(level["first"]))
    case = os.path.join(root, f"{level['name']}_naca{name}_a{alpha:+.0f}")
    os.makedirs(root, exist_ok=True)
    if reuse and os.path.isdir(case) and any(d.replace(".", "").isdigit() and d != "0" for d in os.listdir(case)):
        ni = GridSpec(n_surf=int(level["n_surf"]), n_te=2, nj=int(level["nj"]), radius=float(level["radius"]),
                      first=float(level["first"])).ni
        from pinneapple_design.aero.mesh import grid_points
        got = aero_case.read_coefficients(case, grid_points(shape, GridSpec(
            n_surf=int(level["n_surf"]), n_te=2, nj=int(level["nj"]), radius=float(level["radius"]),
            first=float(level["first"]))), 1.0 / reynolds, float(alpha), ni)
        return {"CL": got.get("Cl"), "CD": got.get("Cd"), "cells": int(ni * int(level["nj"])), "seconds": 0.0,
                "status": "read back from disk", "converged_in": None, "error": got.get("error"), "case": case}
    t0 = time.time()
    got = aero_case.run_case(case, shape, float(alpha), reynolds=reynolds, iterations=iterations, grid=grid,
                             timeout=timeout)
    cells = grid.ni * grid.nj
    return {"CL": got.get("Cl"), "CD": got.get("Cd"), "cells": int(cells), "seconds": round(time.time() - t0, 2),
            "status": got.get("status"), "converged_in": got.get("converged_in"),
            "error": got.get("error"), "case": case}


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman's rank correlation: the correlation of the ranks, which is what a design search needs.

    Ties are given their average rank. Returns ``nan`` when either column has no variation — a set of one design, or a
    rung that predicts the same number for every design, which cannot rank anything (and that is the finding).
    """

    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(1, len(x) + 1, dtype=float)
        for value in np.unique(x):                          # average rank within a tie
            mask = x == value
            if mask.sum() > 1:
                r[mask] = r[mask].mean()
        return r

    x = np.asarray([v for v in a], dtype=float)
    y = np.asarray([v for v in b], dtype=float)
    if x.size != y.size or x.size < 2:
        return float("nan")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom else float("nan")


def ladder(designs: Iterable[design], root: str, *, levels: Sequence[dict[str, Any]] = LEVELS,
           reynolds: float = 4e6, iterations: int = 3000, timeout: float = 3600.0, reuse: bool = False,
           log=print) -> dict[str, Any]:
    """Run every design on every rung and return the cost-accuracy table.

    ``reference`` is the finest RANS level: the error column is against it, not against an experiment, because the
    question the ladder answers is "which rung is a cheaper stand-in for the expensive one", not "is the expensive one
    right" (that is what :mod:`~pinneapple_simulation.cfd_validation.reference` and the TMR cases are for).
    """
    designs = list(designs)
    rows: list[dict[str, Any]] = []
    for d in designs:
        t0 = time.time()
        got = thin_airfoil(d.section, d.alpha)
        rows.append({"design": d.to_dict(), "rung": "thin-airfoil", **got,
                     "seconds": round(time.time() - t0, 4)})
        for level in levels:
            log(f"  {level['name']} NACA {d.section} alpha = {d.alpha:+.1f}")
            try:
                got = rans_airfoil(d.section, d.alpha, level, root, reynolds=reynolds, iterations=iterations,
                                   timeout=timeout, reuse=reuse)
            except (RuntimeError, OSError) as exc:
                got = {"CL": None, "CD": None, "cells": 0, "seconds": 0.0, "status": "failed", "error": str(exc)}
            rows.append({"design": d.to_dict(), "rung": level["name"], **got})
    finest = levels[-1]["name"]
    ref = {(r["design"]["section"], r["design"]["alpha"]): r for r in rows if r["rung"] == finest}
    for r in rows:
        base = ref.get((r["design"]["section"], r["design"]["alpha"]), {})
        for q in ("CL", "CD"):
            a, b = r.get(q), base.get(q)
            r[f"{q}_error"] = None if a is None or b in (None, 0) else float(abs(a - b) / abs(b))
            # the absolute error is the one to read for CL: a section's lift crosses zero between two of these
            # designs, so a relative error there divides by nothing and says nothing
            r[f"{q}_error_abs"] = None if a is None or b is None else float(abs(a - b))
    table = []
    for name in ["thin-airfoil"] + [lv["name"] for lv in levels]:
        mine = [r for r in rows if r["rung"] == name]
        ok = [r for r in mine if r.get("CL") is not None]
        entry = {
            "rung": name,
            "designs": len(mine),
            "converged": sum(1 for r in mine if r.get("status") == "converged"),
            "seconds": round(float(sum(r["seconds"] for r in mine)), 2),
            "seconds_per_design": round(float(sum(r["seconds"] for r in mine) / max(1, len(mine))), 2),
            "cells": int(mine[0]["cells"]) if mine else 0,
            "CL_error_rel": _stat([r.get("CL_error") for r in ok]) if name != finest else 0.0,
            "CL_error_abs": _stat([r.get("CL_error_abs") for r in ok]) if name != finest else 0.0,
            "CD_error_rel": _stat([r.get("CD_error") for r in ok]) if name != finest else 0.0,
            "CD_error_abs": _stat([r.get("CD_error_abs") for r in ok]) if name != finest else 0.0,
        }
        for quantity in ("CL", "CD"):
            # a rung that does not produce the quantity cannot be ranked by it: the closed-form rung has a lift and no
            # drag, and a `nan` in that column is the honest answer rather than a zero that would read as agreement
            pairs = [(r.get(quantity), ref[(r["design"]["section"], r["design"]["alpha"])].get(quantity))
                     for r in ok]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            entry[f"spearman_{quantity}"] = (spearman([a for a, _ in pairs], [b for _, b in pairs])
                                             if len(pairs) == len(ref) else float("nan"))
        table.append(entry)
    return {"family": f"NACA 0012, Re = {reynolds:g}, angles of attack", "reynolds": reynolds,
            "reference_rung": finest, "table": table, "rows": rows,
            "multifidelity": MULTIFIDELITY,
            "not_in_the_table": ["vlm (a finite wing's CL, not a section's)",
                                 f"lbm (a low-Reynolds regime, not Re = {reynolds:g})"]}


def _stat(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def calibrate(rows: Sequence[dict[str, Any]], rung: str, *, quantity: str = "CL",
              designs: Sequence[float] | None = None) -> dict[str, Any]:
    """Fit the offset (and slope) of one rung against the fine level on a few designs, and report the ranking twice.

    A bias that is the same for every design leaves the ranking alone; one that varies does not, and the change in
    Spearman after the correction is exactly the difference between the two. ``designs`` limits the fit to a few
    angles of attack (the expensive runs a portfolio can afford) while the reported ranking uses all of them.
    """
    by_alpha: dict[float, dict[str, Any]] = {}
    for r in rows:
        by_alpha.setdefault(float(r["design"]["alpha"]), {})[r["rung"]] = r
    ref_rung = "rans-fine"
    alphas = sorted(by_alpha)
    fit_alphas = [a for a in alphas if designs is None or a in set(designs)]
    if not fit_alphas:
        raise ValueError(f"no design in {designs!r} among {alphas}")
    coarse = np.asarray([by_alpha[a][rung][quantity] for a in fit_alphas], dtype=float)
    fine = np.asarray([by_alpha[a][ref_rung][quantity] for a in fit_alphas], dtype=float)
    offset = float(np.mean(fine - coarse))                       # the bias, which no ranking can see
    slope = 1.0
    if len(fit_alphas) >= 3 and np.ptp(coarse) > 0:
        slope = float(np.polyfit(coarse, fine, 1)[0])             # the part that can move a ranking
    intercept = 0.0 if slope == 1.0 else float(fine.mean() - slope * coarse.mean())
    before = [by_alpha[a][rung][quantity] for a in alphas]
    after = [slope * by_alpha[a][rung][quantity] + intercept + (0.0 if slope != 1.0 else offset) for a in alphas]
    truth = [by_alpha[a][ref_rung][quantity] for a in alphas]
    return {"rung": rung, "quantity": quantity, "fitted_on": [float(a) for a in fit_alphas],
            "offset": offset, "slope": slope, "intercept": intercept,
            "mean_abs_error_before": _stat([abs(x - y) for x, y in zip(before, truth, strict=False)]),
            "mean_abs_error_after": _stat([abs(x - y) for x, y in zip(after, truth, strict=False)]),
            "spearman_before": spearman(before, truth), "spearman_after": spearman(after, truth)}
