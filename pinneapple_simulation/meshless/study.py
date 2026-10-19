"""Refinement study of meshless and mesh-based methods on one problem and one domain.

For every node spacing ``h`` and every method: the error at independent evaluation points (each method's own
interpolant) and at its nodes, against the manufactured solution or, without one, against a fine finite-element
reference (whose own error is estimated and reported); the setup and solve time; the 1-norm condition number of the
system matrix and the noise amplification. Then the observed order of convergence, the growth of the condition
number, a shape-parameter sweep for Kansa, and where each method wins (no family is assumed to be better: each
criterion names its own winner, with the numbers).
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .domains import Domain2D
from .methods import METHODS, discretize, solve
from .problems import Problem

__all__ = ["MeshlessStudy", "meshless_study"]


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else None


@dataclass
class MeshlessStudy:
    problem: dict[str, Any]
    domain: dict[str, Any]
    rows: list[dict[str, Any]]
    convergence: dict[str, dict[str, Any]]
    eps_sweep: list[dict[str, Any]]
    wins: list[dict[str, Any]]
    notes: dict[str, list[str]]
    reference: dict[str, Any]
    fields: dict[str, Any] = field(default_factory=dict)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    tolerance: float = 1e-3
    wall_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("problem", "domain", "rows", "convergence", "eps_sweep", "wins", "notes",
                                               "reference", "fields", "skipped", "tolerance", "wall_s")} | \
            {"methods": {k: v for k, v in METHODS.items()}}


def _run(problem, dom, nodes, method, opts, truth, E, seed):
    disc = discretize(method, dom, nodes, **opts)
    out = solve(problem, disc, seed=seed)
    with np.errstate(all="ignore"):
        v = out["evaluate"](E)
        un = out["u"]
        tn = truth(disc.nodes.X)
    finite = bool(np.isfinite(v).all() and np.isfinite(un).all())
    err = _rel(v, truth(E)) if finite else float("inf")
    row = {"method": method, "label": METHODS[method]["label"], "family": METHODS[method]["family"],
           "h": float(nodes.h), "n_nodes": int(disc.nodes.n), "n_unknowns": int(disc.n_unknowns),
           "error": err, "error_max": float(np.max(np.abs(v - truth(E))) / max(np.max(np.abs(truth(E))), 1e-300))
           if finite else float("inf"),
           "error_nodes": _rel(un, tn) if finite else float("inf"),
           "setup_s": float(disc.setup_s), "solve_s": float(out["solve_s"]),
           "total_s": float(disc.setup_s + out["solve_s"]), "cond": out.get("cond"),
           "noise_amplification": out.get("noise_amplification"), "stable": finite and err < 1e3,
           "nnz": disc.info.get("nnz"), "info": {k: v for k, v in disc.info.items() if k != "nnz"}}
    if "train" in out:
        row["train"] = out["train"]
    return row, (v if finite else None), disc


def meshless_study(problem: Problem, dom: Domain2D, h: Sequence[float] | None = None, *,
                   methods: Sequence[str] = ("kansa", "rbf_fd", "fd", "fem", "pinn"),
                   options: dict[str, dict[str, Any]] | None = None, pinn_levels: str = "finest",
                   n_eval: int = 2000, tolerance: float = 1e-3, eps_sweep: bool = True, seed: int = 0,
                   progress: Callable[[str], None] | None = None) -> MeshlessStudy:
    """Run every method at every spacing in ``h`` (default: three halvings from diameter / 10). ``options``: per
    method, e.g. ``{"kansa": {"eps": 2.0}, "rbf_fd": {"degree": 3}, "pinn": {"budget_s": 5}}``. ``pinn_levels``:
    ``finest`` (default, the PINN trains once) or ``all``."""
    t_start = time.perf_counter()
    options = {m: dict(v) for m, v in (options or {}).items()}
    skipped: list[dict[str, Any]] = []
    if dom.cloud is not None:
        hs = [dom.cloud.h]
        if "fd" in methods:
            skipped.append({"method": "fd", "reason": "embedded finite differences need the geometry, not a point cloud"})
        methods = [m for m in methods if m != "fd"]
    else:
        hs = sorted({float(x) for x in (h or [dom.diameter / 10 / 2 ** k for k in range(3)])}, reverse=True)
    E = dom.sample_eval(n_eval)
    # the truth: the manufactured solution, or a fine FEM reference with its own error estimated
    reference: dict[str, Any] = {"kind": "manufactured solution", "expression": problem.exact}
    if problem.has_exact:
        truth = lambda P: problem.u(P)  # noqa: E731
    else:
        if not dom.has_geometry:
            raise ValueError("without an exact solution the reference is a fine FEM solution, which needs the geometry")
        h_ref = min(hs) / 3
        if progress:
            progress(f"fine FEM reference (h = {h_ref:.4g})")
        ref = solve(problem, discretize("fem", dom, dom.nodes(h_ref)))
        ref2 = solve(problem, discretize("fem", dom, dom.nodes(2 * h_ref)))
        rv = ref["evaluate"](E)
        est = _rel(ref2["evaluate"](E), rv) / 3.0                       # P1 is second order: e(h) ~ e(2h) / 4
        cache = {"E": rv}

        def truth(P, _r=ref):
            return cache["E"] if P is E else _r["evaluate"](P)
        reference = {"kind": "fine FEM reference", "h": h_ref, "n_nodes": int(len(ref["u"])),
                     "estimated_error": est,
                     "note": "errors below the reference's own estimated error are not resolved"}
    rows, fields_out, _discs = [], {}, {}
    for lev, hh in enumerate(hs):
        nodes = dom.nodes(hh)
        for m in methods:
            if m == "pinn" and pinn_levels == "finest" and lev != len(hs) - 1:
                continue
            if progress:
                progress(f"{METHODS.get(m, {}).get('label', m)}, h = {hh:.4g}")
            try:
                row, v, disc = _run(problem, dom, nodes, m, options.get(m, {}), truth, E, seed)
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as ex:
                msg = str(ex)
                if "NaN" in msg or "inf" in msg.lower() or isinstance(ex, np.linalg.LinAlgError):
                    rows.append({"method": m, "label": METHODS[m]["label"], "family": METHODS[m]["family"],
                                 "h": hh, "n_nodes": nodes.n, "error": float("inf"), "stable": False,
                                 "failure": msg[:160]})
                else:
                    skipped.append({"method": m, "h": hh, "reason": msg[:200]})
                continue
            rows.append(row)
            if v is not None:                                           # keeps the finest stable level
                fields_out[m] = {"h": hh, "values": v, "nodes": disc.nodes.X, "boundary": disc.nodes.boundary}
    # convergence per method
    conv: dict[str, dict[str, Any]] = {}
    for m in methods:
        rr = [r for r in rows if r["method"] == m and r.get("stable") and np.isfinite(r["error"]) and r["error"] > 0]
        if not rr:
            continue
        hh = [r["h"] for r in rr]
        c = {"levels": len(rr), "order": _slope(hh, [r["error"] for r in rr]) if len(rr) >= 2 else None,
             "order_last": (float(np.log(rr[-2]["error"] / rr[-1]["error"]) / np.log(rr[-2]["h"] / rr[-1]["h"]))
                            if len(rr) >= 2 else None),
             "cond_growth": (-_slope(hh, [r["cond"] for r in rr]) if len(rr) >= 2 and all(r.get("cond") for r in rr)
                             else None),
             "finest_error": rr[-1]["error"], "finest_h": rr[-1]["h"]}
        conv[m] = c
    # Kansa shape parameter: accuracy against conditioning (the trade-off principle)
    sweep = []
    if eps_sweep and "kansa" in methods and not problem.time_dependent:
        lev = min(1, len(hs) - 1)
        nodes = dom.nodes(hs[lev])
        if nodes.n <= 2500:
            from .methods import default_eps
            e0 = float(options.get("kansa", {}).get("eps") or default_eps(dom))
            for f in (0.25, 0.5, 1.0, 2.0, 4.0):
                try:
                    r, _, _ = _run(problem, dom, nodes, "kansa", {"eps": e0 * f}, truth, E, seed)
                    sweep.append({"eps": e0 * f, "h": hs[lev], "error": r["error"], "cond": r["cond"]})
                except (ValueError, np.linalg.LinAlgError):
                    sweep.append({"eps": e0 * f, "h": hs[lev], "error": float("inf"), "cond": None})
    wins = _wins(rows, conv, tolerance, reference)
    notes = _notes(rows, conv, sweep, problem)
    # fields for the maps: the finest level of each method, at the evaluation points
    ev = truth(E)
    fl: dict[str, Any] = {"points": E.tolist(), "truth": ev.tolist(), "methods": {}}
    for m, f in fields_out.items():
        fl["methods"][m] = {"h": f["h"], "values": f["values"].tolist(), "error": (f["values"] - ev).tolist(),
                            "nodes": f["nodes"].tolist(), "boundary": f["boundary"].tolist()}
    return MeshlessStudy(problem.describe(), dom.describe(), rows, conv, sweep, wins, notes, reference, fl, skipped,
                         float(tolerance), time.perf_counter() - t_start)


def _wins(rows, conv, tol, reference) -> list[dict[str, Any]]:
    ok = [r for r in rows if r.get("stable")]
    out = []
    if not ok:
        return out
    floor = reference.get("estimated_error", 0.0) or 0.0
    hmin = min(r["h"] for r in ok)
    fin = [r for r in ok if r["h"] == hmin or (r["method"] == "pinn")]
    best = sorted(fin, key=lambda r: r["error"])
    if best:
        b = best[0]
        out.append({"criterion": "Accuracy at the finest spacing", "method": b["method"], "value": b["error"],
                    "unit": "relative L2", "runner_up": best[1]["method"] if len(best) > 1 else None,
                    "runner_up_value": best[1]["error"] if len(best) > 1 else None,
                    "resolved": b["error"] > floor})
    under = [r for r in ok if r["error"] <= tol]
    if under:
        b = min(under, key=lambda r: r["total_s"])
        per = {}
        for r in under:
            per[r["method"]] = min(per.get(r["method"], np.inf), r["total_s"])
        ranked = sorted(per.items(), key=lambda kv: kv[1])
        out.append({"criterion": f"Cheapest to reach {tol:g} relative error", "method": b["method"],
                    "value": b["total_s"], "unit": "s", "detail": f"at h = {b['h']:.4g} ({b['n_nodes']} nodes)",
                    "runner_up": ranked[1][0] if len(ranked) > 1 else None,
                    "runner_up_value": ranked[1][1] if len(ranked) > 1 else None})
    else:
        out.append({"criterion": f"Cheapest to reach {tol:g} relative error", "method": None, "value": None,
                    "unit": "s", "detail": "no method reached the tolerance on these spacings"})
    oc = sorted(((m, c["order"]) for m, c in conv.items() if c.get("order") is not None), key=lambda kv: -kv[1])
    if oc:
        out.append({"criterion": "Fastest convergence under refinement", "method": oc[0][0], "value": oc[0][1],
                    "unit": "observed order", "runner_up": oc[1][0] if len(oc) > 1 else None,
                    "runner_up_value": oc[1][1] if len(oc) > 1 else None})
    cc = sorted((r for r in ok if r["h"] == hmin and r.get("cond")), key=lambda r: r["cond"])
    if cc:
        out.append({"criterion": "Best conditioned system at the finest spacing", "method": cc[0]["method"],
                    "value": cc[0]["cond"], "unit": "1-norm condition number",
                    "runner_up": cc[1]["method"] if len(cc) > 1 else None,
                    "runner_up_value": cc[1]["cond"] if len(cc) > 1 else None})
    na = sorted((r for r in ok if r["h"] == hmin and r.get("noise_amplification") is not None),
                key=lambda r: r["noise_amplification"])
    if na:
        out.append({"criterion": "Least sensitive to perturbed data", "method": na[0]["method"],
                    "value": na[0]["noise_amplification"], "unit": "relative change of u per relative change of data",
                    "runner_up": na[1]["method"] if len(na) > 1 else None,
                    "runner_up_value": na[1]["noise_amplification"] if len(na) > 1 else None})
    return out


def _notes(rows, conv, sweep, problem) -> dict[str, list[str]]:
    n: dict[str, list[str]] = {}
    for r in rows:
        if not r.get("stable"):
            n.setdefault(r["method"], []).append(
                f"h = {r['h']:.4g}: unstable or failed ({r.get('failure', 'non-finite or exploding solution')})" +
                (" -- time-stepping the Kansa nodal operator is a known instability" if r["method"] == "kansa"
                 and problem.time_dependent else ""))
        elif r.get("cond") and r["cond"] > 1e14:
            n.setdefault(r["method"], []).append(
                f"h = {r['h']:.4g}: condition number {r['cond']:.1e}, near the limit of double precision; "
                f"refining further stops paying")
    for m, c in conv.items():
        if c.get("order") is not None:
            n.setdefault(m, []).append(f"observed order {c['order']:.2f} over {c['levels']} spacings")
    if sweep:
        fin = [s for s in sweep if np.isfinite(s["error"])]
        if fin:
            b = min(fin, key=lambda s: s["error"])
            n.setdefault("kansa", []).append(f"best shape parameter of the sweep: eps = {b['eps']:.3g} "
                                             f"(error {b['error']:.2e}, condition {b['cond'] or 0:.1e})")
    for r in rows:
        if r["method"] == "pinn" and "train" in r:
            t = r["train"]
            n.setdefault("pinn", []).append(f"trained {t['steps']} steps in {t['used_s']:.1f} s (budget "
                                            f"{t['budget_s']:g} s, stopped by {t['stopped_by']})")
    return n
