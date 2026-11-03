"""The standard multiphysics-coupling workflow (#1209): one coupled case, every strategy, and the checks.

1. **Coupling strength** first: the contraction factor of the plain partitioned map (:func:`.coupling.coupling_strength`).
2. **Strategies**, all run on the same discrete case: one-way, two-way partitioned Gauss-Seidel (plain, and with
   Aitken relaxation), and monolithic (both residuals together). Each is timed.
3. **Recommendation** (:func:`.coupling.recommend_strategy`) from the measured contraction and the measured error
   of the one-way result against the two-way one, so the user sees when one-way is enough.
4. **Checks**: convergence of the coupling iteration (with the interface residual history); partitioned against
   monolithic; the interface and global energy / force balances of the case; the coupled closed-form solution.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from .cases import CASES
from .coupling import coupling_strength, gauss_seidel, monolithic, one_way, recommend_strategy


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    out["seconds"] = time.perf_counter() - t0
    return out


def _qoi_diff(q: Mapping[str, float], ref: Mapping[str, float]) -> dict[str, float]:
    return {k: abs(q[k] - ref[k]) / max(abs(ref[k]), 1e-300) for k in ref}


def multiphysics_workflow(*, case: str = "joule_wire", case_params: Mapping[str, Any] | None = None,
                          one_way_tolerance: float = 0.01, coupling_tolerance: float = 1e-10,
                          balance_tolerance: float = 1e-3, closed_form_tolerance: float = 1e-3,
                          max_iter: int = 200) -> dict[str, Any]:
    """Run the protocol on ``case`` (``joule_wire``, ``bimetal_strip``, ``conjugate_channel``) with
    ``case_params`` passed to its builder. Returns the strategies, the recommendation, the checks and the
    measured numbers. ``one_way_tolerance``: the largest relative error on the quantities of interest at which
    one-way coupling is accepted."""
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}: use one of {sorted(CASES)}")
    c = CASES[case](**dict(case_params or {}))
    strength = coupling_strength(c.a, c.b, c.y_ref)
    ow = _timed(lambda: one_way(c.a, c.b, c.y_ref))
    gs_plain = _timed(lambda: gauss_seidel(c.a, c.b, c.y_ref, aitken=False, omega=1.0, tol=coupling_tolerance,
                                           max_iter=max_iter))
    gs = _timed(lambda: gauss_seidel(c.a, c.b, c.y_ref, aitken=True, tol=coupling_tolerance, max_iter=max_iter))
    mono = _timed(lambda: monolithic(c.a, c.b, ow["state_a"], ow["state_b"]))
    two_way = gs if gs["converged"] else mono
    q_two = c.qoi(two_way["state_a"], two_way["state_b"])
    q_one = c.qoi(ow["state_a"], ow["state_b"])
    q_mono = c.qoi(mono["state_a"], mono["state_b"])
    ow_err = _qoi_diff(q_one, q_two)
    ow_max = float(max(ow_err.values()))
    strategy, why = recommend_strategy(strength["contraction"], ow_max, one_way_tolerance)

    checks: list[dict[str, Any]] = []
    checks.append({"code": "coupling.convergence", "status": "pass" if gs["converged"] else "fail",
                   "detail": f"Gauss-Seidel + Aitken: {gs['iterations']} iterations to an interface residual "
                             f"{gs['residual_history'][-1]:.2g} (<= {coupling_tolerance:g}); plain Gauss-Seidel: "
                             f"{gs_plain['iterations']} iterations"
                             + ("" if gs_plain["converged"] else " without converging")})
    checks.append({"code": "coupling.monolithic", "status": "pass" if mono["converged"] else "fail",
                   "detail": f"monolithic residual {mono['residual_norm']:.2g} (from {mono['initial_residual_norm']:.2g}), "
                             f"{mono['iterations']} residual evaluations"})
    pm = _qoi_diff(q_mono, q_two) if gs["converged"] else {}
    if pm:
        dm = float(max(pm.values()))
        checks.append({"code": "coupling.partitioned_vs_monolithic", "status": "pass" if dm <= 1e-6 else "fail",
                       "detail": f"largest relative difference of the quantities of interest {dm:.2g} (<= 1e-6)"})
    for bal in c.balances(two_way["state_a"], two_way["state_b"]):
        checks.append({"code": bal["code"], "status": "pass" if bal["value"] <= balance_tolerance else "fail",
                       "detail": f"{bal['detail']}; relative imbalance {bal['value']:.2g} (<= {balance_tolerance:g})"})
    ex = c.exact(two_way["state_a"], two_way["state_b"])
    ex_one = c.exact(ow["state_a"], ow["state_b"])
    if "skipped" in ex:
        checks.append({"code": "closed_form", "status": "skip", "detail": ex["skipped"]})
    else:
        qe = _qoi_diff(q_two, ex["qoi"])
        worst = float(max(max(qe.values()), ex["field_error"]))
        checks.append({"code": "closed_form", "status": "pass" if worst <= closed_form_tolerance else "fail",
                       "detail": f"two-way against {ex['formula']}: field error {ex['field_error']:.2g}, "
                                 + ", ".join(f"{k} {v:.2g}" for k, v in qe.items())
                                 + f" (<= {closed_form_tolerance:g})"})
    checks.append({"code": "one_way_vs_two_way", "status": "info",
                   "detail": "one-way error against two-way: " + ", ".join(f"{k} {v:.3g}" for k, v in ow_err.items())
                             + (f"; one-way against the closed form: field error {ex_one['field_error']:.3g}"
                                if "skipped" not in ex_one else "")})
    checks.append({"code": "recommendation", "status": "info", "detail": f"{strategy}: {why}"})
    failed = [x["code"] for x in checks if x["status"] == "fail"]

    def strat(r):
        return {"converged": bool(r["converged"]), "iterations": int(r["iterations"]), "seconds": float(r["seconds"]),
                "qoi": c.qoi(r["state_a"], r["state_b"]), "residual_history": [float(v) for v in r["residual_history"]],
                **({"relaxation_history": [float(w) for w in r["relaxation_history"]]}
                   if "relaxation_history" in r else {})}
    out = {"case": c.name, "physics": c.physics, "participants": [
               {"name": p.name, "reads": p.field_in, "writes": p.field_out} for p in (c.a, c.b)],
           "params": c.params, "feedback": c.feedback_note, "coupling_strength": strength,
           "strategies": {"one_way": strat(ow), "gauss_seidel": strat(gs_plain), "gauss_seidel_aitken": strat(gs),
                          "monolithic": strat(mono)},
           "one_way_error": ow_err, "recommended": strategy, "why": why,
           "closed_form": ex if "skipped" in ex else {k: v for k, v in ex.items()},
           "checks": checks, "verdict": "fail" if failed else "pass", "failed": failed}
    out["summary"] = (f"{c.physics} ({c.name}): rho = {strength['contraction']:.3g}, one-way error {ow_max:.3g} -> "
                      f"{strategy}; GS+Aitken {gs['iterations']} it (plain {gs_plain['iterations']}); "
                      + ("closed form " + (f"{ex['field_error']:.2g}" if "skipped" not in ex else "skipped"))
                      + ("; fail: " + ", ".join(failed) if failed else "; every check passes"))
    return out


def residual_table(result: Mapping[str, Any]) -> list[list[Any]]:
    """Interface residual per iteration for the partitioned strategies (rows: iteration, plain, Aitken)."""
    a = result["strategies"]["gauss_seidel"]["residual_history"]
    b = result["strategies"]["gauss_seidel_aitken"]["residual_history"]
    n = max(len(a), len(b))
    return [[i + 1, a[i] if i < len(a) else None, b[i] if i < len(b) else None] for i in range(n)]


__all__ = ["multiphysics_workflow", "residual_table"]
