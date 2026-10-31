"""Surrogates of the axial-flux PM machine in 3-D, with calibrated uncertainty and physics checks.

The simulator is the check in full 3-D of :func:`.afpm_design.verify_3d` (with the quasi-3D estimate on the same
grid), over four design variables of :class:`.afpm.AFPMMachine` (:data:`VARIABLES`); everything else keeps its
default. Outputs per design: the cogging peak-to-peak and the tooth flux linkage in 3-D and in quasi-3D, and their
ratios (what quasi-3D overstates for that design).

The campaign, the dataset with a held-out region and the comparison of surrogate families come from
:mod:`pinneapple_autolab.study` (space-filling design, resumable runs, cross-conformal 90 % intervals, a written
selection rule). :func:`physics_checks` are what a surrogate must respect to be eligible, evaluated on a Sobol
sample of the training region:

* every predicted cogging and flux linkage is positive;
* the flux linkage rises with the magnet thickness and falls with the gap (more magnet, more flux; a wider gap,
  less), on at least 90 % of the sample when the variable moves by 5 % of its range;
* the predicted quasi-3D / 3-D ratios stay within the band the campaign measured, widened by 50 %.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

__all__ = ["VARIABLES", "OUTPUTS", "simulator", "space", "run", "physics_checks", "certify"]

# name: (low, high, unit, description)
VARIABLES: dict[str, tuple[float, float, str, str]] = {
    "magnet_arc": (0.6, 0.9, "", "magnet arc as a share of the pole pitch"),
    "slot_fraction": (0.3, 0.7, "", "slot opening as a share of the slot pitch"),
    "gap": (0.0007, 0.0015, "m", "air gap"),
    "magnet_thickness": (0.003, 0.006, "m", "magnet thickness"),
}
OUTPUTS = ("cogging_pkpk_3d", "linkage_3d", "cogging_pkpk_q3d", "linkage_q3d", "ratio_cogging", "ratio_linkage")


def run(params: Mapping[str, float]) -> dict[str, float]:
    """One design: the quasi-3D estimate and the check in 3-D on the coarse grid of the designer
    (:func:`.afpm_design.grid_for`, whose magnet and tooth edges are shares of the cells they cut, so the outputs
    move continuously with ``magnet_arc`` and ``slot_fraction``), about 35 s."""
    from .afpm_design import estimate, verify_3d

    p = {k: float(v) for k, v in params.items()}
    e = estimate(p)
    v = verify_3d(p, against=e)
    return {"cogging_pkpk_3d": v["cogging_pkpk_Nm"], "linkage_3d": v["tooth_linkage_Wb"],
            "cogging_pkpk_q3d": e["cogging_pkpk_Nm"], "linkage_q3d": e["tooth_linkage_Wb"],
            "ratio_cogging": v["estimate_over_3d"]["cogging_pkpk"],
            "ratio_linkage": v["estimate_over_3d"]["tooth_linkage"]}


def simulator():
    """A :class:`pinneapple_autolab.simulator.FunctionSimulator` of :func:`run` that refuses designs outside the
    variables' bounds."""
    from pinneapple_autolab.simulator import FunctionSimulator

    def checker(params):
        return [f"{k} = {params[k]:g} outside [{lo:g}, {hi:g}]" for k, (lo, hi, *_) in VARIABLES.items()
                if k in params and not lo - 1e-12 <= params[k] <= hi + 1e-12]
    return FunctionSimulator(run, OUTPUTS, name="AFPM, 3-D check on the coarse grid", checker=checker)


def space():
    """The design space as a :class:`pinneapple_data.contracts.ParameterSpace`."""
    from pinneapple_data.contracts import Parameter, ParameterSpace

    return ParameterSpace(name="AFPM design", parameters=[Parameter(name=k, low=lo, high=hi, unit=u, description=d)
                                                          for k, (lo, hi, u, d) in VARIABLES.items()])


def physics_checks(measured_ratios: Mapping[str, tuple[float, float]] | None = None, n: int = 256,
                   seed: int = 0) -> dict[str, list[Callable[[Callable], Any]]]:
    """``{target: [check(predict) -> Check]}`` for :func:`pinneapple_autolab.study.benchmark.benchmark`; inputs in the
    unit cube of the variables (their order in :data:`VARIABLES`). ``measured_ratios``: ``{target: (min, max)}`` of
    the ratios in the campaign, for the band check."""
    from scipy.stats import qmc

    from pinneapple_analysis.trust import Check

    U = qmc.Sobol(len(VARIABLES), seed=seed).random(n)
    names = list(VARIABLES)
    src = "surrogate predictions on a Sobol sample of the training region"

    def _c(name, ok, value, limit, reason):
        return Check(name, "supports" if ok else "contradicts", value=value, limit=limit, source=src, reason=reason)

    def positive(target):
        def chk(predict):
            m, _ = predict(U)
            return _c(f"{target}_positive", bool(np.all(m > 0)), float(np.min(m)), 0.0, f"min predicted {target}")
        return chk

    def moves(target, var, sign):
        j = names.index(var)

        def chk(predict):
            m, _ = predict(U)
            U2 = U.copy()
            U2[:, j] = np.minimum(U2[:, j] + 0.05, 1.0)
            m2, _ = predict(U2)
            moved = U2[:, j] > U[:, j]
            frac = float(np.mean((np.sign(m2 - m) == sign)[moved]))
            word = "rises" if sign > 0 else "falls"
            return _c(f"{target}_{word}_with_{var}", frac >= 0.9, frac, 0.9,
                      f"share of the sample where {target} {word} when {var} rises by 5 % of its range")
        return chk

    def band(target, lo, hi):
        w = 0.5 * (hi - lo)

        def chk(predict):
            m, _ = predict(U)
            ok = bool(np.all((m >= lo - w) & (m <= hi + w)))
            return _c(f"{target}_in_measured_band", ok, float(np.max(np.abs(m - 0.5 * (lo + hi)))),
                      0.5 * (hi - lo) + w, f"predicted {target} within the campaign's range [{lo:.4f}, {hi:.4f}] "
                                           "widened by 50 %")
        return chk

    out: dict[str, list] = {
        "cogging_pkpk_3d": [positive("cogging_pkpk_3d")],
        "cogging_pkpk_q3d": [positive("cogging_pkpk_q3d")],
        "linkage_3d": [positive("linkage_3d"), moves("linkage_3d", "magnet_thickness", +1),
                       moves("linkage_3d", "gap", -1)],
        "linkage_q3d": [positive("linkage_q3d"), moves("linkage_q3d", "magnet_thickness", +1),
                        moves("linkage_q3d", "gap", -1)],
    }
    for t, (lo, hi) in (measured_ratios or {}).items():
        out.setdefault(t, []).append(band(t, lo, hi))
    return out


def certify(board, ds, target: str, family: str | None = None, r2_min: float = 0.9,
            ledger=None, tool: str = "pinneapple_simulation.electromagnetics.machines3d.afpm_surrogate") -> dict[str, Any]:
    """The checks of one surrogate as a trust decision and as entries of an evidence ledger.

    Checks: each physics check of the benchmark (critical); on the held-out test designs, ``R^2 >= r2_min``
    (critical) and the coverage of the conformal 90 % interval at least ``0.9 - 2 sqrt(0.09 / n)`` (two binomial
    standard deviations below nominal for ``n`` test designs); the same two on the extrapolation designs, not
    critical (the region the surrogate was not fitted on). The decision is
    :class:`pinneapple_analysis.trust.TrustReport`'s (APPROVED / REVIEW / REJECT); every check is recorded in
    ``ledger`` (a :class:`pinneapple_veriphysics.ledger.EvidenceLedger`, new if None) under ``f"{target}/..."``
    with the SHA-256 of the dataset as its inputs, so a report can say "passed" only with the record of the run."""
    import math

    from pinneapple_analysis.trust import Check, TrustReport
    from pinneapple_veriphysics.ledger import EvidenceLedger, record_checks, sha256_json, utc_now

    family = family or board.select()["choice"][target]
    key = f"{target}/{family}"
    row = board.rows[key]
    started = utc_now()
    checks = [Check(c["name"], "supports" if c["passed"] else "contradicts", critical=True, value=c["value"],
                    limit=c["limit"], source="benchmark physics check", reason=c["reason"])
              for c in board.physics.get(key, [])]
    for split, critical in (("test", True), ("extrapolation", False)):
        m = row.get(split)
        if not m or not m.get("n"):
            checks.append(Check(f"{split}_r2", "not_run", reason=f"no {split} designs"))
            continue
        n = m["n"]
        cov_min = 0.9 - 2 * math.sqrt(0.09 / n)
        checks.append(Check(f"{split}_r2", "supports" if m["r2"] >= r2_min else "contradicts", critical=critical,
                            value=m["r2"], limit=r2_min, source=f"{n} {split} designs solved in 3-D",
                            reason=f"R^2 of the surrogate on the {split} designs"))
        checks.append(Check(f"{split}_coverage90", "supports" if m["coverage90_conformal"] >= cov_min
                            else "contradicts", value=m["coverage90_conformal"], limit=cov_min,
                            source=f"{n} {split} designs solved in 3-D",
                            reason="share of the designs inside the conformal 90 % interval (binomial 2-sigma floor)"))
    report = TrustReport(checks, model=key)
    led = ledger if ledger is not None else EvidenceLedger()
    inputs = sha256_json({"ids": ds.ids, "X": ds.X.tolist(), "Y": ds.Y.tolist(), "split": ds.split})
    record_checks(led, [{"name": c.name, "passed": None if c.status == "not_run" else c.status == "supports",
                         "value": c.value, "threshold": c.limit, "critical": c.critical, "detail": c.reason}
                        for c in checks], prefix=target, inputs_sha256=inputs, started_at=started,
                  finished_at=utc_now(), tool=tool)
    return {"target": target, "family": family, "decision": report.decisao, "score": report.score,
            "reasons": report.motivos,
            "checks": [{"name": c.name, "status": c.status, "critical": c.critical, "value": c.value,
                        "limit": c.limit, "reason": c.reason} for c in checks],
            "ledger_status": {c.name: led.status(f"{target}/{c.name}") for c in checks}, "ledger": led}
