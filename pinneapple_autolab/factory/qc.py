"""Quality control of a factory run: the checks on each run, and on the dataset they add up to.

The vocabulary is the library's (:class:`pinneapple_analysis.trust.Check`): a check **supports**, **contradicts** or
was **not_run**, and the third is a statement, not a pass. What is checked, and what each verdict costs:

===================================  =========  ==================================================================
check                                critical   contradicts when
===================================  =========  ==================================================================
parameters within validity limits    yes        the geometry template refuses the design (a reference limit)
simulation ran                       yes        the solver failed or refused the design; nothing is measured
outputs complete                     yes        the run came back without one of the declared outputs, or with NaN
converged                            no         the recorded residuals are above the limit, or ``converged`` is False
mesh quality                         no         ``checkMesh`` failed or the cell count is zero
mesh convergence (GCI)               no         the grid-convergence index is above the limit
===================================  =========  ==================================================================

A row with a critical contradiction is **rejected**: it is listed with its reasons and never written to the
dataset, and the run it came from is still on record (a dropped run is a lie about how many were spent). A row whose
convergence or mesh check is ``not_run`` is kept and the gap travels with it — the dataset-level report then says
how many rows carry it, and the trust verdict drops to ``REVIEW`` when the coverage is too low to sign anything.

The solver contract this reads is :class:`~pinneapple_autolab.simulator.SimOutcome`'s ``meta``, all optional:

* ``converged`` (bool), or ``final_residuals`` / ``residuals`` as ``{name: value}`` against ``residual_limit``;
* ``mesh`` as ``{"cells": int, "ok": bool}``, or ``mesh_ok`` and ``cells`` at the top of ``meta``;
* ``gci`` (the grid-convergence index of the quantity the decision is made on) against ``gci_limit``.

A solver that records none of them is not penalised for being honest: every one of those checks comes back
``not_run`` with the reason, and the dataset says so instead of assuming the run converged.

The two limits that are project decisions, not physics, are in :data:`LIMITS` and can be passed in: the residual
floor is the one the Ahmed campaign uses, and 10 % of the answer is what a study here accepts as numerical
uncertainty (both overridable per run through ``meta``, which is what a solver that knows its own numbers does).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pinneapple_analysis.trust.trust_report import (
    CONTRADICTS,
    NOT_RUN,
    SUPPORTS,
    Check,
    TrustReport,
)

__all__ = ["LIMITS", "KEPT", "REJECTED", "FactoryQc", "row_verdict"]

#: Default limits of the two checks that are not physics: the residual floor of the Ahmed campaign and the share of
#: the answer a study accepts as mesh-induced uncertainty. ``meta`` overrides both per run.
LIMITS = {"residual": 1e-3, "gci": 0.10}

#: What happened to a row.
KEPT, REJECTED = "kept", "rejected"


def _finite(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def row_verdict(checks: Sequence[Check]) -> tuple[str, list[str]]:
    """``(verdict, reasons)``: a critical contradiction rejects the row, and the reasons are what failed."""
    bad = [c for c in checks if c.status == CONTRADICTS and c.critical]
    reasons = [f"{c.name}: {c.reason}" if c.reason else c.name for c in bad]
    return (REJECTED if bad else KEPT), reasons


class FactoryQc:
    """The checks of one run and of the dataset they add up to. Stateless; the limits are the only configuration."""

    def __init__(self, limits: Mapping[str, float] | None = None):
        self.limits = {**LIMITS, **{str(k): float(v) for k, v in dict(limits or {}).items()}}

    # ------------------------------------------------------------------------------ one run
    def row_checks(self, artifact: Any, outcome: Any, spec: Any) -> list[Check]:
        """The geometry's checks plus the run's: did it run, is every output there, did it converge, is the mesh ok."""
        checks = list(getattr(artifact, "checks", []) or [])
        ok = bool(getattr(outcome, "ok", False))
        meta = dict(getattr(outcome, "meta", None) or {})
        outputs = dict(getattr(outcome, "outputs", None) or {})
        declared = spec.evaluator.output_names

        checks.append(Check("simulation ran", SUPPORTS if ok else CONTRADICTS, critical=True,
                            reason="" if ok else (getattr(outcome, "message", "") or "the solver returned no "
                                                                                   "output and no reason")))
        absent = [k for k in declared if k not in outputs]
        not_finite = [k for k in declared if k in outputs and not _finite(outputs[k])]
        complete = not absent and not not_finite
        checks.append(Check("outputs complete", SUPPORTS if complete else CONTRADICTS, critical=True,
                            value=float(len(declared) - len(absent) - len(not_finite)), limit=float(len(declared)),
                            reason="" if complete else ("missing " + ", ".join(absent) + "; " if absent else "") +
                            ("not a finite number: " + ", ".join(not_finite) if not_finite else "")))

        checks.append(self._converged(meta))
        checks.append(self._mesh(meta))
        checks.append(self._gci(meta))
        return checks

    def _converged(self, meta: Mapping[str, Any]) -> Check:
        limit = float(meta.get("residual_limit", self.limits["residual"]))
        if isinstance(meta.get("converged"), bool):
            return Check("converged", SUPPORTS if meta["converged"] else CONTRADICTS, limit=limit,
                         source="SimOutcome.meta['converged']",
                         reason="" if meta["converged"] else "the solver reports it did not converge")
        res = meta.get("final_residuals") or meta.get("residuals")
        if isinstance(res, Mapping) and res:
            worst = max(float(v) for v in res.values())
            name = max(res, key=lambda k: float(res[k]))
            return Check("converged", SUPPORTS if worst <= limit else CONTRADICTS, value=worst, limit=limit,
                         source="SimOutcome.meta['final_residuals']",
                         reason=f"largest final residual is {name} = {worst:.3g}")
        return Check("converged", NOT_RUN, limit=limit, source="SimOutcome.meta",
                     reason="the solver recorded no residuals and no convergence flag; nothing here says whether "
                            "the run converged")

    def _mesh(self, meta: Mapping[str, Any]) -> Check:
        mesh = meta.get("mesh") if isinstance(meta.get("mesh"), Mapping) else {}
        cells = mesh.get("cells", meta.get("cells"))
        ok = mesh.get("ok", meta.get("mesh_ok"))
        if cells is None and ok is None:
            return Check("mesh quality", NOT_RUN, source="SimOutcome.meta",
                         reason="the solver recorded no mesh: neither its size nor a checkMesh verdict")
        if ok is None:
            return Check("mesh quality", NOT_RUN, value=float(cells), source="SimOutcome.meta",
                         reason=f"{int(cells)} cells recorded but no checkMesh verdict")
        return Check("mesh quality", SUPPORTS if ok else CONTRADICTS, value=float(cells or 0.0),
                     source="SimOutcome.meta", reason="" if ok else "checkMesh reports the mesh is not OK")

    def _gci(self, meta: Mapping[str, Any]) -> Check:
        limit = float(meta.get("gci_limit", self.limits["gci"]))
        gci = meta.get("gci")
        if gci is None:
            return Check("mesh convergence (GCI)", NOT_RUN, limit=limit, source="SimOutcome.meta",
                         reason="no grid-convergence index was recorded for this run; the mesh family a study would "
                                "need to measure one is not part of a single run")
        g = float(gci)
        return Check("mesh convergence (GCI)", SUPPORTS if g <= limit else CONTRADICTS, value=g, limit=limit,
                     source="SimOutcome.meta['gci']",
                     reason=f"GCI {100 * g:.1f} % of the answer against a {100 * limit:.0f} % limit")

    # ------------------------------------------------------------------------------ the dataset
    def dataset_checks(self, rows: Sequence[Any], rejected: Sequence[Any], spec: Any) -> TrustReport:
        """What the rows add up to: are they traceable, complete, distinct, and how much of the mesh evidence exists.

        Called before the writer, so a dataset that would fail is never written.
        """
        from pinneapple_data.provenance import problems
        checks: list[Check] = []
        n = len(rows)

        untraceable = [r.id for r in rows if problems(r.provenance)]
        checks.append(Check("every row has provenance", CONTRADICTS if untraceable else SUPPORTS, critical=True,
                            value=float(n - len(untraceable)), limit=float(n),
                            source="pinneapple_data.provenance.problems",
                            reason=("" if not untraceable else "cannot be traced back: " + ", ".join(untraceable))))

        incomplete = [r.id for r in rows if any(not _finite(v) for v in r.outputs.values()) or not r.outputs]
        checks.append(Check("every row has its outputs", CONTRADICTS if incomplete else SUPPORTS, critical=True,
                            value=float(n - len(incomplete)), limit=float(n), source="the rows of this run",
                            reason="" if not incomplete else "without a finite output: " + ", ".join(incomplete)))

        leaked = [r.id for r in rejected if r.ok]
        checks.append(Check("no rejected row was written", CONTRADICTS if leaked else SUPPORTS, critical=True,
                            value=float(len(leaked)), limit=0.0, source="the rows of this run",
                            reason="" if not leaked else "a rejected row with outputs reached the dataset: " +
                                                          ", ".join(leaked)))

        designs = [tuple(sorted(r.params.items())) for r in rows]
        duplicated = len(designs) - len(set(designs))
        checks.append(Check("the designs are distinct", CONTRADICTS if duplicated else SUPPORTS,
                            value=float(duplicated), limit=0.0, source="the DOE that produced them",
                            reason="" if not duplicated else f"{duplicated} design(s) were evaluated twice"))

        for name, wanted in (("converged", "convergence"), ("mesh quality", "mesh")):
            statuses = [self._status(r, name) for r in rows]
            if any(s == CONTRADICTS for s in statuses):
                checks.append(Check(f"{wanted} evidence on every row", CONTRADICTS, critical=False,
                                    value=float(statuses.count(SUPPORTS)), limit=float(n),
                                    source="SimOutcome.meta",
                                    reason=f"{statuses.count(CONTRADICTS)} of {n} rows contradict it; the rows are "
                                           "kept and the dataset is not signed"))
            else:
                recorded = statuses.count(SUPPORTS)
                checks.append(Check(f"{wanted} evidence on every row", SUPPORTS if n and recorded == n else NOT_RUN,
                                    value=float(recorded), limit=float(n), source="SimOutcome.meta",
                                    reason="" if n and recorded == n else
                                    f"{recorded} of {n} rows record it; the others are kept and flagged"))

        return TrustReport(checks, model=f"factory run of {getattr(spec, 'title', '') or 'the case'}")

    @staticmethod
    def _status(row: Any, name: str) -> str:
        return next((c.status for c in row.checks if c.name == name), NOT_RUN)
