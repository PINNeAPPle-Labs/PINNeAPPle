"""Plan a simulation campaign: which cases to run, in which order, and what they add to what already ran.

    from pinneapple_data.doe import plan_campaign
    plan = plan_campaign(space, n=40, method="maximin", existing=done_runs, existing_id_column="run_id")
    plan.cases                 # case_id, priority, every parameter, nearest existing run and distance
    plan.summary()             # coverage before / after, redundant cases removed, compute saved
    plan.study_result()        # pinneapple_data.contracts.StudyResult (study_type "doe")
    plan.dataset_skeleton(outputs=["Cl", "Cd"])   # DatasetSpec with status "skeleton" for the surrogate step
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contracts import (
    CheckRecord,
    Column,
    DatasetSpec,
    ParameterSpace,
    Provenance,
    StudyResult,
    Table,
)
from .coverage import coverage, prioritize, redundancy
from .designs import Design, design


@dataclass
class CampaignPlan:
    space: ParameterSpace
    design: Design
    cases: Any                                        # DataFrame in priority order
    unit: np.ndarray                                  # unit coordinates of ``cases`` (same order)
    coverage_planned: dict[str, Any]
    coverage_existing: dict[str, Any] | None
    coverage_combined: dict[str, Any] | None
    redundancy: dict[str, Any]
    dropped: Any                                      # planned cases removed as repeats (DataFrame)
    existing_unit: np.ndarray | None = None
    existing_ids: list[str] | None = None
    id: str = field(default_factory=lambda: f"campaign-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}")
    name: str = "campaign"

    def summary(self) -> dict[str, Any]:
        out = {"id": self.id, "name": self.name, "method": self.design.method, "n_cases": int(len(self.cases)),
               "n_generated": int(len(self.design.cases)), "n_dropped_as_repeats": int(len(self.dropped)),
               "design_info": self.design.info, "redundancy": {k: v for k, v in self.redundancy.items()
                                                               if k != "cases"},
               "coverage": {"planned": self.coverage_planned, "existing": self.coverage_existing,
                            "combined": self.coverage_combined}}
        if self.coverage_existing and self.coverage_combined:
            out["fill_distance_reduction"] = (
                1.0 - self.coverage_combined["fill_distance"] / self.coverage_existing["fill_distance"]
                if self.coverage_existing["fill_distance"] else None)
        return out

    def checks(self) -> list[CheckRecord]:
        """Checks that ran on this plan, each with its value."""
        feas = self.space.feasible(self.cases[self.space.names]) if len(self.cases) else np.ones(0, bool)
        out = [CheckRecord(name="constraints satisfied", status="PASS" if feas.all() else "FAIL",
                           value=float(feas.mean()) if len(feas) else 1.0, threshold=1.0,
                           detail="fraction of planned cases inside the bounds and constraints")]
        md = self.coverage_planned.get("min_distance")
        tol = self.redundancy["tol"]
        if md is not None:
            out.append(CheckRecord(name="no repeated case in the plan", status="PASS" if md > tol else "FAIL",
                                   value=md, threshold=tol, detail="smallest distance between planned cases"))
        if self.existing_unit is not None:
            rep = int(self.cases["repeats_existing"].sum()) if "repeats_existing" in self.cases else 0
            out.append(CheckRecord(name="no repeat of an existing run", status="PASS" if rep == 0 else "WARN",
                                   value=rep, threshold=0, detail="planned cases within tol of a case already run"))
        p1 = self.coverage_planned.get("projection_1d") or {}
        if p1:
            worst = min(p1, key=p1.get)
            out.append(CheckRecord(name="1-D projections filled", status="PASS" if p1[worst] >= 0.5 else "WARN",
                                   value=p1[worst], threshold=0.5,
                                   detail=f"worst parameter {worst}: fraction of n bins holding a case"))
        return out

    def study_result(self, app: str = "campaign") -> StudyResult:
        metrics = {"n_cases": int(len(self.cases)), "n_dropped_as_repeats": int(len(self.dropped)),
                   "discrepancy": self.coverage_planned.get("discrepancy"),
                   "min_distance": self.coverage_planned.get("min_distance"),
                   "fill_distance": self.coverage_planned.get("fill_distance")}
        if self.coverage_combined:
            metrics["fill_distance_with_existing"] = self.coverage_combined.get("fill_distance")
            metrics["fill_distance_existing_only"] = self.coverage_existing.get("fill_distance")
        limits = ["Coverage is measured in the unit cube of the parameters (log scales and categories mapped), not "
                  "in the response: a uniform design can still miss a narrow feature of the outputs.",
                  "Adaptive infill weights by how much existing outputs vary between neighbours; it is a "
                  "heuristic, not a model-based error estimate."]
        return StudyResult(id=self.id, study_type="doe", app=app, title=self.name, status="planned",
                           parameter_space=self.space, table=Table.from_frame(self.cases), metrics=metrics,
                           checks=self.checks(), limits=limits,
                           provenance=Provenance(app=app, study_id=self.id, source=f"design {self.design.method}"))

    def dataset_skeleton(self, outputs: Sequence[str] = (), units: dict[str, str] | None = None,
                         app: str = "campaign") -> DatasetSpec:
        """The dataset these runs will make: inputs and declared outputs, no results yet (status "skeleton")."""
        units = units or {}
        cols = [Column(name="case_id", role="id", dtype="str")]
        cols += [Column(name=p.name, role="input", unit=p.unit,
                        dtype="str" if p.kind == "categorical" else "int64" if p.kind == "integer" else "float64")
                 for p in self.space.parameters]
        cols += [Column(name=o, role="output", unit=units.get(o, "")) for o in outputs]
        return DatasetSpec(name=self.name, columns=cols, n_rows=int(len(self.cases)), parameter_space=self.space,
                           status="skeleton",
                           provenance=Provenance(app=app, study_id=self.id, source=f"design {self.design.method}",
                                                 notes="planned cases; outputs filled by the runs"))


def plan_campaign(space: ParameterSpace, n: int | None = None, method: str = "lhs", *, existing=None,
                  existing_id_column: str | None = None, outputs: Sequence[str] = (), seed: int = 0,
                  levels: int | dict[str, int] = 3, tol: float | None = None, drop_repeats: bool = True,
                  feasible: Callable[[Any], np.ndarray] | None = None, name: str | None = None,
                  id_prefix: str = "case") -> CampaignPlan:
    """Design ``n`` cases (default ``space.budget``), drop those repeating an existing run (``drop_repeats``), order
    the rest by priority and measure the coverage of the plan, of what exists, and of both together. Existing cases
    outside the space (bounds or constraints) are left out of the coverage and listed in
    ``redundancy["existing_outside_space"]``."""
    n = int(n if n is not None else (space.budget or 0))
    if method not in ("full_factorial", "fractional_factorial") and n < 1:
        raise ValueError("give the number of cases (or a budget in the parameter space)")
    if existing is not None:
        missing = [p for p in space.names if p not in existing.columns]
        if missing:
            raise KeyError(f"existing cases lack columns {missing}")
        inside = space.feasible(existing[space.names]) if len(existing) else np.ones(0, dtype=bool)
        outside = existing[~inside]
        existing = existing[inside]
    else:
        outside = None
    des = design(space, n, method, seed=seed, existing=existing, outputs=outputs, levels=levels, feasible=feasible)
    ex_unit = space.to_unit(existing) if existing is not None and len(existing) else None
    ex_ids = ([str(v) for v in existing[existing_id_column]] if existing is not None and existing_id_column
              else None)
    red = redundancy(des.unit, ex_unit, tol=tol, existing_ids=ex_ids)
    flags = red["cases"]
    keep = np.array([not (f["repeats_existing"] or f["repeats_planned"]) for f in flags]) if drop_repeats \
        else np.ones(len(flags), dtype=bool)
    cases = des.cases.reset_index(drop=True)
    for col, key in (("nearest_existing", "nearest_existing"), ("distance_to_existing", "distance_to_existing"),
                     ("repeats_existing", "repeats_existing"), ("repeats_planned", "repeats_planned")):
        cases[col] = [f[key] for f in flags]
    dropped = cases[~keep].reset_index(drop=True)
    kept, unit = cases[keep].reset_index(drop=True), des.unit[keep]
    pr = prioritize(unit, ex_unit) if len(unit) else {"order": [], "rank": [], "score": []}
    kept.insert(0, "priority", pr["rank"])
    kept.insert(1, "information", pr["score"])
    kept = kept.sort_values("priority").reset_index(drop=True)
    unit = unit[np.asarray(pr["order"], dtype=int)] if len(unit) else unit
    width = max(3, len(str(len(kept))))
    kept.insert(0, "case_id", [f"{id_prefix}_{i + 1:0{width}d}" for i in range(len(kept))])
    names = [p.name for p in space.free]
    levels = {p.name: len(p.values) if p.kind == "categorical" else int(p.high - p.low) + 1
              for p in space.free if p.kind in ("categorical", "integer")}
    cov_plan = coverage(unit, names, seed=seed, levels=levels) if len(unit) else {"n": 0}
    cov_ex = coverage(ex_unit, names, seed=seed, levels=levels) if ex_unit is not None else None
    cov_all = (coverage(np.vstack([ex_unit, unit]), names, seed=seed, levels=levels)
               if ex_unit is not None and len(unit) else None)
    red_summary = {**red, "cases": [flags[i] for i in np.flatnonzero(keep)]}
    if outside is not None:
        red_summary["n_existing_outside_space"] = int(len(outside))
        red_summary["existing_outside_space"] = ([str(v) for v in outside[existing_id_column]]
                                                 if existing_id_column else outside.index.tolist())
    return CampaignPlan(space=space, design=des, cases=kept, unit=unit, coverage_planned=cov_plan,
                        coverage_existing=cov_ex, coverage_combined=cov_all, redundancy=red_summary, dropped=dropped,
                        existing_unit=ex_unit, existing_ids=ex_ids, name=name or space.name)
