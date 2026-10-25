"""Global leaderboard: every completed experiment of this server, aggregated by
``pinneapple_arena.pointwise.leaderboard.aggregate`` (ranked by mean relative L2, plus per-objective scores).

The database starts empty: no leaderboard rows are shipped with the app, so every entry comes from an experiment
run on this server, with the library version it ran on (``summary.pinneapple_version``).
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query

from pinn_arena.db import SessionLocal
from pinn_arena.orm import ExperimentRecord
from pinn_arena.store import ExperimentStore
from pinneapple_arena.pointwise.architectures import ARCHITECTURES_CATALOG
from pinneapple_arena.pointwise.leaderboard import aggregate
from pinneapple_arena.pointwise.problems import PROBLEMS_CATALOG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])

_EMPTY = {"total_experiments": 0, "total_architectures": 0, "problems_covered": 0, "entries": []}


def _completed_experiments(problem_ids: set | None = None) -> list[dict]:
    """Completed experiments from the database, plus completed in-memory ones not yet saved."""
    db = SessionLocal()
    try:
        q = db.query(ExperimentRecord).filter(ExperimentRecord.status == "COMPLETED",
                                             ExperimentRecord.results.isnot(None))
        if problem_ids is not None:
            q = q.filter(ExperimentRecord.problem_id.in_(problem_ids))
        records = q.all()
    finally:
        db.close()
    out, seen = [], set()
    for r in records:
        try:
            results = json.loads(r.results)
        except (TypeError, ValueError):
            continue
        seen.add(r.id)
        out.append({"id": r.id, "winner": r.winner or "", "problem_id": r.problem_id or "", "model_results": results})
    for exp_id, exp in list(ExperimentStore._experiments.items()):
        if exp_id in seen or str(exp.get("status", "")).upper() != "COMPLETED" or not exp.get("results"):
            continue
        pid = exp.get("config", {}).get("problem_id", "")
        if problem_ids is not None and pid not in problem_ids:
            continue
        out.append({"id": exp_id, "winner": exp.get("winner") or "", "problem_id": pid,
                    "model_results": exp["results"] if isinstance(exp["results"], list) else []})
    return out


def _families() -> dict[str, str]:
    return {a: (i.family.value if hasattr(i.family, "value") else str(i.family)) for a, i in ARCHITECTURES_CATALOG.items()}


@router.get("", summary="Global architecture leaderboard")
async def get_leaderboard(
    problem_id: str | None = Query(None, description="Filter to a specific problem ID"),
    domain: str | None = Query(None, description="Physics domain (e.g. FLUID_MECHANICS)"),
    dimension: str | None = Query(None, description="Dimension (ONE_D, TWO_D, THREE_D)"),
    difficulty: str | None = Query(None, description="Difficulty (BEGINNER, INTERMEDIATE, ADVANCED)"),
    requires_geometry: bool | None = Query(None, description="Whether problem requires mesh geometry"),
):
    """Ranking of the architectures over the completed experiments matching the filters (mean relative L2
    ascending). The mean pools different problems; filter to one problem for a like-for-like ranking."""
    problem_ids: set | None = None
    if any(v is not None for v in (problem_id, domain, dimension, difficulty, requires_geometry)):
        problem_ids = set()
        for pid, prob in PROBLEMS_CATALOG.items():
            if problem_id is not None and pid != problem_id:
                continue
            if domain is not None and prob.domain != domain:
                continue
            if dimension is not None and prob.dimension != dimension:
                continue
            if difficulty is not None and prob.difficulty != difficulty:
                continue
            if requires_geometry is not None and prob.requires_geometry != requires_geometry:
                continue
            problem_ids.add(pid)
        if not problem_ids:
            return dict(_EMPTY)
    try:
        return aggregate(_completed_experiments(problem_ids), families=_families())
    except Exception:
        logger.exception("Leaderboard aggregation failed")
        return dict(_EMPTY)


@router.post("/refresh", summary="Recompute the leaderboard")
def refresh_leaderboard() -> dict:
    """The leaderboard is computed on every request; this returns how many architectures it ranks now."""
    result = aggregate(_completed_experiments(), families=_families())
    return {"ok": True, "architectures_refreshed": len(result["entries"])}
