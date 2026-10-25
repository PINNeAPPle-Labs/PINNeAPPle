"""
PINNeAPPle Arena - Problems API router.

Endpoints for listing and retrieving physics problems, with filtering and
dataset preview generation.
"""
# ruff: noqa: B008  (FastAPI Depends/Query/File defaults)
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from pinn_arena.schemas import PhysicsArea
from pinneapple_arena.pointwise.datasets import DatasetGenerator
from pinneapple_arena.pointwise.problems import PROBLEMS_CATALOG

router = APIRouter(prefix="/problems", tags=["problems"])


@router.get("", summary="List all physics problems")
async def list_problems(
    domain: PhysicsArea | None = Query(None, description="Filter by physics domain"),
    dimension: str | None = Query(None, description="Filter by dimension: ONE_D, TWO_D, THREE_D"),
    difficulty: str | None = Query(None, description="Filter by difficulty: BEGINNER, INTERMEDIATE, ADVANCED"),
    search: str | None = Query(None, description="Search in name or description (case-insensitive)"),
    requires_geometry: bool | None = Query(None, description="Filter by whether a geometry is required"),
):
    """
    List all available physics problems with optional filtering.

    Supports filtering by:
    - Physics domain (fluid mechanics, solid mechanics, etc.)
    - Spatial dimension
    - Difficulty level
    - Free-text search in name and description
    - Whether a 3D geometry is required
    """
    results = list(PROBLEMS_CATALOG.values())

    if domain is not None:
        results = [p for p in results if p.domain == domain]

    if dimension is not None:
        dim_upper = dimension.upper()
        results = [p for p in results if p.dimension.value == dim_upper]

    if difficulty is not None:
        diff_upper = difficulty.upper()
        results = [p for p in results if p.difficulty.value == diff_upper]

    if requires_geometry is not None:
        results = [p for p in results if p.requires_geometry == requires_geometry]

    if search is not None:
        search_lower = search.lower()
        results = [
            p for p in results
            if search_lower in p.name.lower() or search_lower in p.description.lower()
        ]

    return {
        "total": len(results),
        "problems": [p.model_dump() for p in results],
    }


@router.get("/{problem_id}", summary="Get a single physics problem")
async def get_problem(problem_id: str):
    """
    Get the full specification for a specific physics problem by ID.

    Returns all metadata including equations, parameters, boundary conditions,
    initial conditions, and PINNeAPPle preset key.
    """
    problem = PROBLEMS_CATALOG.get(problem_id)
    if problem is None:
        raise HTTPException(
            status_code=404,
            detail=f"Problem '{problem_id}' not found. Available: {list(PROBLEMS_CATALOG.keys())}",
        )
    return problem.model_dump()


@router.get("/{problem_id}/dataset-preview", summary="Preview the reference dataset")
async def preview_dataset(
    problem_id: str,
    n_samples: int = Query(default=500, ge=10, le=5000, description="Number of sample points to return"),
    geometry_id: str | None = Query(None, description="Selected geometry ID (used to customise the dataset generator)"),
):
    """
    Generate a small preview of the reference (ground truth) dataset.

    Uses the DatasetGenerator to produce analytically or numerically computed
    reference solutions. Returns coordinate arrays and solution values suitable
    for frontend visualization.
    """
    if problem_id not in PROBLEMS_CATALOG:
        raise HTTPException(status_code=404, detail=f"Problem '{problem_id}' not found.")

    problem = PROBLEMS_CATALOG[problem_id]

    # Build default parameters from problem parameter specs
    default_params: dict = {}
    for param in problem.parameters:
        default_params[param.name] = param.default

    # Pass geometry_id so generators can branch on geometry type
    if geometry_id is not None:
        default_params["geometry_id"] = geometry_id

    try:
        gen = DatasetGenerator(problem_id, default_params)
        data = gen.generate(n_samples=n_samples)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset generation failed for '{problem_id}': {str(exc)}",
        ) from exc

    # Convert numpy arrays to JSON-serializable lists
    serializable: dict = {"problem_id": problem_id, "n_samples": n_samples}
    for key, value in data.items():
        if hasattr(value, "tolist"):
            serializable[key] = value.tolist()
        elif isinstance(value, dict):
            serializable[key] = value
        else:
            serializable[key] = value

    return serializable


@router.get("/{problem_id}/parameters", summary="Get problem parameter specs")
async def get_problem_parameters(problem_id: str):
    """
    Return the list of configurable parameters for a problem, with defaults,
    min/max ranges, and units. Used to build the frontend parameter form.
    """
    problem = PROBLEMS_CATALOG.get(problem_id)
    if problem is None:
        raise HTTPException(status_code=404, detail=f"Problem '{problem_id}' not found.")
    return {
        "problem_id": problem_id,
        "parameters": [p.model_dump() for p in problem.parameters],
    }
