"""Planning-only Physics AI pipeline architect. No solvers or training are executed."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
import json

@dataclass
class Problem:
    description: str
    domain: str = "unknown"
    geometry: str = "unknown"
    has_geometry: bool | None = None
    has_mesh: bool | None = None
    has_dataset: bool | None = None
    goal: str = "forward"
    varying_geometry: bool = False
    varying_parameters: bool = False
    simulation_seconds: float | None = None
    simulations_needed: int | None = None
    max_compute_seconds: float | None = None
    governing_equations: str | None = None
    initial_conditions: str | None = None
    boundary_conditions: str | None = None

@dataclass
class Step:
    id: str
    title: str
    why: str
    depends_on: list[str] = field(default_factory=list)
    verification: str = ""
    technology: str = ""
    status: str = "proposed"

def plan(problem: Problem) -> dict[str, Any]:
    if not problem.description.strip():
        raise ValueError("Problem description required")
    steps: list[Step] = []
    questions: list[str] = []
    def add(id, title, why, deps=(), verification="", technology=""):
        steps.append(Step(id, title, why, list(deps), verification, technology))
    add("physics", "Review physical formulation", "Confirm equations, assumptions, units, ICs and BCs; propose alternatives for expert approval.", verification="Dimensional consistency, conservation, limiting cases", technology="PhysicalProblem + VeriPhysics")
    if not problem.governing_equations: questions.append("Which governing equations or physical phenomena apply?")
    if not problem.boundary_conditions: questions.append("What boundary conditions and external loads are known?")
    if not problem.initial_conditions: questions.append("Are initial conditions needed, and if so what are they?")
    if problem.geometry == "unknown": questions.append("Is this a 0D/1D/2D/3D problem, and is the geometry fixed?")
    prior = "physics"
    if problem.geometry == "3d" or problem.has_geometry is False:
        if problem.has_geometry is False:
            add("cad", "Create geometry", "Geometry is not supplied.", [prior], "CAD integrity and dimensions", "CAD adapter")
            prior = "cad"
        elif problem.has_geometry is None:
            questions.append("Do you already have CAD geometry?")
    if problem.geometry in ("2d", "3d"):
        if problem.has_mesh is False:
            add("mesh", "Generate mesh", "A discretized spatial solver needs a mesh.", [prior], "Mesh quality and refinement convergence", "Meshing adapter")
            prior = "mesh"
        elif problem.has_mesh is None:
            questions.append("Do you already have a suitable mesh?")
    solvers = {"thermal":"FEniCS / thermal FEM", "fluid":"OpenFOAM", "structural":"CalculiX", "electromagnetic":"EM solver adapter"}
    solver = solvers.get(problem.domain.lower(), "Solver selection pending physics classification")
    add("solver", "Select and benchmark solver", f"Candidate: {solver}. Benchmark a representative case before estimating runtime or cost.", [prior], "Reference case, convergence, conservation", "Solver adapters")
    prior = "solver"
    if problem.has_dataset is False:
        add("dataset", "Design simulation dataset", "Use parameter/geometry sampling, independent held-out groups, metadata and fidelity tracking.", [prior], "Coverage, leakage checks, mesh/units alignment", "Dataset registry")
        prior = "dataset"
    elif problem.has_dataset is None:
        questions.append("Do you have a dataset with provenance, units and train/test separation?")
    estimated = None
    if problem.simulation_seconds is not None and problem.simulations_needed is not None:
        estimated = problem.simulation_seconds * problem.simulations_needed
    cheap = estimated is not None and problem.max_compute_seconds is not None and estimated <= problem.max_compute_seconds
    architecture = None
    if not cheap:
        if problem.goal in ("inverse", "parameter_identification"):
            architecture = "Inverse PINN / adjoint / Bayesian inversion (compare identifiability)"
        elif problem.varying_geometry:
            architecture = "Mesh GNN or geometry-conditioned DeepONet; FNO only after common-grid mapping is justified"
        elif problem.varying_parameters:
            architecture = "DeepONet / FNO / PINO; compare against interpolation and reduced-order baselines"
        else:
            architecture = "PINN or conventional numerical model; benchmark both before selecting"
        add("model", "Evaluate surrogate strategy", architecture, [prior], "Held-out error, physics residuals, OOD and uncertainty", "PINNeAPPle model selection")
        prior = "model"
    else:
        add("direct", "Use conventional solver directly", "Estimated full sweep fits the declared compute budget; surrogate is optional.", [prior], "Convergence and reference comparison", "Solver adapters")
        prior = "direct"
    add("validate", "Define evidence and validation gates", "Specify acceptance tolerances, independent reference and domain of applicability.", [prior], "VeriPhysics evidence; unrun checks remain unvalidated", "pinneapple_veriphysics")
    if problem.goal in ("optimization", "control", "inverse", "parameter_identification", "digital_twin"):
        add("application", "Plan downstream application", f"Goal: {problem.goal}; include observability, uncertainty and safety constraints.", ["validate"], "Independent application-level tests", "Engineering API")
    if estimated is None: questions.append("What is the measured runtime per representative simulation and required case count?")
    if problem.max_compute_seconds is None: questions.append("What compute budget or maximum turnaround time is acceptable?")
    return {"schema_version":"0.1.0","mode":"plan_only","problem":asdict(problem),
            "steps":[asdict(s) for s in steps], "questions":questions,
            "cost_estimate":{"total_serial_seconds":estimated,"basis":"user-supplied measurements" if estimated is not None else "not benchmarked"},
            "architecture_candidate":architecture,"requires_expert_review":True}

def to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False)

def to_mermaid(result: dict[str, Any]) -> str:
    lines = ["flowchart TD"]
    for step in result["steps"]:
        lines.append(f'  {step["id"]}["{step["title"]}"]')
        for dep in step["depends_on"]:
            lines.append(f'  {dep} --> {step["id"]}')
    return "\n".join(lines)
