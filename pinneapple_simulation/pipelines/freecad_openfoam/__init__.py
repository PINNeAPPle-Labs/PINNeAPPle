"""Headless, reproducible pipelines from a parametric FreeCAD model to a checked OpenFOAM result.

    from pinneapple_simulation.pipelines.freecad_openfoam import PipelineSpec, run_pipeline
    spec = PipelineSpec(id="bend", title="Pipe bend", template="pipe_bend",
                        inputs={"Re": {"default": 200, "min": 10, "max": 1000}},
                        physics={"nu": 1e-6, "rho": 1000, "velocity": "Re * nu / (D * 1e-3)"})
    run = run_pipeline(spec, "runs/bend", inputs={"Rc": 30, "Re": 150})
    print(run.state, run.message)            # study_result.json, report.html, figures in runs/bend

Modules: :mod:`.cad` (FreeCAD headless / NumPy stand-in, patch rules, STL), :mod:`.models` (parametric
templates), :mod:`.mesh` (blockMesh + snappyHexMesh / cfMesh), :mod:`.case` (simpleFoam / pimpleFoam, passive
scalar), :mod:`.post` (pressures, fluxes, sections, wall loads, figures), :mod:`.steps` (the pipeline actions),
:mod:`.pipeline` (spec -> automation workflow -> run), :mod:`.sweep` (DOE, one process per case, resume, surrogate),
:mod:`.studies` (the three validated worked studies).
Guide: ``docs/guides/freecad_openfoam_pipelines.md``.
"""
from .cad import CadGeometry, FakeFreeCAD, FreeCADBackend, PatchSpec, assign_patches, get_backend
from .case import Physics, ScalarSpec
from .mesh import MeshSpec
from .models import MODELS, Model, build_document
from .pipeline import FUNCTIONS, PipelineSpec, evaluate, resolve_physics, run_pipeline
from .post import Section
from .studies import STUDIES
from .sweep import Sweep, doe, fit_surrogate

__all__ = ["PipelineSpec", "run_pipeline", "evaluate", "resolve_physics", "FUNCTIONS", "MODELS", "Model",
           "build_document", "CadGeometry", "PatchSpec", "assign_patches", "get_backend", "FreeCADBackend",
           "FakeFreeCAD", "MeshSpec", "Physics", "ScalarSpec", "Section", "Sweep", "doe", "fit_surrogate", "STUDIES"]
