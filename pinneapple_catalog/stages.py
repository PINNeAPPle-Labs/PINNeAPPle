"""The open tools and the public datasets available at each stage of the physics-AI workflow.

This module answers one question per stage — *what can I use here, and how far does this repository already go?* — for
the thirteen sections the Flow palette is built from (:mod:`pinneapple_flow.palette`, whose ``SECTION_IDS`` a test
compares against :data:`STAGE_IDS`, so the two cannot drift apart silently).

Three catalogues already existed and none of them knew about stages:

* ``pinneapple_data.datasets.catalog`` — the *data*: which public dataset exists, in what format, under what licence,
  and whether there is a reader here. Each entry now carries the stages it feeds (``CatalogEntry.stages``).
* ``pinneapple_analysis.verification.tool_recommendation.TOOL_CATALOG`` — the *solvers*, ranked against a
  ``ProblemSpec``. It says so in its own docstring: it has no ``ML/surrogate`` entries, and its scope is "does this
  tool's documentation cover this PDE", not "what stage of the workflow does it serve".
* ``pinneapple_catalog.resources`` — the *licence gate* over datasets, geometry, models and benchmarks.

Neither of the last two carried a stage, and the libraries that make physics-AI work possible at all (neural operator
packages, ROM toolkits, UQ samplers, optimisers, serving runtimes) were catalogued nowhere. This module is that
missing layer, and it stays a *pointer* table: it does not re-describe a tool whose documented scope already lives in
``TOOL_CATALOG`` — an entry that overlaps says ``also_in`` and the test checks that the id is really there.

What every entry states, and what it refuses to state
-----------------------------------------------------
``license`` is an SPDX id (or the licence's own name where there is no id) read from the artefact that governs the
**code** — the repository's ``LICENSE`` file through the GitHub licence API, or the distribution metadata for a
package that ships none. ``license_verified`` is ``True`` only where that read happened; the entries marked ``False``
are the ones whose licence is stated in a place this checkout did not open, and they say so in ``license_source``.
A verified licence is not a commercial clearance — GPL-3.0 and AGPL-3.0 are read correctly and are still unusable in
a closed product, which is why the id is carried rather than a boolean.

``integration`` is one of three, and only one of them is a claim about this repository:

``"wired"``
    a code path here calls it. ``surface`` then names the module, and a test resolves that module's spec, so an
    entry cannot claim wiring to a file that does not exist.
``"named"``
    the repository mentions it — a recommender, a lazy export, a docstring — without a path that would run it.
``"candidate"``
    nothing here touches it. Most of the map is this, on purpose: the point of the map is to show the distance.

The datasets side is projected from the catalogue rather than copied: :func:`datasets` filters
``pinneapple_data.datasets.catalog`` by stage, so a dataset is catalogued once and tagged, and
:func:`unassigned_datasets` names any catalogued dataset that no stage claims — the entry that would otherwise be
invisible from a stage's point of view.

    >>> from pinneapple_catalog import stages
    >>> [s.id for s in stages.stages()][:4]
    ['problem', 'equations', 'solvers', 'physics_ai']
    >>> [t.name for t in stages.tools(stage_id="physics_ai", integration="wired")][:3]
    ['PyTorch Geometric', 'PhysicsNeMo', 'PySINDy']
    >>> stages.coverage("solvers")["wired"] >= 3
    True
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "STAGE_IDS", "Stage", "Tool", "TOOLS", "STAGES", "StageTools",
    "coverage", "datasets", "get", "stage", "stages", "to_json", "tools", "unassigned_datasets",
]

#: The workflow stages, in workflow order, identical to ``pinneapple_flow.palette.SECTION_IDS`` (asserted by a test).
STAGE_IDS: tuple[str, ...] = (
    "problem", "equations", "solvers", "physics_ai", "surrogates", "data", "inverse", "validation",
    "optimization", "twin", "viz", "export", "library",
)

INTEGRATIONS = ("wired", "named", "candidate")


@dataclass(frozen=True)
class Stage:
    """One step of the workflow: the question it answers, what is here, and the gap that is left.

    ``kind`` separates the workflow from the palette's last section: ``library`` is the palette's drawer of generated
    ``auto.*`` calls, not a step of the workflow, so it carries no tools and no question.
    """

    id: str
    title: str
    question: str
    packages: tuple[str, ...] = ()
    gap: str = ""
    kind: str = "workflow"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "question": self.question, "kind": self.kind,
                "packages": list(self.packages), "gap": self.gap}


@dataclass(frozen=True)
class Tool:
    """One open tool, the stages it serves, and how far this repository already goes with it."""

    id: str
    name: str
    stages: tuple[str, ...]
    kind: str
    license: str
    homepage: str
    license_verified: bool = False
    license_source: str = ""
    language: str = ""
    integration: str = "candidate"
    surface: str = ""
    notes: str = ""
    also_in: str = ""

    @property
    def is_open(self) -> bool:
        """Whether the licence is an open-source one, by the id's own family (a custom licence that forbids
        commercial use is not, however source-available it is)."""
        return not self.license.lower().startswith(("commercial", "proprietary", "custom", "non-commercial"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "stages": list(self.stages), "kind": self.kind,
            "license": self.license, "license_verified": self.license_verified,
            "license_source": self.license_source, "homepage": self.homepage, "language": self.language,
            "integration": self.integration, "surface": self.surface, "notes": self.notes,
            "also_in": self.also_in, "open": self.is_open,
        }


_GH = "GitHub licence API (repository LICENSE file)"
_PYPI = "PyPI distribution metadata"

STAGES: dict[str, Stage] = {s.id: s for s in (
    Stage(
        id="problem", title="Problem & geometry",
        question="What is being solved, on what shape, and did the engineer's sentence become a machine-readable "
                 "problem (domain, units, boundary patches, material)?",
        packages=("pinneapple_design", "pinneapple_problemdesign", "pinneapple_data"),
        gap="CAD arrives as a file, not as a parameterised intent: the STEP/BREP edit that a design study needs "
            "requires an external kernel, and just two of them (FreeCAD, CadQuery) are reached from here."),
    Stage(
        id="equations", title="Equations",
        question="Which governing equations, which conditions, and are they written as something a solver or a "
                 "residual can consume — or only as prose?",
        packages=("pinneapple_physics", "pinneapple_problemdesign"),
        gap="`physics.governing` is text with its symbols defined; nothing here turns that prose into a symbolic "
            "form without a human, so SymPy/UFL are catalogued and only named."),
    Stage(
        id="solvers", title="Classical solvers",
        question="What produces the reference numbers that the surrogate will be scored against?",
        packages=("pinneapple_solvers", "pinneapple_simulation"),
        gap="The bridges are per product (CFD, structural), not one backend contract: a new code is a new adapter. "
            "Multi-solver coupling (preCICE) is catalogued and unused."),
    Stage(
        id="physics_ai", title="Physics AI",
        question="Which model family (PINN, neural operator, graph network, differentiable solver) and which "
                 "package actually trains it?",
        packages=("pinneapple_neural", "pinneapple_physics", "pinneapple_models"),
        gap="No common training interface: the user picks the architecture, and each family's own package is a "
            "different dependency tree. The GPU wave solvers (Warp, Taichi) are wired for one demo path each."),
    Stage(
        id="surrogates", title="Surrogates & ROMs",
        question="Which cheap model stands in for the solver once the data exists, and how is it chosen?",
        packages=("pinneapple_models", "pinneapple_neural", "pinneapple_decision"),
        gap="Projection-based ROM (pyMOR, RBniCS, EZyRB) is catalogued and not used; the ROMs here are the library's "
            "own POD/DMD."),
    Stage(
        id="data", title="Data",
        question="Where does the training data come from, in what format, and is the licence one a product may "
                 "ship?",
        packages=("pinneapple_data", "pinneapple_lab"),
        gap="The catalogue names twenty-odd public sources and reads six; the readers are per archive layout, and "
            "each new archive is a new reader."),
    Stage(
        id="inverse", title="Inverse & calibration",
        question="Which parameters, materials or source terms are unknown, and which inference buys them from "
                 "measurements?",
        packages=("pinneapple_physics", "pinneapple_analysis"),
        gap="The library calibrates with its own optimisers; the Bayesian samplers (PyMC, emcee, dynesty) are "
            "catalogued and named, not wired, so no posterior comes out of a PINNeAPPle run."),
    Stage(
        id="validation", title="Validation & UQ",
        question="How far is the prediction from a reference, and what is the number that says so?",
        packages=("pinneapple_analysis", "pinneapple_veriphysics", "pinneapple_check"),
        gap="Data-side and model-side gates exist and report a number; the UQ behind them is a seed-ensemble "
            "disagreement, not a posterior, and no catalogued sampler is wired to make it one."),
    Stage(
        id="optimization", title="Optimization & control",
        question="Given a trustworthy surrogate, which design should be built?",
        packages=("pinneapple_design", "pinneapple_decision"),
        gap="Bayesian optimisation is present but not connected to the trust layer: a candidate can be ranked "
            "without the gate ever deciding whether the surrogate may be believed there."),
    Stage(
        id="twin", title="Digital twin",
        question="Which live signals update the model, and what does the model say before the next measurement?",
        packages=("pinneapple_simulation", "pinneapple_analysis"),
        gap="Streaming is HTTP polling and the telemetry bundle; no message broker, no FMU co-simulation in a twin "
            "loop (FMPy is wired only for the onboarding doctor)."),
    Stage(
        id="viz", title="Visualization",
        question="Can an engineer see the field, the mesh and the difference, in 3-D and as an animation?",
        packages=("pinneapple_lab", "pinneapple_blender"),
        gap="PyVista is wired for the schema's bundles; the CAE readers' own results still go through per-product "
            "plot code."),
    Stage(
        id="export", title="Export & report",
        question="How does the result leave — as a portable model, a mesh, or a document a reviewer signs?",
        packages=("pinneapple_lab", "pinneapple_hub"),
        gap="ONNX is exported and timed with three fallbacks; the rest of the list (OpenUSD, CGNS, Triton) is "
            "catalogued and not wired."),
    Stage(
        id="library", title="Library calls (advanced)",
        question="",
        kind="internal",
        gap="Not a workflow stage: the palette's drawer of generated `auto.*` calls, kept in the list so the stage "
            "ids match the palette exactly."),
)}


def _t(**kw: Any) -> Tool:
    kw["stages"] = tuple(kw["stages"])
    return Tool(**kw)


# Internal, then rebuilt through _t so a mistyped stage id fails at import rather than at query time.
_RAW: tuple[dict[str, Any], ...] = (
    # ── problem ─────────────────────────────────────────────────────────────────────────────────────────────
    dict(id="occt", name="OpenCASCADE (OCCT)", stages=("problem", "export"), kind="geometry kernel",
         license="LGPL-2.1", license_verified=True, license_source=_GH,
         homepage="https://dev.opencascade.org/", language="C++",
         notes="The B-REP kernel under FreeCAD, CadQuery and Salome; STEP/IGES read-write. LGPL-2.1 binds a "
               "modified OCCT, not a program that links it."),
    dict(id="cadquery", name="CadQuery", stages=("problem",), kind="CAD scripting",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://github.com/CadQuery/cadquery", language="Python", integration="named",
         surface="pinneapple", notes="Scripted solids over OCCT (OCP bindings); the parametric shape a design study "
               "wants, without a GUI. Named only as an optional dependency in the top-level namespace -- no node "
               "calls it."),
    dict(id="freecad", name="FreeCAD", stages=("problem",), kind="CAD",
         license="LGPL-2.1", license_verified=True, license_source=_GH,
         homepage="https://www.freecad.org/", language="C++/Python", integration="wired",
         surface="pinneapple_simulation.pipelines.freecad_openfoam",
         notes="Wired as a pipeline: a FreeCAD model becomes an OpenFOAM case. GPL-2.0 applies to the FreeCAD "
               "application itself, LGPL-2.1 to its libraries."),
    dict(id="gmsh", name="Gmsh", stages=("problem", "solvers"), kind="mesher",
         license="GPL-2.0-or-later (with the Gmsh linking exception)", license_verified=True,
         license_source="read 2026-11-25 at gitlab.onelab.info/gmsh/gmsh/-/raw/master/LICENSE.txt: GPL version 2 "
                        "or later, plus an exception for code that links the Gmsh libraries",
         homepage="https://gmsh.info/", language="C++", integration="named", surface="pinneapple",
         notes="The default mesher for the FEM path; the linking exception is what lets it be driven from a "
               "script without GPL-ing the caller. Commercial use is fine, and the GPL reaches a modified Gmsh "
               "and the Gmsh code itself, not the caller the exception names."),
    dict(id="trimesh", name="trimesh", stages=("problem", "viz"), kind="geometry IO",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://github.com/mikedh/trimesh", language="Python", integration="named", surface="pinneapple",
         notes="STL/OBJ/PLY in and out; the interchange format for surface geometry."),
    dict(id="salome", name="SALOME", stages=("problem",), kind="CAD/meshing platform",
         license="LGPL-2.1", license_verified=True,
         license_source="read 2026-11-25 at salome-platform.org/?page_id=2622 (Legal information): GNU Lesser "
                        "General Public License version 2.1",
         homepage="https://www.salome-platform.org/", language="C++/Python",
         notes="OCCT-based CAD plus meshing in one application; the shape-side alternative to a FreeCAD pipeline. "
               "Commercial use is fine; a modified SALOME stays LGPL, and linking to the libraries does not reach "
               "the caller's own code."),

    # ── equations ───────────────────────────────────────────────────────────────────────────────────────────
    dict(id="sympy", name="SymPy", stages=("equations",), kind="computer algebra",
         license="BSD-3-Clause", license_verified=True, license_source=_PYPI,
         homepage="https://www.sympy.org/", language="Python", integration="wired",
         surface="pinneapple_physics.symbolic_pde", notes="`pde_from_sympy` takes a symbolic expression into the "
               "PDE environment, which is the one place where an equation is written by a human and executed."),
    dict(id="ufl", name="UFL (FEniCS form language)", stages=("equations", "solvers"), kind="variational form",
         license="LGPL-3.0", license_verified=True, license_source=_GH,
         homepage="https://github.com/FEniCS/ufl", language="Python", integration="candidate",
         notes="A PDE written as a variational form is executable; this is the notation that makes it so."),
    dict(id="modelica_stdlib", name="Modelica Standard Library", stages=("equations", "twin"),
         kind="acausal model library",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://github.com/modelica/ModelicaStandardLibrary", language="Modelica",
         integration="candidate",
         notes="Acausal component models (fluids, mechanics, thermal) — the equations of a system, not of a field."),
    dict(id="openmodelica", name="OpenModelica", stages=("equations", "twin"), kind="Modelica compiler",
         license="OSMC-PL-1.8", license_verified=True, license_source=_GH,
         homepage="https://openmodelica.org/", language="C++/MetaModelica", integration="wired",
         surface="pinneapple_simulation.external_solvers.modelica",
         notes="OSMC Public License is not an OSI licence and has a copyleft clause: read it before shipping it "
               "inside a product."),
    dict(id="fmpy", name="FMPy", stages=("twin",), kind="FMU co-simulation",
         license="BSD-2-Clause", license_verified=True, license_source=_GH,
         homepage="https://github.com/CATIA-Systems/FMPy", language="Python", integration="wired",
         surface="pinneapple_core.onboarding.doctor",
         notes="Runs a Functional Mock-up Unit; the FMI standard is what makes a vendor model runnable in a twin."),
    dict(id="cantera", name="Cantera", stages=("equations", "solvers"), kind="thermochemistry",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://cantera.org/", language="C++/Python", integration="wired",
         surface="pinneapple_arena.external.models_data",
         notes="Chemical kinetics and thermodynamic properties behind reacting-flow cases."),
    dict(id="coolprop", name="CoolProp", stages=("equations",), kind="fluid properties",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="http://www.coolprop.org/", language="C++/Python", integration="wired",
         surface="pinneapple_catalog.methods",
         notes="Equation-of-state properties for a fluid named in a case; a closure, not a solver."),
    dict(id="sundials", name="SUNDIALS", stages=("equations", "solvers"), kind="ODE/DAE integrators",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://github.com/LLNL/sundials", language="C", integration="candidate",
         notes="CVODE/IDA: the stiff-integrator baseline for a time-dependent problem, and the reference many "
               "PINN papers compare against."),

    # ── solvers ────────────────────────────────────────────────────────────────────────────────────────────
    dict(id="openfoam", name="OpenFOAM", stages=("problem", "solvers", "validation"), kind="finite volume CFD",
         license="GPL-3.0", license_verified=True, license_source=_GH,
         homepage="https://openfoam.org/", language="C++", integration="wired",
         surface="pinneapple_simulation.numerical_solvers.openfoam_bridge",
         notes="The reference CFD code for the CFD product's cases. GPL-3.0 covers the code, not the output data.",
         also_in="openfoam"),
    dict(id="su2", name="SU2", stages=("solvers", "optimization"), kind="compressible CFD + adjoint",
         license="LGPL-2.1", license_verified=True, license_source=_GH,
         homepage="https://su2code.github.io/", language="C++", integration="named",
         surface="pinneapple_analysis.verification.tool_recommendation", also_in="su2",
         notes="Adjoint-based shape optimisation as a first-class workflow, which is the gradient the design stage "
               "needs."),
    dict(id="dolfinx", name="FEniCSx (DOLFINx)", stages=("equations", "solvers"), kind="finite element",
         license="LGPL-3.0", license_verified=True, license_source=_GH,
         homepage="https://fenicsproject.org/", language="C++/Python", integration="wired",
         surface="pinneapple_simulation.numerical_solvers.fenics_bridge",
         notes="A variational form solved as written; the closest thing to an executable equation here.",
         also_in="fenicsx"),
    dict(id="firedrake", name="Firedrake", stages=("equations", "solvers"), kind="finite element",
         license="LGPL-3.0-or-later", license_verified=True, license_source=_GH,
         homepage="https://www.firedrakeproject.org/", language="Python", integration="candidate",
         notes="UFL front end like FEniCSx, with a different assembly and adjoint story; the alternative when "
               "DOLFINx's mesh handling is the limit."),
    dict(id="dealii", name="deal.II", stages=("solvers", "validation"), kind="finite element library",
         license="LGPL-2.1-or-later", license_verified=True, license_source=_GH,
         homepage="https://www.dealii.org/", language="C++", integration="candidate",
         notes="Adaptive hp-refinement and a tutorial suite that is itself a verification corpus."),
    dict(id="calculix", name="CalculiX", stages=("solvers", "validation"), kind="structural FEM",
         license="GPL-2.0", license_verified=True, license_source=_GH,
         homepage="http://www.calculix.de/", language="C/Fortran", integration="wired",
         surface="pinneapple_simulation.external_solvers.calculix",
         notes="Abaqus-compatible decks, which is why it is the structural reference here.", also_in="calculix"),
    dict(id="elmer", name="Elmer", stages=("solvers",), kind="multiphysics FEM",
         license="GPL-2.0", license_verified=True, license_source=_GH,
         homepage="https://www.csc.fi/web/elmer", language="Fortran/C++", integration="candidate",
         notes="Coupled multiphysics (thermal, electromagnetic, structural) with a long verification manual."),
    dict(id="kratos", name="Kratos Multiphysics", stages=("solvers",), kind="multiphysics framework",
         license="BSD-4-Clause (Kratos licence, per application)", license_verified=True, license_source=_GH,
         homepage="https://github.com/KratosMultiphysics/Kratos", language="C++", integration="candidate",
         notes="The core and each application carry their own licence file; a BSD-4-Clause clause is not "
               "compatible with every product policy."),
    dict(id="chrono", name="Project Chrono", stages=("solvers",), kind="multibody dynamics",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://projectchrono.org/", language="C++", integration="candidate",
         notes="Rigid and flexible multibody with granular and vehicle terrain models — the mechanical half of a "
               "vehicle study."),
    dict(id="precice", name="preCICE", stages=("solvers",), kind="solver coupling",
         license="LGPL-3.0", license_verified=True, license_source=_GH,
         homepage="https://precice.org/", language="C++", integration="candidate",
         notes="Couples two independent solvers (fluid and structure, say) across their own processes; the answer "
               "to 'connecting multiple solvers' that this repository does not have."),
    dict(id="jax_cfd", name="JAX-CFD", stages=("solvers", "physics_ai"), kind="differentiable CFD",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://github.com/google/jax-cfd", language="Python/JAX", integration="wired",
         surface="pinneapple_simulation.numerical_solvers.jax_cfd_solver",
         notes="Differentiable, GPU-capable finite-volume/spectral solvers: a solver whose gradient is available "
               "to the optimisation stage."),
    dict(id="phiflow", name="Φ-Flow", stages=("solvers", "physics_ai"), kind="differentiable simulation",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://github.com/tum-pbs/PhiFlow", language="Python", integration="candidate",
         notes="PDEs written once and run on NumPy, PyTorch, JAX or TensorFlow; the same code as solver and as "
               "training loop."),
    dict(id="jax_fem", name="jax-fem", stages=("solvers", "inverse"), kind="differentiable FEM",
         license="GPL-3.0", license_verified=True, license_source=_GH,
         homepage="https://github.com/tianjuxue/jax-fem", language="Python/JAX", integration="candidate",
         notes="GPL-3.0 is a decision, not a detail: a differentiable FEM under GPL is unusable inside a closed "
               "product."),
    dict(id="warp", name="NVIDIA Warp", stages=("solvers", "physics_ai"), kind="GPU kernel language",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://github.com/NVIDIA/warp", language="Python/CUDA", integration="named",
         surface="pinneapple_analysis.verification.tool_recommendation",
         notes="Differentiable kernels for simulation on the GPU; the execution layer under several newer "
               "physics-AI solvers."),
    dict(id="taichi", name="Taichi", stages=("solvers", "physics_ai"), kind="GPU kernel language",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://www.taichi-lang.org/", language="Python", integration="wired",
         surface="pinneapple_arena.external.finetune",
         notes="Embedded DSL that compiles to CUDA/Vulkan; the MPM and SPH demos here use it."),
    dict(id="fluidx3d", name="FluidX3D", stages=("solvers",), kind="lattice Boltzmann CFD",
         license="custom, non-commercial (no SPDX id)", license_verified=True,
         license_source="read 2026-11-25 at github.com/ProjectPhysX/FluidX3D/blob/master/LICENSE.md: free for "
                        "public research, education and personal use, 'Commercial use is not allowed', no military "
                        "use, no AI training on the source; altered versions that are published must publish "
                        "their source too",
         homepage="https://github.com/ProjectPhysX/FluidX3D", language="OpenCL/C++", integration="wired",
         surface="pinneapple_simulation.external_solvers.fluidx3d",
         notes="GPU LBM, and the one catalogue entry whose licence forbids the delivery this repository sells: a "
               "paid study may read and compare it, but may not run or ship it without the author's permission. "
               "Not an open-source licence, so it is not a dependency of anything distributed here."),
    dict(id="openradioss", name="OpenRadioss", stages=("solvers",), kind="explicit crash/impact",
         license="AGPL-3.0-or-later", license_verified=True,
         license_source="read 2026-11-25 at github.com/gabrielrussell/OpenRadioss (mirror of the distribution, "
                        "archived 2026-10-01): COPYRIGHT.md says AGPL version 3 or later, (C) 2026 SISW Siemens; "
                        "openradioss.org and the OpenRadioss repository were withdrawn in 2026-10",
         homepage="https://openradioss.org/", language="Fortran/C", integration="wired",
         surface="pinneapple_simulation.external_solvers.openradioss",
         notes="Explicit dynamics for crash and impact — the counterpart of the explicit solver in this library, "
               "and an AGPL code, which is the strictest decision on this page: a modified binary offered as a "
               "network service must publish its source. Siemens names Simcenter Radioss as the commercial "
               "alternative, and withdrew the open distribution in 2026-10, so the homepage no longer serves it."),
    dict(id="mujoco", name="MuJoCo", stages=("solvers", "twin"), kind="rigid-body simulation",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://mujoco.org/", language="C/Python", integration="wired",
         surface="pinneapple_simulation.external_solvers.mujoco",
         notes="Contact-rich rigid-body dynamics at interactive rates; the physics a controller is trained in."),
    dict(id="newton", name="Newton Physics", stages=("solvers", "physics_ai"), kind="GPU physics engine",
         license="Apache-2.0", license_verified=True,
         license_source="read 2026-11-25 at raw.githubusercontent.com/newton-physics/newton/main/LICENSE.md: "
                        "Apache License version 2.0",
         homepage="https://github.com/newton-physics/newton", language="Python/CUDA", integration="wired",
         surface="pinneapple_simulation.external_solvers.newton_physics",
         notes="Differentiable, GPU-native rigid/soft body engine built on Warp. Apache-2.0 allows commercial use "
               "and redistribution, including inside a closed product, as long as the notice and the statement of "
               "changes travel with it."),
    dict(id="rebound", name="REBOUND", stages=("solvers",), kind="N-body",
         license="GPL-3.0-or-later", license_verified=True,
         license_source="read 2026-11-25 at github.com/hannorein/rebound LICENSE (GPL version 3) and src/rebound.h "
                        "('either version 3 of the License, or (at your option) any later version')",
         homepage="https://github.com/hannorein/rebound", language="C/Python", integration="wired",
         surface="pinneapple_simulation.numerical_solvers.nbody_rebound",
         notes="Symplectic N-body integrators; the reference under the symplectic-model family. GPL-3.0-or-later, "
               "and the Python module links the C library, so shipping it inside a product makes that product "
               "GPL: usable in a study, a decision before it becomes a dependency of delivered software."),
    dict(id="genesis", name="Genesis", stages=("solvers", "twin"), kind="physics engine",
         license="Apache-2.0", license_verified=True,
         license_source="read 2026-11-25 at raw.githubusercontent.com/Genesis-Embodied-AI/Genesis/main/LICENSE: "
                        "Apache License version 2.0",
         homepage="https://github.com/Genesis-Embodied-AI/Genesis", language="Python", integration="wired",
         surface="pinneapple_simulation.external_solvers.genesis",
         notes="Generative physics engine for robotics scenes; young, and its licence history has changed once — "
               "Apache-2.0 today, which permits commercial use and redistribution with the notice kept, but read "
               "it at the pin you install."),

    # ── physics_ai ─────────────────────────────────────────────────────────────────────────────────────────
    dict(id="deepxde", name="DeepXDE", stages=("physics_ai",), kind="PINN library",
         license="LGPL-2.1", license_verified=True, license_source=_GH,
         homepage="https://github.com/lululxvi/deepxde", language="Python", integration="named",
         surface="pinneapple_arena.external",
         notes="The most-used PINN front end: weak and strong forms, inverse problems, and its own benchmark "
               "suite."),
    dict(id="physicsnemo", name="PhysicsNeMo (Modulus)", stages=("physics_ai", "surrogates"), kind="physics-ML framework",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://github.com/NVIDIA/physicsnemo", language="Python", integration="wired",
         surface="pinneapple_data.schema.physicsnemo",
         notes="CUDA-first and multi-gigabyte: this repository converts data into its datapipe contract and does "
               "not install the framework."),
    dict(id="neuraloperator", name="neuraloperator", stages=("physics_ai", "surrogates"), kind="neural operators",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://github.com/neuraloperator/neuraloperator", language="Python", integration="named",
         surface="pinneapple_analysis.benchmark.builtin.fields",
         notes="FNO, TFNO, UNO and their tensorised variants with the resolution-invariance that makes them "
               "surrogates rather than interpolators."),
    dict(id="pytorch_geometric", name="PyTorch Geometric", stages=("physics_ai",), kind="graph learning",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://pyg.org/", language="Python", integration="wired",
         surface="pinneapple_data.graph",
         notes="The message-passing layer a mesh-graph network is built on; wired as the conversion target of a "
               "bundle."),
    dict(id="dgl", name="DGL", stages=("physics_ai",), kind="graph learning",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://www.dgl.ai/", language="Python", integration="candidate",
         notes="The alternative to PyG, with its own sampling story for large graphs."),
    dict(id="equinox", name="Equinox", stages=("physics_ai",), kind="JAX neural networks",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://github.com/patrick-kidger/equinox", language="Python/JAX", integration="candidate",
         notes="PyTorch-like module semantics on JAX; under several differentiable-solver and PINN codebases."),
    dict(id="diffrax", name="Diffrax", stages=("physics_ai", "solvers"), kind="differential equation solvers (JAX)",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://github.com/patrick-kidger/diffrax", language="Python/JAX", integration="candidate",
         notes="Adaptive ODE/SDE solvers with adjoints — the neural-ODE half of a physics-AI model."),
    dict(id="torchdiffeq", name="torchdiffeq", stages=("physics_ai",), kind="neural ODE",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://github.com/rtqichen/torchdiffeq", language="Python", integration="candidate",
         notes="The PyTorch neural-ODE reference, adjoint method included."),
    dict(id="pysindy", name="PySINDy", stages=("physics_ai", "inverse"), kind="equation discovery",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://github.com/dynamicslab/pysindy", language="Python", integration="wired",
         surface="pinneapple_arena.external.demo_datasets",
         notes="Sparse regression that returns an equation rather than weights; the interpretable end of the "
               "spectrum."),
    dict(id="pina", name="PINA", stages=("physics_ai",), kind="PINN library",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://github.com/mathLab/PINA", language="Python", integration="candidate",
         notes="PINNs plus reduced-order models in one API — the only catalogued package that spans the two "
               "stages."),

    # ── surrogates ────────────────────────────────────────────────────────────────────────────────────────
    dict(id="pymor", name="pyMOR", stages=("surrogates",), kind="model order reduction",
         license="BSD-2-Clause AND BSD-3-Clause", license_verified=True, license_source=_PYPI,
         homepage="https://pymor.org/", language="Python", integration="candidate",
         notes="Projection-based ROM with certified error estimators: the number a ROM usually lacks."),
    dict(id="rbnics", name="RBniCS", stages=("surrogates",), kind="reduced basis",
         license="LGPL-3.0", license_verified=True, license_source=_GH,
         homepage="https://github.com/RBniCS/RBniCS", language="Python", integration="candidate",
         notes="Certified reduced basis over FEniCS; the mathematically rigorous branch of ROM."),
    dict(id="ezyrb", name="EZyRB", stages=("surrogates",), kind="reduced order modelling",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://github.com/mathLab/EZyRB", language="Python", integration="candidate",
         notes="POD plus an interpolator (RBF, GPR, ANN) — the smallest useful ROM, and easy to audit."),
    dict(id="scikit_learn", name="scikit-learn", stages=("surrogates", "data"), kind="classical ML",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://scikit-learn.org/", language="Python", integration="wired",
         surface="pinneapple_models",
         notes="The baseline every surrogate should be compared against before a neural one is trained."),
    dict(id="gpytorch", name="GPyTorch", stages=("surrogates", "validation"), kind="Gaussian processes",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://gpytorch.ai/", language="Python", integration="candidate",
         notes="A GP gives a variance with the prediction, which is the cheapest honest UQ a surrogate can have."),
    dict(id="gpy", name="GPy", stages=("surrogates",), kind="Gaussian processes",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://github.com/SheffieldML/GPy", language="Python", integration="candidate",
         notes="Older, sparser GP toolbox; still the shortest path to a heteroscedastic surrogate."),
    dict(id="smt", name="SMT", stages=("surrogates", "optimization"), kind="surrogate modelling toolbox",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://smt.readthedocs.io/", language="Python", integration="candidate",
         notes="Kriging, RBF and gradient-enhanced surrogates with sampling built in — the DOE-to-surrogate "
               "loop in one dependency."),
    dict(id="openturns", name="OpenTURNS", stages=("surrogates", "validation", "inverse"), kind="UQ platform",
         license="LGPL-3.0", license_verified=True, license_source=_GH,
         homepage="https://openturns.github.io/www/", language="C++/Python", integration="named",
         surface="pinneapple_analysis.verification.tool_recommendation",
         notes="Sensitivity, reliability and calibration in one library; the reference implementation of Sobol "
               "indices."),

    # ── data ──────────────────────────────────────────────────────────────────────────────────────────────
    dict(id="h5py", name="h5py / HDF5", stages=("data",), kind="array storage",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://www.h5py.org/", language="Python/C", integration="wired",
         surface="pinneapple_data.datasets.remote_hdf5",
         notes="Every PDE benchmark archive is HDF5; the seekable-file driver is what lets a 1.3 GB file be read "
               "for 65 kB of it."),
    dict(id="zarr", name="Zarr", stages=("data",), kind="chunked array storage",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://zarr.dev/", language="Python", integration="wired",
         surface="pinneapple_data.upd_types",
         notes="Cloud-native chunked arrays; The Well publishes in this format as well as HDF5."),
    dict(id="xarray", name="xarray", stages=("data",), kind="labelled arrays",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://xarray.dev/", language="Python", integration="wired",
         surface="pinneapple_data.schema",
         notes="A bundle becomes a Dataset with units in the attributes; the interop target for the rest of the "
               "scientific Python stack."),
    dict(id="apache_arrow", name="Apache Arrow / Parquet", stages=("data",), kind="columnar storage",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://arrow.apache.org/", language="C++/Python", integration="wired",
         surface="pinneapple_data.unified",
         notes="Columnar export for tables of results; also the transport under several in-memory exchange "
               "protocols."),
    dict(id="adios2", name="ADIOS2", stages=("data",), kind="streaming I/O",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://adios2.readthedocs.io/", language="C++/Python", integration="candidate",
         notes="In-situ and staged I/O for HPC runs: the format a solver writes while it is still running."),
    dict(id="dvc", name="DVC", stages=("data",), kind="data versioning",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://dvc.org/", language="Python", integration="candidate",
         notes="Content-addressed data versions beside git — the same idea as the bundle digest, for the raw "
               "archive."),
    dict(id="mlflow", name="MLflow", stages=("data", "validation"), kind="experiment tracking",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://mlflow.org/", language="Python", integration="candidate",
         notes="Experiment lineage across runs; this library records lineage in its own manifests (INT: experiment "
               "manifests), so the two would have to be reconciled, not duplicated."),
    dict(id="salib", name="SALib", stages=("data", "validation"), kind="sensitivity analysis",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://salib.readthedocs.io/", language="Python", integration="candidate",
         notes="Sobol/Morris screening: which input actually moves the output, before any surrogate is trained."),
    dict(id="chaospy", name="Chaospy", stages=("data", "validation"), kind="polynomial chaos",
         license="MIT", license_verified=True, license_source=_PYPI,
         homepage="https://chaospy.readthedocs.io/", language="Python", integration="candidate",
         notes="Polynomial chaos expansions and quasi-random sampling — UQ that needs far fewer runs than Monte "
               "Carlo."),
    dict(id="pydoe2", name="pyDOE2", stages=("data",), kind="design of experiments",
         license="BSD-3-Clause", license_verified=True, license_source=_PYPI,
         homepage="https://github.com/clicumu/pyDOE2", language="Python", integration="named",
         surface="pinneapple_arena.bundle_uq",
         notes="Latin hypercube and factorial designs; this library has its own LHS, which must keep its "
               "per-column permutations."),

    # ── inverse ───────────────────────────────────────────────────────────────────────────────────────────
    dict(id="pymc", name="PyMC", stages=("inverse", "validation"), kind="Bayesian inference",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://www.pymc.io/", language="Python", integration="candidate",
         notes="A posterior over parameters with the sampler and diagnostics in one place; the tool that would "
               "turn the seed-ensemble 'disagreement' into an uncertainty."),
    dict(id="emcee", name="emcee", stages=("inverse",), kind="MCMC",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://emcee.readthedocs.io/", language="Python", integration="candidate",
         notes="Affine-invariant ensemble sampler: gradient-free, which suits a black-box solver."),
    dict(id="dynesty", name="dynesty", stages=("inverse",), kind="nested sampling",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://dynesty.readthedocs.io/", language="Python", integration="candidate",
         notes="Nested sampling gives the evidence, so two models can be compared rather than only fitted."),
    dict(id="ultranest", name="UltraNest", stages=("inverse",), kind="nested sampling",
         license="GPL-3.0", license_verified=True, license_source=_GH,
         homepage="https://johannesbuchner.github.io/UltraNest/", language="Python", integration="candidate",
         notes="Reactive nested sampling for expensive likelihoods; GPL-3.0 again is a product decision."),
    dict(id="dolfin_adjoint", name="dolfin-adjoint", stages=("inverse", "optimization"), kind="adjoint sensitivities",
         license="LGPL-3.0", license_verified=True, license_source=_GH,
         homepage="https://www.dolfin-adjoint.org/", language="Python", integration="candidate",
         notes="Turns a forward FEniCS solve into a gradient for the inverse problem; the disciplined alternative "
               "to finite-difference sensitivities."),
    dict(id="lmfit", name="lmfit", stages=("inverse",), kind="parameter fitting",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://lmfit.github.io/lmfit-py/", language="Python", integration="candidate",
         notes="Least squares with bounds and parameter expressions — the small inverse problem done honestly."),

    # ── validation ───────────────────────────────────────────────────────────────────────────────────────
    dict(id="pint", name="Pint", stages=("validation", "data"), kind="units",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://pint.readthedocs.io/", language="Python", integration="wired",
         surface="pinneapple_data.schema", notes="The dimensional check behind the bundle's `units` gate; a unit "
               "that does not parse fails the gate."),
    dict(id="arviz", name="ArviZ", stages=("validation", "inverse"), kind="posterior diagnostics",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://python.arviz.org/", language="Python", integration="candidate",
         notes="R-hat, ESS and posterior predictive checks — what makes a claimed uncertainty auditable."),

    # ── optimization ─────────────────────────────────────────────────────────────────────────────────────
    dict(id="nlopt", name="NLopt", stages=("optimization",), kind="nonlinear optimisation",
         license="MIT", license_verified=True, license_source=_PYPI,
         homepage="https://nlopt.readthedocs.io/", language="C/Python", integration="candidate",
         notes="A dozen algorithms behind one interface, gradient-free ones included; relicensed to MIT in 2.8."),
    dict(id="pymoo", name="pymoo", stages=("optimization",), kind="multi-objective optimisation",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://pymoo.org/", language="Python", integration="wired",
         surface="pinneapple_design.design_optimizer", notes="NSGA-II and friends: the Pareto front a design "
               "study reports."),
    dict(id="deap", name="DEAP", stages=("optimization",), kind="evolutionary algorithms",
         license="LGPL-3.0", license_verified=True, license_source=_GH,
         homepage="https://github.com/DEAP/deap", language="Python", integration="candidate",
         notes="Genetic programming as well as genetic algorithms; the toolbox for a non-smooth objective."),
    dict(id="nevergrad", name="Nevergrad", stages=("optimization",), kind="derivative-free optimisation",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://github.com/facebookresearch/nevergrad", language="Python", integration="candidate",
         notes="Many derivative-free optimisers with a budget-aware interface — the right family when one "
               "evaluation is a solver run."),
    dict(id="pycma", name="pycma", stages=("optimization",), kind="CMA-ES",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://github.com/CMA-ES/pycma", language="Python", integration="candidate",
         notes="The reference CMA-ES implementation; strong on ill-conditioned, noisy objectives."),
    dict(id="casadi", name="CasADi", stages=("optimization", "twin"), kind="algorithmic differentiation + NLP",
         license="LGPL-3.0", license_verified=True, license_source=_GH,
         homepage="https://web.casadi.org/", language="C++/Python", integration="wired",
         surface="pinneapple_arena.external.models_data",
         notes="Symbolic differentiation and optimal control; the usual engine under an MPC loop."),
    dict(id="pyomo", name="Pyomo", stages=("optimization",), kind="optimisation modelling",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://www.pyomo.org/", language="Python", integration="candidate",
         notes="Algebraic modelling for the mixed-integer problems a flowsheet or a schedule becomes."),
    dict(id="ipopt", name="Ipopt", stages=("optimization",), kind="interior-point NLP",
         license="EPL-2.0", license_verified=True, license_source=_GH,
         homepage="https://github.com/coin-or/Ipopt", language="C++", integration="candidate",
         notes="The workhorse for large smooth NLPs; EPL-2.0 is weak copyleft, so linking is usually fine."),
    dict(id="highs", name="HiGHS", stages=("optimization",), kind="LP/MIP solver",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://highs.dev/", language="C++", integration="candidate",
         notes="Fast open LP/MIP; the open answer where a commercial solver's licence would otherwise decide."),
    dict(id="cvxpy", name="CVXPY", stages=("optimization",), kind="convex optimisation",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://www.cvxpy.org/", language="Python", integration="candidate",
         notes="A convex problem stated as mathematics and dispatched to a solver."),
    dict(id="scipy", name="SciPy", stages=("optimization", "solvers", "data"), kind="scientific computing",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://scipy.org/", language="Python/C/Fortran", integration="wired",
         surface="pinneapple_simulation.numerical_solvers",
         notes="The baseline under nearly everything else here: `optimize.minimize`, sparse linear algebra, "
               "and the FFT the spectral solvers use."),

    # ── twin ─────────────────────────────────────────────────────────────────────────────────────────────
    dict(id="pykalman", name="pykalman", stages=("twin",), kind="Kalman filtering",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://github.com/pykalman/pykalman", language="Python", integration="candidate",
         notes="The textbook state estimator a twin needs to fuse a sensor reading with a model state."),
    dict(id="filterpy", name="FilterPy", stages=("twin",), kind="Kalman/particle filters",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://github.com/rlabbe/filterpy", language="Python", integration="candidate",
         notes="Kalman, EKF, UKF and particle filters with the book that explains them."),
    dict(id="pyro", name="Pyro", stages=("twin", "inverse"), kind="probabilistic programming",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://pyro.ai/", language="Python", integration="candidate",
         notes="Variational inference on a PyTorch model — the Bayesian layer a neural surrogate can carry."),
    dict(id="mqtt", name="Eclipse Mosquitto (MQTT)", stages=("twin",), kind="sensor messaging",
         license="EPL-2.0 OR EDL-1.0", license_verified=True, license_source=_GH,
         homepage="https://mosquitto.org/", language="C", integration="candidate",
         notes="The protocol most plant sensors speak; the transport a twin's telemetry would use instead of "
               "polling."),
    dict(id="kafka", name="Apache Kafka", stages=("twin",), kind="event streaming",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://kafka.apache.org/", language="Java/Scala", integration="candidate",
         notes="Durable, replayable streams; the difference from a queue is the replay, which is what a twin's "
               "history needs."),
    dict(id="influxdb", name="InfluxDB (open-source core)", stages=("twin",), kind="time-series database",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://github.com/influxdata/influxdb", language="Go/Rust", integration="candidate",
         notes="Downsampling and retention for sensor history; the clustering and several features are under a "
               "separate proprietary licence, so read which edition is deployed."),
    dict(id="airflow", name="Apache Airflow", stages=("twin", "data"), kind="orchestration",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://airflow.apache.org/", language="Python", integration="candidate",
         notes="Schedules the periodical job (recalibrate, retrain, sweep) that a twin needs to stay current."),
    dict(id="prefect", name="Prefect", stages=("twin", "data"), kind="orchestration",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://www.prefect.io/", language="Python", integration="candidate",
         notes="The lighter-weight orchestration alternative when Airflow's scheduler is more machinery than the "
               "job needs."),

    # ── viz ──────────────────────────────────────────────────────────────────────────────────────────────
    dict(id="pyvista", name="PyVista", stages=("viz", "export"), kind="3-D visualisation",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://pyvista.org/", language="Python", integration="wired",
         surface="pinneapple_data.schema.view",
         notes="A bundle becomes a StructuredGrid or UnstructuredGrid and an animation a .pvd/.vtu pair "
               "ParaView opens."),
    dict(id="vtk", name="VTK", stages=("viz", "export", "data"), kind="visualisation toolkit",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://vtk.org/", language="C++/Python", integration="wired",
         surface="pinneapple_data.schema.view",
         notes="The file formats (.vtu/.vts/.pvd) that make a result readable outside this library."),
    dict(id="paraview", name="ParaView", stages=("viz",), kind="post-processing",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://www.paraview.org/", language="C++/Python",
         notes="The viewer an engineer already has; being readable by it is the requirement, not a feature."),
    dict(id="matplotlib", name="Matplotlib", stages=("viz", "export"), kind="2-D plotting",
         license="Matplotlib (PSF-based, no SPDX id)", license_verified=True,
         license_source="read 2026-11-25 at raw.githubusercontent.com/matplotlib/matplotlib/main/LICENSE/LICENSE: "
                        "the Matplotlib Development Team's own agreement for versions 1.3.0 and later, plus a "
                        "separate one for the versions before it",
         homepage="https://matplotlib.org/", language="Python", integration="wired",
         surface="pinneapple_lab",
         notes="The figure in every experiment manifest. Permissive and business-friendly, but not the BSD licence "
               "this row used to call it: use, modification and distribution are granted provided the agreement "
               "and the MDT copyright notice are kept, and a derivative work carries a brief summary of the "
               "changes made."),
    dict(id="plotly", name="Plotly", stages=("viz",), kind="interactive charts",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://plotly.com/python/", language="Python/JavaScript", integration="candidate",
         notes="Interactive figures for a browser report, where a static PNG is not enough."),
    dict(id="vedo", name="vedo", stages=("viz",), kind="3-D visualisation",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://vedo.embl.es/", language="Python", integration="candidate",
         notes="VTK with a scientific-scripting feel; good for quick mesh inspection in a notebook."),
    dict(id="open3d", name="Open3D", stages=("viz", "problem"), kind="3-D data processing",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://www.open3d.org/", language="C++/Python", integration="candidate",
         notes="Registration, reconstruction and point-cloud processing — the geometry side of a scan-based "
               "case."),
    dict(id="yt", name="yt", stages=("viz",), kind="volumetric analysis",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://yt-project.org/", language="Python", integration="candidate",
         notes="Fields and slices from an AMR or particle dataset; built for simulation output that is not a "
               "structured grid."),
    dict(id="napari", name="napari", stages=("viz",), kind="image viewer",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://napari.org/", language="Python", integration="candidate",
         notes="n-dimensional viewer; the practical way to look at a 3-D field slice by slice."),
    dict(id="blender", name="Blender", stages=("viz", "export"), kind="rendering",
         license="GPL-2.0-or-later", license_verified=True,
         license_source="read 2026-11-25 at github.com/blender/blender: COPYING points at "
                        "doc/license/GPL-license.txt (GNU GPL version 2), and the source headers carry "
                        "'SPDX-License-Identifier: GPL-2.0-or-later'",
         homepage="https://www.blender.org/", language="C++/Python", integration="wired",
         surface="pinneapple_blender",
         notes="Rendered stills and clips for a report or a deck. The GPL covers the application and any add-on "
               "that ships inside it, not the image it renders, so a render is not a derived work and can be sold "
               "or published freely."),
    dict(id="f3d", name="F3D", stages=("viz",), kind="fast 3-D viewer",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://f3d.app/", language="C++", integration="candidate",
         notes="Opens a mesh or a VTU in a fraction of a second, without a Python session."),

    # ── export ───────────────────────────────────────────────────────────────────────────────────────────
    dict(id="onnx", name="ONNX", stages=("export",), kind="model interchange",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://onnx.ai/", language="C++/Python", integration="wired",
         surface="pinneapple_arena.serving_bench",
         notes="The graph a trained model leaves as; an FNO's complex spectral layers do not lower to it without "
               "a rewrite."),
    dict(id="onnxruntime", name="ONNX Runtime", stages=("export",), kind="inference runtime",
         license="MIT", license_verified=True, license_source=_PYPI,
         homepage="https://onnxruntime.ai/", language="C++/Python", integration="wired",
         surface="pinneapple_arena.serving_bench",
         notes="Timed against eager PyTorch on the same input, with the max absolute difference reported as the "
               "smoke check."),
    dict(id="openvino", name="OpenVINO", stages=("export",), kind="inference runtime",
         license="Apache-2.0", license_verified=True, license_source=_GH,
         homepage="https://docs.openvino.ai/", language="C++/Python", integration="candidate",
         notes="Intel CPU/iGPU inference; the deployment target that a CUDA-only stack misses."),
    dict(id="triton", name="NVIDIA Triton Inference Server", stages=("export", "twin"), kind="model serving",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://github.com/triton-inference-server/server", language="C++", integration="candidate",
         notes="Multi-model serving with batching and versioning — what a surrogate needs once it is a service."),
    dict(id="openusd", name="OpenUSD", stages=("export", "viz"), kind="scene interchange",
         license="TOST-1.0 (Tomorrow Open Source Technology License)", license_verified=True, license_source=_GH,
         homepage="https://openusd.org/", language="C++/Python", integration="named",
         surface="pinneapple_hub",
         notes="The licence is a modified Apache-2.0 and is not an SPDX id anyone should assume; a scene graph "
               "for downstream 3-D tools."),
    dict(id="cgnslib", name="CGNS", stages=("export", "data"), kind="CFD data standard",
         license="Zlib-style (CGNS licence, derived from zlib/libpng)", license_verified=True,
         license_source="read 2026-11-25 at raw.githubusercontent.com/CGNS/CGNS/develop/license.txt: use for any "
                        "purpose, commercial included, with altered versions marked and the notice kept; the text "
                        "says it is borrowed from the zlib/libpng licence and supersedes the LGPL",
         homepage="https://cgns.github.io/", language="C/Fortran", integration="candidate",
         notes="The standard CFD exchange format, and what NASA's verification cases are published in. Permissive "
               "with no copyleft, so it can be shipped inside a product provided the notice survives and changes "
               "to the library are marked."),
    dict(id="python_docx", name="python-docx", stages=("export",), kind="word documents",
         license="MIT", license_verified=True, license_source=_GH,
         homepage="https://github.com/python-openxml/python-docx", language="Python", integration="candidate",
         notes="A report a reviewer can annotate; the format an engineering sign-off usually takes."),
    dict(id="weasyprint", name="WeasyPrint", stages=("export",), kind="HTML to PDF",
         license="BSD-3-Clause", license_verified=True, license_source=_GH,
         homepage="https://weasyprint.org/", language="Python", integration="candidate",
         notes="HTML plus CSS becomes a paginated PDF — the report path that keeps the source diffable."),
    dict(id="pandoc", name="Pandoc", stages=("export",), kind="document conversion",
         license="GPL-2.0", license_verified=True, license_source=_GH,
         homepage="https://pandoc.org/", language="Haskell", integration="candidate",
         notes="Markdown to anything; GPL-2.0 matters only if it is linked, not if it is invoked."),
    dict(id="xlsxwriter", name="XlsxWriter", stages=("export",), kind="spreadsheets",
         license="BSD-2-Clause", license_verified=True, license_source=_GH,
         homepage="https://xlsxwriter.readthedocs.io/", language="Python", integration="candidate",
         notes="The results table a client actually opens; charts included."),
)

TOOLS: dict[str, Tool] = {}
for _raw in _RAW:
    _tool = _t(**_raw)
    for _stage_id in _tool.stages:
        if _stage_id not in STAGES:
            raise ValueError(f"tool '{_tool.id}' names stage '{_stage_id}', which is not one of {STAGE_IDS}")
    TOOLS[_tool.id] = _tool
del _raw, _tool, _stage_id


@dataclass(frozen=True)
class StageTools:
    """One stage of the map: what the outside world offers here, and what this repository already reaches."""

    stage: Stage
    tools: tuple[Tool, ...]
    datasets: tuple[dict[str, Any], ...]

    @property
    def wired(self) -> tuple[Tool, ...]:
        return tuple(t for t in self.tools if t.integration == "wired")

    @property
    def unverified(self) -> tuple[Tool, ...]:
        """Tools whose licence this checkout did not read — the entries a delivery decision cannot rely on yet."""
        return tuple(t for t in self.tools if not t.license_verified)

    def to_dict(self) -> dict[str, Any]:
        return {**self.stage.to_dict(), "tools": [t.to_dict() for t in self.tools],
                "datasets": list(self.datasets),
                "counts": {"tools": len(self.tools), "wired": len(self.wired),
                           "candidate": sum(1 for t in self.tools if t.integration == "candidate"),
                           "license_unverified": len(self.unverified), "datasets": len(self.datasets)}}


def stages() -> list[Stage]:
    """Every stage, in workflow order."""
    return [STAGES[s] for s in STAGE_IDS]


def stage(stage_id: str) -> Stage:
    try:
        return STAGES[stage_id]
    except KeyError:
        raise KeyError(f"'{stage_id}' is not a stage; known: {list(STAGE_IDS)}") from None


def tools(*, stage_id: str | None = None, integration: str | None = None, kind: str | None = None) -> list[Tool]:
    """The catalogued tools, filtered; sorted by name inside the stage order they serve.

    ``integration="wired"`` is the list of what this repository actually calls, which is the number that makes the
    rest of the map meaningful: everything else is a decision still open.
    """
    if stage_id is not None:
        stage(stage_id)                                  # raises for a typo instead of returning an empty list
    if integration is not None and integration not in INTEGRATIONS:
        raise ValueError(f"integration must be one of {INTEGRATIONS}, not {integration!r}")
    out = [t for t in TOOLS.values()
           if (stage_id is None or stage_id in t.stages)
           and (integration is None or t.integration == integration)
           and (kind is None or t.kind == kind)]
    order = {s: i for i, s in enumerate(STAGE_IDS)}
    return sorted(out, key=lambda t: (min(order[s] for s in t.stages), t.name))


def datasets(stage_id: str | None = None) -> list[dict[str, Any]]:
    """The public datasets that feed a stage, projected from ``pinneapple_data.datasets.catalog``.

    A dataset is catalogued once and tagged with the stages it feeds (``CatalogEntry.stages``), so this is a filter
    over that catalogue rather than a second copy of it. Returns ``[]`` when the dataset catalogue is not importable
    (a partial install): the tool map is still useful without it, and the API reports the missing half separately.
    """
    if stage_id is not None:
        stage(stage_id)                                  # raises for a typo instead of returning an empty list
    try:
        from pinneapple_data.datasets.catalog import entries
    except Exception:                                    # pragma: no cover - only on a partial install
        return []
    return [e.to_dict() for e in entries(stage=stage_id)]


def unassigned_datasets() -> list[dict[str, Any]]:
    """Catalogued datasets that no stage claims — invisible from a stage's point of view, so named here."""
    return [d for d in datasets() if not d.get("stages")]


def coverage(stage_id: str | None = None) -> dict[str, Any]:
    """How far each stage is covered: tools here, tools wired here, and the datasets that feed it.

    With ``stage_id=`` it returns that stage's counts; without it, the totals plus the per-stage counts, which is
    what a reviewer reads before deciding where the next integration should go.
    """
    if stage_id is not None:
        st = stage(stage_id)
        row = StageTools(stage=st, tools=tuple(tools(stage_id=stage_id)), datasets=tuple(datasets(stage_id)))
        return {"stage": st.id, **row.to_dict()["counts"]}
    per = {s: coverage(s) for s in STAGE_IDS}
    total = {
        "stages": len(STAGE_IDS),
        "tools": len(TOOLS),
        "wired": sum(1 for t in TOOLS.values() if t.integration == "wired"),
        "named": sum(1 for t in TOOLS.values() if t.integration == "named"),
        "candidate": sum(1 for t in TOOLS.values() if t.integration == "candidate"),
        "license_unverified": sum(1 for t in TOOLS.values() if not t.license_verified),
        "datasets": len(datasets()),
    }
    return {"totals": total, "stages": per}


def get(tool_id: str) -> Tool:
    try:
        return TOOLS[tool_id]
    except KeyError:
        raise KeyError(f"'{tool_id}' is not catalogued; known: {sorted(TOOLS)}") from None


def to_json(stage_id: str | None = None) -> dict[str, Any]:
    """The map as JSON-ready dicts — the payload the Engineering API serves.

    With ``stage_id=`` one stage; without it, every stage in workflow order plus the totals, so a reader can see
    which step is the thin one without calling thirteen times.
    """
    if stage_id is not None:
        st = stage(stage_id)
        return StageTools(stage=st, tools=tuple(tools(stage_id=stage_id)),
                          datasets=tuple(datasets(stage_id))).to_dict()
    return {
        "stages": [StageTools(stage=STAGES[s], tools=tuple(tools(stage_id=s)),
                              datasets=tuple(datasets(s))).to_dict() for s in STAGE_IDS],
        "coverage": coverage(),
        "unassigned_datasets": [d["id"] for d in unassigned_datasets()],
    }
