"""The simulation factory: one case in, one traceable dataset out — CAD -> DOE -> solver -> dataset -> QC (#152).

The CFD product ran this pipeline for pipe erosion only, hand-wired per case. Generalized, it is four interfaces
(:mod:`.interfaces`) with quality control over each of them (:mod:`.qc`), run once per case (:func:`.factory.run_factory`):

    from pinneapple_autolab.factory import TemplateGeometry, MaximinDoe, run_factory
    run = run_factory(spec, simulator, out="runs/bend", geometry=TemplateGeometry("pipe_bend_90"),
                      doe=MaximinDoe())
    run.status, run.decision, run.dataset["json"]     # ('done', 'APPROVED', 'runs/bend/dataset.json')

What it reuses instead of re-implementing, which is the point of generalizing rather than copying:

* the **spec** — ``pinneapple_analysis.design_study.DesignStudy``: the geometry base with its variables, the
  evaluator with its input mapping and the outputs *with units*, the budget and the seed;
* the **DOE** — ``DesignSpace.sample`` (the Latin hypercube ``run_study`` draws) and
  ``pinneapple_autolab.study.campaign.plan_doe`` (space-filling, with its coverage metrics);
* the **solver** — ``pinneapple_autolab.simulator.Simulator``, unchanged: ``check`` before ``run``, a failure with
  its reason, the cost, the fields;
* the **dataset** — ``pinneapple_autolab.study.dataset`` (``StudyDataset``, ``build_dataset``) and
  ``pinneapple_data.contracts.DatasetSpec``, so a factory dataset is read by everything downstream;
* the **evidence** — ``pinneapple_analysis.trust.Check`` / ``TrustReport`` and
  ``pinneapple_data.provenance`` (solver, version, seed, data hash, commit, mesh), so a row that cannot be traced
  back is refused rather than written.

Where the chain stops, it stops out loud: a case whose solver the library does not have passes the reason to
``run_factory(missing=[...])`` and comes back ``refused``, and a geometry whose CAD kernel is not installed keeps
its solid checks ``not_run`` and says so. See ``pinneapple_autolab.factory.domains.pipe_erosion`` for the case that
does exactly that today.
"""
from __future__ import annotations

from . import domains, qc
from .factory import FactoryRow, FactoryRun, Status, default_geometry, run_factory
from .interfaces import (
    BUILT,
    PARAMETERS_ONLY,
    REFUSED,
    AsIsGeometry,
    DatasetRef,
    DatasetWriter,
    Doe,
    Geometry,
    GeometryArtifact,
    GivenDesigns,
    LatinHypercubeDoe,
    MaximinDoe,
    ProvenanceError,
    StudyDatasetWriter,
    TemplateGeometry,
    design_space,
    template_names,
)
from .qc import KEPT, LIMITS, REJECTED, FactoryQc, row_verdict

__all__ = [
    "run_factory", "FactoryRun", "FactoryRow", "Status", "default_geometry",
    "Geometry", "GeometryArtifact", "TemplateGeometry", "AsIsGeometry", "BUILT", "PARAMETERS_ONLY", "REFUSED",
    "Doe", "LatinHypercubeDoe", "MaximinDoe", "GivenDesigns", "design_space", "template_names",
    "DatasetWriter", "DatasetRef", "StudyDatasetWriter", "ProvenanceError",
    "FactoryQc", "LIMITS", "KEPT", "REJECTED", "row_verdict", "qc", "domains",
]
