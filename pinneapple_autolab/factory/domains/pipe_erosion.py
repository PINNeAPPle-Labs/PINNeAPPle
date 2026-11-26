"""The pipe-erosion case of the CFD product, as far as this library can carry it (#152, CFD-04).

The product's pipeline runs this case today, in ``PINNeAPPle-CFD``: an elbow of a pneumatic conveying line, geometry
from a CadQuery template, a campaign of OpenFOAM runs over diameter and bend ratio, the erosion rate post-processed
with DNV-RP-O501, a dataset of ``delta_p`` and the peak erosion rate, and one validation against the ultrasonic
thickness measurements of Vieira et al. (2016) on a 76.2 mm, R/D = 1.5 elbow.

What is real here, and what is not, measured rather than assumed:

* **the geometry** — :class:`~pinneapple_autolab.factory.TemplateGeometry` on ``pipe_bend_90`` of
  ``pinneapple_design.fitmesh``: the parameters with their published validity limits, the derived development
  lengths (>= 10 D before the bend, >= 5 D after), the exact volume by Pappus, the smallest feature that drives the
  mesh, the reference length of the Reynolds number, the inlet/outlet patches and a point inside the fluid. The
  solid itself needs the optional ``cadquery`` extra; without it the checks that need a solid are ``not_run`` and
  the artifact says why;
* **the DOE** — :func:`plan` draws the designs from the study's own budget, space-filling, with its coverage
  metrics. It runs today;
* **the solver** — **missing**: the library has no erosion solver. ``delta_p`` and the erosion rate come from the
  OpenFOAM case plus the DNV-RP-O501 post-processing of ``PINNeAPPle-CFD`` (``etapas/e04_metodos_solvers/erosao.py``
  and ``etapas/e06_dataset``), which is a separate repository, not a package of this one. :func:`run` therefore
  returns a run refused at that step with :data:`SOLVER_MISSING` as the reason, and no number is invented to fill
  the hole.

So the acceptance of #152 — "the pipe-erosion case runs through the generic factory with the same results as the
current pipeline" — is met up to the solver and not through it: what the factory produces for this case today is the
geometry and the DOE the pipeline needs, wired to the same interfaces every other case uses. Porting the erosion
solver into a package of this library (with its coefficient provenance read again, DNV-RP-O501 rev. 4.2 Tab. 7-2)
closes the last arrow; nothing else does.
"""
from __future__ import annotations

from typing import Any

from pinneapple_analysis.design_study import (
    Base,
    Budget,
    DesignStudy,
    Evaluator,
    Objective,
    Quantity,
    Variable,
)
from pinneapple_autolab.factory.factory import FactoryRun, run_factory
from pinneapple_autolab.factory.interfaces import MaximinDoe, TemplateGeometry

__all__ = ["TEMPLATE", "REF", "SOLVER_MISSING", "CONDITIONS", "spec", "geometry", "doe", "plan", "run", "REFERENCE"]

#: The template of ``pinneapple_design.fitmesh`` this case is built on, and the reference the study names.
TEMPLATE, REF = "pipe_bend_90", "pp.geometries:pipe_bend_90"

#: The operating conditions of the case, fixed by the product: air at 20 m/s carrying sand.
CONDITIONS = {"speed_m_s": 20.0}

#: The reference object of the template: a 76.2 mm, R/D = 1.5 elbow measured by ultrasonic thickness.
REFERENCE = "Vieira et al., Powder Technology 288:339-353 (2016)"

#: Why the last arrow of this case is not here. Passed to :func:`run_factory` as ``missing``, so the run says it.
SOLVER_MISSING = (
    "no erosion solver in this library: delta_p and the peak erosion rate of this case come from the OpenFOAM case "
    "plus the DNV-RP-O501 post-processing of the CFD product (PINNeAPPle-CFD, etapas/e04_metodos_solvers/erosao.py "
    "and etapas/e06_dataset), which is a separate repository and not a package of PINNeAPPle. The factory carries "
    "the geometry and the DOE of the case; until that solver is a package here, the chain stops at the solver and "
    "no output is invented for it")


def spec() -> DesignStudy:
    """The case as a study spec: two variables, two outputs with their units, and the reference design.

    ``delta_p`` in Pa and the peak erosion rate in mm/year are the two quantities the product's dataset carries
    (``PINNeAPPle-CFD`` ``pipeline/jornada.py``, ``UNIDADE_KPI``); the baseline is the reference object of the
    template, so a run of this case is comparable with the published measurements without rescaling anything.
    """
    return DesignStudy(
        kind="surrogate", title="Pipe bend under sand erosion: pressure drop and wall loss",
        statement="Vary the diameter and the bend ratio of a 90-degree elbow of a pneumatic line to learn the "
                  "pressure drop and the peak erosion rate, at the reference object of the template.",
        base=Base(kind="geometry", ref=REF,
                  variables=[Variable("diameter", 0.025, 0.5, unit="m",
                                      description="internal diameter of the pipe"),
                             Variable("bend_ratio", 1.0, 5.0, unit="R/D",
                                      description="radius of the bend over the diameter")],
                  fixed={}, baseline={"diameter": 0.0762, "bend_ratio": 1.5},
                  description="90-degree elbow with straight inlet and outlet runs"),
        evaluator=Evaluator(tool="openfoam.erosion_bend",
                            inputs={"diameter": "diameter", "bend_ratio": "bend_ratio"},
                            outputs=[Quantity("delta_p", "Pa", label="pressure drop"),
                                     Quantity("erosion_rate_max", "mm/year", label="peak wall loss rate")],
                            conditions=dict(CONDITIONS), fidelity="reference",
                            description="OpenFOAM steady RANS of the bend + DNV-RP-O501 erosion post-processing"),
        objectives=[Objective("delta_p", "min", label="pressure drop"),
                    Objective("erosion_rate_max", "min", label="peak wall loss rate")],
        budget=Budget(max_evaluations=8, n_doe=7, n_verify=0, surrogate_evaluations=200, seed=0),
        metadata={"case": "CFD-04", "product": "PINNeAPPle-CFD", "reference": REFERENCE})


def geometry(**kw: Any) -> TemplateGeometry:
    """The geometry end of the case: the ``pipe_bend_90`` template of ``pinneapple_design.fitmesh``."""
    return TemplateGeometry(TEMPLATE, **kw)


def doe() -> MaximinDoe:
    """The DOE end of the case: the space-filling plan of ``pinneapple_autolab.study.campaign.plan_doe``."""
    return MaximinDoe()


def plan(seed: int | None = None) -> dict[str, Any]:
    """The two steps of this case that run today: the geometry template and the designs the budget pays for.

    Nothing is simulated here and nothing is claimed about pressure drop or erosion: this is what the factory knows
    before it calls a solver, which is exactly what a solver needs.
    """
    s = spec()
    geo = geometry()
    designs = doe().plan(s.base.variables, s.budget.planned(len(s.base.names))["doe"],
                         seed=s.budget.seed if seed is None else int(seed), existing=[s.base.baseline_design()])
    first = geo.prepare(s.base.baseline_design(), "")
    return {"geometry": geo.describe(), "baseline": s.base.baseline_design(), "artifact": first.to_dict(),
            "designs": designs, "n_designs": len(designs), "solver_missing": SOLVER_MISSING}


def run(out: str, **kw: Any) -> FactoryRun:
    """Run the case through the factory. Refused at the solver, with :data:`SOLVER_MISSING` as the reason."""
    return run_factory(spec(), None, out=out, geometry=geometry(), doe=doe(), missing=[SOLVER_MISSING], **kw)
