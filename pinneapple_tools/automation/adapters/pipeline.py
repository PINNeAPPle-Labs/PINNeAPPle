"""Adapters of the FreeCAD -> OpenFOAM pipeline (:mod:`pinneapple_simulation.pipelines.freecad_openfoam`):

* ``freecadcmd.export``: FreeCAD **headless** (``freecadcmd``, no display) sets the document's spreadsheet
  parameters, recomputes, exports STEP and the faces; the patches are named by rule and written to a multi-region
  STL. ``backend: fake`` uses the NumPy stand-in (tests, machines without FreeCAD).

and, on the ``openfoam`` adapter (:mod:`.openfoam`), through :func:`pipeline_call`:

* ``openfoam.prepare``: mesh dictionaries and case (internal: blockMesh + snappyHexMesh / cfMesh, simpleFoam /
  pimpleFoam; external: ``ExternalFlow``), with ``Allmesh`` and ``Allrun`` for ``openfoam.mesh`` / ``openfoam.solve``.
* ``openfoam.post``: measured quantities, figures, derived quantities and reference errors.

The work is done by :mod:`pinneapple_simulation.pipelines.freecad_openfoam.steps`; these classes only turn its
results into an :class:`ActionResult`.
"""
from __future__ import annotations

from typing import Any

from ..record import Check, Evidence
from .base import ActionFailed, ActionResult, Adapter, StepContext, register

__all__ = ["FreeCADHeadlessAdapter", "pipeline_call"]


def _result(d: dict[str, Any], ctx: StepContext) -> ActionResult:
    res = ActionResult(outputs=d["outputs"], artifacts=list(d["artifacts"]), actions=list(d["actions"]))
    for name, passed, detail, value, threshold, source in d["checks"]:
        res.checks.append(Check(name, passed, detail, value, threshold, source=source))
    for path, kind, caption in d["figures"]:
        res.evidence.append(Evidence(ctx.rel(path), kind, caption))
    return res


def pipeline_call(fn, params: dict[str, Any], ctx: StepContext) -> ActionResult:
    try:
        return _result(fn(params, ctx.step_dir), ctx)
    except (ValueError, KeyError, RuntimeError, FileNotFoundError) as e:
        raise ActionFailed(f"{type(e).__name__}: {e}") from e


class FreeCADHeadlessAdapter(Adapter):
    name = "freecadcmd"
    mode = "api"
    actions = ("export",)

    def available(self, toolchain) -> tuple[bool, str]:
        return True, "freecadcmd, or the NumPy stand-in with backend: fake"   # the step's requires decides

    def do_export(self, params: dict[str, Any], ctx: StepContext) -> ActionResult:
        from pinneapple_simulation.pipelines.freecad_openfoam.steps import cad_step
        ctx.progress("FreeCAD: recompute and export")
        return pipeline_call(cad_step, params, ctx)


register(FreeCADHeadlessAdapter())
