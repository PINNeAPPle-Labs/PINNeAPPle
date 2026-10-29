"""A parametric FreeCAD model to a checked OpenFOAM result, as one reproducible workflow.

:class:`PipelineSpec` describes the study (document or template, parameters with bounds, patch rules, mesh, physics,
cross sections, derived quantities, references); :meth:`PipelineSpec.to_workflow` turns it into a
:class:`pinneapple_tools.automation.Workflow` whose steps run through the automation adapters:

=========  =========================  ===================================================================
step       action                     what it does
=========  =========================  ===================================================================
cad        ``freecadcmd.export``      set the spreadsheet parameters, recompute, export STEP + faces, name
                                      the patches by rule, write the multi-region STL (m), geometry figure
case       ``openfoam.prepare``       mesh dictionaries (blockMesh + snappyHexMesh, or cfMesh) and the case
                                      (simpleFoam / pimpleFoam, laminar / k-omega SST, passive scalar;
                                      external flows: :class:`ExternalFlow`'s dictionaries)
mesh       ``openfoam.mesh``          ``Allmesh`` (snappyHexMesh in parallel when cores allow) + checkMesh
solve      ``openfoam.solve``         ``Allrun`` (decomposed, MPI) and the convergence checks of the log
post       ``openfoam.post``          pressures, fluxes, sections, forces, y+, Courant, figures, derived
                                      quantities and each reference with its measured error
verify     ``verify.criteria``        acceptance: converged, mass balance, every reference within tolerance
=========  =========================  ===================================================================

:func:`run_pipeline` runs it with :class:`~pinneapple_tools.automation.Runner` (resumable: a rerun in the same folder
skips the passed steps) and returns the run record; the folder holds ``study_result.json``
(:class:`pinneapple_data.contracts.StudyResult`), ``report.html``, every log, file hash and figure.

Expressions (physics values, sections, derived quantities, references) are text over the inputs (document units:
mm, degrees), the physics values (SI) and, after the run, the measured quantities; see :data:`FUNCTIONS` for the
reference correlations they may call.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .cad import rule_eval
from .models import MODELS

__all__ = ["PipelineSpec", "FUNCTIONS", "evaluate", "run_pipeline", "resolve_physics"]


def _functions() -> dict[str, Any]:
    from pinneapple_physics.closed_form import heat_exchanger, particle_drag, pipe_flow
    return {"laminar_f": pipe_flow.laminar, "dean": pipe_flow.dean_number, "white_ratio": pipe_flow.curved_pipe_ratio,
            "hausen_nu": heat_exchanger.hausen_mean_nusselt, "nu_from_T": heat_exchanger.mean_nusselt_from_temperatures,
            "clift_cd": particle_drag.clift_sphere_cd, "schiller_cd": particle_drag.schiller_naumann_cd,
            "log": math.log, "exp": math.exp, "log10": math.log10}


#: reference correlations an expression may call (pinneapple_physics.closed_form)
FUNCTIONS = ("laminar_f(Re)", "dean(Re, D, Rc)", "white_ratio(De)", "hausen_nu(Re, Pr, D_over_L)",
             "nu_from_T(Re, Pr, D_over_L, Tw, Tin, Tout)", "clift_cd(Re)", "schiller_cd(Re)")


def evaluate(expr: Any, names: dict[str, Any]) -> Any:
    """A number, or the value of an expression (see :func:`.cad.rule_eval`) with the reference functions."""
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return expr
    if isinstance(expr, (list, tuple)):
        return [evaluate(e, names) for e in expr]
    return rule_eval(str(expr), names, _functions())


def resolve_physics(physics: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """Physics values with expressions evaluated in order (a later key may use an earlier one, e.g.
    ``velocity = Re * nu / (D * 1e-3)``); nested dicts (the scalar) too."""
    names = dict(inputs)
    text = ("solver", "turbulence", "name", "resolution")

    def one(k, v):
        if v is None or isinstance(v, bool) or k in text or v == "zeroGradient":
            return v
        if isinstance(v, dict):
            return {kk: one(kk, vv) for kk, vv in v.items()}
        return evaluate(v, names)
    out: dict[str, Any] = {}
    for k, v in physics.items():
        out[k] = one(k, v)
        if isinstance(out[k], (int, float)) and not isinstance(out[k], bool):
            names[k] = out[k]
    return out


@dataclass
class PipelineSpec:
    """One FreeCAD -> OpenFOAM study.

    ``template`` names a model of :mod:`.models` (its document is built in the run folder when ``document`` is
    empty, its defaults fill ``parameters`` / ``patches`` / ``solid`` / ``flow``). ``parameters``: document alias
    -> {default, min, max, unit, description} (workflow inputs); ``inputs``: further inputs that are not document
    parameters (e.g. a Reynolds number). ``physics``: :class:`.case.Physics` fields (internal) or
    :class:`ExternalFlow` options (external, plus ``speed`` / ``nu`` / ``rho``), values may be expressions.
    ``sections``: [{name, origin, normal, radius}] in mm, expressions allowed. ``derived``: name -> expression over
    everything measured. ``references``: name -> {measured, reference, tol, source}. ``plane``: {origin, normal}
    (mm) of the field and mesh figures."""
    id: str
    title: str
    template: str | None = None
    document: str | None = None
    solid: str = ""
    flow: str = ""
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    patches: dict[str, Any] = field(default_factory=dict)
    physics: dict[str, Any] = field(default_factory=dict)
    mesh: dict[str, Any] = field(default_factory=dict)
    sections: list[dict[str, Any]] = field(default_factory=list)
    derived: dict[str, str] = field(default_factory=dict)
    references: dict[str, dict[str, Any]] = field(default_factory=dict)
    plane: dict[str, Any] = field(default_factory=lambda: {"origin": [0, 0, 0], "normal": [0, 0, 1]})
    backend: str = "auto"
    procs: int | str = "auto"
    tolerance_mm: float = 0.05
    mass_tol: float = 0.01
    description: str = ""
    limits: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.template:
            m = MODELS[self.template]
            self.solid = self.solid or m.solid
            self.flow = self.flow or m.flow
            params = copy.deepcopy(m.parameters)
            for k, v in self.parameters.items():
                params[k] = {**params.get(k, {}), **v}
            self.parameters = params
            self.patches = self.patches or dict(m.patches)
        if self.flow not in ("internal", "external"):
            raise ValueError(f"{self.id}: flow must be internal or external (or give a template)")
        if not self.parameters:
            raise ValueError(f"{self.id}: no document parameters")

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineSpec:
        return cls(**d)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1))
        return p

    @classmethod
    def load(cls, path: str | Path) -> PipelineSpec:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def defaults(self) -> dict[str, float]:
        return {k: v["default"] for k, v in {**self.parameters, **self.inputs}.items()}

    # ------------------------------------------------------------------ workflow
    def to_workflow(self, folder: str | Path | None = None):
        """The :class:`~pinneapple_tools.automation.Workflow` of this study. ``folder``: where the workflow file
        would live (``{example}`` in templates); documents built from a template go in the run folder."""
        from pinneapple_tools.automation import Workflow
        names = list(self.parameters) + list(self.inputs)
        inp = {k: f"{{inputs.{k}}}" for k in names}
        doc_params = {k: f"{{inputs.{k}}}" for k in self.parameters}
        fc_req = [] if self.backend == "fake" else (["freecadcmd"] if self.backend == "freecad" else [])
        steps = [
            {"id": "cad", "title": "FreeCAD: set the parameters, recompute, export and name the patches",
             "action": "freecadcmd.export", "requires": fc_req,
             "params": {"document": self.document or "", "template": self.template or "", "solid": self.solid,
                        "parameters": doc_params, "patches": self.patches, "backend": self.backend,
                        "tolerance_mm": self.tolerance_mm, "flow": self.flow},
             "expects": ["cad/geometry.stl", "cad/geometry.png"]},
            {"id": "case", "title": "Write the mesh dictionaries and the OpenFOAM case", "action": "openfoam.prepare",
             "after": ["cad"], "requires": ["openfoam"],
             "params": {"flow": self.flow, "stl": "{steps.cad.stl}", "kinds": "{steps.cad.kinds}",
                        "inlet_normals": "{steps.cad.inlet_normals}", "inside": "{steps.cad.inside_m}",
                        "inputs": inp, "physics": self.physics, "mesh": self.mesh, "procs": self.procs},
             "expects": ["case/case/Allmesh", "case/case/Allrun"]},
            {"id": "mesh", "title": "Mesh (blockMesh + snappyHexMesh) and checkMesh", "action": "openfoam.mesh",
             "after": ["case"], "params": {"mesh_case": "{steps.case.case}"}, "timeout_s": 3600},
            {"id": "solve", "title": "Solve", "action": "openfoam.solve", "after": ["mesh"], "confirm": ["expensive"],
             "timeout_s": 4 * 3600,
             "params": {"case": "{steps.case.case}", "residual_tol": "{steps.case.residual_tol}", "compat": False}},
            {"id": "post", "title": "Post-process: pressures, fluxes, sections, forces, y+, figures, references",
             "action": "openfoam.post", "after": ["solve"],
             "params": {"case": "{steps.case.case}", "flow": self.flow, "kinds": "{steps.cad.kinds}",
                        "physics": "{steps.case.physics}", "inputs": inp, "sections": self.sections,
                        "derived": self.derived, "references": self.references, "plane": self.plane,
                        "stl": "{steps.cad.stl}", "title": self.title},
             "expects": ["post/fields.png", "post/mesh.png", "post/measured.json"]},
            {"id": "verify", "title": "Acceptance criteria", "action": "verify.criteria", "after": ["post"]},
        ]
        criteria = [{"id": "converged", "kind": "converged", "title": "solver converged", "params": {"step": "solve"}}]
        if self.flow == "internal":
            criteria.append({"id": "mass_balance", "kind": "mass_balance",
                             "title": f"outlet flow equals inlet flow within {self.mass_tol:g}",
                             "params": {"case": "{steps.case.case}", "inlet": "{steps.cad.inlet}",
                                        "outlet": "{steps.cad.outlet}", "tol": self.mass_tol}})
        if self.physics.get("solver") == "pimpleFoam":
            criteria.append({"id": "courant", "kind": "metric", "title": "largest Courant number at most 1 (or 1.05 x max_courant above 1)",
                             "params": {"step": "post", "output": "courant_max",
                                        "max": "{steps.case.max_courant_allowed}"}})
        for name, r in self.references.items():
            criteria.append({"id": f"ref_{name}", "kind": "metric",
                             "title": f"{name}: within {100 * float(r.get('tol', 0.05)):g} % of "
                                      f"{r.get('source', 'the reference')}",
                             "params": {"step": "post", "output": f"ref_{name}_error",
                                        "max": float(r.get("tol", 0.05))}})
        wf_inputs = {k: {kk: vv for kk, vv in v.items() if kk in ("default", "min", "max", "unit", "description",
                                                                     "choices")}
                     for k, v in {**self.parameters, **self.inputs}.items()}
        d = {"id": self.id, "title": self.title, "description": self.description or self.title,
             "application": "FreeCAD (headless) + OpenFOAM", "inputs": wf_inputs, "steps": steps,
             "criteria": criteria, "limits": list(self.limits)}
        wf = Workflow.from_dict(d, source=str(Path(folder) / "workflow.json") if folder else None)
        return wf


def run_pipeline(spec: PipelineSpec, folder: str | Path, inputs: dict[str, Any] | None = None,
                 approve: bool = True, approver: str = "pipeline"):
    """Run (or resume) the study in ``folder`` with ``inputs`` (document parameters and other inputs). ``approve``:
    every step is approved (the solver is ``expensive``; a resumed step overwrites its own earlier files); without
    it the run stops at the first step that asks. Returns the automation ``RunRecord``."""
    from pinneapple_tools.automation import Runner
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    spec.save(folder / "pipeline.json")
    wf = spec.to_workflow(folder)
    run = Runner(wf, folder, inputs=inputs or None)
    return run.run(approve=[s.id for s in wf.steps] if approve else [], approver=approver)
