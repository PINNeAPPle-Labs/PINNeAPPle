"""Steady compressible flow through an internal passage, solved, checked and written up.

The pipeline answers one question for any axisymmetric passage — a valve, an orifice, a nozzle, a
venturi, a reducer: **what flows, at what pressure drop, and does the answer hold?** It is built from
four parts, each usable alone:

    from pinneapple_simulation.pipelines.compressible_internal import (
        PassageSpec, Span, Ring, StudySpec, Conditions, globe_valve, run_study)

    passage = globe_valve(lift=4.0)                       # or build the spans yourself
    spec = StudySpec(passage, gas="methane", conditions=Conditions(p_out_bar=60, T_in_K=300,
                                                                  p_in_target_bar=70))
    out = run_study(spec, "runs/valve4mm", sweep_values=[1.0, 2.0, 4.0, 8.0])
    print(out["summary"], out["decision"])

Modules: :mod:`.passage` (the axisymmetric passage and its hexahedral sector mesh),
:mod:`.physics` (gas, boundary conditions, Peng-Robinson), :mod:`.case` (the OpenFOAM case and the
runner), :mod:`.post` (patch averages, indicators, CSV), :mod:`.checks` (the physics checks that
decide whether the result may be used), :mod:`.visual` (3-D scene, Blender render, video),
:mod:`.study` (the outer loop and the study), :mod:`.studies` (the passages that ship).

The guide is ``docs/guides/compressible_passages.md``; the worked case, with the customer's own
numbers, is ``PINNeAPPle-CFD/casos/cfd08-valvula-globo``.
"""
from __future__ import annotations

from .case import RunResult, run_case, write_case
from .checks import critical_pressure_ratio, run_checks, stagnation_enthalpy
from .passage import MeshSummary, PassageSpec, Ring, Span, build_mesh
from .physics import GASES, Conditions, Gas, blend, peng_robinson, resolve_gas, sutherland_mu
from .post import FlowResult, axial_profile, read_case, write_tables
from .studies import STUDIES, converging_nozzle, globe_valve, orifice_plate
from .study import CaseOutcome, StudySpec, run_study, solve_once, solve_to_pressure_drop

__all__ = [
    "PassageSpec", "Span", "Ring", "MeshSummary", "build_mesh",
    "Gas", "Conditions", "GASES", "resolve_gas", "blend", "peng_robinson", "sutherland_mu",
    "write_case", "run_case", "RunResult", "read_case", "write_tables", "FlowResult", "axial_profile",
    "run_checks", "critical_pressure_ratio", "stagnation_enthalpy",
    "StudySpec", "CaseOutcome", "solve_once", "solve_to_pressure_drop", "run_study",
    "globe_valve", "orifice_plate", "converging_nozzle", "STUDIES",
]
