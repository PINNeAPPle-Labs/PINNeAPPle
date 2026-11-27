"""Três tipos de exigência, uma ordem fixa, e a etapa que recusou cada projeto (#172).

O aceite tem duas frases e cada uma vira teste aqui: um projeto que viola uma restrição dura é recusado **antes de
qualquer corrida**, e o pipeline registra **qual etapa** recusou cada projeto. A ordem não é convenção de chamada —
o tipo da exigência determina a etapa —, e um objetivo nunca recusa nada: ele é medido, e o que se faz com ele é
ordenar.
"""
from __future__ import annotations

import pytest

from pinneapple_analysis.trust.corrections import Correction
from pinneapple_design.design_gate import (
    ACCEPTED,
    INCONCLUSIVE,
    REJECTED,
    DesignGate,
    Requirement,
    StageRecord,
    summarise,
)

THICK = Requirement.hard("thickness_mm >= 2.0", name="espessura mínima")
MASS = Requirement.hard("mass_kg <= 50", name="massa máxima")
DESIGN = {"thickness_mm": 3.0, "mass_kg": 12.0, "clearance_mm": 8.0, "cells": 120000, "cd": 0.014}


def solver(*, cd: float = 0.014, converged: bool = True, calls: list | None = None):
    def run(design):
        if calls is not None:
            calls.append(dict(design))
        return {"cd": cd, "converged": converged, "residual": 1e-7 if converged else 1e-2}
    return run


# ---------------------------------------------------------------------------------------- a restrição dura primeiro
def test_a_hard_constraint_rejects_before_any_solver_run():
    calls: list = []
    gate = DesignGate([THICK, Requirement.objective("cd")])
    decision = gate.evaluate({**DESIGN, "thickness_mm": 1.4}, run=solver(calls=calls), name="D-01")
    assert decision.status == REJECTED and decision.rejected_by == "geometry"
    assert calls == []                       # o solver nunca foi chamado: é a frase literal do aceite
    assert decision.ran is False
    assert decision.reason.startswith("espessura mínima")
    assert decision.objectives == {}         # nada medido: a etapa parou antes
    assert [r.stage for r in decision.records] == ["geometry"]


def test_a_design_that_passes_geometry_reaches_the_solver_and_the_objective():
    calls: list = []
    gate = DesignGate([THICK, MASS, Requirement.physics("residual <= 1e-5", name="resíduo",
                                                        stage="convergence"),
                       Requirement.objective("cd", sense="min")])
    decision = gate.evaluate(DESIGN, run=solver(calls=calls), name="D-02")
    assert decision.status == ACCEPTED and decision.ran is True and len(calls) == 1
    assert decision.objectives == {"cd": pytest.approx(0.014)}
    assert [r.status for r in decision.records] == ["pass", "empty", "pass", "empty", "pass"]


# ---------------------------------------------------------------------------------------- a ordem e a etapa
def test_the_mesh_stage_rejects_after_geometry_and_before_the_solver():
    calls: list = []
    gate = DesignGate([Requirement.hard("cells <= 200000", name="malha", stage="mesh"),
                       Requirement.physics("residual <= 1e-5", name="convergência"),
                       Requirement.objective("cd")])
    decision = gate.evaluate({"cells": 400000}, run=solver(calls=calls), name="D-03")
    assert decision.rejected_by == "mesh" and calls == [] and decision.ran is False
    assert [r.stage for r in decision.records] == ["geometry", "mesh"]


def test_the_convergence_stage_stops_before_the_physics_is_asked():
    asked: list = []
    gate = DesignGate([Requirement.physics("converged", name="convergência", stage="convergence"),
                       Requirement.physics(func=lambda v: asked.append(v) or True, name="conservação"),
                       Requirement.objective("cd")])
    decision = gate.evaluate({}, run=solver(converged=False), name="D-04")
    assert decision.rejected_by == "convergence" and decision.ran is True
    assert asked == []                       # a física não foi perguntada sobre uma solução que não convergiu


def test_the_pipeline_records_which_stage_rejected_each_design():
    gate = DesignGate([THICK, MASS, Requirement.hard("cells <= 200000", name="malha", stage="mesh"),
                       Requirement.physics("residual <= 1e-5", name="resíduo", stage="convergence"),
                       Requirement.objective("cd")])
    designs = [DESIGN, {**DESIGN, "thickness_mm": 1.0}, {**DESIGN, "mass_kg": 80.0}, {**DESIGN, "cells": 500000}]
    decisions = gate.evaluate_all(designs, run=solver(), names=["D-10", "D-11", "D-12", "D-13"])
    assert [d.status for d in decisions] == [ACCEPTED, REJECTED, REJECTED, REJECTED]
    assert [d.rejected_by for d in decisions] == ["", "geometry", "geometry", "mesh"]
    counts = summarise(decisions)
    assert counts == {"total": 4, "accepted": 1, "rejected": 3, "inconclusive": 0, "ran": 1,
                      "rejected_by_stage": {"geometry": 2, "mesh": 1},
                      "rejected_by_requirement": {"geometry/espessura mínima": 1, "geometry/massa máxima": 1,
                                                  "mesh/malha": 1}}
    assert decisions[1].to_dict()["rejected_by"] == "geometry"


def test_the_log_of_one_design_names_the_stage_and_the_requirement():
    gate = DesignGate([THICK])
    log = gate.evaluate({**DESIGN, "thickness_mm": 0.9}, name="D-20").table()
    assert log.splitlines()[0] == "D-20: rejected em geometry"
    assert "espessura mínima" in log and "FAIL" in log


# ---------------------------------------------------------------------------------------- os objetivos não impõem
def test_an_objective_is_measured_and_never_rejects():
    gate = DesignGate([THICK, Requirement.objective("cd", sense="min", func=lambda v: v["cd"])])
    decision = gate.evaluate({**DESIGN, "cd": 9.9}, run=solver(cd=9.9), name="D-30")
    assert decision.status == ACCEPTED and decision.objectives["cd"] == pytest.approx(9.9)
    assert [r.status for r in decision.records if r.stage == "objectives"] == ["pass"] or decision.records[-1].status == "pass"


def test_an_objective_that_came_out_missing_is_not_a_zero():
    gate = DesignGate([Requirement.objective("cd")])
    decision = gate.evaluate({}, run=lambda d: {}, name="D-31")
    assert decision.status == INCONCLUSIVE and "cd" in decision.reason


# ---------------------------------------------------------------------------------------- os tipos decidem a etapa
def test_a_hard_constraint_cannot_be_checked_on_the_physics_stage():
    with pytest.raises(ValueError, match="é checada em"):
        Requirement(name="x", kind="hard", stage="physics", condition="a > 1")
    with pytest.raises(ValueError, match="é checada em"):
        Requirement(name="x", kind="objective", stage="geometry", condition="a > 1")


def test_an_objective_is_not_a_condition():
    with pytest.raises(ValueError, match="número medido"):
        Requirement(name="x", kind="objective", stage="objectives", condition="cd > 1")
    with pytest.raises(ValueError, match="nem nenhuma"):
        Requirement(name="x", kind="hard", stage="geometry")


def test_a_requirement_that_cannot_be_asked_on_this_design_is_not_a_pass():
    gate = DesignGate([Requirement.hard("clearance_mm > 5", name="folga")])
    decision = gate.evaluate({"thickness_mm": 3.0}, name="D-40")
    assert decision.status == INCONCLUSIVE and decision.rejected_by == "geometry"
    assert "clearance_mm" in decision.reason


def test_the_physics_stage_without_a_solver_is_inconclusive_not_green():
    gate = DesignGate([Requirement.physics("residual <= 1e-5", name="resíduo")])
    decision = gate.evaluate({}, name="D-41")
    assert decision.status == INCONCLUSIVE and decision.ran is False
    assert "sem solver" in decision.records[-1].results[0].reason


def test_a_physics_check_speaks_the_vocabulary_of_a_field_check():
    class FakeCheck:                                   # um FieldCheck: passed None = não rodou
        passed = False
        detail = "fluxo líquido 3,2e-2 contra uma fonte de 1,0"

    gate = DesignGate([Requirement.physics(func=lambda v: FakeCheck(), name="conservação")])
    decision = gate.evaluate({}, run=solver(), name="D-42")
    assert decision.rejected_by == "physics" and "fluxo líquido" in decision.reason

    class Unrun(FakeCheck):
        passed = None

    gate = DesignGate([Requirement.physics(func=lambda v: Unrun(), name="conservação")])
    assert gate.evaluate({}, run=solver(), name="D-43").status == INCONCLUSIVE


def test_a_physics_check_may_answer_with_a_reason_of_its_own():
    gate = DesignGate([Requirement.physics(func=lambda v: (v["cd"] < 0.02, "arrasto acima do esperado"),
                                           name="arrasto plausível")])
    decision = gate.evaluate({}, run=solver(cd=0.05), name="D-44")
    assert decision.rejected_by == "physics" and decision.reason == "arrasto plausível: arrasto acima do esperado"


# ---------------------------------------------------------------------------------------- a correção do #173 entra
def test_a_stored_correction_becomes_a_requirement_and_keeps_its_provenance():
    rule = Correction(rule_id="R-007", condition="thickness_mm >= 2.0", case="bracket*",
                      statement="a 1.5 mm web cracked in test 42", reviewer="A. Ribeiro", date="2026-11-20",
                      source="design D-17")
    gate = DesignGate([Requirement.from_correction(rule)])
    decision = gate.evaluate({**DESIGN, "thickness_mm": 1.5}, name="D-50")
    assert decision.rejected_by == "geometry" and decision.reason.startswith("correction R-007")
    assert "A. Ribeiro" in gate.requirements[0].why and "2026-11-20" in gate.requirements[0].why


def test_the_same_requirement_cannot_be_declared_twice_in_a_stage():
    with pytest.raises(ValueError, match="já existe"):
        DesignGate([THICK, Requirement.hard("thickness_mm >= 1", name="espessura mínima")])


def test_a_stage_with_nothing_to_check_is_recorded_as_empty_not_as_passed():
    decision = DesignGate([THICK]).evaluate(DESIGN, name="D-60")
    assert decision.status == ACCEPTED
    assert [r.status for r in decision.records] == ["pass", "empty", "empty", "empty", "empty"]
    assert isinstance(decision.records[1], StageRecord) and decision.records[1].results == []
