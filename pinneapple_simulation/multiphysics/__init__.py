"""Multiphysics coupling (#1209): two physics participants through named interface fields, coupled one-way,
two-way partitioned (Gauss-Seidel with Aitken relaxation) or monolithically, checked against closed-form coupled
solutions (Joule-heated wire, bimetal strip, conjugate heat transfer in a channel)."""
from .cases import CASES, CoupledCase, bimetal_strip, conjugate_channel, joule_wire
from .coupling import (
           Participant,
           coupling_strength,
           gauss_seidel,
           monolithic,
           one_way,
           recommend_strategy,
)
from .workflow import multiphysics_workflow, residual_table

__all__ = ["CASES", "CoupledCase", "Participant", "bimetal_strip", "conjugate_channel", "coupling_strength",
           "gauss_seidel", "joule_wire", "monolithic", "multiphysics_workflow", "one_way", "recommend_strategy",
           "residual_table"]
