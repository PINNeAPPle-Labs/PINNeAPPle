"""Portuguese text of the library tools the gallery's examples use: each of these had none, and each must keep one
that names a symbol still defined in the code (the Studio shows it on the tool's page and card)."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pinneapple_studio import prose as P

ROOT = Path(__file__).resolve().parents[1]
USED_BY_EXAMPLES = [
    "pinneapple_analysis.dynamics.first_passage:survival_from_solution",
    "pinneapple_analysis.dynamics.koopman:DMD",
    "pinneapple_analysis.inverse_problems.activation_mapping:EikonalPINN",
    "pinneapple_analysis.inverse_problems.activation_mapping:EikonalPINNConfig",
    "pinneapple_analysis.inverse_problems.activation_mapping:interpolate_baseline",
    "pinneapple_analysis.inverse_problems.binn:BINNConfig",
    "pinneapple_analysis.inverse_problems.binn:ReactionDiffusionBINN",
    "pinneapple_analysis.inverse_problems.ode_model:ODEForwardModel",
    "pinneapple_analysis.inverse_problems.sbinn:SBINN",
    "pinneapple_analysis.inverse_problems.sbinn:SBINNConfig",
    "pinneapple_autolab.study.campaign:plan_doe",
    "pinneapple_autolab.study.campaign:run_campaign",
    "pinneapple_autolab.study.dataset:StudyDataset",
    "pinneapple_neural.architectures.pinns.pinnsformer_paper:PINNsFormerPaper",
    "pinneapple_neural.architectures.pinns.pinnsformer_paper:init_official",
    "pinneapple_physics.biology.cell_migration:fit_mechanistic",
    "pinneapple_physics.biology.cell_migration:load_scratch_assay",
    "pinneapple_physics.biology.excitable:Stimulus",
    "pinneapple_physics.biology.excitable:aliev_panfilov_2d",
    "pinneapple_physics.biology.excitable:conduction_velocity",
    "pinneapple_physics.biology.models:minimal_pbpk",
    "pinneapple_physics.biology.models:pk_one_compartment_mm",
    "pinneapple_physics.biology.models:ultradian_glucose_insulin",
    "pinneapple_physics.paper_problems_epi:fit_shooting",
    "pinneapple_physics.paper_problems_epi:synthetic_data",
    "pinneapple_simulation.electromagnetics.fem3d.cases:manufactured_magnetostatic",
    "pinneapple_simulation.electromagnetics.fem3d.cases:slab_eddy",
    "pinneapple_simulation.electromagnetics.fem3d.nedelec:EdgeSpace",
    "pinneapple_simulation.electromagnetics.fem3d.nedelec:solve_magnetostatic",
    "pinneapple_simulation.electromagnetics.fem3d.sphere:SphereCase",
    "pinneapple_simulation.electromagnetics.fem3d.sphere:errors",
    "pinneapple_simulation.electromagnetics.fem3d.sphere:mesh_sphere",
    "pinneapple_simulation.electromagnetics.fem3d.sphere:solve_native",
    "pinneapple_simulation.electromagnetics.fem3d.sphere:uniform_dirichlet",
    "pinneapple_simulation.electromagnetics.machines3d.afpm:AFPM3D",
    "pinneapple_simulation.electromagnetics.machines3d.afpm:AFPMMachine",
    "pinneapple_simulation.electromagnetics.machines3d.endeffects:end_effect_study",
    "pinneapple_simulation.electromagnetics.machines3d.magnet_loss:segmentation_study",
    "pinneapple_simulation.electromagnetics.machines3d.skew:continuous_factor",
    "pinneapple_simulation.electromagnetics.machines3d.skew:skew_study",
    "pinneapple_simulation.electromagnetics.reflector.design:from_config",
    "pinneapple_simulation.electromagnetics.reflector.design:generic_2ghz",
    "pinneapple_simulation.electromagnetics.reflector.design:outside",
    "pinneapple_simulation.electromagnetics.reflector.design:run_design_study",
    "pinneapple_simulation.hemodynamics.aorta:CoupledRun",
    "pinneapple_simulation.hemodynamics.aorta:setup_case",
    "pinneapple_simulation.numerical_solvers.xtfc_general:ConstrainedExpression1D",
    "pinneapple_simulation.numerical_solvers.xtfc_general:sample_calabro",
    "pinneapple_simulation.numerical_solvers.xtfc_inverse:solve_xtfc_inverse",
]


def _defined(symbol: str) -> bool:
    module, _, name = symbol.partition(":")
    path = ROOT / (module.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name for n in tree.body)


@pytest.fixture(scope="module")
def symbols():
    return P.load(ROOT).symbols


@pytest.mark.parametrize("symbol", USED_BY_EXAMPLES)
def test_tool_used_by_an_example_has_portuguese_text(symbols, symbol):
    assert _defined(symbol), symbol
    assert len((symbols.get(symbol) or "").strip()) > 20, symbol
