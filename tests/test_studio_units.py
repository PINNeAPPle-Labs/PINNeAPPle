"""Output units written in the code of library tools (numpy ``Returns`` / ``Attributes``, ``name : type [unit]``),
read back from the source the way the Studio's catalog reads them. Units are taken from the code or its docs, never
guessed: ``-`` is dimensionless, and an output whose unit follows the caller's input has none."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pinneapple_studio.outputs import from_docstring

ROOT = Path(__file__).resolve().parents[1]


def _doc(symbol: str) -> str:
    module, _, name = symbol.partition(":")
    tree = ast.parse((ROOT / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
    return next(ast.get_docstring(n) or "" for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


CF = "pinneapple_physics.closed_form."
PHYSICS = {
    CF + "pipe_flow:darcy_pressure_drop": [("dp", "Pa")],
    CF + "pipe_flow:parallel_plates_pressure_drop": [("dp", "Pa")],
    CF + "pipe_flow:first_cell_height": [("y", "m")],
    CF + "pipe_flow:colebrook": [("f", "-")],
    CF + "pipe_flow:friction_factor": [("f", "-")],
    CF + "pump_npsh:iapws97_psat": [("p_sat", "Pa")],
    CF + "pump_npsh:atmosphere": [("p", "Pa")],
    CF + "rocket_nozzle:choked_mass_flow": [("mdot", "kg/s")],
    CF + "rocket_nozzle:characteristic_velocity": [("c_star", "m/s")],
    CF + "rocket_nozzle:specific_impulse": [("Isp", "s")],
    CF + "rocket_nozzle:area_ratio": [("ratio", "-")],
    CF + "heat:pin_fin_heat": [("q", "W")],
    CF + "heat_exchanger:effectiveness": [("eps", "-")],
    CF + "helmholtz_resonator:helmholtz_resonant_frequency_hz": [("f", "Hz")],
    CF + "helmholtz_resonator:port_volume_m3": [("V", "m3")],
    CF + "membrane_diffusion:steady_state_release_rate_mg_day": [("rate", "mg/day")],
    CF + "membrane_diffusion:concentration_at_membrane_depth": [("C", "kg/m3")],
    CF + "membrane_diffusion:post_depletion_decay_rate_constant_per_day": [("k", "1/day")],
    CF + "cantilever_fatigue:cantilever_root_bending_stress_pa": [("sigma", "Pa")],
    CF + "fin_array_conduction:fin_array_base_temperature_c": [("T_base", "degC")],
    CF + "eddy_coil:skin_depth": [("delta", "m")],
    CF + "thin_airfoil:naca4_camber_slope": [("dzdx", "-")],
    "pinneapple_physics.weather.evaluate:first_crossing": [("lead", "h")],
    "pinneapple_physics.weather.evaluate:acc": [("acc", "-")],
}
RECORDS = {
    "pinneapple_physics.biology.excitable:APSolution": {"x": "-", "y": "-", "t": "-", "u": "-", "v": "-", "dt": "-"},
    "pinneapple_physics.biology.excitable:FHNSolution": {"x": "-", "t": "-", "v": "-", "w": "-"},
}


@pytest.mark.parametrize("symbol", sorted(PHYSICS))
def test_returns_units(symbol):
    assert [(o["name"], o["unit"]) for o in from_docstring(_doc(symbol))] == PHYSICS[symbol]


@pytest.mark.parametrize("symbol", sorted(RECORDS))
def test_record_attribute_units(symbol):
    units = {o["name"]: o["unit"] for o in from_docstring(_doc(symbol), "attributes")}
    assert {k: units.get(k) for k in RECORDS[symbol]} == RECORDS[symbol]


def test_a_unit_that_follows_the_input_stays_empty():
    assert _doc(CF + "pipe_insulation:crf").count("[") == 0        # 1/period or not: left to the caller
    assert "Returns" not in _doc("pinneapple_physics.solid.axisymmetric:lame_parameters")   # same unit as E
