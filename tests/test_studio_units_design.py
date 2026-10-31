"""Output units of the ``pinneapple_design`` tools (numpy ``Returns`` / ``Attributes``, ``name : type [unit]``), read
back from the source the way the Studio's catalog reads them. Each unit comes from the code (``LB`` in kg/lb, the
1.225 kg/m3 of ISA), the names (``_c``, ``_k_w``, ``_pa``, ``_mm``) or the docs; ``-`` is dimensionless."""
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


A, S, T = "pinneapple_design.aero.", "pinneapple_design.surrogate_inverse.", "pinneapple_design.thermal."
RETURNS = {
    A + "aircraft:isa_density": [("rho", "kg/m3")],
    A + "aircraft3d:wing_weight": [("m", "kg")],
    A + "aircraft3d:fuselage_cd0": [("cd0", "-")],
    A + "airliner:sweep_le_from_qc": [("sweep_le", "deg")],
    A + "buildup:sutherland_mu": [("mu", "Pa s")],
    S + "heatsink_fins:junction_temperature_c": [("T_j", "degC")],
    S + "heatsink_fins:thermal_resistance_k_w": [("R", "K/W")],
    S + "brake_rotor:thermal_resistance_k_w": [("R", "K/W")],
    S + "reflex_port:resonant_frequency_hz": [("f", "Hz")],
    S + "suspension_arm:bending_stress_pa": [("sigma", "Pa")],
    S + "suspension_arm:fatigue_life_cycles": [("N", "-")],
    S + "transdermal_patch:release_rate_mg_day": [("rate", "mg/day")],
    S + "transdermal_patch:reservoir_mass_mg": [("m", "mg")],
    T + "board_cooling.scene:fan_positions": [("centres", "mm"), ("radius", "mm")],
}
ATTRIBUTES = {
    A + "panel:PanelResult": {"alpha_deg": "deg", "cp": "-", "cl_pressure": "-"},
    S + "core:SurrogateFit": {"val_mean_rel_err": "-", "seconds": "s", "n_train": "-"},
    S + "heatsink_fins:HeatsinkDesign": {"fin_count": "-", "fin_height_m": "m"},
    S + "suspension_arm:SensitivityResult": {"load_n": "N", "mass_kg": "kg", "d_life_d_width": "1/m", "d_mass_d_width": "kg/m"},
    T + "board_cooling.mission:CoolingMission": {"t_ambient_c": "degC", "max_width_mm": "mm", "tim_k_mm2_w": "K mm2/W"},
    T + "fin_cell:FinCellResult": {"t_junction_c": "degC", "energy_balance_rel_error": "-"},
}


@pytest.mark.parametrize("symbol", sorted(RETURNS))
def test_returns_units(symbol):
    assert [(o["name"], o["unit"]) for o in from_docstring(_doc(symbol))] == RETURNS[symbol]


@pytest.mark.parametrize("symbol", sorted(ATTRIBUTES))
def test_record_attribute_units(symbol):
    units = {o["name"]: o["unit"] for o in from_docstring(_doc(symbol), "attributes")}
    assert {k: units.get(k) for k in ATTRIBUTES[symbol]} == ATTRIBUTES[symbol]
