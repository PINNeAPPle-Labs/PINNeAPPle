"""Output units of the ``pinneapple_systems`` tools (process components, ENSO and time-series helpers) (numpy ``Returns`` / ``Attributes``), read back
from the source the way the Studio's catalog reads them. The units come from the field names (``P_Pa``, ``T_K``,
``h_J_kg``), their comments or the docs; a value in the units of the caller's inputs stays empty."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pinneapple_studio.outputs import from_docstring

ROOT = Path(__file__).resolve().parents[1]
P = "pinneapple_systems.process_components."


def _doc(symbol: str) -> str:
    module, _, name = symbol.partition(":")
    tree = ast.parse((ROOT / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
    return next(ast.get_docstring(n) or "" for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


RETURNS = {
    P + "beam_statics:rectangular_section_properties": [("I", "m4"), ("Q", "m3")],
    P + "control_valve:incompressible_mass_flow": [("w", "kg/s")],
    P + "reaction_kinetics:diprotic_fractions": [("alpha_H2A", "-"), ("alpha_HA", "-"), ("alpha_A", "-")],
    P + "real_gas_eos:mass_flow_to_standard_volumetric_flow": [("q_std", "m3/s")],
    P + "similarity_map:polytropic_head_from_psi": [("head", "J/kg")],
    P + "similarity_map:required_speed_for_head": [("N", "rpm")],
    P + "water_hammer:joukowsky_head_rise": [("dH", "m")],
    P + "water_hammer:wave_period": [("T", "s")],
    "pinneapple_systems.time_series.enso.data:d20_from_profile": [("d20", "m")],
    "pinneapple_systems.time_series.enso.data:soi_from_mslp": [("soi", "-")],
    "pinneapple_systems.time_series.enso.pinn:project_forcing": [("X0", "m/s2"), ("X2", "m/s2")],
    "pinneapple_systems.time_series.metrics_ext.probabilistic:coverage": [("coverage", "-")],
}
ATTRIBUTES = {
    P + "real_gas_eos:GasState": {"P_Pa": "Pa", "T_K": "K", "h_J_kg": "J/kg", "s_J_kgK": "J/(kg K)", "Z": "-"},
    P + "heat_exchanger:HeatExchangerResult": {"Q_W": "W", "dP_Pa": "Pa", "UA_effective_W_K": "W/K", "NTU": "-"},
    P + "water_hammer:OperatingPoint": {"flow": "m3/s", "pump_head": "m", "reynolds": "-"},
    P + "water_hammer:TransientResult": {"head": "m", "flow": "m3/s", "tau": "-", "t": None, "dt": None},
    P + "torsional_stickslip:StickSlipResult": {"t_s": "s", "surface_torque_ts_N_m": "N m"},
    P + "beam_nonlinear_fem:StaticNonlinearBeamResult": {"w_m": "m", "bending_stress": None},   # units of E passed in
    "pinneapple_systems.time_series.enso.data:Gridded": {"lat": "deg", "lon": "deg"},
    "pinneapple_systems.time_series.solar_cycle:HilbertResult": {"phase": "rad", "amplitude": None},
}


@pytest.mark.parametrize("symbol", sorted(RETURNS))
def test_returns_units(symbol):
    assert [(o["name"], o["unit"]) for o in from_docstring(_doc(symbol))] == RETURNS[symbol]


@pytest.mark.parametrize("symbol", sorted(ATTRIBUTES))
def test_record_attribute_units(symbol):
    units = {o["name"]: o["unit"] for o in from_docstring(_doc(symbol), "attributes")}
    assert {k: units.get(k) for k in ATTRIBUTES[symbol]} == ATTRIBUTES[symbol]
