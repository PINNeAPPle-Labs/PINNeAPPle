"""Output units of the ``pinneapple_simulation.electromagnetics`` tools (numpy ``Returns`` / ``Attributes``), read back
from the source the way the Studio's catalog reads them. The package works in SI (``constants``: ``C0`` in m/s, ``MU0``
in H/m), so lengths in m give Hz, s, F, H and T; a value in the units of the caller's inputs stays empty."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pinneapple_studio.outputs import from_docstring

ROOT = Path(__file__).resolve().parents[1]
E = "pinneapple_simulation.electromagnetics."


def _doc(symbol: str) -> str:
    module, _, name = symbol.partition(":")
    tree = ast.parse((ROOT / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
    return next(ast.get_docstring(n) or "" for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


RETURNS = {
    E + "statics:coax_capacitance": [("C", "F/m")],
    E + "statics:coax_inductance": [("L", "H/m")],
    E + "statics:helmholtz_coil_center_B": [("B", "T")],
    E + "statics:mutual_inductance_coaxial_loops": [("M", "H")],
    E + "waves:rect_waveguide_cutoff": [("f_c", "Hz")],
    E + "waves:brewster_angle": [("theta_B", "rad")],
    E + "waves:free_space_wavelength": [("wavelength", "m")],
    E + "fdtd:courant_limit": [("dt", "s")],
    E + "fem:flux_density": [("B", "T")],
    E + "fem3d.team7:coil_current_density": [("J", "A/m2")],
    E + "machines3d.magnet_loss:plate_loss": [("P", "W")],
    E + "analytic:round_conductor_eddy_loss_low_freq": [("P", "W/m")],
    E + "reflector.po:gain_from_q": [("gain", "dBi")],
    E + "periodic.benchmark:worst_case": [("worst", "dB")],
}
ATTRIBUTES = {
    E + "fem:MagnetostaticSolution": {"A": "Wb/m", "B": "T", "nu": "m/H", "energy_functional": "J/m"},
    E + "mie:MieResult": {"x": "-", "qext": "-", "g": "-"},
}


@pytest.mark.parametrize("symbol", sorted(RETURNS))
def test_returns_units(symbol):
    assert [(o["name"], o["unit"]) for o in from_docstring(_doc(symbol))] == RETURNS[symbol]


@pytest.mark.parametrize("symbol", sorted(ATTRIBUTES))
def test_record_attribute_units(symbol):
    units = {o["name"]: o["unit"] for o in from_docstring(_doc(symbol), "attributes")}
    assert {k: units.get(k) for k in ATTRIBUTES[symbol]} == ATTRIBUTES[symbol]


def test_same_unit_as_the_input_stays_empty():
    assert "Returns" not in _doc(E + "antenna.ava:microstrip_width")       # "same unit as h"
    assert "Returns" not in _doc(E + "machines.geometry:polygon_area")
