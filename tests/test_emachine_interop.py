"""Interoperability of the machine models (``machines.interop``): pyleecan files, FEMM scripts, MATLAB, DXF, VTK and
the PINNeAPPle data contracts."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.machine import belt_winding
from pinneapple_simulation.electromagnetics.machines import M400_50A, MAGNET_PRIUS, prius_2004
from pinneapple_simulation.electromagnetics.machines.geometry import polygon_area
from pinneapple_simulation.electromagnetics.machines.interop import (
    cross_section_polygons,
    femm_lua,
    fluxmap_dataset,
    from_mat,
    from_pyleecan,
    read_dxf_polylines,
    read_femm_lua_polygons,
    rotor_parameter_space,
    to_dxf,
    to_mat,
    to_pyleecan,
)

PRIUS_JSON = Path(__file__).parent / "data" / "pyleecan" / "Toyota_Prius.json"


def test_pyleecan_prius_file_gives_our_prius():
    g, w = from_pyleecan(PRIUS_JSON)
    ref = prius_2004()
    assert g == ref  # every dimension, the stack length and the stacking factor
    assert w["turns_per_coil"] == 9 and w["parallel_paths"] == 1 and w["coil_pitch"] == 6
    cps = np.array(w["conductors_per_slot"])  # phase A: +9 +9 in two slots, -9 -9 six slots later
    assert np.allclose(cps[:8, 0], [9, 9, 0, 0, 0, 0, -9, -9])


def test_pyleecan_round_trip():
    g = prius_2004()
    d = to_pyleecan(g)
    d["stator"]["winding"] = {"Ntcoil": 9}
    g2, _ = from_pyleecan(d)
    assert g2 == g


def test_femm_script_draws_the_mesher_polygons():
    g = prius_2004()
    lua = femm_lua(g, M400_50A, MAGNET_PRIUS, belt_winding(48, 4), 9.0, currents=(100.0, -50.0, -50.0))
    polys = read_femm_lua_polygons(lua)
    ours = cross_section_polygons(g)
    assert len(polys) == len(ours) == 8 * 5 + 48 * 2
    assert sum(abs(polygon_area(p)) for p in polys) == pytest.approx(sum(abs(polygon_area(p)) for _, p, _ in ours),
                                                                       rel=1e-9)
    assert lua.count('mi_setblockprop("Magnet"') == 16 and lua.count('mi_setblockprop("Coil"') == 48
    assert lua.count('mi_addbhpoint("M400-50A"') == len(M400_50A.B)
    assert "mi_analyze(1)" in lua


def test_dxf_round_trip(tmp_path):
    g = prius_2004()
    p = to_dxf(tmp_path / "prius.dxf", g)
    layers = read_dxf_polylines(p)
    assert len(layers["MAGNET"]) == 16 and len(layers["SLOT"]) == 48
    ours = [xy for n, xy, _ in cross_section_polygons(g) if n.startswith("magnet")]
    assert sum(abs(polygon_area(x)) for x in layers["MAGNET"]) == pytest.approx(
        sum(abs(polygon_area(x)) for x in ours), rel=1e-9)
    assert abs(polygon_area(layers["MAGNET"][0])) == pytest.approx(0.0189 * 0.0065, rel=1e-9)


def test_mat_round_trip(tmp_path):
    p = to_mat(tmp_path / "m.mat", i_d=np.linspace(-1, 0, 3), T=np.eye(3), meta={"poles": 8})
    d = from_mat(p)
    assert np.allclose(d["T"], np.eye(3)) and int(d["meta"]["poles"]) == 8


def test_contracts():
    from types import SimpleNamespace

    ps = rotor_parameter_space()
    assert [p.name for p in ps.parameters] == ["W4", "H3", "H0", "W0", "H1"]
    gd, gq = np.linspace(-200, 0, 3), np.linspace(0, 200, 3)
    D, Q = np.meshgrid(gd, gq, indexing="ij")
    fm = SimpleNamespace(i_d=gd, i_q=gq, psi_d=0.2 + 1e-3 * D, psi_q=2e-3 * Q, T=6 * (0.2 * Q - 1e-3 * D * Q))
    df, spec = fluxmap_dataset(fm)
    assert len(df) == 9 and spec.n_rows == 9 and {c.name for c in spec.columns} == set(df.columns)
    assert math.isclose(df["torque"].iloc[-1], fm.T[-1, -1])


def test_contract_files_round_trip(tmp_path):
    from pinneapple_data.contracts.models import load

    ps = rotor_parameter_space()
    ps.save(tmp_path / "rotor.space.json")
    assert load(tmp_path / "rotor.space.json") == ps


def test_elmer_case_files(tmp_path):
    from pinneapple_simulation.electromagnetics.machines.interop import elmer_case

    out = elmer_case(tmp_path, prius_2004(), M400_50A, MAGNET_PRIUS, belt_winding(48, 4), 9.0, currents=(100, -50, -50),
                     n_gap=288)
    sif = (tmp_path / "case.sif").read_text()
    import re

    assert len(re.findall(r"^Body \d+ ", sif, re.M)) == 4 + 16 + 48
    assert len(re.findall(r"^Body Force \d+ ", sif, re.M)) == 48
    assert '"MagnetoDynamics2D" "MagnetoDynamics2D"' in sif and "H-B Curve" in sif
    assert (tmp_path / "machine.geo").exists() and out["bodies"][1] == 1
