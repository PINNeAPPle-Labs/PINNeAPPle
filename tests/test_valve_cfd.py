"""Globe valve passage: geometry, mesh, the gas model of the OpenFOAM case, and a short OpenFOAM run (#1467)."""
import math

import numpy as np
import pytest

pytest.importorskip("CoolProp")

from pinneapple_systems.process_components import valve_cfd as vc  # noqa: E402
from pinneapple_systems.process_components.joule_thomson import DEFAULT_GAS, throttle  # noqa: E402


def test_geometry_profile_and_areas():
    g = vc.ValveGeometry()
    pts = g.profile()
    assert len(pts) == 12 and all(r > 0 for _, r in pts)          # annular everywhere: no cell on the axis
    a = g.areas_m2()
    assert math.isclose(a["curtain"], 2 * math.pi * 0.020 * 0.004)
    assert a["curtain"] < a["seat_bore"] < a["pipe"]                 # the lift is the throat
    with pytest.raises(ValueError):
        vc.ValveGeometry(seat_d=70.0).check()


def test_mesh_is_a_wedge_of_hexahedra(tmp_path):
    pytest.importorskip("gmsh")
    m = vc.mesh_valve(vc.ValveGeometry(), str(tmp_path), h_fine_mm=1.0, h_coarse_mm=5.0)
    assert m["cells"] > 500
    txt = open(m["msh"]).read()
    assert '"front"' in txt and '"inlet"' in txt and '"walls"' in txt


def test_pseudo_pure_gas_model():
    mc = vc.mixture_constants(DEFAULT_GAS, 298.15)
    assert 190 < mc["Tc"] < 215 and 4.4e6 < mc["Pc"] < 4.8e6 and 17 < mc["W"] < 19
    # ideal gas: no throttling effect at all
    assert abs(vc.model_throttle(mc, 70e5, 298.15, 20e5, eos="ideal") - 298.15) < 1e-6
    # Peng–Robinson pseudo-pure gas cools like the library's mixture Peng–Robinson (same order, a few K apart)
    T2 = vc.model_throttle(mc, 70e5, 298.15, 20e5, eos="PengRobinson")
    ref = throttle(DEFAULT_GAS, 70e5, 298.15, 20e5, eos="Peng-Robinson")["T2_K"]
    assert -35 < T2 - 298.15 < -15 and abs(T2 - ref) < 4.0
    # departure enthalpy vanishes at low pressure
    assert abs(float(vc.gas_enthalpy(300.0, 1.0, mc)) - float(vc.gas_enthalpy(300.0, 1.0, mc, eos="ideal"))) < 1.0


@pytest.mark.skipif(not vc.openfoam_available(), reason="OpenFOAM not installed")
def test_short_openfoam_run_reads_back(tmp_path):
    r = vc.run_valve_cfd(str(tmp_path), DEFAULT_GAS, 70, 25, 20, eos="ideal", iterations=60, h_fine_mm=1.0,
                         h_coarse_mm=5.0)
    assert r["status"] == "ok", r.get("log_tail")
    assert np.isfinite(r["T_out_C"]) and r["p_in_bar"] > 60 and abs(r["p_out_bar"] - 20) < 1e-6
    assert len(r["history"]["it"]) >= 2 and r["fields"]["T_C"].shape[0] == r["cells"]
