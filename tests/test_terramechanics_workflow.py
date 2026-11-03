"""The terramechanics workflow (#1199): soil identified from plate and shear tests through the inverse workflow,
single-wheel curves, the minimum-mass wheel for a drawbar pull, re-verification and the published single-wheel tests
of Ding et al. (2011); negative controls (one plate width, an impossible requirement)."""
from __future__ import annotations

import math

import pytest

pytest.importorskip("scipy")

from pinneapple_simulation.terramechanics import Wheel, WongReeceModel, get_soil  # noqa: E402
from pinneapple_simulation.terramechanics.workflow import (  # noqa: E402
    PUBLISHED_BAND,
    bekker_pressure,
    design_wheel,
    identify_soil,
    janosi_shear,
    synthetic_soil_tests,
    terramechanics_workflow,
    wheel_mass,
)


def _status(r):
    return {c["code"]: c["status"] for c in r["checks"]}


def test_closed_forms():
    assert bekker_pressure(0.1, 0.02, 1000.0, 5e5, 1.0) == pytest.approx((1e4 + 5e5) * 0.02)
    assert janosi_shear(1e4, 1e9, 500.0, 30.0, 0.01) == pytest.approx(500 + 1e4 * math.tan(math.radians(30)))
    assert janosi_shear(1e4, 0.0, 500.0, 30.0, 0.01) == 0.0
    # rim 2 pi R b t, sides 2 pi R^2 t fill, no lugs, hub
    m = wheel_mass(0.2, 0.1)
    assert m == pytest.approx(2700 * (2 * math.pi * 0.2 * 0.1 * 0.002 + 2 * math.pi * 0.04 * 0.002 * 0.3) + 0.3)


def test_identification_recovers_the_soil_that_made_the_tests():
    soil = get_soil("Ding2011-TYII")
    t = synthetic_soil_tests(soil, seed=0)
    r = identify_soil(t["plates"], t["shear"], base=soil)
    assert all(c["status"] == "pass" for c in r["checks"]), [c for c in r["checks"] if c["status"] != "pass"]
    s = r["soil"]
    # measured: k_c 15634 (truth 15600), k_phi 2.408e6 (2.4074e6), n 1.101 (1.10); c 194 (250), phi 32.45 (31.9)
    assert s.k_c == pytest.approx(soil.k_c, rel=0.05)
    assert s.k_phi == pytest.approx(soil.k_phi, rel=0.03)
    assert s.n0 == pytest.approx(soil.n0, rel=0.02)
    assert s.phi_deg == pytest.approx(soil.phi_deg, abs=1.0)
    assert set(r["identified"]) == {"k_c", "k_phi", "n0", "c", "phi_deg", "K"}


def test_one_plate_width_cannot_separate_k_c_from_k_phi():
    soil = get_soil("Ding2011-TYII")
    t = synthetic_soil_tests(soil, widths=(0.1,), seed=0)
    r = identify_soil(t["plates"], None, base=soil)
    st = _status(r)
    assert st["soil.pressure.structural"] == "fail"
    assert r["pressure"]["estimated"] is False
    assert r["soil"].k_c == soil.k_c                      # nothing estimated: the base soil is kept


def test_workflow_designs_and_reverifies_the_wheel():
    r = terramechanics_workflow()
    st = _status(r)
    assert r["verdict"] == "pass", r["failed"]
    for code in ("soil.truth_recovered", "wheel.load_balance", "wheel.quadrature", "wheel.pull_rises_with_slip",
                 "design.feasible", "verify.full_model_agrees", "verify.pull", "verify.sinkage",
                 "verify.operating_slip", "design.pull_limit_active", "published.ding2011_smooth",
                 "published.ding2011_grousered"):
        assert st[code] == "pass", code
    d, v = r["design"], r["verified"]
    # measured: R 0.138 m, b 0.080 m (lower bound), no lugs, 0.870 kg; the 0.25 x 0.20 m wheel weighs 3.15 kg
    assert d["mass_kg"] < r["wheel"]["mass_kg"]
    assert v["drawbar_pull_N"] >= r["requirement"]["drawbar_pull_N"] * (1 - 1e-3)
    assert v["sinkage_m"] <= r["requirement"]["max_sinkage_m"]
    assert abs(v["load_carried_N"] - 80.0) < 1e-6
    for row in r["published"]:
        for k, e in row["rmse_rel"].items():
            assert e <= PUBLISHED_BAND[k]          # measured: DP 9.6 % / 5.0 %, torque 4.6 % / 4.2 %, sinkage 24-29 %


def test_lighter_wheel_gives_less_pull_so_the_limit_is_active():
    soil = get_soil("Ding2011-TYII")
    d = design_wheel(soil, load=80.0, g=9.81, required_pull=20.0, max_sinkage=0.03, grid=5)
    assert d["feasible"]
    small = WongReeceModel(soil, Wheel(radius=0.9 * d["design"]["radius"], width=d["design"]["width"]), g=9.81)
    assert float(small.steady_state(80.0, 0.3)["drawbar_pull"]) < 20.0


def test_impossible_requirement_is_reported_not_designed():
    r = terramechanics_workflow(identify="none", required_pull=200.0, published=())
    st = _status(r)
    assert st["design.feasible"] == "fail"
    assert r["verdict"] == "fail"
    assert "verified" not in r
