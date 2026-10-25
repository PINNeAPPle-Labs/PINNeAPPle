"""pinneapple_design.thermal.fin_array_speed (ported from the tests of PINNeAPPle-apps poc_surrogate_speed, with the
finite-volume reference in place of the FEniCS subprocess)."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
from pinneapple_design.thermal import fin_array_speed as fas  # noqa: E402


def test_curated_designs_are_manufacturable_and_at_55w_or_more():
    for spec in fas.CURATED_DESIGNS:
        d = fas.FinArrayDesign(spec["fin_count"], spec["fin_height_m"], spec["fin_thickness_m"])
        assert all(v <= 1e-12 for v in fas.constraint_violations(d).values()), spec["name"]
        assert spec["power_w"] >= 55.0
    with pytest.raises(KeyError):
        fas.design_by_name("nope")


@pytest.fixture(scope="module")
def surrogate():
    return fas.train_surrogate(n_samples=3000, epochs=300)


def test_surrogate_validation_and_one_measured_design(surrogate):
    sur, err = surrogate
    assert err < 0.15          # reduced training (3,000 samples, 300 epochs); the full one gave 5.7 % here
    m = fas.measure_design(sur, fas.design_by_name("baseline"), timing_reps=20)
    assert m["constraints_ok"] and m["fea_heat_balance_error"] < 1e-8
    assert m["analytical_vs_fea_diff_pct"] < 1.0 and m["fea_gci"] < 1e-2
    assert m["surrogate_time_s"] < m["fea_wall_time_s"] and m["speedup_factor"] > 1
    s = fas.summarize([m])
    assert s["n_designs"] == 1 and s["max_agreement_diff_pct"] == m["surrogate_vs_fea_diff_pct"]


def test_thermal_field_profile():
    f = fas.thermal_field("baseline")
    assert f["theta_fin_base_c"] > 0 and f["base_temperature_c"] > f["t_ambient_c"] + f["theta_fin_base_c"]
