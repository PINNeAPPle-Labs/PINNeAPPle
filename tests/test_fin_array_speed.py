"""pinneapple_design.thermal.fin_array_speed (ported from the tests of PINNeAPPle-apps poc_surrogate_speed, with the
finite-volume fin pitch of pinneapple_design.thermal.fin_cell in place of the FEniCS subprocess)."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
from pinneapple_design.surrogate_inverse import heatsink_fins as hf  # noqa: E402
from pinneapple_design.thermal import fin_array_speed as fas  # noqa: E402


def test_curated_designs_are_manufacturable_and_at_55w_or_more():
    for spec in fas.CURATED_DESIGNS:
        assert all(v <= 0 for v in hf.constraint_violations(fas.design_of(spec)).values()), spec["name"]
        assert spec["power_w"] >= 55.0
    with pytest.raises(KeyError):
        fas.design_by_name("nope")


@pytest.fixture(scope="module")
def surrogate():
    return fas.train_surrogate(n_samples=3000, epochs=300)


def test_one_measured_design(surrogate):
    model, err = surrogate
    assert err < 0.15          # reduced training (3,000 samples, 300 epochs)
    m = fas.measure_design(model, fas.design_by_name("baseline"), timing_reps=20)
    assert m["constraints_ok"] and m["fea_heat_balance_error"] < 1e-6
    assert m["analytical_vs_fea_diff_pct"] < 1.0 and m["fea_grid_change"] < 1e-2
    assert m["surrogate_time_s"] < m["fea_wall_time_s"] and m["speedup_factor"] > 1
    s = fas.summarize([m])
    assert s["n_designs"] == 1 and s["max_agreement_diff_pct"] == m["surrogate_vs_fea_diff_pct"]


def test_thermal_field_profile():
    f = fas.thermal_field("baseline")
    assert f["theta_fin_base_c"] > 0 and f["base_temperature_c"] > f["t_ambient_c"] + f["theta_fin_base_c"]
