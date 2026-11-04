"""The 3-D multiphysics motor study: the reference chain, the axial thermal network, the surrogate and the
optimisation that runs without the FEA.

The FEA is the slow part, so the tests that need it use the smallest mesh the mesher accepts (``n_gap=192``) and one
design where one is enough; everything else -- the network, the design vector, the surrogate interface, the
optimiser, the model card -- runs without a field solve.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.machines.motor3d import (
    PRIUS,
    Motor3DDesign,
    Motor3DModel,
)
from pinneapple_simulation.electromagnetics.machines.motor3d.design import BOUNDS, NAMES, VARIABLES
from pinneapple_simulation.electromagnetics.machines.motor3d.thermal3d import AxialThermalModel
from pinneapple_simulation.electromagnetics.machines.thermal import MachineThermalData, machine_network

FAST = dict(n_gap=192, n_positions=6, n_segments=4, loss_steps=4)


# ------------------------------------------------------------------------------------------------ design vector
def test_design_vector_round_trip_and_units():
    assert len(NAMES) == len(VARIABLES) == 11
    d = Motor3DDesign.from_vector(np.linspace(0, 1, len(NAMES)))
    assert np.allclose(d.as_vector(), np.linspace(0, 1, len(NAMES)))
    assert Motor3DDesign.from_mapping(d.as_dict()) == d
    with pytest.raises(KeyError):
        Motor3DDesign.from_mapping({"W4": 0.019})
    # the bounds are the box of the variables, in the same order
    assert np.allclose(BOUNDS[:, 0], [v.low for v in VARIABLES])
    clipped = Motor3DDesign.from_vector(BOUNDS[:, 1] * 10).clipped()
    assert clipped.as_vector().max() <= BOUNDS[:, 1].max() + 1e-12


def test_baseline_is_the_prius_and_the_geometry_is_the_library_one():
    from pinneapple_simulation.electromagnetics.machines.geometry import prius_2004

    g = PRIUS.geometry()
    base = prius_2004()
    assert g.length == pytest.approx(base.length)
    assert g.rotor.W4 == pytest.approx(base.rotor.W4)
    assert PRIUS.sector_errors() == []


def test_skew_factor_is_the_classic_sinc():
    d = Motor3DDesign(skew_deg=0.0)
    assert d.skew_factor(4) == pytest.approx(1.0)
    d = Motor3DDesign(skew_deg=10.0)
    x = 0.5 * 4 * np.radians(10.0)
    assert d.skew_factor(4) == pytest.approx(np.sin(x) / x)
    assert Motor3DDesign(skew_deg=30.0).skew_factor(4) < d.skew_factor(4)


# -------------------------------------------------------------------------------------------- thermal network
def test_axial_network_with_one_segment_is_the_library_seven_node_model():
    """The axial network has to reduce to :func:`~.thermal.machine_network` when the stack is one segment: the same
    conductances, the same powers, the same temperatures."""
    g = PRIUS.geometry()
    end = 2 * Motor3DModel(**FAST).winding(PRIUS)["end_turn_m"]
    ref = machine_network(g, MachineThermalData(), end_length=end)
    P = [100.0, 200.0, 3000.0, 1500.0, 5.0, 20.0]
    T_ref = dict(zip(ref.names, ref.steady(P), strict=True))
    ax = AxialThermalModel(g, MachineThermalData(), n_segments=1, end_length=end)
    T = ax.network.steady(ax.powers(*P))
    for part in ("yoke", "teeth", "winding", "rotor", "magnets"):
        assert T[ax.index(part, 0)] == pytest.approx(T_ref[part], rel=1e-9, abs=1e-9)
    assert T[ax.index("end_winding")] == pytest.approx(T_ref["end_winding"], rel=1e-9)


def test_axial_network_is_hotter_inside_the_stack_than_at_the_ends():
    """The point of resolving the stack: the winding runs from a cooled end winding to a hot middle."""
    ax = AxialThermalModel(PRIUS.geometry(), MachineThermalData(), n_segments=6)
    T = ax.network.steady(ax.powers(200.0, 400.0, 8000.0, 3000.0, 20.0, 40.0))
    prof = np.array([T[ax.index("winding", k)] for k in range(6)])
    assert prof[np.argmax(prof)] not in (0, 5)
    assert prof.max() > prof[0] > prof[5] - 1e-9
    assert np.all(np.diff(prof[:3]) > 0) and np.all(np.diff(prof[3:]) < 0)


def test_more_coolant_flow_is_cooler():
    model = Motor3DModel(**FAST)
    hot = model.thermal_model(Motor3DDesign(flow_L_min=0.5))
    cold = model.thermal_model(Motor3DDesign(flow_L_min=12.0))
    P = [100.0, 200.0, 6000.0, 3000.0, 5.0, 20.0]
    Tw_hot = hot.network.steady(hot.powers(*P))[hot.index("winding", 1)]
    Tw_cold = cold.network.steady(cold.powers(*P))[cold.index("winding", 1)]
    assert Tw_cold < Tw_hot - 5.0


# --------------------------------------------------------------------------------------------- reference model
@pytest.fixture(scope="module")
def baseline():
    return Motor3DModel(**FAST).solve(PRIUS)


def test_reference_model_returns_the_physics_of_the_prius(baseline):
    r = baseline
    assert r["valid"], r.get("errors")
    # torque of the 2004 Prius at 250 A and 50 deg, corrected for the magnet temperature and the (zero) skew
    assert 200.0 < r["torque_Nm"] < 380.0
    assert 0.0 <= r["ripple"] < 0.5
    assert 0.5 < r["efficiency"] < 1.0
    assert r["T_winding_C"] > r["T_magnet_C"] > 60.0          # the copper is the hottest part of this design
    assert 0.0 < r["demag_ratio"] < 1.0
    assert 0.5 < r["magnet_mass_kg"] < 2.0
    assert 10.0 < r["active_mass_kg"] < 60.0
    for k in ("torque_slice_Nm", "T_winding_profile_C", "T_magnet_profile_C"):
        assert r[k].shape == (4,)
    # the two independent torques of one solution agree (Maxwell stress against dq flux) to a few per cent
    ex = r["extra"]
    assert abs(ex["i_d"]) > 0 and abs(ex["i_q"]) > 0
    assert ex["v_limit_V"] == pytest.approx(500.0 / np.sqrt(3))


def test_skew_trades_torque_for_ripple():
    model = Motor3DModel(**FAST)
    straight = model.solve(Motor3DDesign(skew_deg=0.0))
    skewed = model.solve(Motor3DDesign(skew_deg=12.0))
    assert skewed["torque_Nm"] < straight["torque_Nm"]              # the fundamental skew factor costs torque
    assert skewed["ripple"] <= straight["ripple"] + 1e-6            # and the slot harmonics cancel
    assert skewed["torque_slice_Nm"].std() > 0.0


def test_coolant_flow_changes_the_operating_temperature_of_the_model():
    model = Motor3DModel(**FAST)
    low = model.solve(Motor3DDesign(flow_L_min=0.5))
    high = model.solve(Motor3DDesign(flow_L_min=12.0))
    assert high["T_winding_C"] < low["T_winding_C"]
    assert high["T_magnet_C"] < low["T_magnet_C"]
    # the colder magnets are the stronger ones, so the torque is higher
    assert high["torque_Nm"] > low["torque_Nm"]


def test_evaluate_returns_nan_for_designs_the_mesher_refuses():
    model = Motor3DModel(**FAST)
    bad = Motor3DDesign(W4=0.0140, H3=0.0075, H0=0.0085, W0=0.0450, H1=0.0010)
    out = model.evaluate(np.vstack([PRIUS.as_vector(), bad.as_vector()]))
    assert out["_valid"][0]
    assert np.isnan(out["torque_Nm"][1]) == (not out["_valid"][1])
    assert out["torque_slice_Nm"].shape == (2, 4)


# -------------------------------------------------------------------------------------------------- surrogate
def _synthetic_dataset(n: int = 160, seed: int = 0):
    """A dataset with smooth analytic 'outputs', so the surrogate's machinery is tested without the FEA."""
    from pinneapple_simulation.electromagnetics.machines.motor3d.dataset import MotorDataset
    from pinneapple_simulation.electromagnetics.machines.motor3d.model import PROFILES, SCALARS

    rng = np.random.default_rng(seed)
    X = BOUNDS[:, 0] + (BOUNDS[:, 1] - BOUNDS[:, 0]) * rng.random((n, len(NAMES)))
    u = (X - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    Y: dict[str, np.ndarray] = {}
    for i, k in enumerate(SCALARS):
        Y[k] = 10.0 + 5.0 * np.sin(2.0 * u[:, i % len(NAMES)] * np.pi) + 2.0 * u[:, (i + 3) % len(NAMES)]
    Y["torque_Nm"] = 200.0 + 100.0 * u[:, 0] + 30.0 * np.cos(3.0 * u[:, 6])
    Y["efficiency"] = 0.8 + 0.1 * u[:, 1]
    Y["demag_ratio"] = 0.5 + 0.2 * u[:, 2]
    Y["ripple"] = 0.1 + 0.05 * u[:, 6]
    for k in PROFILES:
        Y[k] = np.stack([10.0 + 5.0 * np.sin(2.0 * np.pi * u[:, j % len(NAMES)]) for j in range(4)], axis=1)
    return MotorDataset(X=X, Y=Y, valid=np.ones(n, bool), meta={"seed": seed, "n_cases": n})


def test_surrogate_trains_and_reports_held_out_errors(tmp_path):
    from pinneapple_simulation.electromagnetics.machines.motor3d.surrogate import MotorSurrogate3D, SurrogateConfig

    ds = _synthetic_dataset()
    s = MotorSurrogate3D(SurrogateConfig(ensemble=2, epochs=180, hidden=(64, 64), patience=200)).fit(ds)
    card = s.card()
    assert card["n_train"] + card["n_val"] == len(ds)
    assert card["heldout"]["torque_Nm"]["r2"] > 0.8          # a smooth function is learnable
    pred = s.predict(PRIUS.as_vector()[None, :])
    assert pred["torque_Nm"][0].shape == (1,)
    assert pred["T_winding_profile_C"][0].shape == (1, 4)
    assert np.all(pred["torque_Nm"][1] >= 0.0)               # the spread is a standard deviation
    # save/load keeps the predictions and the card
    s.save(tmp_path / "m", version="test")
    again = MotorSurrogate3D.load(tmp_path / "m")
    a, b = s.predict(ds.X[:5]), again.predict(ds.X[:5])
    assert np.allclose(a["torque_Nm"][0], b["torque_Nm"][0])
    assert again.card()["n_val"] == card["n_val"]


def test_surrogate_refuses_weights_that_do_not_match_the_card(tmp_path):
    from pinneapple_simulation.electromagnetics.machines.motor3d.surrogate import MotorSurrogate3D, SurrogateConfig

    ds = _synthetic_dataset(40)
    s = MotorSurrogate3D(SurrogateConfig(ensemble=1, epochs=20, hidden=(16,), patience=10)).fit(ds)
    s.save(tmp_path / "m", version="test")
    blob = tmp_path / "m" / "weights.pt"
    blob.write_bytes(blob.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="SHA-256"):
        MotorSurrogate3D.load(tmp_path / "m")


def test_out_of_box_and_extrapolation():
    from pinneapple_simulation.electromagnetics.machines.motor3d.surrogate import MotorSurrogate3D, SurrogateConfig

    ds = _synthetic_dataset(60)
    s = MotorSurrogate3D(SurrogateConfig(ensemble=1, epochs=20, hidden=(16,), patience=10)).fit(ds)
    assert s.out_of_box(PRIUS.as_vector()[None, :]) == []
    far = PRIUS.as_vector().copy()
    far[NAMES.index("W4")] = BOUNDS[0, 1] * 1.5
    assert "W4" in s.out_of_box(far[None, :])
    assert s.extrapolation(far[None, :])[0] > s.extrapolation(PRIUS.as_vector()[None, :])[0]


# ------------------------------------------------------------------------------------------------- optimiser
def test_optimisation_runs_on_the_surrogate_with_zero_fea_calls():
    from pinneapple_simulation.electromagnetics.machines.motor3d.optimize import (
        SurrogateEvaluator,
        design_problem,
        optimize,
    )
    from pinneapple_simulation.electromagnetics.machines.motor3d.surrogate import MotorSurrogate3D, SurrogateConfig

    ds = _synthetic_dataset(120)
    s = MotorSurrogate3D(SurrogateConfig(ensemble=1, epochs=60, hidden=(32, 32), patience=30)).fit(ds)
    problem = design_problem(evaluator=SurrogateEvaluator(s))
    out = optimize(problem, budget=80, seed=0)
    assert out["status"] == "ok"
    assert out["n_evaluations"] >= 80
    assert out["n_reference_solves"] == 0
    assert out["designs_per_second"] > 0
    for row in out["front"]:
        for name in NAMES:
            assert BOUNDS[NAMES.index(name), 0] - 1e-9 <= row[name] <= BOUNDS[NAMES.index(name), 1] + 1e-9


def test_verification_scores_a_front_against_a_reference():
    from pinneapple_simulation.electromagnetics.machines.motor3d.optimize import (
        SurrogateEvaluator,
        design_problem,
        optimize,
        verify_front,
    )
    from pinneapple_simulation.electromagnetics.machines.motor3d.surrogate import MotorSurrogate3D, SurrogateConfig

    ds = _synthetic_dataset(120)
    s = MotorSurrogate3D(SurrogateConfig(ensemble=1, epochs=60, hidden=(32, 32), patience=30)).fit(ds)

    class Identity(SurrogateEvaluator):
        """A 'reference' that returns the same numbers, so the verification path is tested without an FEA solve."""

        fidelity = "reference"

        def evaluate(self, X):
            return {**super().evaluate(X), "_valid": np.ones(len(np.atleast_2d(X)), bool)}

    out = optimize(design_problem(evaluator=SurrogateEvaluator(s)), budget=60, seed=0)
    ver = verify_front(out, reference=Identity(s), n=3)
    assert ver["status"] == "ok"
    assert ver["n_checked"] == min(3, len(out["front"]))
    # the network sees a batch of three here and a batch of eighty during the search: the same weights give the
    # same numbers up to the last bits of the BLAS kernel, not to the last bit
    assert all(e["mean_rel"] < 1e-6 for e in ver["errors"].values())
    assert ver["still_feasible"] == ver["n_checked"]


# ------------------------------------------------------------------------------------------------- pretrained
def test_pretrained_model_ships_with_a_card_that_matches_its_weights():
    from pinneapple_simulation.electromagnetics.machines.motor3d.surrogate import load_pretrained, pretrained_path

    folder = pretrained_path()
    if not (folder / "weights.pt").exists():
        pytest.skip("the pretrained motor surrogate is not shipped in this checkout")
    s = load_pretrained()
    card = json.loads((folder / "card.json").read_text())
    assert card["n_total"] >= 200
    assert card["inputs"] == list(NAMES)
    mean, _ = s.predict_one(PRIUS)
    assert 150.0 < mean["torque_Nm"] < 450.0
    assert 0.0 < mean["demag_ratio"] < 1.0
    assert s.out_of_box(PRIUS.as_vector()[None, :]) == []
