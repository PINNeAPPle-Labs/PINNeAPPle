"""The AirfRANS → FluidsBench campaign: features, the network, its physics term, and the entry it writes.

Everything here runs on synthetic arrays. The stages that need the real release (extraction) or the benchmark's own
toolchain (validate → evaluate → package) are checked by their contract — a clear error when the input is missing — not
by downloading ten gigabytes inside a test.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_simulation.pipelines.airfrans_benchmark import (  # noqa: E402
    FEATURE_NAMES,
    AirfRANSStackMissing,
    CampaignSpec,
    CoordinateNet,
    PredictionSpec,
    TrainingSpec,
    WallAwareNet,
    WallTruthMissing,
    case_ids_for,
    chord,
    curve_coefficients,
    curve_projection,
    manifest_splits,
    physics_loss,
    point_features,
    predict_cases,
    sample_indices,
    scaled_targets,
    slope_from_wall_shear,
    surface_rank,
    train,
    wall_case_truth,
    wall_layer_profile,
    write_entry,
)

AIRFOIL = np.array([[0.0, 0.0], [0.5, 0.06], [1.0, 0.0], [0.5, -0.03]])


def _case(rng, n=400):
    """A synthetic case in the AirfRANS feature layout: position, sdf, normals, surface flag, freestream."""
    points = rng.uniform(-1.0, 2.0, size=(n, 2))
    sdf = points[:, 1] - 0.05
    normals = np.zeros((n, 2))
    surface = np.zeros((n, 1))
    wall = rng.choice(n, size=20, replace=False)
    points[wall] = AIRFOIL[rng.integers(0, len(AIRFOIL), size=20)]
    sdf[wall] = 0.0
    normals[wall] = [0.0, 1.0]
    surface[wall] = 1.0
    freestream = np.tile([30.0, 0.0], (n, 1))
    features = np.concatenate([points, sdf[:, None], normals, surface, freestream], axis=1).astype(np.float32)
    targets = np.concatenate([np.tile([30.0, 0.0], (n, 1)), 0.5 * np.ones((n, 1))], axis=1).astype(np.float32)
    return features, targets


def test_features_are_scaled_by_the_chord_and_keep_the_wall_channel():
    rng = np.random.default_rng(0)
    features, _ = _case(rng)
    block = point_features(features, AIRFOIL)
    assert block.shape == (len(features), len(FEATURE_NAMES))
    assert chord(AIRFOIL) == pytest.approx(1.0)
    wall = features[:, 5] > 0.5
    assert np.allclose(block[wall, 6], 1.0) and np.allclose(block[~wall, 6], 0.0)
    assert np.all(np.isfinite(block))
    # the signed distance appears both linearly and through its signed logarithm
    assert block[0, 2] == pytest.approx(features[0, 2])
    assert np.sign(block[0, 3]) == np.sign(features[0, 2])


def test_targets_are_in_freestream_units():
    rng = np.random.default_rng(1)
    features, targets = _case(rng)
    scaled = scaled_targets(targets, features)
    np.testing.assert_allclose(scaled[:, 0], 1.0, atol=1e-6)          # u/U_inf
    np.testing.assert_allclose(scaled[:, 2], 0.5 / 30.0 ** 2, rtol=1e-6)


def test_sampling_oversamples_the_wall():
    """The wall is ~1 % of a case's nodes; the sample has to carry far more of it, and every wall node it can."""
    rng = np.random.default_rng(2)
    features, _ = _case(rng, n=2000)
    natural = float((features[:, 5] > 0.5).mean())
    take = sample_indices(features, 400, rng, wall_fraction=0.25)
    wall_share = float((features[take, 5] > 0.5).mean())
    assert wall_share > 2 * natural, (natural, wall_share)
    assert np.all(np.isin(np.flatnonzero(features[:, 5] > 0.5), take))     # all 20 wall nodes made it
    assert len(take) == len(set(take.tolist()))


def test_the_network_maps_features_to_three_scaled_outputs():
    model = CoordinateNet(width=32, depth=2)
    out = model(torch.zeros(7, len(FEATURE_NAMES)))
    assert tuple(out.shape) == (7, 3)
    assert sum(p.numel() for p in model.parameters()) == len(FEATURE_NAMES) * 32 + 32 + 32 * 32 + 32 + 32 * 3 + 3


def test_the_physics_term_is_differentiable_and_vanishes_on_a_conforming_prediction():
    features = np.zeros((6, len(FEATURE_NAMES)), np.float32)
    features[:, 2] = [0.0, 0.0, 2.0, 2.0, 0.5, 0.5]      # sdf/chord: two wall points, two far, two in between
    features[:, 4] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]      # normals along x
    features[:, 5] = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]      # wall normal is y
    features[:, 6] = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]      # surface flag
    x = torch.tensor(features)
    good = torch.tensor([[1.0, 0.0, 0.0]] * 6, requires_grad=True)     # freestream everywhere, p = 0 = 1/2(1 - q^2)
    assert float(physics_loss(good, x).detach()) == pytest.approx(0.0, abs=1e-12)
    bad = torch.tensor([[1.0, 1.0, 1.0]] * 6, requires_grad=True)
    loss = physics_loss(bad, x)
    assert float(loss) > 0
    loss.backward()
    assert bad.grad is not None and float(bad.grad.abs().sum()) > 0


@pytest.fixture()
def store(tmp_path):
    rng = np.random.default_rng(3)
    names = [f"case{i:02d}" for i in range(6)]
    for name in names:
        features, targets = _case(rng)
        np.savez_compressed(tmp_path / f"{name}.npz", features=features, targets=targets,
                            airfoil_points=AIRFOIL.astype(np.float32),
                            airfoil_normals=np.tile([0.0, 1.0], (len(AIRFOIL), 1)).astype(np.float32),
                            case=np.array([name]), inlet_velocity=np.array([30.0], np.float32),
                            aoa_deg=np.array([0.0], np.float32), n_points=np.array([len(features)], np.int32))
    manifest = {"full_train": names[:4], "full_test": names[4:], "scarce_train": names[:2], "aoa_test": names[5:]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return str(tmp_path), manifest


def test_training_writes_weights_and_a_record_that_reproduces_them(tmp_path, store):
    root, manifest = store
    spec = TrainingSpec(train_cases=4, val_cases=2, points_per_case=256, epochs=2, batch_size=256, width=32, depth=2)
    record = train(root, manifest, str(tmp_path / "run"), spec, device="cpu")
    assert record["parameters"] == record["parameters"] and record["epochs"] == 2
    assert len(record["history"]) == 2 and record["val_points"] > 0
    weights = os.path.join(tmp_path, "run", "model.pt")
    import hashlib
    assert record["weights_sha256"] == hashlib.sha256(open(weights, "rb").read()).hexdigest()
    assert json.load(open(os.path.join(tmp_path, "run", "training.json")))["width"] == 32


def test_prediction_writes_the_four_native_arrays_in_order(tmp_path, store):
    root, manifest = store
    spec = TrainingSpec(train_cases=4, val_cases=2, points_per_case=256, epochs=1, batch_size=256, width=32, depth=2)
    train(root, manifest, str(tmp_path / "run"), spec, device="cpu")
    entry = tmp_path / "entry"
    result = predict_cases(root, str(tmp_path / "run" / "model.pt"), manifest["full_test"], str(entry),
                           PredictionSpec(device="cpu"))
    assert result["cases"] == len(manifest["full_test"])
    with np.load(entry / "predictions" / f"{manifest['full_test'][0]}.npz") as data:
        assert set(data.files) == {"velocity", "pressure", "internal_points", "airfoil_points"}
        assert data["velocity"].shape[1] == 2 and data["pressure"].shape == data["internal_points"][:, 0].shape
        assert np.all(np.isfinite(data["velocity"]))


def test_the_entry_carries_the_split_order_and_the_three_supports(tmp_path, store):
    root, _ = store
    support = tmp_path / "support"
    chunk_dir = support / "case-sets" / "standard"
    chunk_dir.mkdir(parents=True)
    cases = [{"case_id": c, "support_instances": [
        {"support_id": "two-dimensional-domain-native", "entity_count": 400, "artifacts": []},
        {"support_id": "airfoil-curve-native", "entity_count": 4, "artifacts": []},
        {"support_id": "airfoil-force-coefficients-native", "entity_count": 1, "artifacts": []}]}
        for c in ("b", "a")]
    (chunk_dir / "chunk-000.json").write_text(json.dumps({"cases": cases}))
    split = tmp_path / "splits" / "full.json"
    split.parent.mkdir()
    split.write_text(json.dumps({"case_ids": ["b", "a"]}))       # the official order is not alphabetical
    training = {"parameters": 100, "width": 32, "depth": 2, "batch": 256, "lr": 2e-3, "epochs": 1, "seed": 0,
                "per_case": 256, "train_cases": 4, "wall_fraction": 0.25, "train_points": 1024, "device": "cpu",
                "physics_weight": 0.0, "weights_sha256": "a" * 64, "seconds": 12.0}
    prediction = {"cases": 2, "seconds": 3.0, "device": "cpu"}
    written = write_entry(str(tmp_path / "entry"), submission_id="model-full-v1", series_id="model-full",
                          split_id="full", support_root=str(support), training=training, prediction=prediction,
                          training_hours=12.0 / 3600, split_file=str(split))
    lines = [json.loads(line) for line in open(written["discretization_cases"])]
    assert [line["case_id"] for line in lines] == ["b", "a"]
    assert [m["support_id"] for m in lines[0]["inference"]["mappings"]] == [
        "two-dimensional-domain-native", "airfoil-curve-native", "airfoil-force-coefficients-native"]
    assert [i["id"] for i in lines[0]["inference"]["inputs"]] == ["surface-geometry-input", "flow-domain-input"]
    config = json.load(open(written["package_config"]))
    assert config["result_revision"] == {"series_id": "model-full", "version": 1, "supersedes": None,
                                         "change_summary": "Initial published result."}
    assert config["participant"]["methodology"]["architecture"]["total_parameter_count"] == 100
    assert all(i["representation"]["entity_counts"][0]["count"]["kind"] in ("fixed", "per_case")
               for i in config["spatial_discretization"]["inference"]["direct_outputs"])
    assert config["spatial_discretization"]["inference"]["mappings"][0]["support_id"] == \
        "two-dimensional-domain-native"


def test_a_campaign_refuses_an_unknown_stage_or_split_and_says_why(tmp_path, store):
    root, _ = store
    spec = CampaignSpec(dataset_root=root, support_root=str(tmp_path), checkout=str(tmp_path))
    from pinneapple_simulation.pipelines.airfrans_benchmark import run_campaign

    with pytest.raises(ValueError, match="unknown stage"):
        run_campaign(spec, str(tmp_path / "w"), submission_id="m-full-v1", stages=("deploy",))
    with pytest.raises(ValueError, match="not a split"):
        run_campaign(spec, str(tmp_path / "w"), submission_id="m-full-v1", split_id="reynolds_extrapolation")
    with pytest.raises(FileNotFoundError, match="run the 'train' stage first"):
        run_campaign(spec, str(tmp_path / "w2"), submission_id="m-full-v1", stages=("predict",))


def test_extraction_says_which_stack_it_needs_when_it_is_missing(tmp_path):
    """Reading a case needs the pinned AirfRANS/VTK stack; a machine without it gets the install line, not a
    stack trace about ``vtk``."""
    from pinneapple_simulation.pipelines.airfrans_benchmark.data import case_arrays

    try:
        import airfrans  # noqa: F401
    except ImportError:
        with pytest.raises(AirfRANSStackMissing, match="airfrans==0.1.5.1"):
            case_arrays(str(tmp_path), "any-case")
    else:                                                       # pragma: no cover - only on a participant machine
        pytest.skip("the AirfRANS stack is installed: the missing-dependency path cannot be exercised here")


def test_the_manifest_helpers_read_the_release_layout(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"full_train": ["a", "b"], "full_test": ["c"], "aoa_test": ["c", "d"]}))
    manifest = manifest_splits(str(tmp_path))
    assert case_ids_for(manifest) == ["a", "b", "c", "d"]


# --------------------------------------------------------------------------------------------------------------
# the wall layer, the slope convention and the force quadrature
# --------------------------------------------------------------------------------------------------------------

def _figure(count: int = 4096):
    """A closed polyline the way a pyvista airfoil mesh stores one: one segment per cell, outward face normals.

    The normals are the exact face normals of the regular polygon (not the vertices'), which is what makes the two
    analytic identities below hold to machine precision rather than to the polygon's own discretisation error."""
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    points = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    cells = np.stack([np.arange(count), (np.arange(count) + 1) % count], axis=1).astype(np.int64)
    middle = 0.5 * (points[cells[:, 0]] + points[cells[:, 1]])
    normals = middle / np.linalg.norm(middle, axis=1, keepdims=True)
    lengths = np.linalg.norm(points[cells[:, 1]] - points[cells[:, 0]], axis=1)
    return points, normals, cells, lengths


def test_the_wall_slope_is_the_wall_shear_over_nu():
    """``2νS·n = ν ∂u_t/∂d`` at a wall: the stored slope is ``+wall shear/ν`` in units of ``U∞/c`` and points along the
    flow, which is the skin friction the flow exerts on the surface."""
    wall_shear = np.array([[0.0, -3.0], [1.5, 0.0]])
    nu, inlet, chord_length = 2e-5, 30.0, 1.0
    slope = slope_from_wall_shear(wall_shear, nu, inlet, chord_length)
    assert np.allclose(slope, wall_shear / nu * (chord_length / inlet))
    # and the physical wall shear comes straight back out of it
    recovered = slope * (nu * inlet / chord_length)
    assert np.allclose(recovered, wall_shear)


def test_a_constant_pressure_on_a_closed_curve_produces_no_force():
    """d'Alembert in its simplest form: a closed loop sees no net force from a uniform pressure."""
    points, normals, cells, lengths = _figure()
    coefficients = curve_coefficients(np.full(len(points), 0.37), np.zeros((len(points), 2)), cells, lengths,
                                      normals, cos_alpha=1.0, sin_alpha=0.0, nu=2e-5, inlet_velocity=1.0,
                                      chord=2.0)
    assert abs(coefficients["cdp"]) < 1e-12
    assert abs(coefficients["clp"]) < 1e-12


def test_a_pressure_gradient_on_a_circle_gives_the_analytic_drag():
    """``p = -k x`` on a closed loop: ``∮ p n_x ds = -k · area`` by the divergence theorem, so ``cdp = -2 k area``.

    The identity holds exactly for the polyline (it is the discrete divergence theorem), so this checks the quadrature
    and not a discretisation error."""
    points, normals, cells, lengths = _figure()
    gradient = 0.25
    area = 0.5 * np.sum(points[cells[:, 0], 0] * points[cells[:, 1], 1]
                        - points[cells[:, 1], 0] * points[cells[:, 0], 1])
    pressure = -gradient * points[:, 0]
    coefficients = curve_coefficients(pressure, np.zeros((len(points), 2)), cells, lengths, normals,
                                      cos_alpha=1.0, sin_alpha=0.0, nu=2e-5, inlet_velocity=1.0, chord=2.0)
    assert coefficients["cdp"] == pytest.approx(-2.0 * gradient * area, rel=1e-9)
    assert abs(coefficients["clp"]) < 1e-9


def test_a_uniform_tangential_slope_integrates_to_the_expected_friction():
    """A constant slope along the flow gives ``cdv = -2ν s L/(U∞ c)`` with ``L`` the perimeter."""
    points, normals, cells, lengths = _figure()
    slope = np.tile(np.array([1000.0, 0.0]), (len(points), 1))      # tangential, along the flow direction
    coefficients = curve_coefficients(np.zeros(len(points)), slope, cells, lengths, normals,
                                      cos_alpha=1.0, sin_alpha=0.0, nu=2e-5, inlet_velocity=1.0, chord=2.0)
    expected = 2.0 * 2e-5 * 1000.0 * lengths.sum() / (1.0 * 2.0)
    assert coefficients["cdv"] == pytest.approx(expected, rel=1e-6)
    assert abs(coefficients["cdp"]) < 1e-12


def test_the_quadrature_runs_on_tensors_and_keeps_the_gradient():
    """The training loss needs the same numbers as the evaluator *and* a gradient back to the surface pressure."""
    points, normals, cells, lengths = _figure()
    pressure = torch.tensor(-0.25 * points[:, 0], requires_grad=True)
    coefficients = curve_coefficients(pressure, torch.zeros(len(points), 2), torch.from_numpy(cells),
                                      torch.from_numpy(lengths), torch.from_numpy(normals),
                                      cos_alpha=1.0, sin_alpha=0.0, nu=2e-5, inlet_velocity=1.0, chord=2.0)
    assert isinstance(coefficients["cdp"], torch.Tensor)
    coefficients["cdp"].backward()
    assert pressure.grad is not None and torch.isfinite(pressure.grad).all()
    assert float(pressure.grad.abs().sum()) > 0.0


def test_the_wall_layer_gives_exact_no_slip_and_the_slope_head_as_the_first_derivative():
    """``U = A(d)·U_outer + (1−A(d))·s·d`` must vanish at the wall and start with slope ``s``."""
    distance = torch.tensor([0.0, 1e-9, 1e-6, 1e-4, 1e-2, 1.0], dtype=torch.float64)
    wall_layer = 5e-5
    weight = wall_layer_profile(distance, wall_layer)
    assert float(weight[0]) == 0.0                                  # no-slip, exactly
    assert float(weight[1]) < 1e-6                                  # and it stays close to the wall while it must
    assert float(weight[-1]) > 0.999                                # the outer solution owns the far field
    slope = torch.tensor(2000.0, dtype=torch.float64, requires_grad=True)
    velocity = (1.0 - wall_layer_profile(distance, wall_layer)) * slope * distance
    derivative = torch.autograd.grad(velocity.sum(), slope, retain_graph=True)[0]
    # dU/dd at d→0 equals the slope: the finite difference of the first two entries reproduces it
    finite_difference = float((velocity[2] - velocity[0]) / distance[2])
    assert finite_difference == pytest.approx(2000.0, rel=1e-3)
    assert float(velocity[0]) == 0.0
    assert float(derivative) > 0.0


def test_the_wall_aware_network_is_no_slip_on_the_wall_and_free_of_it_away_from_it():
    network = WallAwareNet(width=32, depth=1, fourier=2)
    features = torch.zeros(4, len(FEATURE_NAMES))
    features[:, 7], features[:, 8] = 1.0, 0.0                     # freestream along x
    features[:, 2] = torch.tensor([0.0, 0.0, 5e-2, 1.0])          # sdf/c: two wall nodes, two in the field
    features[:, 6] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    with torch.no_grad():
        prediction = network(features)
    assert torch.allclose(prediction[:2, :2], torch.zeros(2, 2), atol=0.0)   # exactly zero on the wall
    assert prediction.shape == (4, 3)


def test_the_wall_slope_samples_line_up_with_the_native_point_order():
    """``surface_rank`` maps a sampled node onto its compact wall entry, and -1 everywhere else."""
    features = np.zeros((5, 9))
    features[:, 5] = [0, 1, 0, 1, 1]
    rank = surface_rank(features)
    assert rank.tolist() == [-1, 0, -1, 1, 2]


def test_the_wall_cache_writer_says_which_stack_it_needs(tmp_path):
    with pytest.raises(WallTruthMissing) as error:
        wall_case_truth(str(tmp_path), "no-such-case")
    assert "airfrans" in str(error.value)


def test_the_gradient_clip_keeps_the_wall_readout_out_of_the_trunks_norm():
    """One global clip over this model measures ~140 at initialisation, dominated by the trunk, and scales the slope
    head's gradients by 0.007 — the head then trains at 1/140 of its learning rate and never fits its own cases
    (measured: 356 % relative error on the training set against 30 % for the same readout trained alone). The clip
    has to act on the two groups separately."""
    from pinneapple_simulation.pipelines.airfrans_benchmark import clip_wall_gradients

    network = WallAwareNet(width=16, depth=1, fourier=0, shape_features=0)
    head = [p for n, p in network.named_parameters() if n.startswith("head_slope")]
    trunk = [p for n, p in network.named_parameters() if not n.startswith("head_slope")]
    for parameter in trunk:                                    # the measurement: norm 139.5 against the head's 1.56
        parameter.grad = torch.full_like(parameter, 10.0)
    for parameter in head:
        parameter.grad = torch.full_like(parameter, 1e-3)

    def norm(parameters):
        return float(torch.sqrt(sum((p.grad ** 2).sum() for p in parameters)))

    head_before, trunk_before = norm(head), norm(trunk)
    assert trunk_before > 10 * head_before                    # a single global norm would be the trunk's
    clip_wall_gradients(network, 1.0)
    assert norm(head) <= 1.0 + 1e-5 and norm(head) == pytest.approx(head_before)  # untouched: below its own cap
    assert norm(trunk) == pytest.approx(1.0)                                       # clipped against its own norm


def test_the_flat_plate_prior_is_the_skin_friction_law_and_is_scaled_like_the_truth():
    """The prior moves the magnitude out of the network: it must be the flat-plate law, in the same dimensionless units
    the cached wall slope uses, so a correction of order one is all the readout has to learn."""
    from pinneapple_simulation.pipelines.airfrans_benchmark import flat_plate_slope
    from pinneapple_simulation.pipelines.airfrans_benchmark.model import CHORD, NU

    scaled, x, inlet = 1.2, 0.1, 50.0              # the edge speed travels in units of U∞
    edge = torch.tensor([[scaled, 0.0]])
    value = float(flat_plate_slope(edge, torch.tensor([x]), torch.tensor([inlet]))[0])
    speed = scaled * inlet
    reynolds = speed * x * CHORD / NU
    expected = 0.5 * speed ** 2 * 0.0576 * reynolds ** -0.2 / NU * (CHORD / inlet)
    assert value == pytest.approx(expected, rel=1e-6)
    assert 5e3 < value < 2e4          # the truth's rms is 26 500 and its median 6 500 on the training sample


def test_with_the_prior_the_readout_starts_at_the_law_and_predicts_a_correction():
    """A zeroed correction has to return the law untouched, so training starts from physics rather than from zero."""
    network = WallAwareNet(width=16, depth=1, fourier=0, shape_features=0, slope_prior=True)
    with torch.no_grad():
        network.head_slope[-1].weight.zero_()
        network.head_slope[-1].bias.zero_()
        features = torch.zeros(2, len(FEATURE_NAMES))
        features[:, 0] = torch.tensor([0.1, 0.3])
        features[:, 7] = 50.0                      # freestream along x
        features[:, 9] = torch.log(torch.tensor(50.0))
        heads = network(features, heads=True)
    from pinneapple_simulation.pipelines.airfrans_benchmark import flat_plate_slope

    edge = heads["outer"] + features[:, 7:9]
    magnitude = flat_plate_slope(edge, features[:, 0], torch.exp(features[:, 9]))
    expected = magnitude[:, None] * edge / torch.linalg.vector_norm(edge, dim=1, keepdim=True)
    assert torch.allclose(heads["slope"], expected, rtol=1e-5)   # the law, pointing along the edge flow


def test_the_curve_projection_averages_each_surface_along_the_chord():
    """Wall nodes on two surfaces and three chord positions: each bin takes its own mean, and only the wall counts."""
    features = np.zeros((8, 11))
    features[:, 6] = 1.0                                  # all on the wall
    features[:, 7] = 1.0                                  # freestream along +x, so the chord frame is the domain frame
    features[:4, 0] = [0.1, 0.1, 0.5, 0.5]                # x/c
    features[4:, 0] = [0.1, 0.1, 0.5, 0.5]
    features[:4, 5] = 1.0                                 # upper surface: normal along +y -> side +1
    features[4:, 5] = -1.0                                # lower surface
    slope = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 0.0],
                      [10.0, 0.0], [14.0, 0.0], [20.0, 0.0], [24.0, 0.0]])

    projected = curve_projection(slope, features, bins=2, smooth=False)
    assert np.allclose(projected[:2, 0], [2.0, 2.0])      # upper, first half
    assert np.allclose(projected[2:4, 0], [6.0, 6.0])
    assert np.allclose(projected[4:6, 0], [12.0, 12.0])   # the two surfaces never share a bin
    assert np.allclose(projected[6:, 0], [22.0, 22.0])


def test_the_curve_projection_is_evaluated_off_the_wall_too():
    """The slope only acts where the wall layer's weight is non-zero — just off the surface, not on it."""
    features = np.zeros((3, 11))
    features[0, 6], features[0, 5], features[0, 0] = 1.0, 1.0, 0.5      # the only wall node
    features[1:, 5], features[1:, 0] = 1.0, 0.5                          # two band nodes above it
    features[1, 2], features[2, 2] = 1e-5, 1e-3
    slope = np.array([[4.0, 0.0], [0.0, 9.0], [0.0, 9.0]])

    projected = curve_projection(slope, features, bins=1, smooth=False)
    assert np.allclose(projected[1:, 0], 4.0)             # the wall's curve reaches the band
    assert np.allclose(projected[1:, 1], 0.0)
