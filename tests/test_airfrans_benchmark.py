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
    case_ids_for,
    chord,
    manifest_splits,
    physics_loss,
    point_features,
    predict_cases,
    sample_indices,
    scaled_targets,
    train,
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
