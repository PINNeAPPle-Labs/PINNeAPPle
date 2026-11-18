"""The field-surrogate benchmark: metrics, containers, the trainer, and the two architectures it ranks."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_arena.fieldbench import (  # noqa: E402
    FieldDataset,
    FieldSample,
    PatchwiseProtocol,
    bootstrap_ci,
    build_model_specs,
    collate_samples,
    extreme_errors,
    extreme_of_field,
    leaderboard,
    mean_relative_l2,
    normalize_geometry,
    paired_bootstrap,
    pooled_r2,
    predict_field_model,
    relative_l2_per_geometry,
    run_benchmark,
    score_model,
    subsample_nodes,
    train_field_model,
)
from pinneapple_neural.architectures.crash_solver import CrashSolverModel  # noqa: E402
from pinneapple_neural.architectures.point_surrogates import PointNetSurrogate  # noqa: E402


def _dataset(n=12, points=200, transient=False, seed=0, categories=("A", "B")) -> FieldDataset:
    rng = np.random.default_rng(seed)
    samples = []
    frames = 4
    for i in range(n):
        pts = rng.normal(size=(points, 3)).astype("float32")
        if transient:
            values = rng.normal(size=(frames, points, 3)).astype("float32")
            times = np.linspace(0.0, 1e-3, frames).astype("float32")
        else:
            values = rng.normal(size=(points, 1)).astype("float32")
            times = None
        samples.append(FieldSample(id=f"s{i}", points=pts, values=values, category=categories[i % len(categories)],
                                   parts=(rng.integers(1, 4, points)).astype(int), times=times))
    return FieldDataset("test", samples)


# --------------------------------------------------------------------------------------------- metrics
def test_relative_l2_is_one_for_a_zero_prediction_and_zero_for_an_exact_one():
    true = [np.arange(10.0)]
    assert relative_l2_per_geometry(true, true)[0] == pytest.approx(0.0)
    assert relative_l2_per_geometry([np.zeros(10)], true)[0] == pytest.approx(1.0)
    assert mean_relative_l2([np.zeros(10)], true) == pytest.approx(1.0)


def test_pooled_r2_is_computed_over_points_not_geometries():
    truths = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
    preds = [np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])]
    # pooled over the six points: SS_res = 2, SS_tot = 4 (mean 1, four of the six points one away from it)
    assert pooled_r2(preds, truths) == pytest.approx(1 - 2 / 4)


def test_bootstrap_interval_narrows_with_more_samples():
    rng = np.random.default_rng(0)
    small = rng.normal(size=20)
    large = rng.normal(size=2000)
    lo_s, hi_s = bootstrap_ci(small, n_boot=2000, rng=np.random.default_rng(1))
    lo_l, hi_l = bootstrap_ci(large, n_boot=2000, rng=np.random.default_rng(1))
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_paired_bootstrap_finds_a_clear_winner_and_reports_a_tie_honestly():
    rng = np.random.default_rng(2)
    a = rng.normal(0.10, 0.01, size=30)
    b = rng.normal(0.20, 0.01, size=30)
    clear = paired_bootstrap(a, b, n_boot=2000, rng=np.random.default_rng(3))
    assert clear["mean_difference"] < 0 and clear["win_rate"] == 1.0
    assert clear["p_permutation"] < 0.01
    tie = paired_bootstrap(a, a + rng.normal(0, 1e-9, size=30), n_boot=2000, rng=np.random.default_rng(4))
    assert abs(tie["mean_difference"]) < 1e-6


def test_three_resolution_protocol_keeps_every_point():
    rng = np.random.default_rng(5)
    pts = rng.normal(size=(100, 3))
    vals = np.sin(pts[:, 0])
    sub = np.arange(0, 100, 2)
    for mode, n_expected in (("subsampled", 50), ("interpolated", 100), ("patchwise", 100)):
        out = PatchwiseProtocol(mode).score(pts, vals, pts[sub], vals[sub])
        assert out["pred"].shape == out["ref"].shape
        if mode == "subsampled":
            assert out["pred"].shape[0] == n_expected


def test_score_model_reports_an_interval_and_a_leaderboard_table():
    rng = np.random.default_rng(6)
    truths = [rng.normal(size=(50, 1)) for _ in range(8)]
    preds = [t + rng.normal(scale=0.3, size=t.shape) for t in truths]
    s = score_model("m", "point", preds, truths, cost={"params_m": 1.0, "latency_ms": 2.0})
    assert s.extra["ci95"][0] < s.rel_l2 < s.extra["ci95"][1]
    table = leaderboard([s])
    assert "| 1 | m | point |" in table and "params (M)" in table


# --------------------------------------------------------------------------------------------- containers
def test_dataset_splits_cover_every_sample_and_hold_out_a_category():
    ds = _dataset(12)
    ids = set(ds.splits["train"]) | set(ds.splits["val"]) | set(ds.splits["test"])
    assert ids == {s.id for s in ds.samples}
    folded = ds.hold_out_category("A")
    assert all(ds[i].category == "A" for i in folded.splits["test"])
    assert not any(ds[i].category == "A" for i in folded.splits["train"])


def test_dataset_round_trips_through_npz(tmp_path):
    ds = normalize_geometry(_dataset(6, transient=True))
    path = ds.save(str(tmp_path / "ds.npz"))
    back = FieldDataset.load(path)
    assert len(back) == len(ds)
    assert back.splits == ds.splits
    a, b = ds[0], back[0]
    assert np.allclose(a.points, b.points) and np.allclose(a.values, b.values)
    assert np.array_equal(a.parts, b.parts)


def test_normalize_geometry_uses_one_transform_for_every_sample():
    ds = normalize_geometry(_dataset(4))
    radii = [float(np.abs(s.points).max()) for s in ds.samples]
    assert max(radii) < 2.0 and ds.normalized


# --------------------------------------------------------------------------------------------- architectures
def test_pointnet_ignores_padded_points():
    model = PointNetSurrogate(dim=32, depth=1).eval()
    x = torch.randn(1, 40, 3)
    padded = torch.cat([x, torch.randn(1, 25, 3)], dim=1)
    mask = torch.zeros(1, 65, dtype=torch.bool)
    mask[0, :40] = True
    with torch.no_grad():
        assert torch.allclose(model(x), model(padded, mask=mask)[:, :40], atol=1e-6)


def test_crash_solver_shapes_tokens_and_adjacency():
    torch.manual_seed(0)
    model = CrashSolverModel(hidden=32, local_slices=8, n_frames=3).eval()
    x = torch.rand(2, 90, 3)
    # three slabs along x: neighbours touch, the two ends do not
    part = torch.where(x[..., 0] < 0.34, 1, torch.where(x[..., 0] < 0.67, 2, 3)).long()
    time = torch.linspace(0, 1e-3, 3).unsqueeze(0).expand(2, -1)
    with torch.no_grad():
        out = model(x, part=part, time=time, return_tokens=True)
    assert out["u"].shape == (2, 3, 90, 3)
    assert out["part_tokens"].shape == (2, 3, 32)
    # a mesh shares its interface nodes, so the distance across an interface is exactly zero; a sampled point cloud
    # needs a tolerance that matches its own spacing (here a tenth of the box)
    adj = CrashSolverModel.part_adjacency(x, part, tol=0.25)
    assert adj.shape == (2, 3, 3)
    assert bool(adj[0, 0, 1]) and bool(adj[0, 1, 2])            # neighbouring slabs share an interface
    assert not bool(adj[0, 0, 2])                               # the two ends do not touch


def test_crash_solver_masked_part_does_not_change_the_others():
    """A padded node of part 3 must not leak into the encoding of parts 1 and 2."""
    torch.manual_seed(0)
    model = CrashSolverModel(hidden=32, local_slices=8, n_frames=1).eval()
    x = torch.randn(1, 60, 3)
    part = torch.cat([torch.ones(1, 20), 2 * torch.ones(1, 20), 3 * torch.ones(1, 20)], dim=1).long()
    x_pad = torch.cat([x, torch.randn(1, 15, 3)], dim=1)
    part_pad = torch.cat([part, torch.zeros(1, 15, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a = model(x, part=part)
        b = model(x_pad, part=part_pad)
    assert torch.allclose(a, b[:, :, :60], atol=1e-5)


# --------------------------------------------------------------------------------------------- training
def test_the_trainer_learns_a_learnable_field():
    """A field that is a smooth function of the coordinates has to be learned — if the loss stops moving, the trainer
    (not the model) is what regressed, and this test says so."""
    rng = np.random.default_rng(7)
    samples = []
    for i in range(10):
        pts = rng.uniform(-1, 1, size=(200, 3)).astype("float32")
        samples.append(FieldSample(id=f"l{i}", points=pts,
                                   values=(0.5 * pts[:, :1] + pts[:, 1:2]).astype("float32"), category="A"))
    ds = normalize_geometry(FieldDataset("learn", samples))
    specs = [{"name": "pointnet", "family": "point", "build": lambda: PointNetSurrogate(dim=64, depth=2),
              "uses_parts": False}]
    result = run_benchmark(specs, ds, epochs=120, batch_size=2, lr=3e-3, device="cpu", patience=120, n_boot=100)
    score = result["scores"][0]
    assert score.rel_l2 < 0.5, f"the trainer did not learn a smooth field (Rel L2 {score.rel_l2:.3f})"


def test_every_spec_builds_and_is_registered_with_a_family():
    specs = build_model_specs(transient=False, scale="small", n_parts=4)
    names = {s["name"] for s in specs}
    assert {"transolver", "transolver_lite", "pointnet", "pointnet_wide", "edgeconv", "part_pointnet",
            "crash_solver"} <= names
    for spec in specs:
        model = spec["build"]()
        assert sum(p.numel() for p in model.parameters()) > 0
        assert spec["family"] in ("operator", "point", "point+part", "hierarchical")


# --------------------------------------------------------------------------------------------- shapes and padding
def test_a_dataset_that_mixes_field_shapes_is_refused_at_load_time():
    """A campaign whose geometries carry different frame counts cannot be batched. The failure used to surface as a
    bare numpy reshape error at the end of a multi-hour run, so the dataset refuses to exist and names the samples."""
    steady = [FieldSample(id="a", points=np.zeros((5, 3)), values=np.zeros((5, 1))),
              FieldSample(id="b", points=np.zeros((6, 3)), values=np.zeros((6, 1)))]
    FieldDataset("steady", steady)                                            # different nodes are fine
    with pytest.raises(ValueError, match="mixes field shapes"):
        FieldDataset("mixed", steady + [FieldSample(id="c", points=np.zeros((5, 3)),
                                                    values=np.zeros((4, 5, 1)), times=np.linspace(0, 1, 4))])
    frames = [FieldSample(id=f"f{i}", points=np.zeros((5, 3)), values=np.zeros((n, 5, 1)),
                          times=np.linspace(0, 1, n)) for i, n in enumerate((10, 10, 9))]
    with pytest.raises(ValueError, match="9 frames"):
        FieldDataset("campaign", frames)


def test_a_prediction_is_trimmed_to_each_samples_own_point_count():
    """The batch is padded to the widest geometry, so a model answers with the padded width: scoring that against the
    reference is the "differ in size (2241 vs 2210)" that killed the surface-pressure campaign, so the prediction is
    sliced back to each sample before it is returned."""
    rng = np.random.default_rng(3)
    samples = [FieldSample(id="small", points=rng.normal(size=(60, 3)).astype("float32"),
                           values=rng.normal(size=(60, 1)).astype("float32")),
               FieldSample(id="wide", points=rng.normal(size=(95, 3)).astype("float32"),
                           values=rng.normal(size=(95, 1)).astype("float32"))]
    ds = FieldDataset("padded", samples)
    model = PointNetSurrogate(dim=32, depth=1)
    preds = predict_field_model(model, ds, ["small", "wide"], device="cpu", batch_size=2)
    assert [p.shape for p in preds] == [(60,), (95,)]
    # a transient batch, where a model may also answer with a single-frame axis the sample does not have
    trans = [FieldSample(id="small", points=rng.normal(size=(60, 3)).astype("float32"),
                         values=rng.normal(size=(3, 60, 1)).astype("float32"), times=np.linspace(0, 1, 3)),
             FieldSample(id="wide", points=rng.normal(size=(80, 3)).astype("float32"),
                         values=rng.normal(size=(3, 80, 1)).astype("float32"), times=np.linspace(0, 1, 3))]
    ds_t = FieldDataset("padded_transient", trans)
    preds_t = predict_field_model(CrashSolverModel(hidden=32, local_slices=8, n_frames=3, out_dim=1), ds_t,
                                  ["small", "wide"], device="cpu", batch_size=2)
    assert [p.shape for p in preds_t] == [(3, 60), (3, 80)]      # a one-component field comes back squeezed


# --------------------------------------------------------------------------------------------- design channels
def test_extra_channels_reach_the_model_and_round_trip_through_npz(tmp_path):
    """A parametric benchmark needs the design vector inside the sample, not in a side file: it is batched with the
    points, saved with them, and every transform of the dataset carries it."""
    rng = np.random.default_rng(5)
    samples = []
    for i in range(4):
        design = np.repeat(rng.random((1, 7)).astype("float32"), 30, axis=0)
        samples.append(FieldSample(id=f"d{i}", points=rng.normal(size=(30, 3)).astype("float32"),
                                   values=rng.normal(size=(3, 30, 3)).astype("float32"), extra=design,
                                   times=np.linspace(0, 1e-3, 3).astype("float32")))
    ds = FieldDataset("designs", samples)
    assert ds[0].inputs()["extra"].shape == (1, 30, 7)
    batch = collate_samples(ds, ["d0", "d1"])
    assert batch["extra"].shape == (2, 30, 7)
    assert np.allclose(batch["extra"][0, 0].numpy(), samples[0].extra[0])

    path = ds.save(str(tmp_path / "designs.npz"))
    back = FieldDataset.load(path)
    assert np.allclose(back["d0"].extra, samples[0].extra)
    assert np.allclose(normalize_geometry(back)["d0"].extra, samples[0].extra)   # not a coordinate
    assert np.allclose(subsample_nodes(back, 10, seed=0)["d0"].extra.shape, (10, 7))


def test_a_dataset_that_mixes_the_extra_width_is_refused():
    """A model is built for one input width; a sample without the design channels cannot be batched with the rest."""
    rng = np.random.default_rng(6)
    samples = [FieldSample(id="a", points=rng.normal(size=(20, 3)).astype("float32"),
                           values=rng.normal(size=(20, 1)).astype("float32"),
                           extra=np.zeros((20, 7), "float32")),
               FieldSample(id="b", points=rng.normal(size=(20, 3)).astype("float32"),
                           values=rng.normal(size=(20, 1)).astype("float32"))]
    with pytest.raises(ValueError, match="extra-channel width"):
        FieldDataset("mixed", samples)


def test_extreme_of_field_reads_the_kpi_a_crash_report_quotes():
    """Peak displacement along the impact direction is the most negative u_x, not the largest magnitude: the two
    differ by the sign and, on a structure that recoils, by the frame they happen in."""
    t = np.zeros((3, 4, 3), "float64")
    t[0, :, 0] = [0.0, -1.0, 0.0, 0.0]
    t[1, :, 0] = [0.0, -3.0, 0.0, 0.0]                  # the peak of the event
    t[2, :, 0] = [0.0, -2.0, 0.0, 0.0]
    assert extreme_of_field(t, mode="min") == pytest.approx(-3.0)
    assert extreme_of_field(t, mode="min", frames="last") == pytest.approx(-2.0)
    assert extreme_of_field(t, mode="min", frames=0) == pytest.approx(-1.0)
    assert extreme_of_field(t, mode="absmax") == pytest.approx(3.0)
    assert np.isnan(extreme_of_field(np.full((3, 4, 3), np.nan)))


def test_extreme_errors_report_the_quantity_error_next_to_the_field_error():
    """A model can shade every point and still miss the peak: the two numbers are reported from the same prediction."""
    rng = np.random.default_rng(7)
    truths = [np.stack([np.zeros((50, 3)), -np.abs(rng.normal(size=(50, 3))) * 1e-2]) for _ in range(6)]
    preds = [t * 1.1 for t in truths]                   # 10 % on every point, so 10 % on the peak as well
    kpi = extreme_errors(preds, truths, channel=0, mode="min", n_boot=200)
    assert kpi["n"] == 6
    assert kpi["mean_relative"] == pytest.approx(0.1, rel=1e-6)
    assert kpi["max_relative"] == pytest.approx(0.1, rel=1e-6)
    lo, hi = kpi["mean_relative_ci95"]
    assert lo <= kpi["mean_relative"] <= hi


def test_a_row_built_for_one_channel_is_refused_on_a_three_component_field():
    """The bug this guards: a displacement field has three components, a row built with ``out_dim=1`` predicts one, torch
    broadcasts it against the other two, and the benchmark then reports a plausible error for a model that never
    predicted the field. Both the loss and the prediction refuse it now."""
    rng = np.random.default_rng(11)
    samples = [FieldSample(id=f"v{i}", points=rng.normal(size=(40, 3)).astype("float32"),
                           values=rng.normal(size=(3, 40, 3)).astype("float32"),
                           times=np.linspace(0, 1e-3, 3).astype("float32")) for i in range(4)]
    ds = FieldDataset("vectors", samples)
    scalar = PointNetSurrogate(dim=16, depth=1)                     # one output channel
    with pytest.raises(ValueError, match="channel"):
        train_field_model(scalar, ds, train_ids=["v0", "v1"], val_ids=["v2"], epochs=1, batch_size=1, device="cpu")
    with pytest.raises(ValueError, match="channel|component"):
        predict_field_model(scalar, ds, ["v3"], device="cpu")
    vector = PointNetSurrogate(dim=16, depth=1, out_dim=3)          # the same row, built for the field it is scored on
    preds = predict_field_model(vector, ds, ["v3"], device="cpu")
    assert preds[0].shape == (1, 40, 3)     # the raw point model answers one frame; the benchmark rows add the time


def test_relative_l2_refuses_a_prediction_of_another_shape_even_when_the_values_add_up():
    """Equal counts are not equal fields: 60 points of one channel against 20 points of three has the same size."""
    with pytest.raises(ValueError, match="shape"):
        relative_l2_per_geometry([np.zeros(60)], [np.zeros((20, 3))])


def test_the_target_is_standardised_with_the_statistics_the_prediction_undoes():
    """The trainer passes training statistics to the batch and the prediction multiplies by the same std to come back
    to physical units. When only the second half ran, the loss was in raw units and every prediction came back scaled
    by the std — a field 1e-3 in size against a reference of 1."""
    rng = np.random.default_rng(13)
    samples = [FieldSample(id=f"s{i}", points=rng.normal(size=(30, 3)).astype("float32"),
                           values=(rng.normal(size=(2, 30, 3)) * 5e-3 + 1e-3).astype("float32"),
                           times=np.linspace(0, 1e-3, 2).astype("float32")) for i in range(4)]
    ds = FieldDataset("scaled", samples)
    mean = np.array([1e-3, 1e-3, 1e-3], "float32")
    std = np.array([5e-3, 5e-3, 5e-3], "float32")
    stats = {"mean": mean, "std": std}
    batch = collate_samples(ds, ["s0", "s1"], standardize=stats)
    valid = batch["mask"][:, None, :, None].expand_as(batch["y"])
    raw = torch.from_numpy(np.stack([np.asarray(ds[i].values) for i in ("s0", "s1")]).astype("float32"))
    expected = (raw - torch.from_numpy(mean).view(1, 1, 1, -1)) / torch.from_numpy(std).view(1, 1, 1, -1)
    assert torch.allclose(batch["y"][valid], expected[valid], atol=1e-6)
    assert float(batch["y"].std()) > 0.5                        # O(1) targets, not 1e-3 ones
