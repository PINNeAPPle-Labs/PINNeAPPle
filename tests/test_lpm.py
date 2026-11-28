"""Large Physics Model blocks: query independence, OOD flags, learning a design space (small, fast)."""
import pytest
import torch

from benchmarks.lpm_ellipse_design_space import run
from pinneapple_neural.lpm import (
    Design,
    DesignBatch,
    FourierEncoding,
    LargePhysicsModel,
    ellipse_design_space,
    fit,
    multiscale_neighbourhood,
    stack_designs,
)


def test_fourier_encoding_shape():
    enc = FourierEncoding(4)
    assert enc(torch.zeros(5, 2)).shape == (5, enc.out_dim(2)) == (5, 18)


def test_query_independence_permutation_and_subsets():
    """The prediction at a point must not depend on which other points are queried with it."""
    torch.manual_seed(0)
    m = LargePhysicsModel(dim=2, n_fields=1, n_heads=2, width=32)
    d = Design.ellipse(1.0, 0.7, 64)
    full = m(d)
    perm = torch.randperm(64)
    d_perm = Design(d.points, d.normals, d.query[perm], d.params)
    torch.testing.assert_close(m(d_perm), full[perm])
    d_sub = Design(d.points, d.normals, d.query[:7], d.params)
    torch.testing.assert_close(m(d_sub), full[:7])


def test_neighbourhood_features_do_not_depend_on_surface_point_order():
    d = Design.ellipse(1.2, 0.8, 50)
    perm = torch.randperm(50)
    a = multiscale_neighbourhood(d.query, d.points, d.normals, (0.2, 1.0))
    b = multiscale_neighbourhood(d.query, d.points[perm], d.normals[perm], (0.2, 1.0))
    torch.testing.assert_close(a, b)


def test_small_lpm_learns_the_design_space_and_flags_far_geometry():
    r = run(n_train=10, n_test=4, n_points=48, epochs=250, heads=2, width=48)
    assert r["heldout_rel_l2_mean"] < 0.35  # full-size benchmark: 0.031 (benchmarks/_out)
    assert r["circle_r3"]["ood"] is True
    assert r["circle_r3"]["ood_score"] > max(r["heldout_ood_scores"])


def test_design_ellipse_and_square():
    """Circle limit of the ellipse family: Cp = 1 - 4 sin^2(theta) (potential flow past a cylinder)."""
    d = Design.ellipse(1.0, 1.0, 64)
    th = torch.atan2(d.points[:, 1], d.points[:, 0])
    torch.testing.assert_close(d.target[:, 0], 1 - 4 * torch.sin(th) ** 2, atol=1e-5, rtol=0)
    torch.testing.assert_close(d.normals, d.points)                     # unit circle: normal = position
    e = Design.ellipse(1.4, 0.6, 32)
    assert float(e.target.max()) == pytest.approx(1.0, abs=1e-6)      # stagnation points at t = 0, pi
    s = Design.square(2.0, 40)
    assert s.points.shape == (40, 2) and s.target is None
    assert float(s.points.abs().max(1).values.min()) == pytest.approx(2.0)   # every point on the boundary


def _ragged_designs():
    """Designs with different surface counts, so the batch path has to pad."""
    return [Design.ellipse(1.0, 0.8, 24), Design.ellipse(0.9, 0.7, 17), Design.ellipse(1.2, 1.1, 31)]


@pytest.mark.parametrize("backbone", ["mlp", "attention"])
def test_batching_and_padding_do_not_change_a_design(backbone):
    """A design evaluated inside a padded batch must see exactly what it sees alone (both backbones)."""
    torch.manual_seed(0)
    m = LargePhysicsModel(dim=2, n_fields=1, radii=(0.2, 0.8), width=32, n_heads=2, backbone=backbone,
                          attn_heads=2, attn_layers=1).eval()
    designs = _ragged_designs()
    batch = stack_designs(designs)
    assert isinstance(batch, DesignBatch) and batch.batch_size == len(designs)
    assert int(batch.point_mask.sum()) == sum(len(d.points) for d in designs)
    together = m(batch)
    for i, d in enumerate(designs):
        torch.testing.assert_close(m(d), together[i, :len(d.query)])
    torch.testing.assert_close(m.geo(batch.points, batch.normals, batch.point_mask),
                               torch.stack([m.geo(d.points, d.normals) for d in designs]))


def test_batched_optimizer_step_equals_per_design_step():
    """One batched step must compute the same loss as the sum over its designs (bagging included)."""
    torch.manual_seed(0)
    designs = _ragged_designs() + [Design.ellipse(0.7, 1.3, 20)]
    m = LargePhysicsModel(dim=2, n_fields=1, radii=(0.2, 0.8), width=24, n_heads=2)
    m2 = LargePhysicsModel(dim=2, n_fields=1, radii=(0.2, 0.8), width=24, n_heads=2)
    m2.load_state_dict(m.state_dict())                    # same weights before the step
    one = fit(m, designs, epochs=1)                       # one design per step
    many = fit(m2, designs, epochs=1, batch_designs=len(designs))   # the same designs in one step
    torch.testing.assert_close(torch.tensor(one[-1]), torch.tensor(many[-1]))


def test_chunked_epoch_equals_one_chunk():
    """Walking the designs chunk by chunk must give the same loss history as putting them in one chunk.

    ``fit`` backpropagates each chunk as it goes and steps once per epoch, which is the same gradient as summing every
    chunk and calling ``backward`` once — but only one chunk's graph is alive at a time. Keeping them all alive is
    what killed the 960-point wing fit at 4.4 GB before it finished an epoch, so this is the property to pin.
    """
    torch.manual_seed(0)
    designs = ellipse_design_space(5, n_points=24, seed=3)
    m = LargePhysicsModel(dim=2, n_fields=1, radii=(0.2, 0.8), width=24, n_heads=3, backbone="attention",
                          attn_heads=2, attn_layers=1)
    m2 = LargePhysicsModel(dim=2, n_fields=1, radii=(0.2, 0.8), width=24, n_heads=3, backbone="attention",
                           attn_heads=2, attn_layers=1)
    m2.load_state_dict(m.state_dict())                          # same weights before the first step
    chunked = fit(m, designs, epochs=3, batch_designs=2)
    whole = fit(m2, designs, epochs=3, batch_designs=len(designs))
    assert len(chunked) == len(whole) == 3
    for a, b in zip(chunked, whole, strict=True):
        assert a == pytest.approx(b, rel=1e-5)


def test_fit_resumes_from_its_checkpoint(tmp_path):
    """A fit that is killed must continue from its checkpoint, not from epoch one.

    The campaign runs for tens of hours on a machine that has killed the process with no crash report, and a family
    refitted from zero each time never finishes. The checkpoint carries the optimizer and scheduler state too, so the
    resumed history continues the interrupted run instead of restarting its cosine schedule. The interruption here is
    an exception thrown from the progress callback, which is the same point in the loop a kill can land.
    """
    designs = ellipse_design_space(5, n_points=24, seed=4)
    torch.manual_seed(0)
    m = LargePhysicsModel(dim=2, n_fields=1, radii=(0.2, 0.8), width=24, n_heads=2, backbone="attention",
                          attn_heads=2, attn_layers=1)
    seen: list[float] = []

    def stop_at_two(epoch, loss):
        seen.append(loss)
        if epoch == 2:
            raise RuntimeError("killed mid-fit")

    with pytest.raises(RuntimeError):
        fit(m, designs, epochs=4, batch_designs=2, checkpoint_dir=tmp_path, checkpoint_every_s=0.0,
            on_epoch=stop_at_two)
    assert len(seen) == 2 and (tmp_path / "fit.pt").exists(), "an interrupted fit leaves its checkpoint behind"

    m2 = LargePhysicsModel(dim=2, n_fields=1, radii=(0.2, 0.8), width=24, n_heads=2, backbone="attention",
                           attn_heads=2, attn_layers=1)
    resumed = fit(m2, designs, epochs=4, batch_designs=2, checkpoint_dir=tmp_path, checkpoint_every_s=0.0)
    assert len(resumed) == 4, "two epochs were already done and two remain"
    assert resumed[:2] == pytest.approx(seen, rel=1e-6), "the resumed fit continues the history it found"
    assert not (tmp_path / "fit.pt").exists(), "a finished fit leaves nothing to resume"

    # a checkpoint of another recipe is not this fit's state, and resuming it would report another run's history
    (tmp_path / "fit.pt").write_bytes(b"")
    assert len(fit(m2, designs, epochs=1, batch_designs=3, checkpoint_dir=tmp_path, checkpoint_every_s=0.0)) == 1


def test_the_neighbourhood_cache_gives_the_same_fit():
    """Caching the multiscale features must not change a single number — it only stops recomputing them.

    ``_multiscale`` rebuilds the query-to-surface distance matrix on every forward, and the geometry never changes
    during a fit. Storing the features once per design is the cache Luminary's 50x came from.
    """
    radii = (0.2, 0.8)
    designs = ellipse_design_space(6, n_points=32, seed=7)
    cached = [Design(d.points, d.normals, d.query, d.params, d.target,
                     multiscale_neighbourhood(d.query, d.points, d.normals, radii))
              for d in designs]

    def fresh():
        torch.manual_seed(0)
        return LargePhysicsModel(dim=2, n_fields=1, radii=radii, width=24, n_heads=2, backbone="attention",
                                 attn_heads=2, attn_layers=1)

    a, b = fresh(), fresh()
    b.load_state_dict(a.state_dict())
    assert fit(a, designs, epochs=2, batch_designs=3) == pytest.approx(
        fit(b, cached, epochs=2, batch_designs=3), rel=1e-6)

    # a cache built for another radius set is ignored, not trusted: the fit equals the uncached one
    wrong = [Design(d.points, d.normals, d.query, d.params, d.target,
                    multiscale_neighbourhood(d.query, d.points, d.normals, (0.5,))) for d in designs]
    c, e = fresh(), fresh()
    e.load_state_dict(c.state_dict())
    assert fit(c, designs, epochs=2, batch_designs=3) == pytest.approx(
        fit(e, wrong, epochs=2, batch_designs=3), rel=1e-6)


def test_design_space_builds_scale_with_the_exact_answer():
    """``ellipse_design_space`` is the benchmark family, with the analytic potential-flow Cp as target."""
    designs = ellipse_design_space(120, n_points=32, seed=5)
    assert len(designs) == 120 and all(d.target.shape == (32, 1) for d in designs)
    a = torch.tensor([float(d.points[:, 0].max()) for d in designs])
    b = torch.tensor([float(d.points[:, 1].max()) for d in designs])
    assert float(a.min()) >= 0.6 and float(a.max()) <= 1.4 and float(b.min()) >= 0.6 and float(b.max()) <= 1.4
    d = designs[7]
    torch.testing.assert_close(d.target[:, 0], Design.ellipse(float(a[7]), float(b[7]), 32).target[:, 0])
    batch = stack_designs(designs[:64])
    assert batch.batch_size == 64 and bool(batch.point_mask.all()) and bool(batch.query_mask.all())


def test_attention_backbone_learns_the_design_space():
    """The geometry-aware transformer backbone learns a design space it never saw, and flags a square."""
    torch.manual_seed(0)
    designs = ellipse_design_space(12, n_points=28, seed=2)
    m = LargePhysicsModel(dim=2, n_fields=1, radii=(0.2, 0.8), width=32, n_heads=1, backbone="attention",
                          attn_heads=2, attn_layers=1)
    hist = fit(m, designs, epochs=200, lr=3e-3, batch_designs=6)
    assert hist[-1] < hist[0]
    m.eval()
    held = ellipse_design_space(4, n_points=28, seed=99)
    rel = [(m.predict(d)["mean"] - d.target).norm() / d.target.norm() for d in held]
    assert float(torch.stack(rel).mean()) < 0.6   # measured full-scale value: benchmarks/_out/lpm_scale_*.json
    assert m.ood(Design.ellipse(3.0, 3.0, 28))["ood"] is True   # outside the family's size range


def test_unknown_backbone_is_refused():
    with pytest.raises(ValueError, match="backbone"):
        LargePhysicsModel(dim=2, n_fields=1, backbone="transformer")
