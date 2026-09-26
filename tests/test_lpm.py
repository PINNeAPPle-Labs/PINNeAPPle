"""Large Physics Model blocks: query independence, OOD flags, learning a design space (small, fast)."""
import torch

from benchmarks.lpm_ellipse_design_space import ellipse, run
from pinneapple_neural.lpm import Design, FourierEncoding, LargePhysicsModel, multiscale_neighbourhood


def test_fourier_encoding_shape():
    enc = FourierEncoding(4)
    assert enc(torch.zeros(5, 2)).shape == (5, enc.out_dim(2)) == (5, 18)


def test_query_independence_permutation_and_subsets():
    """The prediction at a point must not depend on which other points are queried with it."""
    torch.manual_seed(0)
    m = LargePhysicsModel(dim=2, n_fields=1, n_heads=2, width=32)
    d = ellipse(1.0, 0.7, 64)
    full = m(d)
    perm = torch.randperm(64)
    d_perm = Design(d.points, d.normals, d.query[perm], d.params)
    torch.testing.assert_close(m(d_perm), full[perm])
    d_sub = Design(d.points, d.normals, d.query[:7], d.params)
    torch.testing.assert_close(m(d_sub), full[:7])


def test_neighbourhood_features_do_not_depend_on_surface_point_order():
    d = ellipse(1.2, 0.8, 50)
    perm = torch.randperm(50)
    a = multiscale_neighbourhood(d.query, d.points, d.normals, (0.2, 1.0))
    b = multiscale_neighbourhood(d.query, d.points[perm], d.normals[perm], (0.2, 1.0))
    torch.testing.assert_close(a, b)


def test_small_lpm_learns_the_design_space_and_flags_far_geometry():
    r = run(n_train=10, n_test=4, n_points=48, epochs=250, heads=2, width=48)
    assert r["heldout_rel_l2_mean"] < 0.35  # full-size benchmark: 0.031 (benchmarks/_out)
    assert r["circle_r3"]["ood"] is True
    assert r["circle_r3"]["ood_score"] > max(r["heldout_ood_scores"])
