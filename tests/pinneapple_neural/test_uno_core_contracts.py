"""Core coverage (#1129, toward 1.0) for the U-shaped neural operator (UNO).

Structural properties that hold exactly, whatever the weights:
* the spectral convolution is linear, commutes with circular shifts and outputs only its retained modes;
* the point kernel operator is equivariant to a permutation of the points;
* the down and up blocks halve and restore the resolution in 1, 2 and 3 dimensions.
"""
import pytest

torch = pytest.importorskip("torch")

from pinneapple_neural.architectures.neural_operators import uno as U  # noqa: E402


def _seed():
    torch.manual_seed(0)


def test_spectral_conv_is_linear_shift_equivariant_and_band_limited():
    _seed()
    sc = U.SpectralConvND(1, 2, 3, modes=(5,))
    x, z = torch.randn(2, 2, 32), torch.randn(2, 2, 32)
    with torch.no_grad():
        y = sc(x)
        assert torch.allclose(sc(2 * x - z), 2 * y - sc(z), atol=1e-5)                       # linear
        assert torch.allclose(sc(torch.roll(x, 7, dims=-1)), torch.roll(y, 7, dims=-1), atol=1e-5)   # shift
        spec = torch.fft.rfft(y, dim=-1)
    assert y.shape == (2, 3, 32) and spec[..., 5:].abs().max() < 1e-5                         # modes >= 5 empty
    sc2 = U.SpectralConvND(2, 1, 1, modes=(4, 4))
    with torch.no_grad():
        x2 = torch.randn(1, 1, 16, 16)
        assert torch.allclose(sc2(torch.roll(x2, (3, 5), dims=(-2, -1))),
                              torch.roll(sc2(x2), (3, 5), dims=(-2, -1)), atol=1e-5)
    with pytest.raises(ValueError):
        sc(torch.randn(1, 2, 8, 8))                                                             # wrong dimension


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_blocks_halve_and_restore_the_resolution(dim):
    _seed()
    n = 8
    x = torch.randn(1, 4, *([n] * dim))
    modes = (2,) * dim
    assert U.OperatorBlockND(dim, 4, modes)(x).shape == x.shape
    down = U.DownBlockND(dim, 4, 6, modes, depth=1)
    out = down(x)
    coarse, skip = out if isinstance(out, tuple) else (out, None)
    assert coarse.shape == (1, 6, *([n // 2] * dim))
    up = U.UpBlockND(dim, 6, 4, 5, modes, depth=1)
    assert up(coarse, x).shape == (1, 5, *([n] * dim))                                          # back to n


@pytest.mark.parametrize("dim, shape", [(1, (16,)), (2, (16, 16)), (3, (8, 8, 8))])
def test_grid_unet_keeps_the_grid(dim, shape):
    _seed()
    net = U.UNONdGrid(dim, 2, 3, width=8, levels=2, depth_per_level=1)
    y = net(torch.randn(2, 2, *shape))
    assert y.shape == (2, 3, *shape) and torch.isfinite(y).all()


def test_point_operator_is_permutation_equivariant():
    _seed()
    op = U.PointKernelOperator(2, 3, 4, hidden=16, k=5)
    blk = U.PointOperatorBlock(2, 4, k=5)
    x, c = torch.randn(1, 20, 3), torch.rand(1, 20, 2)
    p = torch.randperm(20)
    with torch.no_grad():
        y = op(x, c)
        assert torch.allclose(op(x[:, p], c[:, p]), y[:, p], atol=1e-5)
        h = torch.randn(1, 20, 4)
        assert torch.allclose(blk(h[:, p], c[:, p]), blk(h, c)[:, p], atol=1e-5)
    assert y.shape == (1, 20, 4)


def test_point_unet_and_the_universal_wrapper():
    _seed()
    pts = U.UNONdPoints(2, 1, 2, width=16, levels=2, depth_per_level=1, k=4)
    x, c = torch.randn(1, 32, 1), torch.rand(1, 32, 2)
    assert pts(x, c).shape == (1, 32, 2)
    uni = U.UniversalUNO(2, 1, 2, width_grid=8, levels_grid=2, depth_grid=1, width_points=16, levels_points=2,
                         depth_points=1, k_points=4)
    assert uni(torch.randn(1, 1, 16, 16)).y.shape == (1, 2, 16, 16)                           # grid
    assert uni(x, c).y.shape == (1, 32, 2)                                                    # points
    forced = U.UniversalUNO(2, 1, 2, mesh_mode="points", width_points=16, levels_points=2, depth_points=1,
                            k_points=4, width_grid=8, levels_grid=2, depth_grid=1)
    with pytest.raises(ValueError):
        forced(x)
