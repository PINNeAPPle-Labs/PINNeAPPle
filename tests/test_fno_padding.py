"""Domain padding of the FNOs (#1162): cells of zeros appended after the lifting and removed before the projection,
for non-periodic domains (Li et al. 2021 pad Darcy).

References:
- The spectral convolution is circular: without padding (and without the grid channel) a 1-D FNO commutes with a
  circular shift of its input. With padding the edges no longer wrap around, so it does not.
- Padding 0 is the model as it was: the same weights give the same output.
- The real-arithmetic export (fno_export) of a padded FNO2d computes the same function on the padded grid.
The measured effect on Darcy is in benchmarks/fno_padding_darcy.py and its committed result.
"""
import pytest
import torch

from pinneapple_neural.architectures.neural_operators.fno import FNO2d, FNO3d, FourierNeuralOperator


def _pair(cls, padding, **kw):
    torch.manual_seed(0)
    a = cls(1, 1, width=8, **kw)
    b = cls(1, 1, width=8, padding=padding, **kw)
    b.load_state_dict(a.state_dict())
    return a.eval(), b.eval()


@pytest.mark.parametrize("cls,shape,kw", [
    (FourierNeuralOperator, (2, 1, 32), {"modes": 8}),
    (FNO2d, (2, 1, 16, 20), {"modes1": 4, "modes2": 4}),
    (FNO3d, (1, 1, 8, 8, 10), {"modes1": 2, "modes2": 2, "modes3": 2}),
])
def test_padding_keeps_the_shape_and_padding_zero_is_the_old_model(cls, shape, kw):
    x = torch.randn(*shape)
    a, b = _pair(cls, 4, **kw)
    _, z = _pair(cls, 0, **kw)
    with torch.no_grad():
        ya, yb, yz = a(x).y, b(x).y, z(x).y
    assert ya.shape == yb.shape == x.shape
    assert torch.equal(ya, yz)
    assert (ya - yb).abs().max() > 1e-6


def test_padding_per_axis_and_errors():
    m = FNO2d(1, 1, width=8, modes1=4, modes2=4, padding=(6, 0))
    assert m.padding == (6, 0) and m(torch.randn(1, 1, 16, 16)).y.shape == (1, 1, 16, 16)
    with pytest.raises(ValueError, match="one value per spatial axis"):
        FNO2d(1, 1, padding=(1, 2, 3))
    with pytest.raises(ValueError, match=">= 0"):
        FourierNeuralOperator(1, 1, padding=-1)


def test_without_padding_the_edges_wrap_around_and_with_padding_they_do_not():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64)
    shift = lambda t: torch.roll(t, 7, dims=-1)                                # noqa: E731
    a, b = _pair(FourierNeuralOperator, 16, modes=12)
    with torch.no_grad():
        scale = a(x).y.abs().max().item()
        circ = (a(shift(x)).y - shift(a(x).y)).abs().max().item() / scale
        padded = (b(shift(x)).y - shift(b(x).y)).abs().max().item() / scale
    assert circ < 1e-5                       # float32 round-off: circular
    assert padded > 1e-3 and padded > 1000 * circ   # the edges no longer wrap around


def test_the_real_arithmetic_export_of_a_padded_fno2d_is_the_same_function():
    from pinneapple_neural.architectures.neural_operators.fno_export import _Plain, to_exportable
    torch.manual_seed(0)
    m = FNO2d(1, 1, width=8, modes1=4, modes2=4, padding=(5, 3)).eval()
    x = torch.randn(2, 1, 16, 20)
    real = to_exportable(_Plain(m).eval(), (21, 23))
    with torch.no_grad():
        assert (real(x) - m(x).y).abs().max().item() < 1e-5
