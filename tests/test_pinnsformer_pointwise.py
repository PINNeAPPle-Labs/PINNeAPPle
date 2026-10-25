"""A PINN must be a pointwise function u(x, t): the value at one collocation point cannot depend on which other
points share its batch, otherwise autograd residuals (grad of u.sum() w.r.t. the inputs) pick up cross-point terms.

PINNeAPPle's PINNsFormer splits a flat (N, in_dim) batch into sequences of ``seq_len`` unrelated points and attends
across them, so it is not pointwise. Zhao, Ding & Prakash (ICLR 2024) build the sequence from ONE point and its
k time-shifted copies (official code: ``util.make_time_sequence``), which keeps the model pointwise. Known bug,
reported in examples/paper_reproductions; the xfail flips to a failure (strict) once it is fixed.
"""
import pytest
import torch

from pinneapple_neural.architectures.pinns.pinnsformer import PINNsFormer


@pytest.mark.xfail(strict=True, reason="PINNsFormer attends across unrelated collocation points (known bug)")
def test_pinnsformer_is_pointwise():
    torch.manual_seed(0)
    m = PINNsFormer(2, 1, seq_len=4, d_model=16, nhead=2, num_layers=1, dim_feedforward=32).eval()
    x = torch.rand(8, 2)
    x_other = x.clone()
    x_other[1:4] = torch.rand(3, 2)                     # same first point, other points in its chunk
    with torch.no_grad():
        a = m.forward_tensor(x)[0]
        b = m.forward_tensor(x_other)[0]
    assert torch.allclose(a, b, atol=1e-6)
