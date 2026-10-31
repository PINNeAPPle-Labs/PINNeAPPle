"""A PINN must be a pointwise function u(x, t): the value at one collocation point cannot depend on which other
points share its batch, otherwise autograd residuals (grad of u.sum() w.r.t. the inputs) pick up cross-point terms.

PINNeAPPle's PINNsFormer splits a flat (N, in_dim) batch into sequences of ``seq_len`` unrelated points and attends
across them, so it is not pointwise. Zhao, Ding & Prakash (ICLR 2024) build the sequence from ONE point and its
k time-shifted copies (official code: ``util.make_time_sequence``), which keeps the model pointwise; that model is
``PINNsFormerPaper``. ``PINNsFormer`` keeps its sequence-of-points behaviour (the registry labels it a sequence
model), so its pointwise test stays a strict xfail.
"""
import pytest
import torch

from pinneapple_neural.architectures.pinns.pinnsformer import PINNsFormer


@pytest.mark.xfail(strict=True, reason="PINNsFormer attends across unrelated collocation points (see PINNsFormerPaper)")
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


def test_paper_pinnsformer_is_pointwise_and_matches_the_official_size():
    """PINNsFormerPaper builds the sequence from one point and k time-shifted copies (Zhao et al., eq. 3)."""
    from pinneapple_neural.architectures.pinns.pinnsformer_paper import PINNsFormerPaper

    torch.manual_seed(0)
    m = PINNsFormerPaper().eval()
    assert sum(p.numel() for p in m.parameters()) == 453_561     # the count the official notebooks print
    x = torch.rand(8, 2)
    x_other = x.clone()
    x_other[1:] = torch.rand(7, 2)
    with torch.no_grad():
        assert torch.allclose(m.forward_tensor(x)[0], m.forward_tensor(x_other)[0])
        seq = m.make_sequence(x)
        assert torch.allclose(seq[:, :, 0], x[:, :1].expand(-1, 5))
        assert torch.allclose(seq[:, :, 1] - x[:, 1:], 1e-4 * torch.arange(5.0).expand(8, -1), atol=1e-7)
        assert torch.allclose(m.forward_sequence(x)[:, 0], m.forward_tensor(x))


def test_paper_pinnsformer_has_second_derivatives_and_learnable_wave_activations():
    from pinneapple_neural.architectures.pinns.pinnsformer_paper import PINNsFormerPaper, WaveAct

    torch.manual_seed(0)
    m = PINNsFormerPaper(k=3, d_hidden=16, d_ff=16)
    x = torch.rand(6, 2, requires_grad=True)
    u = m.forward_sequence(x)
    g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    gg = torch.autograd.grad(g[:, 1].sum(), x)[0]                # u_tt exists (wave residual)
    assert torch.isfinite(gg).all()
    acts = [a for a in m.modules() if isinstance(a, WaveAct)]
    assert len(acts) == 12 and all(float(a.w1) == 1.0 and float(a.w2) == 1.0 for a in acts)
