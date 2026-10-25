"""XTFC reports the formulation it actually ran and does not silently ignore its activation argument."""
import pytest
import torch

from pinneapple_neural.architectures.pinns.xtfc import XTFC, XTFCConfig, build_xtfc


class _BrokenCE:
    """Stands in for TFCConstrainedExpression with the real failure: tfc has no ``cf``."""

    def constrained_expression(self, eta_fn):
        raise AttributeError("'utfc' object has no attribute 'cf'")


class _WorkingCE:
    def constrained_expression(self, eta_fn):
        return lambda x: torch.zeros(x.shape[0], 1)


def _model(**kw):
    torch.manual_seed(0)
    return XTFC(1, 1, rf_dim=8, config=XTFCConfig(in_dim=1, out_dim=1, rf_dim=8, use_tfc=False, **kw))


def test_failed_tfc_falls_back_loudly_and_reports_it():
    m = _model()
    m.set_tfc(_BrokenCE())
    x = torch.linspace(0, 1, 5)[:, None]
    with pytest.warns(RuntimeWarning, match="tfc constrained expression failed"):
        out = m(x)
    assert out.extras["tfc_mode"] is False and out.extras["tfc_attached"] is True
    assert torch.allclose(out.y, m._free_function(x))          # g = 0, B = 1: the manual formulation ran


def test_working_tfc_is_reported():
    m = _model()
    m.set_tfc(_WorkingCE())
    out = m(torch.rand(4, 1))
    assert out.extras["tfc_mode"] is True and torch.all(out.y == 0)


def test_activation_rejected_for_fourier_features():
    with pytest.raises(ValueError, match="no effect with rf_kind='rff'"):
        build_xtfc(1, 1, rf_dim=8, activation="tanh", use_tfc=False)
    with pytest.raises(ValueError):
        XTFC(1, 1, rf_dim=8, activation="sin")


def test_defaults_unchanged():
    m = _model()                                               # rff: cos/sin features
    x = torch.rand(6, 1)
    proj = x @ m.W.t() + m.b[None, :]
    assert torch.allclose(m._rf(x), torch.cat([torch.cos(proj), torch.sin(proj)], -1))
    lin = _model(rf_kind="random_linear")                     # activation None -> tanh, as before
    assert torch.allclose(lin._rf(x), torch.tanh(x @ lin.W.t() + lin.b[None, :]))
    sin = _model(rf_kind="random_linear", activation="sin")   # honoured where it applies
    assert torch.allclose(sin._rf(x), torch.sin(x @ sin.W.t() + sin.b[None, :]))
