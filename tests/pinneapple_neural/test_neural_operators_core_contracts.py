"""Core coverage (#1129, toward 1.0) for the neural operators: FNO, its exportable real-arithmetic form, the
Galerkin operator, Transolver blocks, DeepONet's MLP, the base class, the registry and the Noether bridge.

Exact references:
* the real-arithmetic export of an FNO-2D computes the same function as the complex one;
* an FNO is equivariant to circular shifts when it has no coordinate channel;
* the Galerkin operator's basis is orthonormal for its quadrature weights, and its output lies in that basis;
* physics attention, like any attention without positional input, is equivariant to a permutation of the points.
"""
import pytest

torch = pytest.importorskip("torch")

from pinneapple_neural.architectures.neural_operators import base as B  # noqa: E402
from pinneapple_neural.architectures.neural_operators import deeponet as DO  # noqa: E402
from pinneapple_neural.architectures.neural_operators import fno as F  # noqa: E402
from pinneapple_neural.architectures.neural_operators import fno_export as FE  # noqa: E402
from pinneapple_neural.architectures.neural_operators import gno as G  # noqa: E402
from pinneapple_neural.architectures.neural_operators import noether_bridge as NB  # noqa: E402
from pinneapple_neural.architectures.neural_operators import registry as R  # noqa: E402
from pinneapple_neural.architectures.neural_operators import transolver as T  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_fno_1d_is_shift_equivariant_without_a_grid_channel():
    fno = F.FourierNeuralOperator(1, 2, width=8, modes=6, layers=2, use_grid=False)
    u = torch.randn(2, 1, 32)
    with torch.no_grad():
        y = fno(u).y
        assert y.shape == (2, 2, 32)
        assert torch.allclose(fno(torch.roll(u, 5, dims=-1)).y, torch.roll(y, 5, dims=-1), atol=1e-5)
    g = F.FourierNeuralOperator(1, 1, width=8, modes=6, layers=1, use_grid=True)
    assert g(u).y.shape == (2, 1, 32)


def test_real_arithmetic_export_computes_the_same_function():
    model = F.FNO2d(1, 1, width=8, modes1=4, modes2=4, layers=2).eval()
    exp = FE.to_exportable(model, (16, 16))
    assert any(isinstance(m, FE.RealSpectralConv2d) for m in exp.modules())
    x = torch.randn(3, 1, 16, 16)
    with torch.no_grad():
        assert torch.allclose(exp(x).y, model(x).y, atol=1e-5)                       # exact up to rounding
    assert not any(p.requires_grad for p in exp.parameters())


def test_mlp_fno_surrogate_evaluates_at_scattered_points():
    s = F.MLPFNOSurrogate(1, 2, fno_width=8, fno_modes=4, fno_layers=1, mlp_hidden=16, mlp_layers=2)
    out = s(torch.randn(2, 1, 16, 16), torch.rand(2, 10, 2))
    assert out.shape == (2, 10, 2) and torch.isfinite(out).all()


def test_galerkin_basis_is_orthonormal_and_spans_the_output():
    g = G.GalerkinNeuralOperator(in_dim=1, out_dim=2, basis_dim=6, basis_hidden=16, op_hidden=16)
    coords = torch.rand(40, 2)
    w = torch.rand(1, 40) + 0.1
    w = w / w.sum()
    out = g(torch.randn(40, 1), coords=coords, w=w[0])
    Q = out.extras["Phi"][0]                                                           # sqrt(w)-weighted basis
    assert torch.allclose(Q.T @ Q, torch.eye(6), atol=1e-5)                          # orthonormal
    y = out.y                                                                          # unbatched: (N, out)
    assert y.shape == (40, 2) and torch.allclose(Q @ (Q.T @ y), y, atol=1e-5)        # y lies in span(Q)


def test_physics_attention_is_permutation_equivariant():
    att = T.PhysicsAttention(16, heads=4, slices=4).eval()
    blk = T.TransolverBlock(16, 4, 4).eval()
    x = torch.randn(2, 30, 16)
    p = torch.randperm(30)
    with torch.no_grad():
        assert torch.allclose(att(x[:, p]), att(x)[:, p], atol=1e-5)
        assert torch.allclose(blk(x[:, p]), blk(x)[:, p], atol=1e-5)
    with pytest.raises(ValueError):
        T.PhysicsAttention(10, heads=4)


def test_deeponet_mlp_widths_and_the_base_mse():
    m = DO.mlp(3, [8, 5], 2)
    lins = [mod for mod in m if isinstance(mod, torch.nn.Linear)]
    assert [(lin.in_features, lin.out_features) for lin in lins] == [(3, 8), (8, 5), (5, 2)]
    m2 = DO.mlp(3, 4, 1, depth=3, last_act=True)
    assert len([mod for mod in m2 if isinstance(mod, torch.nn.Linear)]) == 4 and isinstance(m2[-1], torch.nn.GELU)
    assert B.NeuralOperatorBase.mse(torch.tensor([1.0, 3.0]), torch.zeros(2)).item() == pytest.approx(5.0)
    assert isinstance(B.NeuralOperatorBase(), torch.nn.Module)


def test_registry_registers_the_family(monkeypatch):
    from pinneapple_neural.architectures.registry import ModelRegistry
    monkeypatch.setattr(ModelRegistry, "_REGISTRY", {})                              # a clean global registry
    R.register_into_global()
    reg = ModelRegistry._REGISTRY
    assert {"fno", "deeponet"} <= set(reg)
    assert reg["pino"].supports_physics_loss and not reg["fno"].supports_physics_loss
    with pytest.raises(KeyError, match="already registered"):
        R.register_into_global()                                                      # names are unique


def test_noether_wrappers_are_research_only_and_lazy(monkeypatch):
    cfg = NB.NoetherModelConfig()
    assert cfg.model_name == "upt"
    monkeypatch.delenv("PINNEAPPLE_COMMERCIAL_MODE", raising=False)
    for cls in (NB.NoetherUPT, NB.NoetherTransformer, NB.NoetherTransolver, NB.NoetherAeroUPT,
                NB.NoetherAeroABUPT, NB.NoetherAeroTransformer, NB.NoetherAeroTransolver):
        with pytest.warns(UserWarning, match="Noether"):
            m = cls(dim=8)                                                            # lazy: nothing built yet
        assert m._inner is None
    try:
        import noether  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="emmiai-noether"):
            m(torch.zeros(1, 4, 3))
    monkeypatch.setenv("PINNEAPPLE_COMMERCIAL_MODE", "1")
    with pytest.raises(RuntimeError, match="COMMERCIAL"):
        NB.NoetherUPT()
