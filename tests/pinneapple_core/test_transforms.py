import numpy as np
import pytest

import pinneapple as pp
from pinneapple_core.data import PhysicsDataset
from pinneapple_core.transforms import (
    Compose, Coordinate, FourierFeatures, Nondimensionalize, NotInvertible, Periodic, Scale, Symmetry, list_pde_rules,
)

rng = np.random.default_rng(0)


def _table(n=50):
    return {"x": rng.uniform(-2, 3, n), "y": rng.uniform(-1, 4, n), "z": rng.uniform(0.5, 2, n), "t": rng.uniform(0, 5, n),
            "u": rng.normal(size=n), "v": rng.normal(size=n)}


def _same(a, b, tol=1e-10):
    assert set(a) == set(b)
    for k in a:
        assert np.allclose(a[k], b[k], atol=tol), k


# -- tables: forward/inverse ---------------------------------------------------------
def test_scale_roundtrip_and_values():
    t = _table()
    s = Scale({"x": 2.0, "u": 0.5}, {"x": 1.0})
    out = s.forward(t)
    assert np.allclose(out["x"], (t["x"] - 1.0) / 2.0) and np.allclose(out["u"], t["u"] / 0.5) and np.allclose(out["y"], t["y"])
    _same(s.inverse(out), t)
    with pytest.raises(ValueError):
        Scale({"x": -1.0})


@pytest.mark.parametrize("kind,ins,outs", [("polar", ("x", "y"), ("r", "th")), ("cylindrical", ("x", "y", "z"), ("r", "th", "z")),
                                           ("spherical", ("x", "y", "z"), ("r", "th", "ph")), ("log", ("z",), ("lz",))])
def test_coordinate_roundtrip(kind, ins, outs):
    t = _table()
    c = Coordinate(kind, ins, outs, center=None)
    out = c.forward(t)
    assert all(o in out for o in outs) and not (set(ins) - set(outs)) & set(out)
    _same(c.inverse(out), t)
    if kind == "polar":
        assert np.allclose(out["r"], np.hypot(t["x"], t["y"]))
    c2 = Coordinate("polar", ("x", "y"), ("r", "th"), center=(1.0, -1.0)) if kind == "polar" else c
    _same(c2.inverse(c2.forward(t)), t)


def test_coordinate_errors():
    with pytest.raises(ValueError):
        Coordinate("polar", ("x",), ("r",))
    with pytest.raises(ValueError):
        Coordinate("bogus", ("x",), ("y",))
    with pytest.raises(ValueError, match="positive"):
        Coordinate("log", ("x",), ("lx",)).forward({"x": np.array([-1.0, 1.0])})
    with pytest.raises(KeyError):
        Coordinate("polar", ("x", "q"), ("r", "t")).forward(_table())


def test_symmetry_reflect_rotate_vectors_and_roundtrip():
    t = _table()
    r = Symmetry.reflect("x", ("x", "y"), vectors=[("u", "v")], center=(0.5, 0.0))
    out = r.forward(t)
    assert np.allclose(out["x"], 1.0 - t["x"]) and np.allclose(out["u"], -t["u"]) and np.allclose(out["v"], t["v"])
    _same(r.forward(out), t)                                              # a reflection is its own inverse
    rot = Symmetry.rotate(np.pi / 2, ("x", "y"), vectors=[("u", "v")])
    out = rot.forward(t)
    assert np.allclose(out["x"], -t["y"]) and np.allclose(out["y"], t["x"]) and np.allclose(out["u"], -t["v"]) and np.allclose(out["v"], t["u"])
    _same(rot.inverse(out), t)
    # speed and radius are invariant
    assert np.allclose(np.hypot(out["u"], out["v"]), np.hypot(t["u"], t["v"]))
    with pytest.raises(ValueError):
        Symmetry.reflect("w", ("x", "y"))
    with pytest.raises(ValueError):
        Symmetry("rotate", ("x", "y", "z"))


def test_periodic_embedding_is_periodic_and_invertible():
    t = {"x": np.linspace(0.0, 2.0, 41, endpoint=False), "u": np.arange(41.0)}
    p = Periodic("x", period=2.0, harmonics=3)
    out = p.forward(t)
    assert set(out) == {"u", "x_cos1", "x_sin1", "x_cos2", "x_sin2", "x_cos3", "x_sin3"}
    shifted = p.forward({"x": t["x"] + 2.0, "u": t["u"]})
    assert all(np.allclose(out[k], shifted[k]) for k in out)               # f(x + period) == f(x)
    _same(p.inverse(out), t)
    with pytest.raises(ValueError):
        Periodic("x", 0.0)


def test_fourier_features_columns_module_and_inverse():
    t = _table(20)
    ff = FourierFeatures(("x", "y"), n_features=5, scale=2.0, seed=7)
    out = ff.forward(t)
    assert len(out) == len(t) + 10 and np.all(np.abs(out["ff_cos_0"]) <= 1)
    _same(ff.inverse(out), t)
    torch = pytest.importorskip("torch")
    x = torch.tensor(np.stack([t["x"], t["y"]], 1), dtype=torch.float32)
    layer = ff.module()
    feats = layer(x).numpy()
    assert feats.shape == (20, 10) and np.allclose(feats[:, 0], out["ff_cos_0"], atol=1e-5)
    assert np.allclose(FourierFeatures(("x", "y"), 5, 2.0, 7).B, ff.B)      # same seed, same frequencies
    assert not np.allclose(FourierFeatures(("x", "y"), 5, 2.0, 8).B, ff.B)
    torch_out = ff.forward({"x": torch.tensor(t["x"]), "y": torch.tensor(t["y"])})
    assert np.allclose(torch_out["ff_cos_0"].numpy(), out["ff_cos_0"])


# -- composition ------------------------------------------------------------------------
def test_compose_order_inverse_and_non_invertible_steps():
    t = _table()
    chain = Scale({"x": 2.0, "y": 2.0}) >> Coordinate("polar", ("x", "y"), ("r", "th")) >> Periodic("th", 2 * np.pi, 2, lo=-np.pi)
    assert isinstance(chain, Compose) and chain.invertible and [s.name for s in chain.steps()] == ["scale", "coordinate", "periodic"]
    out = chain.forward(t)
    assert "th_cos2" in out and np.allclose(out["r"], np.hypot(t["x"], t["y"]) / 2.0)
    _same(chain.inverse(out), t, 1e-9)
    assert [r["name"] for r in chain.provenance()] == ["scale", "coordinate", "periodic"]

    class OneWay(Scale):
        invertible = False
        name = "one_way"
        def inverse(self, table):
            raise NotInvertible("no")

    bad = Scale({"x": 2.0}) >> OneWay({"y": 3.0})
    assert not bad.invertible
    with pytest.raises(NotInvertible, match="one_way"):
        bad.inverse(bad.forward(t))
    with pytest.raises(ValueError):
        Compose([])


# -- datasets -------------------------------------------------------------------------------
def test_dataset_transform_records_provenance_and_keeps_pairs():
    x = rng.uniform(0, 4, (30, 2))
    ds = PhysicsDataset(x, np.sin(x[:, :1]), coords=["x", "y"], fields=["u"])
    chain = Scale({"x": 4.0, "y": 4.0}) >> Scale({"u": 0.5})
    out = chain.apply(ds)
    assert np.allclose(out.inputs.numpy(), x / 4.0) and np.allclose(out.targets.numpy(), np.sin(x[:, :1]) / 0.5)
    assert out.coords == ("x", "y") or list(out.coords) == ["x", "y"]
    assert [r["name"] for r in out.provenance] == ["scale", "scale"] and ds.provenance == []
    a, _ = out.split([0.5, 0.5])
    assert len(a.provenance) == 2                                          # provenance survives splitting
    ff = FourierFeatures(("x", "y"), 3).apply(ds)
    assert ff.inputs.shape[1] == 2 + 6 and ff.provenance[-1]["name"] == "fourier_features"
    with pytest.raises(ValueError, match="coords"):
        Scale({"x": 2.0}).apply(PhysicsDataset(x))


# -- problems -------------------------------------------------------------------------------------
def test_nondimensionalize_burgers_problem_rescales_domain_parameters_and_records_provenance():
    p = pp.PhysicalProblem.from_preset("burgers_1d", nu=0.01)
    nd = Nondimensionalize(p)
    q = nd.apply(p)
    assert q.domain_bounds == {"x": (0.0, 1.0), "t": (0.0, 1.0)} and q.field_ranges["u"] == (-0.5, 0.5)
    L, T, U = 2.0, 1.0, 2.0
    assert q.parameter_values()["nu"] == pytest.approx(0.01 * T / L ** 2)
    assert q.to_pde_spec().pde.params["nu"] == pytest.approx(0.0025)
    rec = q.metadata["transforms"][0]
    assert rec["name"] == "nondimensionalize" and rec["input_fingerprint"] == p.fingerprint() and rec["output_fingerprint"] == q.fingerprint()
    assert p.fingerprint() != q.fingerprint() and q.reference_solver == {} and q.validate() == []
    assert p.parameter_values()["nu"] == 0.01 and p.domain_bounds["x"] == (-1.0, 1.0)         # the original is untouched
    q2 = Scale({"x": 1.0}).apply(q)
    assert [r["name"] for r in q2.metadata["transforms"]] == ["nondimensionalize", "scale"]


def test_nondimensional_burgers_solution_equals_the_original_after_pull_back():
    """u(x,t) of the original problem and the pulled-back solution of the scaled one are the same function."""
    from pinneapple_physics.closed_form.burgers import burgers_sine_exact

    nu = 0.01 / np.pi
    p = pp.PhysicalProblem.from_preset("burgers_1d", nu=nu)
    nd = Nondimensionalize(p)
    # exact solution in the scaled variables: u'(x', t') = u(L x' + x0, T t') / U with the nondimensional nu'
    nu2 = nd.apply(p).parameter_values()["nu"]
    assert nu2 == pytest.approx(nu * 1.0 / 4.0)
    predict_scaled = lambda Xn: (burgers_sine_exact(Xn[:, 0] * 2.0 - 1.0, Xn[:, 1] * 1.0, nu) / 2.0)[:, None]
    # the scaled exact solution satisfies u'_t' + u' u'_x' = nu' u'_x'x' with the rescaled nu' (finite-difference residual)
    X, Tt = np.meshgrid(np.linspace(0.15, 0.85, 40), np.linspace(0.2, 0.8, 20), indexing="ij")
    h = 1e-4
    f = lambda a, b: predict_scaled(np.stack([a.ravel(), b.ravel()], 1)).reshape(a.shape)
    ut = (f(X, Tt + h) - f(X, Tt - h)) / (2 * h)
    ux = (f(X + h, Tt) - f(X - h, Tt)) / (2 * h)
    uxx = (f(X + h, Tt) - 2 * f(X, Tt) + f(X - h, Tt)) / h ** 2
    assert np.abs(ut + f(X, Tt) * ux - nu2 * uxx).max() < 5e-3
    assert np.abs(ut + f(X, Tt) * ux - nu * uxx).max() > 5 * np.abs(ut + f(X, Tt) * ux - nu2 * uxx).max()   # the wrong nu fails
    back = nd.pull_back(predict_scaled, p.coords, p.fields)
    X = np.stack([rng.uniform(-1, 1, 40), rng.uniform(0, 1, 40)], 1)
    assert np.allclose(back(X)[:, 0], burgers_sine_exact(X[:, 0], X[:, 1], nu))


def test_scale_rewrites_conditions_selectors_and_sources():
    from pinneapple_physics.pde_environment.builder import ProblemBuilder

    spec = (ProblemBuilder("poisson_box").domain(x=(0, 2), y=(0, 4)).fields("u").pde("poisson")
            .bc("dirichlet", field="u", value=lambda X, ctx=None: X[:, :1] * 10.0, on="x_min")
            .bc("neumann", field="u", value=3.0, on="x_max").build())
    p = pp.PhysicalProblem.from_pde_spec(spec)
    s = Scale({"x": 2.0, "y": 2.0, "u": 5.0}).apply(p)
    assert s.domain_bounds == {"x": (0.0, 1.0), "y": (0.0, 2.0)}
    dirich, neu = s.conditions
    Xn = np.array([[0.0, 0.5], [1.0, 0.5]])
    assert list(dirich.selector(Xn, None)) == [True, False]                  # x'=0 is the old x_min
    assert np.allclose(dirich.value_fn(np.array([[0.4, 0.5]]), None), (0.8 * 10.0) / 5.0)
    assert np.allclose(neu.value_fn(Xn, None), 3.0 * 2.0 / 5.0)              # g' = g L / U
    f = Scale({"x": 2.0, "y": 2.0, "u": 5.0}).source(lambda X, ctx=None: np.sin(X[:, 0]), p)
    assert np.allclose(f(Xn), np.sin(Xn[:, 0] * 2.0) * (2.0 ** 2 / 5.0))      # f' = f L^2 / U
    with pytest.raises(ValueError, match="anisotropic"):
        Scale({"x": 2.0, "y": 3.0}).apply(p)


def test_problem_level_refusals_are_explicit():
    p = pp.PhysicalProblem.from_preset("burgers_1d")
    with pytest.raises(ValueError, match="U\\*T/L"):
        Scale({"x": 2.0, "u": 5.0}).apply(p)                                  # breaks the convective coefficient
    with pytest.raises(ValueError, match="Galilean"):
        Scale({"x": 2.0, "u": 2.0}, {"u": 1.0}).apply(p)
    with pytest.raises(ValueError, match="no scaling rule"):
        Scale({"x": 2.0}).apply(pp.PhysicalProblem.from_preset("heston_pde_2d"))
    with pytest.raises(NotImplementedError, match="problem-level"):
        Periodic("x", 1.0).apply(p)
    with pytest.raises(NotImplementedError):
        Coordinate("log", ("x",), ("lx",)).apply(p)
    with pytest.raises(TypeError):
        Scale({"x": 1.0}).apply(42)
    assert {"burgers", "poisson", "heat_equation"} <= set(list_pde_rules())


def test_namespace():
    assert pp.transforms.Scale is Scale
