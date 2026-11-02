"""Core coverage (#1129, toward 1.0) for the PINN solver's pieces and the physics front door: coordinate mappings
and UPD sampling, symbolic derivatives, the SymPy backend, domain decomposition, datasets, electromagnetic residuals,
turbulence closures, condition builders and ``solving``'s registry.

Exact references:
* min-max normalisation maps a coordinate onto [-1, 1] with derivative scale 2 / (max - min);
* sampled supervised targets equal the field at the sampled coordinates (a field linear in lat and lon);
* derivatives of a polynomial; the overlap blending of identical subnetworks is the subnetwork itself;
* electrostatics with phi = x^2 + y^2 and rho = -4 eps has zero residual;
* laminar Couette flow with no eddy viscosity has zero Spalart-Allmaras residuals, and WALE's eddy viscosity
  vanishes in pure shear (Nicoud & Ducros 1999).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
sp = pytest.importorskip("sympy")
xr = pytest.importorskip("xarray")

from pinneapple_data.upd_types import SamplingSpec  # noqa: E402
from pinneapple_physics import solving as SO  # noqa: E402
from pinneapple_physics.pde_environment import conditions as CN  # noqa: E402
from pinneapple_physics.pde_environment import turbulence_presets as TP  # noqa: E402
from pinneapple_physics.pde_environment import turbulence_selector as TS  # noqa: E402
from pinneapple_physics.pinn_solver import domino as DM  # noqa: E402
from pinneapple_physics.pinn_solver.compiler import dataset as DS  # noqa: E402
from pinneapple_physics.pinn_solver.compiler import em_kinds as EK  # noqa: E402
from pinneapple_physics.pinn_solver.factory import autodiff as AD  # noqa: E402
from pinneapple_physics.pinn_solver.factory import sympy_backend as SB  # noqa: E402
from pinneapple_physics.pinn_solver.io import mappings as MP  # noqa: E402
from pinneapple_physics.pinn_solver.io import upd_dataset as UD  # noqa: E402
from pinneapple_physics.pinn_solver.schemas import schema_adapter as SA  # noqa: E402


# ------------------------------------------------------------------------------------------------ UPD and mappings
def _shard():
    t = np.array(["2020-01-01T00", "2020-01-01T06", "2020-01-01T12"], dtype="datetime64[ns]")
    lat, lon = np.linspace(-30.0, 30.0, 7), np.linspace(0.0, 90.0, 10)
    T = np.broadcast_to(lat[None, :, None] + 10.0 * lon[None, None, :], (3, 7, 10)).copy()
    return xr.Dataset({"T2M": (("time", "lat", "lon"), T)}, coords={"time": t, "lat": lat, "lon": lon})


class _Item:
    def load_meta(self):
        return {"source": "test"}

    def open_dataset(self):
        return _shard()


def test_coordinate_mapping_normalises_and_scales_derivatives():
    m = MP.build_default_mapping_atmosphere(["T"])
    assert isinstance(m, MP.PINNMapping) and isinstance(m.coord, MP.CoordMapping) and isinstance(m.var, MP.VarMapping)
    assert m.independent_vars() == ["t", "lat", "lon"] and m.dependent_vars() == ["T"]
    ds = _shard()
    c = m.coord.make_coord_arrays(ds, time0=m.infer_time0(ds))
    assert c["lat"].min() == -1.0 and c["lat"].max() == 1.0                              # min-max to [-1, 1]
    assert m.coord.derivative_scale("lat") == pytest.approx(2.0 / np.deg2rad(60.0))      # 2 / (max - min)
    assert m.derivative_scales()["t"] == pytest.approx(2.0 / (12 * 3600.0))
    v = MP.VarMapping(["T"], {"T": "T2M"})
    assert list(v.pick_dataset_vars(ds).data_vars) == ["T2M"]
    with pytest.raises(KeyError):
        MP.VarMapping(["Q"], {"Q": "QV"}).pick_dataset_vars(ds)


def test_upd_dataset_samples_targets_at_the_sampled_points():
    m = MP.build_default_mapping_atmosphere(["T"])
    m.var.upd_to_dep["T"] = "T2M"
    ds = UD.UPDDataset(_Item(), m)
    dims = ds.get_problem_dims()
    assert dims["available_vars"] == ["T2M"] and dims["independent_vars"] == ["t", "lat", "lon"]
    b = ds.sample(SamplingSpec(n_collocation=50, n_data=40, seed=0))
    assert len(b.collocation) == 3 and b.collocation[0].shape == (50, 1)
    (t, la, lo), y = b.data
    lat = (la.numpy().ravel() + 1) / 2 * 60.0 - 30.0                                     # undo deg2rad + min-max
    lon = (lo.numpy().ravel() + 1) / 2 * 90.0
    assert np.allclose(y.numpy().ravel(), lat + 10.0 * lon, atol=1e-3)                   # exact field values


def test_schema_adapter_turns_a_schema_into_a_problem():
    m = MP.build_default_mapping_atmosphere(["T"])
    opts = SA.SchemaAdapterOptions()
    assert opts.default_loss_weights and opts.default_condition_weight == 1.0
    schema = {"equations": {"residuals": ["Derivative(T(t,lat,lon), t)"]},
              "conditions": [{"type": "initial", "name": "ic", "equation": "T(t,lat,lon) - 1"}]}
    problem, sampling = SA.SchemaAdapter(m, opts).to_pinn_problem(schema)
    assert problem.pde_residuals == ["Derivative(T(t,lat,lon), t)"]
    assert problem.independent_vars == ["t", "lat", "lon"] and problem.dependent_vars == ["T"]


# ------------------------------------------------------------------------------------------------ autodiff, SymPy
def test_symbolic_derivatives_are_computed_exactly():
    x, t = sp.symbols("x t")
    u = sp.Function("u")(x, t)
    d = sp.Derivative(u, x, t, t)
    dc = AD.DerivativeComputer(["x", "t"], ["u"], torch.device("cpu"))
    assert dc.sympy_derivative_to_key(d) == AD.DerivativeKey("u", (("t", 2), ("x", 1)))

    class Poly(torch.nn.Module):
        def forward(self, x, t):
            return x ** 2 * t ** 3
    X, T = torch.rand(8, 1), torch.rand(8, 1)
    out = dc.compute(Poly(), (X, T), [d, sp.Derivative(u, x, 2)], {"u": u})
    assert torch.allclose(out[d], 12 * X * T)                                            # d/dx d2/dt2 (x^2 t^3)
    assert torch.allclose(out[sp.Derivative(u, x, 2)], 2 * T ** 3)
    a = torch.zeros(3)
    AD.ensure_requires_grad([a])
    assert a.requires_grad
    with pytest.raises(TypeError):
        AD.ensure_requires_grad([np.zeros(2)])


def test_sympy_backend_makes_a_compiler():
    comp = SB.SympyBackend().make_compiler(independent_vars=["x"], dependent_vars=["u"])
    eq = comp.compile("Derivative(u(x), x) - u(x)")
    assert isinstance(eq, SB.CompiledEquation) and len(eq.derivatives) == 1
    ce = SB.CompiledEquation(expr=sp.Symbol("a"), derivatives=set(), call=lambda a: a, args_order=[sp.Symbol("a")])
    assert ce.call(3) == 3 and ce.exogenous == [] and ce.runtime_constants == []


# ------------------------------------------------------------------------------------------------ domain decomposition
def test_domino_partition_and_blending():
    subs = DM.DoMINO.partition(bounds=[(0, 1), (0, 1)], n_splits=(2, 2), overlap=0.1)
    assert len(subs) == 4
    torch.manual_seed(0)
    shared = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    dom = DM.DoMINO(subs, model_factory=lambda: shared)                                   # identical subnetworks
    x = torch.rand(30, 2)
    with torch.no_grad():
        y = dom(x)
        assert torch.allclose(y if torch.is_tensor(y) else y.y, shared(x), atol=1e-6)     # weights sum to one
    s0 = DM.SubdomainPINN(subs[0], shared)
    assert torch.equal(s0(x), shared(x))


# ------------------------------------------------------------------------------------------------ datasets
def test_single_batch_dataset_cycles_the_small_sets():
    batch = {"x_col": torch.arange(10.0)[:, None], "x_bc": torch.arange(3.0)[:, None] + 100,
             "y_bc": torch.zeros(3, 1)}
    ds = DS.SingleBatchDataset(batch)
    assert len(ds) == 10 and float(ds[4]["x_bc"]) == 101.0                               # 4 % 3 = 1
    out = DS.dict_collate([ds[0], ds[1]])
    assert out["x_col"].tolist() == [[0.0], [1.0]] and out["x_bc"].tolist() == [[100.0], [101.0]]


# ------------------------------------------------------------------------------------------------ residuals
def test_electrostatics_residual_vanishes_for_an_exact_pair():
    X = torch.rand(20, 2, requires_grad=True)
    phi = (X ** 2).sum(dim=1, keepdim=True)                                              # laplacian = 4
    r = EK.em_residuals("electrostatics", fields={"phi": phi}, xcol=X, coords=["x", "y"],
                        params={"eps": 2.0, "rho_fn": lambda Xn, ctx: np.full((len(Xn), 1), -8.0)}, ctx={},
                        model=None, field_names=["phi"], split_fields=None)
    assert torch.allclose(r[0], torch.zeros_like(r[0]), atol=1e-5)                       # eps lap(phi) + rho = 0
    with pytest.raises(ValueError):
        EK.em_residuals("electrostatics", fields={}, xcol=X, coords=["x", "y"], params={}, ctx={}, model=None,
                        field_names=["phi"], split_fields=None)


def test_turbulence_closures_on_laminar_couette_flow():
    def couette2d(x):                                    # u = y, v = 0, p = 0, nu_tilde = 0
        return torch.cat([x[:, 1:2], 0 * x[:, :1], 0 * x[:, :1], 0 * x[:, :1]], dim=1)
    sa = TP.get_rans_preset("sa", nu=1e-3)
    assert isinstance(sa, TP.SpalartAllmarasResiduals)
    x = torch.rand(25, 2) + 0.1
    r = sa(couette2d, x.requires_grad_(True), wall_distance_fn=lambda x: x[:, 1:2])
    for k, v in r.items():
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-6), k                      # exact laminar solution
    wale = TP.WALEResiduals(nu=1e-3, delta=0.05)

    def couette3d(x):                                    # u = y, v = w = p = 0 (x, y, z, t)
        return torch.cat([x[:, 1:2], 0 * x[:, :1], 0 * x[:, :1], 0 * x[:, :1]], dim=1)
    rw = wale(couette3d, (torch.rand(25, 4)).requires_grad_(True))
    for k, v in rw.items():
        if torch.is_tensor(v) and v.numel() > 1:
            assert torch.allclose(v, torch.zeros_like(v), atol=1e-6), k                  # nu_sgs = 0 in pure shear
    assert isinstance(TP.get_rans_preset("wale"), TP.WALEResiduals)
    with pytest.raises((KeyError, ValueError)):
        TP.get_rans_preset("nope")
    assert TS.TurbulenceModel("les_wale") is TS.TurbulenceModel.LES_WALE and TS.TurbulenceModel.LAMINAR == "laminar"


# ------------------------------------------------------------------------------------------------ conditions, solving
def test_condition_builders():
    n = CN.NeumannBC({"u": 1.0})
    ic = CN.InitialCondition({"u": 0.0})
    dc = CN.DataConstraint({"u": 2.0})
    assert (n.kind, ic.kind, dc.kind) == ("neumann", "initial", "data")
    assert n.fields == ic.fields == dc.fields == ("u",)


def test_solving_registry_and_problem_conversion():
    names = SO.list_analytic()
    assert names == [n for n, _, _ in SO._ANALYTIC]

    @SO.register_analytic("_test_never", lambda p: False)
    def _factory(problem):
        return lambda X: X

    try:
        assert SO.list_analytic()[-1] == "_test_never"
    finally:
        SO._ANALYTIC.pop()
    p = SO.as_problem("burgers_1d")
    assert SO.as_problem(p) is p
    with pytest.raises(TypeError):
        SO.as_problem(p, nu=0.1)
    with pytest.raises(TypeError):
        SO.as_problem(3.14)
    assert issubclass(SO.MethodNotAvailable, RuntimeError)
    with pytest.raises(SO.MethodNotAvailable):
        raise SO.MethodNotAvailable("no solver output")
