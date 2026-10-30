"""The shared foundation of pinneapple_analysis.dynamics: method contract, backend detection, trajectories.

No numerical method is tested here (none exists yet in the package); these tests pin the contract the methods
will follow: validation of inputs, missing values and censoring, interpolation that never extrapolates, adapters
that keep units and provenance, and run records that reproduce.
"""
from __future__ import annotations

import ast
import json
import pathlib
import types

import numpy as np
import pytest

from pinneapple_analysis import dynamics as dyn
from pinneapple_analysis.dynamics import (
    Assumption,
    Diagnostic,
    DynamicsResult,
    MethodCard,
    Trajectory,
    TrajectoryEnsemble,
)

PKG = pathlib.Path(dyn.__file__).parent


# --------------------------------------------------------------------------- contract
def _card(**kw):
    base = dict(name="demo", role="primitive", inputs={"x": "array"},
                assumptions=[Assumption("iid", "samples are independent")], config={"n": "int"},
                outputs={"y": "float"}, failure_modes=["n < 2"])
    base.update(kw)
    return MethodCard(**base)


def test_method_card_requires_role_inputs_outputs_failures():
    assert "fails or does not apply when" in str(_card())
    json.dumps(_card().to_dict())
    with pytest.raises(ValueError, match="role"):
        _card(role="model")
    with pytest.raises(ValueError, match="inputs"):
        _card(inputs={})
    with pytest.raises(ValueError, match="failure"):
        _card(failure_modes=[])


def test_result_ok_reflects_diagnostics_and_assumptions():
    r = DynamicsResult("demo", 1.0, diagnostics=[Diagnostic("cond", 10.0, "ok")],
                       assumptions=[Assumption("iid", "independent", checked=True, evidence="Ljung-Box p=0.4"),
                                    Assumption("stationary", "stationary")])
    assert r.ok and [a.name for a in r.unchecked_assumptions] == ["stationary"]
    r2 = DynamicsResult("demo", 1.0, diagnostics=[Diagnostic("cond", 1e14, "fail", "singular")])
    assert not r2.ok
    r3 = DynamicsResult("demo", 1.0, assumptions=[Assumption("iid", "independent", checked=False)])
    assert not r3.ok and r3.violated_assumptions[0].name == "iid"
    r4 = DynamicsResult("demo", None, applicable=False, failures=["fewer than 2 samples"])
    assert not r4.ok and "NOT APPLICABLE" in r4.summary()
    with pytest.raises(ValueError, match="applicable=False"):
        DynamicsResult("demo", None, failures=["x"])
    with pytest.raises(ValueError, match="status"):
        Diagnostic("d", 1, "bad")
    with pytest.raises(KeyError):
        r.diagnostic("missing")


def test_result_to_dict_is_json_with_nonfinite_as_null():
    r = DynamicsResult("demo", {"tau": np.float64(2.5), "curve": np.arange(3.0), "z": 1 + 2j},
                       uncertainty={"ci95": (np.nan, np.inf)}, diagnostics=[Diagnostic("n", np.int64(4))])
    d = json.loads(r.to_json())
    assert d["estimate"] == {"tau": 2.5, "curve": [0.0, 1.0, 2.0], "z": {"re": 1.0, "im": 2.0}}
    assert d["uncertainty"]["ci95"] == [None, None]
    assert d["diagnostics"][0]["value"] == 4


def test_to_study_result_maps_checks_and_limits():
    pytest.importorskip("pydantic")
    rep = dyn.reproducibility("demo", {"n": 3}, [np.arange(3.0)], seed=7)
    r = DynamicsResult("demo", {"tau": 2.0}, uncertainty={"std": 0.1},
                       diagnostics=[Diagnostic("residual", 1e-9, "ok", threshold=1e-6),
                                    Diagnostic("bias", 0.3, "warn"), Diagnostic("n", 3, "info")],
                       assumptions=[Assumption("markov", "Markov process"),
                                    Assumption("gaussian", "Gaussian noise", checked=True, evidence="KS p=0.5")],
                       limits=["one horizon only"], reproducibility=rep)
    s = r.to_study_result(study_type="uncertainty")
    status = {c.name: c.status for c in s.checks}
    assert status == {"residual": "PASS", "bias": "WARN", "n": "N/A", "assumption: gaussian": "PASS"}
    assert s.passed and s.metrics["estimate.tau"] == 2.0 and s.metrics["uncertainty.std"] == 0.1
    assert any("markov" in lim for lim in s.limits) and "one horizon only" in s.limits
    failed = DynamicsResult("demo", None, applicable=False, failures=["too short"]).to_study_result()
    assert failed.status == "failed" and failed.limits[0] == "not applicable: too short"


def test_reproducibility_hash_and_seed():
    a = np.linspace(0, 1, 5)
    r1 = dyn.reproducibility("m", {"k": 1}, [a], seed=3)
    r2 = dyn.reproducibility("m", {"k": 1}, [a.copy()], seed=3)
    assert r1.input_hash == r2.input_hash and r1.versions["numpy"] == np.__version__
    assert dyn.reproducibility("m", {"k": 2}, [a]).input_hash != r1.input_hash
    assert dyn.fingerprint(a) != dyn.fingerprint(a.astype(np.float32))
    g1, s1 = dyn.make_rng(11)
    g2, _ = dyn.make_rng(11)
    assert s1 == 11 and np.array_equal(g1.normal(size=4), g2.normal(size=4))
    g3, s3 = dyn.make_rng(None)                     # an unseeded run still records its seed
    g4, _ = dyn.make_rng(s3)
    assert np.array_equal(g3.normal(size=4), g4.normal(size=4))


# --------------------------------------------------------------------------- backends
def test_capabilities_detect_without_importing(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "jax", raising=False)
    caps = dyn.capabilities()
    assert caps.numpy == np.__version__ and caps.has("scipy")
    assert "jax" not in sys.modules                 # detection did not import it
    assert set(caps.autodiff) <= {"torch", "jax"}
    assert set(caps.to_dict()) == {"numpy", "scipy", "sympy", "torch", "jax"}
    with pytest.raises(ValueError):
        dyn.require("tensorflow")


def test_require_missing_backend_names_the_install(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "jax" else real(name, *a))
    with pytest.raises(dyn.BackendUnavailable, match=r'pip install "jax\[cpu\]"'):
        dyn.require("jax", "autodiff Fisher information")


def test_to_numpy_converts_torch_tensors():
    torch = pytest.importorskip("torch")
    x = torch.arange(3.0, requires_grad=True) * 2
    assert np.array_equal(dyn.to_numpy(x), [0.0, 2.0, 4.0])
    assert dyn.to_numpy([1, 2], dtype=None).dtype.kind == "i"


def test_dynamics_modules_do_not_import_torch_or_jax_at_module_level():
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:                       # module level only; imports inside functions are lazy
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(n.split(".")[0] in ("torch", "jax") for n in names), f"{path.name} imports {names}"


# --------------------------------------------------------------------------- trajectories
def test_trajectory_validates_its_inputs():
    with pytest.raises(ValueError, match="strictly increasing"):
        Trajectory([0.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        Trajectory([0.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match="shape"):
        Trajectory([0.0, 1.0], np.zeros((3, 1)))
    with pytest.raises(ValueError, match="inf"):
        Trajectory([0.0, 1.0], [1.0, np.inf])
    with pytest.raises(ValueError, match="names"):
        Trajectory([0.0, 1.0], np.zeros((2, 2)), names=("a",))
    with pytest.raises(ValueError, match="observed_until"):
        Trajectory([0.0, 1.0], [1.0, 2.0], observed_until=0.5)
    with pytest.raises(ValueError, match="source kind"):
        Trajectory([0.0, 1.0], [1.0, 2.0], source={"kind": "guess"})
    with pytest.raises(ValueError, match="unknown components"):
        Trajectory([0.0, 1.0], [1.0, 2.0], names=("T",), units={"p": "Pa"})
    tr = Trajectory([0.0, 1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        tr.x[0, 0] = 5.0                             # immutable


def test_trajectory_sampling_missing_and_censoring():
    t = np.array([0.0, 0.1, 0.3, 0.35, 1.0])
    x = np.column_stack([np.sin(t), [1.0, np.nan, 3.0, 4.0, 5.0]])
    tr = dyn.from_array(t, x, ("s", "p"), observed_until=2.0, units={"t": "s", "p": "Pa"})
    assert not tr.is_regular() and tr.dt is None
    assert tr.sampling()["irregularity"] == pytest.approx(0.65 / 0.05)
    assert tr.has_missing and tr.missing.sum() == 1 and tr.censored and tr.time_unit == "s"
    assert tr.dropna().n == 4 and tr.dropna(how="all").n == 5
    w = tr.window(0.1, 0.5)
    assert w.n == 3 and w.observed_until == 0.5
    assert tr.select(["p"]).units == {"t": "s", "p": "Pa"}
    reg = dyn.from_array(np.arange(5) * 0.2, np.arange(5.0))
    assert reg.is_regular() and reg.dt == pytest.approx(0.2) and not reg.censored
    with pytest.raises(KeyError):
        tr.component("q")
    with pytest.raises(ValueError, match="no samples"):
        tr.window(5.0, 6.0)


def test_resample_is_exact_on_linear_data_and_never_extrapolates():
    t = np.array([0.0, 0.4, 1.0, 1.5, 3.0])
    x = np.column_stack([2.0 * t + 1.0, np.where(t > 1.2, np.nan, t)])
    tr = dyn.from_array(t, x, ("a", "b"), kind="experimental")
    grid = np.linspace(-0.5, 3.5, 17)
    r = tr.resample(grid)
    inside = (grid >= 0) & (grid <= 3)
    assert np.allclose(r.component("a")[inside], 2.0 * grid[inside] + 1.0, atol=1e-14)
    assert np.all(np.isnan(r.component("a")[~inside]))
    b_inside = (grid >= 0) & (grid <= 1.0)          # component b has data only up to t = 1.0
    assert np.all(np.isnan(r.component("b")[~b_inside]))
    assert r.source["resampled"]["method"] == "linear" and r.kind == "experimental"
    held = tr.resample([0.2, 0.9], method="previous")
    assert np.allclose(held.component("a"), [1.0, 1.8])


def test_ensemble_common_grid_and_mixed_grids():
    t = np.linspace(0, 1, 11)
    paths = np.random.default_rng(0).normal(size=(4, 11))
    ens = TrajectoryEnsemble.from_paths(t, paths, ("X",), seed=0)
    assert len(ens) == 4 and ens.on_common_grid() and ens.stack().shape == (4, 11, 1)
    assert ens[0].kind == "stochastic" and ens[0].source["seed"] == 0 and len(ens[1:3]) == 2
    mixed = TrajectoryEnsemble([ens[0], dyn.from_array(t[::2], paths[1, ::2], ("X",))], labels=[{"k": 1}, {"k": 2}])
    assert not mixed.on_common_grid() and np.allclose(mixed.horizons, 1.0)
    with pytest.raises(ValueError, match="different time grids"):
        mixed.stack()
    with pytest.raises(ValueError, match="components"):
        TrajectoryEnsemble([ens[0], dyn.from_array(t, paths[0], ("Y",))])
    with pytest.raises(ValueError, match="labels"):
        TrajectoryEnsemble([ens[0]], labels=[{}, {}])
    assert dyn.fingerprint(ens) == dyn.fingerprint(TrajectoryEnsemble.from_paths(t, paths, ("X",), seed=0))


def test_from_table_and_csv_read_units_and_missing_values(tmp_path):
    p = tmp_path / "run.csv"
    p.write_text("time [s],T_out [°C],p (bar),flag\n"
                 "0.0,20.0,1.0,1\n0.5,NA,1.1,0\n,22.0,1.2,1\n1.5,23.0,,1\n")
    tr = dyn.from_csv(str(p), "time", ["T_out", "p"])
    assert tr.names == ("T_out", "p") and tr.units == {"T_out": "°C", "p": "bar", "t": "s"}
    assert tr.n == 3 and tr.source["dropped_rows"] == 1 and tr.missing.sum() == 2 and tr.kind == "experimental"
    assert tr.source["path"] == str(p)
    tab = dyn.from_table({"t": [1.0, 0.0], "y": [3.0, 2.0]}, "t", sort=True)
    assert np.array_equal(tab.t, [0.0, 1.0]) and np.array_equal(tab.component("y"), [2.0, 3.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        dyn.from_table({"t": [1.0, 0.0], "y": [3.0, 2.0]}, "t")
    with pytest.raises(KeyError):
        dyn.from_table({"t": [0.0]}, "time")
    bad = tmp_path / "bad.csv"
    bad.write_text("t,y\n0,1\n1\n")
    with pytest.raises(ValueError, match="row 3"):
        dyn.from_csv(str(bad), "t")


def test_from_table_accepts_a_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"t [h]": [0.0, 1.0, 2.0], "c [mol/m3]": [1.0, 0.5, np.nan]})
    tr = dyn.from_table(df, "t")
    assert tr.names == ("c",) and tr.units == {"c": "mol/m3", "t": "h"} and tr.has_missing


def _fake_solution(method="pinn", kind="neural"):
    problem = types.SimpleNamespace(coords=("x", "t"), fields=("u", "v"), name="heat1d")

    def predict(pts):
        x, t = pts[:, 0], pts[:, 1]
        return np.column_stack([np.exp(-t) * np.sin(np.pi * x), x + t])

    return types.SimpleNamespace(problem=problem, method=method, kind=kind, predict=predict)


def test_from_solution_samples_a_point_over_time():
    t = np.linspace(0, 1, 6)
    tr = dyn.from_solution(_fake_solution(), t, at={"x": 0.5}, fields=["u"])
    assert tr.kind == "pinn" and tr.names == ("u",) and tr.source["problem"] == "heat1d"
    assert np.allclose(tr.component("u"), np.exp(-t))
    assert dyn.from_solution(_fake_solution("fdm", "classical"), t, at={"x": 0.0}).kind == "simulation"
    assert dyn.from_solution(_fake_solution("analytic", "analytic"), t, at={"x": 0.0}).kind == "analytic"
    with pytest.raises(ValueError, match="missing"):
        dyn.from_solution(_fake_solution(), t)
    with pytest.raises(ValueError, match="time coordinate"):
        dyn.from_solution(_fake_solution(), t, at={"x": 0.1}, time_coord="tau")


def test_from_model_and_torch_network():
    tr = dyn.from_model(lambda t: np.column_stack([np.cos(t), np.sin(t)]), np.linspace(0, 1, 5), ("c", "s"),
                        kind="analytic")
    assert tr.dim == 2 and np.allclose(tr.component("c"), np.cos(np.linspace(0, 1, 5)))
    with pytest.raises(ValueError, match="first axis"):
        dyn.from_model(lambda t: np.vstack([t, t, t]), np.linspace(0, 1, 5))
    torch = pytest.importorskip("torch")
    net = torch.nn.Linear(1, 2).double()
    trn = dyn.from_model(lambda t: net(torch.as_tensor(t)), np.linspace(0, 1, 4), column=True, kind="pinn")
    with torch.no_grad():
        ref = net(torch.linspace(0, 1, 4, dtype=torch.float64)[:, None]).numpy()
    assert trn.kind == "pinn" and np.allclose(trn.x, ref)


def test_from_snapshots_roundtrips_the_grid():
    t = np.linspace(0, 1, 4)
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 2, 5)
    U = np.exp(-t)[:, None, None] * np.sin(x)[None, :, None] * np.cos(y)[None, None, :]
    tr = dyn.from_snapshots(t, U, kind="fno", coords={"x": x, "y": y}, unit="K")
    assert tr.dim == 15 and tr.names[0] == "u[0]" and tr.units["u[14]"] == "K"
    assert np.array_equal(tr.snapshots(), U) and tr.source["spatial_shape"] == (3, 5)
    with pytest.raises(ValueError, match="snapshots"):
        dyn.from_array(t, t).snapshots()
    with pytest.raises(ValueError, match="n_t"):
        dyn.from_snapshots(t, U[:3])


def test_from_solution_on_a_real_pp_solve_result():
    Solution = pytest.importorskip("pinneapple_physics.solving").Solution

    problem = types.SimpleNamespace(coords=("x", "t"), fields=("u",), name="manual")
    sol = Solution(problem=problem, method="analytic", predict_fn=lambda p: np.exp(-p[:, 1]), wall_time_s=0.0,
                   kind="analytic")
    tr = dyn.from_solution(sol, [0.0, 1.0], at={"x": 0.3})
    assert tr.kind == "analytic" and np.allclose(tr.component("u"), [1.0, np.exp(-1.0)])
