"""ONNX models through the trained-model workflows (#1225), end to end with a real ``.onnx`` file: small torch
models exported by ``torch.onnx.export``, opened with ``open_model("onnx")``, their predictions compared with torch,
then gated, used inverse, graded by VeriPhysics and benchmarked, plus the signature edge cases (batch dimension,
float64 data into a float32 model and the reverse, names that do not match the declared card).

Locally the module is skipped when onnxruntime is missing; CI sets ``PINNEAPPLE_REQUIRE_ONNXRUNTIME=1`` so that a
missing runtime is a failure there, not a silent skip."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if os.environ.get("PINNEAPPLE_REQUIRE_ONNXRUNTIME") == "1":
    import onnxruntime  # noqa: F401  - required in CI: an ImportError fails the run
else:
    pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from pinneapple_analysis.surrogate.model_source import (  # noqa: E402
    gate,
    onnx_function,
    open_model,
    use_model,
)

ALPHA = 0.1


class Heat(torch.nn.Module):
    """u(x, t[, alpha]) = exp(-alpha pi^2 t) sin(pi x): the exact solution of u_t = alpha u_xx on [0, 1]."""

    def __init__(self, alpha: float | None = ALPHA):
        super().__init__()
        self.alpha = alpha

    def forward(self, X):
        a = X[:, 2:3] if self.alpha is None else self.alpha
        return torch.exp(-a * math.pi ** 2 * X[:, 1:2]) * torch.sin(math.pi * X[:, 0:1])


def heat_np(x, t, alpha=ALPHA):
    return np.exp(-alpha * np.pi ** 2 * t) * np.sin(np.pi * x)


def _mlp(d_in=3, d_out=2, dtype=torch.float32):
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(d_in, 16), torch.nn.Tanh(), torch.nn.Linear(16, d_out)).to(dtype)


def _export(module, path, d_in, *, dtype=torch.float32, dynamic=True, input_names=("X",), output_names=("Y",),
            example=None):
    example = example if example is not None else (torch.rand(4, d_in, dtype=dtype),)
    kw = {"dynamic_axes": {n: {0: "batch"} for n in (*input_names, *output_names)}} if dynamic else {}
    torch.onnx.export(module.eval(), example, str(path), input_names=list(input_names),
                      output_names=list(output_names), dynamo=False, **kw)
    return str(path)


BOX = {"x": (0.0, 1.0), "t": (0.0, 1.0), "alpha": (0.02, 0.5)}


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    return {"mlp": _export(_mlp(), d / "mlp.onnx", 3),
            "heat3": _export(Heat(None), d / "heat3.onnx", 3),
            "heat2": _export(Heat(), d / "heat2.onnx", 2),
            "fixed": _export(_mlp(), d / "fixed.onnx", 3, dynamic=False, example=(torch.rand(1, 3),)),
            "f64": _export(_mlp(dtype=torch.float64), d / "f64.onnx", 3, dtype=torch.float64)}


def test_onnx_predictions_match_torch(files):
    """The ONNX file and the torch module agree to float32 rounding on 1000 random points."""
    m = _mlp()
    p = open_model("onnx", path=files["mlp"], inputs=["x", "t", "alpha"], outputs=["u", "v"], bounds=BOX)
    X = np.random.default_rng(0).random((1000, 3))
    got = p.predict({"x": X[:, 0], "t": X[:, 1], "alpha": X[:, 2]})
    with torch.no_grad():
        ref = m(torch.tensor(X, dtype=torch.float32)).numpy()
    diff = max(float(np.max(np.abs(got[o] - ref[:, j]))) for j, o in enumerate(("u", "v")))
    print(f"ONNX vs torch, max abs difference over 1000 points: {diff:.3g} (outputs of size "
          f"{float(np.max(np.abs(ref))):.3g})")
    assert got["u"].dtype == np.float64 and got["u"].shape == (1000,)
    assert diff <= 1e-5 * max(1.0, float(np.max(np.abs(ref))))


def test_batch_dimension_fixed_or_dynamic(files):
    """A file exported with a fixed batch of 1 gives the same predictions as one with a dynamic batch."""
    X = np.random.default_rng(1).random((37, 3))
    dyn = onnx_function(files["mlp"], ["x", "t", "a"], ["u", "v"])(X)
    fixed = onnx_function(files["fixed"], ["x", "t", "a"], ["u", "v"])(X)
    assert dyn.shape == fixed.shape == (37, 2)
    assert np.max(np.abs(dyn - fixed)) <= 1e-6
    one = onnx_function(files["mlp"], ["x", "t", "a"], ["u", "v"])(X[0])          # one point, no batch axis
    assert one.shape == (1, 2) and np.allclose(one, dyn[:1])


def test_float64_data_into_float32_model_and_the_reverse(files):
    X = np.random.default_rng(2).random((50, 3))
    f32 = onnx_function(files["mlp"], ["x", "t", "a"], ["u", "v"])
    assert f32(X).dtype == np.float64 and np.allclose(f32(X), f32(X.astype(np.float32)))
    m64 = _mlp(dtype=torch.float64)
    with torch.no_grad():
        ref = m64(torch.tensor(X)).numpy()
    got = onnx_function(files["f64"], ["x", "t", "a"], ["u", "v"])(X.astype(np.float32).astype(np.float64))
    assert np.max(np.abs(got - ref)) < 1e-6                     # the float64 model runs in float64


def test_a_card_that_does_not_match_the_file_is_refused_with_the_names(files, tmp_path):
    with pytest.raises(ValueError, match=r"'X' with 3 columns.*\['x', 't'\]"):
        open_model("onnx", path=files["mlp"], inputs=["x", "t"], outputs=["u", "v"], bounds=BOX)
    with pytest.raises(ValueError, match=r"'Y' with 2 columns.*\['u'\]"):
        open_model("onnx", path=files["mlp"], inputs=["x", "t", "alpha"], outputs=["u"], bounds=BOX)

    class Two(torch.nn.Module):                                 # one tensor per input and per output, named
        def forward(self, x, t):
            return x * t, x + t
    path = tmp_path / "named.onnx"
    torch.onnx.export(Two(), (torch.rand(3, 1), torch.rand(3, 1)), str(path), input_names=["x", "t"],
                      output_names=["prod", "sum"], dynamo=False,
                      dynamic_axes={n: {0: "b"} for n in ("x", "t", "prod", "sum")})
    p = open_model("onnx", path=str(path), inputs=["t", "x"], outputs=["sum", "prod"],
                   bounds={"x": (0, 1), "t": (0, 1)})
    out = p.predict({"x": np.array([0.5, 0.2]), "t": np.array([0.1, 0.3])})
    assert np.allclose(out["sum"], [0.6, 0.5]) and np.allclose(out["prod"], [0.05, 0.06])
    with pytest.raises(ValueError, match=r"\['x', 't'\].*\['x', 'time'\]"):
        open_model("onnx", path=str(path), inputs=["x", "time"], outputs=["sum", "prod"], bounds={"x": (0, 1)})
    with pytest.raises(ValueError, match=r"\['prod', 'sum'\].*\['sum', 'difference'\]"):
        open_model("onnx", path=str(path), inputs=["x", "t"], outputs=["sum", "difference"], bounds={"x": (0, 1)})


def _heat3(files):
    return open_model("onnx", path=files["heat3"], inputs=["x", "t", "alpha"], outputs=["u"], bounds=BOX,
                      claimed_rel_error=1e-5)


def test_gate_and_inverse_through_an_onnx_model(files):
    p = _heat3(files)

    def reference(x, t, alpha):
        return {"u": float(heat_np(x, t, alpha))}
    q = {"x": [0.2, 0.5, 0.7, 0.4], "t": [0.1, 0.3, 0.5, 0.2], "alpha": [0.1, 0.2, 0.05, 0.3]}
    g = gate(p, q, reference=reference, n_reference=4)
    st = {c["code"]: c["status"] for c in g["checks"]}
    assert g["ok"] and st["gate.reference"] == "pass", g
    assert not gate(p, dict(q, alpha=[0.9] * 4))["ok"]                        # outside the declared box
    r = use_model(p, mode="inverse", observed={"u": float(heat_np(0.5, 0.4, 0.13))}, unknown=["alpha"],
                  known={"x": 0.5, "t": 0.4}, reference=reference)
    assert r["used"], r["checks"]
    assert r["estimate"]["alpha"] == pytest.approx(0.13, rel=1e-3)
    assert r["reverified"]["u"]["rel_diff"] < 1e-4


def test_veriphysics_grades_an_onnx_model(files):
    from pinneapple_physics.symbolic_pde.text import parse_system
    from pinneapple_veriphysics.model_grade import as_callable, grade_model
    sys_ = parse_system(["u_t = alpha*lap(u)"], [{"kind": "initial", "on": "t_min", "equation": "u = sin(pi*x)"},
                                                {"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}],
                        coords=["x", "t"], fields=["u"], coefficients={"alpha": ALPHA})
    box = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
    p = open_model("onnx", path=files["heat2"], inputs=["x", "t"], outputs=["u"], bounds=box)
    XR = np.random.default_rng(1).random((300, 2))
    ref = (XR, heat_np(XR[:, 0], XR[:, 1])[:, None])
    r = grade_model(as_callable(p, ["x", "t"]), sys_, box, reference=ref, validity=box)
    st = {c["code"]: c["status"] for c in r["checks"]}
    assert r["decision"] == "APPROVED" and st["grade.pde_residual"] == "pass", r["checks"]
    wrong = parse_system(["u_t = alpha*lap(u)"], [], coords=["x", "t"], fields=["u"], coefficients={"alpha": 0.2})
    bad = grade_model(as_callable(p, ["x", "t"]), wrong, box)
    assert {c["code"]: c["status"] for c in bad["checks"]}["grade.pde_residual"] == "fail"


def test_benchmark_ranks_an_onnx_predictor(files):
    """The exact solution through ONNX against a quadratic fit of the same tool: the ONNX predictor wins."""
    from pinneapple_analysis.surrogate.benchmark_workflow import benchmark_models
    p = _heat3(files)
    r = benchmark_models(families=["quadratic"], predictors=[p], tool=lambda x, t, alpha: {"u": heat_np(x, t, alpha)},
                         bounds=BOX, output="u", n_train=40, n_test=60, seeds=(0,), n_boot=200)
    assert r["winner"] == f"onnx:{p.name}", r["verdict"]
    onnx_row = next(x for x in r["summary"] if x["candidate"] == r["winner"])
    assert onnx_row["rel_rmse_mean"] < 1e-5


def _run(tmp_path, node):
    import pinneapple_flow as pf
    r = pf.run(pf.Flow("onnx", nodes=[node], edges=[]), workdir=tmp_path, cache=False)
    assert r.status == "done", {k: v["error"] for k, v in r.nodes.items() if v["error"]}
    return {k: v["value"] if isinstance(v, dict) and "value" in v else v
            for k, v in r.nodes[node["id"]]["outputs"]["metrics"].items()}


def test_the_flow_nodes_take_an_onnx_file(files, tmp_path):
    card = {"inputs": ["x", "t", "alpha"], "outputs": ["u"], "bounds": BOX, "claimed_rel_error": 1e-5}
    m = _run(tmp_path / "use", {"id": "use", "type": "model.use", "params": {
        "source": "onnx", "path": files["heat3"], "card": card, "mode": "inverse", "known": {"x": 0.5, "t": 0.4},
        "unknown": ["alpha"], "observed": {"u": float(heat_np(0.5, 0.4, 0.13))}, "reference": ""}})
    assert m["verdict"] == "pass" and m["estimate.alpha"] == pytest.approx(0.13, rel=1e-3)
    m = _run(tmp_path / "grade", {"id": "grade", "type": "veriphysics.grade", "params": {
        "model": "onnx", "path": files["heat2"], "validity": {"x": [0, 1], "t": [0, 1]}}})
    assert m["decision"] == "APPROVED", m
