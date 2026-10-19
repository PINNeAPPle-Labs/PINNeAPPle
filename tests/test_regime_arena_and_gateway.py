"""pinneapple_arena.regimes (regime arena) and pinneapple_registry.gateway (model gateway)."""
import copy

import numpy as np
import pytest

pytest.importorskip("sklearn")

from pinneapple_analysis import surrogate as S
from pinneapple_arena.regimes import NOT_IN_LIBRARY, run_regime_arena
from pinneapple_data.audit import constraint_mask
from pinneapple_registry.gateway import ModelGateway, ood_verdict


def _scalar(n=120):
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (n, 2))
    Y = np.stack([np.exp(X[:, 0]) + X[:, 1], 0.5 + X[:, 0] ** 2], 1)
    return S.SurrogateData(X, Y, ["a", "b"], ["f", "g"], groups=np.repeat(np.arange(n // 4), 4),
                           meta={"constraints": ["g > 0"]})


def test_constraint_mask():
    m = constraint_mask("Cd > 0", {"Cd": np.array([0.1, -0.1])})
    assert m.tolist() == [True, False]
    assert constraint_mask("abs(u) <= 1", {"u": np.array([[0.5, 1.2]])}).tolist() == [[True, False]]


def test_regime_arena_scalar():
    r = run_regime_arena(_scalar(), methods=["ridge", "mlp"], budget={"steps": 60}, seeds=2)
    o = r.to_dict()
    av = {g["id"]: g for g in o["regimes"]}
    assert av["interpolation"]["available"] and av["few_data"]["available"] and av["geometry_change"]["available"]
    assert not av["steep_fronts"]["available"] and "scalar" in av["steep_fronts"]["reason"]
    for reg, rows in o["rankings"].items():
        errs = [x["error_mean"] for x in rows]
        assert errs == sorted(errs) and all(x["n_seeds"] == 2 for x in rows), reg
    mlp = next(m for m in o["methods"] if m["id"] == "mlp")
    assert len(mlp["results"]["interpolation"]["errors"]) == 2 and mlp["results"]["interpolation"]["error_std"] > 0
    assert set(mlp["violations"]) == {"g > 0"} and mlp["cost"]["infer_s_per_query"] > 0
    assert o["audit"]["decision"] in ("pass", "warn", "fail") and o["audit"]["checks"]
    assert o["protocol"]["budget"]["steps"] == 60 and o["hardware"] and o["created"]
    assert set(NOT_IN_LIBRARY) <= set(o["not_run"])
    md = r.to_markdown("demo")
    assert md.startswith("# demo") and "## Rankings" in md and "steep fronts" in md and "R-OAMS" in md


def test_regime_arena_fields_with_baseline():
    d = S.examples.burgers_fields(n=24, nx=16, nt=6)
    solve = S.examples.burgers_fdm_baseline(nx=81)
    r = run_regime_arena(d, methods=["gp", "deeponet"], budget={"steps": 30}, seeds=1, regimes=["interpolation",
                         "extrapolation", "steep_fronts"], baselines={"fdm": solve})
    o = r.to_dict()
    assert set(o["rankings"]) == {"interpolation", "extrapolation", "steep_fronts"}
    bl = o["baselines"][0]
    assert bl["id"] == "fdm" and 0 < bl["error_mean"] < 0.2 and bl["s_per_query"] > 0
    assert next(m for m in o["methods"] if m["id"] == "gp")["pde_residual_ratio"] > 0


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    st = S.run_study(_scalar(), methods=["ridge"], budget={"steps": 10}, seeds=1)
    p = tmp_path_factory.mktemp("m") / "model"
    st.model.save(p)
    return st.model, p


def test_gateway_publish_predict_policies_and_metrics(tmp_path, model):
    sm, path = model
    g = ModelGateway(str(tmp_path / "gw"))
    v1 = g.publish("demo", path, ood_policy="warn")
    v2 = g.publish("demo", sm, ood_policy="refuse", stage="staging")
    assert g.versions("demo") == [v1, v2] and g.resolve("demo") == v2 and g.resolve("demo", "production") == v1
    r = g.predict("demo", {"a": [0.0, 9.0], "b": 0.0}, version=v1)
    assert r["outputs"]["f"][1] is not None and r["ood"][1]["ood"] and r["warnings"]
    r2 = g.predict("demo", [{"a": 0.0, "b": 0.0}, {"a": 9.0, "b": 0.0}])
    assert r2["model"]["version"] == v2 and r2["outputs"]["f"][1] is None
    with pytest.raises(PermissionError, match="cannot relax"):
        g.predict("demo", {"a": 0.0, "b": 0.0}, ood_policy="warn")
    assert g.predict("demo", {"a": 0.0, "b": 0.0}, version=v1, ood_policy="refuse")["ood_policy"] == "refuse"
    with pytest.raises(ValueError, match="missing inputs"):
        g.predict("demo", {"a": 0.0})
    m = g.metrics("demo")
    assert m["calls"] == 5 and m["errors"] == 2
    row = next(x for x in m["by_version"] if x["version"] == v1)
    assert row["ood_rows"] == 1 and row["latency_ms_p95"] >= row["latency_ms_p50"] > 0
    assert len(g.runs("demo")) == 5
    ref = g.describe("demo")["model_ref"]
    assert ref["inputs"] == ["a", "b"] and ref["sha256"]


def test_ood_verdict_is_what_predict_does():
    assert ood_verdict({"ood": False}, "refuse") == "in range"
    assert ood_verdict({"ood": True}, "warn") == "warn" and ood_verdict({"ood": True}, "refuse") == "refuse"
    with pytest.raises(ValueError):
        ood_verdict({"ood": True}, "ignore")


def test_gateway_reject_needs_override(tmp_path, model):
    sm = copy.deepcopy(model[0])
    sm.card["trust"] = {"decision": "REJECT", "checks": [{"name": "held-out error", "status": "contradicts"}]}
    g = ModelGateway(str(tmp_path / "gw"))
    with pytest.raises(PermissionError):
        g.publish("bad", sm)
    assert g.models() == []
    v = g.publish("bad", sm, override_reason="exploratory use only")
    assert g.metadata("bad", v)["trust_override"]["reason"] == "exploratory use only"


def test_gateway_zoo_model(tmp_path):
    g = ModelGateway(str(tmp_path / "gw"))
    g.publish("airfoil", "airfoil_cst_rans_mlp", kind="pinneapple.zoo")
    naca2412 = dict(T0=0.169591, T1=0.144959, T2=0.131433, M0=0.024922, M1=0.071088, M2=0.058960)
    r = g.predict("airfoil", {**naca2412, "alpha_deg": [4.0, 20.0]})
    assert abs(r["outputs"]["Cl"][0] - 0.6625) < 0.05            # OpenFOAM: Cl 0.6625 (zoo card)
    assert r["ood"][1]["outside"] == ["alpha_deg"]


def test_gateway_zoo_model_with_derived_validity(tmp_path):
    """The heatsink card bounds derived quantities (fin gap), not n_fins: the zoo model's own check decides."""
    import pinneapple as pp
    g = ModelGateway(str(tmp_path / "gw"))
    g.publish("hs", "heatsink_forced_mlp", kind="pinneapple.zoo", ood_policy="refuse")
    zm = pp.zoo.load("heatsink_forced_mlp")
    row = {k: float(np.mean(v)) for k, v in zm.bounds().items()}
    row.update(n_fins=12.0, source_width_mm=0.5 * row["base_width_mm"], source_depth_mm=0.5 * row["base_depth_mm"])
    ok = g.predict("hs", row)
    assert ok["ood"][0]["ood"] == (not bool(zm.in_range({k: [v] for k, v in row.items()})[0]))
    crowded = g.predict("hs", {**row, "n_fins": 400.0})            # fins far too dense for the base width
    assert crowded["ood"][0]["ood"] and crowded["outputs"][zm.output_names[0]] == [None]
    assert crowded["ood"][0]["reasons"]


def test_gateway_rest_routes(tmp_path, model):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pinneapple_registry.gateway_api import count_rows, gateway_router
    sm, _ = model
    g = ModelGateway(str(tmp_path / "gw"))
    v = g.publish("demo", sm)
    seen = []
    app = FastAPI()
    app.include_router(gateway_router(g, on_rows=lambda req, n: seen.append(n), caller=lambda req: "tester"))
    c = TestClient(app)
    assert c.get("/v1/models").json()["models"][0]["name"] == "demo"
    assert c.get(f"/v1/models/demo/versions/{v}").json()["model_ref"]["inputs"] == ["a", "b"]
    r = c.post("/v1/models/demo/predict", json={"inputs": {"a": [0.1, 0.2, 9.0], "b": 0.0}})
    assert r.status_code == 200 and r.json()["ood"][2]["ood"] and seen == [3]
    assert c.post("/v1/models/nope/predict", json={"inputs": {"a": 0}}).status_code == 404
    assert c.post("/v1/models/demo/predict", json={"inputs": {"a": 0}}).status_code == 422
    assert c.get("/v1/models/demo/runs").json()["runs"][0]["caller"] == "tester"
    assert c.get("/v1/metrics").json()["calls"] == 2
    assert c.get(f"/v1/models/demo/versions/{v}/export/onnx").status_code == 409     # ridge: no export
    assert count_rows([{"a": 1}, {"a": 2}]) == 2 and count_rows([1.0, 2.0]) == 1


@pytest.fixture()
def served(tmp_path, model):
    """A second gateway serving ``model`` over real HTTP (uvicorn in a thread)."""
    pytest.importorskip("uvicorn")
    import socket
    import threading
    import time

    import uvicorn
    from fastapi import FastAPI

    from pinneapple_registry.gateway_api import gateway_router
    src = ModelGateway(str(tmp_path / "src"))
    src.publish("demo", model[0])
    app = FastAPI()
    app.include_router(gateway_router(src))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", src
    server.should_exit = True
    th.join(5)


def test_client_remote_kind_and_publish_ref(tmp_path, served):
    from pinneapple_registry.gateway_client import GatewayClient, GatewayError
    base, src = served
    c = GatewayClient(base)
    assert [m["name"] for m in c.models()] == ["demo"]
    r = c.predict("demo", {"a": [0.0, 9.0], "b": 0.0})
    assert r["ood"][1]["ood"] and r["outputs"]["f"][0] is not None
    with pytest.raises(GatewayError) as e:
        c.predict("nope", {"a": 0})
    assert e.value.status == 404
    gw = ModelGateway(str(tmp_path / "front"))
    v = gw.publish("proxied", f"{base}/v1/models/demo", kind="remote", ood_policy="refuse")
    out = gw.predict("proxied", {"a": [0.0, 9.0], "b": 0.0})
    direct = src.predict("demo", {"a": 0.0, "b": 0.0})
    assert out["outputs"]["f"][0] == pytest.approx(direct["outputs"]["f"][0]) and out["outputs"]["f"][1] is None
    assert gw.metadata("proxied", v)["remote"]["model"] == "demo" and gw.metrics()["calls"] == 1
    ref = {"name": "from-ref", "version": "1", "uri": f"{base}/v1/models/demo", "trust_decision": "REVIEW",
           "inputs": [{"name": "a", "low": -0.5, "high": 0.5}, {"name": "b", "low": -1, "high": 1}]}
    v2 = gw.publish_ref(ref)
    m = gw.metadata("from-ref", v2)
    assert m["validity_bounds"]["a"] == [-0.5, 0.5] and m["trust"] == "REVIEW" and m["stage"] == "production"
    assert gw.predict("from-ref", {"a": 0.8, "b": 0.0})["ood"][0]["outside"] == ["a"]   # the ref's tighter range
    z = gw.publish_ref({"name": "zoo-airfoil", "uri": "zoo:airfoil_cst_rans_mlp"})
    assert gw.metadata("zoo-airfoil", z)["kind"] == "pinneapple.zoo"
    with pytest.raises(ValueError, match="publish_ref takes"):
        gw.publish_ref({"name": "x", "uri": "/tmp/model"})
