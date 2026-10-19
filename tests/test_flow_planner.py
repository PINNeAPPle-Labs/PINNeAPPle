"""Planner: heuristic detection (EN / PT), every proposal validates, the LLM path validates / repairs / falls back,
and the dispatch reaches OpenAI-compatible endpoints."""
import json

import pytest

import pinneapple_flow as pf
from pinneapple_flow.planner import detect, plan, repair


@pytest.mark.parametrize("text,family,dim,intent", [
    ("Transient heat conduction in a 2D plate with edges held at 0 degrees", "heat", 2, None),
    ("Steady temperature in a square plate with heat generation", "poisson", 2, None),
    ("Vibrating string fixed at both ends", "wave", 1, None),
    ("Viscous Burgers equation with shocks, nu=0.003", "burgers", None, None),
    ("Pollutant transport by advection and diffusion in a 1D channel", "advection_diffusion", 1, None),
    ("Identify the unknown thermal diffusivity of a steel rod from temperature measurements", "heat", 1, "inverse"),
    ("Calor transiente numa barra isolada, quero estimar a difusividade a partir de medições", "heat", 1, "inverse"),
    ("Fast surrogate (neural operator) of heat in a plate for many diffusivities with hyperparameter tuning",
     "heat", 2, "hpo"),
])
def test_detection(text, family, dim, intent):
    d = detect(text)
    assert d["family"] == family
    if dim:
        assert d["dimension"] == dim
    if intent:
        assert intent in d["intents"]


def test_quantities_and_assignments():
    d = detect("heat in a 0.5 m rod, diffusivity 1.2e-5 m^2/s, for 600 s, alpha = 2e-5")
    assert d["length"] == pytest.approx(0.5)
    assert d["duration"] == pytest.approx(600)
    assert d["coefficients"]["alpha"] == pytest.approx(2e-5)


@pytest.mark.parametrize("text", [
    "heat in a 2D plate", "steady heat in a plate with a source, then switch from 2D to 3D",
    "identify the diffusivity of a rod", "a digital twin of a water tank that detects leaks",
    "forecast El Nino", "aircraft take-off", "surrogate of a vibrating string, hybrid ROM, hyperparameter tuning",
    "something completely unrelated to physics",
])
def test_every_proposal_validates(text):
    res = plan(text, backend="heuristic")
    assert res["planner"] == "heuristic" and res["proposals"]
    for p in res["proposals"]:
        assert [i for i in pf.validate_flow(p["flow"]) if i.level == "error"] == []
        assert p["flow"]["meta"]["planner"] == "heuristic"


def test_dimension_switch_is_planned():
    res = plan("steady heat in a plate with a heat source, then switch from 2D to 3D", backend="heuristic")
    assert res["detected"]["dimension_change"] == [2, 3]
    eq = next(n for n in res["proposals"][0]["flow"]["nodes"] if n["type"] == "equations.system")
    assert eq["params"]["follow_template"] is True


def _fake_llm(payload):
    def call(prompt, **kw):
        assert "Node catalog" in prompt and kw["json_mode"] is True
        return payload if isinstance(payload, str) else json.dumps(payload)
    return call


def test_llm_valid_output_is_labelled():
    flow = {"name": "heat", "nodes": [
        {"id": "geo", "type": "geometry.domain", "params": {"dimension": 1}},
        {"id": "eq", "type": "equations.system", "params": {"template": "heat", "pdes": ["u_t = alpha*u_xx"],
                                                            "conditions": [{"kind": "initial", "on": "t_min",
                                                                            "equation": "u = sin(pi*x)"}],
                                                            "coefficients": {"alpha": 0.1}}},
        {"id": "fd", "type": "solver.classical", "params": {"method": "fd"}}],
        "edges": [{"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"},
                  {"source": "eq", "source_port": "problem", "target": "fd", "target_port": "problem"}]}
    res = plan("heat in a rod", backend="llm", provider="ollama", model="m",
               call=_fake_llm({"proposals": [{"title": "FD", "rationale": "r", "flow": flow}]}))
    assert res["planner"] == "llm:ollama/m"
    assert res["proposals"][0]["planner"] == "llm:ollama/m"
    assert res["proposals"][0]["flow"]["meta"]["planner"] == "llm:ollama/m"


def test_llm_output_is_repaired():
    flow = {"name": "x", "nodes": [
        {"id": "geo", "type": "geometry.domain", "params": {"dimension": 2, "evil": "__import__('os')"}},
        {"id": "eq", "type": "equations.system", "params": {"template": "heat", "pdes": ["os.system('rm')"]}},
        {"id": "magic", "type": "solver.quantum_oracle", "params": {}},
        {"id": "fd", "type": "solver.classical", "params": {"method": "fd"}}],
        "edges": [{"source": "geo", "source_port": "geometry", "target": "fd", "target_port": "problem"}]}
    res = plan("heat", backend="llm", provider="openai_compatible", model="local",
               call=_fake_llm({"proposals": [{"title": "t", "rationale": "", "flow": flow}]}))
    p = res["proposals"][0]
    assert p["planner"].endswith("+repair")
    types = [n["type"] for n in p["flow"]["nodes"]]
    assert "solver.quantum_oracle" not in types
    geo = next(n for n in p["flow"]["nodes"] if n["id"] == "geo")
    assert "evil" not in geo["params"]
    eq = next(n for n in p["flow"]["nodes"] if n["id"] == "eq")
    assert eq["params"].get("follow_template") is True and "pdes" not in eq["params"]
    assert [i for i in pf.validate_flow(p["flow"]) if i.level == "error"] == []


@pytest.mark.parametrize("payload", ["this is not json", {"proposals": []},
                                     {"proposals": [{"flow": {"nodes": [{"id": "a", "type": "nope"}]}}]}])
def test_llm_failure_falls_back_to_heuristic(payload):
    res = plan("transient heat in a 2D plate", backend="llm", provider="ollama", call=_fake_llm(payload))
    assert res["planner"] == "heuristic" and res["fallback_from"].startswith("llm:ollama")
    assert "LLM planner failed" in res["warnings"][0]


def test_llm_provider_error_falls_back():
    def boom(prompt, **kw):
        raise ConnectionError("no server")
    res = plan("heat in a rod", backend="llm", provider="ollama", call=boom)
    assert res["planner"] == "heuristic" and "no server" in res["warnings"][0]


def test_repair_wires_missing_required_inputs():
    doc, notes = repair({"nodes": [{"id": "geo", "type": "geometry.domain"},
                                   {"id": "eq", "type": "equations.system", "params": {"template": "heat"}}],
                         "edges": []})
    assert any("wired" in n for n in notes)
    assert [i for i in pf.validate_flow(doc) if i.level == "error"] == []


def test_dispatch_openai_compatible_needs_base_url(monkeypatch):
    from pinneapple_llm import call_llm
    with pytest.raises(ValueError, match="base_url"):
        call_llm("hi", provider="openai_compatible")
    seen = {}

    class FakeClient:
        def __init__(self, api_key=None, base_url=None):
            seen.update(api_key=api_key, base_url=base_url)
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, **kw):
            seen.update(kw)

            class R:
                choices = [type("C", (), {"message": type("M", (), {"content": "{}"})()})()]
            return R()

    import sys
    import types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))
    assert call_llm("hi", provider="openai_compatible", base_url="http://localhost:8000/v1", model="m",
                    json_mode=True, max_tokens=50) == "{}"
    assert seen["base_url"] == "http://localhost:8000/v1" and seen["max_tokens"] == 50
    assert seen["response_format"] == {"type": "json_object"}


def test_sensor_count_read_from_the_text():
    from pinneapple_flow.planner.heuristic import sensor_count, sensor_layout
    assert sensor_count("a twin from 5 temperature sensors along the rod") == 5
    assert sensor_count("three sensors") == 3 and sensor_count("com seis sensores") == 6
    assert sensor_count("monitor the rod") is None and sensor_count("sensor fusion") is None
    assert sensor_layout(3, 1) == [0.25, 0.5, 0.75] and len(sensor_layout(5, 2)) == 5
    res = pf.plan("A 1 m metal rod cools down by heat conduction. Build a digital twin from 5 temperature sensors "
                  "along the rod and detect when the diffusivity changes.", backend="heuristic")
    flow = res["proposals"][0]["flow"]
    live = next(n for n in flow["nodes"] if n["type"] == "data.stream")
    assert len(live["params"]["sensors"]) == 5
    assert any(d["decided_by"]["id"] == "twin.sensor_layout" for d in flow["meta"]["decisions"])
