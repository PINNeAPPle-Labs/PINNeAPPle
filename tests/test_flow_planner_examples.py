"""The generic creation modes for the example domains: a short description of each converted example's problem, fed
to the heuristic planner, gives a valid flow that uses that example's domain nodes (through the public capability
hook, pinneapple_flow.planner.extensions); the guided wizard reaches the same flows; capabilities record their
decisions with references or "no reference"."""
import pytest

from pinneapple_flow.planner import extensions as X
from pinneapple_flow.planner import plan
from pinneapple_flow.planner.wizard import STEPS, describe, plan_from_answers
from pinneapple_flow.schema import as_flow, check_flow

# (example folder, description, the domain node types the first proposal must use)
CASES = [
    ("flight_dynamics", "Simulate the take-off of an A320 and compare the distance to 35 ft with Raymer",
     {"flight.aircraft", "flight.takeoff_cases"}),
    ("flight_dynamics", "Four aircraft flying through severe MIL-F-8785C turbulence at 3000 m",
     {"flight.aircraft", "flight.turbulence_encounter"}),
    ("flight_dynamics", "the light aircraft, the turboprop, the airliner and the glider through the same 1-cosine gust",
     {"flight.same_gust", "render.blender_clip"}),
    ("flight_dynamics", "trim and stability derivatives of a light aircraft",
     {"flight.trim_stability"}),
    ("enso_forecasting", "Forecast El Niño: the Niño 3.4 index with the ENSO models", {"forecast.enso"}),
    ("enso_forecasting", "check the ENSO model: Kelvin and Rossby wave speeds, the Gill response, the recharge "
                         "oscillator", {"enso.solver_checks"}),
    ("near_random_forecasting", "forecast a chaotic Lorenz series with noise; measure predictability first",
     {"forecast_lab.case"}),
    ("motor_magnetics", "cogging torque, torque against current angle and back-EMF of a 24-slot surface PM motor",
     {"em.spm_machine"}),
    ("electrodynamics", "torque-speed envelope, efficiency map and WLTC energy of the Prius traction drive",
     {"em.prius_drive"}),
    ("electrodynamics", "interior PM motor of the 2004 Prius: MTPA and losses", {"em.prius_machine"}),
    ("black_hole_weather", "simulate the accretion flow around a black hole (RIAF torus)", {"blackhole.simulate"}),
]


@pytest.mark.parametrize("example, text, nodes", CASES, ids=[f"{c[0]}:{c[1][:30]}" for c in CASES])
def test_description_gives_the_examples_domain_flow(example, text, nodes):
    res = plan(text, backend="heuristic")
    p = res["proposals"][0]
    flow = check_flow(as_flow(p["flow"]))                   # valid: types, ports, required inputs, no cycles
    types = {n.type for n in flow.nodes}
    assert nodes <= types, (p["title"], sorted(types))
    meta = p["flow"]["meta"]
    assert meta.get("example") == example and meta.get("capability")
    rule = f"capability.{meta['capability']}"
    dec = [d for d in meta["decisions"] if (d.get("decided_by") or {}).get("id") == rule]
    assert dec and all(d["references"] or d["no_reference"] for d in dec), meta["decisions"][:2]


def test_named_aircraft_are_kept():
    p = plan("take-off of the A320 against Raymer", backend="heuristic")["proposals"][0]
    ac = next(n for n in p["flow"]["nodes"] if n["type"] == "flight.aircraft")
    assert ac["params"]["aircraft"] == ["a320"]
    cmp_ = next(n for n in p["flow"]["nodes"] if n["type"] == "examples.compare")
    assert cmp_["params"]["comparisons"] and all("light_ga" not in c["flow"] for c in cmp_["params"]["comparisons"])


def test_capabilities_declare_their_metadata():
    caps = X.capabilities()
    assert {"flight.takeoff", "flight.turbulence", "enso.forecast", "em.spm_machine", "blackhole.accretion"} <= \
        {c.id for c in caps}
    from pinneapple_flow.registry import REGISTRY
    from pinneapple_flow.rules import RULES
    for c in caps:
        assert c.keywords and c.nodes and (c.references or c.uncited), c.id
        assert all(t in REGISTRY for t in c.nodes), c.id
        assert c.rule_id in RULES
    d = X.describe()
    assert all({"id", "keywords", "nodes", "inputs", "downstream", "references"} <= set(x) for x in d)


def test_register_a_capability_from_another_package():
    from pinneapple_flow.planner.build import FlowBuilder

    def build(text, det):
        fb = FlowBuilder("custom")
        fb.add("v", "flight.verification", {})
        return fb
    X.register_capability(X.Capability(id="test.custom_cap", title="custom", domain="test",
                                       keywords={"zzqx custom capability": 5}, nodes=("flight.verification",),
                                       uncited="test", build=build))
    try:
        p = plan("please run the zzqx custom capability", backend="heuristic")["proposals"][0]
        assert p["flow"]["meta"]["capability"] == "test.custom_cap"
    finally:
        X.CAPABILITIES.pop("test.custom_cap")


def test_guided_wizard_reaches_the_same_flows():
    assert [s["id"] for s in STEPS] == ["problem", "physics", "geometry", "methods", "validation", "outputs"]
    r = plan_from_answers({"problem": "", "capability": "flight.takeoff"})
    assert r["proposals"][0]["flow"]["meta"]["capability"] == "flight.takeoff"
    r = plan_from_answers({"problem": "heat in a plate", "physics": "heat", "geometry": "2d",
                           "methods": ["pinn", "classical"], "validation": ["exact"], "outputs": ["3d"]})
    text = describe({"problem": "heat in a plate", "physics": "heat", "geometry": "2d",
                     "methods": ["pinn", "classical"]})
    assert "PINN" in text and "finite differences" in text
    types = {n["type"] for p in r["proposals"] for n in p["flow"]["nodes"]}
    assert {"equations.system"} <= types
    check_flow(as_flow(r["proposals"][0]["flow"]))
