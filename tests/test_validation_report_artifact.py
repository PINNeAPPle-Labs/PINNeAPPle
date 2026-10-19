"""The validation report as an artifact (#706): schema, provenance, diff between versions, the zoo release gate, the
Hub card section, the badge and signatures."""
from __future__ import annotations

import copy
import json
import warnings

import numpy as np
import pytest

from pinneapple_analysis.validation.report_artifact import (
    badge,
    check_report,
    diff,
    hash_data,
    hash_model,
    model_card_section,
    shields_endpoint,
    sign_report,
    verify_report,
)
from pinneapple_analysis.validation.validate import validate

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def heatsink():
    import pinneapple as pp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pp.zoo.load("heatsink_forced_mlp")


@pytest.fixture(scope="module")
def reports(heatsink):
    from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob
    a = ob.threshold_coefficient(ob.periodic_grf(60, (16, 16), alpha=2.0, tau=3.0, seed=1))
    u = np.stack([ob.darcy_2d(x, 1.0) for x in a])
    solver = lambda x: np.stack([ob.darcy_2d(y, 1.0) for y in x])        # noqa: E731  (the exact operator)
    out = {"zoo": validate(heatsink, level="standard"),
           "darcy": validate(solver, "darcy_2d", data={"train": {"a": a[:40], "u": u[:40]},
                                                        "test": {"a": a[40:], "u": u[40:]}}, level="thorough"),
           "tabular": validate(lambda x: {"y": 2 * np.asarray(x["x"])}, {"valid_range": {"x": (0, 1)}},
                               query={"x": [0.5, 2.0]}, level="quick")}
    return out


def test_schema_validates_every_report_the_library_produces(reports):
    for name, rep in reports.items():
        assert check_report(rep) == [], name
        assert check_report(json.loads(rep.to_json())) == [], name
    for name in ("airfoil_cst_rans_mlp", "heatsink_forced_mlp", "heatsink_natural_mlp", "rover_contact_grc1_viper"):
        from pinneapple_hub.zoo.validation_reports import stored
        assert check_report(stored(name)) == [], name


def test_schema_rejects_broken_reports(reports):
    d = reports["zoo"].to_dict()
    bad = copy.deepcopy(d)
    bad["checks"][0]["status"] = "ok"
    bad["checks"].append({**d["checks"][0], "id": "made.up.check"})
    bad["trust_decision"] = "REJECT" if d["decision"] != "fail" else "APPROVED"
    del bad["provenance"]["model_hash"]
    errs = check_report(bad)
    assert any("status" in e or "'ok'" in e for e in errs)
    assert any("made.up.check" in e for e in errs)
    assert any("trust_decision" in e for e in errs)
    assert any("model_hash" in e for e in errs)


def test_provenance_hashes(heatsink, reports):
    p = reports["zoo"].provenance
    assert p["model_hash"] == hash_model(heatsink) and len(p["model_hash"]) == 64
    assert p["physics"]["tolerances"]["reference"] == 0.1
    assert reports["darcy"].provenance["model_hash"] is None                # a plain callable has no weights
    x = {"train": {"a": np.ones((2, 3))}}
    assert hash_data(x) == hash_data(copy.deepcopy(x)) != hash_data({"train": {"a": np.zeros((2, 3))}})
    m = torch.nn.Linear(2, 2)
    h = hash_model(m)
    with torch.no_grad():
        m.weight[0, 0] += 1e-6
    assert hash_model(m) != h


def test_diff_flags_a_lost_check_and_moved_metrics(reports):
    old = reports["zoo"].to_dict()
    new = copy.deepcopy(old)
    i = next(k for k, c in enumerate(new["checks"]) if c["id"] == "applicability.range" or c["status"] == "pass")
    new["checks"][i]["status"] = "fail"
    j = next(k for k, c in enumerate(new["checks"]) if isinstance(c["value"], float) and c["value"] and k != i)
    new["checks"][j]["value"] = 2 * new["checks"][j]["value"]
    new["provenance"]["validity_range"] = {**old["provenance"]["validity_range"], "power_w": [10.0, 300.0]}
    d = diff(old, new)
    assert d.regressed and [c["id"] for c in d.lost] == [new["checks"][i]["id"]]
    assert d.moved and d.moved[0]["rel"] == pytest.approx(1.0)
    assert any("power_w" in e["what"] for e in d.envelope)
    assert "**regressed**" in d.to_markdown()
    same = diff(old, copy.deepcopy(old))
    assert not same.regressed and not same.lost and not same.moved
    back = diff(new, old)
    assert not back.regressed and back.gained


def test_release_gate_blocks_a_model_that_loses_a_check(heatsink):
    from pinneapple_hub.zoo import validation_reports as vr
    from pinneapple_hub.zoo._models import copy_model
    d, _ = vr.check_release("heatsink_forced_mlp")
    assert not d.regressed                                              # the stored report is current
    new = copy_model(heatsink)                                          # "v1.1": a sign error in the resistance
    forward = new._forward
    new._forward = lambda X: {**forward(X), "r_base_k_w": -np.asarray(forward(X)["r_base_k_w"])}
    d, _ = vr.check_release("heatsink_forced_mlp", new)
    assert d.regressed and d.lost and d.decision[1] == "fail"
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "zoo" / "validation_reports.py"
    spec = importlib.util.spec_from_file_location("zoo_validation_reports_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["check", "heatsink_forced_mlp"]) == 0             # the command the publishing workflow runs


def test_every_zoo_hub_card_carries_the_validation_section():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "zoo"))
    import publish_to_hub
    items = [it for it in publish_to_hub.plan(None) if it["kind"] == "models"]
    assert items
    for it in items:
        assert "## Physics validation" in it["readme"], it["name"]


def test_card_section_badge_and_signature(reports):
    rep = reports["zoo"]
    md = model_card_section(rep)
    assert md.startswith("## Physics validation") and "| applicability |" in md and rep.verdict in md
    svg = badge(rep)
    ok = sum(c.status == "pass" for c in rep.checks)
    n = sum(c.status != "not-applicable" for c in rep.checks)
    assert f"{ok}/{n} checks" in svg and svg.startswith("<svg")
    assert shields_endpoint(rep) == {"schemaVersion": 1, "label": "physics-validated", "message": f"{ok}/{n} checks",
                                     "color": "brightgreen"}
    from pinneapple_security.signing import VerificationError, hmac_key
    key = hmac_key(b"a test secret of 32 bytes ......")
    env = sign_report(rep, key)
    assert verify_report(env, key)["verdict"] == rep.verdict
    env["payload"] = env["payload"][:-4] + "AAAA"
    with pytest.raises((VerificationError, ValueError)):
        verify_report(env, key)
    from pinneapple_security.signing import sign_json
    other = sign_json({"verdict": "pass"}, key)                         # same key, another document type
    with pytest.raises(ValueError, match="not a validation report"):
        verify_report(other, key)
    import pinneapple as pp
    assert pp.validate.diff is diff                                     # pp.validate.diff(old, new)
