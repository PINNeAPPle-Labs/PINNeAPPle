"""U-DR-1 compiler: extraction from the three fictitious V-101 documents, conflicts, gaps, Claude quote check, filling.

Filling needs the user's copy of the fillable ASME Form U-DR-1 (not distributed): those tests run when the
environment variable UDR_FORM_PDF points to it and are skipped otherwise.
"""
import json
import os
import types

import pytest

pytest.importorskip("pdfplumber")
pytest.importorskip("pypdf")

from pinneapple_data import udr1  # noqa: E402
from pinneapple_data.udr1.catalog import FIELD_BY_KEY  # noqa: E402
from pinneapple_data.udr1.llm import extract_with_claude, verify_quote  # noqa: E402

EX = os.path.join(os.path.dirname(__file__), "..", "apps", "design_requirements", "examples")
ORDER = [("owner_specification_V-101.pdf", "client specification"), ("process_datasheet_V-101.pdf", "process datasheet"),
         ("mechanical_datasheet_V-101.pdf", "mechanical datasheet")]
FORM = os.environ.get("UDR_FORM_PDF")


@pytest.fixture(scope="module")
def docs():
    return [udr1.read_document(os.path.join(EX, n), name=n, role=r) for n, r in ORDER]


@pytest.fixture(scope="module")
def comp(docs):
    return udr1.compile_requirements(docs, [c for d in docs for c in udr1.extract_rules(d)])


def test_example_finds_the_required_items_with_their_sources(comp):
    s = comp.summary()
    assert s["required"] == 28 and s["required_filled"] == 27 and s["nozzles"] == 6
    d = comp.decisions
    expect = {"design_p_int": "15 barg", "design_t_int": "120 °C", "mdmt1_temp": "-29 °C", "diameter": "1800 mm",
              "length_tt": "5400 mm", "mat_shell": "SA-516 Gr. 70N", "joint_eff_shell": "1.0",
              "design_p_ext": "Full vacuum", "op2_tmin": "-20 °C", "wind_speed": "42 m/s"}
    for k, v in expect.items():
        assert d[k].display == v, k
    assert d["orientation"].value == "vertical" and d["support"].value == "skirt" and d["pwht"].value == "per code"
    assert d["nb_registration"].value == "yes" and d["lethal"].value == "no" and d["overpressure"].value == ["valve"]
    assert d["wind_code"].value == "asce 7" and d["mdmt1_due_to"].value == "ambient temperature"
    assert d["mawp_int_basis"].value == "calculated by manufacturer"
    src = d["design_p_int"].chosen
    assert src.doc == "process_datasheet_V-101.pdf" and src.page == 1 and "15 barg" in src.snippet


def test_planted_conflict_and_gap_are_reported_and_nothing_else(comp):
    assert [c.key for c in comp.conflicts()] == ["ca_shell_int"]
    ca = comp.decisions["ca_shell_int"]
    assert ca.display == "6 mm" and ca.chosen.doc == "owner_specification_V-101.pdf"     # higher priority wins
    assert sorted(c.display() for c in ca.candidates) == ["3 mm", "6 mm"]
    assert [g.key for g in comp.gaps()] == ["cyclic_service"]


def test_nozzle_schedule_including_the_row_continued_on_the_next_page(comp):
    rows = [n.value for n in comp.nozzles]
    assert [r["description"] for r in rows] == ["N1 Gas/liquid inlet", "N2 Gas outlet", "N3 Liquid outlet",
                                                "N4 PSV connection", "N5 Level instrument", "M1 Manway"]
    assert rows[4]["number"] == "2" and rows[5]["size"] == "24 in" and {r["class"] for r in rows} == {"300"}


def test_document_order_decides_conflicts(docs):
    flipped = [docs[2], docs[1], docs[0]]
    comp = udr1.compile_requirements(flipped, [c for d in flipped for c in udr1.extract_rules(d)])
    assert comp.decisions["ca_shell_int"].display == "3 mm" and comp.decisions["ca_shell_int"].status == "conflict"


def test_engineer_overrides_win_and_close_gaps(docs):
    comp = udr1.compile_requirements(docs, [c for d in docs for c in udr1.extract_rules(d)],
                                     overrides={"cyclic_service": "no", "ca_shell_int": "6 mm"})
    assert comp.gaps() == [] and comp.conflicts() == []
    assert comp.decisions["cyclic_service"].note == "entered by engineer"


@pytest.mark.parametrize("key,text,expected", [
    ("design_p_int", "1.5 MPa(g)", (1.5, 1.5e6)), ("design_p_int", "220 psig", (220.0, 220 * 6894.757293168361)),
    ("design_p_int", "16 bar(a)", (16.0, 16e5 - 101325.0)), ("mdmt1_temp", "-20 °F", (-20.0, (-20 - 32) * 5 / 9 + 273.15)),
    ("wind_speed", "100 mph", (100.0, 44.704)), ("diameter", "72 in", (72.0, 1.8288)),
])
def test_quantities_are_compared_in_si_with_gauge_pressure(key, text, expected):
    p = udr1.parse_value(FIELD_BY_KEY[key], text)
    assert p["value"] == pytest.approx(expected[0]) and p["si"] == pytest.approx(expected[1])


def test_yes_no_and_choices():
    assert udr1.parse_value(FIELD_BY_KEY["lethal"], "No")["value"] == "no"
    assert udr1.parse_value(FIELD_BY_KEY["nb_registration"], "Required")["value"] == "yes"
    assert udr1.parse_value(FIELD_BY_KEY["orientation"], "Horizontal drum")["value"] == "horizontal"
    assert udr1.parse_value(FIELD_BY_KEY["orientation"], "vertical or horizontal") is None     # ambiguous: no guess
    assert udr1.parse_value(FIELD_BY_KEY["pwht"], "Not required")["value"] == "none"
    assert sorted(udr1.parse_value(FIELD_BY_KEY["overpressure"], "PSV and rupture disc")["value"]) == ["rupture disk", "valve"]


def test_every_catalog_item_maps_to_a_form_answer():
    for f in udr1.FIELDS:
        assert f.pdf or f.states, f.key
        if f.kind in ("bool", "choice", "multi"):
            assert f.states, f.key


def test_plain_text_documents_are_read():
    doc = udr1.read_document(b"Design pressure: 10 barg\nDesign temperature: 200 C\nMDMT: -46 C", name="notes.txt")
    cands = udr1.extract_rules(doc)
    got = {c.key: c.display() for c in cands}
    assert got["design_p_int"] == "10 barg" and got["design_t_int"] == "200 C" and got["mdmt1_temp"] == "-46 C"


def test_a_heading_with_a_label_word_is_not_an_answer():
    doc = udr1.read_document(b"Owner Project Specification - Pressure Vessels\nOwner: ACME", name="t.txt")
    owners = [c.display() for c in udr1.extract_rules(doc) if c.key == "owner"]
    assert owners == ["ACME"]


# --------------------------------------------------------------------------- Claude path (fake client, no network)
def _fake_client(payload):
    class Msgs:
        def create(self, **kw):
            Msgs.kw = kw
            block = types.SimpleNamespace(type="text", text=json.dumps(payload))
            return types.SimpleNamespace(stop_reason="end_turn", content=[block])
    return types.SimpleNamespace(beta=types.SimpleNamespace(messages=Msgs())), Msgs


def test_claude_values_are_kept_only_when_the_quote_is_in_the_document(docs):
    doc = docs[1]
    pdf = open(os.path.join(EX, ORDER[1][0]), "rb").read()
    client, msgs = _fake_client({"answers": [
        {"key": "design_p_int", "value": "15 barg", "quote": "Internal design pressure 15 barg", "page": 1},
        {"key": "mdmt1_temp", "value": "-46 °C", "quote": "MDMT -46 °C impact tested", "page": 1},       # fabricated
        {"key": "lethal", "value": "maybe", "quote": "Lethal service No", "page": 1}],                    # bad value
        "nozzles": []})
    acc, rej = extract_with_claude(doc, pdf, client=client, model="test-model")
    assert [c.key for c in acc] == ["design_p_int"] and acc[0].method == "claude" and acc[0].si == pytest.approx(15e5)
    assert {r["key"]: r["reason"] for r in rej} == {"mdmt1_temp": "quote not found in the document text",
                                                    "lethal": "value does not parse as bool"}
    kw = msgs.kw
    assert kw["model"] == "test-model" and kw["fallbacks"] == "default"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert verify_quote(doc, "hp gas/condensate   SEPARATOR", 5) == 1        # whitespace/case-insensitive, any page


def test_claude_needs_a_model(docs, monkeypatch):
    monkeypatch.delenv("UDR_CLAUDE_MODEL", raising=False)
    with pytest.raises(ValueError, match="UDR_CLAUDE_MODEL"):
        extract_with_claude(docs[1], b"%PDF", client=object())


# --------------------------------------------------------------------------- filling the official form
@pytest.mark.skipif(not FORM or not os.path.exists(FORM), reason="set UDR_FORM_PDF to your copy of Form U-DR-1")
def test_fill_writes_values_and_check_boxes(comp):
    blank = open(FORM, "rb").read()
    out = udr1.fill_form(blank, comp)
    v = udr1.read_filled(out)
    assert v["Owner"] == "Demo Energy Ltd. (fictitious)" and v["PressureInternal Design Pressure"] == "15 barg"
    assert v["roup6"] == "/Choice1" and v["Group10"] == "/Choice1" and v["group69"] == "/Choice3"
    assert v["TYPE OF JOINT Use  T ypes as Described in U W 12Category A.0"] == "Type 1"
    assert v["DescriptionRow6"] == "M1 Manway" and v["1"] == "900 mm"     # parent.child names do not collide
    assert "Coating specification: SPEC-PT-004 system C" in v["GENERAL NOTESRow1"]     # too long for its box
    for k in ("Date", "User"):
        assert k not in v                                                    # left for the engineer
    with pytest.raises(ValueError, match="not the fillable Form U-DR-1"):
        udr1.fill_form(open(os.path.join(EX, ORDER[0][0]), "rb").read(), comp)


# --------------------------------------------------------------------------- web app
@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import sys
    from fastapi.testclient import TestClient
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "design_requirements"))
    from design_requirements.api import app
    return TestClient(app)


def _uploads(order):
    return [("files", (n, open(os.path.join(EX, n), "rb").read(), "application/pdf")) for n, _ in order]


def test_api_example_compile_and_overrides(client):
    meta = client.get("/api/meta").json()
    assert len(meta["fields"]) == 129 and meta["claude_available"] in (True, False)
    r = client.get("/api/example/compile").json()
    assert r["summary"]["required_filled"] == 27 and r["gaps"] == ["cyclic_service"]
    roles = json.dumps([r for _, r in ORDER])
    r = client.post("/api/compile", files=_uploads(ORDER),
                    data={"roles": roles, "overrides": json.dumps({"cyclic_service": "no"})}).json()
    assert r["gaps"] == []
    bad = client.post("/api/compile", files=_uploads(ORDER),
                      data={"roles": roles, "overrides": json.dumps({"orientation": "diagonal"})})
    assert bad.status_code == 422


def test_api_compile_uploaded_files_in_priority_order(client):
    r = client.post("/api/compile", files=_uploads(list(reversed(ORDER))), data={"roles": json.dumps([r for _, r in reversed(ORDER)])}).json()
    ca = r["decisions"]["ca_shell_int"]
    assert ca["display"] == "3 mm" and ca["status"] == "conflict"


def test_api_fill_rejects_a_pdf_that_is_not_the_form(client):
    pdf = open(os.path.join(EX, ORDER[0][0]), "rb").read()
    r = client.post("/api/example/fill", files={"form": ("x.pdf", pdf, "application/pdf")})
    assert r.status_code == 422
