"""Form compiler (pinneapple_data.formfill): the built-in ASME U-DR-1 template on three fictitious V-101 documents
(conflicts, gaps, tables, filling), any fillable PDF through a generated spec, JSON specs, and the local-LLM path
against a fake Ollama server (quote and value checks).

Filling U-DR-1 needs the user's copy of the fillable ASME form (not distributed): that test runs when UDR_FORM_PDF
points to it and is skipped otherwise. The generic tests build their own fillable PDF with reportlab.
"""
import http.server
import json
import os
import threading

import pytest

pytest.importorskip("pdfplumber")
pytest.importorskip("pypdf")

from pinneapple_data import formfill as ff  # noqa: E402
from pinneapple_data import udr1  # noqa: E402
from pinneapple_data.udr1 import FIELD_BY_KEY  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "apps", "design_requirements", "examples")
EX = os.path.join(EXAMPLES, "asme_u-dr-1")
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


# --------------------------------------------------------------------------- the other supported formats
FORMAT_SETS = {   # template: (documents in priority order, planted conflict, planted gap, values checked)
    "psv": (["relief_load_summary_PSV-101_scanned.pdf", "psv_sizing_PSV-101.pdf", "valve_specification_PSV-101.pdf"],
            ("set_pressure", ["15 barg", "14.5 barg"]), "valve_type",
            {"tag": "PSV-101", "governing_case": "fire", "fluid_state": "gas", "relieving_rate": "18500 kg/h",
             "k_ratio": "1.27", "z_factor": "0.92", "relieving_temp": "160 °C", "overpressure": "21 % (fire case)",
             "bp_built_up": "0.7 barg", "area_required": "2650 mm2", "orifice": "P", "inlet": "4 in CL300 RF",
             "trim_material": "SS 316", "lifting_lever": "yes", "test_gag": "no"}),
    "shell_tube": (["process_datasheet_E-101.pdf", "mechanical_datasheet_E-101.pdf"],
                   ("tube_p_design", ["15 barg", "16 barg"]), "tema_class",
                   {"tema_type": "AES", "duty": "1.45 MW", "shell_fluid": "Cooling water", "tube_fluid": "Natural gas",
                    "shell_flow": "125000 kg/h", "tube_t_in": "85 °C", "shell_t_out": "40 °C",
                    "tube_fouling": "0.00018 m2K/W", "shell_p_design": "7 barg", "tube_mdmt": "-10 °C",
                    "tube_passes": "2", "shell_inlet_nozzle": "8 in CL150 RF", "tube_layout": "30",
                    "tube_material": "SA-179", "baffle_cut": "25 %", "weight_empty": "7800 kg"}),
    "tank": (["tank_specification.pdf", "process_datasheet_T-201.pdf", "mechanical_datasheet_T-201.pdf"],
             ("ca_shell", ["3 mm", "2 mm"]), "empty_rate",
             {"product": "Diesel oil", "sg": "0.85", "nominal_capacity": "10000 m3", "design_p": "20 mbarg",
              "design_vac": "2.5 mbarg", "dmt": "5 °C", "roof_type": "fixed cone", "bottom_type": "cone up",
              "annular_plate": "yes", "wind_speed": "38 m/s", "roof_live_load": "1 kPa", "heating": "no"}),
    "pump": (["process_datasheet_P-101.pdf", "pump_specification.pdf", "hydraulic_calculation_P-101.pdf"],
             ("npsha", ["4.2 m", "3.8 m"]), "viscosity",
             {"tag": "P-101 A/B", "flow_rated": "55 m3/h", "head": "225 m", "vapor_pressure": "11.5 bar(a)",
              "p_suction": "12.3 barg", "material_class": "S-6", "driver": "electric motor",
              "area_class": "Zone 2, IIA T3", "seal": "API 682 Category 2, Arrangement 2"}),
}


needs_ocr = pytest.mark.skipif(not ff.ocr_available(), reason="needs the tesseract program (OCR)")


@pytest.mark.parametrize("template", [pytest.param("psv", marks=needs_ocr), "shell_tube", "tank", "pump"])
def test_supported_format_finds_its_values_the_planted_conflict_and_gap(template):
    names, (ckey, cvals), gap, values = FORMAT_SETS[template]
    spec = ff.get_spec(template)
    docs = [ff.read_document(os.path.join(EXAMPLES, template, n), name=n) for n in names]
    comp = ff.compile(docs, [c for d in docs for c in ff.extract_rules(d, spec)], spec)
    assert [d.key for d in comp.conflicts()] == [ckey]
    assert [c.display() for c in comp.decisions[ckey].candidates] == cvals       # first document wins
    assert [d.key for d in comp.gaps()] == [gap]
    for k, v in values.items():
        assert comp.decisions[k].display == v, k
    assert ff.FormSpec.loads(spec.dumps()).to_dict() == spec.to_dict()
    pdf = ff.render_datasheet(comp)
    assert pdf.startswith(b"%PDF")


# --------------------------------------------------------------------------- scanned PDFs (OCR)
def _scan(pdf_bytes, angle=0.7, seed=0):
    """Image-only copy of a PDF that looks scanned (rotated, speckled, blurred, JPEG)."""
    import io
    import random

    import pypdfium2 as pdfium
    from PIL import Image, ImageFilter
    rnd, pages = random.Random(seed), []
    for page in pdfium.PdfDocument(pdf_bytes):
        im = page.render(scale=200 / 72).to_pil().convert("L").rotate(angle, expand=True, fillcolor=255)
        px = im.load()
        for _ in range(im.width * im.height // 400):
            px[rnd.randrange(im.width), rnd.randrange(im.height)] = rnd.choice((0, 180, 255))
        buf = io.BytesIO()
        im.filter(ImageFilter.GaussianBlur(0.6)).save(buf, "JPEG", quality=60)
        pages.append(Image.open(buf))
    out = io.BytesIO()
    pages[0].save(out, "PDF", resolution=200, save_all=True, append_images=pages[1:])
    return out.getvalue()


@needs_ocr
def test_scanned_table_is_read_like_the_native_pdf():
    """The scanned relief load summary gives the values the native one has, each with an OCR confidence."""
    scan = os.path.join(EXAMPLES, "psv", "relief_load_summary_PSV-101_scanned.pdf")
    doc = ff.read_document(scan)
    assert list(doc.ocr_pages) == [1] and doc.ocr_pages[1] > 85 and doc.unread_pages == []
    got = {c.key: c for c in ff.extract_rules(doc, ff.get_spec("psv"))}
    want = {"tag": "PSV-101", "pid": "PID-100-002", "governing_case": "fire", "fluid": "Hydrocarbon vapour",
            "relieving_rate": "18500 kg/h", "molecular_weight": "19.6", "k_ratio": "1.27", "z_factor": "0.92",
            "operating_temp": "85 °C", "relieving_temp": "160 °C", "operating_pressure": "12 barg",
            "set_pressure": "15 barg", "overpressure": "21 % (fire case)", "bp_superimposed": "0.5 barg",
            "bp_built_up": "0.7 barg"}
    assert {k: got[k].display() for k in want} == want
    assert all(0 < got[k].ocr <= 100 for k in want)
    # a real OCR slip on this scan ("HP." for "HP"): low confidence, so the compiled item asks for a check
    pe = got["protected_equipment"]
    assert pe.display() in ("V-101 HP Gas/Condensate Separator", "V-101 HP. Gas/Condensate Separator")
    if pe.display().endswith("HP. Gas/Condensate Separator"):
        d = ff.compile([doc], [pe], ff.get_spec("psv")).decisions["protected_equipment"]
        assert pe.ocr < 85 and "check it against the scan" in d.note


@needs_ocr
def test_scanned_borderless_table_is_split_at_wide_gaps():
    pytest.importorskip("reportlab")
    import io

    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.setFont("Helvetica", 10)
    c.drawString(50, 800, "Pump data (no grid)")
    for i, (k, v) in enumerate((("Rated flow", "55 m3/h"), ("Differential head", "225 m"), ("NPSH available", "4.2 m"),
                                ("Viscosity", "0.45 cP"))):
        c.drawString(50, 780 - 18 * i, k)
        c.drawString(260, 780 - 18 * i, v)
    c.drawString(50, 690, "Note: the pump shall be suitable for continuous duty.")
    c.save()
    doc = ff.read_document(_scan(buf.getvalue(), angle=-0.5))
    got = {c.key: c.display() for c in ff.extract_rules(doc, ff.get_spec("pump"))}
    assert got == {"flow_rated": "55 m3/h", "head": "225 m", "npsha": "4.2 m", "viscosity": "0.45 cP"}


def test_scanned_pages_without_ocr_are_listed_not_guessed():
    scan = os.path.join(EXAMPLES, "psv", "relief_load_summary_PSV-101_scanned.pdf")
    doc = ff.read_document(scan, ocr="never")
    assert doc.unread_pages == [1] and not doc.has_text and doc.ocr_pages == {}
    assert ff.extract_rules(doc, ff.get_spec("psv")) == []


def test_low_ocr_confidence_is_flagged_for_checking():
    spec = ff.get_spec("pump")
    doc = ff.Document("scan.pdf", ["x"])
    cand = ff.Candidate("npsha", 4.2, "4.2 m", "scan.pdf", 1, "NPSH available | 4.2 m", unit="m", si=4.2, ocr=61.0)
    d = ff.compile([doc], [cand], spec).decisions["npsha"]
    assert d.status == "filled" and "OCR with 61% confidence" in d.note


def test_vapour_pressure_in_bar_absolute_is_compared_as_gauge():
    p = udr1.parse_value(ff.get_spec("pump")["vapor_pressure"], "11.5 bar(a)")
    assert p["si"] == pytest.approx(11.5e5 - 101325.0) and p["gauge"] is True
    assert udr1.parse_value(ff.get_spec("tank")["design_p"], "20 mbarg")["si"] == pytest.approx(2000.0)


def test_numbers_with_thousands_separators_and_decimal_commas():
    q = ff.get_spec("psv")["relieving_rate"]
    assert ff.parse_value(q, "18,500 kg/h")["value"] == 18500.0
    assert ff.parse_value(q, "1,250,000.5 kg/h")["value"] == 1250000.5
    assert ff.parse_value(FIELD_BY_KEY["specific_gravity"], "0,72")["value"] == 0.72


def test_milli_and_mega_prefixes_are_not_confused():
    from pinneapple_data.physical_units import try_parse_unit
    assert try_parse_unit("mPa.s").to_si(1.0) == pytest.approx(1e-3)
    assert try_parse_unit("MPa").to_si(1.0) == pytest.approx(1e6)
    assert try_parse_unit("mW").to_si(1.0) == pytest.approx(1e-3) and try_parse_unit("MW").to_si(1.0) == 1e6
    assert ff.parse_value(ff.get_spec("pump")["viscosity"], "0.45 mPa.s")["si"] == pytest.approx(4.5e-4)


def test_side_by_side_columns_need_a_header_naming_two_sides():
    spec = ff.get_spec("shell_tube")
    doc = ff.Document("x", [""], [(1, [["Item", "Shell side", "Tube side"], ["Design pressure (barg)", "7", "15"],
                                       ["Design temperature", "65 °C", "120 °C"]]),
                                  (1, [["Design pressure", "99 barg"]])])
    got = {c.key: c.display() for c in ff.extract_rules(doc, spec)}
    assert got == {"shell_p_design": "7 barg", "tube_p_design": "15 barg", "shell_t_design": "65 °C",
                   "tube_t_design": "120 °C"}      # a plain "Design pressure" row names no side: not guessed


# --------------------------------------------------------------------------- local LLM (fake Ollama server)
class FakeOllama:
    """Real HTTP server on localhost speaking the two Ollama endpoints the client uses."""

    def __init__(self, reply, models=("test-model:latest",)):
        self.reply, self.requests = reply, []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send({"models": [{"name": m} for m in models]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(req)
                self._send({"model": req["model"], "done": True,
                            "message": {"role": "assistant", "content": json.dumps(outer.reply)}})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture()
def ollama():
    servers = []

    def make(reply, **kw):
        servers.append(FakeOllama(reply, **kw))
        return servers[-1]
    yield make
    for s in servers:
        s.close()


def test_llm_values_are_kept_only_when_quote_and_value_are_in_the_document(docs, ollama):
    doc = docs[1]
    srv = ollama({"answers": [
        {"key": "design_p_int", "value": "15 barg", "quote": "Internal design pressure 15 barg", "page": 1},
        {"key": "mdmt1_temp", "value": "-46 °C", "quote": "MDMT -46 °C impact tested", "page": 1},       # fabricated
        {"key": "design_t_int", "value": "150 °C", "quote": "Internal design pressure 15 barg", "page": 1},  # not in quote
        {"key": "lethal", "value": "maybe", "quote": "Lethal service No", "page": 1}],                    # bad value
        "nozzle": [{"description": "N1 Gas/liquid inlet", "number": "1", "size": "8 in", "flange_type": "WNRF",
                    "class": "300", "quote": "Gas/liquid inlet | 1 | 8 in", "page": 1}]})
    client = ff.OllamaClient(url=srv.url, model="test-model")
    assert client.available() and not ff.OllamaClient(url=srv.url, model="other").available()
    acc, rej = ff.extract_with_llm(doc, udr1.SPEC, client)
    assert [c.key for c in acc if c.key != "nozzle"] == ["design_p_int"]
    assert acc[0].method == "llm" and acc[0].si == pytest.approx(15e5)
    assert {r["key"]: r["reason"] for r in rej} == {"mdmt1_temp": "quote not found in the document text",
                                                    "design_t_int": "value is not in the quoted text",
                                                    "lethal": "value does not parse as bool",
                                                    "nozzle": "table row quote not found in the document text"}
    req = srv.requests[0]
    assert req["model"] == "test-model" and req["stream"] is False and req["options"]["temperature"] == 0
    assert req["format"]["properties"]["answers"]["items"]["properties"]["key"]["enum"][0] == "owner"
    assert "=== Page 1 ===" in req["messages"][1]["content"]           # the text is sent, the PDF never leaves
    assert ff.verify_quote(doc, "hp gas/condensate   SEPARATOR", 5) == 1    # whitespace/case-insensitive, any page


def test_llm_needs_a_model_and_a_server():
    with pytest.raises(ff.OllamaError, match="no model"):
        ff.OllamaClient(url="http://127.0.0.1:9", model=None).chat_json("s", "u", {})
    with pytest.raises(ff.OllamaError, match="cannot reach Ollama"):
        ff.OllamaClient(url="http://127.0.0.1:9", model="m", timeout=2).chat_json("s", "u", {})
    assert not ff.OllamaClient(url="127.0.0.1:9", model="m").available(timeout=1)


def test_long_documents_are_sent_in_page_chunks(docs, ollama):
    srv = ollama({"answers": [], "nozzle": []})
    ff.extract_with_llm(docs[2], udr1.SPEC, ff.OllamaClient(url=srv.url, model="m"), max_chars=1500)
    assert len(srv.requests) == docs[2].n_pages >= 2


# --------------------------------------------------------------------------- any fillable PDF
@pytest.fixture(scope="module")
def blank_hx(tmp_path_factory):
    """A small fillable form the tool has never seen (heat exchanger request), built with reportlab."""
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas
    path = tmp_path_factory.mktemp("forms") / "hx.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(50, 800, "Heat exchanger data request (test form)")
    y = 760
    for name, tip in (("DesignPressureShell", "Design pressure, shell side"), ("DesignTempTube", ""),
                      ("TemaType", "TEMA type"), ("Text3", "")):
        c.acroForm.textfield(name=name, tooltip=tip, x=250, y=y, width=200, height=16)
        y -= 30
    c.acroForm.checkbox(name="Insulated", x=250, y=y, size=14)
    c.acroForm.radio(name="Orientation", value="Horizontal", selected=False, x=250, y=y - 30, size=14)
    c.acroForm.radio(name="Orientation", value="Vertical", selected=False, x=300, y=y - 30, size=14)
    c.save()
    return path.read_bytes()


def test_any_fillable_pdf_becomes_a_spec(blank_hx):
    spec = ff.FormSpec.from_pdf(blank_hx)
    got = {f.key: (f.label, f.kind, bool(f.patterns)) for f in spec.fields}
    assert got == {"design_pressure_shell_side": ("Design pressure, shell side", "text", True),
                   "design_temp_tube": ("Design Temp Tube", "text", True), "tema_type": ("TEMA type", "text", True),
                   "text3": ("Text3", "text", False), "insulated": ("Insulated", "bool", True),
                   "orientation": ("Orientation", "choice", True)}
    assert spec["orientation"].options == ("horizontal", "vertical")
    assert ff.FormSpec.loads(spec.dumps()).to_dict() == spec.to_dict()
    with pytest.raises(ValueError, match="no fillable form fields"):
        ff.FormSpec.from_pdf(open(os.path.join(EX, ORDER[0][0]), "rb").read())


def test_any_fillable_pdf_is_filled_from_documents(blank_hx):
    spec = ff.FormSpec.from_pdf(blank_hx)
    doc = ff.read_document(b"Design pressure shell side: 12 barg\nTEMA type: AES\nInsulated: yes\n"
                           b"Orientation: horizontal", name="hx_datasheet.txt")
    comp = ff.compile([doc], ff.extract_rules(doc, spec), spec)
    assert {k: d.display for k, d in comp.decisions.items() if d.value is not None} == {
        "design_pressure_shell_side": "12 barg", "tema_type": "AES", "insulated": "yes", "orientation": "horizontal"}
    v = ff.read_filled(ff.fill_pdf(blank_hx, comp))
    assert v == {"DesignPressureShell": "12 barg", "TemaType": "AES", "Insulated": "/Yes", "Orientation": "/Horizontal"}
    rec = comp.record()                                    # the data for other forms and calculations
    assert rec["tema_type"] == {"value": "AES", "display": "AES", "unit": None, "si": None, "source": "hx_datasheet.txt p.1"}


def test_json_spec_round_trip_and_custom_patterns(blank_hx):
    spec = ff.FormSpec.from_dict({**ff.FormSpec.from_pdf(blank_hx).to_dict(), "id": "hx"})
    d = spec.to_dict()
    for f in d["fields"]:
        if f["key"] == "text3":
            f.update(label="Shell material", patterns=["shell material", "shell moc"], required=True)
        if f["key"] == "design_temp_tube":
            f.update(kind="quantity", quantity="temperature", patterns=["design temperature,? tube side"])
    custom = ff.FormSpec.from_dict(d)
    doc = ff.read_document(b"Shell MOC: SA-516 Gr. 70\nDesign temperature, tube side: 250 degC", name="x.txt")
    comp = ff.compile([doc], ff.extract_rules(doc, custom), custom)
    assert comp.decisions["text3"].display == "SA-516 Gr. 70" and comp.summary()["required_filled"] == 1
    assert comp.decisions["design_temp_tube"].chosen.si == pytest.approx(523.15)
    assert ff.get_spec("asme_u-dr-1").to_dict() == ff.FormSpec.loads(ff.get_spec("asme_u-dr-1").dumps()).to_dict()
    with pytest.raises(KeyError, match="unknown form template"):
        ff.get_spec("nope")
    with pytest.raises(ValueError, match="kind must be one of"):
        ff.Field("x", "X", kind="date")


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
    with pytest.raises(ValueError, match="not the fillable form"):
        udr1.fill_form(open(os.path.join(EX, ORDER[0][0]), "rb").read(), comp)


# --------------------------------------------------------------------------- web app
@pytest.fixture(scope="module")
def api():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "design_requirements"))
    from design_requirements import api
    return api


@pytest.fixture(scope="module")
def client(api):
    from fastapi.testclient import TestClient
    return TestClient(api.app)


def _uploads(order):
    return [("files", (n, open(os.path.join(EX, n), "rb").read(), "application/pdf")) for n, _ in order]


def test_api_example_compile_and_overrides(client):
    meta = client.get("/api/meta").json()
    assert len(meta["spec"]["fields"]) == 129 and meta["llm_available"] is False
    assert [t["id"] for t in meta["formats"]] == ["asme_u-dr-1", "psv", "shell_tube", "tank", "pump"]
    assert all(f["example"] for f in meta["formats"]) and meta["ocr_available"] == ff.ocr_available()
    r = client.get("/api/example/compile").json()
    assert r["summary"]["required_filled"] == 27 and r["gaps"] == ["cyclic_service"] and len(r["tables"]["nozzle"]) == 6
    assert r["record"]["design_p_int"]["si"] == pytest.approx(15e5)
    roles = json.dumps([r for _, r in ORDER])
    r = client.post("/api/compile", files=_uploads(ORDER),
                    data={"roles": roles, "overrides": json.dumps({"cyclic_service": "no"})}).json()
    assert r["gaps"] == []
    bad = client.post("/api/compile", files=_uploads(ORDER),
                      data={"roles": roles, "overrides": json.dumps({"orientation": "diagonal"})})
    assert bad.status_code == 422 and "must be one of" in bad.json()["detail"]
    assert client.post("/api/compile", files=_uploads(ORDER), data={"use_llm": "true"}).status_code == 503


def test_api_compile_uploaded_files_in_priority_order(client):
    r = client.post("/api/compile", files=_uploads(list(reversed(ORDER))),
                    data={"roles": json.dumps([r for _, r in reversed(ORDER)])}).json()
    ca = r["decisions"]["ca_shell_int"]
    assert ca["display"] == "3 mm" and ca["status"] == "conflict"


def test_api_datasheet_for_every_format(client):
    for t in ("asme_u-dr-1", "psv", "shell_tube", "tank", "pump"):
        r = client.post("/api/example/datasheet", data={"template": t})
        assert r.status_code == 200 and r.content.startswith(b"%PDF"), t
    r = client.get("/api/example/compile", params={"template": "pump",
                                                   "overrides": json.dumps({"viscosity": "0.45 cP"})}).json()
    assert r["gaps"] == [] and r["decisions"]["viscosity"]["note"] == "entered by engineer"
    files = [("files", (n, open(os.path.join(EXAMPLES, "pump", n), "rb").read(), "application/pdf"))
             for n in FORMAT_SETS["pump"][0]]
    r = client.post("/api/datasheet", files=files, data={"template": "pump"})
    assert r.status_code == 200 and "pump_P-101_A_B" in r.headers["content-disposition"]
    r = client.post("/api/fill", files=files, data={"template": "pump"})
    assert r.status_code == 422 and "no fillable official form" in r.json()["detail"]
    assert client.get("/api/example/files/tank/tank_specification.pdf").status_code == 200
    assert client.get("/api/example/files/tank/../asme_u-dr-1/x.pdf").status_code == 404


def test_api_fill_rejects_a_pdf_that_is_not_the_form(client):
    pdf = open(os.path.join(EX, ORDER[0][0]), "rb").read()
    r = client.post("/api/example/fill", files={"form": ("x.pdf", pdf, "application/pdf")})
    assert r.status_code == 422


def test_api_any_fillable_pdf_and_custom_spec(client, blank_hx):
    doc = ("hx.txt", b"Design pressure shell side: 12 barg\nTEMA type: AES", "text/plain")
    form = ("hx.pdf", blank_hx, "application/pdf")
    r = client.post("/api/compile", files=[("files", doc), ("form", form)], data={"template": "auto"}).json()
    assert r["spec"]["id"] == "auto" and r["decisions"]["tema_type"]["display"] == "AES"
    pdf = client.post("/api/fill", files=[("files", doc), ("form", form)], data={"template": "auto"})
    assert pdf.status_code == 200 and ff.read_filled(pdf.content)["TemaType"] == "AES"
    assert client.post("/api/compile", files=[("files", doc)], data={"template": "auto"}).status_code == 422
    spec = client.post("/api/spec/from-pdf", files={"form": form})
    assert spec.status_code == 200 and spec.json()["id"] == "hx"
    r = client.post("/api/compile", files=[("files", doc), ("spec", ("hx.spec.json", spec.content, "application/json"))],
                    data={"template": "custom"}).json()
    assert r["spec"]["id"] == "hx" and r["decisions"]["design_pressure_shell_side"]["display"] == "12 barg"
    bad = client.post("/api/compile", files=[("files", doc), ("spec", ("s.json", b"{}", "application/json"))],
                      data={"template": "custom"})
    assert bad.status_code == 422
    assert client.get("/api/spec").json()["id"] == "asme_u-dr-1"


def test_api_local_llm_when_ollama_is_configured(api, client, ollama, monkeypatch):
    srv = ollama({"answers": [
        {"key": "insulation_external", "value": "Mineral wool, 50 mm", "quote": "External insulation Mineral wool, 50 mm",
         "page": 1},
        {"key": "cyclic_service", "value": "no", "quote": "Cyclic service: No", "page": 1}], "nozzle": []})   # made up
    monkeypatch.setenv("UDR_OLLAMA_URL", srv.url)
    monkeypatch.setenv("UDR_OLLAMA_MODEL", "test-model")
    api._LLM_CACHE["t"] = 0.0
    assert client.get("/api/meta").json()["llm_available"] is True
    r = client.post("/api/compile", files=_uploads(ORDER[2:]), data={"use_llm": "true"}).json()
    methods = [c["method"] for c in r["decisions"]["insulation_external"]["candidates"]]
    assert methods == ["rule", "llm"] and r["decisions"]["insulation_external"]["status"] == "filled"   # they agree
    assert [(x["key"], x["doc"], x["reason"]) for x in r["rejected"]] == [
        ("cyclic_service", ORDER[2][0], "quote not found in the document text")]
    assert "cyclic_service" in r["gaps"]
    api._LLM_CACHE["t"] = 0.0
    assert client.get("/api/meta").json()["llm_available"] is True
    r = client.post("/api/compile", files=_uploads(ORDER[2:]), data={"use_llm": "true"}).json()
    assert srv.requests and r["rejected"] == [] or all(x["doc"] == ORDER[2][0] for x in r["rejected"])
    api._LLM_CACHE["t"] = 0.0
