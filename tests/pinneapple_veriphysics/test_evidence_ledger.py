"""Evidence ledger, Validation Lab and evidence package: a test reads "passed" only when an intact record shows that
it ran (inputs SHA-256, outputs, time); every other case reads "not run"."""
from __future__ import annotations

import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from pinneapple_veriphysics.evidence_package import (
    build_package,
    build_report,
    load_source,
    make_figures,
    render_html,
    render_pdf,
    to_study_result,
    verify_package,
)
from pinneapple_veriphysics.ledger import (
    NOT_RUN,
    EvidenceEntry,
    EvidenceLedger,
    sha256_bytes,
    utc_now,
)
from pinneapple_veriphysics.validation_lab import (
    DISCLAIMER,
    CodeNotAllowed,
    category_of,
    config_from_spec,
    run_validation,
    to_toml,
)
from pinneapple_veriphysics.validation_lab import render_html as lab_html

if sys.version_info >= (3, 11):
    import tomllib
else:                                                                # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
EX = ROOT / "examples" / "physics_check"
H = sha256_bytes(b"inputs")


# ---------------------------------------------------------------------------------------------- ledger
def test_status_comes_only_from_complete_intact_records():
    led = EvidenceLedger()
    led.record("ok", passed=True, inputs_sha256=H, outputs={"r": 1e-4}, started_at=utc_now(), finished_at=utc_now())
    led.record("bad", passed=False, inputs_sha256=H, outputs={"r": 1.0}, started_at=utc_now(), finished_at=utc_now())
    assert led.status("ok") == "passed" and led.status("bad") == "failed"
    assert led.status("never") == NOT_RUN and led.claim("never") == "never: not run"

    nohash = led.add(EvidenceEntry("nohash", outputs={}, started_at=utc_now(), finished_at=utc_now(),
                                   passed=True).seal())
    assert "no SHA-256 of the inputs" in nohash.problems() and led.status("nohash") == NOT_RUN
    noout = led.add(EvidenceEntry("noout", inputs_sha256=H, started_at=utc_now(), finished_at=utc_now(),
                                  passed=True).seal())
    assert led.status("noout") == NOT_RUN and "no recorded outputs" in noout.problems()
    led.add(EvidenceEntry("notime", inputs_sha256=H, outputs={}, passed=True).seal())
    assert led.status("notime") == NOT_RUN
    unsealed = led.add(EvidenceEntry("unsealed", inputs_sha256=H, outputs={}, started_at="t", finished_at="t",
                                     passed=True))
    assert led.status("unsealed") == NOT_RUN and "record altered after it was sealed" in unsealed.problems()


def test_edited_record_no_longer_counts_after_a_round_trip():
    led = EvidenceLedger()
    led.record("residual", passed=False, inputs_sha256=H, outputs={"r": 0.3}, started_at=utc_now(),
               finished_at=utc_now())
    d = json.loads(json.dumps(led.to_dict()))
    assert EvidenceLedger.from_dict(d).status("residual") == "failed"
    d["entries"][0]["passed"] = True                                  # someone flips the verdict
    back = EvidenceLedger.from_dict(d)
    assert back.status("residual") == NOT_RUN
    assert "record invalid: record altered after it was sealed" in back.claim("residual")


def test_run_context_records_time_and_a_crash_is_not_a_pass():
    led = EvidenceLedger()
    with led.run("fast", inputs_sha256=H) as rec:
        rec.outputs, rec.passed = {"x": 1}, True
    e = led.latest("fast")
    assert e.started_at <= e.finished_at and led.status("fast") == "passed"
    with pytest.raises(ZeroDivisionError):
        with led.run("crash", inputs_sha256=H) as rec:
            rec.outputs = {}
            raise ZeroDivisionError("division by zero")
    assert led.status("crash") == NOT_RUN and "ZeroDivisionError" in led.entries[-1].error


# ---------------------------------------------------------------------------------------------- validation lab
@pytest.fixture(scope="module")
def lab_pass(tmp_path_factory):
    pytest.importorskip("meshio")
    w = tmp_path_factory.mktemp("lab") / "pass"
    shutil.copytree(EX / "pass", w)
    return run_validation(w, allow_code=True).to_dict()


def test_rule_groups():
    assert category_of("PC301") == category_of("PC403") == category_of("PC801") == "verification"
    assert category_of("PC701") == category_of("PC601") == "validation"
    assert category_of("PC101") == category_of("PC201") == category_of("PC401") == category_of("PC501") == "physics"
    assert category_of("PC002") is None


def test_lab_separates_the_three_groups_and_records_every_outcome(lab_pass):
    lab = lab_pass
    assert lab["decision"] == "pass" and lab["disclaimer"] == DISCLAIMER
    assert set(lab["categories"]) == {"verification", "validation", "physics"}
    for cat in lab["categories"].values():
        assert cat["checks"] > 0 and cat["passed"] == cat["checks"]
    assert "fields/heat_plate.npz" in lab["categories"]["validation"]["not_run"]      # no reference given there
    led = EvidenceLedger.from_dict(lab["ledger"])
    assert len(led.entries) == lab["summary"]["checks"] and not led.invalid()
    assert all(led.status(o["check_id"]) == "passed" for o in lab["outcomes"])
    assert len({f["sha256"] for f in lab["provenance"]["inputs"]}) > 10
    page = lab_html(lab)
    assert "Numerical verification" in page and "Physical consistency" in page and "not run" in page


def test_lab_on_defective_results_fails_in_the_right_groups(tmp_path):
    pytest.importorskip("meshio")
    shutil.copytree(EX / "fail", tmp_path / "fail")
    lab = run_validation(tmp_path / "fail", allow_code=True).to_dict()
    assert lab["decision"] == "fail"
    bad = {o["rule"] for o in lab["outcomes"] if not o["passed"]}
    assert {"PC101", "PC301", "PC701"} <= bad
    cats = lab["categories"]
    assert cats["physics"]["errors"] and cats["verification"]["errors"] and cats["validation"]["errors"]
    led = EvidenceLedger.from_dict(lab["ledger"])
    assert led.status(next(o["check_id"] for o in lab["outcomes"] if o["rule"] == "PC101" and not o["passed"])) \
        == "failed"


def test_uploaded_code_is_refused(tmp_path):
    shutil.copytree(EX / "pass", tmp_path / "w")
    with pytest.raises(CodeNotAllowed):
        run_validation(tmp_path / "w")


def test_config_from_spec_is_valid_toml_and_runs(tmp_path):
    pytest.importorskip("meshio")
    shutil.copy(EX / "pass" / "fields" / "heat_plate.npz", tmp_path)
    toml = config_from_spec({"results": [{"path": "heat_plate.npz", "physics": "heat",
                                          "parameters": {"k": "10 W/(m*K)", "q": "1000 W/m3"},
                                          "units": {"T": "K"}, "bounds": {"T": [250, 400]},
                                          "bc": {"x=max": {"T": "300 + 25*(1 - y**2)"}}}]})
    doc = tomllib.loads(toml)
    assert doc["result"][0]["bc"]["x=max"]["T"].startswith("300")
    (tmp_path / "pinneapple-check.toml").write_text(toml)
    lab = run_validation(tmp_path).to_dict()
    rules = {o["rule"] for o in lab["outcomes"]}
    assert {"PC103", "PC201", "PC301", "PC401", "PC501"} <= rules and lab["decision"] == "pass"
    assert tomllib.loads(to_toml({"check": {"x": float("inf")}}))["check"]["x"] == float("inf")
    with pytest.raises(ValueError):
        config_from_spec({"models": [{"path": "evil.py:f"}]})


# ---------------------------------------------------------------------------------------------- evidence report
def _src(name, obj):
    return load_source(name, json.dumps(obj).encode())


def test_a_claimed_pass_without_a_record_is_printed_not_run(tmp_path):
    claim_only = {"type": "StudyResult", "name": "candidate", "tests": [{"id": "mass-balance", "passed": True}]}
    led = EvidenceLedger()
    led.record("energy-balance", passed=True, inputs_sha256=H, outputs={"imbalance": 2e-4}, started_at=utc_now(),
               finished_at=utc_now())
    with_record = {"type": "StudyResult", "name": "baseline", "tests": [{"id": "energy-balance", "passed": True}],
                   "evidence": led.to_dict()}
    srcs = [_src("candidate.json", claim_only), _src("baseline.json", with_record)]
    assert [s.contract for s in srcs] == ["StudyResult", "StudyResult"]
    rep = build_report({"title": "t"}, srcs)
    by = {t["check_id"]: t for t in rep["tests"]}
    assert by["mass-balance"]["status"] == NOT_RUN and by["mass-balance"]["source_claim"] == "passed"
    assert by["mass-balance"]["why"] == "no evidence record"
    assert by["energy-balance"]["status"] == "passed" and by["energy-balance"]["inputs_sha256"] == H
    assert rep["test_counts"] == {"passed": 1, "failed": 0, "not run": 1}
    page = render_html(rep)
    row = page[page.index("mass-balance"):].split("</tr>")[0]
    assert "not run" in row and "class='passed'" not in row
    pytest.importorskip("reportlab")
    pdf = render_pdf(rep)
    assert pdf[:4] == b"%PDF"


def test_records_are_matched_per_source():
    lab_led = EvidenceLedger()
    lab_led.record("PC101/case/mass", passed=True, inputs_sha256=H, outputs={}, started_at=utc_now(),
                   finished_at=utc_now())
    a = {"type": "StudyResult", "name": "a", "evidence": lab_led.to_dict()}
    b = {"tool": "pinneapple check", "checks": [{"rule": "PC101", "target": "case", "subject": "mass",
                                                 "passed": True, "message": "ok"}]}
    rep = build_report({}, [_src("a.json", a), _src("b.json", b)])
    st = {(t["source"], t["check_id"]): t["status"] for t in rep["tests"]}
    assert st[("a.json", "PC101/case/mass")] == "passed"
    assert st[("b.json", "PC101/case/mass")] == NOT_RUN                 # someone else's run is not evidence


def test_tampered_record_in_a_source_reads_not_run():
    led = EvidenceLedger()
    led.record("residual", passed=False, inputs_sha256=H, outputs={"r": 0.4}, started_at=utc_now(),
               finished_at=utc_now())
    d = led.to_dict()
    d["entries"][0]["passed"] = True
    rep = build_report({}, [_src("ledger.json", d)])
    (t,) = rep["tests"]
    assert t["status"] == NOT_RUN and "altered" in t["why"] and rep["invalid_records"] == ["residual"]


def test_baseline_comparison_and_package_round_trip(tmp_path):
    csv = b"run,rel_l2,max_err\nFEM n=24,0.0012,0.003\nPINN 200 epochs,0.035,0.08\n"
    led = EvidenceLedger()
    led.record("trust-card", passed=True, inputs_sha256=H, outputs={"decision": "APPROVED"}, started_at=utc_now(),
               finished_at=utc_now())
    srcs = [load_source("metrics.csv", csv), _src("ledger.json", led.to_dict()), load_source("mesh.msh", b"\x00\x01")]
    assert [s.kind for s in srcs] == ["metrics", "ledger", "file"]
    rep = build_report({"title": "Poisson", "baseline": "FEM", "candidate": "PINN", "references": ["doi:x"]}, srcs)
    comp = {r["metric"]: r for r in rep["comparison"]}
    assert comp["rel_l2"]["baseline"] == pytest.approx(0.0012) and comp["rel_l2"]["candidate"] == pytest.approx(0.035)
    both = _src("study.json", {"type": "StudyResult", "name": "FEM vs PINN study", "metrics": {"rel_l2": 9.0}})
    rep2 = build_report({"baseline": "FEM", "candidate": "PINN"}, [srcs[0], both])
    assert {r["metric"]: r["candidate"] for r in rep2["comparison"]}["rel_l2"] == pytest.approx(0.035)
    assert {a["sha256"] for a in rep["artifacts"]} == {s.sha256 for s in srcs}
    pytest.importorskip("matplotlib")
    figs = make_figures(rep, tmp_path)
    assert set(figs) == {"comparison", "tests"}
    z = build_package(rep, srcs, html=render_html(rep, figs), figures=figs)
    assert verify_package(z) == []
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        assert zf.read("sources/mesh.msh") == b"\x00\x01"
        files = {n: zf.read(n) for n in zf.namelist()}
    files["sources/metrics.csv"] = b"run,rel_l2\nPINN,0.0001\n"           # tamper with a source
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n, v in files.items():
            zf.writestr(n, v)
    assert verify_package(buf.getvalue()) == ["sources/metrics.csv: SHA-256 differs from the manifest"]


def test_lab_report_feeds_the_evidence_report(lab_pass):
    rep = build_report({"title": "x"}, [_src("lab.json", lab_pass)])
    assert rep["test_counts"]["passed"] == lab_pass["summary"]["checks"] and rep["test_counts"]["not run"] == 0
    assert rep["configs"] and "pinneapple check --config" in rep["configs"][0]["reproduce"]
    assert any("no validation against data check ran" in x for x in rep["limitations"])


def test_uploads_are_written_safely_and_tables_become_csv(tmp_path):
    from pinneapple_veriphysics.validation_lab import tables_to_csv, write_inputs

    pd = pytest.importorskip("pandas")
    with pytest.raises(ValueError):
        write_inputs(tmp_path, {"../evil.txt": b"x"})
    rows = [{"x": 0.0, "y": 0.0, "z": 0.0, "T": 300.0}, {"x": 1.0, "y": 0.0, "z": 0.0, "T": 310.0}]
    write_inputs(tmp_path, {"probe.json": json.dumps(rows).encode(), "manifest.json": b'{"a": "b"}'})
    cfg, done = tables_to_csv(tmp_path, '[[result]]\npath = "probe.json"\n')
    assert done == {"probe.json": "probe.csv"} and '"probe.csv"' in cfg
    assert pd.read_csv(tmp_path / "probe.csv")["T"].tolist() == [300.0, 310.0]


def test_trust_card_checks_are_recorded_and_na_reads_not_run():
    np = pytest.importorskip("numpy")
    import pinneapple as pp
    from pinneapple_veriphysics.ledger import record_checks

    x = y = np.linspace(0, 1, 21)
    X, Y = np.meshgrid(x, y, indexing="ij")
    led = EvidenceLedger()
    t0 = utc_now()
    card = pp.trust_card({"x": x, "y": y}, X + Y, pde="laplace", dirichlet={"x_min": lambda P: P[:, 1]})
    record_checks(led, card.checks, prefix="plate", inputs_sha256=sha256_bytes((X + Y).tobytes()), started_at=t0,
                  finished_at=utc_now(), tool="pp.trust_card")
    assert led.status("plate/PDE residual") == "passed" and led.latest("plate/PDE residual").category == "verification"
    assert led.status("plate/boundary condition on x_min") == "passed"
    assert led.status("plate/reference (relative L2)") == NOT_RUN          # N/A: no reference was given


def test_shared_contract_shapes_are_read():
    from pinneapple_veriphysics.validation_lab import model_ref_target

    ref = {"kind": "ModelRef", "name": "hs", "uri": "zoo:heatsink_forced_mlp",
           "inputs": [{"name": "air_velocity_m_s"}], "outputs": [{"name": "r_base_k_w"}]}
    t = model_ref_target(ref)
    assert t["zoo"] == "heatsink_forced_mlp" and t["inputs"] == ["air_velocity_m_s"] and t["outputs"] == ["r_base_k_w"]
    with pytest.raises(ValueError):
        model_ref_target({"kind": "ModelRef", "name": "m", "uri": "file:model.pt"})
    study = {"kind": "StudyResult", "id": "s1", "title": "sweep", "study_type": "campaign",
             "checks": [{"name": "mass balance", "status": "PASS", "evidence": "run 3"}],
             "limits": ["one seed only"]}
    rep = build_report({}, [_src("study.json", study)])
    (t,) = rep["tests"]
    assert t["status"] == NOT_RUN and t["source_claim"] == "passed" and t["why"] == "no evidence record"
    assert "study.json: one seed only" in rep["limitations"] and rep["contracts"][0]["name"] == "sweep"


def test_contracts_are_validated_and_the_report_exports_a_study_result():
    pytest.importorskip("pydantic")
    from pinneapple_data.contracts import StudyResult, validate

    good = {"kind": "StudyResult", "id": "s1", "study_type": "campaign", "title": "sweep",
            "checks": [{"name": "mass balance", "status": "PASS", "value": 1e-5}]}
    bad = {"kind": "StudyResult", "id": "s2", "study_type": "not-a-type"}
    srcs = [_src("good.json", good), _src("bad.json", bad)]
    assert srcs[0].problems == [] and srcs[1].problems and "contract_problems" in srcs[1].ref()
    led = EvidenceLedger()
    led.record("energy", passed=True, inputs_sha256=H, outputs={}, started_at=utc_now(), finished_at=utc_now())
    rep = build_report({"title": "t"}, [*srcs, _src("ledger.json", led.to_dict())])
    sr = to_study_result(rep)
    assert validate(sr) == [] and StudyResult.from_dict(sr).study_type == "evidence"
    st = {c["name"]: c["status"] for c in sr["checks"]}
    assert st["good.json: mass balance"] == "NOT_RUN" and st["ledger.json: energy"] == "PASS"
    assert sr["metrics"]["passed"] == 1 and sr["metrics"]["not_run"] == 1
    with zipfile.ZipFile(io.BytesIO(build_package(rep, srcs))) as zf:
        assert validate(json.loads(zf.read("study_result.json"))) == []
