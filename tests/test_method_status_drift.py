"""The method-status snapshot does not drift, and a refresh is one reviewable commit (#1884).

``pinneapple_catalog/method_status.json`` is a generated artifact that two readers treat as a contract: the PC901
rule of ``pinneapple_check`` and the fixtures in ``examples/physics_check/fail``. So a fresh scan that disagrees
with the committed file is a build that will go red for whoever regenerates it -- which is why the file stayed
stale long enough for the issue to be filed.

These tests hold the two properties the issue asks for: the committed snapshot agrees with a fresh scan of the
tree (so regenerating it changes nothing but dates), and the comparison that says so is a command that can be run
in a job, reporting drift without failing on it.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_method_status.py"
STATUS = ROOT / "pinneapple_catalog" / "method_status.json"


def _script():
    spec = importlib.util.spec_from_file_location("_build_method_status_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _payload():
    return json.loads(STATUS.read_text(encoding="utf-8"))


def test_the_committed_snapshot_agrees_with_a_fresh_scan():
    """If this fails, the refresh is the fix -- and the status diff is what the review is for."""
    mod = _script()
    d = mod.drift()
    assert d["statuses"] == {}, f"statuses changed: {d['statuses']} -- refresh with --public --markdown"
    assert d["missing"] == [], f"methods gone from the catalog: {d['missing']}"
    assert d["tests"] == {}, f"evidence changed for {len(d['tests'])} method(s); the snapshot is stale"
    assert d["clean"]


def test_the_snapshot_keeps_the_evidence_every_status_is_audited_by():
    data = _payload()
    assert data["methods"], "the snapshot has no methods in it"
    for method_id, entry in data["methods"].items():
        assert entry["status"] in ("validated", "tested", "untested"), method_id
        assert isinstance(entry["reference_tests"], list) and isinstance(entry["other_tests"], list)
        if entry["status"] == "validated":
            assert entry["reference_tests"], f"{method_id} is validated with no reference test behind it"


def test_a_reported_status_change_is_seen_as_drift(tmp_path):
    """The comparison has to notice a change, which is the only reason it is worth running in a job."""
    mod = _script()
    doctored = json.loads(STATUS.read_text(encoding="utf-8"))
    victim = next(k for k, v in doctored["methods"].items() if v["status"] == "validated")
    doctored["methods"][victim]["status"] = "untested"
    doctored["methods"][victim]["reference_tests"] = []
    path = tmp_path / "method_status.json"
    path.write_text(json.dumps(doctored), encoding="utf-8")

    d = mod.drift(committed_path=str(path))
    assert not d["clean"]
    assert d["statuses"][victim] == ("untested", "validated")


def test_the_report_says_what_to_run_and_what_to_update(tmp_path):
    """A report is only useful if it names the fix; the fixture is the part people forget."""
    mod = _script()
    doctored = json.loads(STATUS.read_text(encoding="utf-8"))
    victim = next(k for k, v in doctored["methods"].items() if v["status"] == "untested")
    doctored["methods"][victim]["status"] = "validated"
    doctored["methods"][victim]["reference_tests"] = ["x::y"]
    path = tmp_path / "method_status.json"
    path.write_text(json.dumps(doctored), encoding="utf-8")

    buffer = io.StringIO()
    mod.report(mod.drift(committed_path=str(path)), out=buffer)
    text = buffer.getvalue()
    assert victim in text and "build_method_status.py" in text
    assert "expected.json" in text, "the physics-check fixtures move with a status, and the report must say so"


def test_the_public_page_is_rendered_from_the_snapshot():
    """docs/validation_status.md is the same file read twice; tests/test_validation_status_page.py owns the exact
    agreement, this only checks the renderer is fed by the snapshot rather than by a scan."""
    mod = _script()
    page = mod.render_public_page()
    data = _payload()
    total = sum(1 for m in data["methods"].values() if m["status"] == "validated")
    assert f"**{total}**" in page
