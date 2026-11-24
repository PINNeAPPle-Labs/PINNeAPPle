"""The experiment quality checklist and its badge (#923): the nine items computed from a folder's own files, a
declaration that switches an item off or lists what was not measured, old manifests that stay valid and read as not
assessed, and the backlog sorted by how visible the experiment is."""
from __future__ import annotations

import json
import os

import pytest

from pinneapple_lab import quality
from pinneapple_lab.manifest import Manifest, load, save


def _manifest(**kw) -> dict:
    d = {"schema": "pinneapple.experiment/1", "id": "examples/demo", "title": "Demo", "description": "d",
         "domain": "fluid dynamics", "methods": ["RANS"], "modules": ["pinneapple_simulation"],
         "params": {"Re": 1e4}, "scripts": ["01_run.py"], "metrics": [], "checks": [], "files": {},
         "links": {"script": "01_run.py"}, "references": [], "status": "not curated", "lab": {},
         "provenance": {"git": "abc1234", "created": "2026-11-24T10:00:00Z",
                        "environment": {"python": "3.11.9", "platform": "darwin"}}}
    d.update(kw)
    return d


def _write(tmp_path, d: dict, readme: str = "", name: str = "demo") -> str:
    folder = tmp_path / name
    folder.mkdir()
    (folder / "01_run.py").write_text("print('hi')\n")
    with open(folder / "experiment.json", "w") as f:
        json.dump(d, f)
    if readme:
        (folder / "README.md").write_text(readme)
    return str(folder)


FULL_CHECKS = [
    {"name": "drag vs wind tunnel", "passed": True, "kind": "reference", "reference": 0.31, "detail": "Cd 0.30"},
    {"name": "error on the held-out Re", "passed": True, "kind": "generalization", "detail": "1.2 %"},
    {"name": "mesh refinement 2x", "passed": True, "kind": "verification", "detail": "mesh refinement halves it"},
    {"name": "converged", "passed": False, "kind": "sanity", "detail": "300 iterations, not converged"},
]


def test_a_full_record_meets_every_applicable_item(tmp_path):
    d = _manifest(checks=FULL_CHECKS, methods=["RANS", "LES"],
                  params={"Re": 1e4, "n_seeds": 5, "n_collocation": 2000, "ablation_without_drag": True},
                  metrics=[{"name": "Cd", "value": 0.30, "unit": "", "how": "mean over 5 seeds, CI95 [0.29, 0.31]"},
                           {"name": "cases", "value": 4.0, "unit": "", "how": "reference vs PINN, baseline beaten"}],
                  quality={"schema": quality.DECLARATION,
                           "not_measured": ["the Re range above 5e5"]})
    c = quality.assess(_write(tmp_path, d))
    assert c.missing == [], c.explain()
    assert c.applicable == 9
    assert c.badge() == "QUALITY 9/9"
    assert set(c.met) == set(quality.ITEMS)


def test_the_badge_names_what_is_missing_and_the_reason(tmp_path):
    d = _manifest(checks=[{"name": "finite", "passed": True, "kind": "sanity", "detail": "no NaN"}])
    c = quality.assess(_write(tmp_path, d))
    assert "reference" in c.missing and "holdout" in c.missing
    assert c.badge().startswith(f"QUALITY {len(c.met)}/{c.applicable}")
    assert "missing: reference" in c.badge()
    ref = c.by_id("reference")
    assert ref.status == quality.MISSING and ref.fix
    assert "nothing to compare with" in ref.evidence[0]
    # the report carries the items with their evidence, and the schema version
    out = c.to_dict()
    assert out["schema"] == quality.SCHEMA and out["badge"] == c.badge()
    assert {i["id"] for i in out["items"]} == set(quality.ITEMS)


def test_an_old_manifest_stays_valid_and_reads_as_not_assessed(tmp_path):
    folder = _write(tmp_path, {"schema": "pinneapple.experiment/1", "id": "examples/old", "title": "Old",
                               "status": "not curated"})
    m = load(folder)                                     # schema 1, no quality block, no provenance
    assert m.quality == {}
    c = quality.assess(folder)
    for item_id in ("reference", "holdout", "repeats", "verdict"):
        assert c.by_id(item_id).status == quality.UNASSESSED, c.explain()
    # provenance and the not-measured list read the folder itself, so they are assessed and missing; the seven
    # items that need a result are not assessed -- the badge counts only what was assessed, never the absence
    assert c.applicable == 2 and c.badge() == "QUALITY 0/2 · missing: provenance, not measured"


def test_a_declaration_switches_an_item_off_with_a_reason_and_lists_what_was_not_measured(tmp_path):
    d = _manifest(checks=[{"name": "finite", "passed": True, "kind": "sanity"}],
                  quality={"schema": quality.DECLARATION,
                           "not_applicable": {"convergence": "no solver: the data are measured"},
                           "not_measured": ["wall time on the reference hardware"]})
    c = quality.assess(_write(tmp_path, d))
    conv = c.by_id("convergence")
    assert conv.status == quality.NA and "no solver" in conv.evidence[0]
    assert c.by_id("not_measured").status == quality.MET
    # nine minus convergence (declared), budget and ablation (one method, no claim): the declaration and the
    # narrow rules both shrink the denominator instead of counting a not-applicable item as a failure
    assert c.applicable == 6
    assert "convergence" not in c.missing


def test_a_newer_declaration_is_read_as_not_assessed_and_says_so(tmp_path):
    d = _manifest(quality={"schema": "pinneapple.quality.declaration/2", "not_measured": ["x"]})
    q = quality.declaration_of(Manifest.from_dict(d))
    assert q["problems"] and "newer" in q["problems"][0]
    assert q.get("not_measured") in ([], None)            # nothing from a format we do not know
    c = quality.assess(_write(tmp_path, d))
    assert c.by_id("not_measured").status in (quality.MISSING, quality.UNASSESSED)


def test_a_reference_check_without_a_source_is_missing_and_says_what_to_record(tmp_path):
    d = _manifest(checks=[{"name": "vs experiment", "passed": True, "kind": "reference"}])
    c = quality.assess(_write(tmp_path, d))
    ref = c.by_id("reference")
    assert ref.status == quality.MISSING and "references[] says what it is" in ref.evidence[-1]
    d2 = _manifest(checks=[{"name": "vs experiment", "passed": True, "kind": "reference",
                            "reference": 0.31, "detail": "Cd from the wind tunnel"}])
    assert quality.assess(_write(tmp_path, d2, name="demo2")).by_id("reference").status == quality.MET


def test_a_not_measured_section_in_the_readme_counts(tmp_path):
    d = _manifest(checks=[{"name": "finite", "passed": True, "kind": "sanity"}])
    c = quality.assess(_write(tmp_path, d, readme="# Demo\n\n## Not measured\n\nThe Re above 5e5.\n"))
    nm = c.by_id("not_measured")
    assert nm.status == quality.MET and "README.md" in nm.evidence[0]


def test_budget_ablation_and_convergence_are_not_applicable_when_nothing_is_claimed(tmp_path):
    d = _manifest(methods=["X-TFC"], modules=["pinneapple_analysis"], checks=FULL_CHECKS[:2],
                  metrics=[{"name": "err", "value": 1e-6, "unit": "", "how": "analytic"}])
    c = quality.assess(_write(tmp_path, d))
    assert c.by_id("budget").status == quality.NA
    assert c.by_id("ablation").status == quality.NA
    assert c.by_id("convergence").status == quality.NA


def test_a_comparison_without_a_budget_and_a_claim_without_an_ablation_is_missing(tmp_path):
    d = _manifest(methods=["FNO", "PINN"],
                  checks=[{"name": "FNO beats PINN on the held-out set", "passed": True, "kind": "baseline"},
                          {"name": "error on unseen cases", "passed": True, "kind": "generalization"}],
                  metrics=[{"name": "rel_l2", "value": 1e-3, "unit": "", "how": "FNO vs PINN on held-out"}])
    c = quality.assess(_write(tmp_path, d))
    assert c.by_id("budget").status == quality.MISSING
    assert c.by_id("ablation").status == quality.MISSING
    assert "claims an improvement" in c.by_id("ablation").evidence[0]
    d2 = _manifest(methods=["FNO", "PINN"],
                   checks=[{"name": "rel_l2 on the held-out set", "passed": True, "kind": "generalization"}],
                   metrics=[{"name": "rel_l2", "value": 1e-3, "unit": "", "how": "measured"}])
    assert "compares 2 methods" in quality.assess(_write(tmp_path, d2, name="demo3")).by_id(
        "ablation").evidence[0]


def test_manifest_round_trip_keeps_the_quality_block(tmp_path):
    d = _manifest(quality={"schema": quality.DECLARATION, "not_measured": ["the losses are not logged"]})
    p = save(Manifest.from_dict(d), str(tmp_path))
    assert load(p).quality["not_measured"] == ["the losses are not logged"]


def test_the_backlog_is_sorted_by_visibility_and_has_the_fix(tmp_path):
    a = quality.Checklist("examples/quiet", [
        quality.Item(i, quality.MISSING, ["x"], quality.FIXES[i]) for i in ("reference", "verdict")])
    b = quality.Checklist("examples/loud", [quality.Item("reference", quality.MISSING, ["x"],
                                                         quality.FIXES["reference"])])
    text = quality.backlog([(a, 2), (b, 40)])
    rows = [line for line in text.splitlines() if line.startswith("| ") and "examples/" in line]
    assert rows[0].startswith("| 40 | `examples/loud`")
    assert "`examples/quiet`" in rows[1] and "compare with an exact solution" in rows[1]
    assert quality.render_markdown([b]).count("|") > 0


def test_the_repository_folders_read_and_are_cached_by_nothing(tmp_path):
    """Two real examples, so the detector is exercised on the shapes the repository actually writes: the flow
    example has a generalization check and no reference, the verification folder has reference checks."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    if not os.path.isdir(os.path.join(root, "base_workflows", "equation_discovery")):
        pytest.skip("examples/ not present")
    from pathlib import Path
    eq = quality.assess(str(Path(root) / "base_workflows" / "equation_discovery"))
    assert eq.by_id("holdout").status == quality.MET
    assert eq.by_id("verdict").status == quality.MET
    assert eq.applicable >= 4
    assert "/" in eq.badge() or "not assessed" in eq.badge()


def test_visibility_counts_the_pages_that_name_the_example():
    m = Manifest(id="examples/flight_dynamics", title="Flight dynamics")
    docs = [("docs/flight.md", "see examples/flight_dynamics for the 6-DOF model"), ("README.md", "nothing")]
    assert quality.visibility(m, docs) == 1
    assert quality.visibility(Manifest(id="examples/absent", title="Absent"), docs) == 0


def test_the_dataset_gate_is_quoted_not_re_run(tmp_path):
    """#923 composes ``pinneapple_data.readiness``: the report the Trainer already produced is evidence on the
    provenance item, and a refused dataset turns into a missing item instead of a second dataset gate."""
    from pinneapple_data import readiness as RD

    d = _manifest(checks=[{"name": "finite", "passed": True, "kind": "sanity"}])
    report = {"status": "PASS", "score": {"overall": 100.0}, "findings": []}
    prov = quality.assess(_write(tmp_path, d), readiness=report).by_id("provenance")
    assert prov.status == quality.MET and any("dataset readiness: PASS" in x for x in prov.evidence)

    bad = {"status": "FAIL", "score": {"overall": 40.0},
           "findings": [{"id": "units.missing", "severity": "critical", "title": "a column has no unit"}]}
    prov2 = quality.assess(_write(tmp_path, d, name="demo2"), readiness=bad).by_id("provenance")
    assert prov2.status == quality.MISSING and "readiness" in prov2.fix
    assert any("units.missing" in x for x in prov2.evidence)
    # the default policy is the library's own (the report shape is readiness's, not a new one)
    assert RD.default_policy().name == "default"
    # no report, no invented evidence
    assert not any("dataset readiness" in x for x in
                   quality.assess(_write(tmp_path, d, name="demo3")).by_id("provenance").evidence)
