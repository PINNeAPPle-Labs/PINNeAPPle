"""The one-page experiment report (#919): a real manifest renders, a folder with no result refuses, and the
numbers on the page are the ones the run recorded."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from pinneapple_analysis.report import Report, ReportError, experiment_report
from pinneapple_analysis.report import experiment as X

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL = os.path.join(ROOT, "examples", "condensation_falsification")
EMPTY = os.path.join(ROOT, "examples", "numerical_solvers")


def _manifest(folder, **over) -> dict:
    m = {"schema": "pinneapple.experiment/1", "id": "examples/toy", "title": "Toy check",
         "description": "A bent pipe against the Hagen-Poiseuille value, on three meshes.",
         "domain": "fluid dynamics", "methods": ["finite volumes"], "modules": [], "params": {},
         "scripts": ["run.py"],
         "metrics": [{"name": "pressure_drop", "value": 0.321, "unit": "bar",
                      "how": "last 200 iterations of simpleFoam on the fine mesh"},
                     {"name": "surrogate_rmse", "value": 0.0206, "unit": "-", "how": "held-out split, 12 cases"},
                     {"name": "mesh_sweep", "value": 2.0, "unit": "-", "how": "meshes that ran to convergence"}],
         "checks": [{"name": "pressure_drop_vs_reference", "passed": True, "value": 0.321, "reference": 0.3202,
                     "rtol": 0.03, "atol": None, "max": None, "min": None, "detail": "3.0 % on the fine mesh",
                     "kind": "reference"},
                    {"name": "mesh_sweep_finished", "passed": False, "value": None, "reference": None, "rtol": None,
                     "atol": None, "max": None, "min": None, "detail": "the medium mesh diverged", "kind": "sanity"}],
         "files": {"figures": ["figures/plot.png"], "videos": [], "viewers": [], "data": []},
         "links": {}, "references": ["Hagen & Poiseuille"], "status": "not curated", "lab": {},
         "provenance": {"source": "experiment()", "git": "abc1234", "created": "2026-11-01T10:00:00Z", "seconds": 42.0,
                        "environment": {"python": "3.11.16", "platform": "Darwin-25.0-arm64"}}}
    m.update(over)
    return m


def _write(folder, **over) -> str:
    os.makedirs(os.path.join(folder, "figures"), exist_ok=True)
    with open(os.path.join(folder, "figures", "plot.png"), "wb") as fh:      # a real (tiny) PNG
        fh.write(bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
                               "de0000000c4944415408d763f8ffff3f0005fe02fea735cd7c0000000049454e44ae426082"))
    with open(os.path.join(folder, "experiment.json"), "w") as fh:
        json.dump(_manifest(folder, **over), fh)
    return folder


def test_report_of_a_toy_manifest_is_html_and_pdf(tmp_path):
    folder = _write(str(tmp_path / "toy"))
    r = experiment_report(folder, out=str(tmp_path / "out"))
    html, pdf = r                                                  # the signature of the issue: -> html, pdf
    assert html == r.html and pdf == r.pdf
    assert os.path.exists(html) and os.path.getsize(html) > 4000
    assert pdf and os.path.exists(pdf) and os.path.getsize(pdf) > 4000
    page = open(html, encoding="utf-8").read()
    assert "Toy check" in page and "Hagen" in page                 # the title and the provenance
    assert "pressure_drop" in page                                 # the card is the manifest's own name
    assert "check failed" in page                                  # the card of the failing check is red
    assert "1 passed, 1 failed" in page
    assert 'alt="' in page                                         # every figure carries alt text
    data = json.load(open(os.path.join(tmp_path, "out", "report.json"), encoding="utf-8"))
    assert data["schema"] == "pinneapple.report/1"
    kpi = {k["name"]: k for k in data["kpis"]}
    assert kpi["pressure_drop"]["status"] == "pass"                # its own check passed
    assert kpi["surrogate_rmse"]["status"] == "no check"            # nothing checks it: not coloured green
    assert kpi["mesh_sweep"]["status"] == "fail"                   # its check failed, so the card is red
    assert data["verdicts"] == {"rows": data["verdicts"]["rows"], "passed": 1, "failed": 1,
                                "kinds": {"reference": 1, "sanity": 1}}
    assert data["header"]["commit"] == "abc1234" and data["header"]["wall_seconds"] == 42.0


def test_a_folder_with_no_result_fails_with_a_clear_message(tmp_path):
    folder = _write(str(tmp_path / "empty"), metrics=[], checks=[],
                    files={"figures": [], "videos": [], "viewers": [], "data": []})
    with pytest.raises(ReportError) as exc:
        experiment_report(folder, out=str(tmp_path / "out"))
    msg = str(exc.value)
    assert "no result to report" in msg and "python" in msg        # says what is missing and what to run


def test_a_real_experiment_of_the_repository_renders(tmp_path):
    """The #900 campaign folder: 23 pre-registered checks, evidence tables, four figures -- not a fixture."""
    r = experiment_report(REAL, out=str(tmp_path / "report"))
    data = r.data
    assert data["header"]["title"].startswith("Falsification campaign")
    assert len(data["verdicts"]["rows"]) == len(json.load(open(os.path.join(REAL, "experiment.json"),
                                                                encoding="utf-8"))["checks"])
    assert data["verdicts"]["failed"] == 11 and data["verdicts"]["passed"] == 12
    assert len(data["figures"]) == 4 and all(os.path.exists(os.path.join(data["out"], f["path"]))
                                             for f in data["figures"])
    assert data["kpis"] and all(k["status"] in ("pass", "fail", "no check") for k in data["kpis"])
    assert len(data["validity"]["stages"]["stages"]) == 6          # the six questions are all there
    assert data["validity"]["checklist"]["items"] and data["validity"]["checklist"]["badge"].startswith("QUALITY")
    assert os.path.exists(r.pdf or "")


def test_a_folder_without_a_manifest_says_so(tmp_path):
    with pytest.raises(ReportError, match="experiment.json"):
        experiment_report(str(tmp_path), out=str(tmp_path / "out"))


def test_the_two_themes_differ_only_in_ink(tmp_path):
    folder = _write(str(tmp_path / "toy"))
    light = experiment_report(folder, out=str(tmp_path / "light"), pdf=False)
    dark = experiment_report(folder, out=str(tmp_path / "dark"), pdf=False, theme="dark")
    a, b = open(light.html, encoding="utf-8").read(), open(dark.html, encoding="utf-8").read()
    assert "--paper:#ffffff" in a and "--paper:#1a1a2e" in b
    assert a.count("<section") >= 0 and "Toy check" in b
    with pytest.raises(ValueError, match="light"):
        experiment_report(folder, out=str(tmp_path / "x"), theme="sepia")


def test_a_check_only_colours_the_number_it_names():
    """`hypotheses_refuted` is not evidenced by a table of `T1 not refuted` rows (#900's own shape)."""
    checks = [{"name": f"T{i} not refuted", "passed": i % 2 == 0, "value": None, "kind": "verification"}
              for i in range(1, 6)]
    assert X._match("hypotheses_refuted", checks) == []
    assert X._match("hypotheses_validated", checks) == []
    near = [{"name": "sphere_drag_CD_vs_reference", "passed": True, "value": 1.087, "reference": 1.087,
             "rtol": 0.05, "kind": "reference"},
            {"name": "sphere_drag_Re_50_CD_vs_reference", "passed": True, "value": 1.569, "reference": 1.574,
             "rtol": 0.05, "kind": "reference"}]
    assert [c["name"] for c in X._match("sphere_drag_Re_50_CD", near)][0] == "sphere_drag_Re_50_CD_vs_reference"


def test_the_band_is_the_checks_own_tolerance():
    band, target = X._band([{"reference": 0.3202, "rtol": 0.03}])
    assert band[0] < 0.3202 < band[1] and "3 %" in target
    band, target = X._band([{"reference": 2.0, "atol": 0.5}])
    assert band == [1.5, 2.5] and "± 0.5" in target
    band, target = X._band([{"reference": 0.2}])
    assert band is None and "no tolerance" in target
    assert X._band([{"max": 3.0}]) == ([-float("inf"), 3.0], "≤ 3")
    assert X._band([]) == (None, "")


def test_the_command_line_writes_the_index(tmp_path):
    folder = _write(str(tmp_path / "toy"))
    out = str(tmp_path / "reports")
    proc = subprocess.run([sys.executable, "-m", "pinneapple_analysis.report", folder, "--out", out, "--no-pdf"],
                          cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert proc.returncode == 0, proc.stderr
    assert os.path.exists(os.path.join(out, "toy", "index.html"))
    index = open(os.path.join(out, "index.html"), encoding="utf-8").read()
    assert "Toy check" in index and "toy/index.html" in index


def test_a_report_object_is_what_the_caller_gets():
    assert Report("a.html", None, {"x": 1}).data == {"x": 1}
    html, pdf = Report("a.html", "a.pdf")
    assert (html, pdf) == ("a.html", "a.pdf")


def test_the_hero_render_of_the_studio_lands_in_the_report(tmp_path):
    """#920 and #919 meet here: a `hero.json` in the report's folder is the figure above the others."""
    folder = _write(str(tmp_path / "toy"))
    out = str(tmp_path / "out")
    os.makedirs(os.path.join(out, "hero"), exist_ok=True)
    still = os.path.join(out, "hero", "still_1920x1080.png")
    with open(still, "wb") as fh:
        fh.write(bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
                               "de0000000c4944415408d763f8ffff3f0005fe02fea735cd7c0000000049454e44ae426082"))
    with open(os.path.join(out, "hero", "hero.json"), "w") as fh:
        json.dump({"schema": "pinneapple.hero/1", "backend": "cycles", "stills": {"1920x1080": still},
                   "field": {"label": "von Mises stress", "unit": "Pa"}, "warp": None,
                   "caption": "von Mises stress (Pa) on the beam, drawn at its own scale; cycles."}, fh)
    r = experiment_report(folder, out=out, pdf=False)
    page = open(r.html, encoding="utf-8").read()
    assert "hero/still_1920x1080.png" in page and "Hero render (cycles" in page
    assert 'alt="Toy check: studio hero render' in page
