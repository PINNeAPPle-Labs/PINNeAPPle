"""pinneapple_studio: the inventory, the classification of every public symbol as a tool, forms, status from tests,
search in Portuguese and English, requirements, the gallery, the job runner and the JSON API.

Most tests build a small repository in a temporary folder (``PINNEAPPLE_ROOT``); the ones marked "real" read this
checkout."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from pinneapple_studio import classify as C
from pinneapple_studio import requirements as R
from pinneapple_studio import runner
from pinneapple_studio.search import Index, expand, tokens

ROOT = Path(__file__).resolve().parents[1]


def _w(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A tiny checkout: one library package with each kind of tool, tests, an example, prose and a glossary."""
    r = tmp_path / "repo"
    _w(r / "pyproject.toml", """
        [project]
        name = "demo"
        version = "0.0.1"
        [project.optional-dependencies]
        cad = ["cadquery>=2"]
    """)
    _w(r / "pinneapple" / "__init__.py", '__version__ = "0.0.1"\n_LAZY_ATTRS = {"fin_heat": ("pinneapple_demo.heat", '
                                         '"fin_heat")}\n')
    _w(r / "pinneapple_catalog" / "__init__.py", "")
    _w(r / "pinneapple_catalog" / "methods.py", "def _m(*a):\n    return a\n")
    _w(r / "pinneapple_catalog" / "method_status.json", json.dumps({"methods": {}}))
    _w(r / "pinneapple_demo" / "__init__.py", '"""Demo package."""\n')
    _w(r / "pinneapple_demo" / "heat.py", '''
        """Heat transfer helpers."""
        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Literal

        import numpy as np


        @dataclass
        class FinDesign:
            length: float
            thickness: float = 0.002


        def fin_heat(h: float, k: float, design: FinDesign, mode: Literal["tip", "adiabatic"] = "tip") -> float:
            """Heat through a fin.

            h: convection coefficient [W/m2K]
            k: conductivity of the fin [W/mK]
            """
            return h * k * design.length


        def mean_temperature(T: np.ndarray) -> float:
            """Mean of a temperature field [K]."""
            return float(np.mean(T))


        def apply_model(model, x: float):
            return model(x)


        class FinError(ValueError):
            pass


        class FinStudy:
            def __init__(self, n: int = 3):
                self.n = n

            def run(self) -> list[float]:
                return [float(i) for i in range(self.n)]
    ''')
    _w(r / "pinneapple_demo" / "_internal.py", "def helper(x):\n    return x\n")
    _w(r / "pinneapple_demo" / "cli.py", "def main(argv=None):\n    return 0\n")
    _w(r / "tests" / "test_heat.py", '''
        from pinneapple_demo.heat import FinDesign, fin_heat, mean_temperature


        def test_fin_against_reference():
            reference = 2.0
            assert fin_heat(1.0, 2.0, FinDesign(1.0)) == reference


        def test_mean_runs():
            assert mean_temperature([1.0, 3.0]) == 2.0
    ''')
    _w(r / "examples" / "fins" / "run_fins.py", '''
        import pinneapple as pp
        import cadquery
        from pinneapple_demo.heat import FinStudy

        print(pp.fin_heat, FinStudy().run(), cadquery)
    ''')
    _w(r / "examples" / "fins" / "experiment.json", json.dumps({"title": "Fins", "description": "A fin study",
                                                                 "domain": "thermal", "scripts": ["run_fins.py"],
                                                                 "metrics": [], "params": {}}))
    _w(r / "docs" / "pt" / "_mapa" / "prosa" / "pinneapple_demo.json", json.dumps({
        "pacote": "pinneapple_demo", "objetivo": "Pacote de demonstração.", "limitacoes": ["Só exemplo."],
        "modulos": {"pinneapple_demo.heat": "Ajudantes de transferência de calor."},
        "simbolos": {"pinneapple_demo.heat:fin_heat": "Calor que passa por uma aleta.",
                     "pinneapple_demo.heat:gone": "Símbolo que não existe mais."}}))
    _w(r / "docs" / "pt" / "_mapa" / "glossario.json", json.dumps([{"termo": "Aleta (fin)", "definicao": "x"}]))
    for d in ("apps", "scripts", "benchmarks", ".github/workflows"):
        (r / d).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PINNEAPPLE_ROOT", str(r))
    monkeypatch.setenv("PINNEAPPLE_STUDIO_CACHE", str(tmp_path / "cache"))
    monkeypatch.syspath_prepend(str(r))
    for m in [m for m in sys.modules if m.startswith("pinneapple_demo")]:
        del sys.modules[m]
    return r


@pytest.fixture()
def cat(repo):
    from pinneapple_studio import catalog
    return catalog.build(repo)


def test_every_symbol_gets_a_category_with_the_reason(cat):
    t = cat["tools"]
    assert t["pinneapple_demo.heat:fin_heat"]["category"] == "executable"          # FinDesign is a nested form
    assert t["pinneapple_demo.heat:mean_temperature"]["category"] == "executable"  # an array is a list of numbers
    assert t["pinneapple_demo.heat:FinStudy"]["category"] == "executable"
    assert t["pinneapple_demo.heat:apply_model"]["category"] == "composite"
    assert "no type annotation" in t["pinneapple_demo.heat:apply_model"]["category_reason"]
    assert t["pinneapple_demo.heat:FinError"]["category"] == "reference"
    assert t["pinneapple_demo.heat:FinDesign"]["category"] == "reference"          # a plain data record
    assert t["pinneapple_demo._internal:helper"]["category"] == "reference"
    assert t["pinneapple_demo.cli:main"]["category"] == "reference"
    assert all(x["category"] in C.CATEGORIES and x["category_reason"] for x in t.values())


def test_form_has_types_units_defaults_and_nested_objects(cat):
    f = cat["tools"]["pinneapple_demo.heat:fin_heat"]["form"]
    assert f["required"] == ["h", "k", "design"]
    assert f["properties"]["h"]["x-unit"] == "W/m2K" and f["properties"]["h"]["type"] == "number"
    assert f["properties"]["mode"] == {"enum": ["tip", "adiabatic"], "default": "tip",
                                       "x-annotation": "Literal['tip', 'adiabatic']"}
    d = f["properties"]["design"]
    assert d["x-class"] == "pinneapple_demo.heat:FinDesign" and d["required"] == ["length"]
    assert C.valid_form(f) == []


def test_status_is_earned_from_tests_and_examples(cat):
    t = cat["tools"]
    assert t["pinneapple_demo.heat:fin_heat"]["status"] == "verified"         # compared with a reference
    assert t["pinneapple_demo.heat:mean_temperature"]["status"] == "tested"
    assert t["pinneapple_demo.heat:FinStudy"]["status"] == "discovered"       # only an example uses it
    assert t["pinneapple_demo.heat:FinStudy"]["examples"] == ["fins"]
    assert t["pinneapple_demo.heat:fin_heat"]["examples"] == ["fins"]          # through pp.fin_heat


def test_prose_is_shown_whole_and_stale_prose_is_listed(cat):
    assert cat["tools"]["pinneapple_demo.heat:fin_heat"]["pt"] == "Calor que passa por uma aleta."
    assert cat["tools"]["pinneapple_demo.heat:mean_temperature"]["pt"] == ""    # no prose: nothing is made up
    assert cat["packages"]["pinneapple_demo"]["pt"]["limitacoes"] == ["Só exemplo."]
    assert cat["prose_unmatched"]["symbols"] == ["pinneapple_demo.heat:gone"]


def test_search_in_portuguese_and_english(cat):
    idx = Index(cat)
    assert idx.search("calor na aleta")[0]["id"] == "pinneapple_demo.heat:fin_heat"
    assert idx.search("fin heat")[0]["id"] == "pinneapple_demo.heat:fin_heat"
    assert {h["id"] for h in idx.search("", category="reference")} >= {"pinneapple_demo.heat:FinError"}
    assert "thermal" in idx.facets()["area"]
    assert "heat" in expand(tokens("calor"))


def test_requirements_name_the_fix(repo):
    rows = R.check(R.requirements([repo / "examples" / "fins" / "run_fins.py"], repo))
    cad = next(r for r in rows if r["name"] == "cadquery")
    assert cad["fix"] == 'pip install -e ".[cad]"'
    assert not any(r["name"] in ("pinneapple", "pinneapple_demo") for r in rows)
    src = "import subprocess\nsubprocess.run(['simpleFoam'])\nimport os\nos.environ['MY_API_KEY']\n"
    (repo / "x.py").write_text(src)
    kinds = {r["kind"]: r for r in R.requirements([repo / "x.py"], repo)}
    assert kinds["program"]["name"] == "OpenFOAM" and kinds["credential"]["name"] == "MY_API_KEY"


def test_gallery_blocks_with_the_reason_and_the_command(repo, monkeypatch):
    from pinneapple_studio import gallery
    monkeypatch.setattr(gallery, "_flow_rows", lambda: ({}, [], {}))
    monkeypatch.setattr(R.importlib.util, "find_spec", lambda name: None if name == "cadquery" else object())
    e = next(x for x in gallery.entries(repo) if x["key"] == "fins")
    assert e["state"] == "blocked" and e["title"] == "Fins"
    assert e["blocked_by"][0]["fix"] == 'pip install -e ".[cad]"'


def test_job_environment_has_no_secrets(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@h/db")
    monkeypatch.setenv("SOME_RANDOM_VAR", "y")
    env = runner.clean_env()
    assert "GITHUB_TOKEN" not in env and "DATABASE_URL" not in env and "SOME_RANDOM_VAR" not in env
    assert env["MPLBACKEND"] == "Agg" and "PATH" in env


def test_form_values_become_call_arguments(cat):
    import numpy as np
    form = cat["tools"]["pinneapple_demo.heat:fin_heat"]["form"]
    args = runner.build_inputs(form, {"h": 10.0, "k": 2.0, "design": {"length": 0.5}})
    assert type(args["design"]).__name__ == "FinDesign" and args["design"].length == 0.5
    form2 = cat["tools"]["pinneapple_demo.heat:mean_temperature"]["form"]
    assert isinstance(runner.build_inputs(form2, {"T": [1, 2]})["T"], np.ndarray)


def test_api_lists_shows_compares_and_runs_a_tool(repo, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pinneapple_core.service.jobs import JobQueue
    from pinneapple_studio import api, gallery
    monkeypatch.setattr(gallery, "_flow_rows", lambda: ({}, [], {}))
    q = JobQueue(tmp_path / "jobs.sqlite", app="studio-test", executor="thread")
    q.register("tool", runner.tool_job)
    app = FastAPI()
    app.include_router(api.router(q), prefix="/api/studio")
    c = TestClient(app)
    s = c.get("/api/studio/summary").json()
    assert s["counts"]["category"]["executable"] >= 3 and s["gallery"]["by_kind"]["example"] == 1
    hits = c.get("/api/studio/tools", params={"q": "aleta", "category": "executable"}).json()
    assert hits["items"][0]["id"] == "pinneapple_demo.heat:fin_heat"
    card = c.get("/api/studio/tools/pinneapple_demo.heat:fin_heat").json()
    assert card["package_pt"]["objetivo"] == "Pacote de demonstração."
    cmp_ = c.get("/api/studio/compare", params={"ids": "pinneapple_demo.heat:fin_heat,pinneapple_demo.heat:FinStudy"})
    assert [x["name"] for x in cmp_.json()] == ["fin_heat", "FinStudy"]
    assert c.post("/api/studio/jobs", json={"kind": "tool", "params": {"tool": "pinneapple_demo.heat:apply_model"}}
                  ).status_code == 400                                           # composite: not run alone
    job = c.post("/api/studio/jobs", json={"kind": "tool", "params": {
        "tool": "pinneapple_demo.heat:fin_heat", "inputs": {"h": 10.0, "k": 2.0, "design": {"length": 0.5}}}})
    assert job.status_code == 202
    res = q.wait(job.json()["job_id"], timeout=120)
    assert res["status"] == "done", res
    assert res["result"]["result"] == 10.0 and res["result"]["evidence"]["outcome"] == "not_run"
    assert any(f["path"] == "result.json" for f in res["result"]["evidence"]["files"])
    blocked = c.post("/api/studio/jobs", json={"kind": "gallery", "params": {"entry": "example:fins"}})
    assert blocked.status_code in (202, 409)


# ------------------------------------------------------------------------------------------------- real
def test_real_every_public_symbol_has_a_category():
    """No symbol of the library is left without a category (the classification never skips one)."""
    from pinneapple_studio import inventory
    inv = inventory.build(ROOT)
    n = 0
    for rec in inv["modules"].values():
        for kind, key in (("function", "functions"), ("class", "classes")):
            for sym in rec[key]:
                cat, why = C.classify(sym, rec, kind)
                assert cat in C.CATEGORIES and why, (rec["module"], sym["name"])
                n += 1
    assert n == inv["totals"]["public_functions"] + inv["totals"]["public_classes"]


@pytest.mark.skipif(not (ROOT / "docs" / "pt" / "_mapa" / "prosa").is_dir(), reason="no Portuguese map")
def test_real_prose_loads_for_every_package():
    from pinneapple_studio import prose
    p = prose.load(ROOT)
    assert len(p.packages) >= 30 and len(p.symbols) > 1000 and p.glossary


def test_exclusion_pattern_comes_from_outside_the_repository(monkeypatch, tmp_path):
    from pinneapple_studio import inventory
    monkeypatch.delenv("PINNEAPPLE_MAPA_EXCLUDE", raising=False)
    assert inventory.exclude_pattern(tmp_path).search("anything") is None          # nothing configured
    monkeypatch.setenv("PINNEAPPLE_MAPA_EXCLUDE", r"acme|\bxyz\b")
    assert inventory.exclude_pattern(tmp_path).search("the ACME case") and not inventory.exclude_pattern(
        tmp_path).search("xyzzy")
    monkeypatch.delenv("PINNEAPPLE_MAPA_EXCLUDE")
    f = tmp_path / "docs" / "pt" / "_mapa" / "exclude.local.txt"
    f.parent.mkdir(parents=True)
    f.write_text("# local only\nfoo\n")
    assert inventory.exclude_pattern(tmp_path).search("a Foo line")


def test_real_cli_dispatch(capsys):
    from pinneapple_check.cli import main
    with pytest.raises(SystemExit) as e:                                         # argparse exits after --help
        main(["studio", "--help"])
    assert e.value.code == 0 and "run-tool" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------- coverage gate
def test_coverage_gate_ratchets(cat, repo, monkeypatch, tmp_path):
    from pinneapple_studio import coverage as COV
    from pinneapple_studio import smoke
    monkeypatch.setattr(COV, "BASELINE", tmp_path / "baseline.txt")
    monkeypatch.setattr(COV, "REASONS", tmp_path / "reasons.json")
    monkeypatch.setattr(smoke, "INPUTS", tmp_path / "inputs.json")
    gal = []
    rows = {r["id"]: r for r in COV.tool_rows(cat)}
    assert rows["pinneapple_demo.heat:FinStudy"]["state"] == "missing"            # runs with defaults, never ran
    assert "no inputs recorded" in rows["pinneapple_demo.heat:fin_heat"]["detail"]
    assert any("FinStudy" in p for p in COV.problems(cat, gal))
    # the baseline takes today's gaps; afterwards nothing new can enter
    COV.write_baseline(cat, gal)
    assert COV.problems(cat, gal) == []
    # a recorded run of the current code leaves the baseline: the stale line becomes a problem
    t = cat["tools"]["pinneapple_demo.heat:FinStudy"]
    (repo / "data" / "studio" / "tool_runs").mkdir(parents=True)
    (repo / "data" / "studio" / "tool_runs" / "pinneapple_demo.json").write_text(json.dumps(
        {t["id"]: {"ok": True, "code_hash": t["code_hash"], "inputs": {}, "returned": "list", "seconds": 0.1}}))
    assert COV.tool_rows(cat)[[r["id"] for r in COV.tool_rows(cat)].index(t["id"])]["state"] == "ran"
    assert any("now ran" in p for p in COV.problems(cat, gal))
    # a declared reason counts; a side-effect name is a reason by rule
    (tmp_path / "reasons.json").write_text(json.dumps({"pinneapple_demo.heat": "needs a wind tunnel"}))
    assert {r["state"] for r in COV.tool_rows(cat) if r["package"] == "pinneapple_demo"} <= {"ran", "reason"}
    assert COV.declared_reason({"id": "m:download_data", "module": "m", "name": "download_data"}, {})
    md = COV.markdown(cat, gal)
    assert "executable" in md and "| tool | state | detail |" in md
    row = COV.record_history(cat, gal, repo)
    assert row["baseline"] == len(COV.baseline()) and COV.history(repo)[-1] == row
    assert "The baseline over time" in COV.markdown(cat, gal)


def test_smoke_run_records_a_tool(cat, repo, monkeypatch):
    from pinneapple_studio import smoke
    stats = smoke.record(cat, ids=["pinneapple_demo.heat:FinStudy"], log=lambda *a: None)
    assert stats == {"batches": 1, "tools": 1, "ok": 1, "failed": 0, "writes": 0}
    rec = smoke.read_runs(repo)["pinneapple_demo.heat:FinStudy"]
    assert rec["ok"] and rec["returned"] == "list" and rec["code_hash"] == cat["tools"]["pinneapple_demo.heat:FinStudy"]["code_hash"]


def test_example_rows_say_why(repo, monkeypatch):
    from pinneapple_studio import coverage as COV
    from pinneapple_studio import gallery
    monkeypatch.setattr(gallery, "_flow_rows", lambda: ({}, [], {}))
    monkeypatch.setattr(R.importlib.util, "find_spec", lambda name: None if name == "cadquery" else object())
    rows = COV.example_rows(gallery.entries(repo), repo)
    assert rows == [{"id": "example:fins", "title": "Fins", "state": "needs",
                     "how": 'cadquery: `pip install -e ".[cad]"`'}]


def test_smoke_marks_a_tool_that_writes_into_the_checkout(tmp_path, monkeypatch):
    """A tool run in a disposable git checkout that writes there is recorded with the files, and the copy is
    restored for the next tool."""
    import subprocess as sp

    from pinneapple_studio import smoke
    r = tmp_path / "co"
    _w(r / "pkg_w.py", "def touch(n: int = 1):\n    open('MARK.txt', 'w').write('x')\n    return n\n")
    sp.run(["git", "init", "-q", str(r)], check=True)
    sp.run(["git", "-C", str(r), "add", "."], check=True)
    sp.run(["git", "-C", str(r), "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "x"], check=True)
    monkeypatch.chdir(r)                                     # the tool writes in its working folder: the copy
    monkeypatch.setenv("PINNEAPPLE_SMOKE_COPY", str(r))
    monkeypatch.syspath_prepend(str(r))
    spec = tmp_path / "todo.json"
    spec.write_text(json.dumps([[{"id": "pkg_w:touch", "kind": "function", "name": "touch", "module": "pkg_w",
                                  "methods": [], "form": {"properties": {}}}, {}]]))
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        smoke.child([str(spec)])
    rec = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert rec["ok"] and rec["changes_files"] == ["MARK.txt"]
    assert not (r / "MARK.txt").exists()                     # restored


def test_inputs_are_harvested_from_calls_with_literals(cat, repo):
    from pinneapple_studio import inputs
    found = inputs.harvest(cat, repo, hub=False)
    assert found["pinneapple_demo.heat:mean_temperature"] == {"inputs": {"T": [1.0, 3.0]},
                                                              "source": "tests/test_heat.py:10"}
    assert "pinneapple_demo.heat:fin_heat" not in found        # FinDesign(1.0) is not a literal
    src = "import numpy as np\nimport pytest\nfrom pinneapple_demo.heat import mean_temperature as mt\n" \
          "def test_x():\n    t = np.linspace(0, 1, 3)\n    mt(t)\n    with pytest.raises(ValueError):\n        mt(None)\n"
    calls = inputs.calls_in(src)
    assert calls == [("pinneapple_demo.heat:mean_temperature", 6, [[0.0, 0.5, 1.0]], {})]


def test_gallery_shows_the_commands_of_the_readme(tmp_path):
    from pinneapple_studio.gallery import run_commands
    _w(tmp_path / "README.md", "# R\n\n```bash\npython examples/r/run.py --part elm\n```\n\n```python\nimport x\n```\n")
    assert run_commands(tmp_path, tmp_path) == ["python examples/r/run.py --part elm"]


def test_catalog_builds_without_the_git_program(repo, monkeypatch, tmp_path):
    """A source tarball in an image without git (the Engineering API's build) still gives a catalog."""
    from pinneapple_studio import catalog
    empty = tmp_path / "no-git-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    cat = catalog.build(repo)
    assert cat["tools"]


def test_outputs_known_answers():
    """Outputs come from the code: the docstring's Returns (numpy or Google, unit in brackets), a NamedTuple or
    dataclass annotation, the module's X_OUTPUTS, or nothing."""
    from pinneapple_studio.outputs import outputs
    nothing = lambda name: None  # noqa: E731
    numpy_doc = "Time constant.\n\nReturns\n-------\ntau : float [s]\n    Time constant of the circuit.\n"
    assert outputs({"name": "f", "doc": numpy_doc, "returns": "float"}, None, nothing) == (
        [{"name": "tau", "type": "float", "unit": "s", "description": "Time constant of the circuit."}], "docstring")
    assert outputs({"name": "f", "doc": "Nothing said.", "returns": ""}, None, nothing) == ([], "none")
    google = "Period.\n\nArgs:\n    x (float): in.\n\nReturns:\n    float [s]: the period.\n"
    assert outputs({"name": "f", "doc": google, "returns": ""}, None, nothing)[0] == [
        {"name": "", "type": "float", "unit": "s", "description": "the period."}]
    two = ("Returns\n-------\nu : numpy.ndarray [m/s]\n    velocity in [0, 1]\np : list[float] [Pa]\n    pressure\n\n"
           "Example::\n\n    u, p = f()\n")
    assert [(o["name"], o["type"], o["unit"]) for o in outputs({"name": "f", "doc": two}, None, nothing)[0]] == [
        ("u", "numpy.ndarray", "m/s"), ("p", "list[float]", "Pa")]
    interval = "Returns\n-------\ntorch.Tensor\n    Scaled to the [0, 1] interval.\n"
    assert outputs({"name": "f", "doc": interval}, None, nothing)[0][0]["unit"] == ""
    res = {"name": "Res", "bases": ["NamedTuple"], "decorators": [], "doc": "Attributes\n----------\ntau : float [s]\n",
           "params": [{"name": "tau", "ann": "float"}, {"name": "gain", "ann": "float"}]}
    rows, src = outputs({"name": "f", "doc": "", "returns": "Res"}, None, lambda n: res if n == "Res" else None)
    assert src == "annotation" and [(o["name"], o["unit"]) for o in rows] == [("tau", "s"), ("gain", "")]
    assert outputs({"name": "f", "doc": "", "returns": "None"}, None, nothing) == ([], "annotation")
    assert outputs({"name": "f", "doc": numpy_doc}, {"f": [{"name": "q", "unit": "W"}]}, nothing) == (
        [{"name": "q", "type": "", "unit": "W", "description": ""}], "declared")


def test_every_tool_has_outputs_and_their_source(cat):
    from pinneapple_studio.outputs import SOURCES
    assert all(t["outputs_source"] in SOURCES and isinstance(t["outputs"], list) for t in cat["tools"].values())


def test_outputs_helpers_on_literals():
    """The section reader, the record-class test and the class finder, called the way a form would."""
    from pinneapple_studio.catalog import make_class_finder
    from pinneapple_studio.outputs import from_docstring, is_record_class
    assert from_docstring("Returns\n-------\ntau : float [s]\n    time constant\n") == [
        {"name": "tau", "type": "float", "unit": "s", "description": "time constant"}]
    assert is_record_class({"name": "Res", "bases": ["NamedTuple"], "decorators": []})
    assert not is_record_class({"name": "Solver", "bases": ["object"], "decorators": []})
    finder = make_class_finder({"modules": {"m": {"module": "m", "classes": [{"name": "Res"}], "imports": []}}})
    assert finder({"module": "m", "classes": [], "imports": ["m:Res"]})("Res") == {"name": "Res"}
