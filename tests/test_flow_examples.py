"""examples/* as flows (pinneapple_flow.examples): every example folder has a flow or a stated reason, every example
flow (and its quick configuration) validates, they are listed in Flow Studio's Examples grouped by domain, and the
quick variants of a representative subset run and reproduce the example's committed numbers."""
import importlib.util
import json
import shutil

import pytest

import pinneapple_flow as pf
from pinneapple_flow import examples as EX
from pinneapple_flow.schema import as_flow, check_flow


def test_every_example_folder_has_a_flow_or_a_reason():
    assert EX.problems() == []


def test_not_examples_are_left_out():
    folders = EX.example_folders()
    assert "flight_dynamics" in folders
    assert not [f for f in folders if f.startswith("_") or f == "scratch" or f.startswith("results")]


@pytest.mark.parametrize("name", [f["name"] for f in EX.flows()])
def test_example_flow_and_its_quick_configuration_validate(name):
    doc = json.loads(EX.flow_path(name).read_text())
    assert doc["meta"]["example"] in EX.example_folders()
    check_flow(as_flow(doc))
    q = EX.with_quick(doc)
    known = {n["id"] for n in doc["nodes"]}
    assert set(doc["meta"].get("quick") or {}) <= known, "quick overrides name nodes of the flow"
    check_flow(as_flow(q))


def test_examples_are_listed_in_the_studio_grouped_by_domain(tmp_path):
    from pinneapple_flow.service import FlowService
    svc = FlowService(db_url=f"sqlite:///{tmp_path / 'flows.db'}", workdir=tmp_path / "work")
    lst = svc.examples()
    ex = [x for x in lst if x["name"].startswith("examples/")]
    assert {x["name"].split("/", 1)[1] for x in ex} == {f["name"] for f in EX.flows()}
    assert all(x["group"] for x in lst)
    assert svc.example(ex[0]["name"])["meta"]["example"]
    with pytest.raises(KeyError):
        svc.example("examples/../../etc/passwd")


def test_inventory_doc_is_current():
    doc = EX.ROOT / "docs" / "flow_studio" / "EXAMPLES_AS_FLOWS.md"
    assert doc.is_file(), "python -m pinneapple_flow example-flows doc"
    assert doc.read_text() == EX.markdown(), "regenerate: python -m pinneapple_flow example-flows doc"


def test_compare_node_reads_results_files_and_manifest_metrics():
    from pinneapple_flow.nodes_ext.examples import example_dir, reference_value
    d = example_dir("flight_dynamics")
    assert reference_value(d, "manifest:distance_35ft_a320") == pytest.approx(1691.2169164612756)
    assert reference_value(d, "results/turbulence.json:pratt[aircraft=a320].dn_sim") > 0
    assert reference_value(d, "results/verification.json:conservation.0.02.energy_rel_error") > 0
    from pinneapple_core.workflows import StepFailed
    with pytest.raises(StepFailed):
        reference_value(d, "../../pyproject.toml:x")


def _needs(*mods):
    for m in mods:
        if importlib.util.find_spec(m) is None:
            pytest.skip(f"needs {m}")


# quick variants of a representative subset (the flight study: trim and stability in full, the take-off quick)
@pytest.mark.parametrize("name, quick", [("flight_dynamics/02_trim_and_stability", False),
                                         ("flight_dynamics/03_takeoff", True)])
def test_quick_variants_reproduce_the_committed_numbers(name, quick, tmp_path, monkeypatch):
    _needs("pinneapple_simulation.flight")
    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    rec = EX.run_example(name, quick=quick, workdir=tmp_path / "work", record=False)
    assert rec["status"] == "done", rec["nodes"]
    assert rec["comparisons"] and rec["matched"] == len(rec["comparisons"]), \
        [c for c in rec["comparisons"] if c["outcome"] != "match"]
    assert "Lab item manifest/flow/" in rec["manifest"]


def test_render_nodes_say_why_they_do_not_run(tmp_path):
    spec = pf.REGISTRY["render.blender_clip"]
    ok, why = spec.available()
    if importlib.util.find_spec("bpy") is None:
        assert not ok and "bpy" in why
    ok, why = pf.REGISTRY["render.clip_grid"].available()
    assert ok == (shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None)
