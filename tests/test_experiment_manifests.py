"""Experiment manifests: the format and its validation, capture by ``pinneapple_lab.experiment(...)`` (manifest, Lab
item, grade), the Lab runner / workflow / solve / compare / Flow hooks, discovery with no code change,
scaffold -> run -> curated and on the site, and the CI gate catching a missing manifest."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from pinneapple_lab import LabStore, experiment, run
from pinneapple_lab import manifest as M

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _fig(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- format
def test_validate_catches_what_the_site_cannot_show(tmp_path):
    m = M.Manifest(id="x", title="X")
    assert M.validate(m, str(tmp_path)) == []
    m.metric("Cl", float("nan"), "", "OpenFOAM")
    m.metrics.append({"name": "Cd", "value": None, "unit": "", "how": "OpenFOAM"})
    m.metrics.append({"name": "St", "value": 0.2, "unit": ""})
    m.files["figures"].append("missing.png")
    errors = M.validate(m, str(tmp_path))
    assert any("'Cl' has no measured value" in e for e in errors) and any("'Cd' has no measured" in e for e in errors)
    assert any("'St' does not say how" in e for e in errors) and any("missing.png does not exist" in e for e in errors)
    p = M.save(M.Manifest(id="y", title="Y"), str(tmp_path))
    assert M.load(str(tmp_path)).to_dict() == json.load(open(p))
    bad = json.load(open(p)) | {"schema": "other/0"}
    json.dump(bad, open(p, "w"))
    with pytest.raises(ValueError):
        M.load(p)


# ---------------------------------------------------------------------------------------------- capture
def test_experiment_context_writes_a_manifest_and_curates(tmp_path):
    lab = str(tmp_path / "lab")
    out = tmp_path / "study"
    (out / "results").mkdir(parents=True)
    with experiment("demo_study", title="Demo study", domain="numerical methods", methods=["FEM"],
                    params={"n": 8}, out_dir=str(out), lab=lab) as ex:
        ex.log_metric("max_error", 1e-4, unit="", how="max |u - exact| on 8 points")
        ex.check("u_vs_exact", value=1e-4, max=1e-3, kind="reference", label="u vs exact")
        ex.check("energy_balance", True, kind="physics")
        _fig(str(out / "results" / "u.png"))                      # captured by the savefig hook
    m = M.load(str(out))
    assert M.validate(m, str(out)) == [] and m.files["figures"] == ["results/u.png"]
    assert m.metrics[0]["value"] == 1e-4 and m.provenance["source"] == "experiment()" and m.provenance["seconds"] >= 0
    cur = json.load(open(os.path.join(lab, "curation.json")))
    item = cur["items"][ex.item]
    assert ex.item == "manifest/demo_study" and item["tier"] in ("A", "B") and item["next_steps"]
    assert any(f.endswith("u.png") for f in os.listdir(os.path.join(lab, "runs", "manifest", item["best_run"],
                                                                     "figures")))


def test_a_failure_inside_the_block_is_recorded_and_graded_f(tmp_path):
    lab = str(tmp_path / "lab")
    with pytest.raises(ZeroDivisionError):
        with experiment("broken", out_dir=str(tmp_path / "b"), lab=lab) as ex:
            ex.log_metric("x", 1.0, how="set")
            _ = 1 / 0
    assert "ZeroDivisionError" in M.load(str(tmp_path / "b")).provenance["error"]
    assert json.load(open(os.path.join(lab, "curation.json")))["items"]["manifest/broken"]["tier"] == "F"


def test_decorator_form_and_manifest_only(tmp_path):
    @experiment("deco", out_dir=str(tmp_path), register=False)
    def main(ex):
        ex.log_metric("y", 2.0, unit="m", how="computed")
        return 7
    assert main() == 7 and M.load(str(tmp_path)).metrics[0]["unit"] == "m"
    assert not os.path.exists(tmp_path / "lab")


# ---------------------------------------------------------------------------------------------- the other hooks
def test_lab_runner_and_workflow_runs_write_manifests(tmp_path):
    r = run("oscillator", {}, root=str(tmp_path / "lab"))
    m = M.load(os.path.join(r.dir, M.FILENAME))
    assert m.provenance["source"] == "lab-run" and m.lab["experiment"] == "oscillator" and m.metrics
    assert M.validate(m, r.dir) == []
    from pinneapple_core.workflows import Step, Workflow, WorkflowRunner
    wr = WorkflowRunner(tmp_path / "wf")
    wf = Workflow("w", [Step("a", "python", {"function": f"{__name__}:_wf_step", "kwargs": {"x": 3}})])
    res = wr.run(wf)
    wm = M.load(str(tmp_path / "wf" / "runs" / res["id"] / M.FILENAME))
    assert wm.provenance["source"] == "workflow" and {"name": "a.y", "value": 6, "unit": "",
                                                       "how": "output of workflow step a"} in wm.metrics


def _wf_step(kw, ctx):
    return {"y": 2 * kw["x"]}


def test_solve_compare_and_flow_hooks(tmp_path):
    class _Cmp:
        def to_dict(self):
            return {"problem": "poisson", "reference": "exact", "n_points": 64,
                    "rows": [{"method": "fem", "status": "ok", "metrics": {"relative_l2": {"u": 0.01}}},
                             {"method": "pinn", "status": "failed", "metrics": {}}]}
    m = M.from_comparison(_Cmp())
    assert m.metrics == [{"name": "fem.relative_l2.u", "value": 0.01, "unit": "",
                          "how": "pp.compare against exact at 64 points"}]
    from pinneapple_physics.solving import Comparison, Solution
    assert hasattr(Solution, "to_experiment") and hasattr(Comparison, "to_experiment")
    f = M.from_flow_run({"id": "flow-1", "title": "Flow run", "nodes": [{"type": "solve"}],
                         "metrics": [{"name": "e", "value": 0.1, "unit": "", "how": "node"},
                                     {"name": "bad", "value": None}]}, str(tmp_path / "flow"))
    assert f.methods == ["solve"] and [x["name"] for x in f.metrics] == ["e"]
    assert M.load(str(tmp_path / "flow")).provenance["source"] == "flow"


# ---------------------------------------------------------------------------------------------- discovery
def test_scanner_finds_a_new_example_with_no_code_change(tmp_path):
    root = tmp_path / "examples"
    (root / "new_study").mkdir(parents=True)
    M.save(M.Manifest(id="examples/new_study", title="New"), str(root / "new_study"))
    found = {m.id for _, m in M.scan([str(root)], lab="")}
    assert found == {"examples/new_study"}


def test_scaffold_run_appears_in_the_lab_and_on_the_site(tmp_path, monkeypatch):
    root = tmp_path / "examples"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    d = M.scaffold("my_study", str(root))
    assert M.validate(M.load(d), d) == []
    lab = tmp_path / "lab"
    env = {**os.environ, "PINNEAPPLE_LAB": str(lab), "PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run([sys.executable, os.path.join(d, "run.py")], cwd=tmp_path, env=env, check=True, timeout=600)
    m = M.load(d)
    assert m.metrics[0]["name"] == "max_error" and m.files["figures"] == ["results/u.png"]
    cur = json.load(open(lab / "curation.json"))
    assert cur["items"]["manifest/my_study"]["tier"] in ("A", "B", "C")
    from pinneapple_lab.html import write_site
    page = open(write_site(LabStore(str(lab)), str(tmp_path / "site"))).read()
    assert "my_study" in page


# ---------------------------------------------------------------------------------------------- the CI gate
def test_ci_gate_catches_a_missing_or_broken_manifest(tmp_path):
    root = tmp_path / "examples"
    for name in ("good", "missing", "broken", "_private", "scratch"):
        (root / name).mkdir(parents=True)
    M.save(M.Manifest(id="good", title="Good"), str(root / "good"))
    bad = M.Manifest(id="broken", title="Broken")
    bad.files["videos"].append("gone.mp4")
    bad.metrics.append({"name": "x", "value": None, "unit": "", "how": "?"})
    M.save(bad, str(root / "broken"))
    problems = M.check_repo(str(root), lab=str(tmp_path / "nolab"))
    text = "\n".join(problems)
    assert "missing: no experiment.json" in text and "gone.mp4 does not exist" in text and "'x' has no measured" in text
    assert "good" not in text and "_private" not in text and "scratch" not in text
    from pinneapple_lab.__main__ import main as cli
    assert cli(["manifests", "--check", str(root)]) == 1


def test_the_repository_passes_its_own_gate():
    assert M.check_repo() == []
    assert len(M.example_dirs()) >= 60
    flight = M.load(os.path.join(REPO, "examples", "flight_dynamics"))
    enso = M.load(os.path.join(REPO, "examples", "enso_forecasting"))
    assert flight.links["guide"] == "docs/guides/flight_dynamics.md" and 847 in flight.links["pr"]
    assert enso.links["guide"] == "docs/guides/enso_forecasting.md" and 846 in enso.links["pr"]
    assert np.all([m.schema == M.SCHEMA for _, m in M.scan(lab="")])
