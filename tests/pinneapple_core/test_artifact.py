"""The ``.ppx`` container: a result that says where it came from and reproduces elsewhere, or no result at all."""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import pinneapple as pp
from pinneapple_core.artifact import (
    SCHEMA,
    Artifact,
    IntegrityError,
    MissingProvenance,
    load,
    save,
    verify,
)
from pinneapple_physics.pde_environment.builder import ProblemBuilder

ROOT = Path(__file__).resolve().parents[2]
SOURCE = lambda X, ctx=None: -2 * np.pi ** 2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
CTX = {"source_fn": SOURCE}


def _poisson():
    spec = (ProblemBuilder("poisson_artifact").domain(x=(0, 1), y=(0, 1)).fields("u").pde("poisson")
            .bc("dirichlet", field="u", value=0.0, on="x_boundary")
            .bc("dirichlet", field="u", value=0.0, on="y_boundary").build())
    return pp.PhysicalProblem.from_pde_spec(spec)


def test_round_trip_keeps_problem_units_metrics_and_members(tmp_path):
    art = Artifact(problem={"name": "poisson_artifact", "fields": ["u"]}, units={"u": "m"},
                   solver={"method": "fem", "kind": "classical"}, metrics=[{"name": "rel_l2", "value": 0.0031, "unit": ""}],
                   validation=[{"name": "balance", "passed": True}], arrays={"u": np.linspace(0, 1, 5)})
    art.provenance = {"source": "test", "created_at": "2026-11-06T00:00:00+00:00", "environment": {"python": "3"},
                      "git": {"commit": "0" * 40, "dirty": False}}
    path = save(art, tmp_path / "case")
    assert path.suffix == ".ppx" and path.exists()

    back = load(path)
    assert back.problem == art.problem and back.units == {"u": "m"}
    assert back.metrics[0]["value"] == pytest.approx(0.0031) and back.validation == [{"name": "balance", "passed": True}]
    assert np.allclose(back.arrays["u"], art.arrays["u"])
    assert back.provenance["source"] == "test"

    report = verify(path)
    assert [c["name"] for c in report.checks][:4] == ["container", "integrity", "provenance", "sections"]
    assert report.passed and not report.failed


def test_saving_a_solution_records_the_problem_and_reproduces_with_the_callable(tmp_path):
    solution = pp.solve(_poisson(), "fem", ctx=CTX, n=8)
    path = save(solution, tmp_path / "poisson.ppx", samples=24)

    art = load(path)
    assert art.problem["name"] == "poisson_artifact" and art.solver["method"] == "fem"
    assert art.arrays["probe"].shape == (24, 2) and art.reproductions[0]["needs"] == "callable"

    fresh = pp.solve(_poisson(), "fem", ctx=CTX, n=8)          # a second solve on this machine, not the original object
    report = art.reproduce(fresh.predict)
    assert report.passed, report.checks
    assert "max |difference|" in report.checks[0]["detail"]

    # a different answer is not accepted as the same result
    shifted = art.reproduce(lambda X: fresh.predict(X) + 1e-3)
    assert not shifted.passed and shifted.failed[0]["name"] == "reproduction:solution"


def test_a_model_artifact_reproduces_in_another_process_from_another_directory(tmp_path):
    module = tmp_path / "ppx_probe_model.py"
    module.write_text(
        "import torch\n"
        "class Tiny(torch.nn.Module):\n"
        "    def __init__(self, width=8):\n"
        "        super().__init__()\n"
        "        self.net = torch.nn.Sequential(torch.nn.Linear(2, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))\n"
        "    def forward(self, x):\n"
        "        return self.net(x)\n")
    sys.path.insert(0, str(tmp_path))
    try:
        import ppx_probe_model
        import torch

        torch.manual_seed(0)
        model = ppx_probe_model.Tiny(width=8)
        path = save(Artifact.from_model(model, init_kwargs={"width": 8}, problem=_poisson(), units={"u": "m"},
                                        lo=(0, 0), hi=(1, 1), samples=16), tmp_path / "model.ppx")
    finally:
        sys.path.remove(str(tmp_path))

    assert load(path).predict(np.zeros((3, 2))).shape == (3, 1)

    code = ("from pinneapple_core.artifact import load, verify\n"
            f"art = load(r'{path}')\n"
            f"report = verify(r'{path}')\n"
            "print(report.passed, report.checks[-1]['detail'])\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(tmp_path), os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path), env=env,
                         timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[0] == "True"
    assert "max |difference|" in out.stdout


def test_a_file_that_does_not_say_where_it_came_from_is_refused(tmp_path):
    art = Artifact(problem={"name": "p", "fields": ["u"]}, units={"u": "m"}, arrays={"u": np.zeros(3)})
    path = save(art, tmp_path / "anon.ppx")
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))

    def _write(mutated):
        target = tmp_path / "mutated.ppx"
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(target, "w") as dst:
            for name in src.namelist():
                dst.writestr(name, json.dumps(mutated) if name == "manifest.json" else src.read(name))
        return target

    without = dict(manifest)
    del without["provenance"]
    with pytest.raises(MissingProvenance, match="provenance"):
        load(_write(without))
    assert verify(_write(without)).failed[0]["name"] == "provenance"

    stripped = {**manifest, "provenance": {k: v for k, v in manifest["provenance"].items() if k != "source"}}
    with pytest.raises(MissingProvenance, match="source"):
        load(_write(stripped))
    # the same file loads when the caller says the provenance does not matter
    assert load(_write(stripped), strict=False).problem["name"] == "p"


def test_a_changed_member_is_refused(tmp_path):
    art = Artifact(problem={"name": "p", "fields": ["u"]}, units={"u": "m"}, arrays={"u": np.linspace(0, 1, 4)})
    path = save(art, tmp_path / "tampered.ppx")
    with zipfile.ZipFile(path) as z:
        members = {n: z.read(n) for n in z.namelist()}
    members["arrays/u.npy"] = b"not the array that was written"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)

    with pytest.raises(IntegrityError, match="arrays/u.npy"):
        load(path)
    report = verify(path)
    assert not report.passed and report.failed[0]["name"] == "integrity"


def test_units_that_are_missing_are_reported_not_hidden(tmp_path):
    art = Artifact(problem={"name": "p", "fields": ["u", "v"]}, units={"u": "m"}, validation=[{"name": "x", "passed": True}],
                   solver={"method": "fem"}, metrics=[{"name": "m", "value": 1.0}])
    report = verify(save(art, tmp_path / "units.ppx"))
    units = [c for c in report.checks if c["name"] == "units"][0]
    assert units["passed"] is False and "v" in units["detail"]
    assert not report.passed


def test_the_manifest_names_its_schema_and_evidences_every_member(tmp_path):
    art = Artifact(problem={"name": "p", "fields": ["u"]}, units={"u": "m"}, solver={"method": "fem"},
                   metrics=[{"name": "m", "value": 1.0}], arrays={"u": np.zeros(2), "mesh_points": np.zeros((4, 2))})
    path = save(art, tmp_path / "schema.ppx")
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        names = set(z.namelist())
    assert manifest["schema"] == SCHEMA
    assert set(manifest["arrays"]) == {"u", "mesh_points"}
    assert set(manifest["integrity"]) == names - {"manifest.json"}
    assert manifest["arrays"]["mesh_points"]["shape"] == [4, 2]
