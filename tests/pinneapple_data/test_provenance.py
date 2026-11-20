"""Provenance: the minimum a dataset, a run or a model records to be reproducible, and the gate that enforces it (#39).

The rule has to be narrow enough that a producer can always satisfy it and strict enough that a record missing a
field is refused, so these tests hold both ends: the five fields (and the mesh on top, for a solver that builds one),
a record built by the helper passing, and the artifacts in this tree passing -- which is what makes the migration of
the existing model cards and datasets a property of the repository rather than a one-off edit.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pinneapple_data.provenance import (MINIMAL, build_provenance, data_hash, data_hash_of_bytes, git_commit,
                                        problems, require, solver_uses_a_mesh)

ROOT = Path(__file__).resolve().parents[2]


def _complete() -> dict:
    return {"solver": "pinn", "library_version": "0.8.0", "seed": 0, "data_hash": "sha256:" + "0" * 64,
            "commit": "a" * 40}


def test_a_record_without_provenance_says_exactly_what_is_missing():
    missing = problems({})
    assert len(missing) == len(MINIMAL)
    for field in MINIMAL:
        assert any(line.startswith(field) for line in missing), field
    assert problems(None) == [f"no provenance at all: a record needs {', '.join(MINIMAL)}"]
    assert problems("not a mapping") == ["provenance must be a mapping, got str"]


def test_a_complete_record_passes_and_require_only_adds_the_error():
    record = _complete()
    assert problems(record) == []
    assert require(record) is record
    broken = {**record, "data_hash": ""}
    assert problems(broken) == [line for line in problems(broken) if line.startswith("data_hash")]
    with pytest.raises(ValueError, match="incomplete provenance"):
        require(broken)
    with pytest.raises(ValueError, match="build_provenance"):
        require({})


def test_the_seed_says_whether_there_was_any_randomness():
    """A steady solve has no seed; recording 0 would claim a run that could have been repeated differently."""
    assert problems({**_complete(), "seed": "deterministic"}) == []
    assert problems({**_complete(), "seed": 1234}) == []
    assert problems({**_complete(), "seed": 1.5}) == ["seed must be an integer or 'deterministic', got 1.5"]
    assert any(line.startswith("seed") for line in problems({**_complete(), "seed": None}))


def test_a_meshed_solver_has_to_say_which_mesh():
    meshed = {**_complete(), "solver": "openfoam-simpleFoam"}
    assert any(line.startswith("mesh") for line in problems(meshed))
    assert problems({**meshed, "mesh": {"cells": 207080}}) == []
    # the guess is the solver's name; a producer that knows better can say so either way
    assert problems(meshed, uses_a_mesh=False) == []
    assert problems({**_complete(), "solver": "my-unknown-solver"}, uses_a_mesh=True) != []
    assert solver_uses_a_mesh("openfoam-simpleFoam") and solver_uses_a_mesh("FEniCS") and solver_uses_a_mesh("fem")
    assert not solver_uses_a_mesh("pinn") and not solver_uses_a_mesh("closed-form")


def test_the_builder_fills_the_version_the_commit_and_the_input_hash(tmp_path):
    payload = tmp_path / "case.dat"
    payload.write_bytes(b"1 2 3\n")
    record = build_provenance("pinn", seed=3, data_hash_of=payload, app="pinneapple_app")
    assert record["library_version"] and record["seed"] == 3 and record["app"] == "pinneapple_app"
    assert record["data_hash"] == data_hash(payload) == data_hash_of_bytes(b"1 2 3\n")
    assert record["commit"] == git_commit()          # a checkout knows its HEAD
    assert problems(record) == []
    # an explicit empty commit is a statement ("not knowable"), and the gate reports it
    assert any(line.startswith("commit") for line in problems(build_provenance("pinn", seed=3, data_hash="x",
                                                                               commit="")))


def test_the_contract_carries_the_same_rule():
    from pinneapple_data.contracts import Provenance
    assert len(Provenance().problems()) == len(MINIMAL)
    assert Provenance(**_complete()).problems() == []


def test_a_model_card_without_provenance_is_refused(tmp_path):
    from pinneapple_hub.model_card import ModelCard
    card = ModelCard(name="toy", architecture="bench_mlp", validation_metrics={"test/rmse": 0.1},
                     reference_source="held-out runs", pinneapple_version="0.8.0")
    assert any(line.startswith("provenance:") for line in card.validate())
    card.provenance = _complete()
    assert card.validate() == []


def test_the_model_card_governance_script_reports_it(tmp_path):
    from pinneapple_hub.model_card import ModelCard
    path = tmp_path / "model_card.json"
    ModelCard(name="toy", architecture="bench_mlp", validation_metrics={"test/rmse": 0.1},
              reference_source="held-out runs", pinneapple_version="0.8.0").save(str(path))
    run = subprocess.run([sys.executable, "scripts/validate_provenance.py", str(path)], capture_output=True,
                         text=True, cwd=str(ROOT))
    assert run.returncode == 1 and "provenance" in run.stdout and "build_provenance" in run.stdout


def test_every_artifact_in_the_tree_has_complete_provenance():
    """The gate as the repository uses it: a new dataset, model card or model reference cannot land without one."""
    run = subprocess.run([sys.executable, "scripts/validate_provenance.py"], capture_output=True, text=True,
                         cwd=str(ROOT))
    assert run.returncode == 0, run.stdout + run.stderr
    assert "0 without complete provenance" in run.stdout


def test_the_reference_that_points_at_a_card_is_not_asked_for_a_second_copy():
    """A ``ModelRef`` is a pointer; duplicating the provenance there would be a second copy that can disagree."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import validate_provenance as vp

    path = ROOT / "apps" / "control_lab" / "examples" / "tank_rom.json.modelref.json"
    assert vp._check_contract(path) == []
    data = json.loads(path.read_text(encoding="utf-8"))
    stripped = {k: v for k, v in data.items() if k != "provenance"}
    tmp = Path(path).parent / "_vp_tmp.modelref.json"
    try:
        tmp.write_text(json.dumps(stripped), encoding="utf-8")
        assert vp._check_contract(tmp) == []                       # it has a uri: the pointer is the provenance
        pointerless = {k: v for k, v in stripped.items() if k != "uri"}
        tmp.write_text(json.dumps(pointerless), encoding="utf-8")
        assert vp._check_contract(tmp)                              # ... and without one, it is asked for it
    finally:
        tmp.unlink(missing_ok=True)
