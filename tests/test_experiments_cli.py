"""The command that closes the acceptance of #167: "a command reproduces a stored result from its record".

A Python call is not enough for that sentence — the record has to be reproducible from a shell, in a job, with an
exit code that distinguishes the three answers. These tests hold the exit codes, because a script that treats
"refused" as "divergent" passes a run it never reproduced.
"""
from __future__ import annotations

import json

import pytest

import pinneapple as pp
from pinneapple_check.cli import main as pinneapple_main

SMALL_PINN = {"epochs": 10, "n_collocation": 64, "hidden_dim": 8, "n_layers": 2}


def record(tmp_path):
    directory = tmp_path / "run"
    experiment = pp.Experiment("burgers_1d", method="pinn", options=dict(SMALL_PINN), seed=3,
                               reference="analytic", n_eval=64)
    experiment.save(str(directory))
    experiment.run().save(str(directory))
    return directory


def test_the_record_carries_the_commit_the_environment_the_seed_the_hardware_and_the_inputs(tmp_path):
    """What #167 asks the record to contain, checked on a real saved run rather than promised in a docstring."""
    directory = record(tmp_path)
    payload = json.loads((directory / "result.json").read_text())
    env = payload["environment"]
    assert env["commit"], "the commit that ran has to be in the record"
    for key in ("python", "numpy", "platform", "machine", "cpu_count", "pinneapple"):
        assert key in env, key
    assert payload["config"]["seed"] == 3
    assert payload["problem_fingerprint"], "the inputs are identified by the problem fingerprint"
    assert payload["n_eval_points"] > 0


def test_the_command_reproduces_a_stored_result_and_exits_zero(tmp_path, capsys):
    directory = record(tmp_path)
    code = pinneapple_main(["reproduce", str(directory), "--quiet"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "reproduced" in out


def test_a_changed_environment_exits_one(tmp_path, capsys):
    directory = record(tmp_path)
    payload = json.loads((directory / "result.json").read_text())
    payload["environment"]["numpy"] = "0.0.0+fake"
    (directory / "result.json").write_text(json.dumps(payload))
    assert pinneapple_main(["reproduce", str(directory), "--quiet"]) == 1
    assert "numpy" in capsys.readouterr().out


def test_a_record_that_cannot_be_run_exits_two(tmp_path, capsys):
    code = pinneapple_main(["reproduce", str(tmp_path / "nao_existe"), "--quiet"])
    assert code == 2
    assert "refused" in capsys.readouterr().err


def test_the_full_report_and_its_json(tmp_path, capsys):
    directory = record(tmp_path)
    out = tmp_path / "report.json"
    assert pinneapple_main(["reproduce", str(directory), "--json", str(out)]) == 0
    assert "| Metric | Recorded | Now |" in capsys.readouterr().out
    payload = json.loads(out.read_text())
    assert payload["outcome"] == "reproduced" and payload["kind"] == "experiment"
    assert payload["metrics"] and payload["experiment"]


def test_the_command_is_listed_in_the_front_door(tmp_path, capsys):
    pinneapple_main([])
    text = capsys.readouterr().out
    assert "reproduce" in text


def test_an_unknown_flag_is_a_usage_error_not_a_verdict(tmp_path):
    with pytest.raises(SystemExit):
        pinneapple_main(["reproduce", str(tmp_path), "--nao-existe"])
