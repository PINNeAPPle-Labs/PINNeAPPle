"""Autoresearch loop: tunable-block edits, keep/revert, logging, proposers (no network)."""
import json

import pytest

from pinneapple_arena.autoresearch import (
    AutoResearch, LLMProposer, RandomSearchProposer, apply_tunables, init_workdir, read_tunables,
)

STUB = '''import os
# >>> TUNABLE
A = 3.0  # distance to the optimum at A = 1
B = "x"
# <<< TUNABLE
print(f"METRIC score={(A - 1.0) ** 2:.6f}")
'''


def test_tunable_block_read_and_rewrite():
    assert read_tunables(STUB) == {"A": 3.0, "B": "x"}
    out = apply_tunables(STUB, {"A": 1.5})
    assert read_tunables(out) == {"A": 1.5, "B": "x"} and "# distance to the optimum" in out
    with pytest.raises(KeyError, match="not tunable"):
        apply_tunables(STUB, {"LR": 1})


def test_loop_keeps_improvements_reverts_the_rest_and_logs(tmp_path):
    (tmp_path / "train.py").write_text(STUB)
    proposals = iter([{"changes": {"A": 2.0}, "description": "closer"},
                      {"changes": {"A": 5.0}, "description": "worse"},
                      {"changes": {"C": 1}, "description": "invalid"},
                      {"changes": {"A": 1.0}, "description": "optimum"}])
    best = AutoResearch(str(tmp_path), lambda p, c, h: next(proposals), budget_s=30).run(max_trials=4)
    assert best.metric == 0.0 and best.description == "optimum"
    assert read_tunables((tmp_path / "best_train.py").read_text())["A"] == 1.0
    rows = (tmp_path / "results.tsv").read_text().strip().splitlines()[1:]
    kept = [r.split("\t")[3] for r in rows]
    assert kept == ["1", "1", "0", "0", "1"]  # baseline, closer, worse (reverted), invalid, optimum
    assert "bad_proposal" in rows[3]


def test_random_search_changes_only_the_search_space():
    code = init_workdir.__globals__["TRAIN_TEMPLATE"]
    prop = RandomSearchProposer({"LR": [1e-3, 3e-3], "WIDTH": [16, 64]}, seed=1)(
        "", code, [])
    assert set(prop["changes"]) <= {"LR", "WIDTH"} and prop["changes"]


def test_llm_proposer_parses_json_from_any_callable():
    seen = {}

    def fake_llm(prompt, system=""):
        seen["prompt"] = prompt
        return 'Sure. {"changes": {"LR": 0.003}, "description": "higher lr"} done'

    code = init_workdir.__globals__["TRAIN_TEMPLATE"]
    prop = LLMProposer(llm_call=fake_llm)("program text", code, [])
    assert prop == {"changes": {"LR": 0.003}, "description": "higher lr"}
    assert "program text" in seen["prompt"] and json.loads(seen["prompt"].split("\n")[4])["LR"] == 0.001


def test_real_pinn_template_runs_and_reports_the_metric(tmp_path):
    init_workdir(str(tmp_path))
    from pinneapple_arena.autoresearch import run_trial
    metric, status, out, _ = run_trial(str(tmp_path), budget_s=4)
    assert status == "ok" and 0 < metric < 1.5, out
