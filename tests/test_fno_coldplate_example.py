"""examples/fno_coldplate (#1163): the library's FNO2d on example 07's cold plate, trained and checked by
PINNeAPPle without PhysicsNeMo. The smoke size runs here; the full run's numbers are in its results/summary.json."""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples", "fno_coldplate")


def test_smoke_runs_the_whole_chain_and_decides(tmp_path):
    spec = importlib.util.spec_from_file_location("fno_coldplate_run", os.path.join(EX, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = mod.main(smoke=True, base=str(tmp_path))
    assert s["decision"] in ("APPROVED", "REVIEW", "REJECT") and s["model"] == "pinneapple_neural FNO2d"
    assert {c["name"] for c in s["checks"]} >= {"boundary condition u = 0", "reference (relative L2)"}
    assert (tmp_path / "results" / "summary.json").exists()


def test_the_committed_full_run_is_the_documented_one():
    s = json.load(open(os.path.join(EX, "results", "summary.json")))
    assert s["train_steps"] == 2700 and s["padding"] == 8
    assert s["test_rel_l2"] < 0.05 and s["decision"] == "REJECT"          # the README states the decision as measured
