"""The surrogate pipeline: the configuration round-trip, the stage machine, and the gates that decide a run.

The heavy runs (a full parametric study, a PINN of 1200 epochs) live in ``examples/surrogate_pipeline``; here every
test either fails before it trains anything or trains a deliberately tiny model.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from pinneapple_pipeline import (
    FAIL,
    NOT_APPLICABLE,
    NOT_EVALUATED,
    PASS,
    WARN,
    ConfigError,
    PipelineFailed,
    PipelineRejected,
    SurrogateConfig,
    SurrogatePipeline,
    VerificationResult,
)
from pinneapple_pipeline import verification as V
from pinneapple_pipeline.configs import BoundaryConditionConfig, DimensionConfig

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MINIMAL = {"name": "tiny", "problem": {"source": "preset", "preset": "burgers_1d"},
           "data": {"source": "physics_only"}}


# ------------------------------------------------------------------------------------------------- configuration
def test_config_round_trips_through_yaml(tmp_path):
    cfg = SurrogateConfig.from_dict({**MINIMAL, "training": {"epochs": 7}})
    path = tmp_path / "p.yaml"
    cfg.to_yaml(path)
    again = SurrogateConfig.from_yaml(path)
    assert again.name == "tiny" and again.training.epochs == 7
    assert again.to_dict() == cfg.to_dict()


def test_an_unknown_key_is_an_error_not_ignored():
    with pytest.raises(ConfigError, match="unknown top-level key"):
        SurrogateConfig.from_dict({**MINIMAL, "trainingg": {"epochs": 7}})
    with pytest.raises(ConfigError, match="has no field"):
        SurrogateConfig.from_dict({**MINIMAL, "training": {"epoch": 7}})


def test_the_configuration_refuses_a_run_it_cannot_do():
    with pytest.raises(ConfigError, match="preset"):
        SurrogateConfig.from_dict({"name": "x", "problem": {"source": "preset"}})
    with pytest.raises(ConfigError, match="parameters is empty"):
        SurrogateConfig.from_dict({**MINIMAL, "data": {"source": "simulation"}})
    with pytest.raises(ConfigError, match="which preset"):
        SurrogateConfig.from_dict({**MINIMAL, "data": {"source": "simulation"},
                                   "parameters": {"parameters": {"Re": [1, 2]}}})
    with pytest.raises(ConfigError, match="check_every"):
        SurrogateConfig.from_dict({**MINIMAL, "training": {"early_stopping": True, "epochs": 50,
                                                           "check_every": 50}})
    with pytest.raises(ConfigError, match="geometry"):
        SurrogateConfig.from_dict({**MINIMAL, "geometry": {"source": "file", "path": "/tmp/x.obj", "format": "obj"}})
    with pytest.raises(ConfigError, match="not in the catalogue"):
        SurrogateConfig.from_dict({**MINIMAL, "materials": {"materials": {"body": "unobtainium"}}})
    with pytest.raises(ConfigError, match="search_space"):
        SurrogateConfig.from_dict({**MINIMAL, "optimization": {"enabled": True}})
    with pytest.raises(ConfigError, match="physics_only"):
        SurrogateConfig.from_dict({**MINIMAL, "adaptive_learning": {"enabled": True}})
    with pytest.raises(ConfigError, match="deployment.port"):
        SurrogateConfig.from_dict({**MINIMAL, "deployment": {"enabled": True, "target": "http", "port": -1}})
    with pytest.raises(ConfigError, match="formulation"):
        SurrogateConfig.from_dict({**MINIMAL, "problem": {"source": "preset", "preset": "burgers_1d",
                                                          "formulation": "weak"}})


def test_dimension_is_checked_against_the_problem():
    d = DimensionConfig(spatial=3)
    assert d.check_against(("x", "y")) and "spatial=3" in d.check_against(("x", "y"))[0]
    d = DimensionConfig(temporal=False)
    assert d.check_against(("x", "t"))
    assert DimensionConfig().check_against(("x", "t")) == []          # unset: taken from the problem


def test_a_boundary_condition_becomes_a_condition_of_the_problem(tmp_path):
    cfg = SurrogateConfig.from_dict({
        "name": "bc", "problem": {"source": "preset", "preset": "laplace_2d"},
        "data": {"source": "physics_only"},
        "boundary_conditions": [{"name": "top", "kind": "dirichlet", "field": "u", "source": "constant",
                                 "value": 5.0, "on": "y_max"}],
    })
    from pinneapple_pipeline import adapters
    bundle = adapters.build_problem(cfg)
    added = [c for c in bundle.problem.conditions if c.name == "top"]
    assert added and added[0].kind == "dirichlet"
    pts = np.array([[0.5, 1.0], [0.5, 0.5]], dtype=float)
    assert added[0].values(pts, {})[0, 0] == pytest.approx(5.0)


def test_a_symbolic_problem_condition_expression_is_evaluated():
    cond = BoundaryConditionConfig(name="hot", kind="dirichlet", field="u", source="symbolic",
                                   expression="300 + 10*sin(t)", on="x_min")
    spec = cond.to_condition(("x", "t"), {"x": (0.0, 1.0), "t": (0.0, 1.0)})
    values = spec.values(np.array([[0.0, np.pi / 2], [0.0, 0.0]]), {})
    assert values[0, 0] == pytest.approx(310.0) and values[1, 0] == pytest.approx(300.0)


# ---------------------------------------------------------------------------------------------------- pipeline
def test_plan_and_dry_run_say_what_would_happen(tmp_path):
    p = SurrogatePipeline.from_preset("burgers_in_viscosity", execution={"out_dir": str(tmp_path)})
    plan = p.plan()
    assert [s["stage"] for s in plan["steps"]] == ["problem", "data", "preprocess", "model", "training", "export",
                                                   "verification"]
    assert plan["cost"]["reference_solves"] == 18
    report = p.dry_run()
    assert report["ok"], report["checks"]


def test_run_until_stops_where_asked_and_leaves_a_state_file(tmp_path):
    p = SurrogatePipeline.from_preset("burgers_in_viscosity", execution={"out_dir": str(tmp_path)})
    result = p.run_until("problem")
    assert result.stages[0].status == "done"
    assert all(s.status == "pending" for s in result.stages[1:])
    state = json.loads((tmp_path / "burgers_in_viscosity" / "state.json").read_text())
    assert state["stages"][0]["stage"] == "problem"


def test_a_stage_that_fails_names_itself_and_keeps_the_earlier_ones(tmp_path):
    cfg = SurrogateConfig.from_dict({**MINIMAL, "problem": {"source": "preset", "preset": "no_such_preset"}})
    p = SurrogatePipeline(cfg, )
    p.config.execution.out_dir = str(tmp_path)
    with pytest.raises(PipelineFailed) as exc:
        p.run()
    assert exc.value.stage == "problem"
    assert (tmp_path / "tiny" / "state.json").is_file()


def test_the_parametric_pipeline_runs_and_gates_itself(tmp_path):
    """A tiny supervised run: 12 cases, one fast method, then every gate that applies."""
    p = SurrogatePipeline.from_preset(
        "burgers_in_viscosity", execution={"out_dir": str(tmp_path)},
        sampling={"design_sampling": {"method": "lhs", "n_samples": 12},
                  "validation_sampling": {"test_fraction": 0.25, "val_fraction": 0.12, "shift": "auto"}, "seed": 1},
        architecture={"family": "ridge", "candidates": []},
        training={"budget_seconds": 0.5},
        verification={"mode": "balanced", "fail_policy": "report"})
    result = p.run()
    gates = {g.gate: g.status for g in result.verification}
    assert gates["problem"] == PASS and gates["simulation"] == PASS
    assert gates["dataset"] == PASS and gates["model"] == PASS
    assert gates["physics"] == NOT_APPLICABLE          # a tabular surrogate has no PDE residual to check
    assert gates["dataset"] and "no split table" not in " ".join(result.gate("dataset").limitations)
    assert result.artifacts["dataset"].endswith("dataset.npz")
    assert (tmp_path / "burgers_in_viscosity" / "report.md").is_file()
    assert result.to_markdown().startswith("# Surrogate pipeline")
    # and the model answers a query, inside the domain it was measured on
    answer = p.predict(nu=0.03)
    assert answer["outputs"] and answer["n"] == 1


def test_a_failed_gate_blocks_the_run_when_the_policy_says_so(tmp_path):
    """An undertrained model must not be reported as a usable one."""
    p = SurrogatePipeline.from_preset(
        "heat_2d_symbolic", execution={"out_dir": str(tmp_path)},
        training={"epochs": 20, "n_collocation": 200},
        verification={"mode": "balanced", "fail_policy": "block"})
    with pytest.raises(PipelineRejected) as exc:
        p.run()
    result = exc.value.result
    assert result.status == "rejected"
    assert result.gate("physics").status == FAIL
    assert any("physics" in x for x in result.limitations)


# ----------------------------------------------------------------------------------------- the new blocks
def _cube_stl(path, *, drop_last=True):
    """A unit cube as a binary STL: 12 triangles, or 11 of them so the surface has open edges."""
    import struct
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    tris = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    if drop_last:
        tris = tris[:-1]
    data = b"\0" * 80 + struct.pack("<I", len(tris))
    for t in tris:
        data += struct.pack("<3f", 0, 0, 0)
        for i in t:
            data += struct.pack("<3f", *v[i])
        data += struct.pack("<H", 0)
    path.write_bytes(data)
    return path


def test_the_geometry_gate_checks_a_surface_instead_of_shrugging(tmp_path):
    """A closed STL passes with numbers; an open one fails and says how many edges are open."""
    closed, opened = _cube_stl(tmp_path / "cube.stl", drop_last=False), _cube_stl(tmp_path / "open.stl")
    for path, expected in ((closed, PASS), (opened, FAIL)):
        cfg = SurrogateConfig.from_dict({**MINIMAL, "geometry": {"source": "file", "path": str(path), "units": "cm"}})
        state = _state()
        state.cfg, state.bundle = cfg, _bundle()
        gate = V.gate_geometry(state)
        assert gate.status == expected
        assert gate.metrics["faces"] == (12 if path is closed else 11)
        if expected is PASS:
            assert gate.metrics["watertight"] is True
            assert gate.metrics["bounding_box_m"] == [0.01, 0.01, 0.01]      # 1 cm, in metres
        else:
            assert gate.metrics["open_edges"] > 0 and "not closed" in gate.limitations[0]


def test_the_geometry_block_can_build_the_surface_it_checks(tmp_path):
    """``source: builder`` writes the STL into the run folder, and the gate measures that file — not a sketch of it.

    ``n`` arrives as 36 from a JSON config and as 36.0 from a config written in Python; both are a resolution.
    """
    cfg = SurrogateConfig.from_dict({**MINIMAL, "geometry": {
        "source": "builder", "builder": "cylinder", "units": "m",
        "parameters": {"radius": 0.05, "length": 0.3, "axis": "y", "n": 36.0}}})
    state = _state(out_dir=tmp_path)
    state.cfg, state.bundle = cfg, _bundle()
    gate = V.gate_geometry(state)
    assert gate.status == PASS, gate.limitations
    assert gate.metrics["builder"] == "cylinder" and gate.metrics["builder_parameters"]["radius"] == 0.05
    assert gate.metrics["watertight"] is True
    assert gate.metrics["bounding_box_m"] == [0.1, 0.3, 0.1]                 # 100 mm diameter, 300 mm long
    assert Path(gate.metrics["source"]).is_file() and Path(state.geometry_path).is_file()
    with pytest.raises(ConfigError, match="builder"):
        SurrogateConfig.from_dict({**MINIMAL, "geometry": {"source": "builder", "builder": "torus"}})


def test_materials_set_the_problems_own_parameters_with_their_sources():
    from pinneapple_pipeline import adapters
    cfg = SurrogateConfig.from_dict({
        "name": "solid", "data": {"source": "physics_only"},
        "problem": {"source": "symbolic", "coords": ["x", "y"], "fields": ["u"],
                    "domain_bounds": {"x": [0, 1], "y": [0, 1]},
                    "equations": ["lap(u) + conductivity"], "constants": {"conductivity": 1.0},
                    "conditions": [{"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}]},
        "materials": {"materials": {"body": "al6061_t6"}}})
    bundle = adapters.build_problem(cfg)
    assert bundle.system.coefficients["conductivity"] == 167.0
    assert any("datasheet" in n for n in bundle.notes)          # the number carries where it was read
    # a name that only looks like a material property is not touched (nu is viscosity here, not Poisson)
    assert "nu" not in cfg.materials.targets({"nu"})


def test_an_hyperparameter_search_keeps_the_best_trial_and_reports_all(tmp_path):
    p = SurrogatePipeline.from_preset(
        "burgers_physics_only", execution={"out_dir": str(tmp_path)},
        training={"epochs": 60, "n_collocation": 256},
        optimization={"enabled": True, "strategy": "grid", "search_space": {"lr": [0.002, 0.01]}},
        export={"enabled": False, "formats": []},
        verification={"fail_policy": "report"})
    result = p.run()
    search = result.metrics["optimization"]
    assert search["n_trials"] == 2 and len(search["trials"]) == 2
    assert search["best"]["score"] <= min(t["score"] for t in search["trials"])
    assert any("hyperparameter search" in n for n in p.state.notes)


def test_early_stopping_stops_before_the_budget(tmp_path):
    """With an impossible improvement demand every check fails, so the run stops at patience x check_every."""
    p = SurrogatePipeline.from_preset(
        "burgers_physics_only", execution={"out_dir": str(tmp_path)},
        training={"epochs": 400, "n_collocation": 256, "early_stopping": True, "check_every": 50,
                  "patience": 1, "min_delta": 1e9},
        export={"enabled": False, "formats": []}, verification={"fail_policy": "report"})
    result = p.run()
    assert result.metrics["training"]["history_points"] < 400
    note = " ".join(p.state.notes)
    assert "early stopping" in note and "stopped at" in note


def test_adaptive_learning_adds_cases_and_reports_every_round(tmp_path):
    p = SurrogatePipeline.from_preset(
        "burgers_in_viscosity", execution={"out_dir": str(tmp_path)},
        sampling={"design_sampling": {"method": "lhs", "n_samples": 12},
                  "validation_sampling": {"test_fraction": 0.25, "val_fraction": 0.12, "shift": "auto"}, "seed": 1},
        architecture={"family": "ridge", "candidates": []}, training={"budget_seconds": 0.5},
        adaptive_learning={"enabled": True, "max_rounds": 1, "samples_per_round": 2, "target_error": 1e-9},
        verification={"mode": "balanced", "fail_policy": "report"})
    result = p.run()
    adaptive = result.metrics["adaptive_learning"]
    assert adaptive["cases_added"] == 2 and adaptive["total_cases"] == 14
    assert adaptive["rounds"][0]["added"] == 2
    assert p.state.dataset.n == 14                       # the dataset on disk grew with it


def test_the_run_is_served_with_its_own_trust(tmp_path):
    p = SurrogatePipeline.from_preset(
        "burgers_in_viscosity", execution={"out_dir": str(tmp_path)},
        sampling={"design_sampling": {"method": "lhs", "n_samples": 12},
                  "validation_sampling": {"test_fraction": 0.25, "val_fraction": 0.12, "shift": "auto"}, "seed": 1},
        architecture={"family": "ridge", "candidates": []}, training={"budget_seconds": 0.5},
        verification={"mode": "balanced", "fail_policy": "report"})
    p.run()
    deployment = p.deploy()
    inside = p.state.validated_domain["bounds"]["nu"]
    answer = deployment.predict(nu=0.5 * (inside[0] + inside[1]))
    assert answer["trust"]["decision"] == "ACCEPT" and answer["outputs"]["u@3"]
    with pytest.raises(ValueError, match="outside the validated domain"):
        deployment.predict(nu=inside[1] + 1.0)
    card = deployment.service.info()["pinneapple"]
    assert card["inputs"] == ["nu"] and card["validity_range"]["nu"] == inside
    # the run folder is enough to serve it again, without the object that trained it
    again = SurrogatePipeline.from_run(tmp_path / "burgers_in_viscosity")
    assert again.result is not None and again.deploy().service.id == "burgers_in_viscosity"


def test_pinneapple_check_is_run_over_the_run_folder_and_feeds_the_gates(tmp_path):
    p = SurrogatePipeline.from_preset(
        "burgers_in_viscosity", execution={"out_dir": str(tmp_path)},
        sampling={"design_sampling": {"method": "lhs", "n_samples": 12},
                  "validation_sampling": {"test_fraction": 0.25, "val_fraction": 0.12, "shift": "auto"}, "seed": 1},
        architecture={"family": "ridge", "candidates": []}, training={"budget_seconds": 0.5},
        verification={"mode": "balanced", "fail_policy": "report"})
    result = p.run()
    check = result.metrics["pinneapple_check"]
    assert check["n_checks"] > 0
    rules = {o["rule"] for o in check["outcomes"]}
    assert "PC601" in rules and all(o["passed"] for o in check["outcomes"])
    folder = tmp_path / "burgers_in_viscosity" / "verification"
    assert (folder / "pinneapple-check.toml").is_file() and (folder / "predict.py").is_file()
    assert any("pinneapple check agrees" in x for x in result.gate("applicability").evidence)



# ------------------------------------------------------------------------------------------------------- gates
def _bundle():
    """The problem of ``MINIMAL`` as the problem stage would build it (for the gates that need one)."""
    from pinneapple_pipeline import adapters
    return adapters.build_problem(SurrogateConfig.from_dict(MINIMAL))


def _state(**kwargs):
    class _S:
        pass
    s = _S()
    s.cfg = SurrogateConfig.from_dict(MINIMAL)
    s.metrics = {}
    s.reference = {}
    s.dataset = None
    s.model = None
    s.model_object = None
    s.validated_domain = {}
    s.bundle = None
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def test_a_gate_status_is_one_of_five_and_only_pass_passes():
    for status in (PASS, WARN, FAIL, NOT_EVALUATED, NOT_APPLICABLE):
        r = VerificationResult("model", status)
        assert r.passed is (status == PASS)
    with pytest.raises(ValueError):
        VerificationResult("model", "ok")


def test_an_unevaluated_model_gate_says_what_is_missing():
    state = _state()
    r = V.gate_model(state)
    assert r.status == NOT_EVALUATED and r.limitations and r.recommendations


def test_the_dataset_gate_catches_duplicate_cases():
    from pinneapple_analysis import surrogate
    data = surrogate.from_arrays({"X": np.zeros((8, 1)), "Y": np.random.default_rng(0).random((8, 2))},
                                 input_names=["nu"], output_names=["a", "b"])
    cfg = SurrogateConfig.from_dict({**MINIMAL, "data": {"source": "simulation"},
                                     "parameters": {"parameters": {"nu": [0.01, 0.02]}}})
    state = _state(cfg=cfg, dataset=data)
    r = V.gate_dataset(state)
    assert r.status == FAIL and any("duplicate" in x for x in r.limitations)


def test_the_export_gate_compares_the_exported_model():
    cfg = SurrogateConfig.from_dict({**MINIMAL, "export": {"formats": ["onnx"], "verify_export": True}})
    state = _state(cfg=cfg, metrics={"export": {"written": {"onnx": "m.onnx"},
                                                "equivalence": {"format": "onnx", "max_relative_difference": 1e-9}}})
    assert V.gate_export(state).status == PASS
    state.metrics["export"]["equivalence"]["max_relative_difference"] = 1e-2
    assert V.gate_export(state).status == FAIL
    state.metrics["export"] = {"written": {"onnx_error": "RuntimeError: nope"}}
    assert V.gate_export(state).status == WARN


def test_a_gate_that_raises_becomes_not_evaluated_not_a_crash():
    state = _state()

    def boom(_state):
        raise RuntimeError("the evidence reader exploded")

    saved = V.GATES["geometry"]
    V.GATES["geometry"] = boom
    try:
        results = V.run_gates(state, names=["geometry"])
    finally:
        V.GATES["geometry"] = saved
    assert results[0].status == NOT_EVALUATED and "exploded" in results[0].limitations[0]


# ---------------------------------------------------------------------------------------------------- presets
def test_every_preset_builds_and_validates():
    from pinneapple_pipeline.presets import PRESETS, preset_config
    for name in PRESETS:
        cfg = preset_config(name)
        assert cfg.name == name
        assert SurrogatePipeline(cfg).dry_run()["ok"], name


def test_a_pipeline_saves_itself_as_the_yaml_it_reads(tmp_path):
    p = SurrogatePipeline.from_preset("burgers_in_viscosity", execution={"out_dir": str(tmp_path)})
    path = p.save(tmp_path / "study.yaml")
    again = SurrogatePipeline.load(path)
    assert again.config.to_dict() == p.config.to_dict()


def test_a_dataset_from_a_file_is_trained_on(tmp_path):
    """``data.source: file``: a CSV of cases, no reference solve at all."""
    import csv
    rows = [{"nu": float(v), "a": float(np.sin(12 * v)), "b": float(np.cos(9 * v))}
            for v in np.linspace(0.005, 0.06, 14)]
    path = tmp_path / "cases.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["nu", "a", "b"])
        writer.writeheader()
        writer.writerows(rows)
    cfg = SurrogateConfig.from_dict({
        "name": "from_file", "problem": {"source": "preset", "preset": "burgers_1d"},
        "data": {"source": "file", "path": str(path), "inputs": ["nu"], "outputs": ["a", "b"]},
        "parameters": {"parameters": {"nu": [0.005, 0.06]}},
        "architecture": {"family": "ridge", "candidates": []},
        "training": {"budget_seconds": 0.3},
        "verification": {"fail_policy": "report"},
        "export": {"enabled": False, "formats": []},
        "execution": {"out_dir": str(tmp_path)},
    })
    result = SurrogatePipeline(cfg).run()
    assert result.gate("dataset").status in (PASS, WARN)
    assert result.gate("simulation").status == PASS          # read from the recorded summary
    assert result.model is not None


def test_the_run_can_narrate_its_stages(tmp_path, capsys):
    """``run(verbose=True)`` prints one line per stage: what it did and how long it took."""
    p = SurrogatePipeline.from_preset(
        "burgers_in_viscosity", execution={"out_dir": str(tmp_path), "resume": False},
        sampling={"design_sampling": {"method": "lhs", "n_samples": 12},
                  "validation_sampling": {"test_fraction": 0.25, "val_fraction": 0.12, "shift": "auto"}, "seed": 1},
        architecture={"family": "ridge", "candidates": []}, training={"budget_seconds": 0.5},
        verification={"mode": "balanced", "fail_policy": "report"})
    p.run(verbose=True)
    printed = capsys.readouterr().out
    for stage in ("problem", "data", "preprocess", "model", "training", "export", "verification"):
        assert f"[{stage:<12}]" in printed
    assert "verification" in printed and "pass" in printed
