"""The ``pinn`` kind of a parametric study (#136): per-sample fields, a mesh-driven fit, and the error against the
reference fields.

Two things are tested and they fail differently. The *refusals* are cheap and exact: a ``pinn`` study whose evaluator
returns no field, or two different meshes, cannot be fitted, and says so before training anything. The *fit* is a
measurement: the slab example trains for a fixed number of optimiser steps a fold (``evaluator.settings["pinn_steps"]``,
not a wall clock), so its held-out numbers are the same on a busy machine as on an idle one, and every held-out
prediction belongs to a design the fold did not train on.
"""
from __future__ import annotations

import json

import pytest

from pinneapple_analysis.design_study import DesignStudy, Model, run_study
from pinneapple_analysis.design_study.examples import slab_cooling, slab_study


@pytest.fixture(scope="module")
def run():
    return run_study(slab_study(), slab_cooling)


# ---------------------------------------------------------------------------------------------- the fit, measured
def test_the_slab_field_is_fitted_by_a_pinn_and_the_error_is_against_the_reference(run):
    assert run.status == "done" and run.trust["decision"] == "APPROVED", run.reasons
    assert run.spec.kind == "pinn"
    p = run.surrogate["pinn"]
    assert p["method"] == "pinn" and p["family"] == "physics-informed"
    assert p["pde"]["type"] == "heat"
    # the field, the mesh and the equation the fit was informed by
    assert p["mesh"]["field"] == "T" and p["mesh"]["coords"] == ["x", "t"]
    assert p["mesh"]["grid_shape"] == [7, 5] and p["mesh"]["n_points"] == 7 * 5
    assert p["n_samples"] == 13                                  # 1 baseline + 12 DOE designs
    assert p["folds"] == 3 and p["n_heldout"] == 13              # every design was held out exactly once

    # the error is against the reference fields, on designs the fold did not train on, and it is inside the target
    e = p["error"]
    assert e["rel_l2_mean"] <= slab_study().verifier.surrogate_target
    assert e["rel_l2_mean"] < 0.20
    assert e["rel_l2_median"] <= e["rel_l2_max"]
    assert p["train_s_mean"] > 0
    # the training budget is in steps, which is what makes the number above the same on a busy machine
    assert p["budget"] == {"steps": slab_study().evaluator.settings["pinn_steps"], "wall_s": None}

    # the trust verdict reads the field check, not only the scalar outputs
    names = [c["name"] for c in run.trust["checks"]]
    field_check = next(c for c in run.trust["checks"] if "PINN held-out relative L2 of T" in c["name"])
    assert field_check["critical"] and field_check["value"] == pytest.approx(e["rel_l2_mean"])
    assert "held-out prediction" in field_check["source"]
    assert any("surrogate held-out nRMSE of t_end_max" in n for n in names)

    # the mesh the residual was measured on is the mesh of the study, and the reference floor is reported with it
    r = p["residual"]
    assert r is not None and r["data_mean"] > 0 and r["pred_mean"] > 0 and r["ratio"] == pytest.approx(
        r["pred_mean"] / r["data_mean"])


def test_every_held_out_prediction_belongs_to_a_design_its_fold_did_not_train_on(run):
    p = run.surrogate["pinn"]
    folds = [d["fold"] for d in p["per_sample"]]
    designs = [d["design"] for d in p["per_sample"]]
    assert sorted(designs) == list(range(p["n_samples"]))       # each design predicted once, none twice
    assert len(set(folds)) == p["folds"]
    for d in p["per_sample"]:
        assert 0 <= d["rel_l2"] < 2.0                          # a number per design, not the metric repeated
    per_fold = {f: sorted(d["design"] for d in p["per_sample"] if d["fold"] == f) for f in set(folds)}
    assert sum(len(v) for v in per_fold.values()) == p["n_samples"]


def test_the_pinn_result_enters_the_report(run):
    page = run.to_markdown()
    assert "## Physics-informed field" in page
    p = run.surrogate["pinn"]
    assert f"{100 * p['error']['rel_l2_mean']:.2f} %" in page
    assert f"| PDE | `{p['pde']['type']}` |" in page
    assert any("The PINN was fitted to T on 35 mesh point(s)" in n for n in run.notes)
    doc = json.loads(json.dumps(run.to_dict()))                # the report is a document, not a Python object
    assert doc["surrogate"]["pinn"]["folds"] == 3
    assert doc["summary"]["kind"] == "pinn"


def test_a_pinn_study_is_one_round_because_it_has_no_recommendation_to_measure(run):
    assert run.stop["reason"] == "single_pass"
    assert len(run.loop["rounds"]) == 1
    assert run.optimizer == {}                                 # nothing was optimised on the field model


# ---------------------------------------------------------------------------------------------- the refusals
def _spec(**over):
    s = slab_study()
    d = dict(kind=s.kind, title=s.title, base=s.base, evaluator=s.evaluator, objectives=s.objectives,
             constraints=s.constraints, budget=s.budget, verifier=s.verifier, model=s.model,
             statement=s.statement, metadata=s.metadata)
    d.update(over)
    return type(s)(**d)


def test_an_evaluator_that_returns_no_field_is_refused_before_anything_is_trained():
    run = run_study(slab_study(), lambda p: {"t_end_max": 1.0, "t_centre_end": 0.5})
    assert run.status == "refused"
    assert "needs per-sample fields" in run.reasons[0]
    assert "__fields__" in run.reasons[0] and "__coords__" in run.reasons[0]
    assert "pinn" not in run.surrogate


def test_two_meshes_in_one_study_are_refused_with_the_design_that_differs():
    def shifting(params):
        out = slab_cooling(params)
        out["__coords__"] = {k: v * 1.5 for k, v in out["__coords__"].items()}       # same field, another mesh
        return out

    calls = {"n": 0}

    def after_the_first(params):
        calls["n"] += 1
        return slab_cooling(params) if calls["n"] == 1 else shifting(params)

    run = run_study(slab_study(), after_the_first)
    assert run.status == "refused"
    assert "different mesh" in run.reasons[0]


def test_two_fields_in_one_study_are_refused_because_a_pinn_fits_one_field():
    def two_fields(params):
        out = slab_cooling(params)
        out["__fields__"] = {"T": out["__fields__"]["T"], "q": out["__fields__"]["T"] * 2.0}
        return out

    run = run_study(slab_study(), two_fields)
    assert run.status == "refused" and "one scalar field per study" in run.reasons[0]


def test_a_pinn_study_without_the_equation_is_refused_by_the_schema():
    with pytest.raises(ValueError, match="model.pde"):
        _spec(model=Model(families=("gp", "quadratic")))


def test_a_field_whose_mesh_is_not_a_lattice_is_still_fitted_and_says_the_residual_was_not_measured():
    """The fit needs only coordinates; the finite-difference residual needs a tensor-product mesh and says so."""
    import numpy as np

    spec = _spec(budget={"max_evaluations": 8, "n_doe": 7, "n_verify": 0, "surrogate_evaluations": 50, "seed": 2},
                 verifier={"surrogate_target": 0.9, "min_verified": 0, "folds": 3},
                 evaluator={"tool": "heat.slab_analytic", "fidelity": "model",
                            "inputs": {"alpha": "alpha", "amplitude": "amplitude"},
                            "conditions": {"length_m": 1.0, "end_time_s": 0.1},
                            "outputs": [{"name": "t_end_max", "unit": "K"}],
                            "settings": {"pinn_steps": 40, "pinn": {"depth": 2, "hidden": 8}}})
    assert isinstance(spec, DesignStudy)

    def scattered(params):
        import math
        a, al = float(params["amplitude"]), float(params["alpha"])
        rng = np.random.default_rng(0)
        pts = np.stack([rng.uniform(0, 1, 20), rng.uniform(0, 0.1, 20)], 1)   # 20 points, not a lattice
        u = a * np.exp(-al * math.pi ** 2 * pts[:, 1]) * np.sin(math.pi * pts[:, 0])
        return {"t_end_max": float(u.max()), "__fields__": {"T": u},
                "__coords__": {"x": pts[:, 0], "t": pts[:, 1]}}

    run = run_study(spec, scattered)
    p = run.surrogate["pinn"]
    assert p["mesh"]["grid_shape"] is None                     # not a tensor-product mesh
    assert p["residual"] is None                               # and the FD residual is not invented
    assert p["error"]["rel_l2_mean"] == p["error"]["rel_l2_mean"]          # a number, not nan
