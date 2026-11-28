"""The design-space pipeline stages that exist so far: sampling, the proof, the card and the registry."""
from __future__ import annotations

import json

import pytest
import torch

from pinneapple_neural.lpm import Design, LargePhysicsModel, ellipse_design_space, fit
from pinneapple_neural.lpm_stages import card as card_stage
from pinneapple_neural.lpm_stages import registry, space, trust


def _trained(tmp_path=None):
    """A small model that has actually learnt something, for the proof stage to report on."""
    designs = ellipse_design_space(10, n_points=32, seed=3)
    torch.manual_seed(0)
    model = LargePhysicsModel(dim=2, n_fields=1, radii=(0.2, 0.8), width=32, n_heads=3, backbone="attention",
                              attn_heads=2, attn_layers=1)
    fit(model, designs[:6], epochs=60, batch_designs=3, lr=3e-3)
    model.eval()
    return model, designs[:6], designs[6:]


# ------------------------------------------------------------------ stage 1: the space
def test_sobol_sampling_stays_inside_the_family_ranges():
    from pinneapple_neural.lpm_pipeline import families

    family = families()["pp_fin"]
    specs = space.sample(family, 12, 2, method="sobol", seed=1)
    assert len(specs) == 24
    for spec in specs:
        for key, (lo, hi) in family.geometry_params.items():
            assert lo <= spec.geometry[key] <= hi
        for key, (lo, hi) in family.operating_params.items():
            assert lo <= spec.operating[key] <= hi
        assert family.validate(spec) == ""
    # the lattice is deterministic and does not repeat a point
    again = space.sample(family, 12, 2, method="sobol", seed=1)
    assert [s.geometry for s in specs] == [s.geometry for s in again]
    assert len({tuple(round(v, 9) for v in s.geometry.values()) for s in specs}) > 1


def test_sampling_redraws_a_geometry_the_family_refuses():
    """``pp_exchanger3d`` refuses a plate whose bores touch: the sampler must not hand it to the reference."""
    from pinneapple_neural.lpm_families_3d import families_3d

    family = families_3d()["pp_exchanger3d"]
    specs = space.sample(family, 6, 1, method="lhs", seed=0)
    assert len(specs) == 6
    assert all(family.validate(s) == "" for s in specs)


def test_an_unknown_sampler_is_refused():
    with pytest.raises(ValueError, match="unknown sampling method"):
        space.unit(4, 2, "chebyshev", 0)


def test_refine_proposes_where_confidence_sits_at_the_threshold():
    from pinneapple_neural.lpm_pipeline import families

    family = families()["pp_fin"]
    specs = space.sample(family, 8, 1, method="sobol", seed=2)
    scores = [0.05, 0.9, 0.48, 0.2, 0.95, 0.52, 0.1, 0.3]
    proposals = space.refine(scores, specs, 3, geometry_params=family.geometry_params, seed=0)
    assert len(proposals) == 3
    assert all("threshold" in p.reason for p in proposals)
    # the first proposal comes from the design nearest the decision boundary, not from the worst one
    assert proposals[0].source == specs[2].label
    for proposal in proposals:
        for key, (lo, hi) in family.geometry_params.items():
            assert lo <= proposal.geometry[key] <= hi


# ------------------------------------------------------------------ stage 6: the proof
def test_the_confidence_of_a_training_design_is_high_and_of_a_foreign_body_is_not():
    model, train, _ = _trained()
    calibration = trust.calibrate(model, train)
    inside = trust.confidence(model, train[0], calibration)
    outside = trust.confidence(model, Design.square(3.0, n=32), calibration)
    assert inside["confidence"] > 0.5, inside
    assert outside["confidence"] < inside["confidence"], (inside, outside)
    assert calibration.knots[0][0] == 0.0 and calibration.knots[-1][1] <= 0.05
    assert "raw OOD score -> confidence" in calibration.table()


def test_the_held_out_report_puts_the_baseline_beside_the_model():
    model, train, hold = _trained()
    report = trust.held_out_report(model, type("F", (), {"tolerance": 0.05})(), hold, train)
    for key in ("relative_l2", "relative_l2_p90", "constant_field_baseline_relative_l2", "within_tolerance",
                "beats_baseline", "per_design", "calibration"):
        assert key in report
    assert len(report["per_design"]) == len(hold)
    assert report["beats_baseline"] is True                  # a fitted model has to beat the mean field on its own space
    assert report["designs_above_baseline"] < len(hold)


def test_the_gate_table_says_what_ran_and_what_did_not():
    model, train, hold = _trained()
    report = trust.held_out_report(model, type("F", (), {"tolerance": 0.05})(), hold, train)
    gates = {g["check"]: g for g in trust.point_gates(report, object())}
    assert gates["accuracy"]["value"] == report["relative_l2"]
    for name in ("conservation", "boundary", "residual"):
        assert gates[name]["status"] == "not applicable"
        assert "quadrature" in gates[name]["detail"]


# ------------------------------------------------------------------ stage 7: the card and the index
def _card(tmp_path, name="pp_ellipse", usage="internal-benchmark", licence="proprietary", rel=0.04, base=0.6):
    manifest = tmp_path / name / "dataset" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"n_designs": 6, "n_geometries": 6, "conditions_per_geometry": 1,
                                    "family": {"reference": {"name": "closed form"}}, "local": {"radii": [0.2, 0.8]}}))
    designs = ellipse_design_space(3, n_points=16, seed=1)
    report = {"relative_l2": rel, "relative_l2_p90": rel * 1.4, "constant_field_baseline_relative_l2": base,
              "within_tolerance": rel <= 0.05, "beats_baseline": rel < base, "per_design": [],
              "calibration": trust.Calibration(knots=[(0.0, 1.0), (1.0, 0.9)]).to_dict()}
    built = card_stage.build_card(family_info={"name": name, "tolerance": 0.05}, report=report,
                                  config={"epochs": 60, "width": 32}, manifest=json.loads(manifest.read_text()),
                                  manifest_path=manifest, train_designs=designs,
                                  gates=[{"check": "accuracy", "status": "passed"}], licence=licence, usage=usage)
    return card_stage.write_card(built, tmp_path / name / "model" / "card.json")


def test_the_card_carries_the_dataset_identity_and_the_distribution(tmp_path):
    path = _card(tmp_path)
    written = card_stage.read_card(path)
    assert written["schema"] == card_stage.CARD_SCHEMA
    assert written["dataset"]["id"].startswith("sha256:")
    assert written["distribution"]["n"] == 3
    assert written["trust"]["decision"] == "not_verified_until_independent_review"
    assert card_stage.dataset_id(tmp_path / "pp_ellipse" / "dataset" / "manifest.json") == written["dataset"]["id"]


def test_a_non_commercial_licence_cannot_be_declared_client_delivery(tmp_path):
    with pytest.raises(ValueError, match="non-commercial"):
        _card(tmp_path, licence="cc-by-nc-4.0", usage="client-delivery")
    ok = card_stage.read_card(_card(tmp_path, name="pp_wing_shift", licence="cc-by-nc-4.0",
                                    usage="validation-only"))
    assert ok["licence"] == {"data": "cc-by-nc-4.0", "usage": "validation-only"}


def test_the_registry_reads_the_cards_it_finds_and_picks_the_best_one(tmp_path):
    _card(tmp_path, name="a", rel=0.20, base=0.60)
    _card(tmp_path, name="b", rel=0.04, base=0.60)
    _card(tmp_path, name="c", rel=0.50, base=0.60)          # beats the baseline, misses the tolerance
    rows = registry.entries(tmp_path)
    assert {r["family"] for r in rows} == {"a", "b", "c"}
    table = registry.table(rows)
    assert "held-out" in table and "pp_ellipse" not in table
    usable = registry.usable(rows)
    assert {r["family"] for r in usable} == {"b"}          # a beats the baseline but misses the tolerance
    assert registry.best(rows)["family"] == "b"
    assert registry.best(rows, family="a") is None      # a model that misses its tolerance is not "the best"
    assert registry.summary(rows)["n_usable"] == 1
