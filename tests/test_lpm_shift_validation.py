"""The outside corpus: what it reads, what it refuses, and what the number it produces is allowed to say."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from pinneapple_data.ashton.vtk_xml import write_surface
from pinneapple_neural.lpm import LargePhysicsModel, fit
from pinneapple_neural.lpm_pipeline import FieldFamily, Reference
from pinneapple_neural.lpm_stages import shift


def _plane(nu: int = 9, nv: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A flat quad grid in ``[0,1]^2`` at ``z = 0``, its cells, and the Cp on both quadratures."""
    xs, ys = np.linspace(0.0, 1.0, nu), np.linspace(0.0, 1.0, nv)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)
    quads = []
    for j in range(nv - 1):
        for i in range(nu - 1):
            row = j * nu + i
            quads.append([row, row + 1, row + nu + 1, row + nu])
    connectivity = np.asarray(quads, dtype=np.int64).ravel()
    offsets = np.arange(0, connectivity.size + 1, 4, dtype=np.int64)
    x, y, _ = points.T
    cp_points = 1.0 - ((x - 0.5) ** 2 + (y - 0.5) ** 2)
    cp_cells = cp_points[np.asarray(quads)].mean(axis=1)
    return points, connectivity, offsets, cp_points, cp_cells


def _case(directory, *, cell_field: bool = False, name: str = "CpMeanTrim") -> None:
    """One case directory in the layout Luminary publishes."""
    points, connectivity, offsets, cp_points, cp_cells = _plane()
    write_surface(directory / "merged_surfaces.vtp", points=points, connectivity=connectivity, offsets=offsets,
                  point_data=None if cell_field else {name: cp_points},
                  cell_data={name: cp_cells} if cell_field else None)
    (directory / "params.json").write_text(json.dumps({"alpha_deg": 4.0, "mach": 0.3,
                                                       "velocity": {"u": 1.0, "v": 0.0}}))
    (directory / "forces.json").write_text(json.dumps({"Cd": 0.031, "Cl": 0.42}))
    (directory / "metadata.json").write_text(json.dumps({"solver": "synthetic"}))


def _tree(root) -> None:
    """Six cases across four of the six naming schemes the repositories actually use."""
    for i in (1, 2, 3, 4):
        _case(root / "OnShape_luminary_crm_version001" / f"sample_{i:06d}", cell_field=(i == 2))
    _case(root / "RUN_7")
    _case(root / "yaris_x00001")


class _Plate(FieldFamily):
    name = "pp_test_plate"
    dim = 3
    field_names = ("Cp",)
    field_units = ("-",)
    operating_params = {"alpha": (-5.0, 10.0)}
    reference = Reference("synthetic", "a plane with a quadratic Cp", ("nothing is solved here",))
    field_scale = (2.0,)          # the dataset stores the target divided by this, and so does an outside corpus
    tolerance = 0.05


def _designs(root, limit=None, **kwargs):
    return shift.designs(root, _Plate(), {"alpha": "alpha_deg"}, dataset_name="Wing-sample", max_points=24,
                         limit=limit, **kwargs)


# ------------------------------------------------------------------ reading what is on disk
def test_the_walk_finds_a_case_by_its_surface_file_under_every_naming_scheme(tmp_path):
    _tree(tmp_path)
    found = shift.samples(tmp_path, "Wing-sample")
    # sorted by path, so the walk is deterministic without depending on the order the filesystem lists a directory
    assert [s.path for s in found] == sorted(s.path for s in found)
    assert sorted(s.path.name for s in found) == ["RUN_7", "sample_000001", "sample_000002", "sample_000003",
                                                  "sample_000004", "yaris_x00001"]
    assert {s.dataset.name for s in found} == {"Wing-sample"}
    assert all(shift.CASE_SURFACE in s.files for s in found)
    assert found[0].forces["Cd"] == pytest.approx(0.031)
    assert found[0].params["velocity.u"] == pytest.approx(1.0)          # nested keys keep their path


def test_a_directory_without_the_surface_file_says_so_and_lists_what_is_there(tmp_path):
    empty = tmp_path / "sample_000009"
    empty.mkdir()
    (empty / "params.json").write_text("{}")
    with pytest.raises(shift.ShiftDataMissing, match="merged_surfaces.vtp"):
        shift.open_sample(empty)
    with pytest.raises(shift.ShiftDataMissing, match="is not a directory"):
        shift.samples(tmp_path / "nope", "Wing-sample")


def test_the_dataset_name_comes_from_the_path_and_carries_the_licence(tmp_path):
    _case(tmp_path / "luminary-shift" / "Pump-sample" / "sample_000001")
    found = shift.samples(tmp_path / "luminary-shift" / "Pump-sample")
    assert found[0].dataset.name == "Pump-sample"
    assert found[0].dataset.licence == "cc-by-nc-4.0"
    assert found[0].dataset.commercial is False
    assert shift.dataset("pump-sample").measured_bytes == 72_500_000_000   # measured, not estimated
    assert shift.dataset("Submarine-sample").commercial is False           # the sample twin is cc-by-nc
    assert shift.dataset("Submarine").licence == "other"                   # and the full one is not assumed
    with pytest.raises(ValueError, match="unknown SHIFT dataset"):
        shift.dataset("AhmedML")              # a dataset this corpus does not have, not a case variant of one


def test_the_field_is_chosen_by_preference_and_a_missing_one_lists_the_arrays(tmp_path):
    _case(tmp_path / "a")
    _case(tmp_path / "b", name="wallShear")
    good = shift.open_sample(tmp_path / "a", "Wing-sample")
    values, location = good.field()
    assert location == "point" and values.shape == (45,)
    bad = shift.open_sample(tmp_path / "b", "Wing-sample")
    with pytest.raises(ValueError, match="wallShear"):
        bad.field()


# ------------------------------------------------------------------ the design the model sees
def test_a_case_becomes_a_design_with_normalised_conditions(tmp_path):
    _case(tmp_path / "a")
    sample = shift.open_sample(tmp_path / "a", "Wing-sample")
    d = shift.design(sample, _Plate(), {"alpha": "alpha_deg"}, max_points=16)
    assert d.query.shape == d.points.shape == (16, 3)
    assert d.target.shape == (16, 1)
    assert d.points.dtype == torch.float32
    # the target arrives in physical units and is divided by the family's field scale, as the dataset writer does
    expected = _plane()[3][shift._subset(45, 16)] / 2.0
    assert torch.allclose(d.target.ravel(), torch.tensor(expected, dtype=torch.float32), atol=1e-6)
    assert float(d.params[0]) == pytest.approx(0.6, abs=1e-6)         # 4.0 in [-5, 10]
    assert "#16of45" in d.label                                      # the subsample is on the record
    # the normals come from the mesh, not from an assumption: this plate faces +z
    assert float(d.normals[:, 2].min()) == pytest.approx(1.0)


def test_a_cell_located_field_is_read_at_the_cell_centres(tmp_path):
    _case(tmp_path / "a", cell_field=True)
    sample = shift.open_sample(tmp_path / "a", "Wing-sample")
    d = shift.design(sample, _Plate(), {"alpha": "alpha_deg"}, max_points=64)
    assert d.target.shape[0] == 32                       # 8 x 4 quads, all of them kept
    assert d.source["location"] == "cell"


def test_a_condition_the_mapping_does_not_cover_names_itself(tmp_path):
    _case(tmp_path / "a")
    sample = shift.open_sample(tmp_path / "a", "Wing-sample")
    with pytest.raises(KeyError, match="alpha_deg"):
        shift.design(sample, _Plate(), {"alpha": "aoa"})


# ------------------------------------------------------------------ what the number may be used for
def test_only_a_validation_usage_can_be_declared():
    assert shift.require_validation_usage("validation-only") == "validation-only"
    for usage in ("client-delivery", "internal-benchmark"):
        with pytest.raises(ValueError, match="cc-by-nc-4.0"):
            shift.require_validation_usage(usage)
    licence = shift.card_licence("Wing-sample")
    assert licence["usage"] == "validation-only" and licence["data"] == "cc-by-nc-4.0"
    assert licence["url"].endswith("luminary-shift/Wing-sample")


def test_the_report_scores_an_outside_case_beside_the_constant_field(tmp_path):
    _tree(tmp_path)
    designs = _designs(tmp_path)
    torch.manual_seed(0)
    # n_params=1 because the plate conditions on one number (alpha): a model built with none cannot take a design
    # that carries one, and the failure would surface as a shape error inside the trunk rather than as the mismatch
    model = LargePhysicsModel(dim=3, n_fields=1, n_params=1, radii=(0.2, 0.8), width=24, n_heads=2,
                              backbone="attention", attn_heads=2, attn_layers=1)
    fit(model, designs[:4], epochs=40, batch_designs=2, lr=3e-3)
    model.eval()

    out = shift.report(model, _Plate(), designs[4:], train_designs=designs[:4], dataset_name="Wing-sample")
    assert out["outside"]["dataset"] == "Wing-sample"
    assert out["outside"]["usage"] == "validation-only"
    assert out["outside"]["licence"] == "cc-by-nc-4.0"
    assert "not trained on these bodies" in out["outside"]["note"]
    held = out["held_out"]
    assert held["n_holdout_designs"] == len(designs) - 4
    assert math.isfinite(held["relative_l2"]) and math.isfinite(held["constant_field_baseline_relative_l2"])
    assert {g["check"] for g in out["gates"]} >= {"accuracy", "baseline"}
    assert out["licence"]["source"] == "luminary-shift/Wing-sample"


# ------------------------------------------------------------------ the comparison harness
def test_the_harness_puts_the_model_and_the_constant_field_beside_each_other(tmp_path):
    from benchmarks.lpm_compare import LPMMember, compare, table

    _tree(tmp_path)
    designs = _designs(tmp_path)
    torch.manual_seed(0)
    model = LargePhysicsModel(dim=3, n_fields=1, n_params=1, radii=(0.2, 0.8), width=24, n_heads=2,
                              backbone="attention", attn_heads=2, attn_layers=1)
    fit(model, designs[:4], epochs=40, batch_designs=2, lr=3e-3)
    model.eval()

    member = LPMMember(model, _Plate())
    rows = compare([member], designs[4:], train_designs=designs[:4], tolerance=_Plate.tolerance,
                   scales=_Plate().scales)
    assert len(rows) == 1                                   # the constant field is the baseline column, not a row
    row = rows[0]
    assert math.isfinite(row.relative_l2) and row.baseline is not None
    assert row.n_designs == 2
    assert set(row.gates()) == {"accuracy", "baseline", "conservation", "boundary", "residual"}
    assert row.gates()["conservation"].startswith("not applicable")
    assert len(member.reports) == 2                          # the OOD flag is recorded for every design scored
    text = table(rows, corpus="shift fixture")
    # the constant field is a column, not a row: it is the same number for every model, and what a reader needs is
    # the model's error beside it
    assert "shift fixture" in text and "baseline" in text and "beaten" in text


# ------------------------------------------------------------------ getting the corpus onto the disk
def test_the_download_plan_takes_one_case_and_leaves_the_terabytes():
    files = ["OnShape_v1/sample_000001/merged_surfaces.vtp",
             "OnShape_v1/sample_000001/merged_surfaces.stl",
             "OnShape_v1/sample_000001/merged_surfaces_inter_mesh_0.stl",
             "OnShape_v1/sample_000001/merged_volumes.vtu",
             "OnShape_v1/sample_000001/params.json",
             "OnShape_v1/sample_000001/forces.json",
             "OnShape_v1/sample_000001/metadata.json",
             "OnShape_v1/sample_000002/merged_surfaces.vtp",
             "OnShape_v1/sample_000002/params.json",
             "README.md"]
    plan = shift.fetch_plan(files, 1)
    assert plan == sorted([f for f in files if f.startswith("OnShape_v1/sample_000001/")
                           and f.rsplit("/", 1)[-1] in {"merged_surfaces.vtp", "params.json", "forces.json",
                                                        "metadata.json"}])
    assert not any("volumes" in p or "inter_mesh" in p or p.endswith(".stl") for p in plan)
    two = shift.fetch_plan(files, 2)
    assert sum(1 for p in two if p.endswith("merged_surfaces.vtp")) == 2
    assert shift.fetch_plan([], 1) == []
