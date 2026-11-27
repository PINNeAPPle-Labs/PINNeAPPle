"""CFDBench: the layout the project's own loader reads, one case at a time.

The fixture writes what the archive holds -- ``u.npy``/``v.npy`` shaped ``(time, height, width)``, a ``mask.npy``
where the case has one and ``case.json`` with the case's parameters -- because the archive is 2.3 GB of zips and the
layout is what the reader has to follow. What the reader must not do is invent what the archive does not state, so
the tests pin the units being ``unknown`` and the coordinates being the case's own extent or the cell index.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from pinneapple_data.datasets import cfdbench
from pinneapple_data.schema import run_gates

NT, NY, NX = 5, 8, 10


def write_case(root, problem="tube", subset="bc", name="case0000", *, extent=True, mask=False, nt=NT):
    case = root / problem / subset / name
    case.mkdir(parents=True, exist_ok=True)
    t = np.arange(nt, dtype=np.float64).reshape(-1, 1, 1)
    u = np.broadcast_to(np.linspace(0.0, 0.1, NX), (nt, NY, NX)).copy() + 0.01 * t
    v = np.zeros((nt, NY, NX)) + 0.001 * t
    np.save(case / "u.npy", u)
    np.save(case / "v.npy", v)
    if mask:
        m = np.zeros((NY, NX))
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 1.0
        np.save(case / "mask.npy", m)
    params = {"vel_in": 0.1, "density": 100.0, "viscosity": 0.1}
    if extent:
        params |= {"width": 1.0, "height": 0.1}
    (case / "case.json").write_text(json.dumps(params), encoding="utf-8")
    return case


def test_a_case_becomes_a_bundle_with_its_velocity_and_parameters(tmp_path):
    write_case(tmp_path)
    case = next(cfdbench.open_cases(tmp_path))
    bundle = cfdbench.load_bundle(case, dt=0.1)
    assert bundle.field_array("velocity").shape == (NT, NY, NX, 2)
    field = bundle.manifest.fields.get("velocity")
    assert field.components == ["u", "v"] and field.axes == ["t", "y", "x"]
    # every number in case.json is a parameter of the case: the fluid's and the domain's
    assert [p.name for p in bundle.manifest.physics.parameters] == ["density", "height", "vel_in", "viscosity", "width"]
    assert bundle.manifest.provenance.license == "Apache-2.0"
    assert {i.path.split("/")[-1] for i in bundle.manifest.provenance.inputs} == {"u.npy", "v.npy", "case.json"}


def test_the_time_axis_is_seconds_with_dt_and_the_frame_index_without_it(tmp_path):
    write_case(tmp_path)
    case = next(cfdbench.open_cases(tmp_path))
    with_dt = cfdbench.load_bundle(case, dt=0.1)
    without = cfdbench.load_bundle(case)
    assert np.allclose(with_dt.axis_array("t"), np.arange(NT) * 0.1)
    assert with_dt.manifest.mesh.axis("t").unit == "s"
    assert without.manifest.mesh.axis("t").unit == "1"
    assert np.allclose(without.axis_array("t"), np.arange(NT))


def test_the_coordinates_come_from_the_extent_the_case_states(tmp_path):
    write_case(tmp_path)
    bundle = cfdbench.load_bundle(next(cfdbench.open_cases(tmp_path)))
    x, y = bundle.axis_array("x"), bundle.axis_array("y")
    assert len(x) == NX and len(y) == NY
    assert x[0] == pytest.approx(0.5 / NX) and x[-1] == pytest.approx(1.0 - 0.5 / NX)      # width = 1
    assert y[-1] == pytest.approx(0.1 - 0.05 / NY)                                        # height = 0.1
    assert bundle.manifest.mesh.axis("x").unit == "unknown"        # a length the archive does not name
    assert "case.json 0..width" in bundle.manifest.metadata["cfdbench"]["coordinates"]["x"]


def test_without_an_extent_the_axis_is_the_cell_index(tmp_path):
    write_case(tmp_path, extent=False)
    bundle = cfdbench.load_bundle(next(cfdbench.open_cases(tmp_path)))
    assert np.allclose(bundle.axis_array("x"), np.arange(NX))
    assert bundle.manifest.mesh.axis("x").unit == "1"
    assert "cell index" in bundle.manifest.metadata["cfdbench"]["coordinates"]["x"]


def test_the_mask_becomes_a_field_when_the_case_ships_one(tmp_path):
    write_case(tmp_path, mask=True)
    bundle = cfdbench.load_bundle(next(cfdbench.open_cases(tmp_path)))
    mask = bundle.field_array("mask")
    assert mask.shape == (NT, NY, NX) and mask[0, 0, 0] == 1.0 and mask[0, 1, 1] == 0.0


def test_the_units_are_unknown_and_the_gate_says_so_until_they_are_declared(tmp_path):
    write_case(tmp_path)
    case = next(cfdbench.open_cases(tmp_path))
    report = run_gates(cfdbench.load_bundle(case))
    assert report.get("units").passed is False
    assert "unknown unit" in report.get("units").detail
    declared = cfdbench.load_bundle(case, units={"velocity": "m/s", "x": "m", "y": "m"})
    assert run_gates(declared).get("units").passed is True


def test_both_nestings_the_archives_use_are_read(tmp_path):
    """``<root>/<problem>/<subset>/case`` when several problems are unpacked, ``<root>/<subset>/case`` when one is."""
    write_case(tmp_path, problem="tube")
    write_case(tmp_path, problem="dam", subset="prop", name="case0001")
    assert {(c.problem, c.subset) for c in cfdbench.open_cases(tmp_path)} == {("tube", "bc"), ("dam", "prop")}

    single = tmp_path / "single"                      # one problem unpacked: the subsets sit at the top level
    inner = write_case(tmp_path / "elsewhere", problem="tube", subset="geo")
    single.mkdir()
    shutil.copytree(inner.parent, single / "geo")
    cases = list(cfdbench.open_cases(single))
    assert len(cases) == 1 and cases[0].subset == "geo" and cases[0].problem == "single"


def test_the_subset_and_problem_filters_are_honoured(tmp_path):
    write_case(tmp_path, problem="tube", subset="bc")
    write_case(tmp_path, problem="tube", subset="geo")
    write_case(tmp_path, problem="dam", subset="bc")
    assert [c.problem for c in cfdbench.open_cases(tmp_path, problem="tube")] == ["tube", "tube"]
    assert {c.subset for c in cfdbench.open_cases(tmp_path, subset="geo")} == {"geo"}


def test_u_and_v_that_do_not_pair_are_refused(tmp_path):
    case = write_case(tmp_path)
    np.save(case / "v.npy", np.zeros((NT, NY, NX + 1)))
    with pytest.raises(ValueError, match="not two equal"):
        cfdbench.load_bundle(next(cfdbench.open_cases(tmp_path)))


def test_a_real_case_shape_is_what_the_archive_stores(tmp_path):
    """The published tube cases are (107, 64, 64); the reader takes any time count and any grid."""
    write_case(tmp_path, nt=107)
    case_path = tmp_path / "tube" / "bc" / "case0000"
    for name in ("u", "v"):
        values = np.load(case_path / f"{name}.npy")
        np.save(case_path / f"{name}.npy", np.resize(values, (107, 64, 64)))
    bundle = cfdbench.load_bundle(next(cfdbench.open_cases(tmp_path)))
    assert bundle.field_array("velocity").shape == (107, 64, 64, 2)


def test_a_missing_directory_says_where_to_get_the_archive(tmp_path):
    with pytest.raises(FileNotFoundError, match="huggingface"):
        list(cfdbench.open_cases(tmp_path / "nope"))
