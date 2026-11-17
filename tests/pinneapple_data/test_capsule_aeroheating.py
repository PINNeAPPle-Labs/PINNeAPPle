"""The capsule-aeroheating reader: pairing the two fidelities, the units it translates, and the bundle it builds.

The pairing, the scaling of the input channels and the unit translation are tested on a stand-in study, so they run
without ``bijective-gamut`` installed. Reading a file is tested on a small one written here by ``gamut`` itself, which
is skipped when the reader is absent: the point of that test is the reader's own layout, not a 77 MB download.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from pinneapple_data.datasets import capsule_aeroheating as cap
from pinneapple_data.schema import run_gates

# ---------------------------------------------------------------------------
# a stand-in study: the two methods below are what the module reads
# ---------------------------------------------------------------------------

class _FakeStudy:
    """The parts of ``gamut.Study`` that read the table and the bounds of a parameter."""

    def __init__(self, members, fidelities, bounds=None, values=None):
        self._members = np.asarray(members, dtype=str)
        self._fidelities = np.asarray(fidelities, dtype=str)
        self._bounds = bounds or {}
        self._values = values or {}
        self.row_count = int(self._members.size)

    def table(self, names=None, rows=None):
        table = {"member": self._members, "fidelity": self._fidelities, **self._values}
        picked = {name: np.asarray(table[name])[slice(None) if rows is None else rows] for name in (names or table)}
        return picked

    def describe(self, name):
        return self._bounds.get(name, {})


def _bounds_for_all(low=0.0, high=1.0):
    return {name: {"lower": low, "upper": high} for name in cap.PARAMETERS}


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reported, expected", [
    ("W m-2", "W/(m^2)"),
    ("m s-1", "m/s"),
    ("kg m-2 s-1", "kg/(m^2*s)"),
    ("J kg-1", "J/kg"),
    ("kg m-3", "kg/(m^3)"),
    ("Pa", "Pa"),
    ("m", "m"),
    ("K", "K"),
    ("km", "km"),
    ("degree", "degree"),
    ("1", "1"),
])
def test_schema_unit_translates_the_format_s_exponent_spelling(reported, expected):
    assert cap.schema_unit(reported) == expected


@pytest.mark.parametrize("reported", ["", "   ", "not a unit", "W m-x", "furlong fortnight-1"])
def test_schema_unit_says_unknown_instead_of_inventing_one(reported):
    assert cap.schema_unit(reported) == "unknown"


def test_the_file_names_and_fidelities_are_the_published_ones():
    assert cap.FILES["wall"] == "vehicle_capsule_multifidelity.gamut"
    assert cap.FILES["volume"] == "vehicle_capsule_eilmer_volume.gamut"
    assert cap.LOW_FIDELITY == "newtonian_panel" and cap.HIGH_FIDELITY == "eilmer_second_order"
    assert set(cap.FIDELITIES) == {cap.LOW_FIDELITY, cap.HIGH_FIDELITY}
    assert cap.PARAMETERS == ("diameter", "nose_radius", "forebody_angle", "shoulder_radius", "mach", "altitude")
    assert cap.LICENSE == "CC-BY-4.0"


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------

def test_pair_fidelities_returns_the_cheap_run_of_each_expensive_one():
    study = _FakeStudy(["c2", "c1", "c2", "c1"],
                       [cap.HIGH_FIDELITY, cap.LOW_FIDELITY, cap.LOW_FIDELITY, cap.HIGH_FIDELITY])
    high, low = cap.pair_fidelities(study)
    assert study._members[high].tolist() == ["c1", "c2"]                # sorted by member
    assert study._members[low].tolist() == ["c1", "c2"]                 # the same member on each row
    assert (study._fidelities[high] == cap.HIGH_FIDELITY).all()
    assert (study._fidelities[low] == cap.LOW_FIDELITY).all()


def test_pair_fidelities_refuses_a_file_with_no_expensive_run():
    study = _FakeStudy(["c1", "c2"], [cap.LOW_FIDELITY, cap.LOW_FIDELITY])
    with pytest.raises(ValueError, match="engineering model only"):
        cap.pair_fidelities(study)


def test_pair_fidelities_refuses_an_expensive_run_without_its_cheap_twin():
    study = _FakeStudy(["c1", "c2", "c3"], [cap.HIGH_FIDELITY, cap.LOW_FIDELITY, cap.LOW_FIDELITY])
    with pytest.raises(ValueError, match="no low-fidelity run of the same member"):
        cap.pair_fidelities(study)


def test_pair_fidelities_refuses_a_member_with_two_cheap_runs():
    study = _FakeStudy(["c1", "c1", "c2"], [cap.LOW_FIDELITY, cap.LOW_FIDELITY, cap.HIGH_FIDELITY])
    with pytest.raises(ValueError, match="not one per member"):
        cap.pair_fidelities(study)


# ---------------------------------------------------------------------------
# input channels
# ---------------------------------------------------------------------------

def test_field_channels_scales_each_input_by_the_bounds_the_file_declares():
    bounds = {"altitude": {"lower": 40.0, "upper": 80.0}, "mach": {"lower": 6.0, "upper": 15.0}}
    bounds.update({name: {"lower": 0.0, "upper": 10.0} for name in cap.PARAMETERS
                   if name not in ("altitude", "mach")})
    study = _FakeStudy(["c1"], [cap.LOW_FIDELITY], bounds=bounds,
                       values={"altitude": np.array([60.0]), "mach": np.array([6.0]),
                               **{name: np.array([5.0]) for name in cap.PARAMETERS if name not in ("altitude", "mach")}})
    names, channels, how = cap.field_channels(study)
    assert names == list(cap.PARAMETERS) and channels.shape == (1, 6)
    row = dict(zip(names, channels[0].tolist(), strict=True))
    assert row["altitude"] == pytest.approx(0.0)                        # the middle of 40..80 km
    assert row["mach"] == pytest.approx(-1.0)                           # the low end of 6..15
    for name in cap.PARAMETERS:
        assert -1.0 <= row[name] <= 1.0
    assert how["bounds"]["mach"] == (6.0, 15.0)


def test_field_channels_refuses_a_parameter_without_bounds_instead_of_using_the_sample_range():
    bounds = _bounds_for_all()
    bounds["mach"] = {}
    study = _FakeStudy(["c1", "c2"], [cap.LOW_FIDELITY] * 2, bounds=bounds,
                       values={name: np.array([0.0, 1.0]) for name in cap.PARAMETERS})
    with pytest.raises(ValueError, match="declares no usable bounds"):
        cap.field_channels(study)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

def test_download_keeps_a_file_already_on_disk(monkeypatch, tmp_path):
    present = tmp_path / cap.FILES["wall"]
    present.write_bytes(b"not really the file")
    monkeypatch.setattr(cap.urllib.request, "urlopen", lambda *a, **k: pytest.fail("fetched over the network"))
    assert cap.download("wall", tmp_path) == present


def test_download_names_the_files_it_knows(tmp_path):
    with pytest.raises(KeyError, match="not one of"):
        cap.download("everything", tmp_path)


# ---------------------------------------------------------------------------
# reading a file the format wrote
# ---------------------------------------------------------------------------

WALL_FIELDS = {"heat_flux": {"units": "W m-2", "at": "node"}, "pressure": {"units": "Pa", "at": "node"},
               "arc_length": {"units": "m", "at": "node"}}


def _write_study(path, *, fidelity_label=True, fidelities=None):
    """A miniature study in the dataset's own layout: a changing wall, two fidelities of one member."""
    gamut = pytest.importorskip("gamut", reason="the reader of the format is not installed")
    fidelities = fidelities or [cap.LOW_FIDELITY, cap.HIGH_FIDELITY, cap.LOW_FIDELITY]
    members = ["c1", "c1", "c2"][: len(fidelities)]
    bounds = {"altitude": (40.0, 80.0), "mach": (6.0, 15.0), "diameter": (0.8, 5.5), "nose_radius": (0.25, 1.6),
              "shoulder_radius": (0.025, 0.075), "forebody_angle": (55.0, 75.0)}
    labels = {"member": {"role": "group"}}
    if fidelity_label:
        labels["fidelity"] = {"role": "categorical", "categories": [cap.LOW_FIDELITY, cap.HIGH_FIDELITY]}
    types, offsets = np.array([5, 5], np.uint8), np.array([0, 3, 6], np.int64)
    connectivity = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
    with gamut.create(path, writer="pinneapple test", mesh="changing", per="run",
                      coordinates={"units": "m", "axes": 3}, title="synthetic capsule study",
                      parameters={name: {"units": "m" if name == "diameter" else "1", "role": "design",
                                         "lower": bounds[name][0], "upper": bounds[name][1]}
                                  for name in cap.PARAMETERS},
                      labels=labels,
                      scalars={"heat_load": "W", "stagnation_heat_flux": "W m-2"},
                      fields=WALL_FIELDS, constants={"wall_temperature": (300.0, "K")}) as study:
        for i, (member, fidelity) in enumerate(zip(members, fidelities, strict=True)):
            points = np.array([[0.0, 0.0, i], [1.0, 0.0, i], [1.0, 1.0, i], [0.0, 1.0, i]])
            values = {"diameter": 1.0 + i, "mach": 8.0, "altitude": 50.0, "nose_radius": 0.5,
                      "shoulder_radius": 0.05, "forebody_angle": 60.0, "member": member,
                      "heat_load": 10.0 + i, "stagnation_heat_flux": 5.0 + i}
            if fidelity_label:
                values["fidelity"] = fidelity
            study.add_run(values, points=points, cells=(types, offsets, connectivity),
                          fields={"heat_flux": np.array([1.0, 2.0, 3.0, 4.0]) * (i + 1),
                                  "pressure": np.ones(4), "arc_length": np.array([0.0, 0.1, 0.2, 0.3])})
    return path


def test_open_study_refuses_a_file_that_labels_no_fidelity(tmp_path):
    path = _write_study(tmp_path / "one.gamut", fidelity_label=False)
    with pytest.raises(ValueError, match="labels no 'fidelity'"):
        cap.open_study(path)


def test_open_study_refuses_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        cap.open_study(tmp_path / "absent.gamut")


def test_wall_reads_one_run_with_its_units_and_its_inputs(tmp_path):
    path = _write_study(tmp_path / "capsule.gamut")
    with cap.open_study(path) as study:
        high, low = cap.pair_fidelities(study)
        assert len(high) == 1 and len(low) == 1 and int(low[0]) == 0
        one = cap.wall(study, int(high[0]))
        assert one.summary() == "c1/eilmer_second_order: 4 nodes, 2 triangles"
        assert one.points.shape == (4, 3) and one.cells.shape == (2, 3)
        assert one.fields["heat_flux"].tolist() == [2.0, 4.0, 6.0, 8.0]
        assert one.units["heat_flux"] == "W/(m^2)" and one.units["arc_length"] == "m"
        assert one.parameters["mach"] == 8.0 and one.parameters["diameter"] == pytest.approx(2.0)
        assert one.parameter_units["mach"] == "1"
        assert one.scalar_units["heat_load"] == "W"
        assert one.scalars["heat_load"] == pytest.approx(11.0)
        assert one.constants["wall_temperature"] == pytest.approx(300.0)


def test_load_bundle_carries_the_wall_the_inputs_and_the_isothermal_wall(tmp_path):
    path = _write_study(tmp_path / "capsule.gamut")
    bundle = cap.load_bundle(path, member="c1")                          # the expensive run when there is one
    assert bundle.manifest.mesh.kind == "unstructured" and bundle.manifest.mesh.dimension == 3
    assert bundle.manifest.mesh.blocks[0].type == "triangle"
    assert bundle.array("mesh:nodes").shape == (4, 3) and bundle.array("block:triangle").shape == (2, 3)
    assert set(bundle.manifest.fields.names()) == set(WALL_FIELDS)
    assert bundle.manifest.fields.get("heat_flux").unit == "W/(m^2)"
    parameters = {p.name: p for p in bundle.manifest.physics.parameters}
    assert parameters["mach"].value == pytest.approx(8.0) and parameters["mach"].unit == "1"
    conditions = {(c.field, c.kind): c for c in bundle.manifest.physics.boundary_conditions}
    assert conditions[("temperature", "dirichlet")].value == pytest.approx(300.0)
    assert conditions[("temperature", "dirichlet")].unit == "K"
    assert bundle.manifest.provenance.license == cap.LICENSE
    assert bundle.manifest.provenance.adapter.endswith("capsule_aeroheating")
    recorded = bundle.manifest.metadata["capsule_aeroheating"]
    assert recorded["member"] == "c1" and recorded["fidelity"] == cap.HIGH_FIDELITY
    assert recorded["scalars"]["heat_load"] == pytest.approx(11.0)


def test_the_bundle_passes_the_units_and_provenance_gates(tmp_path):
    """The file states every unit, so this adapter is the rare one whose bundle the units gate can pass."""
    path = _write_study(tmp_path / "capsule.gamut")
    report = run_gates(cap.load_bundle(path, member="c1"))
    assert report.get("units").passed is True, report.get("units").detail
    assert report.get("provenance").passed is True, report.get("provenance").detail
    assert report.get("integrity").passed is True, report.get("integrity").detail
    assert report.get("conservation").passed is None                  # nothing conserved to integrate on a wall


def test_load_bundle_refuses_a_row_that_is_not_the_member_asked_for(tmp_path):
    path = _write_study(tmp_path / "capsule.gamut")
    with pytest.raises(ValueError, match="not c2"):
        cap.load_bundle(path, row=0, member="c2")


def test_load_bundle_names_a_member_it_does_not_have(tmp_path):
    path = _write_study(tmp_path / "capsule.gamut")
    with pytest.raises(KeyError, match="no member"):
        cap.load_bundle(path, member="capsule_9999")


def test_load_table_bundle_is_the_run_table_with_its_units(tmp_path):
    path = _write_study(tmp_path / "capsule.gamut")
    bundle = cap.load_table_bundle(path)
    assert bundle.manifest.mesh.kind == "none"
    assert bundle.array("field:heat_load").tolist() == [10.0, 11.0, 12.0]
    assert bundle.manifest.fields.get("heat_load").unit == "W"
    recorded = bundle.manifest.metadata["capsule_aeroheating"]
    assert recorded["rows"] == 3 and recorded["members"] == 2
    assert recorded["fidelities"] == {cap.LOW_FIDELITY: 2, cap.HIGH_FIDELITY: 1}


def test_field_samples_needs_the_field_benchmark_and_says_so(tmp_path):
    pytest.importorskip("gamut", reason="the reader of the format is not installed")
    path = _write_study(tmp_path / "capsule.gamut")
    try:
        import pinneapple_arena.fieldbench  # noqa: F401
    except Exception:
        with pytest.raises(ImportError, match="fieldbench"):
            cap.field_samples(path)
    else:
        pytest.skip("the field benchmark is installed: this test is about the message when it is not")


def _stand_in_field_benchmark(monkeypatch):
    """A ``FieldSample`` with the benchmark's own fields, so the samples can be checked without its model stack.

    ``pinneapple_arena.fieldbench`` lives on the field-benchmark branch, and its model rows pull in more of that
    branch than this reader needs. What the samples have to get right is their shape and what goes in which slot, and
    that is what this pins."""
    from dataclasses import dataclass, field

    @dataclass
    class FieldSample:
        id: str
        points: np.ndarray
        values: np.ndarray
        category: str = ""
        parts: np.ndarray | None = None
        part_ids: np.ndarray | None = None
        thickness: np.ndarray | None = None
        times: np.ndarray | None = None
        extra: np.ndarray | None = None
        meta: dict = field(default_factory=dict)

    import types

    module = types.ModuleType("pinneapple_arena.fieldbench")
    module.FieldSample = FieldSample
    monkeypatch.setitem(sys.modules, "pinneapple_arena.fieldbench", module)
    try:
        import pinneapple_arena
    except Exception:                                       # pragma: no cover - the package is in this repository
        return FieldSample
    monkeypatch.setattr(pinneapple_arena, "fieldbench", module, raising=False)
    return FieldSample


def test_field_samples_put_the_cheap_field_beside_the_inputs_on_every_node(tmp_path, monkeypatch):
    pytest.importorskip("gamut", reason="the reader of the format is not installed")
    _stand_in_field_benchmark(monkeypatch)
    path = _write_study(tmp_path / "capsule.gamut")
    samples = cap.field_samples(path)
    assert [s.id for s in samples] == [f"c1_{cap.HIGH_FIDELITY}"]          # only the expensive run has a cheap twin
    one = samples[0]
    assert one.points.shape == (4, 3) and one.values.shape == (4, 1)
    assert one.thickness.tolist() == [0.0, 0.1, 0.2, 0.3]                  # the arc length, node by node
    assert one.extra.shape == (4, 7)                                       # six inputs, then the cheap model's field
    assert one.meta["channels"][-1] == f"heat_flux_{cap.LOW_FIDELITY}"
    assert one.meta["units"] == "W/(m^2)"
    assert np.allclose(one.extra[:, :6], one.extra[0, :6])                 # one flight condition, broadcast
    assert np.allclose(one.extra[:, 6], [1.0, 2.0, 3.0, 4.0])              # the cheap run's own heat flux
    assert np.allclose(one.values[:, 0], [2.0, 4.0, 6.0, 8.0])             # the expensive run's
    assert one.meta["parameters"]["mach"] == pytest.approx(8.0)


def test_field_samples_without_the_cheap_channel_leave_six_channels(tmp_path, monkeypatch):
    pytest.importorskip("gamut", reason="the reader of the format is not installed")
    _stand_in_field_benchmark(monkeypatch)
    path = _write_study(tmp_path / "capsule.gamut")
    one = cap.field_samples(path, low_fidelity_channel=False)[0]
    assert one.extra.shape == (4, 6) and len(one.meta["channels"]) == 6
    every = cap.field_samples(path, fidelity=cap.LOW_FIDELITY)
    assert len(every) == 2 and all(s.extra.shape == (4, 6) for s in every)
