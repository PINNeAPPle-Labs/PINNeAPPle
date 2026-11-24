"""The yt reader (#515): the conversion is tested everywhere, the reading needs yt and says so when it is missing.

Two kinds of test here on purpose. The axis order, the units and the ``PhysicalSample`` shape are checked on plain
arrays, so they run on a machine without yt. The reading is checked on datasets yt builds in memory from a dict
(``yt.load_uniform_grid``), which needs no download and no network: skipped with the install line when yt is absent.
"""
import numpy as np
import pytest

from pinneapple_data.astro import yt_reader
from pinneapple_data.astro.yt_reader import (
    YT_INSTALL_HINT,
    available,
    grid_to_dataset,
    list_fields,
    load,
    points_to_dataset,
)

YT_REASON = f"the yt reader is optional and yt is not installed here: {YT_INSTALL_HINT}"


def _uniform(box=((0, 1), (0, 1), (0, 1))):
    """A 5 x 3 x 2 yt dataset whose density is a function of x alone and whose temperature is a function of z alone:
    the two fields make the axis order of a read-back array unambiguous."""
    yt = pytest.importorskip("yt", reason=YT_REASON)
    nx, ny, nz = 5, 3, 2
    x = np.broadcast_to(np.arange(nx)[:, None, None], (nx, ny, nz)).astype(float)
    z = np.broadcast_to(np.arange(nz)[None, None, :], (nx, ny, nz)).astype(float)
    return yt.load_uniform_grid({"density": (x + 1.0, "g/cm**3"), "temperature": (z + 10.0, "K")},
                                (nx, ny, nz), length_unit="cm", bbox=np.array(box))


# ------------------------------------------------------------------ no yt needed: the conversion and the promise
def test_importing_the_package_does_not_import_yt():
    """An optional dependency that is imported at package import is not optional: the import would fail on a machine
    without yt, and every ``import pinneapple_data`` with it."""
    import subprocess
    import sys

    code = "import pinneapple_data.astro as a; print('yt' in __import__('sys').modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]), timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "False"


def test_available_says_which_package_is_missing_and_how_to_get_it():
    ok, why = available()
    assert isinstance(ok, bool)
    if ok:
        assert why == ""
    else:
        assert "pip install" in why and "yt" in why


def test_load_without_yt_raises_the_install_line_and_not_a_bare_import_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "yt", None)                    # `import yt` now fails, as on a bare machine
    with pytest.raises(ImportError, match="pip install yt"):
        load("some/output")
    with pytest.raises(ImportError, match="pip install yt"):
        list_fields("some/output")
    ok, why = available()
    assert not ok and "pip install yt" in why


def test_the_grid_becomes_a_dataset_with_the_axes_in_the_order_the_array_has():
    """``(x, y, z)`` in, ``("x", "y", "z")`` out, with cell centres as the coordinates: ``x`` runs from the first cell
    centre to the last, never from edge to edge."""
    arr = np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)
    ds = grid_to_dataset({"density": (arr, "g/cm**3")}, left_edge=(0, 0, 0), right_edge=(4, 6, 2), dims=(4, 3, 2),
                         time=1.5, unit_system="cgs")
    assert set(ds.data_vars) == {"density"} and dict(ds.sizes) == {"x": 4, "y": 3, "z": 2}
    np.testing.assert_allclose(ds["x"].values, [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(ds["y"].values, [1.0, 3.0, 5.0])
    np.testing.assert_allclose(ds["z"].values, [0.5, 1.5])
    np.testing.assert_array_equal(ds["density"].values, arr)       # no transpose, no flip
    assert ds.attrs["time"] == 1.5 and ds.attrs["unit_system"] == "cgs"


def test_points_stay_a_table_and_keep_their_unit_system():
    state = points_to_dataset({"x": (np.arange(4.0), "cm"), "density": (np.ones((2, 2)), "g/cm**3")},
                              unit_system="code")
    assert state["x"].shape == state["density"].shape == (4,)
    assert state.unit_system == "code"


# ------------------------------------------------------------------ the reading, on datasets yt builds in memory
def test_a_grid_dataset_becomes_a_grid_sample_with_units_and_provenance():
    pytest.importorskip("yt", reason=YT_REASON)
    s = load(_uniform())
    assert s.is_grid() and s.domain_type() == "grid"
    assert s.domain["dims"] == [5, 3, 2]
    assert set(s.list_variables()) == {"stream_density", "stream_temperature"}
    assert s.schema["units"]["stream_density"] == "g/cm**3"
    assert s.schema["units"]["stream_temperature"] == "K"
    assert s.provenance["reader"] == "yt" and s.provenance["dataset_type"] == "stream"
    assert s.provenance["yt_version"]
    assert s.provenance["length_unit"].endswith("cm")
    # the density is a function of x: it must vary along x and be constant along y and z
    d = np.asarray(s.state["stream_density"].values, float)
    np.testing.assert_allclose(d[:, 0, 0], 1 + np.arange(5.0))
    np.testing.assert_allclose(d[0, :, 0], d[0, 0, 0])
    np.testing.assert_allclose(d[0, 0, :], d[0, 0, 0])
    # the temperature is a function of z: the other way round
    t = np.asarray(s.state["stream_temperature"].values, float)
    np.testing.assert_allclose(t[0, 0, :], 10 + np.arange(2.0))
    np.testing.assert_allclose(t[:, 0, 0], t[0, 0, 0])


def test_the_level_sets_the_resolution_and_the_region_the_box():
    pytest.importorskip("yt", reason=YT_REASON)
    ds = _uniform()
    assert load(ds, level=1).domain["dims"] == [10, 6, 4]
    assert load(ds, dims=(10, 6, 2)).domain["dims"] == [10, 6, 2]
    corner = load(ds, region=((0, 0, 0), (0.5, 0.5, 0.5)))
    x = corner.state["x"].values
    assert x.max() < 0.5 and len(x) == 5                     # the half box, at the level's resolution
    with pytest.raises(ValueError, match="two 3-vectors"):
        load(ds, region=((0, 0), (1, 1)))


def test_particles_and_cell_centres_as_points():
    pytest.importorskip("yt", reason=YT_REASON)
    s = load(_uniform(), to="points", fields=["density"])
    assert s.domain_type() == "points" and s.domain["n"] == 30
    assert s.state["stream_density"].shape == (30,)
    assert s.state.unit_system == "cgs"
    with pytest.raises(ValueError, match="to="):
        load(_uniform(), to="voxels")


def test_the_fields_asked_for_are_the_fields_read_and_an_unknown_one_is_refused():
    pytest.importorskip("yt", reason=YT_REASON)
    ds = _uniform()
    assert list(load(ds, fields=["density"]).list_variables()) == ["stream_density"]
    assert list(load(ds, fields=[("gas", "temperature")]).list_variables()) == ["stream_temperature"]
    with pytest.raises(KeyError, match="no field"):                    # the message names the field asked for
        load(ds, fields=["nope"])
    with pytest.raises(KeyError, match="no field"):
        yt_reader._one_field(list(ds.field_list), "nope")


def test_list_fields_reports_the_units_before_anything_is_read():
    pytest.importorskip("yt", reason=YT_REASON)
    got = {f["field"]: f["unit"] for f in list_fields(_uniform())}
    assert got == {"stream_density": "g/cm**3", "stream_temperature": "K"}


def test_the_physics_audit_gives_mass_and_energy_sums_and_says_what_they_are():
    pytest.importorskip("yt", reason=YT_REASON)
    a = load(_uniform(), physics_audit=True).extras["physics_audit"]
    assert a["stream_density_sum"] == pytest.approx(15.0 * 6)          # 5+4+3+2+1 per (y, z) cell, 6 cells
    assert a["stream_density_max"] == pytest.approx(5.0)
    assert a["stream_density_unit"] == "g/cm**3"
    assert "cell volume" in a["note"]                                  # the caveat travels with the number
    assert a["points"] == 30


def test_the_sample_goes_where_every_other_sample_goes():
    """The point of the reader: it feeds the physics audit and the trainers, so it must be a ``PhysicalSample`` with
    the same fields as the CAE readers'."""
    pytest.importorskip("yt", reason=YT_REASON)
    from pinneapple_data.physical_sample import PhysicalSample

    s = load(_uniform())
    assert isinstance(s, PhysicalSample)
    assert s.summary()["domain_type"] == "grid"
    assert "stream_density" in s.summary()["state"]["vars"]
    assert s.schema["variables"]["stream_density"]["kind"] == "field"
