"""Astropy interop (#522): units on a sample both ways, FITS in and out, and the round trip that proves it.

Three kinds of test here, on purpose:

* the **units** bridge is checked on samples built in memory, so those run everywhere;
* **FITS** reading and writing is checked on files this test writes with astropy itself (a synthetic image, cube and
  table, including a ``Time`` and a ``SkyCoord`` column), which needs no network;
* the **end-to-end round trip** is checked on a real, public FITS file -- downloaded once, cached, its SHA-256 verified
  and its licence recorded (``pinneapple_data.astro.public_data``). Those tests skip, with the reason, when the file
  cannot be fetched; they fail when the bytes are not the ones the catalogue recorded.
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

astropy = pytest.importorskip("astropy", reason='the astropy interop needs astropy: pip install "pinneapple[astro]"')

import xarray as xr  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.io import fits as afits  # noqa: E402
from astropy.table import Table  # noqa: E402
from astropy.time import Time  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from pinneapple_data.astro import fits as pp_fits  # noqa: E402
from pinneapple_data.astro import public_data  # noqa: E402
from pinneapple_data.astro import units as pp_units  # noqa: E402
from pinneapple_data.physical_sample import PhysicalSample  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _fixture(name: str) -> Path:
    """A cached public FITS file; skipped (with the reason) when it cannot be fetched, failed when it is not the file."""
    try:
        return public_data.fetch(name)
    except Exception as exc:
        if "SHA-256" in str(exc):                                # the catalogue says what the bytes must be
            raise
        pytest.skip(f"the public FITS fixture {name!r} could not be fetched here: {exc}")


def _sample(**fields) -> PhysicalSample:
    """A sample whose fields declare their units in ``schema["units"]``, the way the readers here write them."""
    data = {name: (("x",), np.asarray(values, float)) for name, (values, _unit) in fields.items()}
    ds = xr.Dataset(data, coords={"x": np.arange(len(next(iter(fields.values()))[0]), dtype=float)})
    return PhysicalSample(state=ds, schema={"units": {n: uu for n, (_v, uu) in fields.items()}}, provenance={})


def _hide_astropy(monkeypatch) -> None:
    """Make ``import astropy`` fail in this process, the way it fails on a machine without the extra.

    Setting ``sys.modules["astropy"] = None`` is not enough: a submodule that is already imported ``from
    astropy.io import fits`` still resolves, so the submodules are dropped too.
    """
    for name in [m for m in sys.modules if m == "astropy" or m.startswith("astropy.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "astropy", None)


# ------------------------------------------------------------------ the package stays importable without astropy
def test_importing_the_astro_package_does_not_import_astropy():
    """An optional dependency imported at package import is not optional: ``import pinneapple_data`` would fail."""
    code = "import pinneapple_data.astro as a; print('astropy' in __import__('sys').modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO), timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "False"


def test_without_astropy_the_install_line_is_what_comes_out(monkeypatch):
    _hide_astropy(monkeypatch)                                    # `import astropy` now fails, as on a bare machine
    with pytest.raises(ImportError, match="pip install astropy"):
        pp_units.quantity_of(_sample(T=([1.0], "K")), "T")
    with pytest.raises(ImportError, match="pip install astropy"):
        pp_fits.load("anything.fits")
    ok, why = pp_fits.available()
    assert not ok and "pip install astropy" in why


# ------------------------------------------------------------------ units: reading
def test_a_quantity_is_read_in_the_unit_the_sample_declares():
    sample = _sample(T=([300.0, 310.0], "K"), v=([1.0, 2.0], "km/s"))
    assert pp_units.quantity_of(sample, "T")[0].value == 300.0
    assert str(pp_units.quantity_of(sample, "T").unit) == "K"
    assert pp_units.quantity_of(sample, "v").unit == u.km / u.s
    assert set(pp_units.sample_quantities(sample)) == {"T", "v"}


def test_the_units_attribute_of_the_dataset_is_read_before_the_schema():
    ds = xr.Dataset({"T": (("x",), [1.0], {"units": "deg_C"})})
    sample = PhysicalSample(state=ds, schema={"units": {"T": "K"}}, provenance={})
    assert pp_units.declared_unit(sample, "T") == "deg_C"         # what the variable itself says wins


def test_a_field_that_declares_no_unit_is_refused_rather_than_assumed():
    sample = PhysicalSample(state=xr.Dataset({"a": (("x",), [1.0, 2.0])}), schema={})
    assert pp_units.declared_unit(sample, "a") == ""
    with pytest.raises(pp_units.UnitMismatch) as err:
        pp_units.quantity_of(sample, "a")
    assert "declares no unit" in str(err.value) and "Fix:" in str(err.value)
    assert pp_units.quantity_of(sample, "a", unit="m").unit == u.m   # the caller who knows a unit may say so


def test_reading_in_another_unit_converts_and_a_length_is_refused():
    sample = _sample(v=([1.0, 2.0], "km/s"))
    np.testing.assert_allclose(pp_units.quantity_of(sample, "v", unit="m/s").value, [1000.0, 2000.0])
    with pytest.raises(pp_units.UnitMismatch):
        pp_units.quantity_of(sample, "v", unit="K")
    np.testing.assert_allclose(sample.state["v"].values, [1.0, 2.0])  # reading changed nothing


def test_a_string_column_has_no_magnitude_to_convert():
    sample = PhysicalSample(state=xr.Dataset({"name": (("row",), np.array(["a", "b"]))}),
                            schema={"units": {"name": "1"}}, provenance={})
    with pytest.raises(pp_units.UnitMismatch, match="carry no magnitude"):
        pp_units.quantity_of(sample, "name")


# ------------------------------------------------------------------ units: writing back
def test_a_compatible_quantity_is_converted_and_the_conversion_is_recorded():
    sample = _sample(T=([0.0, 0.0], "deg_C"))
    record = pp_units.set_quantity(sample, "T", 300 * u.K)
    assert record["action"] == "converted" and record["scale"] == 1.0
    assert record["offset"] == -273.15                            # value in degC = value in K * scale + offset
    np.testing.assert_allclose(sample.state["T"].values, [26.85, 26.85])
    assert sample.state["T"].attrs["units"] == "deg_C"
    assert sample.provenance["unit_conversions"][0]["from_unit"] == "K"
    assert sample.provenance["unit_conversions"][0]["offset"] == -273.15


def test_the_same_unit_is_written_untouched_and_records_no_conversion():
    sample = _sample(v=([1.0, 2.0], "km/s"))
    assert pp_units.set_quantity(sample, "v", [3.0, 4.0] * u.km / u.s)["action"] == "same"
    np.testing.assert_allclose(sample.state["v"].values, [3.0, 4.0])
    assert "unit_conversions" not in sample.provenance


def test_an_incompatible_unit_is_refused_and_nothing_is_written():
    sample = _sample(v=([1.0, 2.0], "km/s"))
    with pytest.raises(pp_units.UnitMismatch) as err:
        pp_units.set_quantity(sample, "v", 300 * u.K)
    assert "different dimensions" in str(err.value)
    np.testing.assert_allclose(sample.state["v"].values, [1.0, 2.0])
    assert "unit_conversions" not in sample.provenance
    assert sample.schema["units"]["v"] == "km/s"


def test_a_bare_number_is_refused_because_only_the_caller_knows_its_unit():
    sample = _sample(T=([0.0], "K"))
    with pytest.raises(pp_units.UnitMismatch, match="bare number"):
        pp_units.set_quantity(sample, "T", 300.0)


def test_a_new_field_adopts_the_unit_and_writes_it_down():
    sample = _sample(T=([1.0, 2.0], "K"))
    record = pp_units.set_quantity(sample, "rho", 1.2 * u.kg / u.m ** 3, dims=("x",))
    assert record["action"] == "declared" and sample.schema["units"]["rho"] == "kg / m3"
    assert sample.state["rho"].attrs["units"] == "kg / m3"
    np.testing.assert_allclose(sample.state["rho"].values, [1.2, 1.2])


def test_a_temperature_difference_keeps_its_magnitude():
    sample = _sample(dT=([0.0], "deg_C"))
    pp_units.set_quantity(sample, "dT", 5 * u.K, difference=True)
    np.testing.assert_allclose(sample.state["dT"].values, [5.0])  # not -268.15: an offset does not belong to a delta
    absolute = _sample(dT=([0.0], "deg_C"))
    pp_units.set_quantity(absolute, "dT", 5 * u.K)
    np.testing.assert_allclose(absolute.state["dT"].values, [-268.15])


def test_the_round_trip_through_a_quantity_is_exact():
    sample = _sample(T=([273.15, 373.15], "K"))
    q = pp_units.quantity_of(sample, "T")
    pp_units.set_quantity(sample, "T", q)
    np.testing.assert_allclose(pp_units.quantity_of(sample, "T").value, [273.15, 373.15])


# ------------------------------------------------------------------ the LBM lattice helper, both ways
def test_the_lattice_helper_converts_both_ways_and_refuses_the_wrong_dimension():
    from pinneapple_simulation.numerical_solvers.lbm_tunnel import Units

    lattice = Units.from_si(lbm_length=1.0, lbm_velocity=0.1, lbm_rho=1.0, si_length=1e-3, si_velocity=1.0, si_rho=1.0)
    assert pp_units.to_lattice(lattice, 0.5 * u.m, "x") == pytest.approx(500.0)   # 1e-3 m per lattice unit
    back = pp_units.from_lattice(lattice, 500.0, "x")
    assert back.unit == u.m and back.value == pytest.approx(0.5)
    with pytest.raises(pp_units.UnitMismatch, match="a length is measured in m"):
        pp_units.to_lattice(lattice, 300 * u.K, "x")
    with pytest.raises(pp_units.UnitMismatch, match="no inverse"):
        pp_units.from_lattice(lattice, 1.0, "rho")


# ------------------------------------------------------------------ FITS: the synthetic files every machine can write
def _write_image(path: Path, *, cube: bool = False) -> Path:
    """A small FITS image (or spectral cube) written by astropy: a TAN WCS, a unit, an observation date."""
    w = WCS(naxis=3 if cube else 2)
    w.wcs.ctype = (["RA---TAN", "DEC--TAN", "FREQ"] if cube else ["RA---TAN", "DEC--TAN"])
    w.wcs.crval = ([150.0, 2.0, 1.4e9] if cube else [150.0, 2.0])
    w.wcs.crpix = ([10.0, 10.0, 1.0] if cube else [10.0, 10.0])
    w.wcs.cdelt = ([-0.001, 0.001, 1e6] if cube else [-0.001, 0.001])
    w.wcs.cunit = (["deg", "deg", "Hz"] if cube else ["deg", "deg"])
    data = np.arange(20 * 20 * (3 if cube else 1), dtype=float).reshape((3, 20, 20) if cube else (20, 20))
    header = w.to_header()
    header["BUNIT"] = ("Jy/beam", "unit of the pixels")
    header["DATE-OBS"] = "2020-05-17T04:32:11"
    header["TIMESYS"] = "UTC"
    header["TELESCOP"] = "TEST"
    afits.PrimaryHDU(data=data, header=header).writeto(str(path), overwrite=True)
    return path


def test_an_image_keeps_its_array_order_its_unit_and_its_wcs(tmp_path):
    sample = pp_fits.load(_write_image(tmp_path / "img.fits"))
    assert sample.provenance["format"] == "fits-image"
    assert sample.state["image"].dims == ("y", "x")               # numpy order, never transposed
    assert sample.state["image"].shape == (20, 20)
    assert sample.schema["units"]["image"] == "Jy/beam"
    assert sample.provenance["bunit"] == "Jy/beam"
    assert sample.provenance["sky"]["frame"] == "icrs"
    assert sample.provenance["time"]["scale"] == "utc"
    assert float(pp_fits.time_of(sample).mjd) == pytest.approx(58986.189, abs=1e-3)


def test_a_cube_names_its_third_axis_from_the_wcs_and_keeps_the_world_values(tmp_path):
    sample = pp_fits.load(_write_image(tmp_path / "cube.fits", cube=True))
    assert sample.provenance["format"] == "fits-cube"
    assert sample.state["image"].dims == ("spectral", "y", "x")
    assert sample.provenance["wcs"]["axis_types"] == ["RA", "DEC", "FREQ"]
    spectral = sample.state["spectral_world"]
    assert spectral.attrs["units"] == "Hz"
    np.testing.assert_allclose(spectral.values, [1.4e9, 1.401e9, 1.402e9])


def test_the_wcs_is_recorded_card_for_card_and_rebuilds_after_json(tmp_path):
    sample = pp_fits.load(_write_image(tmp_path / "img.fits"))
    live = pp_fits.wcs_of(sample)
    survived = json.loads(json.dumps(sample.provenance))           # what a saved manifest does to provenance
    rebuilt = pp_fits.wcs_of(PhysicalSample(state=sample.state, provenance=survived))
    pixels = np.array([[0.0, 0.0], [9.5, 9.5], [19.0, 19.0]])
    np.testing.assert_allclose(rebuilt.all_pix2world(pixels, 0), live.all_pix2world(pixels, 0))


def test_a_table_column_keeps_its_unit_and_its_shape(tmp_path):
    table = Table({"flux": [1.0, 2.0] * u.Jy, "n": [1, 2],
                   "spectrum": np.arange(6, dtype=float).reshape(2, 3) * u.K})
    table.write(tmp_path / "t.fits", format="fits", overwrite=True)
    sample = pp_fits.load(tmp_path / "t.fits")
    assert sample.provenance["format"] == "fits-table"
    assert sample.schema["units"] == {"flux": "Jy", "spectrum": "K"}
    assert dict(sample.state.sizes) == {"row": 2, "spectrum_dim": 3}
    np.testing.assert_allclose(sample.state["spectrum"].values, table["spectrum"].value)


def test_a_time_and_a_sky_column_come_back_as_themselves(tmp_path):
    table = Table({"flux": [1.0, 2.0] * u.Jy,
                   "when": Time([58000.0, 58001.0], format="mjd", scale="tdb"),
                   "where": SkyCoord([10.0, 11.0] * u.deg, [-1.0, -2.0] * u.deg)})
    table.write(tmp_path / "mixins.fits", format="fits", overwrite=True)
    sample = pp_fits.load(tmp_path / "mixins.fits")
    kinds = {c["name"]: c["kind"] for c in sample.provenance["columns"]}
    assert kinds["when"] == "time" and kinds["where"] == "sky"
    np.testing.assert_allclose(sample.state["when"].values, [58000.0, 58001.0])
    assert sample.extras["columns"]["when"].scale == "tdb"
    assert isinstance(sample.extras["columns"]["where"], SkyCoord)

    pp_fits.save(sample, tmp_path / "mixins_back.fits")
    again = pp_fits.load(tmp_path / "mixins_back.fits")
    kinds_again = {c["name"]: c["kind"] for c in again.provenance["columns"]}
    assert kinds_again == kinds                                   # the same columns, by the same names
    np.testing.assert_allclose(again.state["when"].values, [58000.0, 58001.0])
    np.testing.assert_allclose(again.extras["columns"]["where"].ra.deg, [10.0, 11.0])


def test_an_hdu_that_holds_no_data_says_so(tmp_path):
    afits.PrimaryHDU().writeto(str(tmp_path / "empty.fits"), overwrite=True)
    with pytest.raises(ValueError, match="no HDU with data"):
        pp_fits.load(tmp_path / "empty.fits")
    path = _write_image(tmp_path / "one.fits")
    with pytest.raises(ValueError, match="out of range"):
        pp_fits.load(path, hdu=7)


def test_a_sample_without_a_wcs_reports_none_rather_than_inventing_one(tmp_path):
    afits.PrimaryHDU(data=np.zeros((4, 4))).writeto(str(tmp_path / "plain.fits"), overwrite=True)
    sample = pp_fits.load(tmp_path / "plain.fits")
    assert "wcs" not in sample.provenance and pp_fits.wcs_of(sample) is None
    assert pp_fits.sky_of(sample) is None and pp_fits.time_of(sample) is None


def test_a_date_the_standard_cannot_parse_is_kept_verbatim_with_the_reason(tmp_path):
    header = afits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"], header["NAXIS2"] = 4, 4
    header["DATE-OBS"] = "13/03/95"                              # the pre-1997 DD/MM/YY form, not a FITS date
    header["TIMESYS"] = "UTC"
    afits.PrimaryHDU(data=np.zeros((4, 4)), header=header).writeto(str(tmp_path / "old.fits"), overwrite=True)
    sample = pp_fits.load(tmp_path / "old.fits")
    assert sample.provenance["time"]["raw"] == "13/03/95"
    assert sample.extras["fits_header"]["DATE-OBS"] == "13/03/95"  # written back as it came in
    assert pp_fits.time_of(sample) is not None                    # through the header's own MJD-OBS


# ------------------------------------------------------------------ the real public file, end to end
def test_the_public_fixture_is_cached_with_its_licence_declared():
    path = _fixture("uit_image")
    assert path.stat().st_size == public_data.FIXTURES["uit_image"].size
    meta = public_data.sidecar("uit_image")
    assert meta["licence"].startswith("Public domain")
    assert meta["sha256"] == public_data.FIXTURES["uit_image"].sha256
    assert meta["retrieved"]


def test_the_public_image_round_trips_with_its_wcs_its_unit_and_its_time_intact(tmp_path):
    """The test #522 asks for: a unit-bearing FITS image -> PhysicalSample -> FITS, with the WCS unchanged."""
    path = _fixture("uit_image")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                           # the file's own RADECSYS/DATFIX warnings
        sample = pp_fits.load(path)
    assert sample.provenance["bunit"] == "ERGS/CM**2/S/ANGSTRM"
    assert sample.schema["units"]["image"] == "ERGS/CM**2/S/ANGSTRM"
    np.testing.assert_array_equal(sample.state["image"].values, afits.getdata(str(path)))
    before_sky = sample.provenance["sky"]
    before_mjd = float(pp_fits.time_of(sample).mjd)

    out = pp_fits.save(sample, tmp_path / "copy.fits")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = pp_fits.load(out)

    assert again.provenance["bunit"] == sample.provenance["bunit"]
    np.testing.assert_array_equal(again.state["image"].values, sample.state["image"].values)
    w_before, w_after = pp_fits.wcs_of(sample), pp_fits.wcs_of(again)
    pixels = np.array([[0.0, 0.0], [0.0, 511.0], [511.0, 0.0], [511.0, 511.0], [255.5, 255.5]])
    np.testing.assert_allclose(w_after.all_pix2world(pixels, 0), w_before.all_pix2world(pixels, 0))
    assert w_after.to_header(relax=True).tostring() == w_before.to_header(relax=True).tostring()
    assert again.provenance["sky"] == before_sky
    assert float(pp_fits.time_of(again).mjd) == before_mjd
    assert again.extras["fits_header"]["DATE-OBS"] == sample.extras["fits_header"]["DATE-OBS"]


def test_the_public_cube_round_trips_with_its_three_axis_wcs(tmp_path):
    path = _fixture("wfpc2_cube")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample = pp_fits.load(path)
    assert sample.provenance["format"] == "fits-cube"
    assert sample.state["image"].shape == (4, 200, 200)           # (NAXIS3, NAXIS2, NAXIS1), not transposed
    assert sample.state["image"].dims == ("group_number", "y", "x")
    assert sample.provenance["time"]["source"] == "DATE-OBS"
    assert "19:03:13" in sample.provenance["time"]["isot"]        # DATE-OBS + TIME-OBS, the whole observation

    out = pp_fits.save(sample, tmp_path / "cube.fits")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = pp_fits.load(out)
    np.testing.assert_array_equal(again.state["image"].values, sample.state["image"].values)
    pixels = np.array([[0.0, 0.0, 0.0], [199.0, 199.0, 3.0], [100.0, 50.0, 2.0]])[:, ::-1]
    np.testing.assert_allclose(pp_fits.wcs_of(again).all_pix2world(pixels, 0),
                               pp_fits.wcs_of(sample).all_pix2world(pixels, 0))
    assert again.provenance["time"]["jd"] == sample.provenance["time"]["jd"]


def test_the_public_table_keeps_its_columns_units_and_values(tmp_path):
    path = _fixture("iue_table")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample = pp_fits.load(path)
    assert sample.provenance["format"] == "fits-table"
    assert sample.provenance["hdu_name"] == "MELO"
    assert set(sample.schema["units"]) == {"WAVELENGTH", "DELTAW", "NET", "BACKGROUND", "SIGMA", "FLUX"}
    assert dict(sample.state.sizes)["NET_dim"] == 640

    out = pp_fits.save(sample, tmp_path / "table.fits")
    with afits.open(str(out)) as hdul:
        assert hdul[1].columns["FLUX"].unit == "ERG/CM2/S/A"
        assert hdul[0].header["TELESCOP"] == "IUE"                # the primary header's keywords, kept
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = pp_fits.load(out)
    for name in sample.schema["units"]:
        np.testing.assert_array_equal(np.asarray(again.state[name].values), np.asarray(sample.state[name].values))


# ------------------------------------------------------------------ the archival unit string, and the guide's block
def test_an_archival_unit_astropy_cannot_parse_is_refused_by_name_and_can_be_read_explicitly(tmp_path):
    """A 1995 header spells its unit ``ERGS/CM**2/S/ANGSTRM``: no parser astropy has accepts it, so the sample keeps
    the text and the refusal says what to do. The caller who read the header can still get a Quantity."""
    path = _fixture("uit_image")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample = pp_fits.load(path)
    assert sample.schema["units"]["image"] == "ERGS/CM**2/S/ANGSTRM"     # preserved as text
    with pytest.raises(pp_units.UnitMismatch, match="not a unit astropy can parse"):
        pp_units.quantity_of(sample, "image")
    q = pp_units.quantity_of(sample, "image", unit="erg / (cm2 s Angstrom)")
    assert q.unit == u.erg / (u.cm ** 2 * u.s * u.AA) and q.max().value > 0
    assert sample.schema["units"]["image"] == "ERGS/CM**2/S/ANGSTRM"     # reading did not rewrite the text


def test_the_guide_example_runs_as_printed(tmp_path, monkeypatch):
    """``docs/guides/astropy_fits.md`` promises lines a reader can paste: this runs exactly those lines."""
    guide = REPO / "docs" / "guides" / "astropy_fits.md"
    blocks = __import__("re").findall(r"```python\n(.*?)```", guide.read_text(encoding="utf-8"), flags=__import__("re").S)
    assert len(blocks) == 1, f"expected one Python block in {guide.name}, found {len(blocks)}"
    _fixture("uit_image")                                        # skip (not fail) when the fixture is unreachable
    monkeypatch.chdir(tmp_path)                                  # the block writes copy.fits
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(blocks[0], str(guide), "exec"), {"__name__": "astropy_fits_guide"})
    assert (tmp_path / "copy.fits").is_file()
