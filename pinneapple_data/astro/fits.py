"""FITS images, cubes and tables as ``PhysicalSample``, with the WCS, the sky coordinates and the time kept (#522).

    from pinneapple_data.astro import fits

    s = fits.load("ngc4151.fits")                  # an image: a TAN WCS and BUNIT in the header
    s = fits.load("wfpc2_cube.fits")               # a cube: (plane, y, x), the numpy order the file has
    s = fits.load("iue_table.fits")                # a binary table: one column per field, units from TUNIT
    fits.save(s, "copy.fits")                      # back out, every WCS card and unit as it came in
    fits.wcs_of(s).pixel_to_world(0, 0)            # the live astropy WCS (or None when the file has none)
    fits.sky_of(s)                                 # the pointing, as a SkyCoord
    fits.time_of(s)                                # DATE-OBS as an astropy Time, with its scale

What the sample carries, and where:

* ``state`` -- an ``xarray.Dataset``. An image or a cube holds one data variable (the pixels, **not** transposed:
  numpy's ``(..., y, x)`` is exactly FITS' ``(..., NAXIS2, NAXIS1)``); a table holds one variable per column, with a
  ``row`` dimension and, for a vector column, one more named ``<column>_dim``.
* ``schema["units"]`` -- ``BUNIT`` for an image, ``TUNIT`` for a column. The same text goes on the variable's
  ``units`` attribute, so ``pinneapple_data.astro.units.quantity_of`` reads either file.
* ``provenance["wcs"]`` -- every WCS card the file had (keyword, value, comment), the axis types and the axis units.
  This is what makes the round trip exact: the header is rebuilt card for card, not recomputed.
* ``provenance["sky"]`` -- the frame, the pointing at the centre of the image and at its four corners, and the pixel
  scale: the coordinates a reader has to have and cannot get from the numbers.
* ``provenance["time"]`` -- the observation time from ``MJD-OBS`` / ``JD-OBS`` / ``DATE-OBS`` (+ ``TIME-OBS``), with
  the time scale ``TIMESYS`` names. A date the standard cannot parse is kept **as text** and marked
  ``parsed: false`` with the reason: the 1995-era ``13/03/95`` is preserved, not guessed at.
* ``extras["wcs"]``, ``extras["sky"]``, ``extras["time"]`` -- the live astropy objects; ``extras["fits_header"]`` the
  original header, which is what :func:`save` starts from so nothing the file said is lost.

A binary table's ``Time`` and ``SkyCoord`` columns are kept as such: the column's numeric part goes into ``state`` and
the astropy object into ``extras["columns"]``, which :func:`save` writes back as the same mixin column. A ``Time``
column is stored as MJD in its own scale (exact to a double) with its original format recorded.

astropy is optional: nothing here imports it until a function is called, so ``import pinneapple_data`` keeps working
on a machine without the extra.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FITS_INSTALL_HINT", "available", "load", "save", "sky_of", "time_of", "wcs_of"]

FITS_INSTALL_HINT = 'the FITS reader needs astropy: pip install astropy (or pip install "pinneapple[astro]")'

#: Cards that describe the file's structure rather than its content; ``save`` drops them and lets astropy write them
#: from the array it is given, which is the only way a header and a shape cannot disagree.
_STRUCTURAL = ("SIMPLE", "XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4", "NAXIS5", "EXTEND",
               "PCOUNT", "GCOUNT", "TFIELDS", "BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM")

#: The header cards copied into ``provenance["header"]``: what a reader asks about a file it did not open.
_CURATED = ("OBJECT", "TELESCOP", "INSTRUME", "DETECTOR", "FILTER", "EXPTIME", "BUNIT", "DATE-OBS", "TIME-OBS",
            "DATE-END", "MJD-OBS", "JD-OBS", "TIMESYS", "RADESYS", "EQUINOX", "OBSERVER", "ORIGIN", "AUTHOR")


def _astro():
    """``(astropy.io.fits, astropy.wcs.WCS)``, or an ImportError that says how to get them."""
    try:
        from astropy.io import fits as afits
        from astropy.wcs import WCS
    except Exception as exc:                                     # ImportError, or a broken install of its own deps
        raise ImportError(FITS_INSTALL_HINT) from exc
    return afits, WCS


def available() -> tuple[bool, str]:
    """Whether astropy (with its FITS and WCS parts) can be imported, and the reason when it cannot."""
    try:
        import astropy.io.fits  # noqa: F401
        import astropy.wcs  # noqa: F401
    except Exception as exc:
        return False, f"{FITS_INSTALL_HINT} ({type(exc).__name__}: {exc})"
    return True, ""


# ---------------------------------------------------------------------------------------------- small pieces
def _plain(value: Any) -> Any:
    """A header value as something JSON can hold: numpy scalars to Python numbers, everything else as it is."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return str(value)


def _cards(header) -> list[list[Any]]:
    """A header as ``[[keyword, value, comment], ...]``, in file order, values made JSON-safe."""
    return [[str(card.keyword), _plain(card.value), str(card.comment)] for card in header.cards
            if card.keyword and str(card.keyword).strip()]


def _header(cards: Sequence[Sequence[Any]]):
    afits, _ = _astro()
    return afits.Header([(str(k), v, str(c)) for k, v, c in cards])


def _pick_hdu(hdul, hdu: None | int | str) -> int:
    """The index of the HDU to read: the one asked for, or the first that holds data."""
    if hdu is None:
        for i, h in enumerate(hdul):
            if h.data is not None:
                return i
        raise ValueError("this FITS file has no HDU with data (only an empty primary HDU)")
    try:
        index = hdul.index_of(hdu) if isinstance(hdu, str) else int(hdu)
    except Exception as exc:
        names = [h.name for h in hdul]
        raise ValueError(f"no HDU {hdu!r} in this file; it has {names}") from exc
    if not 0 <= index < len(hdul):
        raise ValueError(f"HDU {index} is out of range: this file has {len(hdul)} HDU(s) (0-{len(hdul) - 1})")
    return index


def _role(axis_type: str) -> str:
    """The dimension name for a FITS axis, from the axis type the WCS gives it."""
    t = (axis_type or "").strip().upper().replace(" ", "_")
    if not t:
        return "plane"
    if t in ("RA", "DEC", "GLON", "GLAT", "ELON", "ELAT", "SLON", "SLAT"):
        return "sky"
    for prefix, name in (("FREQ", "spectral"), ("WAVE", "spectral"), ("AWAV", "spectral"), ("VOPT", "spectral"),
                         ("ZOPT", "spectral"), ("VELO", "spectral"), ("VRAD", "spectral"), ("ENER", "energy"),
                         ("STOKES", "stokes"), ("TIME", "time"), ("UTC", "time"), ("TAI", "time")):
        if t.startswith(prefix):
            return name
    return t.lower().replace("-", "_")


def _dims_for(ndim: int, wcs) -> tuple[str, ...]:
    """The dimension names of the pixel array, in numpy order: leading FITS axes named by their WCS type, then y, x.

    The array is never transposed, so a cube's ``(plane, y, x)`` is the file's own ``(NAXIS3, NAXIS2, NAXIS1)``. A
    repeated role gets a numeric suffix (``plane``, ``plane_1``) so xarray never sees two dimensions with one name.
    """
    types = list(getattr(wcs, "axis_type_names", []) or []) if wcs is not None else []
    leading: list[str] = []
    for j in range(max(ndim - 2, 0)):                            # numpy axis j <-> FITS axis ndim - j
        fits_axis = ndim - j - 1
        leading.append(_role(types[fits_axis]) if fits_axis < len(types) else "plane")
    seen: dict[str, int] = {}
    unique = []
    for name in leading:
        n = seen.get(name, 0)
        seen[name] = n + 1
        unique.append(name if n == 0 else f"{name}_{n}")
    tail = ("y", "x") if ndim >= 2 else ("x",)
    return tuple(unique) + tail


def _axis_world(wcs, axis: int, n: int) -> tuple[np.ndarray | None, str]:
    """The world value along one FITS axis, at the pixels of that axis, others held at the reference pixel.

    This is the standard reading of a FITS axis and it is exact for a separable axis (a spectral one, almost always);
    for the celestial axes it is *not* the sky grid -- a rotated projection needs both axes at once -- which is why the
    sky is recorded as a coordinate in ``provenance["sky"]`` and the full WCS in ``provenance["wcs"]``. Returns
    ``(None, "")`` for a celestial axis rather than a misleading 1-D cut through the sky.
    """
    if wcs is None or axis >= wcs.naxis:
        return None, ""
    types = [str(t).upper() for t in (getattr(wcs, "axis_type_names", None) or [])]
    if axis < len(types) and types[axis] in ("RA", "DEC", "GLON", "GLAT", "ELON", "ELAT", "SLON", "SLAT"):
        return None, ""
    matrix = getattr(wcs, "axis_correlation_matrix", None)
    if matrix is not None:                                       # separable or nothing: a coupled axis is not a 1-D cut
        row = np.asarray(matrix)[axis]
        if int(row.sum()) != 1 or not bool(row[axis]):
            return None, ""
    pixels = []
    for i in range(wcs.naxis):
        ref = float(wcs.wcs.crpix[i] - 1.0) if wcs.wcs.crpix[i] else 0.0
        pixels.append(np.arange(n, dtype=float) if i == axis else np.full(n, ref, dtype=float))
    world = wcs.all_pix2world(*pixels, 0)
    unit = ""
    try:
        unit = str(wcs.world_axis_units[axis])
    except Exception:                                            # a WCS without CUNIT: dimensionless is the answer
        unit = ""
    return np.asarray(world[axis], dtype=float), unit


def _sky_provenance(wcs, shape: Sequence[int]) -> dict[str, Any]:
    """Frame, centre, corners and pixel scale of a celestial WCS: the pointing a reader cannot compute from numbers.

    When the WCS has no celestial part, ``{}`` -- there is nothing to say. When it has one this module cannot resolve
    into a ``SkyCoord``, the entry says ``{"error": ...}`` instead: a missing pointing is reported, never implied to
    be absent from the file.
    """
    if wcs is None or not getattr(wcs, "has_celestial", False):
        return {}
    try:
        from astropy import units as u

        cel = wcs.celestial
        ny, nx = int(shape[-2]), int(shape[-1])
        centre = cel.pixel_to_world([(nx - 1) / 2.0], [(ny - 1) / 2.0])[0]
        corners = cel.pixel_to_world([0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1])
        unit = str((cel.world_axis_units or ["deg"])[0] or "deg")
        scale = cel.proj_plane_pixel_scales() * u.Unit(unit)
        return {
            "frame": str(centre.frame.name),
            "radesys": str(getattr(cel.wcs, "radesys", "") or ""),
            "center": {"lon_deg": float(centre.spherical.lon.deg),
                       "lat_deg": float(centre.spherical.lat.deg)},
            "corners_deg": [[float(c.spherical.lon.deg), float(c.spherical.lat.deg)] for c in corners],
            "pixel_scale_arcsec": [float(abs(s.to_value(u.arcsec))) for s in scale],
            "units": "deg",
        }
    except Exception as exc:                                     # a WCS astropy cannot turn into a SkyCoord
        return {"error": f"{type(exc).__name__}: {exc}",
                "why": "the header has a celestial WCS that astropy could not resolve into a SkyCoord"}


def _time_provenance(header, wcs=None) -> dict[str, Any]:
    """The observation time in the header, as provenance: scale, format, Julian date, and the raw text it came from.

    ``MJD-OBS``, ``JD-OBS`` and ``JD`` are tried in that order, then ``DATE-OBS`` (the FITS date format, with
    ``TIME-OBS`` appended when the date carries no time of day). The scale is ``TIMESYS``, defaulting to UTC as the FITS
    standard says it must.

    A date written in the pre-1997 ``DD/MM/YY`` convention (``13/03/95``) is not something this module reads: astropy's
    own ``Time`` refuses it. It is **not** guessed at either. What happens is: the raw text is kept, and the value
    astropy's WCS reader derives from it for the file's ``MJD-OBS`` (its ``datfix`` applies that exact historical rule)
    is used, with ``source`` naming that path and ``note`` saying so. A date nothing can resolve is kept verbatim with
    ``parsed: false`` and the reason.
    """
    from astropy.time import Time

    scale = str(header.get("TIMESYS", "UTC") or "UTC").strip().lower()
    if scale not in ("utc", "tai", "tt", "tdb", "tcb", "tcg", "ut1", "local"):
        return {"parsed": False, "scale": scale, "why": f"TIMESYS {scale!r} is not a time scale FITS defines",
                "raw": {"TIMESYS": _plain(header.get("TIMESYS"))}}
    attempts: list[tuple[str, Any, str]] = []
    if "MJD-OBS" in header:
        attempts.append(("MJD-OBS", header["MJD-OBS"], "mjd"))
    if "JD-OBS" in header:
        attempts.append(("JD-OBS", header["JD-OBS"], "jd"))
    if "JD" in header:
        attempts.append(("JD", header["JD"], "jd"))
    raw_date = header.get("DATE-OBS")
    if raw_date is not None:
        text = str(raw_date).strip()
        time_obs = str(header.get("TIME-OBS", "") or "").strip()
        if time_obs and "T" not in text:
            text = f"{text}T{time_obs}"
        attempts.append(("DATE-OBS", text, "fits"))
    why = ""
    for key, value, fmt in attempts:
        try:
            t = Time(value, format=fmt, scale=scale)
        except Exception as exc:
            why = f"{key} {value!r}: {type(exc).__name__}: {exc}"
            continue
        return {"parsed": True, "scale": t.scale, "format": t.format, "source": key,
                "jd": float(t.jd), "mjd": float(t.mjd), "isot": str(t.isot), "raw": _plain(raw_date)}
    if wcs is None:
        try:                                                     # only for the date the WCS reader repairs, below
            _, WCS = _astro()
            wcs = WCS(header)
        except Exception:                                        # a header with no usable WCS at all
            wcs = None
    mjd = float(getattr(getattr(wcs, "wcs", None), "mjdobs", 0.0) or 0.0) if wcs is not None else 0.0
    if np.isfinite(mjd) and mjd != 0.0:
        t = Time(mjd, format="mjd", scale=scale)
        return {"parsed": True, "scale": t.scale, "format": "mjd", "source": "DATE-OBS, DD/MM/YY convention",
                "jd": float(t.jd), "mjd": float(t.mjd), "isot": str(t.isot), "raw": _plain(raw_date),
                "note": (f"DATE-OBS {raw_date!r} is not the FITS date-time format; astropy's WCS reader applied the "
                         "pre-1997 DD/MM/YY rule to derive MJD-OBS, which is what is recorded here")}
    if not attempts:
        return {}
    return {"parsed": False, "scale": scale, "source": key if attempts else "", "format": "",
            "raw": _plain(raw_date), "why": why or "no observation-time keyword the FITS standard defines"}


# ---------------------------------------------------------------------------------------------- reading
def load(path: str | Path, hdu: None | int | str = None, *, name: str | None = None) -> Any:
    """One FITS HDU as a ``PhysicalSample``: an image, a cube or a table.

    path : the ``.fits`` / ``.fit`` / ``.fts`` file (or ``.gz`` of one) -- a real file, because a table's counts and
           vector columns are described by the file itself.
    hdu  : the HDU to read, by index or ``EXTNAME``; the first HDU with data by default (which is what a single-image
           file has, and what skips an empty primary HDU in front of a table, as the IUE spectra files have).
    name : the name of the data variable; ``EXTNAME`` (lowercased) or ``"image"`` by default.

    Returns a ``PhysicalSample`` whose ``state`` is an ``xarray.Dataset``, with units in ``schema["units"]``, the WCS
    cards and the axis types in ``provenance["wcs"]``, the pointing in ``provenance["sky"]`` and the observation time
    in ``provenance["time"]``. See the module docstring for the layout, and :func:`save` for the way back.
    """
    from pinneapple_data.physical_sample import PhysicalSample

    afits, WCS = _astro()
    path = str(path)
    with afits.open(path) as hdul:
        index = _pick_hdu(hdul, hdu)
        h = hdul[index]
        if isinstance(h, (afits.BinTableHDU, afits.TableHDU)):
            sample = _table_sample(path, hdul, index)
        else:
            if h.data is None:
                raise ValueError(f"HDU {index} ({h.name}) of {path} holds no data; ask for another hdu=")
            sample = _image_sample(path, hdul, index, name=name)
    assert isinstance(sample, PhysicalSample)
    return sample


def _names_an_axis(header, ndim: int) -> bool:
    """Whether the header says what its axes *mean*, i.e. names at least one with ``CTYPEn``.

    A file with no ``CTYPE`` at all has pixel indices and nothing else: astropy will still build a linear WCS for it
    (``CDELT = 1``, ``CTYPE = ""``), and recording that would write world-coordinate cards into the file on the way
    back out that were never in it. So the rule is the FITS one: an axis has a world coordinate when the header names
    its type.
    """
    return any(str(header.get(f"CTYPE{i}", "") or "").strip() for i in range(1, ndim + 1))


def _image_sample(path: str, hdul, index: int, *, name: str | None) -> Any:
    """An image or a cube: one data variable, the WCS cards, the pointing and the time."""
    from pinneapple_data.physical_sample import PhysicalSample

    afits, WCS = _astro()
    import xarray as xr

    hdu = hdul[index]
    header = hdu.header
    data = np.array(hdu.data)                                    # a copy: the HDUList is closed on the way out
    ndim = data.ndim
    if ndim == 0:
        raise ValueError(f"HDU {index} ({hdu.name}) has a scalar array; this is not an image, a cube or a table")
    wcs = None
    if _names_an_axis(header, ndim):
        try:
            wcs = WCS(header, hdul)
            if wcs.naxis == 0:
                wcs = None
        except Exception:                                        # a broken WCS does not stop an image being readable
            wcs = None
    var = name or str(header.get("EXTNAME", "") or "image").strip().lower() or "image"
    if var == "primary":
        var = "image"
    dims = _dims_for(ndim, wcs)
    bunit = str(header.get("BUNIT", "") or "").strip()

    coords: dict[str, Any] = {}
    for j, dim in enumerate(dims):                                # dims is in numpy order: dims[j] <-> shape[j]
        n = int(data.shape[j])
        coords[dim] = (dim, np.arange(n, dtype=float))
        if wcs is not None:
            world, unit = _axis_world(wcs, ndim - j - 1, n)
            if world is not None:
                coords[f"{dim}_world"] = (dim, world, {"units": unit} if unit else {})
    ds = xr.Dataset({var: (dims, data, {"units": bunit} if bunit else {})}, coords=coords)
    sky = _sky_provenance(wcs, data.shape)
    time = _time_provenance(header, wcs)
    kind = "image" if ndim == 2 else ("spectrum" if ndim == 1 else "cube")
    provenance: dict[str, Any] = {
        "reader": "astropy.io.fits",
        "astropy_version": _version(),
        "format": f"fits-{kind}",
        "source": path,
        "hdu": int(index),
        "hdu_name": str(hdu.name),
        "ndim": int(ndim),
        "shape": [int(s) for s in data.shape],
        "dims": list(dims),
        "bunit": bunit,
        "header": {k: _plain(header[k]) for k in _CURATED if k in header},
    }
    if wcs is not None:
        provenance["wcs"] = {"cards": _cards(wcs.to_header(relax=True)),
                             "axis_types": [str(t) for t in (wcs.axis_type_names or [])],
                             "axis_units": [str(u) for u in (wcs.world_axis_units or [])],
                             "naxis": int(wcs.naxis)}
    if sky:
        provenance["sky"] = sky
    if time:
        provenance["time"] = time
    schema: dict[str, Any] = {
        "units": {var: bunit} if bunit else {},
        "variables": {var: {"unit": bunit, "kind": "field", "dims": list(dims)}},
    }
    extras: dict[str, Any] = {"fits_kind": "image", "fits_header": header.copy(), "fits_hdu": int(index)}
    if wcs is not None:
        extras["wcs"] = wcs
    if sky:
        extras["sky"] = _sky_object(wcs, data.shape)
    if time.get("parsed"):
        extras["time"] = _time_object(time)
    return PhysicalSample(
        state=ds, schema=schema,
        domain={"type": "grid", "dims": list(dims), "shape": [int(s) for s in data.shape],
                "kind": kind, "coordinates": "pixel indices; world values in xarray coords and provenance['wcs']"},
        provenance=provenance, extras=extras)


def _version() -> str:
    try:
        import astropy
        return str(astropy.__version__)
    except Exception:                                            # pragma: no cover - astropy is importable here
        return ""


def _sky_object(wcs, shape: Sequence[int]):
    """The pointing at the centre of the array as a ``SkyCoord``, or ``None``."""
    if wcs is None or not getattr(wcs, "has_celestial", False):
        return None
    try:
        ny, nx = int(shape[-2]), int(shape[-1])
        return wcs.celestial.pixel_to_world([(nx - 1) / 2.0], [(ny - 1) / 2.0])[0]
    except Exception:                                            # a WCS without enough cards for a SkyCoord
        return None


def _time_object(prov: dict[str, Any]):
    """The ``Time`` behind ``provenance["time"]``, rebuilt from the Julian date and the scale."""
    from astropy.time import Time

    return Time(prov["jd"], format="jd", scale=prov["scale"])


#: The time scales the FITS standard lets ``TCTYPn`` name on a time column; anything else is a coordinate type.
_TIME_SCALES = ("UTC", "TAI", "TT", "TDB", "TCG", "TCB", "UT1", "LOCAL")


def _restore_time_columns(table, hdu) -> None:
    """Turn the ``jd1``/``jd2`` pair columns of a FITS time axis back into a ``Time``, in the scale ``TCTYPn`` names.

    The FITS standard writes a time coordinate as two doubles (a Julian date split in two) and marks the column with
    ``TCTYPn`` = a time scale. ``Table.read`` returns that pair as a plain ``(n, 2)`` column with unit ``"d"``; this
    restores what the file says it is. A pair the pair does not fit (one value per row, or a scale that is not one)
    is left alone -- guessing here would invent a time.
    """
    from astropy.time import Time

    header = hdu.header
    for i, column in enumerate(hdu.columns, start=1):
        scale = str(header.get(f"TCTYP{i}", "") or "").strip().upper()
        name = str(getattr(column, "name", "") or "").strip()
        if scale not in _TIME_SCALES or not name or name not in table.colnames:
            continue
        col = table[name]
        values = np.asarray(col)
        if values.ndim != 2 or values.shape[1] != 2:
            continue                                             # not the two-double form this convention defines
        table.replace_column(name, Time(values[:, 0], values[:, 1], format="jd", scale=scale.lower()))


def _restore_sky_column_names(table, hdu) -> None:
    """Rename a decoded ``SkyCoord`` from its frame (``"fk5"``) to the column name the file gave it (``"coord"``).

    The standard writes a sky coordinate as ``<name>.ra`` / ``<name>.dec``, and astropy's reader merges the pair into
    one ``SkyCoord`` named after the *frame*, so without this the sample would say ``fk5`` where the file says
    ``coord`` -- and the round trip would write a column with a different name than it read.
    """
    from astropy.coordinates import SkyCoord

    header = hdu.header
    prefixes: list[str] = []
    for i in range(1, len(hdu.columns) + 1):
        name = str(header.get(f"TTYPE{i}", "") or "")
        for suffix in (".ra", ".dec", ".lon", ".lat"):
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]
                if prefix and prefix not in prefixes:
                    prefixes.append(prefix)
                break
    sky = [str(name) for name, col in table.columns.items() if isinstance(col, SkyCoord)]
    for name, prefix in zip(sky, prefixes, strict=False):
        if name == prefix or prefix in table.colnames:
            continue
        table.rename_column(name, prefix)


def _table_sample(path: str, hdul, index: int) -> Any:
    """A binary (or ASCII) table: one data variable per column, TUNIT as the unit, ``Time``/``SkyCoord`` kept.

    Two FITS conventions are applied here that ``astropy.table.Table.read`` alone leaves half-decoded, both taken from
    the standard rather than invented:

    * a **time column** is written as a pair of Julian-date doubles with ``TCTYPn`` naming the time scale; the pair is
      read back as an ``astropy.time.Time`` (``format="jd1_jd2"``, that scale) and stored as MJD;
    * a **sky column** is written as ``<name>.ra`` / ``<name>.dec`` and decoded by astropy as one ``SkyCoord`` named
      after its *frame* (``"fk5"``), not after the column; the column's own name is recovered from ``TTYPEn`` so the
      sample is named the way the file names it.
    """
    import xarray as xr
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    from astropy.time import Time

    from pinneapple_data.physical_sample import PhysicalSample

    hdu = hdul[index]
    table = Table.read(path, hdu=index)                           # Table.read decodes the SkyCoord mixins
    _restore_time_columns(table, hdu)
    _restore_sky_column_names(table, hdu)
    data_vars: dict[str, Any] = {}
    coords: dict[str, Any] = {"row": np.arange(len(table), dtype=int)}
    units: dict[str, str] = {}
    columns: dict[str, Any] = {}
    described: list[dict[str, Any]] = []
    for col_name, col in table.columns.items():
        col_name = str(col_name)
        if isinstance(col, Time):
            values = np.asarray(col.mjd, dtype=float)
            data_vars[col_name] = (("row",) if values.ndim == 1 else ("row", f"{col_name}_dim"), values)
            columns[col_name] = col
            described.append({"name": col_name, "kind": "time", "format": str(col.format), "scale": str(col.scale),
                              "stored_as": "mjd"})
            continue
        if isinstance(col, SkyCoord):
            values = np.empty(len(table), dtype=object)
            for i in range(len(table)):
                values[i] = col[i]
            data_vars[col_name] = ("row", values)
            columns[col_name] = col
            described.append({"name": col_name, "kind": "sky", "frame": str(col.frame.name),
                              "stored_as": "object array of SkyCoord"})
            continue
        values = np.asarray(col)
        dims = ("row",) if values.ndim == 1 else ("row", f"{col_name}_dim")
        unit = str(getattr(col, "unit", "") or "")
        data_vars[col_name] = (dims, values, {"units": unit} if unit else {})
        if unit:
            units[col_name] = unit
        described.append({"name": col_name, "kind": "quantity" if unit else "column", "unit": unit,
                          "dtype": str(values.dtype), "shape": [int(s) for s in values.shape]})
    ds = xr.Dataset(data_vars, coords=coords)
    header = hdu.header
    primary = hdul[0].header                                   # a table's observation keywords live in the primary HDU
    curated: dict[str, Any] = {}
    for source in (header, primary):                           # the extension first, the primary fills the gaps
        for key in _CURATED:
            if key in source and key not in curated:
                curated[key] = _plain(source[key])
    time_prov = _time_provenance(header) or _time_provenance(primary)
    provenance: dict[str, Any] = {
        "reader": "astropy.io.fits",
        "astropy_version": _version(),
        "format": "fits-table",
        "source": path,
        "hdu": int(index),
        "hdu_name": str(hdu.name),
        "rows": int(len(table)),
        "columns": described,
        "header": curated,
    }
    if time_prov:
        provenance["time"] = time_prov
    schema: dict[str, Any] = {
        "units": units,
        "variables": {d["name"]: {"unit": d.get("unit", ""), "kind": d["kind"]} for d in described},
    }
    extras: dict[str, Any] = {"fits_kind": "table", "fits_header": header.copy(), "fits_hdu": int(index),
                              "table": table, "fits_primary_header": primary.copy()}
    if columns:
        extras["columns"] = columns
    if time_prov.get("parsed"):
        extras["time"] = _time_object(time_prov)
    return PhysicalSample(
        state=ds, schema=schema,
        domain={"type": "table", "rows": int(len(table)), "columns": [d["name"] for d in described]},
        provenance=provenance, extras=extras)


# ---------------------------------------------------------------------------------------------- accessors
def wcs_of(sample: Any):
    """The astropy ``WCS`` of a sample: the live object from the read, or one rebuilt from ``provenance["wcs"]``.

    Rebuilding works after the provenance has been through JSON (a saved manifest), which is what makes the WCS part of
    the record rather than a property of the process that wrote it. ``None`` when the sample has no WCS.
    """
    live = (getattr(sample, "extras", None) or {}).get("wcs")
    if live is not None:
        return live
    _, WCS = _astro()
    cards = ((getattr(sample, "provenance", None) or {}).get("wcs") or {}).get("cards")
    if not cards:
        return None
    try:
        return WCS(_header(cards))
    except Exception:                                            # a manifest written by a different astropy
        return None


def sky_of(sample: Any):
    """The pointing at the centre of the array as a ``SkyCoord``: the live object, or one from ``provenance["sky"]``."""
    live = (getattr(sample, "extras", None) or {}).get("sky")
    if live is not None:
        return live
    prov = (getattr(sample, "provenance", None) or {}).get("sky") or {}
    centre = prov.get("center")
    if not centre:
        return None
    from astropy.coordinates import SkyCoord

    return SkyCoord(centre["lon_deg"], centre["lat_deg"], unit="deg", frame=prov.get("frame", "icrs"))


def time_of(sample: Any):
    """The observation time as an astropy ``Time``: the live object, or one from ``provenance["time"]``; else ``None``.

    ``None`` also means the header's date could not be parsed -- ``provenance["time"]`` then says ``parsed: false``
    with the raw text and the reason, and that text is what :func:`save` writes back.
    """
    live = (getattr(sample, "extras", None) or {}).get("time")
    if live is not None:
        return live
    prov = (getattr(sample, "provenance", None) or {}).get("time") or {}
    if not prov.get("parsed"):
        return None
    return _time_object(prov)


# ---------------------------------------------------------------------------------------------- writing
def save(sample: Any, path: str | Path, *, field: str | None = None, overwrite: bool = True) -> Path:
    """Write a sample back to FITS, with the WCS, the unit and the time the sample carries.

    The header :func:`save` starts from is the one the file had (``extras["fits_header"]``), so ``DATE-OBS``,
    ``OBJECT``, the instrument keywords and every WCS card come back exactly as they went in; the structural cards are
    regenerated by astropy from the array actually written, so a header can never disagree with the data. ``BUNIT`` is
    set from the field's declared unit. Without a stored header, the cards in ``provenance["wcs"]`` are used.

    field : which data variable to write when the state holds more than one; the first one otherwise.
    """
    _astro()                                                     # the install line, before anything else
    extras = getattr(sample, "extras", None) or {}
    provenance = getattr(sample, "provenance", None) or {}
    kind = str(extras.get("fits_kind") or provenance.get("format") or "").replace("fits-", "")
    path = Path(path)
    if kind == "table":
        return _save_table(sample, path, overwrite=overwrite)
    return _save_image(sample, path, field=field, overwrite=overwrite)


def _save_image(sample: Any, path: Path, *, field: str | None, overwrite: bool) -> Path:
    afits, _ = _astro()
    state = sample.state
    extras = getattr(sample, "extras", None) or {}
    provenance = getattr(sample, "provenance", None) or {}
    names = [n for n in getattr(state, "data_vars", {}) if getattr(state[n], "ndim", 0) >= 1]
    if not names:
        raise ValueError("this sample holds no array to write: state is empty")
    var = field or names[0]
    if var not in names:
        raise ValueError(f"this sample has no numeric field {var!r} (it has {names})")
    data = np.asarray(state[var].values)
    header = extras["fits_header"].copy() if extras.get("fits_header") is not None else (
        _header(provenance["wcs"]["cards"]) if (provenance.get("wcs") or {}).get("cards") else afits.Header())
    for card in list(header.cards):                              # structure from the array, never from the old file
        key = str(card.keyword).upper()
        if key in _STRUCTURAL or key.startswith("NAXIS"):
            del header[card.keyword]
    unit = (getattr(sample, "schema", None) or {}).get("units", {}).get(var) or (
        state[var].attrs.get("units") if hasattr(state[var], "attrs") else "")
    if unit:
        header["BUNIT"] = str(unit)
    afits.PrimaryHDU(data=data, header=header).writeto(str(path), overwrite=overwrite)
    return path


def _save_table(sample: Any, path: Path, *, overwrite: bool) -> Path:
    from astropy import units as _u
    from astropy.table import Table

    afits, _ = _astro()
    state = sample.state
    extras = getattr(sample, "extras", None) or {}
    units = (getattr(sample, "schema", None) or {}).get("units", {}) or {}
    mixins = extras.get("columns") or {}
    if extras.get("table") is not None:                          # the table as read: mixins, dtypes and order intact
        table = extras["table"]
    else:
        cols = {}
        for name in getattr(state, "data_vars", {}):
            if name in mixins:
                cols[name] = mixins[name]
                continue
            values = np.asarray(state[name].values)
            unit = units.get(name) or (state[name].attrs.get("units") if hasattr(state[name], "attrs") else "")
            cols[name] = values * _u.Unit(unit) if unit else values
        table = Table(cols)
    table.write(str(path), format="fits", overwrite=overwrite)
    with afits.open(str(path), mode="update") as hdul:            # keep the keywords the original HDUs carried
        ext = hdul[1] if len(hdul) > 1 else hdul[0]
        for stored, target in ((extras.get("fits_header"), ext),
                               (extras.get("fits_primary_header"), hdul[0])):
            if stored is not None:
                _copy_cards(stored, target.header)
    return path


def _copy_cards(stored, header) -> None:
    """Copy the cards of a stored header onto a freshly written one, skipping the ones that describe the structure."""
    for card in stored.cards:
        key = str(card.keyword).upper()
        if not key or key in _STRUCTURAL or key.startswith("NAXIS"):
            continue
        if key.startswith(("TTYPE", "TFORM", "TUNIT", "TDIM", "TNULL", "TPOS")):
            continue
        try:
            header[key] = (card.value, str(card.comment))
        except Exception:                                        # a card FITS will not take back as it was
            continue
