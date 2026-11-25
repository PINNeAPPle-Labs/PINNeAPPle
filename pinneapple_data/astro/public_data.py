"""Small public FITS files, downloaded once and cached with their licence declared (#522).

The round trip this module exists for is the one a test cannot fake: a real FITS file, written by a real instrument's
pipeline, read into a ``PhysicalSample`` and written back out with its world coordinate system intact. That needs a
file from outside this repository -- and a file that may be redistributed, whose provenance can be checked.

    from pinneapple_data.astro import public_data
    path = public_data.fetch("uit_image")            # downloads once into the cache, verifies the SHA-256
    public_data.sidecar("uit_image")                 # {"licence", "citation", "url", "sha256", "retrieved", ...}

Three files, all from the FITS Support Office of NASA/GSFC, which publishes them as sample files for testing FITS
readers (``https://fits.gsfc.nasa.gov/samples/``), all obtained from NASA mission archives and in the public domain:

===============  =========  ==========================================================================
name             size       what it exercises
===============  =========  ==========================================================================
``uit_image``    526 kB     a 512x512 image: ``BUNIT``, a ``RA---TAN`` WCS, a 1995 ``DATE-OBS`` in the
                            pre-standard ``DD/MM/YY`` form
``wfpc2_cube``   684 kB     a 200x200x4 cube: a three-axis WCS whose third axis is ``GROUP_NUMBER``, plus an
                            ASCII table extension; ``DATE-OBS`` + ``TIME-OBS``
``iue_table``    48 kB      a binary table of 640-point spectra: vector columns, ``TUNIT`` per column
===============  =========  ==========================================================================

The cache is ``$PINNEAPPLE_FITS_CACHE`` if set, else ``$XDG_CACHE_HOME/pinneapple/fits``, else
``~/.cache/pinneapple/fits``. Every download is checked against the SHA-256 recorded here: a file that does not match
is deleted and refused, because a fixture that silently changes is worse than no fixture. Nothing is committed to the
repository, and nothing is downloaded until :func:`fetch` is called.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["FitsFixture", "FIXTURES", "cache_dir", "fetch", "list_fixtures", "sidecar"]

_LICENCE = ("Public domain: a sample FITS file published by the FITS Support Office of NASA/GSFC for testing FITS "
            "readers (https://fits.gsfc.nasa.gov/samples/), from the publicly archived observations of a NASA mission")


@dataclass(frozen=True)
class FitsFixture:
    """One public FITS file: where it comes from, what it is worth as a test, and the licence it is used under."""

    name: str
    url: str
    sha256: str
    size: int
    licence: str
    citation: str
    description: str
    hdu: int | None = None


FIXTURES: dict[str, FitsFixture] = {
    "uit_image": FitsFixture(
        name="uit_image",
        url="https://fits.gsfc.nasa.gov/samples/UITfuv2582gc.fits",
        sha256="facfd18c66debeb36f31f98d91440a4df9f2f79044ca788412f6453d2dd29826",
        size=538560,
        licence=_LICENCE,
        citation=("Astro-1 Ultraviolet Imaging Telescope (UIT), far-ultraviolet image of NGC 4151, 1995-03-13; "
                  "distributed by the NASA/GSFC FITS Support Office as the sample file UITfuv2582gc.fits"),
        description=("512x512 int16 image with BUNIT=ERGS/CM**2/S/ANGSTRM, a RA---TAN WCS and a DATE-OBS "
                     "('13/03/95') in the pre-1997 DD/MM/YY form: the unit, the WCS and the time a sample has to keep"),
    ),
    "wfpc2_cube": FitsFixture(
        name="wfpc2_cube",
        url="https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits",
        sha256="da7c0f1b6643850856cba100e9b3e8db76b80e91583eb088635c416a2b4161b3",
        size=699840,
        licence=_LICENCE,
        citation=("HST WFPC2 image u5780205r, 1999-02-20; distributed by the NASA/GSFC FITS Support Office as the "
                  "sample file WFPC2u5780205r_c0fx.fits"),
        description=("200x200x4 primary array cube whose third WCS axis is GROUP_NUMBER (one plane per CCD), plus an "
                     "ASCII table extension of world coordinate parameters"),
    ),
    "iue_table": FitsFixture(
        name="iue_table",
        url="https://fits.gsfc.nasa.gov/samples/IUElwp25637mxlo.fits",
        sha256="a1b604f9055035d46c5b95ca8ce0dd45b733b22bd3d772242a1aabf1b8f6517c",
        size=48960,
        licence=_LICENCE,
        citation=("IUE LWP low-dispersion spectrum of the sample file IUElwp25637mxlo.fits, distributed by the "
                  "NASA/GSFC FITS Support Office"),
        description=("binary table, one row, nine columns: 640-point vector columns (NET, BACKGROUND, SIGMA, "
                     "QUALITY, FLUX) with TUNIT per column"),
        hdu=1,
    ),
}


def cache_dir() -> Path:
    """Where the fixtures are kept: ``$PINNEAPPLE_FITS_CACHE``, else ``$XDG_CACHE_HOME/pinneapple/fits``, else
    ``~/.cache/pinneapple/fits``."""
    env = os.environ.get("PINNEAPPLE_FITS_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "pinneapple" / "fits"


def list_fixtures() -> list[dict]:
    """The catalogue as plain dicts -- name, size, licence, citation, URL -- without downloading anything."""
    return [asdict(fx) for fx in FIXTURES.values()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, *, directory: Path | str | None = None, tries: int = 3) -> Path:
    """The local path of a fixture, downloading and verifying it the first time.

    Raises ``KeyError`` for a name that is not in :data:`FIXTURES` (the message lists the ones that are), and
    ``RuntimeError`` when the download does not arrive or does not match the recorded SHA-256 -- in which case the
    partial file is removed, so a corrupted fixture is never left behind for the next call to trust.
    """
    if name not in FIXTURES:
        raise KeyError(f"unknown FITS fixture {name!r}; known: {sorted(FIXTURES)}")
    fx = FIXTURES[name]
    where = Path(directory) if directory is not None else cache_dir()
    where.mkdir(parents=True, exist_ok=True)
    target = where / f"{fx.name}.fits"
    if target.is_file() and _sha256(target) == fx.sha256:
        _write_sidecar(fx, where, retrieved=None)                # keep the licence next to a cache an older run left
        return target
    from pinneapple_data.datasets.download import fetch_cached

    downloaded = fetch_cached(fx.url, where, tries=tries)
    digest = _sha256(downloaded)
    if digest != fx.sha256:
        downloaded.unlink(missing_ok=True)
        raise RuntimeError(f"{fx.url} does not match the SHA-256 recorded for {name!r} "
                           f"(expected {fx.sha256}, got {digest}); the file was deleted, not used")
    os.replace(downloaded, target)
    _write_sidecar(fx, where, retrieved=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return target


def _write_sidecar(fx: FitsFixture, directory: Path, *, retrieved: str | None) -> Path:
    """Write the licence and provenance next to the file. An existing sidecar keeps its ``retrieved`` moment."""
    path = directory / f"{fx.name}.json"
    previous = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text())
        except Exception:                                        # an unreadable sidecar is rewritten, not trusted
            previous = {}
    meta = {**asdict(fx), "retrieved": retrieved or previous.get("retrieved", "")}
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return path


def sidecar(name: str, *, directory: Path | str | None = None) -> dict:
    """The licence and provenance written next to a downloaded fixture (fetching it if it is not there yet).

    What a report needs to cite: the SHA-256 of the bytes used, the URL they came from, the licence they are used
    under and the moment they were retrieved.
    """
    where = Path(directory) if directory is not None else cache_dir()
    fetch(name, directory=where)
    return json.loads((where / f"{name}.json").read_text())
