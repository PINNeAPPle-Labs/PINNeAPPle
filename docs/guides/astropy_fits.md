# FITS, units and sky coordinates: astronomy data as a `PhysicalSample`

An astronomy result arrives as a FITS file: the pixels, a world coordinate system that says what each pixel is
pointing at, the unit of the numbers, and the time of the observation. A pipeline that drops any of the four is
lossy in a way nobody notices until two files disagree. `pinneapple_data.astro.fits` reads all of it into the
library's own container and writes it back, and `pinneapple_data.astro.units` converts between an
`astropy.units.Quantity` and a field of that container -- or refuses, when the units are not the same dimension.

```python
from pinneapple_data.astro import fits, public_data, units

path = public_data.fetch("uit_image")          # a real FITS image, cached; SHA-256 and licence recorded beside it
sample = fits.load(path)                       # -> PhysicalSample

print(sample.provenance["bunit"])              # 'ERGS/CM**2/S/ANGSTRM', the unit the file declares
print(sample.provenance["sky"]["center"])      # the pointing at the centre, in degrees, in its own frame
print(fits.time_of(sample).isot)               # the observation time, with the scale the header names
print(sample.state["image"].dims)              # ('y', 'x'): numpy order, never transposed

# the pixels as an astropy Quantity. This file's 1995 header spells its unit 'ERGS/CM**2/S/ANGSTRM', which astropy
# refuses; so the caller reads the header and says what the numbers are -- the text in the sample is left alone
q = units.quantity_of(sample, "image", unit="erg / (cm2 s Angstrom)")
print(q.unit, q.max().value)

print(units.set_quantity(sample, "image_si", q.si, dims=("y", "x")))   # a new field: its unit is written down
print(sample.schema["units"]["image_si"])      # 'Pa / s' (astropy's SI form of W m-2 A-1), declared on the sample

fits.save(sample, "copy.fits")                 # back out: every WCS card, BUNIT and DATE-OBS as they came in
```

`fits.load(path, hdu=...)` takes an image, a cube or a binary table. A cube keeps the file's own axis order --
`(NAXIS3, NAXIS2, NAXIS1)` becomes `("spectral", "y", "x")`, with the third axis named from `CTYPE3` -- and a
separable axis also gets its world values as a coordinate (`spectral_world`, in Hz). A table becomes one variable per
column, with `TUNIT` as the unit, and a `Time` or `SkyCoord` column is kept as one.

## The round trip is the test

`fits.save` writes back the header the file had, with only the structural cards regenerated from the array being
written, so the WCS comes back card for card and so does everything else the header said. The proof is not a claim in
a docstring: `tests/test_astro_fits_interop.py` loads a real public image, saves it, loads the copy and compares the
world coordinates of five pixels, the `BUNIT`, the pointing and the observation time. It does the same for a
three-axis cube and for a table with vector columns.

## What it refuses, and why

- A field whose declared unit is a **different dimension** is never converted: `set_quantity(sample, "T", 3 * u.m)`
  raises `UnitMismatch` and leaves the field exactly as it was. A bare number raises too, because only the caller
  knows its unit.
- A **compatible** unit is converted, and the conversion is appended to `provenance["unit_conversions"]` with both
  units, the scale and the offset -- a conversion that leaves no trace is how a factor of 1e3 disappears.
- A date the FITS standard cannot parse (the pre-1997 `13/03/95` form) is kept **as text** in the provenance, and the
  value astropy's own WCS reader derives from it is recorded with that rule named. It is not guessed at.
- A file with no `CTYPE` has no world system worth recording, so no WCS is invented for it.

## Dependencies

`astropy` is optional and imported only when one of these functions is called: `import pinneapple_data` works on a
machine without it, and the call raises an `ImportError` that carries the install line. `pip install
"pinneapple[astro]"` brings astropy, yt (the simulation-code reader, `pinneapple_data.astro.yt_reader`) and galpy
(galactic orbits, `pinneapple_physics.astro.galactic`).

The fixtures in `pinneapple_data.astro.public_data` are sample FITS files published by the NASA/GSFC FITS Support
Office for testing readers; they are in the public domain, downloaded once into `~/.cache/pinneapple/fits` (or
`$PINNEAPPLE_FITS_CACHE`), verified against a recorded SHA-256, and their licence and retrieval time are written next
to them.
