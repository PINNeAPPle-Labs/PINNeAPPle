- **Astropy interop: FITS, units, sky coordinates and time on a `PhysicalSample`, plus galactic dynamics.** New
  `pinneapple_data.astro`: `fits.load(path, hdu=...)` reads an image, a cube or a (binary) table into a
  `PhysicalSample` -- units from `BUNIT`/`TUNIT`, the WCS recorded card for card in `provenance["wcs"]`, the pointing
  in `provenance["sky"]`, the observation time (with its `TIMESYS` scale) in `provenance["time"]`, and a table's
  `Time`/`SkyCoord` columns kept as such; `fits.save(sample, path)` writes it back with the WCS intact, which
  `tests/test_astro_fits_interop.py` checks on a real public FITS file (`public_data`, downloaded once, cached,
  SHA-256 verified, licence recorded). `units.quantity_of` / `units.set_quantity` convert between
  `astropy.units.Quantity` and a field of a sample both ways, converting a compatible unit and **recording** the
  conversion, and refusing a different dimension with `UnitMismatch` instead of writing a wrong number; the LBM
  solver's `Units` helper is covered both ways too. New `pinneapple_physics.astro.galactic` integrates an orbit in a
  mass model through galpy (`integrate_orbit`, `rotation_curve`) and reports the energy and `Lz` drift next to the
  trajectory. astropy, yt and galpy stay optional and are imported lazily: `pip install "pinneapple[astro]"`.
