"""Astrophysics data: the simulation codes' output and the archives' FITS files, read as ``PhysicalSample``.

    from pinneapple_data.astro import fits, public_data, units, yt_reader

    s = fits.load("ngc4151.fits")             # image, cube or table: WCS, units, sky coordinates and time kept
    fits.save(s, "copy.fits")                 # back out, the WCS card for card
    q = units.quantity_of(s, "image")         # the field as an astropy Quantity, in the unit the sample declares
    path = public_data.fetch("uit_image")     # a real public FITS file: cached, SHA-256 verified, licence recorded
    yt = yt_reader.load("output_0042")        # Enzo, FLASH, RAMSES, GADGET, AREPO, Athena++, generic HDF5 ...

``fits``, ``units`` and ``public_data`` need ``astropy``; ``yt_reader`` needs ``yt`` (both arrive with ``pip install
"pinneapple[astro]"``). None of them imports its dependency at package import, so a machine without the extra can still
``import pinneapple_data``: calling one of them raises an ImportError that says how to install it.
"""
from . import fits, public_data, units, yt_reader

__all__ = ["fits", "public_data", "units", "yt_reader"]
