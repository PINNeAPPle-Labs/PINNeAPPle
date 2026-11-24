"""Astrophysics data: the output of the simulation codes, read as ``PhysicalSample``.

    from pinneapple_data.astro import yt_reader
    s = yt_reader.load("output_0042")          # Enzo, FLASH, RAMSES, GADGET, AREPO, Athena++, generic HDF5 ...

The reader needs ``yt`` (``pip install "pinneapple[astro]"``) and imports it only when it is used.
"""
from . import yt_reader

__all__ = ["yt_reader"]
