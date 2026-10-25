"""Reflector antennas and compact ranges by physical optics: geometry, PO currents and near fields, quiet zones.

* :mod:`.geometry` -- offset paraboloid sections (superellipse junction) with blended rolled edges, meshed;
* :mod:`.po` -- ``cos^q`` feeds, plane waves, PO currents, the exact radiated near field;
* :mod:`.analytic` -- closed forms that check them (PO disk on axis, GO aperture taper);
* :mod:`.qz` -- quiet-zone metrics (taper, ripple, phase, usable size);
* :mod:`.compact_range` -- a compact range (feed gain, section size, rolled edge) and its sweeps.

Guide: docs/guides/compact_range.md.
"""
from . import analytic, geometry, po, qz
from .compact_range import CompactRange, evaluate, go_check, qz_field, scaled, sweep
from .geometry import OffsetReflector, mesh_reflector
from .qz import qz_metrics

__all__ = ["analytic", "geometry", "po", "qz", "CompactRange", "evaluate", "go_check", "qz_field", "scaled", "sweep",
           "OffsetReflector", "mesh_reflector", "qz_metrics"]
