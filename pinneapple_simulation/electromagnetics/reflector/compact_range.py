"""A single offset-reflector compact range: feed, reflector, quiet zone, and its parameter studies.

:class:`CompactRange` holds the design (frequency, :class:`~.geometry.OffsetReflector`, a ``cos^q`` feed of a given
gain at the focus aimed at the section centre, the quiet-zone plane, diameter and centre). :func:`evaluate` meshes
the reflector, computes the PO current and the exact near field on the quiet-zone plane (and wider, to find the
usable size) and returns the :func:`~.qz.qz_metrics` with the cost. :func:`scaled` changes the section size keeping
the clearance below it (the feed stays out of the collimated beam) and the quiet zone on the section's centre line.
:func:`sweep` runs a grid of feed gains, section sizes and roll radii into a :class:`~..antenna.runner.ResultTable`.

The default parameters are a generic example (3 GHz, F = 1.5 m, a 1.2 m section), not any particular range.
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from .geometry import OffsetReflector, mesh_reflector
from .po import C0, cosq_feed, po_currents, q_from_gain, radiate, wavenumber
from .qz import disk_grid, qz_metrics

__all__ = ["CompactRange", "evaluate", "qz_field", "scaled", "sweep", "go_check"]


@dataclass
class CompactRange:
    freq_hz: float = 3e9
    reflector: OffsetReflector = None                    # default below
    feed_gain_dbi: float = 12.0
    pol: str = "V"                                       # "V": E along y at boresight; "H": along x
    qz_z: float = 3.0                                    # quiet-zone plane, z [m] (vertex plane z = 0)
    qz_diameter: float = 0.6
    qz_centre_y: float | None = None                     # default: the section centre height
    feed_offset: Sequence[float] = (0.0, 0.0, 0.0)      # feed phase centre minus the focus [m]
    mesh_per_wavelength: float = 8.0
    eval_factor: float = 1.6                             # field computed over eval_factor x the QZ diameter
    qz_step_per_wavelength: float = 6.0

    def __post_init__(self):
        if self.reflector is None:
            self.reflector = OffsetReflector(focal=1.5, half_x=0.6, half_y=0.6, centre_y=0.9, n=6.0, roll=0.15)

    @property
    def wavelength(self) -> float:
        return C0 / self.freq_hz

    @property
    def feed_q(self) -> float:
        return q_from_gain(self.feed_gain_dbi)

    @property
    def qz_centre(self) -> np.ndarray:
        cy = self.reflector.centre_y if self.qz_centre_y is None else self.qz_centre_y
        return np.array([0.0, cy, self.qz_z])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reflector"] = asdict(self.reflector)
        return d


def scaled(cr: CompactRange, side: float | None = None, roll: float | None = None,
           feed_gain_dbi: float | None = None) -> CompactRange:
    """``cr`` with a section of ``side`` (square, same exponent), roll radius ``roll`` and feed gain; the clearance
    between the axis and the section's lower junction stays, the quiet zone follows the section centre line."""
    r = cr.reflector
    if side is not None:
        clear = r.centre_y - r.half_y
        r = replace(r, half_x=side / 2, half_y=side / 2, centre_y=clear + side / 2)
    if roll is not None:
        r = replace(r, roll=roll)
    return replace(cr, reflector=r, feed_gain_dbi=cr.feed_gain_dbi if feed_gain_dbi is None else feed_gain_dbi,
                   qz_centre_y=None if side is not None else cr.qz_centre_y)


def qz_field(cr: CompactRange, points_xy: np.ndarray | None = None) -> dict[str, Any]:
    """PO field on the quiet-zone plane: ``{"xy", "E" (M, 3), "mesh", "seconds"}``; ``xy`` relative to the QZ
    centre (default: a disk of ``eval_factor`` x the QZ diameter, step lambda / ``qz_step_per_wavelength``)."""
    t0 = time.perf_counter()
    lam = cr.wavelength
    k = wavenumber(cr.freq_hz)
    mesh = mesh_reflector(cr.reflector, lam / cr.mesh_per_wavelength)
    feed = cr.reflector.focus + np.asarray(cr.feed_offset, float)
    axis = cr.reflector.centre - feed
    pol = (0.0, 1.0, 0.0) if cr.pol.upper() == "V" else (1.0, 0.0, 0.0)
    _, H = cosq_feed(mesh.centres, k, position=feed, axis=axis, q=cr.feed_q, pol=pol)
    J = po_currents(mesh.normals, mesh.centres, H, source=feed)
    if points_xy is None:
        points_xy = disk_grid(cr.eval_factor * cr.qz_diameter, lam / cr.qz_step_per_wavelength)
    c = cr.qz_centre
    obs = np.c_[points_xy[:, 0] + c[0], points_xy[:, 1] + c[1], np.full(len(points_xy), c[2])]
    E = radiate(mesh.centres, mesh.areas, J, obs, k)
    return {"xy": points_xy, "E": E, "mesh": mesh, "seconds": time.perf_counter() - t0}


def evaluate(cr: CompactRange, *, amp_tol: float = 0.5, phase_tol: float = 5.0) -> dict[str, Any]:
    """Quiet-zone metrics of ``cr`` (co-polar: E_y for "V", E_x for "H") with the mesh size and the cost."""
    f = qz_field(cr)
    co, xp = (1, 0) if cr.pol.upper() == "V" else (0, 1)
    m = qz_metrics(f["xy"], f["E"][:, co], cr.qz_diameter, E_x=f["E"][:, xp], amp_tol=amp_tol, phase_tol=phase_tol)
    r = cr.reflector
    m.update({"feed_gain_dBi": cr.feed_gain_dbi, "feed_q": cr.feed_q, "side": 2 * r.half_x, "roll": r.roll,
              "n_triangles": int(len(f["mesh"].areas)), "n_qz_points": int(len(f["xy"])),
              "eval_diameter": cr.eval_factor * cr.qz_diameter, "seconds": round(f["seconds"], 3),
              "f_GHz": cr.freq_hz / 1e9})
    return m


def sweep(base: CompactRange, gains: Iterable[float], sides: Iterable[float], rolls: Iterable[float | None] = (None,),
          *, log=None):
    """Every combination; returns a :class:`~pinneapple_simulation.electromagnetics.antenna.runner.ResultTable`
    (``tag``, inputs, metrics; ``f_GHz`` constant)."""
    from ..antenna.runner import ResultTable
    rows = []
    for roll in rolls:
        for s in sides:
            for g in gains:
                cr = scaled(base, side=s, roll=roll, feed_gain_dbi=g)
                m = evaluate(cr)
                rows.append({"tag": f"g{g:g}_s{s:g}_r{cr.reflector.roll:g}", "feed_gain_dBi": g, "side": s,
                             "roll": cr.reflector.roll, **{k: v for k, v in m.items()
                                                            if k not in ("feed_gain_dBi", "side", "roll")}})
                if log:
                    log(f"gain {g:g} dBi, side {s:g} m, roll {cr.reflector.roll:g} m: ripple "
                        f"{m['ripple_pp_dB']:.2f} dB, phase {m['phase_pp_deg']:.1f} deg, usable "
                        f"{m['usable_diameter']:.2f} m ({m['seconds']:.1f} s)")
    return ResultTable(rows)


def go_check(cr: CompactRange) -> dict[str, float]:
    """PO against geometrical optics on the quiet zone: the quadratic fit of the PO amplitude (its smooth part)
    against the GO aperture taper of :func:`~.analytic.go_aperture_taper_db` at the same (x, y); returns the
    largest and RMS difference in dB over the QZ disk (the remainder is the diffraction ripple)."""
    from .analytic import go_aperture_taper_db
    from .qz import _fit_resid
    f = qz_field(cr)
    co = 1 if cr.pol.upper() == "V" else 0
    c = cr.qz_centre
    xy = f["xy"]
    inside = np.hypot(xy[:, 0], xy[:, 1]) <= cr.qz_diameter / 2
    A = 20 * np.log10(np.abs(f["E"][inside, co]))
    fit, _ = _fit_resid(xy[inside, 0], xy[inside, 1], A)
    feed = cr.reflector.focus + np.asarray(cr.feed_offset, float)
    go = go_aperture_taper_db(xy[inside, 0] + c[0], xy[inside, 1] + c[1], cr.reflector.focal, cr.feed_q,
                              cr.reflector.centre - feed)
    j = int(np.argmin(np.hypot(xy[inside, 0], xy[inside, 1])))
    d = (fit - fit[j]) - (go - go[j])
    return {"max_abs_dB": float(np.abs(d).max()), "rms_dB": float(np.sqrt(np.mean(d ** 2))),
            "go_taper_dB": float(np.ptp(go)), "po_taper_dB": float(np.ptp(fit))}
