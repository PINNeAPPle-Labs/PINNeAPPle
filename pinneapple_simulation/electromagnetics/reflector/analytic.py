"""Closed forms that check the physical-optics code of :mod:`.po`.

* :func:`disk_on_axis` -- the exact field on the axis of a flat PEC disk carrying the PO current of a normally
  incident plane wave (``J = 2 E0 / eta`` uniform). With ``A_x = mu J (e^{-jkz} - e^{-jkR_a}) / (2 j k)``,
  ``R_a = sqrt(z^2 + a^2)``, and ``d^2 A_x / dx^2 = -(k^2 A_x + d^2 A_x / dz^2) / 2`` on the axis (Helmholtz and
  symmetry)::

      E_x(z) = -j k eta J g / 2  +  eta J a^2 e^{-jkR_a} (k^2 / R_a^2 - j k / R_a^3) / (4 j k (j k))

  with ``g = (e^{-jkz} - e^{-jkR_a}) / (j k)``, which simplifies to
  ``E_x = -E0 [e^{-jkz} - (1 + z^2 / R_a^2 + j a^2 / (k R_a^3)) e^{-jkR_a} / 2]``: the wave reflected by the plane
  plus the edge ring, whose weight tends to ``(1 + cos theta_e) / 2`` and does not vanish for a large disk (unlike
  the obliquity factor of scalar Kirchhoff diffraction). This checks the quadrature and the near-field Green's function of :func:`.po.radiate`.
* :func:`go_aperture_taper_db` -- geometrical optics of a paraboloid fed at its focus by a ``cos^q`` feed: a ray
  leaving the focus at angle ``psi`` from the feed axis reaches the aperture plane with relative amplitude
  ``sqrt(D(psi)) / rho``, ``rho = F + z`` the focus-to-surface distance (feed pattern times the space attenuation
  of the spherical wave; C. A. Balanis, *Antenna Theory*, 4th ed., ch. 15, the aperture distribution of a
  paraboloid). In a compact range the
  collimated field keeps this taper along the rays, so it is the smooth part of the quiet-zone amplitude; the PO
  field adds the ripple of the edge diffraction.
"""
from __future__ import annotations

import numpy as np

from .po import ETA0

__all__ = ["disk_on_axis", "go_aperture_taper_db"]


def disk_on_axis(z, radius: float, k: float, E0: float = 1.0) -> np.ndarray:
    """E_x (complex) at heights ``z`` > 0 on the axis of a PEC disk of ``radius`` under a normally incident
    x-polarised plane wave of amplitude ``E0`` (PO current, exact integral)."""
    z = np.asarray(z, float)
    J = 2 * E0 / ETA0
    Ra = np.sqrt(z ** 2 + radius ** 2)
    f = np.exp(-1j * k * Ra)
    g = (np.exp(-1j * k * z) - f) / (1j * k)
    S = -(1 / (1j * k)) * f * radius ** 2 * (k ** 2 / Ra ** 2 - 1j * k / Ra ** 3)       # k^2 g + g''
    return -1j * k * ETA0 * J / 2 * g - ETA0 * J / (4j * k) * S


def go_aperture_taper_db(x, y, focal: float, q: float, feed_axis) -> np.ndarray:
    """Relative GO amplitude [dB, 0 at the maximum over the given points] at aperture points ``(x, y)`` of the
    paraboloid ``z = (x^2 + y^2) / (4F)`` fed at its focus by a ``cos^q`` feed pointing along ``feed_axis``."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = (x ** 2 + y ** 2) / (4 * focal)
    d = np.stack([x, y, z - focal], -1)
    rho = np.linalg.norm(d, axis=-1)
    a = np.asarray(feed_axis, float) / np.linalg.norm(feed_axis)
    cpsi = (d @ a) / rho
    amp = np.where(cpsi > 0, np.clip(cpsi, 0, 1) ** q, 0.0) / rho
    db = 20 * np.log10(np.maximum(amp, 1e-300))
    return db - db.max()
