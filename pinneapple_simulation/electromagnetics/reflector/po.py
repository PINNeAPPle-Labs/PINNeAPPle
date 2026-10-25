"""Physical optics on a triangulated PEC surface: feed fields, PO currents and their exact radiated field.

* :func:`cosq_feed` -- an ideal ``cos^q`` feed (Ludwig-3 co-polarisation, no back radiation), normalised to 1 W
  radiated: ``|E| r = sqrt(eta D(psi) / (2 pi))`` with ``D(psi) = 2 (2q + 1) cos^(2q) psi`` for ``psi < 90 deg``;
  :func:`q_from_gain` / :func:`gain_from_q` convert between ``q`` and the feed gain;
* :func:`po_currents` -- ``J = 2 n x H_inc`` on the lit triangles (normal facing the source), zero in shadow;
* :func:`radiate` -- the field of surface currents at any point, near or far, with the complete free-space dyadic
  Green's function (no far-field approximation)::

      E(r) = -j k eta / (4 pi) sum_i A_i e^{-jkR}/R [ (1 - j/(kR) - 1/(kR)^2) J_i
                                                      - (1 - 3j/(kR) - 3/(kR)^2) (J_i . R^) R^ ]

  one-point (centroid) quadrature per triangle, so triangles of lambda/8 or smaller; computed in chunks.

``exp(+j omega t)`` convention, SI units. PO neglects the fringe (non-uniform) edge currents (PTD) and multiple
reflections; the edge diffraction it does contain is that of the truncated PO current.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ETA0", "C0", "wavenumber", "cosq_feed", "q_from_gain", "gain_from_q", "po_currents", "radiate",
           "plane_wave"]

C0 = 299_792_458.0
ETA0 = 376.730313412


def wavenumber(f_hz: float) -> float:
    return 2 * np.pi * f_hz / C0


def q_from_gain(gain_dbi: float) -> float:
    """``q`` of a ``cos^q`` feed of directivity ``gain_dbi`` (``D = 2 (2q + 1)``)."""
    return (10 ** (gain_dbi / 10) / 2 - 1) / 2


def gain_from_q(q: float) -> float:
    return float(10 * np.log10(2 * (2 * q + 1)))


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cosq_feed(points, k: float, *, position, axis, q: float, pol=(0.0, 1.0, 0.0)):
    """E and H (N, 3) complex of a ``cos^q`` feed at ``position`` pointing along ``axis``.

    ``pol``: the co-polar direction at boresight (projected perpendicular to ``axis``); the field is Ludwig-3
    polarised (co-polar everywhere in the forward hemisphere), radiating 1 W."""
    P = np.asarray(points, float)
    a = _unit(axis)
    pc = np.asarray(pol, float) - np.dot(pol, a) * a
    pc /= np.linalg.norm(pc)
    pxc = np.cross(pc, a)                                          # local x: (pxc, pc, a) right-handed
    d = P - np.asarray(position, float)
    r = np.linalg.norm(d, axis=1)
    rh = d / r[:, None]
    cpsi = rh @ a
    D = np.where(cpsi > 0, 2 * (2 * q + 1) * np.clip(cpsi, 0, 1) ** (2 * q), 0.0)
    amp = np.sqrt(ETA0 * D / (2 * np.pi)) / r * np.exp(-1j * k * r)
    # Ludwig-3 co-polar unit vector: theta^ sin(phi) + phi^ cos(phi) about the feed axis, phi from pxc
    st = np.sqrt(np.clip(1 - cpsi ** 2, 0, None))
    cphi = np.where(st > 1e-12, (rh @ pxc) / np.maximum(st, 1e-12), 1.0)
    sphi = np.where(st > 1e-12, (rh @ pc) / np.maximum(st, 1e-12), 0.0)
    th = (cpsi[:, None] * (cphi[:, None] * pxc + sphi[:, None] * pc) - st[:, None] * a)
    ph = -sphi[:, None] * pxc + cphi[:, None] * pc
    e = sphi[:, None] * th + cphi[:, None] * ph
    E = amp[:, None] * e
    H = np.cross(rh, E) / ETA0
    return E, H


def plane_wave(points, k: float, direction, pol, amplitude: float = 1.0):
    """E, H of a plane wave travelling along ``direction`` with E along ``pol`` (made transverse)."""
    kh = _unit(direction)
    p = np.asarray(pol, float) - np.dot(pol, kh) * kh
    p /= np.linalg.norm(p)
    ph = np.exp(-1j * k * (np.asarray(points, float) @ kh))
    E = amplitude * ph[:, None] * p
    H = np.cross(kh, E) / ETA0
    return E, H


def po_currents(normals, centres, H_inc, source=None):
    """``J = 2 n x H`` (N, 3) on lit triangles. ``source``: a point (feed) or ``None`` for a field arriving
    against the normals (plane wave): a triangle is lit when its normal faces the source."""
    n = np.asarray(normals, float)
    J = 2 * np.cross(n, H_inc)
    if source is not None:
        lit = np.einsum("ij,ij->i", n, np.asarray(source, float) - centres) > 0
        J[~lit] = 0
    return J


def radiate(centres, areas, J, obs, k: float, chunk: int = 2_000_000):
    """E (M, 3) complex at the points ``obs`` (M, 3) radiated by currents ``J`` (N, 3) on triangles of
    ``areas`` (N,) at ``centres`` (N, 3), with the complete dyadic Green's function."""
    C = np.asarray(centres, float)
    JA = np.asarray(J) * np.asarray(areas, float)[:, None]
    keep = np.any(JA != 0, axis=1)
    C, JA = C[keep], JA[keep]
    Ob = np.atleast_2d(np.asarray(obs, float))
    out = np.zeros((len(Ob), 3), complex)
    step = max(1, chunk // max(len(C), 1))
    for i in range(0, len(Ob), step):
        Rv = Ob[i:i + step, None, :] - C[None, :, :]                       # (m, n, 3)
        R = np.linalg.norm(Rv, axis=2)
        Rh = Rv / R[..., None]
        kR = k * R
        G = np.exp(-1j * kR) / R
        a = 1 - 1j / kR - 1 / kR ** 2
        b = 1 - 3j / kR - 3 / kR ** 2
        JdR = np.einsum("nc,mnc->mn", JA, Rh)
        out[i:i + step] = np.einsum("mn,nc->mc", G * a, JA) - np.einsum("mn,mnc->mc", G * b * JdR, Rh)
    return out * (-1j * k * ETA0 / (4 * np.pi))
