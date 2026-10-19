"""End windings: resistance and leakage inductance of the coil ends, outside the 2-D cross-section.

Each coil end is a thin conductor on the cylinder at the mean slot radius ``r_c``: axially out of the stack by the
overhang ``e``, then a half circle (in the developed cylinder surface) over the coil span, then back. Its length per
coil side, ``e + pi s / 4`` with ``s`` the developed coil span (half of ``2 e + pi s / 2`` per coil end), is the
end-turn length used by :func:`~.drive.prius_phase_resistance`, so resistance and inductance share one geometry.

:func:`end_winding_inductance` integrates the phase inductances of these conductors (Neumann's formula with the
thin-wire kernel, :func:`~..statics.filament_inductance`), both ends, with the coil bundle as a round conductor of
the slot's copper area, in air and with the stator end face as an infinitely permeable mirror; the two bound the
real laminated end face. The dq leakage inductance of a balanced three-phase winding is ``L_aa - M_ab``; it adds to
L_d and L_q (``Drive(..., L_end=...)``).
"""
from __future__ import annotations

import math

import numpy as np

from ..statics import filament_inductance

__all__ = ["coil_end_path", "end_winding_paths", "end_winding_inductance"]

TWO_PI = 2.0 * math.pi


def coil_end_path(r_c: float, phi_a: float, phi_b: float, overhang: float, n_arc: int = 48) -> np.ndarray:
    """Polyline (n, 3) of one coil end from the stack face at angle ``phi_a`` to ``phi_b`` (z >= 0 outside)."""
    span = r_c * (phi_b - phi_a)  # developed, signed
    xc = r_c * phi_a + 0.5 * span
    a = np.linspace(math.pi, 0.0, n_arc + 1) if span > 0 else np.linspace(0.0, math.pi, n_arc + 1)
    x = xc + 0.5 * abs(span) * np.cos(a)
    z = overhang + 0.5 * abs(span) * np.sin(a)
    if span < 0:
        x = xc - 0.5 * abs(span) * np.cos(a)
    phi = x / r_c
    pts = [(r_c * math.cos(phi_a), r_c * math.sin(phi_a), 0.0)]
    pts += [(r_c * math.cos(p), r_c * math.sin(p), zz) for p, zz in zip(phi, z, strict=True)]
    pts += [(r_c * math.cos(phi_b), r_c * math.sin(phi_b), 0.0)]
    return np.array(pts)


def end_winding_paths(layout: list, slots: int, coil_pitch: int, r_c: float, overhang: float) -> list[tuple[int, np.ndarray]]:
    """(phase, path) of every coil end of a single-layer winding: each slot with a positive coil side is joined to the
    slot ``coil_pitch`` further on, which must carry the same phase with the opposite sign."""
    out = []
    for k, (ph, sg) in enumerate(layout):
        if sg > 0:
            j = (k + coil_pitch) % slots
            if layout[j] != (ph, -1):
                raise ValueError(f"slot {j} does not return the coil of slot {k}")
            out.append((ph, coil_end_path(r_c, TWO_PI * k / slots, TWO_PI * (k + coil_pitch) / slots, overhang)))
    return out


def end_winding_inductance(geometry, layout: list, turns_per_coil: float, coil_pitch: int, overhang: float,
                           pieces_per_metre: float = 300.0) -> dict:
    """Phase self and mutual inductances of both coil ends (H), and the dq end leakage ``L_aa - M_ab``, in air and
    with the end face as an iron mirror."""
    st = geometry.stator
    r_c = st.r_in + st.H0 + st.H1 + 0.5 * st.H2
    paths = end_winding_paths(layout, st.slots, coil_pitch, r_c, overhang)
    radius = math.sqrt(st.slot_area() / math.pi)
    out = {"r_c": r_c, "bundle_radius": radius, "coils": len(paths),
           "end_turn_length_per_coil_end": float(np.sum(np.linalg.norm(np.diff(paths[0][1], axis=0), axis=1)))}
    for label, mirror in (("air", None), ("iron_face", 0.0)):
        M = filament_inductance([p for _, p in paths], radius, pieces_per_metre, mirror_z=mirror)
        ph = np.array([q for q, _ in paths])
        L = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                L[a, b] = M[np.ix_(ph == a, ph == b)].sum()
        L *= 2.0 * turns_per_coil ** 2  # both ends; every coil has the same turns, all coils of a phase in series
        out[label] = {"L_aa": float(np.mean(np.diag(L))), "M_ab": float(np.mean(L[~np.eye(3, dtype=bool)])),
                      "L_end_dq": float(np.mean(np.diag(L)) - np.mean(L[~np.eye(3, dtype=bool)]))}
    return out
