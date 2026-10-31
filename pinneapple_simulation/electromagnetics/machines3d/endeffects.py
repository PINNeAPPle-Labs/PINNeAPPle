"""End effects of a surface-PM machine against the stack length: the magnet flux a full-pitch search coil links in
3-D (stack plus air beyond each end) divided by the 2-D value times the stack length.

The search coil is a loop at a given radius (default: the middle of the slots): two axial sides a pole pitch
apart, centred on the axis of magnet 0 with the rotor aligned, joined by arcs beyond the stack ends at ``overhang``
from the end faces (0: in the end faces). Its flux linkage is the line integral of ``A`` around the loop, so it
counts the magnet flux that leaves the rotor through the end faces and enters the stator outside the 2-D
cross-section. The 2-D reference is the same stack without the end air (flux-parallel end planes, so the field
does not vary along the axis), so the ratio measures only what 2-D neglects.
"""
from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable, Sequence

import numpy as np

from ..machine import SPMMachine
from .spm import SPM3D, _fields

__all__ = ["search_coil_linkage", "line_integral", "end_effect_study"]


def search_coil_linkage(model: SPM3D, sol, radius: float | None = None, span: float | None = None,
                        overhang: float = 0.0, n: int = 96) -> float:
    """Flux linkage (Wb) of the search coil: ``span`` (default one pole pitch) centred on angle 0, at ``radius``
    (default mid-slot), the arcs ``overhang`` beyond the stack ends (they must lie inside the mesh)."""
    m = model.machine
    r = 0.5 * (m.r_bore + m.r_bore + m.slot_depth) if radius is None else radius
    span = math.pi / m.pole_pairs if span is None else span
    z1 = m.length / 2 + overhang
    if z1 > model.z[-1] + 1e-12:
        raise ValueError("the search coil's arcs lie outside the mesh: overhang > end_length")
    z1 = min(z1, model.z[-1] - 1e-9)                     # on the outer plane A is not defined beyond the mesh
    a, b = -span / 2, span / 2
    ps = np.linspace(a, b, n + 1)
    # the loop, counter-clockwise seen from +r on the side at angle a: up the side at b, back along the arc
    # at +z1, down the side at a, back along the arc at -z1 (positive flux: radially outward); the arcs are
    # polygons of n chords on the cylinder, the sides straight lines (sampled finely: they cross every layer)
    side_b = np.array([[r * math.cos(b), r * math.sin(b), -z1], [r * math.cos(b), r * math.sin(b), z1]])
    side_a = np.array([[r * math.cos(a), r * math.sin(a), z1], [r * math.cos(a), r * math.sin(a), -z1]])
    arc_top = np.column_stack([r * np.cos(ps[::-1]), r * np.sin(ps[::-1]), np.full_like(ps, z1)])
    arc_bot = np.column_stack([r * np.cos(ps), r * np.sin(ps), np.full_like(ps, -z1)])
    sp = model.space
    return (line_integral(sp, sol.a, side_b, samples=4000) + line_integral(sp, sol.a, arc_top)
            + line_integral(sp, sol.a, side_a, samples=4000) + line_integral(sp, sol.a, arc_bot))


def line_integral(space, a: np.ndarray, P: np.ndarray, samples: int = 64) -> float:
    """``int A . dl`` along the polyline ``P`` for the edge-element field ``a``, exact: ``A`` is linear in each
    tetrahedron, so every segment is cut where it crosses element faces (found by bisection between ``samples``
    points per segment that lie in different elements) and each piece is integrated by the midpoint rule. A plain
    midpoint rule is not enough: the tree gauge puts large gradients in ``A`` in the air, which a closed loop
    cancels only if every piece is integrated exactly (400 points per side gave 0.5 % of error on the search
    coil). ``samples`` must resolve the elements along a segment."""
    mesh = space.mesh
    P = np.asarray(P, float)
    p0, d = P[:-1], np.diff(P, axis=0)                      # (S, 3)
    S = len(d)
    t = np.linspace(0.0, 1.0, samples + 1)
    pts = p0[:, None, :] + t[None, :, None] * d[:, None, :]
    cells = mesh.locate(pts.reshape(-1, 3)).reshape(S, samples + 1)
    if (cells < 0).any():
        raise ValueError("line point outside the mesh")
    seg, k = np.nonzero(cells[:, 1:] != cells[:, :-1])
    lo, hi = t[k].copy(), t[k + 1].copy()
    cl = cells[seg, k]
    for _ in range(40):                                        # bisection to 1e-12 of the segment
        mid = 0.5 * (lo + hi)
        left = mesh.locate(p0[seg] + mid[:, None] * d[seg]) == cl
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
    total = 0.0
    for i in range(S):                                         # break points of segment i: samples and crossings
        bp = np.unique(np.r_[t, hi[seg == i]])
        mids = 0.5 * (bp[1:] + bp[:-1])
        A = space.evaluate(a, p0[i] + mids[:, None] * d[i])
        total += float(np.sum((A @ d[i]) * np.diff(bp)))
    return total


def end_effect_study(machine: SPMMachine | None = None, ratios: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
                     end_length: float = 0.03, end_layers: int = 3, layer_dz: float = 0.015, n_theta: int = 240,
                     overhangs: Sequence[float] = (0.0, 0.01), mu_r_iron: float = 1000.0,
                     progress: Callable[[str], None] | None = None) -> dict:
    """For each L/D (D: bore diameter ``2 r_bore``) the search-coil flux linkage in 3-D against 2-D, the
    ratio ``k = lambda_3D / lambda_2D`` per overhang and the mean gap flux density in the stack. The 2-D value is
    the same layered stack without end air (flux-parallel end planes: axially invariant), so ``k`` holds only the
    end effect and none of the in-plane discretisation. Returns ``{"D", "rows": [...], ...}``."""
    m0 = machine or SPMMachine()
    D = 2 * m0.r_bore
    rows = []
    for ld in ratios:
        t0 = time.perf_counter()
        L = ld * D
        m = SPMMachine(**{**_fields(m0), "length": L, "n_theta": n_theta})
        layers = max(2, math.ceil(L / layer_dz))
        ref = SPM3D(m, n_theta=n_theta, stack_layers=layers, mu_r_iron=mu_r_iron)    # no end air: 2-D
        sr = ref.solve(0.0)
        lam2, gap2 = search_coil_linkage(ref, sr), ref.gap_flux(sr)
        ref.release()
        del ref, sr                                   # one model and its factorisation in memory at a time
        gc.collect()
        model = SPM3D(m, n_theta=n_theta, stack_layers=layers, end_length=end_length, end_layers=end_layers,
                      mu_r_iron=mu_r_iron)
        s = model.solve(0.0)
        row = {"L_over_D": float(ld), "length_m": L, "stack_layers": layers, "tets": int(len(model.mesh.tets)),
               "unknowns": s.info["unknowns"], "gap_flux_T": model.gap_flux(s), "gap_flux_2d_T": gap2,
               "linkage_2d_Wb": lam2, "linkage_2d_per_m": lam2 / L}
        for o in overhangs:
            lam = search_coil_linkage(model, s, overhang=o)
            row[f"linkage_Wb_o{o * 1e3:g}mm"] = lam
            row[f"k_o{o * 1e3:g}mm"] = lam / lam2
        row["k_gap_flux"] = row["gap_flux_T"] / row["gap_flux_2d_T"]
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        model.release()
        del model, s
        gc.collect()
        if progress:
            progress(f"L/D {ld:g}: {row['tets']} tetrahedra, k = "
                     + ", ".join(f"{row[f'k_o{o * 1e3:g}mm']:.4f} (overhang {o * 1e3:g} mm)" for o in overhangs)
                     + f" ({row['seconds']:.0f} s)")
    return {"D": D, "end_length": end_length, "end_layers": end_layers, "n_theta": n_theta, "rows": rows}
