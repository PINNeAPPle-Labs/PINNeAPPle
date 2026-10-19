"""Step skew, manufacturing tolerances and a 3-D view of the field of an IPM machine.

Step skew (multi-slice 2-D): a rotor of ``K`` axial segments, segment ``k`` turned by ``delta_k = s (k - (K-1)/2)``
(step ``s``, total skew ``(K-1) s``). Each segment is a 2-D slice of the same stator with the same phase currents,
so the torque at rotor angle ``theta`` is ``(1/K) sum_k T_2D(theta + delta_k; i_abc(theta))`` and the flux linkage
the mean of the slices (end effects and the inter-segment axial flux are neglected, as in every multi-slice model).
For a quantity with a harmonic of order ``h`` per period ``lam`` the mean over the slices multiplies that harmonic by
the step-skew factor ``|sin(K h pi s / lam) / (K sin(h pi s / lam))|`` (:func:`step_skew_factor`).

Tolerances: :func:`tolerance_model` maps samples of (remanence scale, magnet thickness, magnet width, air gap) to the
FEA mean torque and ripple; with :func:`pinneapple_analysis.uncertainty.tolerance.propagate` (JCGM 101 Monte Carlo,
Wilson interval of the yield) it gives the output distributions directly from the FEA or from a surrogate trained on
it.

3-D view: :func:`skewed_field_3d` extrudes the 2-D solution of every segment into prisms (wedges) along the stack,
each segment turned by its skew angle; :func:`~.export.write_vtu` writes it for ParaView.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np

from .geometry import MachineGeometry, geometry_errors
from .pmsm import IPMMachine

__all__ = ["step_offsets", "step_skew_factor", "skewed_torque", "skewed_no_load", "tolerance_model",
           "skewed_field_3d", "TOLERANCE_VARIABLES"]

TWO_PI = 2.0 * math.pi
TOLERANCE_VARIABLES = ("Br_scale", "H3", "W4", "airgap")


def step_offsets(steps: int, step_angle: float) -> np.ndarray:
    """Rotor angles of the ``steps`` segments, centred on zero."""
    return step_angle * (np.arange(steps) - 0.5 * (steps - 1))


def step_skew_factor(h, steps: int, step_angle: float, period: float):
    """Factor on harmonic ``h`` (per ``period``) of the segment-mean of a quantity: ``|sin(K x) / (K sin x)|``,
    ``x = h pi step_angle / period``."""
    x = np.asarray(h, float) * math.pi * step_angle / period
    small = np.abs(np.sin(x)) < 1e-15
    return np.where(small, 1.0, np.abs(np.sin(steps * x) / (steps * np.where(small, 1.0, np.sin(x)))))


def skewed_no_load(machine: IPMMachine, thetas: Sequence[float], steps: int, step_angle: float) -> dict:
    """Cogging torque and phase flux linkages of the step-skewed rotor at the rotor angles ``thetas``."""
    off = step_offsets(steps, step_angle)
    T = np.zeros(len(thetas))
    psi = np.zeros((len(thetas), 3))
    for d in off:
        A0 = None
        for i, th in enumerate(thetas):
            ms = machine.solve(th + d, A0=A0)
            A0 = ms.A
            T[i] += machine.torque(ms) / steps
            psi[i] += machine.flux_linkage(ms) / steps
    return {"thetas": np.asarray(thetas), "cogging": T, "psi": psi, "offsets": off}


def skewed_torque(machine: IPMMachine, i_d: float, i_q: float, thetas: Sequence[float], steps: int,
                  step_angle: float) -> np.ndarray:
    """Torque of the step-skewed rotor at rotor angles ``thetas`` (mechanical), with phase currents fixed in the dq
    frame of the rotor at ``theta`` (the segments see the same stator currents, turned by their offset)."""
    off = step_offsets(steps, step_angle)
    T = np.zeros(len(thetas))
    for d in off:
        A0 = None
        for i, th in enumerate(thetas):
            ms = machine.solve(th + d, machine.currents_dq(th, i_d, i_q), A0=A0)
            A0 = ms.A
            T[i] += machine.torque(ms) / steps
    return T


def tolerance_model(base: MachineGeometry, I_peak: float = 250.0, gamma_deg: float = 50.0, n_gap: int = 288,
                    n_positions: int = 6, variables: Sequence[str] = TOLERANCE_VARIABLES, **machine_kw):
    """``fn(X) -> (n, 2)`` [mean torque (N m), ripple] for rows of ``variables``: ``Br_scale`` (remanence factor),
    ``H3`` magnet thickness, ``W4`` magnet width (m), ``airgap`` (m, the rotor radius changes, the stator stays).
    Rows whose geometry is invalid give NaN. The function counts its FEA calls in ``fn.calls``."""

    def fn(X):
        X = np.atleast_2d(X)
        out = np.full((len(X), 2), np.nan)
        for i, x in enumerate(X):
            p = dict(zip(variables, map(float, x), strict=True))
            rot = base.rotor
            kw = {k: p[k] for k in ("H3", "W4") if k in p}
            if "airgap" in p:
                kw["r_out"] = base.stator.r_in - p["airgap"]
            rot = dataclasses.replace(rot, **kw)
            if geometry_errors(rot.sector(1e-3)):
                continue
            m = IPMMachine(dataclasses.replace(base, rotor=rot), n_gap=n_gap, **machine_kw)
            if "Br_scale" in p:
                m.magnet = dataclasses.replace(m.magnet, Br20=m.magnet.Br20 * p["Br_scale"])
            op = m.operating_point(I_peak=I_peak, gamma=math.radians(gamma_deg), n_positions=n_positions)
            fn.calls += 1
            out[i] = op["torque_mean"], op["ripple"]
        return out

    fn.calls = 0
    return fn


def skewed_field_3d(machine: IPMMachine, theta: float, currents, steps: int, step_angle: float,
                    layers_per_step: int = 1) -> dict:
    """Prism (wedge) mesh of the stack with the 2-D field of every segment: points (N, 3), wedges (M, 6), cell data
    ``B`` (M, 3), ``|B|`` and region index. Segment ``k`` occupies ``z`` in ``[k, k+1] L / K``."""
    L = machine.length
    pts, cells, B, reg = [], [], [], []
    count = 0
    for k, d in enumerate(step_offsets(steps, step_angle)):
        ms = machine.solve(theta + d, currents)
        mesh = ms.mesh
        n = mesh.n_nodes
        zs = np.linspace(k * L / steps, (k + 1) * L / steps, layers_per_step + 1)
        for j in range(layers_per_step):
            for z in (zs[j], zs[j + 1]):
                pts.append(np.column_stack([mesh.points, np.full(n, z)]))
            t = mesh.tris + count
            cells.append(np.hstack([t, t + n]))
            B.append(np.column_stack([ms.B, np.zeros(len(ms.B))]))
            reg.append(mesh.region)
            count += 2 * n
    B = np.vstack(B)
    return {"points": np.vstack(pts), "wedges": np.vstack(cells), "B": B, "Bmag": np.hypot(B[:, 0], B[:, 1]),
            "region": np.concatenate(reg), "region_names": machine.mm.region_names}
