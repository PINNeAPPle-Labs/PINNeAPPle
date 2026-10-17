"""Verification of the machine FEA: closed forms for the mesher and solver, internal consistency of the torque,
flux-linkage and inductance post-processing, and a cross-check against a published benchmark machine.

``run(quick=...)`` returns :class:`~..verification.CaseResult` rows (the same table format as the
electromagnetics suite); ``examples/electrodynamics/10_pmsm_prius.py --write-docs`` writes them into
docs/guides/electromagnetics.md. Published reference: torque against current angle of the 2004 Toyota Prius
motor at 250 A peak (Z. Yang, M. Krishnamurthy, I. P. Brown, "Electromagnetic and vibrational characteristic of
IPM over full torque-speed range", IEEE IEMDC 2013, pp. 295-302), values as tabulated in pyleecan's test suite
(``Tests/Validation/Electrical/test_EEC_PMSM.py``, Apache-2.0).
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import numpy as np

from ..analytic import magnetized_ring
from ..fem import Geometry, solve_magnetostatic
from ..materials import NU0
from ..verification import CaseResult, observed_orders
from .geometry import Polygon, SectorGeometry, arc, prius_2004
from .mesher import mesh_sector, replicate_sector
from .pmsm import IPMMachine

__all__ = ["CASES", "run", "YANG_2013_PHI0_DEG", "YANG_2013_TORQUE_NM", "prius_torque_curve"]

CASES: dict[str, Callable[[bool], CaseResult]] = {}

YANG_2013_PHI0_DEG = np.linspace(60.0, 180.0, 13)  # current angle from the d-axis (pyleecan Phi0), 250 A peak
YANG_2013_TORQUE_NM = np.array([79, 125, 160, 192, 237, 281, 319, 343, 353, 332, 266, 164, 22], float)

_CACHE: dict = {}


def case(name: str):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


def run(names: Sequence[str] | None = None, quick: bool = False) -> list[CaseResult]:
    out = []
    for n in names or list(CASES):
        t0 = time.perf_counter()
        r = CASES[n](quick)
        r.seconds = time.perf_counter() - t0
        out.append(r)
    return out


def _machine(n_gap: int) -> IPMMachine:
    if n_gap not in _CACHE:
        _CACHE[n_gap] = IPMMachine(prius_2004(), n_gap=n_gap)
    return _CACHE[n_gap]


def prius_torque_curve(n_gap: int = 960, I_peak: float = 250.0, phi0_deg=YANG_2013_PHI0_DEG,
                       n_positions: int = 12) -> dict:
    """Mean Maxwell-stress and dq torque of the Prius model against the current angle (pyleecan ``Phi0``)."""
    key = ("curve", n_gap, I_peak, tuple(np.round(phi0_deg, 6)), n_positions)
    if key not in _CACHE:
        m = _machine(n_gap)
        T, Tdq = [], []
        for p in phi0_deg:
            op = m.operating_point(I_peak=I_peak, gamma=math.radians(p - 90.0), n_positions=n_positions)
            T.append(op["torque_mean"])
            Tdq.append(op["torque_dq"])
        _CACHE[key] = {"phi0_deg": np.asarray(phi0_deg, float), "torque": np.array(T), "torque_dq": np.array(Tdq)}
    return _CACHE[key]


@case("Magnetised ring, sector mesher + P1 FEM")
def _ring(quick: bool) -> CaseResult:
    r0, a, b, R, Br, N = 0.01, 0.02, 0.03, 0.05, 1.0, 8
    span = 2 * math.pi / N
    half = span / 2
    errs = []
    for h in ((4e-3, 2e-3, 1e-3) if quick else (4e-3, 2e-3, 1e-3, 5e-4)):
        ring = np.vstack([[[a * math.cos(-half), a * math.sin(-half)], [b * math.cos(-half), b * math.sin(-half)]],
                          arc(0j, b, -half, half, h)[1:-1],
                          [[b * math.cos(half), b * math.sin(half)], [a * math.cos(half), a * math.sin(half)]],
                          arc(0j, a, half, -half, h)[1:-1]])
        geo = SectorGeometry(r0, R, span, "air", [Polygon("magnet", ring, 0.0)])
        sm = mesh_sector(geo, h, max(1, round(span * r0 / h)), round(span * R / h))
        mesh = replicate_sector(sm, N, span)
        mag = mesh.region == 1
        Brv = np.zeros((mesh.n_elements, 2))
        Brv[mag, 0] = Br
        sol = solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), Br=Brv)
        g = Geometry.of(mesh)
        e2 = n2 = 0.0
        T, P = mesh.tris, mesh.points
        for i, j in ((0, 1), (1, 2), (2, 0)):  # edge-midpoint rule, exact for quadratics
            q = 0.5 * (P[T[:, i]] + P[T[:, j]])
            Ae = magnetized_ring(q[:, 0], q[:, 1], Br, a, b, r0, R)
            e2 += np.sum(g.area * (0.5 * (sol.A[T[:, i]] + sol.A[T[:, j]]) - Ae) ** 2) / 3
            n2 += np.sum(g.area * Ae ** 2) / 3
        errs.append(math.sqrt(e2 / n2))
    return CaseResult("Magnetised ring, sector mesher + P1 FEM", "relative L2 error of A_z, h halved",
                      "closed form (two magnetised cylinders + harmonic)", errs[-1], 2e-3,
                      observed_orders(errs), 2.0, "8 sector copies, magnet edges on mesh edges")


@case("Machine sector geometry (Prius slot and magnet areas)")
def _areas(quick: bool) -> CaseResult:
    from .mesher import MachineMesh

    g = prius_2004()
    exact_slot = g.stator.slot_area()
    exact_mag = g.rotor.W4 * g.rotor.H3
    errs, mag_err = [], 0.0
    for n in ((288, 576, 1152) if quick else (288, 576, 1152, 2304)):
        mm = MachineMesh(g, n_gap=n, smooth=1)
        mesh = mm.mesh(0.0)
        area = Geometry.of(mesh).area
        s0 = area[mesh.mask("slot0")].sum()
        errs.append(abs(s0 - exact_slot) / exact_slot)
        m0 = area[mesh.mask("magnet0")].sum() / 2
        mag_err = max(mag_err, abs(m0 - exact_mag) / exact_mag)
    return CaseResult("Machine sector geometry (Prius slot and magnet areas)", "slot area, gap nodes doubled",
                      "exact area (trapezoid + half disc)", errs[-1], 1e-3, observed_orders(errs), 2.0,
                      f"magnet area (straight edges) exact to {mag_err:.1e}")


@case("Prius IPM: Maxwell stress vs co-energy torque")
def _coenergy(quick: bool) -> CaseResult:
    m = _machine(480 if quick else 960)
    i_peak, gm = 250.0, math.radians(50.0)
    devs = []
    for th in m.positions(3, period=m.load_period):
        cur = m.phase_currents(th, i_peak, gm)
        Ta, Tc = m.torque(m.solve(th, cur)), m.torque_coenergy(th, cur)
        devs.append(abs(Ta - Tc) / abs(Tc))
    return CaseResult("Prius IPM: Maxwell stress vs co-energy torque", "Arkkio torque at 250 A, 3 rotor angles",
                      "virtual work dW'/dtheta (4th-order differences)", max(devs), 0.02, detail=(
                          "max relative difference over the angles"))


@case("Prius IPM: dq power balance and frozen permeability")
def _dq(quick: bool) -> CaseResult:
    m = _machine(480 if quick else 960)
    op = m.operating_point(I_peak=250.0, gamma=math.radians(50.0), n_positions=12)
    dev = abs(op["torque_dq"] - op["torque_mean"]) / op["torque_mean"]
    ind = m.inductances(op["i_d"], op["i_q"])
    return CaseResult("Prius IPM: dq power balance and frozen permeability",
                      "mean Maxwell-stress torque over 60 el. deg", "1.5 p (psi_d i_q - psi_q i_d)", dev, 0.015,
                      detail=f"frozen-permeability superposition error {ind['superposition_error']:.1e}; "
                             f"L_d = {1e3 * ind['L_dd']:.3f} mH, L_q = {1e3 * ind['L_qq']:.3f} mH, "
                             f"psi_pm = {ind['psi_pm_d']:.4f} Wb at this point")


@case("Prius IPM: torque vs current angle (Yang et al. 2013)")
def _yang(quick: bool) -> CaseResult:
    n = 480 if quick else 960
    c = prius_torque_curve(n, n_positions=12)
    T = c["torque"]
    k = int(np.argmax(T))
    rms = float(np.sqrt(np.mean((T - YANG_2013_TORQUE_NM) ** 2)) / YANG_2013_TORQUE_NM.max())
    peak = float((T.max() - YANG_2013_TORQUE_NM.max()) / YANG_2013_TORQUE_NM.max())
    return CaseResult("Prius IPM: torque vs current angle (Yang et al. 2013)",
                      "mean torque, 250 A, Phi0 = 60..180 deg", "published FEA (IEMDC 2013), 13 angles",
                      abs(peak), 0.05, detail=(
                          f"peak {T.max():.1f} N m at Phi0 = {c['phi0_deg'][k]:.0f} deg (published 353 N m at 140 deg); "
                          f"rms deviation {100 * rms:.1f} % of the peak"))


@case("Prius IPM: mesh convergence of the peak torque")
def _mesh(quick: bool) -> CaseResult:
    ns = (288, 576, 1152) if quick else (480, 960, 1920)
    T = [_machine(n).operating_point(I_peak=250.0, gamma=math.radians(50.0), n_positions=12)["torque_mean"]
         for n in ns]
    d1, d2 = abs(T[1] - T[0]), abs(T[2] - T[1])
    return CaseResult("Prius IPM: mesh convergence of the peak torque", "mean torque at 250 A, 50 deg, gap nodes doubled",
                      "successive differences", d2 / abs(T[2]), 0.02, detail=(
                          ", ".join(f"{n}: {t:.2f} N m" for n, t in zip(ns, T, strict=True))
                          + f"; difference ratio {d1 / max(d2, 1e-30):.2f}"))
