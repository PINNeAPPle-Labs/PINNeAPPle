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

__all__ = ["CASES", "run", "YANG_2013_PHI0_DEG", "YANG_2013_TORQUE_NM", "prius_torque_curve", "yang_comparison"]

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


def yang_comparison(curve: dict) -> dict:
    """Per-point errors of a torque-angle curve against Yang et al. 2013, and the angle offset ``delta`` (deg) that
    minimises the rms error of ``T(Phi0 + delta)`` (cubic interpolation of the curve)."""
    from scipy.interpolate import CubicSpline
    from scipy.optimize import minimize_scalar

    phi, T = curve["phi0_deg"], curve["torque"]
    ref = YANG_2013_TORQUE_NM
    err = T - ref
    sp = CubicSpline(phi, T)
    inner = (phi >= phi[0] + 5) & (phi <= phi[-1] - 5)  # keep the shifted samples inside the computed range
    res = minimize_scalar(lambda d: float(np.sqrt(np.mean((sp(phi[inner] + d) - ref[inner]) ** 2))),
                          bounds=(-5.0, 5.0), method="bounded")
    k = int(np.argmax(np.abs(err)))
    return {"phi0_deg": phi.tolist(), "torque": T.tolist(), "reference": ref.tolist(), "error_Nm": err.tolist(),
            "relative_error": (err / ref).tolist(), "max_abs_error_Nm": float(np.abs(err).max()),
            "max_abs_error_at_deg": float(phi[k]), "rms_error_Nm": float(np.sqrt(np.mean(err ** 2))),
            "best_offset_deg": float(res.x), "rms_at_best_offset_Nm": float(res.fun),
            "rms_no_offset_inner_Nm": float(np.sqrt(np.mean(err[inner] ** 2)))}


@case("Prius IPM: torque vs current angle (Yang et al. 2013)")
def _yang(quick: bool) -> CaseResult:
    n = 480 if quick else 960
    c = prius_torque_curve(n, n_positions=12)
    T = c["torque"]
    k = int(np.argmax(T))
    y = yang_comparison(c)
    peak = float((T.max() - YANG_2013_TORQUE_NM.max()) / YANG_2013_TORQUE_NM.max())
    return CaseResult("Prius IPM: torque vs current angle (Yang et al. 2013)",
                      "mean torque, 250 A, Phi0 = 60..180 deg", "published FEA (IEMDC 2013), 13 angles",
                      abs(peak), 0.05, detail=(
                          f"peak {T.max():.1f} N m at Phi0 = {c['phi0_deg'][k]:.0f} deg (published 353 N m at 140 deg); "
                          f"largest deviation {y['max_abs_error_Nm']:+.1f} N m at {y['max_abs_error_at_deg']:.0f} deg; "
                          f"rms {y['rms_error_Nm']:.1f} N m; an angle offset of {y['best_offset_deg']:+.2f} deg fits best "
                          f"(rms {y['rms_no_offset_inner_Nm']:.1f} -> {y['rms_at_best_offset_Nm']:.1f} N m over 70..170 deg)"))


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


# ------------------------------------------------------------------------------------------------ skew
@case("Step skew: multi-slice cogging vs skew factor")
def _skew(quick: bool) -> CaseResult:
    from .robustness import skewed_no_load

    m = _machine(576)
    lam, N = m.cogging_period, 12
    th = m.positions(N, period=lam)
    c0 = np.array([m.torque(m.solve(t)) for t in th])
    F0 = np.fft.rfft(c0) / N
    worst = 0.0
    for K, steps in ((2, 6), (3, 4)):
        s = steps * m.mm.band_step
        Fs = np.fft.rfft(skewed_no_load(m, th, K, s)["cogging"]) / N
        x = np.arange(len(F0)) * math.pi * s / lam
        with np.errstate(all="ignore"):
            fac = np.where(np.abs(np.sin(x)) < 1e-12, np.cos(K * x) / np.cos(x), np.sin(K * x) / (K * np.sin(x)))
        worst = max(worst, float(np.abs(Fs - F0 * fac).max() / np.abs(F0[1:]).max()))
    return CaseResult("Step skew: multi-slice cogging vs skew factor", "cogging harmonics, 2 and 3 steps",
                      "sin(K h pi s / lam) / (K sin(h pi s / lam)) x straight rotor", worst, 1e-6,
                      detail="offsets are multiples of the band step (identical band triangles)")


# ------------------------------------------------------------------------------------------------ drive model
def _linear_drive(R: float = 0.0):
    from .drive import Drive, LinearDQ, v_max_svpwm

    m = LinearDQ(4, 0.2, 0.6e-3, 2.4e-3)  # psi_m > L_d I_max: no maximum-torque-per-volt region below the limit
    return m, Drive(m, R=R, I_max=250.0, V_max=v_max_svpwm(500.0))


@case("Drive: torque envelope vs closed form (linear dq)")
def _envelope(quick: bool) -> CaseResult:
    m, d = _linear_drive()
    errs = []
    for rpm in (500, 1500, 2500, 4000, 6000, 8000, 10000):
        w = rpm * 2 * math.pi / 60
        T = d.max_torque(w)["T"]
        Tc = m.max_torque_closed_form(m.pole_pairs * w, d.I_max, d.V_max)[0]
        Tg = -d.max_torque(w, generating=True)["T"]
        errs += [abs(T - Tc) / Tc, abs(Tg - Tc) / Tc]
    return CaseResult("Drive: torque envelope vs closed form (linear dq)", "max torque, 500..10000 rpm, both signs",
                      "MTPA / circle-ellipse intersection (R = 0)", max(errs), 1e-9,
                      detail="L_d = 0.6 mH, L_q = 2.4 mH, psi_m = 0.2 Wb, 250 A, 500 V SVPWM")


@case("Drive: least-loss currents vs MTPA closed form")
def _mtpa_drive(quick: bool) -> CaseResult:
    m, d = _linear_drive(R=0.05)
    errs = []
    for amp in (25.0, 100.0, 175.0, 250.0):
        i_d, i_q = m.mtpa_closed_form(amp)
        op = d.operating_point(float(m.torque(i_d, i_q)), 50.0, include_core_loss=False)
        errs.append(math.hypot(op["i_d"] - i_d, op["i_q"] - i_q) / amp)
    return CaseResult("Drive: least-loss currents vs MTPA closed form", "currents of least copper loss, 4 torques",
                      "i_d = (psi - sqrt(psi^2 + 8 (L_q - L_d)^2 I^2)) / (4 (L_q - L_d))", max(errs), 1e-6)


@case("Drive cycle: WLTC class 3b energy balance")
def _cycle(quick: bool) -> CaseResult:
    from .drive import Vehicle, cycle_energy, wltc_class3b

    t, v = wltc_class3b()
    m, d = _linear_drive(R=0.0)
    veh = Vehicle(mass=1450.0, c_rr=0.009, cd_a=0.58, wheel_radius=0.30, gear_ratio=4.113, gear_efficiency=1.0,
                  rotating_mass_factor=0.0)
    E = cycle_energy(d, veh, t, v, envelope_speeds=9 if quick else 25)
    vm = 0.5 * (v[1:] + v[:-1])
    road = np.sum((veh.mass * veh.g * veh.c_rr * (vm > 0) + 0.5 * veh.air_density * veh.cd_a * vm ** 2) * vm * np.diff(t))
    err = abs(E["battery_J"] - road) / road
    return CaseResult("Drive cycle: WLTC class 3b energy balance", "battery energy of a lossless drive",
                      "road-load work (kinetic energy returns to zero)", err, 1e-9, detail=(
                          f"trace: {E['distance_m']:.1f} m (GTR 15: 23 266 m), max {3.6 * v.max():.1f} km/h, "
                          f"sum {np.sum(3.6 * v):.1f} km/h s"))


@case("Prius flux map: interpolation vs direct FEA")
def _fluxmap(quick: bool) -> CaseResult:
    from .drive import FluxMap

    m = _machine(288 if quick else 480)
    npos = 6 if quick else 12  # the same rotor angles for the map and the direct solutions
    fm = FluxMap.from_fea(m, 250.0, n_d=7, n_q=7, n_positions=npos, loss_n=0)
    errs = []
    for i_d, i_q in ((-60.0, 90.0), (-150.0, 170.0), (-200.0, 40.0)):
        op = m.operating_point(i_d=i_d, i_q=i_q, n_positions=npos)
        errs.append(abs(float(fm.torque(i_d, i_q)) - op["torque_mean"]) / 300.0)
    return CaseResult("Prius flux map: interpolation vs direct FEA", "torque at 3 off-grid currents (7 x 7 grid)",
                      "FEA at the same currents", max(errs), 0.01, detail="error relative to 300 N m")


# ------------------------------------------------------------------------------------------------ thermal
@case("Thermal network: first-order transient")
def _rc(quick: bool) -> CaseResult:
    from .thermal import ThermalNetwork

    C, G, P, Tb = 500.0, 2.0, 100.0, 20.0
    net = ThermalNetwork.empty(["n"], [C]).to_boundary("n", G, Tb)
    tau = C / G
    exact = Tb + P / G * (1 - math.exp(-1.0))
    errs = []
    for dt in (tau / 10, tau / 20, tau / 40, tau / 80):
        _, T = net.transient([P], tau, dt, Tb, method="trapezoidal")
        errs.append(abs(T[-1, 0] - exact) / (P / G))
    eul = []
    for dt in (tau / 20, tau / 40):
        _, T = net.transient([P], tau, dt, Tb, method="euler")
        eul.append(abs(T[-1, 0] - exact))
    return CaseResult("Thermal network: first-order transient", "node temperature at t = tau, dt halved",
                      "T_b + P/G (1 - e^-1) (closed form)", errs[-1], 1e-4, observed_orders(errs), 2.0,
                      f"trapezoidal rule; backward Euler order {math.log2(eul[0] / eul[1]):.2f}")


@case("Thermal network: cylindrical conduction")
def _shell(quick: bool) -> CaseResult:
    from .thermal import cylinder_conductance

    k, r1, r2, L = 28.0, 0.08, 0.13, 0.1
    G = cylinder_conductance(k, r1, r2, L)
    errs = []
    for n in (8, 16, 32, 64):
        r = np.linspace(r1, r2, n + 1)
        rm = 0.5 * (r[1:] + r[:-1])
        errs.append(abs(1.0 / np.sum(np.diff(r) / (2 * math.pi * rm * L * k)) - G) / G)
    return CaseResult("Thermal network: cylindrical conduction", "series of thin flat shells, n doubled",
                      "2 pi k L / ln(r2/r1)", errs[-1], 1e-4, observed_orders(errs), 2.0)


@case("Demagnetisation field in a magnetised ring")
def _demag(quick: bool) -> CaseResult:
    from .thermal import reverse_field

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
        mesh = replicate_sector(mesh_sector(geo, h, max(1, round(span * r0 / h)), round(span * R / h)), N, span)
        mag = mesh.region == 1
        Brv = np.zeros((mesh.n_elements, 2))
        Brv[mag, 0] = Br
        sol = solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), Br=Brv)
        H_fem = reverse_field(sol.B[mag], np.zeros(mag.sum()), Br, 1.0)
        c = mesh.centroids()[mag]
        d = 1e-7  # central differences of the closed-form A at the centroids
        dAdx = (magnetized_ring(c[:, 0] + d, c[:, 1], Br, a, b, r0, R) - magnetized_ring(c[:, 0] - d, c[:, 1], Br, a, b,
                                                                                            r0, R)) / (2 * d)
        dAdy = (magnetized_ring(c[:, 0], c[:, 1] + d, Br, a, b, r0, R) - magnetized_ring(c[:, 0], c[:, 1] - d, Br, a, b,
                                                                                            r0, R)) / (2 * d)
        H_ex = reverse_field(np.column_stack([dAdy, -dAdx]), np.zeros(mag.sum()), Br, 1.0)
        area = Geometry.of(mesh).area[mag]
        errs.append(math.sqrt(np.sum(area * (H_fem - H_ex) ** 2) / np.sum(area * H_ex ** 2)))
    return CaseResult("Demagnetisation field in a magnetised ring", "reverse field in the magnet, h halved",
                      "closed form of the magnetised ring", errs[-1], 0.05, observed_orders(errs), 1.0,
                      "element-wise B of P1 elements: first order")
