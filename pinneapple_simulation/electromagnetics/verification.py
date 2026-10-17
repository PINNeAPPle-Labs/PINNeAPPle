"""Verification suite of the electromagnetics package: each case runs a solver or a reference implementation
against an independent result (closed form, a published value, another library, a limit) and reports the
measured error and, where a mesh or step is refined, the measured order of convergence.

``run()`` returns :class:`CaseResult` objects; ``to_markdown()`` renders the table of
docs/guides/electromagnetics.md, which is produced by ``examples/electrodynamics/07_em_verification.py``, so every
number in that table comes from a run. ``tests/test_em_verification.py`` asserts each case's tolerance.

Cases are registered with :func:`case`; later modules (FDTD, edge elements, particles, ...) add theirs.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 has only trapz

__all__ = ["CaseResult", "CASES", "case", "run", "to_markdown", "observed_orders", "write_markdown_block"]


@dataclass
class CaseResult:
    name: str
    what: str                     # quantity compared
    reference: str                # against what
    error: float                  # final (finest) relative error, or the measured deviation
    tolerance: float
    orders: list[float] = field(default_factory=list)   # observed orders between successive refinements
    expected_order: float | None = None
    detail: str = ""
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        ok = bool(np.isfinite(self.error) and self.error <= self.tolerance)
        if self.expected_order is not None and self.orders:
            ok = ok and abs(self.orders[-1] - self.expected_order) <= 0.15 * self.expected_order
        return ok

    def row(self) -> str:
        order = ", ".join(f"{o:.2f}" for o in self.orders) if self.orders else "-"
        exp = f" (expected {self.expected_order:g})" if self.expected_order is not None else ""
        cell = lambda t: t.replace("|", "\\|")  # noqa: E731  (|B| inside a Markdown table)
        return (f"| {cell(self.name)} | {cell(self.what)} | {cell(self.reference)} | {self.error:.2e} | "
                f"{order}{exp} | {'pass' if self.passed else 'FAIL'} |")


CASES: dict[str, Callable[[bool], CaseResult]] = {}


def case(name: str):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


def observed_orders(errors: Sequence[float], ratio: float = 2.0) -> list[float]:
    e = np.asarray(errors, float)
    return [float(v) for v in np.log(e[:-1] / e[1:]) / math.log(ratio)]


def run(names: Sequence[str] | None = None, quick: bool = False) -> list[CaseResult]:
    out = []
    for n in names or list(CASES):
        t0 = time.perf_counter()
        r = CASES[n](quick)
        r.seconds = time.perf_counter() - t0
        out.append(r)
    return out


def to_markdown(results: Sequence[CaseResult]) -> str:
    head = ("| Case | Quantity | Reference | Error (finest) | Observed order | Status |\n"
            "|---|---|---|---|---|---|")
    return "\n".join([head] + [r.row() for r in results])


def write_markdown_block(path, results: Sequence[CaseResult], marker: str = "verification-table") -> None:
    """Replace the text between ``<!-- marker:start -->`` and ``<!-- marker:end -->`` in a Markdown file with the
    table of ``results`` and their notes, so a page shows exactly what the last run measured."""
    from pathlib import Path

    p = Path(path)
    text = p.read_text()
    start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    i, j = text.index(start) + len(start), text.index(end)
    notes = "\n".join(f"- {r.name}: {r.detail}" for r in results if r.detail)
    block = "\n" + to_markdown(results) + "\n\nNotes from the same run:\n\n" + notes + "\n"
    p.write_text(text[:i] + block + text[j:])


# ============================================================================================ statics
@case("Coaxial capacitor (P1 FEM)")
def _coax(quick: bool) -> CaseResult:
    from .electrostatics import capacitance, solve_electrostatic
    from .mesh import Ring, polar_mesh
    from .statics import coax_capacitance

    a, b, eps_r = 1e-3, 3.5e-3, 2.1
    errs = []
    for n in ((16, 32, 64) if quick else (16, 32, 64, 128)):
        rings = [Ring(a + (b - a) * k / (n // 4), n) for k in range(n // 4 + 1)]
        m = polar_mesh(rings, lambda ly, r, p: "dielectric")
        s = solve_electrostatic(m, eps_r, {"inner": (m.rings[0], 1.0), "outer": (m.rings[-1], 0.0)})
        errs.append(abs(capacitance(s, 1.0) / coax_capacitance(a, b, eps_r) - 1))
    return CaseResult("Coaxial capacitor (P1 FEM)", "capacitance per metre", "2 pi eps / ln(b/a)", errs[-1],
                      2e-3, observed_orders(errs), 2.0, f"errors {', '.join(f'{e:.2e}' for e in errs)}")


@case("Two-dielectric parallel-plate capacitor (P1 FEM)")
def _plates(quick: bool) -> CaseResult:
    from pinneapple_physics.closed_form.electromagnetics import two_dielectric_capacitor

    from .electrostatics import solve_electrostatic
    from .mesh import rect_mesh

    d = 1e-3
    m = rect_mesh(np.linspace(0, 0.01, 7), np.linspace(0, d, 9), lambda x, y: "a" if y < d / 2 else "b")
    y = m.points[:, 1]
    s = solve_electrostatic(m, np.where(m.mask("a"), 1.0, 4.0), {"g": (np.flatnonzero(y == 0), 0.0),
                                                                 "p": (np.flatnonzero(y == d), 1.0)})
    err = float(np.max(np.abs(s.phi - two_dielectric_capacitor(y, 1.0, d, d / 2, 1.0, 4.0))))
    return CaseResult("Two-dielectric parallel-plate capacitor (P1 FEM)", "potential (max abs, V)",
                      "D uniform, E_i = D / eps_i (exact in P1)", err, 1e-12)


@case("Circular loop by Biot-Savart (polygon)")
def _loop(quick: bool) -> CaseResult:
    from .statics import biot_savart, circle_segments, loop_field

    pts = np.array([[0.03, 0.01, 0.04], [0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [0.12, -0.05, -0.02]])
    exact = loop_field(pts, 0.1, 2.0)
    errs = []
    for n in (45, 90, 180, 360):
        s, e = circle_segments(0.1, n)
        errs.append(float(np.max(np.abs(biot_savart(pts, s, e, 2.0) - exact)) / np.max(np.abs(exact))))
    return CaseResult("Circular loop by Biot-Savart (polygon)", "B off the axis",
                      "elliptic-integral loop field (Jackson 5.37)", errs[-1], 1e-4, observed_orders(errs), 2.0)


@case("Infinite wire and finite solenoid (Biot-Savart)")
def _solenoid(quick: bool) -> CaseResult:
    from .statics import biot_savart, finite_solenoid_on_axis_B, helix_segments, infinite_wire_B

    p = np.array([[0.02, 0.01, 0.0], [-0.05, 0.03, 0.1]])
    Bw = infinite_wire_B(p, 3.0)
    e_wire = float(np.max(np.abs(biot_savart(p, [[0, 0, -1e4]], [[0, 0, 1e4]], 3.0) - Bw)) / np.max(np.abs(Bw)))
    z = np.array([0.0, 0.03, 0.05, 0.08])
    sheet = finite_solenoid_on_axis_B(z, 0.02, 0.1, 200, 1.5)
    s, e = helix_segments(0.02, 0.1, 200, n_per_turn=96)
    B = biot_savart(np.column_stack([0 * z, 0 * z, z]), s, e, 1.5)
    err = float(np.max(np.abs(B[:, 2] / sheet - 1)))
    return CaseResult("Infinite wire and finite solenoid (Biot-Savart)", "B of a 200-turn helix on the axis",
                      "current-sheet solenoid (closed form)", err, 5e-3,
                      detail=f"20 km segment vs mu0 I / (2 pi s): {e_wire:.1e}")


@case("Coaxial mutual inductance")
def _mutual(quick: bool) -> CaseResult:
    from scipy.integrate import quad

    from .constants import MU0
    from .statics import mutual_inductance_coaxial_loops

    r1, r2, d = 0.1, 0.07, 0.05
    M = MU0 / 2 * quad(lambda t: r1 * r2 * math.cos(t) / math.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * math.cos(t)
                                                                   + d * d), 0, 2 * math.pi, epsabs=1e-14)[0]
    err = abs(mutual_inductance_coaxial_loops(r1, r2, d) / M - 1)
    return CaseResult("Coaxial mutual inductance", "M (elliptic integrals)", "Neumann double integral", err, 1e-9)


@case("Axisymmetric coil (FDM, eddy_current_fdm at DC)")
def _coil(quick: bool) -> CaseResult:
    from pinneapple_simulation.numerical_solvers.eddy_current_fdm import (
        MU0,
        annular_current_source,
        axial_flux_density,
        solve_axisymmetric_eddy_current,
    )

    errs = []
    for n in ((121, 241) if quick else (121, 241, 481)):
        r = np.linspace(0, 0.6, n)
        z = np.linspace(-0.6, 0.6, 2 * n - 1)
        J = annular_current_source(r, z, 0.03, 0.0, 0.004, 0.002, 10.0, 0.5)
        dr, dz = r[1] - r[0], z[1] - z[0]
        RR = np.ones((n, len(z)))
        A = solve_axisymmetric_eddy_current(r, z, 0.0, MU0 * RR, 0 * RR, J)
        Bz = axial_flux_density(A, r, z).real
        k = int(np.argmin(np.abs(z - 0.05)))
        ii, jj = np.nonzero(J)
        ref = np.sum(J[ii, jj] * dr * dz * MU0 * r[ii] ** 2 / (2 * (r[ii] ** 2 + (z[jj] - z[k]) ** 2) ** 1.5))
        errs.append(abs(Bz[0, k] / ref - 1))
    return CaseResult("Axisymmetric coil (FDM, eddy_current_fdm at DC)", "on-axis B_z",
                      "sum of current-loop fields (closed form)", errs[-1], 2e-3,
                      detail=f"errors {', '.join(f'{e:.2e}' for e in errs)} (finite box: A = 0 at 20 coil radii)")


# ============================================================================================ waves
@case("Mie sphere vs Bohren & Huffman / miepython")
def _mie(quick: bool) -> CaseResult:
    from .mie import mie

    r = mie(1.55, 2 * math.pi * 0.525 / 0.6328)
    pub = (3.10543, 3.10543, 2.92534)    # BHMIE sample output, Bohren & Huffman (1983) appendix A
    err = max(abs(r.qsca - pub[0]) / pub[0], abs(r.qext - pub[1]) / pub[1], abs(r.qback - pub[2]) / pub[2])
    detail = "published 5-digit values"
    try:
        import miepython

        e2 = 0.0
        for m, x in [(1.55, 5.213), (1.5 - 0.1j, 3.0), (1.33, 50.0), (4 - 3j, 1.2)]:
            qe, qs, qb, _ = miepython.efficiencies_mx(m, x)
            rr = mie(m, x)
            e2 = max(e2, abs(rr.qext / qe - 1), abs(rr.qsca / qs - 1), abs(rr.qback / qb - 1))
        detail += f"; miepython {miepython.__version__}, 4 spheres incl. absorbing: max rel. diff {e2:.1e}"
    except ImportError:
        detail += "; miepython not installed"
    return CaseResult("Mie sphere vs Bohren & Huffman / miepython", "Q_sca, Q_ext, Q_back",
                      "BHMIE sample (B&H app. A)", err, 2e-5, detail=detail)


@case("Mie small-sphere (Rayleigh) limit")
def _rayleigh(quick: bool) -> CaseResult:
    from .mie import mie, mie_pec, rayleigh_qsca

    xs = (0.1, 0.05, 0.025)
    errs = [abs(mie(1.5, x).qsca / rayleigh_qsca(1.5, x) - 1) for x in xs]
    pec = [abs(mie_pec(x).qsca / (10 / 3 * x ** 4) - 1) for x in xs]
    return CaseResult("Mie small-sphere (Rayleigh) limit", "Q_sca / Rayleigh - 1 vs x",
                      "8/3 x^4 |(m^2-1)/(m^2+2)|^2", errs[-1], 1e-4, observed_orders(errs), 2.0,
                      f"PEC sphere vs 10/3 x^4: orders {', '.join(f'{o:.2f}' for o in observed_orders(pec))}")


@case("Fresnel coefficients and Brewster angle")
def _fresnel(quick: bool) -> CaseResult:
    from .waves import brewster_angle, fresnel

    worst = 0.0
    for n1, n2 in ((1.0, 1.5), (1.0, 3.4), (1.5, 1.0)):
        th = np.linspace(0, 1.5, 61)
        rs, rp, ts, tp = fresnel(n1, n2, th)
        s = n1 * np.sin(th) / n2
        ok = s < 1
        f = (n2 * np.sqrt(1 - s[ok] ** 2 + 0j)).real / (n1 * np.cos(th[ok]))
        worst = max(worst, float(np.max(np.abs(np.abs(rs[ok]) ** 2 + f * np.abs(ts[ok]) ** 2 - 1))),
                    float(np.max(np.abs(np.abs(rp[ok]) ** 2 + f * np.abs(tp[ok]) ** 2 - 1))))
    rb = abs(fresnel(1.0, 1.5, brewster_angle(1.0, 1.5))[1])
    return CaseResult("Fresnel coefficients and Brewster angle", "R + T - 1 (s and p, 3 interfaces)",
                      "energy conservation; r_p(atan n2/n1) = 0", max(worst, rb), 1e-12,
                      detail=f"|r_p| at Brewster {rb:.1e}")


@case("Bragg mirror (characteristic matrices)")
def _bragg(quick: bool) -> CaseResult:
    from .waves import multilayer

    lam, nH, nL, N, ns = 1e-6, 2.3, 1.38, 6, 1.52
    stack = [nH, nL] * N
    R = multilayer(stack, [lam / (4 * n) for n in stack], lam, n_out=ns).R
    Y = (nH / nL) ** (2 * N) * ns
    err = abs(R / ((1 - Y) / (1 + Y)) ** 2 - 1)
    return CaseResult("Bragg mirror (characteristic matrices)", "R at the design wavelength",
                      "((1 - Y)/(1 + Y))^2, Y = (nH/nL)^2N ns", err, 1e-12)


@case("Hertzian dipole radiated power")
def _hertz(quick: bool) -> CaseResult:
    from .constants import C0
    from .waves import hertzian_dipole_fields, hertzian_dipole_power

    omega = 2 * math.pi * 1e9
    k = omega / C0
    err = 0.0
    for r in (0.05 / k, 0.5 / k, 20 / k):
        th = np.linspace(0, math.pi, 2001)
        _, Eth, Hph = hertzian_dipole_fields(r, th, 1e-3, omega)
        P = _trapz(0.5 * np.real(Eth * np.conj(Hph)) * 2 * math.pi * r * r * np.sin(th), th)
        err = max(err, abs(P / hertzian_dipole_power(1e-3, omega) - 1))
    return CaseResult("Hertzian dipole radiated power", "flux of the complete fields at kr = 0.05, 0.5, 20",
                      "eta k^2 |I l|^2 / (12 pi)", err, 1e-6)


@case("Half-wave dipole radiation resistance")
def _halfwave(quick: bool) -> CaseResult:
    from .constants import ETA0
    from .waves import dipole_pattern, half_wave_dipole_radiation_resistance

    th = np.linspace(1e-9, math.pi - 1e-9, 200001)
    R = 2 * ETA0 / (4 * math.pi) * _trapz(dipole_pattern(th, 0.5) ** 2 * np.sin(th), th)
    Rc = half_wave_dipole_radiation_resistance()
    return CaseResult("Half-wave dipole radiation resistance", "R_rad from the pattern",
                      "eta Cin(2 pi) / (4 pi) = 73.08 ohm", abs(R / Rc - 1), 1e-6, detail=f"{Rc:.3f} ohm")


@case("Dispersive ADE recursions (Drude, Debye, Lorentz)")
def _ade(quick: bool) -> CaseResult:
    from .constants import EPS0
    from .media import Debye, Drude, Lorentz

    d = Drude(1e15, 1e14)
    errs = []
    for dt in (1e-16, 5e-17, 2.5e-17):
        ka, kb = d.ade(dt)
        z = np.exp(1j * 3e14 * dt)
        jw = 2 / dt * (1 - 1 / z) / (1 + 1 / z)
        errs.append(abs(kb * (1 + 1 / z) / (1 - ka / z) / (jw * EPS0) / d.chi(3e14) - 1))
    deb = Debye(70.0, 8e-12)
    e_deb = []
    for dt in (1e-12, 5e-13, 2.5e-13):
        ka, kb = deb.ade(dt)
        z = np.exp(1j * 2e10 * dt)
        e_deb.append(abs(kb * (1 + 1 / z) / (1 - ka / z) / EPS0 / deb.chi(2e10) - 1))
    lz = Lorentz(1.5, 4e15, 1e14)
    e_lz = []
    for dt in (2e-17, 1e-17, 5e-18):
        a1, a2, a3 = lz.ade(dt)
        z = np.exp(1j * 1e15 * dt)
        e_lz.append(abs(a3 / (1 - a1 / z - a2 / z ** 2) / z / EPS0 / lz.chi(1e15) - 1))
    det = (f"Debye orders {', '.join(f'{o:.2f}' for o in observed_orders(e_deb))}; Lorentz "
           f"{', '.join(f'{o:.2f}' for o in observed_orders(e_lz))}")
    return CaseResult("Dispersive ADE recursions (Drude, Debye, Lorentz)", "discrete chi(omega) vs dt (Drude)",
                      "continuous chi(omega)", errs[-1], 1e-4, observed_orders(errs), 2.0, det)


@case("Uniaxial crystal indices")
def _uniaxial(quick: bool) -> CaseResult:
    from .media import Anisotropic

    no, ne = 1.5, 1.7
    u = Anisotropic.uniaxial(no, ne)
    err = 0.0
    for th in np.linspace(0, math.pi / 2, 9):
        n = sorted(u.indices([math.sin(th), 0, math.cos(th)]).real)
        ext = 1 / math.sqrt(math.cos(th) ** 2 / no ** 2 + math.sin(th) ** 2 / ne ** 2)
        err = max(err, float(np.max(np.abs(np.array(n) - sorted([no, ext])))))
    return CaseResult("Uniaxial crystal indices", "n_o, n_e(theta) from the wave-normal eigenproblem",
                      "index ellipsoid", err, 1e-12)


# ============================================================================================ magnetics
@case("Round wire, A_z P1 FEM")
def _wire(quick: bool) -> CaseResult:
    from .analytic import wire_field
    from .fem import Geometry, solve_magnetostatic
    from .materials import NU0
    from .mesh import Ring, polar_mesh

    a, R, cur = 0.01, 0.05, 100.0
    errs = []
    for n in (32, 64, 128):
        radii = np.unique(np.concatenate([np.linspace(a / (n // 8), a, n // 8), np.linspace(a, R, n // 4)]))
        mesh = polar_mesh([Ring(0.0, 1)] + [Ring(r, n) for r in radii], lambda ly, r, p: "cu" if r < a else "air")
        g = Geometry.of(mesh)
        J = np.where(mesh.mask("cu"), 1.0, 0.0)
        J *= cur / np.sum(J * g.area)
        sol = solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), J=J)
        A_ex, _ = wire_field(np.hypot(*mesh.points.T), cur, a, R)
        errs.append(float(np.abs(sol.A - A_ex).max() / np.abs(A_ex).max()))
    return CaseResult("Round wire, A_z P1 FEM", "A_z (max relative)", "Ampere's law (closed form)", errs[-1], 1e-3,
                      observed_orders(errs), 2.0)


@case("Skin effect in a round wire (A_z eddy FEM)")
def _skin(quick: bool) -> CaseResult:
    from .analytic import skin_effect_wire
    from .constants import MU0
    from .eddy import solve_time_harmonic
    from .materials import NU0
    from .mesh import Ring, polar_mesh

    a, R, sigma = 0.005, 0.02, 5.8e7
    delta = a / 3
    omega = 2 / (MU0 * sigma * delta ** 2)
    n = 128
    radii = np.concatenate([np.linspace(a / (n // 4), a, n // 4), np.geomspace(a, R, n // 8)[1:]])
    mesh = polar_mesh([Ring(0.0, 1)] + [Ring(r, n) for r in radii], lambda ly, r, p: "cu" if r < a else "air")
    m = mesh.mask("cu")
    sol = solve_time_harmonic(mesh, np.full(mesh.n_elements, NU0), np.full(mesh.n_elements, sigma), omega, [m], [1.0])
    _, Z = skin_effect_wire(0.0, 1.0, a, sigma, omega)
    Zt = Z + 1j * omega * MU0 / (2 * math.pi) * math.log(R / a)
    err = max(abs(sol.u[0].real / Zt.real - 1), abs(sol.u[0].imag / Zt.imag - 1))
    return CaseResult("Skin effect in a round wire (A_z eddy FEM)", "impedance R + j X at a / delta = 3",
                      "Bessel solution (closed form)", err, 3e-3)


@case("Froehlich-Kennelly iron ring (Newton)")
def _fk(quick: bool) -> CaseResult:
    from .fem import Geometry, solve_magnetostatic
    from .hysteresis import FrohlichKennelly
    from .materials import NU0
    from .mesh import Ring, polar_mesh

    fk = FrohlichKennelly.from_saturation(3000.0, 1.9)
    a, r1, r2, R, n = 0.005, 0.01, 0.02, 0.03, 64
    radii = np.unique(np.concatenate([np.linspace(a / 4, a, 4), np.linspace(a, R, n // 2)[1:], [r1, r2]]))
    mesh = polar_mesh([Ring(0.0, 1)] + [Ring(r, n) for r in radii],
                      lambda ly, r, p: "cu" if r < a else ("fe" if r1 < r < r2 else "air"))
    g = Geometry.of(mesh)
    J = np.where(mesh.mask("cu"), 1.0, 0.0)
    J *= 3000.0 / np.sum(J * g.area)
    sol = solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), J=J, nonlinear=[(mesh.mask("fe"), fk)])
    c = mesh.centroids()[mesh.mask("fe")]
    H = 3000.0 / (2 * math.pi * np.hypot(c[:, 0], c[:, 1]))
    err = float(np.max(np.abs(np.hypot(*sol.B[mesh.mask("fe")].T) / fk.b(H) - 1)))
    return CaseResult("Froehlich-Kennelly iron ring (Newton)", "|B| in saturated iron", "B(H) at H = I / (2 pi r)",
                      err, 0.03, detail=f"Newton: {sol.iterations} iterations to residual {sol.residual:.1e}")


@case("Jiles-Atherton integrator")
def _ja(quick: bool) -> CaseResult:
    from .hysteresis import JilesAtherton

    ja = JilesAtherton(1.6e6, 1100.0, 400.0, 0.2, 1.6e-3)
    H = np.concatenate([np.linspace(0, 5e3, 101), np.linspace(5e3, -5e3, 201)[1:], np.linspace(-5e3, 5e3, 201)[1:]])
    ref = ja.trace(H, substeps=256).M
    errs = [float(np.max(np.abs(ja.trace(H, substeps=s).M - ref)) / ja.Ms) for s in (2, 4, 8, 16, 32)]
    rev = JilesAtherton(1.6e6, 1100.0, 400.0, 1.0, 1.6e-3)
    lr = rev.trace(H)
    dev = float(np.max(np.abs(lr.M - rev.anhysteretic(lr.H + rev.alpha * lr.M))) / rev.Ms)
    return CaseResult("Jiles-Atherton integrator", "M along a major loop vs RK4 sub-steps",
                      "256 sub-steps; c = 1 stays anhysteretic", errs[-1], 1e-8, observed_orders(errs),
                      detail=f"c = 1 deviation from the anhysteretic curve {dev:.1e} Ms (orders irregular: the "
                             "pinning switch makes dM/dH non-smooth)")


# ============================================================================================ checks
@case("Maxwell residual checks on an exact cavity mode")
def _checks(quick: bool) -> CaseResult:
    from pinneapple_physics import maxwell_checks as mc

    C = 299_792_458.0
    kx, ky = math.pi, 2 * math.pi
    w = C * math.hypot(kx, ky)
    vals, bad = [], None
    for n in (21, 41, 81):
        x = y = np.linspace(0, 1, n)
        t = np.linspace(0, 0.7 * 2 * math.pi / w, n)
        X, Y, T = np.meshgrid(x, y, t, indexing="ij")

        def fields(wf, X=X, Y=Y, T=T):
            return {"Ez": np.sin(kx * X) * np.sin(ky * Y) * np.cos(wf * T),
                    "Bx": -ky * np.sin(kx * X) * np.cos(ky * Y) * np.sin(wf * T) / wf,
                    "By": kx * np.cos(kx * X) * np.sin(ky * Y) * np.sin(wf * T) / wf}
        ch = {c.key: c for c in mc.maxwell_checks({"x": x, "y": y, "t": t}, fields(w))}
        vals.append(ch["ampere_maxwell"].value)
        if n == 41:
            bad = {c.key: c for c in mc.maxwell_checks({"x": x, "y": y, "t": t}, fields(1.2 * w))}
    return CaseResult("Maxwell residual checks on an exact cavity mode", "Ampere-Maxwell residual vs grid",
                      "exact TM12 mode (residual -> 0)", vals[-1], 1e-3, observed_orders(vals), 2.0,
                      f"a 20 % wrong frequency gives {bad['ampere_maxwell'].value:.3f} (rejected) and Poynting "
                      f"imbalance {bad['poynting_balance'].value:.3f}")


# ============================================================================================ FDTD
def _fdtd_reflection_1d(ncl: int, medium, f0: float, bw: float, periods: float, freqs) -> np.ndarray:
    """Reflection coefficient of a half-space from two 1-D runs (with and without it), DFT one wavelength in
    front of the interface."""
    from .constants import C0
    from .fdtd import FDTD, YeeGrid, modulated_gaussian

    lam = C0 / f0
    dx = lam / ncl
    N = int(5 * lam / dx) + 40
    g = YeeGrid((N, 1, 1), (dx, dx, dx))
    im = N // 2 - int(lam / dx)
    out = []
    for with_medium in (True, False):
        s = FDTD(g, {"x": "cpml", "y": "periodic", "z": "periodic"}, npml=20)
        if with_medium:
            m = np.zeros(g.shape, bool)
            m[N // 2:] = True
            s.set_material(m, medium)
        s.add_source("Ey", (30, 0, 0), modulated_gaussian(f0, bw * f0), "soft")
        mon = s.add_dft(["Ey"], freqs, (slice(im, im + 1), slice(None), slice(None)))
        s.run_until(periods / f0)
        out.append(mon.phasor("Ey").ravel())
    return (out[0] - out[1]) / out[1]


@case("FDTD 1-D dielectric half-space (Fresnel)")
def _fdtd_fresnel(quick: bool) -> CaseResult:
    from .media import Medium
    from .waves import fresnel

    f0 = 1e9
    freqs = np.linspace(0.6, 1.4, 5) * f0
    errs = []
    for ncl in ((20, 40) if quick else (20, 40, 80)):
        r = _fdtd_reflection_1d(ncl, Medium(4.0), f0, 0.8, 12, freqs)
        errs.append(float(np.max(np.abs(np.abs(r) - abs(fresnel(1.0, 2.0, 0.0)[0])))))
    return CaseResult("FDTD 1-D dielectric half-space (Fresnel)", "|r(f)|, 0.6-1.4 f0",
                      "(n1 - n2)/(n1 + n2)", errs[-1], 2e-2, observed_orders(errs), 2.0,
                      f"errors {', '.join(f'{e:.2e}' for e in errs)} at 20/40/80 cells per wavelength")


@case("FDTD dispersive half-spaces (Drude, Debye, Lorentz ADE)")
def _fdtd_dispersive(quick: bool) -> CaseResult:
    from .media import Debye, Lorentz, Medium, drude_metal
    from .waves import fresnel

    f0 = 1e15
    w0 = 2 * math.pi * f0
    media = {"Drude": drude_metal(1.5 * w0, 0.1 * w0),
             "Debye": Medium(2.0, poles=(Debye(3.0, 1 / w0),)),
             "Lorentz": Medium(1.5, poles=(Lorentz(2.0, 1.3 * w0, 0.1 * w0),))}
    freqs = np.linspace(0.6, 1.4, 5) * f0
    ladder = (40, 80) if quick else (40, 80, 160)
    worst, orders, det = 0.0, {}, []
    for name, med in media.items():
        rex = fresnel(1.0, med.n(2 * math.pi * freqs), 0.0)[0]
        errs = [float(np.max(np.abs(np.abs(_fdtd_reflection_1d(n, med, f0, 0.8, 40, freqs)) - np.abs(rex))))
                for n in ladder]
        orders[name] = observed_orders(errs)
        worst = max(worst, errs[-1])
        det.append(f"{name}: errors {', '.join(f'{e:.1e}' for e in errs)}, orders "
                   f"{', '.join(f'{o:.2f}' for o in orders[name])}")
    return CaseResult("FDTD dispersive half-spaces (Drude, Debye, Lorentz ADE)", "|r(f)| of each medium, 0.6-1.4 f0",
                      "Fresnel with eps(omega) of the pole", worst, 2e-2, orders["Drude"], 2.0, "; ".join(det))


@case("FDTD 2-D PEC cavity resonance (TM11)")
def _fdtd_cavity(quick: bool) -> CaseResult:
    from .constants import C0
    from .fdtd import FDTD, YeeGrid, gaussian_pulse

    a, b = 1.0, 0.7
    errs, disc = [], []
    for nx in ((21, 41) if quick else (21, 41, 81)):
        dx = a / (nx - 1)
        ny = int(round(b / dx)) + 1
        dy = b / (ny - 1)
        g = YeeGrid((nx, ny, 1), (dx, dy, dx))
        s = FDTD(g)
        s.add_source("Ez", (nx // 3, ny // 3, 0), gaussian_pulse(20 * g.dt, 6 * g.dt), "soft")
        p = s.add_probe("Ez", (nx // 5 + 1, 2 * ny // 3, 0))
        s.run(20000)
        fm = _spectral_peak(np.asarray(p.values), g.dt, C0 / 2 * math.hypot(1 / a, 1 / b))
        fex = C0 / 2 * math.hypot(1 / a, 1 / b)
        wd = 2 / g.dt * math.asin(C0 * g.dt * math.hypot(math.sin(math.pi / a * dx / 2) / dx,
                                                         math.sin(math.pi / b * dy / 2) / dy))
        errs.append(abs(fm / fex - 1))
        disc.append(abs(fm / (wd / (2 * math.pi)) - 1))
    return CaseResult("FDTD 2-D PEC cavity resonance (TM11)", "resonance frequency", "c/2 sqrt(1/a^2 + 1/b^2)",
                      errs[-1], 1e-3, observed_orders(errs), 2.0,
                      f"against the scheme's own dispersion relation: max {max(disc):.1e}")


def _spectral_peak(v: np.ndarray, dt: float, near: float) -> float:
    """Frequency of the spectral peak of ``v`` within 10 % of ``near`` (Hann window, zero padding, parabolic fit
    of the log spectrum)."""
    nfft = 8 * len(v)
    fr = np.fft.rfftfreq(nfft, dt)
    sp = np.abs(np.fft.rfft(v * np.hanning(len(v)), nfft))
    k = int(np.argmax(sp * (np.abs(fr - near) < 0.1 * near)))
    y0, y1, y2 = np.log(sp[k - 1:k + 2])
    return (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * (fr[1] - fr[0])


@case("FDTD 3-D PEC box resonance (TE101)")
def _fdtd_box(quick: bool) -> CaseResult:
    from .fdtd import FDTD, YeeGrid, gaussian_pulse
    from .waves import rect_cavity_resonance

    a, b, d = 1.0, 0.5, 0.75
    errs = []
    for n in ((9, 17) if quick else (9, 17, 33)):
        dx = a / (n - 1)
        ny, nz = int(round(b / dx)) + 1, int(round(d / dx)) + 1
        g = YeeGrid((n, ny, nz), (dx, b / (ny - 1), d / (nz - 1)))
        s = FDTD(g)
        s.add_source("Ey", (n // 3, ny // 2, nz // 3), gaussian_pulse(20 * g.dt, 6 * g.dt), "soft")
        p = s.add_probe("Ey", (2 * n // 3, ny // 2, 2 * nz // 3))
        s.run(12000)
        fex = rect_cavity_resonance(1, 0, 1, a, b, d)
        errs.append(abs(_spectral_peak(np.asarray(p.values), g.dt, fex) / fex - 1))
    return CaseResult("FDTD 3-D PEC box resonance (TE101)", "resonance frequency", "rectangular cavity formula",
                      errs[-1], 5e-3, observed_orders(errs), 2.0)


@case("FDTD rectangular waveguide TE10 propagation constant")
def _fdtd_guide(quick: bool) -> CaseResult:
    from .constants import C0
    from .fdtd import FDTD, YeeGrid, modulated_gaussian

    b = 0.02286
    fc = C0 / (2 * b)
    f = 1.5 * fc
    bex = 2 * math.pi * f / C0 * math.sqrt(1 - (fc / f) ** 2)
    errs, disc = [], []
    for ny in ((11, 21) if quick else (11, 21, 41)):
        dy = b / (ny - 1)
        dx = dy
        lamg = 2 * math.pi / bex
        N = int(6 * lamg / dx) + 40
        g = YeeGrid((N, ny, 1), (dx, dy, dx))
        s = FDTD(g, {"x": "cpml", "y": "pec"}, npml=20)
        y = np.arange(ny) * dy
        w = modulated_gaussian(f, 0.15 * f)
        for j in range(ny):
            s.add_current("Ez", (25, j, 0), w, math.sin(math.pi * y[j] / b))
        i1, i2 = N // 2 - int(lamg / dx), N // 2 + int(lamg / dx)
        m = s.add_dft(["Ez"], [f], (slice(None), slice(ny // 2, ny // 2 + 1), slice(None)))
        s.run_until(60 / f)
        ph = np.unwrap(np.angle(m.phasor("Ez")[0, i1:i2 + 1, 0, 0]))
        beta = -(ph[-1] - ph[0]) / ((i2 - i1) * dx)
        lhs = (math.sin(math.pi * f * g.dt) / (C0 * g.dt)) ** 2 - (math.sin(math.pi * dy / (2 * b)) / dy) ** 2
        bd = 2 / dx * math.asin(math.sqrt(lhs) * dx)
        errs.append(abs(beta / bex - 1))
        disc.append(abs(beta / bd - 1))
    return CaseResult("FDTD rectangular waveguide TE10 propagation constant", "beta at 1.5 f_c (2-D section)",
                      "k sqrt(1 - (fc/f)^2)", errs[-1], 3e-4, observed_orders(errs),
                      detail=f"errors {', '.join(f'{e:.1e}' for e in errs)}; against the scheme's own dispersion "
                             f"relation max {max(disc):.1e} (the phase-slope measurement floor)")


@case("FDTD bulk plasma (Drude) dispersion")
def _fdtd_plasma(quick: bool) -> CaseResult:
    from .constants import C0
    from .fdtd import FDTD, YeeGrid, modulated_gaussian
    from .media import drude_metal

    f0 = 1e15
    wp = 2 * math.pi * 0.6e15
    kex = math.sqrt((2 * math.pi * f0) ** 2 - wp ** 2) / C0
    errs = []
    for ncl in ((40, 80) if quick else (40, 80, 160)):
        lam = C0 / f0
        dx = lam / ncl
        N = int(8 * lam / dx)
        g = YeeGrid((N, 1, 1), (dx, dx, dx))
        s = FDTD(g, {"x": "cpml", "y": "periodic", "z": "periodic"}, npml=20)
        m = np.zeros(g.shape, bool)
        m[60:] = True
        s.set_material(m, drude_metal(wp, 0.0))
        s.add_source("Ey", (30, 0, 0), modulated_gaussian(f0, 0.3 * f0), "soft")
        mon = s.add_dft(["Ey"], [f0])
        s.run_until(80 / f0)
        i1, i2 = int(2 * lam / dx), int(5 * lam / dx)
        ph = np.unwrap(np.angle(mon.phasor("Ey")[0, i1:i2 + 1, 0, 0]))
        errs.append(abs(-(ph[-1] - ph[0]) / ((i2 - i1) * dx) / kex - 1))
    return CaseResult("FDTD bulk plasma (Drude) dispersion", "k at omega = 1.67 omega_p",
                      "omega^2 = omega_p^2 + c^2 k^2", errs[-1], 1e-3, observed_orders(errs),
                      detail=f"errors {', '.join(f'{e:.1e}' for e in errs)}")


@case("FDTD dielectric cylinder, TF/SF plane wave")
def _fdtd_cylinder(quick: bool) -> CaseResult:
    from .constants import C0
    from .fdtd import FDTD, PlaneWaveTFSF, YeeGrid, fill_fraction, modulated_gaussian
    from .media import Medium
    from .waves import cylinder_scattering_tm

    f0 = 1e9
    lam = C0 / f0
    k0 = 2 * math.pi / lam
    a, epsc = 0.3 * lam, 3.0
    errs = []
    for ncl in ((20, 30) if quick else (20, 30, 40, 60)):
        dx = lam / ncl
        N = int(3 * lam / dx) + 50
        g = YeeGrid((N, N, 1), (dx, dx, dx))
        s = FDTD(g, {"x": "cpml", "y": "cpml"}, npml=20)
        c = N // 2
        s.set_material(fill_fraction(g, lambda x, y, z, c=c, dx=dx: np.hypot(x - c * dx, y - c * dx) <= a, 16),
                       Medium(epsc))
        s.add_plane_wave(PlaneWaveTFSF("x", "Ez", modulated_gaussian(f0, 0.6 * f0), ((30, N - 30), (30, N - 30),
                                                                                       (0, 1))))
        inc = s.incident_monitor([f0])
        mon = s.add_dft(["Ez"], [f0])
        s.run_until(40 / f0)
        Ez = mon.phasor("Ez")[0, :, :, 0] / inc.phasor("Ey")[0, c, 0, 0]
        ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        ii = np.round(c + 0.5 * lam * np.cos(ang) / dx).astype(int)
        jj = np.round(c + 0.5 * lam * np.sin(ang) / dx).astype(int)
        ex = cylinder_scattering_tm((ii - c) * dx, (jj - c) * dx, k0, a, epsc)
        errs.append(float(np.max(np.abs(Ez[ii, jj] - ex)) / np.max(np.abs(ex))))
    return CaseResult("FDTD dielectric cylinder, TF/SF plane wave", "total Ez on a ring r = lambda/2",
                      "series of cylinder functions (Harrington 5-9)", errs[-1], 4e-2, observed_orders(errs, 1.5)
                      if len(errs) == 2 else [math.log(errs[i] / errs[i + 1]) / math.log(r) for i, r in
                                              enumerate((1.5, 4 / 3, 1.5))], 2.0,
                      f"eps = 3, a = 0.3 lambda, sub-cell (arithmetic) averaging of eps; errors "
                      f"{', '.join(f'{e:.1e}' for e in errs)} at 20/30/40/60 cells per wavelength")


@case("FDTD Hertzian dipole (3-D)")
def _fdtd_dipole(quick: bool) -> CaseResult:
    from .constants import C0
    from .fdtd import FDTD, YeeGrid, modulated_gaussian
    from .waves import hertzian_dipole_fields

    f0 = 1e9
    lam = C0 / f0
    errs = []
    ladder = (10, 15) if quick else (10, 15, 20)
    for ncl in ladder:
        dx = lam / ncl
        N = int(2.4 * lam / dx) + 20
        c = N // 2
        s = FDTD(YeeGrid((N, N, N), (dx, dx, dx)), {"x": "cpml", "y": "cpml", "z": "cpml"}, npml=10)
        w = modulated_gaussian(f0, 0.5 * f0)
        s.add_current("Ez", (c, c, c), w, 1.0)
        mon = s.add_dft(["Ez"], [f0], (slice(None), slice(c, c + 1), slice(None)))
        s.run_until(14 / f0)
        tj = (np.arange(s.n) + 0.5) * s.dt
        Il = np.sum(np.array([w(x) for x in tj]) * np.exp(-2j * np.pi * f0 * tj)) * s.dt * dx ** 3
        e = 0.0
        for th in np.radians([30, 60, 90, 120]):
            i = int(round(c + 0.8 * lam * math.sin(th) / dx))
            k = int(round(c + 0.8 * lam * math.cos(th) / dx))
            x, z = (i - c) * dx, (k - c) * dx
            rr, tt = math.hypot(x, z), math.atan2(x, z)
            Er, Et, _ = hertzian_dipole_fields(rr, tt, Il, 2 * np.pi * f0)
            Ez_ex = Er * math.cos(tt) - Et * math.sin(tt)
            e = max(e, abs(mon.phasor("Ez")[0, i, 0, k] - Ez_ex) / abs(Ez_ex))
        errs.append(float(e))
    return CaseResult("FDTD Hertzian dipole (3-D)", "phasor E_z at r = 0.8 lambda, theta = 30-120 deg",
                      "complete dipole fields (Balanis 4-8)", errs[-1], 5e-2, observed_orders(errs, 1.5)
                      if len(errs) == 2 else [math.log(errs[0] / errs[1]) / math.log(1.5),
                                              math.log(errs[1] / errs[2]) / math.log(4 / 3)], 2.0,
                      f"errors {', '.join(f'{e:.1e}' for e in errs)} at {', '.join(map(str, ladder))} cells per "
                      "wavelength")


@case("FDTD Bloch-periodic oblique incidence (Floquet)")
def _fdtd_bloch(quick: bool) -> CaseResult:
    from .constants import C0
    from .fdtd import FDTD, YeeGrid, modulated_gaussian
    from .media import Medium
    from .waves import fresnel

    f0 = 1e9
    lam = C0 / f0
    th0 = math.radians(25)
    ky = 2 * math.pi * f0 / C0 * math.sin(th0)
    errs = []
    for ncl in ((20, 40) if quick else (20, 40, 80)):
        dx = lam / ncl
        ny = 4
        N = int(4 * lam / dx) + 40
        g = YeeGrid((N, ny, 1), (dx, dx, dx))
        im = N // 2 - int(0.7 * lam / dx)
        out = []
        for slab in (True, False):
            s = FDTD(g, {"x": "cpml", "y": "bloch"}, npml=20, bloch_k={"y": ky})
            if slab:
                m = np.zeros(g.shape, bool)
                m[N // 2:] = True
                s.set_material(m, Medium(4.0))
            w = modulated_gaussian(f0, 0.15 * f0)
            for j in range(ny):
                s.add_current("Ez", (30, j, 0), w, np.exp(-1j * ky * j * dx))
            mon = s.add_dft(["Ez"], [f0], (slice(im, im + 1), slice(0, 1), slice(None)))
            s.run_until(60 / f0)
            out.append(mon.phasor("Ez").ravel()[0])
        r = (out[0] - out[1]) / out[1]
        errs.append(abs(abs(r) - abs(fresnel(1.0, 2.0, th0)[0])))
    return CaseResult("FDTD Bloch-periodic oblique incidence (Floquet)", "|r_s| at 25 degrees",
                      "Fresnel coefficient", errs[-1], 5e-3, observed_orders(errs), 2.0,
                      f"errors {', '.join(f'{e:.1e}' for e in errs)}")


@case("FDTD CPML reflection")
def _fdtd_cpml(quick: bool) -> CaseResult:
    from .constants import C0
    from .fdtd import FDTD, YeeGrid, modulated_gaussian

    f0 = 1e9
    dx = C0 / f0 / 20

    def probe(N, npml):
        s = FDTD(YeeGrid((N, N, 1), (dx, dx, dx)), {"x": "cpml", "y": "cpml"}, npml=npml)
        c = N // 2
        s.add_current("Ez", (c, c, 0), modulated_gaussian(f0, 0.8 * f0), 1.0)
        p = s.add_probe("Ez", (c + 15, c + 15, 0))
        s.run(400)
        return np.asarray(p.values)

    # the reference is large enough that nothing returns from its boundary within the 400 steps (0.7 cells/step)
    ref = probe(2 * 300 + 40, 10)
    levels = {n: float(np.max(np.abs(probe(2 * n + 40, n) - ref)) / np.max(np.abs(ref))) for n in (5, 10, 20)}
    return CaseResult("FDTD CPML reflection", "max |E - E_ref| / max |E_ref| 5 cells from the PML",
                      "same run in a domain too large to reflect", levels[10], 1e-4,
                      detail=", ".join(f"{n} cells: {v:.1e} ({20 * math.log10(v):.0f} dB)" for n, v in levels.items()))


@case("FDTD fields through the Maxwell checks")
def _fdtd_checks(quick: bool) -> CaseResult:
    from pinneapple_physics import maxwell_checks as mc

    from .constants import C0
    from .fdtd import FDTD, YeeGrid

    vals, pb = [], []
    for nx in ((21, 41) if quick else (21, 41, 81)):
        dx = 1.0 / (nx - 1)
        s = FDTD(YeeGrid((nx, nx, 1), (dx, dx, dx)))
        x = np.arange(nx) * dx
        X, Y = np.meshgrid(x, x, indexing="ij")
        s.E["Ez"][:, :, 0] = np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
        frames, ts = [], []
        for _ in range(int(0.25 * 2 / (C0 * math.hypot(1, 2) / 2) / s.dt)):
            s.step()
            t, f = s.snapshot()
            frames.append(f)
            ts.append(t)
        fields = {c: np.stack([fr[c][:, :, 0] for fr in frames], -1) for c in ("Ez", "Hx", "Hy")}
        ch = {c.key: c for c in mc.maxwell_checks({"x": x, "y": x, "t": np.array(ts)}, fields)}
        vals.append(ch["ampere_maxwell"].value)
        pb.append(ch["poynting_balance"].value)
    return CaseResult("FDTD fields through the Maxwell checks", "Ampere-Maxwell residual of the snapshots",
                      "0 for a solution of Maxwell's equations", vals[-1], 2e-2, observed_orders(vals),
                      detail=f"Poynting imbalance {', '.join(f'{v:.0e}' for v in pb)}; residuals "
                             f"{', '.join(f'{v:.1e}' for v in vals)}")


@case("FDTD near-to-far-field transform (dipole pattern and power)")
def _fdtd_ntff(quick: bool) -> CaseResult:
    from .constants import C0, ETA0
    from .fdtd import FDTD, NearToFarField, YeeGrid, modulated_gaussian
    from .waves import hertzian_dipole_power

    f0 = 1e9
    lam = C0 / f0
    errs, perr = [], []
    ladder = (10, 15) if quick else (10, 15, 20)
    for ncl in ladder:
        dx = lam / ncl
        N = int(1.2 * lam / dx) + 28
        c = N // 2
        s = FDTD(YeeGrid((N, N, N), (dx, dx, dx)), {"x": "cpml", "y": "cpml", "z": "cpml"}, npml=10)
        w = modulated_gaussian(f0, 0.5 * f0)
        s.add_current("Ez", (c, c, c), w, 1.0)
        b = 14
        nt = NearToFarField(s, [f0], ((b, N - b), (b, N - b), (b, N - b)))
        s.run_until(14 / f0)
        tj = (np.arange(s.n) + 0.5) * s.dt
        Il = np.sum(np.array([w(x) for x in tj]) * np.exp(-2j * np.pi * f0 * tj)) * s.dt * dx ** 3
        th = np.linspace(0.05, np.pi - 0.05, 37)
        Et, _ = nt.far_field(th, 0 * th)
        ex = ETA0 * (2 * np.pi / lam) * abs(Il) * np.sin(th) / (4 * np.pi)
        errs.append(float(np.max(np.abs(np.abs(Et[0]) - ex)) / ex.max()))
        perr.append(abs(float(nt.radiated_power()[0]) / hertzian_dipole_power(Il, 2 * np.pi * f0) - 1))
    orders = [math.log(errs[0] / errs[1]) / math.log(1.5)]
    if len(errs) == 3:
        orders.append(math.log(errs[1] / errs[2]) / math.log(4 / 3))
    return CaseResult("FDTD near-to-far-field transform (dipole pattern and power)", "far-field E_theta(theta)",
                      "eta k |I l| sin(theta) / (4 pi)", errs[-1], 2e-2, orders, 2.0,
                      f"radiated power errors {', '.join(f'{e:.1e}' for e in perr)} vs eta k^2 |I l|^2 / (12 pi)")


@case("FDTD dielectric sphere vs Mie (3-D, TF/SF + near-to-far field)")
def _fdtd_mie(quick: bool) -> CaseResult:
    from .constants import C0, ETA0
    from .fdtd import FDTD, NearToFarField, PlaneWaveTFSF, YeeGrid, fill_fraction, modulated_gaussian
    from .media import Medium
    from .mie import mie

    f0 = 1e9
    lam = C0 / f0
    a, eps = 0.25 * lam, 4.0
    ref = mie(2.0, 2 * math.pi * a / lam).qsca
    errs = []
    ladder = (10,) if quick else (10, 15, 20)
    for ncl in ladder:
        dx = lam / ncl
        npml = 10
        N = int(2 * a / dx) + 12 + 2 * npml + 10
        c = N // 2
        g = YeeGrid((N, N, N), (dx, dx, dx))
        s = FDTD(g, {"x": "cpml", "y": "cpml", "z": "cpml"}, npml=npml)
        s.set_material(fill_fraction(g, lambda X, Y, Z, c=c, dx=dx: (X - c * dx) ** 2 + (Y - c * dx) ** 2
                                     + (Z - c * dx) ** 2 <= a * a, 6), Medium(eps))
        tf = npml + 5
        s.add_plane_wave(PlaneWaveTFSF("x", "Ez", modulated_gaussian(f0, 0.5 * f0), ((tf, N - tf),) * 3))
        inc = s.incident_monitor([f0])
        nt = NearToFarField(s, [f0], ((npml + 2, N - npml - 2),) * 3)   # in the scattered-field region
        s.run_until(20 / f0)
        E0 = inc.phasor("Ey")[0, c, 0, 0]
        Q = float(nt.radiated_power()[0]) / (abs(E0) ** 2 / (2 * ETA0) * math.pi * a * a)
        errs.append(abs(Q / ref - 1))
    return CaseResult("FDTD dielectric sphere vs Mie (3-D, TF/SF + near-to-far field)",
                      "Q_sca, eps = 4, x = 1.57", "Mie series", errs[-1], 3e-2,
                      detail=f"errors {', '.join(f'{e:.1e}' for e in errs)} at {', '.join(map(str, ladder))} cells per "
                             "wavelength: a plateau near 2 %, not a convergence order (harmonic/arithmetic sub-cell "
                             "averaging is not the full anisotropic smoothing an oblique surface needs; arithmetic "
                             "averaging alone gives 9.7 % at 10 cells)")
