"""Electro- and magnetostatics, academic level: Coulomb and superposition, Gauss's law, the parallel-plate
capacitor with its edge effect, Biot-Savart (wire, loop, solenoid), Ampere's law and the Hall effect.

Every experiment computes its quantity by a numerical route (superposition of discrete charges, surface
quadrature, a finite-volume field solution, polygonal filaments) and compares it with the textbook formula, with
a refinement ladder that shows the error going to zero. Field values come from
:mod:`pinneapple_simulation.electromagnetics` (``pp.em``) where it has them; the Gauss residual of a field on a
grid is :func:`pinneapple_physics.maxwell_checks.gauss_electric` / ``gauss_magnetic``.
"""
from __future__ import annotations

import math

import numpy as np

from pinneapple_simulation.electromagnetics import statics as S
from pinneapple_simulation.electromagnetics.constants import EPS0, MU0

from .. import plots as P
from ..core import Experiment, Result, observed_order
from ..numerics import TensorGrid, field_energy, graded_axis, solve_potential
from ..registry import register

AREA = "physics.electromagnetism"
E_CHARGE = 1.602176634e-19          # C, exact (SI 2019)


def _gauss_legendre(n: int, a: float, b: float):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


# ===================================================================================== Coulomb
def ring_potential(points, Q: float, R: float) -> np.ndarray:
    """Potential of a uniformly charged ring (radius ``R``, charge ``Q``, plane z = 0, centred on the z axis):
    ``phi = Q / (4 pi eps0) (2 / pi) K(m) / sqrt((R + s)^2 + z^2)``, ``m = 4 R s / ((R + s)^2 + z^2)``
    (Jackson, *Classical Electrodynamics*, 3rd ed., sec. 3.3 / problem 3.3; Smythe, *Static and Dynamic
    Electricity*, sec. 5.10)."""
    from scipy.special import ellipk

    p = np.asarray(points, float)
    s = np.hypot(p[..., 0], p[..., 1])
    z = p[..., 2]
    D = (R + s) ** 2 + z ** 2
    return Q / (4 * math.pi * EPS0) * (2 / math.pi) * ellipk(4 * R * s / D) / np.sqrt(D)


@register
class CoulombSuperposition(Experiment):
    name = "em.coulomb_superposition"
    title = "Coulomb's law and superposition"
    area = AREA
    topic = "electrostatics"
    dims = ("3D", "2D")
    description = ("A charged ring (3-D) and a pair of finite line charges (2-D cross-section) built from N point "
                   "charges by superposition, against the closed forms; the error falls as N grows.")
    references = ("Griffiths, Introduction to Electrodynamics, 4th ed., ex. 2.2 (finite line) and ex. 2.5 (ring)",
                  "Jackson, Classical Electrodynamics, 3rd ed., sec. 3.3 (ring potential, elliptic integral)")
    params = {"Q": 1e-9, "R": 0.05, "ladder": [16, 32, 64, 128, 256], "lam": 1e-9, "half_length": 1.0,
              "separation": 0.04}
    quick = {"ladder": [16, 32, 64, 128]}

    def _run(self, res: Result, dim: str, p):
        (self._ring if dim == "3D" else self._lines)(res, p)

    def _ring(self, res: Result, p):
        Q, R = p["Q"], p["R"]
        th = np.linspace(0, 2 * np.pi, 13)[:-1]
        pts = np.stack([0.6 * R * np.cos(th), 0.6 * R * np.sin(th), np.linspace(-0.4, 0.8, 12) * R], axis=1)
        pts = np.concatenate([pts, [[0, 0, 0.7 * R], [1.8 * R, 0, 0.3 * R]]])
        ref = ring_potential(pts, Q, R)
        errs = []
        for n in p["ladder"]:
            a = 2 * np.pi * (np.arange(n) + 0.5) / n
            at = np.stack([R * np.cos(a), R * np.sin(a), np.zeros(n)], axis=1)
            phi = sum(S.point_charge_field(pts, Q / n, at=c)[0] for c in at)
            errs.append(float(np.max(np.abs(phi / ref - 1))))
        n = p["ladder"][-1]
        res.value("max_rel_error_potential", errs[-1], "", f"max over {len(pts)} off-axis points, N = {n} charges")
        p0 = np.array([[0.6 * R, 0.0, 0.3 * R]])
        a0 = 2 * np.pi * (np.arange(n) + 0.5) / n
        phi0 = sum(S.point_charge_field(p0, Q / n, at=(R * math.cos(t), R * math.sin(t), 0.0))[0][0] for t in a0)
        res.compare("ring potential at s = 0.6 R, z = 0.3 R", phi0, float(ring_potential(p0, Q, R)[0]), tol=1e-6,
                    unit="V", source="Jackson sec. 3.3: phi = Q/(4 pi eps0) (2/pi) K(m)/sqrt((R+s)^2+z^2)",
                    how=f"superposition of {n} point charges (pp.em.statics.point_charge_field)")
        # the axial field: exact for every N by symmetry; compare the field from the N charges
        z = 0.75 * R
        a = 2 * np.pi * np.arange(n) / n
        at = np.stack([R * np.cos(a), R * np.sin(a), np.zeros(n)], axis=1)
        Ez = sum(S.point_charge_field(np.array([[0, 0, z]]), Q / n, at=c)[1][0, 2] for c in at)
        res.compare("axial field E_z(z = 0.75 R)", Ez, Q * z / (4 * math.pi * EPS0 * (z * z + R * R) ** 1.5),
                    tol=1e-10, unit="V/m", source="Griffiths ex. 2.5: E_z = Q z / (4 pi eps0 (z^2 + R^2)^(3/2))")
        far = np.array([[0, 0, 40 * R]])
        Efar = sum(S.point_charge_field(far, Q / n, at=c)[1][0, 2] for c in at)
        res.compare("far field -> point charge (r = 40 R)", Efar, Q / (4 * math.pi * EPS0 * (40 * R) ** 2),
                    tol=1e-3, kind="limit", unit="V/m", source="Coulomb's law, multipole limit r >> R")
        orders = observed_order(errs)
        res.value("convergence_rate", orders[-1] if orders else float("nan"), "",
                  "observed order of the potential error in N (exponential convergence shows as a growing order)")
        res.check("error falls to round-off (below 1e-12) as N grows", errs[-1] < 1e-12 and errs[0] > 1e3 * errs[-1],
                  value=errs[-1], limit=1e-12, detail=", ".join(f"N={k}: {e:.1e}" for k, e in zip(p["ladder"], errs, strict=True)))
        # figure: potential in the x-z plane and the convergence
        x = np.linspace(-2 * R, 2 * R, 161)
        zz = np.linspace(-1.5 * R, 1.5 * R, 121)
        X, Z = np.meshgrid(x, zz, indexing="ij")
        G = np.stack([X, np.zeros_like(X), Z], axis=-1)
        phi = sum(S.point_charge_field(G, Q / n, at=c)[0] for c in at)
        fig, ax = P.figure(2, width=4.3)
        im = ax[0].contourf(X / R, Z / R, np.log10(phi), 30, cmap=P.SEQUENTIAL)
        ax[0].plot([-1, 1], [0, 0], "o", color=P.REFERENCE, ms=5)
        ax[0].set(xlabel="x / R", ylabel="z / R", title=f"log10 potential of {n} charges on a ring")
        ax[0].set_aspect("equal")
        fig.colorbar(im, ax=ax[0], shrink=0.8)
        ax[1].loglog(p["ladder"], errs, "o-", color=P.COMPUTED, lw=2, ms=6, label="max |phi/phi_ring - 1|")
        P.integer_ticks(ax[1], p["ladder"])
        ax[1].set(xlabel="number of point charges N", ylabel="relative error", title="superposition -> ring")
        ax[1].legend(fontsize=8)
        P.badge(ax[1], f"N = {n}: {errs[-1]:.1e}", errs[-1] < 1e-6)
        fig.tight_layout()
        res.figure("coulomb_ring", fig)
        res.data("convergence", N=p["ladder"], max_rel_error=errs)

    def _lines(self, res: Result, p):
        lam, L, sep = p["lam"], p["half_length"], p["separation"]
        x = np.linspace(-2.5 * sep, 2.5 * sep, 11)
        y = np.linspace(0.3 * sep, 1.6 * sep, 5)
        X, Y = np.meshgrid(x, y, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)

        def finite_line(pts, x0, q):
            """Griffiths ex. 2.2 generalised: field of a line segment z in [-L, L] at x = x0, in the plane z = 0."""
            d = pts[:, :2] - [x0, 0.0]
            s = np.linalg.norm(d, axis=1)
            return (q / (2 * math.pi * EPS0 * s) * L / np.sqrt(L * L + s * s))[:, None] * d / s[:, None]

        ref = finite_line(pts, -sep / 2, lam) + finite_line(pts, sep / 2, -lam)
        errs = []
        for n in p["ladder"]:
            zc = -L + (np.arange(4 * n) + 0.5) * (2 * L / (4 * n))
            E = np.zeros((len(pts), 3))
            for x0, q in ((-sep / 2, lam), (sep / 2, -lam)):
                for z in zc:
                    E += S.point_charge_field(pts, q * 2 * L / len(zc), at=(x0, 0.0, z))[1]
            errs.append(float(np.max(np.linalg.norm(E[:, :2] - ref, axis=1) / np.linalg.norm(ref, axis=1))))
        n = 4 * p["ladder"][-1]
        P1 = np.array([[0.0, 0.5 * sep, 0.0]])
        E1 = np.zeros(3)
        zc = -L + (np.arange(n) + 0.5) * (2 * L / n)
        for x0, q in ((-sep / 2, lam), (sep / 2, -lam)):
            for z in zc:
                E1 += S.point_charge_field(P1, q * 2 * L / n, at=(x0, 0.0, z))[1][0]
        ref1 = (finite_line(P1, -sep / 2, lam) + finite_line(P1, sep / 2, -lam))[0]
        res.value("max_rel_error_field", errs[-1], "", f"max over {len(pts)} points, {n} charges per line")
        res.compare("line-pair field E_x at (0, d/2)", E1[0], ref1[0], tol=1e-6, unit="V/m",
                    source="Griffiths ex. 2.2: E = lam/(2 pi eps0 s) L/sqrt(L^2+s^2) per segment, superposed",
                    how=f"{n} point charges per line (midpoint rule along z)")
        e_inf = S.line_charge_field(np.array([[0.0, 0.01, 0.0]]), lam)[0, 1]
        e_fin = finite_line(np.array([[0.0, 0.01, 0.0]]), 0.0, lam)[0, 1]
        res.compare("finite line -> infinite line (s = 0.01 m, L = 1 m)", e_fin, e_inf, tol=1e-3, kind="limit",
                    unit="V/m", source="Gauss's law for an infinite line: lam / (2 pi eps0 s)")
        res.check("worst error over the 55 points below 1e-6", errs[-1] < 1e-6, value=errs[-1], limit=1e-6,
                  detail=", ".join(f"{4 * k}: {e:.1e}" for k, e in zip(p["ladder"], errs, strict=True)) +
                  " (faster than any power until the charges are closer than the points are to the line)")
        g = np.linspace(-3 * sep, 3 * sep, 121)
        GX, GY = np.meshgrid(g, g, indexing="ij")
        Gp = np.stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)], axis=1)
        Ef = (finite_line(Gp, -sep / 2, lam) + finite_line(Gp, sep / 2, -lam)).reshape(GX.shape + (2,))
        fig, ax = P.figure(2, width=4.3)
        mag = np.log10(np.linalg.norm(Ef, axis=-1))
        ax[0].contourf(GX / sep, GY / sep, mag, 30, cmap=P.SEQUENTIAL)
        ax[0].streamplot(g / sep, g / sep, Ef[..., 0].T, Ef[..., 1].T, color=P.INK, linewidth=0.7, density=1.1,
                         arrowsize=0.8)
        ax[0].plot([-0.5, 0.5], [0, 0], "o", color=P.REFERENCE, ms=6)
        ax[0].set(xlabel="x / d", ylabel="y / d", title="field of +lam / -lam line charges (2-D)")
        ax[0].set_aspect("equal")
        counts = [4 * k for k in p["ladder"]]
        ax[1].loglog(counts, errs, "o-", color=P.COMPUTED, lw=2, ms=6, label="superposition")
        P.integer_ticks(ax[1], counts)
        ax[1].set(xlabel="point charges per line", ylabel="max relative error (55 points)", title="convergence")
        ax[1].legend(fontsize=8, loc="lower left")
        P.badge(ax[1], f"{4 * p['ladder'][-1]} charges: {errs[-1]:.1e}", errs[-1] < 1e-6, loc="upper right")
        fig.tight_layout()
        res.figure("coulomb_lines_2d", fig)
        res.data("convergence", N=p["ladder"], max_rel_error=errs)


# ===================================================================================== Gauss
@register
class GaussLaw(Experiment):
    name = "em.gauss_law"
    title = "Gauss's law"
    area = AREA
    topic = "electrostatics"
    dims = ("3D", "2D")
    description = ("The flux of E through a closed surface (cube, 3-D) or curve (square, 2-D) by Gauss-Legendre "
                   "quadrature equals the enclosed charge over eps0; charges outside contribute nothing. The "
                   "divergence residual of a charged ball's field on a grid is checked with pp.maxwell.")
    references = ("Griffiths, Introduction to Electrodynamics, 4th ed., sec. 2.2", "Purcell & Morin, Electricity "
                  "and Magnetism, 3rd ed., sec. 1.10")
    params = {"ladder": [4, 8, 16, 32], "L": 1.0}
    quick = {"ladder": [4, 8, 16]}

    charges3 = ((2e-9, (0.2, -0.1, 0.15)), (-0.5e-9, (-0.3, 0.25, -0.2)), (1e-9, (0.8, 0.1, 0.0)),
                (-3e-9, (0.0, 0.0, 1.4)))

    def _run(self, res: Result, dim: str, p):
        (self._cube if dim == "3D" else self._square)(res, p)

    def _cube(self, res: Result, p):
        h = p["L"] / 2
        qin = sum(q for q, at in self.charges3 if max(map(abs, at)) < h)
        errs, fluxes = [], []
        for n in p["ladder"]:
            flux = 0.0
            # composite Gauss-Legendre: n panels x 4 points per face direction
            u, w = [], []
            for k in range(n):
                a, b = -h + 2 * h * k / n, -h + 2 * h * (k + 1) / n
                uu, ww = _gauss_legendre(4, a, b)
                u.append(uu)
                w.append(ww)
            u, w = np.concatenate(u), np.concatenate(w)
            U, V = np.meshgrid(u, u, indexing="ij")
            W = np.outer(w, w)
            for ax in range(3):
                for sgn in (-1, 1):
                    pts = np.zeros(U.shape + (3,))
                    o = [i for i in range(3) if i != ax]
                    pts[..., ax] = sgn * h
                    pts[..., o[0]], pts[..., o[1]] = U, V
                    E = sum(S.point_charge_field(pts, q, at=at)[1] for q, at in self.charges3)
                    flux += sgn * float(np.sum(E[..., ax] * W))
            fluxes.append(flux)
            errs.append(abs(flux * EPS0 / qin - 1))
        res.value("flux_times_eps0", fluxes[-1] * EPS0, "C", f"composite Gauss-Legendre, {p['ladder'][-1]}x4 "
                  "points per face edge")
        res.compare("flux through the cube x eps0", fluxes[-1] * EPS0, qin, tol=1e-6, unit="C",
                    source="Gauss's law: oint E.dA = Q_enc / eps0 (Griffiths eq. 2.13)",
                    how="2 charges inside, 2 outside (one 0.4 L from a face)")
        res.check("charges outside contribute zero flux", errs[-1] < 1e-6, value=errs[-1], limit=1e-6,
                  detail="the outside charges are summed into every face")
        # divergence residual of a uniformly charged ball's field on a grid (pp.maxwell)
        from pinneapple_physics.maxwell_checks import gauss_electric

        g = np.linspace(-1, 1, 41)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        from scipy.special import erf

        r = np.maximum(np.sqrt(X ** 2 + Y ** 2 + Z ** 2), 1e-12)
        sg, rho0 = 0.3, 1e-9                 # Gaussian cloud rho0 exp(-r^2/s^2): Q(r) in closed form
        Qr = rho0 * math.pi ** 1.5 * sg ** 3 * (erf(r / sg) - 2 * r / (math.sqrt(math.pi) * sg) * np.exp(-(r / sg) ** 2))
        Er = Qr / (4 * math.pi * EPS0 * r * r)
        f = {"Ex": Er * X / r, "Ey": Er * Y / r, "Ez": Er * Z / r, "rho": rho0 * np.exp(-(r / sg) ** 2)}
        ck = gauss_electric({"x": g, "y": g, "z": g}, f)
        res.check("div E = rho/eps0, Gaussian cloud on a 41^3 grid (pp.maxwell.gauss_electric)", ck.passed, value=ck.value,
                  limit=ck.limit, source="pinneapple_physics.maxwell_checks", detail=ck.detail)
        bad = dict(f, rho=0.8 * f["rho"])
        ckb = gauss_electric({"x": g, "y": g, "z": g}, bad)
        res.check("the same check rejects a 20 % wrong charge density", not ckb.passed,
                  source="pinneapple_physics.maxwell_checks",
                  detail=f"residual {ckb.value:.3g} > limit {ckb.limit:.3g}: rejected")
        fig, ax = P.figure(2, width=4.3, three_d=(0,))
        a3 = ax[0]
        for q, at in self.charges3:
            a3.scatter(*at, s=60, color=P.COMPUTED if q > 0 else P.REFERENCE, depthshade=False)
        c = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(np.abs(c[i] - c[j]) > 0) == 1:
                    a3.plot(*zip(c[i], c[j], strict=True), color=P.INK2, lw=0.8)
        a3.set(title="cube, charges inside (+ blue) and outside", xlabel="x", ylabel="y", zlabel="z")
        nn = np.asarray(p["ladder"]) * 4
        ax[1].loglog(nn, np.maximum(errs, 1e-16), "o-", color=P.COMPUTED, lw=2, ms=6)
        ax[1].set(xlabel="quadrature points per face edge", ylabel="|flux eps0 / Q_enc - 1|",
                  title="Gauss's law by surface quadrature")
        P.badge(ax[1], f"{errs[-1]:.1e} at {nn[-1]} points", errs[-1] < 1e-6)
        fig.tight_layout()
        res.figure("gauss_cube", fig)
        res.data("convergence", points_per_edge=nn, rel_error=errs)

    def _square(self, res: Result, p):
        lines = ((3e-9, (0.1, 0.2)), (-1e-9, (-0.3, -0.25)), (2e-9, (0.75, 0.0)))
        h = p["L"] / 2
        lin = sum(q for q, at in lines if max(map(abs, at)) < h)
        errs = []
        for n in p["ladder"]:
            flux = 0.0
            for k in range(n):
                a, b = -h + 2 * h * k / n, -h + 2 * h * (k + 1) / n
                u, w = _gauss_legendre(4, a, b)
                for ax in range(2):
                    for sgn in (-1, 1):
                        pts = np.zeros((len(u), 3))
                        pts[:, ax] = sgn * h
                        pts[:, 1 - ax] = u
                        E = sum(S.line_charge_field(pts - [*at, 0.0], q) for q, at in lines)
                        flux += sgn * float(np.sum(E[:, ax] * w))
            errs.append(abs(flux * EPS0 / lin - 1))
        res.compare("flux per unit length x eps0", (1 + errs[-1]) * lin, lin, tol=1e-6, unit="C/m",
                    source="Gauss's law in 2-D: oint E.n dl = lam_enc / eps0", how="2 line charges inside, 1 outside")
        res.check("error decreases with quadrature points", errs[-1] < errs[0], value=errs[-1])
        fig, ax = P.figure(1, width=4.6)
        nn = np.asarray(p["ladder"]) * 4
        ax[0].loglog(nn, np.maximum(errs, 1e-16), "o-", color=P.COMPUTED, lw=2, ms=6)
        ax[0].set(xlabel="quadrature points per side", ylabel="|flux eps0 / lam_enc - 1|",
                  title="Gauss's law around line charges (2-D)")
        P.badge(ax[0], f"{errs[-1]:.1e}", errs[-1] < 1e-6, loc="upper right")
        fig.tight_layout()
        res.figure("gauss_square_2d", fig)
        res.data("convergence", points_per_side=nn, rel_error=errs)


# ===================================================================================== capacitor
def palmer_strip_capacitance(w: float, d: float, eps_r: float = 1.0) -> float:
    """Capacitance per unit length of two parallel thin strips (width ``w``, gap ``d``) with the edge
    correction of Palmer (1937): ``eps (w/d) [1 + d/(pi w) (1 + ln(2 pi w / d))]`` (F/m). Asymptotic for
    ``w >> d``. H. B. Palmer, "The capacitance of a parallel-plate capacitor by the Schwartz-Christoffel
    transformation", Trans. AIEE 56, 363 (1937)."""
    r = w / d
    return eps_r * EPS0 * r * (1 + (1 + math.log(2 * math.pi * r)) / (math.pi * r))


def kirchhoff_disc_capacitance(a: float, d: float, eps_r: float = 1.0) -> float:
    """Capacitance of two thin coaxial discs (radius ``a``, gap ``d``), Kirchhoff (1877):
    ``eps [pi a^2 / d + a (ln(16 pi a / d) - 1)]`` (F), asymptotic for ``d << a``. G. Kirchhoff, Monatsber.
    Akad. Wiss. Berlin (1877) 144; see also Jackson, Am. J. Phys. 68, 789 (2000)."""
    return eps_r * EPS0 * (math.pi * a * a / d + a * (math.log(16 * math.pi * a / d) - 1))


def strip_capacitor_2d(w: float, d: float, h: float, L: float = 40.0, growth: float = 1.08):
    """``(C per unit length, grid, phi)`` of two thin strips at +-1/2 V by finite volumes on a graded quarter
    domain (mirror at x = 0, antisymmetry phi = 0 at y = 0, grounded box at distance ``L``)."""
    x = graded_axis(w / 2 + 4 * d, h, L, growth)
    y = graded_axis(d / 2 + 4 * d, h, L, growth)
    g = TensorGrid((x, y))
    X, Y = g.mesh()
    fixed = np.zeros(g.shape, bool)
    val = np.zeros(g.shape)
    fixed[:, 0] = True
    plate = np.isclose(Y, d / 2) & (X <= w / 2 + 1e-12)
    if not np.isclose(x, w / 2).any() or not np.isclose(y, d / 2).any():
        raise ValueError(f"h = {h:g} does not put nodes on the plate (use an even number of cells across d)")
    fixed |= plate
    val[plate] = 0.5
    fixed[-1, :] = fixed[:, -1] = True
    phi = solve_potential(g, fixed, val)
    return 4 * 2 * EPS0 * field_energy(g, phi), g, phi


def disc_capacitor(a: float, d: float, h: float, *, cartesian: bool, L: float = 12.0):
    """Capacitance of two coaxial thin discs at +-1/2 V: an axisymmetric (r, z) solution, or a 3-D Cartesian
    one on one eighth of the domain (mirror planes x = 0, y = 0, antisymmetry at z = 0, staircase disc edge)."""
    if cartesian:
        x = graded_axis(a + 2 * d, h, L, 1.15)
        z = graded_axis(d / 2 + 2 * d, h, L, 1.15)
        g = TensorGrid((x, x, z))
        X, Y, Z = g.mesh()
        plate = np.isclose(Z, d / 2) & (X ** 2 + Y ** 2 <= a * a + 1e-12)
        if not np.isclose(z, d / 2).any():
            raise ValueError(f"h = {h:g} does not put nodes on the plates (use an even number of cells across d)")
        fixed = plate.copy()
        fixed[:, :, 0] = fixed[-1] = fixed[:, -1] = fixed[:, :, -1] = True
        phi = solve_potential(g, fixed, np.where(plate, 0.5, 0.0))
        return 8 * 2 * EPS0 * field_energy(g, phi), g, phi
    r = graded_axis(a + 2 * d, h, 3 * L, 1.08)
    z = graded_axis(d / 2 + 2 * d, h, 3 * L, 1.08)
    g = TensorGrid((r, z))
    R, Z = g.mesh()
    plate = np.isclose(Z, d / 2) & (R <= a + 1e-12)
    if not np.isclose(z, d / 2).any() or not np.isclose(r, a).any():
        raise ValueError(f"h = {h:g} does not put nodes on the plates (use an even number of cells across d)")
    fixed = plate.copy()
    fixed[:, 0] = fixed[-1] = fixed[:, -1] = True
    phi = solve_potential(g, fixed, np.where(plate, 0.5, 0.0), eps=R, mean="arithmetic")
    return 2 * 2 * math.pi * 2 * EPS0 * field_energy(g, phi, eps=R, mean="arithmetic"), g, phi


@register
class ParallelPlateCapacitor(Experiment):
    name = "em.parallel_plate_capacitor"
    title = "Parallel-plate capacitor with edge effect"
    area = AREA
    topic = "electrostatics"
    dims = ("2D", "3D")
    description = ("Capacitance of thin plates from the field energy of a finite-volume solution, refined three "
                   "times and Richardson-extrapolated: strips in 2-D against Palmer's edge correction, discs in "
                   "3-D against Kirchhoff's formula. The excess over eps A / d is the fringing field.")
    references = ("H. B. Palmer, Trans. AIEE 56, 363 (1937)", "G. Kirchhoff, Monatsber. Akad. Wiss. Berlin "
                  "(1877) 144", "Roache, ASME J. Fluids Eng. 116, 405 (1994) (Richardson extrapolation, GCI)")
    params = {"w_over_d": 10.0, "a_over_d": 10.0, "d": 0.1, "cells_per_gap": [10, 20, 40],
              "cells_per_gap_3d": [4, 6, 8]}
    quick = {"cells_per_gap": [6, 12, 24], "cells_per_gap_3d": [2, 4, 6]}

    def _run(self, res: Result, dim: str, p):
        from pinneapple_analysis.verification.convergence import richardson_extrapolate

        d = p["d"]
        if dim == "2D":
            w = p["w_over_d"] * d
            cs = [strip_capacitor_2d(w, d, d / k) for k in p["cells_per_gap"]]
            C = [c[0] for c in cs]
            rich = richardson_extrapolate(C[0], C[1], C[2], 2.0)
            ref = palmer_strip_capacitance(w, d)
            ideal = EPS0 * w / d
            res.value("C_per_length_fine", C[-1], "F/m", f"field energy, {p['cells_per_gap'][-1]} cells across the gap")
            res.value("observed_order", rich.observed_order, "", "Roache three-grid observed order")
            res.value("gci_fine", rich.gci_fine, "", "grid convergence index of the fine grid (Fs = 1.25)")
            res.value("fringe_excess", float(rich.extrapolated_value) / ideal - 1, "",
                      "C / (eps0 w / d) - 1, extrapolated")
            res.compare("C per unit length (extrapolated)", float(rich.extrapolated_value), ref, tol=1e-2,
                        unit="F/m", source="Palmer (1937): eps0 (w/d) [1 + d/(pi w)(1 + ln(2 pi w/d))]",
                        how=f"finite volumes, {p['cells_per_gap']} cells per gap, Richardson")
            res.check("solution in the asymptotic range (Roache)", rich.observed_order > 0.5, value=rich.observed_order,
                      detail=f"asymptotic ratio {rich.asymptotic_ratio:.3f}; a thin plate edge (phi ~ r^1/2) "
                             "limits the order to about 1")
            g, phi = cs[-1][1], cs[-1][2]
            X, Y = g.mesh()
            m = (X < 1.6 * w / 2) & (Y < 2.0 * d)
            nx, ny = int(m[:, 0].sum()), int(m[0].sum())
            fig, ax = P.figure(2, width=4.6)
            xx, yy, pp = X[:nx, :ny], Y[:nx, :ny], phi[:nx, :ny]
            xs = np.concatenate([-xx[::-1, :], xx], 0)
            ys = np.concatenate([yy[::-1, :], yy], 0)
            ps = np.concatenate([pp[::-1, :], pp], 0)
            full_x = np.concatenate([xs, xs[:, ::-1]], 1)
            full_y = np.concatenate([ys, -ys[:, ::-1]], 1)
            full_p = np.concatenate([ps, -ps[:, ::-1]], 1)
            cf = ax[0].contourf(full_x / d, full_y / d, full_p, 31, cmap=P.DIVERGING)
            ax[0].contour(full_x / d, full_y / d, full_p, 15, colors="white", linewidths=0.4)
            ax[0].plot([-w / 2 / d, w / 2 / d], [0.5, 0.5], color=P.INK, lw=2.5)
            ax[0].plot([-w / 2 / d, w / 2 / d], [-0.5, -0.5], color=P.INK, lw=2.5)
            ax[0].set(xlabel="x / d", ylabel="y / d", title="potential (V), strips at +-0.5 V")
            fig.colorbar(cf, ax=ax[0], shrink=0.8)
            hh = [1 / k for k in p["cells_per_gap"]]
            ax[1].plot(hh, np.asarray(C) / ideal, "o-", color=P.COMPUTED, lw=2, ms=6, label="finite volumes")
            ax[1].axhline(float(rich.extrapolated_value) / ideal, color=P.COMPUTED, ls=":", lw=1.2,
                          label="Richardson")
            ax[1].axhline(ref / ideal, color=P.REFERENCE, ls="--", lw=1.5, label="Palmer (1937)")
            ax[1].axhline(1.0, color=P.INK2, lw=1, label="eps0 w / d (no fringing)")
            ax[1].set(xlabel="h / d", ylabel="C / (eps0 w / d)", title=f"w/d = {p['w_over_d']:g}")
            ax[1].legend(fontsize=7.5)
            err = abs(float(rich.extrapolated_value) / ref - 1)
            P.badge(ax[1], f"vs Palmer {100 * err:.2f} %", err < 1e-2, loc="lower left")
            fig.tight_layout()
            res.figure("capacitor_2d", fig)
            res.data("ladder", cells_per_gap=p["cells_per_gap"], C=C, palmer=ref, ideal=ideal)
        else:
            a = p["a_over_d"] * d
            C = [disc_capacitor(a, d, d / k, cartesian=True)[0] for k in p["cells_per_gap_3d"]]
            r = [p["cells_per_gap_3d"][1] / p["cells_per_gap_3d"][0], p["cells_per_gap_3d"][2] / p["cells_per_gap_3d"][1]]
            ax_ladder = [8, 16, 32] if p["cells_per_gap_3d"][0] >= 4 else [4, 8, 16]
            Cax = [disc_capacitor(a, d, d / k, cartesian=False)[0] for k in ax_ladder]
            rax = richardson_extrapolate(Cax[0], Cax[1], Cax[2], 2.0)
            ref = kirchhoff_disc_capacitance(a, d)
            ideal = EPS0 * math.pi * a * a / d
            for k, c in zip(p["cells_per_gap_3d"], C, strict=True):
                res.value(f"C_cartesian_{k}_cells", c, "F", "3-D Cartesian finite volumes (one eighth, staircase edge)")
            res.value("C_cartesian_fine", C[-1], "F", f"3-D Cartesian finite volumes, {p['cells_per_gap_3d'][-1]} "
                      "cells across the gap (staircase disc edge)")
            res.value("C_axisymmetric_extrapolated", float(rax.extrapolated_value), "F",
                      f"(r, z) finite volumes {ax_ladder} cells per gap, Richardson")
            res.value("fringe_excess", float(rax.extrapolated_value) / ideal - 1, "", "C / (eps0 pi a^2 / d) - 1")
            res.compare("C, 3-D Cartesian (finest) vs axisymmetric", C[-1], float(rax.extrapolated_value), tol=1e-2,
                        kind="convergence", unit="F", source="this experiment: axisymmetric solution, Richardson "
                        "extrapolated (independent discretisation of the same 3-D problem)")
            res.compare("C, axisymmetric (extrapolated)", float(rax.extrapolated_value), ref, tol=1.5e-2, unit="F",
                        source="Kirchhoff (1877): eps0 [pi a^2/d + a (ln(16 pi a/d) - 1)], asymptotic d << a")
            # Kirchhoff is asymptotic: its error must shrink as a/d grows (scaling check)
            dev = []
            for ad in (2.5, 5.0, 10.0):
                cc = [disc_capacitor(ad * d, d, d / k, cartesian=False)[0] for k in ax_ladder]
                ce = float(richardson_extrapolate(cc[0], cc[1], cc[2], 2.0).extrapolated_value)
                dev.append(ce / kirchhoff_disc_capacitance(ad * d, d) - 1)
            res.check("Kirchhoff's error falls as a/d grows (asymptotic formula)",
                      abs(dev[2]) < abs(dev[1]) < abs(dev[0]),
                      detail=", ".join(f"a/d={ad:g}: {100 * v:+.2f} %" for ad, v in zip((2.5, 5, 10), dev, strict=True)))
            res.data("scaling", a_over_d=[2.5, 5.0, 10.0], deviation_from_kirchhoff=dev)
            _, g, phi = disc_capacitor(a, d, d / p["cells_per_gap_3d"][-1], cartesian=True)
            fig, ax = P.figure(2, width=4.6)
            X, Y, Z = g.mesh()
            j = 0
            m = (g.axes[0] < 1.5 * a)
            mz = g.axes[2] < 3 * d
            xs, zs, ps = X[m][:, j][:, mz], Z[m][:, j][:, mz], phi[m][:, j][:, mz]
            full_x = np.concatenate([-xs[::-1], xs], 0)
            full_z = np.concatenate([zs[::-1], zs], 0)
            full_p = np.concatenate([ps[::-1], ps], 0)
            full_x = np.concatenate([full_x[:, ::-1], full_x], 1)
            full_z = np.concatenate([-full_z[:, ::-1], full_z], 1)
            full_p = np.concatenate([-full_p[:, ::-1], full_p], 1)
            cf = ax[0].contourf(full_x / d, full_z / d, full_p, 31, cmap=P.DIVERGING)
            ax[0].contour(full_x / d, full_z / d, full_p, 15, colors="white", linewidths=0.4)
            for zp in (0.5, -0.5):
                ax[0].plot([-a / d, a / d], [zp, zp], color=P.INK, lw=2.5)
            ax[0].set(xlabel="x / d", ylabel="z / d", title="3-D solution, plane y = 0 (discs at z = +-d/2)")
            fig.colorbar(cf, ax=ax[0], shrink=0.8)
            ax[1].plot([2.5, 5, 10], 100 * np.asarray(dev), "o-", color=P.COMPUTED, lw=2, ms=6,
                       label="axisymmetric FV vs Kirchhoff")
            ax[1].axhline(0, color=P.INK2, lw=1)
            ax[1].set(xlabel="a / d", ylabel="C / C_Kirchhoff - 1 (%)", title="Kirchhoff is asymptotic in d/a")
            ax[1].legend(fontsize=8)
            P.badge(ax[1], f"a/d = 10: {100 * dev[-1]:+.2f} %", abs(dev[-1]) < 1.5e-2)
            fig.tight_layout()
            res.figure("capacitor_3d", fig)
            res.data("ladder", cells_per_gap=p["cells_per_gap_3d"], C=C, axisym_cells=ax_ladder, C_axisym=Cax,
                     kirchhoff=ref, ideal=ideal, ratios=r)


# ===================================================================================== Biot-Savart
@register
class BiotSavart(Experiment):
    name = "em.biot_savart"
    title = "Biot-Savart: wire, loop and solenoid"
    area = AREA
    topic = "magnetostatics"
    dims = ("3D",)
    description = ("The Biot-Savart integral evaluated numerically (midpoint rule along a straight wire; "
                   "polygonal filaments for a loop and a helical coil) against the closed forms: finite wire, "
                   "loop (elliptic integrals, on and off the axis) and the finite solenoid on its axis.")
    references = ("Griffiths, Introduction to Electrodynamics, 4th ed., ex. 5.5 (finite wire), ex. 5.6 (loop), "
                  "prob. 5.11 (finite solenoid)", "Jackson, Classical Electrodynamics, 3rd ed., sec. 5.5 (loop, "
                  "elliptic integrals)")
    params = {"I": 2.0, "R": 0.05, "length": 0.4, "turns": 400, "ladder": [32, 64, 128, 256]}
    quick = {"turns": 100, "ladder": [32, 64, 128]}

    def _run(self, res: Result, dim: str, p):
        I, R = p["I"], p["R"]  # noqa: E741  (I: current)
        # (a) straight wire z in [-L, L] at distance s: midpoint quadrature of dl x r / r^3
        L, s = 0.3, 0.02
        P0 = np.array([s, 0.0, 0.05])
        werr = []
        for n in p["ladder"]:
            z = -L + (np.arange(n) + 0.5) * 2 * L / n
            r = P0[None] - np.stack([np.zeros(n), np.zeros(n), z], 1)
            dl = np.array([0.0, 0.0, 2 * L / n])
            dB = MU0 * I / (4 * math.pi) * np.cross(dl, r) / np.linalg.norm(r, axis=1)[:, None] ** 3
            By = dB.sum(0)[1]
            t2, t1 = math.atan((L - P0[2]) / s), math.atan((-L - P0[2]) / s)
            ref = MU0 * I / (4 * math.pi * s) * (math.sin(t2) - math.sin(t1))
            werr.append(abs(By / ref - 1))
        res.compare("finite wire B (midpoint rule)", ref * (1 + werr[-1]), ref, tol=1e-4, unit="T",
                    source="Griffiths eq. 5.37: B = mu0 I/(4 pi s) (sin t2 - sin t1)",
                    how=f"{p['ladder'][-1]} midpoint pieces")
        res.value("wire_observed_order", observed_order(werr)[-1], "", "midpoint rule, pieces doubled")
        # (b) loop as an n-gon (pp.em.statics.biot_savart), on and off the axis, against elliptic integrals
        pts = np.array([[0, 0, 0], [0, 0, 0.03], [0.02, 0.0, 0.01], [0.03, 0.02, -0.02], [0.08, 0.0, 0.0]])
        Bref = S.loop_field(pts, R, I)
        lerr = []
        for n in p["ladder"]:
            st, en = S.circle_segments(R, n)
            B = S.biot_savart(pts, st, en, I)
            lerr.append(float(np.max(np.linalg.norm(B - Bref, axis=1) / np.linalg.norm(Bref, axis=1))))
        res.compare("loop B, 5 points on and off axis (worst)", 1 + lerr[-1], 1.0, tol=1e-3, unit="|B|/|B_ref|",
                    source="Jackson sec. 5.5: loop field with complete elliptic integrals (pp.em.statics.loop_field)",
                    how=f"inscribed {p['ladder'][-1]}-gon")
        lo = observed_order(lerr)
        res.value("loop_observed_order", lo[-1], "", "inscribed polygon, sides doubled")
        res.check("polygon loop converges at second order", abs(lo[-1] - 2) < 0.2, value=lo[-1])
        res.compare("loop centre B", S.biot_savart(np.zeros((1, 3)), *S.circle_segments(R, p["ladder"][-1]), I)[0, 2],
                    MU0 * I / (2 * R), tol=1e-3, unit="T", source="Griffiths ex. 5.6: B = mu0 I / (2 R)")
        # (c) helical solenoid on its axis against the current sheet
        Ls, N = p["length"], p["turns"]
        st, en = S.helix_segments(R, Ls, N, n_per_turn=48)
        zs = np.linspace(-0.75 * Ls, 0.75 * Ls, 61)
        ax_pts = np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], 1)
        Bh = S.biot_savart(ax_pts, st, en, I)[:, 2]
        Bs = S.finite_solenoid_on_axis_B(zs, R, Ls, N, I)
        res.compare("solenoid centre B (helix)", Bh[30], Bs[30], tol=2e-3, unit="T",
                    source="Griffiths prob. 5.11: mu0 n I / 2 (cos a2 - cos a1), current sheet",
                    how=f"{N}-turn helix, 48 segments per turn")
        res.compare("solenoid end B (helix)", float(np.interp(Ls / 2, zs, Bh)), float(np.interp(Ls / 2, zs, Bs)),
                    tol=1e-2, unit="T", source="Griffiths prob. 5.11 at z = L/2 (about half the centre value)")
        res.compare("long-solenoid limit mu0 n I", Bs[30], MU0 * N / Ls * I, tol=0.05, kind="limit", unit="T",
                    source="Ampere's law for an infinite solenoid (Griffiths ex. 5.9); L/R = "
                           f"{Ls / R:g} gives a finite-length deficit")
        from pinneapple_physics.maxwell_checks import gauss_magnetic

        g = np.linspace(-0.02, 0.02, 21)
        G = np.stack(np.meshgrid(g, g, g + 0.1, indexing="ij"), -1)
        Bg = S.biot_savart(G, *S.circle_segments(R, 128), I)
        ck = gauss_magnetic({"x": g, "y": g, "z": g + 0.1}, {"Bx": Bg[..., 0], "By": Bg[..., 1], "Bz": Bg[..., 2]})
        res.check("div B = 0 near the loop (pp.maxwell.gauss_magnetic)", ck.passed, value=ck.value, limit=ck.limit,
                  source="pinneapple_physics.maxwell_checks", detail=ck.detail)
        # figure
        fig, ax = P.figure(3, width=4.0)
        ax[0].loglog(p["ladder"], werr, "o-", color=P.COMPUTED, lw=2, ms=6, label="wire, midpoint")
        ax[0].loglog(p["ladder"], lerr, "s-", color=P.THIRD, lw=2, ms=6, label="loop, polygon")
        ax[0].set(xlabel="pieces", ylabel="relative error", title="Biot-Savart quadrature")
        ax[0].legend(fontsize=8)
        ax[1].plot(zs / Ls, Bh * 1e3, color=P.COMPUTED, lw=2, label=f"{N}-turn helix")
        ax[1].plot(zs / Ls, Bs * 1e3, "--", color=P.REFERENCE, lw=1.5, label="current sheet (closed form)")
        ax[1].axvspan(-0.5, 0.5, color=P.GRID, alpha=0.4)
        ax[1].set(xlabel="z / L", ylabel="B_z on axis (mT)", title="finite solenoid")
        ax[1].legend(fontsize=8)
        x = np.linspace(-2.5 * R, 2.5 * R, 41)
        z = np.linspace(-0.8 * Ls, 0.8 * Ls, 61)
        Xg, Zg = np.meshgrid(x, z, indexing="ij")
        Gp = np.stack([Xg, np.zeros_like(Xg), Zg], -1)
        st2, en2 = S.helix_segments(R, Ls, min(N, 100), n_per_turn=36)
        Bf = S.biot_savart(Gp, st2, en2, I * N / min(N, 100))
        ax[2].contourf(x * 1e3, z * 1e3, np.log10(np.linalg.norm(Bf, axis=-1)).T, 30, cmap=P.SEQUENTIAL)
        ax[2].streamplot(x * 1e3, z * 1e3, Bf[..., 0].T, Bf[..., 2].T, color="white", density=1.1, linewidth=0.7)
        ax[2].plot([R * 1e3] * 2, [-Ls / 2 * 1e3, Ls / 2 * 1e3], color=P.REFERENCE, lw=2)
        ax[2].plot([-R * 1e3] * 2, [-Ls / 2 * 1e3, Ls / 2 * 1e3], color=P.REFERENCE, lw=2)
        ax[2].set(xlabel="x (mm)", ylabel="z (mm)", title="log10 |B| and field lines, plane y = 0")
        fig.tight_layout()
        res.figure("biot_savart", fig)
        res.data("solenoid_axis", z=zs, B_helix=Bh, B_sheet=Bs)
        res.data("convergence", pieces=p["ladder"], wire=werr, loop=lerr)


# ===================================================================================== Ampere
@register
class AmpereLaw(Experiment):
    name = "em.ampere_law"
    title = "Ampere's law"
    area = AREA
    topic = "magnetostatics"
    dims = ("2D", "3D")
    description = ("The circulation of B around closed paths equals mu0 times the current they enclose. 2-D: "
                   "B from a finite-volume solution of lap A_z = -mu0 J_z for a rectangular conductor, integrated "
                   "around a circle, a square and a path that encloses nothing. 3-D: B of a square loop "
                   "(Biot-Savart) around paths that link it once, twice (a path winding twice) and not at all.")
    references = ("Griffiths, Introduction to Electrodynamics, 4th ed., sec. 5.3", "Jackson, Classical "
                  "Electrodynamics, 3rd ed., sec. 5.3")
    params = {"I": 10.0, "cells": [40, 80, 160], "n_path": 400}
    quick = {"cells": [20, 40, 80], "n_path": 200}

    def _run(self, res: Result, dim: str, p):
        I = p["I"]  # noqa: E741  (I: current)
        if dim == "2D":
            from scipy.interpolate import RegularGridInterpolator

            def circulation(n, paths):
                Lb = 1.0
                x = np.linspace(-Lb, Lb, 2 * n + 1)
                g = TensorGrid((x, x))
                X, Y = g.mesh()
                cond = (np.abs(X - 0.05) <= 0.15 + 1e-9) & (np.abs(Y + 0.02) <= 0.05 + 1e-9)
                area = 0.3 * 0.1
                J = np.where(cond, I / area, 0.0)
                fixed = np.zeros(g.shape, bool)
                fixed[0] = fixed[-1] = fixed[:, 0] = fixed[:, -1] = True
                A = MU0 * solve_potential(g, fixed, 0.0, rho=J)
                # discrete current actually imposed (the control volumes of the conductor nodes)
                vol = np.outer(g.dual_widths(0), g.dual_widths(1))
                I_disc = float(np.sum(J * vol))
                Bx, By = np.gradient(A, x, x, edge_order=2)[1], -np.gradient(A, x, x, edge_order=2)[0]
                fx = RegularGridInterpolator((x, x), Bx, method="cubic")
                fy = RegularGridInterpolator((x, x), By, method="cubic")
                out = []
                for path in paths:
                    t = np.linspace(0, 1, p["n_path"], endpoint=False)
                    pts, dl = path(t)
                    out.append(float(np.sum(fx(pts) * dl[:, 0] + fy(pts) * dl[:, 1])))
                return out, I_disc, (X, Y, A, Bx, By)

            def circle(c, r):
                def f(t):
                    a = 2 * np.pi * t
                    pts = np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)
                    dl = np.stack([-r * np.sin(a), r * np.cos(a)], 1) * (2 * np.pi / len(t))
                    return pts, dl
                return f

            def square(c, h):
                def f(t):
                    s = 4 * t
                    k = np.floor(s).astype(int)
                    u = s - k
                    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]]) + c
                    pts = corners[k] + (corners[k + 1] - corners[k]) * u[:, None]
                    dl = (corners[k + 1] - corners[k]) * (4 / len(t))
                    return pts, dl
                return f

            paths = [circle((0.0, 0.0), 0.4), square((0.03, 0.0), 0.3), circle((0.55, 0.45), 0.2)]
            rows = []
            for n in p["cells"]:
                circ, I_disc, fields = circulation(n, paths)
                rows.append((circ, I_disc))
            circ, I_disc = rows[-1]
            res.value("imposed_current", I_disc, "A", "sum of J over the conductor's control volumes")
            res.compare("circle around the conductor / mu0", circ[0] / MU0, I_disc, tol=2e-3, unit="A",
                        source="Ampere's law: oint B.dl = mu0 I_enc (Griffiths eq. 5.55)",
                        how=f"finite volumes {2 * p['cells'][-1]}^2 cells, cubic interpolation, {p['n_path']} points")
            res.compare("square around the conductor / mu0", circ[1] / MU0, I_disc, tol=2e-3, unit="A",
                        source="Ampere's law (the path shape does not matter)")
            res.compare("circle enclosing nothing / mu0", circ[2] / MU0, 0.0, tol=0.0, atol=2e-3 * I, unit="A",
                        source="Ampere's law: I_enc = 0", kind="conservation")
            errs = [abs(r[0][0] / MU0 / r[1] - 1) for r in rows]
            res.check("circulation error decreases with the grid", errs[-1] < errs[0],
                      detail=", ".join(f"{2 * n}^2: {e:.1e}" for n, e in zip(p["cells"], errs, strict=True)))
            X, Y, A, Bx, By = fields
            fig, ax = P.figure(2, width=4.5)
            Bm = np.hypot(Bx, By)
            cf = ax[0].contourf(X, Y, Bm * 1e6, 30, cmap=P.SEQUENTIAL)
            ax[0].contour(X, Y, A, 20, colors="white", linewidths=0.4)
            for path, col in zip(paths, (P.COMPUTED, P.THIRD, P.REFERENCE), strict=True):
                pts, _ = path(np.linspace(0, 1, 200, endpoint=False))
                ax[0].plot(pts[:, 0], pts[:, 1], color=col, lw=1.8)
            ax[0].add_patch(__import__("matplotlib").patches.Rectangle((-0.1, -0.07), 0.3, 0.1, fc="none",
                                                                         ec=P.INK, lw=1.2))
            ax[0].set(xlabel="x (m)", ylabel="y (m)", title="|B| (uT) and field lines; three paths", xlim=(-0.8, 0.8),
                      ylim=(-0.8, 0.8))
            ax[0].set_aspect("equal")
            fig.colorbar(cf, ax=ax[0], shrink=0.8)
            names = ["circle", "square", "outside"]
            vals = [c / MU0 for c in circ]
            ax[1].bar(names, vals, color=[P.COMPUTED, P.THIRD, P.REFERENCE], width=0.55)
            ax[1].axhline(I_disc, color=P.INK2, ls="--", lw=1.2, label="I enclosed")
            ax[1].set(ylabel="circulation / mu0 (A)", title="Ampere's law, 2-D")
            ax[1].legend(fontsize=8)
            fig.tight_layout()
            res.figure("ampere_2d", fig)
            res.data("circulation", cells=p["cells"], rel_error=errs, values_over_mu0=vals)
        else:
            a = 0.1
            corners = np.array([[a, a, 0], [-a, a, 0], [-a, -a, 0], [a, -a, 0], [a, a, 0]])
            st, en = S.polyline_segments(corners)

            def circ(path, n):
                """Trapezoidal rule on a closed parametrised path (spectrally accurate for a smooth periodic
                integrand), with the exact tangent."""
                t = 2 * np.pi * np.arange(n) / n
                pts, dpts = path(t)
                return float(np.sum(S.biot_savart(pts, st, en, I) * dpts) * 2 * np.pi / n)

            n = p["n_path"]
            # the side x = a carries the current towards +y; each path circles it right-handed about +y
            def link1(t):
                return (np.stack([a + 0.04 * np.sin(t), 0 * t, 0.04 * np.cos(t)], 1),
                        np.stack([0.04 * np.cos(t), 0 * t, -0.04 * np.sin(t)], 1))

            def link1b(t):              # a tilted ellipse, still linking once
                return (np.stack([a + 0.06 * np.sin(t), 0.03 * np.sin(t), 0.02 * np.cos(t) + 0.01], 1),
                        np.stack([0.06 * np.cos(t), 0.03 * np.cos(t), -0.02 * np.sin(t)], 1))

            def link2(t):               # winds twice around the wire before closing
                r = 0.03 + 0.01 * np.cos(t)
                dr = -0.01 * np.sin(t)
                return (np.stack([a + r * np.sin(2 * t), 0.01 * np.sin(t), r * np.cos(2 * t)], 1),
                        np.stack([dr * np.sin(2 * t) + 2 * r * np.cos(2 * t), 0.01 * np.cos(t),
                                  dr * np.cos(2 * t) - 2 * r * np.sin(2 * t)], 1))

            def unlinked(t):            # a circle above the loop's interior: links nothing
                return (np.stack([0.05 * np.cos(t), 0.05 * np.sin(t), 0 * t + 0.05], 1),
                        np.stack([-0.05 * np.sin(t), 0.05 * np.cos(t), 0 * t], 1))

            vals = [circ(f, n) / MU0 for f in (link1, link1b, link2, unlinked)]
            src = "Ampere's law: oint B.dl = mu0 I x (linking number) (Jackson sec. 5.3)"
            res.compare("circle around one side", vals[0], I, tol=1e-6, unit="A", source=src,
                        how=f"Biot-Savart of the square loop, {n}-point periodic trapezoid with the exact tangent")
            res.compare("tilted ellipse, linking once", vals[1], I, tol=1e-6, unit="A", source=src)
            res.compare("path winding twice", vals[2], 2 * I, tol=1e-6, unit="A", source=src)
            res.compare("path not linked (a circle inside the loop, above it)", vals[3], 0.0, tol=0.0, atol=1e-9 * I,
                        unit="A", source=src, kind="conservation")
            fig, ax = P.figure(2, width=4.4, three_d=(0,))
            a3 = ax[0]
            a3.plot(*corners.T, color=P.INK, lw=2.5)
            tt = np.linspace(0, 2 * np.pi, 301)
            for path, col in zip((link1, link1b, link2, unlinked), (P.COMPUTED, P.THIRD, P.SERIES[6], P.REFERENCE),
                                 strict=True):
                pp_ = path(tt)[0]
                a3.plot(*pp_.T, color=col, lw=1.6)
            a3.set(title="square loop (black) and four paths", xlabel="x", ylabel="y", zlabel="z")
            ax[1].bar(["once", "ellipse", "twice", "unlinked"], vals, color=[P.COMPUTED, P.THIRD, P.SERIES[6],
                                                                              P.REFERENCE], width=0.55)
            ax[1].axhline(I, color=P.INK2, ls="--", lw=1)
            ax[1].axhline(2 * I, color=P.INK2, ls=":", lw=1)
            ax[1].set(ylabel="circulation / mu0 (A)", title="linking number x I")
            fig.tight_layout()
            res.figure("ampere_3d", fig)
            res.data("circulation", paths=["once", "ellipse", "twice", "unlinked"], values_over_mu0=vals)


# ===================================================================================== Hall
def hall_bar(L: float, W: float, beta: float, n: int, V: float = 1.0, nx: int | None = None):
    """Potential in a Hall bar ``[0, L] x [0, W]`` (contacts at x = 0 and x = L held at 0 and ``V``; insulating
    edges y = 0, W) for Hall parameter ``beta = mu B``: Laplace inside, ``d phi/dy = beta d phi/dx`` on the
    edges (``J_y = 0`` with the Drude conductivity tensor ``sigma0/(1+beta^2) [[1, beta], [-beta, 1]]``).
    Second-order finite differences with ghost nodes (``n`` cells across the width, about as many per unit length
    along it, or ``nx`` along it: a fixed ``nx`` makes the solution a smooth function of ``L``); returns
    ``(x, y, phi)``."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    nx = nx or max(int(round(n * L / W)), 2)
    ny = n
    x = np.linspace(0, L, nx + 1)
    y = np.linspace(0, W, ny + 1)
    hx, hy = x[1] - x[0], y[1] - y[0]
    cx, cy = 1 / hx ** 2, 1 / hy ** 2
    g = beta * hy / hx              # ghost: phi(i, j +- 1) = phi(i, j -+ 1) +- beta (hy/hx) (phi(i+1) - phi(i-1))
    idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    rows, cols, vals = [], [], []
    b = np.zeros(idx.size)

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for i in range(nx + 1):
        for j in range(ny + 1):
            k = idx[i, j]
            if i == 0 or i == nx:
                add(k, k, 1.0)
                b[k] = 0.0 if i == 0 else V
                continue
            add(k, k, -2 * (cx + cy))
            add(k, idx[i - 1, j], cx)
            add(k, idx[i + 1, j], cx)
            for jj, sgn in ((j + 1, 1), (j - 1, -1)):
                if 0 <= jj <= ny:
                    add(k, idx[i, jj], cy)
                else:
                    add(k, idx[i, j - sgn], cy)
                    add(k, idx[i + 1, j], sgn * g * cy)
                    add(k, idx[i - 1, j], -sgn * g * cy)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(idx.size, idx.size))
    phi = spla.spsolve(A.tocsc(), b).reshape(nx + 1, ny + 1)
    return x, y, phi


@register
class HallEffect(Experiment):
    name = "em.hall_effect"
    title = "Hall effect"
    area = AREA
    topic = "transport in a magnetic field"
    dims = ("2D",)
    description = ("A current through a semiconductor bar in a perpendicular field B: the 2-D potential with the "
                   "Drude conductivity tensor (insulating edges carry the Hall condition), the Hall voltage at "
                   "mid-length and the Hall coefficient against 1/(n q); the contacts short the Hall field in a "
                   "short bar, shown by sweeping the length.")
    references = ("Ashcroft & Mermin, Solid State Physics (1976), ch. 1, eqs. 1.15-1.19 (Drude Hall coefficient)",
                  "Popovic, Hall Effect Devices, 2nd ed. (2004), sec. 4.2 (geometrical factor)")
    params = {"n_carriers": 1e22, "mobility": 0.14, "carrier": "electron", "B": 0.5, "L": 5e-3, "W": 1e-3,
              "t": 1e-4, "V": 1.0, "cells": [10, 20, 40]}
    quick = {"cells": [8, 16, 32]}

    def _run(self, res: Result, dim: str, p):
        q = -E_CHARGE if p["carrier"] == "electron" else E_CHARGE
        mu = math.copysign(p["mobility"], q)               # signed mobility q tau / m
        beta = mu * p["B"]
        sigma0 = p["n_carriers"] * q * mu
        L, W, t, V = p["L"], p["W"], p["t"], p["V"]
        RH, VH = [], []
        for n in p["cells"]:
            x, y, phi = hall_bar(L, W, beta, n, V)
            i = len(x) // 2
            Ex = -np.gradient(phi, x, axis=0)[i]
            Ey = -np.gradient(phi, y, axis=1)[i]
            Jx = sigma0 / (1 + beta ** 2) * (Ex + beta * Ey)
            I = float(np.trapezoid(Jx, y) * t)  # noqa: E741  (I: current)
            vh = float(phi[i, -1] - phi[i, 0])
            VH.append(vh)
            RH.append(-vh * t / (I * p["B"]))       # R_H = E_y / (J_x B), E_y = -V_H / W
        res.value("current", I, "A", "integral of J_x across the mid-section x thickness")
        res.value("hall_voltage", VH[-1], "V", "phi(top) - phi(bottom) at mid-length")
        res.value("hall_angle_deg", math.degrees(math.atan(abs(beta))), "deg", "atan(mu B)")
        res.compare("Hall coefficient R_H = E_y / (J_x B)", RH[-1], 1 / (p["n_carriers"] * q), tol=2e-3,
                    unit="m^3/C", source="Drude: R_H = 1/(n q) (Ashcroft & Mermin eq. 1.15)",
                    how=f"finite differences, {p['cells'][-1]} cells across, L/W = {L / W:g}")
        herr = [abs(v * p["n_carriers"] * q - 1) for v in RH]
        res.check("R_H error falls with the grid", herr[-1] < herr[0],
                  detail=", ".join(f"{k} cells: {e:.1e}" for k, e in zip(p["cells"], herr, strict=True)))
        res.check("sign of V_H gives the carrier sign", (RH[-1] < 0) == (q < 0),
                  detail=f"{p['carrier']}s: R_H {'<' if q < 0 else '>'} 0")
        # geometric factor: short bars
        ratios = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        G = []
        for r in ratios:
            x, y, phi = hall_bar(r * W, W, beta, p["cells"][-1], V)
            i = len(x) // 2
            Ex = -np.gradient(phi, x, axis=0)[i]
            Ey = -np.gradient(phi, y, axis=1)[i]
            I = float(np.trapezoid(sigma0 / (1 + beta ** 2) * (Ex + beta * Ey), y) * t)  # noqa: E741  (I: current)
            G.append(-(phi[i, -1] - phi[i, 0]) * t / (I * p["B"]) * p["n_carriers"] * q)
        res.check("contacts short the Hall voltage in short bars (G rises with L/W)",
                  all(b > a for a, b in zip(G, G[1:], strict=False)) and G[0] < 0.9,
                  detail=", ".join(f"L/W={r:g}: G={g:.3f}" for r, g in zip(ratios, G, strict=True)))
        res.data("geometric_factor", L_over_W=ratios, G=G)
        x, y, phi = hall_bar(L, W, beta, p["cells"][-1], V)
        fig, ax = P.figure(2, width=4.6)
        X, Y = np.meshgrid(x, y, indexing="ij")
        cf = ax[0].contourf(X * 1e3, Y * 1e3, phi, 30, cmap=P.SEQUENTIAL)
        ax[0].contour(X * 1e3, Y * 1e3, phi, 21, colors="white", linewidths=0.6)
        ax[0].set(xlabel="x (mm)", ylabel="y (mm)", title="potential (V); equipotentials tilt by the Hall angle",
                  ylim=(0, W * 1e3))
        fig.colorbar(cf, ax=ax[0], orientation="horizontal", pad=0.25, aspect=30)
        ax[1].plot(ratios, G, "o-", color=P.COMPUTED, lw=2, ms=6, label="computed V_H / V_H(long bar)")
        ax[1].axhline(1.0, color=P.REFERENCE, ls="--", lw=1.5, label="1/(n q) (long bar)")
        ax[1].set(xlabel="L / W", ylabel="geometrical factor G", title="contacts short the Hall field")
        ax[1].legend(fontsize=8)
        err = abs(RH[-1] * p["n_carriers"] * q - 1)
        P.badge(ax[1], f"R_H vs 1/(nq): {100 * err:.3f} %", err < 2e-3)
        fig.tight_layout()
        res.figure("hall_effect", fig)
