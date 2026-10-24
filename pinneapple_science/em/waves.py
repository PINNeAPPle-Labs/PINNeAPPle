"""Waves and optics, academic level: radiation of an oscillating dipole, a plane wave in a lossy medium,
reflection and refraction at an interface (Fresnel and Snell), waveguide modes, cavity resonances and Mie
scattering.

The wave experiments run the FDTD solver of ``pp.em`` (:mod:`pinneapple_simulation.electromagnetics.fdtd`) or a
finite-difference eigen-solver and measure the quantity the way a laboratory would (a fitted decay rate, a phase
gradient, a spectral peak), then compare it with the closed form of :mod:`pinneapple_simulation.electromagnetics.waves`
or :mod:`.mie`, on a refinement ladder whose observed order is reported.
"""
from __future__ import annotations

import math

import numpy as np

from pinneapple_simulation.electromagnetics import waves as W
from pinneapple_simulation.electromagnetics.constants import C0

from .. import plots as P
from ..core import Experiment, Result, observed_order
from ..instruments import linear_fit
from ..numerics import helmholtz_eigs, spectral_peaks
from ..registry import register

AREA = "physics.electromagnetism"


# ===================================================================================== dipole
def dipole_cartesian(X, Y, Z, Il: complex, omega: float):
    """Phasor E and H (Cartesian components) of a z-directed Hertzian dipole at the origin
    (:func:`pinneapple_simulation.electromagnetics.waves.hertzian_dipole_fields`)."""
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    th = np.arccos(Z / r)
    ph = np.arctan2(Y, X)
    Er, Et, Hp = W.hertzian_dipole_fields(r, th, Il, omega)
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    E = (Er * st * cp + Et * ct * cp, Er * st * sp + Et * ct * sp, Er * ct - Et * st)
    H = (-Hp * sp, Hp * cp, 0 * Hp)
    return E, H


@register
class DipoleRadiation(Experiment):
    name = "em.dipole_radiation"
    title = "Oscillating dipole and radiation"
    area = AREA
    topic = "radiation"
    dims = ("3D", "3D+t")
    description = ("The complete near- and far-zone fields of an oscillating (Hertzian) dipole. 3-D: the "
                   "time-averaged Poynting flux integrated over spheres from kr = 0.3 (reactive near zone) to "
                   "kr = 30 is the same radiated power, eta k^2 |I l|^2 / (12 pi); the directivity is 1.5. 3D+t: "
                   "the real fields in space and time, the energy through a sphere over one period, and Faraday "
                   "and Ampere-Maxwell residuals of the 4-D field (pp.maxwell).")
    references = ("Balanis, Antenna Theory, 3rd ed., sec. 4.2 (eqs. 4-8, 4-16)", "Jackson, Classical "
                  "Electrodynamics, 3rd ed., sec. 9.2")
    params = {"f": 1e9, "Il": 1e-3, "n_theta": 32, "n_t": 48, "frames": 60}
    quick = {"n_t": 24, "frames": 0}

    def _run(self, res: Result, dim: str, p):
        w = 2 * math.pi * p["f"]
        k = w / C0
        lam = C0 / p["f"]
        Il = p["Il"]
        Pref = W.hertzian_dipole_power(Il, w)
        x, gw = np.polynomial.legendre.leggauss(p["n_theta"])
        th = np.arccos(x)
        if dim == "3D":
            radii = np.array([0.3, 1.0, 3.0, 30.0]) / k
            Pr = []
            for r in radii:
                Er, Et, Hp = W.hertzian_dipole_fields(np.full_like(th, r), th, Il, w)
                Sr = 0.5 * np.real(Et * np.conj(Hp))
                Pr.append(float(2 * math.pi * r * r * np.sum(Sr * gw)))
            for r, pw in zip(radii, Pr, strict=True):
                res.compare(f"radiated power through the sphere kr = {k * r:g}", pw, Pref, tol=1e-9, unit="W",
                            source="Balanis eq. 4-16: P = eta k^2 |I l|^2 / (12 pi)",
                            how=f"{p['n_theta']}-point Gauss-Legendre in theta of Re(E x H*)/2")
            # directivity from the far-zone intensity
            tq = np.linspace(0, np.pi, 1801)
            Er, Et, Hp = W.hertzian_dipole_fields(np.full_like(tq, 1e4 / k), tq, Il, w)
            U = 0.5 * np.real(Et * np.conj(Hp)) * (1e4 / k) ** 2
            D = 4 * math.pi * U.max() / Pr[-1]
            res.compare("directivity", D, 1.5, tol=1e-3, unit="", source="Balanis eq. 4-31: D0 = 3/2",
                        how="4 pi U_max / P, U on a 0.1 deg grid at kr = 1e4")
            # Maxwell residuals of the phasor fields on a 3-D grid off the origin
            from pinneapple_physics.maxwell_checks import maxwell_checks

            g = np.linspace(0.4 * lam, 0.9 * lam, 26)
            X, Y, Z = np.meshgrid(g, g - 0.25 * lam, g - 0.6 * lam, indexing="ij")
            (Ex, Ey, Ez), (Hx, Hy, Hz) = dipole_cartesian(X, Y, Z, Il, w)
            cks = maxwell_checks({"x": g, "y": g - 0.25 * lam, "z": g - 0.6 * lam},
                                 {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}, omega=w,
                                 laws=["gauss_electric", "gauss_magnetic", "faraday", "ampere_maxwell"])
            for c in cks:
                res.check(f"{c.name} (phasors, 26^3 grid)", c.passed, value=c.value, limit=c.limit,
                          source="pinneapple_physics.maxwell_checks", detail=c.detail)
            fig, ax = P.figure(2, width=4.4)
            kr = np.logspace(-1, 2, 200)
            Pk = []
            for r in kr / k:
                Er, Et, Hp = W.hertzian_dipole_fields(np.full_like(th, r), th, Il, w)
                Pk.append(2 * math.pi * r * r * np.sum(0.5 * np.real(Et * np.conj(Hp)) * gw))
                Pk[-1] = Pk[-1] if True else 0
            Qk = []
            for r in kr / k:
                Er, Et, Hp = W.hertzian_dipole_fields(np.full_like(th, r), th, Il, w)
                Qk.append(2 * math.pi * r * r * np.sum(0.5 * np.abs(np.imag(Et * np.conj(Hp))) * gw))
            ax[0].loglog(kr, np.asarray(Pk) / Pref, color=P.COMPUTED, lw=2, label="Re flux / P (radiated)")
            ax[0].loglog(kr, np.asarray(Qk) / Pref, color=P.REFERENCE, lw=2, label="|Im flux| / P (reactive)")
            ax[0].set(xlabel="k r", ylabel="power / P_rad", title="flux through spheres of radius r")
            ax[0].legend(fontsize=8)
            tt = np.linspace(0, 2 * np.pi, 361)
            Er, Et, Hp = W.hertzian_dipole_fields(np.full_like(tt, 1e4 / k), tt, Il, w)
            Uu = np.real(Et * np.conj(Hp))
            ax[1].remove()
            ax1 = fig.add_subplot(1, 2, 2, projection="polar")
            ax1.plot(tt, Uu / Uu.max(), color=P.COMPUTED, lw=2, label="computed U(theta)")
            ax1.plot(tt, np.sin(tt) ** 2, "--", color=P.REFERENCE, lw=1.5, label="sin^2 theta")
            ax1.set_theta_zero_location("N")
            ax1.set_title(f"radiation pattern, D = {D:.4f}", fontsize=10)
            ax1.legend(fontsize=7, loc="lower right")
            fig.tight_layout()
            res.figure("dipole_power", fig)
            res.data("flux_vs_kr", kr=kr, real=np.asarray(Pk) / Pref, reactive=np.asarray(Qk) / Pref)
        else:
            T = 1 / p["f"]
            t = np.arange(p["n_t"]) * T / p["n_t"]
            r = lam
            Er, Et, Hp = W.hertzian_dipole_fields(np.full_like(th, r), th, Il, w)
            ph = np.exp(1j * w * t)[:, None]
            Sr_t = np.real(Et[None] * ph) * np.real(Hp[None] * ph)     # instantaneous radial Poynting vector
            energy = float(np.sum(2 * math.pi * r * r * Sr_t @ gw) * T / p["n_t"])
            res.compare("energy through the sphere r = lambda in one period", energy, Pref * T, tol=1e-9, unit="J",
                        source="Balanis eq. 4-16: P T", how=f"real fields, {p['n_t']} instants x {p['n_theta']} Gauss "
                        "points: a 3D+t integral")
            # Maxwell residuals of the real 4-D field (x, y, z, t)
            from pinneapple_physics.maxwell_checks import maxwell_checks

            g = np.linspace(0.4 * lam, 0.9 * lam, 21)
            gy, gz = g - 0.25 * lam, g - 0.6 * lam
            X, Y, Z = np.meshgrid(g, gy, gz, indexing="ij")
            (Ex, Ey, Ez), (Hx, Hy, Hz) = dipole_cartesian(X, Y, Z, Il, w)
            tg = np.arange(4 * p["n_t"] // 3) * T / p["n_t"]
            e = np.exp(1j * w * tg)
            f4 = {n: np.real(F[..., None] * e) for n, F in (("Ex", Ex), ("Ey", Ey), ("Ez", Ez), ("Hx", Hx),
                                                            ("Hy", Hy), ("Hz", Hz))}
            cks = maxwell_checks({"x": g, "y": gy, "z": gz, "t": tg}, f4,
                                 laws=["gauss_electric", "gauss_magnetic", "faraday", "ampere_maxwell",
                                       "poynting_balance"])
            for c in cks:
                res.check(f"{c.name} (x, y, z, t grid 21^3 x {len(tg)})", c.passed, value=c.value, limit=c.limit,
                          source="pinneapple_physics.maxwell_checks", detail=c.detail)
            bad = dict(f4, Hx=1.3 * f4["Hx"], Hy=1.3 * f4["Hy"])
            ckb = maxwell_checks({"x": g, "y": gy, "z": gz, "t": tg}, bad, laws=["faraday", "ampere_maxwell"])
            res.check("the checks reject H scaled by 1.3", not any(c.passed for c in ckb),
                      source="pinneapple_physics.maxwell_checks",
                      detail="; ".join(f"{c.key} residual {c.value:.3g} > {c.limit:.3g}" for c in ckb))
            fig, ax = P.figure(1, width=5.0)
            Pt = 2 * math.pi * r * r * Sr_t @ gw
            ax[0].plot(t / T, Pt * 1e6, "o-", color=P.COMPUTED, lw=2, ms=4, label="instantaneous flux, r = lambda")
            ax[0].axhline(Pref * 1e6, color=P.REFERENCE, ls="--", lw=1.5, label="P (time average)")
            ax[0].set(xlabel="t / T", ylabel="power through the sphere (uW)", title="dipole: power in time")
            ax[0].legend(fontsize=8)
            fig.tight_layout()
            res.figure("dipole_time", fig)
            if p["frames"]:
                res.movie("dipole_3d_t", self._frames(Il, w, lam, p["frames"]), fps=15)

    @staticmethod
    def _frames(Il, w, lam, n):
        g = np.linspace(-2 * lam, 2 * lam, 161)
        X, Z = np.meshgrid(g, g, indexing="ij")
        Y = np.zeros_like(X) + 1e-9
        (Ex, Ey, Ez), _ = dipole_cartesian(X + 1e-9, Y, Z, Il, w)
        r = np.sqrt(X ** 2 + Z ** 2)
        frames = []
        for kk in range(n):
            ph = np.exp(1j * 2 * np.pi * kk / n)
            F = np.real(Ez * ph) * (r / lam) * (r > 0.15 * lam)
            v = np.nanmax(np.abs(F)) * 0.6
            fig, ax = P.figure(1, width=5.0, height=4.6)
            ax[0].pcolormesh(g / lam, g / lam, F.T, cmap=P.DIVERGING, vmin=-v, vmax=v, shading="auto")
            ax[0].plot([0, 0], [-0.05, 0.05], color=P.INK, lw=3)
            ax[0].set(xlabel="x / lambda", ylabel="z / lambda",
                      title=f"oscillating dipole: E_z r (plane y = 0), t = {kk / n:.2f} T")
            ax[0].set_aspect("equal")
            fig.tight_layout()
            frames.append(P.to_rgb(fig))
            P.close(fig)
        return frames


# ===================================================================================== lossy medium
@register
class LossyPlaneWave(Experiment):
    name = "em.lossy_plane_wave"
    title = "Plane wave in a lossy medium"
    area = AREA
    topic = "wave propagation"
    dims = ("1D",)
    description = ("A pulse enters a conducting dielectric (1-D FDTD with CPML). From the DFT of the field "
                   "inside it, the attenuation constant (slope of ln|E|) and the phase constant (slope of the "
                   "unwrapped phase) are fitted and compared with gamma = sqrt(j omega mu (sigma + j omega eps)); "
                   "the skin depth follows.")
    references = ("Pozar, Microwave Engineering, 4th ed., sec. 1.4 (eqs. 1.47-1.52)", "Taflove & Hagness, "
                  "Computational Electrodynamics, 3rd ed., 2005 (FDTD, CPML)")
    params = {"f0": 1e9, "eps_r": 4.0, "loss_tangent": 0.5, "cells_per_wavelength": [20, 40, 80]}
    quick = {"cells_per_wavelength": [40, 80]}

    def _run(self, res: Result, dim: str, p):
        from pinneapple_simulation.electromagnetics.constants import EPS0
        from pinneapple_simulation.electromagnetics.fdtd import FDTD, YeeGrid, modulated_gaussian
        from pinneapple_simulation.electromagnetics.media import Medium

        f0 = p["f0"]
        w0 = 2 * math.pi * f0
        sigma = p["loss_tangent"] * w0 * EPS0 * p["eps_r"]
        med = Medium(p["eps_r"], sigma=sigma)
        k = med.k(w0)
        alpha, beta = -float(np.imag(k)), float(np.real(k))
        lam0 = C0 / f0
        rows = []
        for ncl in p["cells_per_wavelength"]:
            dx = lam0 / ncl
            N = int(6 * lam0 / dx) + 60
            g = YeeGrid((N, 1, 1), (dx, dx, dx))
            s = FDTD(g, {"x": "cpml", "y": "periodic", "z": "periodic"}, npml=20)
            i0 = int(1.5 * lam0 / dx) + 25
            m = np.zeros(g.shape, bool)
            m[i0:] = True
            s.set_material(m, med)
            s.add_source("Ey", (25, 0, 0), modulated_gaussian(f0, 0.8 * f0), "soft")
            a0, a1 = i0 + int(0.25 * lam0 / dx), i0 + int(1.5 * lam0 / dx)
            mon = s.add_dft(["Ey"], [f0], (slice(a0, a1), slice(None), slice(None)))
            s.run_until(40 / f0)
            E = mon.phasor("Ey")[0].ravel()
            xx = (np.arange(a0, a1)) * dx
            fa = linear_fit(xx, np.log(np.abs(E)))
            fb = linear_fit(xx, np.unwrap(np.angle(E)))
            rows.append((-fa.slope, -fb.slope))
        a_m, b_m = rows[-1]
        res.compare("attenuation constant alpha", a_m, alpha, tol=5e-3, unit="Np/m",
                    source="Pozar eq. 1.49: gamma = alpha + j beta = j omega sqrt(mu eps (1 - j sigma/(omega eps)))",
                    how=f"1-D FDTD, {p['cells_per_wavelength'][-1]} cells per free-space wavelength, fit of ln|E| "
                        "over 1.25 wavelengths in the medium")
        res.compare("phase constant beta", b_m, beta, tol=5e-3, unit="rad/m", source="Pozar eq. 1.49",
                    how="fit of the unwrapped phase")
        res.value("skin_depth", 1 / a_m, "m", "1 / alpha (measured)")
        errs = [abs(a / alpha - 1) + abs(b / beta - 1) for a, b in rows]
        oo = observed_order(errs)
        res.value("observed_order", oo[-1] if oo else float("nan"), "", "of |alpha err| + |beta err|, cells doubled")
        res.check("error falls with the grid", errs[-1] < errs[0],
                  detail=", ".join(f"{n} cells/lambda: {e:.1e}" for n, e in zip(p["cells_per_wavelength"], errs,
                                                                                    strict=True)))
        fig, ax = P.figure(2, width=4.5)
        ax[0].semilogy(xx / lam0, np.abs(E) / np.abs(E[0]), color=P.COMPUTED, lw=2, label="|E| (FDTD, DFT at f0)")
        ax[0].semilogy(xx / lam0, np.exp(-alpha * (xx - xx[0])), "--", color=P.REFERENCE, lw=1.5,
                       label="exp(-alpha x), closed form")
        ax[0].set(xlabel="x / lambda0", ylabel="|E| / |E(x0)|", title=f"loss tangent {p['loss_tangent']}")
        ax[0].legend(fontsize=8)
        ax[1].loglog(p["cells_per_wavelength"], errs, "o-", color=P.COMPUTED, lw=2, ms=6)
        ax[1].set(xlabel="cells per wavelength", ylabel="|d alpha/alpha| + |d beta/beta|", title="convergence")
        P.badge(ax[1], f"order {oo[-1]:.2f}" if oo else "", errs[-1] < 5e-3, loc="upper right")
        fig.tight_layout()
        res.figure("lossy_plane_wave", fig)
        res.data("ladder", cells_per_wavelength=p["cells_per_wavelength"], alpha=[r[0] for r in rows],
                 beta=[r[1] for r in rows], alpha_ref=alpha, beta_ref=beta)


# ===================================================================================== Fresnel / Snell
def _oblique_run(ncl, n2, theta, pol, f0=1e9, with_medium=True, monitor_inside=False):
    """2-D FDTD with Bloch-periodic y: an oblique plane wave of angle ``theta`` (s: E_z, p: H_z) hits a
    half-space of index ``n2``. Returns the DFT phasors at f0 in front of the interface (and inside it)."""
    from pinneapple_simulation.electromagnetics.fdtd import FDTD, YeeGrid, modulated_gaussian
    from pinneapple_simulation.electromagnetics.media import Medium

    lam = C0 / f0
    ky = 2 * math.pi * f0 / C0 * math.sin(theta)
    dx = lam / ncl
    ny = 4
    N = int(4 * lam / dx) + 40
    g = YeeGrid((N, ny, 1), (dx, dx, dx))
    im = N // 2 - int(0.7 * lam / dx)
    s = FDTD(g, {"x": "cpml", "y": "bloch"}, npml=20, bloch_k={"y": ky})
    if with_medium:
        m = np.zeros(g.shape, bool)
        m[N // 2:] = True
        s.set_material(m, Medium(n2 ** 2))
    # with a fixed k_y, frequencies below f0 sin(theta) are evanescent along x (and near it the wave crawls):
    # keep the pulse's band above that, and run long enough for the narrower pulse to pass
    bw = min(0.15, 0.4 * (1 - math.sin(theta))) if theta > 0 else 0.15
    wv = modulated_gaussian(f0, bw * f0)
    comp = "Ez" if pol == "s" else "Ey"
    for j in range(ny):
        s.add_current(comp, (30, j, 0), wv, np.exp(-1j * ky * j * dx))
    c = "Ez" if pol == "s" else "Hz"
    mon = s.add_dft([c], [f0], (slice(im, im + 1), slice(0, 1), slice(None)))
    mon2 = None
    if monitor_inside:
        a0, a1 = N // 2 + int(0.2 * lam / dx), N // 2 + int(1.0 * lam / dx)
        mon2 = s.add_dft([c], [f0], (slice(a0, a1), slice(0, 1), slice(None)))
    s.run_until(9.0 / bw / f0)
    out = mon.phasor(c).ravel()[0]
    if mon2 is not None:
        xs = (np.arange(a0, a1)) * dx
        return out, xs, mon2.phasor(c)[0].ravel()
    return out


@register
class FresnelSnell(Experiment):
    name = "em.fresnel_snell"
    title = "Fresnel reflection and Snell's law"
    area = AREA
    topic = "optics"
    dims = ("2D",)
    description = ("An oblique plane wave on a dielectric half-space in a 2-D FDTD with Bloch-periodic sides. "
                   "The reflection coefficient for s and p polarisation is the ratio of two runs (with and "
                   "without the medium); at Brewster's angle p is not reflected; the transmitted angle comes from "
                   "the phase gradient of the field inside the medium, against Snell's law.")
    references = ("Born & Wolf, Principles of Optics, 7th ed., sec. 1.5", "Hecht, Optics, 5th ed., sec. 4.6",
                  "Taflove & Hagness, 3rd ed., sec. 13 (periodic structures, Bloch boundaries)")
    params = {"n2": 1.5, "angles_deg": [0.0, 30.0, 45.0, 60.0], "cells_per_wavelength": [20, 40]}
    quick = {"angles_deg": [0.0, 45.0], "cells_per_wavelength": [20, 30]}

    def _run(self, res: Result, dim: str, p):
        n2 = p["n2"]
        ncl = p["cells_per_wavelength"]
        thB = math.atan(n2)
        rows = []
        for ang in p["angles_deg"]:
            th = math.radians(ang)
            rs_ref, rp_ref, _, _ = W.fresnel(1.0, n2, th)
            rec = {"angle": ang}
            for pol, ref in (("s", rs_ref), ("p", rp_ref)):
                errs = []
                for n in ncl:
                    e1 = _oblique_run(n, n2, th, pol)
                    e0 = _oblique_run(n, n2, th, pol, with_medium=False)
                    r = abs((e1 - e0) / e0)
                    errs.append(abs(r - abs(ref)))
                rec[pol] = (r, abs(complex(ref)), errs)
            rows.append(rec)
        for rec in rows:
            for pol in ("s", "p"):
                r, ref, errs = rec[pol]
                res.compare(f"|r_{pol}| at {rec['angle']:g} deg", r, ref, tol=0.0, atol=5e-3, unit="",
                            source="Fresnel equations (Born & Wolf eqs. 1.5.20-21), pp.em.waves.fresnel",
                            how=f"2-D FDTD, {ncl[-1]} cells per wavelength, two runs")
        # Brewster: |r_p| ~ 0
        e1 = _oblique_run(ncl[-1], n2, thB, "p")
        e0 = _oblique_run(ncl[-1], n2, thB, "p", with_medium=False)
        rB = abs((e1 - e0) / e0)
        res.compare(f"|r_p| at Brewster's angle {math.degrees(thB):.2f} deg", rB, 0.0, tol=0.0, atol=5e-3, unit="",
                    source="Brewster: tan(theta_B) = n2 / n1 (Hecht eq. 4.65)", kind="closed_form")
        # Snell: transmitted angle from the phase gradient inside the medium
        snell = []
        for ang in [a for a in p["angles_deg"] if a > 0]:
            th = math.radians(ang)
            _, xs, Ein = _oblique_run(ncl[-1], n2, th, "s", monitor_inside=True)
            kx2 = -linear_fit(xs, np.unwrap(np.angle(Ein))).slope
            ky = 2 * math.pi / (C0 / 1e9) * math.sin(th)
            tt = math.degrees(math.atan2(ky, kx2))
            ref = math.degrees(math.asin(math.sin(th) / n2))
            snell.append((ang, tt, ref))
            res.compare(f"transmitted angle for {ang:g} deg incidence", tt, ref, tol=5e-3, unit="deg",
                        source="Snell's law n1 sin(t1) = n2 sin(t2)", how="atan(k_y / k_x) with k_x from the phase "
                        "gradient of the DFT field in the medium")
        fig, ax = P.figure(2, width=4.5)
        aa = np.linspace(0, 89.5, 300)
        rs, rp, _, _ = W.fresnel(1.0, n2, np.radians(aa))
        ax[0].plot(aa, np.abs(rs), color=P.COMPUTED, lw=1.5, label="|r_s| Fresnel")
        ax[0].plot(aa, np.abs(rp), color=P.REFERENCE, lw=1.5, label="|r_p| Fresnel")
        ax[0].plot([r["angle"] for r in rows], [r["s"][0] for r in rows], "o", color=P.COMPUTED, ms=8,
                   label="FDTD s")
        ax[0].plot([r["angle"] for r in rows] + [math.degrees(thB)], [r["p"][0] for r in rows] + [rB], "s",
                   color=P.REFERENCE, ms=8, label="FDTD p")
        ax[0].axvline(math.degrees(thB), color=P.INK2, ls=":", lw=1)
        ax[0].set(xlabel="incidence angle (deg)", ylabel="|r|", title=f"air -> n = {n2}")
        ax[0].legend(fontsize=7.5)
        ax[1].plot(aa, np.degrees(np.arcsin(np.sin(np.radians(aa)) / n2)), color=P.REFERENCE, lw=1.5,
                   label="Snell")
        ax[1].plot([s[0] for s in snell], [s[1] for s in snell], "o", color=P.COMPUTED, ms=8, label="FDTD phase gradient")
        ax[1].set(xlabel="incidence angle (deg)", ylabel="transmitted angle (deg)", title="refraction")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        res.figure("fresnel_snell", fig)
        res.data("reflection", angles=[r["angle"] for r in rows], rs=[r["s"][0] for r in rows],
                 rp=[r["p"][0] for r in rows], rs_ref=[r["s"][1] for r in rows], rp_ref=[r["p"][1] for r in rows])


# ===================================================================================== waveguide
@register
class WaveguideModes(Experiment):
    name = "em.waveguide_modes"
    title = "Rectangular waveguide modes"
    area = AREA
    topic = "guided waves"
    dims = ("2D",)
    description = ("Cut-off wavenumbers of the TE and TM modes of a WR-90 guide from a finite-difference "
                   "Helmholtz eigenproblem on its cross-section (Neumann for H_z, Dirichlet for E_z), refined "
                   "twice, against k_c = sqrt((m pi/a)^2 + (n pi/b)^2); the dispersion beta(f) follows.")
    references = ("Pozar, Microwave Engineering, 4th ed., sec. 3.3 (eqs. 3.84, 3.106)",)
    params = {"a": 22.86e-3, "b": 10.16e-3, "scale": [4, 8, 16], "n_modes": 5}
    quick = {"scale": [2, 4, 8]}

    def _run(self, res: Result, dim: str, p):
        a, b = p["a"], p["b"]
        nm = p["n_modes"]
        te_ref = sorted(math.hypot(m * math.pi / a, n * math.pi / b) for m in range(6) for n in range(6) if m + n > 0)[:nm]
        tm_ref = sorted(math.hypot(m * math.pi / a, n * math.pi / b) for m in range(1, 6) for n in range(1, 6))[:3]
        errs_te, errs_tm = [], []
        for s in p["scale"]:
            nx, ny = 9 * s, 4 * s          # a / b = 2.25 = 9 / 4: square cells
            te = helmholtz_eigs(nx, ny, a, b, "TE", nm)
            tm = helmholtz_eigs(nx, ny, a, b, "TM", 3)
            errs_te.append(np.abs(te / te_ref - 1))
            errs_tm.append(np.abs(tm / tm_ref - 1))
        names_te = []
        for kc in te_ref:
            mm = [(m, n) for m in range(6) for n in range(6) if m + n > 0 and
                  abs(math.hypot(m * math.pi / a, n * math.pi / b) - kc) < 1e-9]
            names_te.append(f"TE{mm[0][0]}{mm[0][1]}")
        for i, kc in enumerate(te_ref):
            res.compare(f"{names_te[i]} cut-off frequency", C0 * te[i] / (2 * math.pi), C0 * kc / (2 * math.pi),
                        tol=1e-3, unit="Hz", source="Pozar eq. 3.84: f_c = c/(2 pi) sqrt((m pi/a)^2 + (n pi/b)^2)",
                        how=f"FD eigenproblem {9 * p['scale'][-1]} x {4 * p['scale'][-1]} cells")
        for i, kc in enumerate(tm_ref):
            res.compare(f"TM mode {i + 1} cut-off frequency", C0 * tm[i] / (2 * math.pi), C0 * kc / (2 * math.pi),
                        tol=1e-3, unit="Hz", source="Pozar eq. 3.106")
        o = observed_order([float(e.max()) for e in errs_te])
        res.value("observed_order_TE", o[-1], "", "worst TE mode, cells doubled")
        res.check("second-order convergence", abs(o[-1] - 2) < 0.2, value=o[-1])
        kc, modes = helmholtz_eigs(9 * p["scale"][-1], 4 * p["scale"][-1], a, b, "TE", 4, modes=True)
        fig, ax = P.figure(3, width=3.8)
        for i in range(2):
            ax[i].imshow(modes[i].T, origin="lower", extent=(0, a * 1e3, 0, b * 1e3), cmap=P.DIVERGING,
                         vmin=-np.abs(modes[i]).max(), vmax=np.abs(modes[i]).max())
            ax[i].set(xlabel="x (mm)", ylabel="y (mm)", title=f"{names_te[i]}: H_z, f_c = {C0 * kc[i] / 2 / math.pi / 1e9:.3f} GHz")
            ax[i].grid(False)
        f = np.linspace(5e9, 20e9, 300)
        for i, kc_ in enumerate(te[:3]):
            fc = C0 * kc_ / (2 * math.pi)
            beta = np.where(f > fc, 2 * math.pi / C0 * np.sqrt(np.maximum(f ** 2 - fc ** 2, 0)), np.nan)
            ax[2].plot(f / 1e9, beta, color=P.SERIES[i], lw=2, label=names_te[i])
        ax[2].plot(f / 1e9, 2 * math.pi * f / C0, ":", color=P.INK2, lw=1, label="light line")
        ax[2].set(xlabel="f (GHz)", ylabel="beta (rad/m)", title="dispersion (computed f_c)")
        ax[2].legend(fontsize=8)
        fig.tight_layout()
        res.figure("waveguide_modes", fig)
        res.data("ladder", scale=p["scale"], te_err=[e.tolist() for e in errs_te], tm_err=[e.tolist() for e in errs_tm])


# ===================================================================================== cavity
@register
class CavityResonance(Experiment):
    name = "em.cavity_resonance"
    title = "Resonant cavity"
    area = AREA
    topic = "resonators"
    dims = ("2D", "3D")
    description = ("A PEC cavity rung by a short pulse (FDTD). The spectrum of a probe signal shows the "
                   "resonances; each peak is matched to the nearest mode of the closed form and its frequency "
                   "error measured. 2-D: TM_mn modes of a rectangle; 3-D: TE/TM_mnp modes of a box.")
    references = ("Pozar, Microwave Engineering, 4th ed., sec. 6.3 (eq. 6.40)", "Taflove & Hagness, 3rd ed., "
                  "sec. 4.6 (numerical dispersion)")
    params = {"a": 1.0, "b": 0.7, "d": 0.5, "cells": [40, 80], "cells_3d": [16, 32], "steps": 16000,
              "steps_3d": 8000}
    quick = {"cells": [20, 40], "cells_3d": [16, 32], "steps": 12000, "steps_3d": 6000}

    def _run(self, res: Result, dim: str, p):
        from pinneapple_simulation.electromagnetics.fdtd import FDTD, YeeGrid, gaussian_pulse

        a, b, d = p["a"], p["b"], p["d"]
        if dim == "2D":
            ref = sorted(C0 / 2 * math.hypot(m / a, n / b) for m in range(1, 5) for n in range(1, 5))[:5]
            lab = []
            for f in ref:
                lab.append(next(f"TM{m}{n}" for m in range(1, 5) for n in range(1, 5)
                                if abs(C0 / 2 * math.hypot(m / a, n / b) - f) < 1e-6))
        else:
            cand = {}
            for m in range(3):
                for n in range(3):
                    for q in range(3):
                        if (m > 0) + (n > 0) + (q > 0) >= 2:
                            cand.setdefault(round(W.rect_cavity_resonance(m, n, q, a, b, d), 3), f"({m}{n}{q})")
            ref = sorted(cand)[:4]
            lab = [cand[f] for f in ref]
        errs, peaks_last = [], None
        for nc in (p["cells"] if dim == "2D" else p["cells_3d"]):
            dx = a / nc
            if dim == "2D":
                g = YeeGrid((nc + 1, int(round(b / dx)) + 1, 1), (dx, b / round(b / dx), dx))
                comp, src, prb = "Ez", (nc // 3, g.shape[1] // 5, 0), (4 * nc // 5, 2 * g.shape[1] // 3, 0)
                steps = p["steps"]
            else:
                g = YeeGrid((nc + 1, int(round(b / dx)) + 1, int(round(d / dx)) + 1),
                            (dx, b / round(b / dx), d / round(d / dx)))
                comp = "Ey"
                src = (nc // 3, g.shape[1] // 3, g.shape[2] // 3)
                prb = (2 * nc // 3 + 1, 2 * g.shape[1] // 3, 3 * g.shape[2] // 4)
                steps = p["steps_3d"]
            s = FDTD(g)
            s.add_source(comp, src, gaussian_pulse(20 * g.dt, 6 * g.dt), "soft")
            if dim == "3D":
                s.add_source("Ex", (nc // 4, g.shape[1] // 2, g.shape[2] // 3), gaussian_pulse(20 * g.dt, 6 * g.dt))
                s.add_source("Ez", (nc // 2, g.shape[1] // 4, g.shape[2] // 2), gaussian_pulse(20 * g.dt, 6 * g.dt))
            if dim == "2D":
                probes = [s.add_probe(comp, prb), s.add_probe(comp, (nc // 7 + 1, g.shape[1] // 2 + 1, 0))]
            else:                       # one probe per component: every mode has at least one of them
                probes = [s.add_probe(c, (2 * nc // 3 + 1, 2 * g.shape[1] // 3, 3 * g.shape[2] // 4 - 1))
                          for c in ("Ex", "Ey", "Ez")]
            s.run(steps)
            sig = sum(np.asarray(q.values) / np.abs(q.values).max() for q in probes)
            pk = spectral_peaks(sig, g.dt, 1.15 * ref[-1], n_peaks=12, min_rel=1e-3)
            found = [pk[np.argmin(np.abs(pk - f))] if len(pk) else np.nan for f in ref]
            errs.append([abs(fp / f - 1) for fp, f in zip(found, ref, strict=True)])
            peaks_last, sig_last, dt_last = found, sig, g.dt
        for f, fp, lb in zip(ref, peaks_last, lab, strict=True):
            res.compare(f"{lb} resonance", fp, f, tol=2e-3, unit="Hz",
                        source="Pozar eq. 6.40: f = c/2 sqrt((m/a)^2 + (n/b)^2 + (l/d)^2)",
                        how="nearest spectral peak of the FDTD ring-down (Hann window, parabolic interpolation)")
        worst = [max(e) for e in errs]
        oo = observed_order(worst)
        res.value("observed_order", oo[-1] if oo else float("nan"), "", "worst mode, cells doubled")
        res.check("frequency error falls with the grid (numerical dispersion ~ h^2)", worst[-1] < worst[0],
                  detail=", ".join(f"{e:.1e}" for e in worst))
        fig, ax = P.figure(1, width=6.4)
        nfft = 8 * len(sig_last)
        fr = np.fft.rfftfreq(nfft, dt_last)
        spc = np.abs(np.fft.rfft(sig_last * np.hanning(len(sig_last)), nfft))
        m = fr < 1.15 * ref[-1]
        ax[0].semilogy(fr[m] / 1e6, spc[m] / spc[m].max(), color=P.COMPUTED, lw=1.5, label="FDTD probe spectrum")
        for f, lb in zip(ref, lab, strict=True):
            ax[0].axvline(f / 1e6, color=P.REFERENCE, ls="--", lw=1)
            ax[0].text(f / 1e6, 1.3, lb, rotation=90, fontsize=7, color=P.INK2, ha="center", va="bottom")
        ax[0].set(xlabel="f (MHz)", ylabel="normalised spectrum", title=f"{dim} PEC cavity: ring-down spectrum",
                  ylim=(1e-5, 30))
        ax[0].legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        res.figure(f"cavity_{dim.lower()}", fig)
        res.data("modes", labels=lab, f_ref=ref, f_fdtd=peaks_last, errors_per_level=errs)


# ===================================================================================== Mie
@register
class MieScattering(Experiment):
    name = "em.mie_scattering"
    title = "Mie scattering by a sphere"
    area = AREA
    topic = "optics"
    dims = ("3D",)
    description = ("Scattering and extinction efficiencies of a dielectric sphere from the Mie series "
                   "(pp.em.mie): against the published BHMIE sample of Bohren & Huffman, the Rayleigh law for "
                   "small spheres, the optical theorem (extinction from the forward amplitude) and the "
                   "extinction paradox Q_ext -> 2 for large spheres.")
    references = ("Bohren & Huffman, Absorption and Scattering of Light by Small Particles (1983), sec. 4.4 and "
                  "app. A", "van de Hulst, Light Scattering by Small Particles (1957), sec. 8.22 (Q_ext -> 2)")
    params = {"m": 1.33, "x_max": 30.0}
    quick = {}

    def _run(self, res: Result, dim: str, p):
        from pinneapple_simulation.electromagnetics.mie import mie, rayleigh_qsca

        r = mie(1.55, 2 * math.pi * 0.525 / 0.6328)
        src = "Bohren & Huffman (1983) app. A, BHMIE sample output (m = 1.55, x = 5.213)"
        res.compare("Q_sca, BHMIE sample", r.qsca, 3.10543, tol=2e-5, unit="", source=src, kind="published")
        res.compare("Q_back, BHMIE sample", r.qback, 2.92534, tol=2e-5, unit="", source=src, kind="published")
        xs = 0.02
        rs = mie(p["m"], xs)
        res.compare(f"Q_sca at x = {xs} (Rayleigh)", rs.qsca, rayleigh_qsca(p["m"], xs), tol=1e-3, kind="limit",
                    unit="", source="Rayleigh: Q_sca = 8/3 x^4 |(m^2 - 1)/(m^2 + 2)|^2 (B&H eq. 5.8)")
        # optical theorem: Q_ext = 4 / x^2 Re S(0) (B&H eq. 4.76)
        ra = mie(1.5 - 0.02j, 8.0)
        S1, _ = ra.amplitudes([0.0])
        res.compare("optical theorem: 4 Re S(0) / x^2 (absorbing sphere)", 4 / 64.0 * S1.real[0], ra.qext, tol=1e-10,
                    kind="conservation", unit="", source="B&H eq. 4.76 (optical theorem)")
        big = mie(1.33 - 0.01j, 3000.0)
        res.compare("Q_ext at x = 3000 (absorbing)", big.qext, 2.0, tol=1e-2, kind="limit", unit="",
                    source="van de Hulst sec. 8.22: Q_ext -> 2 (extinction paradox); correction ~ x^(-2/3)")
        x = np.linspace(0.05, p["x_max"], 600)
        q = np.array([mie(p["m"], v).qext for v in x])
        res.value("first_extinction_peak_x", float(x[np.argmax(q * (x < 10))]), "",
                  f"size parameter of the first maximum of Q_ext for m = {p['m']}")
        fig, ax = P.figure(2, width=4.5)
        ax[0].plot(x, q, color=P.COMPUTED, lw=1.5, label=f"Q_ext, m = {p['m']}")
        ax[0].axhline(2.0, color=P.REFERENCE, ls="--", lw=1.2, label="large-sphere limit 2")
        ax[0].set(xlabel="size parameter x = 2 pi a / lambda", ylabel="Q_ext", title="Mie extinction")
        ax[0].legend(fontsize=8)
        xx = np.logspace(-2, 0, 40)
        ax[1].loglog(xx, [mie(p["m"], v).qsca for v in xx], "o", color=P.COMPUTED, ms=4, label="Mie series")
        ax[1].loglog(xx, [rayleigh_qsca(p["m"], v) for v in xx], "--", color=P.REFERENCE, lw=1.5, label="Rayleigh x^4")
        ax[1].set(xlabel="x", ylabel="Q_sca", title="small spheres: Rayleigh limit")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        res.figure("mie", fig)
        res.data("extinction", x=x[::4], qext=q[::4])
