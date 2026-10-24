"""Induction and circuits, academic level: Faraday's law with a magnet falling through a loop (3-D + time) and
the RC, RL and series RLC circuits (lumped, 0-D in space).

The falling magnet is a uniformly magnetised cylinder, i.e. a sheet of bound current ``K = M`` on its side
(Griffiths sec. 6.2), discretised into polygonal rings. At every time step the flux through the loop is the
circulation of their vector potential around the loop (Stokes: ``Phi = oint A . dl``, exact A of straight
filaments, ``pp.em.statics.vector_potential``), and the EMF is ``-dPhi/dt`` by central differences: a 3-D field,
in time; the animation shows B (``pp.em.statics.biot_savart``). The
reference is Maxwell's mutual-inductance formula of coaxial circles (``pp.em.statics.mutual_inductance_coaxial_loops``)
summed over the rings, and the kinematics of free fall.
"""
from __future__ import annotations

import math

import numpy as np

from pinneapple_simulation.electromagnetics import statics as S
from pinneapple_simulation.electromagnetics.constants import MU0

from .. import plots as P
from ..core import Experiment, Result
from ..instruments import linear_fit
from ..registry import register

AREA = "physics.electromagnetism"
G0 = 9.80665                            # m/s^2, standard gravity (CGPM 1901)


def magnet_rings(a: float, length: float, Br: float, n_rings: int, n_seg: int, z_center: float = 0.0):
    """Segments and per-segment currents of a cylindrical magnet (radius ``a``, ``length``, remanence ``Br``)
    as ``n_rings`` polygonal rings of the bound surface current ``K = Br / mu0`` (A/m)."""
    zr = z_center - length / 2 + (np.arange(n_rings) + 0.5) * length / n_rings
    I_ring = Br / MU0 * length / n_rings
    st, en = [], []
    for z in zr:
        s, e = S.circle_segments(a, n_seg, z=z)
        st.append(s)
        en.append(e)
    return np.concatenate(st), np.concatenate(en), I_ring, zr


def loop_flux(segs, I_ring, R: float, n_path: int = 64) -> float:
    """Flux through a circular loop of radius ``R`` in the plane z = 0 by Stokes' theorem, ``oint A . dl``
    (:func:`pinneapple_simulation.electromagnetics.statics.flux_through_path`). A surface integral of B_z would
    be singular when the magnet's current sheet crosses the loop's disc; the line integral is not."""
    ps, pe = S.circle_segments(R, n_path)
    return S.flux_through_path(ps, pe, segs[0], segs[1], I_ring)


@register
class FaradayInduction(Experiment):
    name = "em.faraday_induction"
    title = "Faraday's law: a magnet falling through a loop"
    area = AREA
    topic = "electromagnetic induction"
    dims = ("3D+t",)
    description = ("A neodymium-like cylinder magnet falls along the axis of a wire loop. The 3-D field of the "
                   "magnet is computed at every instant (vector potential and Biot-Savart of its bound surface "
                   "current), the flux through the loop by Stokes' theorem, and the EMF = -dPhi/dt. Checked against Maxwell's "
                   "coaxial-circle formula, the free-fall time at which the EMF changes sign, and the net flux "
                   "change over the whole fall; the field in time is the animation.")
    references = ("Griffiths, Introduction to Electrodynamics, 4th ed., sec. 6.2 (bound currents) and 7.2 "
                  "(Faraday's law)", "J. C. Maxwell, A Treatise on Electricity and Magnetism (1873), vol. 2, "
                  "art. 701 (mutual inductance of coaxial circles)", "Smythe, Static and Dynamic Electricity, "
                  "3rd ed., sec. 8.06")
    params = {"a": 6e-3, "length": 20e-3, "Br": 1.2, "R": 15e-3, "turns": 1, "z0": 0.08, "n_rings": 16,
              "n_seg": 48, "n_t": 160, "frames": 60}
    quick = {"n_rings": 12, "n_seg": 48, "n_t": 80, "frames": 0}

    def _run(self, res: Result, dim: str, p):
        a, ln, Br, R, N, z0 = p["a"], p["length"], p["Br"], p["R"], p["turns"], p["z0"]
        T = math.sqrt(4 * z0 / G0)                       # from +z0 to -z0 (released at rest at +z0)
        t = np.linspace(0, T, p["n_t"])
        dt = t[1] - t[0]
        zc = lambda tt: z0 - 0.5 * G0 * tt ** 2           # noqa: E731
        st, en, I_ring, zr0 = magnet_rings(a, ln, Br, p["n_rings"], p["n_seg"])

        def flux_at(tt):
            dz = zc(tt)
            return N * loop_flux((st + [0, 0, dz], en + [0, 0, dz]), I_ring, R)

        def flux_ref(tt):
            dz = zc(tt)
            return N * I_ring * sum(S.mutual_inductance_coaxial_loops(a, R, z + dz) for z in zr0)

        phi = np.array([flux_at(tt) for tt in t])
        emf = -np.array([(flux_at(tt + 0.5 * dt) - flux_at(tt - 0.5 * dt)) / dt for tt in t])
        phi_ref = np.array([flux_ref(tt) for tt in t])
        h = 1e-7
        emf_ref = -np.array([(flux_ref(tt + h) - flux_ref(tt - h)) / (2 * h) for tt in t])
        k = int(np.argmax(np.abs(emf)))
        res.value("peak_emf", emf[k], "V", "largest |EMF| over the fall (second lobe: the magnet is faster)")
        res.value("max_flux", phi.max(), "Wb", "flux through the loop when the magnet is centred in it")
        res.compare("peak EMF", emf[k], emf_ref[k], tol=1e-2, unit="V",
                    source="Maxwell (1873) art. 701: M of coaxial circles (pp.em.statics.mutual_inductance_coaxial_loops), "
                           "summed over the rings, differentiated in time",
                    how=f"oint A.dl of {p['n_rings']} x {p['n_seg']}-gon rings around a 64-gon loop")
        res.compare("flux at the centre crossing", phi.max(), phi_ref.max(), tol=1e-2, unit="Wb",
                    source="Maxwell (1873) art. 701, summed over the rings")
        # EMF sign change: when the magnet's centre crosses the loop plane, t = sqrt(2 z0 / g)
        sgn = np.nonzero(np.diff(np.sign(emf)) != 0)[0]
        j = sgn[np.argmin(np.abs(t[sgn] - T / 2))]
        tz = t[j] - emf[j] * (t[j + 1] - t[j]) / (emf[j + 1] - emf[j])
        res.compare("time of the EMF sign change", tz, math.sqrt(2 * z0 / G0), tol=5e-3, unit="s",
                    source="free fall from rest: z0 = g t^2 / 2 (the flux is largest with the magnet centred)")
        net = float(np.trapezoid(emf, t))
        res.check("net EMF integral = -(Phi_end - Phi_start)", abs(net + (phi[-1] - phi[0])) <= 2e-2 * phi.max(),
                  value=abs(net + (phi[-1] - phi[0])) / phi.max(), limit=2e-2,
                  detail="integral of EMF dt over the fall against the flux change, relative to the peak flux")
        growing = np.gradient(phi, t) > 0.05 * np.max(np.abs(np.gradient(phi, t)))
        res.check("Lenz: EMF < 0 wherever the flux grows", bool(np.all(emf[growing] < 0)),
                  detail=f"{int(growing.sum())} instants with dPhi/dt > 5 % of its peak")
        # div B = 0 around the magnet at mid-fall
        from pinneapple_physics.maxwell_checks import gauss_magnetic

        g = np.linspace(-0.012, 0.012, 17)
        gz = np.linspace(0.016, 0.04, 17)
        G = np.stack(np.meshgrid(g, g, gz, indexing="ij"), -1)
        Bg = S.biot_savart(G, st, en, I_ring)
        ck = gauss_magnetic({"x": g, "y": g, "z": gz}, {"Bx": Bg[..., 0], "By": Bg[..., 1], "Bz": Bg[..., 2]})
        res.check("div B = 0 above the magnet (pp.maxwell.gauss_magnetic)", ck.passed, value=ck.value,
                  limit=ck.limit, source="pinneapple_physics.maxwell_checks", detail=ck.detail)
        m = Br / MU0 * math.pi * a * a * ln
        res.value("magnetic_moment", m, "A m^2", "M pi a^2 L with M = Br / mu0")
        res.data("trace", t=t, z_magnet=zc(t), flux=phi, emf=emf, flux_ref=phi_ref, emf_ref=emf_ref)
        # figure: the traces
        fig, ax = P.figure(2, width=4.6)
        ax[0].plot(t * 1e3, phi * 1e6, color=P.COMPUTED, lw=2, label="oint A.dl (computed)")
        ax[0].plot(t * 1e3, phi_ref * 1e6, "--", color=P.REFERENCE, lw=1.5, label="Maxwell's formula")
        ax[0].set(xlabel="t (ms)", ylabel="flux (uWb)", title="flux through the loop")
        ax[0].legend(fontsize=8)
        ax[1].plot(t * 1e3, emf * 1e3, color=P.COMPUTED, lw=2, label="-dPhi/dt (computed)")
        ax[1].plot(t * 1e3, emf_ref * 1e3, "--", color=P.REFERENCE, lw=1.5, label="Maxwell's formula, d/dt")
        ax[1].axvline(math.sqrt(2 * z0 / G0) * 1e3, color=P.INK2, lw=1, ls=":", label="centre crosses the loop")
        ax[1].set(xlabel="t (ms)", ylabel="EMF (mV)", title="induced EMF (Faraday)")
        ax[1].legend(fontsize=8)
        err = abs(emf[k] / emf_ref[k] - 1)
        P.badge(ax[1], f"peak {emf[k] * 1e3:.2f} mV, {100 * err:.2f} % vs ref", err < 1e-2, loc="lower left")
        fig.tight_layout()
        res.figure("faraday_traces", fig)
        if p["frames"]:
            res.movie("faraday_3d_t", self._frames(st, en, I_ring, R, a, ln, t, zc, emf, p["frames"]), fps=12)

    @staticmethod
    def _frames(st, en, I_ring, R, a, ln, t, zc, emf, n_frames):
        x = np.linspace(-0.04, 0.04, 61)
        z = np.linspace(-0.09, 0.09, 121)
        X, Z = np.meshgrid(x, z, indexing="ij")
        G = np.stack([X, np.zeros_like(X), Z], -1)
        frames = []
        idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
        lev = np.linspace(-5, 0.3, 30)
        for k in idx:
            dz = zc(t[k])
            B = S.biot_savart(G, st + [0, 0, dz], en + [0, 0, dz], I_ring)
            fig, ax = P.figure(2, width=4.2, height=4.4)
            ax[0].contourf(X * 1e3, Z * 1e3, np.clip(np.log10(np.linalg.norm(B, axis=-1)), -5, 0.3), lev,
                           cmap=P.SEQUENTIAL)
            ax[0].streamplot(x * 1e3, z * 1e3, B[..., 0].T, B[..., 2].T, color="white", density=1.0, linewidth=0.6)
            ax[0].add_patch(__import__("matplotlib").patches.Rectangle(((-a) * 1e3, (dz - ln / 2) * 1e3), 2 * a * 1e3,
                                                                         ln * 1e3, fc=P.REFERENCE, ec=P.INK, lw=1))
            ax[0].plot([-R * 1e3, R * 1e3], [0, 0], "o", color=P.SERIES[5], ms=7)
            ax[0].set(xlabel="x (mm)", ylabel="z (mm)", title=f"log10 |B| (T), plane y = 0, t = {t[k] * 1e3:5.1f} ms")
            ax[0].set_aspect("equal")
            ax[1].plot(t * 1e3, emf * 1e3, color=P.GRID, lw=1.5)
            ax[1].plot(t[:k + 1] * 1e3, emf[:k + 1] * 1e3, color=P.COMPUTED, lw=2)
            ax[1].plot(t[k] * 1e3, emf[k] * 1e3, "o", color=P.COMPUTED, ms=6)
            ax[1].set(xlabel="t (ms)", ylabel="EMF (mV)", title="EMF in the loop")
            fig.tight_layout()
            frames.append(P.to_rgb(fig))
            P.close(fig)
        return frames


# ===================================================================================== circuits
@register
class Circuits(Experiment):
    name = "em.circuits_rc_rl_rlc"
    title = "RC, RL and RLC circuits"
    area = AREA
    topic = "circuits"
    dims = ("0D",)
    description = ("Step responses of RC, RL and series RLC circuits integrated numerically (Kirchhoff's laws "
                   "as ODEs, scipy solve_ivp). The time constants, the damped frequency and the decay rate are "
                   "measured from the waveforms (fits and zero crossings) and compared with the closed forms; "
                   "in the free RLC decay the energy dissipated in R equals the energy first stored in C.")
    references = ("Nilsson & Riedel, Electric Circuits, 10th ed., ch. 7 (RL, RC) and ch. 8 (RLC)",
                  "Griffiths, Introduction to Electrodynamics, 4th ed., sec. 7.1-7.2")
    params = {"V0": 5.0, "R": 1e3, "C": 1e-6, "L": 0.1, "R_rlc": 50.0}
    quick = {}

    def _run(self, res: Result, dim: str, p):
        from scipy.integrate import solve_ivp

        V0, R, C, L = p["V0"], p["R"], p["C"], p["L"]
        kw = dict(rtol=1e-10, atol=1e-13, method="DOP853", dense_output=True)
        # RC: dV/dt = (V0 - V) / (R C)
        tau = R * C
        rc = solve_ivp(lambda t, v: (V0 - v) / (R * C), (0, 6 * tau), [0.0], **kw)
        t = np.linspace(0.05 * tau, 3 * tau, 200)
        v = rc.sol(t)[0]
        f = linear_fit(t, np.log(1 - v / V0))
        res.compare("RC time constant (fit of ln(1 - V/V0))", -1 / f.slope, tau, tol=1e-6, unit="s",
                    source="tau = R C (Nilsson & Riedel sec. 7.2)")
        # RL: dI/dt = (V0 - R I) / L
        tl = L / R
        rl = solve_ivp(lambda t, i: (V0 - R * i) / L, (0, 6 * tl), [0.0], **kw)
        t2 = np.linspace(0.05 * tl, 3 * tl, 200)
        i2 = rl.sol(t2)[0]
        f2 = linear_fit(t2, np.log(1 - i2 * R / V0))
        res.compare("RL time constant", -1 / f2.slope, tl, tol=1e-6, unit="s", source="tau = L / R (Nilsson & Riedel sec. 7.1)")
        res.compare("RL final current", rl.sol(6 * tl)[0] * (1 - math.exp(-6)) ** -1, V0 / R, tol=1e-6, unit="A",
                    source="I(inf) = V0 / R; the value at 6 tau divided by (1 - e^-6)")
        # series RLC step: L q'' + R q' + q / C = V0
        Rr = p["R_rlc"]
        alpha = Rr / (2 * L)
        w0 = 1 / math.sqrt(L * C)
        wd = math.sqrt(w0 * w0 - alpha * alpha)
        Td = 2 * math.pi / wd
        sol = solve_ivp(lambda t, y: [y[1], (V0 - Rr * y[1] - y[0] / C) / L], (0, 8 * Td), [0.0, 0.0], **kw)
        tt = np.linspace(0, 8 * Td, 40001)
        vc = sol.sol(tt)[0] / C
        u = vc - V0
        zc = np.nonzero(np.diff(np.sign(u)) != 0)[0]
        tz = tt[zc] - u[zc] * (tt[zc + 1] - tt[zc]) / (u[zc + 1] - u[zc])
        wd_m = math.pi / np.mean(np.diff(tz))
        # successive extrema of u: ratio exp(-alpha Td / 2)
        du = np.gradient(u, tt)
        ext = np.nonzero(np.diff(np.sign(du)) != 0)[0]
        amps = np.abs(u[ext])
        alpha_m = -np.mean(np.diff(np.log(amps))) / np.mean(np.diff(tt[ext]))
        res.compare("RLC damped angular frequency (zero crossings)", wd_m, wd, tol=1e-4, unit="rad/s",
                    source="omega_d = sqrt(1/(L C) - (R/2L)^2) (Nilsson & Riedel sec. 8.3)")
        res.compare("RLC decay rate (log decrement of extrema)", alpha_m, alpha, tol=1e-3, unit="1/s",
                    source="alpha = R / (2 L) (Nilsson & Riedel sec. 8.3)")
        res.value("quality_factor", w0 / (2 * alpha), "", "Q = omega0 / (2 alpha) from the circuit values")
        # free decay energy balance: C charged to V0, no source
        fd = solve_ivp(lambda t, y: [y[1], (-Rr * y[1] - y[0] / C) / L], (0, 40 / alpha), [C * V0, 0.0], **kw)
        tf = np.linspace(0, 40 / alpha, 200001)
        i = fd.sol(tf)[1]
        Ediss = float(np.trapezoid(Rr * i * i, tf))
        res.compare("energy dissipated in R (free decay)", Ediss, 0.5 * C * V0 ** 2, tol=1e-6, unit="J",
                    kind="conservation", source="energy conservation: 1/2 C V0^2 stored at t = 0")
        res.data("rc", t=t, v=v)
        res.data("rlc", t=tt[::40], v=vc[::40])
        fig, ax = P.figure(3, width=3.9)
        tr = np.linspace(0, 6 * tau, 300)
        ax[0].plot(tr / tau, rc.sol(tr)[0], color=P.COMPUTED, lw=2, label="RC: V_C (ODE)")
        ax[0].plot(tr / tau, V0 * (1 - np.exp(-tr / tau)), "--", color=P.REFERENCE, lw=1.5, label="V0 (1 - e^-t/RC)")
        tl_ = np.linspace(0, 6 * tl, 300)
        ax[0].plot(tl_ / tl, rl.sol(tl_)[0] * R, color=P.THIRD, lw=2, ls="-.", label="RL: R I (ODE)")
        ax[0].set(xlabel="t / tau", ylabel="V", title="first-order step responses")
        ax[0].legend(fontsize=7.5)
        ax[1].plot(tt * 1e3, vc, color=P.COMPUTED, lw=1.6, label="V_C (ODE)")
        env = V0 + V0 * np.exp(-alpha * tt) * math.hypot(1, alpha / wd)
        ax[1].plot(tt * 1e3, env, "--", color=P.REFERENCE, lw=1.2, label="envelope e^(-alpha t)")
        ax[1].plot(tt * 1e3, 2 * V0 - env, "--", color=P.REFERENCE, lw=1.2)
        ax[1].set(xlabel="t (ms)", ylabel="V_C (V)", title=f"series RLC step, Q = {w0 / (2 * alpha):.1f}")
        ax[1].legend(fontsize=7.5)
        e_stored = 0.5 * C * (fd.sol(tf)[0] / C) ** 2 + 0.5 * L * i * i
        from scipy.integrate import cumulative_trapezoid

        ax[2].plot(tf * 1e3, e_stored * 1e6, color=P.COMPUTED, lw=2, label="stored in L and C")
        ax[2].plot(tf * 1e3, cumulative_trapezoid(Rr * i * i, tf, initial=0) * 1e6, color=P.REFERENCE, lw=2,
                   label="dissipated in R")
        ax[2].set(xlabel="t (ms)", ylabel="energy (uJ)", title="free decay: energy balance", xlim=(0, 8 / alpha * 1e3))
        ax[2].legend(fontsize=7.5)
        fig.tight_layout()
        res.figure("circuits", fig)
