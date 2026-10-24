"""Electromagnetism, research level: a configurable FDTD study in 3-D + time with mesh convergence and the
observed order, Monte Carlo uncertainty propagation, an inverse problem (permittivity from a measured reflection
spectrum) and a surrogate with its hold-out error.

Each experiment exposes its configuration as parameters and reports, besides the comparison with a reference,
a convergence study (Richardson extrapolation and Roache's grid convergence index,
:mod:`pinneapple_analysis.verification.convergence`) or an uncertainty (Monte Carlo, Jacobian covariance,
held-out error).
"""
from __future__ import annotations

import math

import numpy as np

from pinneapple_simulation.electromagnetics import waves as W
from pinneapple_simulation.electromagnetics.constants import C0, EPS0

from .. import plots as P
from ..core import Experiment, Result
from ..registry import register
from .fdtd_study import FDTDConfig, run_fdtd

AREA = "physics.electromagnetism"


# ===================================================================================== FDTD 3D+t
@register
class FDTDStudy3D(Experiment):
    name = "em.research_fdtd_3d"
    title = "FDTD in 3-D + time: configurable sources and materials, mesh convergence"
    area = AREA
    topic = "computational electromagnetics"
    level = "research"
    dims = ("3D+t",)
    description = ("A 3-D FDTD study configured by a plain dict (domain, cells per wavelength, sources, "
                   "materials). The default runs a Hertzian dipole in vacuum on a refinement ladder: the radiated "
                   "power (near-to-far-field transform) against the closed form, with the observed order, the "
                   "Richardson-extrapolated value and the GCI; the Maxwell residuals of the field in x, y, z and t "
                   "(pp.maxwell); then the configured case (the dipole beside a dielectric sphere) on the same ladder: "
                   "no closed form exists, so it reports its extrapolated value, observed order and GCI against a "
                   "declared accuracy target. The 3-D field in time is the animation and an "
                   "interactive three.js viewer.")
    references = ("Taflove & Hagness, Computational Electrodynamics, 3rd ed. (2005), ch. 3-4, 7, 8",
                  "Balanis, Antenna Theory, 3rd ed., eq. 4-16 (Hertzian dipole power)",
                  "Roache, ASME J. Fluids Eng. 116, 405 (1994) (GCI)")
    params = {"f0": 1e9, "size": 2.4, "ladder": [10, 14, 20], "periods": 12.0,
              "materials": [{"shape": "sphere", "center": [0.45, 0.0, 0.0], "radius": 0.2, "eps_r": 4.0}],
              "sources": [{"kind": "dipole", "component": "Ez", "at": [0.0, 0.0, 0.0]}],
              "snapshots": 48, "viewer": True, "target_gci": 0.05}
    quick = {"snapshots": 4, "viewer": False}

    def _run(self, res: Result, dim: str, p):
        from pinneapple_analysis.verification.convergence import richardson_extrapolate
        from pinneapple_physics.maxwell_checks import maxwell_checks

        lad = p["ladder"]
        base = dict(f0=p["f0"], size=p["size"], periods=p["periods"], sources=p["sources"])
        vac = []
        for ncl in lad:
            run = run_fdtd(FDTDConfig.from_dict({**base, "cells_per_wavelength": ncl}), snapshots=2,
                           maxwell_block=False)
            Pref = W.hertzian_dipole_power(run.current_moment, 2 * math.pi * p["f0"])
            vac.append((ncl, run.power, Pref, run))
        errs = [abs(pw / pr - 1) for _, pw, pr, _ in vac]
        r21, r32 = lad[2] / lad[1], lad[1] / lad[0]
        rich = richardson_extrapolate(vac[0][1] / vac[0][2], vac[1][1] / vac[1][2], vac[2][1] / vac[2][2],
                                      (r21, r32))
        res.compare("radiated power, dipole in vacuum (finest grid)", vac[-1][1], vac[-1][2], tol=2e-2, unit="W",
                    source="Balanis eq. 4-16 with the current moment the grid actually drove",
                    how=f"near-to-far-field transform at f0, {lad[-1]} cells per wavelength")
        res.compare("P / P_closed_form, Richardson-extrapolated", float(rich.extrapolated_value), 1.0, tol=1e-2,
                    unit="", source="Balanis eq. 4-16", kind="closed_form",
                    how=f"three-grid Richardson over {lad} cells per wavelength")
        res.value("observed_order_vacuum", rich.observed_order, "", "Roache three-grid observed order of P/P_ref")
        res.value("gci_fine_vacuum", rich.gci_fine, "", "grid convergence index of the finest grid (Fs = 1.25)")
        for (ncl, _pw, _pr, _), e in zip(vac, errs, strict=True):
            res.value(f"power_error_{ncl}cpw", e, "", "|P_FDTD / P_closed - 1|")
        res.check("error falls with the grid", errs[-1] < errs[0], detail=", ".join(f"{e:.2e}" for e in errs))
        # far-field pattern of the finest vacuum run against sin^2
        th = np.linspace(0.2, math.pi - 0.2, 40)
        pat = vac[-1][3].pattern(th)
        pat = pat / pat.max()
        perr = float(np.max(np.abs(pat - np.sin(th) ** 2)))
        res.compare("far-field pattern max |U/Umax - sin^2|", perr, 0.0, tol=0.0, atol=2e-2, unit="",
                    source="Hertzian dipole pattern sin^2(theta) (Balanis eq. 4-26a)")
        # the configured case: materials on the same ladder
        conf = []
        for ncl in lad:
            run = run_fdtd(FDTDConfig.from_dict({**base, "cells_per_wavelength": ncl, "materials": p["materials"]}),
                           snapshots=p["snapshots"] if ncl == lad[-1] else 2, maxwell_block=ncl == lad[-1])
            conf.append(run.power / W.hertzian_dipole_power(run.current_moment, 2 * math.pi * p["f0"]))
            last = run
        rc = richardson_extrapolate(conf[0], conf[1], conf[2], (r21, r32))
        res.value("power_ratio_configured", conf[-1], "", "P with the materials / P of the same dipole in vacuum "
                  "(closed form), finest grid")
        res.value("power_ratio_configured_extrapolated", float(rc.extrapolated_value), "",
                  "Richardson extrapolation of the three grids (Roache)")
        res.value("observed_order_configured", rc.observed_order, "", "Roache observed order (the sphere is a "
                  "staircase with fill-fraction averaging: first order is expected)")
        res.value("gci_fine_configured", rc.gci_fine, "", "grid convergence index of the finest grid (Fs = 1.25)")
        res.check(f"configured case: GCI of the finest grid within the target {p['target_gci']:g}",
                  rc.gci_fine <= p["target_gci"], value=rc.gci_fine, limit=p["target_gci"],
                  detail=f"no closed form with the sphere: the reference is the study's own extrapolation, "
                         f"{float(rc.extrapolated_value):.4f}; observed order {rc.observed_order:.2f}")
        res.check("configured case: ratios converge monotonically", (conf[0] - conf[1]) * (conf[1] - conf[2]) > 0,
                  detail=", ".join(f"{n}: {c:.4f}" for n, c in zip(lad, conf, strict=True)))
        # Maxwell residuals of the 4-D field block (finest configured run)
        cks = maxwell_checks(last.volume_grid, last.volume, laws=["gauss_electric", "gauss_magnetic", "faraday",
                                                                  "ampere_maxwell"])
        for c in cks:
            res.check(f"{c.name} (x, y, z, t block 7^3 x 24, FDTD)", c.passed, value=c.value, limit=c.limit,
                      source="pinneapple_physics.maxwell_checks", detail=c.detail)
        e = last.energy
        res.check("discrete energy bounded after the source (CPML absorbs, no growth)",
                  bool(e[-1, 1] <= 1.05 * e[:, 1].max()), detail=f"final / max energy {e[-1, 1] / e[:, 1].max():.3g}")
        res.data("ladder", cells_per_wavelength=lad, vacuum_power_error=errs, configured_ratio=conf)
        res.data("energy", t=e[:, 0], W=e[:, 1])
        # figures
        fig, ax = P.figure(3, width=4.0)
        h = [1 / n for n in lad]
        ax[0].loglog(lad, errs, "o-", color=P.COMPUTED, lw=2, ms=6, label="vacuum dipole vs closed form")
        ax[0].set(xlabel="cells per wavelength", ylabel="|P / P_ref - 1|", title=f"order {rich.observed_order:.2f}")
        ax[0].legend(fontsize=8)
        ax[1].plot(h, conf, "s-", color=P.THIRD, lw=2, ms=6, label="with the sphere: P / P_vacuum")
        ax[1].axhline(float(rc.extrapolated_value), color=P.REFERENCE, ls="--", lw=1.4, label="Richardson")
        ax[1].set(xlabel="h / lambda", ylabel="power ratio", title=f"configured case, GCI {100 * rc.gci_fine:.2f} %")
        ax[1].legend(fontsize=8)
        th2 = np.linspace(0, 2 * np.pi, 361)
        pv = vac[-1][3].pattern(th2)
        pc = last.pattern(th2)
        ax[2].remove()
        a2 = fig.add_subplot(1, 3, 3, projection="polar")
        a2.plot(th2, pv / pv.max(), color=P.COMPUTED, lw=1.8, label="vacuum")
        a2.plot(th2, pc / pv.max(), color=P.THIRD, lw=1.8, label="with sphere")
        a2.plot(th2, np.sin(th2) ** 2, ":", color=P.REFERENCE, lw=1.2, label="sin^2")
        a2.set_theta_zero_location("N")
        a2.set_title("far field, plane phi = 0", fontsize=10)
        a2.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        res.figure("fdtd_convergence", fig)
        if p["snapshots"] > 2:
            res.movie("fdtd_3d_t", _slice_frames(last), fps=12)
            if p["viewer"]:
                res.viewer("fdtd_viewer", lambda path, run=last: slice_viewer(run, path))


def _display_slices(run) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Interior coordinates (in wavelengths, the CPML cropped) and E_z r on the three central planes, the first
    source's cell masked: the radiated field keeps its amplitude across the picture instead of the source cell
    swamping it."""
    n, p = run.n, run.config.npml
    lam = run.config.wavelength
    x = (np.arange(n) - n // 2) * run.dx / lam
    sl = slice(p, n - p)
    U, V = np.meshgrid(x[sl], x[sl], indexing="ij")
    R = np.sqrt(U ** 2 + V ** 2)
    out = {}
    for key, F in run.slices.items():
        out[key] = np.where(R > 0.12, F[:, sl, sl] * R, 0.0)
    return x[sl], out


def _slice_frames(run) -> list:
    """3-D view of E_z r on the three central planes, frame by frame (matplotlib)."""
    import matplotlib
    from matplotlib.colors import Normalize

    x, S = _display_slices(run)
    X, Y = np.meshgrid(x, x, indexing="ij")
    vmax = float(np.percentile(np.abs(np.concatenate([v.ravel() for v in S.values()])), 99.5)) or 1.0
    norm = Normalize(-vmax, vmax)
    cmap = matplotlib.colormaps[P.DIVERGING]
    frames = []
    for k in range(len(run.frame_times)):
        fig, ax = P.figure(1, width=6.0, height=5.4, three_d=(0,))
        a = ax[0]
        Z0 = np.zeros_like(X)
        a.plot_surface(X, Y, Z0, facecolors=cmap(norm(S["xy"][k])), rstride=1, cstride=1, shade=False, alpha=0.9)
        a.plot_surface(X, Z0, Y, facecolors=cmap(norm(S["xz"][k])), rstride=1, cstride=1, shade=False, alpha=0.9)
        a.plot_surface(Z0, X, Y, facecolors=cmap(norm(S["yz"][k])), rstride=1, cstride=1, shade=False, alpha=0.9)
        for m in run.config.materials:
            if m["shape"] == "sphere":
                u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
                c = m["center"]
                r = m["radius"]
                a.plot_wireframe(c[0] + r * np.cos(u) * np.sin(v), c[1] + r * np.sin(u) * np.sin(v),
                                 c[2] + r * np.cos(v), color=P.INK, lw=0.4)
        a.set(xlabel="x / lambda", ylabel="y / lambda", zlabel="z / lambda",
              title=f"E_z r on three planes, t = {run.frame_times[k] * run.config.f0:.2f} periods")
        a.view_init(elev=22, azim=-58 + 0.6 * k)
        fig.tight_layout()
        frames.append(P.to_rgb(fig))
        P.close(fig)
    return frames


def slice_viewer(run, path: str, title: str = "FDTD: E_z in 3-D + time") -> str:
    """A single-file three.js page (Studio viewer): the three central planes as surfaces, E_z r on them animated
    in time (play, pause, time slider)."""
    import os

    from pinneapple_tools.visualization.studio.animation import Animation
    from pinneapple_tools.visualization.studio.scene import Scene, Surface
    from pinneapple_tools.visualization.studio.web import web_viewer

    x, S = _display_slices(run)
    n = len(x)
    U, V = np.meshgrid(x, x, indexing="ij")
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + n, a + 1], [a + 1, a + n, a + n + 1]]
    faces = np.asarray(faces)
    Z0 = np.zeros(n * n)
    planes = {"plane_xy": np.column_stack([U.ravel(), V.ravel(), Z0]),
              "plane_xz": np.column_stack([U.ravel(), Z0, V.ravel()]),
              "plane_yz": np.column_stack([Z0, U.ravel(), V.ravel()])}
    scene = Scene([Surface(k, v, faces, "grey") for k, v in planes.items()], title=title)
    an = Animation(run.frame_times * run.config.f0, time_label="t", time_unit="periods", time_decimals=2)
    vmax = float(np.percentile(np.abs(S["xz"]), 99.5)) or 1.0
    for k, key in (("plane_xy", "xy"), ("plane_xz", "xz"), ("plane_yz", "yz")):
        an.add_surface_field(k, np.clip(S[key].reshape(len(run.frame_times), -1), -vmax, vmax),
                             label="E_z r (V)", color_range=(-vmax, vmax), cmap="coolwarm")
    scene.animation = an
    out_dir = os.path.dirname(os.path.abspath(path))
    return web_viewer(scene, out_dir, single_file=True, filename=os.path.basename(path))


# ===================================================================================== Monte Carlo UQ
@register
class WaveguideUQ(Experiment):
    name = "em.research_uq_waveguide"
    title = "Uncertainty of waveguide cut-offs by Monte Carlo"
    area = AREA
    topic = "uncertainty quantification"
    level = "research"
    dims = ("2D",)
    description = ("Manufacturing tolerances of a dielectric-filled rectangular guide (a, b, eps_r with stated "
                   "standard deviations) propagated through the finite-difference eigen-solver by Monte Carlo: "
                   "mean, standard deviation and 95 % interval of the TE10 and TE20 cut-offs, the sampling error "
                   "of the Monte Carlo itself, against first-order (GUM) propagation with the closed-form "
                   "sensitivities.")
    references = ("JCGM 101:2008 (GUM supplement 1, Monte Carlo propagation)", "JCGM 100:2008 eq. 10",
                  "Pozar, Microwave Engineering, 4th ed., eq. 3.84")
    params = {"a": 22.86e-3, "b": 10.16e-3, "eps_r": 2.2, "u_a": 0.05e-3, "u_b": 0.05e-3, "u_eps": 0.02,
              "samples": 1000, "scale": 6, "seed": 11}
    quick = {"samples": 150, "scale": 4}

    def _run(self, res: Result, dim: str, p):
        from ..numerics import helmholtz_eigs

        rng = np.random.default_rng(p["seed"])
        n = p["samples"]
        A = p["a"] + p["u_a"] * rng.standard_normal(n)
        B = p["b"] + p["u_b"] * rng.standard_normal(n)
        E = p["eps_r"] + p["u_eps"] * rng.standard_normal(n)
        out = np.empty((n, 2))
        s = p["scale"]
        for i in range(n):
            kc = helmholtz_eigs(9 * s, 4 * s, A[i], B[i], "TE", 2)
            out[i] = C0 * kc / (2 * math.pi * math.sqrt(E[i]))
        # GUM first order with closed-form sensitivities: f10 = c/(2 a sqrt(eps)), f20 = 2 f10
        f10 = C0 / (2 * p["a"] * math.sqrt(p["eps_r"]))
        u_rel = math.sqrt((p["u_a"] / p["a"]) ** 2 + (0.5 * p["u_eps"] / p["eps_r"]) ** 2)
        for j, (lab, fnom) in enumerate((("TE10", f10), ("TE20", 2 * f10))):
            m, sd = float(out[:, j].mean()), float(out[:, j].std(ddof=1))
            se_sd = sd / math.sqrt(2 * (n - 1))
            lo, hi = np.percentile(out[:, j], [2.5, 97.5])
            res.value(f"{lab}_mean", m, "Hz", f"{n} Monte Carlo samples through the FD eigen-solver")
            res.value(f"{lab}_std", sd, "Hz", "sample standard deviation", uncertainty=se_sd)
            res.value(f"{lab}_ci95_low", float(lo), "Hz", "2.5 % quantile")
            res.value(f"{lab}_ci95_high", float(hi), "Hz", "97.5 % quantile")
            res.compare(f"{lab}: Monte Carlo std vs first-order GUM", sd, u_rel * fnom,
                        tol=max(3 * se_sd / sd, 0.02), unit="Hz", kind="closed_form",
                        source="JCGM 100 eq. 10 with df/da = -f/a, df/deps = -f/(2 eps) (Pozar eq. 3.84)",
                        how="tolerance = 3 standard errors of the sample std (plus 2 %), a sampling-error test")
            res.compare(f"{lab}: Monte Carlo mean vs nominal", m, fnom, tol=2e-3, unit="Hz",
                        source="Pozar eq. 3.84 at the nominal dimensions (FD discretisation error included)")
        res.check("b does not enter TE10 or TE20 (sensitivity ~ 0)",
                  abs(np.corrcoef(B, out[:, 0])[0, 1]) < 0.15,
                  value=float(abs(np.corrcoef(B, out[:, 0])[0, 1])), limit=0.15,
                  detail="correlation of b with f_TE10 over the samples")
        res.data("samples", a=A[:400], b=B[:400], eps_r=E[:400], f_TE10=out[:400, 0], f_TE20=out[:400, 1])
        fig, ax = P.figure(2, width=4.4)
        ax[0].hist(out[:, 0] / 1e9, bins=40, color=P.COMPUTED, alpha=0.85, density=True, label="Monte Carlo")
        xx = np.linspace(out[:, 0].min(), out[:, 0].max(), 200)
        g = np.exp(-0.5 * ((xx - f10) / (u_rel * f10)) ** 2) / (u_rel * f10 * math.sqrt(2 * math.pi)) * 1e9
        ax[0].plot(xx / 1e9, g, color=P.REFERENCE, lw=1.5, label="first-order GUM")
        ax[0].set(xlabel="f_c TE10 (GHz)", ylabel="density", title=f"{n} samples")
        ax[0].legend(fontsize=8)
        ax[1].scatter(A * 1e3, out[:, 0] / 1e9, s=4, color=P.COMPUTED, alpha=0.5, label="a")
        ax[1].set(xlabel="a (mm)", ylabel="f_c TE10 (GHz)", title="sensitivity to the broad wall")
        fig.tight_layout()
        res.figure("uq_waveguide", fig)


# ===================================================================================== inverse problem
@register
class InversePermittivity(Experiment):
    name = "em.research_inverse_permittivity"
    title = "Inverse problem: permittivity of a slab from its reflection spectrum"
    area = AREA
    topic = "inverse problems"
    level = "research"
    dims = ("1D",)
    description = ("A dielectric slab of unknown permittivity and loss is measured in reflection: the 'measurement' "
                   "is a 1-D FDTD run (with the CPML, the scheme's dispersion and added receiver noise), the model "
                   "is the transfer-matrix reflection of pp.em.waves.multilayer. Least squares recovers eps_r and "
                   "the conductivity with their covariance. The model lacks the scheme's dispersion, which biases the "
                   "fit by about one standard uncertainty: the instrument is simulated at the fitted point and the "
                   "bias of that noiseless refit is removed, with its own uncertainty. The recovered values are "
                   "compared with the truth the measurement was generated from (hidden from the fit), and the fit "
                   "is repeated on noise realisations to check the stated uncertainty (coverage, centring).")
    references = ("Baker-Jarvis et al., NIST Technical Note 1536 (2004), transmission/reflection methods",
                  "Born & Wolf, Principles of Optics, 7th ed., sec. 1.6 (characteristic matrices)",
                  "Tarantola, Inverse Problem Theory (2005), sec. 3.2")
    params = {"eps_true": 3.7, "sigma_true": 0.02, "eps_design": 4.0, "thickness": 10e-3, "f_min": 2e9, "f_max": 12e9, "n_freq": 41,
              "noise": 0.004, "cells_per_wavelength": 60, "repeats": 100, "seed": 5}
    quick = {"repeats": 20}

    def _measure(self, p, eps, sigma, empty=None):
        """``(freqs, r, empty)``: the FDTD reflection of a slab ``(eps, sigma)``, referred to the slab's face.
        The grid depends only on the design permittivity ``eps_design``, never on the slab simulated, so the
        measurement and the bias simulations at the fitted point share one discretisation; ``empty`` (the run
        without the slab) is reused when given."""
        from pinneapple_simulation.electromagnetics.fdtd import FDTD, YeeGrid, modulated_gaussian
        from pinneapple_simulation.electromagnetics.media import Medium

        fmax = p["f_max"]
        lam = C0 / fmax / math.sqrt(p["eps_design"])
        dx = lam / p["cells_per_wavelength"]
        nd = int(round(p["thickness"] / dx))
        dx = p["thickness"] / nd
        N = int(3 * C0 / p["f_min"] / dx) + 2 * nd + 80
        freqs = np.linspace(p["f_min"], p["f_max"], p["n_freq"])
        g = YeeGrid((N, 1, 1), (dx, dx, dx))
        i0 = N // 2
        im = i0 - int(0.5 * C0 / p["f_min"] / dx)
        f0 = 0.5 * (p["f_min"] + p["f_max"])

        def run(slab):
            s = FDTD(g, {"x": "cpml", "y": "periodic", "z": "periodic"}, npml=30)
            if slab:
                m = np.zeros(g.shape, bool)
                m[i0:i0 + nd] = True
                s.set_material(m, Medium(eps, sigma=sigma))
            s.add_source("Ey", (40, 0, 0), modulated_gaussian(f0, 1.2 * (p["f_max"] - p["f_min"])), "soft")
            mon = s.add_dft(["Ey"], freqs, (slice(im, im + 1), slice(None), slice(None)))
            s.run_until(30 / p["f_min"])
            return mon.phasor("Ey").ravel()

        empty = run(False) if empty is None else empty
        r = (run(True) - empty) / empty
        # refer the phase to the slab's face, half a cell before its first node (the E_y nodes i0 .. i0+nd-1 carry
        # the slab, so it spans nd cells): the reflected wave has travelled 2 D more than at the face
        k0 = 2 * math.pi * freqs / C0
        return freqs, r * np.exp(2j * k0 * (i0 - 0.5 - im) * dx), empty

    def _bias(self, p, x, empty):
        """Bias of the fit at ``x``: the noiseless FDTD spectrum of the slab ``x`` refitted with the
        transfer-matrix model, minus ``x`` (the scheme's dispersion and the CPML, which the model lacks)."""
        f, r, _ = self._measure(p, float(x[0]), float(x[1]), empty)
        xb, _, _ = self._fit(f, r, p["thickness"], list(x))
        return xb - np.asarray(x, float)

    @staticmethod
    def _model(freqs, eps, sigma, d):
        n = np.sqrt((eps - 1j * sigma / (2 * math.pi * freqs * EPS0)).astype(complex))
        n = np.where(n.imag > 0, -n, n)
        return np.array([W.multilayer([nn], [d], C0 / f).r for nn, f in zip(n, freqs, strict=True)])

    def _fit(self, freqs, r, d, x0):
        from scipy.optimize import least_squares

        def resid(x):
            m = self._model(freqs, x[0], x[1], d)
            return np.concatenate([(m - r).real, (m - r).imag])

        sol = least_squares(resid, x0, x_scale=[1.0, 0.01], bounds=([1.0, 0.0], [20.0, 5.0]))
        J = sol.jac
        dof = max(len(sol.fun) - 2, 1)
        s2 = float(sol.fun @ sol.fun) / dof
        cov = np.linalg.inv(J.T @ J) * s2
        return sol.x, np.sqrt(np.diag(cov)), cov

    def _run(self, res: Result, dim: str, p):
        from pinneapple_analysis.uncertainty.tolerance import wilson_interval

        freqs, r_meas, empty = self._measure(p, p["eps_true"], p["sigma_true"])
        rng = np.random.default_rng(p["seed"])
        noisy = r_meas + p["noise"] * (rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))) / math.sqrt(2)
        x_raw, u_fit, cov = self._fit(freqs, noisy, p["thickness"], [2.5, 0.0])
        # the transfer-matrix model lacks the scheme's dispersion: simulate the instrument at the fitted point,
        # refit, and remove that bias; its own uncertainty is how much it moves over +-u(eps) around the fit
        b = self._bias(p, x_raw, empty)
        db = [self._bias(p, x_raw + s_ * np.array([u_fit[0], 0.0]), empty) for s_ in (1.0, -1.0)]
        u_b = 0.5 * np.abs(db[0] - db[1])
        x = x_raw - b
        u = np.hypot(u_fit, u_b)
        res.value("eps_r_fit_uncorrected", x_raw[0], "", "least squares on the noisy FDTD reflection spectrum",
                  uncertainty=u_fit[0])
        res.value("eps_r_model_bias", b[0], "", "fit of the noiseless FDTD spectrum simulated at the fitted point, "
                  "minus that point (the transfer-matrix model lacks the FDTD dispersion)", uncertainty=u_b[0])
        res.value("eps_r_recovered", x[0], "", "fit minus the model bias; u combines the fit and the bias",
                  uncertainty=u[0])
        res.value("sigma_recovered", x[1], "S/m", "same, for the conductivity", uncertainty=u[1])
        res.value("eps_sigma_correlation", cov[0, 1] / (u_fit[0] * u_fit[1]), "", "from the Jacobian covariance")
        res.compare("eps_r recovered vs truth", x[0], p["eps_true"], tol=2 * u[0] / p["eps_true"],
                    kind="ground_truth", unit="", source="the permittivity the FDTD measurement was generated with "
                    "(hidden from the fit)", how="compatibility within 2 u (k = 2)")
        res.compare("sigma recovered vs truth", x[1], p["sigma_true"], tol=2 * u[1] / p["sigma_true"],
                    kind="ground_truth", unit="S/m", source="as above")
        # model error without noise: the FDTD reflection against the transfer-matrix model at the truth
        mdl = self._model(freqs, p["eps_true"], p["sigma_true"], p["thickness"])
        merr = float(np.max(np.abs(mdl - r_meas)))
        res.value("model_error_max", merr, "", "max |r_FDTD - r_model| at the true parameters (no noise)")
        res.check("FDTD measurement and transfer-matrix model agree below the noise", merr < p["noise"],
                  value=merr, limit=p["noise"])
        # repeat on noise realisations: is the stated u right? (with and without the bias correction)
        z_raw, z = [], []
        for _ in range(p["repeats"]):
            nz = r_meas + p["noise"] * (rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))) / math.sqrt(2)
            xx, uu, _ = self._fit(freqs, nz, p["thickness"], [2.5, 0.0])
            z_raw.append((xx[0] - p["eps_true"]) / uu[0])
            z.append((xx[0] - b[0] - p["eps_true"]) / math.hypot(uu[0], u_b[0]))
        z_raw, z = np.asarray(z_raw), np.asarray(z)
        n = p["repeats"]
        hits = int(np.sum(np.abs(z) <= 2))
        cover = hits / n
        lo, hi = wilson_interval(hits, n)
        res.value("coverage_2u_eps_uncorrected", float(np.mean(np.abs(z_raw) <= 2)), "",
                  f"fraction of {n} noise realisations within 2 u, fit alone (no bias correction)")
        res.value("mean_z_uncorrected", float(np.mean(z_raw)), "", "mean of (eps_fit - eps_true) / u_fit")
        res.value("coverage_2u_eps", cover, "", f"fraction of {n} noise realisations within 2 u, bias corrected")
        res.value("mean_z", float(np.mean(z)), "", "mean of (eps_corrected - eps_true) / u, bias corrected")
        res.check("bias-corrected z centred (|mean z| < 0.3)", abs(float(np.mean(z))) < 0.3,
                  value=float(np.mean(z)), limit=0.3,
                  detail=f"without the correction mean z = {np.mean(z_raw):.2f}")
        res.check("stated u(eps) consistent with the scatter (std of z between 0.7 and 1.5)",
                  0.7 <= float(np.std(z)) <= 1.5, value=float(np.std(z)),
                  detail="z = (eps_corrected - eps_true) / u; its std is 1 when u is right")
        res.check("2 u covers the truth about 95 % of the time", lo <= 0.9545 <= hi, value=cover,
                  detail=f"{hits}/{n}, Wilson 95 % interval {lo:.3f}-{hi:.3f}; without the bias correction "
                         f"{np.mean(np.abs(z_raw) <= 2):.2f}")
        res.data("spectrum", f=freqs, abs_r=np.abs(noisy), abs_r_fit=np.abs(self._model(freqs, x[0], x[1], p["thickness"])))
        fig, ax = P.figure(2, width=4.5)
        ax[0].plot(freqs / 1e9, np.abs(noisy), "o", color=P.COMPUTED, ms=4, label="measured (FDTD + noise)")
        ff = np.linspace(freqs[0], freqs[-1], 300)
        ax[0].plot(ff / 1e9, np.abs(self._model(ff, x[0], x[1], p["thickness"])), color=P.REFERENCE, lw=1.5,
                   label=f"fit: eps = {x[0]:.3f} +- {u[0]:.3f}")
        ax[0].set(xlabel="f (GHz)", ylabel="|r|", title=f"slab {p['thickness'] * 1e3:g} mm, truth eps = {p['eps_true']}")
        ax[0].legend(fontsize=8)
        bins = np.linspace(-4, 4, 25)
        ax[1].hist(z_raw, bins=bins, color=P.INK2, alpha=0.35, density=True,
                   label=f"fit alone: mean {np.mean(z_raw):+.2f}, {100 * np.mean(np.abs(z_raw) <= 2):.0f} % in 2u")
        ax[1].hist(z, bins=bins, color=P.COMPUTED, alpha=0.75, density=True,
                   label=f"bias corrected: mean {np.mean(z):+.2f}, {100 * cover:.0f} % in 2u")
        zz = np.linspace(-4, 4, 100)
        ax[1].plot(zz, np.exp(-zz ** 2 / 2) / math.sqrt(2 * math.pi), color=P.REFERENCE, lw=1.5, label="N(0, 1)")
        ax[1].set(xlabel="z = (eps - eps_true) / u", ylabel="density", title=f"{n} noise realisations")
        ax[1].set_ylim(0, 1.5 * ax[1].get_ylim()[1])            # room for the legend above the bars
        ax[1].legend(fontsize=7.5, loc="upper left")
        fig.tight_layout()
        res.figure("inverse_permittivity", fig)


# ===================================================================================== surrogate
@register
class HallSurrogate(Experiment):
    name = "em.research_surrogate_hall"
    title = "Surrogate of the Hall geometrical factor with hold-out error"
    area = AREA
    topic = "surrogate models"
    level = "research"
    dims = ("2D",)
    description = ("A Gaussian process (anisotropic kernel) trained on finite-difference Hall-bar solutions over "
                   "the aspect ratio L/W and the Hall parameter mu B (Latin hypercube), its error measured on "
                   "held-out solver runs it never saw, its +-2 sigma coverage (variance calibrated by "
                   "leave-one-out on the training runs) and the long-bar limit G -> 1.")
    references = ("Rasmussen & Williams, Gaussian Processes for Machine Learning (2006), ch. 2 and 5",
                  "Popovic, Hall Effect Devices, 2nd ed. (2004), sec. 4.2")
    params = {"n_train": 40, "n_test": 30, "LW": [0.6, 4.0], "beta": [0.05, 1.0], "cells": 24, "seed": 3}
    quick = {"n_train": 20, "n_test": 10, "cells": 16}

    @staticmethod
    def _G(LW, beta, cells):
        from .statics import hall_bar

        W = 1.0
        x, y, phi = hall_bar(LW * W, W, beta, cells, 1.0, nx=4 * cells)   # fixed nx: G smooth in L/W
        i = len(x) // 2
        Ex = -np.gradient(phi, x, axis=0)[i]
        Ey = -np.gradient(phi, y, axis=1)[i]
        I = float(np.trapezoid((Ex + beta * Ey) / (1 + beta ** 2), y))  # noqa: E741  (I: current, sigma0 = t = 1)
        return float(-(phi[i, -1] - phi[i, 0]) / (beta * I))

    def _run(self, res: Result, dim: str, p):
        from pinneapple_presets.core import lhs
        from pinneapple_presets.surrogate import ARDGP

        lo = np.array([p["LW"][0], p["beta"][0]])
        hi = np.array([p["LW"][1], p["beta"][1]])
        box = [tuple(p["LW"]), tuple(p["beta"])]
        Xtr = lhs(p["n_train"], box, seed=p["seed"])
        Xte = lhs(p["n_test"], box, seed=p["seed"] + 100)
        ytr = np.array([self._G(a, b, p["cells"]) for a, b in Xtr])
        yte = np.array([self._G(a, b, p["cells"]) for a, b in Xte])
        from ..numerics import gp_loo_scale

        gp = ARDGP().fit((Xtr - lo) / (hi - lo), ytr)
        scale = gp_loo_scale(gp)          # calibrated on the training runs only (leave-one-out)
        mu, sd = gp.predict((Xte - lo) / (hi - lo))
        sd = sd * scale
        rel = np.abs(mu / yte - 1)
        cov = float(np.mean(np.abs(mu - yte) <= 2 * sd))
        res.value("holdout_max_rel_error", float(rel.max()), "", f"{p['n_test']} held-out solver runs")
        res.value("holdout_mean_rel_error", float(rel.mean()), "", "same")
        res.value("coverage_2sigma", cov, "", "fraction of held-out runs inside the GP's +-2 sigma (LOO-calibrated)")
        res.value("loo_sigma_scale", scale, "", "leave-one-out calibration factor of the GP's standard deviation "
                  "(Rasmussen & Williams eqs. 5.10-5.12), from the training runs only")
        res.compare("hold-out: worst surrogate G vs solver G", float(mu[np.argmax(rel)]), float(yte[np.argmax(rel)]),
                    tol=1e-2, kind="convergence", unit="", source="the finite-difference solver on held-out inputs",
                    how=f"GP trained on {p['n_train']} Latin-hypercube solver runs")
        Glong = self._G(8.0, 0.3, p["cells"])
        res.compare("solver: long bar G(L/W = 8) -> 1", Glong, 1.0, tol=1e-3, kind="limit", unit="",
                    source="Popovic sec. 4.2: G -> 1 for L/W >> 1")
        res.check("+-2 sigma coverage of the hold-out at least 80 %", cov >= 0.8, value=cov, limit=0.8)
        res.data("holdout", X=Xte, G=yte, G_gp=mu, sd=sd)
        fig, ax = P.figure(2, width=4.4)
        LW = np.linspace(*p["LW"], 60)
        for k, b in enumerate((0.1, 0.5, 1.0)):
            Xg = np.column_stack([LW, np.full_like(LW, b)])
            m_, s_ = gp.predict((Xg - lo) / (hi - lo))
            s_ = s_ * scale
            ax[0].plot(LW, m_, color=P.SERIES[k], lw=2, label=f"GP, mu B = {b}")
            ax[0].fill_between(LW, m_ - 2 * s_, m_ + 2 * s_, color=P.SERIES[k], alpha=0.15)
        ax[0].plot(Xtr[:, 0], ytr, ".", color=P.INK2, ms=4, label="training runs")
        ax[0].set(xlabel="L / W", ylabel="G = V_H / V_H(long bar)", title="surrogate of the solver")
        ax[0].legend(fontsize=7.5)
        ax[1].errorbar(yte, mu, yerr=2 * sd, fmt="o", color=P.COMPUTED, ms=4, capsize=2)
        lims = [min(yte.min(), mu.min()), max(yte.max(), mu.max())]
        ax[1].plot(lims, lims, "--", color=P.REFERENCE, lw=1.2)
        ax[1].set(xlabel="solver G (held out)", ylabel="GP G (+-2 sigma)", title=f"hold-out max {100 * rel.max():.2f} %")
        fig.tight_layout()
        res.figure("surrogate_hall", fig)
