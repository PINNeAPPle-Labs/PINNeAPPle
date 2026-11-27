"""Strong gravitational lensing as a parametric inverse problem: an SIE plus external shear with the physics
checks that decide whether a fit means anything (positive convergence, images that solve the lens equation, and an
image multiplicity that follows the caustics).

The closed forms in :mod:`pinneapple_physics.astro.lensing` answer the forward question (deflection, Einstein
radius, shadow). This module answers the inverse one -- given an image of a background galaxy, what are the lens
parameters, and what says the answer is physical? Everything is built from the singular isothermal ellipsoid (SIE)
with external shear, the workhorse of galaxy-scale lens modelling:

* convergence ``kappa = b / (2 sqrt(q^2 x'^2 + y'^2))`` and deflection (Keeton 2001, astro-ph/0102341, eqs. 2-3),
  which is exactly the pseudo-elliptical isothermal of Kormann et al. (1994);
* ``b`` is the Einstein radius of the equivalent singular isothermal sphere, so for ``q = 1`` it is the textbook
  ``theta_E = 4 pi (sigma / c)^2 D_ls / D_s`` of Narayan & Bartelmann (1996, astro-ph/9606001) and Schneider, Ehlers
  & Falco (1992, *Gravitational Lenses*);
* external shear ``(gamma_1, gamma_2)`` adds no convergence -- only tidal distortion -- so it must not change the
  sign of ``kappa`` anywhere;
* the tangential critical curve (where the lens map is singular) maps to the tangential caustic, and the image of
  the singular centre maps to the radial caustic; the number of images of a point source is set by which of those
  closed curves it lies in.

**What is measured and what is generated.** No HST, Euclid or SLACS image is read here: :data:`LENSES` are three
*generated* lenses with parameters taken from the published analytic isothermal model, and the images are synthetic
renderings of those lenses. :func:`data_requirements` and :func:`load_lens_sample` name exactly what an observed
sample would have to provide (file, release, columns, size) and fail cleanly when it is absent.

    from pinneapple_physics.astro.lens_model import LENSES, lens_image, fit_lens_image, consistency_report
    lens, src = LENSES["galaxy-scale"]
    image, pixel = lens_image(lens, src, n=48)
    fit = fit_lens_image(image, pixel)
    print(fit.recovered["theta_e_arcsec"], consistency_report(fit.lens))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .lensing import C, G, M_SUN, MUAS, PC, deflection_angle_rad, einstein_radius_rad

__all__ = [
    "MPC", "angular_diameter_distance_mpc", "isothermal_einstein_arcsec", "point_mass_crosscheck",
    "sie_deflection", "SIELens", "GaussianSource", "lens_image", "fit_lens_image", "LensFit", "DeclaredLens",
    "consistency_report", "LENSES", "data_requirements", "load_lens_sample",
]

MPC = 3.0856775814913673e22          # metres
KMS = 1.0e3


# ------------------------------------------------------------------------------------------------ cosmology
def _comoving_mpc(z: float, h0: float, omega_m: float, n: int = 400) -> float:
    """Line-of-sight comoving distance (Mpc) of a flat LCDM model, by the trapezoid rule on a fixed grid."""
    if z <= 0.0:
        return 0.0
    zz = np.linspace(0.0, float(z), n)
    e = np.sqrt(omega_m * (1.0 + zz) ** 3 + (1.0 - omega_m))
    return float(C * 1e-3 / h0) * float(np.trapezoid(1.0 / e, zz))


def angular_diameter_distance_mpc(z: float, h0: float = 70.0, omega_m: float = 0.3) -> float:
    """Angular diameter distance (Mpc) of a flat LCDM model.

    The defaults (``h0 = 70``, ``omega_m = 0.3``) are round numbers, not a measurement; the lensing ratio
    ``D_ls / D_s`` depends on them only weakly (a few per cent between plausible cosmologies), and every result
    below that uses distances states the values it used.
    """
    if z <= 0.0:
        return 0.0
    return _comoving_mpc(z, h0, omega_m) / (1.0 + z)


def isothermal_einstein_arcsec(sigma_kms: float, z_lens: float, z_source: float, h0: float = 70.0,
                               omega_m: float = 0.3) -> float:
    """Einstein radius of a singular isothermal sphere: ``theta_E = 4 pi (sigma / c)^2 D_ls / D_s``.

    The SIS Einstein radius does not depend on the lens distance except through ``D_ls / D_s``; that is why a
    measured ``theta_E`` and a measured velocity dispersion give a mass without knowing ``H0`` precisely.
    """
    d_l = angular_diameter_distance_mpc(z_lens, h0, omega_m)
    d_s = angular_diameter_distance_mpc(z_source, h0, omega_m)
    if d_s <= 0.0 or d_l >= d_s:
        raise ValueError(f"need 0 < z_lens < z_source; got z_lens={z_lens}, z_source={z_source}")
    return 4.0 * math.pi * (sigma_kms * KMS / C) ** 2 * (d_s - d_l) / d_s * (180.0 / math.pi) * 3600.0


def point_mass_crosscheck(sigma_kms: float, z_lens: float, z_source: float, h0: float = 70.0,
                          omega_m: float = 0.3) -> dict:
    """The isothermal sphere and a point mass agree exactly when the point mass is the *projected* mass inside.

    A singular isothermal sphere of velocity dispersion ``sigma`` has a flat rotation curve, so its deflection is
    the constant ``4 pi (sigma / c)^2`` at every impact parameter, and the projected mass inside the angular
    radius ``theta`` is ``M(<theta) = pi sigma^2 theta D_l / G`` (surface density ``sigma^2 / (2 G D_l theta)``).
    Feeding that mass and ``b = D_l theta_E`` to
    :func:`pinneapple_physics.astro.lensing.deflection_angle_rad`/``einstein_radius_rad`` -- the closed forms
    ``lensing.eht_comparison`` and the shadow diameter use -- must return the same deflection and the same
    Einstein radius. It is an exact identity, so it checks both formulas and the unit bookkeeping between them.
    """
    theta_sis = isothermal_einstein_arcsec(sigma_kms, z_lens, z_source, h0, omega_m)
    d_l = angular_diameter_distance_mpc(z_lens, h0, omega_m) * MPC
    d_s = angular_diameter_distance_mpc(z_source, h0, omega_m) * MPC
    theta_rad = theta_sis / 3600.0 * math.pi / 180.0
    mass_kg = math.pi * (sigma_kms * KMS) ** 2 * theta_rad * d_l / G
    alpha_point = float(deflection_angle_rad(mass_kg, d_l * theta_rad))
    theta_point = float(einstein_radius_rad(mass_kg, d_l, d_s))
    arcsec = math.pi / 180.0 / 3600.0
    return {"theta_e_arcsec": theta_sis, "projected_mass_msun": mass_kg / M_SUN,
            "deflection_sis_rad": 4.0 * math.pi * (sigma_kms * KMS / C) ** 2,
            "deflection_point_rad": alpha_point,
            "theta_point_mass_arcsec": theta_point / arcsec,
            "theta_relative_difference": abs(theta_point - theta_rad) / theta_rad,
            "deflection_relative_difference":
                abs(alpha_point - 4.0 * math.pi * (sigma_kms * KMS / C) ** 2)
                / (4.0 * math.pi * (sigma_kms * KMS / C) ** 2)}


# --------------------------------------------------------------------------------------------- the SIE lens
def sie_deflection(x, y, theta_e: float = 1.0, q: float = 0.8, pa_deg: float = 0.0):
    """Deflection angle of a singular isothermal ellipsoid, in the same units as ``x`` and ``y``.

    ``theta_e`` is the Einstein radius ``b`` of the equivalent spherical isothermal lens, ``q`` the axis ratio
    (< 1) and ``pa_deg`` the position angle of the major axis. In coordinates aligned with the major axis the
    deflection is (Keeton 2001, eqs. 2-3)

        alpha_1 = b / sqrt(1 - q^2) * arctan(sqrt(1 - q^2) x / psi)
        alpha_2 = b / sqrt(1 - q^2) * artanh(sqrt(1 - q^2) y / psi),    psi = sqrt(q^2 x^2 + y^2)

    whose convergence is ``kappa = b / (2 psi)``: positive everywhere except the singular centre, and reducing to
    the isothermal sphere ``alpha = b x / r`` as ``q -> 1``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1]; got {q}")
    pa = math.radians(pa_deg)
    ct, st = math.cos(pa), math.sin(pa)
    xr = x * ct + y * st
    yr = -x * st + y * ct
    if 1.0 - q * q < 1e-8:                                   # spherical branch: no coordinate singularity
        r = np.maximum(np.hypot(xr, yr), 1e-12)
        a1 = theta_e * xr / r
        a2 = theta_e * yr / r
    else:
        s = math.sqrt(1.0 - q * q)
        psi = np.maximum(np.sqrt(q * q * xr * xr + yr * yr), 1e-12)
        a1 = theta_e / s * np.arctan(s * xr / psi)
        a2 = theta_e / s * np.arctanh(np.clip(s * yr / psi, -1.0 + 1e-12, 1.0 - 1e-12))
    return a1 * ct - a2 * st, a1 * st + a2 * ct


def _sie_jacobian(x, y, theta_e: float, q: float, pa_deg: float):
    """Derivatives of :func:`sie_deflection` with respect to x and y (closed form; see the docstring there)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pa = math.radians(pa_deg)
    ct, st = math.cos(pa), math.sin(pa)
    xr = x * ct + y * st
    yr = -x * st + y * ct
    psi = np.maximum(np.sqrt(q * q * xr * xr + yr * yr), 1e-12)
    if 1.0 - q * q < 1e-8:
        r = np.maximum(np.hypot(xr, yr), 1e-12)
        j11 = theta_e * yr * yr / r ** 3
        j22 = theta_e * xr * xr / r ** 3
        j12 = -theta_e * xr * yr / r ** 3
    else:
        s = math.sqrt(1.0 - q * q)
        den1 = 1.0 + (s * xr / psi) ** 2
        den2 = np.maximum(1.0 - (s * yr / psi) ** 2, q * q)      # = q^2 at psi = |yr|, never singular
        j11 = theta_e * (1.0 / psi - q * q * xr * xr / psi ** 3) / den1
        j22 = theta_e * (1.0 / psi - yr * yr / psi ** 3) / den2
        # curl-free: d alpha_1 / d y = d alpha_2 / d x, so one off-diagonal number serves both
        j12 = -theta_e * xr * yr / psi ** 3
    c2, s2, cs = ct * ct, st * st, ct * st
    jxx = j11 * c2 + j22 * s2 - 2.0 * j12 * cs
    jyy = j11 * s2 + j22 * c2 + 2.0 * j12 * cs
    jxy = (j11 - j22) * cs + j12 * (c2 - s2)
    return jxx, jxy, jxy, jyy


@dataclass
class SIELens:
    """A singular isothermal ellipsoid plus external shear: ``theta_e`` (arcsec), ``q`` (axis ratio),
    ``pa_deg`` (major-axis position angle), and the shear components ``gamma_1``, ``gamma_2``."""
    theta_e: float = 1.0
    q: float = 0.8
    pa_deg: float = 0.0
    gamma1: float = 0.0
    gamma2: float = 0.0
    name: str = ""

    def __post_init__(self):
        if self.theta_e <= 0:
            raise ValueError(f"theta_e must be positive; got {self.theta_e}")
        if not 0.0 < self.q <= 1.0:
            raise ValueError(f"q must be in (0, 1]; got {self.q}")

    # -- forward model -------------------------------------------------------------------------
    def deflection(self, x, y):
        """Total deflection (SIE + shear); the shear only stretches, it adds no surface density."""
        ax, ay = sie_deflection(x, y, self.theta_e, self.q, self.pa_deg)
        return ax + self.gamma1 * x + self.gamma2 * y, ay + self.gamma2 * x - self.gamma1 * y

    def convergence(self, x, y):
        """Surface mass density in units of the critical one: ``kappa = b / (2 psi)``, no shear contribution."""
        pa = math.radians(self.pa_deg)
        xr = np.asarray(x, dtype=float) * math.cos(pa) + np.asarray(y, dtype=float) * math.sin(pa)
        yr = -np.asarray(x, dtype=float) * math.sin(pa) + np.asarray(y, dtype=float) * math.cos(pa)
        psi = np.maximum(np.sqrt(self.q ** 2 * xr * xr + yr * yr), 1e-12)
        return self.theta_e / (2.0 * psi)

    def jacobian(self, x, y):
        """``d alpha / d theta`` (4-tuple ``jxx, jxy, jyx, jyy``), shear included."""
        jxx, jxy, _, jyy = _sie_jacobian(x, y, self.theta_e, self.q, self.pa_deg)
        return jxx + self.gamma1, jxy + self.gamma2, jxy + self.gamma2, jyy - self.gamma1

    def source_position(self, x, y):
        """Inverse lens map: where on the source plane the image-plane point (x, y) came from."""
        ax, ay = self.deflection(x, y)
        return np.asarray(x, dtype=float) - ax, np.asarray(y, dtype=float) - ay

    def magnification(self, x, y):
        """``1 / det(I - d alpha / d theta)``; divergent on the critical curves, negative on inverted images."""
        jxx, jxy, jyx, jyy = self.jacobian(x, y)
        det = (1.0 - jxx) * (1.0 - jyy) - jxy * jyx
        return np.where(np.abs(det) < 1e-12, np.inf, 1.0 / det)

    # -- the curves that decide the physics ------------------------------------------------------
    def critical_curve(self, n: int = 181, r_min: float = 0.02, r_max: float = 3.0):
        """Tangential critical curve: the points where ``det(I - d alpha/d theta) = 0``, found as the first sign
        change of the determinant along rays from the lens centre. Returns an (n, 2) array in arcsec (NaN rows
        for rays with no crossing)."""
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        r = np.linspace(r_min, r_max, 400) * self.theta_e
        out = np.full((n, 2), np.nan)
        for i, ang in enumerate(t):
            det = self._det(r * math.cos(ang), r * math.sin(ang))
            sign_change = np.nonzero(det[:-1] * det[1:] < 0)[0]
            if not len(sign_change):
                continue
            k = sign_change[-1]                                    # outermost crossing = tangential curve
            r0, r1 = r[k], r[k + 1]
            d0, d1 = det[k], det[k + 1]
            rc = r0 - d0 * (r1 - r0) / (d1 - d0)
            out[i] = (rc * math.cos(ang), rc * math.sin(ang))
        return out

    def radial_caustic(self, n: int = 181, epsilon: float = 1e-4):
        """Image of the singular centre: the curve in the source plane where the radial images appear or vanish.

        The SIE has no radial critical curve at a finite radius, so the multiplicity boundary that a non-singular
        lens would put on its radial critical curve sits on the image of ``r -> 0``, which is finite because the
        isothermal deflection does not vanish at the centre."""
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        r = epsilon * self.theta_e
        return np.stack(self.source_position(r * np.cos(t), r * np.sin(t)), axis=1)

    def tangential_caustic(self, n: int = 181, **kw):
        """Image of the tangential critical curve under the lens map: the astroid that quads sit inside."""
        cc = self.critical_curve(n=n, **kw)
        ok = np.isfinite(cc[:, 0])
        if not ok.any():
            raise RuntimeError("no critical curve found for this lens")
        return np.stack(self.source_position(cc[ok, 0], cc[ok, 1]), axis=1)

    def effective_einstein_radius(self, n: int = 181) -> float:
        """Mean radius of the tangential critical curve: what a fit would report as ``theta_E`` of the quad."""
        cc = self.critical_curve(n=n)
        cc = cc[np.isfinite(cc[:, 0])]
        return float(np.mean(np.hypot(cc[:, 0], cc[:, 1])))

    def _det(self, x, y):
        jxx, jxy, jyx, jyy = self.jacobian(x, y)
        return (1.0 - jxx) * (1.0 - jyy) - jxy * jyx

    # -- the inverse problem --------------------------------------------------------------------
    def images(self, beta, n_cc: int = 240, iters: int = 60, explore: int = 2):
        """All image positions of a point source at ``beta = (bx, by)``, in arcsec.

        Images are seeded **on and around the critical curve**, which is where pairs of them appear and disappear:
        a Cartesian start grid misses the merging pair of a source near a caustic, where the map is degenerate and
        the basin is a sliver, while a start on the critical curve falls into it at once. Newton is damped with a
        backtracking line search (plain Newton oscillates on the singular curve), and every root found is then
        perturbed radially and tangentially and re-solved, to pick up siblings the first pass missed.

        Roots closer than 1e-3 theta_E are merged, the singular centre is not reported, and an image is only
        returned when the lens equation ``theta - alpha(theta) = beta`` is satisfied to 1e-9 theta_E.
        """
        bx, by = float(beta[0]), float(beta[1])
        cc = self.critical_curve(n=n_cc)
        cc = cc[np.isfinite(cc[:, 0])]
        r_c = float(np.mean(np.hypot(cc[:, 0], cc[:, 1]))) if len(cc) else self.theta_e
        t = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        seeds = [cc]
        for rr in np.linspace(0.08, 0.92, 12) * r_c:
            seeds.append(np.stack([rr * np.cos(t), rr * np.sin(t)], axis=1))
        for fac in (1.25, 1.7, 2.5, 3.5):
            seeds.append(np.stack([fac * self.theta_e * np.cos(t),
                                   fac * self.theta_e * np.sin(t)], axis=1))
        starts = np.concatenate(seeds, axis=0)
        found = self._roots(starts[:, 0], starts[:, 1], bx, by, iters)
        for _ in range(explore):
            if not len(found):
                break
            base = np.asarray(found)
            extra = [base * 1.07, base * 0.93]
            for phi in (0.15, -0.15):
                c, s = math.cos(phi), math.sin(phi)
                extra.append(np.stack([base[:, 0] * c - base[:, 1] * s,
                                       base[:, 0] * s + base[:, 1] * c], axis=1))
            x0, y0 = np.concatenate(extra, axis=0).T
            add = self._roots(x0, y0, bx, by, iters)
            if not len(add):
                break
            grew = False
            for p in add:
                if not any(math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-3 * self.theta_e for q in found):
                    found.append((float(p[0]), float(p[1])))
                    grew = True
            if not grew:
                break
        return np.array(found) if len(found) else np.zeros((0, 2))

    def _roots(self, x, y, bx, by, iters):
        """Damped Newton from the starts ``(x, y)``; returns the distinct roots that satisfy the lens equation."""
        x = np.asarray(x, dtype=float).copy()
        y = np.asarray(y, dtype=float).copy()
        fx, fy, nf = self._residual(x, y, bx, by)
        for _ in range(iters):
            if np.sqrt(nf.max()) < 1e-14:
                break
            jxx, jxy, jyx, jyy = self.jacobian(x, y)
            j11, j12, j21, j22 = 1.0 - jxx, -jxy, -jyx, 1.0 - jyy
            det = j11 * j22 - j12 * j21
            det = np.where(np.abs(det) < 1e-14, 1e-14, det)
            dx = (j22 * fx - j12 * fy) / det
            dy = (-j21 * fx + j11 * fy) / det
            scale = np.minimum(1.0, 0.25 * self.theta_e / np.maximum(np.hypot(dx, dy), 1e-30))
            dx, dy = dx * scale, dy * scale
            converged = nf < 1e-28
            dx = np.where(converged, 0.0, dx)
            dy = np.where(converged, 0.0, dy)
            step = np.ones_like(x)
            good = np.zeros(x.shape, bool)
            for _ in range(6):
                xn, yn = x - step * dx, y - step * dy
                fxn, fyn, nfn = self._residual(xn, yn, bx, by)
                better = nfn < nf
                good |= better
                if good.all():
                    break
                step = np.where(better, step, 0.5 * step)
            x, y = np.where(good, xn, x), np.where(good, yn, y)
            fx, fy = np.where(good, fxn, fx), np.where(good, fyn, fy)
            nf = np.where(good, nfn, nf)
        ok = (np.sqrt(nf) < 1e-9 * self.theta_e) & (np.hypot(x, y) > 1e-4 * self.theta_e)
        found = []
        for px, py in zip(x[ok], y[ok], strict=False):
            if not any(math.hypot(px - a, py - b) < 1e-3 * self.theta_e for a, b in found):
                found.append((float(px), float(py)))
        return found

    def _residual(self, x, y, bx, by):
        ax, ay = self.deflection(x, y)
        fx, fy = x - ax - bx, y - ay - by
        return fx, fy, fx * fx + fy * fy

    def multiplicity(self, beta, **kw) -> int:
        """Number of images of a point source at ``beta`` (arcsec)."""
        return int(len(self.images(beta, **kw)))

    def distance_to_caustic(self, points, n: int = 361) -> np.ndarray:
        """Distance from each point (arcsec) to the nearest point of either caustic curve.

        The curves are sampled at ``n`` angles; that sampling is the resolution of the test, so a distance smaller
        than a caustic sampling step is "on the caustic" for the purpose of the multiplicity check.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        curves = np.concatenate([self.tangential_caustic(n=n), self.radial_caustic(n=n)], axis=0)
        curves = curves[np.isfinite(curves[:, 0])]
        d = np.hypot(pts[:, None, 0] - curves[None, :, 0], pts[:, None, 1] - curves[None, :, 1])
        return d.min(axis=1)


# ---------------------------------------------------------------------------------------- sources, images
@dataclass
class GaussianSource:
    """An elliptical Gaussian background galaxy: total flux, major-axis sigma, axis ratio and position angle
    (all in arcsec / degrees), at ``(x0, y0)`` on the source plane."""
    flux: float = 1.0
    sigma: float = 0.25
    q: float = 1.0
    pa_deg: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    name: str = ""

    def surface_brightness(self, x, y):
        t = math.radians(self.pa_deg)
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        xr = dx * math.cos(t) + dy * math.sin(t)
        yr = -dx * math.sin(t) + dy * math.cos(t)
        norm = 2.0 * math.pi * self.sigma ** 2 * max(self.q, 1e-6)
        return self.flux * np.exp(-0.5 * (xr * xr + (yr / max(self.q, 1e-6)) ** 2) / self.sigma ** 2) / norm


def lens_image(lens: SIELens, source: GaussianSource, n: int = 48, half: float | None = None):
    """Constant-surface-brightness rendering of a lensed source: for every image-plane pixel the lens map gives
    the source-plane position, whose surface brightness is laid down there (so arcs, rings and multiple images
    appear with the right multiplicity). Returns ``(image, pixel_scale_arcsec)``."""
    half = half if half is not None else 3.0 * lens.theta_e
    grid = np.linspace(-half, half, n)
    x, y = np.meshgrid(grid, grid)
    bx, by = lens.source_position(x, y)
    img = source.surface_brightness(bx, by)
    inside = np.hypot(x, y) < 0.05 * lens.theta_e            # the singular centre is not sampled
    img[inside] = 0.0
    return img, grid[1] - grid[0]


@dataclass
class LensFit:
    """A recovered lens, its uncertainty, the residual of the fit and every parameter's truth when known."""
    lens: SIELens
    source: GaussianSource
    recovered: dict
    error: dict
    covariance: np.ndarray
    rms_residual: float
    names: tuple = ("theta_e", "q", "pa_deg", "gamma1", "gamma2", "src_x0", "src_y0", "src_sigma", "src_q",
                    "src_pa", "src_flux")
    truth: dict = field(default_factory=dict)

    def z_scores(self) -> dict:
        """``(recovered - truth) / error`` per parameter, for the parameters whose truth is known."""
        return {k: (self.recovered[k] - self.truth[k]) / self.error[k]
                for k in self.truth if self.error.get(k, 0.0) > 0.0}


def fit_lens_image(image, pixel: float, guess: SIELens | None = None, bounds: dict | None = None,
                   noise: float | None = None) -> LensFit:
    """Recover lens and source parameters from a rendered image by bounded least squares.

    The model is :func:`lens_image` with the lens centre held at the image centre (the usual first step of a lens
    model, and the only way the fit stays identifiable from a single band): 5 lens parameters and 6 source ones.
    ``error`` comes from the linearised covariance ``sigma^2 (J^T J)^-1`` with ``sigma`` the residual rms (or
    ``noise`` when the noise level is known), so it is a local, Gaussian estimate -- the same approximation the
    Gaussian imaging prior uses on the other side of this issue.
    """
    from scipy.optimize import least_squares

    image = np.asarray(image, dtype=float)
    n = image.shape[0]
    guess = guess or SIELens(theta_e=1.0, q=0.9, pa_deg=0.0, gamma1=0.0, gamma2=0.0)
    half = pixel * n / 2.0
    grid = np.linspace(-half + pixel / 2, half - pixel / 2, n)
    gx, gy = np.meshgrid(grid, grid)
    lo = {"theta_e": 0.3, "q": 0.4, "pa_deg": -90.0, "gamma1": -0.3, "gamma2": -0.3, "src_x0": -0.6,
          "src_y0": -0.6, "src_sigma": 0.02, "src_q": 0.2, "src_pa": -90.0, "src_flux": 0.05}
    hi = {"theta_e": 6.0, "q": 1.0, "pa_deg": 90.0, "gamma1": 0.3, "gamma2": 0.3, "src_x0": 0.6,
          "src_y0": 0.6, "src_sigma": 1.5, "src_q": 1.0, "src_pa": 90.0, "src_flux": 20.0}
    if bounds:
        for k, v in bounds.items():
            lo[k], hi[k] = v

    names = ("theta_e", "q", "pa_deg", "gamma1", "gamma2", "src_x0", "src_y0", "src_sigma", "src_q", "src_pa",
             "src_flux")
    p0 = np.array([guess.theta_e, guess.q, guess.pa_deg, guess.gamma1, guess.gamma2, 0.05, 0.02,
                   0.15, 1.0, 0.0, float(image.sum() * pixel * pixel)])

    def model(p):
        lens = SIELens(theta_e=p[0], q=min(p[1], 1.0), pa_deg=p[2], gamma1=p[3], gamma2=p[4])
        src = GaussianSource(flux=p[10], sigma=p[7], q=p[8], pa_deg=p[9], x0=p[5], y0=p[6])
        bx, by = lens.source_position(gx, gy)
        out = src.surface_brightness(bx, by)
        out[np.hypot(gx, gy) < 0.05 * p[0]] = 0.0
        return out

    def resid(p):
        return (model(p) - image).ravel()

    lobes = ([lo[k] for k in names], [hi[k] for k in names])
    # a start that is inside the box, and nothing more: a wide lens (a group, theta_e ~ 5") spreads the same flux
    # over a larger field, so the flux guess leaves the box and the fit refuses to run. Values already inside it are
    # left exactly as they are -- nudging them by even 1e-6 moves the solution to another basin.
    below = np.array(lobes[0])
    above = np.array(lobes[1])
    span = np.maximum(above - below, 1e-12)
    p0 = np.where(p0 < below, below + 1e-3 * span, np.where(p0 > above, above - 1e-3 * span, p0))
    sol = least_squares(resid, p0, bounds=lobes, x_scale="jac", max_nfev=400)
    p = sol.x.copy()
    p[1] = min(p[1], 1.0)
    lens = SIELens(theta_e=p[0], q=p[1], pa_deg=p[2], gamma1=p[3], gamma2=p[4])
    src = GaussianSource(flux=p[10], sigma=p[7], q=p[8], pa_deg=p[9], x0=p[5], y0=p[6])
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    sigma = float(noise) if noise is not None else max(rms, 1e-12)
    try:
        jtj = sol.jac.T @ sol.jac
        cov = sigma ** 2 * np.linalg.pinv(jtj)
        err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:                      # pragma: no cover - a degenerate image has no information
        cov = np.full((len(p), len(p)), np.nan)
        err = np.full(len(p), np.nan)
    return LensFit(lens=lens, source=src, recovered=dict(zip(names, p.tolist(), strict=False)),
                   error=dict(zip(names, err.tolist(), strict=False)), covariance=cov, rms_residual=rms)


# ------------------------------------------------------------------------------------ physics consistency
def consistency_report(lens: SIELens, radii_inner=(0.02, 0.05), radii_outer=(0.85, 0.95), n_directions: int = 6
                       ) -> dict:
    """The checks a lens model has to pass before its parameters mean anything.

    * **positive convergence**: ``kappa`` is sampled on the image plane (the singular centre excluded, where
      ``kappa -> infinity`` by construction). It must be positive everywhere: the external shear adds no mass, so a
      non-positive value would make the model unphysical rather than merely unconstrained.
    * **the images solve the lens equation**: every image returned by :meth:`SIELens.images` is pushed back
      through the lens map and must land on the requested source position.
    * **the multiplicity follows the caustics**: along ``n_directions`` rays the source is placed close in
      (``radii_inner``, well inside the tangential caustic) and far out (``radii_outer``, well outside it). A
      source inside the tangential caustic of an ellipsoidal lens is seen in four images and a source outside it
      in two; the pairs are created and destroyed on the caustic, which is what a lens model must reproduce. The
      distance from each probe to the nearest caustic curve is reported, so the two groups are visibly on
      different sides rather than assumed to be.
    """
    grid_x, grid_y = np.meshgrid(np.linspace(-2.0, 2.0, 41) * lens.theta_e,
                                 np.linspace(-2.0, 2.0, 41) * lens.theta_e)
    r = np.hypot(grid_x, grid_y)
    kappa = lens.convergence(grid_x, grid_y)[r > 0.05 * lens.theta_e]

    probes = []
    for k in range(n_directions):
        phi = 2.0 * np.pi * k / n_directions
        for group, radii, expected in (("inner", radii_inner, 4), ("outer", radii_outer, 2)):
            for f in radii:
                probes.append((phi, group, float(f), expected))

    rows = []
    for phi, group, f, expected in probes:
        src = (f * lens.theta_e * math.cos(phi), f * lens.theta_e * math.sin(phi))
        imgs = lens.images(src)
        back = lens.source_position(imgs[:, 0], imgs[:, 1]) if len(imgs) else (np.zeros(0), np.zeros(0))
        dev = float(np.max(np.hypot(back[0] - src[0], back[1] - src[1]))) if len(imgs) else float("nan")
        rows.append({"direction_deg": float(math.degrees(phi)), "group": group, "radius_theta_e": f,
                     "source_arcsec": (float(src[0]), float(src[1])), "n_images": len(imgs),
                     "expected": expected, "source_error_arcsec": dev,
                     "distance_to_caustic_arcsec": float(lens.distance_to_caustic(src)[0]),
                     "solves_lens_equation": bool(not len(imgs) or dev < 1e-6)})

    solves = all(row["solves_lens_equation"] for row in rows)
    multiplicity = all(row["n_images"] == row["expected"] for row in rows)
    count_mismatch = [row for row in rows if row["n_images"] != row["expected"]]
    return {"min_convergence": float(kappa.min()), "max_convergence": float(kappa.max()),
            "convergence_positive": bool(kappa.min() > 0.0),
            "solves_lens_equation": bool(solves),
            "multiplicity_ok": bool(multiplicity),
            "inner_counts": sorted({r0["n_images"] for r0 in rows if r0["group"] == "inner"}),
            "outer_counts": sorted({r0["n_images"] for r0 in rows if r0["group"] == "outer"}),
            "n_probes": len(rows), "n_mismatch": len(count_mismatch), "mismatch": count_mismatch,
            "rows": rows,
            "all_ok": bool(kappa.min() > 0.0 and solves and multiplicity)}


# ------------------------------------------------------------------------ declared lenses and data loader
@dataclass
class DeclaredLens:
    """One of the *generated* lenses of :data:`LENSES`: the lens, its source, and the published quantities its
    Einstein radius was built from."""
    lens: SIELens
    source: GaussianSource
    sigma_kms: float
    z_lens: float
    z_source: float
    image: str = ""

    @property
    def theta_e_formula_arcsec(self) -> float:
        return isothermal_einstein_arcsec(self.sigma_kms, self.z_lens, self.z_source)


LENSES = {
    # Generated lenses. The Einstein radius is the published isothermal formula evaluated at the stated velocity
    # dispersion and redshifts; the axis ratio, position angle and shear span the range reported for galaxy-scale
    # and group-scale lenses. Nothing here is a fit to an observed system.
    "galaxy-scale": DeclaredLens(
        lens=SIELens(theta_e=isothermal_einstein_arcsec(250.0, 0.30, 1.20), q=0.78, pa_deg=35.0,
                     gamma1=0.05, gamma2=-0.02, name="galaxy-scale"),
        source=GaussianSource(flux=1.0, sigma=0.22, q=0.85, pa_deg=20.0, x0=0.08, y0=0.05, name="source"),
        sigma_kms=250.0, z_lens=0.30, z_source=1.20, image="generated"),
    "quad-with-shear": DeclaredLens(
        lens=SIELens(theta_e=isothermal_einstein_arcsec(290.0, 0.45, 2.00), q=0.62, pa_deg=-20.0,
                     gamma1=0.09, gamma2=0.06, name="quad-with-shear"),
        source=GaussianSource(flux=1.0, sigma=0.14, q=0.70, pa_deg=-40.0, x0=0.03, y0=-0.02, name="source"),
        sigma_kms=290.0, z_lens=0.45, z_source=2.00, image="generated"),
    "group-scale": DeclaredLens(
        lens=SIELens(theta_e=isothermal_einstein_arcsec(550.0, 0.20, 1.50), q=0.85, pa_deg=70.0,
                     gamma1=0.03, gamma2=0.04, name="group-scale"),
        source=GaussianSource(flux=1.0, sigma=0.30, q=0.95, pa_deg=10.0, x0=-0.10, y0=0.06, name="source"),
        sigma_kms=550.0, z_lens=0.20, z_source=1.50, image="generated"),
}


def data_requirements() -> dict:
    """What a *real* lens sample would have to provide, and what this module can do with it.

    Returned (and printed by :func:`load_lens_sample` when it fails) so the gap is a named list, not a warning:
    each entry is one table of lens models, with the columns the fitter needs and where it is published.
    """
    return {
        "SLACS": {
            "release": "Sloan Lens ACS Survey lens models, released with Auger et al. (2009, ApJ 705, 1099) and "
                       "Auger et al. (2010, ApJ 724, 511); the tables are on the SLACS survey pages",
            "file": "one table of the sample (FITS binary table or CSV) with one row per lens",
            "columns": ["name", "z_lens", "z_source", "theta_E_arcsec", "sigma_kms"],
            "size": "a few hundred kB; the loader accepts 1 kB - 50 MB and rejects anything else",
            "offline": False,
        },
        "Euclid Q1": {
            "release": "Euclid early-release strong-lens catalogue (Euclid Collaboration, 2025), distributed through "
                       "the Euclid Science Archive",
            "file": "the strong-lensing catalogue for a field plus the lens-model table",
            "columns": ["name", "ra_deg", "dec_deg", "z_lens", "z_source", "theta_E_arcsec", "q", "pa_deg"],
            "size": "tens of MB for a field catalogue",
            "offline": False,
        },
        "HST cutouts": {
            "release": "HST/ACS or WFC3 cutouts of the lens fields, MAST",
            "file": "one FITS cutout per lens, same pixel grid as the model",
            "columns": ["image", "pixel_scale_arcsec", "psf_fwhm_arcsec", "noise_per_pixel"],
            "size": "tens of kB per cutout",
            "offline": False,
        },
    }


def load_lens_sample(source: str = "SLACS", path: str | None = None):
    """Load a published lens sample, or fail with the exact list of what is missing.

    The samples named in :func:`data_requirements` are not redistributed with PINNeAPPle and are not downloaded
    here. Without a local table this raises ``NotConfigured`` naming the file, the release and the columns, rather
    than returning an empty sample that a study could mistake for a result.
    """
    from pinneapple_core.errors import NotConfigured

    spec = data_requirements().get(source)
    if spec is None:
        raise NotConfigured(f"unknown lens sample {source!r}",
                            fix=f"use one of {sorted(data_requirements())}, or pass a table yourself")
    if path is None:
        raise NotConfigured(
            f"the {source} lens sample is not bundled and no local file was given",
            fix=(f"point path= at {spec['file']} from {spec['release']}; it must have the columns "
                 f"{', '.join(spec['columns'])} ({spec['size']})"))
    import os

    if not os.path.exists(path):
        raise NotConfigured(f"no lens table at {path}",
                            fix=f"{spec['release']} provides the columns {', '.join(spec['columns'])}")
    size = os.path.getsize(path)
    if not 1024 <= size <= 50 * 1024 * 1024:
        raise NotConfigured(f"{path} has {size} bytes, outside the expected range for a lens table",
                            fix="check that the whole table was downloaded, not an HTML error page")
    text = open(path, "r", errors="replace").read(1 << 20)
    missing = [c for c in spec["columns"] if c not in text]
    if missing:
        raise NotConfigured(f"{path} has no column {', '.join(missing)}",
                            fix=f"the {source} table must carry {', '.join(spec['columns'])}")
    rows = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    return rows
