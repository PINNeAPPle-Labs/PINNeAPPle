"""Transit photometry: the occultation model, a differentiable transit fit, stellar variability, a BLS search and
the vetting checks that decide whether a dip is a planet.

What is modelled, and how each piece is checked:

* **The occultation** -- a star of unit radius with the quadratic limb-darkening law
  ``I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)**2``, blocked by a disk of radius ``p = Rp/R*`` at a sky separation
  ``z``. The blocked flux is the Mandel & Agol (2002) occultation integral, evaluated as a one-dimensional radial
  integral: the circle of radius ``r`` around the star's centre spends an angular measure ``2 theta(r)`` inside the
  planet, so the blocked flux is ``int I(r) 2 theta(r) r dr``. Two things make that exact rather than approximate:
  where the planet's disk contains the star's centre (``r <= p - z``) the whole circle is covered and the integral
  is done in closed form, and the remaining piece is Gauss-Legendre in ``r = lo + (hi - lo)(1 - cos q)/2``, a
  substitution that is quadratic at both contacts and so removes the ``sqrt`` behaviour of ``theta`` there. The
  uniform-disk case has a closed form of its own (:func:`occultation_area`) and the small-planet limit has its own
  series; ``tests/test_astro_transits.py`` checks the integral against the first to 1e-8 and against the second at
  its measured ``O(p**2)`` relative size.
* **The geometry** -- circular orbits only: the sky separation from ``(a/R*, i)``, the total and full durations in
  the closed forms of Winn (2010, eqs. 14-15), and the stellar density a transit implies (Seager &
  Mallen-Ornelas 2003; Winn 2010, eq. 30). The durations are checked against the separation the model evaluates, to
  1e-9 days.
* **The fit** -- :func:`transit_fit` is the differentiable model in torch: ``(period, t0, k, b, T14)`` by Adam, with
  linearised (Fisher, delta-method) uncertainties from the Jacobian at the best fit.
* **The star** -- :func:`stellar_variability` fits a Matern-3/2 Gaussian process by marginal likelihood and returns
  the prediction to subtract. Spots are the physically motivated case for a quasi-periodic kernel; that kernel is
  not implemented here, and the docstring says so.
* **The search** -- :func:`bls` is box least squares (Kovacs, Zucker & Mazeh 2002) in numpy, the baseline every
  transit survey is measured against.
* **The vetting** -- :func:`transit_vetting` returns named flags, each with the physical reason it is a flag, and
  one score: odd-even depth difference at the recovered period, a secondary eclipse, a V-shaped (grazing) eclipse, a
  profile the limb-darkened model cannot reproduce (``chi2/dof`` well above 1, or dimming that continues outside the
  event), a centroid shift during the transit, a depth too deep for a planet, and a stellar density no star has.
  :func:`measurement_check` is the same verdict in the shape ``PhysicsEnsemble`` wants, saying *why*.

Nothing here needs the network. :func:`benchmark` measures recovery and false-positive rates on a *synthetic*
labelled set that this module generates with the labels it wrote itself; the docs page says so, and says what a
real measurement would need -- MAST light curves and the NASA Exoplanet Archive disposition:

* TESS and Kepler light curves: https://mast.stsci.edu/ (``TESS``, ``Kepler`` mission products via the MAST API)
* labels: https://exoplanetarchive.ipac.caltech.edu/ (``PS``/``PSCompPars``, ``koi_disposition``)

:func:`load_light_curve` reads one of those products **from disk** and never downloads it: it names the file and the
release it wanted and raises :class:`LightCurveUnavailable` with the MAST URL when it is not there. astropy is an
optional dependency (``pip install "pinneapple[astro]"``), imported inside the function.

    from pinneapple_physics.astro import transits
    lc = transits.synthetic_light_curve("planet", seed=3)
    found = transits.bls(lc["time"], lc["flux"], lc["flux_err"])
    fit = transits.transit_fit(lc["time"], lc["flux"], lc["flux_err"], period=found["period"])
    transits.transit_vetting(lc["time"], lc["flux"], lc["flux_err"], fit, centroid=lc["centroid"])["score"]
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
from torch.autograd.functional import jacobian

__all__ = [
    "R_SUN_M", "R_EARTH_M", "R_JUP_M", "RHO_SUN_CGS", "KINDS",
    "occultation_area", "flux_of_separation", "transit_model", "impact_parameter", "a_over_rstar",
    "transit_durations", "stellar_density", "transit_fit", "stellar_variability", "bls", "transit_vetting",
    "measurement_check", "MeasurementCheck", "synthetic_light_curve", "analyze", "benchmark", "phase_fold_gif",
    "load_light_curve", "light_curve_path", "LightCurveUnavailable", "LIGHT_CURVE_RELEASES", "MAST_PRODUCT_URL",
    "instrument_systematics",
]

# -------------------------------------------------------------------------------------------------- constants
R_SUN_M = 6.957e8                     # IAU 2015 B3 nominal solar radius
R_EARTH_M = 6.371e6                   # IAU 2015 B3 nominal equatorial Earth radius
R_JUP_M = 7.1492e7                    # IAU 2015 B3 nominal equatorial Jupiter radius
G_SI = 6.67430e-11                    # m^3 kg^-1 s^-2
RHO_SUN_CGS = 1.408                   # g/cm^3, the mean solar density M/(4/3 pi R^3)
RHO_SUN_SI = RHO_SUN_CGS * 1000.0     # kg/m^3
DAY_S = 86400.0

DTYPE = torch.float64
#: The occultation model has square-root behaviour at the four contacts (``mu = sqrt(1 - r^2)``,
#: ``theta = arccos(cos)``), where the derivative is infinite and autograd returns NaN. The clamps stay this far from
#: the singular point: the flux changes by ~1e-6 of itself only at the exact contact, and the derivative stays
#: finite, which is what the delta-method covariance of :func:`transit_fit` needs.
_CONTACT = 1e-12
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)          # nodes/weights of the radial occultation integral
_GL_X = torch.as_tensor(_GL_X, dtype=DTYPE)
_GL_W = torch.as_tensor(_GL_W, dtype=DTYPE)


# -------------------------------------------------------------------------------------------- occultation
def occultation_area(z, p):
    """Area of the overlap of the unit disk and a disk of radius ``p`` whose centres are ``z`` apart.

    Closed form, symmetric in the two radii (so a planet larger than the star is the same problem), e.g. Mandel &
    Agol (2002) appendix:

        A = arccos((z^2 + 1 - p^2) / 2z) + p^2 arccos((z^2 + p^2 - 1) / (2 p z))
            - (1/2) sqrt((1 + p - z)(1 - p + z)(p + 1 - z)(p + 1 + z))

    for ``|1 - p| < z < 1 + p``; ``pi min(p, 1)^2`` where one disk contains the other; 0 where they are apart. This
    is the uniform-disk limit of :func:`flux_of_separation`, and the one part of the transit model that is known
    in closed form.
    """
    z = np.asarray(z, dtype=float)
    p = np.asarray(p, dtype=float)
    z, p = np.broadcast_arrays(z, p)
    out = np.zeros(z.shape, dtype=float)
    contains = z <= np.abs(1.0 - p)
    partial = (z < 1.0 + p) & ~contains
    out[contains] = math.pi * np.minimum(p, 1.0)[contains] ** 2
    if partial.any():
        zp, pp = z[partial], p[partial]
        t1 = np.arccos(np.clip((zp * zp + 1.0 - pp * pp) / (2.0 * zp), -1.0, 1.0))
        t2 = pp * pp * np.arccos(np.clip((zp * zp + pp * pp - 1.0) / (2.0 * zp * pp), -1.0, 1.0))
        prod = np.maximum((1.0 + pp - zp) * (1.0 - pp + zp) * (zp - 1.0 + pp) * (pp + 1.0 + zp), 0.0)
        out[partial] = t1 + t2 - 0.5 * np.sqrt(prod)
    return out


def _inside_disk(R, u1, u2):
    """Flux of a fully covered central disk of radius ``R <= 1``, in closed form:

    ``(1/pi) int_0^R 2 pi r I(r) dr = R^2 - u1 (R^2 - m1) - u2 (R^2 - 2 m1 + m2)`` with
    ``m1 = (2/3)(1 - (1 - R^2)^{3/2})`` and ``m2 = R^2 - R^4/2``; at ``R = 1`` it gives ``1 - u1/3 - u2/6``.
    """
    R2 = R * R
    m1 = (2.0 / 3.0) * (1.0 - torch.clamp(1.0 - R2, min=_CONTACT) ** 1.5)
    m2 = R2 - 0.5 * R2 * R2
    return R2 - u1 * (R2 - m1) - u2 * (R2 - 2.0 * m1 + m2)


def _flux_t(z, p, u1=0.4, u2=0.3):
    """Torch occultation model: ``z`` a 1-D tensor of separations (stellar radii), ``p, u1, u2`` scalars, so a whole
    light curve is one batched evaluation and autograd reaches every parameter."""
    z = torch.as_tensor(z, dtype=DTYPE).reshape(-1)
    p = torch.as_tensor(p, dtype=DTYPE)
    u1 = torch.as_tensor(u1, dtype=DTYPE)
    u2 = torch.as_tensor(u2, dtype=DTYPE)
    total = 1.0 - u1 / 3.0 - u2 / 6.0

    filled = torch.clamp(p - z, min=0.0)                     # disk the planet covers completely
    blocked = _inside_disk(torch.clamp(filled, max=1.0), u1, u2)

    lo = torch.clamp(torch.abs(p - z), max=1.0)              # annulus the planet covers partially
    hi = torch.clamp(z + p, max=1.0)
    span = torch.clamp(hi - lo, min=0.0)
    valid = (hi - lo > 0).to(DTYPE)

    q = 0.5 * math.pi * (_GL_X + 1.0)                        # (n,) in (0, pi)
    r = lo[:, None] + span[:, None] * 0.5 * (1.0 - torch.cos(q))[None, :]
    jac = span[:, None] * 0.5 * torch.sin(q)[None, :] * (0.5 * math.pi) * _GL_W[None, :] / math.pi

    mu = torch.sqrt(torch.clamp(1.0 - r * r, min=_CONTACT))
    inten = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    z_safe = torch.clamp(z, min=1e-12)
    cos_theta = (r * r + z_safe[:, None] ** 2 - p * p) / (2.0 * torch.clamp(r, min=1e-12) * z_safe[:, None])
    theta = torch.where(z[:, None] > 1e-9,
                        torch.arccos(torch.clamp(cos_theta, -1.0 + _CONTACT, 1.0 - _CONTACT)),
                        torch.full_like(cos_theta, math.pi))
    blocked = blocked + valid * torch.sum(jac * 2.0 * theta * r * inten, dim=1)
    return 1.0 - blocked / total


def flux_of_separation(z, p, u1=0.4, u2=0.3):
    """Relative flux of the star when a planet of radius ``p = Rp/R*`` sits at sky separation ``z`` (stellar radii),
    with quadratic limb darkening ``u1, u2``: 1 outside the transit, ``1 - p^2/(1 - u1/3 - u2/6)`` when the planet
    is entirely inside the stellar disk (uniform disk: ``1 - p^2``), 0 when it covers the star."""
    out = _flux_t(z, p, u1, u2).numpy()
    return float(out[0]) if np.ndim(z) == 0 else out


# ----------------------------------------------------------------------------------------------- geometry
def impact_parameter(a_over_rstar, inclination_deg):
    """``b = (a/R*) cos i``: the sky separation at mid-transit, in stellar radii."""
    return float(a_over_rstar) * math.cos(math.radians(float(inclination_deg)))


def a_over_rstar(period_d, density_rho_sun=1.0):
    """``a/R*`` from Kepler's third law for a star of mean density ``density_rho_sun`` times the solar density:
    ``(a/R*)^3 = G P^2 rho / (3 pi)``. For the Sun this gives 4.2 at P = 1 d and 74 at P = 365.25 d."""
    rho = float(density_rho_sun) * RHO_SUN_SI
    return (G_SI * (float(period_d) * DAY_S) ** 2 * rho / (3.0 * math.pi)) ** (1.0 / 3.0)


def transit_durations(period_d, a_over_rstar_, inclination_deg, k):
    """Total duration ``T14`` (first to fourth contact) and full duration ``T23`` (second to third, the flat bottom)
    in days, for a circular orbit (Winn 2010, eqs. 14-15):

        T14 = (P/pi) arcsin[ sqrt((1 + k)^2 - b^2) / ((a/R*) sin i) ]
        T23 = (P/pi) arcsin[ sqrt((1 - k)^2 - b^2) / ((a/R*) sin i) ]

    ``T23`` is 0 for a grazing eclipse (``b > 1 - k``): there is no flat bottom, which is the geometric definition
    of V-shaped. Checked in the tests by evaluating the sky separation at ``T14/2`` and ``T23/2``, which must be
    ``1 + k`` and ``1 - k``.
    """
    a = float(a_over_rstar_)
    k = float(k)
    if a <= 1.0:
        raise ValueError("a/R* must exceed 1 for a transit to happen")
    b = impact_parameter(a, inclination_deg)
    sin_i = math.sin(math.radians(float(inclination_deg)))
    if sin_i <= 0:
        raise ValueError("the inclination must be strictly between 0 and 180 deg")
    t14 = (period_d / math.pi) * math.asin(min(1.0, math.sqrt(max((1.0 + k) ** 2 - b * b, 0.0)) / (a * sin_i)))
    t23_arg = (1.0 - k) ** 2 - b * b
    t23 = 0.0 if t23_arg <= 0 else \
        (period_d / math.pi) * math.asin(min(1.0, math.sqrt(t23_arg) / (a * sin_i)))
    return {"t14_d": float(t14), "t23_d": float(t23), "impact": float(b), "grazing": bool(t23_arg <= 0)}


def stellar_density(period_d, k, b, t14_d):
    """Mean stellar density (g/cm^3) implied by a transit, in the small-angle form of Seager & Mallen-Ornelas
    (2003) / Winn (2010, eq. 30):

        a/R* = (P / (pi T14)) sqrt((1 + k)^2 - b^2) / sin i,    rho* = 3 pi (a/R*)^3 / (G P^2)

    with ``sin i = sqrt(1 - (b/(a/R*))^2)``, iterated twice because ``sin i`` depends on ``a/R*``; the
    approximation is good to a few percent for ``T14 << P``. A grazing fit lands far from any star's density, which
    is what the vetting uses it for.
    """
    k, b = float(k), float(b)
    a = (period_d / (math.pi * float(t14_d))) * math.sqrt(max((1.0 + k) ** 2 - b * b, 0.0))
    for _ in range(2):
        sin_i = math.sqrt(max(1.0 - (b / a) ** 2, 1e-6)) if a > 0 else 1.0
        a = (period_d / (math.pi * float(t14_d))) * math.sqrt(max((1.0 + k) ** 2 - b * b, 0.0)) / sin_i
    return 3.0 * math.pi * a ** 3 / (G_SI * (float(period_d) * DAY_S) ** 2) / 1000.0


def _separation(t, period, t0, k, b, t14):
    """Sky separation (stellar radii) of the linear-in-time transit model, folded over ``period``:

        z(t)^2 = b^2 + (2 (t - t0) / T14)^2 ((1 + k)^2 - b^2)

    which is exact at the four contacts (``z = 1 + k`` at ``t0 +- T14/2``, ``z = b`` at ``t0``) and differs from
    the exact circular orbit by the ``O((T14/P)^2)`` projection curvature in between.
    """
    dt = torch.remainder(torch.as_tensor(t, dtype=DTYPE) - t0 + 0.5 * period, period) - 0.5 * period
    half = 0.5 * torch.clamp(torch.as_tensor(t14, dtype=DTYPE), min=1e-9)
    z2 = b * b + (dt / half) ** 2 * ((1.0 + k) ** 2 - b * b)
    return torch.sqrt(torch.clamp(z2, min=0.0))


def transit_model(time, period, t0, k, b, t14, u1=0.4, u2=0.3):
    """Relative flux of the linear-in-time transit model at ``time`` (days), as a numpy array; the geometry is
    :func:`_separation` and the occultation :func:`flux_of_separation`."""
    z = _separation(torch.as_tensor(np.asarray(time, dtype=float).reshape(-1), dtype=DTYPE),
                    float(period), float(t0), float(k), float(b), float(t14))
    return _flux_t(z, float(k), float(u1), float(u2)).numpy()


# ----------------------------------------------------------------------------------------------------- fit
def _logit(x, lo=1e-4, hi=1.0 - 1e-4):
    x = min(max(float(x), lo), hi)
    return math.log(x / (1.0 - x))


def transit_fit(time, flux, flux_err=None, period=None, t0=None, k=None, b=0.3, t14=None, u1=0.4, u2=0.3,
                fix_period=False, steps=400, lr=0.03, seed=0):
    """Fit the transit model to a detrended, normalised light curve, differentiably, in torch.

    Free parameters: the period (unless ``fix_period``), ``t0``, ``k = Rp/R*``, the impact parameter ``b`` (through
    a sigmoid bounded by ``1 + k``) and the total duration ``T14``, plus one multiplicative level. L-BFGS
    (strong-Wolfe line search) on the chi-square, restarted from a short ladder of impact parameters because
    ``(k, b, T14)`` is close to degenerate -- a shallower transit at a grazing chord looks like a deeper one at
    ``b = 0`` -- and the best chi-square of the starts is kept. The uncertainties are the linearised (Fisher)
    covariance ``(J^T W J)^-1`` at the best fit, inflated by the reduced chi-square, mapped to the physical
    parameters by the delta method on the diagonal -- the matrix is returned so a caller can use the correlations.
    With uniform ``sigma`` this is the usual ``sigma^2 (J^T J)^-1``; putting ``chi2/dof`` there instead of the
    weights is what makes the errors come out 1/sigma too large.

    Returns ``{period, period_err, t0, t0_err, depth, depth_err, rp_rs, rp_rs_err, impact, impact_err,
    duration_d, duration_err_d, level, chi2, dof, snr, stellar_density_rho_sun, r_star_rsun, radius_r_earth,
    radius_err_r_earth, n_points, cov, params}``. ``snr`` is the depth over its fitted uncertainty; the radius
    fields are ``None`` until a caller (or :func:`analyze`) supplies ``r_star_rsun``.

    The model is the linear-in-time geometry of ``_separation``; a transit generated with the exact circular orbit
    over a finite ``T14/P`` differs from it by the projection curvature, which is why the recovery bands in the
    tests are wider than the statistical uncertainties.
    """
    torch.manual_seed(int(seed))
    t = torch.as_tensor(np.asarray(time, dtype=float), dtype=DTYPE)
    y = torch.as_tensor(np.asarray(flux, dtype=float), dtype=DTYPE)
    sig = torch.as_tensor(np.full(len(y), float(np.std(np.asarray(flux, dtype=float))))
                          if flux_err is None else np.asarray(flux_err, dtype=float), dtype=DTYPE)
    w = 1.0 / torch.clamp(sig, min=1e-12) ** 2

    if period is None or t0 is None or k is None or t14 is None:
        found = bls(time, flux, flux_err)
        period = found["period"] if period is None else period
        t0 = found["t0"] if t0 is None else t0
        k = found["sqrt_depth"] if k is None else k
        t14 = found["duration"] if t14 is None else t14
    period, t0 = float(period), float(t0)
    k = min(max(float(k), 1e-3), 0.6)
    t14 = max(float(t14), 1e-4)
    p0 = torch.as_tensor(period, dtype=DTYPE)

    names = ([] if fix_period else ["logP"]) + ["dt0", "logk", "braw", "logt14", "level"]
    idx = {n: i for i, n in enumerate(names)}

    def unpack(th):
        period_m = p0 * torch.exp(th[idx["logP"]]) if not fix_period else p0
        k_m = torch.exp(th[idx["logk"]])
        b_m = torch.sigmoid(th[idx["braw"]]) * (1.0 + k_m)
        return period_m, t0 + th[idx["dt0"]], k_m, b_m, torch.exp(th[idx["logt14"]]), 1.0 + th[idx["level"]]

    def model_of(th):
        period_m, t0_m, k_m, b_m, t14_m, level = unpack(th)
        return level * _flux_t(_separation(t, period_m, t0_m, k_m, b_m, t14_m), k_m, u1, u2)

    best = None
    for b_start in (float(b), 0.0, 0.7):
        x0 = {"logP": 0.0, "dt0": 0.0, "logk": math.log(k), "braw": _logit(b_start / (1.0 + k)),
              "logt14": math.log(t14), "level": 0.0}
        theta = [torch.tensor(x0[n], dtype=DTYPE, requires_grad=True) for n in names]
        opt = torch.optim.LBFGS(theta, lr=1.0, max_iter=int(steps), history_size=20,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            resid = model_of(theta) - y
            loss = torch.sum(w * resid * resid)
            if not torch.isfinite(loss):
                loss = torch.full_like(loss, 1e30)
            loss.backward()
            return loss

        with torch.enable_grad():
            opt.step(closure)
            opt.step(closure)                       # a second epoch: the first can end on a rejected line search
        with torch.no_grad():
            val = float(torch.sum(w * (model_of(theta) - y) ** 2))     # the chi-square *at* the returned point
        if best is None or val < best[0]:
            best = (val, [th.detach().clone() for th in theta])
    if best is None:
        raise RuntimeError("transit fit found no finite chi-square")
    chi2 = best[0]
    theta = [th.detach().requires_grad_(True) for th in best[1]]

    def model_vec(v):
        """The model of a whole light curve from the free parameters packed in one vector, for the Jacobian."""
        return model_of(list(torch.unbind(v)))

    dof = max(len(y) - len(theta), 1)
    reduced_chi2 = chi2 / dof
    packed = torch.stack([th.detach() for th in theta])
    J = jacobian(model_vec, packed, strategy="forward-mode", vectorize=True)   # (N, K): 6 JVPs, not N VJPs
    with torch.no_grad():
        theta = [p.clone().requires_grad_(True) for p in torch.unbind(packed)]
        # (J^T W J)^-1 with W = diag(1/sigma^2), inflated by the reduced chi-square when the fit is poor. The
        # weights belong *inside* the Fisher matrix: ``chi2/dof`` times an unweighted ``(J^T J)^-1`` is in the wrong
        # units and inflates every error by 1/sigma of the photometry (a factor of 4000 at 250 ppm).
        Jw = J / sig[:, None]
        scale = torch.linalg.norm(Jw, dim=0).clamp(min=1e-30)         # the columns differ by decades in scale
        Js = Jw / scale
        cov = reduced_chi2 * torch.linalg.pinv(Js.T @ Js, hermitian=True) / scale[:, None] / scale[None, :]
        # The impact parameter is the flat direction of this model -- a small planet at b = 0 and a slightly larger
        # one at a grazing chord are the same chord, and the Fisher information for b vanishes on the b = 0
        # boundary the fit often lands on. The *depth* is still measured; quoting its marginal error across that
        # degeneracy would call a 5000 ppm transit a 3 sigma detection. So the depth and k carry the conditional
        # (b held at the best fit) error, and ``cov`` keeps the full matrix with the correlation for a caller that
        # wants it.
        keep = [i for i, n in enumerate(names) if n != "braw"]
        sub = Js[:, keep]
        cov_cond = reduced_chi2 * torch.linalg.pinv(sub.T @ sub, hermitian=True)
        pos = {i: j for j, i in enumerate(keep)}
        c_kk = float(cov_cond[pos[idx["logk"]], pos[idx["logk"]]]) / float(scale[idx["logk"]]) ** 2
        period_m, t0_m, k_m, b_m, t14_m, level = unpack(theta)
        rp, bpar, per, dur = float(k_m), float(b_m), float(period_m), float(t14_m)
        sig_k = rp * math.sqrt(max(c_kk, 0.0))
        sig_p = per * math.sqrt(max(float(cov[idx["logP"], idx["logP"]]), 0.0)) if not fix_period else float("nan")
        sig_t0 = math.sqrt(max(float(cov[idx["dt0"], idx["dt0"]]), 0.0))
        sig_t14 = dur * math.sqrt(max(float(cov[idx["logt14"], idx["logt14"]]), 0.0))
        p_b = float(torch.sigmoid(theta[idx["braw"]]))
        sig_b = p_b * (1.0 - p_b) * (1.0 + rp) * math.sqrt(max(float(cov[idx["braw"], idx["braw"]]), 0.0))
    depth = rp * rp
    depth_err = 2.0 * rp * sig_k
    rho = stellar_density(per, rp, bpar, dur) if dur > 0 else float("nan")
    return {"period": per, "period_err": sig_p, "t0": float(t0_m), "t0_err": sig_t0,
            "depth": depth, "depth_err": depth_err, "rp_rs": rp, "rp_rs_err": sig_k,
            "impact": bpar, "impact_err": sig_b, "duration_d": dur, "duration_err_d": sig_t14,
            "level": float(level), "chi2": float(chi2), "dof": int(dof),
            "snr": float(depth / depth_err) if depth_err > 0 else float("inf"),
            "stellar_density_rho_sun": rho / RHO_SUN_CGS if rho == rho else float("nan"),
            "r_star_rsun": None, "radius_r_earth": None, "radius_err_r_earth": None,
            "n_points": int(len(y)), "cov": cov.numpy(), "free": names,
            "params": {"period": per, "t0": float(t0_m), "k": rp, "b": bpar, "t14": dur, "u1": u1, "u2": u2}}


def _radius_r_earth(k, r_star_rsun):
    """``Rp`` in Earth radii from ``k`` and a stellar radius in solar radii."""
    return None if r_star_rsun is None else float(k) * float(r_star_rsun) * R_SUN_M / R_EARTH_M


class MeasurementCheck(NamedTuple):
    """The verdict of the physics check on one light curve, in the shape ``PhysicsEnsemble`` wants.

    ``suspect`` is True when the reading is not a physical transit (the ensemble's ``measurement_check`` contract);
    ``verdict`` names the stage that decided and *why*; ``reasons`` lists the flags that fired, each with its
    physics; ``score`` is the vetting score in [0, 1]. Plug it in as
    ``PhysicsEnsemble(..., measurement_check=lambda q: measurement_check(q).suspect)``.
    """

    suspect: bool
    verdict: str
    score: float
    reasons: tuple[str, ...]
    flags: tuple[dict, ...]
    candidate: dict


def measurement_check(light_curve, result=None, snr_min=7.0, r_star_rsun=None):
    """Is this light curve a planet, and *why*: the physics verdict on one signal.

    Runs :func:`analyze` when ``result`` is not given, then reports the vetting verdict in the shape of
    :class:`MeasurementCheck`. The three outcomes are kept apart on purpose:

    * ``suspect=False`` -- a planet candidate: the search detected it and no check fired;
    * ``suspect=True`` -- a false positive: the checks fired, and ``reasons`` says which and why;
    * ``suspect=True`` with ``verdict`` starting "not detected" -- there is no signal at the threshold, which is a
      rejection even though no vetting check had anything to veto.

    ``centroid`` in the light curve is what enables the blend check; without it that check reports *not run* and the
    verdict says so, rather than passing silently.
    """
    out = analyze(light_curve, seed=0) if result is None else result
    vet = out["vetting"]
    reasons = tuple(f"{f['title']} -- {f['why']}" for f in vet["flags"] if f["fired"])
    skipped = tuple(f["name"] for f in vet["flags"] if not f["detail"].get("run", True))
    verdict = vet["verdict"]
    if skipped and not vet["detected"]:
        verdict = f"{verdict} (checks not run: {', '.join(skipped)})"
    return MeasurementCheck(suspect=not (vet["detected"] and vet["score"] >= 0.5), verdict=verdict,
                            score=float(vet["score"]), reasons=reasons, flags=tuple(vet["flags"]),
                            candidate=dict(vet["candidate"]))


# ------------------------------------------------------------------------------------- stellar variability
def _matern32(t1, t2, amp, ell):
    r = np.abs(np.asarray(t1, dtype=float)[:, None] - np.asarray(t2, dtype=float)[None, :])
    s = math.sqrt(3.0) * r / ell
    return amp * amp * (1.0 + s) * np.exp(-s)


def _matern_draw(time, amp, ell, rng, nugget_frac=0.1):
    """One realisation of the Matern-3/2 process at ``time``, drawn through the eigen-decomposition of the
    correlation matrix.

    The direct route -- ``cholesky(K + jitter) @ normal(n)`` -- is wrong here: at ``ell`` below a few cadences the
    correlation matrix is numerically singular, and the Cholesky factor of an ill-conditioned matrix handed to a
    standard normal returns draws with a standard deviation of hundreds of ppm instead of ``amp``. ``eigh`` on the
    normalised matrix with the negative eigenvalues clipped is stable, and the result has the amplitude and the
    length scale that were asked for.
    """
    t = np.asarray(time, dtype=float)
    n = len(t)
    C = _matern32(t, t, 1.0, ell) + np.diag(np.full(n, float(nugget_frac) ** 2))
    w, V = np.linalg.eigh(C)
    return float(amp) * (V @ (np.sqrt(np.clip(w, 0.0, None)) * rng.standard_normal(n)))


def stellar_variability(time, flux, flux_err=None, max_points=260, n_restarts=2, seed=0, predict_at=None):
    """Matern-3/2 Gaussian process for the star's own variability (granulation, slow drifts), fitted by maximising
    the log marginal likelihood over ``(ln amplitude, ln length scale, ln jitter)`` with L-BFGS.

    Two restarts from different length scales (0.3 d and 3 d), best likelihood kept: the marginal likelihood of a
    Matern kernel is bimodal when a transit is still in the data. The prediction is returned at *every* time in
    ``predict_at`` (default: ``time`` itself, so the caller subtracts ``mean`` and keeps the residuals); :func:`analyze`
    fits on the out-of-transit points and predicts on the whole curve with that argument.

    At most ``max_points`` points enter the likelihood (random subset, seeded): the cost is ``O(n^3)`` in the
    Cholesky and a transit search cannot afford a full 2-minute TESS sector. The kernel a spotted star actually
    wants is quasi-periodic -- a periodic term times the Matern decay -- and a Matern-3/2 fit therefore treats
    rotation as correlated noise rather than as a period; that kernel is not implemented here.
    """
    t = np.asarray(time, dtype=float)
    y = np.asarray(flux, dtype=float)
    sig = np.full_like(y, float(np.std(y))) if flux_err is None else np.asarray(flux_err, dtype=float)
    sig = np.clip(sig, 1e-12, None)
    rng = np.random.default_rng(int(seed))
    sub = np.arange(len(t))
    if len(t) > int(max_points):
        sub = np.sort(rng.choice(len(t), int(max_points), replace=False))
    ts, ys, ss = t[sub], y[sub], sig[sub]
    ymean = float(np.median(ys))
    yr = ys - ymean

    def nll(p):
        amp, ell, jit = math.exp(p[0]), math.exp(p[1]), math.exp(p[2])
        K = _matern32(ts, ts, amp, ell) + np.diag(ss ** 2 + jit ** 2)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e12
        a = np.linalg.solve(L, yr)
        return float(0.5 * a @ a + np.log(np.diag(L)).sum())

    from scipy.optimize import minimize
    best = None
    for ell0 in (0.3, 3.0)[:max(int(n_restarts), 1)]:
        x0 = np.array([math.log(max(np.std(yr), 1e-6)), math.log(ell0), math.log(max(np.median(ss), 1e-6))])
        res = minimize(nll, x0, method="L-BFGS-B",
                       bounds=[(-14.0, 2.0), (math.log(0.02), math.log(50.0)), (math.log(1e-8), math.log(0.1))])
        if best is None or res.fun < best.fun:
            best = res
    amp, ell, jit = (math.exp(v) for v in best.x)
    K = _matern32(ts, ts, amp, ell) + np.diag(ss ** 2 + jit ** 2)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yr))
    at = t if predict_at is None else np.asarray(predict_at, dtype=float)
    mean = ymean + _matern32(at, ts, amp, ell) @ alpha
    return {"mean": mean, "residual": y - (ymean + _matern32(t, ts, amp, ell) @ alpha),
            "amplitude": amp, "length_scale_d": ell, "jitter": jit,
            "log_marginal_likelihood": -float(best.fun), "n_fit": int(len(ts)), "n_points": int(len(t)),
            "predict_at": at}


# --------------------------------------------------------------------------------------------------- search
def bls(time, flux, flux_err=None, period_range=(1.0, 10.0), n_periods=1500,
        durations=(0.03, 0.05, 0.08, 0.12)):
    """Box least squares (Kovacs, Zucker & Mazeh 2002) on a normalised light curve.

    For every trial period and every trial box duration the box is slid over the folded phase, and the statistic is
    the fractional reduction of chi-square it achieves, ``1 - chi2_box / chi2_flat``, from cumulative sums over the
    phase-sorted points: one sort per period and one ``searchsorted`` per duration.

    ``t0`` is the mid-transit time of the first cycle. ``snr`` is the box depth over its uncertainty,
    ``sqrt(1/sum(w_in) + 1/sum(w_out))``. The period grid is logarithmic with ``d ln P = duration_min / baseline``
    -- fine enough that a transit's phase does not drift by more than the shortest box across the baseline.

    Returns ``{period, t0, duration, depth, sqrt_depth, power, snr, periodogram, periods, durations, chi2_flat}``.
    """
    t = np.asarray(time, dtype=float)
    y = np.asarray(flux, dtype=float)
    sig = np.full_like(y, float(np.std(y))) if flux_err is None else np.asarray(flux_err, dtype=float)
    sig = np.clip(sig, 1e-12, None)
    w = 1.0 / sig ** 2
    baseline = float(t.max() - t.min())
    durations = np.atleast_1d(np.asarray(durations, dtype=float))
    lo, hi = float(period_range[0]), min(float(period_range[1]), 0.55 * baseline)
    if hi <= lo:
        raise ValueError("period_range is empty inside the baseline")
    n_periods = int(max(min(n_periods, math.log(hi / lo) / (durations.min() / baseline) + 2), 64))
    periods = np.exp(np.linspace(math.log(lo), math.log(hi), n_periods))

    W = float(w.sum())
    S1, S2 = float(np.sum(w * y)), float(np.sum(w * y * y))
    chi2_flat = S2 - S1 * S1 / W
    power = np.zeros((n_periods, len(durations)))
    best = {"power": -np.inf}
    ny = len(y)
    for i, P in enumerate(periods):
        ph = np.mod(t, P)
        order = np.argsort(ph, kind="stable")
        ph, yy, ww = ph[order], y[order], w[order]
        ph2 = np.concatenate([ph, ph + P])                   # a box may wrap: it is searched over two cycles
        cw = np.concatenate([[0.0], np.cumsum(np.concatenate([ww, ww]))])
        c1 = np.concatenate([[0.0], np.cumsum(np.concatenate([ww * yy, ww * yy]))])
        c2 = np.concatenate([[0.0], np.cumsum(np.concatenate([ww * yy * yy, ww * yy * yy]))])
        starts = np.arange(ny)
        for j, dur in enumerate(durations):
            if dur >= 0.4 * P:
                continue
            ends = np.searchsorted(ph2, ph + dur, side="right")
            n_in = cw[ends] - cw[starts]
            w_out = W - n_in
            ok = (n_in > 0.0) & (w_out > 0.0) & (ends > starts)
            s1 = c1[ends] - c1[starts]
            s2 = c2[ends] - c2[starts]
            mean_in = np.where(ok, s1 / np.maximum(n_in, 1e-30), 0.0)
            mean_out = np.where(ok, (S1 - s1) / np.maximum(w_out, 1e-30), 0.0)
            chi2_box = np.where(ok, (s2 - s1 * mean_in) + ((S2 - s2) - (S1 - s1) * mean_out), chi2_flat)
            pw = np.where(ok, 1.0 - chi2_box / chi2_flat, 0.0)
            power[i, j] = pw.max()
            a = int(np.argmax(pw))
            if pw[a] > best["power"]:
                depth = float(mean_out[a] - mean_in[a])
                var_d = 1.0 / max(float(n_in[a]), 1e-30) + 1.0 / max(float(w_out[a]), 1e-30)
                best = {"power": float(pw[a]), "period": float(P), "duration": float(dur),
                        "phase_start": float(ph[a]), "depth": depth,
                        "snr": float(abs(depth) / math.sqrt(var_d)), "n_in": float(n_in[a])}
    P, dur = best["period"], best["duration"]
    best["t0"] = float(np.mod(best.pop("phase_start") + 0.5 * dur, P))
    best["sqrt_depth"] = math.sqrt(max(best["depth"], 1e-8))
    best.update(periodogram=power, periods=periods, durations=durations, chi2_flat=chi2_flat)
    return best


# -------------------------------------------------------------------------------------------------- vetting
#: Reduced chi-square of the planet fit at and above which the transit model visibly misses the data. A limb-darkened
#: occultation has a flat bottom, a sharp ingress and a return to the continuum at fourth contact; a grazing eclipse, a
#: starspot crossing or a folded systematic does not, and the fit absorbs it into the wrong ``(k, b, T14)``. Because
#: ``(k, b)`` is degenerate -- a V is fitted as a deeper chord across a *larger* companion -- the fitted parameters do
#: not show it; the residual does. The three numbers are the two populations :func:`benchmark` measured: planets land
#: at ``chi2/dof = 1.02-1.28`` and every eclipse false positive above 1.26 on the other side, so the ramp runs from
#: 1.3 to the 1.5 cut. The planet floor above 1.0 is the model's own linear-in-time approximation of the orbit.
_MISFIT_NONE = 1.3
_MISFIT_OK = 1.5
#: Largest share of the event's depth that may remain *outside* the event (band ``T14`` to ``2 T14``). A transit
#: returns to the continuum at fourth contact, so this is ~0.01; a periodic systematic the search folded is not
#: confined to the event at all.
_OUTSIDE_MAX = 0.10
_OUTSIDE_NONE = 0.35
#: Significance (in sigma) at which the centroid check counts as half-convinced. A centroid is a two-axis
#: measurement and it is quoted on a different foot than a photometric depth, so its cut sits at 4 sigma rather than
#: the 3 sigma of the odd/even and secondary checks; that is the "tighter cut" the docstring of
#: :func:`transit_vetting` names.
_CENTROID_Z0 = 4.0

_FLAG_TEXT = {
    "odd_even": ("odd and even eclipses have different depths",
                 "an eclipsing binary at twice the recovered period: in a binary the two eclipses of one orbit are "
                 "the primary and the secondary, and a search reports half the real period"),
    "secondary": ("a second dip at phase 0.5 of the recovered period",
                  "the companion's own light. A planet's secondary eclipse is 1e-5-1e-6 deep; an eclipsing binary "
                  "of the depth the primary allows has one at the percent level"),
    "v_shape": ("no flat bottom: the ingress is a large fraction of the eclipse",
                "a grazing eclipse of a large companion. A planet small enough to be a planet has a flat bottom "
                "(T23/T14 ~ 0.8 at k = 0.1); V-shaped means b > 1 - k, i.e. the companion is large against the "
                "chord it crosses"),
    "shape_inconsistent": ("the dip's profile is not a planet transit's",
                           "a planet's profile is a chord across a limb-darkened disk: flat bottom, sharp ingress, back "
                           "to the continuum at fourth contact, so the limb-darkened model fits it to the noise and "
                           "leaves chi2/dof near 1. A grazing eclipse has a long ingress and no proper flat bottom, a "
                           "starspot crossing is asymmetric, and a folded systematic is a sinusoid: none of them is "
                           "reproduced by the transit model, and the fit hides the mismatch in a larger companion on a "
                           "central chord, so the residual -- not the fitted parameters -- is what says so. The second "
                           "half of the check is dimming that continues at 1-2 T14 outside the event, which an "
                           "occultation by a compact disk cannot produce"),
    "centroid_shift": ("the flux centroid moves during the transit",
                       "the dimming is not from the target star: a background or nearby eclipsing binary inside the "
                       "aperture dilutes its own deep eclipse into a shallow one and pulls the centroid"),
    "depth_too_deep": ("the radius is too large for a planet",
                       "a depth above ~2.5% around a solar-radius star is a companion above ~2.5 Jupiter radii: an "
                       "eclipsing binary or a late M dwarf, not a planet"),
    "stellar_density": ("the implied stellar density is not a star's",
                        "the period, depth, duration and impact parameter fix a/R* and therefore the mean density "
                        "(Seager & Mallen-Ornelas 2003); a grazing or diluted binary fit lands far from the "
                        "0.1-4 solar range of main-sequence stars"),
}


def _evidence(z, z0=3.0, width=0.7):
    """Evidence in [0, 1) that a measurement of significance ``z`` sigma is real, as a logistic step
    ``0.5 * (1 + tanh((z - z0) / (2 width)))`` -- ``1 / (1 + exp(-(z - z0)/width))``.

    Three properties are what make the product ``score = prod(1 - evidence)`` mean "every check found nothing":

    * it is exactly 0.5 at ``z = z0``, so ``fired`` (``evidence >= 0.5``) coincides with "the measurement passed its
      cut" and the score is read on the same scale as the flags;
    * it is ~0 at ``z = 0`` -- 0.014 for ``z0 = 3``. A width comparable to the cut itself (a sigmoid of width 2 gives
      0.18 at ``z = 0``) leaves a *null* measurement carrying 18% evidence, and the three sigma-based checks then cap
      a perfect planet at 0.82^2 * 0.88 = 0.59: no light curve could ever reach the 0.8 "planet candidate" band and
      every planet would read "weak candidate". The width is the cut's third, so a measurement has to be well past
      the cut to count and well short of it to be ignored;
    * it saturates: a 50 sigma measurement is not 20 times as damning as a 5 sigma one, because at that point the
      model, not the noise, is the limit. :func:`transit_vetting` additionally caps every evidence below 0.95, so
      ``score`` never reaches exactly 0 and one fired check cannot hide another.
    """
    x = (float(z) - float(z0)) / float(width)
    if x > 40.0:                     # exp(-40) is already 4e-18; past here the result is 1 to machine precision
        return 1.0
    if x < -40.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-x)))


def _window_mean(flux, err, mask):
    """Weighted mean flux, its uncertainty and the number of points inside a boolean mask."""
    n = int(mask.sum())
    if n < 2:
        return float("nan"), float("nan"), n
    w = 1.0 / np.clip(err[mask], 1e-12, None) ** 2
    return float(np.sum(w * flux[mask]) / np.sum(w)), float(math.sqrt(1.0 / np.sum(w))), n


def _z(value, err):
    return float(abs(value) / err) if err == err and err > 0 else 0.0


def _folded_profile(flux, err, t, period, t0, t14, nbins=16, span=1.2):
    """The folded dip as a binned profile: weighted mean *deficit* below the out-of-transit level in ``nbins`` equal
    phase bins spanning ``+-span T14`` around mid-transit.

    Returns ``(centres, deficit, sigma, base, n_in)``. The profile is what a shape check has to look at: the fitted
    ``(k, b)`` cannot tell a flat bottom from a V, because a V is reproduced almost as well by a deeper chord across
    a *larger* companion -- on the synthetic set of :func:`benchmark` a grazing eclipse is fitted at
    ``k = 0.27, b = 0.27`` instead of the ``b > 1 - k`` it came from. The measured profile does not have that
    degeneracy: the width at half depth over ``T14`` is 0.75-0.85 for a transit and ~0.45 for a triangle, whatever
    the fit says.

    The bin width is ``span T14 / nbins`` ~ 0.15 ``T14``, which at the 30-minute cadence of the synthetic set is one
    to two points per bin in the dip; the width is therefore quoted on a grid of ~0.15 and the check's cut has to
    leave that much room. Empty bins come back as ``nan``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(flux, dtype=float)
    sig = np.clip(np.asarray(err, dtype=float), 1e-12, None)
    dt = np.mod(t - float(t0) + 0.5 * float(period), float(period)) - 0.5 * float(period)
    out = np.abs(dt) > 0.5 * float(t14)
    if int(out.sum()) < 2:
        raise ValueError("no out-of-transit points to measure the profile against")
    w = 1.0 / sig ** 2
    base = float(np.sum(w[out] * y[out]) / np.sum(w[out]))
    edges = np.linspace(-float(span) * float(t14), float(span) * float(t14), int(nbins) + 1)
    inside = np.abs(dt) < float(span) * float(t14)
    which = np.digitize(dt, edges) - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    deficit = np.full(int(nbins), np.nan)
    sigma = np.full(int(nbins), np.nan)
    for b in range(int(nbins)):
        m = inside & (which == b)
        if int(m.sum()) < 1:
            continue
        wb = w[m]
        deficit[b] = base - float(np.sum(wb * y[m]) / np.sum(wb))
        sigma[b] = float(math.sqrt(1.0 / np.sum(wb)))
    return centres, deficit, sigma, base, int((np.abs(dt) < 0.5 * float(t14)).sum())


def _flat_fraction(centres, deficit, t14):
    """Width of the dip at a fraction of its depth, from a :func:`_folded_profile`, by linear interpolation of the
    crossing between the last bin above the level and the first below it on each side of the peak.

    Returns ``(w50_over_t14, w90_over_w50, peak)``: the flat-bottom fraction (0.75-0.85 for a transit, ~0.45 for a
    triangle) and the squareness (width at 90% of the depth over width at 50%, ~0.9 for a flat bottom, ~0.2 for a
    triangle). Interpolating rather than counting bins matters: the outermost-above-bin span overstates the width by
    a whole bin (0.15 ``T14`` here), which is the size of the whole planet-to-V separation.
    """
    c = np.asarray(centres, dtype=float)
    d = np.asarray(deficit, dtype=float)
    finite = np.isfinite(d)
    if not finite.any():
        return float("nan"), float("nan"), float("nan")
    p = int(np.nanargmax(np.where(finite, d, -np.inf)))
    peak = float(d[p])
    if not peak > 0.0:
        return float("nan"), float("nan"), float("nan")

    def side(level, step):
        """Interpolated ``x`` where the profile crosses ``level * peak``, walking out from the peak by ``step``."""
        j = p
        while 0 <= j + step < len(c) and finite[j + step] and d[j + step] >= level * peak:
            j += step
        k = j + step
        if not (0 <= k < len(c)) or not finite[k] or d[k] >= d[j]:
            return float(c[j])                      # the profile never crosses inside the window: a plateau
        frac = (d[j] - level * peak) / (d[j] - d[k])
        return float(c[j] + frac * (c[k] - c[j]))

    def width(level):
        return side(level, +1) - side(level, -1)

    w50, w90 = width(0.5), width(0.9)
    return (w50 / float(t14) if w50 == w50 else float("nan"),
            (w90 / w50) if (w90 == w90 and w50 > 0) else float("nan"), peak)


def _epoch_depths(t, y, err, period, t0, t14, centre_phase=0.0, wing=1.5):
    """Depth of the dip at ``centre_phase`` measured cycle by cycle, each against the out-of-transit points of its
    *own* cycle.

    A single out-of-transit mean for the whole curve is only right when the baseline is constant. A momentum dump
    moves the level of one cycle and not of its neighbour, and a global mean then reports the step as an odd/even
    difference -- in the synthetic set that was the dominant false alarm for real planets. Each epoch here gets its
    own baseline (out-of-transit points within half a period), which is the form the check has to take on real
    photometry.

    Returns ``(epochs, depths, errors)``, numpy arrays; the error of a cycle is the quadrature sum of the scatter
    of its own out-of-transit points and of the transit points, over ``sqrt(n)``.
    """
    centre = t0 + centre_phase * period
    epoch = np.round((t - centre) / period).astype(int)
    phase = np.mod(t - t0 + 0.5 * period, period) - 0.5 * period
    # A cycle's baseline must avoid *both* eclipses: leaving the primary inside it when the secondary is measured
    # dilutes the baseline by the primary's depth and reports a spurious negative secondary.
    quiet = (np.abs(phase) > wing * t14) & (np.abs(np.abs(phase) - 0.5 * period) > wing * t14)
    out_epochs, out_depths, out_errs = [], [], []
    for n in np.unique(epoch):
        cycle = np.abs(t - (centre + n * period)) < 0.5 * period
        if cycle.sum() < 6:
            continue
        dtc = t - (centre + n * period)
        m_in = cycle & (np.abs(dtc) < 0.5 * t14)
        m_out = cycle & quiet
        if m_in.sum() < 2 or m_out.sum() < 5:
            continue
        m_base, e_base, _ = _window_mean(y, err, m_out)
        m_dip, e_dip, _ = _window_mean(y, err, m_in)
        s_out = float(np.std(y[m_out])) / math.sqrt(m_out.sum())
        s_in = float(np.std(y[m_in])) / math.sqrt(m_in.sum())
        out_epochs.append(int(n))
        out_depths.append(m_base - m_dip)
        out_errs.append(math.sqrt(max(e_base, s_out) ** 2 + max(e_dip, s_in) ** 2))
    return np.array(out_epochs, dtype=int), np.array(out_depths), np.array(out_errs)


def transit_vetting(time, flux, flux_err, fit, centroid=None, centroid_err=None, r_star_rsun=1.0, snr_min=7.0):
    """The instrument and physics checks a transit has to pass, as named flags plus one score.

    Each flag carries the physics that makes it a flag (``why``) and the measurement behind it (``value``,
    ``sigma``). The score is ``prod(1 - evidence)``: 1.0 means every check found nothing, 0.0 means at least one
    found it clearly. Each evidence is a fixed function of its significance -- :func:`_evidence`, a logistic step of
    width 0.7 sigma centred on 3 sigma for the odd/even and secondary checks (on 4 sigma for the centroid, which is
    the tighter cut) and a linear ramp for the shape and radius checks -- chosen before the benchmark ran, not
    fitted on it; :func:`benchmark` measures how well the resulting score separates the synthetic planets from the
    synthetic false positives.

    The checks, and what each one is for:

    ``odd_even``        the depth of odd and even cycles differ: an eclipsing binary at twice the reported period
    ``secondary``       a dip at phase 0.5: the companion's own light, which a planet's cannot be at this depth
    ``v_shape``         the fitted ``(k, b)`` leave no flat bottom: a grazing chord across a large companion
    ``shape_inconsistent``  the profile is not a planet transit's: the limb-darkened fit misses it (``chi2/dof``
                        well above 1, which is what a V, a starspot crossing or a curved systematic leaves) or the
                        dimming continues *outside* the event (a periodic systematic folded by the search, not an
                        occultation)
    ``centroid_shift``  the flux centroid moves during the transit: a blended background eclipsing binary
    ``depth_too_deep``  the radius is above what a planet can be
    ``stellar_density`` the implied stellar density is not a star's

    ``centroid`` is the (N, 2) pixel position of the flux centroid on the same cadence as ``time``. Without it the
    centroid check is reported as *not run* (evidence 0), never as a pass.

    Returns ``{flags, score, detected, verdict, candidate}``; ``candidate`` is the planet candidate with its radius,
    period and uncertainties, ready to be reported or rejected.
    """
    t = np.asarray(time, dtype=float)
    y = np.asarray(flux, dtype=float)
    err = np.full_like(y, float(np.std(y))) if flux_err is None else np.asarray(flux_err, dtype=float)
    err = np.clip(err, 1e-12, None)
    P, t0, t14 = float(fit["period"]), float(fit["t0"]), float(fit["duration_d"])
    k, b = float(fit["rp_rs"]), float(fit["impact"])

    dt = np.mod(t - t0 + 0.5 * P, P) - 0.5 * P
    in_transit = np.abs(dt) < 0.5 * t14
    flags = []

    def add(name, value, sigma, evidence, detail=None):
        title, why = _FLAG_TEXT[name]
        flags.append({"name": name, "title": title, "why": why,
                      "value": float(value) if value == value else None, "sigma": float(sigma),
                      "evidence": float(min(max(evidence, 0.0), 0.95)), "fired": bool(evidence >= 0.5),
                      "detail": detail or {}})

    base, base_err, _ = _window_mean(y, err, ~in_transit)

    # --- odd / even at the recovered period, each cycle against its own out-of-transit level
    epochs, depths, derrs = _epoch_depths(t, y, err, P, t0, t14)
    odd_sel, even_sel = np.abs(epochs) % 2 == 1, np.abs(epochs) % 2 == 0
    if odd_sel.sum() >= 1 and even_sel.sum() >= 1:
        d_odd, d_even = float(np.mean(depths[odd_sel])), float(np.mean(depths[even_sel]))
        err_oe = math.sqrt(float(np.sum(derrs[odd_sel] ** 2)) / odd_sel.sum() ** 2
                           + float(np.sum(derrs[even_sel] ** 2)) / even_sel.sum() ** 2)
        z_oe = _z(d_odd - d_even, err_oe)
        add("odd_even", d_odd - d_even, z_oe, _evidence(z_oe),
            {"n_odd": int(odd_sel.sum()), "n_even": int(even_sel.sum()), "depth_odd": d_odd,
             "depth_even": d_even, "baseline": "per cycle"})
    else:
        add("odd_even", float("nan"), 0.0, 0.0, {"run": False, "n_cycles": int(len(epochs))})

    # --- secondary eclipse at phase 0.5, same per-cycle baseline
    s_epochs, s_depths, s_errs = _epoch_depths(t, y, err, P, t0, t14, centre_phase=0.5)
    if len(s_depths) >= 2:
        depth_sec = float(np.mean(s_depths))
        err_sec = max(math.sqrt(float(np.sum(s_errs ** 2))) / len(s_depths),
                      float(np.std(s_depths)) / math.sqrt(len(s_depths)))     # the scatter across cycles is a floor
        z_sec = _z(depth_sec, err_sec)
        add("secondary", depth_sec, z_sec, _evidence(z_sec),
            {"n_secondary": int(len(s_depths)), "depth_secondary": depth_sec})
    else:
        add("secondary", float("nan"), 0.0, 0.0, {"run": False, "n_secondary": int(len(s_depths))})

    # --- shape: no flat bottom (grazing), from the fitted geometry
    t23_arg = (1.0 - k) ** 2 - b * b
    ratio = 0.0 if t23_arg <= 0 else math.sqrt(t23_arg / max((1.0 + k) ** 2 - b * b, 1e-12))
    add("v_shape", ratio, (0.7 - ratio) / 0.1, 0.9 * min(max((0.7 - ratio) / 0.3, 0.0), 1.0),
        {"t23_over_t14": ratio, "grazing": bool(t23_arg <= 0)})

    # --- shape: the data's own profile, independent of the fitted parameters
    m_out, e_out, n_out = _window_mean(y, err, in_transit)                 # the whole event
    m_near, e_near, n_near = _window_mean(y, err, (np.abs(dt) > t14) & (np.abs(dt) < 2.0 * t14))
    d_out = base - m_out
    if n_out < 4 or not (d_out == d_out) or d_out <= 3.0 * max(e_out, e_near, e_out, 1e-12):
        add("shape_inconsistent", float("nan"), 0.0, 0.0,
            {"run": False, "n_outer": n_out, "deficit": d_out,
             "chi2_per_dof": float(fit.get("chi2") or 0.0) / max(int(fit.get("dof") or 1), 1)})
    else:
        centres, profile, profile_sigma, _, _ = _folded_profile(y, err, t, P, t0, t14)
        flat, squareness, peak = _flat_fraction(centres, profile, t14)
        outside = (base - m_near) / d_out                                # ~0 a transit, ~0.5 a sinusoid
        dof = max(int(fit.get("dof") or 1), 1)
        misfit = float(fit.get("chi2") or 0.0) / dof
        # Both sub-tests are ramps whose half point *is* the physical cut, so ``fired`` means "past the cut" and the
        # 0.95 cap is what keeps two fired checks from hiding each other in the product.
        e_misfit = min(max(0.5 * (misfit - _MISFIT_NONE) / (_MISFIT_OK - _MISFIT_NONE), 0.0), 0.95)
        e_outside = min(max(0.5 * (outside - _OUTSIDE_MAX) / (_OUTSIDE_NONE - _OUTSIDE_MAX), 0.0), 0.95)
        add("shape_inconsistent", max(misfit - _MISFIT_OK, outside - _OUTSIDE_MAX),
            max((misfit - _MISFIT_NONE) / 0.1, (outside - _OUTSIDE_MAX) / 0.1),
            max(e_misfit, e_outside),
            {"run": True, "chi2_per_dof": misfit, "misfit_ok": _MISFIT_OK,
             "flat_fraction": flat, "squareness": squareness, "peak_deficit": peak,
             "outside_fraction": outside, "deficit_event": d_out, "n_near": n_near,
             "outside_max": _OUTSIDE_MAX, "n_profile_bins": int(np.isfinite(profile).sum())})

    # --- centroid shift during the transit
    if centroid is None:
        add("centroid_shift", float("nan"), 0.0, 0.0, {"run": False})
    else:
        c = np.asarray(centroid, dtype=float).reshape(-1, 2)
        cerr = np.full(2, float(np.std(c[:, 0]))) if centroid_err is None else \
            np.broadcast_to(np.asarray(centroid_err, dtype=float), (2,)).copy()
        cerr = np.clip(cerr, 1e-9, None)
        n_in, n_out = int(in_transit.sum()), int((~in_transit).sum())
        shift = c[in_transit].mean(axis=0) - c[~in_transit].mean(axis=0)
        err_ax = cerr * math.sqrt(1.0 / max(n_in, 1) + 1.0 / max(n_out, 1))
        z_cen = float(math.sqrt(float(np.sum((shift / err_ax) ** 2))))
        add("centroid_shift", float(np.hypot(*shift)), z_cen,
            _evidence(z_cen, z0=_CENTROID_Z0),
            {"shift_pix": shift.tolist(), "run": True, "n_in": n_in, "n_out": n_out})

    # --- radius, when the caller knows the stellar radius
    rp_earth = _radius_r_earth(k, r_star_rsun)
    if rp_earth is None:
        add("depth_too_deep", float("nan"), 0.0, 0.0, {"run": False})
        rp_jup = float("nan")
    else:
        rp_jup = rp_earth * R_EARTH_M / R_JUP_M
        add("depth_too_deep", rp_jup, (rp_jup - 2.0) / 1.0, 0.9 * min(max((rp_jup - 2.0) / 1.5, 0.0), 1.0),
            {"radius_r_jup": rp_jup, "radius_r_earth": rp_earth})

    # --- stellar density the transit implies
    rho = fit.get("stellar_density_rho_sun")
    rho = float("nan") if rho is None else float(rho)
    if not (rho == rho) or rho <= 0:
        # A check that could not run is reported, never counted: evidence 0, like the centroid and profile checks
        # above. Charging 0.9 here would reject a candidate on a measurement that was never made.
        add("stellar_density", float("nan"), 0.0, 0.0, {"run": False})
        rho = float("nan")
    else:
        dev = abs(math.log10(rho))
        add("stellar_density", rho, dev / 0.1, 0.9 * min(max((dev - 0.4) / 0.6, 0.0), 1.0),
            {"stellar_density_rho_sun": rho})

    score = float(np.prod([1.0 - f["evidence"] for f in flags]))
    snr = float(fit.get("snr") or 0.0)
    detected = bool(snr >= float(snr_min))
    reasons = [f["title"] for f in flags if f["fired"]]
    if not detected:
        verdict = "not detected: the dip is below the signal-to-noise the fit needs"
    elif score < 0.5:
        verdict = "false positive: " + "; ".join(reasons)
    elif score < 0.8:
        verdict = "weak candidate: " + ("; ".join(reasons) if reasons else "no check fired")
    else:
        verdict = "planet candidate: no vetting check fired"
    return {"flags": flags, "score": score, "detected": detected, "verdict": verdict,
            "candidate": {"period_d": P, "period_err_d": float(fit["period_err"]) if "period_err" in fit else
                          float("nan"),
                          "t0": t0, "depth": float(fit["depth"]), "depth_err": float(fit["depth_err"]),
                          "rp_over_rstar": k, "radius_r_earth": rp_earth,
                          "radius_err_r_earth": None if rp_earth is None else
                          float(fit["rp_rs_err"]) * float(r_star_rsun) * R_SUN_M / R_EARTH_M,
                          "duration_d": t14, "impact_parameter": b, "radius_r_jup": rp_jup,
                          "stellar_density_rho_sun": rho, "snr": snr, "score": score}}


# ------------------------------------------------------------------------------------------ synthetic data
KINDS = ("planet", "eb_alternating", "eb_secondary", "eb_grazing", "blend", "systematic_sinusoid")


def _sky_separation(time, period, t0, a, inclination_deg, front_only=True):
    """Exact sky separation (stellar radii) for a circular orbit: ``x = a sin(2 pi dt/P)``,
    ``y = a cos(2 pi dt/P) cos i``, ``z = hypot(x, y)``.

    The sky separation alone cannot tell the two crossings apart -- it is the same at phase 0 and phase 0.5 -- so
    ``front_only`` (the default) returns the separation only while the planet is *in front* of the star
    (``cos(2 pi dt/P) > 0``) and a separation far outside the star for the other half of the orbit. Without it a
    planet transit is drawn twice per orbit, at ``P/2``, and a search recovers half the injected period.
    """
    dt = np.mod(np.asarray(time, dtype=float) - t0 + 0.5 * period, period) - 0.5 * period
    ang = 2.0 * math.pi * dt / period
    z = np.hypot(a * np.sin(ang), a * np.cos(ang) * math.cos(math.radians(inclination_deg)))
    if front_only:
        z = np.where(np.cos(ang) > 0.0, z, 1.0e3)                     # behind the star: an occultation, not a transit
    return z


def _eclipse(time, period, t0, depth, duration, centre_phase, ingress_frac=0.25):
    """A trapezoid eclipse: flat bottom over ``(1 - ingress_frac)`` of its duration, linear ingress and egress.
    ``ingress_frac ~ 0.1`` is a planet-like flat bottom, ``~0.5`` a grazing V."""
    dt = np.mod(np.asarray(time, dtype=float) - (t0 + centre_phase * period) + 0.5 * period, period) - 0.5 * period
    x = np.abs(dt) / (0.5 * duration)                       # 1.0 at the outer contacts
    return depth * np.clip((1.0 - x) / max(ingress_frac, 1e-6), 0.0, 1.0)


#: TESS's orbital period, days: the scattered-light and thermal cycles of a survey light curve repeat with it.
TESS_ORBIT_D = 13.7


def _alternating_eclipse(time, period, t0, depth_a, depth_b, duration, ingress_frac=0.12):
    """A trapezoid eclipse whose depth alternates with the parity of the cycle: the signature of an eclipsing binary
    whose real period is twice the one a search reports. The parity has to be in the *epoch index*; two eclipses
    placed at phases 0 and 1 of the same period land on top of each other and add, which is not a binary at all."""
    dt = np.mod(np.asarray(time, dtype=float) - t0 + 0.5 * period, period) - 0.5 * period
    epoch = np.round((np.asarray(time, dtype=float) - t0) / period).astype(int)
    depth = np.where(epoch % 2 == 0, float(depth_a), float(depth_b))
    x = np.abs(dt) / (0.5 * duration)
    return depth * np.clip((1.0 - x) / max(ingress_frac, 1e-6), 0.0, 1.0)


def _momentum_dump(time, epochs, amplitude, recovery_d):
    """Instrument systematic of a momentum dump: at each thrust epoch the photometry steps down by ``amplitude``
    (fractional) and recovers exponentially with ``recovery_d``, the thermal settle of the detector.

        s(t) = -A exp(-(t - t_i)/tau)   for t >= t_i,  0 before

    A dump is not a dip and not a star: it is one-sided in time, it never repeats at a period, and the step is
    sharp against a recovery many times longer. A box search can still fold several of them into a "period".
    """
    t = np.asarray(time, dtype=float)
    s = np.zeros_like(t)
    for e in np.atleast_1d(np.asarray(epochs, dtype=float)):
        dt = t - float(e)
        after = dt >= 0.0
        s[after] -= float(amplitude) * np.exp(-dt[after] / max(float(recovery_d), 1e-6))
    return s


def _scattered_light(time, amplitude, period_d=TESS_ORBIT_D, t_perigee=0.0):
    """Scattered sunlight from the Earth/Moon limb, brightest at perigee and always additive:

        s(t) = A cos^2(pi (t - t_perigee) / P_orbit)
    """
    return float(amplitude) * np.cos(math.pi * (np.asarray(time, dtype=float) - float(t_perigee))
                                     / float(period_d)) ** 2


def instrument_systematics(time, seed=0, dumps=True, scattered=True, dump_ppm=(200.0, 700.0),
                           recovery_d=(0.15, 0.45), scattered_ppm=(100.0, 400.0), period_d=TESS_ORBIT_D,
                           dump_spacing_d=(2.0, 3.5)):
    """The instrument terms this module defines, as one fractional curve plus the parameters that made it.

    Two systematics, both of them what a photometric survey actually fights:

    * **momentum dumps** (:func:`_momentum_dump`) -- a thruster firing every ``dump_spacing_d`` days steps the
      photometry down by ``dump_ppm`` and the detector settles back in ``recovery_d``;
    * **scattered light** (:func:`_scattered_light`) -- a smooth, strictly positive bump once per spacecraft orbit
      (``period_d``, TESS's 13.7 d by default), brightest at perigee.

    Both are additive in fractional flux. Returns ``{"curve", "momentum_dumps", "dump_ppm", "recovery_d",
    "scattered_ppm", "period_d"}``; every number in it goes into the light curve's truth, so a false positive can be
    traced to the term that made it.
    """
    rng = np.random.default_rng([97, int(seed) % (2 ** 31)])
    t = np.asarray(time, dtype=float)
    lo, hi = float(dump_spacing_d[0]), float(dump_spacing_d[1])
    epochs = []
    if dumps:
        t_now = float(t.min()) + float(rng.uniform(lo, hi))
        while t_now < float(t.max()):
            epochs.append(t_now)
            t_now += float(rng.uniform(lo, hi))
    amp = float(rng.uniform(*dump_ppm)) * 1e-6 if epochs else 0.0
    tau = float(rng.uniform(*recovery_d))
    curve = _momentum_dump(t, epochs, amp, tau) if epochs else np.zeros_like(t)
    scatter = float(rng.uniform(*scattered_ppm)) * 1e-6 if scattered else 0.0
    if scatter:
        curve = curve + _scattered_light(t, scatter, period_d, float(rng.uniform(0.0, period_d)))
    return {"curve": curve, "momentum_dumps": [float(e) for e in epochs], "dump_ppm": 1e6 * amp,
            "recovery_d": tau, "scattered_ppm": 1e6 * scatter, "period_d": float(period_d)}


def synthetic_light_curve(kind="planet", seed=0, baseline_d=18.0, cadence_min=30.0, noise_ppm=250.0,
                          variability_ppm=None, u1=0.4, u2=0.3, systematics=True):
    """One labelled light curve: ``{time, flux, flux_err, centroid, centroid_err, truth}`` with the truth that made
    it.

    ``kind`` is one of :data:`KINDS`:

    * ``planet`` -- a circular orbit over the *exact* sky separation ``(a/R*, i)`` with quadratic limb darkening,
      plus a Matern-3/2 variability realisation and white noise; the centroid is noise only.
    * ``eb_alternating`` -- eclipses at every epoch of the recovered period with alternating depths (a binary whose
      real period is twice the one a search reports): no feature at phase 0.5, so only the odd-even check sees it.
    * ``eb_secondary`` -- a primary at phase 0 and a quarter-depth secondary at phase 0.5 of the same period.
    * ``eb_grazing`` -- a large companion on a grazing chord (``b > 1 - k``): V-shaped and far too deep for a
      planet.
    * ``blend`` -- a deep eclipse of a background star diluted into a shallow one, with the flux centroid moving
      during the transit: the light curve alone says planet, the centroid says otherwise.
    * ``systematic_sinusoid`` -- no eclipse at all: a sinusoidal instrument systematic at a period inside the search
      band (a thermal or pointing cycle at ``P_orbit/2`` or ``P_orbit/3``), which folds and fits like a periodic
      dip. Only the *shape* checks can reject it, and that is what it is in the set for.

    Every curve, whatever the kind, carries the two instrument systematics of :func:`instrument_systematics` -- a
    momentum dump every 2-3.5 d with its thermal recovery and one scattered-light bump per 13.7 d orbit -- unless
    ``systematics=False``.

    The variability is drawn from the GP prior (Cholesky of the Matern-3/2 covariance at the sample times), so a
    light curve has the correlated structure a real star has rather than white noise. Everything is seeded from
    ``(kind, seed)``: the same pair gives the same arrays, bit for bit.

    Sampling defaults to 30 min over 18 d (864 points), not the 2-minute TESS cadence: the benchmark has to stay
    inside a test budget, and the docs say what that costs.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; known: {KINDS}")
    rng = np.random.default_rng([KINDS.index(kind), int(seed) % (2 ** 31)])
    cadence_d = float(cadence_min) / 1440.0
    n = int(float(baseline_d) / cadence_d)
    time = np.arange(n) * cadence_d
    r_star = float(rng.uniform(0.85, 1.15))
    rho_star = float(rng.uniform(0.7, 1.4))                          # in solar densities
    u1 = float(np.clip(rng.normal(u1, 0.05), 0.10, 0.70))
    u2 = float(np.clip(rng.normal(u2, 0.05), 0.00, 0.40))
    period = float(rng.uniform(1.8, 7.5))
    t0 = float(rng.uniform(0.0, period))
    flux = np.ones(n)
    truth = {"kind": kind, "label": 1 if kind == "planet" else 0, "period_d": period, "t0": t0,
             "r_star_rsun": r_star}
    centroid_shift_pix = 0.0

    if kind == "planet":
        k = float(rng.uniform(0.045, 0.13))
        b = float(rng.uniform(0.0, 0.6))
        a = a_over_rstar(period, rho_star)
        inc = math.degrees(math.acos(min(0.999, b / a)))
        dur = transit_durations(period, a, inc, k)
        sep = _sky_separation(time, period, t0, a, inc)
        flux = _flux_t(torch.as_tensor(sep, dtype=DTYPE), k, u1, u2).numpy()
        truth.update(k=k, b=b, a_over_rstar=a, inclination_deg=inc, t14_d=dur["t14_d"], t23_d=dur["t23_d"],
                     depth=k * k, grazing=dur["grazing"], centroid_shift_pix=0.0)
    elif kind == "eb_alternating":
        k1 = float(rng.uniform(0.10, 0.16))
        k2 = k1 * float(rng.uniform(0.55, 0.80))
        duration = float(rng.uniform(0.12, 0.20))
        flux = flux - _alternating_eclipse(time, period, t0, k1 * k1, k2 * k2, duration)
        truth.update(k=k1, b=0.3, depth=k1 * k1, depth_second=k2 * k2, t14_d=duration,
                     centroid_shift_pix=0.0)
    elif kind == "eb_secondary":
        k1 = float(rng.uniform(0.09, 0.13))
        duration = float(rng.uniform(0.12, 0.20))
        flux = flux - _eclipse(time, period, t0, k1 * k1, duration, 0.0, ingress_frac=0.12)
        flux = flux - _eclipse(time, period, t0, 0.25 * k1 * k1, 0.8 * duration, 0.5, ingress_frac=0.12)
        truth.update(k=k1, b=0.3, depth=k1 * k1, depth_secondary=0.25 * k1 * k1, t14_d=duration,
                     centroid_shift_pix=0.0)
    elif kind == "eb_grazing":
        k = float(rng.uniform(0.22, 0.32))
        b = float(rng.uniform(0.86, 0.95))
        duration = float(rng.uniform(0.10, 0.16))
        flux = flux - _eclipse(time, period, t0, k * k, duration, 0.0, ingress_frac=0.5)
        truth.update(k=k, b=b, depth=k * k, t14_d=duration, grazing=True, centroid_shift_pix=0.0)
    elif kind == "systematic_sinusoid":
        # no eclipse: a periodic instrument term, at a harmonic of the spacecraft orbit so it sits in the search band
        harmonic = float(rng.choice([2.0, 3.0]))
        period = TESS_ORBIT_D / harmonic
        t0 = float(rng.uniform(0.0, period))
        amplitude = float(rng.uniform(1500.0, 3500.0)) * 1e-6            # peak-to-peak/2, fractional
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        flux = 1.0 + amplitude * np.sin(2.0 * math.pi * (time - t0) / period + phase)
        truth.update(k=float("nan"), b=float("nan"), depth=amplitude, t14_d=0.5 * period,
                     centroid_shift_pix=0.0, harmonic=harmonic, systematic="sinusoid",
                     systematic_ppm=1e6 * amplitude)
    else:                                                            # blend
        k_bg = float(rng.uniform(0.30, 0.42))
        dilution = float(rng.uniform(8.0, 14.0))
        duration = float(rng.uniform(0.10, 0.16))
        flux = flux - _eclipse(time, period, t0, k_bg * k_bg / dilution, duration, 0.0, ingress_frac=0.12)
        truth.update(k=k_bg / math.sqrt(dilution), b=0.3, depth=k_bg * k_bg / dilution, t14_d=duration,
                     dilution=dilution, background_k=k_bg)
        centroid_shift_pix = 0.09

    if variability_ppm is None:
        variability_ppm = float(rng.uniform(150.0, 800.0))
    amp, ell = variability_ppm * 1e-6, float(rng.uniform(0.7, 3.0))
    flux = flux + _matern_draw(time, amp, ell, rng)
    instr = None
    if systematics:
        instr = instrument_systematics(time, seed=int(seed) + 7 * KINDS.index(kind))
        flux = flux + instr["curve"]
    sigma = float(noise_ppm) * 1e-6
    flux = flux + rng.normal(0.0, sigma, n)
    flux_err = np.full(n, sigma)

    cen_sigma = 0.05
    cen = rng.normal(0.0, cen_sigma, (n, 2))
    if centroid_shift_pix:
        dt = np.mod(time - t0 + 0.5 * period, period) - 0.5 * period
        cen[np.abs(dt) < 0.5 * truth["t14_d"], 1] += centroid_shift_pix
        truth["centroid_shift_pix"] = centroid_shift_pix
    truth.update(variability_ppm=variability_ppm, noise_ppm=float(noise_ppm), baseline_d=float(baseline_d),
                 cadence_min=float(cadence_min),
                 systematics=None if instr is None else
                 {k: v for k, v in instr.items() if k != "curve"})
    return {"time": time, "flux": flux, "flux_err": flux_err, "centroid": cen,
            "centroid_err": np.full(2, cen_sigma), "truth": truth}


# ----------------------------------------------------------------------------------------------- observed data
MAST_PRODUCT_URL = ("https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:{collection}/{product}")
#: Where an observed light curve comes from, per mission and pipeline release. The loader below never downloads: it
#: reads a product that is already on disk and names exactly which one, because a transit search on a file that
#: silently comes from another release is not a measurement.
LIGHT_CURVE_RELEASES = {
    "TESS": {"collection": "TESS-SPOC", "flux": "PDCSAP_FLUX", "time_scale": "BTJD",
             "pattern": "*-s{sector:04d}-{tic:016d}-{cadence}-s_lc.fits",
             "note": "SPOC light curves, named tess{start_utc}-s{sector:04d}-{tic:016d}-{cadence}-s_lc.fits; the "
                     "cadence field is passed through as given and matched loosely when it is not, so a 2-minute and "
                     "a 20-second product of the same sector do not silently trade places. QLP and "
                     "GSFC-ELEANOR-LITE products follow their own naming: pass an explicit path instead"},
    "Kepler": {"collection": "HLSP_Kepler", "flux": "PDCSAP_FLUX", "time_scale": "BKJD",
               "pattern": "kplr{kepler_id:09d}_lc_Q{quarter}.fits",
               "note": "the Kepler pipeline quarterly light curve"},
}


class LightCurveUnavailable(FileNotFoundError):
    """An observed light curve that is not on disk, with the product, the release and the archive URL it would need.

    A ``FileNotFoundError`` subclass so that ``except FileNotFoundError`` -- what a caller plausibly writes around a
    file read -- catches it, while ``str(exc)`` still says which product in which release was wanted and where it
    lives, and that this module does not go and fetch it.
    """


def light_curve_path(target, mission="TESS", sector=None, quarter=None, cadence_code=None, directory=".",
                     release=None):
    """The path of the on-disk product for one target, resolved by the naming convention of the release.

    ``mission="TESS"`` takes the TIC identifier and a ``sector``; the SPOC product is
    ``tess{start_utc}-s{sector:04d}-{tic:016d}-{cadence}-s_lc.fits``. The observation's UTC start time comes from the
    archive and a sector number does not carry it, so the product is found by the rest of its name. ``cadence_code``
    is that product's own 4-digit cadence field, passed through verbatim and matched with a wildcard when it is not
    given: this module does not guess which digits stand for 120 s, and a wrong guess here would read the 20-second
    product as the 2-minute one.

    ``mission="Kepler"`` takes the KIC identifier and a ``quarter``: ``kplr{kepler_id:09d}_lc_Q{quarter}.fits`` is
    fully determined by those two.

    Raises :class:`LightCurveUnavailable` when nothing matches -- naming the directory, the pattern and the release --
    or when several products match, which is the 2-minute/20-second and the original/reprocessed case; then pass
    ``path=`` for the one wanted.
    """
    from pathlib import Path
    root = Path(directory)
    mission = str(mission)
    if mission not in LIGHT_CURVE_RELEASES:
        raise ValueError(f"unknown mission {mission!r}; known: {sorted(LIGHT_CURVE_RELEASES)}")
    info = LIGHT_CURVE_RELEASES[mission]
    if mission == "TESS":
        if sector is None:
            raise ValueError("a TESS search needs the sector, which is part of the product name")
        pattern = info["pattern"].format(sector=int(sector), tic=int(target),
                                         cadence="*" if cadence_code is None else f"{int(cadence_code):04d}")
    else:
        if quarter is None:
            raise ValueError("a Kepler search needs the quarter, which is part of the product name")
        pattern = info["pattern"].format(kepler_id=int(target), quarter=int(quarter))
    hits = sorted(root.glob(pattern)) if root.is_dir() else []
    if not hits:
        where = MAST_PRODUCT_URL.format(collection=info["collection"], product=pattern)
        raise LightCurveUnavailable(
            f"no {mission} light curve for {target} in {root.resolve()}: searched {pattern!r} of the "
            f"{release or info['collection']} release. Download the product from MAST and point directory= at it "
            f"({where}); this module reads files on disk and does not use the network.")
    if len(hits) > 1:
        raise LightCurveUnavailable(
            f"{len(hits)} {mission} products match {pattern!r} in {root.resolve()} (several sectors or a reprocessed "
            f"and an original product): pass path= the one wanted, first candidate {hits[0].name}")
    return hits[0]


def load_light_curve(target=None, mission="TESS", sector=None, quarter=None, cadence_code=None, directory=".",
                     path=None, release=None, column=None, sigma_floor_ppm=0.0):
    """Read one **observed** TESS or Kepler light curve from a FITS product on disk, in the shape :func:`analyze`
    takes: ``{time, flux, flux_err, centroid, centroid_err, truth}``.

    The file is named by :func:`light_curve_path` (or given as ``path``) and read with astropy, which is an optional
    dependency (``pip install "pinneapple[astro]"``); the import is inside the function, so importing this module
    still needs nothing but numpy and torch, and a missing astropy raises an ImportError that says how to get it.

    What is read, and how:

    * ``time`` is the archive's own barycentric time (BTJD for TESS, BKJD for Kepler) rebased to its first cadence,
      in days -- a transit search only needs differences;
    * ``flux`` is the release's systematics-corrected column (``PDCSAP_FLUX`` unless ``column`` names another, e.g.
      ``SAP_FLUX`` or ``KSPSAP_FLUX``), normalised by its median;
    * ``flux_err`` is the matching error column, with ``sigma_floor_ppm`` added in quadrature: the quoted errors of a
      real product do not include the star's own variability, and the vetting's ``chi2/dof`` check reads them
      literally, so a floor is the honest way to say so;
    * ``centroid`` is ``POS_CORR1/2`` (TESS) or ``MOM_CENTR1/2`` (Kepler) -- the motion of the flux centroid in
      pixels, which is what the blend check needs; it is ``None`` when the product has no such column, and the check
      then reports *not run* rather than passing;
    * cadences with a non-zero ``QUALITY`` flag and any non-finite value are dropped.

    ``truth`` carries the provenance and no label: ``{"kind": "observed", "label": None, "period_d": None, ...,
    "source": {...}}``. An observed light curve has no injected answer, which is why :func:`benchmark` cannot use it
    and why the numbers on the docs page are synthetic.
    """
    try:
        from astropy.io import fits
    except Exception as exc:                                            # ImportError, or a broken install of its own
        raise ImportError('reading an observed light curve needs astropy: pip install "pinneapple[astro]" '
                          f"({type(exc).__name__}: {exc})") from exc
    if path is None:
        if target is None:
            raise ValueError("load_light_curve needs either path= or target=")
        path = light_curve_path(target, mission=mission, sector=sector, quarter=quarter, cadence_code=cadence_code,
                                directory=directory, release=release)
    info = LIGHT_CURVE_RELEASES.get(str(mission), {})
    with fits.open(str(path)) as hdul:
        table = None
        for hdu in hdul:
            names = getattr(hdu, "columns", None)
            if names is not None and "TIME" in names.names:
                table = hdu
                break
        if table is None:
            raise LightCurveUnavailable(f"{path} has no binary table with a TIME column: not a light-curve product")
        cols = table.columns.names
        names = {n.upper(): n for n in cols}

        def column_of(*candidates):
            for c in candidates:
                if c.upper() in names:
                    return np.asarray(table.data[names[c.upper()]], dtype=float)
            return None

        time = column_of("TIME")
        flux = column_of(column or info.get("flux", "PDCSAP_FLUX"), "PDCSAP_FLUX", "SAP_FLUX", "FLUX")
        err = column_of((column or info.get("flux", "PDCSAP_FLUX")) + "_ERR", "PDCSAP_FLUX_ERR", "SAP_FLUX_ERR",
                        "FLUX_ERR")
        quality = column_of("QUALITY")
        centroid = column_of("POS_CORR1", "MOM_CENTR1")
        if centroid is not None:
            centroid = np.column_stack([centroid, column_of("POS_CORR2", "MOM_CENTR2")])
        header = table.header
    if time is None or flux is None:
        raise LightCurveUnavailable(f"{path} has no usable TIME/FLUX pair (columns: {list(names)})")
    if err is None:
        err = np.full_like(flux, float(np.nanstd(flux)))
    err = np.sqrt(np.clip(err, 0.0, None) ** 2 + (float(sigma_floor_ppm) * 1e-6) ** 2)

    good = np.isfinite(time) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if quality is not None:
        good &= quality == 0
    if int(good.sum()) < 10:
        raise LightCurveUnavailable(f"{path} has {int(good.sum())} usable cadences after the quality cut: too few "
                                    f"to search")
    level = float(np.nanmedian(flux[good]))
    t = time[good] - float(time[good][0])
    source = {"path": str(path), "mission": mission, "release": release or info.get("collection"),
              "flux_column": column or info.get("flux", "PDCSAP_FLUX"), "time_scale": info.get("time_scale"),
              "n_cadences": int(good.sum()), "n_dropped": int((~good).sum()), "median_flux": level,
              "object": header.get("OBJECT"), "sector": header.get("SECTOR"), "quarter": header.get("QUARTER"),
              "camera": header.get("CAMERA"), "ccd": header.get("CCD")}
    return {"time": t, "flux": flux[good] / level, "flux_err": err[good] / level,
            "centroid": None if centroid is None else centroid[good],
            "centroid_err": None if centroid is None else np.full(2, float(np.nanstd(centroid[good][:, 0]))),
            "truth": {"kind": "observed", "label": None, "period_d": None, "t0": None, "depth": None,
                      "t14_d": None, "r_star_rsun": None, "source": source}}


# ------------------------------------------------------------------------------------------------ pipeline
def analyze(light_curve, seed=0, steps=400, detrend=True):
    """The pipeline a survey runs on one light curve: BLS, GP detrend with the transit masked out, BLS again, the
    differentiable fit, then the vetting checks.

    The two-pass structure is deliberate: the first BLS needs a period only well enough to mask the transits, the
    GP is then fitted on the out-of-transit points alone so it cannot absorb the signal as stellar noise, and the
    second BLS and the fit run on the detrended curve. Where the GP still divides the flux by a local level it takes
    part of the transit with it; the recovery bands in the tests carry that.

    Returns ``{bls, bls_first, fit, vetting, curve, gp}``.
    """
    t, y, e = light_curve["time"], light_curve["flux"], light_curve["flux_err"]
    first = bls(t, y, e)
    dt = np.mod(t - first["t0"] + 0.5 * first["period"], first["period"]) - 0.5 * first["period"]
    outside = np.abs(dt) > 0.75 * first["duration"]
    curve = y.copy()
    gp = None
    if detrend:
        gp = stellar_variability(t[outside], (y - np.median(y[outside]))[outside], e[outside], seed=seed,
                                 predict_at=t)
        curve = y - gp["mean"]
        curve = curve / float(np.median(curve[outside]))
    found = bls(t, curve, e)
    fit = transit_fit(t, curve, e, period=found["period"], t0=found["t0"], k=found["sqrt_depth"],
                      b=0.3, t14=found["duration"], steps=steps, seed=seed)
    r_star = light_curve["truth"].get("r_star_rsun")
    fit["r_star_rsun"] = r_star
    fit["radius_r_earth"] = _radius_r_earth(fit["rp_rs"], r_star)
    fit["radius_err_r_earth"] = None if r_star is None else \
        fit["rp_rs_err"] * r_star * R_SUN_M / R_EARTH_M
    vet = transit_vetting(t, curve, e, fit, centroid=light_curve["centroid"],
                          centroid_err=light_curve["centroid_err"], r_star_rsun=r_star or 1.0)
    return {"bls": found, "bls_first": first, "fit": fit, "vetting": vet, "curve": curve, "gp": gp}


def _auc(scores, labels):
    """Area under the ROC curve by the rank statistic (Mann-Whitney U); no sklearn."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _wilson(k, n, z=1.96):
    """Wilson score interval for ``k`` successes in ``n`` trials, as ``(low, high)`` at the 95% level by default.

    A rate measured on a dozen light curves is not a number to two decimals: the Wilson interval is the honest way
    to say "1 of 3" (0.21 to 0.94 at 95%), it stays inside [0, 1] where the Wald interval does not, and it does not
    collapse to zero width at 0 or n successes, which is exactly the case here.
    """
    n = int(n)
    if n <= 0:
        return (float("nan"), float("nan"))
    p = float(k) / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def benchmark(n_per_kind=3, seed=0, steps=300, **kwargs):
    """Recovery and false-positive rates on a synthetic labelled set, with intervals, and the ROC of the score.

    Three ways of answering "is this a planet?", measured on the same light curves, as a ladder where each rung adds
    one stage:

    * ``bls`` -- the box search alone: a signal is declared when the box depth exceeds 7 sigma. This is the control
      every transit survey is measured against, and a pure search finds *every* eclipse, which is what its
      false-positive rate is.
    * ``bls+fit`` -- the search plus the differentiable physics fit: a transit is declared when the fit returns a
      depth in (0, 1).
    * ``bls+fit+vetting`` -- the same, plus the vetting score at its operating point of ``score >= 0.5``, which is
      where :func:`measurement_check` calls a signal a planet.

    A **recovery** additionally needs the fitted period within 1% and the fitted depth within a factor of two of the
    injected values -- a detection at the wrong period is not a recovery. A **false positive** is any *declaration*
    on a non-planet light curve, with no period condition: an eclipsing binary recovered at twice its reported period
    is still a catalogue entry that would have to be retracted, so requiring the period to match would let the
    hardest cases through. (Requiring it -- as an earlier version of this function did -- hid the alternating binary
    that BLS folds to ``2 P`` and the sinusoid that folds to a harmonic, and reported a false-positive rate half the
    true one.)

    Returns ``{rows, summary, auc, n_cases, n_planet, n_false_positive, by_kind}``. ``summary`` gives, per method,
    ``recovery`` and ``false_positive`` with the counts and the 95% Wilson interval on each. Every number here is a
    *synthetic-set* number: the labels are the ones :func:`synthetic_light_curve` wrote, not an archive's. The docs
    page states what a real measurement would need -- MAST light curves and the NASA Exoplanet Archive disposition.
    """
    rows = []
    for i, kind in enumerate(KINDS):
        for j in range(int(n_per_kind)):
            lc = synthetic_light_curve(kind, seed=1000 * i + j + int(seed), **kwargs)
            out = analyze(lc, seed=j, steps=steps)
            tr, fit, found = lc["truth"], out["fit"], out["bls"]
            period_ok = abs(fit["period"] - tr["period_d"]) / tr["period_d"] < 0.01
            depth_ok = 0.5 < (fit["depth"] / tr["depth"]) < 2.0
            rows.append({"kind": kind, "label": tr["label"], "period_true": tr["period_d"],
                         "period_bls": found["period"], "period_fit": fit["period"],
                         "depth_true": tr["depth"], "depth_fit": fit["depth"], "depth_err": fit["depth_err"],
                         "duration_true": tr["t14_d"], "duration_fit": fit["duration_d"],
                         "radius_r_earth": fit["radius_r_earth"], "period_ok": bool(period_ok),
                         "depth_ok": bool(depth_ok), "snr": fit["snr"], "score": out["vetting"]["score"],
                         "detected": out["vetting"]["detected"], "verdict": out["vetting"]["verdict"],
                         "flags": [f["name"] for f in out["vetting"]["flags"] if f["fired"]]})
    lab = np.array([r["label"] for r in rows])
    det = np.array([r["detected"] for r in rows])
    rec = np.array([r["period_ok"] and r["depth_ok"] for r in rows])
    sane = np.array([0.0 < r["depth_fit"] < 1.0 for r in rows])
    score = np.array([r["score"] for r in rows])
    planet, falsepos = lab == 1, lab == 0

    declared = {"bls": det, "bls+fit": det & sane, "bls+fit+vetting": det & sane & (score >= 0.5)}

    def rate(mask, sel):
        n = int(sel.sum())
        k = int((mask & sel).sum())
        return {"value": (float(k) / n) if n else float("nan"), "k": k, "n": n, "ci95": _wilson(k, n)}

    summary = {name: {"recovery": rate(d & rec, planet), "false_positive": rate(d, falsepos),
                      "n_planet": int(planet.sum()), "n_false_positive": int(falsepos.sum())}
               for name, d in declared.items()}
    by_kind = {kind: {"n": int(sum(r["kind"] == kind for r in rows)),
                      "declared": int(sum(r["detected"] and 0.0 < r["depth_fit"] < 1.0
                                          and r["score"] >= 0.5 for r in rows if r["kind"] == kind))}
               for kind in KINDS}
    return {"rows": rows, "summary": summary, "auc": _auc(score, lab), "n_cases": len(rows),
            "n_planet": int(planet.sum()), "n_false_positive": int(falsepos.sum()), "by_kind": by_kind}


# -------------------------------------------------------------------------------------------------- figure
def phase_fold_gif(path, kind="planet", seed=7, n_frames=36, dpi=90, steps=420):
    """Write the phase-fold GIF: a synthetic light curve being folded on the fitted period, frame by frame, with the
    fitted transit drawn over the fold. Returns the path.

    The case is one :func:`synthetic_light_curve` and the curve over the fold is the model :func:`transit_fit`
    recovered, not a TESS light curve; the docs page says so. Regenerate with
    ``python scripts/astro/make_transit_gif.py PATH``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    lc = synthetic_light_curve(kind, seed=seed)
    out = analyze(lc, seed=1, steps=steps)
    fit = out["fit"]
    t, y, e = lc["time"], out["curve"], lc["flux_err"]
    P = fit["period"]
    phase = (np.mod(t - fit["t0"] + 0.5 * P, P) - 0.5 * P) / P
    order = np.argsort(phase)
    phase, y, e = phase[order], y[order], e[order]
    grid = np.linspace(-0.5, 0.5, 600)
    z = _separation(torch.as_tensor(fit["t0"] + grid * P, dtype=DTYPE), P, fit["t0"], fit["rp_rs"],
                    fit["impact"], fit["duration_d"])
    model = _flux_t(z, fit["rp_rs"], 0.4, 0.3).numpy()

    blue, orange = "#0081be", "#a23a00"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6), dpi=int(dpi))
    writer = PillowWriter(fps=6)
    frames = max(int(n_frames), 6)
    with writer.saving(fig, str(path), int(dpi)):
        for i in range(frames):
            frac = (i + 1) / frames
            n_show = max(int(frac * len(grid)), 2)
            ax1.clear()
            ax1.errorbar(t, y, yerr=e, fmt=".", ms=1.5, lw=0.4, alpha=0.5, color=blue)
            ax1.plot(fit["t0"] + grid[:n_show] * P, model[:n_show], color=orange, lw=1.4)
            ax1.set_xlabel("time [d]")
            ax1.set_ylabel("relative flux (detrended)")
            ax1.set_title(f"detrended light curve, P = {fit['period']:.4f} d")
            ax2.clear()
            keep = n_show * len(phase) // len(grid)
            ax2.errorbar(phase[:keep], y[:keep], yerr=e[:keep], fmt=".", ms=1.8, lw=0.4, alpha=0.5, color=blue)
            ax2.plot(grid[:n_show], model[:n_show], color=orange, lw=1.4,
                     label=f"k = {fit['rp_rs']:.3f}, b = {fit['impact']:.2f}")
            ax2.set_xlim(-0.5, 0.5)
            ax2.set_xlabel("phase")
            ax2.set_ylabel("relative flux")
            ax2.set_title(f"phase fold, depth = {1e6 * fit['depth']:.0f} ppm")
            ax2.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)
    return str(path)
