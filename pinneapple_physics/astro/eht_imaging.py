"""Black-hole imaging as an inverse problem: what the Event Horizon Telescope's own data can decide about an
image, and what only the physics prior can.

The EHT does not take a picture. Every station pair measures one Fourier component of the sky brightness at the
projected baseline, so a night of observations fills a handful of arcs in the (u, v) plane and the image is the
answer to a severely underdetermined inverse problem. :mod:`pinneapple_physics.astro.lensing` gives the forward
physics (null geodesics, the shadow diameter, the EHT comparison); this module builds the measurement side around
it:

* **the array** -- the eight stations of the 2017 campaign, their geodetic coordinates, and the (u, v) tracks that
  Earth rotation draws at 230 GHz (:data:`EHT_STATIONS`, :func:`uv_coverage`);
* **the forward model** -- the geometric crescent that the EHT's own modelling uses (a ring of diameter ``d`` and
  radial width ``w`` whose brightness is modulated by ``1 + a cos(phi - pa)``), whose visibilities are the Bessel
  transform ``2 pi I0 [int r p J0 dr - i a cos(phi_uv - pa) int r p J1 dr]``, checked against a direct discrete
  transform of the rendered image (:func:`crescent_visibilities`, :func:`visibilities_from_image`);
* **the fit**, with and without a Gaussian prior on any parameter (:func:`fit_crescent`), and the paired experiment
  that measures how far each feature moves when the prior is switched on (:func:`prior_dependence_study`);
* **the reproduction** of the published M87* ring diameter with its uncertainty (:func:`reproduce_ring_diameter`)
  and the interval that follows from the Monte Carlo rather than from the linearised covariance, through
  :func:`pinneapple_analysis.uncertainty.conformal.conformal_quantile`.

**What is synthetic and what is published.** The EHT public visibility data are not bundled and are not downloaded
here (:func:`data_requirements` names the release, the files and the size, and :func:`load_visibility_data` refuses
without them). Everything this module fits is *simulated*: visibilities computed from a stated image model on a
stated array with a stated noise model. The published numbers it is compared against -- the 42 +- 3 uas ring of M87*
and the 51.8 +- 2.3 uas ring of Sgr A* (Event Horizon Telescope Collaboration 2019, ApJL 875, L1 and 2022, ApJL
930, L12), and the mass and distance each predicted shadow is computed from -- are inputs, not measurements made
here. The reproduction is therefore a **closure test** of the machinery: it says the forward model, the array model
and the fitter are consistent with each other and land on the published value, and it says nothing new about M87*.

The same caveat applies to the uncertainty. A synthetic data set whose image is exactly in the fitted model family
constrains the ring diameter to a small fraction of a microarcsecond at EHT sensitivity, while the published error
is a few microarcseconds: the published budget is dominated by the spread between analysis methods, not by thermal
noise, and no amount of synthetic noise reproduces it. :func:`reproduce_ring_diameter` reports both numbers side by
side instead of tuning the noise until they agree.

The trust layer that fits here is the uncertainty one: :func:`diameter_interval` turns the Monte Carlo of recovered
diameters into a distribution-free interval with the split-conformal quantile. ``TrustGate`` is not used because it
scores a trained model against its training distribution, and the imaging problem has no training set -- the model
is an analytic image, not a learned one.

    from pinneapple_physics.astro.eht_imaging import observe, uv_coverage, fit_crescent
    uv = uv_coverage(12.39)                       # M87*, dec +12.39 deg, 230 GHz
    obs = observe(uv, seed=4)                     # synthetic visibilities from the published ring
    fit = fit_crescent(obs.visibilities, uv, obs.sigma)
    print(fit.values["diameter_uas"], fit.errors["diameter_uas"])
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .lensing import EHT, MUAS, C, shadow_diameter_rad

__all__ = [
    "GIGALAMBDA", "PC", "M_SUN", "EHT_STATIONS", "EHT_BAND", "EHT_OBJECTS", "SEFD_JY", "DEFAULT_PRIOR",
    "FEATURE_UNITS", "PARAMETERS", "shadow_angular_diameter_uas", "visibility_null_baseline_gl", "station_ecef",
    "shadow_radius_deg", "UVData", "uv_coverage", "sefd_sigma",
    "crescent_visibilities", "crescent_image", "crescent_flux_map", "visibilities_from_image",
    "CrescentTruth", "ImageTruth", "published_crescent", "SyntheticObservation", "observe",
    "CrescentFit", "fit_crescent", "RingReproduction", "reproduce_ring_diameter", "diameter_interval",
    "FeatureShift", "PriorStudy", "prior_dependence_study", "physics_prior",
    "raytraced_ring", "image_ring_diameter", "data_requirements", "load_visibility_data", "report",
]

PC = 3.0856775814913673e22                 # metres
M_SUN = 1.98840987e30                      # kg
G = 6.67430e-11
GIGALAMBDA = 1e9                           # 1 Glambda = 10^9 wavelengths: the unit the EHT plots (u, v) in

# --------------------------------------------------------------------------------------------- the published array
# Geodetic coordinates of the eight stations of the 2017 campaign (EHT Collaboration 2019, ApJL 875, L4, Table 1;
# site coordinates rounded to ~0.001 deg, i.e. to ~100 m, which moves a 230 GHz baseline by ~1e-4 Glambda out of a
# few Glambda). The South Pole station's longitude is degenerate at the pole and any value serves.
EHT_STATIONS = {
    "ALMA": (-23.0290, -67.7540, 5050.0),      # Atacama Large Millimeter/submillimeter Array, Llano de Chajnantor
    "APEX": (-23.0058, -67.7593, 5100.0),      # Atacama Pathfinder Experiment, same plateau
    "JCMT": (19.8228, -155.4761, 4092.0),      # James Clerk Maxwell Telescope, Maunakea
    "SMA": (19.8242, -155.4783, 4080.0),       # Submillimeter Array, Maunakea
    "SMT": (32.7016, -109.8914, 3186.0),       # Submillimeter Telescope, Mount Graham
    "LMT": (18.9850, -97.3147, 4640.0),        # Large Millimeter Telescope, Sierra Negra
    "SPT": (-89.9839, 0.0, 2800.0),            # South Pole Telescope
    "PV": (37.0660, -3.3930, 2850.0),          # IRAM 30 m, Pico Veleta
}

#: Observing band of the 2017 campaign: 1.3 mm, 230 GHz. The noise model below uses the recorded bandwidth and a
#: coherent integration time as stated parameters, not as the paper's measured Tsys.
EHT_BAND = {"frequency_ghz": 230.0, "wavelength_m": C / 230e9, "bandwidth_hz": 2.0e9, "time_s": 10.0}

#: Published ring diameters and the mass / distance their shadows are predicted from. The values are the ones
#: :data:`pinneapple_physics.astro.lensing.EHT` already carries (EHT 2019 VI for M87*; EHT 2022 I and the GRAVITY
#: Collaboration 2019 for Sgr A*), extended with the right ascension and declination that set the (u, v) tracks.
EHT_OBJECTS = {
    "M87*": {"ra_hours": 12.5137, "dec_deg": 12.3911, **EHT["M87*"],
             "source": "EHT Collaboration 2019, ApJL 875, L1 (ring) and L6 (mass); 16.8 Mpc from the surface "
                       "brightness (L5)"},
    "Sgr A*": {"ra_hours": 17.7611, "dec_deg": -28.9362, **EHT["Sgr A*"],
               "source": "EHT Collaboration 2022, ApJL 930, L12 (ring) and L13 (mass); GRAVITY Collaboration 2019 "
                         "for the distance"},
}

#: Per-station system equivalent flux density (Jy), the order of magnitude each site reports at 1.3 mm. These set
#: the thermal noise of a synthetic visibility through the radiometer equation and are a *model* of the array, not
#: the campaign's measured system temperatures.
SEFD_JY = {"ALMA": 90.0, "APEX": 1300.0, "JCMT": 3000.0, "SMA": 700.0, "SMT": 2000.0, "LMT": 600.0,
           "SPT": 1500.0, "PV": 3000.0}

#: The parameters of the geometric crescent, in the order the fitter uses them.
PARAMETERS = ("diameter_uas", "width_uas", "asymmetry", "pa_deg", "flux_jy")

#: The physics prior of the study: the ring diameter sits at the shadow diameter of the published mass and distance
#: (a photon-orbit statement, not a fit), with the spread the published analyses show between methods; the radial
#: width is a few GM/c^2 of emitting gas, centred on the width the published crescent fits report. Both are stated
#: here so a reader can change them and re-run.
DEFAULT_PRIOR = {"width_uas": (8.0, 4.0)}

_BOUNDS = {"diameter_uas": (20.0, 90.0), "width_uas": (0.5, 30.0), "asymmetry": (-0.9, 0.9),
           "pa_deg": (-180.0, 180.0), "flux_jy": (0.05, 2.0)}

_WGS_A = 6378137.0
_WGS_F = 1.0 / 298.257223563


# --------------------------------------------------------------------------------------------- closed forms
def shadow_angular_diameter_uas(mass_msun: float, distance_mpc: float) -> float:
    """Angular diameter (microarcseconds) of a Schwarzschild shadow, ``2 sqrt(27) G M / (c^2 D)``.

    The closed form of :func:`pinneapple_physics.astro.lensing.shadow_diameter_rad`, in the unit the EHT quotes.
    For the published M87* mass and distance it is 39.69 uas -- 0.8 sigma below the measured 42 +- 3 uas ring.
    """
    return float(shadow_diameter_rad(mass_msun * M_SUN, distance_mpc * PC) / MUAS)


def visibility_null_baseline_gl(diameter_uas: float) -> float:
    """Baseline (Glambda) of the first null of a thin ring's visibility amplitude: ``2.4048 / (2 pi R)``.

    A ring of angular radius ``R`` has visibilities ``J0(2 pi b R)``, so the amplitude falls to zero where the
    baseline reaches the first zero of ``J0``; that null is the primary evidence for a ring and it is why the EHT's
    longest baselines (a few Glambda) matter. For the published 42 uas ring it sits at 3.76 Glambda.
    """
    return float(2.404825557695773 / (2.0 * math.pi * diameter_uas / 2.0 * MUAS) / GIGALAMBDA)


# ------------------------------------------------------------------------------------------------ the array
def station_ecef(lat_deg: float, lon_deg: float, height_m: float) -> np.ndarray:
    """Earth-centred, Earth-fixed position (metres) of a geodetic site, on the WGS84 ellipsoid."""
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    e2 = _WGS_F * (2.0 - _WGS_F)
    n = _WGS_A / math.sqrt(1.0 - e2 * math.sin(la) ** 2)
    return np.array([(n + height_m) * math.cos(la) * math.cos(lo),
                     (n + height_m) * math.cos(la) * math.sin(lo),
                     (n * (1.0 - e2) + height_m) * math.sin(la)])


def _enu_axes(lat_deg: float, lon_deg: float) -> np.ndarray:
    """East, north and up unit vectors of a site, as rows, in ECEF coordinates."""
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    east = np.array([-math.sin(lo), math.cos(lo), 0.0])
    north = np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
    up = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    return np.stack([east, north, up])


@dataclass
class UVData:
    """A sampled (u, v) plane: baseline coordinates in Glambda, the scan each sample belongs to, the station pair,
    and the thermal noise of the sample."""
    u: np.ndarray                       # (n_vis,) Glambda, east-west projection
    v: np.ndarray                       # (n_vis,) Glambda, north-south projection
    scan: np.ndarray                    # (n_vis,) int, index of the time sample
    antenna1: np.ndarray                # (n_vis,) int, index into `stations`
    antenna2: np.ndarray                # (n_vis,) int
    sigma: np.ndarray                   # (n_vis,) Jy, thermal noise per visibility
    stations: tuple = ()                # station names, the order `antenna1` / `antenna2` index
    frequency_ghz: float = EHT_BAND["frequency_ghz"]

    @property
    def n_vis(self) -> int:
        return int(self.u.size)

    @property
    def n_scans(self) -> int:
        return int(self.scan.max() + 1) if self.u.size else 0

    @property
    def max_baseline_gl(self) -> float:
        """Longest projected baseline in Glambda; it sets the finest structure the array can see."""
        return float(np.max(np.hypot(self.u, self.v))) if self.u.size else 0.0

    def radial_bins(self, n: int = 12) -> np.ndarray:
        """Mean |V| per baseline-length bin, useful for the amplitude-versus-baseline curve."""
        r = np.hypot(self.u, self.v)
        edges = np.linspace(0.0, r.max() * 1.0001, n + 1)
        return 0.5 * (edges[:-1] + edges[1:])


def uv_coverage(dec_deg: float, ha_hours: tuple = (-3.5, 3.5), n_scans: int = 24, frequency_ghz: float | None = None,
                stations: dict | None = None, min_elevation_deg: float = 10.0, bandwidth_hz: float | None = None,
                time_s: float | None = None) -> UVData:
    """Earth-rotation synthesis of an array over a range of hour angles, for a source at ``dec_deg``.

    For a baseline vector **b** (east, north, up at the array reference, metres) and a source at hour angle ``H``
    and declination ``d``, the projections are ``u = b . e_u / lambda`` and ``v = b . (w x e_u) / lambda`` with
    ``w`` the unit vector to the source and ``e_u`` the east direction, so the visibility phase ``2 pi (u l + v m)``
    is the geometric delay in wavelengths. Only samples with both stations above ``min_elevation_deg`` are kept --
    the horizon is not negotiable -- which is what makes the sampled arcs ragged rather than a clean grid.

    The default eight stations, 24 scans over +-3.5 h and 230 GHz give 327 visibilities on 21 of the 28 baseline
    pairs, out to 7.98 Glambda, which is the shape of the 2017 M87* coverage: ragged arcs, not a grid, because a pair
    only counts when both of its stations are up.
    """
    stations = EHT_STATIONS if stations is None else stations
    freq = EHT_BAND["frequency_ghz"] if frequency_ghz is None else float(frequency_ghz)
    lam = C / (freq * 1e9)
    names = list(stations)
    pos = {k: station_ecef(*stations[k]) for k in names}
    topo = {k: _enu_axes(stations[k][0], stations[k][1]) for k in names}
    unit_up = {k: topo[k][2] for k in names}                       # the local vertical, in ECEF
    # the reference site for the (u, v) frame is the array centroid, which keeps every baseline in one frame
    ref = np.mean([pos[k] for k in names], axis=0)
    lat0 = math.asin(ref[2] / np.linalg.norm(ref))
    lon0 = math.atan2(ref[1], ref[0])
    centre_axes = _enu_axes(math.degrees(lat0), math.degrees(lon0))
    dec = math.radians(dec_deg)
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]

    us, vs, scans, a1, a2 = [], [], [], [], []
    for it, ha in enumerate(np.linspace(ha_hours[0], ha_hours[1], int(n_scans))):
        h = math.radians(ha * 15.0)
        # the source direction in the local east-north-up frame of the array centre
        w = np.array([-math.cos(dec) * math.sin(h),
                      math.sin(dec) * math.cos(lat0) - math.cos(dec) * math.cos(h) * math.sin(lat0),
                      math.sin(dec) * math.sin(lat0) + math.cos(dec) * math.cos(h) * math.cos(lat0)])
        e_u = np.array([1.0, 0.0, 0.0])
        e_v = np.cross(w, e_u)                    # = (0, U_w, -N_w): north, tilted away from the zenith
        # the same direction in ECEF, so each station's own horizon (not the array centre's) decides visibility
        w_ecef = centre_axes.T @ w
        sin_el = np.sin(math.radians(min_elevation_deg))
        above = {k: float(unit_up[k] @ w_ecef) > sin_el for k in names}
        for i, j in pairs:
            if not (above[names[i]] and above[names[j]]):
                continue
            b = pos[names[i]] - pos[names[j]]
            b_enu = centre_axes @ b
            us.append(b_enu @ e_u / lam / GIGALAMBDA)
            vs.append(b_enu @ e_v / lam / GIGALAMBDA)
            scans.append(it)
            a1.append(i)
            a2.append(j)
    u = np.asarray(us)
    v = np.asarray(vs)
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    if not u.size:
        raise ValueError(f"no baseline has both stations above {min_elevation_deg} deg for dec {dec_deg}; "
                         f"check the declination and the hour-angle range")
    bw = float(EHT_BAND["bandwidth_hz"] if bandwidth_hz is None else bandwidth_hz)
    ts = float(EHT_BAND["time_s"] if time_s is None else time_s)
    sefd = np.array([SEFD_JY.get(k, 1000.0) for k in names])
    sigma = sefd_sigma(sefd[a1], sefd[a2], bw, ts)
    return UVData(u=u, v=v, scan=np.asarray(scans), antenna1=a1, antenna2=a2, sigma=sigma, stations=tuple(names),
                  frequency_ghz=freq)


def sefd_sigma(sefd1, sefd2, bandwidth_hz: float = EHT_BAND["bandwidth_hz"], time_s: float = EHT_BAND["time_s"]):
    """Thermal noise of a visibility from the radiometer equation, ``sqrt(SEFD_1 SEFD_2 / (2 B t))`` (Jy).

    ``B`` is the recorded bandwidth and ``t`` the coherent integration time; doubling the pair's sensitivity
    combines the two antennas in quadrature. With the values in :data:`SEFD_JY` and a 2 GHz, 10 s sample the
    shortest baselines carry 2-7 mJy of noise on a 0.6 Jy source.
    """
    return np.sqrt(np.asarray(sefd1, float) * np.asarray(sefd2, float) / (2.0 * float(bandwidth_hz) * float(time_s)))


# ---------------------------------------------------------------------------------------- the crescent model
def crescent_visibilities(params, u, v) -> np.ndarray:
    """Complex visibilities of the geometric crescent, evaluated in closed form.

    The image is a ring of diameter ``d`` and radial Gaussian width ``w`` whose surface brightness is modulated by
    ``1 + a cos(phi - pa)`` -- the family the EHT's geometric modelling uses, with ``a`` standing in for Doppler
    beaming and gravitational lensing. In polar coordinates the Fourier transform separates:

        V(b, phi_b) = F [ S0(b) - i a cos(phi_b - pa) S1(b) ],   Sk = int r p(r) Jk(2 pi b r) dr / int r p(r) dr

    with ``S0`` and ``S1`` Bessel transforms of the radial profile, ``F`` the total flux and ``(b, phi_b)`` the
    polar coordinates of the baseline. The modulation term is purely imaginary and proportional to ``cos`` of the
    angle between the baseline and the bright side: an asymmetry at 90 deg to a baseline leaves that baseline's
    visibility real. The radial integral is a Gauss-Legendre quadrature in ``u = (r - R)/w``, where the profile is
    a Gaussian and the quadrature converges to machine precision with a hundred nodes.

    ``params`` is a mapping or sequence in the order of :data:`PARAMETERS`, with units uas, uas, dimensionless,
    degrees and Jy.
    """
    d, w, a, pa, flux = _as_params(params)
    radius = 0.5 * d * MUAS
    sigma = max(w, 1e-9) * MUAS
    nodes, weights = _legendre(96)
    lo = -min(radius / sigma, 6.0)
    hi = 6.0
    uu = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    ww = 0.5 * (hi - lo) * weights
    r = radius + sigma * uu
    profile = np.exp(-0.5 * uu * uu)
    keep = r > 0.0
    r, ww, profile = r[keep], ww[keep], profile[keep]
    rho = np.hypot(u, v) * GIGALAMBDA
    x = 2.0 * math.pi * rho[:, None] * r[None, :]
    norm = float(np.sum(profile * ww * r))
    j0, j1 = _bessel()
    s0 = np.sum(j0(x) * (profile * ww * r)[None, :], axis=1) / norm
    s1 = np.sum(j1(x) * (profile * ww * r)[None, :], axis=1) / norm
    angle = np.arctan2(v, u) - math.radians(pa)
    return flux * (s0 - 1j * a * np.cos(angle) * s1)


def _legendre(n: int):
    """Cached Gauss-Legendre nodes and weights on [-1, 1]."""
    key = int(n)
    if key not in _LEGENDRE_CACHE:
        nodes, weights = np.polynomial.legendre.leggauss(key)
        _LEGENDRE_CACHE[key] = (nodes, weights)
    return _LEGENDRE_CACHE[key]


_LEGENDRE_CACHE: dict = {}
_BESSEL: list = []


def _bessel():
    """``scipy.special.j0`` and ``j1``, imported once and cached."""
    if not _BESSEL:
        from scipy.special import j0, j1
        _BESSEL.extend([j0, j1])
    return _BESSEL[0], _BESSEL[1]


def _as_params(params) -> tuple:
    """A parameter mapping or sequence to the 5-tuple the model wants."""
    if isinstance(params, dict):
        return tuple(float(params[k]) for k in PARAMETERS)
    p = tuple(float(x) for x in params)
    if len(p) != len(PARAMETERS):
        raise ValueError(f"expected {len(PARAMETERS)} parameters {PARAMETERS}; got {len(p)}")
    return p


def crescent_image(params, n: int = 96, fov_uas: float | None = None, x0_uas: float = 0.0,
                   y0_uas: float = 0.0):
    """Render the crescent on a square grid: the image the visibilities are the transform of.

    Returns ``(image, pixel_uas)`` with the image normalised so that its sum is 1, i.e. a shape, whose transform
    scaled by the total flux is :func:`crescent_visibilities`. The default field is 4 times the ring radius, wide
    enough that the Gaussian ring is not cut by the edge.
    """
    d, w, a, pa, _ = _as_params(params)
    fov = 4.0 * d if fov_uas is None else float(fov_uas)
    pixel = fov / int(n)
    grid = (np.arange(int(n)) - (int(n) - 1) / 2.0) * pixel
    x, y = np.meshgrid(grid, grid)
    r = np.hypot(x - x0_uas, y - y0_uas)
    phi = np.arctan2(y - y0_uas, x - x0_uas)
    profile = np.exp(-0.5 * ((r - 0.5 * d) / max(w, 1e-9)) ** 2)
    image = profile * (1.0 + a * np.cos(phi - math.radians(pa)))
    total = float(image.sum())
    return (image / total if total > 0 else image), pixel


def crescent_flux_map(params, n: int = 96, fov_uas: float | None = None):
    """:func:`crescent_image` scaled to Jy per pixel, so that ``sum * pixel_uas**2`` is the total flux."""
    image, pixel = crescent_image(params, n=n, fov_uas=fov_uas)
    flux = _as_params(params)[4]
    return image * flux, pixel


def visibilities_from_image(image, pixel_uas: float, u, v) -> np.ndarray:
    """Direct discrete Fourier transform of a rendered image at the sampled baselines (Jy).

    ``V = sum_pixels I exp(-2 pi i (u l + v m))`` with ``l, m`` in radians; the image is the sky brightness, so its
    sum is the flux. This is the *definition* of the measurement, used to check the closed form of
    :func:`crescent_visibilities` and to project a ray-traced image onto the array.
    """
    img = np.asarray(image, dtype=float)
    m = img.shape[0]
    if img.shape[0] != img.shape[1]:
        raise ValueError(f"the image must be square; got {img.shape}")
    grid = (np.arange(m) - (m - 1) / 2.0) * float(pixel_uas) * MUAS
    x, y = np.meshgrid(grid, grid)
    phase = np.outer(np.asarray(u, float) * GIGALAMBDA, x.ravel())
    phase += np.outer(np.asarray(v, float) * GIGALAMBDA, y.ravel())
    return np.exp(-2j * math.pi * phase) @ img.ravel()


# ------------------------------------------------------------------------------------- a synthetic observation
class CrescentTruth:
    """A crescent image as the truth behind a synthetic observation."""

    kind = "crescent"

    def __init__(self, params):
        self.params = dict(zip(PARAMETERS, _as_params(params), strict=True))

    def visibilities(self, u, v) -> np.ndarray:
        return crescent_visibilities(self.params, u, v)

    def __repr__(self) -> str:
        p = self.params
        return (f"CrescentTruth(diameter={p['diameter_uas']:.2f} uas, width={p['width_uas']:.2f} uas, "
                f"asymmetry={p['asymmetry']:.2f}, pa={p['pa_deg']:.1f} deg, flux={p['flux_jy']:.3f} Jy)")


class ImageTruth:
    """Any square image (a ray-traced disc, say) as the truth behind a synthetic observation."""

    kind = "image"

    def __init__(self, image, pixel_uas: float, flux_jy: float = 1.0, name: str = ""):
        img = np.asarray(image, dtype=float)
        total = float(img.sum())
        self.image = img / total if total else img
        self.pixel_uas = float(pixel_uas)
        self.flux_jy = float(flux_jy)
        self.name = name

    def visibilities(self, u, v) -> np.ndarray:
        return self.flux_jy * visibilities_from_image(self.image, self.pixel_uas, u, v)

    def __repr__(self) -> str:
        return f"ImageTruth({self.name or 'image'}, {self.image.shape[0]}px, {self.pixel_uas:.3f} uas/px)"


@dataclass
class SyntheticObservation:
    """Synthetic visibilities, the truth that generated them and the noise that was added."""
    visibilities: np.ndarray
    clean: np.ndarray
    truth: object
    sigma: np.ndarray                  # per-visibility noise, thermal and calibration in quadrature
    sigma_thermal: float
    gain_error: float
    phase_error_rad: float
    seed: int | None = None

    @property
    def snr(self) -> float:
        """Median signal-to-noise of a visibility amplitude -- the number that decides what the data can say."""
        amp = np.abs(self.clean)
        return float(np.median(amp / np.asarray(self.sigma))) if np.all(np.asarray(self.sigma) > 0) else float("inf")


def observe(uv: UVData, truth=None, seed: int | None = 0, sigma_scale: float = 1.0, gain_error: float = 0.05,
            phase_error_rad: float = 0.10) -> SyntheticObservation:
    """Simulate a night of visibilities: the truth's transform, thermal noise, and per-station gain and phase
    errors.

    The corruptions are the reason a real interferometer's error budget is not thermal: every visibility is
    multiplied by ``(1 + dA_i)(1 + dA_j) exp(i (dphi_i - dphi_j))`` with one amplitude ``dA`` and one phase ``dphi``
    drawn per station per scan, so a station's error is shared by all the pairs it takes part in. They are drawn but
    not fitted, so the error they inject is carried in ``sigma`` -- the thermal noise and the calibration term
    ``|V| sqrt(2 g_A^2 + 2 g_phi^2)`` in quadrature -- rather than being hidden inside a fit that would then claim a
    chi-square of one.

    The default truth is the published M87* crescent (:func:`published_crescent`). ``sigma_scale`` multiplies the
    thermal noise of every sample, which is how the study below degrades the data: the thermal noise of a single
    visibility falls as ``1/sqrt(t)``, so ``sigma_scale = 10`` is a campaign with a hundredth of the integration
    time.
    """
    truth = published_crescent() if truth is None else truth
    rng = np.random.default_rng(seed)
    clean = truth.visibilities(uv.u, uv.v)
    sigma_thermal = float(np.mean(uv.sigma)) * float(sigma_scale)
    noise = sigma_thermal / math.sqrt(2.0) * (rng.standard_normal(uv.n_vis) + 1j * rng.standard_normal(uv.n_vis))
    data = clean + noise
    if gain_error or phase_error_rad:
        n_st = max(int(uv.antenna1.max()) + 1, int(uv.antenna2.max()) + 1)
        d_a = rng.normal(0.0, gain_error, (uv.n_scans, n_st)) if gain_error else np.zeros((uv.n_scans, n_st))
        d_p = rng.normal(0.0, phase_error_rad, (uv.n_scans, n_st)) if phase_error_rad else np.zeros((uv.n_scans, n_st))
        gain = ((1.0 + d_a[uv.scan, uv.antenna1]) * (1.0 + d_a[uv.scan, uv.antenna2])
                * np.exp(1j * (d_p[uv.scan, uv.antenna1] - d_p[uv.scan, uv.antenna2])))
        data = data * gain
    var_frac = 2.0 * float(gain_error) ** 2 + 2.0 * float(phase_error_rad) ** 2
    sigma = np.sqrt(sigma_thermal ** 2 + np.abs(clean) ** 2 * var_frac)
    return SyntheticObservation(visibilities=data, clean=clean, truth=truth, sigma=sigma,
                                sigma_thermal=sigma_thermal, gain_error=gain_error,
                                phase_error_rad=phase_error_rad, seed=seed)


def published_crescent(object_name: str = "M87*") -> CrescentTruth:
    """The crescent the reproduction is built from: the object's published ring diameter, a 19 uas FWHM ring
    (20 uas, the width the M87* geometric fits report), an asymmetry amplitude of 0.4 and a compact flux density of
    0.6 Jy, with the bright side at position angle 90 deg.

    These are the inputs of the closure test. The diameter and the flux are the published ones; the width, the
    asymmetry and its direction are the order of magnitude the published images show, stated here so that a reader
    can see exactly what was injected.
    """
    obj = EHT_OBJECTS[object_name]
    diameter = float(obj["ring_uas"])
    return CrescentTruth({"diameter_uas": diameter, "width_uas": diameter / 5.25, "asymmetry": 0.4, "pa_deg": 90.0,
                          "flux_jy": 0.6})


# ----------------------------------------------------------------------------------------------- the inverse
@dataclass
class CrescentFit:
    """A crescent recovered from visibilities: the values, their linearised errors, and how well the model fits."""
    values: dict
    errors: dict
    covariance: np.ndarray
    chi2: float
    dof: int
    prior: dict = field(default_factory=dict)
    phases: bool = True
    n_vis: int = 0

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / self.dof if self.dof else float("nan")

    def significance_from(self, other: CrescentFit, name: str) -> float:
        """How far a parameter of ``other`` is from this fit, in units of the combined error."""
        s = math.hypot(self.errors[name], other.errors[name])
        return (self.values[name] - other.values[name]) / s if s > 0 else float("nan")

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={self.values[k]:.3f}+-{self.errors[k]:.3f}" for k in PARAMETERS)
        return f"CrescentFit({body}, chi2/dof={self.reduced_chi2:.2f})"


def _starting_point(data) -> np.ndarray:
    """A start that is not the answer: 45 uas diameter (between the two values the study compares), 8 uas width,
    a mild asymmetry, no preferred direction, and the flux the zero-baseline amplitude implies."""
    return np.array([45.0, 8.0, 0.2, 0.0, float(np.abs(data).max())])


def fit_crescent(visibilities, uv: UVData, sigma, prior: dict | None = None, phases: bool = True,
                 p0=None) -> CrescentFit:
    """Recover the crescent from visibilities by bounded least squares, optionally with a Gaussian prior.

    ``prior`` maps a parameter name to ``(mean, sigma)``; each entry adds one residual row ``(p - mean) / sigma``
    next to the data rows, which is exactly a Gaussian prior written as a pseudo-observation -- so switching the
    prior on and off is a change of one argument and nothing else. ``phases=False`` fits the amplitudes only, which
    is the situation the EHT's geometric modelling was in for M87*: at 1.3 mm the atmospheric phase is not measured
    absolutely, so the amplitudes carry the ring and the phases carry the asymmetry.

    Errors are the linearised ``sqrt(diag((J^T J)^-1))`` with ``sigma`` the known noise of a visibility; that
    approximation is what :func:`prior_dependence_study` checks against a Monte Carlo, and where it fails (a
    reflection of the asymmetry that the amplitudes cannot tell from the original) the fit reports its formal error
    and the study reports the bias next to it.
    """
    from scipy.optimize import least_squares

    data = np.asarray(visibilities, dtype=complex)
    if data.size != uv.n_vis:
        raise ValueError(f"{data.size} visibilities for {uv.n_vis} baselines")
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), data.shape) if np.ndim(sigma) else float(sigma)
    prior = dict(DEFAULT_PRIOR if prior is None else prior)
    u, v = uv.u, uv.v
    lo = np.array([_BOUNDS[k][0] for k in PARAMETERS])
    hi = np.array([_BOUNDS[k][1] for k in PARAMETERS])

    start = _starting_point(data)
    if p0 is not None:
        start = np.asarray([float(x) for x in ([p0[k] for k in PARAMETERS] if isinstance(p0, dict) else p0)])
    start = np.clip(start, lo * 1.001, hi * 0.999)

    def residuals(p):
        model = crescent_visibilities(p, u, v)
        if phases:
            rows = [(model.real - data.real) / sigma, (model.imag - data.imag) / sigma]
        else:
            rows = [(np.abs(model) - np.abs(data)) / sigma]
        out = np.concatenate(rows)
        extra = [(p[PARAMETERS.index(k)] - mu) / sd for k, (mu, sd) in prior.items()]
        return np.concatenate([out, extra]) if extra else out

    sol = least_squares(residuals, start, bounds=(lo, hi), x_scale="jac", max_nfev=300)
    p = np.clip(sol.x, lo, hi)
    n_data = 2 * uv.n_vis if phases else uv.n_vis
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac)
        err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:                                    # pragma: no cover - degenerate start
        cov = np.full((len(p), len(p)), np.nan)
        err = np.full(len(p), np.nan)
    chi2_all = float(np.sum(sol.fun ** 2))
    chi2_prior = float(sum(((p[PARAMETERS.index(k)] - mu) / sd) ** 2 for k, (mu, sd) in prior.items()))
    return CrescentFit(values=dict(zip(PARAMETERS, p.tolist(), strict=True)),
                       errors=dict(zip(PARAMETERS, err.tolist(), strict=True)), covariance=cov,
                       chi2=chi2_all - chi2_prior, dof=max(n_data - len(p), 1), prior=prior, phases=phases,
                       n_vis=uv.n_vis)


def _dataset(object_name: str = "M87*", n_scans: int = 24, ha_hours: tuple = (-3.5, 3.5), **kw) -> UVData:
    """The (u, v) coverage of the object, cached: the study runs the same array many times."""
    key = (object_name, int(n_scans), tuple(ha_hours))
    if key not in _UV_CACHE:
        if len(_UV_CACHE) > 8:
            _UV_CACHE.pop(next(iter(_UV_CACHE)))
        _UV_CACHE[key] = uv_coverage(EHT_OBJECTS[object_name]["dec_deg"], ha_hours=ha_hours, n_scans=n_scans, **kw)
    return _UV_CACHE[key]


_UV_CACHE: dict = {}


def physics_prior(object_name: str = "M87*", diameter_sigma_uas: float = 4.0, width_uas: float = 8.0,
                  width_sigma_uas: float = 4.0) -> dict:
    """The physics prior of the study: the ring sits at the shadow diameter of the published mass and distance,
    and its radial width is a few GM/c^2.

    The diameter prior is a *prediction* -- :func:`shadow_angular_diameter_uas` of the published mass and distance,
    no fit involved -- with a stated spread, so a reader can see the number the prior contributes. The width prior
    is the range the published crescent fits report.
    """
    obj = EHT_OBJECTS[object_name]
    predicted = shadow_angular_diameter_uas(obj["mass_msun"], obj["distance_pc"] / 1e6)
    return {"diameter_uas": (predicted, float(diameter_sigma_uas)), "width_uas": (float(width_uas),
                                                                                 float(width_sigma_uas))}


# -------------------------------------------------------------------------------------- the reproduction
@dataclass
class RingReproduction:
    """The published ring diameter, the diameter this machinery recovers from a synthetic data set built on it,
    and the interval that separates them -- reproduced versus declared, side by side."""
    object: str
    published_uas: float
    published_err_uas: float
    predicted_uas: float
    recovered_uas: float
    recovered_err_uas: float
    conformal_half_width_uas: float
    samples_uas: np.ndarray
    flux_jy: float
    fit: CrescentFit
    observation: SyntheticObservation
    n_realizations: int

    @property
    def publication_sigma(self) -> float:
        """``(recovered - published) / published_err``: how far the closure test lands from the printed number."""
        return float((self.recovered_uas - self.published_uas) / self.published_err_uas)

    @property
    def prediction_sigma(self) -> float:
        """``(recovered - predicted) / recovered_err``: the GR prediction against what the data gave."""
        return (float((self.recovered_uas - self.predicted_uas) / self.recovered_err_uas)
                if self.recovered_err_uas > 0 else float("nan"))

    def table(self) -> list[dict]:
        """One row per number, with the provenance of each: measured here, published, or predicted."""
        return [
            {"quantity": "ring diameter", "value_uas": self.recovered_uas, "err_uas": self.recovered_err_uas,
             "provenance": f"recovered here from {self.n_realizations} synthetic "
                           f"{self.observation.truth.kind} data sets"},
            {"quantity": "ring diameter (conformal 95 %)", "value_uas": self.recovered_uas,
             "err_uas": self.conformal_half_width_uas,
             "provenance": "distribution-free interval over the Monte Carlo (split conformal)"},
            {"quantity": "ring diameter", "value_uas": self.published_uas, "err_uas": self.published_err_uas,
             "provenance": f"published: {EHT_OBJECTS[self.object]['source']}"},
            {"quantity": "shadow diameter", "value_uas": self.predicted_uas, "err_uas": float("nan"),
             "provenance": "2 sqrt(27) GM/c^2 D from the published mass and distance (no fit)"},
        ]


def diameter_interval(samples, center: float | None = None, alpha: float = 0.05) -> float:
    """Distribution-free half-width of ``alpha`` around the median of ``samples``, by the split-conformal
    quantile of the absolute deviations.

    The linearised error of :func:`fit_crescent` assumes the estimator is Gaussian near its minimum; the Monte
    Carlo does not. ``conformal_quantile`` takes the ``ceil((n+1)(1-alpha))``-th smallest deviation, which covers a
    new realization with probability at least ``1 - alpha`` for any distribution, provided ``n`` is large enough
    (``conformal_quantile_valid``: at least 19 realizations for 95 %). Below that the interval is clipped to the
    largest deviation and the guarantee no longer holds, which the reproduction reports rather than hides.
    """
    from pinneapple_analysis.uncertainty.conformal import (
        conformal_quantile,
        conformal_quantile_valid,
    )

    x = np.asarray(samples, dtype=float).ravel()
    mu = float(np.median(x)) if center is None else float(center)
    half = conformal_quantile(np.abs(x - mu), alpha)
    if not conformal_quantile_valid(x.size, alpha):                   # pragma: no cover - small Monte Carlo
        half = float(np.max(np.abs(x - mu)))
    return float(half)


def reproduce_ring_diameter(object_name: str = "M87*", n_realizations: int = 25, sigma_scale: float = 1.0,
                            gain_error: float = 0.05, phase_error_rad: float = 0.10, seed: int = 0,
                            truth=None) -> RingReproduction:
    """Reproduce the object's published ring diameter from synthetic visibilities, with an uncertainty.

    ``n_realizations`` independent noise draws are fitted; the reported diameter is their mean, its error the
    standard error of that mean, and :func:`diameter_interval` adds the distribution-free 95 % interval. The
    predicted shadow diameter is computed from the published mass and distance with no fit at all, so the
    comparison "ring = shadow, within the published error" is a real statement about the published numbers, while
    the recovered diameter is a closure test of this machinery on data that were generated from it
    (:func:`published_crescent` says what was injected).
    """
    uv = _dataset(object_name)
    truth = published_crescent(object_name) if truth is None else truth
    samples, first, first_obs = [], None, None
    for k in range(int(n_realizations)):
        obs = observe(uv, truth=truth, seed=seed + k, sigma_scale=sigma_scale, gain_error=gain_error,
                      phase_error_rad=phase_error_rad)
        fit = fit_crescent(obs.visibilities, uv, obs.sigma, prior={})
        samples.append(fit.values["diameter_uas"])
        if first is None:
            first, first_obs = fit, obs
    x = np.asarray(samples)
    obj = EHT_OBJECTS[object_name]
    mean = float(x.mean())
    err = float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else float(first.errors["diameter_uas"])
    return RingReproduction(object=object_name, published_uas=float(obj["ring_uas"]),
                            published_err_uas=float(obj["ring_err_uas"]),
                            predicted_uas=shadow_angular_diameter_uas(obj["mass_msun"], obj["distance_pc"] / 1e6),
                            recovered_uas=mean, recovered_err_uas=err,
                            conformal_half_width_uas=diameter_interval(x, mean), samples_uas=x,
                            flux_jy=first.values["flux_jy"], fit=first, observation=first_obs,
                            n_realizations=int(n_realizations))


# --------------------------------------------------------------------------------------- the prior experiment
@dataclass
class FeatureShift:
    """One image feature, fitted with the physics prior off and on: how far the prior moves it, and whether that
    move is larger than the uncertainty that the data leave on it."""
    name: str
    unit: str
    off_mean: float
    on_mean: float
    shift: float                       # on - off, paired over the same noise realizations
    shift_err: float                   # standard error of the paired shift
    off_sd: float
    on_sd: float

    @property
    def significance(self) -> float:
        """``|shift| / shift_err``: how many sigma the prior moves the feature. The paired design removes the noise
        common to both fits, so this answers "is there any effect at all", not "does the effect matter"."""
        return abs(self.shift) / self.shift_err if self.shift_err > 0 else float("inf")

    @property
    def moved(self) -> float:
        """``|shift| / off_sd``: how much of the feature's own uncertainty the prior accounts for. This is the
        yardstick the verdict uses -- an analysis that quoted the prior-on value would be wrong by this fraction of
        the error bar if the prior were wrong."""
        return abs(self.shift) / self.off_sd if self.off_sd > 0 else float("inf")

    @property
    def shrink(self) -> float:
        """``off_sd / on_sd``: how much the prior tightened the estimate (1 means it bought no precision)."""
        return self.off_sd / self.on_sd if self.on_sd > 0 else float("inf")

    @property
    def verdict(self) -> str:
        """Three labels from the two measured numbers, with the thresholds stated rather than derived:

        * ``prior-driven`` -- the prior moves the feature by at least half of the feature's own uncertainty
          (``moved >= 0.5``), so the answer is no longer the data's;
        * ``prior-influenced`` -- the move is smaller, but the prior bought at least 1.5x precision
          (``shrink >= 1.5``), so the error bar is partly the prior's;
        * ``data-determined`` -- neither: the prior neither moves the answer nor tightens it.

        Both numbers are printed next to the label so a reader can apply a different threshold.
        """
        if self.moved >= 0.5:
            return "prior-driven"
        if self.shrink >= 1.5:
            return "prior-influenced"
        return "data-determined"

    def row(self) -> dict:
        return {"feature": self.name, "unit": self.unit, "prior off": self.off_mean, "prior off sd": self.off_sd,
                "prior on": self.on_mean, "prior on sd": self.on_sd, "shift": self.shift,
                "shift err": self.shift_err, "moved (shift/sd)": self.moved, "shrink (sd ratio)": self.shrink,
                "significance": self.significance, "verdict": self.verdict}


@dataclass
class PriorStudy:
    """The whole experiment: one :class:`FeatureShift` per image feature, at one data quality."""
    object: str
    truth_kind: str
    phases: bool
    sigma_scale: float
    median_snr: float
    n_realizations: int
    features: list = field(default_factory=list)
    reason: str = ""

    def row(self, name: str) -> dict:
        for f in self.features:
            if f.name == name:
                return f.row()
        raise KeyError(name)

    def verdicts(self) -> dict:
        return {f.name: f.verdict for f in self.features}

    def table(self) -> list[dict]:
        return [f.row() for f in self.features]


FEATURE_UNITS = {"diameter_uas": "uas", "width_uas": "uas", "asymmetry": "", "pa_deg": "deg", "flux_jy": "Jy"}


def prior_dependence_study(object_name: str = "M87*", n_realizations: int = 20, truth=None,
                           sigma_scale: float = 1.0, phases: bool = True, prior: dict | None = None,
                           gain_error: float = 0.05, phase_error_rad: float = 0.10, seed: int = 0) -> PriorStudy:
    """Fit the same synthetic data sets twice -- prior off, prior on -- and report the move of every feature.

    The two fits see identical visibilities, so the difference of their results is caused by the prior alone and
    the standard error of the paired differences is free of the noise that is common to both. ``sigma_scale``
    degrades the data (thermal noise grows as the integration time falls): the same experiment at 1, 10 and 100
    times the noise is what tells a reader *where* the prior starts to matter, rather than asserting that it does.

    ``phases=False`` fits visibility amplitudes only. That is the situation the EHT's geometric modelling was in,
    and it is the case in which a feature can be decided by the prior: the amplitude of an asymmetric ring is
    invariant to the sign of the asymmetry, so a fit to amplitudes alone is free to put the bright side on either
    side while the prior is not.
    """
    uv = _dataset(object_name)
    truth = published_crescent(object_name) if truth is None else truth
    prior = physics_prior(object_name) if prior is None else prior
    off, on, snr = [], [], []
    for k in range(int(n_realizations)):
        obs = observe(uv, truth=truth, seed=seed + k, sigma_scale=sigma_scale, gain_error=gain_error,
                      phase_error_rad=phase_error_rad)
        a = fit_crescent(obs.visibilities, uv, obs.sigma, prior={}, phases=phases)
        b = fit_crescent(obs.visibilities, uv, obs.sigma, prior=prior, phases=phases)
        off.append([a.values[name] for name in PARAMETERS])
        on.append([b.values[name] for name in PARAMETERS])
        snr.append(obs.snr)
    off = np.asarray(off)
    on = np.asarray(on)
    n = max(int(n_realizations), 2)
    features = []
    for i, name in enumerate(PARAMETERS):
        diff = on[:, i] - off[:, i]
        sd_diff = float(diff.std(ddof=1))
        features.append(FeatureShift(name=name, unit=FEATURE_UNITS[name], off_mean=float(off[:, i].mean()),
                                     on_mean=float(on[:, i].mean()), shift=float(diff.mean()),
                                     shift_err=sd_diff / math.sqrt(n) if sd_diff > 0 else 0.0,
                                     off_sd=float(off[:, i].std(ddof=1)), on_sd=float(on[:, i].std(ddof=1))))
    return PriorStudy(object=object_name, truth_kind=str(getattr(truth, "kind", "image")), phases=bool(phases),
                      sigma_scale=float(sigma_scale), median_snr=float(np.median(snr)), n_realizations=int(n),
                      features=features)


# --------------------------------------------------------------------------------------- the ray-traced link
def raytraced_ring(object_name: str = "M87*", inclination_deg: float = 17.0, r_cam: float = 60.0,
                   fov_deg: float = 30.0, n: int = 112, r_in: float = 6.0, r_out: float = 8.0,
                   doppler: bool = True) -> dict:
    """The image a real black hole makes: a thin disc around a Schwarzschild hole, ray-traced exactly.

    Calls :func:`pinneapple_physics.astro.lensing.trace_thin_disc` and ``disc_image`` -- null geodesics integrated
    in the plane of each ray, Keplerian gas with relativistic beaming when ``doppler`` is on -- and rescales the
    render to the object's published mass and distance, so the shadow in the image has the published angular size.
    ``inclination_deg`` is measured from the disc axis: 17 deg is M87*, whose ring is nearly circular. The emitting
    annulus ``r_in`` to ``r_out`` in GM/c^2 is a *choice*: a Schwarzschild thin disc has its inner edge at the ISCO,
    6 GM/c^2, and the emission is cut at 8 GM/c^2 so that the image is the compact ring M87* shows rather than a
    broad disc. The ring diameter that comes out is a property of that choice, and it is deliberately not imposed
    to be the shadow: the two are different quantities (see :func:`image_ring_diameter`).

    Returns the image (Jy per pixel, normalised to unit flux), its pixel scale in uas, the luminance, and the
    angular scale of the render. The mapping is the one physical statement in it: one GM/c^2 at the object's
    distance subtends ``shadow_diameter / (2 sqrt(27))`` microarcseconds, so the render's degrees convert with
    ``uas_per_deg = shadow_diameter_uas / (2 alpha_deg)``, where ``alpha_deg`` is the shadow's angular radius in
    the render (:func:`trace_thin_disc`'s camera).
    """
    from .lensing import disc_image, trace_thin_disc

    obj = EHT_OBJECTS[object_name]
    trace = trace_thin_disc(width=int(n), height=int(n), inclination_deg=inclination_deg, r_cam=r_cam,
                            fov_deg=fov_deg)
    rgb, _coverage = disc_image(trace, r_in=r_in, r_out=r_out, doppler=doppler)
    luminance = rgb.sum(axis=2)
    alpha_deg = shadow_radius_deg(r_cam)
    shadow_uas = shadow_angular_diameter_uas(obj["mass_msun"], obj["distance_pc"] / 1e6)
    uas_per_deg = shadow_uas / (2.0 * alpha_deg)
    pixel_uas = fov_deg / int(n) * uas_per_deg
    return {"image": luminance, "pixel_uas": pixel_uas, "uas_per_deg": uas_per_deg, "shadow_uas": shadow_uas,
            "alpha_deg": alpha_deg, "trace": trace, "r_in": r_in, "r_out": r_out}


def shadow_radius_deg(r_cam: float, b_crit: float | None = None) -> float:
    """Angular radius (degrees) of the shadow seen by a static camera at ``r_cam`` GM/c^2.

    ``asin(b_crit sqrt(1 - 2/r_cam) / r_cam)`` with the critical impact parameter ``b_crit = 3 sqrt(3)``: the same
    expression ``accretion_view`` returns, restated here so the scale mapping of :func:`raytraced_ring` does not
    need a camera.
    """
    b = 3.0 * math.sqrt(3.0) if b_crit is None else float(b_crit)
    return math.degrees(math.asin(min(1.0, b * math.sqrt(max(1.0 - 2.0 / r_cam, 0.0)) / r_cam)))


def image_ring_diameter(image, pixel_uas: float, flux_jy: float | None = None, prior: dict | None = None,
                        uv: UVData | None = None) -> dict:
    """Fit the same crescent to an image (a ray-traced disc, say) and report its diameter in microarcseconds.

    Two numbers come back: the diameter the crescent family finds in the image's own visibilities on the array's
    (u, v) coverage, and the flux-weighted mean radius of the image, which needs no model. A ray-traced disc is not
    a crescent, so the two need not agree and the difference is the model error, not an error of either one.
    """
    img = np.asarray(image, dtype=float)
    total = float(img.sum())
    img = img / total if total else img
    flux = float(flux_jy) if flux_jy is not None else 1.0
    mean_radius = _flux_weighted_radius(img, pixel_uas)
    uv = _dataset("M87*") if uv is None else uv
    clean = flux * visibilities_from_image(img, pixel_uas, uv.u, uv.v)
    sigma = float(np.mean(uv.sigma))
    fit = fit_crescent(clean, uv, sigma, prior={} if prior is None else prior)
    return {"diameter_uas": fit.values["diameter_uas"], "diameter_err_uas": fit.errors["diameter_uas"],
            "width_uas": fit.values["width_uas"], "asymmetry": fit.values["asymmetry"],
            "pa_deg": fit.values["pa_deg"], "flux_jy": fit.values["flux_jy"],
            "mean_radius_uas": mean_radius, "mean_diameter_uas": 2.0 * mean_radius, "fit": fit}


def _flux_weighted_radius(image, pixel_uas: float) -> float:
    """``sum I r / sum I`` over the image, with ``r`` measured from the image centre (uas)."""
    img = np.asarray(image, dtype=float)
    m = img.shape[0]
    grid = (np.arange(m) - (m - 1) / 2.0) * float(pixel_uas)
    x, y = np.meshgrid(grid, grid)
    w = np.clip(img, 0.0, None)
    total = float(w.sum())
    return float(np.sum(w * np.hypot(x, y)) / total) if total > 0 else float("nan")


# ------------------------------------------------------------------------------------- declared data products
def data_requirements() -> dict:
    """What the real data would be, and what this module would do with it.

    Named rather than implied, because the difference between "this study used the EHT data" and "this study used
    synthetic data" is the difference between a measurement and a closure test.
    """
    return {
        "M87* 2017": {
            "release": "Event Horizon Telescope Collaboration 2019, ApJL 875, L1-L6: the M87* 2017 data release "
                       "(April 2019), distributed by the EHT Collaboration through the CDS/VizieR and the EHT "
                       "website in uvfits",
            "file": "SR1_M87_2017_101_hi_hops_netcal_StokesI.uvfits (and the lo-band and LMT-only analogues); one "
                    "file per band, network calibration and Stokes parameter",
            "columns": ["time", "baseline (antenna pair)", "u (lambda)", "v (lambda)", "Stokes I visibility",
                        "thermal sigma"],
            "size": "tens of MB per uvfits file; the loader accepts 1 MB - 500 MB and rejects anything else",
            "offline": False,
        },
        "Sgr A* 2017": {
            "release": "Event Horizon Telescope Collaboration 2022, ApJL 930, L12-L17: the Sgr A* 2017 data "
                       "release, same archive and format",
            "file": "the Sgr A* 2017 uvfits files, one per band and calibration",
            "columns": ["time", "baseline (antenna pair)", "u (lambda)", "v (lambda)", "Stokes I visibility",
                        "thermal sigma"],
            "size": "tens of MB per uvfits file; the loader accepts 1 MB - 500 MB and rejects anything else",
            "offline": False,
        },
    }


def load_visibility_data(object_name: str = "M87* 2017", path: str | None = None):
    """Load a public EHT data release, or fail with the exact list of what is missing.

    Neither release is redistributed with PINNeAPPle and nothing is downloaded here. Without a local uvfits file
    this raises ``NotConfigured`` naming the release, the file and the expected size, rather than returning an
    empty array that a study could mistake for a measurement.
    """
    import os

    from pinneapple_core.errors import NotConfigured

    spec = data_requirements().get(object_name)
    if spec is None:
        raise NotConfigured(f"unknown EHT data release {object_name!r}",
                            fix=f"use one of {sorted(data_requirements())}, or pass a uvfits file yourself")
    if path is None:
        raise NotConfigured(f"the {object_name} visibilities are not bundled and no local file was given",
                            fix=f"point path= at {spec['file']} from the {spec['release']}; it must carry "
                                f"{', '.join(spec['columns'])} ({spec['size']})")
    if not os.path.exists(path):
        raise NotConfigured(f"no uvfits file at {path}",
                            fix=f"{spec['release']} provides the columns {', '.join(spec['columns'])}; "
                                f"e.g. {spec['file']}")
    size = os.path.getsize(path)
    if not 1024 * 1024 <= size <= 500 * 1024 * 1024:
        raise NotConfigured(f"{path} has {size} bytes, outside the size of a uvfits release",
                            fix="check that the whole file was downloaded, not an HTML error page")
    with open(path, "rb") as f:
        head = f.read(4096)
    if b"UVFITS" not in head and b"SIMPLE" not in head:
        raise NotConfigured(f"{path} is not a FITS file (no SIMPLE card in its first block)",
                            fix=f"an EHT visibility release is a FITS binary table: {spec['file']}")
    raise NotConfigured(
        f"{path} looks like a uvfits file, but reading it needs astropy and the release's own conventions, which "
        f"are not part of this module",
        fix="read it with astropy.io.fits, take Stokes I and the thermal sigma per row, and feed u, v (in Glambda) "
            "to uv_coverage's UVData; the fitting side of this module is independent of where the visibilities "
            "came from")


# --------------------------------------------------------------------------------------------------- reporting
def report(object_name: str = "M87*", n_realizations: int = 20, seed: int = 0) -> dict:
    """The whole study as one dictionary of tables: the reproduction, the prior experiment at three data
    qualities, the amplitude-only variant, and the ray-traced link. This is what the documentation page prints."""
    rep = reproduce_ring_diameter(object_name, n_realizations=n_realizations, seed=seed)
    ladder = []
    for scale in (1.0, 10.0, 100.0):
        study = prior_dependence_study(object_name, n_realizations=n_realizations, sigma_scale=scale, seed=seed)
        ladder.append({"sigma_scale": scale, "median_snr": study.median_snr, "phases": True,
                       "features": study.table()})
    amplitudes = prior_dependence_study(object_name, n_realizations=n_realizations, phases=False, seed=seed)
    traced = raytraced_ring(object_name)
    image_fit = image_ring_diameter(traced["image"], traced["pixel_uas"], flux_jy=rep.flux_jy,
                                    uv=_dataset(object_name))
    return {"object": object_name, "reproduction": rep.table(), "reproduction_record": rep,
            "published": EHT_OBJECTS[object_name], "prior": physics_prior(object_name),
            "prior_ladder": ladder,
            "amplitude_only": {"median_snr": amplitudes.median_snr, "features": amplitudes.table()},
            "raytrace": {"diameter_uas": image_fit["diameter_uas"],
                         "diameter_err_uas": image_fit["diameter_err_uas"],
                         "mean_diameter_uas": image_fit["mean_diameter_uas"],
                         "asymmetry": image_fit["asymmetry"], "pa_deg": image_fit["pa_deg"],
                         "width_uas": image_fit["width_uas"], "shadow_uas": traced["shadow_uas"],
                         "pixel_uas": traced["pixel_uas"], "r_in_gm_c2": traced["r_in"],
                         "r_out_gm_c2": traced["r_out"]}}
