"""Bright-star catalogue (Yale BSC5) and stellar astrophysics: distances, absolute magnitudes, effective
temperature from colour, bolometric corrections from a black-body V-band integral, luminosity, radius, Planck
spectra and black-body colours.

    cat = load_bright_stars()
    props = star_properties(cat, cat.index_of(2491))         # Sirius
"""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")

H = 6.62607015e-34
KB = 1.380649e-23
C = 2.99792458e8
SIGMA_SB = 5.670374419e-8
L_SUN = 3.828e26          # W (IAU 2015 B3)
R_SUN = 6.957e8           # m
T_SUN = 5772.0            # K
M_BOL_SUN = 4.74          # IAU 2015 B2
M_V_SUN = 4.81
PC_M = 3.0856775814913673e16


@dataclass
class StarCatalog:
    hr: np.ndarray
    name: np.ndarray
    desig: np.ndarray
    con: np.ndarray
    ra: np.ndarray            # deg, J2000
    dec: np.ndarray
    pmra: np.ndarray          # mas/yr, mu_alpha cos delta
    pmdec: np.ndarray
    v: np.ndarray
    bv: np.ndarray            # nan if missing
    sp: np.ndarray
    plx: np.ndarray           # mas, nan if missing
    plx_dyn: np.ndarray       # True if the BSC notes flag a dynamical parallax
    rv: np.ndarray            # km/s

    def __len__(self):
        return len(self.hr)

    def index_of(self, hr: int) -> int:
        i = np.nonzero(self.hr == int(hr))[0]
        if not len(i):
            raise KeyError(f"HR {hr} not in the catalogue")
        return int(i[0])

    def label(self, i: int) -> str:
        return self.name[i] or self.desig[i]


@lru_cache(maxsize=1)
def load_bright_stars() -> StarCatalog:
    with gzip.open(os.path.join(DATA, "bsc5.json.gz"), "rt", encoding="utf-8") as fh:
        d = json.load(fh)
    f = lambda k: np.array([np.nan if x is None else x for x in d[k]], dtype=float)  # noqa: E731
    return StarCatalog(np.array(d["hr"]), np.array(d["name"], dtype=object), np.array(d["desig"], dtype=object),
                       np.array(d["con"], dtype=object), f("ra"), f("dec"), f("pmra"), f("pmdec"), f("v"), f("bv"),
                       np.array(d["sp"], dtype=object), f("plx"), np.array(d["plx_dyn"], bool), f("rv"))


@lru_cache(maxsize=1)
def load_constellations() -> list:
    """IAU constellation figures with a ``key`` (the IAU abbreviation; Serpens is split into ``Ser1`` = Caput and
    ``Ser2`` = Cauda), the Latin ``name``, the ``names`` per language ({"en": ..., "pt": ...}) and a label point.

    The d3-celestial source gives both halves of Serpens the same name and label point; here each half gets its own
    name and a label at the mean of its figure's vertices."""
    with gzip.open(os.path.join(DATA, "constellations.json.gz"), "rt", encoding="utf-8") as fh:
        cons = json.load(fh)
    names = constellation_names()
    for c in cons:
        key = c["id"]
        if key == "Ser":
            pts = np.array([p for line in c["lines"] for p in line], dtype=float)
            ra = np.degrees(np.arctan2(np.sin(np.radians(pts[:, 0])).mean(), np.cos(np.radians(pts[:, 0])).mean()))
            key = "Ser1" if ra % 360 < 255 else "Ser2"
            c["label"] = [round(float(ra % 360), 3), round(float(pts[:, 1].mean()), 3)]
        c["key"] = key
        c["names"] = names[key]
        c["name"] = names[key]["en"]
    return cons


@lru_cache(maxsize=1)
def _constellation_names_file() -> dict:
    with open(os.path.join(DATA, "constellation_names.json"), encoding="utf-8") as fh:
        return json.load(fh)


def constellation_names(lang: str | None = None) -> dict:
    """Constellation names by key (IAU abbreviation, ``Ser1``/``Ser2`` for the two halves of Serpens): the IAU Latin
    name (``en``) and the Brazilian Portuguese name (``pt``), e.g. ``Cru`` -> Crux / Cruzeiro do Sul. With ``lang``,
    a {key: name} map in that language. Sources: ``constellation_names.json``."""
    names = _constellation_names_file()["names"]
    if lang is None:
        return {k: dict(v) for k, v in names.items()}
    if lang not in ("en", "pt"):
        raise ValueError(f"language {lang!r}: use 'en' or 'pt'")
    return {k: v[lang] for k, v in names.items()}


# --------------------------------------------------------------------------------------------- black body
def planck_lambda(wavelength_m, temperature_k):
    """Spectral radiance B_lambda (W m^-2 sr^-1 m^-1)."""
    lam = np.asarray(wavelength_m, dtype=float)
    T = np.asarray(temperature_k, dtype=float)
    x = H * C / (lam * KB * T)
    with np.errstate(over="ignore"):
        return 2 * H * C**2 / lam**5 / np.expm1(np.minimum(x, 700))


def wien_peak_m(temperature_k):
    return 2.897771955e-3 / np.asarray(temperature_k, dtype=float)


def blackbody_rgb(temperature_k):
    """sRGB colour of a black body (reuses the black-hole renderer's fit, :mod:`pinneapple_physics.blackhole`)."""
    from ..blackhole.raytrace import blackbody_rgb as _bb
    return _bb(temperature_k)


def blackbody_spectrum(temperature_k: float, n: int = 300, lam_min_nm: float = 100.0, lam_max_nm: float = 3000.0):
    lam = np.linspace(lam_min_nm, lam_max_nm, n) * 1e-9
    B = planck_lambda(lam, temperature_k)
    return {"wavelength_nm": lam * 1e9, "radiance": B, "peak_nm": float(wien_peak_m(temperature_k) * 1e9),
            "flux_w_m2": float(SIGMA_SB * temperature_k**4), "rgb": blackbody_rgb(temperature_k).tolist()}


# ----------------------------------------------------------------------------------------- stellar physics
def teff_from_bv(bv):
    """Effective temperature (K) from B-V, Ballesteros (2012, EPL 97, 34008), black-body based; good to ~5%
    for main-sequence and giant stars between B-V = -0.3 and 1.8."""
    bv = np.asarray(bv, dtype=float)
    return 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))


def _v_band():
    # Johnson V approximated by a Gaussian centred at 551 nm with 88 nm FWHM
    lam = np.linspace(380e-9, 760e-9, 400)
    return lam, np.exp(-0.5 * ((lam - 551e-9) / (88e-9 / 2.355)) ** 2)


def bolometric_correction(teff):
    """BC_V for a black body of temperature ``teff``, zero-pointed so that BC_V(Sun, 5772 K) = M_bol,sun - M_V,sun
    = -0.07. A physically transparent approximation (real stellar atmospheres differ by ~0.1-0.3 mag for cool M
    stars and hot O stars)."""
    lam, S = _v_band()
    T = np.atleast_1d(np.asarray(teff, dtype=float))

    def bc_raw(t):
        frac = np.trapezoid(planck_lambda(lam[None, :], t[:, None]) * S, lam, axis=1) / (SIGMA_SB * t**4 / np.pi)
        return 2.5 * np.log10(frac)
    zp = (M_BOL_SUN - M_V_SUN) - bc_raw(np.array([T_SUN]))[0]
    out = bc_raw(T) + zp
    return out if np.ndim(teff) else float(out[0])


def star_properties(cat: StarCatalog, i: int, extinction_av: float = 0.0) -> dict:
    """Distance, absolute magnitude, effective temperature, luminosity and radius of catalogue star ``i``."""
    plx = cat.plx[i]
    out = {"hr": int(cat.hr[i]), "name": cat.label(i), "designation": cat.desig[i], "constellation": cat.con[i],
           "ra_deg": float(cat.ra[i]), "dec_deg": float(cat.dec[i]), "v_mag": float(cat.v[i]),
           "b_v": None if np.isnan(cat.bv[i]) else float(cat.bv[i]), "spectral_type": cat.sp[i],
           "radial_velocity_km_s": None if np.isnan(cat.rv[i]) else float(cat.rv[i]),
           "parallax_mas": None if np.isnan(plx) else float(plx), "parallax_dynamical": bool(cat.plx_dyn[i])}
    if not np.isnan(plx) and plx > 0:
        d_pc = 1000.0 / plx
        Mv = cat.v[i] - 5 * np.log10(d_pc / 10.0) - extinction_av
        out.update(distance_pc=float(d_pc), distance_ly=float(d_pc * 3.261563777), abs_mag_v=float(Mv))
        if out["b_v"] is not None:
            T = float(teff_from_bv(cat.bv[i]))
            bc = bolometric_correction(T)
            L = 10 ** (-0.4 * (Mv + bc - M_BOL_SUN))
            R = np.sqrt(L) * (T_SUN / T) ** 2
            out.update(teff_k=T, bolometric_correction=float(bc), luminosity_lsun=float(L), radius_rsun=float(R),
                       peak_nm=float(wien_peak_m(T) * 1e9), rgb=blackbody_rgb(T).tolist())
    elif out["b_v"] is not None:
        out.update(teff_k=float(teff_from_bv(cat.bv[i])))
    return out


def hr_diagram(cat: StarCatalog, max_rel_parallax_error: float | None = None, min_parallax_mas: float = 5.0):
    """Points (B-V, M_V) for stars with a usable trigonometric parallax (>= ``min_parallax_mas``)."""
    ok = ~np.isnan(cat.bv) & ~np.isnan(cat.plx) & (cat.plx >= min_parallax_mas) & ~cat.plx_dyn
    Mv = cat.v[ok] - 5 * np.log10(1000.0 / cat.plx[ok] / 10.0)
    return {"hr": cat.hr[ok], "bv": cat.bv[ok], "abs_mag": Mv, "v": cat.v[ok]}
