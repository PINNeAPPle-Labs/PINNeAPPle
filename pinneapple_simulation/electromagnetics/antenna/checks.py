"""Consistency checks of one antenna result, as :class:`pinneapple_analysis.trust.Check` / ``TrustReport``.

Each check is a relation the numbers must satisfy whatever solver produced them (IEEE Std 145 definitions):

=============================  ===========================================================  ========
check                          relation                                                     critical
=============================  ===========================================================  ========
``realized_gain``              ``G_realized = G_IEEE + 10 log10(1 - |S11|^2)``                no
``efficiency_is_G_over_D``     ``e_rad = G / D`` (``10 log10 e = G_dBi - D_dBi``)             no
``efficiency_le_1``            ``e_rad <= 1`` (a passive antenna cannot radiate more than    yes
                               it accepts)
``peak_ge_boresight``          the pattern peak is not below the boresight value             no
``directivity_le_hw``          ``D <= 7.28 L / lambda`` (+ 0.5 dB), :mod:`.bounds`           no
``s11_resolved``               an S11 interpolated by a rational (vector) fit is only        no
                               trusted with enough solved frequencies in the band
``power_closure``              radiated power from the far-field pattern = Poynting flux     no
                               through the radiation surface (within 10 %): the mesh
                               resolves the fields the far field is computed from
=============================  ===========================================================  ========

A check whose inputs are missing (``None`` / NaN) is ``not_run`` with the reason; it never passes silently.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pinneapple_analysis.trust.trust_report import (
    CONTRADICTS,
    NOT_RUN,
    SUPPORTS,
    Check,
    TrustReport,
)

from . import bounds

__all__ = ["check_realized_gain", "check_efficiency_ratio", "check_efficiency_le_one", "check_peak_ge_boresight",
           "check_directivity_limit", "check_s11_resolved", "check_power_closure", "antenna_checks"]

SOURCE = "IEEE Std 145-2013 definitions"


def _num(v: Any) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _missing(name: str, critical: bool = False, **inputs: Any) -> Check | None:
    miss = [k for k, v in inputs.items() if _num(v) is None]
    if miss:
        return Check(name, NOT_RUN, critical=critical, reason=f"missing {', '.join(miss)}")
    return None


def check_realized_gain(g_dbi, g_real_dbi, s11_db, *, tol_db: float = 0.1) -> Check:
    """``G_realized = G + 10 log10(1 - |S11|^2)`` within ``tol_db``."""
    name = "realized_gain"
    if (c := _missing(name, G=g_dbi, G_realized=g_real_dbi, S11=s11_db)) is not None:
        return c
    mag2 = 10 ** (float(s11_db) / 10)
    if mag2 >= 1:
        return Check(name, CONTRADICTS, value=float(s11_db), limit=0.0, source=SOURCE,
                     reason="|S11| >= 1: the port reflects all the power, no realized gain is defined")
    expect = float(g_dbi) + 10 * math.log10(1 - mag2)
    err = abs(float(g_real_dbi) - expect)
    return Check(name, SUPPORTS if err <= tol_db else CONTRADICTS, value=err, limit=tol_db, source=SOURCE,
                 reason=f"G_realized {float(g_real_dbi):.2f} dBi vs G + 10log10(1-|S11|^2) = {expect:.2f} dBi")


def check_efficiency_ratio(g_dbi, d_dbi, eff, *, tol_db: float = 0.1) -> Check:
    """``e_rad = G / D`` within ``tol_db``."""
    name = "efficiency_is_G_over_D"
    if (c := _missing(name, G=g_dbi, D=d_dbi, efficiency=eff)) is not None:
        return c
    if float(eff) <= 0:
        return Check(name, CONTRADICTS, value=float(eff), limit=0.0, reason="non-positive efficiency")
    err = abs(10 * math.log10(float(eff)) - (float(g_dbi) - float(d_dbi)))
    return Check(name, SUPPORTS if err <= tol_db else CONTRADICTS, value=err, limit=tol_db, source=SOURCE,
                 reason=f"10log10(e) = {10 * math.log10(float(eff)):.2f} dB vs G - D = "
                        f"{float(g_dbi) - float(d_dbi):.2f} dB")


def check_efficiency_le_one(eff, *, tol: float = 0.01) -> Check:
    """Radiation efficiency at most 1 (+ ``tol`` for the far-field quadrature). Critical."""
    name = "efficiency_le_1"
    if (c := _missing(name, critical=True, efficiency=eff)) is not None:
        return c
    v = float(eff)
    return Check(name, SUPPORTS if v <= 1 + tol else CONTRADICTS, critical=True, value=v, limit=1 + tol,
                 source="energy conservation (passive antenna)",
                 reason="" if v <= 1 + tol else "the antenna radiates more than it accepts: check the port power, "
                 "the far-field surface or the S11")


def check_peak_ge_boresight(peak_dbi, bore_dbi, *, tol_db: float = 0.05) -> Check:
    name = "peak_ge_boresight"
    if (c := _missing(name, peak=peak_dbi, boresight=bore_dbi)) is not None:
        return c
    gap = float(peak_dbi) - float(bore_dbi)
    return Check(name, SUPPORTS if gap >= -tol_db else CONTRADICTS, value=gap, limit=-tol_db,
                 reason=f"peak - boresight = {gap:.2f} dB")


def check_directivity_limit(d_dbi, length_m, f_hz, *, margin_db: float = 0.5, kind: str = "hw") -> Check:
    """Directivity below the end-fire limit of :mod:`.bounds` for a length ``length_m`` (+ ``margin_db``)."""
    name = "directivity_le_hw" if kind == "hw" else f"directivity_le_{kind}"
    if (c := _missing(name, D=d_dbi, L=length_m, f=f_hz)) is not None:
        return c
    lim = float(bounds.endfire_directivity_dbi(float(length_m), float(f_hz), kind)) + margin_db
    v = float(d_dbi)
    return Check(name, SUPPORTS if v <= lim else CONTRADICTS, value=v, limit=lim,
                 source="Balanis, Antenna Theory, ch. 6 (end-fire directivity)",
                 reason=f"D = {v:.2f} dBi vs {kind} limit {lim - margin_db:.2f} dBi + {margin_db} dB")


def check_s11_resolved(n_solved: int | None, *, interpolated: bool, min_points: int = 5) -> Check:
    """An S11 from a rational (vector-fit) interpolation needs at least ``min_points`` solved frequencies.

    With fewer, the fit has as many poles as data and its values between the samples are not evidence."""
    name = "s11_resolved"
    if not interpolated:
        return Check(name, SUPPORTS, value=n_solved, reason="S11 solved directly at this frequency")
    if n_solved is None:
        return Check(name, NOT_RUN, reason="number of solved frequencies unknown")
    ok = int(n_solved) >= min_points
    return Check(name, SUPPORTS if ok else CONTRADICTS, value=int(n_solved), limit=min_points,
                 reason="" if ok else f"S11 interpolated from only {int(n_solved)} solved frequencies: "
                 "not trustworthy between them")


def check_power_closure(p_pattern, p_surface, *, rtol: float = 0.10) -> Check:
    """Far-field power (integral of the pattern) against the near-field Poynting flux through the same surface."""
    name = "power_closure"
    if (c := _missing(name, P_pattern=p_pattern, P_surface=p_surface)) is not None:
        return c
    a, b = float(p_pattern), float(p_surface)
    if b <= 0:
        return Check(name, CONTRADICTS, value=b, reason="non-positive radiated power through the surface")
    err = abs(a - b) / b
    return Check(name, SUPPORTS if err <= rtol else CONTRADICTS, value=err, limit=rtol,
                 source="Poynting theorem (far field vs radiation surface)",
                 reason=f"P_pattern {a:.3g} W vs P_surface {b:.3g} W" + ("" if err <= rtol else
                        ": refine the mesh (lower res) before trusting D, G and the efficiency"))


def antenna_checks(row: Mapping[str, Any], *, length_m: float | None = None, n_solved: int | None = None,
                   interpolated: bool = False, min_points: int = 5, tol_db: float = 0.1,
                   margin_db: float = 0.5, model: str = "") -> TrustReport:
    """Every check above on one table row (the columns of :mod:`.runner`).

    ``length_m``: the antenna length for the directivity limit (default: ``row["Lb"]`` in mm)."""
    g = lambda k: row.get(k)  # noqa: E731
    if length_m is None and _num(g("Lb")) is not None:
        length_m = float(g("Lb")) * 1e-3
    f = _num(g("f_GHz"))
    checks = [
        check_realized_gain(g("G_bore_dBi"), g("Greal_bore_dBi"), g("S11_dB"), tol_db=tol_db),
        check_efficiency_ratio(g("G_bore_dBi"), g("D_bore_dBi"), g("rad_eff"), tol_db=tol_db),
        check_efficiency_le_one(g("rad_eff")),
        check_peak_ge_boresight(g("D_peak_dBi"), g("D_bore_dBi")),
        check_directivity_limit(g("D_peak_dBi"), length_m, None if f is None else f * 1e9, margin_db=margin_db),
        check_s11_resolved(n_solved, interpolated=interpolated, min_points=min_points),
        check_power_closure(g("P_rad_pattern"), g("P_rad_surface")),
    ]
    return TrustReport(checks, model=model or str(g("tag") or ""),
                       details={"f_GHz": f, "tag": g("tag")})
