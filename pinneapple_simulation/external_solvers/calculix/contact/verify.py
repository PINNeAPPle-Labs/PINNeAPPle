"""The checks a contact run is judged by: closed-form comparisons, the Coulomb cone and the Hertz profile fit.

A contact analysis that converges is not a contact analysis that is right — a wrong master surface, a friction
coefficient applied to the wrong interface or a load transferred through the wrong path all converge, and the plot in
PrePoMax looks plausible. What settles it is a number computed independently: ``|F_t| = mu*N`` for a sliding joint,
``p0`` and the half-width for a non-conforming pair, ``T/N = min(tan(theta), mu)`` for an interface on a slope.

Three rules hold everywhere here:

* every comparison names the closed-form value, its formula and how it was obtained;
* a value that could not be measured makes the check ``inconclusive`` — never a pass;
* tolerances are declared by the case, not tuned until it passes. A mesh that does not resolve the contact patch
  fails the check, and the report says the tolerance it missed.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

PASS, FAIL, INCONCLUSIVE = "pass", "fail", "inconclusive"


@dataclass
class Check:
    """One verification, in the shape the job's evidence record reads."""
    name: str
    outcome: str
    value: float | None = None
    reference: float | None = None
    detail: str = ""
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "outcome": self.outcome, "value": self.value, "reference": self.reference,
                "detail": self.detail, "source": self.source}


def finite(*values: Any) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def relative(value: float | None, reference: float | None) -> float | None:
    """``(value - reference) / reference``, or None when either is missing, non-finite or the reference is 0."""
    if not finite(value, reference) or reference == 0:
        return None
    return (value - reference) / reference


def compare(name: str, value: float | None, reference: float | None, tol: float, source: str,
            unit: str = "", what: str = "") -> Check:
    """``value`` against a closed-form ``reference``, passing inside a declared relative ``tol``."""
    rel = relative(value, reference)
    label = f"{what}: " if what else ""
    if rel is None:
        return Check(name, INCONCLUSIVE, value, reference,
                     f"{label}nada a comparar (valor {value!r}, referência {reference!r})", source)
    ok = abs(rel) <= tol
    return Check(name, PASS if ok else FAIL, value, reference,
                 f"{label}{value:.6g} {unit} contra {reference:.6g} {unit} da solução fechada ({source}), "
                 f"tolerância {tol * 100:.1f}%, desvio {rel * 100:+.2f}%", source)


def at_most(name: str, value: float | None, limit: float | None, tol: float, source: str, unit: str = "",
            what: str = "") -> Check:
    """``|value| <= limit`` (an inequality, such as the Coulomb cone), with the same declared slack."""
    if not finite(value, limit):
        return Check(name, INCONCLUSIVE, value, limit, f"{what}: nada a comparar ({value!r} vs {limit!r})", source)
    ok = abs(value) <= abs(limit) * (1 + tol)
    return Check(name, PASS if ok else FAIL, value, limit,
                 f"{what}: |{value:.6g}| {unit} contra o limite {limit:.6g} {unit} ({source}), "
                 f"folga {tol * 100:.1f}%", source)


def between(name: str, ratio: float | None, low: float, high: float, source: str, what: str = "") -> Check:
    """A ratio expected inside a band — the shape of a contact-pressure profile, for instance."""
    if not finite(ratio):
        return Check(name, INCONCLUSIVE, ratio, high, f"{what}: nada a comparar ({ratio!r})", source)
    ok = low <= ratio <= high
    return Check(name, PASS if ok else FAIL, ratio, high,
                 f"{what}: {ratio:.4g}, esperado entre {low:.4g} e {high:.4g} ({source})", source)


# ------------------------------------------------------------------------------------------------ contact extraction
def contact_nodes(contact: dict[int, dict[str, float]], nodes: Iterable[int] | None = None,
                  min_pressure: float = 0.0) -> list[tuple[int, dict[str, float]]]:
    """The nodes actually carrying pressure, restricted to the interface that was asked about."""
    keep = set(nodes) if nodes is not None else None
    return [(n, r) for n, r in sorted(contact.items())
            if (keep is None or n in keep) and r.get("pressure", 0.0) > min_pressure]


def peak_pressure(contact: dict[int, dict[str, float]], nodes: Iterable[int] | None = None) -> float | None:
    rows = [r["pressure"] for _, r in contact_nodes(contact, nodes)]
    return max(rows) if rows else None


def hertz_fit(points: Sequence[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Fit ``p(x) = p0 * sqrt(1 - (x/b)^2)`` to a measured contact-pressure profile: ``(p0, b, r2)``.

    The fit is linear in ``p^2`` against ``x^2`` (``p^2 = p0^2 - (p0^2 / b^2) x^2``), which needs no optimiser and
    gives the peak pressure and the half-width directly — the two numbers Hertz's line-contact solution predicts.
    ``None`` when the profile has fewer than three distinct positions or does not fall away from the centre (a
    contact patch the mesh did not resolve).
    """
    pts = sorted({round(x, 10): p for x, p in points}.items())
    if len(pts) < 3:
        return None
    xs = [x for x, _ in pts]
    ps2 = [p * p for _, p in pts]
    n = len(xs)
    sx = sum(x * x for x in xs)
    sxx = sum(x ** 4 for x in xs)
    sy = sum(ps2)
    sxy = sum(x * x * y for x, y in zip(xs, ps2, strict=True))
    den = n * sxx - sx * sx
    if abs(den) < 1e-30:
        return None
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    if intercept <= 0 or slope >= 0:
        return None
    p0 = math.sqrt(intercept)
    b = math.sqrt(-intercept / slope)
    mean = sy / n
    ss_tot = sum((y - mean) ** 2 for y in ps2)
    fitted = [intercept + slope * x * x for x in xs]
    ss_res = sum((y - f) ** 2 for y, f in zip(ps2, fitted, strict=True))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return p0, b, r2


def hertz_line(P: float, R: float, length: float, E: float, nu: float, rigid: bool = True) -> dict[str, float]:
    """Hertz line contact: half-width ``b`` and peak pressure ``p0`` for a total load ``P`` over ``length``.

    ``b = sqrt(4 P R / (pi L E*))``, ``p0 = 2 P / (pi b L)``, with ``1/E* = (1 - nu^2)/E`` for a rigid indenter and
    ``(1 - nu1^2)/E1 + (1 - nu2^2)/E2`` for two elastic bodies.
    """
    e_star = E / (1 - nu * nu) if rigid else E / (2 * (1 - nu * nu))
    if P <= 0 or R <= 0 or length <= 0 or e_star <= 0:
        return {}
    b = math.sqrt(4 * P * R / (math.pi * length * e_star))
    return {"b": b, "p0": 2 * P / (math.pi * b * length), "e_star": e_star}


def von_mises(sx: float, sy: float, sz: float, sxy: float = 0.0, syz: float = 0.0, szx: float = 0.0) -> float:
    """The equivalent (von Mises) stress of a stress state, tension positive — the quantity ``*PLASTIC`` compares
    with the yield stress."""
    return math.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                     + 3.0 * (sxy * sxy + syz * syz + szx * szx))


def hertz_line_axis(zeta: float, nu: float = 0.3) -> tuple[float, float, float]:
    """The principal stresses on the loaded axis under a 2-D Hertz line contact, in units of ``p0``.

    ``zeta = z/b``, ``z`` measured from the surface into the body, compression negative. The classical field
    (Johnson, *Contact Mechanics*, 1985) written out on the axis, where the shear vanishes and the three principal
    stresses are ``sigma_x`` (across the contact), ``sigma_y = nu (sigma_x + sigma_z)`` (out of plane, plane strain)
    and ``sigma_z`` (into the body).
    """
    root = math.sqrt(1.0 + zeta * zeta)
    sx = -((1.0 + 2.0 * zeta * zeta) / root - 2.0 * zeta)
    sz = -1.0 / root
    return sx, nu * (sx + sz), sz


def first_yield_factor(nu: float = 0.3, steps: int = 4000, z_max: float = 4.0) -> float:
    """``p0 / sigma_y`` at the onset of yield in a 2-D Hertz line contact — Johnson's classic factor.

    The peak of the equivalent stress sits *below* the surface, so the factor is not ``1``: the field above is
    scanned along the axis and the factor is ``1 / max(von Mises / p0)``. For ``nu = 0.3`` it comes out **1.79** by
    von Mises (the criterion CalculiX uses), with the peak at ``z = 0.70 b``; by Tresca it is 1.67 ``sigma_y``, the
    peak shear being ``0.30 p0`` at ``z = 0.79 b``. Same numbers as Johnson, *Contact Mechanics* (Cambridge, 1985),
    the section on the onset of yield.
    """
    peak = max(von_mises(*hertz_line_axis(z_max * i / steps, nu)) for i in range(1, steps + 1))
    return 1.0 / peak if peak > 0 else 0.0


def lame_shrink_fit(interference: float, a: float, b: float, c: float, E: float, nu: float,
                    plane_strain: bool = True, *, E_ring: float | None = None,
                    nu_ring: float | None = None) -> float:
    """Interface pressure of a ring shrunk onto a hollow shaft.

    With ``E_ring``/``nu_ring`` omitted the ring and the shaft share the material.

    Lamé: the radial interference is the sum of the ring's expansion and the shaft's contraction,
    ``delta = p a [(c^2+a^2)/(c^2-a^2) + (a^2+b^2)/(a^2-b^2)] / E'`` — the Poisson terms cancel for a single
    material, which is why the pressure depends on ``nu`` only through ``E'``. In **plane strain** (a long fit, which
    is what a one-element-thick model with its faces held is) ``E' = E/(1 - nu^2)``, and that is a 10 % difference at
    ``nu = 0.3``: a plane-stress formula against a plane-strain model misses by more than the mesh does.
    """
    if E_ring is not None or nu_ring is not None:
        # two materials: the ring no longer cancels against the shaft, so the full Lamé sum is used. With
        # ``E_ring == E`` and ``nu_ring == nu`` it reduces to the one-material result exactly (same float).
        from .lame import interference_fit_pressure
        if not (0 < b < a < c) or interference <= 0:
            return 0.0
        return interference_fit_pressure(
            interference_m=interference, interface_radius_m=a, ring_outer_radius_m=c,
            shaft_inner_radius_m=b, ring_young_pa=E if E_ring is None else E_ring,
            ring_poisson=nu if nu_ring is None else nu_ring, shaft_young_pa=E, shaft_poisson=nu,
            plane_strain=plane_strain)
    if not (0 < b < a < c) or interference <= 0 or E <= 0:
        return 0.0
    e_eff = E / (1 - nu * nu) if plane_strain else E
    ring = (c * c + a * a) / (c * c - a * a)
    shaft = (a * a + b * b) / (a * a - b * b)
    return interference * e_eff / (a * (ring + shaft))


def coulomb_ratio(contact: dict[int, dict[str, float]], nodes: Iterable[int] | None,
                  min_fraction: float = 0.05) -> float | None:
    """The largest ``|tau| / p`` over the loaded nodes: 0 when nothing is stuck or sliding, ``mu`` at the limit.

    Nodes below ``min_fraction`` of the peak pressure are left out: there the pressure is a rounding error and the
    ratio it produces says nothing about friction.
    """
    rows = contact_nodes(contact, nodes)
    if not rows:
        return None
    peak = max(r["pressure"] for _, r in rows)
    if peak <= 0:
        return None
    ratios = [math.hypot(r["shear1"], r["shear2"]) / r["pressure"]
              for _, r in rows if r["pressure"] >= min_fraction * peak]
    return max(ratios) if ratios else None


def tributary_areas(coords: dict[int, tuple[float, float, float]], nodes: Iterable[int],
                    plane: str = "y") -> dict[int, float]:
    """The area each node of a structured surface stands for, so a nodal field can be integrated into a force.

    On a grid the rule is the vertex one: a node owns half of each cell edge it sits on, so its weight is the product
    of the half-spans to its neighbours. For a bilinear field that is the exact integral (each cell hands a quarter of
    itself to each corner), which is what makes ``sum(p_i * A_i)`` the contact force rather than an estimate of it.
    """
    at = {"x": 0, "y": 1, "z": 2}[plane]
    others = [i for i in (0, 1, 2) if i != at]
    keep = list(nodes)
    out: dict[int, float] = {}
    spans: list[dict[float, float]] = []
    for i in others:
        values = sorted({round(coords[n][i], 9) for n in keep})
        span = {}
        for k, v in enumerate(values):
            lo = values[k - 1] if k > 0 else v
            hi = values[k + 1] if k + 1 < len(values) else v
            span[v] = (hi - lo) / 2
        spans.append(span)
    for n in keep:
        a = spans[0][round(coords[n][others[0]], 9)] * spans[1][round(coords[n][others[1]], 9)]
        out[n] = a
    return out


def resultant(contact: dict[int, dict[str, float]], weights: dict[int, float], field: str = "pressure") -> float | None:
    """``sum(field * area)`` over the loaded nodes: the force the interface transmits in that direction."""
    total = 0.0
    used = 0
    for node, row in contact.items():
        if node in weights and row.get("pressure", 0.0) > 0.0:
            value = math.hypot(row.get(field, 0.0), row.get("shear2", 0.0)) if field == "shear1" \
                else row.get(field, 0.0)
            total += value * weights[node]
            used += 1
    return total if used else None


def slip_range(contact: dict[int, dict[str, float]], nodes: Iterable[int] | None) -> float | None:
    """How far the interface actually slid: the spread of the tangential slip over the loaded nodes."""
    rows = [r for _, r in contact_nodes(contact, nodes)]
    if not rows:
        return None
    return max(math.hypot(r["slip1"], r["slip2"]) for r in rows)
