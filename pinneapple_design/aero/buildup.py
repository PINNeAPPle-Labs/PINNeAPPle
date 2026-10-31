"""Zero-lift drag of a library aircraft from its mesh: turbulent flat-plate friction per component (wetted area of
each body, Reynolds number on its reference length, compressibility and Raymer form factors), scaled to the wetted
area of the bodies' boolean union, plus interference and excrescences.

    from pinneapple_design.aero.buildup import friction_drag
    d = friction_drag(pp.geometries.build("airliner_a320"), ac, rho=0.3639, V=230.2, T=216.65, mach=0.78)
    d["CD0"], d["S_wet"]
"""
from __future__ import annotations

import math
from typing import Any

# body-name prefix -> (form factor, reference length in m or one of "length", "mac", "0.7mac")
AIRLINER_COMPONENTS: dict[str, tuple[float, Any]] = {
    "fuselage": (1.10, "length"), "wing": (1.35, "mac"), "htail": (1.30, "0.7mac"), "vtail": (1.30, "mac"),
    "nacelle": (1.25, 4.3), "sharklet": (1.35, 1.0), "pylon": (1.3, 3.0)}


def sutherland_mu(T: float) -> float:
    """Dynamic viscosity of air (Sutherland), Pa.s.

    Returns
    -------
    mu : float [Pa s]
        Dynamic viscosity.
    """
    return 1.458e-6 * T ** 1.5 / (T + 110.4)


def friction_drag(geometry, ac, *, rho: float, V: float, T: float, mach: float,
                  components: dict[str, tuple[float, Any]] | None = None, interference: float = 1.08) -> dict[str, Any]:
    """CD0 on ``ac.wing_area`` of the bodies of ``geometry`` (a :class:`~pinneapple_design.geometry_library.Geometry`)
    whose names start with a key of ``components``; ``ac`` gives the wing area, the MAC and the length."""
    from ..geometry_library import meshops
    comps = AIRLINER_COMPONENTS if components is None else components
    refs = {"length": geometry.params.get("length", getattr(ac, "length", 0.0)), "mac": ac.mac, "0.7mac": 0.7 * ac.mac}
    mu = sutherland_mu(T)
    cd0, per = 0.0, {}
    for b in geometry.bodies:
        key = next((k for k in comps if b.name.startswith(k)), None)
        if key is None:
            continue
        k_form, lref = comps[key]
        lref = refs[lref] if isinstance(lref, str) else float(lref)
        Re = rho * V * lref / mu
        cf = 0.455 / (math.log10(Re) ** 2.58 * (1 + 0.144 * mach ** 2) ** 0.65)
        d = k_form * cf * meshops.area(b.vertices, b.faces) / ac.wing_area
        cd0 += d
        per[key] = per.get(key, 0.0) + d
    # the bodies overlap (wing roots inside the fuselage, pylons in the nacelles): the wetted area is the area of
    # their boolean union (Geometry.solid(), manifold3d), so scale the component sum to it
    area_sum = geometry.area()
    try:
        s_wet = meshops.area(*geometry.solid())
    except ImportError:
        s_wet = area_sum
    scale = s_wet / area_sum * interference
    return {"CD0": cd0 * scale, "S_wet": s_wet, "area_sum": area_sum, "components": {k: v * scale for k, v in per.items()}}


__all__ = ["AIRLINER_COMPONENTS", "friction_drag", "sutherland_mu"]
