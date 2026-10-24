"""Parametric parts as :class:`~pinneapple_tools.nvidia.brep.PartRecipe` data, with their closed-form volumes so
the exact solid can be checked against them."""
from __future__ import annotations

import math

from .brep import PartRecipe

__all__ = ["l_bracket", "l_bracket_volume", "heat_sink", "heat_sink_geometry"]


def l_bracket(*, base=0.08, height=0.06, width=0.05, t=0.006, fillet=0.005, bolt_d=0.009, web_hole_d=0.012,
              bolt_y=(0.015, 0.035), bolt_x=0.05, web_hole_z=0.035) -> PartRecipe:
    """An L bracket: base plate and web of thickness ``t``, an inner fillet, two bolt holes in the base and one
    hole through the web (none when ``web_hole_d`` is 0)."""
    r = (PartRecipe().box("base", (0, 0, 0), (base, width, t)).box("web", (0, 0, 0), (t, width, height))
            .union("L", "base", "web").fillet_edges_near("Lf", "L", [(t, width / 2, t)], fillet)
            .cylinder("h1", (bolt_x, bolt_y[0], -height), bolt_d / 2, 3 * height)
            .cylinder("h2", (bolt_x, bolt_y[1], -height), bolt_d / 2, 3 * height)
            .difference("a", "Lf", "h1").difference("b" if web_hole_d > 0 else "bracket", "a", "h2"))
    if web_hole_d > 0:
        r = (r.cylinder("h3z", (0, 0, 0), web_hole_d / 2, 5 * t).rotate("h3", "h3z", (0, 0, 0), (0, 1, 0), 90)
             .translate("h3t", "h3", (-2 * t, width / 2, web_hole_z)).difference("bracket", "b", "h3t"))
    return r


def l_bracket_volume(*, base=0.08, height=0.06, width=0.05, t=0.006, fillet=0.005, bolt_d=0.009,
                     web_hole_d=0.012, **_) -> float:
    return (base * width * t + t * width * height - t * width * t - 2 * math.pi * (bolt_d / 2) ** 2 * t
            - math.pi * (max(web_hole_d, 0.0) / 2) ** 2 * t + fillet ** 2 * (1 - math.pi / 4) * width)


def heat_sink(*, n_fins: int, fin_height: float, fin_t: float, base_t: float = 0.005, length: float = 0.1,
              width: float = 0.1, root_fillet: float = 0.0) -> PartRecipe:
    """A plate-fin heat sink: a base ``length`` (along the fins, x) by ``width`` (y) and ``n_fins`` equally spaced
    fins of thickness ``fin_t`` and height ``fin_height`` on top; ``root_fillet`` rounds the inner fin roots (an
    extrusion has them; the closed-form thermal model does not)."""
    r = PartRecipe().box("base", (0, 0, 0), (length, width, base_t))
    pitch = (width - fin_t) / (n_fins - 1)
    last = "base"
    for i in range(n_fins):
        r.box(f"fin{i}", (0, i * pitch, base_t - 1e-4), (length, fin_t, fin_height + 1e-4))
        r.union(f"hs{i}", last, f"fin{i}")
        last = f"hs{i}"
    r.result = last
    if root_fillet > 0:
        pts = []
        for i in range(n_fins):
            if i > 0:
                pts.append((length / 2, i * pitch, base_t))
            if i < n_fins - 1:
                pts.append((length / 2, i * pitch + fin_t, base_t))
        r.fillet_edges_near("hs_filleted", last, pts, root_fillet)
    return r


def heat_sink_geometry(*, n_fins: int, fin_height: float, fin_t: float, base_t: float = 0.005, length: float = 0.1,
                       width: float = 0.1, root_fillet: float = 0.0) -> dict:
    """Closed-form volume and areas (``wetted_area``: fins and the exposed base top; ``total_area``: every face).
    The root fillets add ``r^2 (1 - pi/4)`` of section each to the volume; the areas ignore them."""
    vol = length * width * base_t + n_fins * length * fin_t * fin_height \
        + 2 * (n_fins - 1) * root_fillet ** 2 * (1 - math.pi / 4) * length
    fin_area = n_fins * (2 * fin_height * length + fin_t * length + 2 * fin_height * fin_t)
    base_top = length * width - n_fins * length * fin_t
    total = fin_area + base_top + length * width + 2 * (length + width) * base_t
    return {"volume": vol, "wetted_area": fin_area + base_top, "total_area": total,
            "gap": (width - n_fins * fin_t) / (n_fins - 1)}
