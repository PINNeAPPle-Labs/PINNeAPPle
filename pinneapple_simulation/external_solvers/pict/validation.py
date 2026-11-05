"""The checks a PICT cavity result is held to: continuity, the boundary conditions, and Ghia et al. (1982).

The reference data are the values in Ghia, Ghia and Shin, *High-Re solutions for incompressible flow using the
Navier-Stokes equations and a multigrid method*, Journal of Computational Physics 48 (1982) 387-411, Tables I and II,
as reproduced by PICT's own validation suite (``tests/validations.py``, Apache-2.0). They are the u velocity along the
vertical centreline and the v velocity along the horizontal centreline, at the y and x stations in the tables -- not
a dense profile, so a computed field is interpolated onto those stations before comparing.

Nothing here needs PICT installed: the reference data, the reductions and the comparison can all be exercised on an
array (``tests/test_pict_bridge.py`` does exactly that). :func:`validate` is the only function that runs the solver,
and it says so instead of returning numbers when the solver is missing.
"""
from typing import Any

import numpy as np

from .case import _weights

__all__ = [
    "GHIA",
    "ghia_profile",
    "centerline",
    "compare_to_ghia",
    "divergence_max",
    "spacing_for",
    "spacing_from_summary",
    "stations",
    "cavity_checks",
    "validate",
]

# Ghia et al. (1982), Tables I (u along x = 0.5) and II (v along y = 0.5). Stations in stations-of-the-domain
# coordinates; values are normalised by the lid velocity U = 1.
GHIA: dict[str, Any] = {
    "y": [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
          0.9531, 0.9609, 0.9688, 0.9766, 1.0000],
    "u": {
        100: [0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581,
              -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000],
        1000: [0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648, -0.06080,
               0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1.00000],
        5000: [0.00000, -0.41165, -0.42901, -0.43643, -0.40435, -0.33050, -0.22855, -0.07404, -0.03039,
               0.08183, 0.20087, 0.33556, 0.46036, 0.45992, 0.46120, 0.48223, 1.00000],
        10000: [0.00000, -0.42735, -0.42537, -0.41657, -0.38000, -0.32709, -0.23186, -0.07540, 0.03111,
                0.08344, 0.20673, 0.34635, 0.47804, 0.48070, 0.47783, 0.47221, 1.00000],
    },
    "x": [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
          0.9453, 0.9531, 0.9609, 0.9688, 1.0000],
    "v": {
        100: [0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533,
              -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000],
        1000: [0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235, 0.02526, -0.31966,
               -0.42665, -0.51550, -0.39188, -0.33714, -0.27669, -0.21388, 0.00000],
        5000: [0.00000, 0.42447, 0.43329, 0.43648, 0.42951, 0.35368, 0.28066, 0.27280, 0.00945, -0.30018,
               -0.36214, -0.41442, -0.52876, -0.55408, -0.55069, -0.49774, 0.00000],
        10000: [0.00000, 0.43983, 0.43733, 0.43124, 0.41487, 0.35070, 0.28003, 0.27224, 0.00831, -0.30719,
                -0.36737, -0.41496, -0.45863, -0.49099, -0.52987, -0.54302, 0.00000],
    },
}


def ghia_profile(reynolds: int, component: str = "u") -> tuple[np.ndarray, np.ndarray]:
    """``(stations, values)`` of the Ghia table for ``reynolds``; ``component`` is ``"u"`` (along y) or ``"v"``.

    Raises :class:`KeyError` naming the Reynolds numbers the table has instead of interpolating between them.
    """
    if component not in ("u", "v"):
        raise KeyError(f"component must be 'u' or 'v', not {component!r}")
    table = GHIA[component]
    if int(reynolds) not in table:
        raise KeyError(f"no Ghia reference for Re={reynolds}; the table has {sorted(table)}")
    station = GHIA["y"] if component == "u" else GHIA["x"]
    return np.asarray(station, dtype=float), np.asarray(table[int(reynolds)], dtype=float)


def centerline(field: np.ndarray, *, direction: str = "vertical") -> np.ndarray:
    """The middle column (``"vertical"``, along y) or middle row (``"horizontal"``, along x) of a 2-D field.

    For an even number of cells the two middle lines are averaged: the comparison is against a point on the centre,
    and a single line would bias it by half a cell.
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError("a centreline is taken from a 2-D field (ny, nx)")
    if direction == "vertical":
        n = field.shape[1]
        cols = [n // 2 - 1, n // 2] if n % 2 == 0 else [n // 2]
        return field[:, cols].mean(axis=1)
    if direction == "horizontal":
        n = field.shape[0]
        rows = [n // 2 - 1, n // 2] if n % 2 == 0 else [n // 2]
        return field[rows, :].mean(axis=0)
    raise ValueError(f"direction must be 'vertical' or 'horizontal', not {direction!r}")


def compare_to_ghia(computed: np.ndarray, stations: np.ndarray, *, reynolds: int = 100,
                    component: str = "u", tol: float = 0.05) -> dict[str, Any]:
    """Interpolate ``computed`` (sampled at ``stations``) onto the Ghia table and report rms, max and the verdict.

    ``tol`` is the rms the result must stay under, in units of the lid velocity; it is a parameter because the
    accuracy a run owes depends on its grid -- 32x32 cells at Re = 1000 is a screenshot, 256x256 is a result.
    """
    ref_x, ref_u = ghia_profile(reynolds, component)
    computed = np.asarray(computed, dtype=float)
    stations = np.asarray(stations, dtype=float)
    if computed.shape != stations.shape:
        raise ValueError(f"computed {computed.shape} and stations {stations.shape} must match")
    order = np.argsort(stations)
    mine = np.interp(ref_x, stations[order], computed[order])
    diff = mine - ref_u
    rms = float(np.sqrt(np.mean(diff ** 2)))
    return {"reynolds": int(reynolds), "component": component, "rms": rms, "max": float(np.abs(diff).max()),
            "tol": float(tol), "passed": bool(rms <= tol), "reference": "Ghia et al. (1982)"}


def _cell_widths(n_cells: int, span: float, base: float, refine_lo: bool, refine_hi: bool) -> np.ndarray:
    return _weights(int(n_cells), float(base), refine_lo, refine_hi) * float(span)


def spacing_from_summary(summary: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Cell sizes ``(dx, dy)`` of the grid the case built, from the summary in the result."""
    x0, y0, x1, y1 = summary["bounds"]
    refine = ("-x", "+x", "-y", "+y") if summary.get("refine") else ()
    dx = _cell_widths(summary["nx"], x1 - x0, summary.get("base", 1.0), "-x" in refine, "+x" in refine)
    dy = _cell_widths(summary["ny"], y1 - y0, summary.get("base", 1.0), "-y" in refine, "+y" in refine)
    return dx, dy


def divergence_max(u: np.ndarray, v: np.ndarray, dx: np.ndarray, dy: np.ndarray, *,
                   nodes: bool = False) -> float:
    """``max |du/dx + dv/dy|`` over the whole field, central differences inside and one-sided at the border.

    ``dx`` and ``dy`` are the distances between consecutive samples along each axis (so they have one entry fewer than
    the field). ``nodes=False`` means the samples are cell centres and ``dx``/``dy`` are the cell widths; ``nodes=True``
    means they are the node spacings. The border is included on purpose: in a cell-centred field the largest divergence
    sits right below the moving lid, and an interior-only difference would never look there.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    dx = np.asarray(dx, dtype=float).reshape(-1)
    dy = np.asarray(dy, dtype=float).reshape(-1)
    if u.shape != v.shape:
        raise ValueError(f"u {u.shape} and v {v.shape} must have the same shape")
    expected = (dy.size + 1, dx.size + 1) if nodes else (dy.size, dx.size)
    if u.shape != expected:
        kind = "nodes" if nodes else "cells"
        raise ValueError(f"{kind} fields are {u.shape} but the spacing describes {expected}")
    if min(dx.size, dy.size) < 2:
        raise ValueError("a divergence needs at least 3 samples per direction")

    if nodes:
        # A node field carries the boundary conditions in its first and last rows, and the samples next to them come
        # from a different discretisation (a one-sided derivative of the streamfunction) than the walls do (the imposed
        # value). A difference taken across that seam is O(1/h) however well the flow satisfies continuity, so the
        # divergence is taken two nodes in, where every stencil is interior. Written as plain loops on purpose: the
        # vectorised version of this was wrong for a day (a mis-sized denominator that only showed as a plausible
        # number), and this runs once per check.
        hx, hy = np.asarray(dx, float), np.asarray(dy, float)
        n_y, n_x = u.shape
        if min(n_y, n_x) < 6:
            raise ValueError("too few samples to measure a divergence away from the imposed boundary rows")
        worst = 0.0
        for i in range(2, n_y - 2):
            for j in range(2, n_x - 2):
                du_dx = (u[i, j + 1] - u[i, j - 1]) / (hx[j - 1] + hx[j])
                dv_dy = (v[i + 1, j] - v[i - 1, j]) / (hy[i - 1] + hy[i])
                worst = max(worst, abs(du_dx + dv_dy))
        return float(worst)

    # sample i uses its neighbours i-1 and i+1, so the step is dx[i-1] + dx[i]
    du_dx = np.empty_like(u)
    du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (dx[:-2] + dx[1:-1])
    du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx[0]
    du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx[-1]
    dv_dy = np.empty_like(v)
    dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (dy[:-2] + dy[1:-1])[:, None]
    dv_dy[0, :] = (v[1, :] - v[0, :]) / dy[0]
    dv_dy[-1, :] = (v[-1, :] - v[-2, :]) / dy[-1]
    return float(np.abs(du_dx + dv_dy).max())


def spacing_for(result) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(dx, dy, nodes)`` matching the field of ``result``: the description of the grid it was recorded on."""
    summary = result.case_summary
    x0, y0, x1, y1 = summary["bounds"]
    if result.layout == "nodes":
        # the local engine solves on a uniform grid of nodes, so the spacing is the span over the intervals
        return (np.full(max(result.nx - 1, 2), (x1 - x0) / max(result.nx - 1, 1)),
                np.full(max(result.ny - 1, 2), (y1 - y0) / max(result.ny - 1, 1)), True)
    dx, dy = spacing_from_summary(summary)
    return dx, dy, False


def stations(result) -> tuple[np.ndarray, np.ndarray]:
    """Normalised ``(x, y)`` of the recorded samples, in units of the cavity side."""
    x0, y0, x1, y1 = result.case_summary["bounds"]
    xs, ys = result.cell_centers()
    return (xs - x0) / (x1 - x0), (ys - y0) / (y1 - y0)


def cavity_checks(result, *, tol_div: float = 1e-2, tol_wall: float = 5e-2, tol_ghia: float = 0.05,
                  snapshot: int = -1) -> dict[str, Any]:
    """The checks a recorded cavity run has to pass, each with the number it was decided on.

    - ``continuity``: the largest discrete divergence of the velocity, normalised by ``U / L`` (PISO enforces it to
      the pressure tolerance; a value of order 1 means the field is not a solution of the equations).
    - ``lid``: the u velocity on the top row against the lid velocity.
    - ``walls``: the largest speed on the three closed walls -- the bottom row and the side columns below the lid
      (a no-slip wall must be still).
    - ``ghia_u`` / ``ghia_v``: the centreline profiles against Ghia et al. (1982) at the run's Reynolds number, when
      the table has one.

    The checks read the field where the engine put it (``result.layout``): a node-based field has the walls in its
    first and last rows, a cell-centred one has them half a cell outside -- and there the ``lid`` line measures the
    top *cell centre*, which on a coarse grid is not the lid velocity. That is why a PICT run that reports a low
    ``lid`` value on a coarse grid is read together with its grid, not as a boundary-condition failure.
    """
    u, v = result.component("u")[snapshot], result.component("v")[snapshot]
    summary = result.case_summary
    dx, dy, nodes = spacing_for(result)
    lid = float(summary["lid_velocity"])
    length = float(summary["bounds"][2] - summary["bounds"][0])

    div = divergence_max(u, v, dx, dy, nodes=nodes)
    div_normalised = div * length / max(lid, 1e-30)
    # the three closed walls: the bottom row, and the two side columns without their top cell, which belongs to
    # the moving lid and is checked separately
    speed = np.hypot(u, v)
    walls = np.concatenate([speed[0, :], speed[1:-1, 0], speed[1:-1, -1]])
    lid_u = float(u[-1, :].mean())

    out: dict[str, Any] = {
        "continuity": {"value": div_normalised, "tol": tol_div, "passed": bool(div_normalised <= tol_div)},
        "lid": {"value": lid_u, "expected": lid, "tol": 0.1 * abs(lid),
                "passed": bool(abs(lid_u - lid) <= 0.1 * abs(lid))},
        "walls": {"value": float(walls.max()), "tol": tol_wall, "passed": bool(walls.max() <= tol_wall)},
    }

    xs, ys = stations(result)
    reynolds = int(summary["Re"])
    if reynolds in GHIA["u"]:
        out["ghia_u"] = compare_to_ghia(centerline(u, direction="vertical") / lid, ys, reynolds=reynolds,
                                        component="u", tol=tol_ghia)
        out["ghia_v"] = compare_to_ghia(centerline(v, direction="horizontal") / lid, xs, reynolds=reynolds,
                                        component="v", tol=tol_ghia)
    out["passed"] = all(item["passed"] for key, item in out.items() if key != "passed")
    return out


def validate(*, reynolds: tuple[int, ...] = (100, 1000), nx: int = 32, ny: int = 32, iterations: int = 100,
             tol_ghia: float = 0.05, engine: str = "auto") -> dict[str, Any]:
    """Run the cavity at each Reynolds number and check it; the report says which engine produced the numbers.

    ``engine="auto"`` uses PICT when it can run here and this repository's local solver otherwise (a Mac, a CI box,
    any machine without CUDA). Nothing is relabelled: every row carries the engine that ran, and the report names it
    at the top. ``nx``/``ny`` must be large enough for the Reynolds number being claimed -- a 32x32 grid at Re = 1000
    is a smoke test, and the rms against Ghia says so rather than a footnote.
    """
    from .case import CavityCase, pict_availability
    from .runner import PICTRunConfig, run_cavity

    availability = pict_availability()
    resolved = ("pict" if availability.available else "local") if engine == "auto" else engine
    if resolved == "pict" and not availability.available:
        return {"available": False, "engine": resolved, "reason": "; ".join(availability.missing),
                "hint": availability.hint, "rows": []}

    rows = []
    for re in reynolds:
        config = PICTRunConfig(iterations=iterations) if resolved == "pict" else None
        result = run_cavity(CavityCase(nx=nx, ny=ny, Re=re, refine=False), config, engine=resolved)
        checks = cavity_checks(result, tol_ghia=tol_ghia)
        rows.append({"Re": int(re), "nx": nx, "ny": ny, "engine": result.engine,
                     "iterations": iterations if resolved == "pict" else result.meta.get("steps"),
                     "checks": checks, "wall_time_s": result.wall_time_s})
    return {"available": True, "engine": resolved, "rows": rows,
            "passed": all(row["checks"]["passed"] for row in rows)}
