"""What this solver is checked against, and the numbers it reproduces.

The checks are of two kinds. The first four are exact or near-exact properties of the machinery (a panel integral
against a quadrature, the potential flow past a cylinder against ``Cp = 1 - 4 sin^2 theta``, the attached sheet of a
spinning cylinder against its surface speed, and the diffusion drift against the growth of a Gaussian vorticity
patch). The last two are the physics: the drag of a cylinder at ``Re = 40`` against Dennis & Chang and the Strouhal
number at ``Re = 200`` against Williamson & Brown's correlation.

``run(quick=True)`` does the cheap ones (a second); ``run()`` adds the two runs (minutes).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import kernels
from .bodies import Assembly, cylinder, rotating
from .boundary import SourceSheet

__all__ = ["run", "check_panel_influence", "check_potential_flow", "check_spinning_cylinder",
           "check_diffusion", "check_cylinder_drag", "check_strouhal"]


def _quadrature_velocity(points: np.ndarray, a: np.ndarray, b: np.ndarray, strength: np.ndarray,
                         kind: str = "vortex", n: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """The same field as :func:`~.kernels.panel_flux`, by brute-force quadrature of the point kernels."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    u = np.zeros(len(points))
    v = np.zeros(len(points))
    s = (np.arange(n) + 0.5) / n
    for k in range(len(a)):
        pts = a[k][None, :] + s[:, None] * (b[k] - a[k])[None, :]
        dl = np.linalg.norm(b[k] - a[k]) / n
        dr = points[:, None, :] - pts[None, :, :]
        r2 = (dr ** 2).sum(-1)
        g = strength[k] * dl
        if kind == "vortex":
            u += (g / (2 * np.pi)) * (-dr[..., 1] / r2).sum(axis=1)
            v += (g / (2 * np.pi)) * (dr[..., 0] / r2).sum(axis=1)
        else:
            u += (g / (2 * np.pi)) * (dr[..., 0] / r2).sum(axis=1)
            v += (g / (2 * np.pi)) * (dr[..., 1] / r2).sum(axis=1)
    return u, v


def check_panel_influence(seed: int = 3) -> dict[str, Any]:
    """The closed-form panel integrals against a 20 000-point quadrature, away from the panels."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
    vertices = np.column_stack([np.cos(theta), np.sin(theta)])
    a, b = vertices, np.roll(vertices, -1, axis=0)
    strength = rng.normal(size=len(a))
    points = rng.normal(size=(40, 2))
    points = points * (1.0 + 0.6 / np.hypot(*points.T))[:, None]
    errors = {}
    for kind in ("vortex", "source"):
        ua, va = kernels.panel_flux(points, a, b, strength, kind)
        un, vn = _quadrature_velocity(points, a, b, strength, kind)
        errors[kind] = float(max(np.abs(ua - un).max() / np.abs(un).max(),
                                 np.abs(va - vn).max() / np.abs(vn).max()))
    worst = max(errors.values())
    return {"name": "panel_influence", "ok": worst < 1e-6,
            "detail": f"relative error against quadrature: {worst:.2e} (vortex {errors['vortex']:.1e}, "
                      f"source {errors['source']:.1e})", "value": worst}


def check_potential_flow(panels: int = 80) -> dict[str, Any]:
    """``Cp = 1 - 4 sin^2(theta)`` on a cylinder in a uniform stream, from the source sheet alone."""
    assembly = Assembly([cylinder(1.0, panels)])
    panelset = assembly.panels(0.0)
    theta = np.arctan2(panelset.mid[:, 1], panelset.mid[:, 0])
    u_inf = np.array([1.0, 0.0])
    matrix = kernels.source_sheet_system(panelset.mid, panelset.a, panelset.b, panelset.normal)
    sheet = SourceSheet.solve(panelset, u_inf, np.zeros_like(panelset.mid), matrix)
    u, v = sheet.at_midpoints()
    cp = 1.0 - ((u + 1.0) ** 2 + v ** 2)
    exact = 1.0 - 4.0 * np.sin(theta) ** 2
    error = float(np.abs(cp - exact).max())
    return {"name": "potential_flow", "ok": error < 1e-6,
            "detail": f"largest |Cp - (1 - 4 sin^2 theta)| = {error:.2e} on {panels} panels", "value": error}


def check_spinning_cylinder(rate: float = 1.0, panels: int = 64) -> dict[str, Any]:
    """A cylinder spinning in a fluid at rest carries a uniform sheet equal to its surface speed."""
    assembly = Assembly([cylinder(1.0, panels, motion=rotating((0.0, 0.0), rate))])
    panelset = assembly.panels(0.0)
    matrix = kernels.vortex_sheet_system(panelset.mid, panelset.a, panelset.b, panelset.tangent)
    target = (panelset.velocity * panelset.tangent).sum(axis=1)
    gamma = np.linalg.solve(matrix, target)
    expected = 0.5 * rate                      # radius 0.5 times the angular rate
    error = float(np.abs(gamma - expected).max() / abs(expected))
    return {"name": "spinning_cylinder", "ok": error < 0.05,
            "detail": f"sheet {gamma.mean():.4f} against surface speed {expected:.4f} "
                      f"({100 * error:.1f} %), uniform to {gamma.std():.1e}", "value": error}


def check_diffusion(re: float = 200.0, n: int = 3000, sigma: float = 0.2) -> dict[str, Any]:
    """Viscosity as a drift: the diffusion velocity of a Gaussian patch must be ``2 nu r / sigma^2``.

    The vorticity of a patch ``omega = exp(-r^2/sigma^2)`` has ``grad(omega)/omega = -2 r/sigma^2``, so the drift
    ``-nu grad(omega)/omega`` grows linearly with the radius and equals ``2 nu / sigma`` at ``r = sigma``. That is
    what Dynnikova's kernel estimate of ``I2/I1`` has to reproduce, which is the whole of the viscous term.
    """
    nu = 1.0 / re
    rng = np.random.default_rng(1)
    r = sigma * np.abs(rng.normal(size=n)) * 0.8
    theta = rng.uniform(0, 2 * np.pi, n)
    x, y = r * np.cos(theta), r * np.sin(theta)
    # a smooth patch: equal circulations, uniform in angle (a Gaussian in radius is close enough)
    gamma = np.full(n, 1.0)
    eps = np.full(n, sigma * 0.08)
    vx, vy = kernels.vortex_diffusion_drift(x, y, gamma, eps, nu, same_sign=True)
    radial = (vx * x + vy * y) / np.maximum(np.hypot(x, y), 1e-12)
    near = (np.hypot(x, y) > 0.6 * sigma) & (np.hypot(x, y) < 1.4 * sigma)
    measured = float(np.median(radial[near])) if np.any(near) else float("nan")
    expected = 2.0 * nu / sigma
    rel = abs(measured - expected) / expected
    return {"name": "diffusion", "ok": rel < 0.5,
            "detail": f"radial drift at r = sigma: {measured:.3e} against 2 nu / sigma = {expected:.3e} "
                      f"({100 * rel:.0f} %); it points outward, which is what carries vorticity down the gradient",
            "value": measured}


def check_cylinder_drag(re: float = 40.0, panels: int = 96, t_end: float = 10.0,
                        progress: Any = None) -> dict[str, Any]:
    """Steady drag of a cylinder at ``Re = 40`` against Dennis & Chang (1970): ``Cd = 1.3375``."""
    from .cases import cylinder_case
    from .study import run_study

    result = run_study(cylinder_case(re=re, panels=panels, t_end=t_end), progress=progress)
    cd = result.coefficients["cd"]
    dynamic = 0.5 * result.case.config.rho * result.case.config.velocity_scale ** 2 * result.case.config.reference_length
    friction = float(result.simulation.friction[0]) / dynamic
    target = 1.3375
    rel = abs(cd - target) / target
    return {"name": "cylinder_drag", "ok": rel <= 0.15,
            "detail": f"Cd = {cd:.3f} against 1.3375 (Dennis & Chang 1970), {100 * rel:+.1f} % "
                      f"(pressure {cd - friction:.3f} + friction {friction:.3f})",
            "value": cd, "target": target}


def check_strouhal(re: float = 200.0, panels: int = 96, t_end: float = 60.0,
                   progress: Any = None) -> dict[str, Any]:
    """Shedding frequency at ``Re = 200`` against Williamson & Brown's correlation (``St = 0.1968``)."""
    from .cases import cylinder_case
    from .study import run_study

    result = run_study(cylinder_case(re=re, panels=panels, t_end=t_end), progress=progress)
    st = result.coefficients.get("strouhal")
    target = 0.2731 - 1.1129 / np.sqrt(re) + 0.4821 / re
    ok = st is not None and abs(st - target) / target <= 0.15
    detail = (f"St = {st:.3f} against {target:.4f} (Williamson & Brown 1998)" if st is not None
              else "the lift signal never became periodic: the wake did not shed")
    return {"name": "strouhal", "ok": bool(ok), "detail": detail, "value": st, "target": target}


def run(quick: bool = True, **kwargs: Any) -> list[dict[str, Any]]:
    """Every check, cheap ones first. ``quick=False`` adds the two cylinder runs (minutes each)."""
    checks = [check_panel_influence(), check_potential_flow(), check_spinning_cylinder(), check_diffusion()]
    if not quick:
        checks.append(check_cylinder_drag(**kwargs))
        checks.append(check_strouhal(**{k: v for k, v in kwargs.items() if k in ("panels", "t_end")}))
    return checks
