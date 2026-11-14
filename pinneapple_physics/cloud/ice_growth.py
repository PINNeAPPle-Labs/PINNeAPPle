"""Integration of the depositional ice growth law for one crystal or for a set of chamber experiments.

This is the forward model side of the reproduction in ``examples/reproductions/lamb2026_icenode``: given a
:class:`~pinneapple_physics.cloud.ice_deposition.DepositionalIceGrowth` (which carries ``T``, ``p``, ``S_i`` and
the kinetics) and an initial mass, integrate Eq. 7 and return the mass ratio ``m(t)/m0`` that the levitation
diffusion chamber measures directly.

    from pinneapple_physics.cloud import DepositionalIceGrowth, solve_mass_ratio
    model = DepositionalIceGrowth(T=233.0, p=97190.0, Si=1.2, kinetics="nelson_baker")
    t = np.arange(0.0, 501.0)
    ratio = solve_mass_ratio(model, t, m0=1.5e-10)      # 0-500 s mass ratio of one crystal

The growth law is stiff when the crystal is small (the kinetic resistance in :func:`modified_diffusivity` makes
``dm/dt`` vary over decades as ``r`` crosses the vapour jump length), so the default solver is SciPy's implicit
``Radau``; ``method="LSODA"`` is roughly twice as fast on the same problems.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .ice_deposition import DepositionalIceGrowth

__all__ = ["solve_mass", "solve_mass_ratio"]


def solve_mass(model: DepositionalIceGrowth, t, m0: float, method: str = "Radau", rtol: float = 1e-6,
               atol: float = 1e-16) -> np.ndarray:
    """Mass in kg at the times ``t`` (s), starting from ``m0`` (kg)."""
    t = np.asarray(t, dtype=np.float64)
    sol = solve_ivp(lambda _t, y: np.atleast_1d(model.dmdt(y[0])), (float(t[0]), float(t[-1])), [float(m0)],
                    t_eval=t, method=method, rtol=rtol, atol=atol)
    if not sol.success:                                                # pragma: no cover - defensive
        raise RuntimeError(f"depositional growth integration failed: {sol.message}")
    return sol.y[0]


def solve_mass_ratio(model: DepositionalIceGrowth, t, m0: float, **kw) -> np.ndarray:
    """Mass ratio ``m(t)/m0``, the quantity the levitation diffusion chamber measures."""
    return solve_mass(model, t, m0, **kw) / float(m0)
