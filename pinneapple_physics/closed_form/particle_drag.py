"""Drag coefficient of a rigid sphere in a steady uniform flow, ``C_D = F / (0.5 rho U^2 pi d^2 / 4)`` at
``Re = U d / nu``.

* :func:`clift_sphere_cd`: the standard drag curve of R. Clift, J. R. Grace and M. E. Weber, *Bubbles, Drops and
  Particles*, Academic Press (1978), Table 5.2: ``24/Re (1 + 0.1935 Re^0.6305)`` for 20 < Re <= 260.
* :func:`schiller_naumann_cd`: L. Schiller and A. Naumann, Z. Ver. Dtsch. Ing. 77 (1933) 318-320,
  ``24/Re (1 + 0.15 Re^0.687)`` for Re < 800 (as given in Clift et al., Table 5.2 notes).
"""
from __future__ import annotations

import warnings

__all__ = ["clift_sphere_cd", "schiller_naumann_cd"]


def clift_sphere_cd(re: float) -> float:
    """Sphere drag coefficient of Clift, Grace & Weber (1978), Table 5.2, range 20 < Re <= 260 (warns outside).

    Returns
    -------
    Cd : float [-]
        Drag coefficient.
    """
    if not re > 0:
        raise ValueError(f"Re must be positive, got {re}")
    if not 20.0 < re <= 260.0:
        warnings.warn(f"clift_sphere_cd: Re = {re:g} is outside 20 < Re <= 260", RuntimeWarning, stacklevel=2)
    return 24.0 / re * (1.0 + 0.1935 * re ** 0.6305)


def schiller_naumann_cd(re: float) -> float:
    """Sphere drag coefficient of Schiller & Naumann (1933), Re < 800 (warns above).

    Returns
    -------
    Cd : float [-]
        Drag coefficient.
    """
    if not re > 0:
        raise ValueError(f"Re must be positive, got {re}")
    if re >= 800.0:
        warnings.warn(f"schiller_naumann_cd: Re = {re:g} is above 800", RuntimeWarning, stacklevel=2)
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)
