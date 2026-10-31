"""Hyperelastic strain-energy potentials ``W(F)``: stress and tangent from one function.

The first Piola-Kirchhoff stress is ``P = dW/dF`` (depth 1) and the material tangent ``A_iJkL = d2W/dF_iJ dF_kL``
(depth 2, shape ``(3, 3, 3, 3)`` per point). The tangent of a potential has the major symmetry
``A_iJkL = A_kLiJ``; :func:`major_symmetry` measures it for any tangent (from a potential, a material routine or a
learned model).

Neo-Hookean (compressible, Ogden form): ``W = mu/2 (I_1 - 3) - mu ln J + lam/2 (ln J)^2``, with the closed forms
``P = mu (F - F^-T) + lam ln J F^-T`` and ``A_iJkL = mu d_ik d_JL + (mu - lam ln J) Fi_Li Fi_Jk + lam Fi_Ji Fi_Lk``
(``Fi = F^-1``); Bonet & Wood, *Nonlinear Continuum Mechanics for Finite Element Analysis*, 2nd ed. (2008), sections
6.4.3 and 6.5. At ``F = I`` the tangent is the isotropic linear-elastic one, ``mu (d_ik d_JL + d_iL d_Jk) +
lam d_iJ d_kL``.

Mooney-Rivlin (compressible, isochoric invariants): ``W = c10 (I1b - 3) + c01 (I2b - 3) + kappa/2 (J - 1)^2`` with
``I1b = J^(-2/3) I_1``, ``I2b = J^(-4/3) I_2``; small-strain shear modulus ``2 (c10 + c01)``, bulk modulus ``kappa``.
"""
from __future__ import annotations

import numpy as np
import torch

from .core import Potential, Response

__all__ = ["neo_hookean", "mooney_rivlin", "neo_hookean_stress", "neo_hookean_tangent", "isotropic_tangent",
           "major_symmetry"]


def _cauchy(d, x):
    F = x["F"]
    J = torch.linalg.det(F)
    return d[("F",)] @ F.transpose(-1, -2) / J[:, None, None]


def _hyperelastic(fn, name: str, description: str) -> Potential:
    return Potential(fn, {"F": ("1", (3, 3))}, name="W", unit="J/m3", description=description,
                     names={("F",): ("P", 1, "Pa", "first Piola-Kirchhoff stress dW/dF"),
                            ("F", "F"): ("A", 1, "Pa", "material tangent d2W/dFdF")},
                     responses={"sigma": Response(_cauchy, 1, "Pa", "Cauchy stress P F^T / J")})


def neo_hookean(mu: float, lam: float) -> Potential:
    """Compressible Neo-Hookean strain energy ``W(F)`` (J/m3) with shear modulus ``mu`` and Lame ``lam`` (Pa)."""

    def W(F):
        C = F.transpose(-1, -2) @ F
        I1 = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
        lnJ = torch.log(torch.linalg.det(F))
        return 0.5 * mu * (I1 - 3.0) - mu * lnJ + 0.5 * lam * lnJ ** 2

    return _hyperelastic(W, "neo_hookean", f"Neo-Hookean, mu = {mu:g} Pa, lambda = {lam:g} Pa")


def mooney_rivlin(c10: float, c01: float, kappa: float) -> Potential:
    """Compressible Mooney-Rivlin strain energy in isochoric invariants (``c10``, ``c01``, ``kappa`` in Pa)."""

    def W(F):
        C = F.transpose(-1, -2) @ F
        I1 = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
        I2 = 0.5 * (I1 ** 2 - torch.diagonal(C @ C, dim1=-2, dim2=-1).sum(-1))
        J = torch.linalg.det(F)
        return c10 * (J ** (-2.0 / 3.0) * I1 - 3.0) + c01 * (J ** (-4.0 / 3.0) * I2 - 3.0) + 0.5 * kappa * (J - 1) ** 2

    return _hyperelastic(W, "mooney_rivlin", f"Mooney-Rivlin, c10 = {c10:g}, c01 = {c01:g}, kappa = {kappa:g} Pa")


def neo_hookean_stress(F: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """Closed-form ``P = mu (F - F^-T) + lam ln J F^-T`` (``F`` of shape ``(..., 3, 3)``)."""
    F = np.asarray(F, float)
    FiT = np.swapaxes(np.linalg.inv(F), -1, -2)
    lnJ = np.log(np.linalg.det(F))[..., None, None]
    return mu * (F - FiT) + lam * lnJ * FiT


def neo_hookean_tangent(F: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """Closed-form ``A_iJkL`` (Bonet & Wood 2008), shape ``(..., 3, 3, 3, 3)``."""
    F = np.asarray(F, float)
    Fi = np.linalg.inv(F)
    lnJ = np.log(np.linalg.det(F))[..., None, None, None, None]
    eye = np.eye(3)
    t1 = np.einsum("ik,JL->iJkL", eye, eye)
    t2 = np.einsum("...Li,...Jk->...iJkL", Fi, Fi)
    t3 = np.einsum("...Ji,...Lk->...iJkL", Fi, Fi)
    return mu * t1 + (mu - lam * lnJ) * t2 + lam * t3


def isotropic_tangent(mu: float, lam: float) -> np.ndarray:
    """Linear isotropic elasticity ``mu (d_ik d_JL + d_iL d_Jk) + lam d_iJ d_kL``."""
    eye = np.eye(3)
    return (mu * (np.einsum("ik,JL->iJkL", eye, eye) + np.einsum("iL,Jk->iJkL", eye, eye))
            + lam * np.einsum("iJ,kL->iJkL", eye, eye))


def major_symmetry(A: np.ndarray) -> float:
    """Largest ``|A_iJkL - A_kLiJ|`` relative to ``max |A|``: zero for the tangent of a potential."""
    A = np.asarray(A, float)
    At = np.einsum("...iJkL->...kLiJ", A)
    return float(np.abs(A - At).max() / max(np.abs(A).max(), 1e-300))
