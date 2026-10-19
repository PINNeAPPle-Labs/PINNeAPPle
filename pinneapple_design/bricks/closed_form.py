"""Closed-form references for brick statics (used to validate :mod:`.statics`, :mod:`.dynamics` and :mod:`.fea`).

* Tipping: a rigid body on a smooth support stands while its centre of mass is over the support polygon.
* Harmonic stacking of n identical unconnected blocks of length L: the block k-th from the top overhangs the one
  below it by L / (2 k), so the top block overhangs the table edge by (L / 2) H_n (H_n the n-th harmonic number).
  Johnson, "Leaning Tower of Lire", Am. J. Phys. 23, 240 (1955); Paterson and Zwick, "Overhang", Am. Math.
  Monthly 116(1), 19-44 (2009). This is the optimum for one block per level.
* Stud cantilever: a part hanging off the edge of a fixed one, held by a row of studs at x_s inside the contact
  patch [x0, x1], pivots about the outer edge x1. With weight W at x = d > x1 the most favourable distribution
  loads every stud equally: t = W (d - x1) / sum_s (x1 - x_s).
* Simply supported beam of span L, self-weight w per length and a central load P: reactions P / 2 + w L / 2,
  bending moment M(x) = (P / 2 + w L / 2) x - w x^2 / 2 for x <= L / 2, and Euler-Bernoulli mid-span deflection
  P L^3 / (48 E I) + 5 w L^4 / (384 E I), I = second moment of area.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def harmonic(n: int) -> float:
    return float(sum(1.0 / k for k in range(1, n + 1)))


def harmonic_overhang(n: int, length: float) -> float:
    """Maximum overhang of ``n`` unconnected identical blocks, one per level: (L / 2) H_n."""
    return 0.5 * length * harmonic(n)


def harmonic_stack(n: int, length: float, edge: float = 0.0) -> np.ndarray:
    """Left-edge x positions of the blocks (bottom first) of the harmonic stack on a table ending at ``edge``."""
    right = np.empty(n)
    right[n - 1] = edge + harmonic_overhang(n, length)          # top block
    for i in range(n - 2, -1, -1):                              # block i is (n - i)-th from the top
        k = n - 1 - i
        right[i] = right[i + 1] - length / (2 * k)
    return right - length


def tip_over_offset(length: float) -> float:
    """Largest overhang of a single uniform block on a table edge: half its length."""
    return 0.5 * length


def stud_cantilever_tension(weight: float, d: float, studs_x: Sequence[float], x1: float) -> float:
    """Equal tension per stud [N] that holds a part of weight ``weight`` [N] with its centre of mass at ``d``,
    pivoting about the patch edge ``x1`` (0 if the centre of mass is inside the patch)."""
    lever = float(np.sum(x1 - np.asarray(studs_x, float)))
    if d <= x1:
        return 0.0
    if lever <= 0:
        return float("inf")
    return weight * (d - x1) / lever


def beam_reactions(span: float, w: float, P: float) -> float:
    """Each support reaction of a symmetric simply supported beam [N]."""
    return 0.5 * P + 0.5 * w * span


def beam_moment(x: float, span: float, w: float, P: float) -> float:
    """Bending moment at ``x`` (from a support) of the simply supported beam with a central load [N mm]."""
    x = min(x, span - x)
    return (0.5 * P + 0.5 * w * span) * x - 0.5 * w * x * x


def beam_midspan_deflection(span: float, E: float, inertia: float, P: float = 0.0, w: float = 0.0) -> float:
    """Euler-Bernoulli mid-span deflection of a simply supported beam (consistent units)."""
    return P * span ** 3 / (48 * E * inertia) + 5 * w * span ** 4 / (384 * E * inertia)


def tower_tip_force(weight: float, half_base: float, height: float) -> float:
    """Horizontal force at ``height`` that tips a rigid block of weight ``weight`` about the base edge at
    ``half_base`` from its centre of mass line: F = W b / h."""
    return weight * half_base / height
