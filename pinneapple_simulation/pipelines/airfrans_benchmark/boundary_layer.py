"""The wall shear an integral boundary layer predicts along a section, from its edge velocity alone.

The campaign's remaining gap is the wall shear, and three families of fix have been closed by measurement: no
per-case scale (32.67 % → 28.13 % is the best any calibration could do), no extra near-wall sampling (`wf50`: field
3.13 % → 7.96 %, wall shear 22.05 % → 23.38 %), and no chord curve (projecting onto a smooth curve takes the wall shear
from 22.05 % to 78.7 %). What is left is what a flat-plate law cannot express: the traction varies along the chord
because the *pressure gradient* varies, and it is the pressure gradient that thins or thickens the layer and drives
separation on the aft surface.

That is a solved problem in the classical sense. An integral method carries two or three thicknesses along the chord and
closes them with empirical relations; Thwaites for the laminar run, Head for the turbulent one, Michel for transition.
Given an edge velocity — which :mod:`.panel` supplies from the geometry and the freestream alone, with no truth — it
returns ``τ_w(x)``, and the network's wall prior stops being a constant-coefficient law.

Everything here is validated against a form that predates it: Thwaites on a constant edge velocity against Blasius, Head
on a constant edge velocity against the flat-plate skin-friction law the campaign has been using, and the two together
against the release's own wall shear.
"""
from __future__ import annotations

import numpy as np

__all__ = ["thwaites", "head", "edge_solution", "wall_shear_along"]

#: Michel's criterion for natural transition, ``Re_θ ≥ 1.174 (1 + 22400/Re_x) Re_x^0.46``
MICHEL = (1.174, 22400.0, 0.46)


def thwaites(edge: np.ndarray, x: np.ndarray, nu: float) -> dict[str, np.ndarray]:
    """Thwaites' laminar closure: momentum thickness, shape factor and wall shear along a surface.

    ``θ² = 0.45 ν / U_e⁶ · ∫₀^x U_e⁵ dx'`` and ``τ_w = μ U_e l(λ) / θ`` with ``λ = θ²/ν · dU_e/dx``. The empirical
    ``l(λ)`` is the two-branch fit of the standard formulation; ``λ = −0.09`` is separation."""
    edge = np.asarray(edge, float)
    x = np.asarray(x, float)
    speed = np.maximum(np.abs(edge), 1e-9)
    interval = np.diff(x, prepend=x[0])
    integral = np.cumsum(speed ** 5 * interval)
    theta = np.sqrt(np.maximum(0.45 * nu * integral / speed ** 6, 0.0))
    gradient = np.gradient(speed, x)
    lam = np.where(theta > 0, theta ** 2 / nu * gradient, 0.0)
    lam = np.clip(lam, -0.09, 0.25)
    shape_function = np.where(lam <= 0.0,
                              0.22 + 1.57 * lam - 1.8 * lam ** 2,
                              0.22 + 1.402 * lam + 0.018 * lam / (0.107 + lam))
    with np.errstate(divide="ignore", invalid="ignore"):
        shear = np.where(theta > 0, nu * speed * shape_function / np.maximum(theta, 1e-30), 0.0)
    return {"theta": theta, "lambda": lam, "shear": shear, "edge": speed, "x": x}


def head(edge: np.ndarray, x: np.ndarray, nu: float, start: int = 0,
         theta0: float | None = None, shape0: float = 1.4) -> dict[str, np.ndarray]:
    """Head's turbulent closure by marching ``θ`` and ``H`` in ``x``, with Ludwig–Tillmann's ``c_f``.

    The two march forward together: von Kármán's momentum integral for ``θ``, ``dθ/dx = c_f/2 − (H + 2)(θ/U)dU/dx``,
    and Head's entrainment for the product ``θH₁``, ``d(θH₁)/dx = 0.0306 (H₁ − 3)^{-0.653}``, with ``H₁`` Head's shape
    factor — the ratio of the boundary layer's *thickness* to its momentum thickness, ``(δ − δ*)/θ``, which is 5 to 7 in
    a turbulent layer. ``H`` is recovered from ``H₁`` by inverting the empirical relation, and the skin friction comes
    from ``c_f/2 = 0.123 · 10^{-0.678H} · Re_θ^{-0.268}``.

    Marching ``θH₁`` rather than ``H`` is what keeps this stable: solving for ``dH/dx`` instead divides by ``θ`` at every
    step and the shape factor runs to its bounds, which takes the skin friction down by a factor of three — measured, on
    a constant edge velocity, before this form was used."""
    edge = np.asarray(edge, float)
    x = np.asarray(x, float)
    count = len(x)
    theta, shape, shear = np.zeros(count), np.zeros(count), np.zeros(count)
    if count == 0:
        return {"theta": theta, "shape": shape, "shear": shear, "edge": edge, "x": x}

    def head_shape(shape_factor: float) -> float:
        gap = max(shape_factor - 1.1, 1e-3)
        return 3.3 + 0.8234 * gap ** -1.287 if shape_factor <= 1.6 else 3.3 + 1.4144 * gap ** -0.653

    def head_factor(head1: float) -> float:
        gap = max(head1 - 3.3, 1e-3)
        if head1 >= head_shape(1.6):
            return 1.1 + (0.8234 / gap) ** (1.0 / 1.287)
        return 1.1 + (1.4144 / gap) ** (1.0 / 0.653)

    speed = np.maximum(np.abs(edge), 1e-9)
    gradient = np.gradient(speed, x)
    theta[start] = theta0 if theta0 is not None else 1e-5
    shape[start] = shape0
    product = theta[start] * head_shape(shape0)
    friction = 0.123 * 10.0 ** (-0.678 * shape[start]) * max(speed[start] * theta[start] / nu, 1.0) ** -0.268
    shear[start] = friction * speed[start] ** 2

    for i in range(start + 1, count):
        step = abs(x[i] - x[i - 1])
        if step == 0.0:
            theta[i], shape[i] = theta[i - 1], shape[i - 1]
        else:
            friction = 0.123 * 10.0 ** (-0.678 * shape[i - 1]) * max(speed[i - 1] * theta[i - 1] / nu, 1.0) ** -0.268
            momentum = friction - (shape[i - 1] + 2.0) * theta[i - 1] / speed[i - 1] * gradient[i - 1]
            theta[i] = max(theta[i - 1] + step * momentum, 1e-9)
            product = product + step * 0.0306 * max(head_shape(shape[i - 1]) - 3.0, 1e-3) ** -0.653
            shape[i] = float(np.clip(head_factor(product / theta[i]), 1.05, 2.4))
        friction = 0.123 * 10.0 ** (-0.678 * shape[i]) * max(speed[i] * theta[i] / nu, 1.0) ** -0.268
        shear[i] = friction * speed[i] ** 2
    return {"theta": theta, "shape": shape, "shear": shear, "edge": edge, "x": x}


def edge_solution(edge: np.ndarray, x: np.ndarray, nu: float) -> dict[str, np.ndarray]:
    """The whole chord: Thwaites up to transition (Michel), Head from there, ``θ`` continuous across it.

    ``edge`` is the speed just outside the boundary layer at the stations ``x``, in metres per second and metres. The two
    runs are joined at the first station where Michel's criterion holds, which is what keeps the drag from carrying a
    discontinuity at the transition point."""
    edge = np.asarray(edge, float)
    x = np.asarray(x, float)
    laminar = thwaites(edge, x, nu)
    speed = np.maximum(np.abs(edge), 1e-9)
    reynolds_x = np.maximum(speed * np.maximum(x, 0.0) / nu, 1.0)
    reynolds_theta = np.maximum(speed * laminar["theta"] / nu, 0.0)
    coefficient, offset, exponent = MICHEL
    transition = reynolds_theta >= coefficient * (1.0 + offset / reynolds_x) * reynolds_x ** exponent
    first = int(np.argmax(transition)) if transition.any() else len(x)
    if first >= len(x):  # never trips Michel on this surface: the whole run stays laminar, and Head has nothing to march
        return {"shear": laminar["shear"], "theta": laminar["theta"], "transition": np.full(len(x), first),
                "laminar": laminar, "turbulent": None, "edge": speed, "x": x}

    turbulent = head(edge, x, nu, start=first, theta0=float(laminar["theta"][first]), shape0=1.4)
    shear = np.where(np.arange(len(x)) < first, laminar["shear"], turbulent["shear"])
    return {"shear": shear, "theta": np.where(np.arange(len(x)) < first, laminar["theta"], turbulent["theta"]),
            "transition": np.full(len(x), first), "laminar": laminar, "turbulent": turbulent, "edge": speed, "x": x}


def wall_shear_along(edge: np.ndarray, x: np.ndarray, nu: float) -> np.ndarray:
    """Just the wall shear, in the same units as the edge velocity."""
    return edge_solution(edge, x, nu)["shear"]


#: how far the shape correction is allowed to move the law. The integral method returns the *variation* along the chord
#: and the law the *level*; near the leading edge the variation is a ratio of two quantities that both go to zero, and
#: without a band a handful of nose nodes would decide the readout's prior
CORRECTION_BAND = (0.25, 4.0)

#: how much of the integral model's variation to believe. Measured on 24 training cases, node by node, against the
#: release's own wall shear: the flat-plate law's shape error along the chord has rms log 0.401, the integral model's
#: variation correlates with it at 0.568, and one constant fitted on those cases — ``log R = α · log c`` — removes
#: 17.7 % of it at ``α = 0.322``. The model's variation is right in direction and about three times too large in
#: amplitude, which is what a gain below one means. It is a constant of the model, like the law's own coefficient, not a
#: per-case calibration.
CORRECTION_GAIN = 0.322


def shape_correction(edge: np.ndarray, x: np.ndarray, nu: float,
                     coefficient: float = 0.0576, exponent: float = 0.2) -> np.ndarray:
    """``τ_integral(x) / τ_flat(x)`` — the variation the law cannot express, as a multiplier on it.

    The flat-plate law this campaign's wall prior uses has the right *level* and no *shape*: it is the answer for a
    constant edge velocity, and on a section the pressure gradient changes the layer's thickness by a factor of several
    along the chord. An integral method gets the shape and, driven along the whole chord, gets the level wrong — 109.7 %
    on the release's node-wise wall shear, worse than the law it would replace. The two failures are complementary, so
    this returns the ratio and leaves the level to the law.

    Both sides are the *same* solver: the numerator with the case's edge velocity, the denominator with the flat-plate
    relation, so the ratio is 1 wherever the edge velocity is constant and the systematic error the integral method
    carries cancels."""
    edge = np.asarray(edge, float)
    x = np.asarray(x, float)
    speed = np.maximum(np.abs(edge), 1e-9)
    integral = edge_solution(speed, np.maximum(x, 1e-6), nu)["shear"]
    reynolds_x = np.maximum(speed * np.maximum(x, 1e-6) / nu, 1.0)
    flat = 0.5 * speed ** 2 * coefficient * reynolds_x ** (-exponent)
    ratio = np.where(flat > 0, integral / np.maximum(flat, 1e-30), 1.0)
    return np.clip(ratio, *CORRECTION_BAND)
