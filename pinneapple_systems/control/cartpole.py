"""Inverted pendulum on a cart: energy swing-up from hanging, LQR catch at upright, disturbance rejection.

The cart's acceleration is the input, as when a servo drive tracks an acceleration command. The pendulum is a point
mass at distance ``length`` (l) from the pivot, and the angle is measured from upright:

    p'' = u,          theta'' = (g sin(theta) - u cos(theta)) / l + d

``d`` is an angular-acceleration disturbance (a push).

Pipeline, all on :mod:`pinneapple_systems.control`:

1. :func:`~.analysis.linearize` at upright gives ``A, B``. :func:`~.analysis.stability` finds the real pole at
   ``+sqrt(g/l)`` and its doubling time.
2. :func:`~.tuning.lqr` with ``Q, R`` gives the balance gains and the closed-loop poles.
3. An energy-based swing-up (Astrom & Furuta, *Automatica* 36, 2000) pumps the pendulum's energy towards that of
   upright rest, with ``u = k_E E sign(theta' cos theta)`` limited to ``a_swing``. Near upright, the LQR takes over.
4. A push of ``push`` rad/s^2 lasting ``push_duration`` at ``t_push`` tests the disturbance rejection.

Without control, the open loop is simulated from a small tilt, to show how fast it falls.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .analysis import controllability, linearize, stability
from .tuning import lqr

__all__ = ["CartPoleParams", "cartpole_rhs", "cartpole_study"]


@dataclass(frozen=True)
class CartPoleParams:
    """Parameters of the cart-pole study.

    Attributes
    ----------
    g : float [m/s^2]
        Gravity.
    length : float [m]
        Pivot to the pendulum's centre of mass.
    a_balance : float [m/s^2]
        Cart acceleration limit while balancing.
    a_swing : float [m/s^2]
        Cart acceleration limit while pumping energy (gentler, so the swing-up takes several swings).
    k_energy : float [1/(m s^2) scaled]
        Gain of the energy pumping law (cart acceleration per unit of energy error, per kg and per m of l).
    q_position, q_velocity, q_angle, q_rate : float
        LQR state weights (cart position, cart velocity, angle, angular rate).
    r_input : float
        LQR input weight.
    catch_angle : float [rad]
        The LQR takes over below this angle from upright.
    catch_rate : float [rad/s]
        And below this angular rate.
    """
    g: float = 9.81
    length: float = 0.5
    a_balance: float = 15.0
    a_swing: float = 3.0
    k_energy: float = 6.0
    q_position: float = 10.0
    q_velocity: float = 1.0
    q_angle: float = 100.0
    q_rate: float = 1.0
    r_input: float = 0.1
    catch_angle: float = 0.35
    catch_rate: float = 4.0


def cartpole_rhs(x, u, d, t, params: CartPoleParams | None = None):
    """``dx/dt`` of the cart-pole, state ``[p, v, theta, omega]`` (m, m/s, rad from upright, rad/s), input the cart
    acceleration [m/s^2], disturbance an angular acceleration [rad/s^2]."""
    params = params or CartPoleParams()
    p, v, th, w = (float(s) for s in x)
    a = float(np.atleast_1d(u)[0]) if np.size(u) else 0.0
    dd = float(np.atleast_1d(d)[0]) if np.size(d) else 0.0
    return np.array([v, a, w, (params.g * math.sin(th) - a * math.cos(th)) / params.length + dd])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def cartpole_study(params: CartPoleParams | None = None, theta0_deg: float = 177.0, t_end: float = 16.0,
                   dt: float = 0.005, t_push: float = 12.0, push: float = -25.0, push_duration: float = 0.1,
                   free_tilt_deg: float = 1.0) -> dict[str, Any]:
    """Swing-up from (near) hanging, LQR catch, a push, and the uncontrolled fall for comparison.

    Parameters
    ----------
    params : CartPoleParams
        Model and controller parameters.
    theta0_deg : float [deg]
        Initial angle from upright (180 is hanging straight down).
    t_end, dt : float [s]
        Duration and fixed RK4 step.
    t_push : float [s]
        Time of the push.
    push : float [rad/s^2]
        Angular-acceleration disturbance during the push.
    push_duration : float [s]
        Length of the push.
    free_tilt_deg : float [deg]
        Initial tilt of the uncontrolled fall.

    Returns
    -------
    unstable_pole : float [1/s]
        Largest real part of the open-loop eigenvalues (``sqrt(g/l)``).
    doubling_time : float [s]
        ``ln 2 / unstable_pole``.
    controllable : bool
        Rank of the controllability matrix is full.
    lqr_gain : list of float
        ``u = -K x`` gains (m/s^2 per m, per m/s, per rad, per rad/s).
    closed_loop_poles : list of complex
        Eigenvalues of ``A - B K`` (1/s).
    catch_time : float [s]
        When the LQR took over.
    settled_time : float [s]
        From then on within 2 deg of upright until the push (``nan`` if never).
    push_peak_deg : float [deg]
        Largest angle after the push.
    push_recovery : float [s]
        Time after the push to stay within 2 deg.
    max_accel : float [m/s^2]
        Largest cart acceleration used.
    cart_travel : list of float [m]
        Minimum and maximum cart position.
    free_fall_time_to_90deg : float [s]
        Uncontrolled: time from ``free_tilt_deg`` to 90 deg.
    t, x, u, mode : arrays
        The controlled trajectory (s; state units; m/s^2; 'swing' or 'lqr' per step).
    """
    P = params or CartPoleParams()

    def f(x, u, d=0.0, t=0.0):
        return cartpole_rhs(x, u, d, t, P)

    lin = linearize(f, [0, 0, 0, 0], [0.0])
    A, B = lin["A"], lin["B"]
    st = stability(A)
    K = lqr(A, B, np.diag([P.q_position, P.q_velocity, P.q_angle, P.q_rate]), [[P.r_input]])["K"]
    cl = stability(A - B @ K)

    def energy(th, w):            # zero at upright rest, per unit mass
        return 0.5 * (P.length * w) ** 2 + P.g * P.length * (math.cos(th) - 1.0)

    def control(x, mode):
        p, v, th, w = x
        thw = _wrap(th)
        if mode == "lqr" or (abs(thw) < P.catch_angle and abs(w) < P.catch_rate):
            u = float(-(K @ np.array([p, v, thw, w]))[0])
            return max(-P.a_balance, min(P.a_balance, u)), "lqr"
        u = P.k_energy * energy(thw, w) * math.copysign(1.0, w * math.cos(thw) + 1e-12) - 2.0 * p - 2.0 * v
        return max(-P.a_swing, min(P.a_swing, u)), "swing"

    def rk4(x, u, d):
        k1 = f(x, u, d)
        k2 = f(x + dt / 2 * k1, u, d)
        k3 = f(x + dt / 2 * k2, u, d)
        k4 = f(x + dt * k3, u, d)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    n = int(round(t_end / dt))
    x = np.array([0.0, 0.0, math.radians(theta0_deg), 0.0])
    T, X, U, M = [], [], [], []
    mode = "swing"
    for i in range(n):
        t = i * dt
        u, mode = control(x, mode)
        d = push if t_push <= t < t_push + push_duration else 0.0
        x = rk4(x, u, d)
        T.append(t + dt)
        X.append(x.copy())
        U.append(u)
        M.append(mode)
    T, X, U = np.array(T), np.array(X), np.array(U)
    th = np.array([_wrap(a) for a in X[:, 2]])
    two = math.radians(2.0)
    catch = M.index("lqr") if "lqr" in M else None
    i_push = int(round(t_push / dt))
    settled = float("nan")
    if catch is not None and catch < i_push:
        seg = np.abs(th[catch:i_push]) < two
        bad = np.where(~seg)[0]
        k = 0 if not bad.size else bad[-1] + 1
        if k < seg.size:
            settled = float(T[catch + k])
    post = np.abs(th[i_push:])
    peak = float(np.degrees(post.max())) if post.size else float("nan")
    badp = np.where(post >= two)[0]
    recovery = float((badp[-1] + 1) * dt) if badp.size and badp[-1] + 1 < post.size else (0.0 if post.size else
                                                                                           float("nan"))
    # uncontrolled fall from a small tilt
    xf = np.array([0.0, 0.0, math.radians(free_tilt_deg), 0.0])
    fall = float("nan")
    for i in range(int(round(5.0 / dt))):
        xf = rk4(xf, 0.0, 0.0)
        if abs(xf[2]) >= math.pi / 2:
            fall = (i + 1) * dt
            break
    mr = st["max_real"]
    return {"params": asdict(P), "unstable_pole": mr, "doubling_time": math.log(2) / mr if mr > 0 else math.inf,
            "controllable": controllability(A, B)["controllable"], "lqr_gain": K.ravel().tolist(),
            "closed_loop_poles": [complex(s) for s in cl["s"]], "closed_loop_classification": cl["classification"],
            "catch_time": float(T[catch]) if catch is not None else float("nan"), "settled_time": settled,
            "push_peak_deg": peak, "push_recovery": recovery, "max_accel": float(np.max(np.abs(U))),
            "cart_travel": [float(X[:, 0].min()), float(X[:, 0].max())], "free_fall_time_to_90deg": fall,
            "t": T, "x": X, "u": U, "mode": M}
