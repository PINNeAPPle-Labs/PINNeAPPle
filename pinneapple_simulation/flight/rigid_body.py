"""6-DOF rigid-body equations in body axes with a unit quaternion for the attitude.

Frames: earth NED (x north, y east, z down, flat non-rotating earth), body (x forward, y right wing, z down). The
attitude quaternion q = [q0, q1, q2, q3] (scalar first) rotates body vectors into NED: v_ned = C(q) v_body. Euler
angles (yaw psi, pitch theta, roll phi, 3-2-1 order) are only an output: the quaternion has no singularity at
theta = +-90 deg, and :func:`quat_to_euler` stays finite there (it clips the pitch sine and splits yaw / roll by the
quaternion, not by the division that blows up).

State vector (13): position NED [m], velocity in body axes (u, v, w) [m/s], quaternion, angular velocity in body axes
(p, q, r) [rad/s]:

    x_dot = C(q) v_b
    v_b_dot = F_b / m - omega x v_b              (F_b includes gravity)
    q_dot = 1/2 q (x) [0, omega] - k (|q|^2 - 1) q
    omega_dot = I^-1 (M_b - omega x I omega)

The last term of q_dot pulls the quaternion back to unit norm (Baumgarte-type); :func:`rk4_step` also renormalises.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

G0 = 9.80665
IDX_POS, IDX_VEL, IDX_QUAT, IDX_OMEGA = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return np.array([a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                     a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                     a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
                     a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Rotation matrix body -> NED (columns: the body axes in NED)."""
    q0, q1, q2, q3 = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]])


def euler_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """Roll, pitch, yaw [rad] (3-2-1: yaw, then pitch, then roll) -> quaternion body -> NED."""
    cr, sr = math.cos(phi / 2), math.sin(phi / 2)
    cp, sp = math.cos(theta / 2), math.sin(theta / 2)
    cy, sy = math.cos(psi / 2), math.sin(psi / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """(phi, theta, psi) [rad]. At the gimbal lock (|theta| = 90 deg) only phi -+ psi is defined: the split puts it
    all in psi (phi = 0), so the result stays finite and round-trips the attitude."""
    q0, q1, q2, q3 = np.asarray(q, float) / np.linalg.norm(q)
    s = 2 * (q0 * q2 - q1 * q3)
    if abs(s) >= 1 - 1e-10:
        theta = math.copysign(math.pi / 2, s)
        psi = 2 * math.atan2(q3, q0)                    # = psi - phi (theta = +90) or psi + phi (theta = -90)
        return 0.0, theta, (psi + math.pi) % (2 * math.pi) - math.pi
    phi = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    theta = math.asin(s)
    psi = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    return phi, theta, psi


def state_vector(pos=(0, 0, 0), vel_body=(0, 0, 0), euler=(0, 0, 0), omega=(0, 0, 0)) -> np.ndarray:
    return np.r_[np.asarray(pos, float), np.asarray(vel_body, float), euler_to_quat(*euler),
                 np.asarray(omega, float)]


def rigid_body_derivative(x: np.ndarray, F_b: np.ndarray, M_b: np.ndarray, mass: float, inertia: np.ndarray,
                          inertia_inv: np.ndarray | None = None, k_norm: float = 1.0) -> np.ndarray:
    """Time derivative of the 13-state for total body force F_b (gravity included) and moment M_b about the CG."""
    v = x[IDX_VEL]
    q = x[IDX_QUAT]
    w = x[IDX_OMEGA]
    C = quat_to_dcm(q)
    Iinv = np.linalg.inv(inertia) if inertia_inv is None else inertia_inv
    dx = np.empty(13)
    dx[IDX_POS] = C @ v
    dx[IDX_VEL] = F_b / mass - np.cross(w, v)
    dx[IDX_QUAT] = 0.5 * quat_multiply(q, np.r_[0.0, w]) - k_norm * (q @ q - 1.0) * q
    dx[IDX_OMEGA] = Iinv @ (M_b - np.cross(w, inertia @ w))
    return dx


def gravity_body(q: np.ndarray, mass: float, g: float = G0) -> np.ndarray:
    return quat_to_dcm(q).T @ np.array([0.0, 0.0, mass * g])


def rk4_step(f: Callable[[float, np.ndarray], np.ndarray], t: float, x: np.ndarray, dt: float) -> np.ndarray:
    """One classical Runge-Kutta step; the quaternion is renormalised afterwards."""
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)
    y = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    y[IDX_QUAT] /= np.linalg.norm(y[IDX_QUAT])
    return y


def kinetic_energy(x: np.ndarray, mass: float, inertia: np.ndarray) -> float:
    v, w = x[IDX_VEL], x[IDX_OMEGA]
    return 0.5 * mass * float(v @ v) + 0.5 * float(w @ inertia @ w)


def angular_momentum_ned(x: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    """Angular momentum about the CG in NED (constant for a torque-free body)."""
    return quat_to_dcm(x[IDX_QUAT]) @ (inertia @ x[IDX_OMEGA])
