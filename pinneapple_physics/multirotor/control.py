"""A cascaded controller, so the vehicle actually flies while its faults are being diagnosed.

Position -> desired acceleration -> a thrust vector and an attitude -> body rates -> torques -> the mixer. Each loop is
proportional (with damping on the velocity) because this is not the subject of the issue: what matters is that the
vehicle holds a trajectory under a fault until the fault is detected, so the residual is measured on a flying vehicle
rather than on an open-loop divergence.

The controller is deliberately given **no knowledge of the faults** -- it sees the nominal model and the sensors -- so a
payload or a degraded motor makes it work harder, which is exactly what shows up in the residual.
"""
from __future__ import annotations

import math

import numpy as np

from .model import Multirotor, State

__all__ = ["Controller", "Trajectory", "quaternion_error"]


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quaternion_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """``conj(current) * target``: the attitude error as a rotation vector **in the body frame**.

    The order matters and was measured: ``target * conj(current)`` gives the same error in the *world* frame, and using
    it as a body-rate command couples a yaw error into roll and pitch -- the vehicle tumbles as soon as the heading
    moves, which is exactly what the circle trajectory does.
    """
    wt, xt, yt, zt = target
    wc, xc, yc, zc = current
    w = wc * wt + xc * xt + yc * yt + zc * zt
    x = wc * xt - xc * wt - yc * zt + zc * yt
    y = wc * yt + xc * zt - yc * wt - zc * xt
    z = wc * zt - xc * yt + yc * xt - zc * wt
    if w < 0.0:                                   # the short way round
        w, x, y, z = -w, -x, -y, -z
    return 2.0 * np.array([x, y, z])


class Trajectory:
    """Where the vehicle is asked to be: hover, a circle, a climb, or a step -- enough to exercise the dynamics."""

    def __init__(self, kind: str = "hover", *, altitude: float = 5.0, radius: float = 4.0, period_s: float = 20.0,
                 climb_rate: float = 0.5, yaw_rate: float = 0.0):
        self.kind = kind
        self.altitude = altitude
        self.radius = radius
        self.period_s = period_s
        self.climb_rate = climb_rate
        self.yaw_rate = yaw_rate

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """``(position, velocity, yaw)`` at time ``t``."""
        if self.kind == "circle":
            w = 2 * math.pi / self.period_s
            p = np.array([self.radius * math.cos(w * t), self.radius * math.sin(w * t), -self.altitude])
            v = np.array([-self.radius * w * math.sin(w * t), self.radius * w * math.cos(w * t), 0.0])
            return p, v, _wrap(w * t + math.pi / 2)
        if self.kind == "climb":
            return (np.array([0.0, 0.0, -(self.altitude + self.climb_rate * t)]), np.array([0.0, 0.0, -self.climb_rate]),
                    self.yaw_rate * t)
        return np.array([0.0, 0.0, -self.altitude]), np.zeros(3), self.yaw_rate * t


class Controller:
    """The cascade. Gains are per loop and have units, so they can be read: ``kp_pos`` is 1/s, ``kp_att`` is 1/s."""

    def __init__(self, model: Multirotor, *, kp_pos: float = 2.0, kd_pos: float = 3.2, kp_att: float = 6.0,
                 kp_rate: float = 0.12, ki_rate: float = 0.25, max_tilt_rad: float = 0.55,
                 max_thrust_factor: float = 2.2):
        self.model = model
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_att = kp_att
        self.kp_rate = kp_rate
        self.ki_rate = ki_rate
        self.max_tilt = max_tilt_rad
        self.max_thrust_factor = max_thrust_factor
        self._integral = np.zeros(3)

    def command(self, state: State, target_position: np.ndarray, target_velocity: np.ndarray, yaw: float,
                dt: float) -> np.ndarray:
        """The four motor speeds (normalised to ``omega_max``) that chase the target."""
        af = self.model.airframe
        e_pos = np.asarray(target_position, float) - state.position
        e_vel = np.asarray(target_velocity, float) - state.velocity
        accel = self.kp_pos * e_pos + self.kd_pos * e_vel
        accel = np.clip(accel, -8.0, 8.0)
        force = af.mass_kg * (accel + np.array([0.0, 0.0, -af.gravity]))

        magnitude = float(np.linalg.norm(force))
        magnitude = min(max(magnitude, 0.15 * af.weight_n), self.max_thrust_factor * af.weight_n)
        # the direction the body -z axis has to point (body -z is "up" in the world when level)
        direction = force / max(float(np.linalg.norm(force)), 1e-9)
        tilt = math.acos(min(max(float(direction @ np.array([0.0, 0.0, -1.0])), -1.0), 1.0))
        if tilt > self.max_tilt:
            flat = direction + np.array([0.0, 0.0, 1.0])
            flat = flat / max(float(np.linalg.norm(flat)), 1e-9)
            direction = math.cos(self.max_tilt) * np.array([0.0, 0.0, -1.0]) + math.sin(self.max_tilt) * flat
            direction = direction / np.linalg.norm(direction)
            magnitude = min(magnitude, af.weight_n / max(math.cos(self.max_tilt), 1e-3))

        cy, sy = math.cos(yaw), math.sin(yaw)
        target = self._attitude_from(direction, np.array([cy, sy, 0.0]))
        rate_target = self.kp_att * quaternion_error(target, state.attitude)
        rate_target = np.clip(rate_target, -4.0, 4.0)
        self._integral = np.clip(self._integral + (rate_target - state.rates) * dt, -0.3, 0.3)
        torque = self.kp_rate * (rate_target - state.rates) + self.ki_rate * self._integral
        torque = np.clip(torque, -0.6, 0.6)
        return self.model.allocate(magnitude, torque)

    @staticmethod
    def _attitude_from(up: np.ndarray, heading: np.ndarray) -> np.ndarray:
        """The quaternion whose body ``-z`` is ``up`` and whose body ``x`` follows ``heading`` as closely as it can."""
        z_body = -up
        z_body = z_body / max(float(np.linalg.norm(z_body)), 1e-9)
        x_body = heading - float(heading @ z_body) * z_body
        if float(np.linalg.norm(x_body)) < 1e-6:
            x_body = np.array([1.0, 0.0, 0.0]) - z_body[0] * z_body
        x_body = x_body / max(float(np.linalg.norm(x_body)), 1e-9)
        y_body = np.cross(z_body, x_body)
        rotation = np.column_stack([x_body, y_body, z_body])
        # matrix -> quaternion (Shepperd)
        trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            q = np.array([0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s,
                          (rotation[0, 2] - rotation[2, 0]) / s, (rotation[1, 0] - rotation[0, 1]) / s])
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            q = np.array([(rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s,
                          (rotation[0, 1] + rotation[1, 0]) / s, (rotation[0, 2] + rotation[2, 0]) / s])
        elif rotation[1, 1] > rotation[2, 2]:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            q = np.array([(rotation[0, 2] - rotation[2, 0]) / s, (rotation[0, 1] + rotation[1, 0]) / s,
                          0.25 * s, (rotation[1, 2] + rotation[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            q = np.array([(rotation[1, 0] - rotation[0, 1]) / s, (rotation[0, 2] + rotation[2, 0]) / s,
                          (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s])
        return q / np.linalg.norm(q)
