"""Analytic checks of a MuJoCo run — the PINNeAPPle trust layer for rigid-body simulation.

Each case builds a minimal MJCF whose answer is known in closed form, runs it through :class:`~.runner.MuJoCoRunner`
and compares. When a case fails, the model, the time step or the contact settings are wrong — not the paper — so the
checks are run before a MuJoCo-generated dataset is trusted as training data.

Cases
-----
``free_fall``        z(t) = z0 - g t^2 / 2 and v = -g t for a free body.
``pendulum_period``  small-oscillation period T = 2 pi sqrt(L / g), measured from the zero crossings.
``sliding_friction`` stopping distance v0^2 / (2 mu g) of a box sliding on a plane.
``gravity_torque``   the torque a pendulum needs to be held at an angle, m g L sin(theta), against ``qfrc_bias``.
``torque_free_spin`` kinetic energy and angular momentum of a free body, with no torque acting on it.
``resting_contact``  a box settled on a plane stays put (no sinking, contacts present).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .loader import MuJoCoConfig, load_model, make_data, require_mujoco
from .runner import MuJoCoRunner

__all__ = ["CASES", "validate", "summary"]

CASES = ("free_fall", "pendulum_period", "sliding_friction", "gravity_torque", "torque_free_spin", "resting_contact")

GRAVITY = 9.81


def _row(case: str, value: float, limit: float, passed: bool | None = None, note: str = "") -> dict:
    return {"case": case, "value": float(value), "limit": float(limit),
            "passed": bool(value <= limit) if passed is None else bool(passed), "note": note}


def _error(measured: float, expected: float) -> float:
    return abs(measured - expected) / max(abs(expected), 1e-12)


# ---------------------------------------------------------------------------- cases
def free_fall(dt: float = 1e-3) -> dict:
    """A body in free fall for 0.3 s: position and velocity against the closed form."""
    duration, z0 = 0.3, 1.0
    xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 {-GRAVITY}"/><worldbody>
    <body name="ball" pos="0 0 {z0}"><freejoint/><geom type="sphere" size="0.05" mass="1"/></body>
    </worldbody></mujoco>"""
    cfg = MuJoCoConfig(model_xml=xml, n_steps=int(round(duration / dt)), dt=dt)
    traj = MuJoCoRunner(cfg).simulate()
    z, vz = traj.xpos[-1, 1, 2], traj.cvel[-1, 1, 5]
    value = max(_error(z, z0 - 0.5 * GRAVITY * duration ** 2), _error(vz, -GRAVITY * duration))
    return _row("free_fall", value, 5e-3, note="relative error of z(t) and v(t)")


def pendulum_period(dt: float = 1e-3, length: float = 0.5, amplitude: float = 0.05) -> dict:
    """Small-oscillation period of a point mass on a massless rod, from the zero crossings of the angle."""
    xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 {-GRAVITY}"/><worldbody>
    <body name="bob" pos="0 0 0"><joint name="hinge" type="hinge" axis="0 1 0" damping="0"/>
    <geom type="sphere" size="0.02" mass="1" pos="0 0 {-length}"/></body></worldbody></mujoco>"""
    duration = 4.0
    cfg = MuJoCoConfig(model_xml=xml, n_steps=int(round(duration / dt)), dt=dt)
    traj = MuJoCoRunner(cfg).simulate(q0=[amplitude])
    crossings = np.flatnonzero(np.sign(traj.qpos[:-1, 0]) != np.sign(traj.qpos[1:, 0]))
    if crossings.size < 3:
        return _row("pendulum_period", float("inf"), 1e-2, False, "not enough zero crossings to measure a period")
    times = traj.t[crossings]
    measured = float(np.mean(np.diff(times[::2]))) if len(times) >= 3 else float("inf")
    expected = 2 * np.pi * np.sqrt(length / GRAVITY)
    return _row("pendulum_period", _error(measured, expected), 1e-2,
                note=f"T = {measured:.4f} s against 2*pi*sqrt(L/g) = {expected:.4f} s")


def sliding_friction(dt: float = 1e-3, mu: float = 0.4, v0: float = 2.0) -> dict:
    """Stopping distance of a box sliding on a plane, v0^2 / (2 mu g)."""
    xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 {-GRAVITY}"/><worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" friction="{mu} 0.005 0.0001"/>
    <body name="box" pos="0 0 0.1"><freejoint/>
    <geom type="box" size="0.1 0.1 0.1" mass="1" friction="{mu} 0.005 0.0001"/></body>
    </worldbody></mujoco>"""
    cfg = MuJoCoConfig(model_xml=xml, n_steps=int(round(1.2 / dt)), dt=dt)
    traj = MuJoCoRunner(cfg).simulate(qd0=[v0, 0, 0, 0, 0, 0])
    speed = np.linalg.norm(traj.qvel[:, :2], axis=1)
    moving = np.flatnonzero(speed > 1e-3)
    stopped_at = int(moving[-1]) if moving.size else 0
    # the box keeps a small residual creep; take the position where it effectively stopped
    distance = float(traj.qpos[stopped_at, 0])
    expected = v0 ** 2 / (2 * mu * GRAVITY)
    return _row("sliding_friction", _error(distance, expected), 5e-2,
                note=f"stopped at {distance:.4f} m against {expected:.4f} m")


def gravity_torque(theta_deg: float = 30.0, length: float = 0.5, dt: float = 1e-3) -> dict:
    """Generalized torque a pendulum needs to be held at ``theta``: m g L sin(theta), read from ``qfrc_bias``."""
    mujoco = require_mujoco()
    xml = f"""<mujoco><option gravity="0 0 {-GRAVITY}"/><worldbody>
    <body name="bob" pos="0 0 0"><joint name="hinge" type="hinge" axis="0 1 0"/>
    <geom type="sphere" size="0.02" mass="1" pos="0 0 {-length}"/></body></worldbody></mujoco>"""
    model = load_model(MuJoCoConfig(model_xml=xml))
    data = make_data(model)
    theta = np.radians(theta_deg)
    data.qpos[0] = theta
    mujoco.mj_forward(model, data)
    measured = float(data.qfrc_bias[0])
    expected = GRAVITY * length * np.sin(theta)
    return _row("gravity_torque", _error(abs(measured), expected), 1e-6,
                note=f"|qfrc_bias| = {abs(measured):.6f} N m against m g L sin(theta) = {expected:.6f} N m")


def torque_free_spin(dt: float = 1e-3, duration: float = 1.0) -> dict:
    """A free body spinning with no torque: kinetic energy and angular momentum stay constant."""
    xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 0" integrator="implicitfast"/><worldbody>
    <body name="body"><freejoint/><geom type="box" size="0.3 0.15 0.05" mass="2"/></body>
    </worldbody></mujoco>"""
    cfg = MuJoCoConfig(model_xml=xml, n_steps=int(round(duration / dt)), dt=dt)
    traj = MuJoCoRunner(cfg).simulate(qd0=[0, 0, 0, 3.0, 1.0, 0.5])
    kinetic = traj.energy()["kinetic"]
    if kinetic.size == 0:
        return _row("torque_free_spin", float("inf"), 1e-2, False, "energy was not recorded")
    energy_drift = float(np.abs(kinetic - kinetic[0]).max() / max(abs(kinetic[0]), 1e-12))
    momentum = _angular_momentum(traj, body=1)
    momentum_drift = float(np.abs(np.linalg.norm(momentum, axis=1) - np.linalg.norm(momentum[0])).max()
                           / max(np.linalg.norm(momentum[0]), 1e-12))
    return _row("torque_free_spin", max(energy_drift, momentum_drift), 1e-2,
                note="relative drift of kinetic energy and |L|")


def _angular_momentum(traj, body: int = 1) -> np.ndarray:
    """World-frame angular momentum of a body, ``I_world(q) * omega_world``, per record."""
    out = np.empty((len(traj.t), 3))
    for k in range(len(traj.t)):
        w, x, y, z = traj.xquat[k, body]
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        inertia_world = rot @ traj.inertia[body] @ rot.T
        out[k] = inertia_world @ traj.cvel[k, body, :3]
    return out


def resting_contact(dt: float = 1e-3) -> dict:
    """A box dropped from 1 cm settles on the plane and then does not sink or bounce."""
    xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 {-GRAVITY}"/><worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.11"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
    </worldbody></mujoco>"""
    settle, check = 0.4, 0.2
    cfg = MuJoCoConfig(model_xml=xml, n_steps=int(round((settle + check) / dt)), dt=dt)
    traj = MuJoCoRunner(cfg).simulate()
    start = int(round(settle / dt))
    z = traj.xpos[start:, 1, 2]
    drift = float(np.abs(z - z.mean()).max())
    contacts = int(traj.ncon[start:].min()) if traj.ncon.size else 0
    passed = drift <= 1e-4 and contacts >= 4
    note = f"sinking {drift * 1e3:.4f} mm with {contacts} contacts (at least 4 expected)"
    return _row("resting_contact", drift, 1e-4, passed, note)


# ---------------------------------------------------------------------------- driver
def validate(cases: Sequence[str] | None = None, dt: float = 1e-3, log=None) -> list:
    """Run the analytic cases and return one row per case ``{"case", "value", "limit", "passed", "note"}``.

    ``cases`` defaults to every entry of :data:`CASES`; ``log`` is called with each row when given.
    """
    selected = list(CASES if cases is None else cases)
    unknown = [c for c in selected if c not in CASES]
    if unknown:
        raise ValueError(f"unknown cases {unknown}; available: {list(CASES)}")
    rows = []
    for name in selected:
        row = _CASES[name](dt=dt)
        rows.append(row)
        if log is not None:
            log(row)
    return rows


_CASES = {
    "free_fall": free_fall,
    "pendulum_period": pendulum_period,
    "sliding_friction": sliding_friction,
    "gravity_torque": gravity_torque,
    "torque_free_spin": torque_free_spin,
    "resting_contact": resting_contact,
}


def summary(rows: Sequence[dict]) -> dict:
    """Pass/fail roll-up of :func:`validate`: ``{"decision", "passed", "failed"}``."""
    failed = [r["case"] for r in rows if not r["passed"]]
    return {"decision": "APPROVED" if not failed else "REJECT", "passed": len(rows) - len(failed), "failed": failed}
