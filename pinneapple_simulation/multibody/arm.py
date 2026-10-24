"""Fixed-base serial arms on :class:`~pinneapple_simulation.multibody.tree.MultibodyTree`: inverse and forward
dynamics, a closed-form planar two-link reference, quintic trajectories, and the energy balance of a motion.

``fixed_base_inverse_dynamics``  joint torques ``tau = M_jj qdd + h_j - Q_g,j`` with the base held still.
``fixed_base_forward``           integrate the joint equations (RK4) under given torques (base held still).
``planar_two_link_torques``      the closed-form torques of a planar two-link arm (Spong, Hutchinson & Vidyasagar,
                                 *Robot Modeling and Control*, 2006, sec. 7.3), an independent check of the tree.
``quintic``                      rest-to-rest quintic polynomial between two joint configurations.
``energy_balance``               work done by the joint torques against the change of kinetic + potential energy.
"""
from __future__ import annotations

import numpy as np

from .tree import MultibodyTree

__all__ = ["fixed_base_inverse_dynamics", "fixed_base_forward", "planar_two_link_torques", "quintic",
           "energy_balance", "link_world_points", "lumped_link", "link_root_checks"]


def _state(tree: MultibodyTree, q):
    qf, v = tree.initial_state(angles=np.asarray(q, float))
    return qf, v


def fixed_base_inverse_dynamics(tree: MultibodyTree, q, qd, qdd) -> np.ndarray:
    """``(n, nj)`` joint torques along a trajectory ``q, qd, qdd`` (each ``(n, nj)``), base fixed at the origin."""
    q, qd, qdd = (np.atleast_2d(np.asarray(a, float)) for a in (q, qd, qdd))
    out = np.zeros_like(q)
    for k in range(len(q)):
        qf, v = _state(tree, q[k])
        v[6:] = qd[k]
        kin = tree.kinematics(qf, v)
        M = tree.mass_matrix(kin)
        h = tree.bias(kin, v)
        g = tree.gravity_force(kin)
        out[k] = M[6:, 6:] @ qdd[k] + h[6:] - g[6:]
    return out


def fixed_base_forward(tree: MultibodyTree, q0, qd0, tau_fn, t) -> tuple[np.ndarray, np.ndarray]:
    """Joint angles and rates at the times ``t`` (RK4 between them) under torques ``tau_fn(t, q, qd)``."""
    def f(tk, y):
        n = tree.nj
        q, qd = y[:n], y[n:]
        qf, v = _state(tree, q)
        v[6:] = qd
        kin = tree.kinematics(qf, v)
        M = tree.mass_matrix(kin)[6:, 6:]
        rhs = tau_fn(tk, q, qd) + tree.gravity_force(kin)[6:] - tree.bias(kin, v)[6:]
        return np.r_[qd, np.linalg.solve(M, rhs)]

    y = np.r_[np.asarray(q0, float), np.asarray(qd0, float)]
    Y = [y]
    for a, b in zip(t[:-1], t[1:], strict=True):
        h = b - a
        k1 = f(a, y)
        k2 = f(a + h / 2, y + h / 2 * k1)
        k3 = f(a + h / 2, y + h / 2 * k2)
        k4 = f(b, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Y.append(y)
    Y = np.asarray(Y)
    return Y[:, :tree.nj], Y[:, tree.nj:]


def planar_two_link_torques(m1, m2, l1, lc1, lc2, I1, I2, g, q, qd, qdd) -> np.ndarray:
    """Closed-form torques of a planar two-link arm in a vertical plane; angles from the horizontal (``q2``
    relative), ``I1``, ``I2`` about the centres of mass and normal to the plane. Shapes ``(n, 2)``."""
    q, qd, qdd = (np.atleast_2d(np.asarray(a, float)) for a in (q, qd, qdd))
    c2, s2 = np.cos(q[:, 1]), np.sin(q[:, 1])
    M11 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * c2) + I1 + I2
    M12 = m2 * (lc2 ** 2 + l1 * lc2 * c2) + I2
    M22 = m2 * lc2 ** 2 + I2
    h = -m2 * l1 * lc2 * s2
    g1 = (m1 * lc1 + m2 * l1) * g * np.cos(q[:, 0]) + m2 * lc2 * g * np.cos(q[:, 0] + q[:, 1])
    g2 = m2 * lc2 * g * np.cos(q[:, 0] + q[:, 1])
    t1 = M11 * qdd[:, 0] + M12 * qdd[:, 1] + h * qd[:, 1] ** 2 + 2 * h * qd[:, 0] * qd[:, 1] + g1
    t2 = M12 * qdd[:, 0] + M22 * qdd[:, 1] - h * qd[:, 0] ** 2 + g2
    return np.stack([t1, t2], 1)


def quintic(q0, q1, T: float, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rest-to-rest quintic ``q(t)`` from ``q0`` to ``q1`` in ``T`` seconds: ``(q, qd, qdd)``, each ``(n, nj)``."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    s = np.clip(np.asarray(t, float) / T, 0, 1)[:, None]
    p = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    dp = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
    ddp = (60 * s - 180 * s ** 2 + 120 * s ** 3) / T ** 2
    d = q1 - q0
    return q0 + p * d, dp * d, ddp * d


def energy_balance(tree: MultibodyTree, t, q, qd, tau) -> dict:
    """``W = integral of tau . qd dt`` (trapezoid) against ``Delta(T + U)``; ``rel`` is the mismatch relative to the
    largest of ``|W|`` and the peak energy swing."""
    E = []
    for k in range(len(t)):
        qf, v = _state(tree, q[k])
        v[6:] = qd[k]
        T, U = tree.energy(tree.kinematics(qf, v), v)
        E.append(T + U)
    E = np.asarray(E)
    P = np.sum(np.asarray(tau) * np.asarray(qd), axis=1)
    W = np.r_[0.0, np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(t))]
    scale = max(np.abs(W).max(), np.abs(E - E[0]).max(), 1e-12)
    return {"work": W, "energy": E - E[0], "rel": float(np.abs(W - (E - E[0])).max() / scale)}


def link_world_points(tree: MultibodyTree, q, points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """World positions of body-frame points ``{body name: (m, 3)}`` for each configuration in ``q (n, nj)``."""
    out = {k: [] for k in points}
    for qk in np.atleast_2d(q):
        qf, _ = _state(tree, qk)
        kin = tree.kinematics(qf)
        for name, P in points.items():
            i = tree.index(name)
            out[name].append(np.asarray(P, float) @ kin.R[i].T + kin.p[i])
    return {k: np.asarray(v) for k, v in out.items()}


def lumped_link(name: str, length: float, mass_per_m: float, point_masses=(), width: float = 0.05):
    """A :class:`~pinneapple_simulation.multibody.tree.Body` for a straight link along its x axis: a uniform beam
    (``mass_per_m`` over ``length``, square section ``width``) plus point masses ``[(x, m), ...]`` (motors,
    payload). Mass, centre of mass and inertia about it (parallel-axis theorem)."""
    from .tree import Body
    mb = mass_per_m * length
    xs = [length / 2] + [float(x) for x, _ in point_masses]
    ms = [mb] + [float(m) for _, m in point_masses]
    m = float(sum(ms))
    xc = float(np.dot(xs, ms) / m)
    i_axis = mb * width ** 2 / 6                                # thin-walled square tube about its own axis
    i_bend = mb * length ** 2 / 12 + mb * (length / 2 - xc) ** 2 + sum(mi * (xi - xc) ** 2
                                                                      for xi, mi in zip(xs[1:], ms[1:], strict=True))
    inr = np.diag([i_axis, i_bend + i_axis / 2, i_bend + i_axis / 2])
    return Body(name, m, [xc, 0.0, 0.0], inr)


def link_root_checks(tau, section, *, yield_strength: float, fatigue_strength: float | None = None,
                     safety: float = 3.0, names=None) -> tuple[np.ndarray, list[dict]]:
    """Bending stress at each link root from the joint torques ``tau (T, nj)`` (the torque a joint delivers is the
    bending moment it passes into the child link), checked against ``yield / safety`` and, with
    ``fatigue_strength`` (fully reversed, at the design life), the stress amplitude (half the range over the
    cycle) against ``fatigue_strength / safety``. Returns ``(stress (T, nj) [Pa], checks)``."""
    tau = np.atleast_2d(np.asarray(tau, float))
    sig = np.abs(tau) * section.c_m / section.I_m4
    signed = tau * section.c_m / section.I_m4
    names = names or [f"joint {j}" for j in range(tau.shape[1])]
    checks = []
    for j, nm in enumerate(names):
        u = sig[:, j].max() * safety / yield_strength
        checks.append({"name": f"{nm} link root: peak stress x SF {safety:g} <= yield", "value": float(u),
                       "limit": 1.0, "passed": bool(u <= 1.0), "detail": f"{sig[:, j].max() / 1e6:.1f} MPa"})
        if fatigue_strength is not None:
            amp = 0.5 * (signed[:, j].max() - signed[:, j].min())
            v = amp * safety / fatigue_strength
            checks.append({"name": f"{nm} link root: stress amplitude x SF {safety:g} <= fatigue strength",
                           "value": float(v), "limit": 1.0, "passed": bool(v <= 1.0),
                           "detail": f"amplitude {amp / 1e6:.1f} MPa per cycle"})
    return sig, checks
