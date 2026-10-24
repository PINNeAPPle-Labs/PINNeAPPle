"""PINNeAPPle Robot Sim: robots simulated with NVIDIA Newton (CPU or GPU through Warp), recorded with the
quantities PINNeAPPle checks (joint torques, energy, body poses, IMU), and an Isaac Sim backend with the same
interface for machines that have it.

    sim = RobotSim.from_spec(spec)                     # or from_urdf(path) / from_usd(path) / from_mjcf(path)
    run = sim.run(t_end=0.8, dt=5e-4, q0=q0, controller=pd_tracking(ref, kp, kd, feedforward=tau_fn))
    run.q, run.qd, run.tau, run.energy(), run.imu("link2")

Controllers are plain functions ``controller(t, q, qd) -> tau`` (joint torques before the actuator limits); the
recorded ``tau`` is what the actuators delivered (clipped to each joint's effort limit, and the clipping is counted
in ``run.saturated``). Helpers: :func:`torque_trajectory`, :func:`pd_tracking`. The solver is Newton's
``SolverFeatherstone`` (reduced coordinates; semi-implicit, first order in ``dt``: halve ``dt`` and the error of a
smooth motion halves, which :func:`convergence_order` measures).
"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spec import ArticulationSpec, spec_to_urdf

__all__ = ["RobotSim", "RobotRun", "pd_tracking", "torque_trajectory", "convergence_order", "IsaacSimBackend",
           "newton_available"]


def newton_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None


def _newton():
    try:
        import newton
        import warp as wp
    except ImportError as e:  # pragma: no cover
        raise ImportError("PINNeAPPle Robot Sim runs on NVIDIA Newton: pip install newton (CPU works)") from e
    wp.config.quiet = True
    return newton, wp


def torque_trajectory(fn: Callable[[float], np.ndarray]):
    """Open-loop joint torques ``fn(t)``."""
    return lambda t, q, qd: np.asarray(fn(t), float)


def pd_tracking(reference: Callable[[float], tuple[np.ndarray, np.ndarray]], kp, kd,
                feedforward: Callable[[float], np.ndarray] | None = None):
    """``tau = ff(t) + kp (q_ref - q) + kd (qd_ref - qd)``; ``reference(t) -> (q_ref, qd_ref)``."""
    kp, kd = np.asarray(kp, float), np.asarray(kd, float)

    def ctrl(t, q, qd):
        qr, qdr = reference(t)
        tau = kp * (qr - q) + kd * (qdr - qd)
        return tau + (np.asarray(feedforward(t), float) if feedforward is not None else 0.0)
    return ctrl


@dataclass
class RobotRun:
    t: np.ndarray
    q: np.ndarray                 # (T, ndof) joint positions [rad]
    qd: np.ndarray                # (T, ndof)
    tau: np.ndarray               # (T, ndof) delivered joint torques [N m] (applied over the step after t)
    body_pos: np.ndarray          # (T, nbody, 3) body origins
    body_quat: np.ndarray         # (T, nbody, 4) (x, y, z, w) as Newton stores them
    body_vel: np.ndarray          # (T, nbody, 3) linear velocity of the centre of mass, world
    body_omega: np.ndarray        # (T, nbody, 3) angular velocity, world
    body_names: list[str]
    joint_names: list[str]
    masses: np.ndarray
    coms: np.ndarray              # (nbody, 3) centre of mass in the body frame
    inertias: np.ndarray          # (nbody, 3, 3) about the com, body frame
    gravity: float
    dt: float
    saturated: int = 0
    wall_s: float = 0.0
    backend: str = "newton"
    info: dict[str, Any] = field(default_factory=dict)

    def _rot(self, k):
        x, y, z, w = (self.body_quat[k, :, i] for i in range(4))
        return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                         np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                         np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)

    def com_world(self) -> np.ndarray:
        return np.stack([self.body_pos[k] + np.einsum("nij,nj->ni", self._rot(k), self.coms)
                         for k in range(len(self.t))])

    def energy(self) -> dict[str, np.ndarray]:
        """Kinetic, potential and total mechanical energy of every body over time [J]."""
        T, U = np.zeros(len(self.t)), np.zeros(len(self.t))
        for k in range(len(self.t)):
            R = self._rot(k)
            c = self.body_pos[k] + np.einsum("nij,nj->ni", R, self.coms)
            vc = self.body_vel[k]
            Iw = np.einsum("nij,njk,nlk->nil", R, self.inertias, R)
            T[k] = 0.5 * np.sum(self.masses * np.sum(vc * vc, 1)) + \
                0.5 * np.sum(np.einsum("ni,nij,nj->n", self.body_omega[k], Iw, self.body_omega[k]))
            U[k] = np.sum(self.masses * self.gravity * c[:, 2])
        return {"kinetic": T, "potential": U, "total": T + U}

    def actuator_work(self) -> np.ndarray:
        """Cumulative work of the joint torques [J] (torque held over each step, rate at the step's midpoint)."""
        qdm = 0.5 * (self.qd[1:] + self.qd[:-1])
        return np.r_[0.0, np.cumsum(np.sum(self.tau[:-1] * qdm, 1) * np.diff(self.t))]

    def imu(self, body: str) -> dict[str, np.ndarray]:
        """Simulated IMU at a body's centre of mass: specific force (acceleration minus gravity) and angular rate,
        both in the body frame, by central differences of the recorded velocities."""
        i = self.body_names.index(body) if body in self.body_names else \
            [n.split("/")[-1] for n in self.body_names].index(body)
        a = np.gradient(self.body_vel[:, i], self.t, axis=0) - np.array([0.0, 0.0, -self.gravity])
        R = np.stack([self._rot(k)[i] for k in range(len(self.t))])
        return {"specific_force": np.einsum("tji,tj->ti", R, a), "angular_rate": np.einsum("tji,tj->ti", R,
                                                                                            self.body_omega[:, i])}


class RobotSim:
    """A Newton model of a robot (fixed or floating base) on an optional ground plane."""

    def __init__(self, builder_fn: Callable[[Any], None], *, gravity: float = 9.81, ground: bool = True,
                 device: str = "cpu", name: str = "robot"):
        newton, wp = _newton()
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -float(gravity)))
        builder_fn(b)
        if ground:
            b.add_ground_plane()
        self.model = b.finalize(device=device)
        self.gravity, self.device, self.name, self.ground = float(gravity), device, name, bool(ground)
        m = self.model
        self.body_names = list(getattr(m, "body_label", [f"body{i}" for i in range(m.body_count)]))
        labels = list(getattr(m, "joint_label", []))
        jt = m.joint_type.numpy()
        self.joint_names = [labels[i].split("/")[-1] for i in range(len(jt)) if jt[i] == int(newton.JointType.REVOLUTE)] \
            if labels else [f"q{i}" for i in range(m.joint_dof_count)]
        self.effort_limit = m.joint_effort_limit.numpy().astype(float)
        self.free_base = m.joint_dof_count != len(self.joint_names)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_urdf(cls, source: str, *, fixed_base: bool = True, **kw) -> RobotSim:
        """``source``: a URDF file or URDF text."""
        if not os.path.exists(source):
            d = tempfile.mkdtemp(prefix="robot_urdf_")
            path = os.path.join(d, "robot.urdf")
            with open(path, "w") as f:
                f.write(source)
            source = path
        return cls(lambda b: b.add_urdf(source, floating=not fixed_base, enable_self_collisions=False), **kw)

    @classmethod
    def from_mjcf(cls, path: str, **kw) -> RobotSim:
        return cls(lambda b: b.add_mjcf(path), **kw)

    @classmethod
    def from_usd(cls, path: str, **kw) -> RobotSim:
        """A UsdPhysics stage (e.g. one written by ``pinneapple_tools.nvidia.isaac.author_articulation``): the same
        file Omniverse and Isaac Sim open. The stage's own ground is ignored when ``ground=True`` adds one."""
        ground = kw.get("ground", True)
        return cls(lambda b: b.add_usd(path, ignore_paths=["/World/Ground"] if ground else None,
                                       enable_self_collisions=False), **kw)

    @classmethod
    def from_spec(cls, spec: ArticulationSpec, **kw) -> RobotSim:
        kw.setdefault("gravity", spec.gravity)
        kw.setdefault("name", spec.name)
        return cls.from_urdf(spec_to_urdf(spec), fixed_base=spec.fixed_base, **kw)

    @property
    def ndof(self) -> int:
        return int(self.model.joint_dof_count)

    def mass_properties(self) -> dict[str, np.ndarray]:
        m = self.model
        return {"mass": m.body_mass.numpy().astype(float), "com": m.body_com.numpy().astype(float),
                "inertia": m.body_inertia.numpy().astype(float)}

    # ------------------------------------------------------------------ simulation
    def run(self, t_end: float, dt: float, *, q0=None, qd0=None,
            controller: Callable[[float], np.ndarray] | None = None, record_every: int = 1,
            torque_at_midpoint: bool = True, contacts: bool | None = None) -> RobotRun:
        """Integrate to ``t_end`` with step ``dt``; ``controller(t, q, qd) -> tau`` is evaluated at the step's
        midpoint time (with the state at its start) when ``torque_at_midpoint``. ``contacts``: run collision
        detection every step (default: when there is a ground plane or a floating base)."""
        import time

        newton, wp = _newton()
        m = self.model
        s0, s1, ctrl = m.state(), m.state(), m.control()
        solver = newton.solvers.SolverFeatherstone(m)
        nq = m.joint_coord_count
        q = m.joint_q.numpy().astype(float).copy() if q0 is None else np.asarray(q0, float)
        qd = np.zeros(self.ndof) if qd0 is None else np.asarray(qd0, float)
        if len(q) != nq:
            raise ValueError(f"q0 has {len(q)} values; the model has {nq} coordinates")
        s0.joint_q.assign(q.astype(np.float32))
        s0.joint_qd.assign(qd.astype(np.float32))
        newton.eval_fk(m, s0.joint_q, s0.joint_qd, s0)
        n = int(round(t_end / dt))
        rec: dict[str, list] = {k: [] for k in ("t", "q", "qd", "tau", "bq", "bqd")}
        lim = self.effort_limit
        sat = 0
        t0 = time.perf_counter()
        collide = (self.ground or self.free_base) if contacts is None else bool(contacts)
        cts = None
        for k in range(n + 1):
            tk = k * dt
            qk, qdk = s0.joint_q.numpy().astype(float), s0.joint_qd.numpy().astype(float)
            tau = np.zeros(self.ndof)
            if controller is not None:
                tau = np.asarray(controller(tk + (dt / 2 if torque_at_midpoint else 0.0), qk, qdk), float)
                clipped = np.clip(tau, -lim, lim) if lim.size == tau.size else tau
                sat += int(np.any(clipped != tau))
                tau = clipped
            if k % record_every == 0:
                bq, bqd = s0.body_q.numpy().astype(float), s0.body_qd.numpy().astype(float)
                rec["t"].append(tk)
                rec["q"].append(qk)
                rec["qd"].append(qdk)
                rec["tau"].append(tau)
                rec["bq"].append(bq)
                rec["bqd"].append(bqd)
            if k == n:
                break
            ctrl.joint_f.assign(tau.astype(np.float32))
            s0.clear_forces()
            if collide:
                cts = m.collide(s0)
            solver.step(s0, s1, ctrl, cts, dt)
            s0, s1 = s1, s0
        wall = time.perf_counter() - t0
        bq, bqd = np.asarray(rec["bq"]), np.asarray(rec["bqd"])
        mp = self.mass_properties()
        # Newton body_qd: (v, w), v the velocity of the centre of mass (checked against finite differences)
        return RobotRun(np.asarray(rec["t"]), np.asarray(rec["q"]), np.asarray(rec["qd"]), np.asarray(rec["tau"]),
                        bq[..., :3], bq[..., 3:7], bqd[..., :3], bqd[..., 3:6], self.body_names, self.joint_names,
                        mp["mass"], mp["com"], mp["inertia"], self.gravity, dt * record_every, sat, wall)


def convergence_order(errors: Sequence[float], dts: Sequence[float]) -> float:
    """Observed order of accuracy: slope of log(error) against log(dt) (least squares)."""
    return float(np.polyfit(np.log(np.asarray(dts, float)), np.log(np.asarray(errors, float)), 1)[0])


class IsaacSimBackend:
    """The same ``run`` interface on NVIDIA Isaac Sim (needs Isaac Sim and an RTX GPU). Here it writes the standalone
    script (:func:`pinneapple_tools.nvidia.isaac.isaac_replay_script`) and raises, so a caller can fall back to
    :class:`RobotSim` and record that Isaac Sim did not run."""

    def __init__(self, usd_path: str, robot_prim: str):
        self.usd_path, self.robot_prim = usd_path, robot_prim

    @staticmethod
    def available() -> bool:
        import importlib.util
        return importlib.util.find_spec("isaacsim") is not None

    def write_script(self, traj_npz: str, out_npz: str, script_path: str, mode: str = "effort") -> str:
        from pinneapple_tools.nvidia.isaac import isaac_replay_script
        with open(script_path, "w") as f:
            f.write(isaac_replay_script(self.usd_path, self.robot_prim, traj_npz, out_npz, mode=mode))
        return script_path

    def run(self, *a, **kw):
        if not self.available():
            raise RuntimeError("Isaac Sim is not installed here (it needs an RTX GPU); run the script written by "
                               "write_script() on an Isaac Sim machine, or use RobotSim (Newton)")
        raise NotImplementedError("inside Isaac Sim, run the script written by write_script()")
