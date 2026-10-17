"""Run a :class:`RigidScene` with one of Newton's rigid-body solvers and record the trajectory."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .scene import RigidScene

SOLVERS = ("xpbd", "semi_implicit", "featherstone", "mujoco")


@dataclass
class RigidTrajectory:
    """Recorded states. Quaternions are ``(w, x, y, z)``; velocities are of the centre of mass, world frame."""

    names: list[str]
    t: np.ndarray            # (T,)
    pos: np.ndarray          # (T, N, 3) body origin [m]
    quat: np.ndarray         # (T, N, 4)
    vel: np.ndarray          # (T, N, 3) [m/s]
    omega: np.ndarray        # (T, N, 3) [rad/s]
    mass: np.ndarray         # (N,)
    inertia: np.ndarray      # (N, 3, 3) body frame, about the centre of mass
    com: np.ndarray          # (N, 3) centre of mass in the body frame
    gravity: float
    solver: str
    dt: float
    wall_time_s: float
    meta: dict = field(default_factory=dict)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def com_world(self) -> np.ndarray:
        from .scene import _rot
        out = np.empty_like(self.pos)
        for k in range(len(self.t)):
            for i in range(len(self.names)):
                out[k, i] = self.pos[k, i] + _rot(self.quat[k, i]) @ self.com[i]
        return out

    def energy(self) -> dict[str, np.ndarray]:
        """Kinetic, potential (m g z of the centre of mass) and total energy per recorded time [J]."""
        from .scene import _rot
        z = self.com_world()[..., 2]
        ke = 0.5 * (self.mass * (self.vel ** 2).sum(-1))
        rot = np.zeros_like(ke)
        for k in range(len(self.t)):
            for i in range(len(self.names)):
                R = _rot(self.quat[k, i])
                Iw = R @ self.inertia[i] @ R.T
                w = self.omega[k, i]
                rot[k, i] = 0.5 * w @ Iw @ w
        pe = self.mass * self.gravity * z
        return {"kinetic": (ke + rot).sum(-1), "potential": pe.sum(-1), "total": (ke + rot + pe).sum(-1)}

    def to_rigid_body_state(self, k: int = -1):
        """State at record ``k`` as the PyTorch :class:`RigidBodyState` used by the differentiable solver."""
        import torch

        from pinneapple_simulation.particle_dynamics.rigid_body import RigidBodyState

        s = RigidBodyState(n_bodies=len(self.names), dim=3)
        s.pos = torch.as_tensor(self.pos[k], dtype=torch.float32)
        s.quat = torch.as_tensor(self.quat[k], dtype=torch.float32)
        s.vel = torch.as_tensor(self.vel[k], dtype=torch.float32)
        s.omega = torch.as_tensor(self.omega[k], dtype=torch.float32)
        return s

    def to_upd(self):
        """The trajectory as a PINNeAPPle :class:`PhysicalSample` (time coordinate, one field per quantity)."""
        import torch

        from pinneapple_data.physical_sample import PhysicalSample

        T = len(self.t)
        state = {k: torch.as_tensor(getattr(self, k).reshape(T, -1), dtype=torch.float32)
                 for k in ("pos", "quat", "vel", "omega")}
        return PhysicalSample(
            state=state,
            domain={"type": "grid", "coords": {"time": self.t.astype(np.float32)}},
            provenance={"version": "0.1", "source": "newton", "solver": f"newton.solvers.{self.solver}",
                        "bodies": list(self.names), "dt": self.dt, **self.meta},
            schema={"units": {"pos": "m", "quat": "1", "vel": "m/s", "omega": "rad/s", "time": "s"}},
        )

    def save(self, path) -> None:
        np.savez_compressed(path, names=np.array(self.names), t=self.t, pos=self.pos, quat=self.quat, vel=self.vel,
                            omega=self.omega, mass=self.mass, inertia=self.inertia, com=self.com,
                            gravity=self.gravity, solver=self.solver, dt=self.dt)


def _make_solver(newton, model, solver: str, iterations: int):
    if solver == "xpbd":
        return newton.solvers.SolverXPBD(model, iterations=iterations, enable_restitution=True)
    if solver == "semi_implicit":
        return newton.solvers.SolverSemiImplicit(model, angular_damping=0.0)
    if solver == "featherstone":
        return newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    if solver == "mujoco":
        return newton.solvers.SolverMuJoCo(model)       # needs: pip install "newton[sim]"
    raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")


def simulate(scene: RigidScene, duration: float, dt: float = 1e-3, solver: str = "xpbd", iterations: int = 8,
             record_every: int = 10, device: str = "cpu", log=None, wrenches=None) -> RigidTrajectory:
    """Integrate ``scene`` for ``duration`` seconds with step ``dt`` and record every ``record_every`` steps.

    ``solver``: ``"xpbd"`` (position-based, robust contacts, default), ``"semi_implicit"`` (penalty contacts,
    needs a small ``dt``), ``"featherstone"`` (reduced coordinates, best for joint chains) or ``"mujoco"``
    (MuJoCo-Warp, needs the ``newton[sim]`` extra). The first call on a device compiles Warp kernels
    (seconds to a minute on CPU); later calls reuse the cache.

    ``wrenches``: constant external loads ``{body index: (force [N], torque [N m])}`` in the world frame, the torque
    taken about the body's centre of mass, applied at every step (e.g. a steady push).
    """
    import newton

    model, ids = scene.build(device=device)
    sol = _make_solver(newton, model, solver, iterations)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    # initial velocities, written on the maximal-coordinate state and, for joint-space solvers, on the free joints
    qd = s0.body_qd.numpy()
    moving = False
    for i, b in enumerate(scene.bodies):
        v, w = np.asarray(b.vel, float), np.asarray(b.omega, float)
        if v.any() or w.any():
            moving = True
            qd[ids[i]] = np.concatenate([v, w])
    if moving:
        if solver in ("featherstone", "mujoco"):
            raise NotImplementedError("initial body velocities are set for the maximal-coordinate solvers "
                                      "(xpbd, semi_implicit); for joint-space solvers set the joint velocities")
        s0.body_qd.assign(qd)

    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    n_steps = int(round(duration / dt))
    rec_t, rec_q, rec_qd = [], [], []

    def record(k):
        rec_t.append(k * dt)
        rec_q.append(s0.body_q.numpy()[ids].copy())
        rec_qd.append(s0.body_qd.numpy()[ids].copy())

    ext = None
    if wrenches:
        ext = np.zeros((model.body_count, 6), np.float32)
        for i, (f, tau) in wrenches.items():
            ext[ids[i]] = np.concatenate([np.asarray(f, float), np.asarray(tau, float)])

    t0 = time.time()
    record(0)
    for k in range(1, n_steps + 1):
        s0.clear_forces()
        if ext is not None:
            s0.body_f.assign(ext)
        pipe.collide(s0, contacts)
        sol.step(s0, s1, ctrl, contacts, dt)
        s0, s1 = s1, s0
        if k % record_every == 0 or k == n_steps:
            record(k)
            if log and k % (record_every * 50) == 0:
                log(f"t = {k * dt:.3f} s")
    wall = time.time() - t0

    q = np.asarray(rec_q, np.float64)
    qd = np.asarray(rec_qd, np.float64)
    quat = q[..., [6, 3, 4, 5]]                                  # (x, y, z, w) -> (w, x, y, z)
    inertia = model.body_inertia.numpy()[ids].reshape(-1, 3, 3)
    return RigidTrajectory(
        names=[b.name for b in scene.bodies], t=np.asarray(rec_t), pos=q[..., :3], quat=quat,
        vel=qd[..., :3], omega=qd[..., 3:], mass=model.body_mass.numpy()[ids].astype(float),
        inertia=inertia.astype(float), com=model.body_com.numpy()[ids].astype(float),
        gravity=scene.gravity, solver=solver, dt=dt, wall_time_s=wall,
        meta={"newton_version": getattr(newton, "__version__", "?"), "device": device, "iterations": iterations},
    )
