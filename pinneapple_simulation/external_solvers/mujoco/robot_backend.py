"""MuJoCo as a PINNeAPPle robot simulator.

:class:`MuJoCoRobotSim` gives MuJoCo 4.0 the same interface as
:class:`~pinneapple_simulation.robot_sim.sim.RobotSim` (NVIDIA Newton) and
:class:`~pinneapple_simulation.robot_sim.sim.IsaacSimBackend`: build a robot from an :class:`ArticulationSpec`, an
MJCF file, a URDF file or an XML string, run it with a torque controller, and get back the same
:class:`~pinneapple_simulation.robot_sim.sim.RobotRun` — joint positions, velocities, delivered torques, body poses,
body velocities and mass properties — so the PINNeAPPle checks (energy balance, torque replay, actuator limits,
``imu``) run unchanged, and the same study can be solved by two independent engines.

    from pinneapple_simulation.robot_sim.arms import TwoLinkArm
    from pinneapple_simulation.external_solvers.mujoco import MuJoCoRobotSim

    arm = TwoLinkArm(payload=5.0)
    sim = MuJoCoRobotSim.from_spec(arm.spec(), ground=False)
    run = sim.run(0.8, 5e-4, controller=lambda t, q, qd: kp * (q_ref(t) - q) - kd * qd)
    run.energy(), run.actuator_work(), run.imu("link2")

The controller is ``controller(t, q, qd) -> tau`` (joint torques, clipped to each joint's effort limit; the clipping
is counted in ``run.saturated``), exactly as in Newton's ``RobotSim.run``.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from .loader import model_summary, require_mujoco

__all__ = ["MuJoCoRobotSim", "spec_to_mjcf"]

_AXES = {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}


def _f(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def spec_to_mjcf(spec, ground: bool | None = None) -> str:
    """MJCF text for an :class:`~pinneapple_simulation.robot_sim.spec.ArticulationSpec`.

    Boxes and cylinders become MuJoCo geoms (MuJoCo cylinders are along z, so the spec's y-axis cylinders are turned),
    each joint becomes a hinge at the child link frame, each drive an effort ``motor`` with the joint's force limit as
    its control range, and a ground plane is added when ``spec.ground`` is set (or ``ground`` overrides it).
    """
    want_ground = spec.ground if ground is None else bool(ground)
    links = {L.name: L for L in spec.links}
    if len(links) != len(spec.links):
        raise ValueError("duplicate link names in the ArticulationSpec")
    by_child = {J.child: J for J in spec.joints}
    for J in spec.joints:
        if J.parent not in links or J.child not in links:
            raise ValueError(f"joint {J.name}: unknown link {J.parent!r} or {J.child!r}")
        if np.linalg.norm(np.asarray(J.anchor, float) - np.asarray(links[J.child].position, float)) > 1e-9:
            raise ValueError(f"joint {J.name}: the child link frame must sit on the joint anchor")
    roots = [L for L in spec.links if L.name not in by_child]
    if len(roots) > 1:
        raise ValueError(f"the spec has {len(roots)} roots ({[r.name for r in roots]}); a robot has one")
    if not roots:
        raise ValueError("the spec has no root link (every link is a child of a joint)")

    children: dict = {}
    for J in spec.joints:
        children.setdefault(J.parent, []).append(J)

    lines = [f'<mujoco model="{escape(spec.name)}">',
             # the spec stores angles in radians; MJCF's default angle unit is degrees
             '  <compiler angle="radian" autolimits="true"/>',
             f'  <option gravity="0 0 {-float(spec.gravity):.9g}" timestep="0.001"/>',
             '  <worldbody>']
    if want_ground:
        lines.append('    <geom name="ground" type="plane" size="20 20 0.1" rgba="0.55 0.55 0.58 1"/>')

    def emit(link, offset, depth, root: bool, joint=None) -> None:
        pad = "  " * depth
        rgba = f'{float(link.color[0]):.3g} {float(link.color[1]):.3g} {float(link.color[2]):.3g} 1'
        lines.append(f'{pad}<body name="{escape(link.name)}" pos="{_f(offset)}">')
        if joint is not None:
            # a <joint> element moves the body it sits in, so it goes inside the child body it drives
            axis = _AXES[joint.axis.upper()]
            lo, hi = (np.radians(joint.limits_deg) if joint.limits_deg is not None else (-np.pi, np.pi))
            lines.append(f'{pad}  <joint name="{escape(joint.name)}" type="hinge" pos="0 0 0" axis="{axis}" '
                         f'range="{lo:.9g} {hi:.9g}" limited="{"true" if joint.limits_deg is not None else "false"}"/>')
        inertia = link.principal_inertia()
        lines.append(f'{pad}  <inertial pos="{_f(link.com)}" mass="{float(link.mass):.9g}" '
                     f'diaginertia="{_f(inertia)}"/>')
        if link.shape == "cylinder":
            geom = (f'<geom name="{escape(link.name)}_geom" type="cylinder" '
                    f'size="{float(link.size[0]) / 2:.9g} {float(link.size[1]) / 2:.9g}" '
                    f'pos="{_f(link.box_offset)}" euler="1.5707963267948966 0 0" rgba="{rgba}" '
                    f'contype="{1 if link.collide else 0}" conaffinity="{1 if link.collide else 0}"/>')
        else:
            geom = (f'<geom name="{escape(link.name)}_geom" type="box" '
                    f'size="{_f(np.asarray(link.size, float) / 2)}" pos="{_f(link.box_offset)}" rgba="{rgba}" '
                    f'contype="{1 if link.collide else 0}" conaffinity="{1 if link.collide else 0}"/>')
        lines.append(f"{pad}  {geom}")
        if root and not spec.fixed_base:
            lines.append(f'{pad}  <freejoint name="{escape(link.name)}_root"/>')
        for J in children.get(link.name, []):
            child = links[J.child]
            # the child body frame sits on the joint anchor, so its offset is measured from the parent's frame
            offset = np.asarray(J.anchor, float) - np.asarray(link.position, float)
            emit(child, offset, depth + 1, root=False, joint=J)
        lines.append(f"{pad}</body>")

    emit(roots[0], np.asarray(roots[0].position, float), 3, root=True)
    lines.append("  </worldbody>")

    if spec.joints:
        lines.append("  <actuator>")
        for J in spec.joints:
            force = float(J.max_force)
            lines.append(f'    <motor name="{escape(J.name)}_motor" joint="{escape(J.name)}" '
                         f'ctrllimited="true" ctrlrange="{-force:.9g} {force:.9g}" gear="1"/>')
        lines.append("  </actuator>")
    lines.append("</mujoco>")
    return "\n".join(lines) + "\n"


class MuJoCoRobotSim:
    """A MuJoCo model of a robot with the ``RobotSim.run`` interface.

    Parameters
    ----------
    xml : MJCF text, a path to an MJCF/URDF file, or ``None`` when the classmethod already compiled the model.
    gravity : gravity magnitude [m/s^2]; ignored for a model loaded from a file (its ``<option>`` wins).
    ground : add a ground plane (only used when building from a spec/xml without one).
    name : label kept in the run metadata.
    """

    def __init__(self, xml: str | None = None, *, gravity: float = 9.81, ground: bool = True,
                 name: str = "robot") -> None:
        self.name = name
        self.gravity = float(gravity)
        self._xml = xml
        self._model = None
        self._data = None
        self._joint_ids: list = []
        self._actuator_of_joint: dict = {}

    # ------------------------------------------------------------------ construction
    @staticmethod
    def available() -> bool:
        """Whether MuJoCo is installed (and therefore this backend can run)."""
        import importlib.util

        return importlib.util.find_spec("mujoco") is not None

    @classmethod
    def from_xml(cls, xml: str, **kw) -> MuJoCoRobotSim:
        """A robot from MJCF text."""
        return cls(xml, **kw)

    @classmethod
    def from_mjcf(cls, path: str, **kw) -> MuJoCoRobotSim:
        """A robot from an MJCF (or URDF) file: the path is kept and loaded by MuJoCo, so meshes resolve."""
        with open(path) as f:
            text = f.read()
        sim = cls(text, **kw)
        sim._path = path
        return sim

    @classmethod
    def from_urdf(cls, path_or_text: str, **kw) -> MuJoCoRobotSim:
        """A robot from a URDF file or URDF text (MuJoCo compiles URDF directly)."""
        if path_or_text.lstrip().startswith("<"):
            return cls(path_or_text, **kw)
        with open(path_or_text) as f:
            text = f.read()
        sim = cls(text, **kw)
        sim._path = path_or_text
        return sim

    @classmethod
    def from_spec(cls, spec, **kw) -> MuJoCoRobotSim:
        """A robot from an :class:`ArticulationSpec` (the same object Newton's ``RobotSim.from_spec`` takes)."""
        kw.setdefault("gravity", spec.gravity)
        kw.setdefault("name", spec.name)
        ground = kw.pop("ground", None)
        return cls(spec_to_mjcf(spec, ground=ground), ground=spec.ground if ground is None else ground, **kw)

    # ------------------------------------------------------------------ model
    @property
    def model(self):
        if self._model is None:
            mujoco = require_mujoco()
            path = getattr(self, "_path", None)
            if path is not None:
                self._model = mujoco.MjModel.from_xml_path(path)
            elif self._xml is not None:
                self._model = mujoco.MjModel.from_xml_string(self._xml)
            else:  # pragma: no cover - construction guards against this
                raise ValueError("no model source: pass XML text, a path, or build from a spec")
            self._index()
        return self._model

    @property
    def data(self):
        if self._data is None:
            mujoco = require_mujoco()
            self._data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self._data)
        return self._data

    def _index(self) -> None:
        mujoco = require_mujoco()
        m = self._model
        self._joint_ids = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        self._joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint{i}"
                             for i in self._joint_ids]
        self._body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
                            for i in range(m.nbody)]
        self._qpos_adr = np.array([m.jnt_qposadr[i] for i in self._joint_ids], dtype=int)
        self._dof_adr = np.array([m.jnt_dofadr[i] for i in self._joint_ids], dtype=int)
        self._actuator_of_joint = {}
        limits = []
        for a in range(m.nu):
            joint = int(m.actuator_trnid[a, 0])
            if joint >= 0:
                self._actuator_of_joint[joint] = a
        for i in self._joint_ids:
            actuator = self._actuator_of_joint.get(i)
            if actuator is not None and bool(m.actuator_ctrllimited[actuator]):
                limits.append(float(m.actuator_ctrlrange[actuator, 1]))
            else:
                limits.append(float("inf"))
        self._effort_limit = np.asarray(limits, dtype=float)

    @property
    def joint_names(self) -> list[str]:
        """Names of the actuated hinge joints, in ``q`` order."""
        _ = self.model
        return list(self._joint_names)

    @property
    def body_names(self) -> list[str]:
        """Names of every body, in MuJoCo's index order (body 0 is the world)."""
        _ = self.model
        return list(self._body_names)

    @property
    def effort_limit(self) -> np.ndarray:
        """Control range of each joint's motor (``inf`` when the actuator is not limited)."""
        _ = self.model
        return self._effort_limit.copy()

    @property
    def ndof(self) -> int:
        """Number of actuated hinge joints (the length of ``q`` in a run)."""
        return len(self.joint_ids)

    @property
    def joint_ids(self) -> list:
        _ = self.model
        return self._joint_ids

    def mass_properties(self) -> dict:
        """Body masses, centres of mass and full inertia tensors (about the com, body frame)."""
        mujoco = require_mujoco()
        m = self.model
        inertia = np.zeros((m.nbody, 3, 3))
        for i in range(m.nbody):
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, np.asarray(m.body_iquat[i], dtype=float))
            rot = rot.reshape(3, 3)
            inertia[i] = rot @ np.diag(np.asarray(m.body_inertia[i], dtype=float)) @ rot.T
        return {"mass": np.asarray(m.body_mass, dtype=float).copy(),
                "com": np.asarray(m.body_ipos, dtype=float).copy(), "inertia": inertia}

    # ------------------------------------------------------------------ simulation
    def run(self, t_end: float, dt: float, *, q0=None, qd0=None,
            controller: Callable | None = None, record_every: int = 1,
            torque_at_midpoint: bool = True, contacts: bool | None = None) -> Any:
        """Integrate to ``t_end`` with step ``dt`` and return a ``RobotRun``.

        ``controller(t, q, qd) -> tau`` is evaluated at the step's midpoint when ``torque_at_midpoint``; the
        returned torques are what the motors delivered (clipped to each joint's control range, counted in
        ``run.saturated``). ``contacts=False`` disables contact detection (``mjDSBL_CONTACT``).
        """
        from pinneapple_simulation.robot_sim.sim import RobotRun

        mujoco = require_mujoco()
        m, d = self.model, self.data
        m.opt.timestep = float(dt)
        if contacts is False:
            m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mujoco.mj_resetData(m, d)

        q = np.zeros(len(self.joint_ids)) if q0 is None else np.asarray(q0, dtype=float).reshape(-1)
        if q.size != len(self.joint_ids):
            raise ValueError(f"q0 has {q.size} values; the model has {len(self.joint_ids)} hinge joints")
        qd = np.zeros(len(self.joint_ids)) if qd0 is None else np.asarray(qd0, dtype=float).reshape(-1)
        if qd.size != len(self.joint_ids):
            raise ValueError(f"qd0 has {qd.size} values; the model has {len(self.joint_ids)} hinge joints")
        d.qpos[self._qpos_adr] = q
        d.qvel[self._dof_adr] = qd
        mujoco.mj_forward(m, d)

        n = int(round(t_end / dt))
        record_every = max(1, int(record_every))
        limit = self.effort_limit
        rec = {"t": [], "q": [], "qd": [], "tau": [], "pos": [], "quat": [], "vel": [], "omega": []}
        saturated = 0
        t0 = time.perf_counter()
        for k in range(n + 1):
            tk = k * dt
            qk = np.array(d.qpos[self._qpos_adr], dtype=float)
            qdk = np.array(d.qvel[self._dof_adr], dtype=float)
            tau = np.zeros(len(self.joint_ids))
            if controller is not None:
                tau = np.asarray(controller(tk + (dt / 2 if torque_at_midpoint else 0.0), qk, qdk), dtype=float)
                if tau.size != len(self.joint_ids):
                    raise ValueError(f"controller returned {tau.size} torques; expected {len(self.joint_ids)}")
                clipped = np.clip(tau, -limit, limit)
                saturated += int(np.any(clipped != tau))
                tau = clipped
            if k % record_every == 0 or k == n:
                rec["t"].append(tk)
                rec["q"].append(qk)
                rec["qd"].append(qdk)
                rec["tau"].append(tau)
                rec["pos"].append(np.asarray(d.xpos, dtype=float).copy())
                rec["quat"].append(np.asarray(d.xquat, dtype=float).copy())
                rec["vel"].append(np.asarray(d.cvel[:, 3:], dtype=float).copy())
                rec["omega"].append(np.asarray(d.cvel[:, :3], dtype=float).copy())
            if k == n:
                break
            if m.nu:
                d.ctrl[:] = 0.0
                for j, joint in enumerate(self._joint_ids):
                    actuator = self._actuator_of_joint.get(joint)
                    if actuator is not None:
                        d.ctrl[actuator] = tau[j]
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)

        wall = time.perf_counter() - t0
        props = self.mass_properties()
        quat = np.asarray(rec["quat"], dtype=float)
        # MuJoCo quaternions are (w, x, y, z); RobotRun stores Newton's (x, y, z, w)
        quat_xyzw = quat[..., [1, 2, 3, 0]]
        # the model's own gravity (a file may override the spec's), taken as a magnitude along -z
        gravity = abs(float(m.opt.gravity[2]))
        return RobotRun(
            t=np.asarray(rec["t"], dtype=float),
            q=np.asarray(rec["q"], dtype=float),
            qd=np.asarray(rec["qd"], dtype=float),
            tau=np.asarray(rec["tau"], dtype=float),
            body_pos=np.asarray(rec["pos"], dtype=float),
            body_quat=quat_xyzw,
            body_vel=np.asarray(rec["vel"], dtype=float),
            body_omega=np.asarray(rec["omega"], dtype=float),
            body_names=list(self.body_names),
            joint_names=list(self.joint_names),
            masses=props["mass"], coms=props["com"], inertias=props["inertia"],
            gravity=gravity, dt=dt * record_every, saturated=saturated, wall_s=wall,
            backend="mujoco", info=model_summary(m),
        )
