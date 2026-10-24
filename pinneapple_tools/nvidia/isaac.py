"""Isaac Sim-ready robots in OpenUSD (``UsdPhysics``), their checks, and the Isaac Sim script that replays a motion.

Isaac Sim simulates articulations described with the ``UsdPhysics`` schemas (rigid bodies with mass properties,
colliders, revolute joints with drives, an articulation root, a physics scene). This module writes those from a
plain description, so the masses and inertias Isaac Sim uses are the ones PINNeAPPle computed with:

``Link`` / ``Joint`` / ``ArticulationSpec`` (from :mod:`pinneapple_simulation.robot_sim.spec`)
    links as boxes or cylinders (visual + collider) with mass, centre of mass and principal inertia; revolute
    joints with an anchor point, axis, limits and an angular drive (position or effort).
``author_articulation(stage, spec)``
    adds the robot under ``/World/<name>`` of a :class:`pinneapple_tools.nvidia.omniverse.FieldStage` (or any
    ``Usd.Stage``) with a ``PhysicsScene``, a ground plane, a fixed joint to the world for a fixed base, and the
    articulation root on the robot prim.
``validate_physics(path)``
    :func:`omniverse.validate_stage` plus the physics a simulator will read: ``UsdPhysics.LoadUsdPhysicsFromRange``
    (the USD physics parser) finds every body, joint and articulation; masses positive; inertias physical
    (triangle inequality); every joint between rigid bodies; a gravity direction and magnitude.
``isaac_replay_script(...)``
    a standalone Isaac Sim script (``isaacsim.SimulationApp``, ``isaacsim.core.api.World``,
    ``isaacsim.core.prims.SingleArticulation``) that opens the stage, applies the joint torques PINNeAPPle computed
    (or tracks the positions with the drives), and saves the joint positions and measured efforts to ``.npz``.
    Needs Isaac Sim (RTX GPU); written against the 5.x API, which 6.x still ships (deprecated).
``compare_replay(npz, t, q, tau)``
    tracking and effort errors of an Isaac Sim replay against PINNeAPPle's trajectory.
"""
from __future__ import annotations

import os

import numpy as np

from pinneapple_simulation.robot_sim.spec import (  # noqa: F401
    ArticulationSpec,
    Joint,
    Link,
    box_inertia,
    cylinder_inertia,
)

from .omniverse import _name, require_pxr, validate_stage

__all__ = ["Link", "Joint", "ArticulationSpec", "box_inertia", "cylinder_inertia", "author_articulation", "validate_physics",
           "isaac_replay_script", "compare_replay"]


def author_articulation(stage, spec: ArticulationSpec) -> str:
    """Write ``spec`` under ``/World/<name>``; returns that prim path. ``stage``: a ``FieldStage`` or ``Usd.Stage``."""
    require_pxr()
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt
    st = getattr(stage, "stage", stage)
    UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
    scene = UsdPhysics.Scene.Define(st, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(float(spec.gravity))
    if spec.ground:
        g = UsdGeom.Cube.Define(st, "/World/Ground")
        g.CreateSizeAttr(1.0)
        g.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
        g.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.1))
        g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.35, 0.35, 0.37)]))
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    root = f"/World/{_name(spec.name)}"
    robot = UsdGeom.Xform.Define(st, root)
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    for k, v in spec.info.items():
        robot.GetPrim().CreateAttribute(f"pinneapple:info:{_name(k)}", Sdf.ValueTypeNames.String).Set(str(v))
    pos = {}
    for L in spec.links:
        p = f"{root}/{_name(L.name)}"
        x = UsdGeom.Xform.Define(st, p)
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, L.position)))
        pos[L.name] = (p, np.asarray(L.position, float))
        prim = x.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr(float(L.mass))
        m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, L.com)))
        inr = L.principal_inertia()
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inr)))
        m.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
        if L.shape == "cylinder":
            r, h = float(L.size[0]) / 2, float(L.size[1])
            c = UsdGeom.Cylinder.Define(st, f"{p}/geom")
            c.CreateRadiusAttr(r)
            c.CreateHeightAttr(h)
            c.CreateAxisAttr("Y")
            c.AddTranslateOp().Set(Gf.Vec3d(*map(float, L.box_offset)))
            c.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-r, -h / 2, -r), Gf.Vec3f(r, h / 2, r)]))
        elif L.shape == "box":
            c = UsdGeom.Cube.Define(st, f"{p}/geom")
            c.CreateSizeAttr(1.0)
            c.AddTranslateOp().Set(Gf.Vec3d(*map(float, L.box_offset)))
            c.AddScaleOp().Set(Gf.Vec3f(*map(float, L.size)))
            c.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)]))
        else:
            raise ValueError(f"link {L.name}: shape is 'box' or 'cylinder', got {L.shape!r}")
        c.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, L.color))]))
        if L.collide:
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if spec.fixed_base:
        base = spec.links[0]
        fj = UsdPhysics.FixedJoint.Define(st, f"{root}/root_joint")
        fj.CreateBody1Rel().SetTargets([pos[base.name][0]])
        fj.CreateLocalPos0Attr(Gf.Vec3f(*map(float, base.position)))
        fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    for J in spec.joints:
        (p0, x0), (p1, x1) = pos[J.parent], pos[J.child]
        a = np.asarray(J.anchor, float)
        j = UsdPhysics.RevoluteJoint.Define(st, f"{root}/{_name(J.name)}")
        j.CreateBody0Rel().SetTargets([p0])
        j.CreateBody1Rel().SetTargets([p1])
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, a - x0)))
        j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, a - x1)))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateAxisAttr(J.axis.upper())
        if J.limits_deg is not None:
            j.CreateLowerLimitAttr(float(J.limits_deg[0]))
            j.CreateUpperLimitAttr(float(J.limits_deg[1]))
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        d.CreateTypeAttr("force")
        d.CreateMaxForceAttr(float(J.max_force))
        d.CreateStiffnessAttr(float(J.stiffness) if J.drive == "position" else 0.0)
        d.CreateDampingAttr(float(J.damping) if J.drive in ("position", "velocity") else 0.0)
        if J.drive == "position":
            d.CreateTargetPositionAttr(0.0)
        elif J.drive == "velocity":
            d.CreateTargetVelocityAttr(0.0)
        j.GetPrim().CreateAttribute("pinneapple:drive_mode", Sdf.ValueTypeNames.String).Set(J.drive)
    return root


def validate_physics(path: str, *, expect_bodies: int | None = None, expect_joints: int | None = None) -> dict:
    """:func:`validate_stage` plus the ``UsdPhysics`` parse and the mass / joint checks a simulator relies on."""
    require_pxr()
    from pxr import Usd, UsdPhysics
    rep = validate_stage(path)
    problems = list(rep["problems"])
    stage = Usd.Stage.Open(os.fspath(path))
    parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, ["/"])
    counts: dict[str, int] = {}
    for kind, (paths, _descs) in parsed.items():
        counts[str(kind).rsplit(".", 1)[-1]] = len(paths)
    bodies = counts.get("RigidBody", 0)
    joints = counts.get("RevoluteJoint", 0) + counts.get("FixedJoint", 0)
    if counts.get("Scene", 0) != 1:
        problems.append(f"{counts.get('Scene', 0)} physics scenes (need 1)")
    if counts.get("Articulation", 0) < 1:
        problems.append("no articulation found by the UsdPhysics parser")
    if expect_bodies is not None and bodies != expect_bodies:
        problems.append(f"{bodies} rigid bodies, expected {expect_bodies}")
    if expect_joints is not None and counts.get("RevoluteJoint", 0) != expect_joints:
        problems.append(f"{counts.get('RevoluteJoint', 0)} revolute joints, expected {expect_joints}")
    total_mass = 0.0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(prim)
            mass = m.GetMassAttr().Get() if m else None
            inr = np.asarray(m.GetDiagonalInertiaAttr().Get() or (0, 0, 0), float) if m else np.zeros(3)
            if not mass or mass <= 0:
                problems.append(f"{prim.GetPath()}: mass {mass}")
            else:
                total_mass += float(mass)
            if np.any(inr <= 0) or inr[0] + inr[1] < inr[2] - 1e-12 or inr[1] + inr[2] < inr[0] - 1e-12 or inr[0] + inr[2] < inr[1] - 1e-12:
                problems.append(f"{prim.GetPath()}: inertia {inr.tolist()} is not physical")
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            for rel in (j.GetBody0Rel(), j.GetBody1Rel()):
                for tgt in rel.GetTargets():
                    b = stage.GetPrimAtPath(tgt)
                    if not b or not b.HasAPI(UsdPhysics.RigidBodyAPI):
                        problems.append(f"{prim.GetPath()}: target {tgt} is not a rigid body")
    scene = UsdPhysics.Scene(stage.GetPrimAtPath("/World/PhysicsScene"))
    g = scene.GetGravityMagnitudeAttr().Get() if scene else None
    if g is None or not g > 0:
        problems.append("gravity magnitude not set")
    return {**rep, "ok": not problems, "problems": problems, "physics": counts, "rigid_bodies": bodies,
            "joints": joints, "total_mass": total_mass, "gravity": g}


def isaac_replay_script(usd_path: str, robot_prim: str, traj_npz: str, out_npz: str, *, mode: str = "effort",
                        physics_dt: float = 1.0 / 600.0, headless: bool = True) -> str:
    """Standalone Isaac Sim script (run with Isaac Sim's ``python.sh``). ``traj_npz`` holds ``t``, ``q`` [rad],
    ``tau`` [N m] (joint order of the USD articulation); ``mode="effort"`` applies ``tau`` open loop, ``"position"``
    tracks ``q`` with the drives. Writes ``t``, ``q`` and ``effort`` (``get_measured_joint_efforts``) to
    ``out_npz``."""
    if mode not in ("effort", "position"):
        raise ValueError("mode is 'effort' or 'position'")
    return f'''# Isaac Sim standalone replay, written by pinneapple_tools.nvidia.isaac. Needs Isaac Sim 5.x / 6.x and an RTX GPU:
#   ~/isaacsim/python.sh {os.path.basename(out_npz).replace(".npz", "")}_replay.py
from isaacsim import SimulationApp

app = SimulationApp({{"headless": {bool(headless)}}})

import numpy as np                                                        # noqa: E402
import omni.usd                                                           # noqa: E402
from isaacsim.core.api import World                                       # noqa: E402
from isaacsim.core.prims import SingleArticulation                        # noqa: E402
from isaacsim.core.utils.types import ArticulationAction                  # noqa: E402

omni.usd.get_context().open_stage({os.path.abspath(usd_path)!r})
for _ in range(10):
    app.update()
world = World(stage_units_in_meters=1.0, physics_dt={physics_dt!r}, rendering_dt={physics_dt * 10!r})
robot = SingleArticulation(prim_path={robot_prim!r}, name="robot")
world.scene.add(robot)
world.reset()
data = np.load({os.path.abspath(traj_npz)!r})
t_ref, q_ref, tau_ref = data["t"], data["q"], data["tau"]
robot.set_joint_positions(q_ref[0])
robot.set_joint_velocities(np.zeros_like(q_ref[0]))
T, Q, E = [], [], []
n = int(round(t_ref[-1] / {physics_dt!r}))
for k in range(n + 1):
    tk = k * {physics_dt!r}
    if {mode!r} == "effort":
        tau = np.array([np.interp(tk, t_ref, tau_ref[:, j]) for j in range(tau_ref.shape[1])])
        robot.set_joint_efforts(tau)
    else:
        qk = np.array([np.interp(tk, t_ref, q_ref[:, j]) for j in range(q_ref.shape[1])])
        robot.apply_action(ArticulationAction(joint_positions=qk))
    world.step(render=False)
    T.append(tk)
    Q.append(np.asarray(robot.get_joint_positions()))
    E.append(np.asarray(robot.get_measured_joint_efforts()))
np.savez({os.path.abspath(out_npz)!r}, t=np.asarray(T), q=np.asarray(Q), effort=np.asarray(E),
         dof_names=np.asarray(robot.dof_names))
app.close()
'''


def compare_replay(npz: str, t, q, tau) -> dict:
    """Max tracking error [rad] and RMS effort error relative to the peak torque, Isaac Sim replay vs PINNeAPPle."""
    d = np.load(npz)
    qi = np.stack([np.interp(d["t"], t, q[:, j]) for j in range(q.shape[1])], 1)
    ti = np.stack([np.interp(d["t"], t, tau[:, j]) for j in range(tau.shape[1])], 1)
    return {"max_tracking_rad": float(np.abs(d["q"] - qi).max()),
            "effort_rel_rms": float(np.sqrt(np.mean((d["effort"] - ti) ** 2)) / max(np.abs(ti).max(), 1e-12))}
