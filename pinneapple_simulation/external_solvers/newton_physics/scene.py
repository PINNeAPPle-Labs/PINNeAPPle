"""Scene description for the Newton rigid-body backend.

A :class:`RigidScene` is a plain Python description (bodies, shapes, joints, ground) that does not import
Newton or Warp until :meth:`RigidScene.build` is called, so scenes can be created, saved and inspected on a
machine without the solver installed.

Conventions: SI units, the z axis points up, gravity is ``-g`` along z, and quaternions are given as
``(w, x, y, z)`` like :class:`pinneapple_simulation.particle_dynamics.rigid_body.RigidBodyState` (Newton
stores ``(x, y, z, w)``; the conversion happens here and in the runner).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

_IDENTITY: Quat = (1.0, 0.0, 0.0, 0.0)


@dataclass
class Shape:
    """One collision shape of a body, in the body frame (centred on the body origin)."""

    kind: str                      # "sphere" | "box" | "capsule" | "cylinder" | "mesh" | "compound"
    size: tuple[float, ...]        # sphere (r,), box (hx, hy, hz), capsule/cylinder (r, half_height)
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    boxes: list | None = None      # compound: [((hx, hy, hz), (x, y, z) in the body frame, mass), ...]


@dataclass
class Body:
    name: str
    shape: Shape
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = _IDENTITY         # (w, x, y, z)
    vel: Vec3 = (0.0, 0.0, 0.0)    # linear velocity of the centre of mass, world frame [m/s]
    omega: Vec3 = (0.0, 0.0, 0.0)  # angular velocity, world frame [rad/s]
    density: float = 1000.0        # [kg/m^3]; ignored when ``mass`` is given
    mass: float | None = None   # [kg]; rescales the density so the body has this mass
    mu: float = 0.5                # Coulomb friction coefficient
    restitution: float = 0.0


@dataclass
class Joint:
    kind: str                      # "revolute" | "fixed"
    parent: int                    # body index, or -1 for the world
    child: int
    anchor: Vec3                   # joint position in world coordinates at t = 0
    axis: Vec3 = (0.0, 1.0, 0.0)   # rotation axis in world coordinates at t = 0
    damping: float = 0.0


@dataclass
class RigidScene:
    """Bodies, joints and an optional ground plane, built into a Newton model on demand.

    >>> scene = RigidScene()
    >>> scene.ground(mu=0.4)
    >>> box = scene.box("crate", (0.2, 0.2, 0.2), pos=(0, 0, 0.2), vel=(2.0, 0, 0))
    """

    gravity: float = 9.81
    bodies: list[Body] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    ground_mu: float | None = None
    ground_restitution: float = 0.0

    # ------------------------------------------------------------------ building the description
    def ground(self, mu: float = 0.5, restitution: float = 0.0) -> RigidScene:
        self.ground_mu, self.ground_restitution = float(mu), float(restitution)
        return self

    def _add(self, name: str, shape: Shape, **kw) -> int:
        if any(b.name == name for b in self.bodies):
            raise ValueError(f"body name {name!r} is already used")
        q = np.asarray(kw.pop("quat", _IDENTITY), float)
        q = tuple(q / np.linalg.norm(q))
        self.bodies.append(Body(name=name, shape=shape, quat=q, **kw))
        return len(self.bodies) - 1

    def sphere(self, name: str, radius: float, **kw) -> int:
        return self._add(name, Shape("sphere", (float(radius),)), **kw)

    def box(self, name: str, half_extents: Sequence[float], **kw) -> int:
        hx, hy, hz = (float(v) for v in half_extents)
        return self._add(name, Shape("box", (hx, hy, hz)), **kw)

    def capsule(self, name: str, radius: float, half_height: float, **kw) -> int:
        return self._add(name, Shape("capsule", (float(radius), float(half_height))), **kw)

    def cylinder(self, name: str, radius: float, half_height: float, **kw) -> int:
        return self._add(name, Shape("cylinder", (float(radius), float(half_height))), **kw)

    def mesh(self, name: str, vertices, faces, **kw) -> int:
        """A closed triangle mesh (e.g. from an STL of the part), vertices in the body frame [m]."""
        v = np.asarray(vertices, np.float32).reshape(-1, 3)
        f = np.asarray(faces, np.int32).reshape(-1, 3)
        return self._add(name, Shape("mesh", (), vertices=v, faces=f), **kw)

    def compound(self, name: str, boxes, **kw) -> int:
        """One rigid body made of several boxes ``[(half_extents, position in the body frame, mass), ...]``; its mass,
        centre of mass and inertia follow from the boxes (``mass`` / ``density`` keywords are not used)."""
        bx = [(tuple(float(v) for v in h), tuple(float(v) for v in p), float(m)) for h, p, m in boxes]
        return self._add(name, Shape("compound", (), boxes=bx), **kw)

    def revolute(self, parent: int, child: int, anchor: Sequence[float], axis: Sequence[float] = (0, 1, 0),
                 damping: float = 0.0) -> int:
        """Hinge between ``parent`` (``-1`` = world) and ``child`` through ``anchor`` about ``axis`` (world, t = 0)."""
        a = np.asarray(axis, float)
        self.joints.append(Joint("revolute", int(parent), int(child), tuple(map(float, anchor)),
                                 tuple(a / np.linalg.norm(a)), float(damping)))
        return len(self.joints) - 1

    def fixed(self, parent: int, child: int, anchor: Sequence[float] | None = None) -> int:
        """Weld ``child`` to ``parent`` (``-1`` = world) in their current relative pose."""
        anc = tuple(map(float, anchor)) if anchor is not None else self.bodies[child].pos
        self.joints.append(Joint("fixed", int(parent), int(child), anc))
        return len(self.joints) - 1

    # ------------------------------------------------------------------ checks
    def check(self) -> list[str]:
        """Problems that would make the model invalid or the result meaningless (empty list = fine)."""
        out = []
        children = [j.child for j in self.joints]
        if len(children) != len(set(children)):
            out.append("a body is the child of more than one joint (closed loops are not supported here)")
        for j in self.joints:
            if not (-1 <= j.parent < len(self.bodies)) or not (0 <= j.child < len(self.bodies)):
                out.append(f"joint {j.kind} refers to a body that does not exist")
            if j.parent == j.child:
                out.append(f"joint {j.kind} connects body {j.child} to itself")
        for b in self.bodies:
            s = b.shape.size
            if b.shape.kind == "compound":
                if not b.shape.boxes or any(min(h) <= 0 or m <= 0 for h, _, m in b.shape.boxes):
                    out.append(f"{b.name}: a compound needs boxes with positive sizes and masses")
                continue
            if b.shape.kind != "mesh" and (len(s) == 0 or min(s) <= 0):
                out.append(f"{b.name}: shape sizes must be positive, got {s}")
            if b.shape.kind == "mesh" and (b.shape.vertices is None or len(b.shape.faces) < 4):
                out.append(f"{b.name}: a mesh needs vertices and at least 4 faces")
            if b.mass is not None and b.mass <= 0:
                out.append(f"{b.name}: mass must be positive")
            if b.mu < 0 or not (0 <= b.restitution <= 1):
                out.append(f"{b.name}: friction must be >= 0 and restitution in [0, 1]")
        return out

    # ------------------------------------------------------------------ Newton model
    def build(self, device: str = "cpu"):
        """Build the Newton model. Returns ``(model, body_ids)`` with ``body_ids[i]`` the Newton index of body i."""
        problems = self.check()
        if problems:
            raise ValueError("invalid scene: " + "; ".join(problems))
        import newton
        import warp as wp

        wp.config.quiet = True
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -float(self.gravity)))
        if self.ground_mu is not None:
            cfg = newton.ModelBuilder.ShapeConfig(mu=self.ground_mu, restitution=self.ground_restitution)
            builder.add_ground_plane(cfg=cfg)

        jointed = {j.child for j in self.joints} | {j.parent for j in self.joints if j.parent >= 0}
        ids = []
        for i, b in enumerate(self.bodies):
            xf = _transform(wp, b.pos, b.quat)
            make = builder.add_link if i in jointed else builder.add_body
            bid = make(xform=xf, label=b.name)
            cfg = newton.ModelBuilder.ShapeConfig(density=b.density, mu=b.mu, restitution=b.restitution)
            s = b.shape
            if s.kind == "sphere":
                builder.add_shape_sphere(bid, radius=s.size[0], cfg=cfg)
            elif s.kind == "box":
                builder.add_shape_box(bid, hx=s.size[0], hy=s.size[1], hz=s.size[2], cfg=cfg)
            elif s.kind == "capsule":
                builder.add_shape_capsule(bid, radius=s.size[0], half_height=s.size[1], cfg=cfg)
            elif s.kind == "cylinder":
                builder.add_shape_cylinder(bid, radius=s.size[0], half_height=s.size[1], cfg=cfg)
            elif s.kind == "mesh":
                builder.add_shape_mesh(bid, mesh=newton.Mesh(s.vertices, s.faces.reshape(-1)), cfg=cfg)
            elif s.kind == "compound":
                for (hx, hy, hz), p, m in s.boxes:
                    c = newton.ModelBuilder.ShapeConfig(density=m / (8 * hx * hy * hz), mu=b.mu, restitution=b.restitution)
                    builder.add_shape_box(bid, xform=wp.transform(p=wp.vec3(*p), q=wp.quat_identity()),
                                          hx=hx, hy=hy, hz=hz, cfg=c)
            else:
                raise ValueError(f"unknown shape {s.kind!r}")
            ids.append(bid)

        # one articulation per tree: a jointed body without a parent joint gets a free joint to the world. Newton
        # needs the joints of an articulation to be contiguous, so the joints are created tree by tree.
        by_child = {j.child: k for k, j in enumerate(self.joints)}
        roots = [i for i in sorted(jointed) if i not in by_child]
        groups: dict[int, list[int]] = {i: [] for i in roots}
        for k in _topological(self.joints):
            groups.setdefault(_root(self.joints, k), []).append(k)
        for root, ks in groups.items():
            js = []
            if root >= 0:
                js.append(builder.add_joint_free(ids[root], label=f"{self.bodies[root].name}_free"))
            for k in ks:
                j = self.joints[k]
                pq = self.bodies[j.parent] if j.parent >= 0 else None
                pxf = _joint_frame(wp, pq, j.anchor)
                cxf = _joint_frame(wp, self.bodies[j.child], j.anchor)
                parent = ids[j.parent] if j.parent >= 0 else -1
                if j.kind == "revolute":
                    jid = builder.add_joint_revolute(parent=parent, child=ids[j.child], parent_xform=pxf,
                                                     child_xform=cxf, axis=wp.vec3(*j.axis), damping=j.damping,
                                                     label=f"{self.bodies[j.child].name}_hinge")
                else:
                    jid = builder.add_joint_fixed(parent=parent, child=ids[j.child], parent_xform=pxf,
                                                  child_xform=cxf, label=f"{self.bodies[j.child].name}_weld")
                js.append(jid)
            builder.add_articulation(js, label=f"tree_{root}")

        for i, b in enumerate(self.bodies):            # optional exact mass: rescale density
            if b.mass is not None and b.shape.kind != "compound":
                m0 = builder.body_mass[ids[i]]
                k = b.mass / m0
                builder.body_mass[ids[i]] = b.mass
                builder.body_inv_mass[ids[i]] = 1.0 / b.mass
                inertia = np.asarray(builder.body_inertia[ids[i]], float).reshape(3, 3) * k
                builder.body_inertia[ids[i]] = wp.mat33(*inertia.reshape(-1))
                builder.body_inv_inertia[ids[i]] = wp.mat33(*np.linalg.inv(inertia).reshape(-1))
        return builder.finalize(device=device), ids


def _transform(wp, pos, quat_wxyz):
    w, x, y, z = quat_wxyz
    return wp.transform(p=wp.vec3(*pos), q=wp.quat(x, y, z, w))


def _rot(quat_wxyz) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _joint_frame(wp, body: Body | None, anchor):
    """Joint frame in the body (or world) frame, aligned with the world axes at t = 0."""
    if body is None:
        return wp.transform(p=wp.vec3(*anchor), q=wp.quat_identity())
    R = _rot(body.quat)
    p = R.T @ (np.asarray(anchor, float) - np.asarray(body.pos, float))
    w, x, y, z = body.quat
    return wp.transform(p=wp.vec3(*p), q=wp.quat(-x, -y, -z, w))     # conj(q): world -> body


def _topological(joints: list[Joint]) -> list[int]:
    done, order, children = set(), [], {j.child for j in joints}
    pending = list(range(len(joints)))
    while pending:
        progressed = False
        for k in list(pending):
            p = joints[k].parent
            if p < 0 or p not in children or p in done:
                order.append(k)
                done.add(joints[k].child)
                pending.remove(k)
                progressed = True
        if not progressed:
            raise ValueError("joint graph has a cycle")
    return order


def _root(joints: list[Joint], k: int) -> int:
    by_child = {j.child: j for j in joints}
    j = joints[k]
    while j.parent >= 0 and j.parent in by_child:
        j = by_child[j.parent]
    return j.parent if j.parent >= 0 else -(1 + j.child)
