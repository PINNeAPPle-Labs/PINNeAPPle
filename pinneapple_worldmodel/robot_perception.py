"""Robot perception data from the robot sim: arm motions simulated by PINNeAPPle Robot Sim (NVIDIA Newton), written
as OpenUSD timelines with a camera, seen by a randomized synthetic camera, labelled with the simulator's exact joint
angles, and checked for what a perception model of a robot must respect.

``arm_motions(arm, n, seed)``
    random point-to-point moves of :class:`~pinneapple_simulation.robot_sim.TwoLinkArm` (quintic plan, PD + model
    feedforward) simulated in Newton; exact ``q``, ``qd`` and link poses per camera frame.
``CameraVariation`` / ``sample_camera(rng)``
    zoom, framing offset, gain, noise, blur, link albedo, background and sometimes an occluder in front of the arm.
``write_motion_stage(path, arm, motion, var)``
    the motion as a USD timeline (one animated Xform per link, the occluder, a side ``UsdGeom.Camera``) with the
    joint angles time-sampled as ``pinneapple:label:q``.
``camera_frames(path, motion, var)``
    the camera images, rasterized from the stage at each time code (``rasterize_topdown(view="side")``).
``visibility(arm, motion, var)``
    the label check: a frame is kept only if the elbow and the tool point are inside the image and not behind the
    occluder (an image that does not show the joints cannot carry their angles as a label).
``ImageRegressor``
    a small CNN; here it reads ``(sin, cos)`` of both joint angles from a frame.
``perception_checks(q_pred, motion, arm)``
    joint angle error, tool point error through the forward kinematics, the speed the predicted angle sequence
    implies against Newton's joint rates, and frame-to-frame jumps beyond what the joint rate limit allows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["arm_motions", "CameraVariation", "sample_camera", "write_motion_stage", "camera_frames", "visibility",
           "forward_points", "payload_track_check", "arm_camera_sample", "ImageRegressor", "perception_checks", "VISUAL_THICKNESS"]

VISUAL_THICKNESS = 0.09      # m: links drawn thicker than the 50 mm tube so they span a few pixels


def forward_points(arm, q) -> dict[str, np.ndarray]:
    """Elbow and tool point ``(T, 2)`` in the x-z plane for joint angles ``q (T, 2)`` (the simulator's convention:
    angles from the horizontal, positive turning x towards -z)."""
    q = np.atleast_2d(q)
    a1, a12 = q[:, 0], q[:, 0] + q[:, 1]
    elbow = np.c_[arm.l1 * np.cos(a1), arm.shoulder_z - arm.l1 * np.sin(a1)]
    tool = elbow + np.c_[arm.l2 * np.cos(a12), -arm.l2 * np.sin(a12)]
    return {"elbow": elbow, "tool": tool}


def arm_motions(arm, n: int, seed: int, *, duration: float = 0.8, dt: float = 1e-3, fps: float = 30.0,
                kp=(350.0, 110.0), kd=(5.0, 2.5), q_box=((-1.3, 1.3), (-1.6, 1.6)), moves=None) -> list[dict]:
    """``n`` random point-to-point moves simulated in Newton, sampled at ``fps`` (or the given ``moves``:
    ``[(q_start, q_end)]``)."""
    from pinneapple_simulation.multibody import arm as mb
    from pinneapple_simulation.robot_sim import RobotSim, pd_tracking
    rng = np.random.default_rng(seed)
    sim = RobotSim.from_spec(arm.spec(), ground=False)
    tree = arm.tree()
    every = max(int(round(1 / (fps * dt))), 1)
    out = []
    for k in range(n if moves is None else len(moves)):
        if moves is None:
            qa = tuple(rng.uniform(*b) for b in q_box)
            qb = tuple(rng.uniform(*b) for b in q_box)
        else:
            qa, qb = (tuple(map(float, x)) for x in moves[k])
        plan = lambda t, qa=qa, qb=qb: tuple(-x for x in mb.quintic(qa, qb, duration, np.atleast_1d(t)))  # noqa: E731
        feed = lambda t, qa=qa, qb=qb: -mb.fixed_base_inverse_dynamics(tree, *mb.quintic(qa, qb, duration, [t]))[0]  # noqa: E731
        ref = lambda t, plan=plan: tuple(x[0] for x in plan(t)[:2])  # noqa: E731
        r = sim.run(duration, dt, q0=plan(0.0)[0][0], controller=pd_tracking(ref, kp, kd, feedforward=feed),
                    record_every=every)
        out.append({"t": r.t, "q": r.q, "qd": r.qd, "body_pos": r.body_pos, "body_quat": r.body_quat,
                    "dt": float(r.t[1] - r.t[0]), "saturated": r.saturated})
    return out


@dataclass
class CameraVariation:
    zoom: float = 1.0
    offset: tuple = (0.0, 0.0)       # m, framing offset in x, z
    gain: float = 1.0
    noise: float = 0.0
    blur: float = 0.0
    albedo: tuple = (0.8, 0.8)       # link1, link2 grey levels
    background: float = 0.1
    occluder: tuple | None = None    # (x, z, width, height) in m, in front of the arm

    def bounds(self, arm) -> tuple[float, float, float, float]:
        half = 1.25 / self.zoom
        cx, cz = self.offset[0], arm.shoulder_z + self.offset[1]
        return (cx - half, cx + half, cz - half, cz + half)

    def to_dict(self) -> dict:
        return asdict(self)


def sample_camera(rng, *, occluder_p: float = 0.3) -> CameraVariation:
    occ = None
    if rng.random() < occluder_p:
        occ = (float(rng.uniform(-0.9, 0.9)), float(rng.uniform(-0.5, 1.2)), float(rng.uniform(0.15, 0.4)),
               float(rng.uniform(0.15, 0.4)))
    return CameraVariation(zoom=float(rng.uniform(0.9, 1.15)), offset=(float(rng.uniform(-0.1, 0.1)),
                                                                      float(rng.uniform(-0.1, 0.1))),
                           gain=float(rng.uniform(0.7, 1.2)), noise=float(rng.uniform(0.0, 0.04)),
                           blur=float(rng.uniform(0.0, 0.8)), albedo=(float(rng.uniform(0.55, 0.95)),
                                                                      float(rng.uniform(0.55, 0.95))),
                           background=float(rng.uniform(0.0, 0.25)), occluder=occ)


def _box(size, offset):
    a, b, c = (s / 2 for s in size)
    V = np.array([[x, y, z] for x in (-a, a) for y in (-b, b) for z in (-c, c)], float) + offset
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def write_motion_stage(path: str, arm, motion: dict, var: CameraVariation, *, payload=None) -> str:
    """``payload``: ``(V, F, grey)``, a mesh carried at the tool point (in the tool frame: x along the forearm)."""
    from pxr import Gf, Sdf, UsdGeom

    from pinneapple_tools.nvidia.omniverse import FieldStage
    fps = 1.0 / motion["dt"]
    st = FieldStage(path, "Arm motion from PINNeAPPle Robot Sim (Newton), synthetic camera", fps=fps)
    w = VISUAL_THICKNESS
    shapes = {"base": ((0.25, 0.05, arm.shoulder_z - 0.06), (0, 0.15, (arm.shoulder_z - 0.06) / 2), (0.3, 0.3, 0.3)),
              "link1": ((arm.l1, w, w), (arm.l1 / 2, 0, 0), (var.albedo[0],) * 3),
              "link2": ((arm.l2, w, w), (arm.l2 / 2, 0, 0), (var.albedo[1],) * 3)}
    for i, (name, (size, off, col)) in enumerate(shapes.items()):
        V, F = _box(size, off)
        st.add_rigid_animation(name, V, F, motion["t"], motion["body_pos"][:, i], motion["body_quat"][:, i],
                               color=col)
    if payload is not None:
        V, F, grey = payload
        st.add_rigid_animation("Payload", np.asarray(V, float) + [arm.l2, 0, 0], F, motion["t"],
                               motion["body_pos"][:, 2], motion["body_quat"][:, 2], color=(grey,) * 3)
    if var.occluder is not None:
        x, z, ww, hh = var.occluder
        V, F = _box((ww, 0.02, hh), (x, -0.3, z))
        st.add_mesh("Occluder", V, F, color=(0.45, 0.4, 0.35))
    cam = UsdGeom.Camera.Define(st.stage, "/World/SideCamera")
    lo, r, b, t = var.bounds(arm)
    cam.AddTranslateOp().Set(Gf.Vec3d((lo + r) / 2, -4.0, (b + t) / 2))
    cam.AddRotateXOp().Set(90.0)
    cam.CreateProjectionAttr("orthographic")
    cam.CreateHorizontalApertureAttr(float((r - lo) * 10))     # tenths of a scene unit (USD camera convention)
    cam.CreateVerticalApertureAttr(float((t - b) * 10))
    prim = st.stage.GetPrimAtPath("/World")
    qa = prim.CreateAttribute("pinneapple:label:q", Sdf.ValueTypeNames.Float2)
    for k, tk in enumerate(motion["t"]):
        qa.Set(Gf.Vec2f(*map(float, motion["q"][k])), float(tk * fps))
    for k, v in var.to_dict().items():
        prim.CreateAttribute(f"pinneapple:camera:{k}", Sdf.ValueTypeNames.String).Set(str(v))
    return st.save()


def camera_frames(path: str, arm, motion: dict, var: CameraVariation, H: int = 72, W: int = 72,
                  seed: int = 0) -> np.ndarray:
    """``(T, H, W)`` grey frames of the stage at each recorded time."""
    from scipy import ndimage

    from pinneapple_tools.nvidia.omniverse import rasterize_topdown
    rng = np.random.default_rng(seed)
    fps = 1.0 / motion["dt"]
    out = []
    for tk in motion["t"]:
        img = rasterize_topdown(path, H, W, bounds=var.bounds(arm), time=float(tk * fps), view="side",
                                background=(var.background,) * 3).mean(-1)
        if var.blur > 0:
            img = ndimage.gaussian_filter(img, var.blur)
        out.append(np.clip(var.gain * img + rng.normal(0, var.noise, img.shape), 0, 1))
    return np.asarray(out, np.float32)


def visibility(arm, motion: dict, var: CameraVariation, *, margin: float = 0.03) -> np.ndarray:
    """Per frame: are the elbow and the tool point inside the image and not behind the occluder?"""
    P = forward_points(arm, motion["q"])
    lo, r, b, t = var.bounds(arm)
    ok = np.ones(len(motion["q"]), bool)
    for p in P.values():
        ok &= (p[:, 0] > lo + margin) & (p[:, 0] < r - margin) & (p[:, 1] > b + margin) & (p[:, 1] < t - margin)
        if var.occluder is not None:
            x, z, ww, hh = var.occluder
            ok &= ~((np.abs(p[:, 0] - x) < ww / 2 + margin) & (np.abs(p[:, 1] - z) < hh / 2 + margin))
    return ok


def payload_track_check(frames, arm, motion: dict, var: CameraVariation, *, tol_px: float = 2.0,
                        offset=(0.0, 0.0)) -> dict:
    """Does each frame show the carried part where the kinematics put it? The centroid of the pixels brighter than
    the links (the part is drawn white, the links grey: threshold half-way between the two after the camera's gamma
    and gain) against the tool point, plus the part's ``offset`` (x, z in the tool frame), projected into the image."""
    H, W = frames.shape[1:]
    link = max(var.albedo) ** (1 / 2.2) * var.gain
    part = min(var.gain, 1.0)
    if part - link < 0.08:
        raise ValueError("the part is not brighter than the links in this camera (lower the albedo or the gain)")
    level = (link + part) / 2
    lo, r, b, t = var.bounds(arm)
    q = np.atleast_2d(motion["q"])
    a12 = q[:, 0] + q[:, 1]
    c, sn = np.cos(a12), np.sin(a12)
    tool = forward_points(arm, q)["tool"] + np.c_[c * offset[0] + sn * offset[1], -sn * offset[0] + c * offset[1]]
    px = np.c_[(tool[:, 0] - lo) / (r - lo) * (W - 1), (t - tool[:, 1]) / (t - b) * (H - 1)]
    err = []
    for f, p in zip(frames, px, strict=True):
        rr, cc = np.nonzero(f > level)
        err.append(np.hypot(cc.mean() - p[0], rr.mean() - p[1]) if len(rr) else np.inf)
    err = np.asarray(err)
    return {"error_px": err, "max_px": float(err.max()), "passed": bool(err.max() <= tol_px)}


def arm_camera_sample(payload: float = 5.0, seed: int = 0, occluder: bool = False,
                      usd_path: str = "arm_camera_sample.usda") -> dict:
    """One labelled perception sample: a random arm move in the robot sim (Newton), its USD timeline with a side
    camera (random variation, an occluder if asked) at ``usd_path``, the camera frames, and the label checks (the
    forward kinematics against the simulator's link poses, the frames that show the elbow and the tool point)."""
    from pinneapple_simulation.robot_sim import TwoLinkArm
    arm = TwoLinkArm(payload=float(payload))
    m = arm_motions(arm, 1, seed)[0]
    var = sample_camera(np.random.default_rng(seed), occluder_p=1.0 if occluder else 0.0)
    path = write_motion_stage(usd_path, arm, m, var)
    frames = camera_frames(path, arm, m, var, seed=seed)
    vis = visibility(arm, m, var)
    err = float(np.abs(forward_points(arm, m["q"])["elbow"] - m["body_pos"][:, 2][:, [0, 2]]).max())
    return {"decision": "APPROVED" if err <= 1e-6 else "REJECT", "label_error_m": err, "frames": int(len(frames)),
            "visible_fraction": float(vis.mean()), "camera": var.to_dict(), "usd": path}


class ImageRegressor:
    """A small CNN from a grey image ``(H, W)`` to ``n_out`` numbers."""

    def __init__(self, H: int, W: int, n_out: int, seed: int = 0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Conv2d(1, 16, 5, 2, 2), nn.SiLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
                                 nn.Conv2d(32, 48, 3, 2, 1), nn.SiLU(), nn.Flatten(),
                                 nn.Linear(48 * ((H + 7) // 8) * ((W + 7) // 8), 96), nn.SiLU(), nn.Linear(96, n_out))

    def fit(self, X, Y, *, steps: int = 3000, batch: int = 64, lr: float = 2e-3, seed: int = 0):
        import torch
        X = torch.tensor(np.asarray(X, np.float32)[:, None])
        Y = torch.tensor(np.asarray(Y, np.float32))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        rng = np.random.default_rng(seed)
        self.losses = []
        for k in range(steps):
            idx = torch.from_numpy(rng.integers(0, len(X), batch))
            loss = torch.mean((self.net(X[idx]) - Y[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for g_ in opt.param_groups:
                g_["lr"] = lr * 0.5 * (1 + np.cos(np.pi * (k + 1) / steps))
            self.losses.append(float(loss.detach()))
        self.net.eval()
        return self

    def predict(self, X) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self.net(torch.tensor(np.asarray(X, np.float32)[:, None])).numpy()


def angle_features(q) -> np.ndarray:
    q = np.atleast_2d(q)
    return np.c_[np.sin(q[:, 0]), np.cos(q[:, 0]), np.sin(q[:, 1]), np.cos(q[:, 1])]


def feature_angles(f) -> np.ndarray:
    f = np.atleast_2d(f)
    return np.c_[np.arctan2(f[:, 0], f[:, 1]), np.arctan2(f[:, 2], f[:, 3])]


def perception_checks(q_pred, motion: dict, arm, *, qd_max: float = 12.0) -> dict:
    """Errors of a predicted angle sequence for one motion, all frames visible."""
    q, qd, dt = motion["q"], motion["qd"], motion["dt"]
    err = np.abs(np.angle(np.exp(1j * (q_pred - q))))
    tip = np.hypot(*(forward_points(arm, q_pred)["tool"] - forward_points(arm, q)["tool"]).T)
    v_pred = np.gradient(np.unwrap(q_pred, axis=0), dt, axis=0)
    v_rel = float(np.sqrt(np.mean((v_pred - qd) ** 2)) / max(np.sqrt(np.mean(qd ** 2)), 1e-12))
    jumps = np.abs(np.diff(np.unwrap(q_pred, axis=0), axis=0)).max(1) / dt
    return {"angle_rms": float(np.sqrt(np.mean(err ** 2))), "angle_max": float(err.max()),
            "tool_rms": float(np.sqrt(np.mean(tip ** 2))), "velocity_rel_rms": v_rel,
            "jumps_over_limit": int(np.sum(jumps > qd_max))}
