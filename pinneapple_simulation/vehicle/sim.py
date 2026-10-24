"""Time-domain simulation of a wheeled vehicle on deformable terrain: multibody dynamics + Wong-Reece wheels.

Per step (``dt``, linearly implicit Euler, :meth:`MultibodyTree.step`):

1. Wheel geometry on the undisturbed surface: lowest rim point, terrain normal in the wheel plane, rolling
   direction, sinkage below the undisturbed surface; the soil memory ahead of the wheel gives its prior sinkage
   (multi-pass).
2. Slip from the wheel's angular velocity and its centre velocity, slip angle from the lateral velocity. The slip
   that sets the slip-sinkage exponent is the slip lagged over ``sink_length`` metres of rim travel: slip-sinkage
   builds up as the wheel turns and digs, a stopped wheel neither builds nor loses it, and the slip ratio of a
   nearly stopped wheel (a ratio of noise) cannot feed back into the sinkage (a model setting).
3. Wong-Reece forces (:class:`~pinneapple_simulation.terramechanics.WongReeceModel`): drawbar pull and normal force in
   the wheel plane, lateral force along the axle, soil torque about it; their derivatives with respect to the wheel
   velocities enter the implicit step, so millisecond steps stay stable while the wheels slip.
4. Vertical soil damping ``c = zeta 2 sqrt(k m)`` (``k = dF_z/dz``, ``m`` the vehicle mass per wheel) -- granular soil
   dissipates the energy of a bouncing wheel; the Wong-Reece laws are quasi-static and have none. ``zeta``
   (``soil_damping``) is a model setting, not a measured soil property (Chrono's SCM soil uses the same idea).
5. Rigid rocks: penalty contact (spring-damper, Coulomb friction) on the rim (grousers included).
6. Actuators: wheel speed PI with the motor torque limit, steering PD (heading hold), active suspension PD with a
   load-levelling loop (an assumed control law: the arm angles move to equalise the wheel loads), all torque-limited.
7. The terrain records the plastic trace of every wheel (ruts) and a grouser mark each time a grouser passes the
   bottom of a loaded wheel.

Ground made of several soils: pass a list of soils and ``soil_map(x, y) -> index`` (each wheel gets the Wong-Reece
law of the soil under its contact point). A route: ``run(..., route=[(x, y), ...])`` steers by pure pursuit along the
polyline and stops at its end; ``stop(sample) -> reason or None`` ends a run early (e.g. a traction monitor).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..multibody.tree import cross, quat_to_matrix
from ..terramechanics.soils import Soil
from ..terramechanics.terrain import DeformableTerrain
from ..terramechanics.wong_reece import WongReeceModel, slip_angle, slip_ratio
from .build import VehicleModel


@dataclass
class VehicleRun:
    """Recorded samples (arrays with a leading time axis; wheel arrays are (T, n_wheels))."""
    model: VehicleModel
    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    wheel: dict
    joints: dict
    chassis: dict
    stresses: dict
    terrain_events: list
    meta: dict = field(default_factory=dict)

    @property
    def corner_names(self):
        return self.model.spec.corner_names

    def distance(self) -> np.ndarray:
        xy = self.q[:, :2]
        return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]

    def energy(self) -> np.ndarray:
        """Cumulative mechanical drive energy, sum over wheels of max(tau omega, 0) [J] (no regeneration)."""
        P = np.clip(self.wheel["drive_torque"] * self.wheel["omega_joint"], 0, None).sum(1)
        return np.r_[0.0, np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(self.t))]

    def summary(self) -> dict:
        moving = self.chassis["speed"] > 0.2 * max(self.meta.get("speed_cmd", 0.1), 1e-3)
        sl = self.wheel["slip"][moving] if moving.any() else self.wheel["slip"]
        d = self.distance()[-1]
        return {"duration_s": float(self.t[-1] - self.t[0]), "distance_m": float(d),
                "mean_speed_m_s": float(d / max(self.t[-1] - self.t[0], 1e-9)),
                "mean_slip": float(np.mean(sl)), "max_slip": float(np.max(self.wheel["slip"])),
                "max_sinkage_mm": float(1e3 * np.max(self.wheel["sinkage"])),
                "mean_sinkage_mm": float(1e3 * np.mean(self.wheel["sinkage"][moving] if moving.any() else 0.0)),
                "energy_J": float(self.energy()[-1]), "energy_per_metre_J_m": float(self.energy()[-1] / max(d, 1e-9)),
                "max_load_N": float(self.wheel["load"].max()), "min_load_N": float(self.wheel["load"].min()),
                "max_pitch_deg": float(np.degrees(np.abs(self.chassis["pitch"])).max()),
                "max_roll_deg": float(np.degrees(np.abs(self.chassis["roll"])).max()),
                "max_drive_torque_Nm": float(np.abs(self.wheel["drive_torque"]).max()),
                "max_rock_force_N": float(self.wheel["rock_force"].max())}

    def to_csv(self, path: str) -> str:
        cols = {"t_s": self.t, "x_m": self.q[:, 0], "y_m": self.q[:, 1], "z_m": self.q[:, 2],
                "speed_m_s": self.chassis["speed"], "pitch_deg": np.degrees(self.chassis["pitch"]),
                "roll_deg": np.degrees(self.chassis["roll"]), "yaw_deg": np.degrees(self.chassis["yaw"])}
        units = {"sinkage": ("sinkage_mm", 1e3), "slip": ("slip", 1.0), "load": ("load_N", 1.0),
                 "drawbar_pull": ("drawbar_pull_N", 1.0), "lateral": ("lateral_N", 1.0),
                 "drive_torque": ("drive_torque_Nm", 1.0), "slip_angle": ("slip_angle_deg", 180 / math.pi)}
        for k, (nm, sc) in units.items():
            for j, c in enumerate(self.corner_names):
                cols[f"{nm}_{c}"] = sc * self.wheel[k][:, j]
        A = np.stack(list(cols.values()), 1)
        np.savetxt(path, A, delimiter=",", header=",".join(cols), comments="", fmt="%.6g")
        return path

    def replay_terrain(self, terrain: DeformableTerrain, t_end: float, start: int = 0) -> int:
        """Apply the recorded terrain events with time <= ``t_end`` (from index ``start``) to ``terrain``; returns
        the index of the next event (pass it back as ``start`` to continue incrementally)."""
        ev = self.terrain_events
        k = start
        while k < len(ev) and ev[k][0] <= t_end:
            _, kind, args = ev[k]
            if kind == "rut":
                terrain.record_wheel(*args)
            else:
                terrain.stamp_grouser(*args)
            k += 1
        return k


class VehicleSimulator:
    """``model`` from :func:`~.build.build_vehicle`, ``terrain`` a :class:`DeformableTerrain` (rocks included),
    ``soil`` one :class:`Soil`, or a list of soils with ``soil_map(x, y) -> index`` into it (arrays in, int array
    out) for ground made of several soils. ``wheel_speeds``: "equal" (every wheel gets the speed command) or
    "kinematic" (each wheel at the speed its distance to the turning centre asks for)."""

    def __init__(self, model: VehicleModel, terrain: DeformableTerrain, soil: Soil | Sequence[Soil], g: float = 1.62,
                 soil_damping: float = 0.7, rock_stiffness: float = 3e5, rock_damping: float = 4e3,
                 rock_friction: float = 0.6, stamp_every: int = 5, imprint_ratio: float = 0.6, n_gl: int = 16,
                 level_gain: float = 0.6, v_eps: float = 0.02, multipass_shear: bool = True,
                 sink_length: float = 0.15, soil_map: Callable | None = None, wheel_speeds: str = "equal"):
        self.m = model
        self.spec = model.spec
        self.tree = model.tree
        self.terrain = terrain
        soils = list(soil) if isinstance(soil, (list, tuple)) else [soil]
        if len(soils) > 1 and soil_map is None:
            raise ValueError("several soils need soil_map(x, y) -> index")
        self.soils = soils
        self.soil = soils[0]
        self.soil_map = soil_map
        if wheel_speeds not in ("equal", "kinematic"):
            raise ValueError("wheel_speeds must be 'equal' or 'kinematic'")
        self.wheel_speeds = wheel_speeds
        self.g = float(g)
        self.wrs = [WongReeceModel(sl, self.spec.wheel, g=g, n_gl=n_gl) for sl in soils]
        self.wr = self.wrs[0]
        self.terrain.rebound = self.wr.rebound
        self.zeta = soil_damping
        self.k_rock, self.c_rock, self.mu_rock = rock_stiffness, rock_damping, rock_friction
        self.stamp_every = int(stamp_every)
        self.imprint_ratio = imprint_ratio
        self.level_gain = level_gain
        self.v_eps = float(v_eps)              # slip regularisation near standstill [m/s]
        self.sink_length = float(sink_length)  # rim travel over which slip-sinkage follows the slip [m]
        self.s_sink = np.zeros(len(self.spec.corners))
        self.multipass_shear = bool(multipass_shear)
        self.nw = len(self.spec.corners)
        self._rim = np.linspace(0, 2 * np.pi, 48, endpoint=False)

    # ------------------------------------------------------------------ geometry
    def _ground(self, c, a):
        T = self.terrain
        p = np.column_stack([c[:, 0], c[:, 1], T.height0(c[:, 0], c[:, 1])])
        for _ in range(3):
            n = T.normal0(p[:, 0], p[:, 1])
            d = np.einsum("ij,ij->i", c - p, n)
            foot = c - d[:, None] * n
            p = np.column_stack([foot[:, 0], foot[:, 1], T.height0(foot[:, 0], foot[:, 1])])
        n = T.normal0(p[:, 0], p[:, 1])
        nw = n - np.einsum("ij,ij->i", n, a)[:, None] * a
        nw /= np.linalg.norm(nw, axis=1, keepdims=True)
        dist = np.einsum("ij,ij->i", c - p, nw)
        z = self.spec.wheel.radius - dist
        t = cross(a, nw)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        return z, p, nw, t

    # ------------------------------------------------------------------ forces
    def _soil_index(self, P) -> np.ndarray:
        if self.soil_map is None:
            return np.zeros(len(P), int)
        return np.clip(np.asarray(self.soil_map(P[:, 0], P[:, 1]), int).reshape(len(P)), 0, len(self.soils) - 1)

    def _forces(self, idx, slip, sinkage, beta, zu, j0, *, sinkage_slip, stresses: bool = False):
        """Wong-Reece forces of each sample on its own soil (``idx``), merged into one result."""
        groups = np.unique(idx)
        if len(groups) == 1:
            return self.wrs[groups[0]].forces(slip, sinkage, beta, zu, j0, stresses=stresses,
                                              sinkage_slip=sinkage_slip)
        parts = {g: self.wrs[g].forces(slip[idx == g], sinkage[idx == g], beta[idx == g], zu[idx == g],
                                       j0[idx == g], stresses=stresses, sinkage_slip=sinkage_slip[idx == g])
                 for g in groups}
        out = parts[groups[0]]
        merged = {}
        for name, val in vars(out).items():
            if not isinstance(val, np.ndarray):
                continue
            full = np.zeros((len(idx),) + val.shape[1:], val.dtype)
            for g, r in parts.items():
                full[idx == g] = getattr(r, name)
            merged[name] = full
        for name, val in merged.items():
            setattr(out, name, val)
        return out

    def _wheel_forces(self, kin, v, dt, rec):
        tree, M = self.tree, self.m
        nv = tree.nv
        R = self.spec.wheel.radius
        b = self.spec.wheel.width
        wb = M.wheel_body
        c = kin.p[wb]
        a = kin.axis[M.wheel]
        z, P, nw, t = self._ground(c, a)
        Jc = np.stack([tree.point_jacobian(kin, wb[i], c[i]) for i in range(self.nw)])
        Jw = kin.Jw[wb]
        vc = np.einsum("wij,j->wi", Jc, v)
        om = np.einsum("wij,j,wi->w", Jw, v, a)
        v_long = np.einsum("ij,ij->i", vc, t)
        v_lat = np.einsum("ij,ij->i", vc, a)
        ahead = np.sqrt(np.clip(2 * R * np.clip(z, 0, None), 0, None)) + 2 * self.terrain.cell
        th = t[:, :2] / np.maximum(np.linalg.norm(t[:, :2], axis=1, keepdims=True), 1e-9)
        front_pt = P[:, :2] + th * ahead[:, None]              # beyond the entry point of the arc (own trace)
        zu = self.terrain.precompaction(front_pt, t[:, :2], 0.5 * b)
        j0 = self.terrain.prior_shear(front_pt, t[:, :2], 0.5 * b) if self.multipass_shear else np.zeros(self.nw)
        du = np.array([2e-3, 2e-3, 2e-3 / R])
        U = np.stack([v_long, v_lat, om], 1)
        Us = np.concatenate([U] + [U + du[k] * np.eye(3)[k] for k in range(3)])
        s_all = slip_ratio(Us[:, 0], Us[:, 2], R, self.v_eps)
        b_all = slip_angle(Us[:, 0], Us[:, 1], self.v_eps)
        s_sink = np.tile(self.s_sink, 4)                       # lagged state: no derivative w.r.t. the velocities
        sidx = self._soil_index(P)
        res = self._forces(np.tile(sidx, 4), s_all, np.tile(z, 4), b_all, np.tile(zu, 4), np.tile(j0, 4),
                           sinkage_slip=s_sink)
        F0 = np.stack([res.Fx, res.Fy, res.Fz, res.My], 1).reshape(4, self.nw, 4)
        base = F0[0]
        dF = np.stack([(F0[k + 1] - base) / du[k] for k in range(3)], -1)        # (w, 4 quantities, 3 inputs)
        Q = np.zeros(nv)
        D = np.zeros((nv, nv))
        on = res.z_eff[: self.nw] > 0
        ndamp = np.zeros(self.nw)
        mshare = tree.total_mass() / self.nw
        for i in range(self.nw):
            Fx, Fy, Fz, My = base[i]
            Fw = Fx * t[i] + Fz * nw[i] + Fy * a[i]
            Q += Jc[i].T @ Fw - Jw[i].T @ (My * a[i])
            Ju = np.stack([t[i] @ Jc[i], a[i] @ Jc[i], a[i] @ Jw[i]])
            dFw = np.outer(t[i], dF[i, 0] @ Ju) + np.outer(a[i], dF[i, 1] @ Ju) + np.outer(nw[i], dF[i, 2] @ Ju)
            D += Jc[i].T @ dFw - Jw[i].T @ np.outer(a[i], dF[i, 3] @ Ju)
            if on[i]:
                n_exp = float(self.wrs[sidx[i]].sinkage_exponent(s_sink[i]))
                k = (n_exp + 0.5) * max(Fz, 0.0) / max(res.z_eff[i], 2e-3)
                cz = self.zeta * 2.0 * math.sqrt(max(k, 0.0) * mshare)
                Jn = nw[i] @ Jc[i]
                Q -= cz * (Jn @ v) * Jn
                D -= cz * np.outer(Jn, Jn)
                ndamp[i] = cz
        # rocks: penalty contact on the rim (with grousers)
        rock_f = np.zeros(self.nw)
        if self.terrain.rocks:
            Ro = self.spec.wheel.outer_radius
            for i in range(self.nw):
                e1, e2 = t[i], cross(a[i], t[i])
                pts = c[i] + Ro * (np.cos(self._rim)[:, None] * e1 + np.sin(self._rim)[:, None] * e2)
                best = (0.0, None, None)
                for rk in self.terrain.rocks:
                    if math.hypot(rk.x - c[i, 0], rk.y - c[i, 1]) > Ro + max(rk.a, rk.b) + 0.05:
                        continue
                    dep, nr = rk.penetration(pts)
                    j = int(np.argmax(dep))
                    if dep[j] > best[0]:
                        best = (float(dep[j]), pts[j], nr[j])
                if best[1] is None:
                    continue
                dep, q, nr = best
                Jq = tree.point_jacobian(kin, wb[i], q)
                vq = Jq @ v
                vn = nr @ vq
                fn = max(self.k_rock * dep - self.c_rock * vn, 0.0)
                vt = vq - vn * nr
                vt_n = np.linalg.norm(vt)
                vreg = 0.02
                ft = -self.mu_rock * fn * vt / max(vt_n, vreg)
                Q += Jq.T @ (fn * nr + ft)
                Jn = nr @ Jq
                D -= self.c_rock * np.outer(Jn, Jn) + self.k_rock * dt * np.outer(Jn, Jn)
                if vt_n < vreg:
                    Pt = np.eye(3) - np.outer(nr, nr)
                    D -= self.mu_rock * fn / vreg * Jq.T @ Pt @ Jq
                rock_f[i] = fn
        if rec is not None:
            rec.update(z=z, P=P, nw=nw, t=t, a=a, c=c, zu=zu, j0=j0, j_exit=res.j_exit[: self.nw],
                       slip=s_all[: self.nw], s_sink=s_sink[: self.nw], beta=b_all[: self.nw], soil_idx=sidx,
                       F=base, z_eff=res.z_eff[: self.nw], om=om, rock=rock_f, v_long=v_long, damp=ndamp)
        return Q, D

    # ------------------------------------------------------------------ actuators
    def _actuators(self, q, v, dt, state, cmd, rec):
        nv = self.tree.nv
        Q = np.zeros(nv)
        D = np.zeros(nv)
        spec = self.spec
        R = spec.wheel.radius
        tau_drive = np.zeros(self.nw)
        yaw = _yaw(q)
        heading = cmd.get("heading")
        delta_cmd = 0.0
        if heading is not None:
            err = (heading - yaw + math.pi) % (2 * math.pi) - math.pi
            if cmd.get("path_y") is not None:
                hd = math.radians(0)  # noqa: F841
                err += -cmd.get("k_path", 0.8) * (q[1] - cmd["path_y"]) * math.cos(heading)
            delta_cmd = float(np.clip(cmd.get("k_heading", 1.5) * err, -0.35, 0.35))
        scale = self._wheel_speed_scale(delta_cmd)
        for i, c in enumerate(spec.corners):
            # drive: PI on the joint rate with torque saturation and anti-windup
            j = self.m.wheel[i]
            w = v[6 + j]
            w_cmd = min(cmd["speed"] * scale[i] / R, c.drive.speed_limit)
            e = w_cmd - w
            u = c.drive.kp * e + state["I"][i]
            tau = float(np.clip(u, -c.drive.torque_limit, c.drive.torque_limit))
            if abs(u) <= c.drive.torque_limit:
                state["I"][i] += c.drive.ki * e * dt
                D[6 + j] -= c.drive.kp
            Q[6 + j] += tau
            tau_drive[i] = tau
            # steering PD
            j = self.m.steer[i]
            dcmd = delta_cmd * c.front
            u = c.steer.kp * (dcmd - q[7 + j]) - c.steer.kd * v[6 + j]
            Q[6 + j] += float(np.clip(u, -c.steer.torque_limit, c.steer.torque_limit))
            if abs(u) < c.steer.torque_limit:
                D[6 + j] -= c.steer.kd + dt * c.steer.kp
            # suspension PD towards the levelled set point
            j = self.m.susp[i]
            u = c.susp.kp * (state["susp_cmd"][i] - q[7 + j]) - c.susp.kd * v[6 + j]
            Q[6 + j] += float(np.clip(u, -c.susp.torque_limit, c.susp.torque_limit))
            if abs(u) < c.susp.torque_limit:
                D[6 + j] -= c.susp.kd + dt * c.susp.kp
        if rec is not None:
            rec["tau_drive"] = tau_drive
            rec["delta_cmd"] = delta_cmd
        return Q, np.diag(D)

    def _wheel_speed_scale(self, delta: float) -> np.ndarray:
        """Wheel speed / chassis speed per corner. ``wheel_speeds="equal"``: 1 for every wheel. ``"kinematic"``: each
        wheel at the speed of its distance to the instantaneous centre of rotation the steering command sets (front
        wheels at +delta, rear wheels at -delta about the mid-wheelbase point), so wheels do not fight in a turn."""
        if self.wheel_speeds == "equal" or abs(delta) < 1e-4:
            return np.ones(self.nw)
        P = np.array([c.wheel_centre[:2] for c in self.spec.corners], float)
        P = P - P.mean(0)
        xf = np.abs(P[:, 0]).max()
        Rc = xf / math.tan(delta)                                    # signed: > 0 turning left
        d = np.hypot(P[:, 0], Rc - P[:, 1])
        return d / abs(Rc)

    def _level(self, state, Fz, dt):
        """Load-levelling of the warp mode: with four wheels the front/rear and left/right load split is fixed by
        the centre of mass; only the diagonal (warp) share, (FL + RR) - (FR + RL), can be equalised by the arms.
        The arms move along the warp vector (lifting the more loaded diagonal) at a rate proportional to it."""
        if self.level_gain <= 0 or Fz.sum() <= 0:
            return
        up = np.array([c.side for c in self.spec.corners], float)     # +theta lifts a left wheel (y > 0)
        w = np.array([c.side * c.front for c in self.spec.corners], float)
        w /= np.linalg.norm(w)
        warp = float(w @ Fz) / max(Fz.mean(), 1e-6)
        state["lev"] += self.level_gain * warp * w * dt
        lim = [c.susp_limits for c in self.spec.corners]
        state["susp_cmd"] = np.array([np.clip(up[i] * state["lev"][i], lim[i][0] * 0.9, lim[i][1] * 0.9)
                                      for i in range(self.nw)])

    # ------------------------------------------------------------------ run
    def initial_state(self, x=0.0, y=0.0, yaw_deg=0.0, lift: float = 0.0):
        """Chassis aligned with the terrain normal at (x, y) and heading ``yaw_deg``, raised until the deepest wheel
        has its grousers in the surface (rim ``lift`` above it)."""
        n = self.terrain.normal0(np.array([x]), np.array([y]))[0]
        f = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0])
        f = f - (f @ n) * n
        f /= np.linalg.norm(f)
        R = np.column_stack([f, cross(n, f), n])
        h = float(self.terrain.height0(x, y))
        q, v = self.tree.initial_state(p=(x, y, h), R=R)
        kin = self.tree.kinematics(q)
        c = kin.p[self.m.wheel_body]
        z = self._ground(c, kin.axis[self.m.wheel])[0]
        q[2] += float(np.max(z)) + lift
        return q, v

    def run(self, duration: float, speed=0.1, *, dt: float = 2e-3, output_dt: float = 0.05, start=(0.0, 0.0),
            yaw_deg: float = 0.0, settle: float = 2.0, ramp: float = 2.0, heading_hold: bool = True,
            path_y: float | None = None, record_stresses: bool = True, progress=None, q0=None, v0=None,
            route=None, lookahead: float = 1.5, goal_tol: float = 0.3, stop=None) -> VehicleRun:
        """Settle at rest for ``settle`` s, then drive at the wheel-speed command ``speed`` [m/s] (a float or a
        function of time) for ``duration`` s, ramping over ``ramp`` s. Samples every ``output_dt`` s.

        ``route``: waypoints (x, y) [m]; the rover starts at the first one facing the second (``start`` and
        ``yaw_deg`` are ignored), steers towards the point ``lookahead`` metres ahead on the polyline (pure pursuit)
        and the run ends within ``goal_tol`` of the last one or once its progress along the route passes the end. ``stop(sample)`` is called at every output sample with
        ``{"t", "x", "y", "speed", "slip", "sinkage", "soil_idx"}`` (wheel arrays) and ends the run when it returns
        a reason; ``meta["ended"]`` records why the run ended ("duration", "goal" or that reason)."""
        tree = self.tree
        path = None
        if route is not None:
            path = np.asarray(route, float)
            if path.ndim != 2 or len(path) < 2:
                raise ValueError("route needs at least two (x, y) waypoints")
            seg = np.diff(path, axis=0)
            start = tuple(path[0])
            yaw_deg = math.degrees(math.atan2(seg[0, 1], seg[0, 0]))
            s_cum = np.r_[0.0, np.cumsum(np.linalg.norm(seg, axis=1))]
        ended = "duration"
        if q0 is None:
            q, v = self.initial_state(start[0], start[1], yaw_deg, lift=0.002)
        else:
            q, v = q0.copy(), v0.copy()
        self.s_sink = np.zeros(self.nw)
        state = {"I": np.zeros(self.nw), "lev": np.zeros(self.nw), "susp_cmd": np.zeros(self.nw),
                 "spin": None, "energy": 0.0}
        heading = math.radians(yaw_deg) if heading_hold else None
        spd = speed if callable(speed) else (lambda tt, s=speed: s)
        t_total = settle + duration
        n = int(round(t_total / dt))
        every = max(1, int(round(output_dt / dt)))
        rec_all = {k: [] for k in ("t", "q", "v")}
        W = {k: [] for k in ("sinkage", "z_eff", "slip", "slip_angle", "drawbar_pull", "lateral", "load",
                             "soil_torque", "drive_torque", "omega", "omega_joint", "precompaction", "rock_force",
                             "contact_point", "normal", "tangent", "spin", "centre", "soil_idx")}
        J = {k: [] for k in ("susp", "steer")}
        C = {k: [] for k in ("speed", "pitch", "roll", "yaw", "com")}
        S = {k: [] for k in ("theta", "sigma", "tau_x")}
        events = []
        pitch_g = self.spec.wheel.grousers.pitch_angle() if self.spec.wheel.grousers.active else None
        for k in range(n + 1):
            tt = k * dt
            drive_t = tt - settle
            if path is not None:
                heading = _pursuit_heading(path, s_cum, q[0], q[1], lookahead)
            cmd = {"speed": 0.0 if drive_t < 0 else spd(drive_t) * min(1.0, drive_t / max(ramp, 1e-9)),
                   "heading": heading, "path_y": path_y}
            info = {}

            def forces(kin, qq, vv, _info=info, _cmd=cmd):
                Qw, Dw = self._wheel_forces(kin, vv, dt, _info)
                Qa, Da = self._actuators(qq, vv, dt, state, _cmd, _info)
                return Qw + Qa, Dw + Da

            q_new, v_new, kin = tree.step(q, v, dt, forces)
            self._level(state, np.clip(info["F"][:, 2], 0, None), dt)
            # terrain: ruts and grouser marks
            spin = np.array([q[7 + j] for j in self.m.wheel])
            if k % self.stamp_every == 0:
                for i in range(self.nw):
                    if info["z_eff"][i] > 0:
                        args = (float(info["P"][i, 0]), float(info["P"][i, 1]), info["t"][i][:2].copy(),
                                self.spec.wheel.radius, self.spec.wheel.width, float(info["z"][i]),
                                *_peak_stress(self.wrs[info["soil_idx"][i]], info, i), float(info["j_exit"][i]))
                        self.terrain.record_wheel(*args)
                        events.append((tt, "rut", args))
            if pitch_g is not None:
                if state["spin"] is not None:
                    g = self.spec.wheel.grousers
                    for i in range(self.nw):
                        if info["z_eff"][i] > 0 and math.floor(spin[i] / pitch_g) != math.floor(state["spin"][i]
                                                                                              / pitch_g):
                            args = (float(info["P"][i, 0]), float(info["P"][i, 1]), info["t"][i][:2].copy(),
                                    self.spec.wheel.width, self.imprint_ratio * g.height, g.thickness * 2.5, g.shape,
                                    g.chevron_angle_deg)
                            self.terrain.stamp_grouser(*args)
                            events.append((tt, "grouser", args))
                state["spin"] = spin
            if k % every == 0 and tt >= settle - 1e-9:
                rec_all["t"].append(tt - settle)
                rec_all["q"].append(q.copy())
                rec_all["v"].append(v.copy())
                F = info["F"]
                W["sinkage"].append(np.clip(info["z"], 0, None))
                W["z_eff"].append(info["z_eff"])
                W["slip"].append(info["slip"])
                W["slip_angle"].append(info["beta"])
                W["drawbar_pull"].append(F[:, 0])
                W["lateral"].append(F[:, 1])
                W["load"].append(F[:, 2])
                W["soil_torque"].append(F[:, 3])
                W["drive_torque"].append(info["tau_drive"])
                W["omega"].append(info["om"])
                W["omega_joint"].append(np.array([v[6 + j] for j in self.m.wheel]))
                W["precompaction"].append(info["zu"])
                W["rock_force"].append(info["rock"])
                W["contact_point"].append(info["P"])
                W["normal"].append(info["nw"])
                W["tangent"].append(info["t"])
                W["spin"].append(spin)
                W["centre"].append(info["c"])
                W["soil_idx"].append(info["soil_idx"])
                J["susp"].append(np.array([q[7 + j] for j in self.m.susp]))
                J["steer"].append(np.array([q[7 + j] for j in self.m.steer]))
                Rb = quat_to_matrix(q[3:7])
                C["yaw"].append(math.atan2(Rb[1, 0], Rb[0, 0]))
                C["pitch"].append(math.asin(np.clip(Rb[2, 0], -1, 1)))
                C["roll"].append(math.atan2(Rb[2, 1], Rb[2, 2]))
                hd = np.array([math.cos(C["yaw"][-1]), math.sin(C["yaw"][-1])])
                C["speed"].append(float(v[:2] @ hd + (cross(v[3:6], kin.c[0] - kin.p[0])[:2] @ hd)))
                C["com"].append(kin.c[0].copy())
                if record_stresses:
                    r = self._forces(info["soil_idx"], info["slip"], info["z"], info["beta"], info["zu"], info["j0"],
                                     stresses=True, sinkage_slip=info["s_sink"])
                    S["theta"].append(r.theta)
                    S["sigma"].append(r.sigma)
                    S["tau_x"].append(r.tau_x)
                if progress:
                    progress(tt - settle, duration)
                if path is not None and drive_t > 0 and (
                        math.hypot(q[0] - path[-1, 0], q[1] - path[-1, 1]) < goal_tol
                        or _route_progress(path, s_cum, q[0], q[1]) >= s_cum[-1] - 1e-3):
                    ended = "goal"
                if stop is not None and drive_t > 0 and ended == "duration":
                    why = stop({"t": tt - settle, "x": float(q[0]), "y": float(q[1]), "speed": C["speed"][-1],
                                "slip": info["slip"], "sinkage": np.clip(info["z"], 0, None),
                                "soil_idx": info["soil_idx"]})
                    if why:
                        ended = str(why)
                if ended != "duration":
                    break
            # slip-sinkage state: first-order lag over the rim travel of this step
            a = np.clip(np.abs(info["om"]) * self.spec.wheel.radius * dt / max(self.sink_length, 1e-9), 0.0, 1.0)
            self.s_sink = self.s_sink + a * (np.clip(info["slip"], -0.99, 0.99) - self.s_sink)
            q, v = q_new, v_new
        arr = lambda d: {kk: np.array(vv) for kk, vv in d.items()}  # noqa: E731
        return VehicleRun(self.m, np.array(rec_all["t"]), np.array(rec_all["q"]), np.array(rec_all["v"]), arr(W),
                          arr(J), arr(C), arr(S) if record_stresses else {}, events,
                          {"dt": dt, "speed_cmd": speed if not callable(speed) else None, "soil": self.soil.name,
                           "soils": [sl.name for sl in self.soils], "g": self.g, "settle_s": settle,
                           "soil_damping": self.zeta, "ended": ended})


def _peak_stress(wr, info, i):
    """Peak normal and shear stress of wheel ``i`` [Pa] (for the terrain contact maps)."""
    r = wr.forces(info["slip"][i:i + 1], info["z"][i:i + 1], info["beta"][i:i + 1], info["zu"][i:i + 1],
                  info["j0"][i:i + 1], stresses=True, sinkage_slip=info["s_sink"][i:i + 1])
    return float(r.sigma.max()), float(np.abs(r.tau_x).max())


def _route_progress(path, s_cum, x, y, extend: float = 5.0):
    """Arc length [m] of the point of ``path`` closest to (x, y); the last segment is extended by ``extend`` metres
    so a vehicle that passed the end has progress beyond the route length."""
    a, d = path[:-1], np.diff(path, axis=0)
    L = np.sqrt(np.maximum((d ** 2).sum(1), 1e-12))
    hi = np.ones(len(d))
    hi[-1] = 1.0 + extend / L[-1]
    u = np.clip(((np.array([x, y]) - a) * d).sum(1) / L ** 2, 0.0, hi)
    foot = a + u[:, None] * d
    k = int(np.argmin(((foot - [x, y]) ** 2).sum(1)))
    return float(s_cum[k] + u[k] * L[k])


def _pursuit_heading(path, s_cum, x, y, lookahead):
    """Heading [rad] from (x, y) to the point ``lookahead`` metres along ``path`` past the closest point on it."""
    s_t = min(min(_route_progress(path, s_cum, x, y, extend=0.0), s_cum[-1]) + lookahead, s_cum[-1])
    target = np.array([np.interp(s_t, s_cum, path[:, 0]), np.interp(s_t, s_cum, path[:, 1])])
    return math.atan2(target[1] - y, target[0] - x)


def _yaw(q):
    R = quat_to_matrix(q[3:7])
    return math.atan2(R[1, 0], R[0, 0])


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


__all__ = ["VehicleSimulator", "VehicleRun"]
