"""Parametric 3-D vehicles for CFD and photoreal rendering: a road car and a launch vehicle.

Both return ``Part`` lists (``pinneapple_design.aero.airframe.Part``: name, vertices, faces, material, group, aero),
in metres, x along the vehicle from the nose (the flow direction), z up, symmetric about y = 0. The parts with
``aero=True`` are closed surfaces that form the CFD geometry; the others (rims, the rocket's engine bells and plume)
are only drawn. Materials are names from ``VEHICLE_MATERIALS`` (registered with the studio on import).

    from pinneapple_design.geometry.vehicles3d import RoadCar, LaunchVehicle
    car = RoadCar(style="fastback", slant_deg=22)
    parts = car.parts()                      # body (paint / glass / lights / trim faces), four wheels
    bodies = car.cfd_bodies()                # {"body": (V, F), "wheels": (V, F)} for ExternalFlow (ground at z = 0)
    rocket = LaunchVehicle()
    rocket.barrowman()                       # normal-force slope and centre of pressure, Barrowman's equations
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pinneapple_design.aero.airframe import Part

VEHICLE_MATERIALS = {
    "car_paint": {"color": [0.62, 0.05, 0.06], "metallic": 0.35, "roughness": 0.18, "clearcoat": 1.0},
    "car_glass": {"color": [0.03, 0.045, 0.06], "metallic": 0.0, "roughness": 0.03, "clearcoat": 1.0},
    "car_trim": {"color": [0.05, 0.055, 0.06], "metallic": 0.0, "roughness": 0.5},
    "tyre": {"color": [0.035, 0.035, 0.038], "metallic": 0.0, "roughness": 0.85},
    "rim": {"color": [0.78, 0.79, 0.81], "metallic": 1.0, "roughness": 0.22},
    "headlight": {"color": [0.95, 0.96, 1.0], "metallic": 0.0, "roughness": 0.1, "emissive": [0.95, 0.96, 1.0]},
    "taillight": {"color": [0.7, 0.02, 0.02], "metallic": 0.0, "roughness": 0.2, "emissive": [0.9, 0.04, 0.03]},
    "rocket_white": {"color": [0.95, 0.95, 0.94], "metallic": 0.0, "roughness": 0.35},
    "rocket_black": {"color": [0.06, 0.06, 0.07], "metallic": 0.0, "roughness": 0.45},
    "nozzle": {"color": [0.42, 0.40, 0.38], "metallic": 1.0, "roughness": 0.35},
    "flame": {"color": [1.0, 0.55, 0.2], "metallic": 0.0, "roughness": 1.0, "emissive": [1.0, 0.45, 0.12],
              "alpha": 0.6, "double_sided": True},
    "flame_core": {"color": [1.0, 0.93, 0.75], "metallic": 0.0, "roughness": 1.0, "emissive": [1.0, 0.9, 0.7]},
    "asphalt": {"color": [0.16, 0.165, 0.17], "metallic": 0.0, "roughness": 0.9},
}


def _register_materials() -> None:
    try:
        from pinneapple_tools.visualization.studio.scene import MATERIALS
        MATERIALS.update({k: v for k, v in VEHICLE_MATERIALS.items() if k not in MATERIALS})
    except Exception:                                       # noqa: BLE001 - the studio is optional
        pass


_register_materials()


def _smooth(xk, yk):
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(np.asarray(xk, float), np.asarray(yk, float))


def _loft_rings(rings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed surface through rings (n_rings, n_pts, 3) ordered along +x; the first and last rings are capped."""
    nr, n, _ = rings.shape
    V = rings.reshape(-1, 3)
    F = []
    for i in range(nr - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            F += [[a, b, a + n], [b, b + n, a + n]]
    V = list(V)
    for i in (0, nr - 1):
        c = len(V)
        V.append(rings[i].mean(0))
        for j in range(n):
            F.append([c, i * n + (j + 1) % n, i * n + j])
    V, F = np.asarray(V), np.asarray(F, np.int64)
    # orient every face outward (consistent with the signed volume and per-face against the local axis)
    ctr = np.c_[V[:, 0], np.zeros(len(V)), np.zeros(len(V))]
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = tri.mean(1) - np.c_[tri.mean(1)[:, 0], np.full(len(F), 0.0), np.full(len(F), rings[..., 2].mean())]
    side = slice(0, 2 * (nr - 1) * n)
    if (np.einsum("ij,ij->i", nrm[side], out[side]) < 0).mean() > 0.5:
        F[side] = F[side, ::-1]
    for k, i in enumerate((0, nr - 1)):
        cs = slice(2 * (nr - 1) * n + k * n, 2 * (nr - 1) * n + (k + 1) * n)
        tri = V[F[cs]]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        want = -1.0 if i == 0 else 1.0
        if (nrm[:, 0] * want < 0).mean() > 0.5:
            F[cs] = F[cs, ::-1]
    del ctr
    return V, F


def _revolve(x: np.ndarray, r: np.ndarray, n: int = 64, axis_z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Body of revolution about the x axis (at height ``axis_z``) through radii r(x); ends capped."""
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rings = np.stack([np.c_[np.full(n, xi), ri * np.cos(th), axis_z + ri * np.sin(th)] for xi, ri in zip(x, r, strict=True)])
    return _loft_rings(rings)


def _box_fin(root_x: float, root_c: float, tip_x: float, tip_c: float, r0: float, span: float, t: float,
             angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """A trapezoidal fin (flat plate with bevelled edges) on a body of radius r0, rotated about x by angle."""
    prof = []                                   # (x, y) of the planform, root at y = r0 - small overlap
    y0, y1 = r0 * 0.85, r0 + span
    pts = [(root_x, y0), (root_x + root_c, y0), (tip_x + tip_c, y1), (tip_x, y1)]
    V = []
    for zz in (-t / 2, t / 2):
        for (xx, yy) in pts:
            V.append((xx, yy, zz))
    V = np.array(V, float)
    # thin the leading/trailing edges: move the edge vertices of each face to the mid-plane
    F = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                  [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    a = math.radians(angle_deg)
    R = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    V = V @ R.T
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    c = V.mean(0)
    flip = np.einsum("ij,ij->i", nrm, tri.mean(1) - c) < 0
    F[flip] = F[flip][:, ::-1]
    del prof
    return V, F


def _resample(P: np.ndarray, n: int) -> np.ndarray:
    """n points evenly spaced by arc length along the polyline P, the last point excluded (rings are closed)."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.linspace(0, d[-1], n + 1)[:-1]
    return np.c_[np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])]


def _merge(meshes) -> tuple[np.ndarray, np.ndarray]:
    Vs, Fs, off = [], [], 0
    for V, F in meshes:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def _split_by(V, F, mats, name, group="", aero=True) -> list[Part]:
    out = []
    mats = np.asarray(mats, dtype=object)
    for m in dict.fromkeys(mats.tolist()):
        sel = F[mats == m]
        used, inv = np.unique(sel, return_inverse=True)
        out.append(Part(f"{name}_{m}", V[used], inv.reshape(-1, 3), m, group or name, aero))
    return out


def road(length: float = 9.0, width: float = 6.5, x0: float = -2.2) -> Part:
    """A flat asphalt plane at z = 0 (drawn only)."""
    V = np.array([[x0, -width / 2, 0], [x0 + length, -width / 2, 0], [x0 + length, width / 2, 0], [x0, width / 2, 0]],
                 float)
    return Part("road", V, np.array([[0, 1, 2], [0, 2, 3]]), "asphalt", "scenery", False)


# ============================================================================ road car
@dataclass
class RoadCar:
    """A four-door coupé / sedan: ``style`` "fastback" (roof runs into a slanted rear window down to the tail) or
    "notchback" (rear window, then a trunk deck). Dimensions in metres."""
    length: float = 4.6
    width: float = 1.86
    roof_height: float = 1.36
    clearance: float = 0.13
    hood_height: float = 0.92
    deck_height: float = 1.0
    windshield_deg: float = 28.0         # rake of the windshield from the horizontal
    slant_deg: float = 22.0              # fastback / rear-window angle from the horizontal
    style: str = "fastback"
    boat_tail: float = 0.06              # half-width reduction at the tail
    diffuser_deg: float = 6.0
    wheel_radius: float = 0.35
    wheel_width: float = 0.25
    wheelbase: float = 2.80
    front_overhang: float = 0.92
    n_x: int = 300
    n_ring: int = 160

    # -------------------------------------------------------------- profiles along x
    def _stations(self):
        L = self.length
        x_cowl = 1.25
        x_roof0 = x_cowl + (self.roof_height - self.hood_height) / math.tan(math.radians(self.windshield_deg))
        if self.style == "fastback":
            x_tail_top = L - 0.18
            x_roof1 = max(x_roof0 + 0.5, x_tail_top - (self.roof_height - self.deck_height) /
                          math.tan(math.radians(self.slant_deg)))
            xs = [0.0, 0.06, 0.25, 0.7, x_cowl, x_roof0, 0.5 * (x_roof0 + x_roof1), x_roof1, x_tail_top, L]
            zt = [0.58, 0.70, 0.80, 0.88, self.hood_height, self.roof_height, self.roof_height + 0.02,
                  self.roof_height, self.deck_height, self.deck_height - 0.05]
            x_glass = (x_cowl + 0.05, x_tail_top - 0.25)
        else:
            x_rw1 = L - 1.0
            x_roof1 = max(x_roof0 + 0.6, x_rw1 - (self.roof_height - self.deck_height) /
                          math.tan(math.radians(self.slant_deg)))
            xs = [0.0, 0.06, 0.25, 0.7, x_cowl, x_roof0, 0.5 * (x_roof0 + x_roof1), x_roof1, x_rw1, L - 0.15, L]
            zt = [0.58, 0.70, 0.80, 0.88, self.hood_height, self.roof_height, self.roof_height + 0.02,
                  self.roof_height, self.deck_height, self.deck_height + 0.01, self.deck_height - 0.06]
            x_glass = (x_cowl + 0.05, x_rw1 - 0.03)
        top = _smooth(xs, zt)
        belt = _smooth([0, 0.7, x_cowl, L - 0.4, L], [0.6, 0.86, self.hood_height - 0.02, self.deck_height - 0.06,
                                                       self.deck_height - 0.1])
        c = self.clearance
        xd = L - 0.75
        bottom = _smooth([0, 0.12, 0.45, xd, L], [0.36, 0.22, c, c, c + 0.75 * math.tan(math.radians(self.diffuser_deg))])
        W = self.width / 2
        half = _smooth([0, 0.08, 0.5, 1.2, 2.4, L - 0.6, L], [0.55, 0.72, 0.88, W, W, W - 0.02, W - self.boat_tail])
        return top, belt, bottom, half, x_glass

    def _body(self):
        top, belt, bottom, half, x_glass = self._stations()
        L = self.length
        u = np.linspace(0, 1, self.n_x)
        x = L * (0.5 - 0.5 * np.cos(math.pi * u))               # dense near the nose and the tail
        n_up, n_lo = self.n_ring // 2, self.n_ring - self.n_ring // 2
        ex = 2 / 4.5
        rings = []
        for xi in x:
            zt, zb, zbelt, b = float(top(xi)), float(bottom(xi)), float(belt(xi)), float(half(xi))
            zt = max(zt, zbelt + 0.01)
            nose = min(1.0, math.sqrt(max(xi, 1e-4) / 0.30))     # round the nose in plan and elevation
            zm = zbelt
            k = 0.35 + 0.65 * nose
            zt_, zb_ = zm + (zt - zm) * k, zm - (zm - zb) * k
            b = b * (0.25 + 0.75 * nose)
            upper, lower = [], []
            for phi in np.linspace(0, math.pi, 400):                             # cabin: right belt -> roof -> left
                cs, sn = math.cos(phi), math.sin(phi)
                f = abs(sn) ** ex
                w = b * (1.0 - 0.30 * f ** 1.3) if zt - zbelt > 0.05 else b
                upper.append((w * math.copysign(abs(cs) ** ex, cs), zm + (zt_ - zm) * f))
            for phi in np.linspace(math.pi, 2 * math.pi, 400):                   # body: left belt -> floor -> right
                cs, sn = math.cos(phi), math.sin(phi)
                lower.append((b * math.copysign(abs(cs) ** ex, cs), zm - (zm - zb_) * abs(sn) ** ex))
            ring = [(xi, yy, zz) for yy, zz in _resample(np.asarray(upper), n_up)] + \
                   [(xi, yy, zz) for yy, zz in _resample(np.asarray(lower), n_lo)]
            rings.append(ring)
        V, F = _loft_rings(np.asarray(rings))
        nr, n = len(x), self.n_ring
        n_side = 2 * (nr - 1) * n
        st = np.full(len(F), -1)
        jj = np.full(len(F), -1)
        for fi in range(n_side):                                  # station and ring index of each side face
            a = F[fi].min()
            st[fi], jj[fi] = a // n, F[fi][F[fi] < (a // n + 1) * n].min() % n if (F[fi] < (a // n + 1) * n).any() else a % n
        xs_f = np.where(st >= 0, 0.5 * (x[np.clip(st, 0, nr - 1)] + x[np.clip(st + 1, 0, nr - 1)]), V[F].mean(1)[:, 0])
        cen = V[F].mean(1)
        mats = np.full(len(F), "car_paint", dtype=object)
        up = (jj >= 0) & (jj < n_up)
        margin = 3                                                # paint between the glass and the belt / roof
        g0, g1 = x_glass
        rel = jj / max(n_up - 1, 1)
        side_win = up & (jj >= margin) & (jj < n_up - margin) & (np.abs(rel - 0.5) > 0.17)
        x_ws = g0 + 0.6 * (self.roof_height - self.hood_height) / math.tan(math.radians(self.windshield_deg))
        x_rw = g1 - 0.45
        screens = up & (jj >= margin) & (jj < n_up - margin) & ((xs_f < x_ws) | (xs_f > x_rw))
        glass = (xs_f > g0) & (xs_f < g1) & (side_win | screens)
        pillar = (np.abs(xs_f - (g0 + 0.50 * (g1 - g0))) < 0.05) & side_win & ~screens
        apillar = (np.abs(xs_f - x_ws) < 0.035) & up
        mats[glass & ~pillar & ~apillar] = "car_glass"
        lo_band = (jj >= n_up) & ((jj - n_up < 5) | (n - 1 - jj < 5))            # just below the belt, both sides
        mats[lo_band & (xs_f > 0.02) & (xs_f < 0.40)] = "headlight"
        mats[lo_band & (xs_f > L - 0.09)] = "taillight"
        bottom_band = (jj >= n_up) & (np.abs(jj - (n_up + n_lo / 2)) < n_lo * 0.32)
        mats[bottom_band] = "car_trim"
        mats[(st < 0) & (cen[:, 0] > L / 2)] = "car_trim"                       # the tail cap (base)
        # wheel arches: dark wells around the wheels on the body sides
        R = self.wheel_radius
        for xc, _ in self.wheel_positions()[::2]:
            d = np.hypot(cen[:, 0] - xc, cen[:, 2] - (R - 0.012))
            mats[(d < R + 0.075) & (np.abs(cen[:, 1]) > 0.55) & (mats != "car_glass")] = "car_trim"
        return V, F, mats

    def _wheel(self, xc: float, yc: float) -> list[Part]:
        R, Wd = self.wheel_radius, self.wheel_width
        zc = R - 0.012                                               # a contact patch on the road
        n = 48
        th = np.linspace(0, 2 * math.pi, n, endpoint=False)
        prof = [(-Wd / 2, 0.70), (-Wd / 2, 0.93), (-Wd / 2 + 0.02, 0.99), (0.0, 1.0), (Wd / 2 - 0.02, 0.99),
                (Wd / 2, 0.93), (Wd / 2, 0.70)]
        rings = np.stack([np.c_[xc + R * f * np.cos(th), np.full(n, yc + dy), zc + R * f * np.sin(th)]
                          for dy, f in prof])
        # loft along y: reorder so the helper sees an "x" axis -> swap axes temporarily
        Vt, Ft = _loft_rings(rings[:, :, [1, 0, 2]])
        Vt = Vt[:, [1, 0, 2]]
        Ft = Ft[:, ::-1]
        side = math.copysign(1.0, yc)
        rim_y = yc + side * (Wd / 2 + 0.004)
        rr = np.r_[np.linspace(0.08, 0.66, 3)]
        rim_rings = np.stack([np.c_[xc + R * r * np.cos(th), np.full(n, rim_y - side * 0.01 * k), zc + R * r * np.sin(th)]
                              for k, r in enumerate(rr)])
        Vr, Fr = _loft_rings(rim_rings[:, :, [1, 0, 2]])
        Vr = Vr[:, [1, 0, 2]]
        name = f"wheel_{'f' if xc < self.length / 2 else 'r'}{'l' if yc > 0 else 'r'}"
        return [Part(f"{name}_tyre", Vt, Ft, "tyre", "wheels", True), Part(f"{name}_rim", Vr, Fr[:, ::-1], "rim", "wheels", False)]

    def wheel_positions(self):
        y = self.width / 2 - self.wheel_width / 2 - 0.035
        xf = self.front_overhang
        return [(xf, y), (xf, -y), (xf + self.wheelbase, y), (xf + self.wheelbase, -y)]

    def parts(self) -> list[Part]:
        V, F, mats = self._body()
        parts = _split_by(V, F, mats, "body")
        for xc, yc in self.wheel_positions():
            parts += self._wheel(xc, yc)
        return parts

    def cfd_bodies(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        V, F, _ = self._body()
        wheels = _merge([(p.vertices, p.faces) for p in self.parts() if p.material == "tyre"])
        return {"body": (V, F), "wheels": wheels}

    def frontal_area(self) -> float:
        """Projected frontal area (body and wheels), from a fine raster of the y-z projection."""
        pts = np.vstack([p.vertices for p in self.parts() if p.aero])
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts[:, 1:])
        return float(hull.volume) * 0.93                 # the hull overestimates the rounded corners slightly


# ============================================================================ launch vehicle
@dataclass
class LaunchVehicle:
    """A two-stage launcher with a tangent-ogive fairing and four tail fins in a + configuration."""
    length: float = 42.0
    diameter: float = 3.0
    fairing_length: float = 7.5
    fairing_diameter: float = 3.6
    interstage_x: float = 15.0
    fin_root: float = 4.2
    fin_tip: float = 1.8
    fin_span: float = 2.4
    fin_sweep: float = 2.0              # leading-edge sweep distance (root LE to tip LE along x)
    fin_thickness: float = 0.22
    n_engines: int = 7
    plume_length: float = 30.0

    def _profile(self):
        Lf, Rf, R = self.fairing_length, self.fairing_diameter / 2, self.diameter / 2
        x_ogive = np.linspace(0, Lf * 0.72, 60)
        rho = (Rf ** 2 + (Lf * 0.72) ** 2) / (2 * Rf)                 # tangent ogive radius
        r_ogive = np.sqrt(np.maximum(rho ** 2 - (Lf * 0.72 - x_ogive) ** 2, 0)) + Rf - rho
        r_ogive[0] = 0.02
        x_cyl = np.linspace(Lf + 0.8, self.length - 0.3, int((self.length - Lf) / 0.25))
        x = list(x_ogive) + [Lf * 0.75, Lf] + list(x_cyl) + [self.length]
        r = list(r_ogive) + [Rf, Rf] + [R] * len(x_cyl) + [R * 0.97]
        return np.asarray(x), np.asarray(r)

    def body(self) -> tuple[np.ndarray, np.ndarray]:
        x, r = self._profile()
        return _revolve(x, r, n=72)

    def fins(self) -> list[tuple[np.ndarray, np.ndarray]]:
        R = self.diameter / 2
        root_x = self.length - self.fin_root - 0.2
        tip_x = root_x + self.fin_sweep
        return [_box_fin(root_x, self.fin_root, tip_x, self.fin_tip, R, self.fin_span, self.fin_thickness, a)
                for a in (0, 90, 180, 270)]

    def parts(self, plume: bool = True) -> list[Part]:
        V, F = self.body()
        cen = V[F].mean(1)
        mats = np.full(len(F), "rocket_white", dtype=object)
        band = (np.abs(cen[:, 0] - self.interstage_x) < 0.9)
        mats[band] = "rocket_black"
        roll = (cen[:, 0] > self.length - 6.5) & (cen[:, 0] < self.length - 0.2) & \
               (np.sin(4 * np.arctan2(cen[:, 2], cen[:, 1])) > 0) & (cen[:, 0] < self.length - 4.6)
        mats[roll] = "rocket_black"
        parts = _split_by(V, F, mats, "rocket")
        for k, (Vf, Ff) in enumerate(self.fins()):
            parts.append(Part(f"fin_{k}", Vf, Ff, "rocket_black", "fins", True))
        # engines: a ring of bells and a centre one (drawn only)
        R = self.diameter / 2
        x0 = self.length
        n_ring = max(0, self.n_engines - 1)
        centres = [(0.0, 0.0)] + [(0.62 * R * math.cos(2 * math.pi * k / n_ring), 0.62 * R * math.sin(2 * math.pi * k / n_ring))
                                  for k in range(n_ring)]
        bells = []
        for cy, cz in centres:
            xb = np.linspace(x0 - 0.1, x0 + 1.5, 12)
            rb = 0.18 + 0.32 * ((xb - xb[0]) / (xb[-1] - xb[0])) ** 0.7
            Vb, Fb = _revolve(xb, rb, n=32)
            Vb[:, 1] += cy
            Vb[:, 2] += cz
            bells.append((Vb, Fb))
        Vb, Fb = _merge(bells)
        parts.append(Part("engines", Vb, Fb, "nozzle", "engines", False))
        if plume:
            xp = np.linspace(x0 + 1.5, x0 + 1.5 + self.plume_length, 50)
            s = (xp - xp[0]) / self.plume_length
            rp = R * (0.7 + 1.5 * s ** 0.55) * (1 - 0.6 * s ** 4)
            Vp, Fp = _revolve(xp, rp, n=48)
            parts.append(Part("plume", Vp, Fp, "flame", "plume", False))
            xc = np.linspace(x0 + 1.5, x0 + 1.5 + 0.35 * self.plume_length, 30)
            sc = (xc - xc[0]) / (xc[-1] - xc[0])
            rc = R * 0.62 * (1 + 0.3 * sc) * np.sqrt(np.maximum(1 - sc ** 2, 0.02))
            Vc, Fc = _revolve(xc, rc, n=40)
            parts.append(Part("plume_core", Vc, Fc, "flame_core", "plume", False))
        return parts

    def upright(self, parts: list[Part]) -> list[Part]:
        """The parts stood on their tail (nose up, z), for launch renders."""
        L = self.length
        return [Part(p.name, np.c_[p.vertices[:, 2], p.vertices[:, 1], L - p.vertices[:, 0]], p.faces, p.material,
                     p.group, p.aero) for p in parts]

    def cfd_bodies(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {"rocket": self.body(), "fins": _merge(self.fins())}

    def ref_area(self) -> float:
        return math.pi * (self.diameter / 2) ** 2

    def barrowman(self) -> dict[str, float]:
        """Normal-force slope CN_alpha (per rad, on the body cross-section) and centre of pressure from the nose,
        by Barrowman's equations (subsonic, small angles): ogive nose CN_a = 2 at 0.466 of its length; the fairing
        shoulder (boat-tail to the stage diameter) as a conical transition; four fins with body interference."""
        Lf, Rf, R = self.fairing_length, self.fairing_diameter / 2, self.diameter / 2
        d = self.diameter
        Ln = 0.72 * Lf
        cn_nose = 2.0 * (Rf / R) ** 2
        x_nose = 0.466 * Ln
        # transition from Rf to R between Lf and Lf + 0.8 (negative for a boat-tail)
        xt, lt = Lf, 0.8
        cn_tr = 2.0 * ((R / R) ** 2 - (Rf / R) ** 2)
        x_tr = xt + lt / 3 * (1 + (1 - Rf / R) / (1 - (Rf / R) ** 2))
        # fins
        s, cr, ct = self.fin_span, self.fin_root, self.fin_tip
        xr = self.fin_sweep
        lm = math.hypot(s, xr + ct / 2 - cr / 2)                       # mid-chord line length
        cn_f = (1 + R / (s + R)) * (4 * 4 * (s / d) ** 2) / (1 + math.sqrt(1 + (2 * lm / (cr + ct)) ** 2))
        x_root = self.length - self.fin_root - 0.2
        x_f = x_root + xr * (cr + 2 * ct) / (3 * (cr + ct)) + (cr + ct - cr * ct / (cr + ct)) / 6
        cn = cn_nose + cn_tr + cn_f
        xcp = (cn_nose * x_nose + cn_tr * x_tr + cn_f * x_f) / cn
        return {"CN_alpha": cn, "x_cp": xcp, "CN_alpha_nose": cn_nose, "CN_alpha_transition": cn_tr,
                "CN_alpha_fins": cn_f, "x_cp_fins": x_f}
