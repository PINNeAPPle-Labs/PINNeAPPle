"""Parametric single-aisle airliner (A320 / 737 class) in 3D: circular fuselage with cabin and cockpit windows and a
livery, low swept wing with dihedral, washout and sharklets, two under-wing turbofans on pylons, swept tails sized by
volume coefficient. Same interface as Airframe (vortex lattice, exports), so the whole pipeline applies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.interpolate import PchipInterpolator

from .airframe import MATERIALS, Airframe, Part, _signed_volume, _split
from .geometry import REFERENCE

MATERIALS.update({
    "window": {"color": [0.03, 0.04, 0.06], "metallic": 0.0, "roughness": 0.05, "clearcoat": 1.0},
    "belly": {"color": [0.62, 0.64, 0.67], "metallic": 0.1, "roughness": 0.35, "clearcoat": 0.6},
    "livery": {"color": [0.02, 0.18, 0.45], "metallic": 0.0, "roughness": 0.3, "clearcoat": 1.0},
    "livery2": {"color": [0.98, 0.72, 0.05], "metallic": 0.0, "roughness": 0.3, "clearcoat": 1.0},
    "nacelle": {"color": [0.86, 0.87, 0.89], "metallic": 0.2, "roughness": 0.25, "clearcoat": 1.0},
    "fan": {"color": [0.05, 0.05, 0.06], "metallic": 0.7, "roughness": 0.35},
    "exhaust": {"color": [0.35, 0.33, 0.31], "metallic": 0.9, "roughness": 0.45},
    "wingmetal": {"color": [0.72, 0.74, 0.77], "metallic": 0.05, "roughness": 0.5, "clearcoat": 0.2},
})


def sweep_le_from_qc(sweep_qc: float, ar: float, taper: float) -> float:
    return math.degrees(math.atan(math.tan(math.radians(sweep_qc)) + (1 - taper) / (ar * (1 + taper))))


@dataclass
class Airliner(Airframe):
    length: float = 37.57
    width: float = 3.95
    height: float = 4.14
    wing_area: float = 122.6
    aspect_ratio: float = 9.5
    taper: float = 0.24
    sweep_le: float = 28.6
    dihedral: float = 5.0
    twist: float = -4.0
    incidence: float = 2.0
    wing_x: float = 12.3
    high_wing: bool = False
    section: np.ndarray = field(default_factory=lambda: REFERENCE["NACA 2412"].copy())
    htail_volume: float = 1.00
    vtail_volume: float = 0.080
    htail_ar: float = 5.0
    vtail_ar: float = 1.6
    tail_arm_frac: float = 0.915
    htail_sweep: float = 32.0
    vtail_sweep: float = 38.0
    engine_station: float = 0.34        # fraction of the half-span
    fan_diameter: float = 1.95
    sharklets: bool = True
    gear: bool = False                  # in flight

    _XI = np.array([0.0, 0.01, 0.03, 0.06, 0.10, 0.14, 0.70, 0.80, 0.88, 0.95, 1.0])
    _W = np.array([0.10, 0.42, 0.70, 0.88, 0.97, 1.00, 1.00, 0.86, 0.62, 0.36, 0.14])
    _H = np.array([0.10, 0.45, 0.72, 0.89, 0.97, 1.00, 1.00, 0.88, 0.66, 0.42, 0.20])
    _ZC = np.array([-0.12, -0.10, -0.07, -0.04, -0.01, 0.0, 0.0, 0.08, 0.17, 0.25, 0.30])

    def _fus_w(self, xi):
        return _AW(np.clip(xi, 0, 1)) * self.width

    def _fus_h(self, xi):
        return _AH(np.clip(xi, 0, 1)) * self.height

    def _fus_zc(self, xi):
        return _AZ(np.clip(xi, 0, 1)) * self.height

    def wing_z_root(self) -> float:
        xi = self.wing_x / self.length
        return float(self._fus_zc(xi) - 0.30 * self._fus_h(xi))

    def htail(self):
        x_qc = self.tail_arm_frac * self.length
        arm = x_qc - (self.mac_le_x + 0.25 * self.mac)
        S = self.htail_volume * self.wing_area * self.mac / arm
        b = math.sqrt(S * self.htail_ar)
        lam = 0.32
        cr = 2 * S / (b * (1 + lam))
        xi = x_qc / self.length
        return {"area": S, "span": b, "taper": lam, "root_chord": cr, "x_qc": x_qc, "arm": arm,
                "x_le": x_qc - 0.25 * cr,
                "z": float(self._fus_zc(xi)) + 0.05 * self.height, "sweep_le": self.htail_sweep, "dihedral": 6.0}

    def vtail(self):
        x_qc = (self.tail_arm_frac - 0.015) * self.length
        arm = x_qc - (self.mac_le_x + 0.25 * self.mac)
        S = self.vtail_volume * self.wing_area * self.span / arm
        h = math.sqrt(S * self.vtail_ar)
        lam = 0.35
        cr = 2 * S / (h * (1 + lam))
        xi = (x_qc - 0.25 * cr) / self.length
        return {"area": S, "height": h, "taper": lam, "root_chord": cr, "x_le": x_qc - 0.25 * cr, "arm": arm,
                "z": float(self._fus_zc(xi) + 0.38 * self._fus_h(xi))}

    def ground_z(self) -> float:
        return float(self._fus_zc(0.4) - 0.5 * self.height) - 1.7

    # ------------------------------------------------------------------ build
    def build(self, detail: str = "high") -> List[Part]:
        n = {"high": 1.0, "cfd": 0.6, "low": 0.45}[detail]
        parts = self._airliner_fuselage(int(440 * n), int(96 * n), windows=detail != "cfd")
        parts += self._wing(int(80 * n), int(40 * n))
        parts += self._airliner_tails(int(50 * n), int(16 * n))
        if detail != "cfd":
            parts += self._engines()
            if self.sharklets:
                parts += self._sharklets()
        for p in parts:
            if p.name.startswith("wing") and p.material == "paint":
                p.material = "wingmetal"
        return parts

    def _airliner_fuselage(self, ns, nt, windows=True) -> List[Part]:
        L = self.length
        u = np.linspace(0, 1, ns)
        xi = np.where(u < 0.25, 0.14 * (1 - np.cos(np.pi * u / 0.5)), u)          # dense at the nose
        xi = np.unique(np.r_[0.14 * 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, int(ns * 0.18)))),
                             np.linspace(0.14, 0.70, int(ns * 0.62)), 0.70 + 0.30 * np.linspace(0, 1, int(ns * 0.2)) ** 1.0])
        w, h, zc = self._fus_w(xi), self._fus_h(xi), self._fus_zc(xi)
        th = np.linspace(0, 2 * math.pi, nt, endpoint=False)
        cy, sz = np.cos(th), np.sin(th)
        ns = len(xi)
        V = np.empty((ns, nt, 3))
        V[..., 0] = (xi * L)[:, None]
        V[..., 1] = 0.5 * w[:, None] * cy[None]
        V[..., 2] = zc[:, None] + 0.5 * h[:, None] * sz[None]
        verts = V.reshape(-1, 3)
        F, mats = [], []
        pitch, wwin, z0, z1 = 0.533, 0.24, 0.18, 0.36
        for i in range(ns - 1):
            x = 0.5 * (xi[i] + xi[i + 1])
            xm = x * L
            for j in range(nt):
                a, b = i * nt + j, i * nt + (j + 1) % nt
                c, d = (i + 1) * nt + (j + 1) % nt, (i + 1) * nt + j
                F += [(a, d, c), (a, c, b)]
                zr = 0.5 * (sz[j] + sz[(j + 1) % nt])
                yr = 0.5 * (cy[j] + cy[(j + 1) % nt])
                m = "paint"
                if zr < -0.62:
                    m = "belly"
                elif -0.30 < zr < -0.12 and 0.04 < x < 0.86:
                    m = "livery"
                elif -0.40 < zr < -0.30 and 0.06 < x < 0.80:
                    m = "livery2"
                if windows:
                    if 0.17 < x < 0.69 and z0 < zr < z1 and abs(yr) > 0.5 and ((xm - 6.6) % pitch) < wwin:
                        m = "window"                                    # cabin windows, 21 in pitch
                    if 0.028 < x < 0.062 and 0.28 < zr < 0.62 and abs(yr) > 0.12:
                        m = "window"                                    # cockpit
                if x > 0.93:
                    m = "belly"                                         # APU / tail cone
                mats += [m, m]
        nose = len(verts); verts = np.vstack([verts, [[0.0, 0.0, zc[0]]]])
        tail = len(verts); verts = np.vstack([verts, [[L, 0.0, zc[-1]]]])
        for j in range(nt):
            F.append((nose, j, (j + 1) % nt)); mats.append("paint")
            k = (ns - 1) * nt
            F.append((tail, k + (j + 1) % nt, k + j)); mats.append("belly")
        F, mats = np.array(F), np.array(mats, dtype=object)
        if _signed_volume(verts, F) < 0:
            F = F[:, ::-1]
        return _split(verts, F, mats, "fuselage")

    def _airliner_tails(self, npts, nspan) -> List[Part]:
        ht, vt = self.htail(), self.vtail()
        loop = self._section_loop(npts, None, symmetric_thick=0.10)
        eta = np.linspace(0, 1, nspan)
        b2 = ht["span"] / 2
        hs = [(ht["x_le"] + e * b2 * math.tan(math.radians(ht["sweep_le"])), e * b2,
               ht["z"] + e * b2 * math.tan(math.radians(ht["dihedral"])), ht["root_chord"] * (1 - (1 - ht["taper"]) * e), 0.0)
              for e in eta]
        parts = self._lifting_surface("htail", loop, hs, "paint")
        V, F = [], []
        m = len(loop)
        for e in eta:
            c = vt["root_chord"] * (1 - (1 - vt["taper"]) * e)
            xle = vt["x_le"] + e * vt["height"] * math.tan(math.radians(self.vtail_sweep))
            V.append(np.c_[xle + loop[:, 0] * c, loop[:, 1] * c, np.full(m, vt["z"] + e * vt["height"])])
        V = np.vstack(V)
        for i in range(nspan - 1):
            for j in range(m - 1):
                a, b, c_, d = i * m + j, i * m + j + 1, (i + 1) * m + j + 1, (i + 1) * m + j
                F += [(a, b, c_), (a, c_, d)]
        for i, flip in ((0, True), (nspan - 1, False)):
            cidx = len(V)
            V = np.vstack([V, V[i * m:(i + 1) * m].mean(0)])
            for j in range(m - 1):
                tri = (cidx, i * m + j, i * m + j + 1)
                F.append(tri[::-1] if flip else tri)
        F = np.array(F)
        if _signed_volume(V, F) < 0:
            F = F[:, ::-1]
        cz = V[F].mean(1)[:, 2]
        mats = np.where(cz > vt["z"] + 0.18 * vt["height"], "livery", "paint").astype(object)
        mats[(cz > vt["z"] + 0.18 * vt["height"]) & (cz < vt["z"] + 0.30 * vt["height"])] = "livery2"
        parts += _split(V, F, mats, "vtail")
        return parts

    def _revolve(self, name, xs, rs, axis_origin, mat_fn, nt=48, close_front=True, close_back=True):
        """Body of revolution about the x axis through axis_origin; mat_fn(i_station) gives the material."""
        th = np.linspace(0, 2 * math.pi, nt, endpoint=False)
        o = np.asarray(axis_origin, float)
        V = np.c_[np.repeat(xs, nt) + o[0], (np.outer(rs, np.cos(th))).ravel() + o[1], (np.outer(rs, np.sin(th))).ravel() + o[2]]
        F, mats = [], []
        n = len(xs)
        for i in range(n - 1):
            for j in range(nt):
                a, b = i * nt + j, i * nt + (j + 1) % nt
                c, d = (i + 1) * nt + (j + 1) % nt, (i + 1) * nt + j
                F += [(a, b, c), (a, c, d)]
                mats += [mat_fn(i)] * 2
        if close_front:
            cf = len(V); V = np.vstack([V, [[xs[0] + o[0], o[1], o[2]]]])
            F += [(cf, (j + 1) % nt, j) for j in range(nt)]; mats += ["fan"] * nt
        if close_back:
            cb = len(V); V = np.vstack([V, [[xs[-1] + o[0], o[1], o[2]]]])
            k = (n - 1) * nt
            F += [(cb, k + j, k + (j + 1) % nt) for j in range(nt)]; mats += ["exhaust"] * nt
        F = np.array(F)
        if _signed_volume(V, F) < 0:
            F = F[:, ::-1]
        return _split(V, F, np.array(mats, dtype=object), name)

    def _engines(self) -> List[Part]:
        out = []
        R = self.fan_diameter / 2 * 1.08
        Ln = 4.3
        for side in (1, -1):
            eta = self.engine_station
            wp = self._wing_point(eta, 0.0)
            ctr = np.array([wp[0] - 0.55 * Ln, side * wp[1], wp[2] - 0.62 * R * 2 + 0.25])
            xs = np.r_[np.linspace(0, 0.35, 6) ** 1.0, np.linspace(0.5, Ln, 14)]
            rs = R * np.r_[0.88 + 0.12 * np.sin(np.linspace(0, math.pi / 2, 6)), 1.0 - 0.25 * ((np.linspace(0.5, Ln, 14) - 0.5) / (Ln - 0.5)) ** 1.6]
            tag = "r" if side > 0 else "l"
            out += self._revolve(f"nacelle_{tag}", xs, rs, ctr, lambda i: "nacelle")
            # exhaust plug
            xs2 = np.linspace(Ln - 0.2, Ln + 1.2, 10)
            rs2 = 0.42 * R * (1 - np.linspace(0, 1, 10) ** 1.3) + 0.01
            out += self._revolve(f"plug_{tag}", xs2, rs2, ctr, lambda i: "exhaust", nt=32, close_front=False)
            # pylon: a thin swept slab from the nacelle top to the wing
            top = ctr + [0.8, 0, R * 0.92]
            p0 = np.array([ctr[0] + 0.6, ctr[1], ctr[2] + R * 0.95])
            p1 = np.array([wp[0] + 1.6, ctr[1], wp[2] - 0.05])
            p2 = np.array([ctr[0] + Ln + 0.4, ctr[1], ctr[2] + R * 0.55])
            p3 = np.array([wp[0] + 3.4, ctr[1], wp[2] - 0.05])
            t = 0.22
            P = np.array([p0, p1, p3, p2])
            V = np.vstack([P + [0, -t / 2, 0], P + [0, t / 2, 0]])
            F = np.array([(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
                          (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)])
            if _signed_volume(V, F) < 0:
                F = F[:, ::-1]
            out.append(Part(f"pylon_{tag}", V, F, "paint"))
        for p in out:
            p.aero = False
        return out

    def _sharklets(self) -> List[Part]:
        b2 = self.span / 2
        tip = self._wing_point(1.0, 0.0)
        ct = self.root_chord * self.taper
        loop = self._section_loop(40, None, symmetric_thick=0.09)
        hgt, cant = 2.43, math.radians(15)
        out = []
        for side in (1, -1):
            V, F = [], []
            n = 10
            m = len(loop)
            for k in range(n):
                e = k / (n - 1)
                yy = tip[1] + e * hgt * math.sin(cant)                 # canted outward
                zz = tip[2] + e * hgt * math.cos(cant)
                c = ct * (1 - 0.68 * e)
                xle = tip[0] + e * 1.6                                 # swept back
                # section chord along x, thickness across (y), stacked up the sharklet
                V.append(np.c_[xle + loop[:, 0] * c, side * (yy + loop[:, 1] * c), np.full(m, zz)])
            V = np.vstack(V)
            for i in range(n - 1):
                for j in range(m - 1):
                    a, b, c_, d = i * m + j, i * m + j + 1, (i + 1) * m + j + 1, (i + 1) * m + j
                    F += [(a, b, c_), (a, c_, d)]
            cidx = len(V)
            V = np.vstack([V, V[(n - 1) * m:].mean(0)])
            for j in range(m - 1):
                F.append((cidx, (n - 1) * m + j, (n - 1) * m + j + 1))
            F = np.array(F)
            if _signed_volume(V, F) < 0:
                F = F[:, ::-1]
            p = Part(f"sharklet_{'r' if side > 0 else 'l'}", V, F, "livery")
            p.aero = False
            out.append(p)
        return out

    def summary(self):
        s = super().summary()
        s.update(sweep_qc=math.degrees(math.atan(math.tan(math.radians(self.sweep_le)) - (1 - self.taper) / (self.aspect_ratio * (1 + self.taper)))))
        return s


_AW = PchipInterpolator(Airliner._XI, Airliner._W)
_AH = PchipInterpolator(Airliner._XI, Airliner._H)
_AZ = PchipInterpolator(Airliner._XI, Airliner._ZC)
