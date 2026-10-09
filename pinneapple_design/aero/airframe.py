"""Parametric light aircraft in 3D: fuselage, wing (the optimized airfoil, span, taper, sweep, dihedral, twist),
tails sized by volume coefficient, propeller, landing gear. Exports glTF binary (.glb, PBR materials: Blender, Unreal,
Unity, three.js), OpenUSD (.usda, UsdPreviewSurface: Omniverse) and STL (CFD).

Axes: x aft from the nose, y to the right wing, z up; metres.
"""
from __future__ import annotations

import dataclasses
import io
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

from .geometry import REFERENCE, surfaces

MATERIALS = {
    "paint": {"color": [0.93, 0.94, 0.95], "metallic": 0.0, "roughness": 0.28, "clearcoat": 1.0},
    "stripe": {"color": [0.04, 0.22, 0.52], "metallic": 0.0, "roughness": 0.3, "clearcoat": 1.0},
    "accent": {"color": [0.75, 0.08, 0.06], "metallic": 0.0, "roughness": 0.3, "clearcoat": 1.0},
    "glass": {"color": [0.02, 0.035, 0.05], "metallic": 0.0, "roughness": 0.02, "clearcoat": 1.0},
    "nav_red": {"color": [1.0, 0.05, 0.03], "metallic": 0.0, "roughness": 0.2, "emissive": [1.0, 0.05, 0.03]},
    "nav_green": {"color": [0.05, 1.0, 0.25], "metallic": 0.0, "roughness": 0.2, "emissive": [0.05, 1.0, 0.25]},
    "prop": {"color": [0.06, 0.06, 0.07], "metallic": 0.3, "roughness": 0.45},
    "prop_tip": {"color": [0.95, 0.75, 0.05], "metallic": 0.0, "roughness": 0.4},
    "chrome": {"color": [0.92, 0.92, 0.94], "metallic": 1.0, "roughness": 0.12},
    "tire": {"color": [0.03, 0.03, 0.03], "metallic": 0.0, "roughness": 0.92},
    "strut": {"color": [0.55, 0.56, 0.58], "metallic": 0.8, "roughness": 0.35},
}


@dataclass
class Part:
    name: str
    vertices: np.ndarray            # (n, 3)
    faces: np.ndarray               # (m, 3) int, outward (counter-clockwise seen from outside)
    material: str
    group: str = "airframe"         # "propeller" parts spin about the x axis through `pivot`
    aero: bool = True               # part of the clean CFD shape


@dataclass
class Airframe:
    # fuselage
    length: float = 8.2
    width: float = 1.12
    height: float = 1.42
    # wing
    wing_area: float = 16.2
    aspect_ratio: float = 7.5
    taper: float = 0.70
    sweep_le: float = 0.0           # deg, leading edge
    dihedral: float = 1.5           # deg
    twist: float = -3.0             # deg at the tip (washout negative)
    incidence: float = 1.5          # deg at the root
    wing_x: float = 2.30            # m, root leading edge
    high_wing: bool = True
    section: np.ndarray = field(default_factory=lambda: REFERENCE["NACA 2412"].copy())
    # tails
    htail_volume: float = 0.70
    vtail_volume: float = 0.040
    htail_ar: float = 4.2
    vtail_ar: float = 1.6
    tail_arm_frac: float = 0.90     # tail quarter-chord station / fuselage length
    # extras (visual only)
    prop_diameter: float = 1.90
    gear: bool = True

    # ------------------------------------------------------------------ derived wing numbers
    @property
    def span(self) -> float:
        return math.sqrt(self.wing_area * self.aspect_ratio)

    @property
    def root_chord(self) -> float:
        return 2 * self.wing_area / (self.span * (1 + self.taper))

    @property
    def mac(self) -> float:
        l = self.taper
        return 2 / 3 * self.root_chord * (1 + l + l * l) / (1 + l)

    @property
    def mac_y(self) -> float:
        l = self.taper
        return self.span / 6 * (1 + 2 * l) / (1 + l)

    @property
    def mac_le_x(self) -> float:
        return self.wing_x + self.mac_y * math.tan(math.radians(self.sweep_le))

    def wing_z_root(self) -> float:
        zc, h = self._fus_zc(self.wing_x / self.length), self._fus_h(self.wing_x / self.length)
        return zc + 0.5 * h - 0.04 * self.root_chord if self.high_wing else zc - 0.5 * h + 0.06 * self.root_chord

    # ------------------------------------------------------------------ fuselage shape (normalised stations)
    _XI = np.array([0.0, 0.03, 0.08, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0])
    _W = np.array([0.30, 0.62, 0.86, 0.98, 1.00, 0.95, 0.74, 0.50, 0.30, 0.10])
    _H = np.array([0.34, 0.55, 0.76, 0.93, 1.00, 0.96, 0.76, 0.55, 0.38, 0.20])
    _ZC = np.array([-0.08, -0.06, -0.03, 0.0, 0.02, 0.04, 0.09, 0.15, 0.21, 0.26])

    def _fus_w(self, xi):
        return _PW(np.clip(xi, 0, 1)) * self.width

    def _fus_h(self, xi):
        return _PH(np.clip(xi, 0, 1)) * self.height

    def _fus_zc(self, xi):
        return _PZ(np.clip(xi, 0, 1)) * self.height

    # ------------------------------------------------------------------ tails
    def htail(self) -> Dict[str, float]:
        x_qc = self.tail_arm_frac * self.length
        arm = x_qc - (self.mac_le_x + 0.25 * self.mac)
        S = self.htail_volume * self.wing_area * self.mac / arm
        b = math.sqrt(S * self.htail_ar)
        lam = 0.70
        cr = 2 * S / (b * (1 + lam))
        return {"area": S, "span": b, "taper": lam, "root_chord": cr, "x_qc": x_qc, "arm": arm,
                "x_le": x_qc - 0.25 * cr, "z": self._fus_zc(x_qc / self.length) + 0.05}

    def vtail(self) -> Dict[str, float]:
        x_qc = (self.tail_arm_frac - 0.02) * self.length
        arm = x_qc - (self.mac_le_x + 0.25 * self.mac)
        S = self.vtail_volume * self.wing_area * self.span / arm
        h = math.sqrt(S * self.vtail_ar)
        lam = 0.55
        cr = 2 * S / (h * (1 + lam))
        xi = x_qc / self.length
        return {"area": S, "height": h, "taper": lam, "root_chord": cr, "x_le": x_qc - 0.25 * cr, "arm": arm,
                "z": self._fus_zc(xi) + 0.35 * self._fus_h(xi)}

    # ------------------------------------------------------------------ build
    def build(self, detail: str = "high") -> List[Part]:
        n = {"high": 1.0, "cfd": 0.8, "low": 0.5}[detail]
        parts = self._fuselage(int(110 * n), int(64 * n))
        parts += self._wing(int(70 * n), int(36 * n))
        if detail != "cfd":
            parts += self._wing_details()
        parts += self._tails(int(50 * n), int(14 * n))
        if detail != "cfd":
            parts += self._propeller()
            if self.gear:
                parts += self._gear()
        return parts

    def _fuselage(self, ns: int, nt: int) -> List[Part]:
        L = self.length
        xi = 0.5 * (1 - np.cos(np.linspace(0, math.pi, ns)))       # clustered at nose and tail
        w, h, zc = self._fus_w(xi), self._fus_h(xi), self._fus_zc(xi)
        th = np.linspace(0, 2 * math.pi, nt, endpoint=False)
        ex = 2.6                                                    # superellipse exponent (boxier cabin)
        cy = np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2 / ex)
        sz = np.sign(np.sin(th)) * np.abs(np.sin(th)) ** (2 / ex)
        V = np.empty((ns, nt, 3))
        V[..., 0] = (xi * L)[:, None]
        V[..., 1] = 0.5 * w[:, None] * cy[None]
        V[..., 2] = zc[:, None] + 0.5 * h[:, None] * sz[None]
        verts = V.reshape(-1, 3)
        F, mats = [], []
        zrel = sz                                                    # -1 .. 1 around the section
        for i in range(ns - 1):
            for j in range(nt):
                a, b = i * nt + j, i * nt + (j + 1) % nt
                c, d = (i + 1) * nt + (j + 1) % nt, (i + 1) * nt + j
                F += [(a, d, c), (a, c, b)]
                x = 0.5 * (xi[i] + xi[i + 1])
                zr = 0.5 * (zrel[j] + zrel[(j + 1) % nt])
                yr = 0.5 * (cy[j] + cy[(j + 1) % nt])
                m = "paint"
                if 0.12 < x < 0.40 and zr > 0.30 and (abs(yr) > 0.55 or x < 0.20):
                    m = "glass"                                     # windshield and side windows
                if (0.08 < x < 0.97) and (-0.12 < zr < -0.02) and abs(yr) > 0.5:
                    m = "stripe"
                mats += [m, m]
        # caps
        nose_c = len(verts); verts = np.vstack([verts, [[0.0, 0.0, zc[0]]]])
        tail_c = len(verts); verts = np.vstack([verts, [[L, 0.0, zc[-1]]]])
        for j in range(nt):
            F.append((nose_c, j, (j + 1) % nt)); mats.append("paint")
            k = (ns - 1) * nt
            F.append((tail_c, k + (j + 1) % nt, k + j)); mats.append("paint")
        F, mats = np.array(F), np.array(mats, dtype=object)
        if _signed_volume(verts, F) < 0:
            F = F[:, ::-1]
        return _split(verts, F, mats, "fuselage")

    def _section_loop(self, npts: int, shape: np.ndarray, symmetric_thick: Optional[float] = None) -> np.ndarray:
        """Closed airfoil loop in the (x, z) plane, unit chord: TE -> upper -> LE -> lower -> TE."""
        b = np.linspace(0, math.pi, npts)
        x = 0.5 * (1 - np.cos(b))
        if symmetric_thick is not None:
            t = symmetric_thick
            yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
            yu, yl = yt, -yt
        else:
            yu, yl = surfaces(shape, x)
        up = np.c_[x[::-1], yu[::-1]]
        lo = np.c_[x[1:], yl[1:]]
        return np.vstack([up, lo])                                  # first and last point both at the TE

    def _lifting_surface(self, name, loop, stations, mat, mirror=True) -> List[Part]:
        """Loft a section loop through stations (x_le, y, z, chord, twist_deg); caps at both ends."""
        V, F = [], []
        m = len(loop)
        for (xle, y, z, c, tw) in stations:
            t = math.radians(tw)
            px = (loop[:, 0] - 0.25) * c
            pz = loop[:, 1] * c
            rx = xle + 0.25 * c + px * math.cos(t) + pz * math.sin(t)
            rz = -px * math.sin(t) + pz * math.cos(t)
            V.append(np.c_[rx, np.full(m, y), z + rz])
        V = np.vstack(V)
        ns = len(stations)
        for i in range(ns - 1):
            for j in range(m - 1):
                a, b, c_, d = i * m + j, i * m + j + 1, (i + 1) * m + j + 1, (i + 1) * m + j
                F += [(a, b, c_), (a, c_, d)]
        # caps (fan around the section centroid)
        for i, flip in ((0, True), (ns - 1, False)):
            cidx = len(V)
            V = np.vstack([V, V[i * m:(i + 1) * m].mean(0)])
            for j in range(m - 1):
                tri = (cidx, i * m + j, i * m + j + 1)
                F.append(tri[::-1] if flip else tri)
        F = np.array(F)
        # orient outward: the loop runs TE->upper->LE->lower, along +y that is outward-consistent; check volume sign
        if _signed_volume(V, F) < 0:
            F = F[:, ::-1]
        parts = [Part(name + ("_right" if mirror else ""), V, F, mat)]
        if mirror:
            Vm = V * [1, -1, 1]
            parts.append(Part(name + "_left", Vm, F[:, ::-1].copy(), mat))
        return parts

    def _wing(self, npts: int, nspan: int) -> List[Part]:
        b2, cr, lam = self.span / 2, self.root_chord, self.taper
        loop = self._section_loop(npts, self.section)
        eta = np.sin(np.linspace(0, math.pi / 2, nspan))            # denser at the tip
        z0 = self.wing_z_root()
        st = [(self.wing_x + e * b2 * math.tan(math.radians(self.sweep_le)), e * b2,
               z0 + e * b2 * math.tan(math.radians(self.dihedral)), cr * (1 - (1 - lam) * e),
               self.incidence + self.twist * e) for e in eta]
        return self._lifting_surface("wing", loop, st, "paint")

    def _wing_point(self, eta: float, xc: float) -> np.ndarray:
        """A point on the wing reference plane at span fraction eta and chord fraction xc."""
        b2 = self.span / 2
        c = self.root_chord * (1 - (1 - self.taper) * eta)
        return np.array([self.wing_x + eta * b2 * math.tan(math.radians(self.sweep_le)) + xc * c, eta * b2,
                         self.wing_z_root() + eta * b2 * math.tan(math.radians(self.dihedral))])

    def _wing_details(self) -> List[Part]:
        out = []
        for side, mat in ((1, "nav_green"), (-1, "nav_red")):     # green on the right wing, red on the left
            p = self._wing_point(1.0, 0.12) * [1, side, 1] + [0, side * 0.02, 0]
            out.append(_ellipsoid(f"navlight_{'r' if side > 0 else 'l'}", p, (0.07, 0.03, 0.03), mat))
            if self.high_wing:                                      # lift struts, fuselage bottom side -> 45 % span
                xi = (self.wing_x + 0.3 * self.root_chord) / self.length
                root = np.array([self.wing_x + 0.3 * self.root_chord, side * 0.45 * float(self._fus_w(xi)),
                                 float(self._fus_zc(xi) - 0.30 * self._fus_h(xi))])
                tip = self._wing_point(0.45, 0.30) * [1, side, 1] - [0, 0, 0.06]
                out.append(_tube(f"wingstrut_{'r' if side > 0 else 'l'}", root, tip, 0.045, "paint"))
        for p in out:
            p.aero = False
        return out

    def _tails(self, npts: int, nspan: int) -> List[Part]:
        ht, vt = self.htail(), self.vtail()
        loop = self._section_loop(npts, None, symmetric_thick=0.10)
        eta = np.linspace(0, 1, nspan)
        hs = [(ht["x_le"] + e * ht["span"] / 2 * math.tan(math.radians(8)), e * ht["span"] / 2, ht["z"],
               ht["root_chord"] * (1 - (1 - ht["taper"]) * e), 0.0) for e in eta]
        parts = self._lifting_surface("htail", loop, hs, "paint")
        vs = [(vt["x_le"] + e * vt["height"] * math.tan(math.radians(35)), 0.0, vt["z"] + e * vt["height"],
               vt["root_chord"] * (1 - (1 - vt["taper"]) * e), 0.0) for e in eta]
        # vertical: section in the x-y plane, stacked along z
        V, F = [], []
        m = len(loop)
        for (xle, _, z, c, _) in vs:
            V.append(np.c_[xle + loop[:, 0] * c, loop[:, 1] * c, np.full(m, z)])
        V = np.vstack(V)
        for i in range(len(vs) - 1):
            for j in range(m - 1):
                a, b, c_, d = i * m + j, i * m + j + 1, (i + 1) * m + j + 1, (i + 1) * m + j
                F += [(a, b, c_), (a, c_, d)]
        for i, flip in ((0, True), (len(vs) - 1, False)):
            cidx = len(V)
            V = np.vstack([V, V[i * m:(i + 1) * m].mean(0)])
            for j in range(m - 1):
                tri = (cidx, i * m + j, i * m + j + 1)
                F.append(tri[::-1] if flip else tri)
        F = np.array(F)
        if _signed_volume(V, F) < 0:
            F = F[:, ::-1]
        mats = np.array(["paint"] * len(F), dtype=object)
        top = V[F].mean(1)[:, 2] > vt["z"] + 0.62 * vt["height"]
        mats[top] = "accent"
        parts += _split(V, F, mats, "vtail")
        return parts

    def _propeller(self) -> List[Part]:
        zc = float(self._fus_zc(0.0))
        R = self.prop_diameter / 2
        parts = []
        # spinner: paraboloid of revolution from x = -0.42 to 0.02
        nx, nt = 24, 40
        xs = np.linspace(-0.42, 0.02, nx)
        r = 0.19 * np.sqrt(np.clip((xs + 0.42) / 0.44, 0, 1))
        th = np.linspace(0, 2 * math.pi, nt, endpoint=False)
        V = np.c_[np.repeat(xs, nt), np.outer(r, np.cos(th)).ravel(), zc + np.outer(r, np.sin(th)).ravel()]
        F = []
        for i in range(nx - 1):
            for j in range(nt):
                a, b = i * nt + j, i * nt + (j + 1) % nt
                c, d = (i + 1) * nt + (j + 1) % nt, (i + 1) * nt + j
                F += [(a, b, c), (a, c, d)]
        cap = len(V); V = np.vstack([V, [[0.02, 0, zc]]])
        k = (nx - 1) * nt
        F += [(cap, k + (j + 1) % nt, k + j) for j in range(nt)]
        F = np.array(F)
        if _signed_volume(V, F) < 0:
            F = F[:, ::-1]
        parts.append(Part("spinner", V, F, "chrome", group="propeller", aero=False))
        # two blades: twisted thin sections from r = 0.15 to R
        rr = np.linspace(0.15, R, 22)
        loop = self._section_loop(24, None, symmetric_thick=0.10)
        for k, ang in enumerate((0.0, math.pi)):
            V, F, mats = [], [], []
            for r_ in rr:
                e = (r_ - 0.15) / (R - 0.15)
                c = 0.17 * (1 - 0.55 * e) * (0.6 + 1.6 * e * (1 - e))
                pitch = math.radians(48 * (1 - e) + 14 * e)
                px, pz = (loop[:, 0] - 0.35) * c, loop[:, 1] * c
                # blade frame: radial along local axis, chord in the rotation plane, pitched
                xb = px * math.sin(pitch) + pz * math.cos(pitch)
                tb = px * math.cos(pitch) - pz * math.sin(pitch)
                y = r_ * math.cos(ang) - tb * math.sin(ang)
                z = zc + r_ * math.sin(ang) + tb * math.cos(ang)
                V.append(np.c_[-0.20 + xb, y, z])
            V = np.vstack(V)
            m = len(loop)
            for i in range(len(rr) - 1):
                for j in range(m - 1):
                    a, b, c_, d = i * m + j, i * m + j + 1, (i + 1) * m + j + 1, (i + 1) * m + j
                    F += [(a, b, c_), (a, c_, d)]
                    mats += ["prop_tip" if i >= len(rr) - 3 else "prop"] * 2
            for i, flip in ((0, True), (len(rr) - 1, False)):
                cidx = len(V)
                V = np.vstack([V, V[i * m:(i + 1) * m].mean(0)])
                for j in range(m - 1):
                    tri = (cidx, i * m + j, i * m + j + 1)
                    F.append(tri[::-1] if flip else tri)
                    mats.append("prop_tip" if i else "prop")
            F = np.array(F)
            if _signed_volume(V, F) < 0:
                F = F[:, ::-1]
            for p in _split(V, F, np.array(mats, dtype=object), f"blade{k}"):
                p.group, p.aero = "propeller", False
                parts.append(p)
        return parts

    def _gear(self) -> List[Part]:
        xi_m = (self.wing_x + 0.55 * self.root_chord) / self.length
        z_bot = float(self._fus_zc(xi_m) - 0.5 * self._fus_h(xi_m))
        ground = z_bot - 0.78
        parts = []
        for side in (1, -1):
            c = np.array([self.wing_x + 0.55 * self.root_chord + 0.15, side * 1.25, ground + 0.22])
            parts.append(_ellipsoid(f"wheelpant_{'r' if side > 0 else 'l'}", c, (0.42, 0.11, 0.20), "paint"))
            parts.append(_tube(f"strut_{'r' if side > 0 else 'l'}", c + [0, -side * 0.08, 0.15],
                               np.array([c[0] - 0.05, side * 0.45, z_bot + 0.05]), 0.035, "strut"))
        cn = np.array([0.75, 0.0, ground + 0.20])
        parts.append(_ellipsoid("wheelpant_nose", cn, (0.34, 0.09, 0.18), "paint"))
        parts.append(_tube("strut_nose", cn + [0, 0, 0.12], np.array([0.62, 0.0, float(self._fus_zc(0.08)) - 0.3]), 0.035, "strut"))
        for p in parts:
            p.aero = False
        return parts

    def ground_z(self) -> float:
        xi_m = (self.wing_x + 0.55 * self.root_chord) / self.length
        return float(self._fus_zc(xi_m) - 0.5 * self._fus_h(xi_m)) - 0.78

    def summary(self) -> Dict[str, float]:
        ht, vt = self.htail(), self.vtail()
        return {"span": self.span, "root_chord": self.root_chord, "tip_chord": self.root_chord * self.taper,
                "mac": self.mac, "mac_le_x": self.mac_le_x, "htail_area": ht["area"], "htail_span": ht["span"],
                "htail_arm": ht["arm"], "vtail_area": vt["area"], "vtail_height": vt["height"]}


_PW = PchipInterpolator(Airframe._XI, Airframe._W)
_PH = PchipInterpolator(Airframe._XI, Airframe._H)
_PZ = PchipInterpolator(Airframe._XI, Airframe._ZC)


# ---------------------------------------------------------------------- helpers
def _signed_volume(V, F) -> float:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def _split(V, F, mats, name) -> List[Part]:
    out = []
    mats = np.asarray(mats, dtype=object)
    for m in dict.fromkeys(mats.tolist()):
        sel = F[mats == m]
        used, inv = np.unique(sel, return_inverse=True)
        out.append(Part(f"{name}" if m in ("paint",) else f"{name}_{m}", V[used], inv.reshape(-1, 3), m))
    return out


def _ellipsoid(name, c, r, mat, nu=28, nv=16) -> Part:
    u = np.linspace(0, 2 * math.pi, nu, endpoint=False)
    v = np.linspace(-math.pi / 2, math.pi / 2, nv)
    V = [np.array([c[0], c[1], c[2] - r[2]])]
    for vv in v[1:-1]:
        for uu in u:
            V.append(c + np.array([r[0] * math.cos(vv) * math.cos(uu), r[1] * math.cos(vv) * math.sin(uu), r[2] * math.sin(vv)]))
    V.append(np.array([c[0], c[1], c[2] + r[2]]))
    V = np.array(V)
    F = []
    rings = nv - 2
    for j in range(nu):
        F.append((0, 1 + (j + 1) % nu, 1 + j))
    for i in range(rings - 1):
        for j in range(nu):
            a, b = 1 + i * nu + j, 1 + i * nu + (j + 1) % nu
            c_, d = 1 + (i + 1) * nu + (j + 1) % nu, 1 + (i + 1) * nu + j
            F += [(a, b, c_), (a, c_, d)]
    top = len(V) - 1
    for j in range(nu):
        F.append((top, 1 + (rings - 1) * nu + j, 1 + (rings - 1) * nu + (j + 1) % nu))
    F = np.array(F)
    if _signed_volume(V, F) < 0:
        F = F[:, ::-1]
    return Part(name, V, F, mat)


def _tube(name, p0, p1, r, mat, nt=12) -> Part:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    ax /= np.linalg.norm(ax)
    u = np.cross(ax, [0, 0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    th = np.linspace(0, 2 * math.pi, nt, endpoint=False)
    ring = np.outer(np.cos(th), u) * r + np.outer(np.sin(th), w) * r
    V = np.vstack([p0 + ring, p1 + ring, p0, p1])
    F = []
    for j in range(nt):
        a, b, c, d = j, (j + 1) % nt, nt + (j + 1) % nt, nt + j
        F += [(a, b, c), (a, c, d), (2 * nt, b, a), (2 * nt + 1, d, c)]
    F = np.array(F)
    if _signed_volume(V, F) < 0:
        F = F[:, ::-1]
    return Part(name, V, F, mat)


def vertex_normals(V, F) -> np.ndarray:
    n = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------- exports
# The writers are the library's generic ones (pinneapple_tools.visualization.studio); these keep the aircraft axes
# (x aft, y right, z up), the aircraft materials and the spinning propeller group.
def _surfaces(parts: List[Part], fields: Dict[str, Dict[str, np.ndarray]]):
    from pinneapple_tools.visualization.studio.scene import Surface
    out = []
    for p in parts:
        s = Surface(p.name, p.vertices, p.faces, p.material, group=p.group)
        for an, per_part in fields.items():
            if p.name in per_part:
                s.fields[an.lstrip("_")] = np.asarray(per_part[p.name], np.float32)
        out.append(s)
    return out


def to_glb(parts: List[Part], scalars: Optional[Dict[str, np.ndarray]] = None,
           fields: Optional[Dict[str, Dict[str, np.ndarray]]] = None) -> bytes:
    """glTF 2.0 binary with PBR materials (clearcoat, transmission). Axes converted to glTF's y-up.
    ``scalars``: optional per-part vertex values stored as _CP attribute (e.g. surface pressure).
    ``fields``: more per-vertex values, {attribute name (e.g. "_CF"): {part name: values}}."""
    from pinneapple_tools.visualization.studio.scene import AXES, write_glb
    fields = dict(fields or {})
    if scalars:
        fields["_CP"] = scalars
    prop = [i for i, p in enumerate(parts) if p.group == "propeller"]
    return write_glb(_surfaces(parts, fields), AXES["aircraft"], MATERIALS,
                     generator="PINNeAPPle pinneapple_design.aero.airframe", groups={"propeller": prop} if prop else None)


def to_usda(parts: List[Part], title: str = "aircraft") -> str:
    """OpenUSD text: one Mesh per part, UsdPreviewSurface materials; z up, metres (Omniverse, usdview)."""
    from pinneapple_tools.visualization.studio.scene import write_usda
    return write_usda(_surfaces(parts, {}), MATERIALS, f"{title}: aircraft design", root="Aircraft")


def to_stl(parts: List[Part], solid: str = "aircraft") -> bytes:
    """Binary STL of the given parts (one closed surface per part)."""
    from pinneapple_tools.visualization.studio.scene import write_stl
    return write_stl(parts, solid)
