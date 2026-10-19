"""Quasi-steady aerodynamics of a whole aircraft for flight simulation, from the vortex lattice of
:mod:`pinneapple_design.aero.vlm`.

At every evaluation the lattice of wing + horizontal tail is solved with the local relative air velocity at each
control point (flight velocity, rotation p, q, r and the wind / gust at that point), so the lift, pitching and rolling
moments, the pitch and roll damping, the downwash at the tail, the dihedral effect and the response to a gust that
differs from one wing to the other all come from the same linear potential-flow model. The influence matrix depends
only on the geometry: it is LU-factorised once. Control surfaces rotate the normals of their panels (thin-airfoil
flap effectiveness tau). Forces: Kutta-Joukowski on the bound legs with the local velocity.

Added by component build-up (the lattice has neither): zero-lift drag from the parasite-drag build-up passed in
(``cd0``), induced drag K CL^2 with K from the lattice's Trefftz-plane drag (and McCormick's ground-effect factor), the
vertical tail as a lifting line at its aerodynamic centre (Helmbold lift slope with the effective aspect ratio
1.55 AR for the end-plate effect of fuselage and tailplane; rudder through tau), and the fuselage Munk moments in
pitch and yaw (Raymer, Aircraft Design, 6th ed., eqs. 16.25 and 16.47).

Axes: the lattice uses the airframe axes A (x aft, y right, z up), the dynamics the body axes B (x forward, y right,
z down); B = T (A - cg) with T = diag(-1, 1, -1), a proper rotation, so forces and moments map with T as well.

Limits (quasi-steady): no stall (the angle of attack at the section cl max is reported by the caller), no unsteady
lift build-up (Wagner, Kussner), no downwash lag (Cm_alpha_dot), no aeroelasticity, ground effect on induced drag only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from pinneapple_design.aero.vlm import build_lattice, horseshoe_induced
from pinneapple_design.aero.vlm import solve as vlm_solve

T_AB = np.diag([-1.0, 1.0, -1.0])


def flap_effectiveness(cf_c: float, correction: float = 0.85) -> float:
    """Thin-airfoil flap effectiveness d(alpha_0)/d(delta) for a plain flap of chord ratio cf/c:
    tau = 1 - (theta_f - sin theta_f) / pi, cos theta_f = 2 cf/c - 1, times an empirical ``correction`` for real
    viscous flow at moderate deflections (an assumption; 0.7-0.9 is the usual range)."""
    th = math.acos(2 * cf_c - 1)
    return correction * (1 - (th - math.sin(th)) / math.pi)


def helmbold_lift_slope(aspect_ratio: float, sweep_half_chord_deg: float = 0.0, eta: float = 0.95) -> float:
    """Lift-curve slope [/rad] of a finite wing (Helmbold / DATCOM form, Raymer eq. 12.6, subsonic, M ~ 0)."""
    A = aspect_ratio
    t = math.tan(math.radians(sweep_half_chord_deg))
    return 2 * math.pi * A / (2 + math.sqrt(4 + A * A * (1 + t * t) / (eta * eta)))


@dataclass
class ControlSurface:
    """Panels of the lattice moved by one control: ``surface`` 0 wing / 1 tail, spanwise range |eta| in
    [eta_in, eta_out] of that surface's half-span, ``sign`` per side (+1 symmetric, -1 antisymmetric: right side
    gets +delta), chord ratio for the effectiveness. A positive deflection is trailing edge down on the right side."""
    name: str
    surface: int
    eta_in: float
    eta_out: float
    cf_c: float = 0.25
    antisymmetric: bool = False
    tau: float | None = None

    def effectiveness(self) -> float:
        return self.tau if self.tau is not None else flap_effectiveness(self.cf_c)


@dataclass
class Fin:
    """Vertical tail as one lifting line: area [m^2], height [m], aerodynamic centre in airframe axes, rudder chord
    ratio, quarter-chord sweep [deg]."""
    area: float
    height: float
    ac: np.ndarray
    rudder_cf_c: float = 0.30
    sweep_deg: float = 35.0
    stall_deg: float = 20.0

    @property
    def lift_slope(self) -> float:
        return helmbold_lift_slope(1.55 * self.height ** 2 / self.area, self.sweep_deg)


@dataclass
class AeroModel:
    """Built by :meth:`from_airframe`; :meth:`loads` gives forces and moments about the CG in body axes."""
    af: object
    cg: np.ndarray                      # airframe axes
    cd0: float
    fin: Fin | None
    controls: dict[str, ControlSurface]
    cma_fus: float = 0.0                # /rad, fuselage Munk moment on the wing area and MAC
    cnb_fus: float = 0.0                # /rad, fuselage yawing moment on the wing area and span
    cd_gear: float = 0.0
    cd_flap_per_rad: float = 0.0
    nc: int = 4
    ns_wing: int = 14
    ns_tail: int = 6
    tail: bool = True
    k_induced: float = field(init=False)

    def __post_init__(self):
        af = self.af
        self.S, self.c, self.b = af.wing_area, af.mac, af.span
        L = build_lattice(af, self.nc, self.ns_wing, self.ns_tail, self.tail)
        self.L = L
        Aind = horseshoe_induced(L.C, L)
        self.lu = lu_factor(np.einsum("mnk,mk->mn", Aind, L.nrm0))
        self.mid = 0.5 * (L.P1 + L.P2)
        self.dl = L.P2 - L.P1
        self.r_mid = self.mid - self.cg                      # A axes, from the CG
        self.r_ctl = L.C - self.cg
        t_span = self.dl / np.linalg.norm(self.dl, axis=1, keepdims=True)
        self.dn = np.cross(t_span, L.nrm0)                    # d(normal)/d(nose-up rotation of the panel)
        self.n_strips = len(L.chord)
        # strip reference points (quarter-chord midpoints) for the wind: one wind sample per strip
        sp = np.zeros((self.n_strips, 3))
        np.add.at(sp, L.strip, self.mid)
        cnt = np.bincount(L.strip, minlength=self.n_strips)
        self.strip_points = sp / cnt[:, None]
        half = {0: af.span / 2, 1: (af.htail()["span"] / 2 if self.tail else 1.0)}
        eta = np.abs(self.strip_points[:, 1]) / np.array([half[s] for s in L.strip_surf])
        self.strip_eta = eta
        self.ctl = {}
        for name, cs in self.controls.items():
            sel = (L.strip_surf == cs.surface) & (eta >= cs.eta_in) & (eta <= cs.eta_out)
            side = np.where(self.strip_points[:, 1] >= 0, 1.0, -1.0) if cs.antisymmetric else np.ones(self.n_strips)
            w = (sel * side)[L.strip] * cs.effectiveness()
            # trailing edge down = nose-up rotation of the chord line by tau delta on the right side
            self.ctl[name] = w
        # induced drag factor from the lattice: CDi = K CL^2 (Trefftz plane, wing + tail, alpha sweep at i_t = 0)
        sol = vlm_solve(af, float(self.cg[0]), self.nc, self.ns_wing, self.ns_tail, self.tail)
        self.vlm = sol
        al = np.radians([0.0, 4.0, 8.0])
        CL = np.array([sol.coefficients(a, 0.0)["CL"] for a in al])
        CDi = np.array([sol.coefficients(a, 0.0)["CDi"] for a in al])
        self.k_induced = float(np.polyfit(CL ** 2, CDi, 1)[0])
        self.oswald = 1 / (math.pi * af.aspect_ratio * self.k_induced)

    # ------------------------------------------------------------------ the solve
    def circulation(self, V_ctl: np.ndarray, deflections: dict[str, float]) -> np.ndarray:
        """Bound circulation [m^2/s] for relative air velocities at the control points (N, 3) in A axes."""
        n = self.L.nrm0.copy()
        for name, d in deflections.items():
            if d and name in self.ctl:
                n = n + (self.ctl[name] * d)[:, None] * self.dn
        rhs = -np.einsum("ij,ij->i", n, V_ctl)
        return lu_solve(self.lu, rhs)

    def loads(self, v_b: np.ndarray, omega_b: np.ndarray, rho: float, wind_b_strips: np.ndarray | None = None,
              wind_b_fin: np.ndarray | None = None, wind_b_cg: np.ndarray | None = None,
              deflections: dict[str, float] | None = None, height_agl: float = np.inf,
              gear_down: bool = False) -> dict[str, object]:
        """Aerodynamic force and moment about the CG in body axes.

        ``v_b``: aircraft velocity over the ground in body axes, ``omega_b`` body rates, ``wind_b_*`` the wind in body
        axes at the strip reference points (n_strips, 3), at the fin and at the CG (default: none)."""
        defl = deflections or {}
        vA = T_AB @ v_b
        wA = T_AB @ omega_b
        L = self.L
        wind_s = np.zeros((self.n_strips, 3)) if wind_b_strips is None else wind_b_strips @ T_AB
        wind_p = wind_s[L.strip]
        V_ctl = wind_p - (vA + np.cross(wA, self.r_ctl))
        V_mid = wind_p - (vA + np.cross(wA, self.r_mid))
        G = self.circulation(V_ctl, defl)
        F_p = rho * G[:, None] * np.cross(V_mid, self.dl)
        F_A = F_p.sum(0)
        M_A = np.cross(self.r_mid, F_p).sum(0)
        # free stream at the CG
        w_cg = np.zeros(3) if wind_b_cg is None else T_AB @ wind_b_cg
        Vc = w_cg - vA
        Vn = float(np.linalg.norm(Vc)) + 1e-9
        q = 0.5 * rho * Vn * Vn
        alpha = math.atan2(Vc[2], Vc[0])                 # A axes: air towards +x, upward = positive alpha
        beta = math.asin(np.clip(-Vc[1] / Vn, -1, 1))     # air towards -y (from the right) = positive beta
        e_D = Vc / Vn                                     # drag acts along the relative wind
        e_L = np.array([-Vc[2], 0.0, Vc[0]]) / (math.hypot(Vc[0], Vc[2]) + 1e-12)
        lift = float(F_A @ e_L)
        CL = lift / (q * self.S)
        # induced drag with McCormick's ground effect, zero-lift drag, gear, flaps
        h = height_agl / self.b
        k_ge = (16 * h) ** 2 / (1 + (16 * h) ** 2) if np.isfinite(h) else 1.0
        CD = self.cd0 + k_ge * self.k_induced * CL * CL + (self.cd_gear if gear_down else 0.0) \
            + self.cd_flap_per_rad * abs(defl.get("flap", 0.0))
        F_A = F_A + q * self.S * CD * e_D
        # vertical tail (lifting line at its aerodynamic centre)
        fin_force = 0.0
        if self.fin is not None:
            r = self.fin.ac - self.cg
            wf = np.zeros(3) if wind_b_fin is None else T_AB @ wind_b_fin
            Vf = wf - (vA + np.cross(wA, r))
            Vfn = float(np.linalg.norm(Vf)) + 1e-9
            sig = math.atan2(Vf[1], Vf[0])                 # flow angle towards +y at the fin
            lim = math.radians(self.fin.stall_deg)
            ang = float(np.clip(sig + flap_effectiveness(self.fin.rudder_cf_c) * defl.get("rudder", 0.0), -lim, lim))
            Yf = 0.5 * rho * Vfn * Vfn * self.fin.area * self.fin.lift_slope * ang
            Ff = np.array([0.0, Yf, 0.0])
            F_A = F_A + Ff
            M_A = M_A + np.cross(r, Ff)
            fin_force = Yf
        F_b = T_AB @ F_A
        M_b = T_AB @ M_A
        # fuselage Munk moments (destabilising): nose-up with alpha, nose towards the sideslip with beta
        M_b = M_b + np.array([0.0, q * self.S * self.c * self.cma_fus * alpha, q * self.S * self.b * self.cnb_fus * beta])
        return {"F": F_b, "M": M_b, "alpha": alpha, "beta": beta, "V": Vn, "q": q, "CL": CL, "CD": CD,
                "gamma": G, "fin_force": fin_force}

    def panel_dcp(self, gamma: np.ndarray, V: float) -> np.ndarray:
        """Pressure-coefficient jump across each panel, dCp = rho Gamma V |dl| / (q area) = 2 Gamma / (V dx)."""
        L = self.L
        return 2 * gamma / (max(V, 1e-6) * L.chord[L.strip] / self.nc)

    def panel_quads(self) -> np.ndarray:
        """(N, 4, 3) corner points of every panel in airframe axes (for visualisation of the loading)."""
        L = self.L
        ch = (L.chord[L.strip] / self.nc)[:, None] * np.array([1.0, 0, 0])
        a = L.P1 - 0.25 * ch
        b = L.P2 - 0.25 * ch
        return np.stack([a, b, b + ch, a + ch], 1)


def fuselage_munk_factor(wing_qc_fraction: float) -> float:
    """Raymer's K_fus [per degree] against the wing root quarter-chord position as a fraction of the fuselage length
    (Aircraft Design fig. 16.14), interpolated between the two values the aero-optimizer models use (0.012 at 30 %,
    0.025 at 45 %): an approximation of the chart, clipped to that range."""
    return float(np.interp(wing_qc_fraction, [0.30, 0.45], [0.012, 0.025]))


def fuselage_moments(length: float, max_width: float, volume: float, wing_area: float, mac: float, span: float,
                     depth: float, wing_qc_fraction: float):
    """(Cm_alpha, Cn_beta) of the fuselage [/rad]: Raymer eq. 16.25 Cm_a = K_f W_f^2 L_f / (c S) (per degree x
    57.3) and eq. 16.47 Cn_b = -1.3 (volume / (S b)) (D_f / W_f) (sign: body axes, a positive Cn_beta is stable,
    the fuselage is unstable so the value is negative)."""
    cma = fuselage_munk_factor(wing_qc_fraction) * max_width ** 2 * length / (mac * wing_area) * 180 / math.pi
    cnb = -1.3 * volume / (wing_area * span) * (depth / max_width)
    return cma, cnb
