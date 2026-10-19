"""Lumped thermal network of a radial-flux machine, and the demagnetisation check of its magnets.

:class:`ThermalNetwork` is a general network of isothermal nodes: heat capacities ``C`` (J/K), conductances ``G``
(W/K) between nodes and to fixed-temperature boundaries, heat sources ``P`` (W). Steady state solves
``(L + G_b) T = P + G_b T_b`` (``L`` the conductance Laplacian); transients use backward Euler or the trapezoidal
rule (A-stable, orders 1 and 2), with time-dependent sources.

:func:`machine_network` builds the classic seven-node model of a water-jacketed machine (stator yoke, teeth,
winding in the slots, end windings, rotor, magnets, shaft/bearing path) from the cross-section and stated material and
contact data: conduction through cylindrical shells ``G = 2 pi k L / ln(r2/r1)``, through the slot insulation
(equivalent conductivity of the winding), axially along the conductors to the end windings, and convection across the
air gap and at the end windings with given heat-transfer coefficients. The coefficients are inputs, not
correlations fitted here: a validated model needs them from measurements or a CFD study of the actual machine.

:func:`demagnetisation_margin` evaluates the FEA field inside every magnet element: the field opposing the
magnetisation, ``H_rev = -(B . m - Br) / (mu0 mu_r)``, against the intrinsic coercivity ``Hcj(T)`` of the grade (the
knee of a square-loop NdFeB magnet); margin < 0 marks elements that would demagnetise irreversibly.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..constants import MU0

__all__ = ["ThermalNetwork", "cylinder_conductance", "machine_network", "MachineThermalData",
           "demagnetisation_margin", "reverse_field", "NDFEB_HCJ_MIN_KAM", "intrinsic_coercivity",
           "magnet_temperature_sweep", "demagnetisation_refinement"]

# Minimum intrinsic coercivity of sintered NdFeB at 20 degC by grade suffix (the grade-letter convention of magnet
# manufacturers): N 12, M 14, H 17, SH 20, UH 25, EH 30, AH 35 kOe (1 kOe = 79.577 kA/m).
NDFEB_HCJ_MIN_KAM = {k: v * 1e3 / (4 * math.pi) for k, v in
                     {"N": 12.0, "M": 14.0, "H": 17.0, "SH": 20.0, "UH": 25.0, "EH": 30.0, "AH": 35.0}.items()}


def intrinsic_coercivity(Hcj20: float, temperature: float, beta: float = -0.0055) -> float:
    """``Hcj(T) = Hcj20 (1 + beta (T - 20))`` (A/m); ``beta`` is the reversible temperature coefficient of Hcj
    (typically -0.5 to -0.6 %/K for sintered NdFeB; take the grade's datasheet value)."""
    return Hcj20 * (1.0 + beta * (temperature - 20.0))


def cylinder_conductance(k: float, r1: float, r2: float, length: float, fraction: float = 1.0) -> float:
    """Radial conduction through a cylindrical shell (or ``fraction`` of it): ``2 pi k L f / ln(r2 / r1)``."""
    return 2.0 * math.pi * k * length * fraction / math.log(r2 / r1)


@dataclass
class ThermalNetwork:
    names: list[str]
    C: np.ndarray                                   # J/K per node
    links: list[tuple[int, int, float]] = field(default_factory=list)       # (i, j, G)
    boundaries: list[tuple[int, float, float]] = field(default_factory=list)  # (i, G, T_boundary)

    @classmethod
    def empty(cls, names: Sequence[str], C: Sequence[float]) -> ThermalNetwork:
        return cls(list(names), np.asarray(C, float))

    def index(self, name: str) -> int:
        return self.names.index(name)

    def connect(self, a: str, b: str, G: float) -> ThermalNetwork:
        self.links.append((self.index(a), self.index(b), float(G)))
        return self

    def to_boundary(self, a: str, G: float, T: float) -> ThermalNetwork:
        self.boundaries.append((self.index(a), float(G), float(T)))
        return self

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """``K`` (W/K) and the boundary load ``b`` (W): ``C dT/dt = -K T + b + P``."""
        n = len(self.names)
        K = np.zeros((n, n))
        b = np.zeros(n)
        for i, j, G in self.links:
            K[i, i] += G
            K[j, j] += G
            K[i, j] -= G
            K[j, i] -= G
        for i, G, T in self.boundaries:
            K[i, i] += G
            b[i] += G * T
        return K, b

    def steady(self, P: Sequence[float]) -> np.ndarray:
        K, b = self.matrix()
        return np.linalg.solve(K, b + np.asarray(P, float))

    def transient(self, P: Callable[[float], Sequence[float]] | Sequence[float], t_end: float, dt: float,
                  T0: Sequence[float] | float, method: str = "trapezoidal") -> tuple[np.ndarray, np.ndarray]:
        """Temperatures at ``t = 0, dt, ..., t_end``; ``P`` constant or ``P(t)``; ``method``: ``"euler"`` (backward)
        or ``"trapezoidal"`` (Crank-Nicolson)."""
        K, b = self.matrix()
        n = len(self.names)
        if callable(P):
            Pf = P
        else:
            P_const = np.asarray(P, float)

            def Pf(t):
                return P_const
        steps = int(round(t_end / dt))
        T = np.empty((steps + 1, n))
        T[0] = np.full(n, float(T0)) if np.isscalar(T0) else np.asarray(T0, float)
        Cd = np.diag(self.C / dt)
        theta = 1.0 if method == "euler" else 0.5
        A = Cd + theta * K
        lu = np.linalg.inv(A) if n <= 64 else None
        for k in range(steps):
            t = k * dt
            rhs = Cd @ T[k] - (1 - theta) * K @ T[k] + b + theta * np.asarray(Pf(t + dt), float) \
                + (1 - theta) * np.asarray(Pf(t), float)
            T[k + 1] = lu @ rhs if lu is not None else np.linalg.solve(A, rhs)
        return np.arange(steps + 1) * dt, T


@dataclass
class MachineThermalData:
    """Material and boundary data of :func:`machine_network` (SI). Defaults are typical textbook values, stated as
    assumptions: laminations radial k = 28 W/m K, winding (copper, resin, insulation) equivalent k = 1.0 W/m K across
    the slot, copper k = 390 W/m K along the conductors with copper fill 0.45, air-gap h = 120 W/m^2 K, end-winding
    h = 40 W/m^2 K, water jacket to stator back h A = 300 W/K with coolant at 65 degC, densities and specific heats
    of steel 7650 / 460, copper 8900 / 385, NdFeB 7500 / 450."""

    k_lam: float = 28.0
    k_winding: float = 1.0
    k_cu: float = 390.0
    copper_fill: float = 0.45
    h_gap: float = 120.0
    h_end: float = 40.0
    jacket_GA: float = 300.0
    coolant_T: float = 65.0
    k_magnet_contact: float = 0.5   # W/m K, glue/air layer around the magnet in its pocket
    contact_gap: float = 0.1e-3     # m
    shaft_G: float = 2.0            # W/K rotor to housing through shaft and bearings
    c_steel: tuple = (7650.0, 460.0)
    c_cu: tuple = (8900.0, 385.0)
    c_mag: tuple = (7500.0, 450.0)


def machine_network(geometry, data: MachineThermalData | None = None, end_length: float = 0.16) -> ThermalNetwork:
    """Seven-node network of a machine with a water jacket on the stator back (see the module docstring)."""
    d = data or MachineThermalData()
    st, rt, L = geometry.stator, geometry.rotor, geometry.length
    S = st.slots
    r_bore, r_out = st.r_in, st.r_out
    r_slot_top = r_bore + st.H0 + st.H1
    r_slot_bot = r_slot_top + st.H2
    A_slot = st.slot_area()
    tooth_frac = 1.0 - S * 0.5 * (st.W1 + st.W2) / (2 * math.pi * 0.5 * (r_slot_top + r_slot_bot))
    V_yoke = math.pi * (r_out ** 2 - r_slot_bot ** 2) * L
    V_teeth = (math.pi * (r_slot_bot ** 2 - r_bore ** 2) - S * A_slot) * L
    V_cu_slot = S * A_slot * d.copper_fill * L
    V_cu_end = S * A_slot * d.copper_fill * end_length
    n_mag = 2 * rt.poles
    V_mag = n_mag * rt.W4 * rt.H3 * L
    V_rotor = math.pi * (rt.r_out ** 2 - rt.r_in ** 2) * L - V_mag
    C = [d.c_steel[0] * d.c_steel[1] * V_yoke, d.c_steel[0] * d.c_steel[1] * V_teeth,
         d.c_cu[0] * d.c_cu[1] * V_cu_slot, d.c_cu[0] * d.c_cu[1] * V_cu_end,
         d.c_steel[0] * d.c_steel[1] * V_rotor, d.c_mag[0] * d.c_mag[1] * V_mag]
    net = ThermalNetwork.empty(["yoke", "teeth", "winding", "end_winding", "rotor", "magnets"], C)
    r_mid_yoke = 0.5 * (r_out + r_slot_bot)
    net.to_boundary("yoke", 1.0 / (1.0 / d.jacket_GA + 1.0 / cylinder_conductance(d.k_lam, r_mid_yoke, r_out, L)),
                    d.coolant_T)
    net.connect("yoke", "teeth", 1.0 / (1.0 / cylinder_conductance(d.k_lam, r_slot_bot, r_mid_yoke, L)
                                        + 1.0 / cylinder_conductance(d.k_lam, 0.5 * (r_bore + r_slot_bot), r_slot_bot, L,
                                                                     tooth_frac)))
    # winding to iron: across half the slot width to the slot walls (two sides + bottom), equivalent conductivity
    w_half = 0.25 * (st.W1 + st.W2)
    perim = 2 * st.H2 + 0.5 * (st.W1 + st.W2)
    G_slot = S * d.k_winding * perim * L / w_half
    net.connect("winding", "teeth", 0.7 * G_slot)    # side walls
    net.connect("winding", "yoke", 0.3 * G_slot)     # slot bottom
    A_cu = S * A_slot * d.copper_fill
    net.connect("winding", "end_winding", d.k_cu * A_cu / (0.5 * (L + end_length)))
    A_end = S * 2 * (2 * st.H2 + st.W2) * end_length  # wetted surface of both end windings, approximately
    net.to_boundary("end_winding", d.h_end * A_end, d.coolant_T)
    net.connect("teeth", "rotor", d.h_gap * 2 * math.pi * rt.r_out * L)  # air-gap convection, film on both sides
    A_mag = n_mag * 2 * (rt.W4 + rt.H3) * L
    net.connect("rotor", "magnets", d.k_magnet_contact * A_mag / d.contact_gap)
    net.to_boundary("rotor", d.shaft_G, d.coolant_T)
    return net


def reverse_field(B: np.ndarray, angle: np.ndarray, Br: float, mu_r: float) -> np.ndarray:
    """Field opposing the magnetisation (A/m) in linear magnet elements: ``H_rev = -(B . m - Br) / (mu0 mu_r)`` with
    ``m = (cos angle, sin angle)``, from ``B = mu0 mu_r H + Br m``."""
    Bm = B[:, 0] * np.cos(angle) + B[:, 1] * np.sin(angle)
    return -(Bm - Br) / (MU0 * mu_r)


def demagnetisation_margin(machine, ms, Hcj: float, margin_ratio: float = 0.9) -> dict:
    """Field opposing the magnetisation in every magnet element of the solution ``ms`` of ``machine`` (an
    :class:`~.pmsm.IPMMachine`), against ``Hcj`` (A/m at the magnet temperature; the knee of a square-loop NdFeB
    magnet). Returns the worst reverse field, the margin ``Hcj - H_rev`` (A/m), the demagnetised area fraction
    (``H_rev > Hcj``), the area fraction within ``margin_ratio`` of the knee (``H_rev > margin_ratio Hcj``; less
    sensitive to single corner elements than the knee crossing), and per magnet element the element index and
    ``H_rev / Hcj``."""
    mask = machine._magnet
    H_rev = reverse_field(ms.B[mask], machine.mm.magnet_angle[mask] + ms.theta, machine.Br, machine.magnet.mu_r)
    area = ms.sol.geometry.area[mask]
    bad = H_rev > Hcj
    return {"H_rev_max": float(H_rev.max()), "margin": float(Hcj - H_rev.max()), "Hcj": float(Hcj),
            "demagnetised_fraction": float(area[bad].sum() / area.sum()),
            "margin_ratio": float(margin_ratio),
            "fraction_above_margin": float(area[H_rev > margin_ratio * Hcj].sum() / area.sum()),
            "elements": np.where(mask)[0], "ratio": H_rev / Hcj}


def demagnetisation_refinement(geometry, magnet_temperature: float, currents_d: dict, Hcj: dict,
                               gradings=(0.2, 0.1, 0.05, 0.025), n_gap: int = 960, margin_ratio: float = 0.9,
                               magnet=None) -> list[dict]:
    """Mesh study of :func:`demagnetisation_margin`: the machine meshed with decreasing ``grading`` (element size
    growth away from the gap, :class:`~.mesher.MachineMesh`), so the magnet elements shrink at a fixed gap mesh.
    ``currents_d``: label -> d-axis current (A, rotor at 0); ``Hcj``: grade -> intrinsic coercivity (A/m). Per level:
    number and mean size of the magnet elements, and per case and grade the worst ``H_rev / Hcj``, the demagnetised
    fraction and the fraction above ``margin_ratio``."""
    from .pmsm import IPMMachine

    out = []
    for gr in gradings:
        m = IPMMachine(geometry, n_gap=n_gap, grading=float(gr), magnet_temperature=magnet_temperature,
                       **({"magnet": magnet} if magnet else {}))
        level = {"grading": float(gr), "magnet_elements": int(m._magnet.sum()), "cases": {}}
        for label, i_d in currents_d.items():
            ms = m.solve(0.0, m.currents_dq(0.0, float(i_d), 0.0))
            area = ms.sol.geometry.area[m._magnet]
            level["magnet_element_size_m"] = float(np.sqrt(2.0 * area.mean()))
            level["cases"][label] = {}
            for grade, H in Hcj.items():
                d = demagnetisation_margin(m, ms, H, margin_ratio)
                level["cases"][label][grade] = {"ratio_max": float(d["ratio"].max()),
                                                "demagnetised_fraction": d["demagnetised_fraction"],
                                                "fraction_above_margin": d["fraction_above_margin"]}
        out.append(level)
    return out


def magnet_temperature_sweep(geometry, temperatures, I_peak: float = 250.0, n_gap: int = 960,
                             n_positions: int = 12, n_flux: int = 24, magnet=None) -> list[dict]:
    """Machine performance against the magnet temperature (``Br(T)`` of the grade): MTPA torque and angle at
    ``I_peak``, torque ripple, magnet flux linkage and back-EMF constant (fundamental, from ``n_flux`` angles over an
    electrical period), frozen-permeability L_d, L_q at the MTPA point."""
    from .pmsm import IPMMachine

    out = []
    for T in temperatures:
        m = IPMMachine(geometry, n_gap=n_gap, magnet_temperature=float(T), **({"magnet": magnet} if magnet else {}))
        gm, _ = m.mtpa(I_peak, n_positions=4, tol=math.radians(1.0))
        op = m.operating_point(I_peak=I_peak, gamma=gm, n_positions=n_positions)
        ind = m.inductances(op["i_d"], op["i_q"])
        th = m.positions(n_flux, period=2 * math.pi / m.pole_pairs)
        psi = np.array([m.flux_linkage(m.solve(t)) for t in th])
        emf = m.back_emf(psi, th, 1000.0)
        out.append({"magnet_temperature_C": float(T), "Br_T": m.Br, "mtpa_gamma_deg": math.degrees(gm),
                    "torque_Nm": op["torque_mean"], "ripple": op["ripple"], "L_d_mH": 1e3 * ind["L_dd"],
                    "L_q_mH": 1e3 * ind["L_qq"], "psi_pm_Wb": ind["psi_pm_d"],
                    "emf_phase_fundamental_peak_V_1000rpm": emf["fundamental_peak"]})
    return out
