"""Axial-resolved lumped thermal network of the machine: the 2-D radial network of :mod:`.thermal`, cut into segments
along the stack.

:func:`~..thermal.machine_network` models one node per part (yoke, teeth, winding, end windings, rotor, magnets): the
stack is isothermal along ``z`` and the only axial information is the separate end-winding node. That is enough for a
hot-spot *level* and it is not enough for a hot-spot *place*: the middle of the stack sits between two cooled ends,
the ends of the stack sit next to the end windings, and the winding's axial conduction (copper, 390 W/m K along the
conductors) spreads heat over the whole length. This module keeps the same radial conductances and gives every part
``n_segments`` nodes along the stack, linked to their neighbours by ``k A / dz``.

The network is quasi-3-D: resolved in ``z`` and lumped in ``r`` and ``theta`` (every slot and every pole is the same).
It is solved with the steady-state solver of :class:`~..thermal.ThermalNetwork`, so all the checks of that class
apply, and the transient solver gives the thermal step response for free.

Assumptions, stated where they are used: the axial conductivity of the lamination stack is far below its in-plane one
(the sheets are insulated), so the axial conductance of the iron rings uses its own ``k_lam_axial`` (default 4 W/m K,
a typical value for a coated lamination stack; the in-plane 28 W/m K of :class:`~..thermal.MachineThermalData` is the
radial one), while copper and magnets conduct at their handbook values. The jacket is spread over the outer surface
of every yoke segment in proportion to its length.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..thermal import MachineThermalData, ThermalNetwork, cylinder_conductance

__all__ = ["AxialThermalModel", "axial_network"]

RADIAL = ("yoke", "teeth", "winding", "rotor", "magnets")


@dataclass
class AxialThermalModel:
    """The nodes, their conductances and their volumes of the axial network, ready to be solved.

    Attributes after construction: ``names`` (``"yoke[0]"``... in radial order per segment, then ``"end_winding"``),
    ``K`` (segments), ``network`` (:class:`~..thermal.ThermalNetwork`), ``volumes`` (m^3 per node, for the loss
    density and the heat capacity) and ``segment_length`` (m).
    """
    geometry: Any
    data: MachineThermalData = field(default_factory=MachineThermalData)
    n_segments: int = 4
    end_length: float = 0.16
    k_lam_axial: float = 4.0
    k_magnet: float = 9.0
    _extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_segments < 1:
            raise ValueError("n_segments must be at least 1")
        self.K = int(self.n_segments)
        self.L = float(self.geometry.length)
        self.dz = self.L / self.K
        self._build()

    # --------------------------------------------------------------------------------------------- areas
    def _areas(self) -> dict[str, float]:
        """Cross-section areas (m^2) of each part, from the same cross-section the mesher uses."""
        st, rt, d = self.geometry.stator, self.geometry.rotor, self.data
        r_bore, r_out = st.r_in, st.r_out
        r_slot_top = r_bore + st.H0 + st.H1
        r_slot_bot = r_slot_top + st.H2
        A_slot = st.slot_area()
        S = st.slots
        n_mag = 2 * rt.poles
        return {"yoke": math.pi * (r_out ** 2 - r_slot_bot ** 2),
                "teeth": math.pi * (r_slot_bot ** 2 - r_bore ** 2) - S * A_slot,
                "winding": S * A_slot * d.copper_fill,
                "rotor": math.pi * (rt.r_out ** 2 - rt.r_in ** 2) - n_mag * rt.W4 * rt.H3,
                "magnets": n_mag * rt.W4 * rt.H3}

    def _build(self) -> None:
        st, rt, d, dz = self.geometry.stator, self.geometry.rotor, self.data, self.dz
        A = self._areas()
        r_bore, r_out = st.r_in, st.r_out
        r_slot_top = r_bore + st.H0 + st.H1
        r_slot_bot = r_slot_top + st.H2
        r_mid_yoke = 0.5 * (r_out + r_slot_bot)
        S, n_mag = st.slots, 2 * rt.poles

        names: list[str] = []
        for k in range(self.K):
            names += [f"{p}[{k}]" for p in RADIAL]
        names.append("end_winding")
        volumes = np.zeros(len(names))
        C = np.zeros(len(names))
        net = ThermalNetwork.empty(names, C)
        idx = {(p, k): RADIAL.index(p) + len(RADIAL) * k for k in range(self.K) for p in RADIAL}
        i_end = len(names) - 1

        rho_c = {p: 0.0 for p in RADIAL}
        rho_c["yoke"] = rho_c["teeth"] = rho_c["rotor"] = d.c_steel[0] * d.c_steel[1]
        rho_c["winding"] = d.c_cu[0] * d.c_cu[1]
        rho_c["magnets"] = d.c_mag[0] * d.c_mag[1]
        for p in RADIAL:
            for k in range(self.K):
                volumes[idx[(p, k)]] = A[p] * dz

        # ------------------------------------------------------------------ radial (independent per segment)
        tooth_frac = 1.0 - S * 0.5 * (st.W1 + st.W2) / (2 * math.pi * 0.5 * (r_slot_top + r_slot_bot))
        w_half = 0.25 * (st.W1 + st.W2)
        perim = 2 * st.H2 + 0.5 * (st.W1 + st.W2)
        G_slot = S * d.k_winding * perim * dz / w_half
        A_cu = A["winding"]
        A_mag = n_mag * 2 * (rt.W4 + rt.H3) * dz
        g_radial = {
            ("yoke", "teeth"): 1.0 / (1.0 / cylinder_conductance(d.k_lam, r_slot_bot, r_mid_yoke, dz)
                                      + 1.0 / cylinder_conductance(d.k_lam, 0.5 * (r_bore + r_slot_bot), r_slot_bot,
                                                                   dz, tooth_frac)),
            ("winding", "teeth"): 0.7 * G_slot,
            ("winding", "yoke"): 0.3 * G_slot,
            ("teeth", "rotor"): d.h_gap * 2 * math.pi * rt.r_out * dz,
            ("rotor", "magnets"): d.k_magnet_contact * A_mag / d.contact_gap,
        }
        for k in range(self.K):
            for (a, b), G in g_radial.items():
                if G > 0:
                    net.connect(f"{a}[{k}]", f"{b}[{k}]", G)
            # water jacket on the outer surface of this yoke segment, flow-scaled by the caller through ``data``
            net.to_boundary(f"yoke[{k}]",
                            1.0 / (1.0 / (d.jacket_GA * dz / self.L)
                                   + 1.0 / cylinder_conductance(d.k_lam, r_mid_yoke, r_out, dz)),
                            d.coolant_T)
            # shaft/bearing path, shared equally by the segments
            net.to_boundary(f"rotor[{k}]", d.shaft_G / self.K, d.coolant_T)

        # ------------------------------------------------------------------ axial (inside each radial ring)
        k_axial = {"yoke": self.k_lam_axial, "teeth": self.k_lam_axial, "rotor": self.k_lam_axial,
                   "winding": d.k_cu, "magnets": self.k_magnet}
        for p in RADIAL:
            if self.K > 1:
                G = k_axial[p] * A[p] / dz
                for k in range(self.K - 1):
                    net.connect(f"{p}[{k}]", f"{p}[{k + 1}]", G)
        # the copper of the end segments leaves the stack into the end winding (with a single segment both ends are
        # the same node: one link, not two)
        for k in sorted({0, self.K - 1}):
            net.connect(f"winding[{k}]", "end_winding", d.k_cu * A_cu / (0.5 * (dz + self.end_length)))

        # end windings: copper volume of one end overhang per end, cooled by convection everywhere
        V_end = 2.0 * S * A_cu * self.end_length
        volumes[i_end] = V_end
        A_wetted = S * 2 * (2 * st.H2 + st.W2) * self.end_length
        net.to_boundary("end_winding", d.h_end * A_wetted, d.coolant_T)

        for i, v in enumerate(volumes):
            C[i] = v * (d.c_cu[0] * d.c_cu[1] if names[i].startswith("winding") or names[i] == "end_winding"
                        else d.c_mag[0] * d.c_mag[1] if names[i].startswith("magnets")
                        else d.c_steel[0] * d.c_steel[1])
        net.C = C
        self.names, self.network, self.volumes = names, net, volumes
        self.areas = A

    # --------------------------------------------------------------------------------------------- solving
    def index(self, part: str, k: int | None = None) -> int:
        return self.names.index(f"{part}[{k}]" if k is not None else part)

    def powers(self, p_yoke: float, p_teeth: float, p_winding_slot: float, p_winding_end: float,
               p_rotor: float, p_magnets: float) -> np.ndarray:
        """Power (W) into every node from the losses of one operating point. The stator iron is spread over the
        segments in proportion to their length; the slot copper likewise; the end-winding copper goes to the single
        end-winding node."""
        P = np.zeros(len(self.names))
        for k in range(self.K):
            P[self.index("yoke", k)] += p_yoke / self.K
            P[self.index("teeth", k)] += p_teeth / self.K
            P[self.index("winding", k)] += p_winding_slot / self.K
            P[self.index("rotor", k)] += p_rotor / self.K
            P[self.index("magnets", k)] += p_magnets / self.K
        P[self.index("end_winding")] += p_winding_end
        return P

    def steady(self, P: np.ndarray) -> dict[str, Any]:
        """Node temperatures (degC) and the summaries a design decision needs: the hottest winding and magnet node
        (they are not at mid-stack when the ends are cooled) and their axial profiles."""
        T = self.network.steady(P)
        out = {"T": T, "names": list(self.names)}
        for part in RADIAL:
            prof = np.array([T[self.index(part, k)] for k in range(self.K)])
            out[f"T_{part}_profile_C"] = prof
            out[f"T_{part}_max_C"] = float(prof.max())
            out[f"T_{part}_mean_C"] = float(prof.mean())
        out["T_end_winding_C"] = float(T[self.index("end_winding")])
        return out


def axial_network(geometry, data: MachineThermalData | None = None, n_segments: int = 4,
                  end_length: float = 0.16, **kw) -> AxialThermalModel:
    """The axial network of ``geometry`` (see :class:`AxialThermalModel`)."""
    return AxialThermalModel(geometry, data or MachineThermalData(), n_segments=n_segments,
                             end_length=end_length, **kw)
