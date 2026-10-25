"""Steady temperature of a vented brake rotor from its cooling vanes, as a fin array
(:mod:`.fin_array_conduction`): the vanes are fins of length = vent gap across the airflow, ``n`` of them along the
mean circumference, each with the radial flow length as its other dimension, on a friction ring of given thickness.

Scope: steady state under a sustained braking power (a long descent), one film coefficient for the whole vent, no
radiation, no transient heating within a stop, no thermal contact with the hub. The default rotor (36 cast-iron vanes,
8 mm gap, 4 mm thick, 0.754 m mean circumference) and the 1,480 W / h = 40 W/(m2 K) case come from the source
project (PINNeAPPle-apps ``brakecool_ai`` / ``poc_verification_gap``); they are a worked case, not a measured rotor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from . import fin_array_conduction as fac

__all__ = ["VentedRotor", "rotor_temperature_c", "radial_profile"]


@dataclass(frozen=True)
class VentedRotor:
    vane_count: float = 36.0
    vane_gap_m: float = 0.008
    vane_thickness_m: float = 0.004
    k_w_mk: float = 50.0                     # grey cast iron
    mean_circumference_m: float = 0.754
    radial_flow_length_m: float = 0.06
    ring_thickness_m: float = 0.006
    t_ambient_c: float = 25.0

    def with_(self, **kw) -> VentedRotor:
        return replace(self, **kw)

    @property
    def r_mean_m(self) -> float:
        return self.mean_circumference_m / (2.0 * math.pi)


def _params(rotor: VentedRotor, power_w: float, h_conv: float) -> dict[str, float]:
    return fac.fin_array_profile_params(
        k_material_w_mk=rotor.k_w_mk, h_conv_w_m2k=h_conv, n_fins=rotor.vane_count,
        fin_thickness_m=max(rotor.vane_thickness_m, 1e-6), fin_length_m=max(rotor.vane_gap_m, 1e-6),
        transverse_extent_m=rotor.radial_flow_length_m, spacing_extent_m=rotor.mean_circumference_m,
        base_thickness_m=rotor.ring_thickness_m, power_w=power_w)


def rotor_temperature_c(rotor: VentedRotor, power_w: float, h_conv: float) -> float:
    """Friction-ring (vane root side) temperature."""
    return fac.fin_array_base_temperature_c(_params(rotor, power_w, h_conv), rotor.t_ambient_c, power_w)


def radial_profile(rotor: VentedRotor, power_w: float, h_conv: float) -> dict[str, float]:
    """Parameters of the vane temperature along its radial extent (same fin parameter m, radial corrected length
    L + t/2), for a viewer: T(r) = T_amb + theta_b cosh(m (Lc - x)) / cosh(m Lc), x from the inner edge."""
    p = _params(rotor, power_w, h_conv)
    lc = rotor.radial_flow_length_m + rotor.vane_thickness_m / 2.0
    r_mean_mm = rotor.r_mean_m * 1000.0
    return {"t_ambient_c": rotor.t_ambient_c, "theta_vane_base_c": p["theta_fin_base_c"], "m": p["m"],
            "lc_radial_m": lc, "radial_flow_length_m": rotor.radial_flow_length_m,
            "r_outer_mm": r_mean_mm + rotor.radial_flow_length_m * 500.0,
            "r_inner_mm": r_mean_mm - rotor.radial_flow_length_m * 500.0}
