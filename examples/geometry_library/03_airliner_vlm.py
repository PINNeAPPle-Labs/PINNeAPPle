"""Vortex-lattice aerodynamics of the library airliner plus a friction-drag estimate from its mesh.

    python examples/geometry_library/03_airliner_vlm.py

Rebuilds the ``airliner_a320`` entry's aircraft (the same parameters as ``pp.geometries.build``), solves the
vortex lattice of wing + horizontal tail (``pinneapple_design.aero.vlm``): lift slope, neutral point, trim at a
cruise lift coefficient, induced drag. The wetted area of the library mesh (fuselage, wing, tails, nacelles) gives
a turbulent flat-plate friction drag with form factors at Mach 0.78, 11 000 m: the zero-lift drag, and L/D.
Published values for comparison (A320 class): CL_alpha about 5.5-6 /rad, cruise L/D about 16-18. Seconds.
"""
import math

import pinneapple as pp
from pinneapple_design.aero.airliner import Airliner
from pinneapple_design.aero.buildup import friction_drag
from pinneapple_design.aero.vlm import solve

g = pp.geometries.build("airliner_a320")
p = g.params
ac = Airliner(length=p["length"], wing_area=p["wing_area"], aspect_ratio=p["aspect_ratio"], taper=p["taper"],
              sweep_le=p["sweep_le"], dihedral=p["dihedral"], twist=p["twist"], fan_diameter=p["fan_diameter"],
              engine_station=p["engine_station"], sharklets=p["sharklets"])
x_ref = ac.mac_le_x + 0.25 * ac.mac
sol = solve(ac, x_ref)
print(g)
print(f"span {ac.span:.2f} m, S {ac.wing_area} m^2, MAC {ac.mac:.2f} m, reference x {x_ref:.2f} m")
print(f"CL_alpha {sol.CL[1]:.2f} /rad, neutral point x = {sol.neutral_point:.2f} m "
      f"(wing + tail only: the fuselage's destabilising moment is not in the lattice)")

# cruise: ISA 11 000 m, Mach 0.78, mass 64 t
T, rho, a_snd = 216.65, 0.3639, 295.07
V = 0.78 * a_snd
q = 0.5 * rho * V * V
CL = 64000 * 9.81 / (q * ac.wing_area)
alpha, it = sol.trim(CL)
c = sol.coefficients(alpha, it)
# friction drag from the library mesh: wetted area per component, Re on its length, form factors (Raymer), scaled to
# the wetted area of the bodies' union, interference and excrescences (pinneapple_design.aero.buildup)
d = friction_drag(g, ac, rho=rho, V=V, T=T, mach=0.78)
CD0, S_wet = d["CD0"], d["S_wet"]
CD = CD0 + c["CDi"] / 0.85                                       # span efficiency of the real wing vs the VLM
print(f"cruise M 0.78 FL360, 64 t: CL {CL:.3f}, alpha {math.degrees(alpha):.2f} deg, tail incidence "
      f"{math.degrees(it):.2f} deg")
print(f"wetted area {S_wet:.0f} m^2 (sum of the body areas {g.area():.0f} m^2), CD0 {CD0:.4f}, CDi {c['CDi']:.4f}, CD {CD:.4f}, L/D {CL / CD:.1f}")
