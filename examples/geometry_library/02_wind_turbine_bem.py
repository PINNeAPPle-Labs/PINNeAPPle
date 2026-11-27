"""Blade-element momentum (BEM) on the NREL 5-MW turbine of the geometry library: power and thrust curves.

    python examples/geometry_library/02_wind_turbine_bem.py

Takes the rotor of ``pp.geometries.build("wind_turbine_nrel5mw")`` (radius, hub, the published chord / twist /
thickness table it is lofted from, precone), runs a steady BEM with Prandtl tip and hub losses and the Glauert /
Buhl high-induction correction, a variable-speed schedule (optimal tip-speed ratio up to the rated rotor speed) and
collective pitch above rated, and compares with the reference turbine: 5 MW rated at 11.4 m/s, 12.1 rpm
(Jonkman et al. 2009). Airfoil polars are simple thin-airfoil / flat-plate models (lift slope 2 pi x 0.9, stall at
about 12 deg, cylinder drag at the root), so expect the power within 10-15 %, not better. No solver to install:
seconds on a laptop.
"""
import math
import os

import numpy as np

import pinneapple as pp
from pinneapple_design.geometry_library.entries.wind_turbine import BLADE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
os.makedirs(OUT, exist_ok=True)

g = pp.geometries.build("wind_turbine_nrel5mw")
p = g.params
R, Rh = p["rotor_diameter"] / 2, p["hub_diameter"] / 2
B, rho = 3, 1.225
blade = next(b for b in g.bodies if b.name == "blade_1")
print(f"{g}\nblade surface {0.5 * sum(np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) for t in blade.vertices[blade.faces]):.0f} m^2"
      f" (both sides), rotor radius {R} m, hub radius {Rh} m")

r_tab, twist_tab, chord_tab, tc_tab = BLADE.T
r = np.linspace(Rh + 0.5, R - 0.3, 40)
dr = np.gradient(r)
chord = np.interp(r, r_tab, chord_tab)
twist = np.radians(np.interp(r, r_tab, twist_tab))
tc = np.interp(r, r_tab, tc_tab)


def polar(alpha, tc):
    """cl, cd of a section of thickness ratio tc at angle of attack alpha (rad): thin airfoil + stall + cylinder."""
    a0 = -math.radians(3.0) * np.clip((0.5 - tc) / 0.32, 0, 1)        # zero-lift angle of the cambered sections
    cl_lin = 2 * math.pi * 0.9 * (alpha - a0)
    stall = math.radians(12.0)
    cl = np.where(np.abs(alpha - a0) < stall, cl_lin, np.sign(alpha - a0) * (1.1 * np.abs(np.sin(2 * (alpha - a0)))
                                                                           + 0.2))
    cd = 0.006 + 0.02 * tc + 0.012 * (alpha - a0) ** 2 + np.where(np.abs(alpha - a0) > stall, 1.2 * np.sin(alpha - a0) ** 2, 0)
    cyl = np.clip((tc - 0.40) / 0.6, 0, 1)                              # root: towards a cylinder, cd 0.5, cl 0
    return cl * (1 - cyl), cd * (1 - cyl) + 0.5 * cyl


def bem(U, omega, pitch):
    a = np.full_like(r, 0.3)
    ap = np.zeros_like(r)
    for _ in range(300):
        phi = np.arctan2((1 - a) * U, (1 + ap) * omega * r)
        alpha = phi - twist - pitch
        cl, cd = polar(alpha, tc)
        cn = cl * np.cos(phi) + cd * np.sin(phi)
        ct = cl * np.sin(phi) - cd * np.cos(phi)
        sig = B * chord / (2 * math.pi * r)
        s = np.maximum(np.abs(np.sin(phi)), 1e-4)
        F = (2 / math.pi * np.arccos(np.exp(-B / 2 * (R - r) / (r * s)))) * \
            (2 / math.pi * np.arccos(np.exp(-B / 2 * (r - Rh) / (Rh * s))))
        F = np.maximum(F, 1e-4)
        CT = sig * (1 - a) ** 2 * cn / s ** 2
        a_new = np.where(CT > 0.96 * F,                                   # Buhl's Glauert correction
                         (18 * F - 20 - 3 * np.sqrt(np.abs(CT * (50 - 36 * F) + 12 * F * (3 * F - 4)))) / (36 * F - 50),
                         1 / (4 * F * s ** 2 / (sig * cn) + 1))
        ap_new = 1 / (4 * F * s * np.cos(phi) / (sig * ct) - 1)
        if np.max(np.abs(a_new - a)) < 1e-6:
            break
        a, ap = 0.7 * a + 0.3 * a_new, 0.7 * ap + 0.3 * ap_new
    W2 = ((1 - a) * U) ** 2 + ((1 + ap) * omega * r) ** 2
    dT = 0.5 * rho * W2 * chord * cn * B * dr
    dQ = 0.5 * rho * W2 * chord * ct * B * r * dr
    return float(dQ.sum() * omega), float(dT.sum())


tsr_opt, omega_max, P_rated = 7.55, 12.1 * 2 * math.pi / 60, 5.296e6     # aerodynamic power at rated (Jonkman 7.1)
rows = []
pitch = 0.0
for U in np.arange(4.0, 25.01, 1.0):
    omega = min(tsr_opt * U / R, omega_max)
    P, T = bem(U, omega, pitch)
    if P > P_rated:                     # feather from the previous pitch until the power drops to rated, then refine
        lo = pitch
        while P > P_rated and pitch < math.radians(35):
            lo, pitch = pitch, pitch + math.radians(0.5)
            P, T = bem(U, omega, pitch)
        hi = pitch
        for _ in range(25):
            pitch = 0.5 * (lo + hi)
            P, T = bem(U, omega, pitch)
            lo, hi = (pitch, hi) if P > P_rated else (lo, pitch)
    Cp = P / (0.5 * rho * math.pi * R ** 2 * U ** 3)
    rows.append((U, omega * 60 / (2 * math.pi), math.degrees(pitch), P / 1e6, T / 1e3, Cp))
print(f"{'U m/s':>6} {'rpm':>6} {'pitch':>6} {'P MW':>7} {'T kN':>7} {'Cp':>6}")
for row in rows:
    print("{:6.1f} {:6.2f} {:6.2f} {:7.3f} {:7.1f} {:6.3f}".format(*row))
cp_max = max(r_[5] for r_ in rows)
u_rated = next(r_[0] for r_ in rows if r_[3] >= 0.99 * P_rated / 1e6)
print(f"\nmax Cp {cp_max:.3f} (reference 0.482), rated reached at {u_rated:.1f} m/s (reference 11.4 m/s)")
np.savetxt(os.path.join(OUT, "nrel5mw_bem_power_curve.csv"), np.array(rows), delimiter=",",
           header="U_m_s,rpm,pitch_deg,P_MW,T_kN,Cp", comments="")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_tools.visualization.style import use_theme
    use_theme("report")   # one theme for every figure of the library (#918)
    A = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(A[:, 0], A[:, 3], "o-", label="BEM on the library blade")
    ax[0].axhline(5.0, color="k", ls="--", lw=1, label="5 MW rated (electrical)")
    ax[0].set_xlabel("wind speed [m/s]")
    ax[0].set_ylabel("power [MW]")
    ax[0].legend()
    ax[1].plot(A[:, 0], A[:, 4], "o-")
    ax[1].set_xlabel("wind speed [m/s]")
    ax[1].set_ylabel("thrust [kN]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nrel5mw_bem_power_curve.png"), dpi=130)
    print("wrote", os.path.join(OUT, "nrel5mw_bem_power_curve.png"))
except ImportError:
    pass
