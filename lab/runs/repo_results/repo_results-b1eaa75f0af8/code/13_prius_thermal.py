"""Magnet temperature, demagnetisation and a lumped thermal network of the Prius motor.

1. Performance against the magnet temperature (20 to 150 degC; Br(T) of the grade): MTPA torque at 250 A, back-EMF
   constant, L_d and L_q.
2. Temperatures over repeated WLTC class 3b cycles from the drive model's losses (copper split between slots and end
   windings by length, iron between yoke, teeth and rotor by the FEA loss density at 250 A, magnet loss to the
   magnets), and the time to reach the winding limit at peak load from the end-of-cycle state.
3. Demagnetisation at the hottest magnet temperature of 2.: the reverse field in every magnet element against
   Hcj(T) of the NdFeB grades N to UH, at full negative d-axis current (250 A) and at the peak of a three-phase short
   circuit estimated as 2 psi_pm / L_d.

Needs the flux map of ``11_prius_drive.py``. Thermal data are the stated defaults of ``MachineThermalData``
(water jacket, coolant 65 degC): illustrative, not measured on a Prius. Hcj: minimum of each grade letter at 20 degC
with a temperature coefficient of -0.55 %/K (a typical datasheet value for sintered NdFeB, an assumption here).

    python examples/electrodynamics/13_prius_thermal.py            # about 30 min
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.machines import IPMMachine, prius_2004
from pinneapple_simulation.electromagnetics.machines.drive import (
    Drive,
    FluxMap,
    Vehicle,
    cycle_energy,
    prius_phase_resistance,
    v_max_svpwm,
    wltc_class3b,
)
from pinneapple_simulation.electromagnetics.machines.thermal import (
    NDFEB_HCJ_MIN_KAM,
    demagnetisation_margin,
    demagnetisation_refinement,
    intrinsic_coercivity,
    machine_network,
    magnet_temperature_sweep,
)

OUT = Path(__file__).parent / "results"
BETA_HCJ = -0.0055
N_GAP = 960


def iron_split(m: IPMMachine) -> dict:
    """Fractions of the iron loss in the stator yoke, teeth and rotor at 250 A MTPA (50.3 deg), 1500 rpm."""
    gm = math.radians(50.3)
    r = m.losses(1500.0, -250 * math.sin(gm), 250 * math.cos(gm), steps=24)
    mesh = m.mm.mesh(0.0)
    area = mesh.areas()
    c = mesh.centroids()
    rc = np.hypot(c[:, 0], c[:, 1])
    st = m._stator_iron
    s = m.geometry.stator
    r_slot_bot = s.r_in + s.H0 + s.H1 + s.H2
    dens = r["stator_loss_density"]
    yoke = float(np.sum((area[st] * dens)[rc[st] >= r_slot_bot]))
    teeth = float(np.sum((area[st] * dens)[rc[st] < r_slot_bot]))
    rotor = float(np.sum(area[m._rotor_iron] * r["rotor_loss_density"]))
    tot = yoke + teeth + rotor
    return {"yoke": yoke / tot, "teeth": teeth / tot, "rotor": rotor / tot}


def main() -> dict:
    t0 = time.perf_counter()
    g = prius_2004()
    sweep = magnet_temperature_sweep(g, (20.0, 60.0, 100.0, 150.0), n_gap=N_GAP)
    for r in sweep:
        print(f"magnets {r['magnet_temperature_C']:5.0f} degC: Br {r['Br_T']:.3f} T, MTPA {r['torque_Nm']:.1f} N m at "
              f"{r['mtpa_gamma_deg']:.1f} deg, EMF {r['emf_phase_fundamental_peak_V_1000rpm']:.1f} V pk at 1000 rpm, "
              f"L_d {r['L_d_mH']:.3f} mH, L_q {r['L_q_mH']:.3f} mH", flush=True)

    fm = FluxMap.from_npz(OUT / "11_prius_fluxmap.npz")
    R, wdata = prius_phase_resistance(100.0)
    drive = Drive(fm, R=R, I_max=250.0, V_max=v_max_svpwm(500.0))
    t, v = wltc_class3b()
    veh = Vehicle(mass=1450.0, c_rr=0.009, cd_a=0.58, wheel_radius=0.30, gear_ratio=4.113, gear_efficiency=0.97)
    E = cycle_energy(drive, veh, t, v)
    m20 = IPMMachine(g, n_gap=N_GAP)
    split = iron_split(m20)
    end = wdata["end_turn_per_side_m"]
    f_slot = g.length / (g.length + end)
    net = machine_network(g, end_length=end)

    def node_losses(cu, fe, mg):
        return {"yoke": fe * split["yoke"], "teeth": fe * split["teeth"], "winding": cu * f_slot,
                "end_winding": cu * (1 - f_slot), "rotor": fe * split["rotor"], "magnets": mg}

    n = len(E["t"])
    P_cycle = np.array([[node_losses(E["loss_copper"][k], E["loss_iron"][k], E["loss_magnet"][k])[nm]
                         for nm in net.names] for k in range(n)])
    cycles = 4

    def P_t(time_s):
        return P_cycle[min(int(time_s), cycles * n - 1) % n]

    T0 = net.boundaries[0][2]
    times, T = net.transient(P_t, cycles * n - 1, 1.0, T0)
    hot = {nm: float(T[:, i].max()) for i, nm in enumerate(net.names)}
    print("WLTC x4 peak temperatures (degC):", {k: round(v_, 1) for k, v_ in hot.items()}, flush=True)
    w = 1500 * 2 * math.pi / 60
    op = drive.operating_point(0.999 * drive.max_torque(w)["T"], w)
    Pk = node_losses(op["copper"], op["iron"], op["magnet"])
    ts, Tp = net.transient([Pk[nm] for nm in net.names], 1800.0, 1.0, T[-1])
    hot_cu = np.maximum(Tp[:, net.index("winding")], Tp[:, net.index("end_winding")])  # the hotter winding node
    reach = np.nonzero(hot_cu >= 180.0)[0]
    t_limit = float(ts[reach[0]]) if len(reach) else None
    print(f"peak load {op['T']:.1f} N m at 1500 rpm: hottest winding node reaches 180 degC after {t_limit} s",
          flush=True)

    T_mag = float(max(hot["magnets"], Tp[:, net.index("magnets")].max()))
    m = IPMMachine(g, n_gap=N_GAP, magnet_temperature=T_mag)
    ind = m.inductances(0.0, 0.0)
    i_sc = 2.0 * ind["psi_pm_d"] / ind["L_dd"]
    cases = {"i_d = -250 A": -250.0, f"short-circuit peak 2 psi/L_d = -{i_sc:.0f} A": -i_sc}
    demag, maps = {}, {}
    for label, i_d in cases.items():
        ms = m.solve(0.0, m.currents_dq(0.0, i_d, 0.0))
        demag[label] = {}
        for grade in ("N", "M", "H", "SH", "UH"):
            Hcj = intrinsic_coercivity(NDFEB_HCJ_MIN_KAM[grade] * 1e3, T_mag, BETA_HCJ)
            d = demagnetisation_margin(m, ms, Hcj)
            demag[label][grade] = {k: d[k] for k in ("H_rev_max", "margin", "Hcj", "demagnetised_fraction")}
            print(f"{label}, grade {grade}: Hcj {Hcj / 1e3:.0f} kA/m, worst reverse field {d['H_rev_max'] / 1e3:.0f} "
                  f"kA/m, demagnetised fraction {d['demagnetised_fraction']:.4f}", flush=True)
            maps[(label, grade)] = (ms, d)
    hcj = {gr: intrinsic_coercivity(NDFEB_HCJ_MIN_KAM[gr] * 1e3, T_mag, BETA_HCJ) for gr in ("N", "M", "H", "SH", "UH")}
    study = demagnetisation_refinement(g, T_mag, cases, hcj, n_gap=N_GAP)
    for lv in study:
        print(f"grading {lv['grading']}: {lv['magnet_elements']} magnet elements, "
              f"{1e3 * lv['magnet_element_size_m']:.2f} mm;",
              {c: {gr: (round(v["ratio_max"], 3), round(v["demagnetised_fraction"], 4),
                        round(v["fraction_above_margin"], 4)) for gr, v in r.items()} for c, r in lv["cases"].items()},
              flush=True)
    res = {"magnet_temperature_sweep": sweep, "iron_split": split, "wltc_cycles": cycles,
           "peak_temperatures_C": hot,
           "mean_losses_W": {"copper": float(np.mean(E["loss_copper"])), "iron": float(np.mean(E["loss_iron"])),
                             "magnet": float(np.mean(E["loss_magnet"]))},
           "peak_load": {"torque_Nm": op["T"], "rpm": 1500, "copper_W": op["copper"], "iron_W": op["iron"],
                         "magnet_W": op["magnet"], "time_to_180C_winding_s": t_limit},
           "demagnetisation": {"magnet_temperature_C": T_mag, "beta_Hcj_per_K": BETA_HCJ,
                               "short_circuit_peak_estimate_A": i_sc, "cases": demag,
                               "mesh_refinement": study},
           "seconds": time.perf_counter() - t0}
    OUT.mkdir(exist_ok=True)
    (OUT / "13_prius_thermal.json").write_text(json.dumps(res, indent=1, default=float))
    _figures(times, T, net.names, ts, Tp, t_limit, sweep, maps, m, list(cases), study, T0)
    return res


def _figures(times, T, names, ts, Tp, t_limit, sweep, maps, m, labels, study, T0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
    for i, nm in enumerate(names):
        # rotor and magnets coincide (contact conductance ~340 W/K against ~0.2 W magnet loss): magnets dashed on top
        kw = {"ls": (0, (3, 2)), "lw": 1.6} if nm == "magnets" else {"lw": 2.6 if nm == "rotor" else 1.2}
        ax[0].plot(times / 60, T[:, i], label=nm, **kw)
        ax[1].plot(ts / 60, Tp[:, i], label=nm, **kw)
    ax[0].set_xlabel("time (min), 4 x WLTC class 3b")
    ax[0].set_ylabel("temperature (degC)")
    ax[0].legend(fontsize=8)
    ax[1].axhline(180, color="k", ls="--", lw=0.8)
    if t_limit is not None:
        ax[1].axvline(t_limit / 60, color="k", lw=0.8)
    ax[1].set_xlabel("time at peak load (min)")
    ax[1].set_ylabel("temperature (degC)")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_title("250 A, 1500 rpm from the end-of-cycle state (dashed: 180 degC)")
    Tm = [r["magnet_temperature_C"] for r in sweep]
    ax[2].plot(Tm, [r["torque_Nm"] / sweep[0]["torque_Nm"] for r in sweep], "o-", label="MTPA torque, 250 A")
    ax[2].plot(Tm, [r["emf_phase_fundamental_peak_V_1000rpm"] / sweep[0]["emf_phase_fundamental_peak_V_1000rpm"]
                    for r in sweep], "s-", label="back-EMF")
    ax[2].plot(Tm, [r["L_d_mH"] / sweep[0]["L_d_mH"] for r in sweep], "^-", label="L_d")
    ax[2].plot(Tm, [r["L_q_mH"] / sweep[0]["L_q_mH"] for r in sweep], "v-", label="L_q")
    ax[2].set_xlabel("magnet temperature (degC)")
    ax[2].set_ylabel("relative to 20 degC")
    ax[2].legend(fontsize=8)
    fig.text(0.01, 0.01, f"All nodes start at the coolant temperature {T0:.0f} degC. Boundaries: water jacket on the "
             f"stator back (h A 300 W/K, coolant {T0:.0f} degC), end windings to {T0:.0f} degC air (h 40 W/m^2 K), "
             f"rotor through shaft and bearings (2 W/K to {T0:.0f} degC). Rotor and magnet curves coincide (dashed "
             "magnets over the rotor): the magnet loss is about 0.2 W, the rotor-magnet contact about 340 W/K. "
             "Thermal data are illustrative defaults (MachineThermalData), not measured on a Prius.", fontsize=7,
             wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "13_prius_thermal.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    panels = [(label, grade) for label in labels for grade in ("H", "SH")]
    for ax, (label, grade) in zip(axs.ravel(), panels, strict=True):
        ms, d = maps[(label, grade)]
        mesh = ms.mesh
        el = d["elements"]
        bg = PolyCollection(mesh.points[mesh.tris] * 1e3, facecolors="0.85", edgecolors="none")
        ax.add_collection(bg)
        pc = PolyCollection(mesh.points[mesh.tris[el]] * 1e3, array=d["ratio"], cmap="RdYlGn_r", edgecolors="none")
        pc.set_clim(0, 1.2)
        ax.add_collection(pc)
        ax.set_xlim(40, 85)
        ax.set_ylim(-25, 25)
        ax.set_aspect("equal")
        ax.set_title(f"grade {grade}, magnets at {m.magnet_temperature:.0f} degC\n{label}\n"
                     f"demagnetised area {100 * d['demagnetised_fraction']:.2f} %", fontsize=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(pc, ax=ax, shrink=0.8, label="H_rev / Hcj (> 1: irreversible)")
    fig.tight_layout()
    fig.savefig(OUT / "13_prius_demag.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    h = np.array([1e3 * lv["magnet_element_size_m"] for lv in study])
    for label, ls in zip(labels, ("-", "--"), strict=True):
        for grade in ("M", "H", "SH"):
            r = [lv["cases"][label][grade] for lv in study]
            short = label.split(",")[0] if label.startswith("i_d") else "short circuit"
            axs[0].plot(h, [x["ratio_max"] for x in r], "o" + ls, label=f"{grade}, {short}")
            axs[1].plot(h, [100 * x["demagnetised_fraction"] for x in r], "o" + ls, label=f"{grade}, {short}")
            axs[2].plot(h, [100 * x["fraction_above_margin"] for x in r], "o" + ls, label=f"{grade}, {short}")
    for a, t in zip(axs, ("worst element H_rev / Hcj", "area with H_rev > Hcj (%)",
                          "area with H_rev > 0.9 Hcj (%)"), strict=True):
        a.set_xlabel("mean magnet element size (mm)")
        a.set_title(t, fontsize=10)
        a.invert_xaxis()
    axs[0].axhline(1.0, color="k", lw=0.6)
    axs[0].legend(fontsize=7)
    fig.text(0.01, 0.01, "Mesh refinement inside the magnets (element-size growth away from the gap reduced at a fixed "
             f"gap mesh of {N_GAP} nodes); magnets at {m.magnet_temperature:.0f} degC, rotor at 0. Solid: i_d = -250 A; "
             "dashed: short-circuit peak estimate.", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "13_prius_demag_mesh.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
