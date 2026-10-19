"""End effects of the Prius motor, outside the 2-D cross-section.

1. End-winding leakage inductance of both coil ends (Neumann's formula on the coil-end conductors,
   ``machines.endwinding``), in air and with the stator end face as an iron mirror; the end-turn length is the one
   of the phase resistance.
2. Its effect on the drive: torque envelope and efficiency at high speed with ``Drive(..., L_end=...)``.
3. End leakage of the magnet flux: a 3-D magnetostatic model of half the stack with air beyond the end face (gmsh +
   GetDP), against the same extruded model without it, on three meshes and with a longer end-air region.
4. The correction this gives to the 2-D model: the remanence scaled by the 3-D ratio in the 2-D FEA (the magnet flux
   at no load falls by that ratio; saturation and the reluctance torque follow), MTPA torque at 250 A and back-EMF.

    python examples/electrodynamics/16_prius_end_effects.py          # needs gmsh and getdp; about 2 h
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.machine import belt_winding
from pinneapple_simulation.electromagnetics.machines import (
    M400_50A,
    MAGNET_PRIUS,
    IPMMachine,
    prius_2004,
)
from pinneapple_simulation.electromagnetics.machines import getdp as G
from pinneapple_simulation.electromagnetics.machines.drive import (
    Drive,
    FluxMap,
    prius_phase_resistance,
    v_max_svpwm,
)
from pinneapple_simulation.electromagnetics.machines.endwinding import end_winding_inductance

OUT = Path(__file__).parent / "results"
RPM = 2 * math.pi / 60


def main() -> dict:
    t0 = time.perf_counter()
    g = prius_2004()
    ew = {ppm: end_winding_inductance(g, belt_winding(48, 4), 9.0, 6, 0.019366, pieces_per_metre=ppm)
          for ppm in (150, 300)}
    e = ew[300]
    print(f"end winding (both ends): L_aa {1e6 * e['air']['L_aa']:.1f} uH, M_ab {1e6 * e['air']['M_ab']:.1f} uH, "
          f"dq {1e6 * e['air']['L_end_dq']:.1f} uH in air; dq {1e6 * e['iron_face']['L_end_dq']:.1f} uH with the iron "
          f"face", flush=True)

    fm = FluxMap.from_npz(OUT / "11_prius_fluxmap.npz")
    R, _ = prius_phase_resistance(100.0)
    speeds = np.array([1000, 2000, 3000, 4000, 5000, 6000]) * RPM
    env = {}
    for label, L in (("2d", 0.0), ("air", e["air"]["L_end_dq"]), ("iron_face", e["iron_face"]["L_end_dq"])):
        d = Drive(fm, R=R, I_max=250.0, V_max=v_max_svpwm(500.0), L_end=L)
        env[label] = [d.max_torque(w)["T"] for w in speeds]
        print(label, [round(x, 1) for x in env[label]], flush=True)
    d0 = Drive(fm, R=R, I_max=250.0, V_max=v_max_svpwm(500.0))
    d1 = Drive(fm, R=R, I_max=250.0, V_max=v_max_svpwm(500.0), L_end=e["air"]["L_end_dq"])
    eff = {}
    for rpm, T in ((4000, 100.0), (6000, 60.0)):
        a, b = d0.operating_point(T, rpm * RPM), d1.operating_point(T, rpm * RPM)
        eff[f"{rpm}rpm_{T:.0f}Nm"] = {"2d": a["efficiency"], "with_end_winding": b["efficiency"],
                                      "i_d_2d": a["i_d"], "i_d_end": b["i_d"]}

    levels = []
    for n_gap, ls, la, ea in ((144, 4, 4, 0.04), (240, 6, 6, 0.04), (336, 8, 8, 0.04), (240, 6, 9, 0.06)):
        r = G.end_leakage_3d(g, M400_50A, 1.24, 1.05, n_gap=n_gap, layers_stack=ls, end_air=ea, layers_air=la)
        r.update({"n_gap": n_gap, "layers_stack": ls, "layers_air": la, "end_air_m": ea})
        r.pop("workdir", None)
        levels.append(r)
        print(f"3-D n_gap {n_gap}, {ls} layers, end air {1e3 * ea:.0f} mm: magnet flux ratio {r['ratio']:.4f} "
              f"({r['end_air']['prisms']} prisms)", flush=True)
    k = levels[2]["ratio"]

    corr = {}
    for label, scale in (("2d", 1.0), ("end_leakage", k)):
        m = IPMMachine(g, magnet=dataclasses.replace(MAGNET_PRIUS, Br20=MAGNET_PRIUS.Br20 * scale), n_gap=960)
        gm, T = m.mtpa(250.0, n_positions=12)
        nl = m.no_load(12, n_flux=120)
        emf = m.back_emf(nl["psi"], nl["theta"], 1000.0)
        corr[label] = {"Br_T": m.Br, "mtpa_gamma_deg": math.degrees(gm), "torque_Nm": T,
                       "emf_fundamental_peak_V_1000rpm": emf["fundamental_peak"]}
        print(label, corr[label], flush=True)
    res = {"end_winding": {str(k_): v for k_, v in ew.items()}, "envelope_rpm": (speeds / RPM).tolist(),
           "envelope_Nm": env, "efficiency_points": eff, "magnet_end_leakage_3d": levels,
           "flux_correction": {"half_stack_m": g.length / 2, "ratio_finest": k,
                               "note": "the 3-D half stack includes one end; the full stack has two, so the ratio "
                                       "applies to the whole machine"},
           "corrected_2d": corr,
           "seconds": time.perf_counter() - t0}
    OUT.mkdir(exist_ok=True)
    (OUT / "16_prius_end_effects.json").write_text(json.dumps(res, indent=1, default=float))
    _figure(res)
    return res


def _figure(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    rpm = res["envelope_rpm"]
    for label, name in (("2d", "2-D model"), ("air", "+ end windings in air"), ("iron_face", "+ end windings, iron face")):
        ax[0].plot(rpm, res["envelope_Nm"][label], "o-", label=name)
    ax[0].set_xlabel("speed (rpm)")
    ax[0].set_ylabel("maximum torque, 250 A, 500 V (N m)")
    ax[0].legend(fontsize=8)
    lv = [x for x in res["magnet_end_leakage_3d"] if x["end_air_m"] == 0.04]
    ax[1].plot([x["end_air"]["prisms"] for x in lv], [x["ratio"] for x in lv], "o-", label="end air 40 mm")
    x6 = [x for x in res["magnet_end_leakage_3d"] if x["end_air_m"] != 0.04]
    ax[1].plot([x["end_air"]["prisms"] for x in x6], [x["ratio"] for x in x6], "s", label="end air 60 mm")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("prisms in the 3-D model")
    ax[1].set_ylabel("3-D magnet flux / extruded 2-D")
    ax[1].legend(fontsize=8)
    c = res["corrected_2d"]
    q = ["torque_Nm", "emf_fundamental_peak_V_1000rpm"]
    ax[2].bar(np.arange(2) - 0.2, [c["2d"][k] for k in q], 0.4, label="2-D")
    ax[2].bar(np.arange(2) + 0.2, [c["end_leakage"][k] for k in q], 0.4, label="2-D with Br x 3-D ratio")
    ax[2].set_xticks(range(2), ["MTPA torque at 250 A (N m)", "EMF at 1000 rpm (V peak)"])
    for i, k in enumerate(q):
        ax[2].text(i + 0.2, c["end_leakage"][k], f"{100 * (c['end_leakage'][k] / c['2d'][k] - 1):+.1f} %",
                   ha="center", va="bottom", fontsize=8)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "16_prius_end_effects.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
