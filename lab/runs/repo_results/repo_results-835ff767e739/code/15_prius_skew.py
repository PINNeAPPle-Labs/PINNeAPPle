"""Step skew of the Prius rotor by multi-slice 2-D FEA: cogging, torque ripple and back-EMF harmonics against the skew
angle, the closed-form skew factor, and a 3-D view of the skewed rotor with |B|.

A rotor of K axial segments, segment k turned by delta_k = s (k - (K-1)/2); each segment is a 2-D slice of the same
stator carrying the same phase currents, and the machine's torque and flux linkage are the means over the slices
(``machines.robustness``).

    python examples/electrodynamics/15_prius_skew.py           # about 20 min
    python examples/electrodynamics/15_prius_skew.py --quick
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.machines import IPMMachine, prius_2004
from pinneapple_simulation.electromagnetics.machines.export import write_vtu
from pinneapple_simulation.electromagnetics.machines.robustness import (
    skewed_field_3d,
    skewed_no_load,
    skewed_torque,
    step_skew_factor,
)

OUT = Path(__file__).parent / "results"
TWO_PI = 2 * math.pi


def main(quick: bool) -> dict:
    t0 = time.perf_counter()
    m = IPMMachine(prius_2004(), n_gap=480 if quick else 960)
    slot = TWO_PI / m.geometry.stator.slots
    gm = math.radians(50.3)
    i_d, i_q = -250 * math.sin(gm), 250 * math.cos(gm)
    th_load = m.positions(6 if quick else 12)
    th_cog = m.positions(6 if quick else 12, period=m.cogging_period)
    rows = []
    fractions = (0.0, 0.5, 1.0) if quick else (0.0, 0.25, 0.5, 0.75, 1.0)
    for K in (2, 3):
        for frac in fractions:
            total = frac * slot
            s = total / (K - 1)
            cog = skewed_no_load(m, th_cog, K, s)["cogging"]
            T = skewed_torque(m, i_d, i_q, th_load, K, s)
            rows.append({"steps": K, "skew_mech_deg": math.degrees(total), "skew_slot_pitches": frac,
                         "cogging_pkpk_Nm": float(np.ptp(cog)), "torque_mean_Nm": float(T.mean()),
                         "ripple_pkpk": float(np.ptp(T) / T.mean())})
            r = rows[-1]
            print(f"K={K} skew {r['skew_mech_deg']:5.3f} deg: cogging {r['cogging_pkpk_Nm']:.3f} N m, torque "
                  f"{r['torque_mean_Nm']:.2f} N m, ripple {100 * r['ripple_pkpk']:.1f} %", flush=True)

    # back-EMF of a two-step skew by one slot pitch, directly from the slices, against the closed-form factor
    nf = 40 if quick else 120
    th_e = m.positions(nf, period=TWO_PI / m.pole_pairs)
    psi0 = np.array([m.flux_linkage(m.solve(t)) for t in th_e])
    K, s = 2, slot
    psis = skewed_no_load(m, th_e, K, s)["psi"]
    e0, es = m.back_emf(psi0, th_e, 1000.0), m.back_emf(psis, th_e, 1000.0)
    nh = min(25, len(e0["harmonics_phase"]) - 1)  # below the Nyquist order
    h = np.arange(1, nh + 1)
    H0, Hs = np.array(e0["harmonics_phase"][:nh]), np.array(es["harmonics_phase"][:nh])
    fac = step_skew_factor(h, K, s, TWO_PI / m.pole_pairs)
    sig = H0 > 1e-3 * H0[0]
    dev = np.abs(Hs[sig] - fac[sig] * H0[sig]) / H0[0]
    emf = {"steps": K, "skew_mech_deg": math.degrees(s), "positions": nf,
           "harmonics_straight_V": H0.tolist(), "harmonics_skewed_V": Hs.tolist(), "skew_factor": fac.tolist(),
           "max_deviation_from_factor_rel_fundamental": float(dev.max()),
           "fundamental_ratio": float(Hs[0] / H0[0]), "factor_fundamental": float(fac[0]),
           "thd_straight": e0["thd"], "thd_skewed": es["thd"]}
    print(f"EMF: fundamental x{emf['fundamental_ratio']:.4f} (factor {emf['factor_fundamental']:.4f}), THD "
          f"{100 * e0['thd']:.1f} % -> {100 * es['thd']:.1f} %, max deviation from the factor "
          f"{emf['max_deviation_from_factor_rel_fundamental']:.2e} of the fundamental", flush=True)

    # 3-D view: three-step skew by one slot pitch at 250 A, 50.3 deg
    f3 = skewed_field_3d(m, 0.0, m.currents_dq(0.0, i_d, i_q), 3, slot / 2)
    OUT.mkdir(exist_ok=True)
    write_vtu(OUT / "15_prius_skew_3d.vtu", f3["points"], f3["wedges"], {"Bmag": f3["Bmag"], "region": f3["region"]})
    res = {"sweep": rows, "back_emf": emf, "seconds": time.perf_counter() - t0,
           "vtu": "15_prius_skew_3d.vtu", "load_point": {"I_peak_A": 250.0, "gamma_deg": 50.3}}
    (OUT / "15_prius_skew.json").write_text(json.dumps(res, indent=1, default=float))
    _figures(rows, emf, m, f3)
    return res


def _figures(rows, emf, m, f3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for K, mk in ((2, "o-"), (3, "s-")):
        r = [x for x in rows if x["steps"] == K]
        a = [x["skew_mech_deg"] for x in r]
        ax[0].plot(a, [x["cogging_pkpk_Nm"] for x in r], mk, label=f"{K} steps")
        ax[1].plot(a, [100 * x["ripple_pkpk"] for x in r], mk, label=f"{K} steps")
    ax[0].set_xlabel("total skew (mech. deg; slot pitch 7.5)")
    ax[0].set_ylabel("cogging peak to peak (N m)")
    ax[1].set_xlabel("total skew (mech. deg)")
    ax[1].set_ylabel("torque ripple at 250 A (% pk-pk)")
    ax[0].legend()
    ax[1].legend()
    H0, Hs = np.array(emf["harmonics_straight_V"]), np.array(emf["harmonics_skewed_V"])
    h = np.arange(1, len(H0) + 1)
    ax[2].bar(h - 0.2, H0, 0.4, label="straight")
    ax[2].bar(h + 0.2, Hs, 0.4, label="2-step skew, 1 slot pitch")
    ax[2].plot(h, np.array(emf["skew_factor"]) * H0, "k_", ms=12, label="straight x skew factor")
    ax[2].set_yscale("log")
    ax[2].set_ylim(1e-2, 200)
    ax[2].set_xlabel("harmonic order")
    ax[2].set_ylabel("phase back-EMF at 1000 rpm (V)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "15_prius_skew.png", dpi=130)
    plt.close(fig)

    # 3-D: top face of each segment (rotor and stator teeth near the gap), coloured by |B|
    P, W, B = f3["points"], f3["wedges"], f3["Bmag"]
    c = P[W[:, :3]].mean(1)
    r = np.hypot(c[:, 0], c[:, 1])
    keep = r < m.geometry.rotor.r_out  # the rotor of every segment: whole circles, so no ragged edges
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    faces = P[W[keep][:, 3:]] * 1e3
    pc = Poly3DCollection(faces, cmap="viridis", linewidths=0)
    pc.set_array(B[keep])
    pc.set_clim(0, 2.2)
    ax.add_collection3d(pc)
    lim = 1e3 * m.geometry.rotor.r_out
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, 1e3 * m.length)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.view_init(elev=28, azim=-60)
    fig.colorbar(pc, ax=ax, shrink=0.6, label="|B| (T)")
    ax.set_title("3-step skewed rotor (1 slot pitch), 250 A: |B| on the top face of each segment")
    fig.savefig(OUT / "15_prius_skew_3d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main("--quick" in sys.argv)
