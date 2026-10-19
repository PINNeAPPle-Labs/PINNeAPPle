"""2004 Toyota Prius traction motor (48 slots, 8 poles, V-shaped NdFeB, M400-50A): nonlinear 2-D FEA.

Composes ``pinneapple_simulation.electromagnetics.machines``: no-load cogging and back-EMF, torque against current
angle compared with the published curve (Yang et al., IEEE IEMDC 2013), MTPA, frozen-permeability L_d / L_q,
torque ripple, iron (catalogue table and Bertotti fit) and magnet losses, figures, and the verification table.

    python examples/electrodynamics/10_pmsm_prius.py                # about 10 min on one core
    python examples/electrodynamics/10_pmsm_prius.py --quick        # coarse mesh, fewer points
    python examples/electrodynamics/10_pmsm_prius.py --write-docs   # also rewrite the table in the guide
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.electromagnetics.machines import (
    M400_50A,
    IPMMachine,
    fit_iron_loss,
    prius_2004,
)
from pinneapple_simulation.electromagnetics.machines import verification as mv
from pinneapple_simulation.electromagnetics.verification import to_markdown

OUT = Path(__file__).parent / "results"
GUIDE = Path(__file__).parents[2] / "docs" / "guides" / "electromagnetics.md"
START, END = "<!-- machine-verification:start -->", "<!-- machine-verification:end -->"


def main(quick: bool, write_docs: bool) -> dict:
    t0 = time.perf_counter()
    n_gap = 480 if quick else 960
    m = IPMMachine(prius_2004(), n_gap=n_gap)
    res: dict = {"n_gap": n_gap, "elements": int(m.mm.n_elements)}

    nl = m.no_load(12 if quick else 24, n_flux=48 if quick else 120)
    emf = m.back_emf(nl["psi"], nl["theta"], rpm=1000.0)
    res["no_load"] = {"cogging_peak_to_peak_Nm": float(np.ptp(nl["cogging"])),
                      "psi_pm_peak_Wb": float(np.max(np.abs(nl["psi"][:, 0]))),
                      "flux_positions_per_electrical_period": len(nl["theta"]),
                      "emf_line_rms_V_at_1000rpm": emf["line_rms"],
                      "emf_line_fundamental_rms_V_at_1000rpm": emf["line_fundamental_rms"],
                      "emf_phase_fundamental_peak_V_at_1000rpm": emf["fundamental_peak"],
                      "emf_thd_phase": emf["thd"], "emf_thd_line": emf["thd_line"],
                      "emf_phase_harmonics_V": emf["harmonics_phase"][:25]}

    curve = mv.prius_torque_curve(n_gap, n_positions=12)
    res["torque_vs_angle"] = {"phi0_deg": curve["phi0_deg"].tolist(), "torque_Nm": curve["torque"].tolist(),
                              "torque_dq_Nm": curve["torque_dq"].tolist(),
                              "yang_2013_Nm": mv.YANG_2013_TORQUE_NM.tolist(),
                              "comparison": mv.yang_comparison(curve)}
    for p_, t_, r_ in zip(curve["phi0_deg"], curve["torque"], mv.YANG_2013_TORQUE_NM, strict=True):
        print(f"Phi0 {p_:5.0f} deg: {t_:7.1f} N m, published {r_:5.0f}, error {t_ - r_:+6.1f} N m", flush=True)

    mtpa = []
    for amp in ((100.0, 250.0) if quick else (50.0, 100.0, 150.0, 200.0, 250.0)):
        g, _ = m.mtpa(amp, n_positions=4, tol=math.radians(1.0))
        op = m.operating_point(I_peak=amp, gamma=g, n_positions=6 if quick else 12)
        ind = m.inductances(op["i_d"], op["i_q"])
        mtpa.append({"I_peak_A": amp, "gamma_deg": math.degrees(g), "torque_Nm": op["torque_mean"],
                     "ripple_pk_pk": op["ripple"], "i_d": op["i_d"], "i_q": op["i_q"], "psi_d": op["psi_d"],
                     "psi_q": op["psi_q"], "L_d_mH": 1e3 * ind["L_dd"], "L_q_mH": 1e3 * ind["L_qq"],
                     "psi_pm_Wb": ind["psi_pm_d"]})
        print(f"MTPA {amp:5.0f} A: gamma {mtpa[-1]['gamma_deg']:5.1f} deg, T {op['torque_mean']:6.1f} N m, "
              f"ripple {100 * op['ripple']:4.1f} %, L_d {mtpa[-1]['L_d_mH']:.3f} mH, L_q {mtpa[-1]['L_q_mH']:.3f} mH",
              flush=True)
    res["mtpa"] = mtpa

    fit = fit_iron_loss(M400_50A, f_max=1000.0)
    top = mtpa[-1]
    losses = {}
    for rpm in (1500.0,):
        tab = m.losses(rpm, top["i_d"], top["i_q"], steps=24 if quick else 48)
        bert = m.losses(rpm, top["i_d"], top["i_q"], steps=24 if quick else 48, loss_model=fit.model)
        losses[f"{rpm:.0f}rpm_{top['I_peak_A']:.0f}A"] = {
            "frequency_Hz": tab["frequency_Hz"], "stator_iron_W_table": tab["stator_iron_W"],
            "rotor_iron_W_table": tab["rotor_iron_W"], "stator_iron_W_bertotti": bert["stator_iron_W"],
            "rotor_iron_W_bertotti": bert["rotor_iron_W"], "magnet_W": tab["magnet_W"]}
        keep = tab
    res["losses"] = losses
    res["bertotti_fit"] = {"k_h": fit.fitted[0], "alpha": fit.fitted[1], "beta": fit.fitted[2], "k_c": fit.fitted[3],
                           "k_e": fit.fitted[4], "rms_rel_error": fit.rms_rel_error, "max_rel_error": fit.max_rel_error}

    _figures(m, nl, emf, curve, mtpa, keep)
    results = mv.run(quick=quick)
    table = to_markdown(results)
    print(table)
    notes = "\n".join(f"- {r.name}: {r.detail}" for r in results if r.detail)
    print(notes)
    res["verification"] = [{"case": r.name, "error": r.error, "orders": r.orders, "passed": r.passed,
                            "detail": r.detail} for r in results]
    res["seconds"] = time.perf_counter() - t0
    OUT.mkdir(exist_ok=True)
    (OUT / "10_pmsm_prius.json").write_text(json.dumps(res, indent=1, default=float))
    if write_docs:
        text = GUIDE.read_text()
        i, j = text.index(START) + len(START), text.index(END)
        GUIDE.write_text(text[:i] + "\n" + table + "\n\nNotes from the same run:\n\n" + notes + "\n" + text[j:])
    return res


def _figures(m, nl, emf, curve, mtpa, loss_run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    OUT.mkdir(exist_ok=True)
    top = mtpa[-1]
    sol = m.solve(0.0, m.currents_dq(0.0, top["i_d"], top["i_q"]))
    mesh = sol.mesh
    Bmag = np.hypot(sol.B[:, 0], sol.B[:, 1])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    # the whole cross-section, cropped by the axes (no element selection, so no partial elements at the edges)
    pc = PolyCollection(mesh.points[mesh.tris] * 1e3, array=Bmag, cmap="viridis", edgecolors="none")
    pc.set_clim(0, 2.2)
    ax.add_collection(pc)
    ax.tricontour(mesh.points[:, 0] * 1e3, mesh.points[:, 1] * 1e3, mesh.tris, sol.A, levels=24, colors="w",
                  linewidths=0.5)
    t = np.linspace(0, 2 * math.pi, 721)
    g = m.geometry
    for r in (g.rotor.r_in, g.stator.r_out):
        ax.plot(1e3 * r * np.cos(t), 1e3 * r * np.sin(t), "k-", lw=0.8)
    ax.set_xlim(0, 140)
    ax.set_ylim(-70, 70)
    ax.set_aspect("equal")
    fig.colorbar(pc, ax=ax, label="|B| (T)")
    ax.set_title(f"Prius motor, MTPA at {top['I_peak_A']:.0f} A: |B| and flux lines")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.savefig(OUT / "10_pmsm_field.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    ax[0].plot(curve["phi0_deg"], curve["torque"], "o-", label="this FEA (Maxwell stress)")
    ax[0].plot(curve["phi0_deg"], mv.YANG_2013_TORQUE_NM, "s--", label="Yang et al. 2013")
    ax0b = ax[0].twinx()
    ax0b.bar(curve["phi0_deg"], curve["torque"] - mv.YANG_2013_TORQUE_NM, width=4, alpha=0.3, color="gray")
    ax0b.set_ylabel("difference (N m), bars")
    ax[0].set_xlabel("current angle Phi0 (deg)")
    ax[0].set_ylabel("mean torque (N m)")
    ax[0].set_title("250 A peak")
    ax[0].legend()
    ax[1].plot(np.degrees(nl["theta_cogging"]), nl["cogging"], "o-")
    ax[1].set_xlabel("rotor angle (mech. deg)")
    ax[1].set_ylabel("cogging torque (N m)")
    ax[1].set_title("one cogging period")
    th = np.degrees(nl["theta"]) * m.pole_pairs
    for k, c in enumerate("ABC"):
        ax[2].plot(th, emf["e"][:, k], label=c)
    ax[2].set_xlabel("electrical angle (deg)")
    ax[2].set_ylabel("phase back-EMF at 1000 rpm (V)")
    ax[2].set_title(f"{len(nl['theta'])} rotor angles per electrical period")
    ax[2].legend()
    h = np.array(emf["harmonics_phase"][:25])
    ax[3].bar(np.arange(1, len(h) + 1), 100 * h / h[0])
    ax[3].set_yscale("log")
    ax[3].set_ylim(0.01, 200)
    ax[3].set_xlabel("harmonic order")
    ax[3].set_ylabel("phase back-EMF harmonic (% of fundamental)")
    ax[3].set_title(f"THD {100 * emf['thd']:.1f} % (phase), {100 * emf['thd_line']:.1f} % (line)")
    fig.tight_layout()
    fig.savefig(OUT / "10_pmsm_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot([p["i_d"] for p in mtpa], [p["i_q"] for p in mtpa], "o-")
    ax[0].set_xlabel("i_d (A)")
    ax[0].set_ylabel("i_q (A)")
    ax[0].set_title("MTPA locus")
    ax[0].set_aspect("equal")
    ax[1].plot([p["I_peak_A"] for p in mtpa], [p["L_d_mH"] for p in mtpa], "o-", label="L_d")
    ax[1].plot([p["I_peak_A"] for p in mtpa], [p["L_q_mH"] for p in mtpa], "s-", label="L_q")
    ax[1].set_xlabel("current amplitude (A)")
    ax[1].set_ylabel("frozen-permeability inductance (mH)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "10_pmsm_mtpa.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    r = main("--quick" in sys.argv, "--write-docs" in sys.argv)
    print(json.dumps({k: v for k, v in r.items() if k in ("no_load", "losses", "bertotti_fit", "seconds")},
                     indent=1, default=float))
    sys.exit(0 if all(v["passed"] for v in r["verification"]) else 1)
