"""3-D multiphysics optimisation of the Prius-class interior-PM motor on its pretrained surrogate: a design search
of thousands of candidates in seconds, with no FEA per candidate.

The script walks the whole chain, with the numbers printed at every step:

1. the reference model (2-D magnetostatic FEA per axial slice, skew, iron/magnet/copper loss, axial thermal network,
   magnet temperature back on torque and demagnetisation) on the baseline design;
2. the pretrained surrogate on the same design, next to it;
3. NSGA-II on the surrogate alone -- torque density and efficiency up, magnet mass down, winding and magnet
   temperature, ripple and the demagnetisation ratio as limits -- with the reference solves it cost (zero);
4. the front re-checked with the reference model, which is where FEA is spent, and by how much the surrogate was
   off;
5. the axial picture: where the hot spot is along the stack, for the baseline and for the best design.

Writes PNGs, ``results/summary.json`` and a ``results/front.csv``. Runs in about a minute on one core once the
pretrained model is present; ``--train`` builds a small surrogate from scratch instead (a few minutes).

    python examples/motor_magnetics/03_multiphysics3d_surrogate.py
    python examples/motor_magnetics/03_multiphysics3d_surrogate.py --train
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pinneapple_simulation.electromagnetics.machines.motor3d import (  # noqa: E402
    PRIUS,
    Motor3DDesign,
    Motor3DModel,
    SurrogateConfig,
    build_dataset,
    design_problem,
    load_pretrained,
    optimize,
    verify_front,
)

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def main(train: bool = False, budget: int = 2000) -> dict:
    t0 = time.time()

    # --- 1. the reference model on the baseline ------------------------------------------------------
    ref = Motor3DModel()
    base = ref.solve(PRIUS)
    print(f"reference model ({ref.description})")
    for k in ("torque_Nm", "ripple", "efficiency", "T_winding_C", "T_magnet_C", "demag_ratio", "magnet_mass_kg",
              "active_mass_kg"):
        print(f"  {k:22s} {base[k]:10.3f}")
    print(f"  one design: {base['seconds']:.2f} s")

    # --- 2. the surrogate on the same design ---------------------------------------------------------
    if train:
        ds = build_dataset(240, seed=0, model=ref, workers=1)
        from pinneapple_simulation.electromagnetics.machines.motor3d import MotorSurrogate3D

        surrogate = MotorSurrogate3D(SurrogateConfig(ensemble=4)).fit(ds)
        print(f"trained a surrogate on {len(ds)} designs: held-out mean relative error "
              f"{100 * surrogate.card_['mean_rel_overall']:.2f} %")
    else:
        surrogate = load_pretrained()
        print(f"pretrained surrogate ({len(surrogate.models)} networks, version {surrogate.card_['version']})")
    mean, spread = surrogate.predict_one(PRIUS)
    print(f"  surrogate torque     {mean['torque_Nm']:.3f} +/- {spread['torque_Nm']:.3f} N m "
          f"(reference {base['torque_Nm']:.3f})")
    print(f"  surrogate T_winding  {mean['T_winding_C']:.2f} +/- {spread['T_winding_C']:.2f} degC "
          f"(reference {base['T_winding_C']:.2f})")

    # --- 3. the optimisation, on the surrogate only --------------------------------------------------
    t1 = time.perf_counter()
    out = optimize(design_problem(surrogate=surrogate), budget=budget, seed=0)
    print(f"\nNSGA-II on the surrogate: {out['n_evaluations']} designs in {out['wall_seconds']:.2f} s "
          f"({out['designs_per_second']:.0f} designs/s), {out['n_reference_solves']} FEA calls")
    print(f"  designs on the front: {len(out['front'])}")
    if out["best_torque_density"]:
        b = out["best_torque_density"]
        print(f"  best torque density {b['torque_density_Nm_kg']:.2f} N m/kg: W4={b['W4'] * 1e3:.1f} mm "
              f"H3={b['H3'] * 1e3:.2f} mm L={b['L'] * 1e3:.1f} mm skew={b['skew_deg']:.1f} deg "
              f"rpm={b['rpm']:.0f} flow={b['flow_L_min']:.1f} L/min")
        print(f"    torque {b['torque_Nm']:.1f} N m, magnet {b['magnet_mass_kg']:.2f} kg, "
              f"T_winding {b['T_winding_C']:.0f} degC, demag {b['demag_ratio']:.2f}")
    print(f"  wall time of the search: {time.perf_counter() - t1:.2f} s")

    # --- 4. the front re-checked with the reference model --------------------------------------------
    ver = verify_front(out, reference=ref, n=6)
    print(f"\nre-checked {ver['n_checked']} designs of the front with the reference model "
          f"({ver['reference_seconds']:.1f} s): {ver['still_feasible']} still feasible, "
          f"rank agreement {ver['rank_agreement']:.3f}")
    for k in ("torque_Nm", "T_winding_C", "demag_ratio", "efficiency"):
        e = ver["errors"][k]
        print(f"  {k:14s} mean {100 * e['mean_rel']:5.2f} %  max {100 * e['max_rel']:5.2f} %")

    # --- 5. figures ----------------------------------------------------------------------------------
    E = out["result"].evaluations
    ok = E.feasible & ~E.relaxed
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(E.Y["torque_density_Nm_kg"][~ok], E.Y["efficiency"][~ok], s=6, c="0.8", label="outside the limits")
    ax[0].scatter(E.Y["torque_density_Nm_kg"][ok], E.Y["efficiency"][ok], s=6, c="C0", alpha=0.5, label="feasible")
    idx = out["result"].pareto()
    ax[0].scatter(E.Y["torque_density_Nm_kg"][idx], E.Y["efficiency"][idx], s=26, c="C3", marker="*",
                  label="Pareto front")
    ax[0].set(xlabel="torque density [N m/kg]", ylabel="efficiency [-]", title="Surrogate search")
    ax[0].legend(fontsize=8)
    z = np.linspace(-0.5, 0.5, 4)
    for label, d, c in (("baseline", PRIUS, "0.4"), ("best torque density", Motor3DDesign.from_vector(E.X[
            out["result"].best(0)]), "C3")):
        p = surrogate.predict(d.as_vector()[None, :])
        ax[1].plot(z, p["T_winding_profile_C"][0][0], "o-", color=c, label=f"{label}: winding")
        ax[1].plot(z, p["T_magnet_profile_C"][0][0], "s--", color=c, label=f"{label}: magnets")
    ax[1].set(xlabel="axial position z / L [-]", ylabel="temperature [degC]", title="Along the stack")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "motor3d_surrogate.png", dpi=130)
    plt.close(fig)

    summary = {"reference": ref.info(), "baseline": {k: float(np.max(base[k])) for k in
                                                     ("torque_Nm", "efficiency", "T_winding_C", "T_magnet_C",
                                                      "demag_ratio", "magnet_mass_kg", "active_mass_kg",
                                                      "torque_density_Nm_kg")},
               "surrogate_card": {k: surrogate.card_[k] for k in ("version", "n_total", "n_train", "n_val",
                                                                  "mean_rel_overall", "worst_output")},
               "surrogate_baseline": mean, "optimization": {k: out[k] for k in
                                                            ("n_evaluations", "wall_seconds", "designs_per_second",
                                                             "n_reference_solves")},
               "best_torque_density": out["best_torque_density"], "verification": {
                   "n_checked": ver["n_checked"], "still_feasible": ver["still_feasible"],
                   "rank_agreement": ver["rank_agreement"],
                   "errors": {k: ver["errors"][k] for k in ("torque_Nm", "T_winding_C", "demag_ratio")}},
               "seconds": round(time.time() - t0, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    if out["front"]:
        import csv

        with (OUT / "front.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out["front"][0]))
            w.writeheader()
            w.writerows(out["front"])
    print(f"\nwrote {OUT}/motor3d_surrogate.png, summary.json, front.csv  ({summary['seconds']} s)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train a small surrogate instead of loading the pretrained")
    ap.add_argument("--budget", type=int, default=2000)
    main(**vars(ap.parse_args()))
