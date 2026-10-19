"""NASA VIPER on the Moon: real-scale CAD, Wong-Reece wheels on GRC-1, multibody dynamics over a crater.

A thin script over library features (every model, solver and renderer lives in the library):

    pinneapple_simulation.vehicle.viper          VIPER from public figures (VIPER_SOURCES), parametric CAD
    pinneapple_simulation.vehicle                multibody model, simulator, scenarios, renders and animations
    pinneapple_simulation.terramechanics         soils with provenance, Wong-Reece wheel, testbed, PINN surrogate
    pinneapple_tools.visualization.studio        lunar lighting presets, shadows, planetary renders

Writes docs/assets/lunar_rover/ (outputs/viper/media with --quick) and outputs/viper/ (CSV, glTF/STL, surrogate):

    viper_beauty.png, viper_side.png        CAD on regolith, low Sun (lunar_south_pole preset)
    viper.glb / viper.stl                   CAD with PBR materials (glb in docs, stl in outputs)
    testbed_viper.png                       single-wheel curves of the VIPER wheel on GRC-1 and JSC-1A
    validation_single_wheel.png             model vs published single-wheel tests: cited soils (dashed) and the
                                            slip-sinkage calibration on every other point (solid), held-out RMSE
    surrogate.png                           PINN vs Wong-Reece on the held-out set
    grade_sweep.png                         slip, speed and torque vs grade (full vehicle)
    crater_series.png, crossslope_series.png  per-wheel time series of the dynamic runs
    crater_crossing.gif / .mp4              the crater run, 3-D ruts shaded by the low Sun
    results.json                            every number quoted in docs/core_concepts/lunar_rover_terramechanics.md

    python examples/use_cases/terramechanics/lunar_rover_3d.py            # about 25 min on 4 CPU cores
    python examples/use_cases/terramechanics/lunar_rover_3d.py --quick    # shorter runs, fewer frames
Rendering needs pyvista (and a display or xvfb-run on a headless machine).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pinneapple_simulation.terramechanics import WongReeceModel, get_soil, testbed  # noqa: E402
from pinneapple_simulation.terramechanics.surrogate import (  # noqa: E402
    compare_on_states,
    train_contact_surrogate,
)
from pinneapple_simulation.vehicle import (  # noqa: E402
    VehicleSimulator,
    build_vehicle,
    scenarios,
    viper,
    viz,
)

OUT = HERE / "outputs" / "viper"


def log(*a):
    print(*a, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main(quick: bool = False):
    t0 = time.time()
    media = OUT / "media" if quick else ROOT / "docs" / "assets" / "lunar_rover"
    OUT.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)
    R: dict = {}

    # ------------------------------------------------------------------ the vehicle
    spec = viper()
    tot = spec.cad.total()
    R["vehicle"] = {"name": spec.name, "mass_kg": tot["mass"], "com_m": tot["com"],
                    "inertia_about_com_kg_m2": np.diag(tot["inertia"]), "track_m": spec.track(),
                    "wheelbase_m": spec.wheelbase(), "envelope_m": np.subtract(*spec.cad.bounds()[::-1]),
                    "bodies": {b: {"mass_kg": v["mass"], "com_m": v["com"]}
                               for b, v in spec.cad.mass_properties().items()},
                    "sources": spec.sources}
    spec.cad.export(str(media / "viper.glb"))
    spec.cad.export(str(OUT / "viper.stl"))
    # front three-quarter view (mast, camera head, TRIDENT drill) and side view, the low Sun turned onto the faces seen
    viz.beauty(spec, str(media / "viper_beauty.png"), camera=((4.3, -3.9, 1.75), (0.1, 0.0, 0.95)),
               sun_azimuth_deg=-55.0, title="NASA VIPER, parametric model from public figures")
    viz.beauty(spec, str(media / "viper_side.png"), camera=((0.4, -5.6, 1.15), (0.0, 0.0, 1.05)),
               sun_azimuth_deg=-35.0, title="")
    log("CAD:", {k: R["vehicle"][k] for k in ("mass_kg", "track_m", "wheelbase_m", "envelope_m")})

    # ------------------------------------------------------------------ single wheel: VIPER wheel and validation
    W = tot["mass"] * 1.62 / 4
    slips = np.linspace(0.0, 0.8, 33)
    curves = {}
    for name in ("GRC-1", "JSC-1A"):
        curves[name] = WongReeceModel(get_soil(name), spec.wheel).steady_state(np.full_like(slips, W), slips)
    viz.plot_testbed(curves, str(media / "testbed_viper.png"),
                     f"VIPER wheel (R 0.25 m, b 0.20 m, 24 grousers), load {W:.0f} N (lunar g)")
    R["testbed_viper"] = {k: {q: np.round(v[q], 4) for q in ("slip", "drawbar_pull", "torque", "sinkage")}
                          for k, v in curves.items()}
    reps, cals = [], []
    for ds, wk in testbed.list_datasets():
        d = testbed.load_dataset(ds, wk)
        cals.append(testbed.calibrate_holdout(d))          # n0, n1 (slip-sinkage) and K on every other point
        reps.append(cals[-1]["report_before"])
    testbed.plot_validation(reps, str(media / "validation_single_wheel.png"),
                            "Single-wheel tests: slip-sinkage calibrated on every other point, the rest held out",
                            calibrated=cals)
    R["validation"] = {r.data.label: r.summary() for r in reps}
    R["slip_sinkage_calibration"] = {
        c["dataset"]: {"params": c["params"], "at_bound": c["at_bound"], "train_slips": reps[i].data.slip[c["train"]],
                       "held_out_slips": reps[i].data.slip[~c["train"]], "held_out_before": c["test_before"],
                       "held_out_after": c["test_after"], "train_after": c["train_after"]}
        for i, c in enumerate(cals)}
    log("held-out sinkage rmse [mm] before -> after:",
        {c["dataset"]: (round(1e3 * c["test_before"]["sinkage_m"]["rmse"], 2),
                        round(1e3 * c["test_after"]["sinkage_m"]["rmse"], 2)) for c in cals})
    log("held-out DP rmse / load before -> after:",
        {c["dataset"]: (round(c["test_before"]["drawbar_pull_N"]["rmse_rel_scale"], 3),
                        round(c["test_after"]["drawbar_pull_N"]["rmse_rel_scale"], 3)) for c in cals})

    # ------------------------------------------------------------------ PINN surrogate for the VIPER wheel
    wr = WongReeceModel(get_soil("GRC-1"), spec.wheel)
    sur, rep = train_contact_surrogate(wr, (-0.4, 0.9), (0.002, 0.06), epochs=800 if quick else 4000)
    sur.save(str(OUT / "viper_contact_surrogate"))
    R["surrogate"] = {k: v for k, v in rep.items() if k != "history"}
    log("surrogate held-out:", {k: round(v["rmse_over_range"], 5) for k, v in rep["test"].items()})

    # ------------------------------------------------------------------ full vehicle: grades, cross slope, crater
    grades = (0, 5, 10, 15) if quick else (0, 5, 10, 12.5, 15, 17.5, 20)
    R["grade_sweep"] = scenarios.grade_sweep(viper, "GRC-1", grades, duration=8.0 if quick else 14.0, workers=4)
    log("grades:", [(g["grade_deg"], round(g["slip_mean"], 3)) for g in R["grade_sweep"]])
    _plot_grades(R["grade_sweep"], media / "grade_sweep.png")

    soil = get_soil("GRC-1")
    cross = scenarios.flat_site(x_range=(-2, 6), slope_deg=10.0, slope_dir_deg=90.0)
    run_x = VehicleSimulator(build_vehicle(spec), cross, soil).run(8.0 if quick else 20.0, speed=0.1)
    run_x.to_csv(str(OUT / "crossslope.csv"))
    viz.plot_run(run_x, str(media / "crossslope_series.png"), "10 deg cross slope at 0.1 m/s on GRC-1 (heading hold)")
    R["cross_slope_10deg"] = run_x.summary() | {
        "slip_angle_deg_mean": np.degrees(run_x.wheel["slip_angle"][run_x.t > 3].mean(0)),
        "lateral_N_mean": run_x.wheel["lateral"][run_x.t > 3].mean(0), "corners": run_x.corner_names}

    site = scenarios.crater_site()
    run_c = VehicleSimulator(build_vehicle(spec), site, soil).run(20.0 if quick else 45.0, speed=0.2)
    run_c.to_csv(str(OUT / "crater.csv"))
    viz.plot_run(run_c, str(media / "crater_series.png"), "Crater crossing at 0.2 m/s on GRC-1 (rock under the left "
                                                           "wheels, crater D 4 m, depth 0.32 m)")
    R["crater"] = run_c.summary()
    log("crater:", {k: round(v, 3) for k, v in R["crater"].items()})

    # surrogate against the reference at the states met along the crater run (same inputs)
    cmp_ = compare_on_states(sur, wr, run_c.wheel["slip"], run_c.wheel["sinkage"])
    R["surrogate_along_crater_run"] = {"errors_inside_box": cmp_["errors_inside"],
                                       "fraction_outside_box": cmp_["fraction_outside"]}
    _plot_surrogate(sur, wr, rep, media / "surrogate.png")

    # ------------------------------------------------------------------ animation
    log("animation ...")
    viz.animate(run_c, site, str(media / "crater_crossing.gif"), mp4=str(media / "crater_crossing.mp4"),
                field=None, frames=50 if quick else 150, fps=15, size=(960, 540),
                progress=lambda j, n: log(f"  frame {j}/{n}") if j % 25 == 0 else None)
    R["wall_time_s"] = time.time() - t0
    with open(media / "results.json", "w") as f:
        json.dump(jsonable(R), f, indent=1)
    shutil.copy(media / "results.json", OUT / "results.json")
    log(f"done in {R['wall_time_s']:.0f} s -> {media}")
    return R


def _plot_grades(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = [r["grade_deg"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(g, [r["slip_mean"] for r in rows], "o-", color="#2a78d6")
    ax[0].axhline(0.4, color="#9a9893", lw=0.8, ls="--")
    ax[0].text(0.3, 0.385, "40 % slip, VIPER requirement at 15 deg", fontsize=7.5, color="#52514e", va="top")
    ax[1].plot(g, [r["speed_m_s"] for r in rows], "o-", color="#eb6834")
    ax[2].plot(g, [r["drive_torque_Nm"] for r in rows], "o-", color="#1baf7a", label="mean")
    ax[2].plot(g, [r["max_drive_torque_Nm"] for r in rows], "s--", color="#1baf7a", ms=4, label="peak")
    ax[2].legend(fontsize=8, frameon=False)
    for a, t in zip(ax, ("Mean slip [-]", "Speed (command 0.1 m/s) [m/s]", "Drive torque per wheel [N m]"),
                    strict=True):
        a.set_title(t, fontsize=10, loc="left")
        a.set_xlabel("grade [deg]", fontsize=9)
        a.grid(True, color="#e4e3df", lw=0.6)
    fig.suptitle("VIPER climbing uniform grades of GRC-1 (multibody, lunar g)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_surrogate(sur, wr, rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = np.linspace(-0.4, 0.9, 131)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for z, c in ((0.01, "#2a78d6"), (0.025, "#eb6834"), (0.045, "#1baf7a")):
        ref = wr.forces(s, np.full_like(s, z)).as_array()
        pr = sur.predict(s, np.full_like(s, z))
        for j in range(3):
            ax[j].plot(s, ref[:, j], color=c, lw=2.2, alpha=0.45, label=f"Wong-Reece, z = {1e3 * z:.0f} mm")
            ax[j].plot(s, pr[:, j], color=c, lw=1.0, ls="--", label="PINN" if z == 0.01 else None)
    for j, (t, u) in enumerate((("Fx", "N"), ("Fz", "N"), ("My", "Nm"))):
        e = rep["test"][t]
        ax[j].set_title(f"{t} [{u.replace('Nm', 'N m')}]   held-out rmse {e[f'rmse_{u}']:.2f} "
                        f"({100 * e['rmse_over_range']:.2f} % of range)", fontsize=9, loc="left")
        ax[j].set_xlabel("slip [-]", fontsize=9)
        ax[j].grid(True, color="#e4e3df", lw=0.6)
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    main(quick=a.quick)
