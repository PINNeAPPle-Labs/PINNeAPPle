"""Regenerate (and optionally re-execute) the two demonstration notebooks.

    python examples/wind_farm_penmanshiel/notebooks/build_notebooks.py
    python examples/wind_farm_penmanshiel/notebooks/build_notebooks.py --execute

The source of the notebooks is here, as Python strings, so a change to an API or
a number is a one-line edit in a file a reviewer can diff -- not a JSON patch
against a checked-in ``.ipynb``. ``--execute`` needs ``nbformat``, ``nbclient``
and ``ipykernel``; without it the script only rewrites the cell sources and the
committed outputs are left as they are.

The notebooks read the Penmanshiel archives (see the example's README) and reuse
the pickle cache that ``run_poc.py`` fills.
"""
from __future__ import annotations

import argparse
import json
import pathlib


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


PREAMBLE = '''
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt

from pinneapple_systems import wind
from pinneapple_systems.wind import figures

warnings.filterwarnings("ignore", category=FutureWarning)

RAW = Path(os.environ.get("PENMANSHIEL_RAW", Path.home() / "pn-wind-data" / "raw"))
CACHE = Path(os.environ.get("PENMANSHIEL_CACHE", Path.home() / "pn-wind-data" / "cache"))
assert (RAW / "Penmanshiel_WT_static.csv").exists(), (
    f"no Penmanshiel data in {RAW}; see examples/wind_farm_penmanshiel/README.md for the download"
)
RATED, ROTOR, HUB = 2050.0, 82.0, 59.0
'''

TURBINE_CELLS = [
    md("""
# One turbine: five power curves and the price of physics

This notebook fits five power models to one Penmanshiel turbine on a **training
window only** and scores them on a later window that no model saw. It runs the
same pipeline as `run_poc.py`, in fewer cells and without the farm.

Data: Plumley & Takeuchi, *Penmanshiel wind farm data*, Zenodo,
[doi:10.5281/zenodo.16807304](https://doi.org/10.5281/zenodo.16807304), CC-BY-4.0.
The download commands are in `examples/wind_farm_penmanshiel/README.md`.
"""),
    md("## Setup"),
    code(PREAMBLE),
    md("""
## Load one turbine and apply the recorded operating mask

The export carries the operator's own availability, downtime and curtailment
columns, so the mask is evidence rather than a heuristic. Every rule reports how
much it removed.
"""),
    code('''
TURBINE = "T01"
raw = wind.load_turbine(RAW, TURBINE, cache_dir=CACHE)
raw = wind.add_density_correction(raw, elevation_m=204.0)

report = wind.qc_report(raw, rated_power_kw=RATED)
print(f"{report['rows']:,} rows, {report['span']['start'][:10]} to {report['span']['end'][:10]}")
print(f"operating: {report['operating_samples']:,} ({report['operating_fraction']:.1%})"
      f"   curtailed: {report['curtailed_samples']:,}")
print("mask provenance:", report["mask_provenance"]["curtailment_columns"],
      report["mask_provenance"]["downtime_columns"])

raw["operating"] = wind.operating_mask(raw, rated_power_kw=RATED)
data = raw[raw["operating"]]

# Region 2 only: below rated speed the machine is chasing maximum Cp, above it
# the pitch controller is deliberately spilling power, so a Cp averaged over both
# regions measures the controller, not the rotor.
region2 = data[(data.u_corrected > 5.0) & (data.u_corrected < 9.0)]
cp = region2.power_kw / (0.5 * wind.RHO_REF * np.pi * (ROTOR / 2) ** 2 * region2.u_corrected ** 3 / 1000)
print(f"median Cp in region 2 (5-9 m/s): {cp.median():.3f}   "
      f"implied by {len(region2):,} samples, against the Betz limit {wind.betz_limit():.3f}")
'''),
    code('''
fig = figures.plot_wind_rose(data, title=f"{TURBINE}: nacelle wind speed")
plt.show()
'''),
    md("""
## The split

Train on everything before July 2017, validate on July-August, test on
September-December. Nothing is shuffled and no lagged target is used: the test
window is 3-15 months after the last training sample.
"""),
    code('''
TRAIN_END, VAL_END = pd.Timestamp("2017-07-01"), pd.Timestamp("2017-09-01")
train = data[data.timestamp < TRAIN_END]
val = data[(data.timestamp >= TRAIN_END) & (data.timestamp < VAL_END)]
test = data[data.timestamp >= VAL_END]
print(f"train {len(train):,}   validation {len(val):,}   test {len(test):,}")
'''),
    md("""
## Fit the five models

- **reference** — the analytic curve a datasheet carries (cubic to rated, Cp = 0.45);
- **IEC binned** — the 61400-12-1 measured curve, 0.5 m/s bins;
- **logistic** — five parameters, the smooth fit most commercial tools use;
- **physics-informed** — a monotone hinge network whose constraints hold by construction;
- **boosted trees** — the conventional ML baseline, on the same input.

Plus the operator's own `Potential power default PC` column as a yardstick that
this package did **not** fit.
"""),
    code('''
from pinneapple_systems import wind
from pinneapple_systems.wind.models import feature_matrix

speed, power = train.u_corrected.to_numpy(float), train.power_kw.to_numpy(float)

reference = wind.ReferencePowerCurve(RATED, ROTOR, cut_in=3.0, cut_out=25.0)
binned = wind.fit_binned_curve(speed, power, cut_out=25.0, min_count=20)
logistic = wind.fit_logistic_curve(speed, power, rated_power_kw=RATED, cut_in=3.0, cut_out=25.0)
pi_model = wind.PhysicsInformedPowerModel(
    rated_power_kw=RATED, rotor_diameter=ROTOR, hub_height=HUB, cut_in=3.0, cut_out=25.0,
).fit(feature_matrix(train, ("u_corrected",)), power)
gbm = wind.GradientBoostedPowerModel(feature_names=("u_corrected",)).fit(
    feature_matrix(train, ("u_corrected",)), power)

models = {
    "reference (analytic)":   wind.SpeedCurveModel(reference.power, name="reference", kind="reference"),
    "IEC binned":             wind.SpeedCurveModel(binned.power, name="binned", kind="empirical"),
    "logistic":               wind.SpeedCurveModel(logistic.power, name="logistic", kind="empirical"),
    "physics-informed":       pi_model,
    "boosted trees":          gbm,
    "operator potential PC":  wind.FrameColumnModel(),
}
print("fitted; the physics-informed region-2 Cp is",
      round(pi_model.diagnostics["implied_cp_region2"], 3), "against Betz", round(wind.betz_limit(), 3))
'''),
    md("## Score them on the test window"),
    code('''
def predict(model, frame):
    if getattr(model, "reads_frame", False):
        return model.predict(frame)
    return model.predict(feature_matrix(frame, model.feature_names))

rows = {}
for label, model in models.items():
    m = wind.point_metrics(test.power_kw.to_numpy(float), predict(model, test))
    rows[label] = {
        "RMSE [kW]": round(m.rmse_kw, 1),
        "MAE [kW]": round(m.mae_kw, 1),
        "RMSE weighted by energy [kW]": round(m.energy_weighted_rmse_kw, 1),
        "R2": round(m.r2, 4),
        "energy error": f"{m.energy_error_fraction:+.1%}",
    }
pd.DataFrame(rows).T
'''),
    md("""
## Where each model stands on the physics

The interesting column is the last one. The IEC binned curve — the industry's
definition of a power curve — is not monotone, is not zero outside the cut-in /
cut-out window, and its noisy low-wind bins imply a Cp above the Betz limit.
The physics-informed model cannot make any of those mistakes, and it pays about
0.4 % of rated power for that.
"""),
    code('''
def speed_curve(model, frame):
    if getattr(model, "reads_frame", False):
        return None
    names = tuple(model.feature_names)
    if names == ("u_corrected",):
        return lambda u: model.predict(np.column_stack([u]))
    med = {n: float(np.nanmedian(feature_matrix(frame, (n,))[:, 0])) for n in names}
    return lambda u: model.predict(np.column_stack(
        [u if n == "u_corrected" else np.full_like(u, med[n]) for n in names]))

checks = {}
for label, model in models.items():
    curve = speed_curve(model, train)
    if curve is None:
        continue
    checks[label] = {
        c["name"].replace("conservation_", ""): ("pass" if c["passed"] else "FAIL")
        for c in wind.physical_guardrails(curve, rated_power_kw=RATED, cut_in=3.0, cut_out=25.0,
                                          rotor_diameter=ROTOR)
    }
pd.DataFrame(checks).T
'''),
    code('''
curves = {
    "reference (analytic)": reference.power,
    "IEC binned": binned.power,
    "logistic": logistic.power,
    "physics-informed": lambda u: pi_model.predict(np.column_stack([u])),
}
fig = figures.plot_power_curves(curves, train, rated_power_kw=RATED,
                                     subtitle=f"{TURBINE}, training window only ({len(train):,} samples)")
plt.show()
'''),
    code('''
fig = figures.plot_error_by_speed(
    {label: wind.per_bin_error(test.u_corrected.to_numpy(float), test.power_kw.to_numpy(float),
                               predict(model, test), bin_width=1.0)
     for label, model in models.items()},
    title=f"Bias against wind speed, {TURBINE}, test window")
plt.show()
'''),
    md("""
## How much the prediction deserves to be trusted

A split-conformal interval is built on the **validation** residuals and its
coverage is measured on the **test** window. The gap between the target and the
measured coverage is the honest part: a year-later window is not exchangeable
with the calibration one.
"""),
    code('''
residual_val = val.power_kw.to_numpy(float) - predict(pi_model, val)
half_width = wind.conformal_interval(residual_val, level=0.9)
residual_test = test.power_kw.to_numpy(float) - predict(pi_model, test)
print(f"half-width {half_width:.0f} kW at a 90 % target")
print(wind.calibration_report(residual_test, half_width))
'''),
    md("""
## The applicability map

The last word belongs to VeriPhysics: what was verified, what is only inferred,
and where the model must not be asked to extrapolate.
"""),
    code('''
envelope = {
    "wind_speed": (float(train.u_corrected.quantile(0.001)), float(train.u_corrected.quantile(0.999))),
    "relative_density": (float(train.rho_relative.quantile(0.001)), float(train.rho_relative.quantile(0.999))),
}
m = wind.point_metrics(test.power_kw.to_numpy(float), predict(pi_model, test))
app = wind.power_applicability_map(
    architecture="physics-informed monotone hinge network (25 knots, hard cap and operating gate)",
    train=train, test_metrics=m.to_dict(),
    guardrails=wind.physical_guardrails(lambda u: pi_model.predict(np.column_stack([u])),
                                        rated_power_kw=RATED, cut_in=3.0, cut_out=25.0,
                                        rotor_diameter=ROTOR),
    calibration={"half_width_kw": half_width, "target_coverage": 0.9,
                 "coverage": wind.calibration_report(residual_test, half_width)["coverage"]},
    bounds=envelope, rmse_threshold_kw=0.05 * RATED, loss_history=pi_model.history,
    epochs=pi_model.epochs, n_parameters=int(pi_model.theta.size * 2),
    loss_weights={"data": 1.0, "cubic_region2": pi_model.cubic_weight,
                  "smoothness": pi_model.smoothness_weight, "betz": pi_model.betz_weight},
)
print(app.render())
'''),
    md("""
### What this notebook does not establish

The test window lies entirely inside the training envelope, so this is an
**interpolation** test. The operator's potential-power column is fitted by the
operator over years of data this notebook does not use; it is a yardstick, not a
competitor. And no curve here has been checked against a met mast.
"""),
]

WAKE_CELLS = [
    md("""
# Wakes: two analytical models against what the farm actually did

Jensen/PARK and Bastankhah & Porté-Agel on the published Penmanshiel layout,
compared sector by sector with the farm efficiency the SCADA implies. The wake
decay constant is fitted on a **validation** window and scored on a later test
window.

These are analytical engineering models, not CFD. The notebook says so once and
then measures how far off they are.
"""),
    md("## Setup"),
    code(PREAMBLE),
    code('''
layout = wind.static_layout(RAW)
print(layout, "|", layout.installed_capacity_mw, "MW")
nearest = wind.pair_geometry(layout)
pd.DataFrame(nearest)[["turbine", "nearest", "distance_in_diameters", "bearing_deg"]].round(2).head(6)
'''),
    md("""
## One wake, two shapes

The Jensen wake is a cone with a top-hat profile; the Gaussian one recovers in the
centre faster than at the edge. Both conserve momentum in a depth-averaged sense.
"""),
    code('''
x = np.linspace(0.5, 20, 400) * ROTOR
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.plot(x / ROTOR, wind.jensen_deficit(x, 0.0, ROTOR, 0.8), lw=2,
        label="Jensen/PARK, on the axis", color="#004c6d")
ax.plot(x / ROTOR, wind.gaussian_deficit(x, 0.0, ROTOR, 0.8), lw=2,
        label="Gaussian, on the axis", color="#a23a00")
ax.plot(x / ROTOR, wind.jensen_deficit(x, 1.5 * ROTOR, ROTOR, 0.8), lw=1.4, ls="--",
        color="#004c6d", label="Jensen, 1.5 D off-axis")
ax.plot(x / ROTOR, wind.gaussian_deficit(x, 1.5 * ROTOR, ROTOR, 0.8), lw=1.4, ls="--",
        color="#a23a00", label="Gaussian, 1.5 D off-axis")
ax.set_xlabel("downstream distance [rotor diameters]")
ax.set_ylabel("velocity deficit [-]")
ax.set_title("One wake, CT = 0.8", loc="left", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
plt.show()
'''),
    md("## The whole farm against wind direction"),
    code('''
curve = wind.ReferencePowerCurve(RATED, ROTOR).power
result = wind.farm_power(layout, 8.0, 270.0, curve)
print(f"8 m/s from the west: {result.total_power_kw/1000:.2f} MW of "
      f"{result.total_power_no_wake_kw/1000:.2f} MW wake-free "
      f"-> {result.wake_loss_fraction:.1%} loss")

polar = wind.direction_polar(layout, [6.0, 8.0, 10.0, 12.0], curve, n_directions=72)
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.plot(polar["directions_deg"], polar["efficiencies"], color="#004c6d", lw=1.8)
ax.set_xlabel("wind direction the wind comes from [deg]")
ax.set_ylabel("farm efficiency [-]")
ax.set_title("Modelled farm efficiency, 6-12 m/s", loc="left", fontsize=10)
ax.grid(alpha=0.25)
plt.show()
'''),
    md("""
## What the farm actually did

The comparison is built from metered power only. For each sector, the reference is
the group of three most-upwind turbines; their power is inverted through the
site's curves to a reference speed, and the farm efficiency is observed power over
what every machine would make at that speed. Anemometry is deliberately left out:
nacelle anemometers are not calibrated against each other, and in region 2 a 5 %
speed offset becomes a 15 % power offset.
"""),
    code('''
layout_names = layout.names
farm = wind.load_farm(RAW, turbines=layout_names, cache_dir=CACHE)
farm = wind.add_density_correction(
    farm, elevation_m=float(np.mean([t.elevation for t in layout if t.elevation is not None])))
farm["operating"] = pd.concat([
    pd.Series(wind.operating_mask(g, rated_power_kw=RATED), index=g.index)
    for _, g in farm.groupby("turbine", sort=False)
]).sort_index()

TRAIN_END, VAL_END = pd.Timestamp("2017-07-01"), pd.Timestamp("2017-09-01")
train_rows = farm[farm.timestamp < TRAIN_END]
curves = {}
for name in layout_names:
    g = train_rows[(train_rows.turbine == name) & train_rows.operating]
    curves[name] = wind.fit_binned_curve(g.u_corrected, g.power_kw, cut_out=25.0, min_count=20).power
print("one measured curve per turbine, fitted on the training window only")
'''),
    code('''
def window(lo, hi):
    part = farm[(farm.timestamp >= lo) & (farm.timestamp < hi)].copy()
    part["operating"] = pd.concat([
        pd.Series(wind.operating_mask(g, rated_power_kw=RATED), index=g.index)
        for _, g in part.groupby("turbine", sort=False)
    ]).sort_index()
    return part

obs_val = wind.farm_efficiency_observed(window(TRAIN_END, VAL_END), layout, curves,
                                        n_sectors=36, speed_edges=(4.0, 11.0, 1.0),
                                        min_samples=40, reference_group=3)
obs_test = wind.farm_efficiency_observed(window(VAL_END, pd.Timestamp("2018-01-01")), layout, curves,
                                         n_sectors=36, speed_edges=(4.0, 11.0, 1.0),
                                         min_samples=40, reference_group=3)
print(f"validation cells {np.isfinite(obs_val.efficiency).sum()}, "
      f"test cells {np.isfinite(obs_test.efficiency).sum()} "
      f"from {obs_test.meta['n_timestamps_used']:,} timestamps")
'''),
    md("""
## Fit one constant on validation, score it on test

The textbook Gaussian constant is 0.0325. A farm with a published layout and
SCADA can measure its own instead — as long as the fit is not then scored on its
own data.
"""),
    code('''
calibration = wind.calibrate_wake_decay(obs_val, layout, curves, kind="gaussian")
print(f"best k* on validation: {calibration['best_k']:.4f} "
      f"(textbook {calibration['textbook_k']:.4f})")
pd.DataFrame(calibration["sweep"]).round(4)
'''),
    code('''
rows = {}
for label, cfg in [
    ("Gaussian, textbook k*", wind.WakeConfig(kind="gaussian")),
    (f"Gaussian, k*={calibration['best_k']:.3f} fitted", wind.WakeConfig(
        kind="gaussian", k=calibration["best_k"], gaussian_k=calibration["best_k"])),
    ("Jensen, textbook k", wind.WakeConfig(kind="jensen")),
]:
    c = wind.compare_with_model(obs_test, layout, curves, config=cfg)
    model = np.array([np.nan if v is None else v for v in c["model_efficiency"]], dtype=float)
    seen = np.array([np.nan if v is None else v for v in c["observed_efficiency"]], dtype=float)
    rows[label] = {
        "modelled": np.nanmean(model).round(3),
        "observed": np.nanmean(seen).round(3),
        "bias [pp]": round(100 * c["bias"], 1),
        "RMSE": round(c["rmse"], 3),
        "correlation": round(c["correlation"], 2),
    }
pd.DataFrame(rows).T
'''),
    code('''
comparison = wind.compare_with_model(obs_test, layout, curves, config=wind.WakeConfig(kind="gaussian"))
fig = figures.plot_wake_polar(obs_test, comparison)
plt.show()
'''),
    md("""
## The same model, in 3D

The twin3d scene carries the plan-view wake footprint on a ground grid, swept over
a full direction cycle. `outputs/wind_farm_demo.mp4` is this scene rendered frame
by frame; here one frame is drawn inline.
"""),
    code('''
from pinneapple_systems.wind import twin, video

directions = np.arange(0.0, 360.0, 15.0)
scene = twin.build_farm_scene(layout, directions, wind_speed=8.0,
                              power_curves=[curves[n] for n in layout.names], n_x=72, n_y=54)
print(f"{len(directions)} steps, ground field {scene.parts[0].fields['velocity_ratio'].shape}, "
      f"{len(scene.sensors)} turbine sensors")

fig = video.render_frame(scene, layout, 18, wind_speed=8.0,
                         title=f"{layout.name}: {len(layout)} turbines, {layout.installed_capacity_mw:.1f} MW")
plt.show()
'''),
    md("""
### What this does not establish

The observed efficiency is an **upper bound** on wake loss: it absorbs curtailment
that survived the filter, icing, and genuine per-turbine under-performance. The
correlation across sectors is weak, so the model's *direction dependence* is not
reproduced even where its average is. And nothing here is a flow solution.
"""),
]


def main() -> None:
    folder = pathlib.Path(__file__).resolve().parent
    folder.mkdir(parents=True, exist_ok=True)
    for name, cells in (("01_turbine_power_model.ipynb", TURBINE_CELLS),
                        ("02_wake_models.ipynb", WAKE_CELLS)):
        (folder / name).write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
        print("wrote", folder / name)


def execute(folder: pathlib.Path) -> None:
    """Run both notebooks in place so their outputs are committed with them."""
    import nbformat
    from nbclient import NotebookClient

    for name in ("01_turbine_power_model.ipynb", "02_wake_models.ipynb"):
        path = folder / name
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=2400, kernel_name="python3",
                       resources={"metadata": {"path": str(pathlib.Path.cwd())}}).execute()
        nbformat.write(nb, path)
        print("executed", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="also run the notebooks and commit their outputs")
    args = ap.parse_args()
    main()
    if args.execute:
        execute(pathlib.Path(__file__).resolve().parent)

