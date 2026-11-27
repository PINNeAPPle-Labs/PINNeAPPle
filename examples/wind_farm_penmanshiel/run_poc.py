"""Penmanshiel wind-farm POC: one turbine modelled, one farm's wakes checked against SCADA.

What this script does, in order, and what each step is allowed to claim:

1. **Load** the public Penmanshiel dataset (14 Senvion MM82, CC-BY-4.0) and apply
   the recorded operating mask (availability, downtime, curtailment) instead of a
   guessed one.
2. **Explore**: per-turbine coverage, wind rose, speed distribution.
3. **Fit** four single-turbine power models on the training window only -- a
   reference (analytic) curve, the IEC binned curve, a five-parameter logistic,
   a gradient-boosted baseline and the physics-informed monotone model -- and
   score them on a **later** window that no model ever saw.
4. **Check** the physics-informed model's claims (cap, monotonicity, operating
   window, Betz) by re-measuring them, and assemble the VeriPhysics applicability
   map from the checks that actually ran.
5. **Run the farm**: Jensen and Gaussian wake models over the published layout,
   a physics-informed surrogate of the farm efficiency, and a curtailment study.
6. **Compare** the wake model with the efficiency the farm actually achieved,
   sector by sector, and report what that comparison can and cannot establish.

Every number written to ``results.json`` carries the window, the filter and the
model that produced it. Anything that could not be computed is written as
``null`` with a reason, never as a zero.

Usage::

    python examples/wind_farm_penmanshiel/run_poc.py \\
        --raw ~/pn-wind-data/raw --out examples/wind_farm_penmanshiel/outputs
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pinneapple_systems import wind
from pinneapple_systems.wind import figures
from pinneapple_systems.wind.models import feature_matrix

# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------

RATED_KW = 2050.0
ROTOR_M = 82.0
HUB_M = 59.0
CUT_IN = 3.0
CUT_OUT = 25.0

TRAIN_END = "2017-07-01"
VAL_END = "2017-09-01"

SPEED_BINS = np.arange(4.0, 11.5, 1.0)

MODEL_LABELS = {
    "reference": "reference (analytic cubic)",
    "binned": "IEC binned power curve",
    "logistic": "logistic (5-parameter)",
    "physics-informed": "physics-informed (monotone hinge)",
    "conventional-ml": "gradient-boosted trees (u only)",
    "conventional-ml-full": "gradient-boosted trees (met features)",
    "constant": "constant (training mean)",
    "operator": "operator potential power (reference)",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", required=True, help="folder holding the Penmanshiel zip and static CSV")
    parser.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    parser.add_argument("--cache", default=None, help="parse cache folder (one pickle per turbine-year)")
    parser.add_argument("--sectors", type=int, default=36)
    parser.add_argument("--quick", action="store_true", help="one turbine, no farm section")
    args = parser.parse_args(argv)

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: dict = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {"name": "Penmanshiel wind farm", "doi": wind.PENMANSHIEL_DOI,
                    "license": "CC-BY-4.0"},
        "case": {"rated_power_kw": RATED_KW, "rotor_diameter_m": ROTOR_M, "hub_height_m": HUB_M,
                 "cut_in": CUT_IN, "cut_out": CUT_OUT,
                 "train_end": TRAIN_END, "validation_end": VAL_END},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }

    # -- 1. data ------------------------------------------------------------
    layout = wind.static_layout(args.raw)
    results["farm"] = {
        "turbines": len(layout),
        "installed_capacity_mw": layout.installed_capacity_mw,
        "turbine_table": layout.to_records(),
        "nearest_neighbour": wind.pair_geometry(layout),
    }

    print(f"[1/6] loading {len(layout)} turbines from {args.raw}")
    farm = wind.load_farm(args.raw, turbines=layout.names, cache_dir=args.cache)
    elevation_m = float(np.mean([t.elevation for t in layout if t.elevation is not None] or [200.0]))
    farm = wind.add_density_correction(farm, elevation_m=elevation_m)
    farm["operating"] = _grouped_mask(farm, RATED_KW)
    results["data"] = {
        "rows": int(len(farm)),
        "span": [str(farm["timestamp"].min()), str(farm["timestamp"].max())],
        "per_turbine": wind.summarise_by_turbine(farm, RATED_KW),
        "micro_quality": _qc_digest(farm, layout),
    }

    window = _split(farm)
    results["data"]["windows"] = {k: {"rows": int(v["n"]),
                                      "span": [str(v["start"]), str(v["end"])]}
                                  for k, v in window.items()}

    # -- 2. explore ---------------------------------------------------------
    print("[2/6] exploring")
    operating = farm[farm["operating"]]
    figures.save(figures.plot_wind_rose(operating), out / "figures" / "01_wind_rose.png")

    focus = _focus_turbine(farm, layout)
    results["focus_turbine"] = focus
    focus_operating = operating[operating["turbine"] == focus]

    # -- 3. single-turbine models ------------------------------------------
    print("[3/6] fitting power models")
    frames = {
        name: focus_operating[(focus_operating["timestamp"] >= lo) & (focus_operating["timestamp"] < hi)]
        for name, (lo, hi) in window.items()
    }
    models, curves, fits = _fit_models(frames["train"])
    results["models"] = fits

    scores = {}
    for name in ("validation", "test"):
        scores[name] = _score(models, frames[name])
    results["scores"] = scores
    results["scores"]["conformal"] = _conformal(models["physics-informed"], frames)

    figures.save(
        figures.plot_power_curves(
            curves, frames["train"], rated_power_kw=RATED_KW,
            subtitle=f"turbine {focus}, training window only ({len(frames['train']):,} samples)",
        ),
        out / "figures" / "02_power_curves.png",
    )
    figures.save(
        figures.plot_error_by_speed({
            MODEL_LABELS[k]: _bin_rows(models[k], frames["test"]) for k in
            ("reference", "binned", "logistic", "physics-informed", "conventional-ml", "operator")
            if k in models
        }, title=f"Test-window bias against wind speed, turbine {focus}"),
        out / "figures" / "03_error_by_speed.png",
    )
    figures.save(figures.plot_learning_curve(models["physics-informed"].history),
                 out / "figures" / "04_pi_convergence.png")

    # -- 4. verification ----------------------------------------------------
    print("[4/6] verifying")
    pi_model = models["physics-informed"]
    guardrails = wind.physical_guardrails(
        lambda u: pi_model.predict(np.column_stack([u])),
        rated_power_kw=RATED_KW, cut_in=CUT_IN, cut_out=CUT_OUT, rotor_diameter=ROTOR_M,
    )
    results["guardrails"] = guardrails
    results["domain"] = wind.domain_report(
        frames["train"], frames["test"],
        variables=("wind_speed_ms", "u_corrected", "ambient_temperature_c"),
    )
    results["domain"]["envelope"] = {
        "wind_speed": [float(frames["train"]["u_corrected"].quantile(0.001)),
                       float(frames["train"]["u_corrected"].quantile(0.999))],
        "relative_density": [float(frames["train"]["rho_relative"].quantile(0.001)),
                             float(frames["train"]["rho_relative"].quantile(0.999))],
    }
    applicability = wind.power_applicability_map(
        architecture="physics-informed monotone hinge network (25 knots, hard cap and operating gate)",
        train=frames["train"],
        test_metrics=scores["test"]["physics-informed"],
        guardrails=guardrails,
        calibration=results["scores"]["conformal"],
        bounds=results["domain"]["envelope"],
        rmse_threshold_kw=0.05 * RATED_KW,
        final_loss=pi_model.history[-1] if pi_model.history else None,
        loss_history=pi_model.history,
        epochs=pi_model.epochs,
        n_parameters=int(pi_model.theta.size * 2) if pi_model.theta is not None else None,
        loss_weights={"data": 1.0, "cubic_region2": pi_model.cubic_weight,
                      "smoothness": pi_model.smoothness_weight, "betz": pi_model.betz_weight},
    )
    results["applicability"] = applicability.to_dict()
    (out / "applicability_map.txt").write_text(applicability.render() + "\n", encoding="utf-8")

    if args.quick:
        _write(out, results, started, partial=True)
        return 0

    # -- 5. farm / wakes ----------------------------------------------------
    print("[5/6] running the farm models")
    turbine_curves = _farm_curves(farm, layout, frames["train"])
    test_farm = farm[(farm["timestamp"] >= pd.Timestamp(window["test"]["start"]))
                     & (farm["timestamp"] <= pd.Timestamp(window["test"]["end"]))].copy()
    test_farm["operating"] = _grouped_mask(test_farm, RATED_KW)

    observation = wind.farm_efficiency_observed(
        test_farm, layout, turbine_curves, n_sectors=args.sectors,
        speed_edges=(SPEED_BINS[0], SPEED_BINS[-1], SPEED_BINS[1] - SPEED_BINS[0]),
        min_samples=40, reference="power",
    )
    results["wakes"] = {"observation": observation.to_dict(), "models": {}}
    for kind in ("gaussian", "jensen"):
        cfg = wind.WakeConfig(kind=kind)
        comparison = wind.compare_with_model(observation, layout, turbine_curves, config=cfg)
        results["wakes"]["models"][kind] = _trim_comparison(comparison, observation)
        figures.save(
            figures.plot_wake_polar(observation, comparison,
                                    title=f"Farm efficiency vs direction: {cfg.label}\n"
                                          f"observed on the test window, {observation.meta['n_timestamps_used']:,} timestamps"),
            out / "figures" / f"05_wake_polar_{kind}.png",
        )

    # per-turbine map at the sector with the strongest predicted deficit
    worst_sector = _worst_sector(results["wakes"]["models"]["gaussian"], observation)
    if worst_sector is not None:
        k = worst_sector
        direction = float(observation.sectors_deg[k])
        speed = float(np.nanmedian(SPEED_BINS))
        single = wind.farm_power(layout, speed, direction, [turbine_curves[n] for n in layout.names])
        figures.save(
            figures.plot_farm_deficit_map(layout, single.wake_deficit, direction_deg=direction,
                                          wind_speed=speed),
            out / "figures" / "06_farm_deficit_map.png",
        )
        results["wakes"]["worst_sector"] = {
            "sector_deg": direction, "wind_speed_ms": speed,
            "efficiency": single.efficiency, "per_turbine_deficit": single.wake_deficit.tolist(),
            "turbines": layout.names,
        }
    figures.save(figures.plot_layout(layout), out / "figures" / "07_layout.png")

    # -- 6. surrogate and optimisation --------------------------------------
    print("[6/6] surrogate and curtailment study")
    surrogate = wind.fit_surrogate(layout, [turbine_curves[n] for n in layout.names],
                                   speeds=(5.0, 7.0, 9.0, 11.0, 13.0))
    results["surrogate"] = {"name": surrogate.name, **surrogate.diagnostics}
    study = wind.curtailment_study(
        layout,
        directions=[0.0, 90.0, 180.0, 270.0],
        wind_speeds=[6.0, 8.0, 10.0],
        power_curves=[turbine_curves[n] for n in layout.names],
    )
    results["curtailment"] = study

    _write(out, results, started)
    print(f"\nwritten to {out}")
    return 0


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def _split(farm: pd.DataFrame) -> dict:
    """Temporal split: train, then a later validation window, then a later test window."""
    train_end = pd.Timestamp(TRAIN_END)
    val_end = pd.Timestamp(VAL_END)
    frames = {
        "train": (pd.Timestamp("1900-01-01"), train_end),
        "validation": (train_end, val_end),
        "test": (val_end, pd.Timestamp("2100-01-01")),
    }
    out = {}
    for name, (lo, hi) in frames.items():
        part = farm[(farm["timestamp"] >= lo) & (farm["timestamp"] < hi)]
        out[name] = {"n": len(part), "start": part["timestamp"].min(), "end": part["timestamp"].max()}
    return out


def _grouped_mask(farm: pd.DataFrame, rated_kw: float) -> pd.Series:
    """Operating mask applied per turbine, returned aligned to the frame's index."""
    parts = [
        pd.Series(wind.operating_mask(g, rated_power_kw=rated_kw), index=g.index)
        for _, g in farm.groupby("turbine", sort=False)
    ]
    return pd.concat(parts).sort_index()


def _qc_digest(farm: pd.DataFrame, layout) -> dict:
    """QC counts per turbine, plus the mask provenance that justifies the filter."""
    by_turbine = {}
    for name, g in farm.groupby("turbine"):
        report = wind.qc_report(g, rated_power_kw=RATED_KW)
        by_turbine[name] = {
            "rows": report["rows"],
            "operating": report["operating_samples"],
            "operating_fraction": report["operating_fraction"],
            "curtailed": report["curtailed_samples"],
            "missing_power": next((r["removed"] for r in report["rules"] if r["name"] == "missing_power_kw"), None),
        }
    return {
        "provenance": wind.mask_provenance(farm),
        "by_turbine": by_turbine,
    }


def _focus_turbine(farm: pd.DataFrame, layout) -> str:
    """The turbine with the most operating samples -- stated, not assumed."""
    counts = farm[farm["operating"]].groupby("turbine").size()
    return str(counts.idxmax())


def _fit_models(train: pd.DataFrame) -> tuple[dict, dict, dict]:
    speed = train["u_corrected"].to_numpy(dtype=float)
    power = train["power_kw"].to_numpy(dtype=float)

    binned = wind.fit_binned_curve(speed, power, cut_out=CUT_OUT, min_count=20)
    logistic = wind.fit_logistic_curve(speed, power, rated_power_kw=RATED_KW,
                                       cut_in=CUT_IN, cut_out=CUT_OUT)
    reference = wind.ReferencePowerCurve(RATED_KW, ROTOR_M, cut_in=CUT_IN, cut_out=CUT_OUT)
    pi_model = wind.PhysicsInformedPowerModel(
        rated_power_kw=RATED_KW, rotor_diameter=ROTOR_M, hub_height=HUB_M,
        cut_in=CUT_IN, cut_out=CUT_OUT,
    ).fit(feature_matrix(train, ("u_corrected",)), power)
    gbm = wind.GradientBoostedPowerModel(feature_names=("u_corrected",)).fit(
        feature_matrix(train, ("u_corrected",)), power)
    gbm_full = wind.GradientBoostedPowerModel(feature_names=wind.ALTERNATING_MET_FEATURES).fit(
        feature_matrix(train, wind.ALTERNATING_MET_FEATURES), power)
    constant = wind.ConstantPowerModel().fit(feature_matrix(train, ("u_corrected",)), power)

    models = {
        "reference": wind.SpeedCurveModel(reference.power, name=MODEL_LABELS["reference"], kind="reference"),
        "binned": wind.SpeedCurveModel(binned.power, name=MODEL_LABELS["binned"], kind="empirical"),
        "logistic": wind.SpeedCurveModel(logistic.power, name=MODEL_LABELS["logistic"], kind="empirical"),
        "physics-informed": pi_model,
        "conventional-ml": gbm,
        "conventional-ml-full": gbm_full,
        "constant": constant,
        "operator": wind.FrameColumnModel(),
    }
    curves = {
        MODEL_LABELS["reference"]: reference.power,
        MODEL_LABELS["binned"]: binned.power,
        MODEL_LABELS["logistic"]: logistic.power,
        MODEL_LABELS["physics-informed"]: lambda u: pi_model.predict(np.column_stack([u])),
    }
    fits = {
        "reference": {"rated_speed_ms": reference.rated_speed, "cp": reference.cp,
                      "rotor_area_m2": reference.rotor_area},
        "binned": {"support": binned.support(), "bins": binned.to_records()},
        "logistic": {"a": logistic.a, "b": logistic.b, "c": logistic.c},
        "physics-informed": pi_model.to_dict(),
        "conventional-ml": gbm.to_dict(),
        "conventional-ml-full": gbm_full.to_dict(),
        "constant": constant.to_dict(),
        "operator": wind.FrameColumnModel().to_dict(),
    }
    return models, curves, fits


class _CurveModel:
    """Adapter so a plain speed->power callable can be scored like a fitted model."""


def _predict(model, frame) -> np.ndarray:
    """Dispatch: reference columns read the frame, models read a feature matrix."""
    if getattr(model, "reads_frame", False):
        return model.predict(frame)
    return model.predict(feature_matrix(frame, model.feature_names))


def _score(models: dict, frame: pd.DataFrame) -> dict:
    measured = frame["power_kw"].to_numpy(dtype=float)
    out: dict = {}
    for key, model in models.items():
        out[key] = {**wind.point_metrics(measured, _predict(model, frame)).to_dict(),
                    "label": MODEL_LABELS[key], "kind": getattr(model, "kind", "model")}
    return out

def _bin_rows(model, frame: pd.DataFrame) -> list[dict]:
    return wind.per_bin_error(frame["u_corrected"].to_numpy(dtype=float),
                              frame["power_kw"].to_numpy(dtype=float),
                              _predict(model, frame), bin_width=1.0)


def _conformal(model, frames: dict) -> dict:
    """Half-width from the validation residuals, coverage measured on the test window."""
    val_residual = frames["validation"]["power_kw"].to_numpy() - _predict(model, frames["validation"])
    half = wind.conformal_interval(val_residual, level=0.9)
    test_residual = frames["test"]["power_kw"].to_numpy() - _predict(model, frames["test"])
    report = wind.calibration_report(test_residual, half)
    return {"half_width_kw": half, "target": 0.9, "coverage": report["coverage"],
            "n_validation": int(len(val_residual)), "n_test": report["n"],
            "mean_abs_residual_kw": report["mean_abs_residual_kw"]}


def _farm_curves(farm: pd.DataFrame, layout, train_frame: pd.DataFrame) -> dict:
    """One measured curve per turbine, fitted on the training rows of that turbine."""
    lo = train_frame["timestamp"].min()
    hi = train_frame["timestamp"].max()
    curves = {}
    for name in layout.names:
        g = farm[(farm["turbine"] == name) & farm["operating"]
                 & (farm["timestamp"] >= lo) & (farm["timestamp"] <= hi)]
        curves[name] = wind.fit_binned_curve(g["u_corrected"], g["power_kw"],
                                             cut_out=CUT_OUT, min_count=20).power
    return curves


def _trim_comparison(comparison: dict, observation) -> dict:
    grid = np.array(comparison["model_efficiency"], dtype=float)
    out = {k: v for k, v in comparison.items() if k not in ("model_efficiency", "per_turbine_deficit_predicted")}
    out["mean_model_efficiency_by_sector"] = np.nanmean(grid, axis=1).tolist()
    out["per_turbine_deficit_predicted_by_sector"] = comparison["per_turbine_deficit_predicted"]
    return out


def _worst_sector(comparison: dict, observation) -> int | None:
    values = comparison.get("mean_model_efficiency_by_sector")
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).any():
        return None
    return int(np.nanargmin(array))


def _write(out: Path, results: dict, started: float, *, partial: bool = False) -> None:
    results["runtime_seconds"] = round(time.time() - started, 1)
    results["partial"] = partial
    (out / "results.json").write_text(json.dumps(results, indent=2, default=_json_default) + "\n",
                                      encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


if __name__ == "__main__":
    sys.exit(main())
