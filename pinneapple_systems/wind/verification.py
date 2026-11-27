"""Checking the wind models against data, and against the domain they were fitted on.

Two different questions are answered here, and they must not be confused:

*Which turbine states is the model allowed to speak about?*
    :func:`domain_report` measures how much of a test window lies outside the
    envelope the model was trained on. A 10-minute sample at 28 m/s or at
    -12 degC is not the same claim as one at 8 m/s in April, and a farm model
    fitted on September-December should not silently be used for a July
    afternoon.

*Does the farm model reproduce what the farm actually did?*
    :func:`farm_efficiency_observed` builds an efficiency curve from SCADA alone
    -- observed farm power divided by what the same machines would make at the
    free-stream speed -- and :func:`compare_with_model` puts the analytical wake
    model next to it, sector by sector.

That second comparison is *not* a validation of the wake model's physics: the
"free-stream" speed is a nacelle anemometer on the most upwind turbine, the
expected power comes from a curve fitted to the same instruments, and the
observed efficiency also absorbs curtailment, icing, availability and
instrument bias. What it can show is whether the model reproduces the *shape* of
the direction dependence and the right order of magnitude of the loss. It cannot
show that a 7 % loss is 7.0 % and not 6.4 %.

The VeriPhysics glue at the bottom assembles the facts this module measured into
an applicability map, so the claims, the evidence and the gaps travel together.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .layout import WindFarmLayout
from .wake import WakeConfig, farm_power

__all__ = [
    "domain_report",
    "curve_inverse",
    "FarmObservation",
    "farm_efficiency_observed",
    "compare_with_model",
    "calibrate_wake_decay",
    "power_applicability_map",
    "physical_guardrails",
]


# ---------------------------------------------------------------------------
# Domain of validity
# ---------------------------------------------------------------------------

def domain_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    variables: Sequence[str] = ("wind_speed_ms", "ambient_temperature_c"),
    margin: float = 0.0,
) -> dict:
    """Fraction of the test window outside the training envelope of each variable.

    The envelope is the closed interval spanned by the *training* data, widened
    by ``margin`` (in units of the variable). Coverage is reported per variable
    on its own and jointly, and the joint number is the one an out-of-domain
    claim rests on: being inside both ranges separately is weaker than being
    inside their joint convex hull, which is why this makes no convex-hull claim.
    """
    rows = []
    joint = np.ones(len(test), dtype=bool)
    for var in variables:
        if var not in train or var not in test:
            rows.append({"variable": var, "available": False})
            continue
        t = train[var].to_numpy(dtype=float)
        s = test[var].to_numpy(dtype=float)
        t = t[np.isfinite(t)]
        lo, hi = float(np.min(t)) - margin, float(np.max(t)) + margin
        inside = np.isfinite(s) & (s >= lo) & (s <= hi)
        joint &= inside
        rows.append({
            "variable": var,
            "available": True,
            "train_min": lo,
            "train_max": hi,
            "test_min": float(np.nanmin(s)) if np.isfinite(s).any() else None,
            "test_max": float(np.nanmax(s)) if np.isfinite(s).any() else None,
            "inside_fraction": float(inside.mean()) if inside.size else 0.0,
            "outside_count": int((~inside).sum()),
        })
    return {
        "variables": rows,
        "joint_inside_fraction": float(joint.mean()) if joint.size else 0.0,
        "n_test": int(len(test)),
    }


# ---------------------------------------------------------------------------
# SCADA-observed farm efficiency
# ---------------------------------------------------------------------------

def curve_inverse(curve, power_kw, *, grid: np.ndarray | None = None) -> np.ndarray:
    """Effective wind speed for a measured power, by inverting a power curve.

    The curve is sampled on a grid and interpolated backwards; power above rated
    maps to the rated speed (the curve is flat there, so the inversion is
    genuinely ambiguous -- returning the rated speed is the documented choice).
    """
    g = np.linspace(0.0, 30.0, 601) if grid is None else np.asarray(grid, dtype=float)
    p = np.asarray(curve(g), dtype=float)
    p = np.maximum.accumulate(p)  # enforce monotone before inverting
    out = np.interp(np.asarray(power_kw, dtype=float), p, g, left=0.0, right=g[-1])
    return out


@dataclass
class FarmObservation:
    """Observed farm efficiency on a (direction sector, speed bin) grid."""

    sectors_deg: np.ndarray
    speed_centres: np.ndarray
    efficiency: np.ndarray            # (n_sectors, n_speeds), NaN where unsupported
    counts: np.ndarray                # (n_sectors, n_speeds)
    reference_turbine: list[str]
    energy_weight: np.ndarray         # (n_sectors,) measured energy per sector
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sectors_deg": self.sectors_deg.tolist(),
            "speed_centres": self.speed_centres.tolist(),
            "efficiency": np.where(np.isfinite(self.efficiency), self.efficiency, None).tolist(),
            "counts": self.counts.tolist(),
            "reference_turbine": self.reference_turbine,
            "energy_weight": self.energy_weight.tolist(),
            "meta": self.meta,
        }


def farm_efficiency_observed(
    farm: pd.DataFrame,
    layout: WindFarmLayout,
    curves: dict,
    *,
    n_sectors: int = 36,
    speed_edges: Sequence[float] = (4.0, 11.0, 1.0),
    min_samples: int = 40,
    operating_column: str = "operating",
    reference: str = "power",
    reference_group: int = 3,
    speed_column: str = "u_corrected",
) -> FarmObservation:
    """Farm efficiency implied by SCADA, per direction sector and wind-speed bin.

    Steps, all of them visible in the returned object:

    1. Keep timestamps where every turbine in ``layout`` is operating.
    2. For each sector, pick the most upwind turbine as the free-stream reference.
    3. Expected power of turbine ``i`` = its own curve at the reference speed;
       observed power is the measured one. Their ratio, summed over turbines, is
       the observed farm efficiency.
    4. Group by reference-speed bin so the result can be compared with the wake
       model evaluated at the same speeds.

    ``reference="power"`` (the default) derives the reference speed by inverting
    the reference turbine's *power*: ``U_ref = curve_ref^-1(P_ref)``. That is
    deliberate. Nacelle anemometers are not calibrated against each other, and in
    region 2 a 5 % anemometer offset becomes a 15 % efficiency offset -- larger
    than the wake loss the comparison is trying to see. Active power is metered
    and comparable across machines, so the power ratio is the more trustworthy
    signal, at the cost of needing a monotone curve to invert. ``reference=
    "anemometer"`` keeps the direct-instrument variant for comparison; it is
    noisier for the reason above and every figure using it should say so.

    ``curves`` maps turbine name -> callable(speeds) -> kW. Using each machine's
    own curve is deliberate: it absorbs the per-machine performance differences
    that a shared curve would turn into a fake wake deficit.

    The sector is assigned from the *circular* mean direction over the turbines
    (``atan2`` of the mean unit vector), not from the arithmetic mean of the
    angles: 359 deg and 1 deg average to 180 deg arithmetically, which would put
    every westerly sample in the wrong sector.
    """
    if reference not in ("power", "anemometer"):
        raise ValueError(f"unknown reference {reference!r}")
    names = [n for n in layout.names if n in set(farm["turbine"])]
    if len(names) < 2:
        raise ValueError("need at least two turbines with data to estimate a farm efficiency")
    sub_layout = _subset_layout(layout, names)
    needed = {"timestamp", "turbine", "power_kw", "wind_direction_deg", operating_column, speed_column}
    missing = needed - set(farm.columns)
    if missing:
        raise KeyError(f"farm frame is missing columns: {sorted(missing)}")

    wide_power = farm.pivot_table(index="timestamp", columns="turbine", values="power_kw", aggfunc="last")
    # pivot_table drops all-NaN rows, so each pivot can come back on a *different*
    # timestamp index. Reindex everything onto the power index before any
    # arithmetic, or label alignment silently mixes neighbouring samples.
    index = wide_power.index
    wide_speed = farm.pivot_table(index="timestamp", columns="turbine", values=speed_column,
                                  aggfunc="last").reindex(index)
    wide_dir = farm.pivot_table(index="timestamp", columns="turbine", values="wind_direction_deg",
                                aggfunc="last").reindex(index)
    wide_op = farm.pivot_table(index="timestamp", columns="turbine", values=operating_column,
                               aggfunc="last").reindex(index)
    cols = [c for c in names if c in wide_power]
    complete = wide_op[cols].fillna(False).astype(bool).all(axis=1)
    complete &= wide_power[cols].notna().all(axis=1) & wide_speed[cols].notna().all(axis=1)

    # Circular mean of the direction: average the unit vectors, then take atan2.
    radians = np.radians(wide_dir[cols].to_numpy(dtype=float))
    mean_dir = np.degrees(np.arctan2(np.nanmean(np.sin(radians), axis=1),
                                     np.nanmean(np.cos(radians), axis=1))) % 360.0
    mean_dir = pd.Series(mean_dir, index=index)
    sector_width = 360.0 / n_sectors
    sector_index = np.floor(mean_dir / sector_width).astype(int).clip(0, n_sectors - 1)
    keep = complete & sector_index.notna()
    sector_index = sector_index[keep].astype(int)
    wide_power = wide_power.loc[keep]
    wide_speed = wide_speed.loc[keep]

    edges = np.arange(speed_edges[0], speed_edges[1] + 1e-9, speed_edges[2])
    centres = 0.5 * (edges[:-1] + edges[1:])
    efficiency = np.full((n_sectors, centres.size), np.nan)
    counts = np.zeros((n_sectors, centres.size), dtype=int)
    energy = np.zeros(n_sectors)
    refs: list[str] = []

    for k in range(n_sectors):
        direction = (k + 0.5) * sector_width
        order = sub_layout.upwind_order(direction)
        group = [sub_layout.names[i] for i in order[:max(1, reference_group)]]
        refs.append("+".join(group))
        sel = sector_index.to_numpy() == k
        if sel.sum() < min_samples:
            continue
        group_power = wide_power.loc[sel, group].mean(axis=1).to_numpy(dtype=float)
        if reference == "anemometer":
            u_ref = wide_speed.loc[sel, group].mean(axis=1).to_numpy(dtype=float)
        else:
            u_ref = curve_inverse(_mean_curve([curves[g] for g in group]), group_power)
        expected = np.zeros_like(u_ref)
        observed = np.zeros_like(u_ref)
        for name in cols:
            expected += np.asarray(curves[name](u_ref), dtype=float)
            observed += wide_power.loc[sel, name].to_numpy(dtype=float)
        valid = np.isfinite(u_ref) & np.isfinite(expected) & (expected > 0)
        energy[k] = float(np.nansum(observed[valid]))
        which = np.digitize(u_ref[valid], edges) - 1
        ratio = observed[valid] / np.maximum(expected[valid], 1e-9)
        for b in range(centres.size):
            mask = which == b
            counts[k, b] = int(mask.sum())
            if mask.sum() >= min_samples:
                efficiency[k, b] = float(np.mean(ratio[mask]))

    return FarmObservation(
        sectors_deg=(np.arange(n_sectors) + 0.5) * sector_width,
        speed_centres=centres,
        efficiency=efficiency,
        counts=counts,
        reference_turbine=refs,
        energy_weight=energy,
        meta={
            "n_sectors": n_sectors,
            "speed_edges": [float(edges[0]), float(edges[-1]), float(edges[1] - edges[0])],
            "min_samples": min_samples,
            "reference": reference,
            "reference_group": int(reference_group),
            "n_timestamps_used": int(keep.sum()),
            "turbines": cols,
            "caveat": (
                "Efficiency is observed farm power over the sum of per-turbine curves at the "
                "most-upwind turbine's reference speed. With reference='power' that speed is "
                "inverted from metered power, which removes the nacelle-anemometer calibration "
                "spread but inherits whatever the per-turbine curves got wrong. It still absorbs "
                "curtailment, icing, availability and genuine per-turbine performance differences, "
                "so it is an upper bound on the wake loss, not an independent measurement of it."
            ),
        },
    )


def compare_with_model(
    observation: FarmObservation,
    layout: WindFarmLayout,
    curves: dict,
    *,
    config: WakeConfig | None = None,
    fit_scale: bool = True,
) -> dict:
    """Run the wake model on the same (sector, speed) grid and report the gap.

    ``fit_scale`` is the important argument. The observed efficiency carries a
    multiplicative calibration factor: the free-stream reference is one turbine's
    nacelle anemometer, and those instruments are not calibrated against each
    other. In region 2 the effect is cubic in the speed ratio, so a 5 % anemometer
    offset becomes a 15 % efficiency offset -- larger than the wake loss itself.

    With ``fit_scale=True`` the comparison is therefore reported twice:

    * ``rmse``/``bias`` -- the raw gap, which mixes wake-model error with that
      calibration offset and must not be quoted as wake-model accuracy;
    * ``scale``, ``rmse_scaled`` and ``correlation`` -- a single multiplicative
      scale fitted over all cells. The scale absorbs the constant instrument
      bias (it is constant as long as the reference turbine does not change);
      what survives is the **direction dependence**, which is the part of the
      wake model this data can actually test.

    Nothing here identifies a per-turbine calibration: that needs a met mast or a
    manufacturer calibration, neither of which this dataset provides.
    """
    names = observation.meta["turbines"]
    sub_layout = _subset_layout(layout, names)
    per_turbine_curves = [curves[n] for n in names]
    sectors = observation.sectors_deg
    speeds = observation.speed_centres
    model_eff = np.full_like(observation.efficiency, np.nan)
    model_eff_freestream = np.full_like(observation.efficiency, np.nan)
    per_turbine_deficit = np.zeros((sectors.size, len(names)))
    index_of = {n: i for i, n in enumerate(names)}
    for k, direction in enumerate(sectors):
        group = [index_of[g] for g in observation.reference_turbine[k].split("+") if g in index_of]
        deficits = np.zeros(len(names))
        for b, u in enumerate(speeds):
            result = farm_power(sub_layout, float(u), float(direction), per_turbine_curves, config)
            model_eff_freestream[k, b] = result.efficiency
            # The observed efficiency divides by the *reference group's* production,
            # and that group is itself waked in row-aligned sectors. Evaluating the
            # model at the group's own (deficit-reduced) inflow keeps both sides of
            # the comparison defined the same way; without it the observed curve is
            # flattened exactly where the model predicts the deepest dip.
            delta_ref = float(np.mean([result.wake_deficit[i] for i in group])) if group else 0.0
            reference_speed = u * (1.0 - delta_ref)
            denominator = sum(float(np.asarray(curves[n](np.array([reference_speed]))).ravel()[0])
                              for n in names)
            model_eff[k, b] = result.total_power_kw / denominator if denominator > 0 else np.nan
            deficits += result.wake_deficit
        per_turbine_deficit[k] = deficits / max(speeds.size, 1)

    ok = np.isfinite(model_eff) & np.isfinite(observation.efficiency)
    diff = model_eff[ok] - observation.efficiency[ok]
    weights = np.where(ok, observation.counts, 0).astype(float)[ok]
    out = {
        "model_efficiency": np.where(np.isfinite(model_eff), model_eff, None).tolist(),
        "model_efficiency_freestream": np.where(np.isfinite(model_eff_freestream),
                                                model_eff_freestream, None).tolist(),
        "observed_efficiency": np.where(np.isfinite(observation.efficiency), observation.efficiency, None).tolist(),
        "n_cells": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(diff**2))) if ok.any() else None,
        "bias": float(np.mean(diff)) if ok.any() else None,
        "weighted_rmse": float(np.sqrt(np.average(diff**2, weights=weights))) if ok.any() and weights.sum() > 0 else None,
        "correlation": float(np.corrcoef(model_eff[ok], observation.efficiency[ok])[0, 1])
        if ok.sum() >= 3 else None,
        "per_turbine_deficit_predicted": per_turbine_deficit.tolist(),
        "turbines": names,
        "config": (config or WakeConfig()).label,
    }
    if fit_scale and ok.any():
        m = model_eff[ok]
        o = observation.efficiency[ok]
        scale = float((m * o).sum() / max((m * m).sum(), 1e-12))
        residual = scale * m - o
        out["scale"] = scale
        out["rmse_scaled"] = float(np.sqrt(np.mean(residual**2)))
        out["bias_scaled"] = float(np.mean(residual))
        out["mean_observed_efficiency"] = float(np.mean(o))
        out["mean_model_efficiency"] = float(np.mean(m))
        out["direction_correlation"] = float(np.corrcoef(m, o)[0, 1]) if ok.sum() >= 3 else None
    return out


def calibrate_wake_decay(
    observation: FarmObservation,
    layout: WindFarmLayout,
    curves: dict,
    *,
    kind: str = "gaussian",
    k_grid: Sequence[float] = (0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.09, 0.11, 0.13, 0.16),
) -> dict:
    """Fit the single wake-decay constant to observed farm efficiency.

    One parameter, fitted on a **validation** window and reported on a later
    test window by the caller. The textbook values (0.05 offshore, 0.075 onshore)
    are defaults from other sites; a farm with a published layout and five years
    of SCADA is exactly the case where that constant should be measured instead
    of assumed -- as long as the fit is not then scored on its own data.

    Both ``k`` and ``gaussian_k`` are swept together so the returned constant
    means the same thing for either model family.
    """
    rows = []
    best = None
    for k in k_grid:
        cfg = WakeConfig(kind=kind, k=float(k), gaussian_k=float(k))
        comparison = compare_with_model(observation, layout, curves, config=cfg, fit_scale=False)
        if comparison["n_cells"] < 3 or comparison["rmse"] is None:
            continue
        rows.append({"k": float(k), "rmse": comparison["rmse"], "bias": comparison["bias"],
                     "n_cells": comparison["n_cells"]})
        if best is None or comparison["rmse"] < best["rmse"]:
            best = {"k": float(k), "rmse": comparison["rmse"], "bias": comparison["bias"],
                    "n_cells": comparison["n_cells"]}
    if best is None:
        raise ValueError("no wake-decay value could be scored on this observation")
    return {
        "kind": kind,
        "best_k": best["k"],
        "best_rmse": best["rmse"],
        "best_bias": best["bias"],
        "sweep": rows,
        "textbook_k": 0.075 if kind == "jensen" else 0.0324555,
        "note": (
            "Fitted on the validation window only. The comparison absorbs reference-group and "
            "instrument effects, so the fitted k is an effective constant for this site, not a "
            "physical measurement of wake decay."
        ),
    }


def _mean_curve(curves: Sequence) -> Callable:
    """Pointwise mean of several power curves, on the 0-30 m/s grid used for inversion."""

    def curve(speeds):
        speeds = np.asarray(speeds, dtype=float)
        return np.mean([np.asarray(c(speeds), dtype=float) for c in curves], axis=0)

    return curve


def _subset_layout(layout: WindFarmLayout, names: Sequence[str]) -> WindFarmLayout:
    keep = set(names)
    return WindFarmLayout(layout.name + " (subset)", [t for t in layout.turbines if t.name in keep])


# ---------------------------------------------------------------------------
# Physics guardrails
# ---------------------------------------------------------------------------

def physical_guardrails(
    predict,
    *,
    rated_power_kw: float,
    cut_in: float,
    cut_out: float,
    rotor_diameter: float,
    rho: float = 1.225,
    n_grid: int = 601,
) -> list[dict]:
    """Re-measure the physical claims on the *fitted* model, independently.

    By-construction properties are still checked rather than asserted: a bug in
    the cap or a bad knot array would show up here, and a claim that is never
    re-measured is a claim that rots. Each entry follows the guardrail schema
    VeriPhysics consumes: ``name``, ``passed``, ``detail``.
    """
    u = np.linspace(0.0, 32.0, n_grid)
    p = np.asarray(predict(u), dtype=float)
    checks: list[dict] = []

    peak = float(np.nanmax(p))
    checks.append({
        "name": "conservation_power_cap",
        "passed": bool(peak <= rated_power_kw * 1.001 + 1e-6),
        "detail": f"max predicted power {peak:.1f} kW vs rated {rated_power_kw:.1f} kW",
    })

    window = (u >= cut_in) & (u <= cut_out)
    d = np.diff(p[window])
    worst = float(d.min()) if d.size else 0.0
    checks.append({
        "name": "conservation_monotonicity",
        "passed": bool(worst >= -1e-6),
        "detail": f"largest decrease of P(U) inside [{cut_in:g}, {cut_out:g}] m/s is {worst:.3e} kW",
    })

    below = p[u < cut_in]
    above = p[u > cut_out]
    outside = max(float(np.max(np.abs(below))) if below.size else 0.0,
                  float(np.max(np.abs(above))) if above.size else 0.0)
    checks.append({
        "name": "conservation_operating_window",
        "passed": bool(outside <= 1e-6),
        "detail": f"max |P| outside [{cut_in:g}, {cut_out:g}] m/s is {outside:.3e} kW",
    })

    area = np.pi * (rotor_diameter / 2.0) ** 2
    flux = 0.5 * rho * area * u**3 / 1000.0
    cp = np.where(flux > 0, np.clip(p / np.maximum(flux, 1e-12), 0.0, 10.0), 0.0)
    cp_max = float(np.max(cp))
    checks.append({
        "name": "conservation_betz_limit",
        "passed": bool(cp_max <= 16.0 / 27.0 + 1e-6),
        "detail": f"implied Cp reaches {cp_max:.3f}; Betz limit is {16.0 / 27.0:.3f}",
    })
    return checks


# ---------------------------------------------------------------------------
# VeriPhysics
# ---------------------------------------------------------------------------

def power_applicability_map(
    *,
    architecture: str,
    train: pd.DataFrame,
    test_metrics: dict,
    guardrails: Sequence[dict],
    calibration: dict | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    rmse_threshold_kw: float,
    final_loss: float | None = None,
    loss_history: Sequence[float] | None = None,
    epochs: int | None = None,
    n_parameters: int | None = None,
    loss_weights: dict[str, float] | None = None,
):
    """Assemble the VeriPhysics applicability map from what this POC measured.

    ``heldout_error`` is the temporal-holdout error against a *declared*
    threshold; ``reference_data_match`` is the same fact in guardrail form, so
    the map's error item is ``VERIFIED`` only when a real holdout was scored.
    """
    from pinneapple_veriphysics.applicability import build_applicability_map

    domain_bounds = bounds or {
        "wind_speed": (float(train["wind_speed_ms"].quantile(0.001)),
                       float(train["wind_speed_ms"].quantile(0.999))),
        "relative_density": (float(train["rho_relative"].quantile(0.001)),
                             float(train["rho_relative"].quantile(0.999))),
    }
    checks = list(guardrails)
    rmse = float(test_metrics["rmse_kw"])
    checks.append({
        "name": "reference_data_match",
        "passed": bool(rmse <= rmse_threshold_kw),
        "detail": (
            f"temporal holdout RMSE {rmse:.1f} kW against a declared threshold "
            f"of {rmse_threshold_kw:.1f} kW (energy-weighted "
            f"{test_metrics['energy_weighted_rmse_kw']:.1f} kW)"
        ),
    })
    confidence: list[dict] = []
    if calibration is not None and calibration.get("coverage") is not None:
        target = float(calibration.get("target_coverage", calibration.get("target", 0.9)))
        confidence.append({
            "name": "uq_calibration",
            "score": float(1.0 - abs(calibration["coverage"] - target)),
            "source_summary": (
                f"split-conformal interval from the validation window: "
                f"{calibration['coverage']:.1%} empirical coverage on the test window "
                f"against a {target:.0%} target "
                f"(half-width {calibration.get('half_width_kw', float('nan')):.0f} kW)"
            ),
        })
    calibration_payload = None
    if calibration is not None and calibration.get("coverage") is not None:
        target = float(calibration.get("target_coverage", calibration.get("target", 0.9)))
        calibration_payload = {
            "ece": abs(float(calibration["coverage"]) - target),
            "coverage": float(calibration["coverage"]),
            "target_coverage": target,
        }
    return build_applicability_map(
        architecture=architecture,
        coords=tuple(domain_bounds),
        fields=("power_kw",),
        domain_bounds=domain_bounds,
        physical_parameters={"cut_out_ms": 25.0},
        guardrail_checks=checks,
        confidence_components=confidence,
        final_loss=final_loss,
        loss_history=loss_history,
        epochs=epochs,
        n_parameters=n_parameters,
        loss_weights=loss_weights,
        calibration=calibration_payload,
        heldout_error={
            "metric": "rmse_kw",
            "value": rmse,
            "threshold": rmse_threshold_kw,
            "description": "temporal holdout (later period than training), operating samples only",
        },
        extrapolation=None,
    )
