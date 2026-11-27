"""Wind energy: SCADA power curves, analytical wake models and the farm behind them.

This is the vertical of PINNeAPPle that speaks wind-farm engineering. It answers
three questions with the same objects, so a result can always be traced back to
its inputs:

1. **How much power does one turbine make at this wind?**
   :mod:`~pinneapple_systems.wind.power_curve` holds the reference (analytic) and
   IEC binned curves; :mod:`~pinneapple_systems.wind.models` holds a
   physics-informed monotone model and a conventional machine-learning baseline
   to judge it against.
2. **How much of that reaches the next turbine?**
   :mod:`~pinneapple_systems.wind.wake` implements the Jensen/PARK and
   Gaussian (Bastankhah & Porté-Agel) wakes, root-sum-square superposition, and
   the farm model that runs them over a layout.
3. **Can I trust either number here?**
   :mod:`~pinneapple_systems.wind.verification` measures the training envelope,
   compares the farm model with SCADA-observed efficiency, and assembles the
   VeriPhysics applicability map from the checks that actually ran.

The dataset loader is :mod:`~pinneapple_systems.wind.penmanshiel` (14 Senvion
MM82 in the Scottish Borders, CC-BY, doi:10.5281/zenodo.16807304).

What this package is not
------------------------
The wake models are analytical, first-order engineering models -- not CFD. They
reproduce the direction dependence and the order of magnitude of wake losses and
are wrong in detail. Nothing here has been validated as a high-fidelity flow
solution, and a report that shows a wake number must carry the model name and
its configuration next to it.

Quick start
-----------
::

    from pinneapple_systems import wind

    layout = wind.static_layout("data/penmanshiel")
    curve = wind.ReferencePowerCurve(rated_power_kw=2050, rotor_diameter=82)
    result = wind.farm_power(layout, 8.0, 270.0, curve.power)
    result.total_power_kw, result.wake_loss_fraction
"""
from __future__ import annotations

from .layout import WindFarmLayout, Turbine, wind_vector
from .power_curve import (
    RHO_REF,
    BinnedPowerCurve,
    LogisticPowerCurve,
    ReferencePowerCurve,
    air_density,
    betz_limit,
    density_corrected_speed,
    fit_binned_curve,
    fit_logistic_curve,
    relative_density,
    standard_pressure,
)
from .models import (
    ALTERNATING_MET_FEATURES,
    ConstantPowerModel,
    FrameColumnModel,
    GradientBoostedPowerModel,
    PhysicsInformedPowerModel,
    SpeedCurveModel,
    feature_matrix,
)
from .metrics import calibration_report, conformal_interval, per_bin_error, point_metrics
from .wake import (
    FarmPowerResult,
    WakeConfig,
    direction_polar,
    farm_efficiency,
    farm_power,
    gaussian_deficit,
    jensen_deficit,
    pair_geometry,
    wake_deficits,
)
from .penmanshiel import (
    PENMANSHIEL_DOI,
    add_density_correction,
    curtailment_mask,
    discover_archives,
    load_farm,
    load_turbine,
    mask_provenance,
    operating_mask,
    qc_report,
    static_layout,
    summarise_by_turbine,
)
from .optimization import (
    CurtailmentResult,
    FarmSurrogate,
    annual_energy_production,
    curtailment_study,
    fit_surrogate,
    optimize_curtailment,
)
from .verification import (
    FarmObservation,
    compare_with_model,
    curve_inverse,
    domain_report,
    farm_efficiency_observed,
    physical_guardrails,
    power_applicability_map,
)

__all__ = [
    # layout
    "WindFarmLayout", "Turbine", "wind_vector",
    # power curves
    "RHO_REF", "ReferencePowerCurve", "BinnedPowerCurve", "LogisticPowerCurve",
    "air_density", "betz_limit", "density_corrected_speed", "fit_binned_curve",
    "fit_logistic_curve", "relative_density", "standard_pressure",
    # models
    "ALTERNATING_MET_FEATURES", "ConstantPowerModel", "FrameColumnModel", "GradientBoostedPowerModel",
    "PhysicsInformedPowerModel", "SpeedCurveModel", "feature_matrix",
    # metrics
    "calibration_report", "conformal_interval", "per_bin_error", "point_metrics",
    # wake
    "FarmPowerResult", "WakeConfig", "direction_polar", "farm_efficiency", "farm_power",
    "gaussian_deficit", "jensen_deficit", "pair_geometry", "wake_deficits",
    # data
    "PENMANSHIEL_DOI", "add_density_correction", "curtailment_mask", "discover_archives",
    "load_farm", "load_turbine", "mask_provenance", "operating_mask", "qc_report",
    "static_layout", "summarise_by_turbine",
    # verification
    "FarmObservation", "compare_with_model", "curve_inverse", "domain_report",
    "farm_efficiency_observed", "physical_guardrails", "power_applicability_map",
    # optimisation and surrogate
    "CurtailmentResult", "FarmSurrogate", "annual_energy_production", "curtailment_study",
    "fit_surrogate", "optimize_curtailment",
]
