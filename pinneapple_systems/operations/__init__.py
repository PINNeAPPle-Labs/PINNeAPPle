"""Operations: run validated physical models and surrogates next to live plant data.

One shared plant (a heated buffer tank with mass and energy balances, sensors and faults) and three tools built on
it, each usable alone and all speaking ``pinneapple_data.contracts``:

* :mod:`.twin`     -- digital twin runtime: EnKF state estimation, forecasts, alerts, drift monitoring and
                      revalidation, history in SQLite/Postgres;
* :mod:`.anomaly`  -- physics anomaly detection: statistics, time-series models, conservation balances and model
                      residuals, events with verifiable explanations, a labelled fault benchmark;
* :mod:`.control`  -- model predictive control (linear and nonlinear) against PID and rule-based baselines, in
                      simulation / recommendation mode only.

:mod:`.rom` fits input-driven reduced models (operator inference with inputs), :mod:`.series` chains the days
13-31 apps (campaign -> surrogate -> uncertainty -> optimizer -> twin) and :mod:`.contracts` converts models and
studies to ``ModelRef`` / ``StudyResult``. The embedded MQTT broker lives in
:mod:`pinneapple_systems.digital_twin.io.mqtt_broker`.
"""
from __future__ import annotations

from .anomaly import (
    BalanceRule,
    DetectionResult,
    DetectorConfig,
    Event,
    detect,
    fault_benchmark,
    run_benchmark,
    score_events,
    tank_balance_rules,
    tank_constants,
)
from .control import (
    MODE,
    ControlScenario,
    ControlSpec,
    LinearMPC,
    NonlinearMPC,
    PIDBaseline,
    RuleBasedBaseline,
    compare_controllers,
    recommend,
    run_closed_loop,
)
from .plant import (
    Fault,
    Scenario,
    SensorNoise,
    TankParams,
    TankSimulator,
    demo_scenario,
    energy_balance_residual,
    mass_balance_residual,
    simulate,
    steady_state,
)
from .rom import ControlledOpInf, fit_controlled_opinf, residual_sigma
from .twin import (
    AlertRules,
    LumpedTankModel,
    SurrogateModel,
    TwinRuntime,
    TwinStore,
    model_from_dict,
)

__all__ = [
    "TankParams", "Fault", "Scenario", "SensorNoise", "TankSimulator", "simulate", "demo_scenario", "steady_state",
    "mass_balance_residual", "energy_balance_residual",
    "ControlledOpInf", "fit_controlled_opinf", "residual_sigma",
    "TwinRuntime", "TwinStore", "AlertRules", "LumpedTankModel", "SurrogateModel", "model_from_dict",
    "BalanceRule", "DetectorConfig", "DetectionResult", "Event", "detect", "fault_benchmark", "score_events",
    "run_benchmark", "tank_balance_rules", "tank_constants",
    "ControlSpec", "ControlScenario", "PIDBaseline", "RuleBasedBaseline", "LinearMPC", "NonlinearMPC",
    "run_closed_loop", "compare_controllers", "recommend", "MODE",
]
