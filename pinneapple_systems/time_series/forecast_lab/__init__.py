"""Forecasting near-random series without fooling yourself: one interface for baselines, data-driven, data +
physics, physical and hybrid forecasters, and the protocol that compares them (purged walk-forward tuning, an
untouched test period, point and probabilistic scores, Diebold-Mariano tests with multiple-comparison control).

See ``docs/guides/near_random_forecasting.md``.
"""
from .base import Forecast, Forecaster, SeriesData, forecast_origins, targets_at, train_end
from .baselines import (
                        ARForecaster,
                        ARIMAForecaster,
                        Climatology,
                        ExponentialSmoothing,
                        Persistence,
                        PhaseClimatology,
                        SeasonalNaive,
                        dominant_period,
)
from .hybrid import ResidualCorrectionForecaster, UDEForecaster
from .learned import (
                        GBMForecaster,
                        LagFeatures,
                        NeuralForecaster,
                        PhysicsLossMLP,
                        train_torch_regressor,
)
from .physical import (
                        DelayODEForecaster,
                        EnKFForecaster,
                        LinearSSMForecaster,
                        ODEForecaster,
                        PhysicsFeatures,
                        SDEDriftForecaster,
                        SINDyForecaster,
)

__all__ = [
    "Forecast", "Forecaster", "SeriesData", "forecast_origins", "targets_at", "train_end",
    "Persistence", "Climatology", "ARForecaster", "ARIMAForecaster", "ExponentialSmoothing", "SeasonalNaive",
    "PhaseClimatology", "dominant_period",
    "LagFeatures", "GBMForecaster", "NeuralForecaster", "PhysicsLossMLP", "train_torch_regressor",
    "ODEForecaster", "DelayODEForecaster", "LinearSSMForecaster", "SDEDriftForecaster", "EnKFForecaster",
    "SINDyForecaster", "PhysicsFeatures", "ResidualCorrectionForecaster", "UDEForecaster",
]
