"""Data assimilation: one problem object over the variational and the ensemble analysis (``assimilation``),
the two kernels it runs -- 4D-Var through autograd (``var4d``) and the ensemble Kalman analysis from
``pinneapple_analysis.state_estimation`` -- a model trained here as the forward operator (``neural``), the ensemble
methods in ``pinneapple_analysis.inverse_problems`` (EKI), plus Gaussian-process field reconstruction from sparse
sensors and greedy sensor placement (``sensor_placement``)."""
from .assimilation import METHODS, AnalysisReport, Assimilation, cyclic_localisation
from .models import lorenz63_step, lorenz96_step, lorenz96_tendency
from .neural import (
    PeriodicTendency,
    lorenz96_trajectory,
    neural_twin_experiment,
    surrogate_step,
    train_tendency_surrogate,
)
from .sensor_placement import (
    GaussianProcessField,
    SensorPlacement,
    greedy_sensor_placement,
    matern_kernel,
)
from .var4d import Observation, Var4D, Var4DResult, gradient_check, twin_experiment

__all__ = ["Observation", "Var4D", "Var4DResult", "gradient_check", "twin_experiment", "lorenz96_step",
           "lorenz96_tendency", "lorenz63_step", "GaussianProcessField", "SensorPlacement", "greedy_sensor_placement",
           "matern_kernel", "Assimilation", "AnalysisReport", "METHODS", "cyclic_localisation",
           "PeriodicTendency", "lorenz96_trajectory", "neural_twin_experiment", "surrogate_step",
           "train_tendency_surrogate"]
