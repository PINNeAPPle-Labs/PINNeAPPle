"""Global and local sensitivity analysis: which parameters really control the result.

=========================================  =====================================================================
``sobol_indices(fn, space, n)``            first-order, total (and second-order) Sobol indices, bootstrap intervals
``saltelli_design`` / ``sobol_from_runs``  the same from runs done elsewhere (OpenFOAM, CalculiX, a campaign)
``first_order_given_data(X, y)``           first-order indices from an existing dataset (no designed experiment)
``morris(fn, space, r)``                   elementary-effects screening (mu*, sigma), ``r (d + 1)`` evaluations
``effect_curves(fn, space)``               main effects (partial dependence) and ICE curves
``local_derivatives(fn, x0, torch_fn=)``   Jacobian by autodiff (torch) or central differences
``freeze_recommendation`` / ``verify_freeze``  which inputs can be fixed, and the direct check of that claim
``testfunctions``                          Ishigami and Sobol' G with their exact indices
``sensitivity_study`` / ``study_cost``     all of the above on one model, and its cost in evaluations
``gp_posterior_paths`` / ``ensemble_paths``  surrogate-error sources: GP posterior paths, one path per ensemble member
``paths_from_residuals``                   the same from calibration residuals (the weak source: no correlation)
``sobol_from_paths`` / ``sobol_with_paths``  Sobol indices whose intervals also cover the surrogate's error
``ranking_stability``                      rank-unchanged and top-k probabilities, Kendall tau across the draws
=========================================  =====================================================================

Inputs are described as in :mod:`pinneapple_analysis.uncertainty.distributions` (uniform, normal, truncated normal,
lognormal, triangular). The estimators are checked against the exact indices of the Ishigami and G functions in
``tests/test_sensitivity.py``; the coverage of the intervals that include the surrogate's error is counted in
``tests/test_sensitivity_surrogate_error.py``. The older
:class:`pinneapple_analysis.inverse_problems.GlobalSensitivity` (uniform bounds, no intervals) remains for its
callers.
"""
from pinneapple_analysis.sensitivity import testfunctions
from pinneapple_analysis.sensitivity.effects import (
    effect_curves,
    freeze_recommendation,
    local_derivatives,
    verify_freeze,
)
from pinneapple_analysis.sensitivity.morris import MorrisResult, morris, morris_design
from pinneapple_analysis.sensitivity.sobol import (
    SobolIndices,
    first_order_given_data,
    saltelli_design,
    sobol_from_evaluations,
    sobol_from_runs,
    sobol_indices,
)
from pinneapple_analysis.sensitivity.study import sensitivity_study, study_cost
from pinneapple_analysis.sensitivity.surrogate_error import (
    SurrogateError,
    ensemble_paths,
    gp_posterior_paths,
    paths_from_residuals,
    ranking_stability,
    sobol_from_paths,
    sobol_with_paths,
)

__all__ = ["SobolIndices", "sobol_indices", "saltelli_design", "sobol_from_evaluations", "sobol_from_runs",
           "first_order_given_data",
           "MorrisResult", "morris", "morris_design", "effect_curves", "local_derivatives", "freeze_recommendation",
           "verify_freeze", "testfunctions", "sensitivity_study", "study_cost",
           "SurrogateError", "gp_posterior_paths", "ensemble_paths", "paths_from_residuals", "ranking_stability",
           "sobol_from_paths", "sobol_with_paths"]
