"""Extreme-value analysis for climate and environmental series (#1198): GEV and GPD by maximum likelihood with
profile-likelihood intervals, covariates in the GEV location and scale, block maxima and peaks over a threshold
with declustering, copulas for compound events, a probability-ratio attribution, and the standard workflow
(:func:`extreme_climate_workflow`)."""
from .copula import CopulaFit, fit_copula, joint_exceedance
from .extraction import (
           block_maxima,
           choose_threshold,
           decluster_runs,
           mean_excess,
           threshold_stability,
)
from .models import GEVFit, GPDFit, fit_gev, fit_gpd, gev_return_level, gpd_return_level
from .workflow import CASES, extreme_climate_workflow, load_case

__all__ = ["CopulaFit", "fit_copula", "joint_exceedance", "block_maxima", "choose_threshold", "decluster_runs",
           "mean_excess", "threshold_stability", "GEVFit", "GPDFit", "fit_gev", "fit_gpd", "gev_return_level",
           "gpd_return_level", "CASES", "extreme_climate_workflow", "load_case"]
