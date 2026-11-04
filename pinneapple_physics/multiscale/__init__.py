"""Multiscale physics for adaptive ensembles: filters, closures, multiscale residuals, cross-scale consistency and
scale-aware model selection.

=====================  ==========================================================================================
module                 contents
=====================  ==========================================================================================
:mod:`.filters`        box, Gaussian and sharp spectral filters with exact transfer functions; restriction and
                       prolongation; post-filtering from one width to another and its realizability; variable-width
                       filtering and the commutation error
:mod:`.ns2d`           pseudo-spectral 2-D vorticity DNS / LES, energy spectrum, residual of the vorticity equation
:mod:`.closures`       Smagorinsky, dynamic Smagorinsky, gradient (Clark) model; a-priori and a-posteriori tests;
                       label-free closure health (enstrophy growth, spectral pile-up, Germano-identity error)
:mod:`.residuals`      residuals of the filtered equations at several widths, split into subfilter, commutation
                       and model parts
:mod:`.consistency`    cross-scale consistency: filter(fine prediction) against the coarse prediction
:mod:`.scales`         scale profiles of models, scale queries, scale compatibility, :class:`ScaleAwareRouter`
                       with a fidelity ladder (surrogate -> LES with closure -> resolved solver)
:mod:`.learned`        (PyTorch, imported on demand) scale-conditioned FNO with a filtering-consistency loss,
                       differentiable LES and a closure trained a posteriori through the solver; periodic CNN,
                       torus graph and a Fourier-Galerkin PINN for the benchmark surrogates
:mod:`.turbulence`     decaying 2-D turbulence benchmark: DNS data, filtered pairs, query streams that shift Re
                       and scale, closure benchmark, scale-conditioning study
:mod:`.sgs3d`          3-D a-priori benchmark of the subgrid closures that pinneapple ships for the PINN path
                       (Smagorinsky, dynamic Smagorinsky, WALE): Taylor-Green DNS, exact subfilter stress through
                       an aliasing-free sharp filter, the a-priori metrics, and a learned closure scored on a
                       held-out snapshot
:mod:`.comparison`     cached expert evaluation, scale-blind vs scale-aware ensembles on one stream, metrics,
                       tables and the summary figure
:mod:`.viz`            animation of a run (requested scale, chosen model, spectra, weights)
=====================  ==========================================================================================

Plain numpy except :mod:`.learned`. See ``docs/core_concepts/multiscale_ensembles.md``.
"""
from __future__ import annotations

from .closures import (
    AposterioriReport,
    AprioriReport,
    ClosureHealth,
    DynamicSmagorinsky,
    GradientModel,
    NoModel,
    Smagorinsky,
    aposteriori_test,
    apriori_test,
    closure_health,
    germano_error,
    spectrum_error,
    true_subfilter,
)
from .consistency import VORTICITY_INTEGRALS, ConsistencyReport, cross_scale_consistency
from .filters import (
    FILTER_KINDS,
    Filter,
    apply_filter,
    commutation_error,
    filter_variable_width,
    post_filter_transfer,
    prolong,
    realizability_mismatch,
    refilter,
    restrict,
    transfer,
)
from .ns2d import (
    VorticityNS2D,
    energy_spectrum,
    random_vorticity,
    velocity_from_vorticity,
    vorticity_residual,
)
from .residuals import MultiscaleResidual, multiscale_residuals, ns2d_multiscale_residuals
from .scales import (
    RUNGS,
    ScaleAwareRouter,
    ScaleCheck,
    ScaleProfile,
    ScaleQuery,
    ScaleRoute,
    scale_compatibility,
)

# ``sgs3d`` is imported lazily. Eagerly importing it here makes
# ``python -m pinneapple_physics.multiscale.sgs3d`` (the documented way to run the benchmark) load the module twice
# and emit a RuntimeWarning about it, which reads like a failure on a command that worked.
_SGS3D_NAMES = frozenset({
    "AprioriCase", "apriori_case", "apriori_metrics", "closure_eddy_viscosity", "gradient_model_stress",
    "ideal_eddy_viscosity", "learn_sgs_stress", "modelled_stress", "periodic_stencil", "run_benchmark",
    "sharp_filter", "spectral_gradient", "taylor_green_dns", "taylor_green_snapshots",
})


def __getattr__(name):
    if name in _SGS3D_NAMES:
        from . import sgs3d

        return getattr(sgs3d, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FILTER_KINDS", "Filter", "transfer", "apply_filter", "restrict", "prolong", "post_filter_transfer",
    "realizability_mismatch", "refilter", "filter_variable_width", "commutation_error",
    "VorticityNS2D", "random_vorticity", "energy_spectrum", "velocity_from_vorticity", "vorticity_residual",
    "Smagorinsky", "DynamicSmagorinsky", "GradientModel", "NoModel", "true_subfilter", "apriori_test",
    "AprioriReport", "aposteriori_test", "AposterioriReport", "closure_health", "ClosureHealth", "germano_error",
    "spectrum_error",
    "MultiscaleResidual", "multiscale_residuals", "ns2d_multiscale_residuals",
    "ConsistencyReport", "cross_scale_consistency", "VORTICITY_INTEGRALS",
    "RUNGS", "ScaleProfile", "ScaleQuery", "ScaleCheck", "scale_compatibility", "ScaleRoute", "ScaleAwareRouter",
    "AprioriCase", "apriori_case", "apriori_metrics", "closure_eddy_viscosity", "ideal_eddy_viscosity",
    "learn_sgs_stress", "modelled_stress", "periodic_stencil", "run_benchmark", "sharp_filter",
    "spectral_gradient", "taylor_green_dns", "taylor_green_snapshots",
]
