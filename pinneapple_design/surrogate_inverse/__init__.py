"""Surrogate-driven inverse design on closed-form physics, with the answer re-checked against the physics.

Five worked cases share one method (:mod:`.core`): sample the design space, label it with a closed-form model from
:mod:`pinneapple_physics.closed_form`, train a small MLP, freeze it, run Adam on the design variables through it, and
grade the final design against the closed form (:class:`.core.VerificationCard`, also as a
:class:`pinneapple_analysis.trust.TrustReport`). Each case adds one analysis of its own:

=====================  =========================================================  ===================================
module                 design                                                     extra analysis
=====================  =========================================================  ===================================
``heatsink_fins``      straight-fin heat sink (fin count, height, thickness)      2-D finite-volume fin-cell check
``brake_rotor``        vented brake rotor vanes (count, gap, thickness)           Monte Carlo of machining tolerances
``transdermal_patch``  membrane-controlled patch (area, membrane, concentration)  release over the wear time
``suspension_arm``     rectangular cantilever arm (width, thickness), min. mass   local sensitivities (elasticities)
``reflex_port``        bass-reflex port (radius, length) for a tuning frequency   volume-vs-accuracy trade-off sweep
=====================  =========================================================  ===================================

Each case module also builds its part as a triangle mesh in millimetres (:mod:`.meshes`) for a viewer or STL file.
The models are textbook closed forms with stated assumptions (each module docstring); none is validated against
measurements. The apps ``apps/heatsink_design_ai``, ``apps/brakecool_ai``, ``apps/patchdose_ai``,
``apps/ridetune_ai`` and ``apps/soundshape_ai`` serve them.
"""
from __future__ import annotations

from . import (
           brake_rotor,
           core,
           heatsink_fins,
           meshes,
           reflex_port,
           suspension_arm,
           transdermal_patch,
)
from .core import MLPSurrogate, SurrogateFit, VerificationCard, descend, fit_surrogate

__all__ = ["core", "meshes", "heatsink_fins", "brake_rotor", "transdermal_patch", "suspension_arm", "reflex_port",
           "MLPSurrogate", "SurrogateFit", "VerificationCard", "descend", "fit_surrogate"]
