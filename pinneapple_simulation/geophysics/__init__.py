"""Earth-system building blocks, each verified against an exact solution or a standard test:

* ``sphere_shallow_water``: spectral-transform shallow-water model on the sphere and the Williamson et al. (1992)
  test cases 2, 5 and 6, with mass / energy / potential-enstrophy diagnostics;
* ``richards``: 1-D Richards equation (mixed form, Celia et al. 1990) with van Genuchten-Mualem, Brooks-Corey,
  Clapp-Hornberger/Campbell and Gardner soils;
* ``energy_balance``: two-layer energy-balance model of global warming (Held et al. 2010; Geoffroy et al. 2013) with
  exact integration, closed-form step response, ECS/TCR and parameter fitting.
"""
from .energy_balance import F2X_DEFAULT, TwoLayerEBM, fit_two_layer
from .richards import (
                       CELIA_1990_SOIL,
                       Boundary,
                       BrooksCorey,
                       Campbell,
                       ClappHornberger,
                       Gardner,
                       RichardsResult,
                       VanGenuchten,
                       gardner_steady_infiltration,
                       solve_richards,
)
from .sphere_shallow_water import (
                       EARTH,
                       SphereShallowWater,
                       SphericalHarmonics,
                       height_errors,
                       rossby_haurwitz_speed,
                       stable_dt,
                       williamson_case2,
                       williamson_case5,
                       williamson_case6,
)

__all__ = ["SphericalHarmonics", "SphereShallowWater", "williamson_case2", "williamson_case5", "williamson_case6",
           "height_errors", "rossby_haurwitz_speed", "stable_dt", "EARTH",
           "VanGenuchten", "BrooksCorey", "ClappHornberger", "Campbell", "Gardner", "Boundary", "RichardsResult",
           "solve_richards", "gardner_steady_infiltration", "CELIA_1990_SOIL",
           "TwoLayerEBM", "fit_two_layer", "F2X_DEFAULT"]
