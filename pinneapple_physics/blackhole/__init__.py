"""Black-hole accretion: simulations and "black hole weather" forecasting with a trust horizon.

Reproduces and extends Duarte, Nemmen & Navarro (2022, MNRAS 512, 5848; code MIT,
github.com/black-hole-group/DL_BH_fluids). See ``examples/black_hole_weather/`` and issue #399.

    from pinneapple_physics.blackhole import RIAFConfig, AccretionFlow, torus_state
    flow = AccretionFlow(RIAFConfig(nr=128, ntheta=64, alpha=0.1, viscosity="SS"))
    flow.set_primitives(torus_state(flow))
    out = flow.run(t_end=2000.0, every=20.0)      # frames (T, 5, nr, ntheta): rho, v_r, v_theta, v_phi, p
"""
from .hydro import AccretionFlow, RIAFConfig, bondi_pw, torus_state

__all__ = ["AccretionFlow", "RIAFConfig", "bondi_pw", "torus_state"]
