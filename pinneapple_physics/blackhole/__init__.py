"""Black-hole accretion: simulations and "black hole weather" forecasting with a trust horizon.

Reproduces and extends Duarte, Nemmen & Navarro (2022, MNRAS 512, 5848; code MIT,
github.com/black-hole-group/DL_BH_fluids). See ``examples/black_hole_weather/`` and issue #399.

    from pinneapple_physics.blackhole import RIAFConfig, AccretionFlow, torus_state
    flow = AccretionFlow(RIAFConfig(nr=128, ntheta=64, alpha=0.1, viscosity="SS"))
    flow.set_primitives(torus_state(flow))
    out = flow.run(t_end=2000.0, every=20.0)      # frames (T, 5, nr, ntheta): rho, v_r, v_theta, v_phi, p

Submodules: ``runs`` (generate / resume / read simulation runs), ``convergence`` (resolution study: conservative
remap, observed order and GCI, statistical convergence), ``dataset`` (the zoo dataset of whole trajectories),
``forecast`` (the U-Net, training with rollout loss / noise / mass penalty, rollouts, trust horizon, mass envelope),
``skill`` (skill against persistence, climatology and DMD with bootstrap intervals, physics checks), ``family``
(the model family -- the paper's U-Net, a Fourier neural operator and the weather module's SphereUNet backbone -- and
the equal-weight and adaptive ensembles over them), ``twin``, ``raytrace``, ``capture``.
"""
from .hydro import AccretionFlow, RIAFConfig, bondi_pw, torus_state

__all__ = ["AccretionFlow", "RIAFConfig", "bondi_pw", "torus_state"]
