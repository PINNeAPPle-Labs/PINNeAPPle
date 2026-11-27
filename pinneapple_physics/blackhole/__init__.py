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
``skill`` (skill against persistence, climatology and DMD with bootstrap intervals, physics checks), ``twin``,
``raytrace``, ``capture``. ``hydro_numba`` and ``hydro_warp`` are the two optional backends of the solver
(``backend="numba"``, ``backend="warp"``; the second runs NVIDIA Warp kernels, on a GPU and behind ``wp.Tape``).
"""
from .hydro import AccretionFlow, RIAFConfig, accreted_mass, bondi_pw, torus_state

__all__ = ["AccretionFlow", "RIAFConfig", "accreted_mass", "bondi_pw", "torus_state"]
