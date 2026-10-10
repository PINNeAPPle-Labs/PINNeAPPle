"""Built-in experiments. Each module registers its experiments on import.

| name | what | dataset |
|---|---|---|
| ``oscillator`` | damped oscillator, RK4 / Euler / symplectic Euler vs the exact solution | trajectories |
| ``heat_xtfc`` | 1D heat equation solved by X-TFC vs the exact solution | fields |
| ``bondi_accretion`` | black-hole hydro solver vs exact Bondi accretion (Paczynski-Wiita) | profiles |
| ``accretion_flow`` | torus accreting onto a black hole; density movie, accretion rate, conservation | frames |
| ``cylinder_lbm`` | lattice-Boltzmann flow past a cylinder; regime, Strouhal number, vorticity images | vorticity |
| ``bh_forecast`` | Duarte et al. U-Net forecast of accretion flows, scored against persistence (source checkout) | - |
"""
from . import accretion, blackhole_forecast, cylinder, ode, pde  # noqa: F401
