"""Built-in experiments. Each module registers its experiments on import.

| name | what | dataset |
|---|---|---|
| ``oscillator`` | damped oscillator, RK4 / Euler / symplectic Euler vs the exact solution | trajectories |
| ``heat_xtfc`` | 1D heat equation solved by X-TFC vs the exact solution | fields |
| ``bondi_accretion`` | black-hole hydro solver vs exact Bondi accretion (Paczynski-Wiita) | profiles |
| ``accretion_flow`` | torus accreting onto a black hole; density movie, accretion rate, conservation | frames |
| ``cylinder_lbm`` | lattice-Boltzmann flow past a cylinder; regime, Strouhal number, vorticity images | vorticity |
| ``bh_forecast`` | Duarte et al. U-Net forecast of accretion flows, scored against persistence (source checkout) | - |
| ``repo_results`` | results already produced by repository scripts, re-validated and stored as datasets | per source |
| ``example_script`` | any script of ``examples/`` (examples and use cases): figures, JSON metrics, arrays, console, code | artifacts |
| ``car_lbm`` / ``car_surrogate`` | parametric 2-D car body in an LBM wind tunnel; FNO + MLP surrogates, design search verified by LBM | flow, vorticity, predictions |
| ``vehicle_cfd`` | 3-D road car / launch vehicle in OpenFOAM: coefficients, skin Cp, streamlines, Blender renders, 3-D viewer | surface, streamlines |
| ``bar_wear`` / ``bar_wear_ranking`` | Archard wear of a crowned bar on a counterface, per material; running-in, steady wear, ranking | wear history, ranking |
| ``benchmark_case`` | landing-page cases from PINNeAPPle-Benchmark and PINNeAPPle-Climate, headline claim re-checked | arrays, sweep, lead skill |
"""
from . import (  # noqa: F401
    accretion,
    blackhole_forecast,
    car,
    cylinder,
    discovery,
    examples,
    landing,
    ode,
    pde,
    repo_results,
    vehicles3d,
    wear,
)
