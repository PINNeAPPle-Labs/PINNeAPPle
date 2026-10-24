# PINNeAPPle science: experiments

Every row is a quantity computed by an experiment of `pinneapple_science` (`pp.science`) and compared with its reference; the values come from the committed `result.json` files. Regenerate with `python -m pinneapple_science report all`. The map of every area (available, partial, planned) is [ATLAS.md](ATLAS.md) and [atlas.html](atlas.html).

| experiment | dim | level | kind | quantity | computed | reference | error | tol | reference kind | trust |
|---|---|---|---|---|---|---|---|---|---|---|
| [em.ampere_law](electromagnetism/ampere_law/README.md) | 2D | academic | theory | circle around the conductor / mu0 | 10.2076 | 10.2083 | 7.61e-05 | 0.002 | closed_form | APPROVED |
| [em.ampere_law](electromagnetism/ampere_law/README.md) | 2D | academic | theory | square around the conductor / mu0 | 10.2075 | 10.2083 | 8.15e-05 | 0.002 | closed_form | APPROVED |
| [em.ampere_law](electromagnetism/ampere_law/README.md) | 2D | academic | theory | circle enclosing nothing / mu0 | 3.80556e-05 | 0 | 3.81e-05 (abs.) | 0.02 (abs.) | conservation | APPROVED |
| [em.ampere_law](electromagnetism/ampere_law/README.md) | 3D | academic | theory | circle around one side | 10 | 10 | 0.00e+00 | 1e-06 | closed_form | APPROVED |
| [em.ampere_law](electromagnetism/ampere_law/README.md) | 3D | academic | theory | tilted ellipse, linking once | 10 | 10 | 1.78e-16 | 1e-06 | closed_form | APPROVED |
| [em.ampere_law](electromagnetism/ampere_law/README.md) | 3D | academic | theory | path winding twice | 20 | 20 | 0.00e+00 | 1e-06 | closed_form | APPROVED |
| [em.ampere_law](electromagnetism/ampere_law/README.md) | 3D | academic | theory | path not linked (a circle inside the loop, above it) | -7.94093e-18 | 0 | 7.94e-18 (abs.) | 1e-08 (abs.) | conservation | APPROVED |
| [em.bench_mu0_solenoid](electromagnetism/bench_mu0_solenoid/README.md) | 3D | academic | practice | mu0 (bench) vs CODATA 2022 | 1.25169e-06 | 1.25664e-06 | 3.94e-03 | 0.01 | codata | APPROVED |
| [em.bench_rc_discharge](electromagnetism/bench_rc_discharge/README.md) | 0D | academic | practice | C (bench) vs the capacitor's true value | 4.30371e-07 | 4.31051e-07 | 1.58e-03 | 0.0029 | ground_truth | APPROVED |
| [em.bench_rc_discharge](electromagnetism/bench_rc_discharge/README.md) | 0D | academic | practice | C vs the nominal label (470 nF +- 10 %) | 4.30371e-07 | 4.7e-07 | 8.43e-02 | 0.1 | ground_truth | APPROVED |
| [em.biot_savart](electromagnetism/biot_savart/README.md) | 3D | academic | theory | finite wire B (midpoint rule) | 1.99519e-05 | 1.99519e-05 | 4.38e-08 | 0.0001 | closed_form | APPROVED |
| [em.biot_savart](electromagnetism/biot_savart/README.md) | 3D | academic | theory | loop B, 5 points on and off axis (worst) | 1.00017 | 1 | 1.71e-04 | 0.001 | closed_form | APPROVED |
| [em.biot_savart](electromagnetism/biot_savart/README.md) | 3D | academic | theory | loop centre B | 2.5134e-05 | 2.51327e-05 | 5.02e-05 | 0.001 | closed_form | APPROVED |
| [em.biot_savart](electromagnetism/biot_savart/README.md) | 3D | academic | theory | solenoid centre B (helix) | 0.00243844 | 0.00243823 | 8.39e-05 | 0.002 | closed_form | APPROVED |
| [em.biot_savart](electromagnetism/biot_savart/README.md) | 3D | academic | theory | solenoid end B (helix) | 0.00124696 | 0.00124693 | 2.19e-05 | 0.01 | closed_form | APPROVED |
| [em.biot_savart](electromagnetism/biot_savart/README.md) | 3D | academic | theory | long-solenoid limit mu0 n I | 0.00243823 | 0.00251327 | 2.99e-02 | 0.05 | limit | APPROVED |
| [em.cavity_resonance](electromagnetism/cavity_resonance/README.md) | 2D | academic | theory | TM11 resonance | 2.61385e+08 | 2.61388e+08 | 1.35e-05 | 0.002 | closed_form | APPROVED |
| [em.cavity_resonance](electromagnetism/cavity_resonance/README.md) | 2D | academic | theory | TM21 resonance | 3.68407e+08 | 3.68416e+08 | 2.38e-05 | 0.002 | closed_form | APPROVED |
| [em.cavity_resonance](electromagnetism/cavity_resonance/README.md) | 2D | academic | theory | TM12 resonance | 4.53665e+08 | 4.53749e+08 | 1.86e-04 | 0.002 | closed_form | APPROVED |
| [em.cavity_resonance](electromagnetism/cavity_resonance/README.md) | 2D | academic | theory | TM31 resonance | 4.97998e+08 | 4.98071e+08 | 1.47e-04 | 0.002 | closed_form | APPROVED |
| [em.cavity_resonance](electromagnetism/cavity_resonance/README.md) | 2D | academic | theory | TM22 resonance | 5.22748e+08 | 5.22776e+08 | 5.36e-05 | 0.002 | closed_form | APPROVED |
| [em.cavity_resonance](electromagnetism/cavity_resonance/README.md) | 3D | academic | theory | (110) resonance | 2.6131e+08 | 2.61388e+08 | 2.99e-04 | 0.002 | closed_form | APPROVED |
| [em.cavity_resonance](electromagnetism/cavity_resonance/README.md) | 3D | academic | theory | (101) resonance | 3.34943e+08 | 3.35178e+08 | 7.02e-04 | 0.002 | closed_form | APPROVED |
| [em.cavity_resonance](electromagnetism/cavity_resonance/README.md) | 3D | academic | theory | (011) resonance | 3.68213e+08 | 3.68416e+08 | 5.51e-04 | 0.002 | closed_form | APPROVED |
| [em.cavity_resonance](electromagnetism/cavity_resonance/README.md) | 3D | academic | theory | (111) resonance | 3.97636e+08 | 3.97743e+08 | 2.69e-04 | 0.002 | closed_form | APPROVED |
| [em.circuits_rc_rl_rlc](electromagnetism/circuits_rc_rl_rlc/README.md) | 0D | academic | theory | RC time constant (fit of ln(1 - V/V0)) | 0.001 | 0.001 | 1.52e-11 | 1e-06 | closed_form | APPROVED |
| [em.circuits_rc_rl_rlc](electromagnetism/circuits_rc_rl_rlc/README.md) | 0D | academic | theory | RL time constant | 0.0001 | 0.0001 | 1.86e-11 | 1e-06 | closed_form | APPROVED |
| [em.circuits_rc_rl_rlc](electromagnetism/circuits_rc_rl_rlc/README.md) | 0D | academic | theory | RL final current | 0.005 | 0.005 | 2.29e-12 | 1e-06 | closed_form | APPROVED |
| [em.circuits_rc_rl_rlc](electromagnetism/circuits_rc_rl_rlc/README.md) | 0D | academic | theory | RLC damped angular frequency (zero crossings) | 3152.38 | 3152.38 | 1.96e-11 | 0.0001 | closed_form | APPROVED |
| [em.circuits_rc_rl_rlc](electromagnetism/circuits_rc_rl_rlc/README.md) | 0D | academic | theory | RLC decay rate (log decrement of extrema) | 250 | 250 | 7.96e-11 | 0.001 | closed_form | APPROVED |
| [em.circuits_rc_rl_rlc](electromagnetism/circuits_rc_rl_rlc/README.md) | 0D | academic | theory | energy dissipated in R (free decay) | 1.25e-05 | 1.25e-05 | 5.85e-11 | 1e-06 | conservation | APPROVED |
| [em.coulomb_superposition](electromagnetism/coulomb_superposition/README.md) | 3D | academic | theory | ring potential at s = 0.6 R, z = 0.3 R | 183.158 | 183.158 | 6.21e-16 | 1e-06 | closed_form | APPROVED |
| [em.coulomb_superposition](electromagnetism/coulomb_superposition/README.md) | 3D | academic | theory | axial field E_z(z = 0.75 R) | 1380.49 | 1380.49 | 6.59e-16 | 1e-10 | closed_form | APPROVED |
| [em.coulomb_superposition](electromagnetism/coulomb_superposition/README.md) | 3D | academic | theory | far field -> point charge (r = 40 R) | 2.24478 | 2.24689 | 9.37e-04 | 0.001 | limit | APPROVED |
| [em.coulomb_superposition](electromagnetism/coulomb_superposition/README.md) | 2D | academic | theory | line-pair field E_x at (0, d/2) | 898.396 | 898.396 | 3.81e-10 | 1e-06 | closed_form | APPROVED |
| [em.coulomb_superposition](electromagnetism/coulomb_superposition/README.md) | 2D | academic | theory | finite line -> infinite line (s = 0.01 m, L = 1 m) | 1797.42 | 1797.51 | 5.00e-05 | 0.001 | limit | APPROVED |
| [em.dipole_radiation](electromagnetism/dipole_radiation/README.md) | 3D | academic | theory | radiated power through the sphere kr = 0.3 | 0.00438953 | 0.00438953 | 3.95e-16 | 1e-09 | closed_form | APPROVED |
| [em.dipole_radiation](electromagnetism/dipole_radiation/README.md) | 3D | academic | theory | radiated power through the sphere kr = 1 | 0.00438953 | 0.00438953 | 9.88e-16 | 1e-09 | closed_form | APPROVED |
| [em.dipole_radiation](electromagnetism/dipole_radiation/README.md) | 3D | academic | theory | radiated power through the sphere kr = 3 | 0.00438953 | 0.00438953 | 9.88e-16 | 1e-09 | closed_form | APPROVED |
| [em.dipole_radiation](electromagnetism/dipole_radiation/README.md) | 3D | academic | theory | radiated power through the sphere kr = 30 | 0.00438953 | 0.00438953 | 5.93e-16 | 1e-09 | closed_form | APPROVED |
| [em.dipole_radiation](electromagnetism/dipole_radiation/README.md) | 3D | academic | theory | directivity | 1.5 | 1.5 | 4.44e-16 | 0.001 | closed_form | APPROVED |
| [em.dipole_radiation](electromagnetism/dipole_radiation/README.md) | 3D+t | academic | theory | energy through the sphere r = lambda in one period | 4.38953e-12 | 4.38953e-12 | 9.20e-16 | 1e-09 | closed_form | APPROVED |
| [em.faraday_induction](electromagnetism/faraday_induction/README.md) | 3D+t | academic | theory | peak EMF | 0.00524896 | 0.00526995 | 3.98e-03 | 0.01 | closed_form | APPROVED |
| [em.faraday_induction](electromagnetism/faraday_induction/README.md) | 3D+t | academic | theory | flux at the centre crossing | 7.72419e-05 | 7.7425e-05 | 2.36e-03 | 0.01 | closed_form | APPROVED |
| [em.faraday_induction](electromagnetism/faraday_induction/README.md) | 3D+t | academic | theory | time of the EMF sign change | 0.127727 | 0.127732 | 3.82e-05 | 0.005 | closed_form | APPROVED |
| [em.fresnel_snell](electromagnetism/fresnel_snell/README.md) | 2D | academic | theory | |r_s| at 0 deg | 0.201868 | 0.2 | 1.87e-03 (abs.) | 0.005 (abs.) | closed_form | APPROVED |
| [em.fresnel_snell](electromagnetism/fresnel_snell/README.md) | 2D | academic | theory | |r_p| at 0 deg | 0.201868 | 0.2 | 1.87e-03 (abs.) | 0.005 (abs.) | closed_form | APPROVED |
| [em.fresnel_snell](electromagnetism/fresnel_snell/README.md) | 2D | academic | theory | |r_s| at 30 deg | 0.24226 | 0.240408 | 1.85e-03 (abs.) | 0.005 (abs.) | closed_form | APPROVED |
| [em.fresnel_snell](electromagnetism/fresnel_snell/README.md) | 2D | academic | theory | |r_p| at 30 deg | 0.160767 | 0.1589 | 1.87e-03 (abs.) | 0.005 (abs.) | closed_form | APPROVED |
| [em.fresnel_snell](electromagnetism/fresnel_snell/README.md) | 2D | academic | theory | |r_s| at 45 deg | 0.30508 | 0.303337 | 1.74e-03 (abs.) | 0.005 (abs.) | closed_form | APPROVED |
| [em.fresnel_snell](electromagnetism/fresnel_snell/README.md) | 2D | academic | theory | |r_p| at 45 deg | 0.0939397 | 0.0920134 | 1.93e-03 (abs.) | 0.005 (abs.) | closed_form | APPROVED |
| [em.fresnel_snell](electromagnetism/fresnel_snell/README.md) | 2D | academic | theory | |r_s| at 60 deg | 0.421591 | 0.420204 | 1.39e-03 (abs.) | 0.005 (abs.) | closed_form | APPROVED |
| [em.fresnel_snell](electromagnetism/fresnel_snell/README.md) | 2D | academic | theory | |r_p| at 60 deg | 0.0401845 | 0.0424492 | 2.26e-03 (abs.) | 0.005 (abs.) | closed_form | APPROVED |
| [em.fresnel_snell](electromagnetism/fresnel_snell/README.md) | 2D | academic | theory | |r_p| at Brewster's angle 56.31 deg | 0.00212959 | 0 | 2.13e-03 (abs.) | 0.005 (abs.) | closed_form | APPROVED |
| [em.fresnel_snell](electromagnetism/fresnel_snell/README.md) | 2D | academic | theory | transmitted angle for 30 deg incidence | 19.4437 | 19.4712 | 1.41e-03 | 0.005 | closed_form | APPROVED |
| [em.fresnel_snell](electromagnetism/fresnel_snell/README.md) | 2D | academic | theory | transmitted angle for 45 deg incidence | 28.0944 | 28.1255 | 1.11e-03 | 0.005 | closed_form | APPROVED |
| [em.fresnel_snell](electromagnetism/fresnel_snell/README.md) | 2D | academic | theory | transmitted angle for 60 deg incidence | 35.2326 | 35.2644 | 9.00e-04 | 0.005 | closed_form | APPROVED |
| [em.gauss_law](electromagnetism/gauss_law/README.md) | 3D | academic | theory | flux through the cube x eps0 | 1.5e-09 | 1.5e-09 | 0.00e+00 | 1e-06 | closed_form | APPROVED |
| [em.gauss_law](electromagnetism/gauss_law/README.md) | 2D | academic | theory | flux per unit length x eps0 | 2e-09 | 2e-09 | 2.07e-16 | 1e-06 | closed_form | APPROVED |
| [em.hall_effect](electromagnetism/hall_effect/README.md) | 2D | academic | theory | Hall coefficient R_H = E_y / (J_x B) | -0.000623757 | -0.000624151 | 6.31e-04 | 0.002 | closed_form | APPROVED |
| [em.lossy_plane_wave](electromagnetism/lossy_plane_wave/README.md) | 1D | academic | theory | attenuation constant alpha | 10.2083 | 10.183 | 2.48e-03 | 0.005 | closed_form | APPROVED |
| [em.lossy_plane_wave](electromagnetism/lossy_plane_wave/README.md) | 1D | academic | theory | phase constant beta | 43.1632 | 43.1361 | 6.29e-04 | 0.005 | closed_form | APPROVED |
| [em.mie_scattering](electromagnetism/mie_scattering/README.md) | 3D | academic | theory | Q_sca, BHMIE sample | 3.10543 | 3.10543 | 1.44e-06 | 2e-05 | published | APPROVED |
| [em.mie_scattering](electromagnetism/mie_scattering/README.md) | 3D | academic | theory | Q_back, BHMIE sample | 2.92534 | 2.92534 | 2.22e-07 | 2e-05 | published | APPROVED |
| [em.mie_scattering](electromagnetism/mie_scattering/README.md) | 3D | academic | theory | Q_sca at x = 0.02 (Rayleigh) | 1.77577e-08 | 1.77582e-08 | 2.94e-05 | 0.001 | limit | APPROVED |
| [em.mie_scattering](electromagnetism/mie_scattering/README.md) | 3D | academic | theory | optical theorem: 4 Re S(0) / x^2 (absorbing sphere) | 1.9868 | 1.9868 | 2.24e-16 | 1e-10 | conservation | APPROVED |
| [em.mie_scattering](electromagnetism/mie_scattering/README.md) | 3D | academic | theory | Q_ext at x = 3000 (absorbing) | 2.00955 | 2 | 4.78e-03 | 0.01 | limit | APPROVED |
| [em.parallel_plate_capacitor](electromagnetism/parallel_plate_capacitor/README.md) | 2D | academic | theory | C per unit length (extrapolated) | 1.03434e-10 | 1.0303e-10 | 3.93e-03 | 0.01 | closed_form | APPROVED |
| [em.parallel_plate_capacitor](electromagnetism/parallel_plate_capacitor/README.md) | 3D | academic | theory | C, 3-D Cartesian (finest) vs axisymmetric | 3.26247e-10 | 3.2703e-10 | 2.39e-03 | 0.01 | convergence | APPROVED |
| [em.parallel_plate_capacitor](electromagnetism/parallel_plate_capacitor/README.md) | 3D | academic | theory | C, axisymmetric (extrapolated) | 3.2703e-10 | 3.24381e-10 | 8.17e-03 | 0.015 | closed_form | APPROVED |
| [em.research_fdtd_3d](electromagnetism/research_fdtd_3d/README.md) | 3D+t | research | theory | radiated power, dipole in vacuum (finest grid) | 1.59015e-26 | 1.58494e-26 | 3.29e-03 | 0.02 | closed_form | APPROVED |
| [em.research_fdtd_3d](electromagnetism/research_fdtd_3d/README.md) | 3D+t | research | theory | P / P_closed_form, Richardson-extrapolated | 1.00106 | 1 | 1.06e-03 | 0.01 | closed_form | APPROVED |
| [em.research_fdtd_3d](electromagnetism/research_fdtd_3d/README.md) | 3D+t | research | theory | far-field pattern max |U/Umax - sin^2| | 0.00856227 | 0 | 8.56e-03 (abs.) | 0.02 (abs.) | closed_form | APPROVED |
| [em.research_inverse_permittivity](electromagnetism/research_inverse_permittivity/README.md) | 1D | research | theory | eps_r recovered vs truth | 3.69982 | 3.7 | 4.78e-05 | 0.00091 | ground_truth | APPROVED |
| [em.research_inverse_permittivity](electromagnetism/research_inverse_permittivity/README.md) | 1D | research | theory | sigma recovered vs truth | 0.0189475 | 0.02 | 5.26e-02 | 0.065 | ground_truth | APPROVED |
| [em.research_surrogate_hall](electromagnetism/research_surrogate_hall/README.md) | 2D | research | theory | hold-out: worst surrogate G vs solver G | 0.510358 | 0.509955 | 7.90e-04 | 0.01 | convergence | APPROVED |
| [em.research_surrogate_hall](electromagnetism/research_surrogate_hall/README.md) | 2D | research | theory | solver: long bar G(L/W = 8) -> 1 | 0.999994 | 1 | 5.74e-06 | 0.001 | limit | APPROVED |
| [em.research_uq_waveguide](electromagnetism/research_uq_waveguide/README.md) | 2D | research | theory | TE10: Monte Carlo std vs first-order GUM | 2.2863e+07 | 2.23e+07 | 2.52e-02 | 0.067 | closed_form | APPROVED |
| [em.research_uq_waveguide](electromagnetism/research_uq_waveguide/README.md) | 2D | research | theory | TE10: Monte Carlo mean vs nominal | 4.41927e+09 | 4.42082e+09 | 3.52e-04 | 0.002 | closed_form | APPROVED |
| [em.research_uq_waveguide](electromagnetism/research_uq_waveguide/README.md) | 2D | research | theory | TE20: Monte Carlo std vs first-order GUM | 4.57067e+07 | 4.46001e+07 | 2.48e-02 | 0.067 | closed_form | APPROVED |
| [em.research_uq_waveguide](electromagnetism/research_uq_waveguide/README.md) | 2D | research | theory | TE20: Monte Carlo mean vs nominal | 8.8348e+09 | 8.84165e+09 | 7.75e-04 | 0.002 | closed_form | APPROVED |
| [em.waveguide_modes](electromagnetism/waveguide_modes/README.md) | 2D | academic | theory | TE10 cut-off frequency | 6.55701e+09 | 6.55714e+09 | 1.98e-05 | 0.001 | closed_form | APPROVED |
| [em.waveguide_modes](electromagnetism/waveguide_modes/README.md) | 2D | academic | theory | TE20 cut-off frequency | 1.31132e+10 | 1.31143e+10 | 7.93e-05 | 0.001 | closed_form | APPROVED |
| [em.waveguide_modes](electromagnetism/waveguide_modes/README.md) | 2D | academic | theory | TE01 cut-off frequency | 1.47521e+10 | 1.47536e+10 | 1.00e-04 | 0.001 | closed_form | APPROVED |
| [em.waveguide_modes](electromagnetism/waveguide_modes/README.md) | 2D | academic | theory | TE11 cut-off frequency | 1.61437e+10 | 1.61451e+10 | 8.71e-05 | 0.001 | closed_form | APPROVED |
| [em.waveguide_modes](electromagnetism/waveguide_modes/README.md) | 2D | academic | theory | TE30 cut-off frequency | 1.96679e+10 | 1.96714e+10 | 1.78e-04 | 0.001 | closed_form | APPROVED |
| [em.waveguide_modes](electromagnetism/waveguide_modes/README.md) | 2D | academic | theory | TM mode 1 cut-off frequency | 1.61437e+10 | 1.61451e+10 | 8.71e-05 | 0.001 | closed_form | APPROVED |
| [em.waveguide_modes](electromagnetism/waveguide_modes/README.md) | 2D | academic | theory | TM mode 2 cut-off frequency | 1.97378e+10 | 1.97396e+10 | 9.11e-05 | 0.001 | closed_form | APPROVED |
| [em.waveguide_modes](electromagnetism/waveguide_modes/README.md) | 2D | academic | theory | TM mode 3 cut-off frequency | 2.45856e+10 | 2.45893e+10 | 1.50e-04 | 0.001 | closed_form | APPROVED |
