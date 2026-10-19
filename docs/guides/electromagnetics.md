# Electromagnetics and electrodynamics

`pinneapple_simulation.electromagnetics` (`pp.em`) solves electromagnetic problems in SI units and checks the
results against independent references; `pinneapple_physics.maxwell_checks` (`pp.maxwell`) checks any
electromagnetic field, from a solver or a neural network, against Maxwell's equations without a reference
solution. This page gives the theory the code implements, what is verified and to which order, and the limits.

## Conventions

* SI units throughout. Constants are CODATA 2022 (`pp.em.constants`): `mu0 = 1.25663706127e-6` N/A² (measured
  since the 2019 SI redefinition), `eps0 = 1 / (mu0 c²)`, `eta0 = mu0 c`.
* Phasors are `Re[X exp(+j omega t)]` (engineering convention, as in Taflove & Hagness, Pozar, scikit-rf). A lossy
  medium has `Im(eps_r) < 0`; a wave travelling to `+z` is `exp(-j k z)`. Physics texts (Jackson, Bohren & Huffman)
  use `exp(-i omega t)`; their `eps` is the complex conjugate of the one here. `pp.em.mie` takes the index in this
  package's convention (`m = n - j k`, the same as miepython).
* `pp.em.units.check_equations()` verifies that every term of every law below has the same SI dimension
  (`pinneapple_data.physical_units`, now with T, Wb, coulomb, farad, henry and siemens), and
  `check_units({"B": "mT", ...})` flags fields declared in the wrong unit or scale before residuals are computed.

## Maxwell's equations

Differential form (Jackson, *Classical Electrodynamics*, 3rd ed., sec. 6.1; Griffiths, *Introduction to
Electrodynamics*, 4th ed., ch. 7):

| Law | Differential | Integral |
|---|---|---|
| Gauss | `div D = rho` | `oint_S D . dA = Q_enc` |
| Gauss (magnetic) | `div B = 0` | `oint_S B . dA = 0` |
| Faraday | `curl E = -dB/dt` | `oint_C E . dl = -d/dt int_S B . dA` |
| Ampere-Maxwell | `curl H = J + dD/dt` | `oint_C H . dl = I_enc + d/dt int_S D . dA` |

with `D = eps E`, `B = mu H`, `J = sigma E + J_s`. Charge continuity `d rho/dt + div J = 0` follows from the
divergence of Ampere-Maxwell plus Gauss; Poynting's theorem `du/dt + div S + J . E = 0`, `S = E x H`,
`u = (E . D + B . H) / 2`, from the curl equations (Jackson sec. 6.7).

Interface conditions (Jackson sec. I.5): `n x (E1 - E2) = 0`, `n x (H1 - H2) = K`, `n . (D1 - D2) = sigma_s`,
`n . (B1 - B2) = 0`. A perfect electric conductor (PEC) has `n x E = 0`.

### Formulations

| Regime | Unknown | Equation | Where |
|---|---|---|---|
| Electrostatics | `phi`, `E = -grad phi` | `-div(eps grad phi) = rho` | `solve_electrostatic` (P1 FEM), kinds `electrostatics`, `electrostatics_integral` |
| Magnetostatics (2-D) | `A_z`, `B = curl(A_z e_z)` | `-div(nu grad A) = J + curl(nu Br)` | `solve_magnetostatic` (P1 FEM, Newton for nonlinear B-H), kind `magnetostatics_az` |
| Magneto-quasistatics | phasor `A_z` | `-div(nu grad A) + j omega sigma A = J_s + sigma u_c` | `solve_time_harmonic` (solid conductors with imposed currents), `eddy_current_fdm` (axisymmetric), kind `eddy_current_az` |
| Full wave, time domain | `E, H` | curl equations | kinds `maxwell_tm`, `maxwell_te`, `maxwell_3d`, `maxwell_tm_integral` |
| Full wave, frequency domain | phasor `E_z` | `lap E + k0^2 eps_r E = f` | kind `maxwell_tm_frequency` |

The A-formulation in a solid conductor carries the voltage per metre `u_c` as an extra unknown fixed by its net
current (Jin, *The Finite Element Method in Electromagnetics*, 3rd ed., ch. 4 and 8), which is the 2-D A-V
formulation.

### Integral form as a PINN loss

The compiler kinds `electrostatics_integral` and `maxwell_tm_integral` impose the integral laws on a square
control volume around each collocation point (Gauss-Legendre quadrature on its edges). They need only first
derivatives and they see material interfaces: with `eps` jumping from 1 to 4 across a capacitor, `phi = y`
satisfies `eps lap(phi) = 0` on both sides, so the differential residual is exactly zero for this wrong field,
while the flux of `D` through the squares that straddle the interface is not
(`tests/test_maxwell_checks.py::test_integral_gauss_sees_the_dielectric_interface_the_differential_form_misses`).

### Presets with closed-form solutions

`pp.PhysicalProblem.from_preset(name)`; `pp.solve(problem, "analytic")` returns the exact field, so
`pp.compare(problem, [...], reference="analytic")` scores any method:

| Preset | Physics | Exact solution |
|---|---|---|
| `em_two_dielectric_capacitor` (`_differential`) | Gauss across a dielectric interface | `D` uniform, `E_i = D / eps_i` (Griffiths ex. 4.7) |
| `em_wire_az` | magnetostatics of a round conductor | Ampere's law |
| `em_tm_cavity` (`_integral`) | TMz standing wave in a PEC square | `Ez = sin sin cos`, Pozar sec. 6.3 |
| `em_skin_effect` | eddy currents into a conductor | `A0 exp(-(1 + j) x / delta)` |
| `em_cylinder_scattering` | plane wave on a dielectric cylinder, TMz | series of cylinder functions (Harrington sec. 5-9) |

## Materials

* `Medium(eps_inf, mu_r, sigma, poles=...)` with `Drude(omega_p, gamma)`, `Lorentz(delta_eps, omega0, delta)` and
  `Debye(delta_eps, tau)` poles (Taflove & Hagness, *Computational Electrodynamics*, 3rd ed., ch. 9). Each pole
  carries the coefficients of its auxiliary differential equation for time-domain solvers; the trapezoidal
  recursions reproduce `chi(omega)` at second order in `dt` (table below). `is_passive` checks `Im(eps_r) <= 0`.
* `Anisotropic(eps_r, mu_r, sigma)` tensors, `uniaxial(n_o, n_e, axis)`, rotations, and the two refractive
  indices of a plane wave along any direction from Fresnel's equation of wave normals. `is_reciprocal` is the
  symmetry of the tensors (a gyrotropic medium is not reciprocal).
* Magnetic: linear `LinearMaterial`, `Magnet` (`B = mu0 mu_r H + Br`), measured `BHCurve` (monotone PCHIP),
  `BrauerCurve`, and `FrohlichKennelly` (`B = mu0 H + H / (alpha + beta |H|)`, every quantity of the Newton
  solver in closed form). `JilesAtherton(Ms, a, k, c, alpha)` integrates the hysteresis model of Jiles & Atherton
  (J. Magn. Magn. Mater. 61, 1986) along an applied-field path; its `dM/dH` is derived from the model's own
  definitions (`M = Mrev + Mirr`), and `c = 1` reproduces the anhysteretic curve.

## References in closed form

`pp.em.statics`: capacitors (parallel plate, coax, spheres), point and line charges, infinite wire, exact
Biot-Savart of straight segments (`biot_savart`, with the cancellation-free form beside long segments), circles,
helices, the off-axis loop field in elliptic integrals, finite solenoid, Helmholtz pair, Maxwell's mutual
inductance of coaxial loops, loop self-inductance.
`pp.em.waves`: Fresnel coefficients, Brewster and critical angles, planar multilayers by characteristic matrices
(Born & Wolf), skin depth and surface impedance, rectangular and circular waveguide cut-offs and modes,
rectangular and cylindrical cavity resonances and the TE101 conductor Q (Pozar), Hertzian dipole fields and power,
thin-dipole patterns, half-wave dipole resistance and directivity (Balanis), TMz scattering by a cylinder,
surface-plasmon and plasma dispersion. `pp.em.mie`: Mie series for dielectric and perfectly conducting spheres
(Bohren & Huffman; Wiscombe's series length), efficiencies, asymmetry parameter, amplitude functions.

## Physics checks of any field

`pp.maxwell.maxwell_checks(grid, fields, eps=..., mu=..., sigma=..., omega=None)` judges a field without a
reference, like the Darcy checks: Gauss's laws, Faraday, Ampere-Maxwell and continuity as relative residuals
(`||sum of terms|| / sum of ||term||`, second-order differences), Poynting's energy balance over the grid box,
PEC faces, and optionally a reference. Every residual is also computed on the grid with every other point: the
ratio `r(2h) / r(h)` is about 4 when the residual is discretisation error (the field obeys the law) and about 1
when the field violates it, and the message says which. A law whose terms all vanish for the given components is
reported as not informative instead of passed. `tangential_continuity`, `pec_boundary`, `reciprocity` and
`passivity` cover interfaces and S-parameters. `maxwell_trust_report` gives the deploy decision.

In CI:

```toml
[[result]]
path = "fields.npz"           # x, y, (z), (t) axes; Ex..Hz / Bx.., Dx.., Jx.., rho; phasors as Ez_re / Ez_im
physics = "maxwell"
parameters = { eps_r = 1.0, mu_r = 1.0, sigma = 0.0 }   # omega = <rad/s> for phasors
```

`pinneapple check` reports Gauss's laws under PC102, Faraday / Ampere-Maxwell / continuity under PC301 and
Poynting's theorem under PC103 (tolerances `maxwell-gauss`, `maxwell-faraday`, `maxwell-ampere`,
`maxwell-continuity`, `maxwell-energy`). `save_maxwell_npz` writes the file with SI units and
`check_config_entry` the configuration block. It runs without torch.

## Time domain: FDTD on the Yee grid

`pp.em.fdtd` solves the curl equations by the leapfrog scheme of Yee (1966): E at integer, H at half time
steps, components staggered by half a cell (Taflove & Hagness, *Computational Electrodynamics*, 3rd ed., ch. 3).
One code path covers 1-D, 2-D and 3-D: an axis with one node is a direction of invariance.

```python
import numpy as np
from pinneapple_simulation.electromagnetics import fdtd, media

grid = fdtd.YeeGrid((200, 200, 1), (1.5e-2, 1.5e-2, 1.5e-2))     # dt = 0.99 x the Courant limit
sim = fdtd.FDTD(grid, {"x": "cpml", "y": "cpml"}, npml=10)
sim.set_material(fdtd.fill_fraction(grid, lambda x, y, z: np.hypot(x - 1.5, y - 1.5) < 0.3), media.Medium(3.0))
sim.add_plane_wave(fdtd.PlaneWaveTFSF("x", "Ez", fdtd.modulated_gaussian(1e9, 0.6e9), ((30, 170), (30, 170), (0, 1))))
dft = sim.add_dft(["Ez", "Hx", "Hy"], [1e9])
sim.run_until(40e-9)
```

* **Stability.** `dt <= 1 / (c sqrt(sum 1/dx_i^2))` (TH eq. 4.65). `YeeGrid` refuses a larger step; the test
  shows grid-scale noise growing by 1e6 in 400 steps at 1.02 times the limit and staying bounded at 0.999.
* **Numerical dispersion.** The scheme's waves obey `(sin(omega dt/2) / (c dt))^2 = sum_i (sin(k_i dx_i/2) / dx_i)^2`
  (TH eq. 4.12, `numerical_dispersion`). The FDTD resonances match this relation to 1e-7 and the continuum at
  second order.
* **Energy.** The scheme conserves `(1/2) sum eps E^{n-1} . E^n + (1/2) sum mu |H^{n-1/2}|^2` exactly in a
  lossless cavity (`energy()`; constant to 1e-12 in the test).
* **Boundaries.** `pec`, `pmc`, `periodic`, `bloch` (Floquet phase `exp(-j k_b L)`, complex fields) and `cpml`,
  the convolutional PML of Roden & Gedney (2000) with polynomial `sigma`, `kappa` and linear `alpha` grading
  (TH sec. 7.9). Measured reflection 5 cells from the PML: -51 dB (5 cells), -87 dB (10), -106 dB (20).
* **Dispersive media** by auxiliary differential equations solved with the E update: Drude (polarisation current,
  trapezoidal), Debye (polarisation, trapezoidal), Lorentz (polarisation, central), TH sec. 9.4-9.6.
* **Sources and plane waves.** Soft and hard E sources, impressed currents (a Hertzian dipole is one cell of `J`).
  `PlaneWaveTFSF` is a total-field / scattered-field box driven by an auxiliary 1-D line with the same `dx` and
  `dt`. The masked-curl form works for any box in 1-D, 2-D and 3-D, and the leakage into the scattered-field
  region is 1e-16.
* **Sub-cell averaging.** `fill_fraction` gives the fraction of each cell inside a shape, and `set_material` mixes
  `eps` arithmetically with it at the nodes. Each E component then takes the harmonic mean of its two nodes (the
  series rule for E normal to an interface; `harmonic_eps=False` switches it off). For E tangential to the surface
  this is the exact first-order correction (Farjadpour et al., Opt. Lett. 31, 2006):
  - the TMz cylinder converges at order 2, where the staircased one jumped between 1.2 % and 3.4 %;
  - on a sphere (E normal to part of the surface) harmonic averaging cuts the Q_sca error from 9.7 % to 2 %, but
    the error then plateaus near 2 % instead of converging. Full anisotropic (Kottke) smoothing is not implemented.
* **Monitors and output.**
  - Probes and running DFTs (phasors in this package's `exp(+j omega t)` convention, E and H stamped at their own
    times).
  - `poynting_flux_dft` gives the power through a plane.
  - `NearToFarField` gives far-field patterns and radiated power from equivalent currents on a closed box (Balanis,
    *Advanced Engineering Electromagnetics*, eq. 6-122).
  - `snapshot()` returns node- and time-centred fields in the layout of `maxwell_checks`, so an FDTD run can be
    checked like any other field.

Figures and animations: `pinneapple_tools.visualization.electromagnetics` (`field_movie` writes GIF with Pillow
or MP4 with ffmpeg, `field_map` adds a material outline and Poynting arrows, `radiation_pattern` draws polar
patterns). `examples/electrodynamics/08_fdtd_gallery.py` writes `results/fdtd_cylinder_scattering.gif` / `.mp4`,
`results/fdtd_cylinder_field_map.png` and `results/fdtd_dipole_pattern.png`. Each figure comes with its check:
the cylinder's field against the series of cylinder functions (1.7 % at 30 cells per wavelength), the snapshots
through `maxwell_trust_report` (APPROVED), and the dipole pattern and power against the closed forms.

## Electric machines

The A_z solver includes a polar ring mesher with a moving band, magnets, coils, (anti-)periodic constraints,
Maxwell-stress (Arkkio) and co-energy torque, flux linkage, back-EMF, cogging, Steinmetz/Bertotti iron loss,
resistance-limited magnet eddy loss, and an `SPMMachine` model with adjoint torque sensitivities used by the rotor
topology optimisation (`pinneapple_design.topology`, [guide](topology_optimization.md)). `tests/test_motor_magnetics.py`
asserts: slotless SPM torque within 1e-3 of the exact subdomain solution by both Maxwell stress and co-energy (360
angular steps, rotor off the grid); Maxwell-stress and co-energy torque of the slotted 24-slot machine within
0.3 %; skin-effect impedance of a round wire within 0.3 % of the Bessel solution (the table below measures
8.55e-4); adjoint rotor-torque sensitivities within 1e-5 of finite differences, for linear and nonlinear iron.
`examples/topology_optimization/03_reluctance_rotor.py` (run 2026-10-17) optimises a reluctance rotor to a mean
torque of 38.10 N m with linear iron; re-evaluated with the nonlinear B-H curve the same rotor gives 13.44 N m, so
saturation must be in the optimisation loop for thin-featured designs.

### Interior-PM traction motor: the 2004 Toyota Prius

`pinneapple_simulation.electromagnetics.machines` models interior-magnet machines on conforming meshes:

* `geometry`: parametric V-magnet rotors (`VRotor`: V opening and depth, magnet width and thickness, bridges, ribs,
  centre post, magnet segments) and round-bottom slots (`RoundBottomSlots`), the parametrisations of pyleecan's
  `HoleM50` and `SlotW11`; `prius_2004()` holds the dimensions of pyleecan's `Toyota_Prius.json` (48 slots, 8 poles,
  0.75 mm gap, 83.82 mm stack, stacking factor 0.95).
* `mesher`: every polygon edge is a mesh edge (conforming Delaunay with Gabriel-edge recovery, concentric-shell
  splitting at small angles). Half a pole and half a slot pitch are meshed, mirrored and copied around, so all
  poles and slots have identical, symmetric elements: the no-load torque of the aligned rotor is 6e-13 N m
  and the cogging curve is antisymmetric with zero mean. The air gap holds three ring layers; only the middle
  one (moving band) is re-triangulated as the rotor turns.
* `steel`: M400-50A (EN 10106) B-H curve and specific-loss table (50 Hz to 2.5 kHz), the data of pyleecan's material
  library (Apache-2.0), and the Prius magnet grade (Br 1.24 T at 20 degC, -0.1 %/K); `stacked` applies the
  stacking factor; `TableIronLoss` interpolates the catalogue table; `fit_iron_loss` fits
  `k_h f^alpha B^beta + k_c (f B)^2 + k_e (f B)^1.5` with the classical term from resistivity and thickness.
* `pmsm.IPMMachine`: Newton solution with the nonlinear B-H curve, Arkkio and co-energy torque, flux linkage,
  back-EMF, the dq frame (d-axis on the magnet flux), torque ripple, frozen-permeability flux split and L_d / L_q,
  MTPA by golden-section search on the FEA torque, iron loss per element in its own frame (table or fit) and
  resistance-limited magnet eddy loss.

The FEA is checked against a published benchmark here and against a second, independent code (gmsh + GetDP) in the
next section. FEMM (Windows-only) and pyleecan (its wheel needs a git-only dependency) do not run in this
environment. The published reference is the
torque against current angle of the Prius motor at 250 A peak computed by Yang, Krishnamurthy and Brown (IEEE IEMDC
2013), as tabulated in pyleecan's test suite. Their model and pyleecan's geometry are not the same file (centre
post, magnet temperature, steel data), so agreement within a few per cent is the expectation, not a mesh error.

`python examples/electrodynamics/10_pmsm_prius.py --write-docs` (run on 2026-10-17, 960 gap nodes, 29 936
elements) wrote the table below and `examples/electrodynamics/results/10_pmsm_prius.json`; figures:
`10_pmsm_field.png` (|B| and flux lines at MTPA, 250 A), `10_pmsm_curves.png` (torque against current angle with the
published curve, cogging, back-EMF), `10_pmsm_mtpa.png` (MTPA locus, L_d and L_q).

Results of that run (magnets at 20 degC, 1.24 T):

| Peak current (A) | MTPA angle (deg from q) | Mean torque (N m) | Ripple pk-pk | L_d (mH) | L_q (mH) |
|---|---|---|---|---|---|
| 50 | 27.6 | 70.4 | 24.9 % | 1.945 | 4.871 |
| 100 | 37.1 | 147.8 | 25.5 % | 1.741 | 3.697 |
| 150 | 43.1 | 221.3 | 22.5 % | 1.623 | 3.101 |
| 200 | 47.1 | 287.9 | 22.8 % | 1.511 | 2.729 |
| 250 | 50.3 | 345.6 | 22.9 % | 1.386 | 2.435 |

L_d and L_q are the apparent frozen-permeability values at each MTPA point (saturation lowers L_q from 4.87 to
2.44 mH). The d-axis is the fundamental of the magnet flux linkage, from the winding layout (`pm_flux_angle`): a single
no-load solution's flux vector carries the slot harmonics and sits 0.79 electrical degrees off, which shifted the
torque at pure negative d-axis current by -12 N m; with the fundamental the mean torque at Phi0 = 180 deg is 0.0
N m, as the rotor's mirror symmetry requires.

Torque against current angle at 250 A, against Yang et al. (2013):

| Phi0 (deg) | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 | 160 | 170 | 180 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| this FEA (N m) | 92.0 | 133.6 | 175.8 | 217.3 | 256.0 | 289.9 | 318.2 | 338.1 | 345.6 | 330.1 | 267.1 | 151.4 | 0.0 |
| published (N m) | 79 | 125 | 160 | 192 | 237 | 281 | 319 | 343 | 353 | 332 | 266 | 164 | 22 |
| difference | +13.0 | +8.6 | +15.8 | +25.3 | +19.0 | +8.9 | -0.8 | -4.9 | -7.4 | -1.9 | +1.1 | -12.6 | -22.0 |

The peak agrees within 2.1 %; the largest deviation is +25.3 N m (+13 %) at 90 deg (pure q-axis current), the rms
deviation 13.3 N m. Shifting the curve in angle does not remove it: the best-fitting offset (-1.32 deg) lowers the
rms over 70 to 170 deg only from 12.2 to 9.7 N m. The differences are in magnitude where the magnet torque dominates
(low Phi0, more torque here) and at 180 deg, where a machine symmetric about its d-axis has no mean torque but the
published curve gives 22 N m (an angle convention or an asymmetry of their model). Plausible causes, not
resolved here: magnet remanence or temperature of the published model, its steel curve, the centre post (W1 = 0 in
pyleecan's file).

No load: cogging 1.38 N m peak to peak (antisymmetric, zero mean), magnet flux linkage 0.209 Wb peak per phase.
Back-EMF from the flux linkage at 120 rotor angles per electrical period (spectral derivative): 87.6 V phase peak
fundamental at 1000 rpm (107.3 V line-to-line rms fundamental, 0.1073 V/rpm), THD 9.3 % phase and 8.5 % line;
largest harmonics 11th 6.5 %, 5th 4.2 %, 7th 3.6 %, 3rd 2.4 % of the fundamental (the 3rd cancels between phases).
No published Prius back-EMF constant could be retrieved here to compare with. Losses at 1500 rpm (100 Hz), 250 A
MTPA: stator iron 391 W, rotor iron 29 W with the catalogue table (332 W and 32 W with the Bertotti fit), magnet eddy
loss 37.9 W (resistance limited, one piece per magnet). The Bertotti fit to the M400-50A table up to 1 kHz has
8.2 % rms and 22 % maximum relative error; the table model is the default. Mesh convergence of the torque is not
monotone (480, 960, 1920 gap nodes: 343.42, 345.65, 344.22 N m): treat the FEA torque as converged to about 0.4 %.

<!-- machine-verification:start -->
| Case | Quantity | Reference | Error (finest) | Observed order | Status |
|---|---|---|---|---|---|
| Magnetised ring, sector mesher + P1 FEM | relative L2 error of A_z, h halved | closed form (two magnetised cylinders + harmonic) | 1.91e-04 | 1.89, 1.91, 1.99 (expected 2) | pass |
| Machine sector geometry (Prius slot and magnet areas) | slot area, gap nodes doubled | exact area (trapezoid + half disc) | 5.70e-05 | 1.99, 1.81, 1.90 (expected 2) | pass |
| Prius IPM: Maxwell stress vs co-energy torque | Arkkio torque at 250 A, 3 rotor angles | virtual work dW'/dtheta (4th-order differences) | 1.96e-03 | - | pass |
| Prius IPM: dq power balance and frozen permeability | mean Maxwell-stress torque over 60 el. deg | 1.5 p (psi_d i_q - psi_q i_d) | 1.96e-04 | - | pass |
| Prius IPM: torque vs current angle (Yang et al. 2013) | mean torque, 250 A, Phi0 = 60..180 deg | published FEA (IEMDC 2013), 13 angles | 2.08e-02 | - | pass |
| Prius IPM: mesh convergence of the peak torque | mean torque at 250 A, 50 deg, gap nodes doubled | successive differences | 4.15e-03 | - | pass |

Notes from the same run:

- Magnetised ring, sector mesher + P1 FEM: 8 sector copies, magnet edges on mesh edges
- Machine sector geometry (Prius slot and magnet areas): magnet area (straight edges) exact to 4.4e-16
- Prius IPM: Maxwell stress vs co-energy torque: max relative difference over the angles
- Prius IPM: dq power balance and frozen permeability: frozen-permeability superposition error 4.1e-11; L_d = 1.386 mH, L_q = 2.420 mH, psi_pm = 0.2140 Wb at this point
- Prius IPM: torque vs current angle (Yang et al. 2013): peak 345.6 N m at Phi0 = 140 deg (published 353 N m at 140 deg); largest deviation +25.3 N m at 90 deg; rms 13.3 N m; an angle offset of -1.32 deg fits best (rms 12.2 -> 9.7 N m over 70..170 deg)
- Prius IPM: mesh convergence of the peak torque: 480: 343.42 N m, 960: 345.65 N m, 1920: 344.22 N m; difference ratio 1.56
<!-- machine-verification:end -->

### Second code: gmsh + GetDP, and interoperability

`pinneapple_simulation.electromagnetics.machines.getdp` solves the same machine with gmsh 4.12 and GetDP 3.2,
external GPL programs called as executables (`apt install gmsh getdp`):

* `write_geo` writes the same polygons as the mesher;
* `write_pro` writes a 2-D A-formulation with the M400-50A curve as an Akima table of `nu(|B|^2)` and
  Newton-Raphson (`JacNL`, tolerance 1e-9), a moving air gap, and the Arkkio torque and phase flux linkages
  computed in GetDP;
* `solve` runs both programs.

Mesh, nonlinear solver and torque integral are GetDP's own; only the geometry, the materials and the currents are
shared.

`python examples/electrodynamics/14_prius_getdp_crosscheck.py` (run on 2026-10-18) uses 960 gap nodes in both
codes: 29 936 elements here, 58 309 in GetDP. Two further runs add their results to
`results/14_prius_getdp_crosscheck.json`: `--cogging-convergence` and `--cogging-diagnosis`. Figures:
`14_prius_getdp_crosscheck.png` and `14_prius_cogging_convergence.png`. Mean torque at 250 A against current angle,
over 12 rotor angles:

| Phi0 (deg) | PINNeAPPle (N m) | GetDP (N m) | Difference | Yang 2013 (N m) |
|---|---|---|---|---|
| 60 | 91.97 | 90.94 | -1.03 | 79 |
| 70 | 133.64 | 132.56 | -1.09 | 125 |
| 80 | 175.84 | 174.76 | -1.09 | 160 |
| 90 | 217.26 | 216.19 | -1.07 | 192 |
| 100 | 255.97 | 254.94 | -1.03 | 237 |
| 110 | 289.94 | 289.03 | -0.90 | 281 |
| 120 | 318.15 | 317.34 | -0.81 | 319 |
| 130 | 338.14 | 337.38 | -0.76 | 343 |
| 140 | 345.65 | 344.98 | -0.66 | 353 |
| 150 | 330.14 | 329.58 | -0.55 | 332 |
| 160 | 267.13 | 267.29 | +0.16 | 266 |
| 170 | 151.40 | 151.25 | -0.15 | 164 |
| 180 | 0.00 | -0.24 | -0.24 | 22 |

Per quantity:

| Quantity | PINNeAPPle | GetDP | Difference | Published |
|---|---|---|---|---|
| Torque vs angle, max / rms difference | | | 1.09 / 0.81 N m (0.19 % at the peak) | |
| Torque vs Yang 2013, rms / max | 13.3 / 25.3 N m | 12.9 / 24.2 N m | | Yang 2013 |
| Back-EMF fundamental, 1000 rpm (60 positions) | 87.60 V | 87.64 V | +0.04 % | |
| Phase back-EMF THD | 9.35 % | 9.04 % | -0.31 points | |
| No-load d-axis flux linkage `psi_d0` | 0.20914 Wb | 0.20921 Wb | +0.03 % | |
| Apparent `L_d` at 250 A, 50 deg | 1.370 mH | 1.385 mH | +1.1 % | |
| Apparent `L_q` at 250 A, 50 deg | 1.856 mH | 1.864 mH | +0.4 % | |
| Cogging pk-pk, 1920 gap nodes | 1.341 N m | 1.004 N m | -25 % | |

The apparent inductances are `(psi_d - psi_d0) / i_d` and `psi_q / i_q` at the same point in both codes. They
differ from the differential values in the MTPA table above.

The two codes agree on everything except cogging. Cogging is the small difference of large forces, 0.4 % of the
peak torque here. The cogging study:

| Gap nodes | PINNeAPPle (elements) | GetDP (elements) |
|---|---|---|
| 480 | 1.698 N m (13 120) | 1.538 N m (28 777) |
| 960 | 1.353 N m (29 936) | 1.046 N m (58 309) |
| 1920 | 1.341 N m (77 904) | 1.004 N m (153 737) |

* Both codes converge under mesh refinement (0.9 % and 4 % change in the last step), but to values 25 % apart.
* With linear iron (`mu_r` 2500, no saturation) at 960 nodes they agree: 1.2225 against 1.2111 N m, 0.9 %.
* A 10 times denser GetDP B-H table changes its cogging by 1e-5 N m.

So the difference lies in the saturated iron: the 1.5 mm bridges carry about 2.4 T at no load. A symmetric machine
has odd cogging. PINNeAPPle's mirrored mesh gives exactly odd cogging. GetDP's mesh is not symmetric, and its
cogging carries an even part of up to 0.089 N m with nonlinear iron and 0.020 N m with linear iron, a measure of
its discretisation error. Even comparing odd parts only, the codes differ by 23 %. Which value is closer to the real
machine is not settled here: that needs a measurement or a third code. Cogging from either code carries this
uncertainty.

`pinneapple_simulation.electromagnetics.machines.interop`:

* pyleecan: `from_pyleecan` reads pyleecan's `Toyota_Prius.json`, included under `tests/data/pyleecan`
  (Apache-2.0). The test checks that it gives exactly `prius_2004()`. `to_pyleecan` writes the parameters back, and
  the round trip is exact.
* FEMM: `femm_lua` writes a Lua script for FEMM's magnetics module: geometry, B-H curve, magnets, circuits and the
  analysis. It is checked by re-reading its nodes against the mesher's polygons, not by running FEMM.
* ElmerFEM: `elmer_case` writes the gmsh geometry and a `MagnetoDynamics2D` `.sif`. It is not run here.
* MATLAB: `to_mat` / `from_mat`. DXF: `to_dxf` / `read_dxf_polylines`, with one layer per region. Both round trips
  are tested.
* Data contracts: `rotor_parameter_space` gives a `ParameterSpace` of the rotor variables. `fluxmap_dataset` gives a
  flux map as a `DatasetSpec` table, written to CSV, Parquet or HDF5 by `pinneapple_data.contracts.tables`.

### Drive model: MTPA, flux weakening, efficiency map, WLTC energy

`pinneapple_simulation.electromagnetics.machines.drive` turns a machine into a drive under inverter limits:

* `FluxMap.from_fea(machine, I_max)`: mean torque and dq flux linkages of the FEA on an `(i_d, i_q)` grid (saturation
  and cross-saturation included, odd/even symmetry in `i_q`), and per grid point the speed-free iron-loss sums and
  magnet-loss coefficient (`IPMMachine.loss_components`), so the core loss at any speed is
  `k_h f^alpha S_h + k_c f^2 S_c + k_e f^1.5 S_e + c_m f^2` without new field solutions. `LinearDQ` is the
  constant-inductance model with closed-form MTPA and flux-weakening torque.
* `Drive(machine, R, I_max, V_max)`: steady state `v_d = R i_d - w psi_q`, `v_q = R i_q + w psi_d`, current and
  voltage amplitude limits (`v_max_svpwm(V_dc) = V_dc / sqrt 3`). `max_torque(speed)` searches the outer boundary of
  the feasible current set (current circle, voltage ellipse, so maximum-torque-per-volt operation is included);
  `operating_point(T, speed)` returns the currents of least copper + core loss for a torque (MTPA at low speed,
  flux weakening at high speed); `efficiency_map` evaluates a speed x torque grid.
* `Vehicle` and `cycle_energy`: road load and inertia over a speed trace, a single gear, regeneration up to the
  braking envelope (the rest goes to the friction brakes), battery-side energy of the drive.
  `wltc_class3b()` is the WLTC class 3b trace of UN GTR No. 15, Annex 1 (1800 s, 23.266 km, 131.3 km/h).

`python examples/electrodynamics/11_prius_drive.py --write-docs` (run on 2026-10-17) builds the flux map of the
Prius model (9 x 9 currents up to 250 A, 480 gap nodes, 12 rotor angles; loss data on 5 x 5 points, 24 angles per
electrical period) and evaluates the drive with stated assumptions: 250 A peak, 500 V DC link with SVPWM (288.7 V
phase peak), phase resistance 93.6 mOhm at 100 degC from the winding data (72 series turns, 8.49 mm^2 conductors,
end turn 0.16 m per side estimated), and an illustrative compact car (1450 kg, C_rr 0.009, C_d A 0.58 m^2, wheel
0.30 m, gear 4.113, 97 %). Results (`results/11_prius_drive.json`, `11_prius_drive.png`):

* envelope: 343.6 N m up to 2000 rpm, then flux weakening (181.0 N m at 3750 rpm, 104.4 N m at 6000 rpm); peak
  electromagnetic power 77.7 kW at 2500 rpm under these limits. The Prius's published rating (50 kW, 400 N m,
  quoted in pyleecan's test suite) comes from its own inverter current and thermal limits, which are not modelled;
* peak efficiency 96.2 % at 2750 rpm, 70.7 N m (copper + iron + magnet loss; no inverter or mechanical loss; the map
  is evaluated up to the motoring and braking envelopes at every speed from 100 to 6000 rpm);
* WLTC class 3b: motoring 11.52 MJ, regenerated 2.97 MJ, drive losses 0.98 MJ, no friction braking needed within
  the braking envelope, battery-side 102.1 Wh/km for this car.

The optimiser is checked against the closed forms of the linear machine and against energy bookkeeping:

<!-- drive-verification:start -->
| Case | Quantity | Reference | Error (finest) | Observed order | Status |
|---|---|---|---|---|---|
| Drive: torque envelope vs closed form (linear dq) | max torque, 500..10000 rpm, both signs | MTPA / circle-ellipse intersection (R = 0) | 1.31e-14 | - | pass |
| Drive: least-loss currents vs MTPA closed form | currents of least copper loss, 4 torques | i_d = (psi - sqrt(psi^2 + 8 (L_q - L_d)^2 I^2)) / (4 (L_q - L_d)) | 1.97e-08 | - | pass |
| Drive cycle: WLTC class 3b energy balance | battery energy of a lossless drive | road-load work (kinetic energy returns to zero) | 0.00e+00 | - | pass |
| Prius flux map: interpolation vs direct FEA | torque at 3 off-grid currents (7 x 7 grid) | FEA at the same currents | 2.72e-04 | - | pass |

Notes from the same run:

- Drive: torque envelope vs closed form (linear dq): L_d = 0.6 mH, L_q = 2.4 mH, psi_m = 0.2 Wb, 250 A, 500 V SVPWM
- Drive cycle: WLTC class 3b energy balance: trace: 23266.3 m (GTR 15: 23 266 m), max 131.3 km/h, sum 83758.6 km/h s
- Prius flux map: interpolation vs direct FEA: error relative to 300 N m
<!-- drive-verification:end -->

### Step skew, tolerances and a 3-D view

`pinneapple_simulation.electromagnetics.machines.robustness`:

* Step skew by multi-slice 2-D FEA. A rotor of `K` axial segments, with segment `k` turned by
  `delta_k = s (k - (K-1)/2)`. Each segment is a 2-D slice with the same stator and phase currents. The torque and
  flux linkage are the means over the slices. `skewed_torque` gives load torque and ripple, `skewed_no_load` gives
  cogging and flux linkage. Inter-segment axial flux and end effects are neglected, as in every multi-slice model.
  For a harmonic of order `h` per period `lam`, the slice mean multiplies the straight rotor's harmonic by the
  step-skew factor `|sin(K h pi s / lam) / (K sin(h pi s / lam))|` (`step_skew_factor`).
* `tolerance_model(geometry)`: the FEA mean torque and ripple as a function of remanence scale, magnet thickness,
  magnet width and air gap, for `pinneapple_analysis.uncertainty.tolerance.propagate` (JCGM 101 Monte Carlo).
* `skewed_field_3d` extrudes each segment's 2-D solution into prisms turned by its skew angle.
  `export.write_vtu` and `export.solution_to_vtu` write VTK unstructured grids for ParaView.

`python examples/electrodynamics/15_prius_skew.py` (run on 2026-10-18; 960 gap nodes, 12 rotor angles per period,
250 A at 50.3 deg) gives `results/15_prius_skew.json`, `15_prius_skew.png`, `15_prius_skew_3d.png` and
`15_prius_skew_3d.vtu`. The skew is the total `(K - 1) s`; the slot pitch, which is also the cogging period, is
7.5 deg:

| Total skew (deg) | Cogging pk-pk, K = 2 (N m) | K = 3 (N m) | Torque, K = 2 (N m) | K = 3 (N m) | Ripple, K = 2 | K = 3 |
|---|---|---|---|---|---|---|
| 0 | 1.353 | 1.353 | 345.6 | 345.6 | 22.9 % | 22.9 % |
| 1.875 | 0.639 | 0.877 | 343.9 | 344.5 | 18.7 % | 20.1 % |
| 3.75 | 0.624 | 0.472 | 338.8 | 341.1 | 8.8 % | 13.3 % |
| 5.625 | 0.639 | 0.420 | 330.1 | 335.3 | 11.2 % | 7.9 % |
| 7.5 | 1.353 | 0.846 | 317.0 | 326.6 | 16.2 % | 7.3 % |

At the classic step-skew optimum, a total skew of `(K-1)/K` of the cogging period, the cogging falls:

* two steps at 3.75 deg: 1.35 to 0.62 N m, with the load ripple down from 22.9 to 8.8 % for 2.0 % less torque;
* three steps, near 5 deg (5.625 deg computed): 1.35 to 0.42 N m, ripple 7.9 %, torque -3.0 %.

Two steps over a full slot pitch reproduce the straight rotor's cogging exactly, because the two slices are a whole
cogging period apart.

With the same two-step skew over one slot pitch, the back-EMF (120 positions per electrical period) shows:

* the fundamental falls by 0.96593, against the closed-form factor `cos 15 deg = 0.96593`;
* every harmonic of the skewed rotor equals the straight harmonic times the factor, to within 1.5e-9 of the
  fundamental;
* the phase THD drops from 9.3 to 7.2 %.

The verification case below checks the same identity on the cogging harmonics of 2- and 3-step skews (1.0e-8).

| Case | Quantity | Reference | Error (finest) | Observed order | Status |
|---|---|---|---|---|---|
| Step skew: multi-slice cogging vs skew factor | cogging harmonics, 2 and 3 steps | sin(K h pi s / lam) / (K sin(h pi s / lam)) x straight rotor | 1.01e-08 | - | pass |

The identity is exact because the slice offsets are multiples of the moving-band step, so every slice uses the same
band triangles. It verifies the multi-slice composition, not the 2-D field itself.

<!-- tolerances:start -->
`python examples/electrodynamics/17_prius_tolerances.py` (run on 2026-10-18) propagates manufacturing tolerances
through the FEA by Monte Carlo (JCGM 101, `propagate`): 200 samples, one nonlinear FEA each, with 480 gap nodes and
12 rotor angles at 250 A and 50.3 deg. The tolerances are normal inputs and are stated assumptions, not measured
data. One standard deviation of each:

* remanence 1 % (a +-2 % batch spread at two sigma);
* magnet thickness 0.025 mm;
* magnet width 0.05 mm;
* air gap 0.025 mm, on a 0.75 mm gap (the rotor radius moves).

Results are in `results/17_prius_tolerances.json` and `17_prius_tolerances.png`:

| Output | Nominal | Mean | Std | 1 % / 99 % | Standardised regression coefficients (remanence, thickness, width, air gap) | R^2 |
|---|---|---|---|---|---|---|
| Mean torque (N m) | 343.35 | 343.31 | 1.96 | 338.9 / 347.4 | 0.82, 0.15, 0.24, -0.54 | 0.999 |
| Ripple (pk-pk) | 20.1 % | 20.2 % | 0.47 % | 19.1 / 21.2 % | -0.19, -0.07, -0.02, -0.95 | 0.983 |

The torque spread (0.57 % of nominal) is set by the remanence first and the air gap second. The ripple spread is
set almost entirely by the air gap: a larger gap smooths the slot harmonics. Both outputs are close to linear in
these inputs (R^2 >= 0.98), so a linear surrogate fitted to a few FEA solutions would reproduce this study. All 200
samples met the stated specification: torque >= 98 % of nominal and ripple <= 25 %. The yield is therefore 1.000,
with a Wilson 95 % interval of 0.981 to 1.000. The nominal torque differs from the 960-node value (345.6 N m) by
the mesh; the spread is a difference between samples on the same mesh.
<!-- tolerances:end -->
### Magnet temperature, thermal network and demagnetisation

`pinneapple_simulation.electromagnetics.machines.thermal`:

* `ThermalNetwork`: isothermal nodes with heat capacities, conductances between nodes and to fixed-temperature
  boundaries, time-dependent sources; steady state and transients (backward Euler or trapezoidal rule).
  `machine_network(geometry)` builds the seven-node model of a water-jacketed machine (stator yoke, teeth, winding in
  the slots, end windings, rotor, magnets, shaft path) from the cross-section: cylindrical-shell conduction
  `2 pi k L / ln(r2/r1)`, slot insulation, axial conduction to the end windings, convection across the gap and at the
  end windings. The material, contact and heat-transfer data are the stated defaults of `MachineThermalData`
  (coolant 65 degC). They are inputs, not correlations fitted here, and illustrative: none is measured on a Prius.
* `IPMMachine(..., magnet_temperature=T)`: remanence `Br(T) = Br20 (1 + alpha_Br (T - 20))`, with
  `alpha_Br = -0.1 %/K` (the pyleecan Prius magnet). `magnet_temperature_sweep` evaluates MTPA torque, back-EMF and
  `L_d`, `L_q` with the FEA at each temperature.
* `demagnetisation_margin(machine, solution, Hcj)`: in every magnet element, the field opposing the magnetisation
  `H_rev = -(B . m - Br) / (mu0 mu_r)` against the intrinsic coercivity (the knee of a square-loop sintered NdFeB
  magnet). It returns the per-element ratio `H_rev / Hcj` and the demagnetised area fraction (ratio > 1).
  `intrinsic_coercivity(Hcj20, T, beta) = Hcj20 (1 + beta (T - 20))`. `NDFEB_HCJ_MIN_KAM` holds the minimum `Hcj`
  at 20 degC of each grade letter (N 12, M 14, H 17, SH 20, UH 25, EH 30, AH 35 kOe), the manufacturers' grade
  convention. `beta = -0.55 %/K` is a typical datasheet value for sintered NdFeB and an assumption here: use the
  grade's datasheet.

`python examples/electrodynamics/13_prius_thermal.py` (run on 2026-10-18, 960 gap nodes) gives
`results/13_prius_thermal.json`, `13_prius_thermal.png` and `13_prius_demag.png`.

Performance against the magnet temperature (MTPA at 250 A; EMF is the phase fundamental, peak, at 1000 rpm):

| Magnets (degC) | Br (T) | Torque (N m) | MTPA angle (deg) | EMF (V) | L_d (mH) | L_q (mH) |
|---|---|---|---|---|---|---|
| 20 | 1.240 | 345.6 | 50.3 | 87.6 | 1.386 | 2.435 |
| 60 | 1.190 | 338.7 | 50.6 | 84.0 | 1.386 | 2.443 |
| 100 | 1.141 | 331.6 | 51.0 | 80.2 | 1.388 | 2.458 |
| 150 | 1.079 | 322.6 | 51.4 | 75.4 | 1.389 | 2.474 |

From 20 to 150 degC, Br falls by 13.0 % and the EMF by 13.9 %. The torque at 250 A falls by only 6.7 %, because the
reluctance torque does not depend on the magnets. `L_q` rises by 1.6 % as the weaker magnets saturate the q-axis
iron less.

The thermal network takes the drive model's per-second losses over four WLTC class 3b cycles. Copper loss is split
between the slots and the end windings by length. Iron loss is split by the FEA loss density at 250 A: yoke 26 %,
teeth 63 %, rotor 10 %. The magnet loss goes to the magnets. Mean losses are copper 298 W, iron 247 W and magnet
0.2 W. The peak temperatures are:

* teeth 82.5 degC;
* winding 80.3 degC;
* magnets 78.5 degC;
* end winding 75.1 degC;
* yoke 71.9 degC.

The network then runs at peak load from the end-of-cycle state: 343.2 N m at 1500 rpm, with copper 8.75 kW, iron
382 W and magnet 30 W. The hotter winding node reaches 180 degC (class H) after 237 s. The magnets settle at
127.4 degC.

Demagnetisation at that magnet temperature, at the full d-axis current (`i_d = -250 A`) and at the peak of a
three-phase short circuit estimated as `2 psi_pm / L_d = -165 A` (demagnetised area fraction of the magnets;
`H_rev` is the worst element):

| Grade | Hcj at 127 degC (kA/m) | -250 A, H_rev 719 kA/m | -165 A, H_rev 517 kA/m |
|---|---|---|---|
| N | 391 | 99.4 % | 31.8 % |
| M | 456 | 93.3 % | 3.2 % |
| H | 554 | 18.6 % | 0 |
| SH | 651 | 2.4 % | 0 |
| UH | 814 | 0 | 0 |

The worst elements sit at the outer magnet corners and at the V tip. `13_prius_demag.png` shows the per-element
ratio for grades H and SH in both cases.

The table above uses the default mesh: 784 magnet elements, 2.2 mm mean size. To test how much the fractions depend
on the magnet mesh, `demagnetisation_refinement` refines the magnets over four levels. It lowers the element-size
growth away from the gap, keeping the gap mesh fixed (`13_prius_demag_mesh.png`). Each cell gives the demagnetised
area, the area above `0.9 Hcj` and the worst-element ratio `H_rev / Hcj`:

| Magnet elements (size) | H, -250 A | SH, -250 A | UH, -250 A | M, -165 A | H, -165 A |
|---|---|---|---|---|---|
| 784 (2.24 mm) | 18.6 / 83.4 % / 1.30 | 2.4 / 7.6 % / 1.10 | 0 / 0 % / 0.88 | 3.2 / 10.9 % / 1.13 | 0 / 2.0 % / 0.93 |
| 1920 (1.43 mm) | 19.4 / 85.5 % / 1.43 | 2.6 / 8.3 % / 1.21 | 0 / 0.7 % / 0.97 | 3.3 / 12.3 % / 1.25 | 0.2 / 1.0 % / 1.03 |
| 4224 (0.96 mm) | 18.6 / 85.5 % / 1.46 | 2.0 / 7.5 % / 1.24 | 0 / 0.7 % / 0.99 | 4.1 / 14.0 % / 1.28 | 0.4 / 1.1 % / 1.06 |
| 7216 (0.74 mm) | 18.8 / 85.1 % / 1.52 | 2.3 / 8.6 % / 1.29 | 0.2 / 0.5 % / 1.03 | 3.6 / 14.0 % / 1.33 | 0.3 / 1.0 % / 1.10 |

The area fractions settle. The demagnetised fraction stays within 0.9 percentage points across the four levels,
and within 0.6 between the two finest. The area above `0.9 Hcj` stays within 3.1 points across the levels (M at
-165 A rises from 10.9 to 14.0 %) and within 1.1 between the two finest. The worst-element ratio does not converge. It grows by 4 to 6 % at each level because the field is
singular at the sharp corners where the magnet meets the pocket. A demagnetised fraction of a few tenths of a
percent at the finest mesh is a corner effect of the same kind, as for UH at -250 A and H at the short circuit.

The conclusions therefore rest on the area fractions, not on the worst element. When hot, grade H loses about
19 % of the magnet area at the full d-axis current, and about 85 % of the area lies within 10 % of the knee. Grade
SH loses about 2 %, with 8 % within 10 % of the knee. In UH, only the corners reach the knee. Corner-level
demagnetisation is not resolved by this model: the real corners are rounded and their field is not singular.

The grade rows are a comparison of candidate grades, not a statement about the real car. We found no accessible
public source that states the grade of the 2004 Prius magnets. Neither does the pyleecan model this geometry comes
from: it gives `Br` and `mu_r` only. So the "N demagnetises" row says nothing about the production motor, whose
magnets were chosen by Toyota for its own current and temperature limits.

The network and the demagnetisation field are checked against closed forms (`machines.verification`):

| Case | Quantity | Reference | Error (finest) | Observed order | Status |
|---|---|---|---|---|---|
| Thermal network: first-order transient | node temperature at t = tau, dt halved | T_b + P/G (1 - e^-1) (closed form) | 4.79e-06 | 2.00, 2.00, 2.00 (expected 2) | pass |
| Thermal network: cylindrical conduction | series of thin flat shells, n doubled | 2 pi k L / ln(r2/r1) | 5.08e-06 | 2.00, 2.00, 2.00 (expected 2) | pass |
| Demagnetisation field in a magnetised ring | reverse field in the magnet, h halved | closed form of the magnetised ring | 5.99e-03 | 1.02, 0.89, 0.95 (expected 1) | pass |

The element-wise B of P1 elements is first order, hence the observed order of the demagnetisation field.

### Rotor design: FEA + Gaussian-process surrogate + NSGA-II

`pinneapple_simulation.electromagnetics.machines.design` puts the FEA behind the design-study core of
`pinneapple_design.design_optimizer`:

* `rotor_evaluator` maps five rotor variables (magnet width `W4` and thickness `H3`, V depth `H0` and opening `W0`,
  bridge `H1`) to the FEA mean torque and ripple at a fixed current, plus the exact magnet mass. Invalid
  geometries are rejected without a solve.
* `rotor_design_problem` maximises torque and minimises magnet mass, with ripple at most 30 %.
* `hybrid_vs_fea_only` compares two studies:
  * surrogate-assisted NSGA-II (`surrogate_assisted`): a Latin-hypercube start, then rounds of GP fit, NSGA-II on
    the GP, and FEA on points along the predicted front (adaptive sampling);
  * NSGA-II that calls the FEA for every individual.

  It compares them at equal quality, the hypervolume of the feasible FEA-evaluated designs, and re-evaluates every
  design of the hybrid front with a finer FEA (`verify_candidates`).

`python examples/electrodynamics/12_prius_rotor_design.py` (run on 2026-10-18) uses three seeds. The study FEA has
288 gap nodes and 6 rotor angles at 250 A, 50 deg. The hybrid starts from 24 Latin-hypercube points and runs
4 rounds of 8 FEA verifications. The FEA-only NSGA-II has a budget of 2000 designs. Results are in
`results/12_prius_rotor_design.json` and `12_prius_rotor_design.png`:

| Seed | Hybrid FEA calls | Hybrid hypervolume | FEA calls for NSGA-II to reach it | Saving | NSGA-II final (FEA calls) |
|---|---|---|---|---|---|
| 0 | 53 | 0.8938 | 711 | 92.5 % | 0.9194 (1582) |
| 1 | 52 | 0.8893 | 601 | 91.3 % | 0.9067 (1600) |
| 2 | 51 | 0.9379 | 605 | 91.6 % | 0.9598 (1554) |

At equal quality the hybrid needs 51 to 53 FEA solutions where NSGA-II on the FEA needs 601 to 711, a saving of
91 to 93 %. Given enough calls, the FEA-only GA ends higher: 0.907 to 0.960 after about 1600 calls. The hypervolume
is normalised per seed by the ideal and nadir points of both runs, so it compares the two runs of one seed, not
seeds with each other.

The GP's held-out error on the 8 FEA points of each round, in the last round:

* torque: mean 0.07 to 0.14 %, max 0.35 %;
* magnet mass: 0.01 % or less;
* ripple: mean 2.5 to 4.6 %, max 11 %. In the first round it was up to 46 %, so ripple is the weak output.

All 83 designs of the three hybrid fronts were re-evaluated with the finer FEA (480 gap nodes, 12 angles):

* all remain feasible;
* torque changes by at most 1.1 % (mean 0.8 %);
* ripple rises by 15 to 19 % on average and by up to 30 %, because 6 rotor angles under-sample it.

A design study should therefore apply its ripple limit with that margin or evaluate ripple with 12 angles.

Against the Prius rotor through the same finer FEA (343.4 N m, 1.236 kg, ripple 20.3 %), each front has 6 to 8
designs with more torque and less magnet: about 364 N m (+6 %) at 1.21 kg, with ripple 24 to 25 %. These designs
sit at the lower bound of the bridge, 1.0 mm against the Prius's 1.5 mm. A thinner bridge leaks less magnet flux,
but the bridge carries the centrifugal load. Mechanical stress, demagnetisation and the torque-speed envelope are not
constraints of this study, so it does not claim a better rotor. It shows the method and its FEA-call saving.

### End effects: end windings and magnet end leakage

Two effects outside the 2-D cross-section:

* End windings. `machines.endwinding` models each coil end as a thin conductor on the cylinder at the mean slot
  radius: axially out of the stack by the overhang, a half circle over the coil span, then back.
  * The same path gives the end-turn length of the phase resistance (`prius_phase_resistance`, 0.16 m per coil
    end), so resistance and inductance share one geometry.
  * `end_winding_inductance` integrates the phase inductances with Neumann's double integral, using the thin-wire
    kernel `1 / sqrt(r^2 + g^2)` with the wire's geometric mean distance `g = a e^-1/4` (`statics.filament_inductance`).
    The coil bundle is a round conductor of the slot's copper area. Both ends are included, in air and with the
    stator end face as an infinitely permeable mirror (image currents); the two cases bound the laminated face.
  * The dq leakage `L_end = L_aa - M_ab` adds to both axes in the voltage equations (`Drive(..., L_end=...)`). It
    cancels in the torque `1.5 p (psi_d i_q - psi_q i_d)`.
* Magnet end leakage. `getdp.end_leakage_3d` builds a 3-D magnetostatic model of half the stack, from the mid-plane
  to the end face, with an air region beyond the face: the cross-section extruded into prisms. The solver is GetDP,
  using a magnetic scalar potential and Picard iteration on the secant permeability. The model is compared with the
  same mesh without the end air, which is the extruded 2-D field. The ratio of the no-load air-gap flux under a pole
  corrects the 2-D magnet flux.

The filament inductance is verified against closed forms (`tests/test_emachine_ends.py`):

* a circular loop against Rosa's `mu0 R (ln(8R/a) - 7/4)`, within 2e-3;
* two coaxial loops against Maxwell's elliptic-integral mutual inductance, below 1e-3 and second order in the
  polygon;
* a loop above the iron mirror against the same loop plus the mutual of its image, within 2e-3.

`python examples/electrodynamics/16_prius_end_effects.py` (run on 2026-10-18) gives
`results/16_prius_end_effects.json` and `16_prius_end_effects.png`.

End windings, 24 coils of 9 turns, overhang 19.4 mm:

* `L_aa` 148.2 uH, `M_ab` 13.0 uH, so `L_end` = 135.2 uH in air;
* 115.3 uH with the iron face;
* 150 and 300 pieces per metre agree to 1.4e-4.

In air, `L_end` is 9.8 % of `L_d` (1.386 mH) and 5.6 % of `L_q` at 250 A. With the drive model and the FEA flux map
(250 A, 500 V), the maximum torque is:

| Speed (rpm) | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|---|---|---|---|---|---|
| 2-D model (N m) | 343.6 | 343.6 | 239.3 | 167.4 | 128.5 | 104.4 |
| + end windings in air | 343.6 | 338.1 | 213.3 | 150.1 | 115.7 | 94.2 |
| + end windings, iron face | 343.6 | 339.9 | 216.8 | 152.4 | 117.4 | 95.6 |

The end-winding reactance takes voltage, so the corner speed moves down. Above it the envelope falls by 10 to 11 %
in air: 239.3 to 213.3 N m at 3000 rpm and 104.4 to 94.2 N m at 6000 rpm. At part load in flux weakening, the
efficiency rises slightly because less negative `i_d` is needed:

* 4000 rpm, 100 N m: 94.27 to 94.36 %;
* 6000 rpm, 60 N m: 90.86 to 91.29 %.

The 3-D/2-D ratio of the magnet flux on three meshes:

* 0.9345 with 116 856 prisms;
* 0.9463 with 198 336 prisms;
* 0.9482 with 342 832 prisms.

The steps are 0.0118 then 0.0019. Lengthening the end air from 40 to 60 mm changes the ratio by 1e-4 (0.9462).
The half stack has one end and the full stack two, so the ratio applies to the whole 83.8 mm machine: the end
leakage removes 5.2 % of the 2-D magnet flux.

The correction scales the remanence in the 2-D FEA by that ratio, from 1.24 to 1.176 T, so saturation and
reluctance torque follow:

* the MTPA torque at 250 A falls from 345.7 to 336.7 N m (-2.6 %), with the MTPA angle moving from 49.5 to
  50.1 deg;
* the back-EMF fundamental at 1000 rpm falls from 87.6 to 82.9 V (-5.4 %).

The torque falls less than the flux because the reluctance torque does not depend on the magnets.

## Verification

Each case compares a solver or reference implementation with an independent result; where the mesh or step is
refined the observed order is measured. The table is the output of
`python examples/electrodynamics/07_em_verification.py` (`pinneapple_simulation.electromagnetics.verification`,
asserted by `tests/test_em_verification.py`):

<!-- verification-table:start -->
| Case | Quantity | Reference | Error (finest) | Observed order | Status |
|---|---|---|---|---|---|
| Coaxial capacitor (P1 FEM) | capacitance per metre | 2 pi eps / ln(b/a) | 3.87e-04 | 1.99, 2.00, 2.00 (expected 2) | pass |
| Two-dielectric parallel-plate capacitor (P1 FEM) | potential (max abs, V) | D uniform, E_i = D / eps_i (exact in P1) | 3.11e-15 | - | pass |
| Circular loop by Biot-Savart (polygon) | B off the axis | elliptic-integral loop field (Jackson 5.37) | 5.08e-05 | 1.99, 2.00, 2.00 (expected 2) | pass |
| Infinite wire and finite solenoid (Biot-Savart) | B of a 200-turn helix on the axis | current-sheet solenoid (closed form) | 5.32e-04 | - | pass |
| Coaxial mutual inductance | M (elliptic integrals) | Neumann double integral | 2.22e-16 | - | pass |
| Axisymmetric coil (FDM, eddy_current_fdm at DC) | on-axis B_z | sum of current-loop fields (closed form) | 3.69e-04 | - | pass |
| Mie sphere vs Bohren & Huffman / miepython | Q_sca, Q_ext, Q_back | BHMIE sample (B&H app. A) | 1.44e-06 | - | pass |
| Mie small-sphere (Rayleigh) limit | Q_sca / Rayleigh - 1 vs x | 8/3 x^4 \|(m^2-1)/(m^2+2)\|^2 | 4.41e-05 | 1.98, 2.00 (expected 2) | pass |
| Fresnel coefficients and Brewster angle | R + T - 1 (s and p, 3 interfaces) | energy conservation; r_p(atan n2/n1) = 0 | 6.66e-16 | - | pass |
| Bragg mirror (characteristic matrices) | R at the design wavelength | ((1 - Y)/(1 + Y))^2, Y = (nH/nL)^2N ns | 2.22e-16 | - | pass |
| Hertzian dipole radiated power | flux of the complete fields at kr = 0.05, 0.5, 20 | eta k^2 \|I l\|^2 / (12 pi) | 1.33e-13 | - | pass |
| Half-wave dipole radiation resistance | R_rad from the pattern | eta Cin(2 pi) / (4 pi) = 73.08 ohm | 0.00e+00 | - | pass |
| Dispersive ADE recursions (Drude, Debye, Lorentz) | discrete chi(omega) vs dt (Drude) | continuous chi(omega) | 9.02e-06 | 2.00, 2.00 (expected 2) | pass |
| Uniaxial crystal indices | n_o, n_e(theta) from the wave-normal eigenproblem | index ellipsoid | 4.44e-16 | - | pass |
| Round wire, A_z P1 FEM | A_z (max relative) | Ampere's law (closed form) | 5.18e-04 | 2.06, 2.04 (expected 2) | pass |
| Skin effect in a round wire (A_z eddy FEM) | impedance R + j X at a / delta = 3 | Bessel solution (closed form) | 8.55e-04 | - | pass |
| Froehlich-Kennelly iron ring (Newton) | \|B\| in saturated iron | B(H) at H = I / (2 pi r) | 5.08e-04 | - | pass |
| Jiles-Atherton integrator | M along a major loop vs RK4 sub-steps | 256 sub-steps; c = 1 stays anhysteretic | 1.28e-09 | 2.77, 1.11, 2.88, 2.20 | pass |
| Maxwell residual checks on an exact cavity mode | Ampere-Maxwell residual vs grid | exact TM12 mode (residual -> 0) | 2.33e-04 | 2.28, 2.22 (expected 2) | pass |
| FDTD 1-D dielectric half-space (Fresnel) | \|r(f)\|, 0.6-1.4 f0 | (n1 - n2)/(n1 + n2) | 2.02e-03 | 2.11, 2.03 (expected 2) | pass |
| FDTD dispersive half-spaces (Drude, Debye, Lorentz ADE) | \|r(f)\| of each medium, 0.6-1.4 f0 | Fresnel with eps(omega) of the pole | 8.74e-04 | 2.05, 2.02 (expected 2) | pass |
| FDTD 2-D PEC cavity resonance (TM11) | resonance frequency | c/2 sqrt(1/a^2 + 1/b^2) | 1.36e-05 | 2.01, 1.98 (expected 2) | pass |
| FDTD 3-D PEC box resonance (TE101) | resonance frequency | rectangular cavity formula | 2.37e-04 | 2.01, 2.00 (expected 2) | pass |
| FDTD rectangular waveguide TE10 propagation constant | beta at 1.5 f_c (2-D section) | k sqrt(1 - (fc/f)^2) | 3.89e-05 | 1.53, 1.25 | pass |
| FDTD bulk plasma (Drude) dispersion | k at omega = 1.67 omega_p | omega^2 = omega_p^2 + c^2 k^2 | 9.89e-05 | 1.83, 1.11 | pass |
| FDTD dielectric cylinder, TF/SF plane wave | total Ez on a ring r = lambda/2 | series of cylinder functions (Harrington 5-9) | 4.23e-03 | 1.97, 2.12, 1.95 (expected 2) | pass |
| FDTD Hertzian dipole (3-D) | phasor E_z at r = 0.8 lambda, theta = 30-120 deg | complete dipole fields (Balanis 4-8) | 1.17e-02 | 2.11, 1.98 (expected 2) | pass |
| FDTD Bloch-periodic oblique incidence (Floquet) | \|r_s\| at 25 degrees | Fresnel coefficient | 1.01e-03 | 2.05, 2.01 (expected 2) | pass |
| FDTD CPML reflection | max \|E - E_ref\| / max \|E_ref\| 5 cells from the PML | same run in a domain too large to reflect | 4.25e-05 | - | pass |
| FDTD fields through the Maxwell checks | Ampere-Maxwell residual of the snapshots | 0 for a solution of Maxwell's equations | 1.78e-03 | 1.51, 1.50 | pass |
| FDTD near-to-far-field transform (dipole pattern and power) | far-field E_theta(theta) | eta k \|I l\| sin(theta) / (4 pi) | 2.90e-03 | 2.08, 2.06 (expected 2) | pass |
| FDTD dielectric sphere vs Mie (3-D, TF/SF + near-to-far field) | Q_sca, eps = 4, x = 1.57 | Mie series | 2.41e-02 | - | pass |

Notes from the same run:

- Coaxial capacitor (P1 FEM): errors 2.45e-02, 6.17e-03, 1.55e-03, 3.87e-04
- Infinite wire and finite solenoid (Biot-Savart): 20 km segment vs mu0 I / (2 pi s): 6.3e-12
- Axisymmetric coil (FDM, eddy_current_fdm at DC): errors 1.00e-02, 1.40e-03, 3.69e-04 (finite box: A = 0 at 20 coil radii)
- Mie sphere vs Bohren & Huffman / miepython: published 5-digit values; miepython 3.3.0, 4 spheres incl. absorbing: max rel. diff 7.9e-09
- Mie small-sphere (Rayleigh) limit: PEC sphere vs 10/3 x^4: orders 2.00, 2.00
- Fresnel coefficients and Brewster angle: |r_p| at Brewster 0.0e+00
- Half-wave dipole radiation resistance: 73.079 ohm
- Dispersive ADE recursions (Drude, Debye, Lorentz): Debye orders 2.00, 2.00; Lorentz 2.00, 2.00
- Froehlich-Kennelly iron ring (Newton): Newton: 8 iterations to residual 1.9e-12
- Jiles-Atherton integrator: c = 1 deviation from the anhysteretic curve 1.2e-05 Ms (orders irregular: the pinning switch makes dM/dH non-smooth)
- Maxwell residual checks on an exact cavity mode: a 20 % wrong frequency gives 0.181 (rejected) and Poynting imbalance 0.218
- FDTD 1-D dielectric half-space (Fresnel): errors 3.56e-02, 8.25e-03, 2.02e-03 at 20/40/80 cells per wavelength
- FDTD dispersive half-spaces (Drude, Debye, Lorentz ADE): Drude: errors 2.7e-03, 6.5e-04, 1.6e-04, orders 2.05, 2.02; Debye: errors 6.5e-03, 1.6e-03, 4.0e-04, orders 2.02, 2.01; Lorentz: errors 1.4e-02, 3.5e-03, 8.7e-04, orders 2.03, 2.01
- FDTD 2-D PEC cavity resonance (TM11): against the scheme's own dispersion relation: max 1.6e-07
- FDTD rectangular waveguide TE10 propagation constant: errors 2.7e-04, 9.3e-05, 3.9e-05; against the scheme's own dispersion relation max 2.5e-05 (the phase-slope measurement floor)
- FDTD bulk plasma (Drude) dispersion: errors 7.6e-04, 2.1e-04, 9.9e-05
- FDTD dielectric cylinder, TF/SF plane wave: eps = 3, a = 0.3 lambda, sub-cell (arithmetic) averaging of eps; errors 3.8e-02, 1.7e-02, 9.3e-03, 4.2e-03 at 20/30/40/60 cells per wavelength
- FDTD Hertzian dipole (3-D): errors 4.9e-02, 2.1e-02, 1.2e-02 at 10, 15, 20 cells per wavelength
- FDTD Bloch-periodic oblique incidence (Floquet): errors 1.7e-02, 4.1e-03, 1.0e-03
- FDTD CPML reflection: 5 cells: 2.8e-03 (-51 dB), 10 cells: 4.2e-05 (-87 dB), 20 cells: 5.3e-06 (-106 dB)
- FDTD fields through the Maxwell checks: Poynting imbalance 6e-05, 4e-06, 9e-07; residuals 1.4e-02, 5.0e-03, 1.8e-03
- FDTD near-to-far-field transform (dipole pattern and power): radiated power errors 8.0e-03, 3.6e-03, 2.0e-03 vs eta k^2 |I l|^2 / (12 pi)
- FDTD dielectric sphere vs Mie (3-D, TF/SF + near-to-far field): errors 2.1e-02, 1.8e-02, 2.4e-02 at 10, 15, 20 cells per wavelength: a plateau near 2 %, not a convergence order (harmonic/arithmetic sub-cell averaging is not the full anisotropic smoothing an oblique surface needs; arithmetic averaging alone gives 9.7 % at 10 cells)
<!-- verification-table:end -->

## Fixed while building this

* `eddy_current_fdm.annular_current_source` normalised the coil current density by the nominal band area, so the
  injected current depended on how the band fell on the grid (0.75 N I in one case); it now injects exactly N I.
  `axial_flux_density` returned half the field on the axis (`A / r` with `A = 0` there instead of its limit
  `dA/dr`), and `eddy_current_density` returned `+j omega sigma A`, the eddy current with the wrong sign for the
  solver's `exp(+j omega t)` convention. Together these explain the coil field "about 0.20 times" the loop formula
  recorded in the validation gallery.
* The Newton solver of `solve_magnetostatic` could stall at a residual of 1e-7 because its Armijo test compared
  energy differences below round-off; near convergence it now takes the full Newton step (6 iterations to 3e-13
  on the saturated iron ring).

## Limits

* The FEM solvers are 2-D (planar or axisymmetric for `eddy_current_fdm`), first order (P1), and resolve
  geometry on the mesh (polar grids are staircased in angle).
* FDTD: uniform grids only. The sub-cell averaging is not the full anisotropic smoothing: curved surfaces with E
  normal to them keep an error of a few percent that does not converge (2 % on the Mie sphere below). The CPML is tuned for vacuum; inside
  dispersive or negative-permittivity media its reflection is not measured. Surface plasmon polaritons are
  verified only in closed form: an FDTD SPP run on a lossless Drude interface did not converge cleanly at
  affordable resolutions (4.9 % at 80 cells per wavelength), so no FDTD SPP number is claimed. NumPy, single
  thread: a 60^3 grid takes 19 ms per step with PEC walls and 38 ms with CPML on all sides (measured in the
  development container).
* The differential compiler kinds do not differentiate material maps: jumps of `eps`, `mu` or `nu` are only seen
  by the integral kinds.
* `maxwell_checks` uses central differences on tensor grids: fields with discontinuities (material interfaces,
  PEC edges) leave residuals there that are not errors of the field; crop them or check each region separately.
* Machines: 2-D FEA; end effects as corrections (end-winding inductance from thin filaments, magnet end leakage
  as one 3-D no-load ratio applied through the remanence, not a coupled 3-D solution at load); skew by
  multi-slice, without inter-segment axial flux; magnets at a uniform temperature,
  iron loss from B waveforms with the elliptic-to-alternating split (no minor loops, no rotational-loss data),
  magnet eddy loss resistance-limited (valid while the skin depth exceeds the magnet size). The Bertotti fit to the
  M400-50A table has measured errors (see the notes above); the table model is exact at the catalogue points only.
* Thermal: a lumped network whose material, contact and heat-transfer data are illustrative defaults, not
  measured or fitted to a Prius; one temperature per node (no hot spot inside the slot, magnets uniform). `Hcj(T)`
  is the grade minimum with an assumed -0.55 %/K coefficient. The demagnetisation check is static, a square-loop
  knee test on the 2-D field (no recoil-line modelling, no partial demagnetisation of later operation). The
  worst-element ratio does not converge under mesh refinement (field singular at the sharp magnet corners), so
  only area fractions are reported as results. The
  short-circuit peak `2 psi_pm / L_d` is an estimate, not a transient simulation.
* Drive: steady state only (no current-controller dynamics, no PWM ripple or harmonics losses), no inverter,
  mechanical or battery losses; the voltage limit is that of linear space-vector modulation.
* `JilesAtherton` is the scalar model; its parameters must be fitted to measured loops of the material.
  `illustrative_steel` is a formula, not a measured grade.

## References

* J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley, 1999.
* D. J. Griffiths, *Introduction to Electrodynamics*, 4th ed., Cambridge University Press, 2017.
* A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*,
  3rd ed., Artech House, 2005.
* J.-M. Jin, *The Finite Element Method in Electromagnetics*, 3rd ed., Wiley-IEEE Press, 2014.
* D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley, 2012.
* Z. Yang, M. Krishnamurthy, I. P. Brown, "Electromagnetic and vibrational characteristic of IPM over full
  torque-speed range", IEEE IEMDC 2013, pp. 295-302.
* M. Arkkio, *Analysis of induction motors based on the numerical solution of the magnetic field and circuit
  equations*, Acta Polytechnica Scandinavica EE-59, 1987.
* G. Bertotti, "General properties of power losses in soft ferromagnetic materials", IEEE Trans. Magn. 24(1),
  621-630, 1988.
* J. R. Shewchuk, "Delaunay refinement algorithms for triangular mesh generation", Comput. Geom. 22, 21-74, 2002.
* pyleecan 1.4.2 (Apache-2.0): machine and material library, https://github.com/Eomys/pyleecan.
* C. A. Balanis, *Antenna Theory*, 4th ed., Wiley, 2016.
* R. F. Harrington, *Time-Harmonic Electromagnetic Fields*, McGraw-Hill, 1961.
* M. Born and E. Wolf, *Principles of Optics*, 7th ed., Cambridge University Press, 1999.
* C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small Particles*, Wiley, 1983.
* W. J. Wiscombe, Improved Mie scattering algorithms, *Appl. Opt.* 19 (1980) 1505-1509.
* D. C. Jiles and D. L. Atherton, Theory of ferromagnetic hysteresis, *J. Magn. Magn. Mater.* 61 (1986) 48-60.
* S. A. Maier, *Plasmonics: Fundamentals and Applications*, Springer, 2007.
* E. B. Rosa and F. W. Grover, Formulas and tables for the calculation of mutual and self-inductance, *Bulletin
  of the Bureau of Standards* 8 (1912).
