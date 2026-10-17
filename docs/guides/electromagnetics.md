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
* `JilesAtherton` is the scalar model; its parameters must be fitted to measured loops of the material.
  `illustrative_steel` is a formula, not a measured grade.

## References

* J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley, 1999.
* D. J. Griffiths, *Introduction to Electrodynamics*, 4th ed., Cambridge University Press, 2017.
* A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*,
  3rd ed., Artech House, 2005.
* J.-M. Jin, *The Finite Element Method in Electromagnetics*, 3rd ed., Wiley-IEEE Press, 2014.
* D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley, 2012.
* C. A. Balanis, *Antenna Theory*, 4th ed., Wiley, 2016.
* R. F. Harrington, *Time-Harmonic Electromagnetic Fields*, McGraw-Hill, 1961.
* M. Born and E. Wolf, *Principles of Optics*, 7th ed., Cambridge University Press, 1999.
* C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small Particles*, Wiley, 1983.
* W. J. Wiscombe, Improved Mie scattering algorithms, *Appl. Opt.* 19 (1980) 1505-1509.
* D. C. Jiles and D. L. Atherton, Theory of ferromagnetic hysteresis, *J. Magn. Magn. Mater.* 61 (1986) 48-60.
* S. A. Maier, *Plasmonics: Fundamentals and Applications*, Springer, 2007.
* E. B. Rosa and F. W. Grover, Formulas and tables for the calculation of mutual and self-inductance, *Bulletin
  of the Bureau of Standards* 8 (1912).
