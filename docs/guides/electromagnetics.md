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

The cross-check is against a published benchmark, not against FEMM or pyleecan (neither runs in this
environment: FEMM is Windows-only, pyleecan's wheel needs a git-only dependency). The reference is the
torque against current angle of the Prius motor at 250 A peak computed by Yang, Krishnamurthy and Brown (IEEE IEMDC
2013), as tabulated in pyleecan's test suite. Their model and pyleecan's geometry are not the same file (centre
post, magnet temperature, steel data), so agreement within a few per cent is the expectation, not a mesh error.

`python examples/electrodynamics/10_pmsm_prius.py --write-docs` (run on 2026-10-17, 960 gap nodes, 29 936
elements) wrote the table below and `examples/electrodynamics/results/10_pmsm_prius.json`; figures:
`10_pmsm_field.png` (|B| and flux lines at MTPA, 250 A), `10_pmsm_curves.png` (torque against current angle with the
published curve, cogging, back-EMF), `10_pmsm_mtpa.png` (MTPA locus, L_d and L_q).

Results of that run (magnets at 20 degC, 1.24 T):

| Peak current (A) | MTPA angle (deg from q) | Mean torque (N m) | Ripple pk-pk | L_d (mH) | L_q (mH) | psi_pm (Wb) |
|---|---|---|---|---|---|---|
| 50 | 26.9 | 70.4 | 24.9 % | 1.948 | 4.870 | 0.2141 |
| 100 | 36.5 | 147.8 | 25.6 % | 1.744 | 3.698 | 0.2184 |
| 150 | 42.4 | 221.3 | 22.4 % | 1.627 | 3.103 | 0.2217 |
| 200 | 46.5 | 287.9 | 22.8 % | 1.515 | 2.731 | 0.2201 |
| 250 | 49.5 | 345.6 | 22.9 % | 1.392 | 2.428 | 0.2138 |

L_d, L_q and psi_pm are the apparent frozen-permeability values at each MTPA point (saturation lowers L_q from 4.87
to 2.43 mH). No load: cogging 1.38 N m peak to peak, magnet flux linkage 0.209 Wb peak per phase, back-EMF 107.7 V
line-to-line rms at 1000 rpm (THD 8.8 %). Losses at 1500 rpm (100 Hz), 250 A MTPA: stator iron 391 W, rotor iron
29 W with the catalogue table (332 W and 32 W with the Bertotti fit), magnet eddy loss 37.9 W (resistance limited,
one piece per magnet). The Bertotti fit to the M400-50A table up to 1 kHz has 8.2 % rms and 22 % maximum relative
error; the table model is the default. Mesh convergence of the torque is not monotone (480, 960, 1920 gap nodes:
343.20, 345.50, 344.08 N m): treat the FEA torque as converged to about 0.4 %.

<!-- machine-verification:start -->
| Case | Quantity | Reference | Error (finest) | Observed order | Status |
|---|---|---|---|---|---|
| Magnetised ring, sector mesher + P1 FEM | relative L2 error of A_z, h halved | closed form (two magnetised cylinders + harmonic) | 1.91e-04 | 1.89, 1.91, 1.99 (expected 2) | pass |
| Machine sector geometry (Prius slot and magnet areas) | slot area, gap nodes doubled | exact area (trapezoid + half disc) | 5.70e-05 | 1.99, 1.81, 1.90 (expected 2) | pass |
| Prius IPM: Maxwell stress vs co-energy torque | Arkkio torque at 250 A, 3 rotor angles | virtual work dW'/dtheta (4th-order differences) | 1.92e-03 | - | pass |
| Prius IPM: dq power balance and frozen permeability | mean Maxwell-stress torque over 60 el. deg | 1.5 p (psi_d i_q - psi_q i_d) | 2.37e-04 | - | pass |
| Prius IPM: torque vs current angle (Yang et al. 2013) | mean torque, 250 A, Phi0 = 60..180 deg | published FEA (IEMDC 2013), 13 angles | 2.12e-02 | - | pass |
| Prius IPM: mesh convergence of the peak torque | mean torque at 250 A, 50 deg, gap nodes doubled | successive differences | 4.13e-03 | - | pass |

Notes from the same run:

- Magnetised ring, sector mesher + P1 FEM: 8 sector copies, magnet edges on mesh edges
- Machine sector geometry (Prius slot and magnet areas): magnet area (straight edges) exact to 4.4e-16
- Prius IPM: Maxwell stress vs co-energy torque: max relative difference over the angles
- Prius IPM: dq power balance and frozen permeability: frozen-permeability superposition error 3.1e-10; L_d = 1.393 mH, L_q = 2.453 mH, psi_pm = 0.2142 Wb at this point
- Prius IPM: torque vs current angle (Yang et al. 2013): peak 345.5 N m at Phi0 = 140 deg (published 353 N m at 140 deg); rms deviation 5.0 % of the peak
- Prius IPM: mesh convergence of the peak torque: 480: 343.20 N m, 960: 345.50 N m, 1920: 344.08 N m; difference ratio 1.62
<!-- machine-verification:end -->

## Verification

Each case compares a solver or reference implementation with an independent result; where the mesh or step is
refined the observed order is measured. The table is the output of
`python examples/electrodynamics/07_em_verification.py` (`pinneapple_simulation.electromagnetics.verification`,
asserted by `tests/test_em_verification.py`):

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
* The differential compiler kinds do not differentiate material maps: jumps of `eps`, `mu` or `nu` are only seen
  by the integral kinds.
* `maxwell_checks` uses central differences on tensor grids: fields with discontinuities (material interfaces,
  PEC edges) leave residuals there that are not errors of the field; crop them or check each region separately.
* Machines: 2-D only (no end windings, no axial flux, no skew in the FEA), magnets at a uniform temperature,
  iron loss from B waveforms with the elliptic-to-alternating split (no minor loops, no rotational-loss data),
  magnet eddy loss resistance-limited (valid while the skin depth exceeds the magnet size). The Bertotti fit to the
  M400-50A table has measured errors (see the notes above); the table model is exact at the catalogue points only.
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
