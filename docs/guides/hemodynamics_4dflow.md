# Pressure and wall shear stress from 4D-flow MRI: what the phantom gives back

4D-flow MRI measures velocity. The two numbers a clinician wants from it — the pressure difference along a vessel and
the wall shear stress — are both derivatives of that velocity, and one of them is a derivative taken **at the wall**,
which is the one place the scanner never samples. The claim that a physics-informed network can supply them is
testable, and this module tests it the only way that means anything: on a case whose answer is known in closed form.

The case is a **synthetic** phantom, not a patient and not a downloaded benchmark: a rigid straight artery with
pulsatile laminar flow, whose exact solution Womersley wrote in 1955. Everything below — the measurement, the
network, the errors — is generated from that closed form, so the truth is not an estimate.

```
-dp/dz = Re[ G e^{i w t} ]        W(r) = (G / (i rho w)) [1 - J0(beta r/R) / J0(beta)],   beta = alpha sqrt(-i)
alpha = R sqrt(w / nu)            WSS  = -mu dW/dr |_{r=R}
```

`pinneapple_physics.hemodynamics` carries the closed form for velocity, flow rate, wall shear stress and pressure
gradient, for one harmonic or for a sum of them (it is linear in the forcing, so a real waveform is exact too).

```python
from pinneapple_physics.hemodynamics import BLOOD, case_from_vessel, DEFAULT_VESSELS, sample_4d_flow

wave = case_from_vessel(DEFAULT_VESSELS[3])            # R = 15 mm, alpha = 20.7, 60 bpm
wave.peak_velocity(), wave.wall_shear(0.0)            # 0.25 m/s, 0.64 Pa
four_d = sample_4d_flow(wave, n_radial=8, n_phases=24, n_planes=3, noise=0.05, seed=1)
four_d.voxel, four_d.wall_gap                         # 1.9 mm, 0.94 mm
```

## What the acquisition leaves out

The samples are **voxel centres**: eight voxels across the radius put the first one 0.94 mm from the wall, and the
wall itself is never measured. The wall shear stress lives in the boundary layer, and its thickness in this flow is
`R/alpha = sqrt(nu/omega)` — independent of the vessel, **0.72 mm** for blood at 60 bpm. So a 4D-flow voxel of one to
three millimetres is the same size as the layer it would have to resolve, on every vessel, and the closer the estimate
looks at the wall the less it has to look at.

That is the measured statement this guide is built on, and it is why the wall shear stress has to come from the
equations rather than from the data.

## The network

The fit is the axisymmetric Navier-Stokes system, in variables that make the phantom's symmetries exact:

* `y = r/R` and `tau = omega t`, the cycle in radians;
* `v = w/W0` (the measured quantity), `u = u_r/W0` (which the exact solution has as zero, and the equations allow
  nothing else) and `g = -(dp/dz)/(rho omega W0)` (the pressure gradient, a function of the phase alone in a
  developed straight tube — learning it **is** recovering the relative pressure, because an incompressible field has
  no absolute one);
* the ansatz `v = (1 - y^2) N_v(y^2, sin tau, cos tau)`, `u = y(1 - y^2) N_u`, `g = N_g(sin tau, cos tau)`: no-slip,
  axis symmetry, wall impermeability and periodicity hold to machine precision, so the optimiser cannot trade them
  away for a better data fit;
* three residuals, from `pinneapple_physics.pinn_solver`'s `PINNFactory` (the repository's symbolic-to-autograd
  compiler), with `A = Re/alpha^2` and `C = 1/alpha^2`:

```
u_y + u/y = 0
u_tau + A (u u_y) = C (u_yy + u_y/y - u/y^2)
v_tau + A (u v_y) = g + C (v_yy + v_y/y)
```

`C` is 2.3e-3 at `alpha = 21`. The viscous term is invisible in the core and decisive inside a layer `1/alpha` thick:
this is the whole difficulty, and it is why the fit is judged on the gradient at the wall and not on the mean residual.

The scale `W0` is the largest measured velocity — it comes from the acquisition, not from the answer. What the network
has to find is the shape and the phase; the slope at the wall follows from them, by autograd.

## What it recovers

Five phantom cases: four vessels whose radii give Womersley numbers from 2.8 to 20.7 at 60 bpm, and one physiological
five-harmonic waveform at the aortic radius. Each is measured at 8 voxels across the radius, 24 cardiac phases, 3
planes and 5 % velocity noise, and fitted by an ensemble of three independently seeded networks. Every error below is
relative L2 over the cardiac cycle against the closed form.

| phantom | alpha | first sample | wall shear stress | two-point difference | pressure gradient | ensemble spread | truth inside 2 sd |
|---|---|---|---|---|---|---|---|
| small artery, R = 2 mm | 2.8 | 0.12 mm (0.2 layers) | **1.2 %** of 1.35 Pa | 27 % | 2.5 % of 36 Pa | 0.2 % | 25 % of the cycle |
| femoral, R = 4 mm | 5.5 | 0.25 mm (0.3 layers) | **2.3 %** of 0.85 Pa | 44 % | 3.0 % of 41 Pa | 0.2 % | 25 % of the cycle |
| carotid, R = 8 mm | 11.0 | 0.50 mm (0.7 layers) | **2.7 %** of 0.68 Pa | 77 % | 3.3 % of 68 Pa | 0.3 % | 21 % of the cycle |
| aorta, R = 15 mm | 20.7 | 0.94 mm (1.3 layers) | **2.8 %** of 0.70 Pa | 99 % | 2.8 % of 137 Pa | 0.5 % | 29 % of the cycle |
| physiological, 5 harmonics | 20.7 | 0.94 mm (1.3 layers) | **7.0 %** of 1.09 Pa | 100 % | 8.4 % of 249 Pa | 2.7 % | 92 % of the cycle |

"Layers" is the gap measured in boundary-layer thicknesses, `wall_gap / (R/alpha)`. The pressure column is the gradient
over the studied segment, ten radii long -- 36 Pa over 2 cm of the small artery, 137 Pa over 15 cm of the aorta -- and
is what a clinician would call the pressure drop.

Read as a whole, the table says three things:

* **the network recovers both targets to a few per cent** from a measurement with no sample at the wall, in an
  eighty-second fit on one CPU core;
* **the error does not grow with the Womersley number**, although the measurement does get worse with it: the aorta's
  first sample sits outside the boundary layer altogether (1.3 layers away) and its wall shear stress is still off by
  2.8 %, the same as the carotid's, whose layer the samples do resolve. What the data pin down is the shape in the
  core and the phase; the slope at the wall comes from the equations and the no-slip condition, which is the entire
  claim being tested;
* **the two-point difference is not in the same league**, and the fatter the vessel the worse it does: at 99 % on the
  aorta it is no better than returning zero.

The multi-harmonic case is the honest counterweight. Its fourth harmonic has `alpha = 41`, a layer 0.18 mm thick, and
the error triples: a real waveform asks for the thin layers a single frequency never shows.

## Where the wall shear stress stops being recoverable

<!--LIMIT-->

## What the measurement corrected

<!--CORRECTED-->

## What this is not

It is not patient data and it is not a scanner: the acquisition is generated from the closed form with Gaussian noise,
so there is no slice profile, no partial volume, no velocity aliasing, no motion and no segmentation error. The vessel
is straight and rigid, the flow is developed and axisymmetric, and the blood is Newtonian — Womersley's assumptions,
which is the price of having an exact answer to compare against. A curved, compliant, bifurcating artery has no closed
form, and nothing here says how the error behaves there.

The uncertainty is a **deep ensemble**: its spread prices the initialisation and the optimisation, not the model form.
A boundary layer that no member resolves is a bias all of them share, and the spread does not report it — which is
exactly what the resolution sweep is for.

The next idea of the issue — pharmacokinetics and pharmacodynamics, compartment ODEs with a learned correction and
regime changes flagged by the residual of `measurement_check` — is **not done**. One measured thing is worth more than
two half-done ones, and the haemodynamics half is the one with a closed-form reference.
