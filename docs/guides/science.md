# Science by area (`pp.science`)

`pinneapple_science` (`pp.science`) organises PINNeAPPle by area of science. Every experiment is a small,
self-checking study: it computes a quantity by a numerical route (a field solution, a quadrature, a time
integration, a fit to noisy readings) and compares it with a cited reference, in one or more dimensions. The
**atlas** maps every area the module will cover, says which are available, which the rest of the library already
covers in part, and which are planned, and points to that code instead of copying it.

```python
import pinneapple as pp

pp.science.atlas()                                     # every area and subarea, with its status and pointers
pp.science.experiments(area="electromagnetism", dim="3D+t")
r = pp.science.run("em.faraday_induction")             # a magnet falling through a loop, 3-D in time
print(r)                                               # computed vs reference, physics checks, trust decision
r.save("out/faraday")                                  # result.json, figures, the animation (MP4 and GIF)
```

```bash
python -m pinneapple_science atlas --html atlas.html --check     # the atlas page; every pointer validated
python -m pinneapple_science list --dim 2D
python -m pinneapple_science run em.hall_effect
python -m pinneapple_science report all                          # rewrites examples/science/ (manifests, seals)
```

Flow Studio: node `science.run` (choose the experiment and the dimension) and `science.atlas`.

## What every experiment declares

| field | values |
|---|---|
| `area` | an atlas area id, e.g. `physics.electromagnetism` |
| `level` | `academic`: didactic, checked against a textbook value; `research`: configurable, with convergence and uncertainty |
| `kind` | `theory`: a numerical or symbolic derivation checked against a closed form, a published value or a limit; `practice`: a virtual bench with instruments (noise, calibration error, resolution) and a fit with its uncertainty |
| `dims` | `0D` (lumped elements, no spatial coordinate), `1D`, `2D`, `3D`, `3D+t` (three space dimensions and time). The same experiment runs in every dimension its physics allows. |

Every result carries:

- **comparisons** with references (`closed_form`, `published`, `codata`, `limit`, `conservation`, `convergence` or,
  for a bench, `ground_truth`), each with its source, the computed and reference values, the measured error and the
  tolerance;
- **physics checks** (conservation, residuals of Maxwell's equations with `pp.maxwell`, convergence order, symmetry);
- the **trust report** (`pinneapple_analysis.trust.TrustReport`): a failed reference comparison is critical
  (REJECT), any other failed check gives REVIEW;
- **figures** and, for `3D+t`, an **animation** of the 3-D field in time;
- the Lab manifest `experiment.json` and the **Lab seal** stamped on the figures (`python -m pinneapple_science
  report`).

No number is typed in: values are computed by the run; references carry their citation.

## Electromagnetism (available)

Built on `pp.em` (`pinneapple_simulation.electromagnetics`: statics, FDTD, Mie, media, waves) and `pp.maxwell`
(`pinneapple_physics.maxwell_checks`). The committed results, with every comparison, are in
[`examples/science/`](../../examples/science/README.md); each experiment folder has its README with the tables
and sealed figures.

| experiment | dims | what is computed | reference |
|---|---|---|---|
| `em.coulomb_superposition` | 3D, 2D | a charged ring and two finite line charges as sums of point charges | ring potential (elliptic integral, Jackson 3.3); finite line (Griffiths ex. 2.2) |
| `em.gauss_law` | 3D, 2D | flux through a cube / square by Gauss-Legendre quadrature; div E residual of a charge cloud | Q_enc / eps0; `pp.maxwell.gauss_electric` |
| `em.parallel_plate_capacitor` | 2D, 3D | C from the field energy of a finite-volume solution, Richardson-extrapolated | Palmer (1937) strips; Kirchhoff (1877) discs; 3-D Cartesian vs axisymmetric |
| `em.biot_savart` | 3D | wire (midpoint rule), loop (polygon), 400-turn helix | Griffiths 5.37, loop elliptic integrals, finite solenoid |
| `em.ampere_law` | 2D, 3D | circulation of B from a FV solution of lap A = -mu0 J; paths linking a square loop once, twice, never | mu0 I_enc x linking number |
| `em.faraday_induction` | 3D+t | flux of a falling magnet's field through a loop (Stokes, oint A.dl), EMF = -dPhi/dt; animation | Maxwell's coaxial-circle formula; free-fall time of the sign change |
| `em.circuits_rc_rl_rlc` | 0D | step responses by ODE integration; tau, omega_d, alpha measured from the waveforms | RC, L/R, sqrt(1/LC - (R/2L)^2), R/2L; energy balance |
| `em.hall_effect` | 2D | Hall bar with the Drude conductivity tensor; Hall coefficient at mid-length; geometric factor vs L/W | R_H = 1/(n q) |
| `em.dipole_radiation` | 3D, 3D+t | Poynting flux through spheres from kr = 0.3 to 30; energy per period from the real fields; 4-D Maxwell residuals; animation | eta k^2 abs(I l)^2 / (12 pi), D = 1.5 |
| `em.lossy_plane_wave` | 1D | FDTD in a conducting dielectric; alpha and beta fitted from the DFT field | gamma = sqrt(j omega mu (sigma + j omega eps)) |
| `em.fresnel_snell` | 2D | Bloch-periodic FDTD at oblique incidence; abs(r) for s and p; Brewster; transmitted angle from the phase gradient | Fresnel, Brewster, Snell |
| `em.waveguide_modes` | 2D | FD Helmholtz eigenproblem of a WR-90 cross-section, TE and TM | f_c = c/(2 pi) sqrt((m pi/a)^2 + (n pi/b)^2) |
| `em.cavity_resonance` | 2D, 3D | FDTD ring-down; spectral peaks matched to modes | rectangular cavity formula |
| `em.mie_scattering` | 3D | Mie series efficiencies | BHMIE published sample; Rayleigh; optical theorem; Q_ext -> 2 |
| `em.bench_mu0_solenoid` | 3D | **bench**: Hall probe, ammeter, caliper; mu0 from the slope of B(I) with a GUM budget; coverage of U checked by Monte Carlo | CODATA 2022 |
| `em.bench_rc_discharge` | 0D | **bench**: 8-bit oscilloscope with noise and time-base error; C = tau / R with its uncertainty; coverage by Monte Carlo | the capacitor's true value (hidden from the fit) |

### Research level

| experiment | dims | what is computed | reference / evidence |
|---|---|---|---|
| `em.research_fdtd_3d` | 3D+t | a 3-D FDTD study configured by a dict (domain, cells per wavelength, `sources`, `materials`: spheres, boxes, cylinders of any medium); a Hertzian dipole on a refinement ladder, radiated power by the near-to-far-field transform; then the dipole beside a dielectric sphere on the same ladder | closed-form dipole power and pattern, observed order, Richardson and Roache's GCI; for the configured case (no closed form) its own extrapolation and GCI against a declared target; Faraday, Ampere-Maxwell and Gauss residuals of a block of the field in x, y, z and t |
| `em.research_uq_waveguide` | 2D | Monte Carlo of a, b, eps_r tolerances through the finite-difference eigen-solver: mean, std, 95 % interval of the TE10 and TE20 cut-offs | first-order GUM propagation with closed-form sensitivities, tolerance from the sampling error of the std |
| `em.research_inverse_permittivity` | 1D | eps_r and sigma of a slab recovered by least squares from a reflection spectrum measured by FDTD with receiver noise, with the Jacobian covariance; repeated on noise realisations | the truth the measurement was generated with (hidden from the fit); coverage of the stated u |
| `em.research_surrogate_hall` | 2D | a Gaussian process of the Hall geometrical factor G(L/W, mu B) trained on solver runs (Latin hypercube), variance calibrated by leave-one-out on the training runs | error on held-out solver runs; +-2 sigma coverage; the long-bar limit |

The FDTD study writes the fourth dimension two ways: an MP4 / GIF of E_z r on the three central planes in time,
and a single-file three.js page (the Studio viewer, `pinneapple_tools.visualization.studio.web_viewer`) with play,
pause and a time slider.

```python
from pinneapple_science.em.fdtd_study import FDTDConfig, run_fdtd

cfg = FDTDConfig.from_dict({"f0": 1e9, "size": 2.4, "cells_per_wavelength": 14,
                            "sources": [{"kind": "dipole", "component": "Ez", "at": [0, 0, 0]}],
                            "materials": [{"shape": "box", "center": [0.5, 0, 0], "size": [0.2, 0.6, 0.6],
                                           "eps_r": 6.0, "sigma": 0.01}]})
run = run_fdtd(cfg, snapshots=24)            # run.power (W at f0), run.pattern(theta), run.slices, run.volume
```

Limits stated by the experiments themselves: Palmer's and Kirchhoff's formulas are asymptotic (the capacitor
experiment measures how Kirchhoff's error falls with a/d: +6.2 %, +2.3 %, +0.8 % at a/d = 2.5, 5, 10 in the
committed run); the thin-plate edge limits the finite-volume order to about 1; the Fresnel comparisons use an
absolute tolerance of 5e-3 on abs(r) at 40 cells per wavelength. The dipole beside a sphere converges at first
order (staircase sphere with fill-fraction averaging), so its finest grid carries a GCI of a few percent; the
surrogate's +-2 sigma band covers 80 % of the held-out runs, at the threshold the experiment declares.

## The atlas

The atlas is versioned data, [`pinneapple_science/data/atlas.json`](../../pinneapple_science/data/atlas.json)
(schema `pinneapple.science.atlas/1`). The full table is [`examples/science/ATLAS.md`](../../examples/science/ATLAS.md)
and the navigable page `examples/science/atlas.html` (search, filters by status and by branch). Statuses:

- **available**: `pp.science` experiments in every dimension listed, each checked against its reference;
- **partial**: library code covers part of the area (a module, a problem preset or a Lab experiment the atlas
  points to); no `pp.science` experiment yet;
- **planned**: mapped, nothing in the library yet.

Areas: classical mechanics; electromagnetism and optics; thermodynamics and statistical physics; fluid mechanics;
continuum and solid mechanics; acoustics and waves; quantum mechanics; special and general relativity; nuclear and
particle physics; condensed matter; plasma physics; astrophysics and cosmology; geophysics; chemistry; materials
science; biology and biophysics; neuroscience; ecology and epidemiology; earth and climate sciences; applied
mathematics and numerical methods.

`pp.science.validate()` (and `tests/test_science.py`) fails when a pointer names a module, preset or Lab
experiment that does not exist, when an `available` subarea has no experiment, or when a registered experiment is
missing from the atlas.

## Adding an area or an experiment

1. **Map it.** Add the area (or the subarea) to `pinneapple_science/data/atlas.json` with its status and the
   pointers to code that already covers it (`{"module": ...}`, `{"preset": ...}`, `{"lab": ...}`). Point to
   existing code; do not copy it. Bump `version`.
2. **Write the experiment** in `pinneapple_science/<area>/<topic>.py` (new physics or numerics go in the library,
   in the package they belong to, with tests; the experiment calls them):

   ```python
   from pinneapple_science.core import Experiment, Reference
   from pinneapple_science.registry import register

   @register
   class Pendulum(Experiment):
       name = "mech.pendulum_period"
       title = "Pendulum period"
       area = "physics.classical_mechanics"
       topic = "oscillators"
       level, kind, dims = "academic", "theory", ("0D",)
       references = ("Taylor, Classical Mechanics (2005), sec. 4.9",)
       params = {"L": 1.0, "theta0": 0.1}

       def _run(self, res, dim, p):
           T = ...                                   # integrate the ODE, measure the period from zero crossings
           res.compare("small-angle period", T, Reference(2 * 3.14159265 * (p["L"] / 9.80665) ** 0.5,
                       "Taylor eq. 5.12", "limit"), tol=1e-3, unit="s")
           res.figure("pendulum", fig)
   ```

3. **Register the module** in `pinneapple_science.registry.MODULES`, list the experiment under its subarea in the
   atlas (`{"experiment": "mech.pendulum_period"}`) and in the Flow node list
   (`pinneapple_flow/nodes/science.py`, `EXPERIMENTS`), and set the subarea to `available` when every dimension
   it claims runs.
4. **Commit the results**: `python -m pinneapple_science report mech.pendulum_period` writes
   `examples/science/<area>/<experiment>/` with `result.json`, figures, the manifest and the sealed figures; then
   `python -m pinneapple_science report all` refreshes the catalogue. Run `python -m pinneapple_lab reindex` first
   in a fresh checkout so the curation sees every committed run.
5. **Test it**: add the name to `FAST` in `tests/test_science.py` (with a `quick` parameter set that runs in
   seconds and still passes its references).
