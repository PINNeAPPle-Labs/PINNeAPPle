# Evidenced audit report

Generated 2026-09-04 by actually running the test suites named below (not
estimated or guessed — every number here is a real test result). See
`ROADMAP_PHYSICS_AI_HUB.md` section 1.1 for the two-tier audit design this
implements (Tier A = breadth/"does it run", Tier B =
`tests/test_manufactured_solutions.py`, physics correctness).

## Astrophysics/space specialization: 7 new PDE/ODE kinds, all verified

PINNeAPPle's chosen initial domain specialization (see
`ROADMAP_PHYSICS_AI_HUB.md` section "Domain specialization"): 7 new,
literature-grounded benchmark presets in
`pinneapple_physics/pde_environment/presets/astrophysics.py`, spanning
industrial space engineering (satellite orbit propagation, space-debris
conjunction assessment, spacecraft attitude control) and research
astrophysics (stellar structure, dark-matter halo potentials, compressible
hydrodynamics). Each needed a new PDE/ODE residual kind in the compiler
(`pinneapple_physics/pinn_solver/compiler/compile.py`), except
`nfw_dark_matter_potential`, which reuses the existing "poisson" kind.

**Every closed-form solution used for validation was derived or verified
with `sympy` this session before being written into code** (substituted
into its governing ODE/PDE, confirmed exact zero residual) — not
recalled from memory and trusted. This caught one real error: the first
version of the J2 perturbation acceleration (from memory) had the wrong
overall sign; the version actually shipped was derived as -grad(V) of the
cited geopotential and is correct by construction.

`tests/test_astrophysics_validation.py` (Tier B, same method as
`test_manufactured_solutions.py`): plug the exact solution into the
compiled residual with no training, assert ~0; plug a deliberately wrong
solution in, assert clearly nonzero. **16/16 pass** (8 presets/kinds x
exact+wrong, except `satellite_j2_perturbation`, which has no closed-form
trajectory — see below):

| Preset | Kind | Exact solution used | Result |
|---|---|---|---|
| `kepler_two_body_orbit` | `kepler_two_body_orbit` | Kepler's equation (Newton-Raphson, differentiable) | ✅ residual ~1e-19 |
| `space_debris_cw_relative_motion` | `space_debris_cw_relative_motion` | Clohessy-Wiltshire closed form | ✅ residual ~0 |
| `satellite_j2_perturbation` | `satellite_j2_perturbation` | no closed form; checked instead: J2=0 reduces exactly to two-body (res ~1e-19), J2=Earth's value measurably perturbs it (res ~3e-11) | ✅ both directions confirmed |
| `spacecraft_attitude_euler_rotation` | `spacecraft_attitude_euler_rotation` | axisymmetric torque-free precession | ✅ residual ~0 (<1e-8) |
| `lane_emden_polytrope` | `lane_emden_polytrope` | n=0 and n=1 closed forms | ✅ both ~0 (n=0 caught a real bug, see below) |
| `nfw_dark_matter_potential` | `poisson` (existing kind) | NFW closed-form potential | ✅ residual ~0 |
| `sod_shock_tube_astro` | `euler_compressible_1d` | smooth advected-pulse MMS (the real Sod solution is discontinuous, not usable for pointwise autograd MMS) | ✅ residual ~0 (<1e-8) |

**Real bug caught by this process**: the Lane-Emden branch's
`theta_pow_n = sign(theta) * abs(theta)**n` "safe negative-base power"
trick is mathematically wrong for even integer n (it returns -1 instead
of the correct +1 for `theta**0` when theta<0) — the n=0 exact-solution
test failed with residual 0.234 instead of ~0 before the fix. Corrected
to use `theta ** int(n)` directly whenever n is an integer (0, 1, 5 all
are), falling back to the sign-preserving regularization only for
genuinely non-integer n. This is exactly the kind of defect Tier B is
for: the code ran without error before the fix (Tier A would have missed
it entirely) but computed the wrong physics.

**Also fixed as a byproduct**: two pre-existing Category-1 "Unsupported
PDE kind" gaps from the Tier A audit below, `sir_ode` and
`pk_two_compartment_ode`, were fixed using the same
compiler-branch pattern built for the new astrophysics ODE kinds (both
now compile and run; Tier A failure count went from 62/137 to 60/137 as a
direct result, confirmed by re-running the full suite).

**Known gaps from the first pass, closed in a follow-up pass** (below):
`satellite_j2_perturbation`'s secular drift-rate formulas and
`lane_emden_polytrope`'s astrophysically-standard n=1.5/n=3 both lacked
any check beyond the instantaneous-residual level. Both are now
independently validated (see "Follow-up pass" below). The CGNS/Exodus/
Fluent/Abaqus readers' "not validated against a real writer's file"
caveat is now closed for three of the four formats (see
"Real-writer validation of the CGNS/Exodus/Fluent/Abaqus readers" below);
unrelated to this specialization either way.

### Real-writer validation of the CGNS/Exodus/Fluent/Abaqus readers

Follow-up to the caveat above: each reader was run against a file from an
independent, real writer (a different program than the one being tested),
not just the self-consistent synthetic files used originally. Full
provenance for every fixture is in `tests/fixtures/cfd_formats/README.md`;
tests are `tests/test_cfd_format_readers.py` (5/5 pass).

- **CGNS: validated for real.** `h5py` was already available, but there is
  no real CGNS-writing tool installable in this sandbox via `pip` (`pycgns`
  needs a C build toolchain this environment doesn't have --
  `ninja`/`hdf5`-dev via `pkg-config` weren't resolvable through pip's
  build isolation). `brew install cgns` *did* work, though (bottled, no
  compilation needed) -- it installs the real CGNS 4.5.2 reference
  library/headers **and** the CGNS project's own CLI tools (`cgnscheck`,
  `cgnslist`, ...). A small C program was written against the real CGNS
  Mid-Level Library API (`cg_open`/`cg_zone_write`/`cg_field_write`/...) to
  produce a file, which `cgnscheck` (the CGNS project's own reference
  validator) confirmed was spec-valid (0 errors, only benign warnings)
  before `cgns_reader.py` was pointed at it. **Result: read correctly on
  the first attempt** -- coordinates and both vertex fields
  (`Temperature`, `Pressure`) matched their known-exact values exactly. No
  reader bug found. Note in passing: meshio's own built-in `.cgns` writer
  was tried first and rejected as invalid evidence -- it omits the
  CGNS-mandated `label`/`type` HDF5 attributes entirely (its own source has
  a literal `# TODO something is missing here`), so it does not produce a
  spec-compliant CGNS file; see the README for the full finding.
- **Exodus II: validated for real, one real bug found and fixed.**
  `meshio` (already a dependency elsewhere in this repo) writes genuine
  Exodus via the real `netCDF4` package. Its default output is
  NetCDF-4/HDF5-based; forcing `netCDF4.Dataset(..., format="NETCDF3_CLASSIC")`
  produced the classic-NetCDF variant `exodus_reader.py` targets. **Read
  correctly** (coordinates + both fields matched exactly) once written.
  The real bug: pointing the reader at the *other* real, valid Exodus file
  meshio produces (NetCDF-4/HDF5-based, no format override) raised a raw,
  confusing `scipy` `TypeError` ("not a valid NetCDF 3 file") instead of
  the clear, documented-as-intentional failure the module's own docstring
  promised ("out of scope... raise on that failure rather than silently
  returning nothing"). Fixed: `read_exodus` now detects the HDF5 magic
  header and the classic-format-only `TypeError` and raises a
  `NotImplementedError` naming the actual problem and the `netCDF4`/
  `h5netcdf` fallback, instead of letting scipy's internal message leak
  through unexplained.
- **Fluent/Gambit mesh: validated for real.** `meshio`'s `ansys` writer
  targets the same TGrid `.msh` format `fluent_mesh_reader.py` implements
  (both cite the same TGrid user-guide appendix); written in ASCII mode
  (`binary=False`) since that's the only encoding the reader supports.
  **Read correctly**, node coordinates matched exactly. No reader bug
  found.
- **Abaqus: partially validated, and the `.odb` half plainly is not
  possible here.** The `.inp` mesh-reading half
  (`read_abaqus_inp_mesh`) was validated against a real `meshio`-written
  `.inp` deck (`*NODE`/`*ELEMENT, TYPE=C3D4`) -- read correctly, node
  coordinates and connectivity matched exactly, no bug found. The `.odb`
  results bridge (`export_odb_fields`, which shells out to a real Abaqus's
  own `abaqus python` interpreter) genuinely cannot be exercised here:
  `which abaqus` and a filesystem search both confirm no Abaqus
  installation exists on this machine (expensive licensed commercial FEA
  software, as expected) -- and faking an `.odb` file by hand would defeat
  the entire point of validating against a real writer, so no attempt was
  made. This half of the caveat remains genuinely open and needs a machine
  with a licensed Abaqus install to close.

### Follow-up pass: end-to-end training + independent numerical validation

Two things the first pass explicitly flagged as missing, both addressed:

**1. Actually training networks end-to-end** (the first pass only proved
the residual/physics implementation is correct, never trained a network
and checked its own output against the truth):

- `examples/pde_environment/05_kepler_orbit_validation.py`: first attempt
  (physics residual + IC only) converged PDE+IC loss to ~0.001 — looking
  done — while position RMSE was **~104% of the semi-major axis**
  (completely wrong trajectory shape). This is a real, reproduced PINN
  pure-IVP failure mode, not a footnote: an IC pins the solution at one
  point only, so small residual+IC loss does not imply the network found
  the correct global trajectory among the many that locally satisfy it.
  Fixed by adding 15 sparse "tracking-data" points from the exact
  solution as a `DataConstraint` (which is also an honest reframing of
  the preset's real industrial use case: orbit determination genuinely
  is fitting dynamics to sparse tracking observations). Final result
  (Adam + cosine LR decay + grad clipping, 3000 epochs, ~45s CPU):
  **1.85% position RMSE, 2.65% velocity RMSE**; conserved quantities not
  directly supervised (specific energy, angular momentum) matched exact
  values to ~1.9%/~0.7%.
- `examples/pde_environment/06_space_debris_cw_validation.py`: repeated
  the same experiment for `space_debris_cw_relative_motion` (a *linear*
  ODE, unlike Kepler's nonlinear one) — the same pure-IVP collapse
  reproduced again (loss ~1e-12, RMSE ~360%), confirming the failure mode
  is about IVP structure, not nonlinearity. A second, independent
  pitfall was found while fixing it: Hill's along-track coordinate y(t)
  has a genuine secular (linearly-growing) term reaching ~-12 km over one
  period while x/z stay within ±1.5 km — the single shared
  position-scale that worked for Kepler badly under-scales y here and
  stalls convergence at ~17% RMSE; per-axis scaling matched to each
  coordinate's actual range fixed it. Final result: **3.00% position
  RMSE**.

**2. Independent numerical validation of the two previously-open gaps**
(both via `scipy.integrate.solve_ivp`, reimplemented independently of
`compile.py` — checking the physics, not the compiler code against
itself):

- `tests/test_j2_secular_validation.py`: integrates the J2-perturbed
  equations of motion over 800/400 orbits, fits the osculating
  RAAN/argument-of-perigee secular trend, compares to Vallado's
  literature formulas. **Nodal regression: 0.45% agreement. Apsidal
  precession: 0.53% agreement.** (Caveat documented in the test itself:
  naively checking apsidal precession near the ~63.435° critical
  inclination gives a spurious "45% error" from the literature formula's
  own near-zero denominator there, not a real discrepancy — the test
  deliberately uses 30° instead.)
- `tests/test_lane_emden_numerical_validation.py`: integrates the
  Lane-Emden equation for n=1.5 (white dwarf) and n=3 (Eddington standard
  model), finds the first zero crossing ξ₁, compares to published
  textbook tables (Chandrasekhar 1939). **n=1.5: 0.0001% agreement. n=3:
  0.00002% agreement.** (The integrator is first sanity-checked against
  the n=0/n=1 closed forms, matching to <1e-6, before being trusted for
  the no-closed-form cases.)

Both test files pass (2/2 and 4/4 respectively), run in a few seconds
each, and are part of the regular suite going forward.

---

## Zero — the pre-existing test suite (`tests/`, excluding the two new
Tier A/B files below): 4 collection-blocking bugs found and fixed, then
**0 failures across the entire suite**

Running `pytest tests/` at the start of this pass didn't run a single
test — it aborted immediately with 4 `ModuleNotFoundError`s at collection
time (pytest aborts collecting the *whole* directory tree on any
module-level import error, so these 4 broken files were silently blocking
every other pre-existing test in the repository, tested or not, from ever
running). All four turned out to be the same underlying pattern —
compatibility shim packages (`pinneapple_train`, `pinneapple_solvers`,
`pinneapple_models`) whose `__init__.py` only re-exports symbols at the
flat top level, while a large number of call sites across the repo
(examples, the library's own internal NLP-to-PDE agent knowledge base,
`pinneapple_tools.benchmark_suite`, and these tests) import from them as
genuine *submodules* (`pinneapple_train.trainer`, `.losses`, `.metrics`;
`pinneapple_solvers.fft`; `pinneapple_models.registry`) that never
existed as real files:

| Broken import | Real location | Blast radius found |
|---|---|---|
| `pinneapple_pinn.factory.pinn_factory` (package doesn't exist at all) | `pinneapple_physics.pinn_solver.factory.pinn_factory` | 1 test file (fixed the import directly — no established shim convention to extend for this one) |
| `pinneapple_train.trainer` / `.losses` / `.metrics` | `pinneapple_neural.trainer.{trainer,losses,metrics}` | 7 example scripts, `pinneapple_tools/benchmark_suite/{timeseries_pipeline,physics_pipeline}.py`, and **the library's own `pinneapple_problemdesign` NLP-to-PDE agent knowledge base** (`knowledge/{pinneapple_capabilities,mapping}.py`) documented this exact broken API — added real `pinneapple_train/{trainer,losses,metrics}.py` submodules |
| `pinneapple_solvers.fft.FFTSolver` | `pinneapple_simulation.numerical_solvers.fft.FFTSolver` | plus a second, independent bug in the same file: `pinneapple_solvers/__init__.py`'s own top-level re-export imported a nonexistent `FFTProcessor` name inside a bare `try/except`, silently leaving it `None` forever with no error ever surfacing — fixed both |
| `pinneapple_models.registry.ModelRegistry` | `pinneapple_neural.architectures.registry.ModelRegistry` | added `pinneapple_models/registry.py` |

Once collection was unblocked, running the full suite surfaced these
further real, independent bugs (all fixed, not just documented):

- **`pinneapple_systems/process_components/real_gas_eos.py`**: `CP.PT_INPUTS`/`CP.HmassP_INPUTS`/`CP.PSmass_INPUTS`
  are evaluated as plain function *arguments* at each `state_from_*` call
  site, i.e. before the function's own internal `_COOLPROP_AVAILABLE`
  check ever runs — so on any machine without CoolProp installed, all 12
  `test_process_components.py` tests failed with a bare, confusing
  `NameError: name 'CP' is not defined` instead of the clear
  `ImportError` the code was already set up to raise internally. Added a
  `_require_coolprop()` guard at the top of all three entry points, added
  `pytest.importorskip("CoolProp")` to the test module (so it skips
  cleanly rather than fails on an environment without it, the same fix
  applied to `tests/pinneapple_geom/test_mesh_ops.py`'s bare `import
  trimesh`), and added `CoolProp` to `pyproject.toml`'s new `process`
  extra (it wasn't listed anywhere before).
- **`pinneapple_neural/trainer/trainer.py`**: `Trainer.fit()` never
  returned per-epoch history at all (only the final `best_val`/
  `best_path`), despite `RunLogger` already computing exactly that data
  every epoch and simply not passing it back to the caller. Added a
  `"history"` key (list of per-epoch `{"epoch", "train_total",
  "val_total", ...metrics}` dicts) to the returned dict.
- **`pinneapple_app/backend/routers/experiments.py`**: the experiment
  progress callback unconditionally read `ev["epoch"]`/`ev["total_epochs"]`/
  `ev["model"]`, but `ExperimentRunner` sends two other, structurally
  different event shapes through the *same* callback for its auto-fix
  advisor loop (`type="advisor"`/`type="retrain"` events, which carry no
  `epoch`/`total_epochs` key at all) — so **any experiment run whose
  auto-fix advisor loop fired at all crashed the entire experiment** with
  `KeyError('epoch')`, reported to the user only as a bare `"failed"`
  status with message `"'epoch'"`. This is arguably the most
  user-visible bug found this session (a core, advertised feature of the
  app silently ate every experiment that used it) and was **not caught by
  any existing test** — `tests/test_app_backend.py`'s
  `TestExperiments` class existed and exercises exactly this path, but
  had never been run successfully before (it couldn't get past the
  collection-blocking bugs above to even execute). Fixed by branching on
  `ev.get("type")` before assuming the training-event shape.

**Net result**: `pytest tests/ --ignore=tests/test_full_library_matrix.py
--ignore=tests/test_manufactured_solutions.py` went from *4 collection
errors, 0 tests run* to **0 failures**, running the entire pre-existing
suite (all of it that doesn't need an unavailable optional dependency,
which now skip cleanly instead of erroring).

**Later re-run note**: in a later re-run this session (while validating
the astrophysics additions above), `tests/test_app_backend.py`'s 34
tests could not be re-verified in this particular environment because
`httpx` (required by `starlette.testclient.TestClient`) was not
installed and this session's sandbox blocked a global `pip install`.
This is an environment gap, not a code regression -- `httpx` was missing
from `pyproject.toml`'s `dev` extra entirely (now added) despite being a
hard requirement for this test file, so `pip install -e .[dev]` would
not have caught it either. The rest of the suite (everything except this
one file) was re-run in full and showed zero regressions from this
session's compile.py changes (60/137 Tier-A failures, down from 62, the
2-failure improvement being the `sir_ode`/`pk_two_compartment_ode` fixes
documented in the astrophysics section above -- not a new problem).

---

## Tier A / Tier B (this session's own new tests)

**Totals**: 90 registered architectures (`ModelRegistry.list()`) + 47
registered PDE presets (`list_presets()`) = 137 Tier-A checks. **62
failed** (45%), the rest passed or were skipped as not applicable to this
generic smoke test's assumptions (see categories below).

This is not a claim that 45% of the library is "broken" -- most of the 62
failures fall into two large, distinct categories, only one of which is a
real defect. Read the categories, not just the count.

## Category 1 — real defect: presets registered but not compilable at all

**~35 of the 62 failures.** These presets are discoverable via
`list_presets()` and `get_preset(name)` succeeds, but
`compile_problem(spec)`'s `pde_kind` dispatch (`pinneapple_physics
.pinn_solver.compiler.compile.py`) has no branch for their `spec.pde.kind`
at all — `solve_pde` (and therefore `pipeline()`, and therefore any normal
usage) raises `ValueError: Unsupported PDE kind: <kind>` immediately.
**This is the single largest real gap this audit found.** Preset names
observed with this exact failure (`pde.kind` values in parentheses; the
underlying kind, not always identical to the preset name):

```
aircraft_wing_aerodynamics, aircraft_wing_structural,
axial_compressor_cascade_2d, axial_compressor_meanline,
axial_compressor_stage_3d, bekker_wong_surrogate_2d, black_scholes_1d,
car_brake_thermal, car_external_aero, car_suspension_fatigue,
climate_atmosphere_2d, climate_ocean_gyre (stommel_gyre_2d),
cpu_heatsink_thermal (heat_equation_steady), crystal_phonon
(phonon_bte_1d_gray), datacenter_airflow_2d, datacenter_cfd_3d,
datacenter_server_thermal, drug_diffusion_tissue, fan_cooler_cfd,
furnace_combustion_zone, heston_pde_2d, industrial_furnace_thermal
(heat_equation_steady), material_fracture_2d (phase_field_fracture_2d),
opinion_dynamics_2d, pcb_thermal (heat_equation_steady), pk_two_compartment
(pk_two_compartment_ode), plane_strain_2d, plane_stress_2d
(linear_elasticity_plane_stress), refractory_lining
(heat_equation_steady), rocket_nozzle_cfd, rocket_structural, sir_epidemic
(sir_ode), thermoelasticity_2d, threaded_coupling_tc50_rotating,
von_mises_2d (linear_elasticity_plane_stress)
```

A second, smaller variant of the same defect: `channel_flow_3d`,
`lid_driven_cavity_3d`, `pipe_flow_3d` are **steady** presets (no `t`
coordinate by design — see the architecture report this project's
`splash-pinneapple` pipeline was built from), but
`navier_stokes_incompressible`'s branch in `compile.py` unconditionally
raises `ValueError: Navier–Stokes expects time coord 't'.` for any spec
without one. These three are registered, physically sensible, steady
Navier-Stokes presets that `solve_pde` cannot run at all.

**Recommendation** (as originally written): tackle in the order real
users would hit them: thermal (`heat_equation_steady` covers several
presets in one fix), structural
(`linear_elasticity_plane_stress`/`plane_strain` covers several), then
the rest.

**Follow-up pass acted on that recommendation.** Added real residual
implementations (`pinneapple_physics/pinn_solver/compiler/compile.py`),
each derived and independently sanity-checked before being written
(details and the exact `sympy`/manufactured-solution verification are in
`tests/test_manufactured_solutions.py`, extended this pass):

| New/fixed `pde_kind` | Presets closed |
|---|---|
| `navier_stokes_incompressible` (removed the unconditional "must have coord 't'" check — steady presets simply drop the du/dt term) | `channel_flow_3d`, `lid_driven_cavity_3d`, `pipe_flow_3d` |
| `heat_equation_steady` (new) — `k*laplacian(T) = -q`, isotropic | `cpu_heatsink_thermal`, `industrial_furnace_thermal`, `datacenter_server_thermal` |
| `heat_equation_steady_anisotropic` (new) — `sum_i k_i*d^2T/dx_i^2 = -q` | `pcb_thermal` |
| `heat_equation_steady_multilayer` (new) — same as `heat_equation_steady`, reading the preset's own precomputed `k_eff` | `refractory_lining` |
| `linear_elasticity_plane_strain` (new alias — mathematically identical to the existing `linear_elasticity` branch at 2 spatial dims: plane strain literally IS "zero out-of-plane strain, full 3D constitutive relation restricted to 2D") | `plane_strain_2d` |
| `linear_elasticity_plane_stress` (new) — same branch, with the standard reduced Lamé parameter `lambda* = 2*lambda*mu/(lambda+2*mu)` (Timoshenko & Goodier) in place of the raw 3D lambda, derived and `sympy`-verified this session by eliminating `eps_zz` from the full 3D relation under the `sigma_zz=0` plane-stress constraint | `plane_stress_2d`, `von_mises_2d`, and — found as a bonus while re-running Tier A, not originally on this list — `aircraft_wing_structural`, `car_suspension_fatigue`, `rocket_structural` (all three reuse the same kind) |
| `thermoelasticity_2d` (new) — steady heat (`laplacian(T)=0`) coupled to plane-stress elasticity with an isotropic thermal strain `eps_th = alpha_T*T*I` subtracted from the constitutive relation | `thermoelasticity_2d` |

**Net result**: Tier A failures went from 60/137 (after the earlier
`sir_ode`/`pk_two_compartment_ode` fix) to **45/137** — 15 more closed in
this pass (12 targeted + 3 bonus), confirmed by re-running the full Tier
A matrix. All new kinds have `test_manufactured_solutions.py` MMS
coverage (exact-solution-gives-~0 / wrong-solution-gives-nonzero, same
method as the original Laplace test) except `thermoelasticity_2d`'s
coupled term specifically, which this codebase's `laplacian()` helper
can't cleanly MMS-test with a spatially-constant temperature field (a
second-derivative-of-a-constant autograd-graph-connectivity limitation,
not a defect in the new code) — tracked as a known gap, not silently
skipped.

**Third batch, same pass**: while triaging `drug_diffusion_tissue`
(kind `"reaction_diffusion_2d"`), found this was never actually a
missing-physics gap at all — a naming collision. An existing
`"reaction_diffusion"` kind already existed in `compile.py`, but it
implements two-species nonlinear Gray-Scott kinetics (fields `u`, `v`);
`drug_diffusion_tissue` needs a completely different, much simpler
single-species linear model (`dC/dt = D*laplacian(C) - lambda*C`, field
`C`). Added `reaction_diffusion_2d` as its own kind rather than
overloading the existing one. Also added `black_scholes_1d` (the real
Black, Scholes 1973 PDE) and `heston_pde_2d` (Heston 1993, with the
correlated mixed derivative term). Both finance presets use `tau`
(time-to-expiry), not `t`, as their time-like coordinate — same
`has_t`/`t_index`-independent pattern as `lane_emden_polytrope`'s `xi`.

`black_scholes_1d`'s MMS test uses the REAL closed-form Black-Scholes
European call price (not an arbitrary manufactured function) — verified
with `sympy` to solve the PDE exactly — which is a stronger check than
usual: it proves the residual is correct against the actual textbook
solution practitioners would compare it to, not just some function that
happens to satisfy the equation. `heston_pde_2d` has no simple closed
form (real Heston pricing needs the Fourier/characteristic-function
method), so it has Tier A coverage (runs, finite loss) but no Tier B MMS
test this session — a known, stated gap, not a defect.

**Net result of this third batch**: Tier A failures **45/137 → 42/137**.
Tests: `test_manufactured_solutions.py` now 14/14 passing (4 new: 2 for
`reaction_diffusion_2d`, 2 for `black_scholes_1d`).

**Fourth batch**: two more naming collisions found the same way (a
preset's `pde.kind` string pointing at physics that already existed
under a *different* kind string) plus one genuinely new coupled-physics
kind:

- `incompressible_navier_stokes_2d` (`aircraft_wing_aerodynamics`,
  `car_external_aero`) — identical physics to the already-implemented
  `navier_stokes_incompressible` at 2 spatial dims (same fields `u,v,p`,
  same params `{nu, Re}`). Aliased into the existing branch.
- `heat_equation_transient` (`car_brake_thermal`) — NOT a pure alias:
  this preset's coords are cylindrical `(r, z, t)` (a brake disc is
  genuinely axisymmetric), so the plain Cartesian `laplacian()` used by
  `heat_equation` would silently give the wrong answer (missing the
  standard `(1/r)*dT/dr` term) — confirmed with `sympy`: `T=ln(r)` is
  exactly harmonic in the axisymmetric sense but has a nonzero Cartesian
  Laplacian (`-1/r²`). Implemented as its own kind with the correct
  axisymmetric term.
- `incompressible_navier_stokes_energy_2d` / `_3d` /
  `navier_stokes_energy_2d` (`datacenter_airflow_2d`, `datacenter_cfd_3d`,
  `furnace_combustion_zone`) — three more distinct kind strings that all
  collapsed onto ONE new implementation: steady incompressible NS coupled
  one-way to a steady advection-diffusion energy equation, with thermal
  diffusivity assumed from a fixed air Prandtl number (Pr=0.71, since none
  of the three presets supply thermal conductivity directly) and an
  optional volumetric heat source term (`furnace_combustion_zone`'s
  combustion heat release).

MMS coverage added for the two new formulations (the NS-2D alias reuses
the existing, already-tested `navier_stokes_incompressible` coverage, so
needed no new test): `heat_equation_transient` verified against
`T=ln(r)` (axisymmetric-harmonic, `sympy`-confirmed); the NS+energy
coupling verified against a genuine potential-flow field
(`u=x²-y², v=-2xy`, velocity potential `x³/3-xy²`) whose stream function
(`x²y-y³/3`) is by construction both the exact energy-equation solution
(constant along streamlines) and independently harmonic — momentum,
continuity, AND energy all satisfied by one `sympy`-verified manufactured
solution simultaneously.

**Net result of this fourth batch**: Tier A failures **42/137 → 36/137**.
`test_manufactured_solutions.py`: 18/18 passing.

**Fifth batch**: climate and social-dynamics presets.

- `shallow_water_2d` (`climate_atmosphere_2d`) — genuinely NOT the same
  as the existing `shallow_water` kind (which uses conservative
  variables `h,hu,hv` and has no rotation term): this is the rotating
  shallow-water system in PRIMITIVE variables `h,u,v` with an explicit
  Coriolis term (f-plane approximation), a real, different physics
  formulation, not a naming collision. Verified with `sympy` against a
  steady geostrophic-balance exact solution (`u=U0` constant, `v=0`,
  `h` linear in `y` with slope `-f*U0/g`) — the textbook balance between
  Coriolis deflection and pressure-gradient force, satisfying continuity
  and both momentum components exactly.
- `stommel_gyre_2d` (`climate_ocean_gyre`) — Stommel's (1948) wind-driven
  barotropic streamfunction model. The preset's own docstring documents a
  default wind-stress-curl forcing formula, but doesn't pass the basin
  width `W` it depends on as a PDE param — implemented reading the
  forcing from `ctx["source_fn"]` (same convention as `poisson`/
  `heat_equation` elsewhere in this compiler), defaulting to zero
  (unforced decay) when not supplied. Verified with `sympy` against
  `psi = exp(m*x)*sin(k*y)`, `m` the positive root of a quadratic derived
  directly from the PDE — an honest closed-form solution, not a
  simplification.
- `opinion_dynamics_2d` — a genuinely nonlinear (cubic reaction term)
  2D PDE; a real closed-form manufactured solution was not derived this
  session (attempts using constant, purely-1D, or quadratic-in-space
  ansätze either hit this codebase's `laplacian()`-helper limitation
  with spatially-constant fields, or couldn't satisfy the cubic term
  in more than one point) — Tier A (compiles and runs) only, a stated
  gap not a silent one.

**Net result of this fifth batch**: Tier A failures **36/137 → 33/137**.
`test_manufactured_solutions.py`: 22/22 passing (4 new, for
`shallow_water_2d` and `stommel_gyre_2d`; `opinion_dynamics_2d` has no
MMS test, per the gap noted above).

**Sixth batch**: `axisymmetric_linear_elasticity_torsion`
(`threaded_coupling_tc50_rotating`) — an axisymmetric threaded-coupling
problem where the meridional displacements (`u_r`, `u_z`) are IDENTICAL
physics to the existing `axisymmetric_linear_elasticity` kind, but the
preset also needs a torsional displacement `u_θ`, which decouples from
the meridional problem in linear elasticity (a standard result, and the
preset's own docstring/meta already state the exact torsional Navier
equation to use). Implemented as its own kind combining both. Verified
with `sympy` against `u_θ = B/r` (the axisymmetric analog of a 2D
irrotational-vortex 1/r falloff, genuinely curved unlike the also-valid
but trivial rigid-rotation solution `u_θ = A*r`).

**Net result of this sixth batch**: Tier A failures **33/137 → 32/137**.
`test_manufactured_solutions.py`: 24/24 passing.

**Seventh batch**: `incompressible_navier_stokes_rotating_frame`
(`fan_cooler_cfd`) — steady incompressible NS in a frame rotating at
angular velocity omega about the z-axis, adding the standard Coriolis
(`2*Omega x u`) and centrifugal (`-omega^2*r`) terms to the existing
`navier_stokes_incompressible` momentum equation. Verified with `sympy`
against the textbook solid-body-rotation solution: zero relative velocity
(`u=v=0`) balanced entirely by a hydrostatic-style pressure field
`p=0.5*omega^2*(x^2+y^2)`.

**Net result of this seventh batch**: Tier A failures **32/137 → 31/137**.
`test_manufactured_solutions.py`: 26/26 passing.

**Eighth batch**: two of the axial-compressor presets, tackled with the
turbomachinery preset's own documented equations doing most of the work:

- `compressor_meanline_1d` (`axial_compressor_meanline`) — the preset's
  own docstring states the exact 3 governing equations (stage-averaged
  energy, continuity, ideal-gas state) for 4 of its 5 fields
  (`T_t, p_t, rho, u`); implemented exactly as documented. The 5th field,
  `c_theta` (tangential velocity), has NO governing equation or boundary
  condition anywhere in the preset itself — rather than invent an
  unstated Euler-turbine-work/swirl equation for it, it was left with no
  PDE residual, matching exactly what the preset specifies rather than
  adding unstated physics. Verified with a real (non-trivial, `sympy`-
  checkable by hand) manufactured solution: `T_t` linear in `s`, `u` and
  `rho` constant, `p_t` from the ideal-gas relation — satisfies all
  3 documented equations simultaneously.
- `compressible_euler_2d` (`axial_compressor_cascade_2d`) — genuinely
  different from the existing `euler_compressible` kind (primitive
  variables `rho,u,v,p,T` here vs. conservative `rho,rho_u,rho_v,E`
  there, and steady here vs. unsteady there). Implemented by building
  the conservative fluxes algebraically from the primitive fields and
  reusing the EXACT SAME flux-divergence formulas as the already-Tier-B-
  verified `euler_compressible` branch (steady, so its time-derivative
  terms are simply dropped), plus the ideal-gas state equation (needed
  since `p` is here an independent field, not derived from `E`).
  **Honest confidence note**: unlike the other new kinds this session,
  this one does NOT have an independent nonlinear closed-form MMS test —
  genuine 2D nonlinear compressible-Euler exact solutions are hard to
  construct quickly (verified by hand: forcing spatially-constant
  velocity/density/pressure via momentum+energy also forces temperature
  constant, i.e. uniform flow is the only 1D-reducible self-consistent
  case without a driving mechanism like an area change — too weak a
  check to claim as verification, since a flux formula with a sign error
  would still trivially satisfy it). Correctness here is inherited from
  the reused, already-verified flux formulas, not independently
  re-derived — Tier A (compiles, runs) confirmed; treat with
  correspondingly moderate confidence.

**Net result of this eighth batch**: Tier A failures **31/137 → 29/137**.
`test_manufactured_solutions.py`: 28/28 passing (2 new, for
`compressor_meanline_1d`; `compressible_euler_2d` intentionally has none,
per the honesty note above).

**Ninth batch**: `phonon_bte_1d_gray` (`crystal_phonon`) — the preset's
own docstring gives the exact gray-medium (Callaway model) phonon
Boltzmann transport equation, a genuinely tractable linear PDE
(advection + relaxation + diffusion). Two parameters the equation needs
(`Cv`, `T_eq`) weren't in the preset's own PDE params — `Cv` defaults to
1 (consistent with the preset's own `ScaleSpec(alpha=k)`, which already
treats `k` as if it were the diffusivity directly), and `T_eq` was
**added to the preset itself** (a small, well-justified fix, not a
compiler-side guess) to match the preset's own initial-condition value
`0.5*(T_hot+T_cold)`, which it was already using without exposing to the
PDE params. Verified with a genuine, nontrivial closed-form solution
derived (not recalled) this session: a decaying traveling wave
`T = T_eq + A*exp(-t/tau_d)*sin(k_wave*x - omega_r*t)` solves the
equation exactly when `omega_r = k_wave*vg` (propagates at the group
velocity) and `tau_d = tau/(1+alpha*k_wave^2*tau)` (a wavenumber-
dependent decay time) — both relations solved for and confirmed with
`sympy`.

**Real numerical-conditioning finding surfaced while building this
test**: at `crystal_phonon`'s own literal SI-unit default parameters
(`vg~3000 m/s`, `tau~1e-12 s`), the wavenumber/decay-time relation spans
10+ orders of magnitude, and the compiled residual (like any autograd
second-derivative PDE) loses enough float32 precision purely from that
scale disparity that a naive test at those exact defaults gave a
misleadingly large "residual" for the EXACT solution (~7e13) that looked
like a bug at first. Switching to float64 confirmed it wasn't: residual
~3e-4 against terms of order ~1e16 (relatively ~1e-20, clean). This is a
real, practical finding for anyone training this preset at its literal
default values (expect to need nondimensionalization or float64), not a
defect in the new code — documented in the test itself so it isn't
mistaken for one later.

**Net result of this ninth batch**: Tier A failures **29/137 → 28/137**.
`test_manufactured_solutions.py`: 30/30 passing.

**Tenth batch**: the two remaining gaps that fit this compiler's
stateless-equality-residual design (`bekker_wong_terramechanics`
genuinely doesn't, see below).

- `phase_field_fracture_2d` (`material_fracture_2d`) — the ORIGINAL
  Bourdin, Francfort & Marigo (2000) AT-2 phase-field fracture model
  (exactly what this preset's own meta cites — NOT the later Miehe 2010
  tension/compression spectral split, which postdates that citation):
  degraded elasticity `div[(1-phi)^2 * C:eps(u)] = 0` coupled to a
  phase-field equation driven by the (undegraded) elastic strain-energy
  density. One deliberate, documented simplification versus a full
  quasi-static solver: the crack-driving-force `H` is the
  **instantaneous** strain energy, not history-max(energy) over the
  loading path — this compiler's residual is a stateless function of
  current field values with nothing to track a running maximum against
  across training/load steps. Exact for monotonic loading (this
  preset's own boundary conditions describe exactly that: fixed bottom,
  prescribed top displacement, no unloading), documented in the code so
  it isn't mistaken for an oversight. No independent nonlinear MMS this
  session (constructing one requires solving a fully-coupled nonlinear
  system for a matching displacement+damage field pair, more derivation
  time than available) — Tier A confirmed, moderate confidence,
  consistent with `compressible_euler_2d`'s treatment above.
- `compressible_euler_axisymmetric` (`rocket_nozzle_cfd`) — steady
  axisymmetric (no swirl) compressible Euler, cylindrical `(r,z)`.
  Derived directly this session (not recalled) from the general 3D
  cylindrical Euler equations with `v_theta=0`: continuity and energy
  use the plain axisymmetric divergence `(1/r)*d(r*F_r)/dr + d(F_z)/dz`,
  but the r-momentum component needs an extra `-p/r` correction term —
  verified algebraically: writing its r-flux as `r*(rho*v^2+p)` and
  dividing by `r` (matching the scalar pattern) adds a spurious extra
  `p/r`, since `(1/r)*d(r*p)/dr = dp/dr + p/r`, not just `dp/dr`;
  subtracting `p/r` back off recovers the correct non-conservative
  cylindrical momentum equation exactly. Passes a real (if weak) sanity
  check: uniform axial flow (`v=0`, `rho,u,p,T` all constant) gives
  exactly zero residual, as it must. No independent nonlinear MMS this
  session (same difficulty as the 2D cascade case) — Tier A confirmed,
  moderate confidence.

**Net result of this tenth batch**: Tier A failures **28/137 → 26/137**.

**Eleventh batch**: `compressible_euler_rotating_3d`
(`axial_compressor_stage_3d`) — the highest-complexity kind added this
session: full 3D compressible Euler WITH swirl, cylindrical `(r,theta,z)`,
in a frame rotating at omega about the z-axis. Rather than hand-derive
simplified Coriolis/centrifugal source terms for all 5 coupled equations
(a real risk of a sign/algebra error), this substitutes the absolute
tangential velocity `u_theta_abs = w + omega*r` (`w` = the model's
relative `u_theta` field, matching the convention already used by
`incompressible_navier_stokes_rotating_frame`) directly into the
STANDARD inertial-frame cylindrical Euler equations and replaces the
time derivative with `d/dt -> -omega*d/dtheta` (the exact rule for a
flow field steady in rotating coordinates), letting autograd
differentiate the substituted expression rather than working from a
pre-simplified closed form.

This substitution rule was independently verified this session (derived
with `sympy`, not recalled) against continuity and r-momentum: continuity
has NO omega-dependence at all after simplification (correct — rigid
rotation can't create or destroy mass), and r-momentum produces exactly
the expected explicit `-omega^2*r*rho` (centrifugal) and `-2*omega*rho*w`
(Coriolis) terms. The implementation was then cross-checked NUMERICALLY
(differential testing) against these two independently-derived closed
forms for a generic smooth field: agreement to machine precision in
float64 (~1e-13 relative; a larger, ~1e-3 gap seen at float32 was
confirmed to be precision noise, not a formula difference, by re-running
in float64 and watching it vanish). Since continuity, r-momentum,
theta-momentum, z-momentum, and energy all use the exact same
substitution mechanism, this is real evidence for the shared mechanism,
not just the two hand-checked equations — a genuinely different (not
weaker) style of verification than a closed-form MMS, appropriate for a
system this coupled.

**Net result of this eleventh batch**: Tier A failures **26/137 → 25/137**.
`test_manufactured_solutions.py`: 31/31 passing.

**Session total for this line of work: Tier A failures 62/137 → 25/137
(37 closed)**, spanning eleven batches, every fix backed by either a
`sympy`-verified closed-form solution (the large majority), a numerical
cross-implementation check (the rotating 3D case, appropriate given its
coupled complexity), or an honestly-flagged lower-confidence Tier-A-only
treatment where neither was achievable in the time available (both
`compressible_euler_2d` variants, explicitly marked as such rather than
overclaiming).

**Twelfth (final) batch**: `bekker_wong_terramechanics`
(`bekker_wong_surrogate_2d`) — the one kind that genuinely isn't a
differential-equation residual: the preset's own meta describes three
INEQUALITY/monotonicity constraints on a semi-empirical surrogate
mapping `(slip, sinkage) -> (Fx, Fz, My)`, not an equality PDE. Rather
than force it into the equality-residual mold every other kind in this
file uses, this implements the standard soft-constraint technique
(squared-hinge penalties, `relu(violation)`, appended UNSQUARED since
the shared aggregation code downstream already squares every `res_list`
entry — a real, easy-to-miss double-squaring bug avoided by checking the
aggregation code first):

- R2 (`Fx <= c*A + Fz*tan(phi)`): needed a contact-patch area `A`, which
  needed sinkage-to-entry-angle geometry the preset doesn't provide
  directly — used the standard rigid-wheel terramechanics relation
  (Wong, "Theory of Ground Vehicles"): `theta1 = arccos(1 - sinkage/R)`
  (from `sinkage = R*(1-cos(theta1))`, the geometric relation between
  how far a wheel of radius `R` sinks and the angle to its first ground
  contact point), `A = b*R*theta1`. Uses only this preset's own `R_m`/
  `b_m` params and the sinkage coordinate directly — no fabricated
  Bekker pressure-sinkage moduli (`kc`, `kphi`, `n`) or Janosi-Hanamoto
  shear modulus (`K`) that a fuller classical Bekker-Wong force
  integration would need but this preset's params don't supply.
- R3 (`dFx/ds >= 0` for `slip<=0.4`): a first-derivative monotonicity
  penalty, masked to the stated slip range.
- R4 (`My >= R*Fx`): a plain algebraic inequality penalty.
- R1 (`Fx(slip=0)=0`) was a genuine point/initial condition, not an
  interior-residual term — **the preset itself was missing it entirely**
  despite listing it in its own meta (`conditions=()`, empty, before this
  fix); added as a real `InitialCondition` using this compiler's
  existing machinery, verified to select exactly the `slip=0` points and
  target `Fx=0` there.

Verified with a genuinely constraint-respecting solution (small,
monotonically increasing `Fx`, comfortably-large `My`) giving exactly
zero penalty, and a deliberately constraint-violating one (`Fx` far
exceeding the shear-strength bound, `My=0` violating the moment
constraint) giving a clearly large one.

**Net result of this twelfth batch**: Tier A failures **25/137 → 24/137**.
`test_manufactured_solutions.py`: 34/34 passing.

## All 7 originally-identified Category 1 gaps are now closed

**Session total for this line of work: Tier A failures 62/137 → 24/137
(38 closed) across twelve batches.** Every fix is backed by one of: a
`sympy`-verified closed-form solution (the large majority), a numerical
cross-implementation check (the rotating-3D compressible Euler case,
appropriate given its coupled complexity), a satisfying/violating pair
for the one genuinely non-equality kind (Bekker-Wong), or an honestly-
flagged lower-confidence Tier-A-only treatment where none of the above
was achievable in the time available (both `compressible_euler_2d`
variants) — every one of those confidence levels stated explicitly in
this report and in the code itself, not overclaimed.

No Category 1 gaps remain from the original list this audit identified.
The broader Tier A number (24/137) still included Category 2
(shape-mismatch and calling-convention false positives) and Category 3
(missing optional dependencies) entries.

## Finalized: every remaining Tier A failure individually confirmed and closed

The earlier version of this section estimated "~20" Category 2 failures
and left a "handful" unverified with a caveat to "treat them as probably
Category 2, not independently confirmed." That caveat is now resolved:
every one of the 24 remaining Tier A failures was individually run,
its exact error message read, and its architecture's source code
checked where the reason wasn't obvious from the error alone (`elm`
specifically — see below). None turned out to be Category 1 (a real,
previously-hidden defect); all 24 fall into one of four precise,
verified reasons:

| Reason | Architectures | Evidence |
|---|---|---|
| Missing optional dependency (`emmiai-noether` not installed) | `noether_abupt`, `noether_aero_abupt`, `noether_aero_transformer`, `noether_aero_transolver`, `noether_aero_upt`, `noether_transformer`, `noether_transolver`, `noether_upt` (8) | `ModuleNotFoundError: No module named 'noether'`, raised at forward() time, not build time — the test's skip check for this only covered the build step before this fix, so these were genuinely showing as `FAILED` until now, not already-handled as the original report implied |
| Sequence/image/minimum-length input shape, not a flat `(N,4)` point cloud | `afno`, `arima`, `conv2d`, `conv3d`, `esn`, `esn_rc`, `koopman`, `transformer`, `havok` (9) | Shape assertions (`conv2d`/`conv3d`: "Expected 4D... input of size: [8,4]"), tuple-unpacking errors from code expecting `(B,T,F) = x.shape` (`arima`/`esn`/`esn_rc`/`koopman`/`transformer`), or a minimum-sequence-length requirement exceeding the test's tiny batch (`havok`: "Need T>=delays. Got T=8, delays=50") |
| Requires extra arguments beyond `forward(x)` (a different, legitimate calling convention, not a bug) | `gno`, `neural_cde`, `ode_rnn` (3) | `TypeError`s naming a genuinely missing argument: `GalerkinNeuralOperator.forward() missing 1 required keyword-only argument: 'coords'`; `NeuralCDE`/`ODERNN.forward() missing 1 required positional argument: 't'` — these architecture families need explicit coordinates/time points by design |
| Closed-form/fit-based model, not gradient-trained (must call `.fit()` on real data first) | `dmd`, `pod`, `hybrid_rbf`, `elm` (4) | `dmd`/`pod`/`hybrid_rbf` raise directly ("DMD not fitted", "POD not fitted", "HybridRBFNetwork.forward() called before fit()"); `elm` is subtler — its forward pass succeeds, but reading `pinneapple_neural/architectures/reservoir_computing/elm.py` shows `self.W_out = nn.Parameter(torch.zeros(...), requires_grad=False)` with the comment `# trained by closed form` and a real `.fit()` method doing ridge regression — genuinely zero trainable parameters until `.fit()` is called, the correct textbook Extreme Learning Machine design, not a bug |

**`tests/test_full_library_matrix.py` was updated to recognize all four
reasons precisely** (previously it only recognized the "sequence/image
shape" and "missing dependency at build time" cases) — each of the 24
now gets a specific, honest skip message naming which of the four
reasons applies and why, instead of either an unexplained `FAILED` or a
generic catch-all skip. Confirmed by re-running the full Tier A suite:
**71 passed, 75 skipped, 0 failed** out of 146 total (137 at the start
of this session's audit — the registry has grown since, mainly the 7
new astrophysics presets — with 62 originally-unexplained failures at
that time).

## Tier B (physics correctness) — passed

`tests/test_manufactured_solutions.py`'s two tests both pass: the
compiled `"laplace"` residual is ~0 (< 1e-8) for the exact harmonic
solution `u = x²-y²`, and clearly nonzero (> 1, in fact ≈16 as the exact
Laplacian predicts) for the non-harmonic `u = x²+y²`. This confirms the
`"laplace"` `pde_kind`'s residual implementation is genuinely correct, not
just "runs" — the one PDE kind this session had time to verify at Tier B
depth. Extending Tier B coverage to more `pde_kind`s (Poisson with a
manufactured source term, Burgers via Cole-Hopf, ...) is listed in
`ROADMAP_PHYSICS_AI_HUB.md` section 1.1 as follow-up work.

## Follow-up pass: 3D `KOmegaSSTResiduals` turbulence closure (P2.4)

`turbulence_presets.py`'s `KOmegaSSTResiduals.__call__` was 2D-only
(`(x,y)->(u,v,p,k,omega)`); `ROADMAP_PHYSICS_AI_HUB.md` section 2.4
called for a genuine 3D generalization, the same shape of upgrade
`WALEResiduals` in the same file already got (2D-helper-style ->
real 3D closure). Done as a dispatch on `x_col.shape[1]`: 2 columns
routes to `_call_2d` (the original code, refactored but numerically
untouched — confirmed by diff: only docstring lines changed inside
it), 3 columns routes to a new `_call_3d` mapping
`(x,y,z)->(u,v,w,p,k,omega)`. The new 3D viscous-stress divergence
(`_viscous_stress_divergence_3d`) is built from full Hessians of
u/v/w via a new `_hessian_components_3d` helper — the same
no-incompressibility-shortcut construction `_viscous_stress_
divergence_2d` already uses, extended with the z-direction cross
terms, since mu_eff varies spatially in both cases. The k-/omega-
equations reuse the existing `_laplacian` unchanged (it already sums
over every column of `x`, so a 3-column `x` "just works") and gain the
3D cross-diffusion `k_x w_x + k_y w_y + k_z w_z`.

**Verification**: the full nonlinear coupled k-omega SST system has no
simple closed-form exact solution, so — same fallback used above for
`compressible_euler_rotating_3d` — this is a cross-implementation
check, not an exact-solution MMS check. A concrete trial field
(trigonometric/polynomial in x, y, z, chosen so every term being
checked is genuinely non-zero, including the cross-diffusion dot
product) was differentiated two independent ways: by `sympy`, working
directly from the equations restated for 3D, and by running the real
`_call_3d` torch/autograd code on an `nn.Module` evaluating the
identical formulas. `F2=0` keeps the eddy-viscosity Bradshaw limiter
on its smooth branch (`nu_t` reduces exactly to `k/omega`, avoiding a
`max()` kink autograd and `sympy` could disagree across); the
realizability and cross-diffusion clips were confirmed numerically
inactive at every sample point before being treated as absent in the
closed form (not assumed). All six residuals (momentum_x/y/z,
continuity, k_eq, omega_eq) agreed to **machine precision in float64**
— max abs diff 4.6e-13 (omega_eq), 3.1e-13 (k_eq), <7e-16 for the
other four — against residual magnitudes of order 1-40. Also checked
in float32 during development: ~8e-6 max abs diff, ordinary float32
roundoff rather than a genuine precision problem (unlike this
session's `phonon_bte_1d_gray` finding, nothing here spans more than
~2 orders of magnitude). A second test supplies a field satisfying
none of the six equations and confirms a clearly nonzero aggregate
residual, matching this file's exact/wrong pair convention.

New tests: `test_audit_physics_k_omega_sst_3d_matches_independent_
closed_form`, `test_audit_physics_k_omega_sst_3d_wrong_solution_gives_
nonzero_residual` in `tests/test_manufactured_solutions.py`
(36/36 passing). `tests/test_full_library_matrix.py` re-run
unaffected: 71 passed, 75 skipped, 0 failed — identical to the count
recorded elsewhere in this report, confirming no regression (this
module isn't part of the `compile_problem`/`pde_kind` registry that
suite audits, so no change there was expected either way).

## What this session fixed vs. what it found but did not fix

Fixed this session (see `ROADMAP_PHYSICS_AI_HUB.md`'s P0 table for the
full list with file locations): `solve_pde()`, `UPDDataset.save(zarr)`,
`Trainer.fit`'s no-grad validation, the `33_rans_turbulence.py` template,
`PeriodicBC` multi-axis chaining, plus new capability additions (WALE LES,
NS body-force hook, `AdaptiveWeights`, stochastic/latent PINN utilities,
binary OpenFOAM + CGNS/Exodus/Fluent/Abaqus readers, model hub,
`pipeline()`, adaptive hyperparameter search, Blender bridge, `pinneapple
_llm`).

**Found but not fixed this session** (Category 1 above): ~20 distinct
`pde_kind`s referenced by registered presets with no `compile_problem`
branch, and the steady-Navier-Stokes gap. This is real, load-bearing
backlog, not a footnote — a preset that cannot be compiled is a preset a
user of `list_presets()`/`pipeline()` will hit and be confused by, and it
is the single biggest reliability gap this audit surfaced.

## `solve_pde()` silently dropped `selector_type="tag"` conditions — now fails loudly, with the tag_masks mechanism itself confirmed real

A downstream-consumer audit (VeriPhysics calling `solve_pde()` on tag-based
presets like `pipe_flow_3d`/`industrial_furnace_thermal` with no `ctx`)
found that `solve_pde()`'s auto-sampler only ever handles conditions whose
`selector_type` is `"all"` or `"callable"` — `"tag"` conditions are
excluded from `auto_conditions` unconditionally, with no error raised.
Since most CFD/industry presets in
`pinneapple_physics/pde_environment/presets/{cfd,structural,industry}.py`
express every boundary condition as `selector_type="tag"`
(`"inlet"`/`"outlet"`/`"wall"`/`"fixed"`/...), calling `solve_pde(get_
preset("pipe_flow_3d"), model)` with no further arguments used to train a
model with **zero boundary conditions enforced** — the PDE-residual loss
still went down, the run "succeeded," and nothing in the returned
`history` distinguished it from a correctly-constrained run. This is
exactly the class of result `PhysicsGuardrail`/trust-score consumers must
never rate as trustworthy without actually knowing it's untrustworthy.

**Before treating this as a bug to patch over, the tag_masks mechanism
itself was verified to work correctly when given real geometry** — it is
a real, load-bearing contract, not a stub. Ran
`examples/pde_environment/04_heat3d_stl_box.py` (a real trimesh box STL ->
`pinneapple_design.geometry.builders.STLDomainBatchBuilder` ->
`compile_problem`) end to end:

```
loss keys: ['pde', 'bc_T_boundary', 'bc_T_inlet_hot', 'total']
total: 186.65467834472656
mesh_info: {'is_watertight': True, 'n_verts': 8, 'n_faces': 12}
tags: {'boundary': 24000, 'inlet': 2110, 'outlet': 2102, 'walls': 19788}
```

and separately, with `inside_mode="bbox"` (avoiding a missing optional
`rtree` dependency for `trimesh_contains`), confirmed the two tag-scoped
Dirichlet losses are genuinely different numbers computed over genuinely
different point sets (`bc_T_boundary=0.0798`, `bc_T_inlet_hot=0.851`, vs.
each other and vs. `pde=0.0029`) — i.e. the per-tag masking is real, not a
coincidental pass-through. A second check built a `ns_incompressible_2d`
batch by hand the same way
`examples/pde_environment/03_ns2d_channel_tags.py` does (manual
inlet/outlet/wall plane masks, no STL) and ran it through `solve_pde()`
itself end to end (5 epochs, loss `98.7 -> 37.3`, monotonically
decreasing) — confirming the *fix* below doesn't just detect the tag
contract, it also lets a caller who does supply real tag geometry train
through `solve_pde()` normally.

**The fix** (`pinneapple_physics/__init__.py`): `solve_pde()` now raises a
new `TagConditionsUnresolved(ValueError)` before starting any training if
*any* condition in `spec.conditions` has `selector_type="tag"` and is not
covered by explicit input (`x_<kind>` passed AND a matching
`mask_<condition_name>` passed via `**cond_masks`) — naming every
uncovered condition, its tag, and exactly which two arguments are needed,
plus a pointer to `STLDomainBatchBuilder`/the two examples above. Nothing
about `"all"`/`"callable"` auto-sampling changed.

**Preset classification, done programmatically** (`list_presets()` +
inspecting `selector_type` per condition on every one of the 65
registered presets — not eyeballed):

| Category | Count | Needs |
|---|---|---|
| `selector_type="tag"` only | 32 | real geometry (STL/mesh via `STLDomainBatchBuilder`, or a hand-built batch) |
| mixed `"tag"` + `"callable"` | 8 | same as above for the tag-scoped conditions; the callable ones still auto-sample |
| `"callable"`/`"all"` only (or no conditions) | 25 | nothing extra — `domain_bounds` alone is enough, unchanged behavior |

Tag-only (32): `aircraft_wing_aerodynamics`, `aircraft_wing_structural`,
`car_external_aero`, `car_suspension_fatigue`, `channel_flow_3d`,
`climate_ocean_gyre`, `cpu_heatsink_thermal`, `darcy_pressure_only_3d`,
`datacenter_airflow_2d`, `datacenter_cfd_3d`, `datacenter_server_thermal`,
`fan_cooler_cfd`, `furnace_combustion_zone`, `helmholtz_acoustics_3d`,
`industrial_furnace_thermal`, `laplace_2d`, `lid_driven_cavity_3d`,
`linear_elasticity_3d`, `linear_elasticity_3d_industry`,
`material_fracture_2d`, `ns_incompressible_2d`, `pcb_thermal`,
`pipe_flow_3d`, `plane_strain_2d`, `plane_stress_2d`, `poisson_2d`,
`refractory_lining`, `rocket_nozzle_cfd`, `rocket_structural`,
`steady_heat_conduction_3d`, `thermoelasticity_2d`, `von_mises_2d`.
Mixed (8): `axial_compressor_cascade_2d`, `car_brake_thermal`,
`climate_atmosphere_2d`, `drug_diffusion_tissue`, `opinion_dynamics_2d`,
`reaction_diffusion_2d`, `transient_heat_3d`, `wave_ultrasound_3d`.
The remaining 25 (`burgers_1d`, `space_debris_cw_relative_motion`,
`black_scholes_1d`, the `threaded_coupling_*`/`axial_compressor_*`
non-tag variants, the astrophysics/finance/biology ODE presets, etc.) are
unaffected — see `TagConditionsUnresolved`'s docstring and `solve_pde()`'s
own docstring for the up-to-date contract statement, since the preset
catalog will keep growing after this report is written.

**Test suite impact, measured before and after, not assumed.** The two
files that call `solve_pde()` directly, run in isolation before touching
anything (`git stash`-equivalent clean checkout of
`pinneapple_physics/__init__.py`):

| File | Before (pass/skip/fail) | After (pass/skip/fail) |
|---|---|---|
| `tests/test_cartesian_breadth.py` | 49 / 21 / 0 (of 70) | 14 / 56 / 0 (of 70) |
| `tests/test_full_library_matrix.py` | 75 / 77 / 7 (of 159) | 42 / 117 / 0 (of 159) |
| Combined | 124 / 98 / 7 (of 229) | 56 / 173 / 0 (of 229) |

The 7 pre-existing failures (`test_full_library_matrix.py`, presets
`darcy_pressure_only_3d`/`helmholtz_acoustics_3d`/
`linear_elasticity_3d_industry`/`reaction_diffusion_2d`/
`steady_heat_conduction_3d`/`transient_heat_3d`/`wave_ultrasound_3d`) were
a `KeyError: 'x'` from `spec.domain_bounds` being entirely absent on those
`presets/industry.py` factories — a separate, pre-existing gap in those
specific presets (out of scope for this fix, left as-is), which the new
check now masks with a clearer, more actionable `TagConditionsUnresolved`
skip instead of a confusing `KeyError` (both are true simultaneously: they
lack `domain_bounds` *and* are 100% tag-based, consistent with being
designed for real-geometry-only use).

Both test files were updated to recognize
`pinneapple_physics.TagConditionsUnresolved` via a new
`_needs_real_geometry_for_tags()` helper (same pattern as the existing
`_is_missing_optional_dep()` etc.) and `pytest.skip()` with the real
reason, instead of either fabricating `tag_masks` to force a pass or
leaving them as unexplained hard failures. The 68 tests that moved from
pass to skip were passing *only* because their tag conditions were being
silently dropped — after the fix they correctly report "this preset needs
real geometry this generic harness doesn't supply," which is the accurate
statement. No `"callable"`/`"all"` preset's pass/fail status changed
(`burgers_1d`, `drug_diffusion_tissue`'s callable half,
`space_debris_cw_relative_motion`, and the Tier A.1 architecture-only
tests are bit-for-bit unaffected).

Full repo-wide `solve_pde()` caller audit (not just the two test files):
`pinneapple_physics/__init__.py`'s own `pipeline()` (only ever called in
its own docstring example with `burgers_1d`, a callable preset — no
change), and `pinneapple_analysis/verification/causal_discrepancy.py`
(already wraps every `solve_pde()` call in a bare `except Exception`,
recording it as `InterventionRunResult(error=...)` rather than crashing —
this fix makes that path correctly report "FAILED to train" for tag-based
presets used without geometry instead of silently reporting a trained-
but-unconstrained result as a success; no dedicated test exercises this
file, so no test-suite delta). No `saas/`, `veriphysics/`, or
`pinneapple_arena` caller of `solve_pde()` exists inside this repo (those
products live in sibling repos, out of scope here).

## Follow-up pass: real analytic geometry for 23 of the 40 tag-based presets (2026-09-17)

The fix above correctly makes 40 of the 65 registered presets (32
tag-only + 8 mixed) unusable via `solve_pde()` until they are given real
geometry. This pass closes that gap for real, for as many of the 40 as
can be done **without fabricating geometry the preset itself doesn't
already specify**: a new preset-by-preset engineering-judgment table,
`pinneapple_physics/pde_environment/presets/tag_geometry.py`
(`TAG_GEOMETRY_FIXTURES`), classifies every one of the 40 as either

- fixable: the preset's own domain is a canonical box/rectangle or
  circular-cross-section cylinder fully described by the preset's own
  `domain_bounds`/params, AND every tag name maps to one specific face of
  that shape without guessing at anything the preset's docstring/comments
  don't already state (every entry in the table cites the exact text the
  mapping is read from) — **23 of 40**, or
- not fixable without fabricating geometry: a real airfoil/car-body/
  furnace-refractory/turbine-blade profile, an internal (non-face) object
  location never given coordinates, an unlocated ambiguous tag (e.g. two
  differently-named-but-otherwise-identical "fixed"/"load" faces with no
  disambiguating text), a domain whose own stated `domain_bounds` don't
  even encode the real shape (`rocket_structural`'s annulus), or a
  separate, unrelated compiler defect that geometry alone can't fix
  (`aircraft_wing_structural` — see below) — **17 of 40**, each with a
  one-line reason in `NOT_FIXABLE_WITHOUT_REAL_GEOMETRY`. For these,
  `TagConditionsUnresolved` continues to fire, correctly.

**Fixed (23)**: `channel_flow_3d`, `climate_atmosphere_2d`,
`climate_ocean_gyre`, `darcy_pressure_only_3d`, `drug_diffusion_tissue`,
`furnace_combustion_zone`, `helmholtz_acoustics_3d`, `laplace_2d`,
`lid_driven_cavity_3d`, `linear_elasticity_3d_industry`,
`material_fracture_2d`, `ns_incompressible_2d`, `opinion_dynamics_2d`,
`pipe_flow_3d`, `plane_strain_2d`, `poisson_2d`, `reaction_diffusion_2d`,
`refractory_lining`, `steady_heat_conduction_3d`, `thermoelasticity_2d`,
`transient_heat_3d`, `von_mises_2d`, `wave_ultrasound_3d`.

**Not fixable without fabricating geometry (17)**, with the specific
reason in each case: `aircraft_wing_aerodynamics` (real airfoil profile),
`aircraft_wing_structural` (geometry IS unambiguous, but the preset's own
Neumann conditions use traction fields `tx`/`ty` that `compile.py`'s
generic Neumann handling can't resolve against the model's actual fields
`ux`/`uy` — confirmed by hand: `KeyError: 'ty'` even with correct geometry
supplied — a separate compiler defect, out of scope for a geometry-only
fix), `axial_compressor_cascade_2d` (curved blade cascade profile),
`car_brake_thermal` (ambiguous friction/cooling-surface assignment to the
disc's faces), `car_external_aero` (bluff-body silhouette),
`car_suspension_fatigue` (wishbone-arm geometry, tags not tied to
specific faces), `cpu_heatsink_thermal` (fin geometry),
`datacenter_airflow_2d`/`datacenter_cfd_3d` (internal rack/CRAC objects,
not box faces), `datacenter_server_thermal` (hotspot zones never given
coordinates), `fan_cooler_cfd` (blade/hub geometry),
`industrial_furnace_thermal` (internal material-layer boundary, needs a
real furnace CAD/mesh), `linear_elasticity_3d`/`plane_stress_2d`
(`"fixed"`/`"load"` are two different, unlocated tags with nothing to
disambiguate which face is which), `pcb_thermal` (component hotspots
given as a power dict, never coordinates), `rocket_nozzle_cfd` (curved
convergent-divergent nozzle contour), `rocket_structural` (the physical
part is an annulus but the preset's own `domain_bounds` is a solid
square that doesn't even encode the hole).

**Verified, not just asserted**: every one of the 17 "not fixable" claims
above was independently confirmed this session by direct inspection of
the preset's source (e.g. `pcb_thermal`'s `q_components` really is a
`{name: watts}` dict with no coordinates anywhere;
`aircraft_wing_structural`'s Neumann conditions really do use `fields=
("ty",)`/`("tx","ty")` while the model's own fields are `(ux, uy)`, and
calling `compile_problem`'s `loss_fn` directly with correct geometry
supplied by hand does raise `KeyError: 'ty'`, confirming the second,
independent defect). Every one of the 23 "fixed" presets was run
end-to-end through `solve_pde()` for real this session (not just
Tier-A "compiles" checked) — see next section.

### New builder: `analytic_domain_batch_builder.py` (mesh-free, exact-coordinate)

The existing `STLDomainBatchBuilder` needs an actual mesh file and always
re-centers the loaded mesh on its bounding-box centroid, which silently
breaks any preset `value_fn` that assumes literal, un-shifted coordinates
(e.g. `channel_flow_3d`'s inlet profile computes `y*(H-y)` assuming
`y in [0, H]`, not a re-centered `[-H/2, H/2]`). For the 23 presets above
— genuinely representable by `spec.domain_bounds` alone — a new
`pinneapple_design/geometry/builders/analytic_domain_batch_builder.py`
samples the literal box/cylinder analytically instead: `sample_box_tag_
batch()` and `sample_cylinder_tag_batch()`. This is still real geometry,
not a mock — the sampled points are genuine interior/boundary points of
the exact solid the preset's own docstring describes, and every
condition's own `mask()`/`values()` is called exactly as
`STLDomainBatchBuilder` does, so nothing about the preset's declared
physics is bypassed.
`pinneapple_physics/pde_environment/presets/tag_geometry.py`'s
`build_tag_batch(name, spec)` is the single entry point that dispatches
each of the 23 fixed presets to the right shape/face mapping; see
`examples/pde_environment/07_tagged_presets_real_geometry.py` for
end-to-end usage through `solve_pde()`.

**A real, independent bug found and fixed while building this**:
`compile.py`'s `loss_fn` uses whatever is in `batch["y_bc"]` (sliced to
each condition's own mask) as the target for **every** condition kind
once `y_bc` is supplied at all — it only calls `cond.values()` itself
when `y_bc` is absent entirely (the `solve_pde()` auto-sampling path).
`STLDomainBatchBuilder._targets_from_conditions` only ever filled `y_bc`
for `kind == "dirichlet"`, silently leaving Neumann/Robin targets as NaN
— a latent gap that happened to never surface in the two existing STL
examples (`04_heat3d_stl_box.py` has no Neumann tag condition;
`03_ns2d_channel_tags.py`'s one Neumann target is genuinely zero, so the
NaN gap was invisible). Confirmed directly (no `trimesh` needed, since
this only touches the post-mask targeting helper): before the fix,
`pipe_flow_3d`'s Neumann `outlet_dp_dn` tag's own field (`p`) stayed NaN
in `y_bc`; after, it correctly reads `0.0` (the real `dp/dn=0` target).
Fixed in both builders (the new analytic one shipped with the fix
already applied; `STLDomainBatchBuilder` patched to match).

**Per-tag residual/loss evidence, not just "it ran"**: every one of the
23 fixed presets was run through `compile_problem`'s `loss_fn` directly
(random-init model, real sampled geometry) and confirmed to produce
**distinct, non-degenerate per-tag losses** — e.g.
`pipe_flow_3d`: `{pde: 3.45, bc_inlet: 0.49, bc_outlet_dp_dn: 1.75,
bc_wall: 0.065}`; `steady_heat_conduction_3d`: `{pde: 1431.0,
bc_T_boundary: 0.22, bc_T_inlet_hot: 1.14}` — never two identical or
all-zero values, which would indicate a tag not actually being applied.
Two representative presets (one box, `laplace_2d`; one cylinder,
`pipe_flow_3d`) were then trained for 200 epochs through `solve_pde()`
end-to-end and confirmed to actually converge, not just run:
`laplace_2d` aggregate loss `670.8 -> 0.256` (ratio `3.8e-4`),
`pipe_flow_3d` `299.6 -> 2.55` (ratio `8.5e-3`), with every per-tag loss
shrinking individually (`pipe_flow_3d`'s three tags: `0.49/1.75/0.065 ->
0.001/0.0005/0.004`) — the per-tag masking is doing real, physically
meaningful work, not a coincidental pass-through. All 17 "not fixable"
presets were independently re-confirmed to still raise
`TagConditionsUnresolved` exactly as before (unaffected, as intended).

### Test suite impact, measured before and after (full `pytest tests/`, not just the two `solve_pde` files)

One pre-existing, unrelated environment issue was found and worked around
to get a clean before/after full-suite comparison at all: **`tests/
test_breadth_six_packages.py::test_breadth_classical_forecasters[xgboost]`
segfaults the Python interpreter outright** (confirmed in isolation,
identically on both the pre-fix and post-fix commit — a native `xgboost`/
OpenMP crash on this machine, unrelated to anything in this session's
change) — `pytest` cannot even catch a segfault, so the whole process
dies mid-run. Both full-suite runs below use `--deselect
"tests/test_breadth_six_packages.py::test_breadth_classical_forecasters[xgboost]"`
to work around it; this is a pre-existing environment gap, not something
this session introduced or fixed, and is unrelated to the tag-geometry
work either way.

Full `pytest tests/` (1667 of 1668 collected, 1 deselected), run at the
pre-fix commit (`bfa19dbd`, clean checkout) and again at this session's
final commit, both in this exact environment:

| | Before (`bfa19dbd`) | After (this session) |
|---|---|---|
| Passed | 1289 | **1340** |
| Failed | 75 | 75 |
| Error (setup/collection) | 39 | 39 |
| Skipped | 263 | **212** |
| xfail | 1 | 1 |
| **Total** | 1667 | 1667 |

**The set of 114 failing/erroring test IDs (75 `FAILED` + 39 `ERROR`) is
bit-for-bit identical before and after** (`diff` of the two sorted ID
lists is empty) — every one of them is a pre-existing, unrelated failure
(e.g. `test_app_backend.py`'s `RuntimeError`s, `test_admin_router.py`,
`test_preset_authoring.py`'s Ollama-dependent tests, ...), confirmed
untouched by this change. **The entire net effect of this pass on the
full suite is exactly 51 tests moving from `skip` to `pass`** (1289 ->
1340), and zero tests moving the other way — those 51 are precisely the
`test_cartesian_breadth.py`/`test_full_library_matrix.py` combinations
for the 23 newly-fixed presets that previously skipped with "needs real
geometry" and now train for real (confirmed directly: those two files
alone went from 56 passed/173 skipped to 107 passed/122 skipped, a
+51/-51 delta that accounts for the entire suite-wide delta with nothing
left over). No `"callable"`/`"all"` preset and no unrelated test file
changed status in either direction.

**Net tally against the original 40**: 23/40 (57.5%) fixed with real,
verified, non-degenerate per-tag training; 17/40 (42.5%) correctly
documented as needing real external geometry (or, in
`aircraft_wing_structural`'s one case, a separate compiler fix) that this
pass does not fabricate. `TagConditionsUnresolved` is unchanged and still
fires for all 17 plus every non-tag-based failure mode it always covered.

## Third follow-up pass: 9 specific presets across 3 pre-decided strategies, from the 17 "not fixable" list (2026-09-17)

This pass targeted 9 named presets out of the 17 above (grouped into 3
strategies, each decided in advance rather than re-litigated by this
session): real published standard geometry for 4 (Group 1), a genuine
compiler defect for 1 (Group 2), and an explicitly-flagged CHOSEN
CONVENTION for a genuinely ambiguous tag on 4 more (Group 3 -- 3 line
items, one of which is 2 presets). **6 of 9 closed for real** (verified
running `solve_pde()`/`compile_problem` end-to-end with distinct,
non-degenerate per-tag losses, the same bar as the 23 above; some also
shown to converge). **1** (`aircraft_wing_aerodynamics`) was investigated
as instructed and found to be genuinely blocked by a missing preset
parameter, not solved by assumption. **2 more** (`car_brake_thermal`,
`rocket_structural`) are genuine partials: each preset's ORIGINALLY-
STATED blocker is now resolved and independently verified, but each
turned out to hide a SECOND, different, pre-existing compiler defect this
pass was not scoped to fix -- precisely isolated and documented rather
than silently left as an unexplained failure.

### Group 1 — real published-standard geometry

- **`aircraft_wing_aerodynamics` — NOT fixed, missing parameter confirmed
  by inspection, not assumed.** The preset's own signature (`Re, Ma,
  alpha_deg, chord, rho_inf, U_inf, nu_air`) has no thickness ratio or
  NACA code at all -- `"NACA-like profile"` in its docstring names a
  *family*, not a specific 4-digit code. Picking "NACA 0012" (as
  instructed to try) means picking `t=0.12`, a number nothing in the
  preset provides or implies; every other 4-digit thickness would be
  equally "NACA-like" and produce a physically different airfoil. This is
  exactly the parameter-gap escape hatch this pass was told to use rather
  than guess -- left in `NOT_FIXABLE_WITHOUT_REAL_GEOMETRY` with the
  specific missing parameter named (not the old vaguer "needs a real
  airfoil profile" reason).
- **`rocket_nozzle_cfd` — fixed.** A simple conical (not Rao-bell, per the
  instruction to only attempt the bell contour if time allowed and the
  cone worked first) convergent-divergent nozzle wall, `r_wall(z) =
  throat_radius + (exit_radius - throat_radius) * z / nozzle_length` --
  every quantity read from the preset's own params (`throat_radius`,
  `exit_radius`, `nozzle_length` from `spec.meta`/`domain_bounds`), the
  half-angle `atan((exit_radius-throat_radius)/nozzle_length)` **derived**
  from them, not chosen freely. One honest scope note: the preset's own 3
  params describe only the *divergent* section unambiguously (no chamber
  radius or throat axial position is given anywhere), so the domain
  modelled is the throat (z=0, r=throat_radius) to exit (z=nozzle_length,
  r=exit_radius) -- "inlet" here is the throat plane with the preset's own
  stagnation p/T imposed there, a standard simplification, not an
  invented parameter. New builder:
  `sample_axisymmetric_wall_tag_batch` (`analytic_domain_batch_builder.py`).
  Verified: distinct per-tag losses at init (`pde=1.68e6, bc_inlet=5.0e13,
  bc_outlet=1.64, bc_wall=0.146, bc_axis=0.464`); trained through
  `solve_pde()` for real. Honest caveat found while verifying: `bc_inlet`'s
  raw physical units (`p_inlet=1e7 Pa`, `T_inlet=3500 K`) are ~13-14 orders
  of magnitude larger than the other 3 tags' targets, so this preset's
  *own* unnormalized units (not the geometry fix) dominate the loss and
  visibly stall Adam training at both `lr=1e-3`/30 epochs and
  `lr=1e-4`/300 epochs -- `bc_outlet`/`bc_wall`/`bc_axis` stay real,
  distinct, and non-degenerate throughout either way. This is the same
  category of finding as `crystal_phonon`'s float32-precision note in the
  first astrophysics pass above: a real, practical training-conditioning
  fact about this preset's literal SI defaults, not a defect in the fix.
- **`car_external_aero` — fixed.** A 2D Ahmed-body-inspired silhouette
  (Ahmed, Ramm & Faltin 1984, SAE 840300 -- the standard reference bluff
  body for automotive external-aero CFD validation), built from its
  published dimension RATIOS (ground clearance 50/1044, nose radius
  100/1044, rear slant length 222/1044 of the body length; the classic
  25-degree slant configuration) applied to this preset's own
  `car_length`/`car_height` (read back exactly out of `domain_bounds`,
  which the preset itself derives from them: `x_max=5*car_length`,
  `y_max=5*car_height`) -- not the real body's literal millimeters, and
  not a freehand silhouette. The front is modelled as two quarter-circle
  fillets of the published radius (front-bottom, front-top) joined by a
  short vertical face -- closer to the real body's actual corner rounding
  than one giant arc spanning the full height. New builder:
  `sample_box_with_curve_tag_batch` plus `_curve_geometry.
  ahmed_body_polygon`/`polygon_perimeter_sample`/`polygon_contains` (a
  vectorized ray-casting point-in-polygon test used to reject interior
  collocation points that fall inside the solid body -- the ONE geometry
  in this batch with real cross-sectional area, unlike the zero-thickness
  blade/nozzle-wall curves). Verified: distinct per-tag losses at init
  (`pde=4.49, bc_inlet=552.6, bc_outlet=2.30, bc_ground=551.1,
  bc_car_body=0.079, bc_top=2.42`); trained 30 epochs through `solve_pde()`
  with visible per-tag movement (`bc_outlet` 2.30->0.47, `bc_ground`
  551->533).
- **`axial_compressor_cascade_2d` — fixed, and trains cleanly.** A
  circular-arc cascade-blade camber line (Dixon & Hall, *Fluid Mechanics
  and Thermodynamics of Turbomachinery*, the standard circular-arc camber
  construction), using ONLY this preset's own `flow_angle_in_deg`,
  `flow_angle_out_deg`, `chord` (read as the zero-incidence/zero-deviation
  design-condition metal angles, a standard assumption) -- turning angle
  `theta_c=beta1-beta2`, camber-circle radius `R=chord/(2*sin(theta_c/2))`
  (the chord-length relation for a circular arc), and stagger
  `zeta=(beta1+beta2)/2`, which is not a separate assumption but a
  *forced consequence* of the arc being circular and symmetric (the
  tangent deviates +-theta_c/2 from the chord at each end, so
  beta1=zeta+theta_c/2 and beta2=zeta-theta_c/2 solve exactly to that
  mean). Modelled as a zero-thickness cambered plate (no preset param
  gives blade thickness or max-camber location), so no interior exclusion
  is needed. `inlet`/`outlet` are `selector_type="callable"` and are
  covered automatically, same as every mixed preset in the pass above.
  Verified with real training, not just non-degenerate losses: aggregate
  loss `387.4 -> 3.43` over 30 epochs, `pde` `363.5 -> 2.02`,
  `bc_blade_surface` `0.131 -> 0.0059` -- both terms shrinking together,
  the strongest convergence evidence in this pass (comparable to
  `laplace_2d`/`pipe_flow_3d`'s representative-convergence bar above).

### Group 2 — compiler defect, not geometry: `aircraft_wing_structural` fixed

`compile.py`'s per-condition loop unconditionally evaluated
`pred = torch.cat([fvals[f] for f in cond.fields], dim=1)` for every
condition BEFORE dispatching on `cond.kind` -- for a Neumann condition
declaring traction fields (`tx`/`ty`) that the model never outputs
directly (the model outputs displacements `ux`/`uy`), this raised
`KeyError: 'ty'` immediately, confirmed by hand last pass even with
correct geometry supplied.

**The generalization** (not a hack scoped to this one preset): added
`ConditionSpec.traction_map` (`pinneapple_physics/pde_environment/
conditions.py`) -- an explicit `{declared_traction_field: displacement_
field}` mapping (e.g. `{"ty": "uy"}`), documented in `ConditionSpec`'s own
docstring as a general mechanism for any elasticity preset whose Neumann
BC targets are tractions rather than literal model fields. `compile.py`
now:

1. Only takes the old eager `fvals[f]`-lookup path when every one of
   `cond.fields` IS a literal model field (`uses_model_fields`) --
   unconditionally true for every pre-existing condition in the repo, so
   this is provably a no-op for everything except the new traction case.
2. For a `kind="neumann", order<=1` condition with `traction_map` set,
   builds the FULL elasticity stress tensor sigma from ALL of the PDE's
   own displacement fields (`_elasticity_displacement_fields`,
   `_elasticity_stress_tensor` -- identical lambda/mu constitutive
   relation, including the plane-stress reduced lambda*, to the interior
   residual's own construction, so a traction-BC prediction can never
   silently disagree with what the PDE residual itself enforces), then
   evaluates `n . sigma` restricted to the row selected by each
   component's mapped axis (`_elasticity_traction_from_stress`).
3. Anything else declaring a non-model field without a `traction_map` now
   raises a clear, actionable error naming the missing mechanism, instead
   of the old bare, confusing `KeyError`.

**Verified two ways**, matching this session's method for a
compiler-level (not physics-content) fix:
- **Closed-form correctness check** (not just "ran without erroring"): a
  hand-picked pure-shear displacement field (`ux=k3*y, uy=k4*x`, giving
  `eps_xx=eps_yy=0`, `eps_xy=0.5*(k3+k4)` exactly) has a known closed-form
  traction at the `x=max` face (`t_y = sigma_xy = mu*(k3+k4)`) --
  `_elasticity_traction_from_stress` reproduces it to **7e-8 relative
  error** (float32 roundoff), confirming the formula itself, independent
  of any trained network.
- **End-to-end on the real preset**: `aircraft_wing_structural` run
  through `compile_problem` with real box geometry (`root_fixed`=x-min,
  `tip_load`=x-max, `free_surface`=y-min+y-max, all unambiguous per the
  preset's own docstring, as already noted in the second pass) --
  distinct, non-degenerate per-tag losses (`bc_root_fixed=0.037,
  bc_tip_load=3.75e20, bc_free_surface=1.75e22`), no `KeyError`. Honest
  caveat, same category as `rocket_nozzle_cfd`'s above: this preset's own
  raw units (`E=70e9 Pa`, `lift_load=50000 N` over a `~5m x 0.1m`
  unnormalized domain) make `pde`/`bc_tip_load`/`bc_free_surface` land at
  `1e20`-`1e24`, so 30 epochs of plain Adam barely move the aggregate --
  a pre-existing units/scale property of this preset, not of the traction
  mechanism (confirmed independently correct by the closed-form check
  above).
- Added `NeumannBC(..., traction_map=...)` plumbing to
  `_tagged_neumann()` (`presets/engineering.py`) and set it on both of
  `aircraft_wing_structural`'s traction conditions
  (`{"ty": "uy"}` for `tip_load`, `{"tx": "ux", "ty": "uy"}` for
  `free_surface`).
- Regression check: `tests/test_manufactured_solutions.py` (36/36) still
  passes unchanged -- every pre-existing condition in the repo has
  `traction_map=None` and takes the exact old code path.

### Group 3 — genuinely ambiguous tag, decision already made (implemented, not re-decided)

Each of these adds an explicit `CHOSEN_CONVENTIONS` entry in
`tag_geometry.py` (separate from the inline per-preset comments used for
the 23+1 presets above, which all cite the preset's OWN text) --
transparently marking these as this session's engineering decision, not
something the preset itself states:

- **`linear_elasticity_3d` and `plane_stress_2d` — fixed.** `"fixed"` =
  domain-min face of the principal axis (x=0, the base/engaste),
  `"load"` = domain-max face (x=1, the free tip) -- the canonical
  cantilever-beam/column setup. Both are simple box fixtures (no new
  builder needed). `linear_elasticity_3d` (fields `ux,uy,uz`, kind
  `"linear_elasticity"`) also exercises `_elasticity_displacement_fields`'s
  3D branch for the first time in this pass, though this preset's own
  `load` condition uses full-vector traction (`ux,uy,uz` all Neumann-
  targeted, all literal model fields already) so it does not need
  `traction_map` itself.
- **`car_brake_thermal` — tag ambiguity resolved; training still blocked
  by a SECOND, independent, pre-existing compiler defect (same pattern as
  `rocket_structural` below).** `"friction_surface"` = the disc's two flat
  faces (`z=min`, `z=max`, where the pad contacts it), `"cooling_surface"`
  = the outer rim (`r=max`, exposed to airflow) -- the standard disc-brake
  thermal-analysis assignment; coords are cylindrical-axisymmetric
  `(r,z,t)`, and the `"initial"` condition is `selector_type="callable"`
  (auto-sampled by `solve_pde()` on its own). A plain box fixture was
  built for this and independently verified geometrically correct
  (`sample_box_tag_batch` with `tag_faces={"friction_surface": [z-min,
  z-max], "cooling_surface": [r-max]}` gives exactly 1000/500 masked
  points for a 500-per-face sample, as expected). **But it hits a second
  defect while training**: `"friction_surface"`/`"cooling_surface"`
  declare `fields=("q_heat",)`/`("h","T_ref")` -- a heat-flux magnitude
  and a convection coefficient+reference temperature, neither a literal
  model field (the model only outputs `"T"`) nor resolvable by the
  `traction_map` mechanism (built for elasticity tractions, not
  heat-flux/convection Neumann BCs). Grepping the preset files shows this
  exact `q_heat`/`h`/`T_ref` convention is ALSO used by
  `cpu_heatsink_thermal`, `pcb_thermal`, `industrial_furnace_thermal`, and
  all 3 `datacenter_*` presets -- a systemic, pre-existing gap across many
  thermal presets, well beyond this pass's authorized scope of resolving
  `car_brake_thermal`'s specific tag ambiguity. The fixture is therefore
  intentionally NOT wired into `TAG_GEOMETRY_FIXTURES`/`build_tag_batch`'s
  dispatch table, `car_brake_thermal` stays in
  `NOT_FIXABLE_WITHOUT_REAL_GEOMETRY` with this precise, narrowed reason,
  and `TagConditionsUnresolved` still correctly fires for it end-to-end --
  confirmed by re-running `test_full_library_matrix.py`'s
  `car_brake_thermal` case: still cleanly SKIPPED, same as before this
  pass, zero change in test-suite outcome.
- **`rocket_structural` — geometry half fixed for real; training still
  blocked by a SECOND, independent, pre-existing compiler defect.** This
  was correctly diagnosed last pass as a data bug, not a tag ambiguity:
  `domain_bounds` is a solid square `[-outer,outer]^2` but the physical
  part is an annulus. The preset already carries `inner_radius=0.2`/
  `outer_radius=0.22` in its own `meta` dict (confirmed by inspection --
  not fabricated), so a new `sample_annulus_tag_batch`
  (`analytic_domain_batch_builder.py`) samples the REAL annulus (rejection
  sampling `inner_radius <= sqrt(x^2+y^2) <= outer_radius` for the
  interior, exact inner/outer rings for the two boundary tags) reading
  those two params straight from `spec.meta`. Verified directly:
  interior collocation points land at `r in [0.20000, 0.21999]`, exactly
  the annulus, not the old square's corners/hole. Verified further that 3
  of its 4 tag conditions (`outer_wall`, `T_inner`, `T_outer`) train
  through `compile_problem` with real, distinct, non-degenerate losses
  (`bc_outer_wall=0.084, bc_T_inner=639633, bc_T_outer=85709`) once the
  annulus geometry is supplied. **But the 4th, `inner_wall`, hits a
  SECOND, independent defect this pass did not authorize fixing**: it
  declares `fields=("p_normal",)`, a pressure MAGNITUDE, not a literal
  model field and not resolvable by the `traction_map` mechanism built
  for Group 2 above (that mechanism derives one traction VECTOR COMPONENT
  along a named axis from the stress tensor; a pressure BC instead needs
  the scalar normal-stress contraction `n^T.sigma.n` compared against
  `-p_internal`, a genuinely different derivation with its own sign
  convention this pass was not asked to design). Confirmed directly: the
  new, clearer compiler error correctly names `'p_normal'` and points at
  `ConditionSpec.traction_map`'s docstring, instead of the old bare,
  unexplained `KeyError`. `rocket_structural` is therefore left in
  `NOT_FIXABLE_WITHOUT_REAL_GEOMETRY` with this precise, narrowed-down
  reason (previously: "domain_bounds doesn't encode the hole"; now: "the
  hole is fixed, a second and different compiler gap remains") --
  `TagConditionsUnresolved` still correctly fires for it via `solve_pde()`
  end-to-end, and the annulus fixture is intentionally NOT wired into
  `TAG_GEOMETRY_FIXTURES`/`build_tag_batch`'s dispatch table (kept as
  verified, tested, standalone infrastructure only) to preserve that
  table's existing invariant that every entry in it trains end-to-end for
  real, same as all 29 (23 + this pass's 6) already in it.

### Net tally for this pass

This pass addressed all 9 distinct preset names named across the task's 3
groups (Group 1: 4, Group 2: 1, Group 3: 4 -- `linear_elasticity_3d` and
`plane_stress_2d` are 2 separate presets under Group 3's first bullet).
**6 of the 9 closed for real, trained end-to-end with real, distinct,
non-degenerate per-tag losses** (some also shown to converge):
`rocket_nozzle_cfd`, `car_external_aero`, `axial_compressor_cascade_2d`
(Group 1), `aircraft_wing_structural` (Group 2), `linear_elasticity_3d`
and `plane_stress_2d` (Group 3). **1 confirmed genuinely blocked by a
specific missing preset parameter, investigated rather than guessed**:
`aircraft_wing_aerodynamics` (Group 1 -- no NACA thickness/code param
exists to read). **2 are genuine partials**: `car_brake_thermal` and
`rocket_structural` (Group 3) both had their ORIGINALLY-STATED blocker
(a tag-face ambiguity; a solid-square-vs-annulus data bug) resolved and
independently verified, but each hides a SECOND, different, pre-existing
compiler defect this pass was not scoped to fix (heat-flux/convection
Neumann fields for the former, a pressure-magnitude Neumann field for the
latter -- both a different derivation than the `traction_map` mechanism
built for Group 2's elasticity-traction case handles) -- documented
precisely, with the new, clearer compiler error naming exactly what's
missing in each case, rather than claimed as closed. Both partials leave
`test_full_library_matrix.py`'s status for them exactly as it was before
this pass (`SKIPPED`, confirmed by re-running both cases directly) -- the
partial fixes are real, verified infrastructure, but deliberately not
wired into `TAG_GEOMETRY_FIXTURES` so no test's outcome changes on their
account.

Updated running tally against the original 40 tag-based presets: **29/40
now trained end-to-end with real geometry** (23 + 6), **11/40** remain
`NOT_FIXABLE_WITHOUT_REAL_GEOMETRY` for a specific, individually-verified
reason each (9 unchanged from before + `car_brake_thermal`'s and
`rocket_structural`'s narrowed, partially-resolved ones).

### Full-suite regression check (before/after, same method as the second pass)

`pytest tests/ --deselect "tests/test_breadth_six_packages.py::test_breadth_classical_forecasters[xgboost]"`,
run at the clean pre-this-pass commit (`571b66ba`, via a separate
`git worktree add --detach` checkout so the comparison needed no stash of
this pass's own in-progress changes) and again at this pass's final
commit, same environment.

**Methodology note, an environment quirk this pass had to work around**:
in this environment, pytest's own final `passed/failed/error/skipped`
tally line (and part of the `short test summary info` listing) did not
get written to disk on either run -- confirmed NOT a resource-contention
artifact (it reproduced identically running completely alone, no other
process active) and NOT a hang (the process genuinely exits; `ps`
confirms it is gone, deterministically right after printing
`test_app_backend.py`'s last `ERROR` line both times). All test EXECUTION
completes normally first (the `.`/`F`/`E`/`s`/`x` dot-progress reaches
100%); whatever is cut short happens only in pytest's own post-run
reporting. Rather than accept an unverifiable total, both runs' dot-
progress characters (the exact `.FEsx` stream pytest prints per test as
it runs, well before any summary section) were extracted directly and
counted, and (since `--collect-only` confirms both commits collect the
exact same 1667 test IDs in the exact same order, as expected --
`TAG_GEOMETRY_FIXTURES` gaining 6 entries doesn't change which tests
`list_presets()`-driven files collect, only how they later behave)
**mapped position-by-position back to the actual test IDs**, turning the
two character streams into a full, exact pass/fail/skip diff, not just a
count.

| | Before (`571b66ba`) | After (this pass) |
|---|---|---|
| Passed | 1341 | 1353 |
| Failed | 74 | 75 |
| Error | 39 | 39 |
| Skipped | 212 | 199 |
| xfail | 1 | 1 |
| Total | 1667 | 1667 |

**Exactly 14 tests changed status, identified individually, not just
counted**: 13 moved `skip -> pass` and are precisely the 6 newly-fixed
presets' test combinations -- `test_cartesian_breadth.py`'s 7
architectures x `plane_stress_2d` (`vanilla_pinn`, `modified_mlp`,
`bench_res_mlp`, `bench_fourier_mlp`, `siren`, `vpinn`, `xtfc`) and
`test_full_library_matrix.py`'s one test per preset for all 6
(`aircraft_wing_structural`, `axial_compressor_cascade_2d`,
`car_external_aero`, `linear_elasticity_3d`, `plane_stress_2d`,
`rocket_nozzle_cfd`) -- exactly matching this pass's 6 real fixes, with
nothing left over. The 14th, and only unexpected change, moved
`pass -> fail`: `tests/pinneapple_analysis/test_architecture_critique.py
::test_real_llm_produces_a_valid_complete_response` -- a completely
unrelated test (architecture-critique NLP review, nothing to do with PDE
presets/geometry/elasticity) that calls a REAL Ollama LLM
(`provider="ollama"`) and asserts its output. Re-ran it standalone to
confirm this is exactly what it looks like: the live LLM occasionally
returns a category name (`'overall_reasoning'`) outside the fixed
checklist the test's own assertion enforces
(`ValueError: LLM named category 'overall_reasoning', not in the real
checklist [...] -- refusing a hallucinated category.`) -- genuine LLM-
output nondeterminism from a real model call, unrelated to and
unaffected by anything in `pinneapple_physics`/`pinneapple_design` this
pass touched. **Net, verified result: 0 regressions from this pass's
changes** (13/13 of the expected new passes landed exactly where
predicted, 0 unexplained losses, and the one unrelated flip has an
independently-confirmed, unrelated root cause).

## Fourth follow-up pass: NACA 0012 literature default, and 2 new Neumann-BC compiler mechanisms (2026-09-17)

Product-owner decision (Yan), not this pass's own choice: 3 named,
pre-decided items from the Third follow-up pass's leftover list above --
(1) `aircraft_wing_aerodynamics` gets a literature-standard NACA 0012
thickness default (the previous pass correctly left this unfixed because
NO thickness parameter existed at all; the decision here is to ADD one,
not to guess a value the preset never had), (2) generalize the compiler
for heat-flux/convection Neumann BCs (`car_brake_thermal`'s second,
independent blocker from last pass), (3) generalize the compiler for a
pressure-magnitude Neumann BC (`rocket_structural`'s second, independent
blocker from last pass). **All 3 closed for real**, verified end-to-end.

### Item 1 -- `aircraft_wing_aerodynamics`: NACA 0012 literature default

Added `naca_thickness: float = 0.12` as an explicit, overridable keyword
parameter (not a hidden internal constant) to the preset's own signature.
0.12 (NACA 0012, a symmetric 12%-thick profile) is documented in the
preset's own docstring, `_curve_geometry.py`, and `tag_geometry.py` as a
**literature default the product owner chose**, not something inherent
to the preset's original parameters (which never included a thickness or
NACA-code field at all, exactly as the Third follow-up pass found) --
the single most common reference/benchmark airfoil in the public
aerodynamics/CFD literature, and readable/overridable by anyone who wants
to supply a real specific profile's thickness later. Uses the same public
NACA 4-digit thickness formula already cited (and investigated) in the
previous pass's `NOT_FIXABLE_WITHOUT_REAL_GEOMETRY` entry for this
preset:

```
y_t(x) = 5*t*(0.2969*sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)^2
               + 0.2843*(x/c)^3 - 0.1015*(x/c)^4)
```

implemented fresh as `_curve_geometry.naca4_symmetric_polygon` (a closed
polygon via cosine-spaced sample points, upper+lower surface, same
construction style as `ahmed_body_polygon`), reusing the already-existing
`polygon_perimeter_sample`/`polygon_contains` helpers unchanged (both are
fully generic, not car-body-specific). Sanity-checked directly: the
polygon's peak half-thickness is 0.06003*chord (i.e. thickness 0.1201*c,
matching the nominal 12% to within cosine-sampling resolution), and
`polygon_contains` correctly classifies a point at the airfoil's midchord
as inside and points ahead of/behind/outside it as outside.

Wired as a `box_with_curve` `TAG_GEOMETRY_FIXTURES` entry: airfoil
leading edge at the coordinate origin (chord along +x, unrotated -- angle
of attack is already encoded in the existing `farfield_inlet` velocity
direction, the standard "rotate the flow, not the body" CFD convention,
unchanged by this pass), consistent with this preset's own asymmetric
domain_bounds (`-5*chord` upstream, `15*chord` downstream of x=0) and
symmetric y-range. One additional, explicitly-flagged **chosen
convention** (added to `CHOSEN_CONVENTIONS`, same transparency bar as
every other genuinely-ambiguous-tag entry in this file): `"wake_outlet"`
has no stated face anywhere in the preset (only named in the docstring's
"Regions" list) -- mapped to the same `x=max` outlet plane as
`"farfield_outlet"` (both are Neumann/zero-gradient conditions on
different fields -- pressure vs. velocity -- at the same physical exit
plane), reusing this file's own already-established pattern of two
distinct tags sharing one face along the same flow axis.

**Verified end-to-end**: `build_tag_batch("aircraft_wing_aerodynamics",
...)` produces all 4 non-degenerate tag masks (400 points each at
`n_bc_per_face=400`: `farfield_inlet`, `farfield_outlet`, `wake_outlet`,
`airfoil`); `compile_problem` gives real, finite, distinct per-tag losses
at init (`pde=4.65, bc_farfield_inlet=5189.5, bc_farfield_outlet=2.12,
bc_airfoil=0.088, bc_wake_outlet=3.81`); trained 20 epochs through
`solve_pde()` with visible per-tag movement on 2 of 4 tags
(`bc_farfield_outlet` 2.12->0.68, `bc_wake_outlet` 3.81->2.78) --
`bc_farfield_inlet` dominates and barely moves because `U_inf=102 m/s`
is unnormalized (same category of finding as `rocket_nozzle_cfd`'s and
`aircraft_wing_structural`'s raw-SI-units notes above, not a defect in
this fix). Moved from `NOT_FIXABLE_WITHOUT_REAL_GEOMETRY` to
`TAG_GEOMETRY_FIXTURES`.

### Items 2 & 3 -- generalizing the compiler for 2 more Neumann-BC families

The Third follow-up pass's `ConditionSpec.traction_map` only covers ONE
vector traction component derived from `n . sigma` along a named axis. It
does not cover (a) a Neumann/Robin condition whose target is a heat flux
or convection law (fields not literal model outputs, and not a traction
at all), or (b) a Neumann condition whose target is a pressure MAGNITUDE
needing the full scalar contraction `n^T . sigma . n` (not one axis-aligned
component). Both are now real, separate mechanisms in
`pinneapple_physics/pde_environment/conditions.py`
(`ConditionSpec.thermal_bc`, `ConditionSpec.normal_stress_field`) and
`compile.py` (two new branches in the per-condition loop, gated the same
way `traction_map` already is: `uses_model_fields=False` +
`kind="neumann"` + `order<=1` + the new field set).

**`ConditionSpec.thermal_bc`** (`{"kind": "flux"|"convection", "T_field":
"T"}`): implements the standard physics directly --

```
flux:       -k * dT/dn = q_heat
convection: -k * dT/dn = h * (T - T_ref)
```

`dT/dn` via the same `norm_dot_grad` autograd machinery the plain Neumann
branch already uses; `k` read with the exact same fallback order
(`k_eff`, then `k`, then 1.0) the `heat_equation_steady*`/
`heat_equation_transient` interior residuals already use, so this can
never silently disagree with the PDE's own conductivity. Generic: any
preset declaring this `q_heat`/`h`+`T_ref` convention gets the mechanism
automatically, not just `car_brake_thermal` -- confirmed by grep this is
the SAME convention used by `cpu_heatsink_thermal`, `pcb_thermal`,
`industrial_furnace_thermal`, and all 3 `datacenter_*` thermal presets
(their `NOT_FIXABLE_WITHOUT_REAL_GEOMETRY` entries in `tag_geometry.py`
now note explicitly that their remaining blocker is geometry only, not
this compiler gap).

**`ConditionSpec.normal_stress_field`** (a single field name, e.g.
`"p_normal"`): implements the pressure-vessel convention
`n^T . sigma . n = -p_internal` via a new
`_elasticity_normal_stress_from_stress` helper (full double contraction,
genuinely different math from `_elasticity_traction_from_stress`'s single
selected row -- not a special case of it). `sigma` is built by the SAME
`_elasticity_stress_tensor` helper `traction_map` already uses, so a
pressure-BC prediction can never silently disagree with the interior
residual's own stress. `rocket_structural`'s actual `pde_kind` is
`"thermoelasticity_2d"`, not one of the 3 plain elasticity kinds
`traction_map`/`_elasticity_stress_tensor` originally supported --
extended `_elasticity_stress_tensor` (and
`_elasticity_displacement_fields`) with a `thermoelasticity_2d` branch
that additionally subtracts the SAME isotropic thermal-strain correction
(`eps_th = alpha_T*T*I`) the interior `"thermoelasticity_2d"` residual
already applies, verified below to be load-bearing (not a silent no-op).

**Closed-form verification** (matching the `traction_map` pure-shear
check's rigor -- exact solution in, ~0 residual out; wrong solution in,
clearly nonzero residual out -- via `compile_problem` directly, no
training involved), 4 independent checks:

| Mechanism | Trial field | Result |
|---|---|---|
| `thermal_bc="flux"` | `T=a*x+0.5*b*x^2+0.5*c*y^2` (quadratic in every direction to avoid this codebase's known autograd second-derivative-of-a-zero-dependence limitation, same reason `test_manufactured_solutions.py`'s own exact solutions are never purely linear); target `q_heat=-k*dT/dn` at `x=1` | exact: loss `0.000e+00`; wrong (`q_heat=999`): loss `1.01e+06` |
| `thermal_bc="convection"` | same T field, position-dependent `h` solved exactly per-point so `h*(T-T_ref)=-k*dT/dn` | exact: loss `0.000e+00`; wrong (`h`x5): loss `1.64e+08` |
| `normal_stress_field` (`linear_elasticity_plane_stress`) | `ux,uy` quadratic-in-both-directions displacement field; target `p_internal=-sigma_xx(X)` computed exactly per-point from the closed-form stress (using the SAME plane-stress reduced `lambda*`) | exact: loss `0.000e+00`; wrong (`p=999`): loss `9.93e+05` |
| `normal_stress_field` (`thermoelasticity_2d`, rocket_structural's real `pde_kind`) | same displacement field plus `T=T0+0.5*d*(x^2+y^2)`; target includes the thermal-strain correction | exact: loss `7.1e-14`; **target missing the thermal term**: loss `44.75` (proves the thermal branch is load-bearing, not a no-op) |

### `car_brake_thermal` -- wired end-to-end

`friction_surface`/`cooling_surface` now declare `thermal_bc` explicitly
in `engineering.py` (`{"kind": "flux", ...}` / `{"kind": "convection",
...}`); the box `TAG_GEOMETRY_FIXTURES` entry built and geometrically
verified last pass (unchanged) is now wired into the dispatch table.
Verified via `build_tag_batch`+`compile_problem`+`solve_pde()`: all 3
non-`callable` conditions get non-degenerate masks
(`friction_surface`=800 pts [both z-faces], `cooling_surface`=400 pts);
real, finite, distinct per-tag losses at init (`pde=2.35,
bc_friction_surface=4.00e12, bc_cooling_surface=5.51e8`) -- the huge
`bc_friction_surface` value is exactly consistent with the untrained
network's near-zero `dT/dn`, so the residual is dominated by
`q_friction=2e6 W/m^2` squared (`(2e6)^2=4e12`, matching to 3 significant
figures); `bc_cooling_surface` similarly consistent with
`h_conv=80`/`T_ref=293` at a near-zero initial `T`. `test_full_library_
matrix.py::test_audit_breadth_preset_trains_a_few_steps[car_brake_thermal]`
now genuinely trains 3 real epochs through `solve_pde()` (previously
skipped via `TagConditionsUnresolved`) -- **passes**. Moved from
`NOT_FIXABLE_WITHOUT_REAL_GEOMETRY` to `TAG_GEOMETRY_FIXTURES`.

### `rocket_structural` -- wired end-to-end

`inner_wall` now declares `normal_stress_field="p_normal"` explicitly in
`engineering.py`; the annulus `TAG_GEOMETRY_FIXTURES` entry built and
geometrically verified last pass (unchanged: real annulus from this
preset's own `meta["inner_radius"]`/`meta["outer_radius"]`) is now wired
into the dispatch table via a new `"annulus"` shape branch in
`build_tag_batch` (reusing `sample_annulus_tag_batch`, already present
and tested as standalone infrastructure since last pass).
`T_inner`/`outer_wall`/`T_outer` share the same inner/outer rings as
`inner_wall` (same 2 physical circles, different fields). Verified via
`build_tag_batch`+`compile_problem`+`solve_pde()`: all 4 tags get 400
non-degenerate points each; real, finite, distinct per-tag losses at init
(`pde=2.16e25, bc_inner_wall=1.32e23, bc_outer_wall=0.056,
bc_T_inner=639568, bc_T_outer=85690`) -- `bc_T_inner`/`bc_T_outer` land
almost exactly on last pass's already-verified values for the SAME 3
non-`inner_wall` tags (`639633`/`85709` then vs. `639568`/`85690` now,
the tiny difference being random model-init noise only), confirming this
pass's annulus wiring reproduces the prior, independently-verified
geometry exactly, with `bc_inner_wall` now a real number instead of a
`KeyError`. The `bc_inner_wall`/`pde` scale (`~1e23`/`~1e25`) is the same
raw-SI-units category of finding as `aircraft_wing_structural`'s
`E=70e9 Pa` note above (`rocket_structural`'s own `E=200e9 Pa`,
`p_internal=10e6 Pa` are similarly unnormalized), not a defect in this
fix. `test_full_library_matrix.py::test_audit_breadth_preset_trains_a_
few_steps[rocket_structural]` now genuinely trains 3 real epochs through
`solve_pde()` (previously skipped) -- **passes**. Moved from
`NOT_FIXABLE_WITHOUT_REAL_GEOMETRY` to `TAG_GEOMETRY_FIXTURES`.

### Net tally for this pass

All 3 items closed for real, each verified with either a closed-form
proof (both new compiler mechanisms, 4/4 checks) or real non-degenerate
per-tag losses confirmed running end-to-end through `solve_pde()`/
`compile_problem` (all 3 presets), matching this line of work's
established bar throughout. Updated running tally against the original
40 tag-based presets: **32/40 now trained end-to-end with real geometry**
(23 + 6 + 3), **8/40** remain `NOT_FIXABLE_WITHOUT_REAL_GEOMETRY` for a
specific, individually-verified reason each (real fin/rack/hotspot/blade
geometry this pass was not asked to fabricate; 4 of those 8's
`thermal_bc`-related fields are now noted as compiler-ready, blocked on
geometry alone).

### Full-suite regression check (before/after, same method as the second and third passes)

`pytest tests/ --deselect "tests/test_breadth_six_packages.py::test_breadth_classical_forecasters[xgboost]"`,
run at the clean pre-this-pass commit (`14e0a131`, via a separate
`git worktree add --detach` checkout, same technique as both previous
passes) and again at this pass's final commit, same environment.

**Same methodology note as the third pass applies again**: this
environment's pytest does not write its own final `passed/failed/error/
skipped` tally line on either run (confirmed again: all test EXECUTION
completes normally, `.FEsx` dot-progress reaches 100%, the process exits
cleanly right after printing the `short test summary info` section's last
line -- only the one-line aggregate afterward is missing). `--collect-only`
(without `-q`, which on this pytest version condenses to per-file counts
once the suite is large) confirms both commits collect the exact same
**1667** test IDs in the exact same order, so the `.FEsx` dot-progress
stream was mapped position-by-position back to the real test IDs, exactly
as the third pass did.

| | Before (`14e0a131`) | After (this pass) |
|---|---|---|
| Passed | 1354 | 1357 |
| Failed | 74 | 74 |
| Error | 39 | 39 |
| Skipped | 199 | 196 |
| xfail | 1 | 1 |
| Total | 1667 | 1667 |

**Exactly 3 tests changed status, identified individually, not just
counted** -- all 3 `skip -> pass`, exactly the 3 presets this pass wired
into `TAG_GEOMETRY_FIXTURES`, one `test_full_library_matrix.py` case each
(none of the 3 appear in `test_cartesian_breadth.py`'s fixed 7-preset
list, so no additional architecture x preset combinations were affected
there):

```
tests/test_full_library_matrix.py::test_audit_breadth_preset_trains_a_few_steps[aircraft_wing_aerodynamics]
tests/test_full_library_matrix.py::test_audit_breadth_preset_trains_a_few_steps[car_brake_thermal]
tests/test_full_library_matrix.py::test_audit_breadth_preset_trains_a_few_steps[rocket_structural]
```

**Zero unexplained changes, zero regressions** -- unlike the third pass
(which had one unrelated `pass -> fail` flip from a live, nondeterministic
Ollama call), this pass's diff is exactly and only the 3 expected flips,
nothing else moved in either direction. `FAILED`/`ERROR` counts are
bit-for-bit identical before and after (74/39), confirming this pass
touched nothing in the pre-existing failure/error set.

## Geometry OOD guardrail: a real gap from the PhysicsNeMo comparison, closed (2026-09-18)

**The finding this closes**, from a separate cross-repo research pass
(`physicsnemo-notes/insights-to-import-into-pinneapple.md`, section 7a,
real code read on both sides, no code ported): NVIDIA PhysicsNeMo ships an
experimental geometry guardrail
(`physicsnemo/experimental/guardrails/geometry/`, ~3561 lines, Apache-2.0)
that flags a CAD/mesh geometry as out-of-distribution (OOD) relative to
the shapes a model was trained on -- shape-feature extraction + a density
model (GMM or Polynomial Chaos Expansion) + OK/WARN/REJECT percentile
classification. PINNeAPPle's trust/verification stack
(`trust_gate.py`/`physics_confidence_score.py`/`evidence_graph.py`, 6969
lines, larger than PhysicsNeMo's own guardrails module) had no equivalent
signal -- `pinneapple_analysis.verification.geometry_intelligence` sounds
adjacent but solves a different problem (semantic region/BC
classification, not whole-shape anomaly detection). This was a real,
confirmed gap, not a duplication.

### What was built

New `pinneapple_analysis/verification/geometry_ood_guardrail.py`
(`GeometryOODGuardrail`/`GeometryOODResult`/`extract_geometry_features`/
`default_reference_meshes`):

- **Feature extraction**: 13 real shape descriptors (centroid, bbox
  extent, 3 PCA eigenvalues of the vertex covariance, surface area, bbox
  volume, aspect ratio, mean curvature proxy), reusing existing
  infrastructure (`pinneapple_design.geometry.core.mesh.MeshData`,
  `pinneapple_design.geometry.ops.features.compute_curvature_proxy`)
  rather than reimplementing mesh feature extraction from scratch.
- **Density model -- a deliberate, documented scope decision**: a
  diagonal Mahalanobis distance (per-feature z-score sum-of-squares) +
  `scipy.stats.chi2.sf`, NOT a full covariance / GMM / PCE. This is the
  exact same statistical recipe `pinneapple_analysis.trust.trust_gate
  .TrustGate._ood_score` already uses for coordinate-space OOD in this
  codebase, applied here to shape features instead -- chosen explicitly
  because PINNeAPPle's current geometry catalog (generated
  primitives/presets) doesn't yet have enough archived real geometries
  per preset to fit a full 13x13 covariance honestly; the module's own
  docstring documents this tradeoff (real cross-feature correlations are
  not modelled) rather than hiding it, and states when to revisit it (a
  real per-preset archive of production geometries).
- **Reference catalog**: `default_reference_meshes()` builds 48 real
  (non-fabricated) box/cylinder/channel meshes via the existing
  `pinneapple_design.geometry.gen.primitives.build_mesh` -- the same
  real mesh-generation infrastructure `geometry_intelligence.py`'s own
  tests already use -- across a deterministic grid of sizes/aspect
  ratios.

### Integration into the trust stack

Wired as a 5th `ConfidenceComponent` into
`pinneapple_analysis.verification.physics_confidence_score`
(`N_POSSIBLE_COMPONENTS` 4 -> 5, new `geometry_ood_result` kwarg on
`compute_physics_confidence()`, new `_component_from_geometry_ood()`
following the exact same "score used as real evidence, `source_summary`
quotes the real numbers, absent check is never fabricated" pattern the
other 4 components already use). `GeometryOODResult.score` is already a
[0, 1] "higher = more trustworthy" value with the identical convention
`TrustGate._ood_score` uses, so no extra transform is needed for it to
slot into the existing arithmetic-mean aggregation.
`pinneapple_analysis.verification.evidence_graph.build_evidence_graph`/
`explain_trust` required **zero code changes** to pick up the new
component -- confirmed directly: they iterate generically over
`confidence.components` by `name`/`score`/`source_summary`, so a REJECT
geometry_ood result correctly renders as a `CONTRADICTING` evidence line
automatically.

### Real verification: normal vs. anomalous geometry

Fit `GeometryOODGuardrail` on the 48-mesh default reference catalog,
queried two real geometries built by the same `build_mesh()`
infrastructure:

| Geometry | mahalanobis_sq | p-value (score) | status |
|---|---|---|---|
| Box `(1.5, 1.2, 0.9)` -- inside the reference range | 3.09 (dof=13) | 0.998 | OK |
| Box `(500, 0.001, 0.001)` -- extreme aspect ratio, 5 orders of magnitude outside the reference range | 1.24e9 | ~0.0 | REJECT |

The score genuinely differentiates the two (not a constant output), and
`top_deviating_features` correctly names `aspect_ratio`/`pca_eigenvalue_1`
as the drivers for the REJECT case -- an honest, inspectable explanation,
never an opaque number. A `woven_tube` mesh (a shape family never
present in the box/cylinder/channel reference catalog) was also confirmed
to score lower than an in-family box, and `build_evidence_graph` was
confirmed to render the REJECT case as a `CONTRADICTING` evidence line.

### Test suite impact, measured before and after

New `tests/test_geometry_ood_guardrail.py` (15 tests, real
`build_mesh()` geometries, `pytest.importorskip("trimesh")` matching this
repo's established convention for the optional `geom` extra).
`tests/test_physics_confidence_score.py` updated for
`N_POSSIBLE_COMPONENTS` 4->5: the one test that hardcoded "4 components =
full coverage" was split into
`test_all_four_original_components_is_partial_coverage_now` (asserts
`coverage == 4/5`, an explicit regression guard for this exact change)
and a new `test_all_five_components_full_coverage` (supplies a real
`geometry_ood_result` to reach true 5/5), plus 3 new geometry_ood-alone
tests. All 23+15 tests pass in isolation.

Full `pytest tests/ --deselect
"tests/test_breadth_six_packages.py::test_breadth_classical_forecasters[xgboost]"`
(the same pre-existing xgboost/OpenMP segfault workaround this line of
work has used since the second pass), run at the clean pre-this-session
commit (`69117744`, via a separate `git worktree add --detach` checkout,
same technique as the prior passes) and again at this session's final
commit, same environment (the optional `trimesh`/`geom` extra was
installed for this comparison, since it's required to exercise the new
module at all -- identical in both runs):

| | Before (`69117744`) | After (this session) |
|---|---|---|
| Collected | 1691 | 1711 |
| Passed | 1375 | 1394 |
| Failed | 78 | 79 |
| Error | 40 | 40 |
| Skipped | 197 | 197 |
| xfail | 1 | 1 |

**The FAILED/ERROR test-ID sets are identical except for exactly one
expected rename/split**: `test_all_four_components_full_coverage`
(FAILED in the before run) was replaced by
`test_all_four_original_components_is_partial_coverage_now` and
`test_all_five_components_full_coverage` (both FAILED in the after run)
-- a `diff` of the two sorted FAILED+ERROR ID lists shows only this one
change, nothing else moved. **Both new tests fail for the exact same
pre-existing, unrelated environment reason** that already breaks 2
pre-existing tests identically in the before run
(`test_calibration_component_well_calibrated_scores_high`,
`test_calibration_component_poorly_calibrated_scores_lower`, both FAILED
in `69117744` too): some earlier test in the full-suite run leaves
PyTorch's default device set to `"mps"`, and
`pinneapple_analysis.uncertainty.calibration._normal_cdf` then raises
`TypeError: Cannot convert a MPS Tensor to float64 dtype` -- confirmed
identical, unrelated to this session's change, and already present before
it (exactly the same class of pre-existing environment gap as the
xgboost segfault worked around above). When run outside the full suite
(`pytest tests/test_physics_confidence_score.py` alone, no MPS-device
leak from an earlier file), all 23 tests -- including both new ones --
pass. The +20 collected-test delta (1711-1691) is the 19 new tests this
session added (15 in the new file, 4 net in the modified file) plus 1
unrelated new parametrized case in `test_breadth_six_packages.py`
(383->384 collected in that file alone, confirmed via a separate
`--collect-only` diff) that this session did not touch -- most likely
environment drift from this machine's shared, concurrently-used
multi-agent worktree setup between the two separate `pytest` process
invocations, not a change caused by this session.

**Scope note**: the diagonal-Mahalanobis choice over a full GMM/PCE is a
deliberate, documented tradeoff for the product's current stage (see the
module's own docstring) -- not a shortcut taken silently. `trust_gate.py`
itself (the weighted-sub-score `TrustGate.score()` API) was left
unmodified; the integration point chosen was
`PhysicsConfidenceScore`/`ConfidenceComponent`, since that is the
componentized aggregator this task's own instructions pointed at and the
one `evidence_graph.py` already consumes generically.
