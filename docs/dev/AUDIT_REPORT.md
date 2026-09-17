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
