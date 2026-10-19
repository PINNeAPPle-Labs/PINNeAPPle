# Validation gallery: one worked example per model family and per physics

`examples/validation/` shows how to validate your kind of model on your kind of physics. Every example has the same
structure:

1. Train or load the model, with data from the library's reference solvers or the zoo.
2. Run `pp.validate`.
3. Read the trust card. `validation.gallery.conclude` writes `report.json`, `trust_card.json`, `report.sarif` and
   `failure.json` to `examples/_out/validation/<example>/`.
4. Run `pinneapple check` in CI.
5. See one real failure the checks catch, on a broken variant of the same model.

Everything runs on a CPU. CI runs the whole gallery at smoke size (`.github/workflows/validation-gallery.yml`; under
3 minutes on a 4-core machine) and fails if an example stops catching its failure. Run one example with
`python examples/validation/<file>.py [--full]`.

| # | Model family | Physics | Validated with | Failure caught |
|---|---|---|---|---|
| 01 | PINN | heat conduction (manufactured) | `{"pinn": poisson_2d()}` | boundary loss weight 0: PDE satisfied, walls not (`pinn.boundary`) |
| 02 | FNO, as an ONNX file | porous flow (Darcy) | `"darcy_2d"` on `fno.onnx` | permeability contrast 100 / 1 outside the training envelope (`applicability.envelope`) |
| 03 | FNO and DeepONet steppers | reaction–diffusion (Allen–Cahn) | `{"rollout": ...}` | DeepONet: spurious high-wavenumber energy before the pointwise error grows (`rollout.spectrum`) |
| 04 | MeshGraphNet | linear elasticity (3-D FEM cantilever) | tabular spec on the tip deflection | 30 kN query, 3× the largest training load: 4.2 mm predicted, 5.6 mm by FEM (`applicability.range`) |
| 05 | POD + DMD | incompressible flow (cylinder wake, LBM) | `{"rollout": ...}` | ROM fitted on the growing transient (\|λ\| = 1.008) keeps amplifying the wake (`rollout.invariants`) |
| 06 | POD + Operator Inference | weather-like advection | `{"rollout": ..., "conserved": ["mass"]}` | basis cut at 95 % of the energy: mass exact, small scales smeared (`rollout.spectrum`) |
| 07 | MLP (zoo) | forced convection (heat sink) | zoo card | 0.2 m/s and 600 W, outside the card's range (`applicability.range`) |
| 08 | Gaussian process | convection-cooled plate | tabular spec with relations | raw-feature GP: hotter with a better conductor, power linearity broken by 36 % (`metamorphic`) |
| 09 | Neural ODE vs Hamiltonian NN | molecular dynamics (Lennard-Jones, NVE) | `{"rollout": ..., "invariants": {"energy": E}}` | plain neural ODE: energy drifts 19 % although its training loss is lower (`rollout.invariant.energy`) |
| 10 | Diffusion model | porous flow, a sample per draw | `{"generative": {"physics": "darcy"}}` | joint (a, u) generator: pressures that do not solve the PDE for their permeability (`data.physics`) |
| 11 | PhysicsNeMo FNO (external) | 2-D magnetostatics, `-div(ν grad A) = J` | `{"pde": "darcy", "f": J}` | iron with μr = 1000, outside the training envelope (`applicability.envelope`) |
| 12 | Least squares vs MLP | structural dynamics (cantilever modes) | tabular spec with scaling laws | MLP breaks f ∝ √(E/ρ) by 6 % (`metamorphic`); the power law recovers the exponents exactly |

`examples/validation/pinneapple-check.toml` runs two of the models (12 and the zoo model of 07) through
`pinneapple check` with the same relations and range checks. The other examples write `report.sarif`, which the
GitHub Action uploads to code scanning.

## Library additions made for the gallery

- `pp.validate(sampler, {"generative": {...}})`. A generative model is a sampler `n -> {column: array}`. Its samples
  are audited like a dataset: every sample against the PDE, diversity (mode collapse) and memorisation (generated
  samples that copy training samples).
- Rollouts of state vectors (`ndim = 0`, e.g. particle systems or ROM coefficients) and user invariants
  (`"invariants": {name: f(traj, ndim)}`) in `pp.validate` rollout specs.
- `numerical_solvers.reaction_diffusion.ReactionDiffusion1D`: Allen–Cahn and Fisher–KPP with ETDRK4. It is checked
  against the stationary tanh interface and the Bramson-corrected KPP front speed.
- `pinneapple_neural.utils.fit_supervised`: a minimal Adam loop on `(x, y)` tensors.

## Honest results

- **The FNO of example 02 is rejected at smoke size.** Its net boundary flux misses the source by 29 %, it has too
  much high-wavenumber energy and it breaks the reflection symmetry by 10 %. The gallery reports that rather than
  tuning it away; the envelope failure is caught regardless.
- **Spectrum floor.** On Allen–Cahn, every network "failed" `rollout.spectrum` until the band was restricted to
  shells with energy above 1e-6 of the peak (amplitudes above 0.1 %). Below that the reference holds only round-off.
  This is now the default of `spectral_drift`.
- **Data loss and physics disagree.** In example 09 the neural ODE fits the training data better than the
  Hamiltonian network (loss 0.011 vs 0.048) and still drifts in energy. In example 12 the MLP passes a loose error
  check on some test sets while breaking the scaling laws.
- **Two solvers were not used, because they failed a sanity check:**
  - `generate_magnetic_flux_leakage_synthetic`: the peak leakage signal is not monotonic in defect depth and changes
    with the grid (36 × 14, 72 × 28, 144 × 56).
  - `eddy_current_fdm` at ω = 0: the on-axis field of a coil was a fraction of the closed-form current-loop field.
    The solver was right; `annular_current_source` normalised the density by the nominal band area, so the current
    it injected depended on how the band fell on the grid. It now injects exactly N·I, and the on-axis field matches
    the loop formula (`tests/test_em_core.py`). Examples 11 and 08 still use the Darcy operator and the heat plate.
- Smoke sizes are small. The FNO and DeepONet numbers (horizons, spectra) change with training length; the tests
  assert that each failure is caught and that the good model does not fail the same check, not the exact values.
