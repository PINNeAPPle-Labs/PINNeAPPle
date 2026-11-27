- Large Physics Model pipeline (`pinneapple_neural.lpm_pipeline`) and the first five families
  (`pinneapple_neural.lpm_families`): a design space becomes a dataset and then a trained model, in five stages that
  each record what they did. `draw` samples a geometry and then several operating points for that same body, `build`
  turns one design into the library's `lpm.Design` (point cloud + normalised conditioning + target), `generate_dataset`
  writes one `.npz` per design plus a manifest with the reference, its citation, its assumptions and every SHA-256,
  `train` fits with batched design-space training and splits **by geometry** (a held-out body is never a training
  body), and `model/card.json` (schema `pinneapple.lpm.family_card/1`) records what the run measured and what it did
  not. Five families ship, each with a reference that someone else wrote: `pp_fin` (1-D fin theory on the
  plate-fin heat sink network), `pp_airfoil` (Joukowski conformal map with the Kutta condition — exact potential
  flow), `pp_nozzle` (quasi-1-D isentropic flow), `pp_exchanger` (Kern + Gnielinski with the 1-2 ε-NTU energy
  equations) and `pp_bracket` (immersed finite-element elasticity, the solver-backed one). Every family is checked
  against an identity that does not come from itself: lift against ρUΓ with zero drag, the area–Mach round trip,
  ε-NTU against its closed form, the fin equation, and load linearity plus mesh convergence of the FEM.
- `LargePhysicsModel.remember_training_set` stores **distinct** geometry codes. A design space runs several operating
  points on one body, and those share a code exactly; the duplicates filled the low end of the nearest-neighbour
  distances the OOD rule measures itself against, so the reference collapsed toward zero and every held-out body was
  flagged as out of distribution.
- `examples/lpm_five_models` (data, fit, held-out error and OOD report for the five families) and
  `docs/guides/lpm_pipeline.md`.
- `pinneapple_neural.lpm_real_data`: the same pipeline fitted on a **published** campaign instead of targets this
  repository computed. `pp_airfoil_rans` reads `airfoil_rans_cst_re4e6_fields` — 436 steady 2-D OpenFOAM RANS runs
  (k-omega SST, y+ < 1) over 100 CST shapes at four incidences plus four NACA sections at nine, cc-by-4.0 — and
  trains on the wall ring of the shared O-grid, with surface Cp as the target and the incidence as the condition.
  `FieldFamily.specs()` and `published_split()` let a family supply the designs and the split, so the campaign's own
  held-out shapes stay held out instead of being re-drawn here. `data_gates()` refuses a run the campaign flags
  unsteady, whose wall pressure does not integrate to the campaign's published Cl, or whose Cp is outside the range a
  RANS section can produce, and reports every refusal with its reason. `Reference` gained `provenance` and `licence`,
  both repeated in every card.
