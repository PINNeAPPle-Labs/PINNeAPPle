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
