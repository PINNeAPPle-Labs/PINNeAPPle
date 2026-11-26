- `examples/end_to_end/airfoil_trust`: a complete run of the verification pipeline on a NACA four-digit family in
  OpenFOAM, with the surrogate as the vehicle rather than the subject -- a mesh family whose grid-convergence study
  measures the reference's own discretisation error, a maximin design of experiments split by design, a POD+MLP
  ensemble checked by the model gates, an evidence domain that answers four queries a marginal-range check cannot tell
  apart, camber opened as a new parameter dimension with its own campaign, evidence added to a frozen model through
  `extend`, and the dossier of `validation_case` / `dossier` as the report. New modules:
  `pinneapple_autolab.domains.airfoil` (`AirfoilAero`, `MESH_LEVELS`, `sample_lattice`), `.airfoil_study` (`PodSurrogate`,
  `physics_checks`, `input_sensitivity`, `calibration_metrics`) and `pinneapple_simulation.numerical_solvers.airfoil2d`
  (gmsh quad/hex meshing of a section, `simpleFoam` case writer, coefficient and probe readers).
  `pinneapple_autolab.study.mesh_study.grid_convergence` gained an optional `sizes=` argument for a family refined in
  fewer than three directions, because the isotropic `cells**(-1/3)` convention overstates the refinement of a
  one-cell-thick extruded mesh. A run whose probe samples are not inside the mesh is now rejected instead of being
  recorded: OpenFOAM reports the points it cannot locate as `-1e300`, which is a number like any other downstream.
- `ValidationDossier.render_markdown` now carries the applicability, uncertainty, limitations and run-provenance
  sections, as `render_text` and `render_html` already did: the Markdown report of a validation case was silently
  dropping the parts of the record that say what the model may *not* be used for.
