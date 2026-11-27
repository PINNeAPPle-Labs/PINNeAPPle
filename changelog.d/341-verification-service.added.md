- Physics verification as a service, first slice (#341): `pinneapple_analysis.trust.certificate.verify(fields, problem)`
  takes somebody else's prediction -- arrays, an `.npz`/`.h5` grid, a VTU/VTK/CSV/XDMF point cloud, the result of a
  case -- and returns a `Certificate`: per check, the number, the limit it was judged against, where in the domain the
  worst point is and where that number came from, as a JSON document with the `Trust` record embedded. Nothing here
  is a second engine: the residuals are `pinneapple_check.physics.equations` on `pinneapple_check.numerics`
  providers, the reading is `pinneapple_check.fields.load_result`, the units are `pinneapple_data.physical_units`,
  the in-domain level follows `pinneapple_physics.ensemble.residual_baseline`, and the domain gate is
  `pinneapple_analysis.trust.trust_gate`. The decision is `Certificate.badge()` -- `no_skill` when a measured check
  says the case left the declared domain, `fading` when a check failed or could not be measured, `good` when every
  check was measured and passed -- and a residual with neither an in-domain level nor a declared tolerance is
  `unmeasured`, never a pass. `verify` also covers what the linter's equations do not: non-finite values,
  positivity, boundary conditions, and a parameter whose unit has the wrong dimension stops the certificate before
  a residual is computed from a differently-scaled number. `calibrate` measures the two numbers the issue asks for
  -- detection rate and false alarms -- case by case against a truth the certificate cannot see: on the periodic
  advection-diffusion field, three predictions that obey the law (round-off, 0.1 % and 0.2 % of noise, residual
  0.0007 / 0.019 / 0.039) are not flagged and three that advect at the wrong speed (1.3, 1.6, 2.0 against the
  declared 1.0, residual 0.206 / 0.260 / 0.344) are, at a declared tolerance of 0.05: detection 3/3, false alarms
  0/3, while the `finite` and `units` gates flag nothing on either set -- the table separates the gate that decides
  from the ones that do not.
  The boundary is measured and stated in the tests: 0.039 passes and 0.095 is flagged, and the same field fails a
  tolerance of 0.02 -- the limit is the caller's, which is why the in-domain level of the model wins over it.
  Shallow water is checked too: a flat lake at rest is certified, and the same lake with a bumpy free surface fails
  the momentum residual, because zero velocity under a slope is not a solution. Not in this slice: the finite-volume
  face sums for an OpenFOAM case folder (its cell values are read as a point cloud), and the HTTP endpoint with the
  upload page -- the library call is the whole surface this round.
