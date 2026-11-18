- `pinneapple_design.suspension`: a double-wishbone corner designed and sized in one loop — the closed-loop multibody
  kinematics through wheel travel (camber, toe, caster, scrub, motion ratio, roll centre), the arm's loads, stress and
  Euler buckling, the bushes' compliance modes (steer and camber under a lateral force), a 3-D hexahedral check of the
  winning arm, and the bounded search for the lightest corner that meets all of them: `suspension_study()` runs it and
  returns its own checks, the 3-D field and the CAD, with the limits of the model stated. Example in
  `examples/suspension_elastokinematics/`, planner capability `suspension.design`.
