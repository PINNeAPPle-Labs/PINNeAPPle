- `pinneapple_design.suspension`: the corner's numbers now come with their error — a second kinematics formulation
  (`pose_from_angles`: joint angles instead of closure unknowns, agreeing with `pose` to 1e-11 deg over the sweep), the
  arm's leg solved by an external solver (`calculix.fem_lower_arm_calculix`: gmsh builds the B-rep, `ccx` solves it,
  within 1.8 % of beam theory and of the library's own hexahedra), a tube against Timoshenko's cantilever built by the
  same pipeline (−0.42 % at second-order elements), and the cases where the answer is a closed-form zero (a
  parallelogram gains no camber, the compliance vanishes as the bushes stiffen). `suspension_verify` reports the three
  cheap ones among its own checks; `verification.run_all()` runs every case and returns the measurement beside its
  reference. New `FEModel.from_gmsh_model` builds a mesh from the gmsh Python API, so a parametric OpenCASCADE part
  stays parametric.
