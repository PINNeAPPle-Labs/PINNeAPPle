- **Related work, and what has been reproduced** (#403): `docs/guides/related_work_reproduction.md` indexes the external
  lines the library builds on — X-TFC (Schiassi, De Florio, Furfaro, Mortari), Ashton's AhmedML/DrivAerML/WindsorML,
  NVIDIA PhysicsNeMo, Warp, Omniverse/Cosmos/Isaac and Earth2Studio — with the rule that a line counts as *reproduced*
  only when a test in this repository computes the published quantity at a declared tolerance, and "declared"
  otherwise. The one reproduced line carries its numbers: X-TFC on the heat equation closes to 5,75e-4 / 5,62e-4 /
  3,05e-4 at `t = 0,25 / 0,5 / 1` against the closed form, with the initial and boundary conditions exact to machine
  precision by construction (asserted at 1e-9 over three seeds), and the Burgers sharp-layer limit is recorded rather
  than smoothed over.
