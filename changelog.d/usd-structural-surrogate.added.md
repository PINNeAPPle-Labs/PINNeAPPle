- `pinneapple_simulation.pipelines.usd_structural_surrogate`: the UR10 mounting pedestal of NVIDIA's OpenUSD asset
  (`Props/Mounts/ur10_mount.usd`) read with OpenUSD and measured for its mounting interface, a four-variable design
  family meshed with gmsh and solved with CalculiX, the result sampled onto one fixed grid, a PhysicsNeMo FNO trained
  on that grid under a `ComputeBudget`, the operator exposed to `virtual_ccx` behind its validity envelope, and the
  physics checks (peak von Mises, clamped-face adherence, load linearity) it has to pass. Ships a dashboard that
  turns the design variables into a live prediction, a real `ccx` solve on demand and a fine-tuning step on it.
- `pinneapple_simulation.external_solvers.calculix.study`: `FEModel.from_gmsh` no longer fails on macOS, where gmsh's
  console script is a `#!/usr/bin/env python` wrapper and no `python` exists; it is run with an interpreter on PATH.
