- `pinneapple_simulation.electromagnetics.machines3d` (issue #929, second step): the surface-PM machine in 3-D on the
  edge-element solver.
  - `SPM3D` extrudes the polar mesh, with magnet skew and end regions; the mesh has no handedness and one mesh
    serves every rotor angle and skew. It also gives Arkkio torque and multi-slice 2-D on the same mesh.
  - `skew.skew_study` compares step and continuous skew in 3-D against multi-slice, with checkpoints.
  - `examples/electrodynamics/20_spm_skew_3d.py` and Flow node `em.spm_skew_3d`. Against the straight 3-D
    cogging, multi-slice is within 0.5 % for step skew and 3.8 % for continuous skew. The best skew (continuous
    over one period) leaves 0.062 N m in 3-D against 0.045 N m in multi-slice.
