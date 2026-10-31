- 3-D eddy currents and end effects (issue #929, third step).
  - `fem3d.team7`: TEAM workshop problem 7 with its measured tables, gmsh mesh, probes and comparison. Example
    `examples/electrodynamics/21_team7.py` refines the plate from 12 to 5 mm (60 000 to 195 000 unknowns, 44 s
    and 7.2 GB at the finest). In-phase `Bz` is within 5 to 8 % of the measurements; quadrature `Bz` within 7.5 %
    at 50 Hz and 40 % at 200 Hz, where the plate has under one element per skin depth. The computed eddy current
    on the face nearer the coil matches the table of the other face (A4-B4) within 10 to 13 %; both pairings are
    reported.
  - `solve_eddy`: the tree gauge now links its root groups (the conductor's floating potential was a null mode:
    residual 1e-4 before, 2e-14 after), and PARDISO factorises the real symmetric form by LDL^T with iterative
    refinement (7.7 GB against 13.6 GB on 197 000 unknowns). Factors are freed explicitly (`SPM3D.release()`).
  - `machines3d.endeffects`: magnet flux linked by a search coil in 3-D against 2-D for L/D from 0.25 to 2
    (`examples/electrodynamics/22_spm_end_effects_3d.py`), with an exact line integral of `A`. The loss is a fixed
    length: 0.286 mm of stack from L/D = 0.5 to 2, so `k = 1 - 0.29 mm / L` for this machine (1.4 % at
    L/D = 0.25, 0.16 % at L/D = 2).
  - Flow nodes `em.team7` and `em.spm_end_effects_3d`; `TetMesh.locate` is vectorised and caches its search tree.
