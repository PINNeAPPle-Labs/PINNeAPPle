# Wing design: least cruise drag within the structural and stall limits

Vortex lattice and friction build-up, a learned model only when its measured cost pays off, the planform optimised for cruise drag within the root-bending, stall and span limits, and the design re-verified (pinneapple_design.aero.wing_design).

Flow: `pinneapple_flow/flows/wing_design.json` (as the example flow `industrial/wing_design`). Run it with `python -m pinneapple_flow example-flows run industrial/wing_design [--quick]`; regenerate this folder with `python -m pinneapple_flow example-flows gallery industrial --entries wing_design`.

Quick configuration (parameter overrides per node): `{"wing": {"maxiter": 12, "popsize": 8}}`.

Every number below was measured by a run of the flow on the commit and date stated; none is typed in.

## quick

done in 7.6 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `wing.design.aspect_ratio` | 8.87157 | optimum, re-solved by the lattice |
| `wing.design.taper` | 0.305771 | optimum, re-solved by the lattice |
| `wing.design.twist` | -0.118131 | optimum, re-solved by the lattice |
| `wing.model.drag_N` | 393.696 | vortex lattice + friction build-up at the design |
| `wing.model.CDi` | 0.00409419 | vortex lattice + friction build-up at the design |
| `wing.model.CD0` | 0.00846214 | vortex lattice + friction build-up at the design |
| `wing.model.e` | 0.982798 | vortex lattice + friction build-up at the design |
| `wing.model.M_root_Nm` | 49231.8 | vortex lattice + friction build-up at the design |
| `wing.model.span_m` | 11.9883 | vortex lattice + friction build-up at the design |
| `wing.model.cl_local_max_landing` | 1.18001 | vortex lattice + friction build-up at the design |
| `wing.model.stall_station` | 0.666347 | vortex lattice + friction build-up at the design |
| `wing.baseline.drag_N` | 417.961 | the library's light aircraft wing (AR 7.5, taper 0.7, -3 deg) |
| `wing.cost.solve_seconds` | 0.0213451 | measured: mean of three lattice solves |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| design.stall | pass | margin 0.34 at the design, from the lattice |
| design.root_bending | pass | margin 0.1795 at the design, from the lattice |
| design.span | pass | margin 0.0117 at the design, from the lattice |
| model.converged | pass | induced drag changes by 0.09% with a lattice 1.6 x finer chordwise and 2 x spanwise |
| model.munk | pass | span efficiency e = 0.9828 <= 1 (Munk) |
| model.lifting_line | pass | lift slope 4.852 /rad against Prandtl's lifting line 5.085 /rad (4.6%; the lattice is lower at low aspect ratio, as lifting-surface theory is) |
| verify.cfd | inconclusive | an OpenFOAM polar of the section (pinneapple_design.aero.verify) is the next fidelity; not run: no OpenFOAM in this environment |

## full

done in 17.8 s (commit d83a70bc, 2026-11-03).

| metric | value | how |
|---|---|---|
| `wing.design.aspect_ratio` | 8.88825 | optimum, re-solved by the lattice |
| `wing.design.taper` | 0.308903 | optimum, re-solved by the lattice |
| `wing.design.twist` | -0.00592937 | optimum, re-solved by the lattice |
| `wing.model.drag_N` | 393.276 | vortex lattice + friction build-up at the design |
| `wing.model.CDi` | 0.00407783 | vortex lattice + friction build-up at the design |
| `wing.model.CD0` | 0.00846513 | vortex lattice + friction build-up at the design |
| `wing.model.e` | 0.98489 | vortex lattice + friction build-up at the design |
| `wing.model.M_root_Nm` | 49355.1 | vortex lattice + friction build-up at the design |
| `wing.model.span_m` | 11.9996 | vortex lattice + friction build-up at the design |
| `wing.model.cl_local_max_landing` | 1.18109 | vortex lattice + friction build-up at the design |
| `wing.model.stall_station` | 0.666347 | vortex lattice + friction build-up at the design |
| `wing.baseline.drag_N` | 417.961 | the library's light aircraft wing (AR 7.5, taper 0.7, -3 deg) |
| `wing.cost.solve_seconds` | 0.018852 | measured: mean of three lattice solves |
| `report.decisions` | 0 |  |
| `report.decisions_without_reference` | 0 |  |
| `report.items` | 1 |  |

| check | status | detail |
|---|---|---|
| design.stall | pass | margin 0.3389 at the design, from the lattice |
| design.root_bending | pass | margin 0.1774 at the design, from the lattice |
| design.span | pass | margin 0.0004313 at the design, from the lattice |
| model.converged | pass | induced drag changes by 0.08% with a lattice 1.6 x finer chordwise and 2 x spanwise |
| model.munk | pass | span efficiency e = 0.9849 <= 1 (Munk) |
| model.lifting_line | pass | lift slope 4.854 /rad against Prandtl's lifting line 5.087 /rad (4.6%; the lattice is lower at low aspect ratio, as lifting-surface theory is) |
| verify.cfd | inconclusive | an OpenFOAM polar of the section (pinneapple_design.aero.verify) is the next fidelity; not run: no OpenFOAM in this environment |
