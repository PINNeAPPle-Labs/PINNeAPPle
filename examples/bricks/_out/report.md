# Self-modifying interlocking-brick builds

Every candidate build is checked by a static equilibrium analysis (rigid parts, stud clutch, friction); rejected candidates are explained (which joint let go, or where the resultant left its support) and replayed with the Newton rigid-body engine.

## Joint model

- Clutch per stud: 0.98 N. Source: 0.98 N per stud: StableLego (Liu et al., IEEE RA-L 9(11), 2024, doi:10.1109/LRA.2024.3414281); BrickGPT (Pun et al., arXiv:2505.05469, 2025).
- Uncertainty: none published; the sensitivity study scales it by 0.5x to 1.5x (an assumed band).
- Friction on smooth contacts: mu = 0.4 (assumed).
- Limits: tension only: a stud joint fails when its pull-out force exceeds clutch_n; shear and torsion at stud joints are carried by the stud geometry (unbounded unless shear_n is set); compression is unlimited (ABS crushing is far above these loads); rigid parts: no joint compliance, so the force distribution is the most favourable one (lower bound); a single stud is a pivot about the vertical (no torsion); contact acts at least edge_inset_mm inside a patch edge (rounded edges, no knife-edge balance).

## Checks against closed forms

Trust report decision: **APPROVED** (score 1.00, coverage 1.00).

| check | simulated | closed form | rel. error | tolerance | pass | note |
|---|---|---|---|---|---|---|
| harmonic stack (6 blocks): stability limit / closed form | 1 x offsets | 1 | 0.00e+00 | 0.001 | yes |  |
| single block overhang limit / length | 0.5  | 0.5 | 0.00e+00 | 0.001 | yes |  |
| stud cantilever brick_1x4 on 1 stud(s): tension per stud | 0.0308034 N | 0.0308034 | 1.13e-16 | 1e-06 | yes |  |
| stud cantilever brick_1x8 on 2 stud(s): tension per stud | 0.0297243 N | 0.0297243 | 1.17e-16 | 1e-06 | yes |  |
| tower of 8 2x2 bricks: tipping push at the top | 0.00940125 N | 0.00940125 | 1.85e-16 | 1e-06 | yes |  |
| plate beam: sum of support reactions | 0.114351 N | 0.114351 | 1.21e-15 | 1e-06 | yes |  |
| voxel FEA plate beam: mid-span deflection (Timoshenko) | 0.0326609 mm | 0.0327931 | 4.03e-03 | 0.02 | yes | Euler-Bernoulli alone: 0.0322 mm |
| Newton harmonic stack (4 blocks): stands iff factor < 1, factors != 1 | 7 cases | 7 | 0.00e+00 | 0.0 | yes | at exactly factor 1 (neutral equilibrium) Newton tips the stack |
| Newton table edge: brick with centre of mass inside stays, outside falls | 1  | 1 | 0.00e+00 | 0.0 | yes |  |

Newton on the harmonic stack (offsets scaled by a factor; the closed form stands for factor <= 1):

| factor | top-block drop [mm] | Newton | closed form | agree |
|---|---|---|---|---|
| 0.9 | 0.17 | stands | stands | yes |
| 0.95 | 0.18 | stands | stands | yes |
| 0.98 | 0.21 | stands | stands | yes |
| 0.99 | 0.11 | stands | stands | yes |
| 1.0 | 108.80 | falls | stands | **no** |
| 1.01 | 108.80 | falls | falls | yes |
| 1.02 | 108.80 | falls | falls | yes |
| 1.05 | 108.80 | falls | falls | yes |

## Statics against Newton on searched candidates

Candidates the statics calls stable (nothing should move) or a mechanism (something should), replayed by Newton with every stud joint intact under the goal's loads. Joint failures are left out: Newton has no clutch model to judge them.

| goal | stable, stays | stable, moves | mechanism, moves | mechanism, stays | agreement | dt [s] |
|---|---|---|---|---|---|---|
| cantilever | 10 | 0 | 8 | 2 | 90% | 0.00025 |
| bridge | 10 | 0 | 10 | 0 | 100% | 0.00025 |
| tower | 10 | 0 | 5 | 5 | 75% | 0.00025 |

Where the simulations disagree with each other or with the closed forms:

- cantilever: 0 of 40 statics mechanisms stand once single-stud joints get a little torsion capacity (a lone stud is a pivot in the statics; Newton welds every intact joint).
- bridge: 0 of 40 statics mechanisms stand once single-stud joints get a little torsion capacity (a lone stud is a pivot in the statics; Newton welds every intact joint).
- tower: 2 of 40 statics mechanisms stand once single-stud joints get a little torsion capacity (a lone stud is a pivot in the statics; Newton welds every intact joint).
- XPBD friction creep under the sustained 10 mN push on the best tower (statics: friction holds): 0.23 mm at dt = 0.001 s, 0.03 mm at dt = 0.00025 s over 0.6 s; it shrinks with dt, so it is a solver artefact.
- Harmonic stack at exactly the closed-form limit (neutral equilibrium): Newton tips it (top block drop 108.8 mm); at 0.99x it stands and at 1.01x it falls.

## Search strategies at equal evaluation budget

Statistic: per strategy, the median of the best score over the seeds, a seed that found no feasible build counted as the worst (the same rule as the chart), and the range over the feasible seeds.

### cantilever

| strategy | median [mm] | range of feasible seeds | feasible seeds | per seed |
|---|---|---|---|---|
| evolution | 111.9 | 95.9 to 135.9 | 5/5 | 127.9, 135.9, 95.9, 111.9, 103.9 |
| surrogate | 103.9 | 71.9 to 127.9 | 5/5 | 103.9, 127.9, 119.9, 71.9, 87.9 |

Budget 400 physics evaluations per run, seeds [0, 1, 2, 3, 4].

Best build (evolution, seed 1): 8 parts, 135.9 [mm]. stands: peak stud tension 0.333 N = 34% of the 0.98 N clutch
Closed-form references: single_block 31.9 mm, harmonic_unconnected 86.7 mm, com_bound 255.2 mm.

Sensitivity of the best build to the joint model:

| clutch [N] | shear cap [N/stud] | result | load factor | utilisation |
|---|---|---|---|---|
| 0.49 | - | stands | - | 0.679 |
| 0.49 | 1 | stands | - | 0.679 |
| 0.49 | 0.3 | stands | - | 0.679 |
| 0.73 | - | stands | - | 0.453 |
| 0.73 | 1 | stands | - | 0.453 |
| 0.73 | 0.3 | stands | - | 0.453 |
| 0.98 | - | stands | - | 0.339 |
| 0.98 | 1 | stands | - | 0.339 |
| 0.98 | 0.3 | stands | - | 0.339 |
| 1.23 | - | stands | - | 0.272 |
| 1.23 | 1 | stands | - | 0.272 |
| 1.23 | 0.3 | stands | - | 0.272 |
| 1.47 | - | stands | - | 0.226 |
| 1.47 | 1 | stands | - | 0.226 |
| 1.47 | 0.3 | stands | - | 0.226 |

### bridge

| strategy | median [parts] | range of feasible seeds | feasible seeds | per seed |
|---|---|---|---|---|
| evolution | 5.0 | 3.0 to 5.0 | 5/5 | 5.0, 3.0, 5.0, 5.0, 5.0 |
| surrogate | 8.0 | 3.0 to 8.0 | 3/5 | 3.0, -, 8.0, -, 3.0 |

Budget 400 physics evaluations per run, seeds [0, 1, 2, 3, 4].

Best build (evolution, seed 1): 3 parts, 3.0 [parts]. carries 1.462 N at mid-span (1.46 x the 1.0 N load)

Sensitivity of the best build to the joint model:

| clutch [N] | shear cap [N/stud] | result | load factor | utilisation |
|---|---|---|---|---|
| 0.49 | - | joint_failure | 0.797 | 1.31 |
| 0.49 | 1 | joint_failure | 0.797 | 1.31 |
| 0.49 | 0.3 | joint_failure | 0.797 | 1.31 |
| 0.73 | - | stands | 1.13 | 0.87 |
| 0.73 | 1 | stands | 1.13 | 0.87 |
| 0.73 | 0.3 | stands | 1.13 | 0.87 |
| 0.98 | - | stands | 1.46 | 0.653 |
| 0.98 | 1 | stands | 1.46 | 0.653 |
| 0.98 | 0.3 | stands | 1.45 | 0.653 |
| 1.23 | - | stands | 1.79 | 0.522 |
| 1.23 | 1 | stands | 1.79 | 0.522 |
| 1.23 | 0.3 | stands | 1.64 | 0.522 |
| 1.47 | - | stands | 2.13 | 0.435 |
| 1.47 | 1 | stands | 2.13 | 0.435 |
| 1.47 | 0.3 | stands | 1.83 | 0.435 |

### tower

| strategy | median [mm] | range of feasible seeds | feasible seeds | per seed |
|---|---|---|---|---|
| evolution | 118.4 | 57.6 to 144.0 | 5/5 | 118.4, 57.6, 76.8, 128.0, 144.0 |
| surrogate | 115.2 | 70.4 to 153.6 | 5/5 | 153.6, 115.2, 70.4, 89.6, 137.6 |

Budget 400 physics evaluations per run, seeds [0, 1, 2, 3, 4].

Best build (surrogate, seed 0): 16 parts, 153.6 [mm]. 153.6 mm tall, survives 16.4 mN at the top (1.64 x the push); stands: peak stud tension 0.202 N = 21% of the 0.98 N clutch

Sensitivity of the best build to the joint model:

| clutch [N] | shear cap [N/stud] | result | load factor | utilisation |
|---|---|---|---|---|
| 0.49 | - | stands | 1.64 | 0.412 |
| 0.49 | 1 | stands | 1.64 | 0.412 |
| 0.49 | 0.3 | stands | 1.64 | 0.412 |
| 0.73 | - | stands | 1.64 | 0.274 |
| 0.73 | 1 | stands | 1.64 | 0.274 |
| 0.73 | 0.3 | stands | 1.64 | 0.274 |
| 0.98 | - | stands | 1.64 | 0.206 |
| 0.98 | 1 | stands | 1.64 | 0.206 |
| 0.98 | 0.3 | stands | 1.64 | 0.206 |
| 1.23 | - | stands | 1.64 | 0.165 |
| 1.23 | 1 | stands | 1.64 | 0.165 |
| 1.23 | 0.3 | stands | 1.64 | 0.165 |
| 1.47 | - | stands | 1.64 | 0.137 |
| 1.47 | 1 | stands | 1.64 | 0.137 |
| 1.47 | 0.3 | stands | 1.64 | 0.137 |

## Files

- comparison chart: `comparison.png`
- cantilever glTF: `best_cantilever.glb`
- cantilever STL: `best_cantilever.stl`
- cantilever LDraw: `best_cantilever.ldr`
- cantilever instructions: `instructions_cantilever/instructions.md`
- bridge glTF: `best_bridge.glb`
- bridge STL: `best_bridge.stl`
- bridge LDraw: `best_bridge.ldr`
- bridge instructions: `instructions_bridge/instructions.md`
- tower glTF: `best_tower.glb`
- tower STL: `best_tower.stl`
- tower LDraw: `best_tower.ldr`
- tower instructions: `instructions_tower/instructions.md`
- cantilever viewer: `viewer_cantilever`
- cantilever GIF: `evolution_cantilever.gif`
- cantilever MP4: `evolution_cantilever.mp4`
- cantilever Cycles render: `best_cantilever_cycles.png`
- cantilever Cycles load path: `best_cantilever_load_path.png`
- bridge viewer: `viewer_bridge`
- bridge GIF: `evolution_bridge.gif`
- bridge MP4: `evolution_bridge.mp4`
- bridge Cycles render: `best_bridge_cycles.png`
- bridge Cycles load path: `best_bridge_load_path.png`
- tower viewer: `viewer_tower`
- tower GIF: `evolution_tower.gif`
- tower MP4: `evolution_tower.mp4`
- tower Cycles render: `best_tower_cycles.png`
- tower Cycles load path: `best_tower_load_path.png`
- turntable GIF: `turntable_cantilever.gif`
- turntable MP4: `turntable_cantilever.mp4`
