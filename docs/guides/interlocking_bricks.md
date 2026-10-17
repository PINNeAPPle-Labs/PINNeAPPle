# Self-modifying brick builds

`pinneapple_design.bricks` builds structures from interlocking toy bricks, judges every build with physics and
redesigns it until it meets a goal: the longest cantilever past a table edge, a bridge that carries a load with the
fewest parts, the tallest tower that survives a push. The geometry follows the LDraw unit system, so builds open in
LDraw-compatible editors and real part shapes can be loaded from the LDraw parts library.

```python
from pinneapple_design.bricks import Assembly, Ground, analyse_assembly

a = Assembly(ground=Ground("table", [(-1000, 0, -1000, 1000)]))   # a table that ends at x = 0
a.add("brick_1x8", -4, 0, 0, colour="blue")                         # half of it past the edge
a.add("brick_1x8", 0, 0, 3, colour="yellow")                        # on top, reaching 64 mm further
r = analyse_assembly(a)
print(r.status, r.explanation)                                      # mechanism: where the centre of mass left
```

## The parts

| quantity | value | source |
|---|---|---|
| stud pitch | 8.0 mm (20 LDU) | LDraw unit system |
| plate / brick height | 3.2 mm / 9.6 mm (8 / 24 LDU) | LDraw unit system |
| stud | 4.8 mm diameter, 1.6 mm high | LDraw `stud.dat` primitive |
| part width | 8 n - 0.2 mm (0.1 mm clearance per side) | StableLego configuration (7.8 mm unit) |
| brick masses | e.g. 2x4: 2.16 g, 1x8: 3.03 g | StableLego brick library (Liu et al. 2024) |
| plate masses | shell model fitted to the brick masses (worst residual +28 % on the 1x1 brick) | `parts.fit_mass_model` |

The catalogue (`parts.CATALOG`) holds rectangular bricks and plates with their LDraw part numbers. Assemblies live on
the stud grid (x, y in stud pitches, z in plate heights). Two parts are joined where the top of one overlaps the
bottom of the other: each overlapping cell is one stud-tube contact. `Assembly.validate()` reports overlaps and parts
with no contact path to the ground; the ground is a table (smooth) or a baseplate (studs), optionally limited to
regions such as a table edge or two banks.

## The physics

**Statics** (`statics.analyse`). Every part is a rigid body. Every contact patch carries compression at its four
corners, and a stud joint also carries a clutch tension at each stud (at most `JointModel.clutch_n`) and a
horizontal force at each stud; smooth contacts carry Coulomb friction. A linear programme finds the force
distribution with the smallest peak stud utilisation. If one exists within the clutch, the build stands (the
lower-bound theorem of plasticity); if none exists even with unbreakable studs, the build is a mechanism and the
report says which support polygon the resultant left and by how much. `capacity` gives the largest factor on the
external loads, and `progressive_failure` releases the critical joints round by round.

**Joint model.** The clutch is 0.98 N per stud, the friction capacity reported for the StableLego analysis (Liu,
Deng, Wang, Liu, IEEE RA-L 9(11) 2024) and reused by BrickGPT (Pun et al., arXiv:2505.05469). No uncertainty is
published, so the sensitivity study scales it by an assumed 0.5x to 1.5x. Limits of the model:

- tension only: a joint fails when its pull-out force exceeds the clutch;
- horizontal stud forces are unbounded by default (as in StableLego); `JointModel(shear_n=...)` caps them;
- a single stud is a pivot about the vertical (no torsion);
- rigid parts and no joint compliance: the force distribution is the most favourable one, so the verdict is
  optimistic for brittle clutch;
- side-by-side parts do not push on each other (there is a 0.2 mm gap);
- contact forces act at least `edge_inset_mm` (0.5 mm, assumed) inside a patch edge. Without it the search drives
  builds onto a knife edge (centre of mass exactly over a table edge), which the statics accepts with zero margin
  and Newton tips; the closed-form checks use sharp edges (`edge_inset_mm=0`).

**Dynamics** (`dynamics.replay`). NVIDIA Newton replays a build: every cluster held together by intact studs is one
rigid body made of its boxes, released joints become box-box contacts with friction, tables have edges and a floor
below, and vertical loads are dead weights. Newton has no breakable joints, so the statics decides which joints let
go and Newton shows what happens next.

**FEA** (`fea.solve`). The voxelised build as one Hex8 solid (incompatible modes), for load paths and stresses in
the plastic. It is an upper bound on stiffness: joint compliance is not modelled.

## The loop

`search.evolve` and `search.surrogate_search` mutate builds with plain edits (add, remove, move, swap, rotate,
recolour) and grammar rules that read the last evaluation (extend toward the goal, add a counterweight, brace the
joint that failed, widen the base, mirror a half of a span). Every candidate is simulated; duplicates are looked
up. Evolution selects survivors with NSGA-II and Deb's constraint handling on the goal's objectives (for example
reach and part count). The surrogate search proposes eight times more children than it can afford, ranks them with
a Gaussian process trained on every build simulated so far (upper confidence bound on a penalised score) and
simulates only the best. `search.compare` runs both with the same evaluation budget and seeds.

## Visuals

- `visuals.evolution_timeline` + `timeline_scene(...).export(folder)`: the twin3d three.js viewer plays the
  evolution generation by generation. Build frames are coloured by stud utilisation (the load path); rejected
  candidates are replayed by Newton with the joints that let go in red. Serve the folder over http.
- `visuals.render_frames`, `write_gif`, `write_mp4`: the same frames from the software renderer.
- `visuals.render_blender`, `turntable_blender`: Blender Cycles renders (`pip install bpy`).
- `visuals.write_instructions`: step images per layer, a part list and an LDraw file with one STEP per layer.

## LDraw files and the parts library

`ldraw.write_ldr` / `write_mpd` write builds; `ldraw.read_ldr` reads .ldr and .mpd files (sub-models expanded,
quarter-turn rotations; parts off the grid or outside the catalogue are listed in `skipped`).
`ldraw.fetch_parts_library()` downloads the LDraw parts library at run time into a cache folder; it is licensed
CC BY 2.0 ("LDraw.org Parts Library") and is not copied into this repository. `ldraw.PartsLibrary(root)`
triangulates real parts for renders.

## The experiment

```bash
python examples/bricks/self_modifying_bricks.py --full       # 5 seeds x 400 evaluations per strategy and goal
python -m pinneapple_lab sweep bricks -g case=cantilever,bridge,tower
```

## Results of the full run

5 seeds, 400 physics evaluations per run and strategy (`examples/bricks/_out/report.md` has every number,
`summary.json` the raw values).

**Closed forms.** All nine checks pass (trust report APPROVED): the LP reproduces the harmonic stacking limit and
the single-block limit exactly, the stud-cantilever tension and the tower tipping push to 1e-16, the beam reactions
to 1e-15; the voxel FEA beam deflection is within 0.4 % of Timoshenko (1.4 % above Euler-Bernoulli alone, which
leaves out shear). Newton agrees with harmonic stacking at 0.9x to 0.99x (stands) and 1.01x to 1.05x (falls). It
disagrees at exactly 1.00x, the neutral equilibrium of the closed form, where it tips the stack.

Statistic: the median of the best score over the 5 seeds, a seed that found no feasible build counted as the
worst, and in brackets the range over the feasible seeds (the same rule as `comparison.png` and the report).

| goal | evolution | surrogate-guided | best build |
|---|---|---|---|
| cantilever, reach past the edge with 8 parts | 111.9 mm (95.9 to 135.9), 5/5 feasible | 103.9 mm (71.9 to 127.9), 5/5 | 135.9 mm, 34 % of the clutch |
| bridge over 96 mm carrying 1 N, parts | 5 (3 to 5), 5/5 feasible | 8 (3 to 8), 3/5 feasible | 3 parts, carries 1.46 N |
| tower that survives a 10 mN push, height | 118.4 mm (57.6 to 144.0), 5/5 | 115.2 mm (70.4 to 153.6), 5/5 | 153.6 mm, 16 parts, 1.64 x the push |

Evolution has the better median on all three goals and is the more reliable on the bridge (the surrogate-guided
search found no bridge in 2 of 5 seeds), but the margins are smaller than the range over seeds and the
surrogate-guided search found the tallest tower, so 5 seeds do not separate the two strategies. For
reference, unconnected blocks of the same length reach (L/2) H_8 = 86.7 mm (harmonic stacking), so the stud
clutch buys 57 % more reach (135.9 mm), still far below the 255 mm centre-of-mass bound.

**Sensitivity.** The best cantilever and tower stand over the whole assumed clutch band (0.5x to 1.5x) and with
stud shear capped at 0.3 N; the best bridge fails at 0.5x clutch (0.49 N per stud, load factor 0.80) and stands
from 0.75x, so its verdict depends on the joint strength.

The three best builds stand in Newton under their goal loads (the bridge with its 1 N dead weight, the tower with
the 10 mN push): largest displacement 0.00, 0.00 and 0.03 mm in 0.6 s, no tilt.

**Statics against Newton** (searched candidates, every joint intact, dt = 0.25 ms): agreement 90 % on cantilevers,
100 % on bridges, 75 % on towers; every disagreement is a statics "mechanism" that Newton keeps standing:

- cantilever (2 cases): the centre of mass is 0.2 and 0.5 mm past the inset support, inside the 0.5 mm edge inset
  that the statics assumes and Newton's sharp boxes do not;
- tower (5 cases): no equilibrium in the statics although the resultant is inside the support; Newton shows a
  slow drift of 0.1 to 0.8 mm without tilting, which points to sliding at the friction limit. The single-stud
  torsion assumption explains 2 of 40 tower mechanisms; the rest is not explained.

XPBD friction creeps under a sustained sideways load: the best tower drifts 0.23 mm at dt = 1 ms and 0.03 mm at
0.25 ms in 0.6 s while the statics says friction holds, so the cross-check uses the small step.

**Limits.** The verdicts rest on the cited 0.98 N clutch, which has no published uncertainty, on rigid parts and
the most favourable force distribution (optimistic for brittle clutch), on unbounded stud shear and on an assumed
friction coefficient. Nothing here was compared with physical builds.
