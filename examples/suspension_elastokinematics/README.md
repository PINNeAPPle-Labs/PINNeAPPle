# Suspension design and sizing: one corner, three models, one loop

A double-wishbone front corner, designed and sized the way a chassis engineer does it — except that the three models
that must agree are one program and the loop closes by itself:

```
python examples/suspension_elastokinematics/01_corner_design.py            # the full search and the check
python examples/suspension_elastokinematics/01_corner_design.py --smoke    # a short search, for CI
```

```
python -c "from pinneapple_design.suspension import suspension_study; suspension_study(out_dir='runs/corner')"
```

## The problem

Place the hardpoints of a double wishbone and choose the arm's tube and the bushings' rate so that the corner is
**light** (the objective) and, at the same time:

| requirement | what it is | what it costs if it is missed |
|---|---|---|
| static margin ≥ 1.5 | outer-fibre stress against yield at the worst load case | the arm yields on a pothole |
| Euler factor ≥ 1.5 | buckling of the compressed member | the arm folds |
| camber gain in a band (−0.33 … −0.27 °/50 mm) | how much camber the wheel gains as it rises | grip and tyre temperature, corner to corner |
| bump steer ≤ 0.10 °/50 mm | the wheel steering itself as it moves | the car tramlines and follows the road |
| toe compliance ≤ 0.45 °/kN | the wheel steering when pushed sideways | wander, and a rear axle that steers under braking |
| wheel rate ≥ 18 N/mm | the ride rate the spring and the motion ratio give | the car sits on its bump stops |

The last two are **elastokinematic**: they exist because the bushings are rubber. And they fight the others — a stiffer
bushing fixes the steer and transmits more road harshness, a thinner tube saves mass and loses the margin, a taller
upper arm gains camber and changes the roll centre.

## What runs

1. **CAD** — `hardpoints.py` holds the corner's geometry (six hardpoints, the spring, the tyre, the tube); `geometry.py`
   turns the winning numbers into a solid model (`.glb`), the arms as tubes of the section the design chose.
2. **Multibody** — `kinematics.py` solves the closed loop (lower arm angle, upper arm angle, upright spin about the
   kingpin) by Newton at every wheel-travel position and reads the wheel plane off the solved pose: camber, toe, caster,
   KPI, scrub radius, trail, half track, motion ratio, wheel rate and the axle's roll centre.
3. **Structure** — `loads.py` turns five load cases at the contact patch into the force each arm carries (the upper arm
   is a two-force member, the lower arm takes the bending); `structure.py` checks the tube for stress and Euler buckling
   and, for the winner, builds the leg as 8-node hexahedra and solves the 3-D elasticity problem
   (`SolidFEM`, incompatible modes) against beam theory.
4. **Elastokinematics** — `compliance.py` spans what the bushes allow with six modes, feeds each of them through the
   same closed-loop solver and solves the elastic equilibrium under a lateral force, a braking force and an aligning
   torque: steer and camber per kN, and which bush causes them.
5. **The loop** — `design.py` searches the five variables (tube OD, wall, bush rate, upper-arm inner-pivot height, tie
   rod inner height) with a bounded compass search: 5.32 kg nominal → **4.77 kg**, feasible, in 214 candidates.
6. **The check** — `study.py` re-checks the winner with the finite-element model, writes the 3-D field of von Mises on
   the deformed leg, the four figures and the JSON, and returns its own checks with the numbers they measured.

## The numbers the nominal corner starts from

```
static camber        -1.00 deg          camber gain        -0.30 deg / 50 mm
caster               +2.68 deg          bump steer         -0.53 deg / 50 mm   (fails: the tie rod is too low)
KPI                  +7.13 deg          toe compliance      0.53 deg / kN      (fails: the bushes are soft)
scrub radius        +25.6 mm            wheel rate         21.95 N/mm
motion ratio          0.581             roll centre         -4.5 mm
```

The search's answer: a 41.7 × 2.0 mm tube, 1057 N/mm bushes, the tie rod 4.2 mm lower and the upper arm 6.4 mm higher
— 4.77 kg for both A-arms and their bushes, with the outer fibre at 280 MPa of the 280 MPa allowed (a static margin of
1.500), the steer compliance at 0.450 °/kN of the 0.45 allowed, and the camber gain at −0.323 °/50 mm inside its band. The
finite-element peak (234 MPa) agrees with beam theory (230 MPa) to 2 %, and the whole loop is deterministic: the same
budget gives the same design, which is what `suspension_verify` re-checks together with the mesh.

## What it does not answer

- linear elasticity, small displacements: no plasticity, contact, welds or heat-affected zones;
- static load cases: a margin against yield is **not** a fatigue life, and the corner's inertia and the damper's
  reaction are outside the cases;
- bushings linear and uncoupled; no rate, temperature or ball-joint friction;
- one corner of a symmetric axle: no roll stiffness, steering system or aerodynamics;
- the arm is a uniform tube on a clamped root with the ball joint's offset as a couple, not a forged production arm.

The model says all of this itself: `suspension_study()["limits"]`.

## Where the code and the numbers live

- library: `pinneapple_design/suspension/` (`hardpoints`, `kinematics`, `loads`, `structure`, `compliance`, `design`,
  `geometry`, `study`);
- tests: `tests/test_suspension_design.py` (closure invariants, mesh volume, finite elements against the beam, Euler
  load, compliance scaling with the bush rate, the loop's feasibility and determinism, the study's files);
- flow node and planner capability: `pinneapple_flow/nodes_ext/suspension.py` (the Studio proposes it from a sentence);
- this example's manifest: [`experiment.json`](experiment.json).
