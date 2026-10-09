# Aircraft Design Optimizer

**A single-aisle airliner (A320 / 737 class), searched by a genetic algorithm, checked in OpenFOAM 3D, shown with its
flow fields.**

Day 13 of the PINNeAPPle 30-app program. Enter a mission (passengers, range, cruise altitude, engine TSFC) and the
requirements; NSGA-II searches 13 numbers (the wing section: 6 CST weights; the wing: area, aspect ratio, taper, sweep,
twist, position; the cruise Mach) for less CO₂ per passenger-km, a faster cruise and a slower approach.

```
wing section (CST) ── 2D OpenFOAM → MeshGraphNet + MLP ensemble (section polars, below)
        │
airliner: fuselage, swept wing with sharklets, turbofans, tails ── glTF / USD / STL
        │
compressible vortex lattice (wing + tail, trimmed) · Korn wave drag · friction at flight Re · transport weights
· Breguet mission · slats/flaps Vref · static margin, stall station, buffet margin, fuel volume, 36 m gate
        │
NSGA-II (48 × 40 = 1920 airliners, ~100 s) → Pareto front, picks, every failing design with its reason
        │
OpenFOAM 3D (snappyHexMesh half model, k-ω SST) for the A320 class and the balanced pick
        → skin Cp and Cf, streamlines, symmetry plane and wake slices, in the browser and in Blender (Cycles)
```

## The airliner model, checked

**Theory**: elliptic loading gives e = 1.0000 in the Trefftz plane; rectangular AR 6: lift slope 4.21 /rad, e 0.984;
lift slope of swept wings at Mach 0.6–0.78 within 2–3 % of DATCOM.

**The A320ceo**, the one calibration (miscellaneous drag 0.0018, wing mass × 1.5 for high lift), kept for every design:

| | model | published |
|---|---|---|
| MTOW · OEW | 73.5 t · 42.3 t | 73.5 t · ≈42.4 t |
| Wing mass | 8.9 t | ≈8.8 t |
| Cruise L/D at Mach 0.78 | 16.4 | ≈16–17 |
| Approach speed Vref | 131 kt | ≈133–138 kt |
| CO₂, 180 passengers × 4800 km | 51.7 g/pkm | — |

**Search** (default mission, 1920 airliners, 920 meet every requirement, 151 on the Pareto front): the greenest
feasible design burns −13 % CO₂ per passenger-km at Mach 0.735; the balanced pick −7.5 % at Mach 0.80 with the same
130 kt approach; the fastest cruises at Mach 0.81; the short-field pick lands at 119 kt.

**OpenFOAM 3D** (half model, ~0.4 M cells, simpleFoam k-ω SST with wall functions, approach at 70 m/s and α 4°, sea
level; nacelles and sharklets left out of the CFD geometry):

| Aircraft | CL vortex lattice (wing + tail) | CL OpenFOAM: wing · aircraft | CD OpenFOAM = pressure + friction |
|---|---|---|---|
| A320 class | 0.558 | 0.501 · 0.589 | 0.0462 = 0.0314 + 0.0148 |
| Balanced pick | 0.605 | 0.555 · 0.640 | 0.0472 = 0.0326 + 0.0146 |

The difference between the whole aircraft and the wing (~0.09) is mostly the fuselage's lift, which the lattice
leaves out; both rank the two aircraft the same way.
Fields in the app (Surface, Streamlines, Slices) use the jet scale, blue low to red high, with a colour bar.

Reproduce: `tools/verify3d_airliner.py WORK --label "A320 class"` (snappyHexMesh + simpleFoam, ~12 min on 4 cores,
writes `model/cfd3d/<id>.npz` and `model/verification3d.json`); `tools/render_airliner.py` (Blender renders into
`model/renders/`).

## The wing-section engine (2D)

The airliner's wing section comes from this loop, the original Day 13 engine for a light aircraft (Cessna 172
class; still served by `POST /api/optimize`):

```
airfoil (6 CST weights) + wing area
   → O-grid, 20k cells, the same topology for every design
   → OpenFOAM simpleFoam, k-ω SST resolved to the wall (y+ < 1), Re 4·10⁶   (436 runs, −2° … 14°)
   → surrogates: MeshGraphNet on the CFD grid (pressure, velocity, turbulence + Cl, Cd, Cm)
                 and an ensemble of 5 MLPs (fast, with an uncertainty)
   → NSGA-II with constraint domination: top speed ↑, CO₂ per 100 km ↓, stall speed ↓
   → flyable / not flyable (and why), the Pareto front, picks
   → OpenFOAM polars for the front → back into the training set (active learning)
```

- **Objectives**: top speed at full power at cruise altitude; kg of CO₂ per 100 km at cruise speed (fuel from the
  BSFC, × 3.1 kg CO₂/kg avgas); stall speed of the clean wing (shorter runways, gentler landings). Speed and CO₂ both
  come from low drag; they fight the low-speed side, so the front is speed vs stall speed, coloured by CO₂.
- **Requirements** (a design that misses one is not flyable): stall speed ≤ 61 kt (the CS-23 single-engine limit,
  applied to the clean wing), thickness ≥ 11 % (spar depth, fuel), |Cm c/4| ≤ 0.10 in cruise (trim), cruise
  cl ≤ 70 % of cl max (gusts, turns). Invalid shapes (crossing or wavy surfaces) are rejected before evaluation.
- **Trust**: five MLPs vote; when they disagree the design is marked "surrogate unsure" and kept off the front. For any
  design the graph network gives the flow field and a second, independent set of coefficients.
- **Verify it yourself**: every design downloads as a ready OpenFOAM case (mesh, setup, a numpy script for Cl/Cd/Cm).

## Validation

**CFD, before it is used** (this grid: O-grid 228 × 88 cells, far field at 200 chords, k-ω SST, y⁺ < 1):

| Case | OpenFOAM here | Reference |
|---|---|---|
| NACA 0012, Re 6·10⁶, α 0°: drag | Cd 0.00812 | ≈ 0.0081 (NASA Turbulence Modeling Resource, SST) |
| NACA 0012, α 10°: lift / drag | Cl 1.086 / Cd 0.0140 | ≈ 1.09 / ≈ 0.0123 |
| NACA 2412, Re 4·10⁶: zero-lift angle, lift slope | −2.0°, 0.110 /° | −2.1°, ~0.105 /° (NACA wind tunnel) |

Drag at high lift reads ~14 % high on this grid (the far field at 25 chords gave +48 %, so it is at 200); the same grid
serves every design, so rankings hold. Aircraft model on the NACA 2412 polar (the Cessna 172's airfoil): 126 kt top
speed, 14 L/100 km at 107 kt (the real aircraft: ~124 kt, ~15 L/100 km).

**Training set**: 436 OpenFOAM runs (100 Latin-hypercube airfoils × 4 angles + 4 NACA polars × 9), ~58 s each on
one core, 0 failures; 32 runs oscillate near stall (averaged), 8 in deep stall are dropped (Cl swing > 0.1).

**Surrogates on airfoils they never saw** (15 held-out airfoils at all their angles, and the NACA airfoils):

| Model | Cl MAE | Cd error, mean / 90 % | Cm MAE |
|---|---|---|---|
| MLP ensemble (5), held-out airfoils | 0.003 | 0.5 % / 1.0 % | 0.0005 |
| MLP ensemble, NACA 2412 / 4412 / 0012 / 4415 | 0.003 | 0.6 % / 1.1 % | 0.0002 |
| MeshGraphNet, held-out airfoils | 0.011 | 1.5 % / 3.9 % | 0.0008 |
| MeshGraphNet, NACA airfoils | 0.010 | 2.0 % / 5.1 % | 0.0011 |

MeshGraphNet fields on the held-out airfoils: RMSE 0.036 in p (range 5.4), 0.035 / 0.022 in Ux / Uy (U∞ = 1).

**Optimizer picks run in OpenFOAM** (6 designs along the Pareto front, 9 angles each, default aircraft): top speed
within 0.1 kt, CO₂ within 0.1 %, stall speed within 0.3 kt of the surrogate, 6/6 with the same flyable verdict. The 54
runs went back into the training set. Against the NACA 2412 on the reference wing (126.4 kt, 31.4 kg CO₂/100 km,
stall 53.7 kt), the fastest flyable design is +11 kt and −11 % CO₂, with a 12 m² wing and a stall speed at the
61 kt limit; the balanced pick is +4.5 kt and −6 % CO₂ at 56 kt stall.

## Reproduce (OpenFOAM v1912+, ~1 min per run on one core)

```bash
python apps/aero_optimizer/tools/generate_dataset.py runs/ --shapes 100 --alphas 4 --workers 4      # 436 runs
python apps/aero_optimizer/tools/train_surrogate.py runs/ apps/aero_optimizer/model/surrogate.pt --gnn-minutes 60
python apps/aero_optimizer/tools/verify.py apps/aero_optimizer/model/surrogate.pt verify/ --designs 6 --round 1
python apps/aero_optimizer/tools/train_surrogate.py runs/ apps/aero_optimizer/model/surrogate.pt --extra verify/   # retrain
python apps/aero_optimizer/tools/build_assets.py runs/ verify/ --out apps/aero_optimizer/model
```

Library: `pinneapple_design.aero` (`geometry`: CST/NACA shapes · `mesh`: O-grid → polyMesh · `case`: OpenFOAM case,
forces from the fields · `aircraft`: performance and requirements · `surrogate`: MeshGraphNet graph + MLP ensemble ·
`optimize`: the engine and NSGA-II search).

## Run

```bash
pip install -e . -r apps/aero_optimizer/requirements.txt          # from the repository root
cd apps/aero_optimizer && uvicorn aero_optimizer.api:app --port 8092
```

Environment: `ADO_USER` / `ADO_PASSWORD` (HTTP Basic login), `ADO_MAX_HEAVY` (concurrent searches, default 2),
`ADO_MODEL` (surrogate bundle). Docker: `docker build -f apps/aero_optimizer/Dockerfile -t aero-optimizer .`;
deployment with the other apps in [`apps/deploy`](../deploy) (service `aero`). Tests: `tests/test_aero_optimizer.py`, `tests/test_aero_optimizer_3d.py`.

## API

Airliner: `POST /api/optimize3d` (mission, requirements, population, generations, seed) → a job (`GET /api/job/{id}`;
the default request returns the stored search at once). `POST /api/aircraft` (`x` = 13 numbers) → performance, checks,
span loading, streamlines; `openfoam` when the design was run in 3D. `GET /api/cfd3d/{id}` → streamlines with speed,
symmetry-plane and wake slices. `GET /api/aircraft.glb?x=…&field=cp|cf|all` (skin fields as vertex attributes
`_CP`, `_CF`), `.usda` (Omniverse), `.stl` (clean CFD geometry).

Wing section: `POST /api/optimize` (aircraft, requirements, wing-area range, population, generations) → every design with its
verdict and reasons, the Pareto front, picks. `POST /api/design` (`x` = 6 shape weights + wing area) → performance,
checks, polar with uncertainty, flow field and coefficients from the graph network. `GET /api/openfoam-case?x=…&alpha=…`
→ zip of a ready case. `GET /api/surrogate` → test-set parity data.
