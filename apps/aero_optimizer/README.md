# Aircraft Design Optimizer

**OpenFOAM → graph neural network → thousands of wing designs → the ones that fly faster and burn less.**

Day 13 of the PINNeAPPle 30-app program. An aircraft-design loop end to end, on a light single-engine aircraft
(Cessna 172 class by default, any mass/power/cruise you enter):

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

RESULTS_PLACEHOLDER

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
deployment with the other apps in [`apps/deploy`](../deploy) (service `aero`). Tests: `tests/test_aero_optimizer.py`.

## API

`POST /api/optimize` (aircraft, requirements, wing-area range, population, generations) → every design with its
verdict and reasons, the Pareto front, picks. `POST /api/design` (`x` = 6 shape weights + wing area) → performance,
checks, polar with uncertainty, flow field and coefficients from the graph network. `GET /api/openfoam-case?x=…&alpha=…`
→ zip of a ready case. `GET /api/surrogate` → test-set parity data.
