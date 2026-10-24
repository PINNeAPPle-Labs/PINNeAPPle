# PINNeAPPle with the NVIDIA stack

Every example puts PINNeAPPle at the centre (the problem, the reference solvers, the geometry checks, the physics
checks and the trust decision) and joins it with a different combination of stacks. The glue lives in
`pinneapple_tools.nvidia`; each example is a thin `run.py` that writes `results/architecture.png` (PINNeAPPle's role
and each stack's role, and what was *not* run here), result figures, `summary.json` and `run_info.json` (versions,
hardware, timings, which components ran).

| Code | Stack | What it is | Runs here |
|---|---|---|---|
| O | NVIDIA Omniverse | OpenUSD stages (`usd-core`), validated with `pxr.UsdValidation`; RTX rendering in Kit needs a GPU | USD: yes. Kit RTX: no |
| P | NVIDIA PhysicsNeMo | `nvidia-physicsnemo` models (FNO, MeshGraphNet, FullyConnected) trained on CPU | yes |
| B | NVIDIA USD BRep | NURBS B-rep kernel (SMLib), `pip install usd-brep` (CPython 3.12 wheel) | yes |
| I | PINNeAPPle Robot Sim | robot simulation on NVIDIA Newton (CPU), with an optional Isaac Sim adapter (same interface) | yes (Isaac Sim: no) |
| N | PINNeAPPle Model Services | local, NIM-compatible model microservices (`/v1/health/ready`, `/v1/models`, `/v1/infer`) with a trust report on every answer; the same client talks to NVIDIA NIM with `NVIDIA_API_KEY` | yes (NIM: no) |
| C | PINNeAPPle World Model | synthetic data and a CPU world model from PINNeAPPle simulations, with physics-consistency checks; optional NVIDIA Cosmos adapter (same interface) | yes (Cosmos: no) |

| # | Stacks | Example | Decision (recorded run) | Status |
|---|---|---|---|---|
| 01 | O | [Pump and pipeline water-hammer twin](01_omniverse_pump_pipeline/README.md) | APPROVED (60 s closure only) | done |
| 02 | P | [Plume forecast, FNO rollout](02_physicsnemo_plume_rollout/README.md) | REVIEW (10-step horizon) | done |
| 03 | I | [Pick-and-place arm in the robot sim, structural checks](03_robotsim_pick_and_place/README.md) | APPROVED (shoulder motor at 95 %) | done |
| 04 | C | Physics-consistent synthetic video and world model | | planned |
| 05 | B | [Strap with a hole: exact geometry, FEM, Heywood](05_brep_plate_with_hole/README.md) | APPROVED (Kt 2.320 vs 2.296) | done |
| 06 | N | Zoo surrogate served NIM-style with its trust report | | planned |
| 07 | O + P | [Cold plate surrogate in a USD scene](07_omniverse_physicsnemo_coldplate/README.md) | REJECT (energy balance) | done |
| 08 | I + P | [Learned arm dynamics, checked physics](08_robotsim_physicsnemo_learned_dynamics/README.md) | REVIEW (start pose at the envelope edge) | done |
| 09 | C + P | PhysicsNeMo world model on PINNeAPPle-generated data | | planned |
| 10 | B + P | [Heat-sink design from exact geometry](10_brep_physicsnemo_heatsink/README.md) | APPROVED (first proposal rejected by verification) | done |
| 11 | N + P | PhysicsNeMo model served NIM-style | | planned |
| 12 | O + I | [One USD scene for Omniverse and the robot sim](12_omniverse_robotsim_arm/README.md) | APPROVED (USD = URDF to the bit) | done |
| 13 | O + C | Synthetic inspection data as USD scene variations | | planned |
| 14 | B + O | [Exact CAD into Omniverse: curvature, wall thickness](14_brep_omniverse_bracket/README.md) | APPROVED (native B-rep layer lacks stage metadata) | done |
| 15 | N + B | Specification extracted by a model service, part built in B-rep, verified | | planned |
| 16 | O + P + B | [Verified heat sink with its temperature field in USD](16_omniverse_physicsnemo_brep_heatsink/README.md) | APPROVED | done |
| 17 | O + I + C | Robot perception with synthetic camera data | | planned |
| 18 | all six | End to end | | planned |

What runs where: PINNeAPPle, PhysicsNeMo (CPU), OpenUSD (`usd-core`) and USD BRep run on a laptop and were run for
the numbers in each README. Omniverse Kit RTX rendering, Isaac Sim, NVIDIA Cosmos and NVIDIA NIM need an NVIDIA GPU
and/or an `NVIDIA_API_KEY`; their adapters are written against the official APIs and drawn hatched in the
architecture figures, and nothing in the results comes from them.

```bash
pip install nvidia-physicsnemo usd-core newton newton-usd-schemas
python3.12 -m venv /opt/brep && /opt/brep/bin/pip install usd-brep numpy   # USD BRep ships a CPython 3.12 wheel
export PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python
pytest tests/test_nvidia_integrations.py -q
```
