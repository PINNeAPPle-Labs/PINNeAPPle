# MeshGraphNet in PINNeAPPle

Implementation:
- `pinneapple_neural/architectures/graphnn/mesh_graph_net.py`: the encoder, processor and decoder network,
  registered as `mgn` / `meshgraphnet`.
- `mgn_dynamics.py`: the transient training recipe of Pfaff et al. It covers normalization, training noise, the Δv
  target, and autoregressive rollout with boundary nodes held fixed.

## Experiments that run (`examples/meshgraphnet/`)

| Script | What it checks | Data |
|---|---|---|
| `01_synthetic_diffusion.py` | MGN learns heat diffusion on random Delaunay meshes and beats the "frozen state" baseline in rollout on unseen meshes | generated on the fly |
| `02_cylinder_flow_deepmind.py` | Transient flow around a cylinder (DeepMind `cylinder_flow`, Pfaff et al. 2021), on a reduced budget | download of a TFRecord prefix |

`tests/test_mesh_dynamics_mgn.py` runs a reduced `01_synthetic_diffusion.run(...)`, which takes about 12 s and
checks that MGN beats the baseline. It also tests the TFRecord reader of `_common.py` on a file the test writes.

### Measured results (CPU, loaded machine)

| Experiment | Configuration | Rollout RMSE | "Frozen initial state" baseline |
|---|---|---|---|
| 01 synthetic diffusion | 24 training meshes, 150 nodes, 4 MP layers, 2500 steps, rollout 20, 6 test meshes | 0.0297 | 0.1104 |
| 02 cylinder_flow | 11 training trajectories, 64 hidden, 6 MP layers, 3000 steps, lr 1e-3→1e-5, noise 0.003, rollout 50, **2** validation trajectories | 0.0711 (1 step: 0.0265) | 0.0962 |

The synthetic case validates the implementation. The cylinder_flow run **only** shows that the pipeline learns from
real data: it is about 26% better than the baseline, with little data and 3000 steps. It does **not** reproduce the
paper, which used:
- 1000 trajectories;
- 15 layers and 128 hidden units;
- about 10⁶ steps;
- noise 0.02.

The hyperparameters were chosen for the reduced budget. A first attempt with lr 1e-4→1e-6 and noise 0.02 did not
converge in 1000 steps (normalized loss about 5) and was stopped.

## Optional dependencies

| Package | Needed by |
|---|---|
| `tfrecord` (`pip install tfrecord`) | `02_cylinder_flow_deepmind.py`, `physicsnemo_parity/vortex_shedding_mgn.py`, `physicsnemo_parity/lagrangian_mgn.py` (DeepMind TFRecords) |
| `pyvista` (`pip install "pinneapple[pyvista]"`) | `physicsnemo_parity/stokes_mgn.py` (`.vtp` meshes) |
| `scipy` | `01_synthetic_diffusion.py` (Delaunay meshes) |

## Outputs

These small results stay versioned in `_out/`, so the numbers above can be checked without rerunning:
- `synthetic_diffusion.json`
- `cylinder_flow/metrics.json`
- `cylinder_flow/rollout.png`, about 280 KB

Downloads (`_data/`) and logs (`_out/*.log`) are in `.gitignore`.

## Literature used as reference for experiments that work

- Pfaff et al., ICLR 2021, *Learning Mesh-Based Simulation with Graph Networks* (arXiv 2010.03409): cylinder_flow,
  airfoil, flag, deforming_plate. Code and data: github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets
- Sanchez-Gonzalez et al., ICML 2020, *Learning to Simulate Complex Physics with Graph Networks* (arXiv 2002.09405):
  Water/Sand/Goop (the basis of `lagrangian_mgn`).
- PhysicsNeMo examples (`examples/cfd/{vortex_shedding_mgn,stokes_mgn,lagrangian_mgn}`).

## Parity scripts with PhysicsNeMo (written, **not run**): `physicsnemo_parity/`

| PhysicsNeMo | PINNeAPPle | Dataset |
|---|---|---|
| `vortex_shedding_mgn` | `vortex_shedding_mgn.py` | full DeepMind cylinder_flow (13.6 GB) |
| `stokes_mgn` | `stokes_mgn.py` | NGC `physicsnemo_datasets_stokes_flow` (FEniCS, .vtp) |
| `lagrangian_mgn` | `lagrangian_mgn.py` | DeepMind Learning-to-Simulate (Water etc.) |

Out of scope:
- `vortex_shedding_mesh_reduced`: the reduced model with a transformer;
- `stokes_mgn/pi_fine_tuning*.py`: fine-tuning on the PDE residual.

`examples/vs_physicsnemo/06_combined_meshgraphnet_valid` also trains PINNeAPPle's `MeshGraphNet`. The example runs
the whole chain: MGN on an airfoil mesh, then physical checks, then TorchScript/ONNX export.

## CANTO (NVIDIA, arXiv 2609.36806)

CANTO is a *CAD-native* transformer operator:
- **Input:** it tokenizes NURBS patches (control points, weights, knots).
- **Output:** it predicts surface and volume fields at arbitrary query points.
- **Datasets:** AhmedML, WindsorML, DrivAerML, HiLiftAeroML.
- **Code:** the page cites no public code.

It is not a GNN and does not depend on a mesh. Relation to MGN: it is the mesh-free alternative for external
aerodynamics. Suggested integration: a backbone in `neural_operators` or an optional bridge (the `noether_bridge`
pattern), with MGN as the mesh baseline in the same benchmark suite.
