# 20 One OpenUSD stage: dataset, model input and receipt

**What.** A moving Gaussian heat source scans a 100 mm steel plate (`k = 45 W/m/K`, `rho c = 3.8 MJ/m3/K`) for 1.92 s
in 12 variants (position, speed, power, and one wavy path). PINNeAPPle solves each scan with its immersed
finite-cell transient solver — no conforming mesh of the plate is ever built: the geometry is a box cut by a
background grid. Each scan is written as an **OpenUSD stage** whose `temperature` primvar is keyed over the timeline.
That folder of `.usda` files is the dataset: the training pairs are read back **out of the files** with `pxr`, a
PhysicsNeMo FNO learns the one-step map, is rolled out over a scan it never saw, and the prediction is written
**back into that same stage** as a second mesh with its own time samples, next to the reference and a trust decision.

**Why.** A result that lives in a notebook cannot be reviewed, and a `.usda` that only holds a picture cannot be
checked. Here the file is all three things at once: the data the model learned from, the field it predicted, and the
verdict on that prediction. The example proves the round trip is exact (the file holds exactly the float32 the solver
wrote), so what a reviewer opens in USD Composer is the same number the checks were run on.

| Step | Who | Ran here |
|---|---|---|
| transient conduction, moving source, no conforming mesh | PINNeAPPle `pinneapple_simulation.immersed.solve_transient_heat` | yes |
| one `.usda` per scan, `temperature` keyed over the timeline, spot track as curves | OpenUSD (`usd-core`) | yes |
| training pairs read back from the stages | `pxr` (`read_stage_field`) | yes |
| one-step operator, 32 x 32, trained from the files | PhysicsNeMo `FNO` (CPU) | yes |
| rollout on 2 held-out scans, written back into the stage | PhysicsNeMo + OpenUSD | yes |
| timeline playback / RTX in Omniverse Kit, USD Composer | Omniverse | **no** (needs an Omniverse install and an RTX GPU) |

```bash
pip install usd-core nvidia-physicsnemo
python examples/nvidia/20_usd_is_the_dataset/run.py            # ~25 min on a busy laptop
python examples/nvidia/20_usd_is_the_dataset/run.py --smoke    # seconds (CI)
```

## Results

(this run: laptop, CPU only, background load from other work)

- 10 training scans x 9 frames = **90 (energy, temperature) pairs**, all read back from the `.usda` files, plus 2
  held-out scans. 32 x 32 grid, 1400 Adam steps, batch 8, 362 s; PhysicsNeMo FNO with 2.37 M parameters.
- **The file holds what the solver wrote**: max |field in the file - solver field| = **0.0** K and 0.0 J/m3 over
  every pair (the stages store float32, so the check compares against the same cast).
- Reference first law, worst frame of any scan: **0.44 %** of the deposited energy — the immersed solver stores every
  joule that goes in, and the two primvars in the file satisfy it.
- Held-out scan `v10` (straight path, speed and power never seen): relative L2 **1.10 %** at the last frame
  (0.02 % at the first), hot spot within **4.4 mm** of the reference, first law on the prediction **1.05 %**.
  **Decision: APPROVED** — written to `/World/Prediction` as `pinneapple:trust:decision`.
- Held-out scan `v11` (the wavy path): relative L2 **3.83 %**, which is *inside* the 5 % reference tolerance, but the
  first law on the prediction is **6.3 %** (limit 5 %) and the field dips **2.8 K** below the initial temperature.
  **Decision: REJECT.** The reference error would have waved it through; the physics did not.
- Stage validation: `UsdValidation` and `ComplianceChecker` clean, 2 meshes, 4 prims, 45 time samples on the shared
  timeline. The decision read back from the file is the decision that was written, and the prediction read back is
  bit-identical to the array written into it.
- **Decision of the example: one stage carries the dataset, the prediction and the verdict, and the verdict
  distinguishes a 3.8 % error that is fine from a 3.8 % error that is not.**

Figures in `results/`: `architecture.png`, `prediction.png` (deposited energy from the file, reference, prediction,
difference), `stage_readback.png` (the held-out `.usda` read back with OpenUSD: reference on the left, prediction on
the right), `usd_timeline.gif` (both over the timeline), `training.png`, `checks_curves.png` (error and first-law
residual along the timeline) and `checks_<scan>.png` (the two trust cards).

The dataset itself is in `dataset/`: 12 `.usda` stages, one per scan, `energy` and `temperature` keyed over
`t = 0 ... 1.92 s`. Open any of them in USD Composer and the timeline plays.

