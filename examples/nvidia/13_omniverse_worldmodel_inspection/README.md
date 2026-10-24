# 13 Synthetic inspection data as USD scene variations: PINNeAPPle + Omniverse (OpenUSD) + PINNeAPPle World Model

**What.** A heated plate has a subsurface void (a disk of low conductivity) and runs hotter above it. PINNeAPPle's
finite-volume solver gives the temperature field and the **physical label**: the point where the void raises the
surface temperature most.

- **USD stage per sample:** the plate with `temperature`, `radiance` and `conductivity` primvars, a side
  `UsdGeom.Camera` with the sample's pose, and the variation and label as attributes.
- **Synthetic camera:** rasterizes the stage from above (`rasterize_topdown`) and applies the variation: emissivity
  gain, reflected ambient, noise, blur, zoom, rotation, offset and, in 25 % of samples, a specular glint.
- **Label check:** the image is compared with what the same camera sees on the defect-free plate. A sample is
  dropped when the strongest anomaly is not at the label (a glint outshines the void) or the label falls on
  saturated pixels.
- **Detector:** a small CNN locates the hot spot, trained on all samples or only on the accepted ones.

**Why.** Augmentation can make a synthetic image lie about its label. Here the physics says where the defect is, and
every augmented image must still show it.

| Step | Who | Ran here |
|---|---|---|
| steady conduction, label, label resolution study | PINNeAPPle | yes |
| 680 USD stages (primvars, camera, label attributes), validated | OpenUSD (`usd-core` 26.11) | yes |
| synthetic camera, label check, CNN detector | PINNeAPPle World Model | yes |
| Omniverse Replicator / RTX render of the same stages | Omniverse | no (needs an RTX GPU) |
| Cosmos Transfer (photoreal augmentation) | NVIDIA Cosmos | no (needs a GPU or `NVIDIA_API_KEY`) |

```bash
pip install usd-core
python examples/nvidia/13_omniverse_worldmodel_inspection/run.py    # ~84 min on 4 CPU cores, shared with other runs
```

## Results (this run: 4-core Xeon 2.1 GHz, no GPU)

- **Physical label:** it moves at most 0.016 plate units between 48 and 96 cells, less than one coarse cell.
- **USD stages:** the stages checked are valid (primvars, camera, `pxr.UsdValidation`).
- **Label check, 600 training samples:**
  - It drops **98.6 %** of the 146 samples with a glint.
  - It keeps **99.7 %** of the 295 glint-free, unsaturated ones.
  - 35 % of glint-free samples saturate at the label and are dropped too: the camera cannot show the defect there.
  - 296 samples are accepted.
- **Detector, on 50 test images that show the defect** (tolerance 4 px):

| Detector | Median error | Within 4 px |
|---|---|---|
| CNN on all samples | 8.7 px | 20 % |
| CNN on accepted samples | **6.1 px** | **26 %** |
| brightest pixel (baseline) | 5.7 px | 44 % |

- **Decision: REVIEW.**
  - The check improves the training data: the CNN trained on accepted samples beats the one trained on all samples.
  - But a small CNN trained on 296 images does not beat the brightest-pixel baseline, and it misses the 90 % target.
  - The data pipeline and the checks pass; the detector does not.

Figures in `results/`: `architecture.png`, `samples.png` (accepted and rejected images with the label),
`usd_preview.png` (a sample stage read back from USD), `detector.png` (location error), `checks.png`.
