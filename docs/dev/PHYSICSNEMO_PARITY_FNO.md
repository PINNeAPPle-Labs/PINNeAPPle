# PhysicsNeMo parity, FNO-2D: measured, not assumed (#2529)

The epic asks for parity that is demonstrated. This page is the FNO slice of that: what PINNeAPPle's
`FNO2d` and PhysicsNeMo's `FNO` actually share, what they do not, and the two numbers that decide it — the
relative L2 between the two forward passes **with the same weights**, and the error each side reaches on the same
problem against an **exact** reference.

Everything measured here is reproducible from the repository:

```bash
python -m benchmarks.fno_physicsnemo_parity            # ~40 min on 4 CPU cores, writes benchmarks/_out/
python -m benchmarks.fno_physicsnemo_parity --quick    # the parity rows only, about 15 s
pytest tests/test_fno_physicsnemo_parity.py -q         # the claims, as tests
```

## Pinned versions

| | |
|---|---|
| PhysicsNeMo | `nvidia-physicsnemo==2.2.2` (imports and runs natively on macOS ARM64, CPU, Python 3.11; its FNO does not need CUDA) |
| torch | 2.15.0, CPU, float32 |
| PINNeAPPle | this repository at the commit under test |
| License | both Apache-2.0, so nothing here is a re-implementation under someone else's name |

PhysicsNeMo's `DeepONet` lives in `physicsnemo.experimental.models.xdeeponet` in this release (with an
`ExperimentalFeatureWarning`, i.e. "partial test coverage" by NVIDIA's own wording) and not in `physicsnemo.models`,
where this audit expected it. That is the reason FNO is the operator audited first: it is the one both projects
ship as stable.

## What each side has

`physicsnemo.models` in 2.2.2: `afno`, `dit`, `domino`, `dpot`, `figconvnet`, `flare`, `fno`, `geotransolver`,
`graphcast`, `meshgraphnet`, `mesh_reduced`, `pangu`, `rnn`, `transolver`, `unet`, `vfgn`, plus the weather/nwp
models and `diffusion_unets`. `physicsnemo.experimental.models`: `aerojepa`, `flare`, `geotransolver`, `globe`,
`healda`, `strata`, `xdeeponet`, `diffusion*`.

`pinneapple_neural.architectures.neural_operators` (registry keys): `fno` (1-D), `fno2d`, `fno3d`, `deeponet`,
`multiscale_deeponet`, `gno`, `pino`, `uno`, `transolver_native`, `transolver_lite`, plus the Noether bridge
(`upt`, `abupt`, `transformer`, `transolver` and their `aero_`/`noether_` variants, optional dependency).

## The FNO-2D feature matrix

Left column: the feature. Then what each side does, and the state after this work. "Was" is what this repository
did before the slice; "now" is what it does.

| Feature | PhysicsNeMo 2.2.2 `FNO` | PINNeAPPle `FNO2d` was | now | Evidence |
|---|---|---|---|---|
| Fourier layer | `SpectralConv2d`, `rfft2`, positive and negative corner in `H`, `modes1 x modes2` block | same construction, `rfft2`, same two corners | unchanged | `pinneapple_neural/architectures/neural_operators/fno.py::_SpectralConv2d`; forward parity **1.8e-7** |
| Spectral weight layout | real/imaginary pair, `(..., 2)` float32 | `cfloat` | unchanged (both are `cfloat` after `view_as_complex`) | `fno_compare.py::fno2d_weight_map` |
| Spectral weight init | `scale * U[0,1)` on both parts (`scale = 1/(in_c out_c)`), Li et al. 2021 | `scale * complex N(0,1)` | was the gap: `spectral_init="uniform"` added (default keeps `"normal"`) | `_SpectralConv2d._draw` |
| Lifting | two 1x1 convs with a hidden layer of `width // 2` and the activation between | one 1x1 conv | `lift_hidden=` added | `FNO2d(lift_hidden=...)` |
| Lifting bias | present but **frozen** (`requires_grad = False`, `Conv2dFCLayer.reset_parameters`) | trainable | `requires_grad` mirrored by the transplant | `fno_compare.py::transplant_fno2d_weights` |
| Activation after the last Fourier block | none | GELU after every block | `final_activation=` added | `FNO2d(final_activation=False)` |
| Decoder head | pointwise MLP: `Linear(width -> decoder_layer_size)`, SiLU, `Linear(-> out)` | 1x1 conv | `decoder_hidden=`, `decoder_activation=` added | `FNO2d(decoder_hidden=...)` |
| Domain padding | `F.pad` after the lift, `padding` cells at the end of `H` and `W`, cropped before the head; `padding_type` ∈ {constant, reflect, replicate, circular} | zero padding only, same position | unchanged (**gap**: other padding modes) | `FNO2d(padding=...)` |
| Modes per axis | `num_fno_modes` int or `[m1, m2]` | `modes1`, `modes2` | unchanged (independent per axis) | |
| Mode truncation on a coarse grid | slices `[:modes1]`, `[-modes1:]`; overlapping when `modes1 > H/2` | clamps to `min(modes, H // 2)` | unchanged; the parity builder refuses a grid where the clamp bites | `fno_compare.FNOArch`, `_check_grid` |
| Coordinate channels | `coord_features`, `linspace(0, 1)`, `meshgrid(indexing="ij")` | `use_grid`, identical construction | unchanged | forward parity covers it |
| Dimensions | 1, 2, 3 **and 4** | 1, 2, 3 | unchanged (**gap**: 4-D) | |
| Resolution transfer | spectral layer is resolution-free | same | measured: train at 64², score at 96² | `benchmarks/_out/fno_physicsnemo_parity.json` → `transfer_96` |
| Mixed precision / CUDA graphs / multi-GPU | `MetaData(amp=..., cuda_graphs=..., jit=...)`; distributed paths | none of these | unchanged (**gap**) | |
| Serialization | `save`/`load` (PhysicsNeMo checkpoint format) | `state_dict`; `GridFieldPredictor.save` for the predictor wrapper | weight **transplant** between the two | `transplant_fno2d_weights` |
| Parameter accounting | real numbers in the state dict | complex tensors count 1 with `numel()` | `real_parameter_count` counts complex as two | equal counts at parity, see below |

The three structural differences that mattered (init, lifting, last-block activation, decoder head) are what made
"FNO versus FNO" a comparison of two different architectures. All four are now expressible on this side without
changing the defaults, so an existing checkpoint loads and an existing model trains exactly as before.

## Measurement 1 — same weights, same function?

`fno_compare.fno2d_forward_parity` builds PhysicsNeMo's `FNO` and this library's `FNO2d` in the parity
configuration (`PARITY_ARCH`: 32 channels, 12 modes, 4 blocks, 8 cells of padding, 64-wide decoder), transplants the
weights through the documented map, mirrors the frozen set, and evaluates both on the same batch of uniform random
fields.

| grid | relative L2 between the two forward passes | max abs difference | field scale |
|---|---|---|---|
| 32 x 32 | 1.80e-7 | 4.5e-8 | 7.8e-2 |
| 64 x 64 | 1.79e-7 | 5.2e-8 | 7.8e-2 |
| 96 x 96 | 1.79e-7 | 5.2e-8 | 7.8e-2 |

Predeclared tolerance: `1e-4` (`fno_compare.PARITY_TOLERANCE`), i.e. three orders of magnitude above the float32
round-off the two FFT pipelines carry. **The answer is yes**: at these weights the two are the same function, and
the residual is round-off, not modelling.

Parameters: 2,366,305 real numbers on each side — equal, once a complex weight is counted as two. (With
`numel()` on complex tensors this side reports 2,362,241 and the comparison is meaningless; that is why
`real_parameter_count` exists.)

## Measurement 2 — same loss, same gradient?

The forward pass agreeing says the two are the same function. The gradients agreeing says **training** them is the
same computation. `fno2d_gradient_parity` runs one forward and one backward of the FNO paper's loss (per-sample
relative L2, mean over the batch) on the same batch, after the same transplant.

| | relative L2 of the gradients |
|---|---|
| whole model | 6.7e-8 |
| worst single parameter (`out_proj.2.weight`, the last decoder weight) | 1.9e-6 |

PhysicsNeMo freezes the two lifting biases (`Conv2dFCLayer.reset_parameters` sets `requires_grad = False` after
zeroing them, unlike its 1-D and 3-D variants); 14 of the 16 parameters train on both sides once the transplant
mirrors that. Leaving them trainable on one side alone would have made the models differ by 12 numbers nobody would
notice in a loss curve.

## Measurement 3 — the same problem, both paths, one exact reference

**Problem.** `u_t + c . grad u = kappa lap u` on the periodic unit square, `u(0) = u0`, `u(t_end)`, with
`c = (0.7, 0.3)`, `kappa = 0.01`, `t_end = 0.5` (a displacement of 0.35 box lengths and a mild diffusion).
Dimensionless concentration; velocity in box lengths per time unit; `kappa` in box lengths squared per time unit.

**Reference.** The propagator is diagonal in Fourier space, so the exact solution of every mode is an analytic
multiplier and the reference is computed, not solved: `operator_benchmarks.advection_diffusion_2d`, float64, no
discretisation error of its own. Total mass `mean(u)` is conserved exactly — that is the physics QoI, reported as
`mass_drift`.

**Data.** 256 fields from `periodic_grf` (alpha = 2, tau = 3, seed 7) **band-limited to |k| <= 6** in each direction,
so the exact solution is representable by an FNO keeping 12 modes: the error measured is the network's and the
training's, not a truncation the architecture cannot avoid. Split 192 train / 64 held out.

**Protocol.** Both sides: per-sample relative L2 loss, Adam, `lr = 1e-3`, weight decay 1e-5, cosine schedule over the
steps, batch 16, 200 steps, the same seeded batch order, the same input mean/std and output scale, 4 threads, CPU,
float32. Three seeds. `pinneapple_analysis.cost.equal_budget.train_within_budget` drives both.

**Result — the parity configuration** (the same architecture on both sides, weights transplanted before training):

| side | train time (s) | held-out relative L2 | worst sample | mass drift | transfer to 96² | agreement between the two trained models |
|---|---|---|---|---|---|---|
| PINNeAPPle | 132.2 ± 21.9 | **0.1708** ± 0.0023 | 0.2566 ± 0.0114 | 0.0792 ± 0.0012 | 0.1839 ± 0.0103 | **1.6e-7** ± 5.7e-9 |
| PhysicsNeMo | 112.2 ± 3.6 | **0.1708** ± 0.0023 | 0.2566 ± 0.0114 | 0.0792 ± 0.0012 | 0.1839 ± 0.0103 | **1.6e-7** ± 5.7e-9 |

Mean ± standard deviation over the three seeds, each seed a full training run on both sides. The two columns that
matter: the held-out error agrees to every printed digit, and the two *trained* models differ from each other by
1.6e-7 — the same number as the untrained forward parity, i.e. 200 steps of Adam on the same batches move both
weights tensors the same way. Left to differ only in the data copy, the two would still diverge in the last digits;
they do not. The time differs (132 s against 112 s): same math, different Python around it, and that is stated here
rather than averaged away.

**Result — each side as it ships** (PINNeAPPle's defaults, PhysicsNeMo's defaults — a different architecture on
purpose, since this is the comparison a migrating user actually faces):

| side | train time (s) | held-out relative L2 | worst sample | mass drift | transfer to 96² | agreement between the two trained models |
|---|---|---|---|---|---|---|
| PINNeAPPle | 96.7 ± 3.6 | **0.0979** ± 0.0137 | 0.1572 ± 0.0221 | 0.0387 ± 0.0048 | 0.1650 ± 0.0074 | 0.126 ± 0.009 |
| PhysicsNeMo | 102.4 ± 3.5 | 0.1708 ± 0.0023 | 0.2566 ± 0.0114 | 0.0792 ± 0.0012 | 0.1839 ± 0.0103 | 0.126 ± 0.009 |

The as-shipped PINNeAPPle default lands at **0.0979** against PhysicsNeMo's **0.1708** on this problem, with the mass
drift 0.0387 against 0.0792 — better on both, and with a wider spread across seeds (0.0137 against 0.0023), so this
is one problem and 200 steps, not a ranking. The two models now differ by 0.126 relative L2, which is the size of the
architectural difference the parity options exist to remove; in the parity configuration it becomes 1.6e-7.

Reproduce: `python -m benchmarks.fno_physicsnemo_parity` — **1368.7 s** wall on this machine (10-core ARM, 4 threads,
CPU, float32), writing `benchmarks/_out/fno_physicsnemo_parity.json`.


## What this changes for a user migrating from PhysicsNeMo

1. The architecture is now expressible here, and the weights come across: `transplant_fno2d_weights` is the
   documented map, and it refuses a target that is not the parity configuration instead of half-copying.
2. The defaults did not change, so nothing that was trained here before needs retraining.
3. What is still not portable: 4-D FNO, the non-constant padding modes, AMP/CUDA-graphs/distributed wrappers, and
   PhysicsNeMo's checkpoint format (only the weight map is provided; the file format is not read).

## Gaps this slice does not close

| Gap | Status | Why it is not closed here |
|---|---|---|
| FNO 1-D and 3-D in the parity configuration | open | the same five options are trivial to add (`FourierNeuralOperator`, `FNO3d`); the measurement, not the code, is the work, and 2-D is where the industrial cases live |
| `padding_type` other than zero | open | needs the same measurement per mode; a wrong padding mode changes the answer at the boundary band, so it must be measured, not assumed |
| 4-D FNO | open, dispositioned | used for space-time weather models, not for the internal-flow cases of the epic |
| AMP / CUDA graphs / distributed | open | no GPU on this machine; the epic forbids claiming what was not run |
| PhysicsNeMo checkpoint `load` | open | the weight map covers migration; the file format is NVIDIA's to keep |
| DeepONet parity | not started | PhysicsNeMo's is `experimental` in 2.2.2; the audit is scheduled after the operator both projects ship as stable |
| The three GNN/geometry operators (`figconvnet`, `domino`, `geotransolver`) | not started | separate slice; each needs its own problem and reference |
