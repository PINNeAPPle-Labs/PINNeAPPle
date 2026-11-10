# Cold plate with the library's FNO

This is the cold plate of `examples/nvidia/07_omniverse_physicsnemo_coldplate`, with the library's own `FNO2d` in place of PhysicsNeMo's FNO. Nothing outside PINNeAPPle is needed.

- **Plate:** 100 mm × 100 mm, copper inserts (400 W/m/K) in an aluminium base (100 W/m/K).
- **Physics:** a uniform heat load, edges held at the coolant temperature, steady conduction `-div(k grad T) = q`.
- **Data:** 400 training and 50 test layouts from the library's finite volumes, the same layouts as example 07.
- **Model:** `FNO2d` (width 32, 12 modes, 4 layers) with 8 cells of domain padding, since the edges are not periodic (#1162).
- **Training:** the same `ComputeBudget` and loss as example 07.
- **Checks:** `pinneapple_physics.darcy_checks` decide whether the predicted fields can be used.

```
python examples/fno_coldplate/run.py               # about 2 min on 4 CPU cores
python examples/fno_coldplate/run.py --steps 8000  # about 7 min
python examples/fno_coldplate/run.py --smoke       # seconds (CI)
```

## Measured (4 CPU cores, 2026-11-03)

| | library FNO2d, 2 700 steps | library FNO2d, 8 000 steps | PhysicsNeMo FNO, 2 700 steps (example 07) | limit |
|---|---|---|---|---|
| parameters | 1.18 M | 1.18 M | 2.37 M | |
| test relative L2 | 1.67 % | 1.40 % | 1.76 % | 5 % ✓ |
| hotspot error | 0.24 K | 0.21 K | 0.24 K | |
| certified energy-error bound (PDE residual) | 0.164 | 0.143 | 0.196 | 0.10 ✗ |
| certified L2 error bound | 0.247 | 0.207 | 0.293 | 0.05 ✗ |
| flux balance | 0.054 | 0.041 | 0.089 | 0.02 ✗ |
| boundary condition, maximum principle, envelope | ✓ | ✓ | ✓ | |
| **decision** | **REJECT** | **REJECT** | REJECT | |

`results/` holds the 2 700-step run: `summary.json`, `fields.png` and `checks.png`.

## What it shows

- **Against PhysicsNeMo's FNO.** At the same budget, the library's FNO is closer to the reference and closer to satisfying the PDE, with half the parameters.
- **Why the checks still reject it.** The fields are accurate on average: 1.4–1.7 % error, about 0.2 K on the hotspot. They do not satisfy the PDE locally. The certified bound comes from the residual of `-div(k grad T) = q` on the finite-volume grid, and it stays above its limit. Training longer on data alone lowers it slowly.
  - The next step is to fine-tune with the PDE's residual as a loss: PINO, already in `pinneapple_neural/architectures/neural_operators/pino.py`. Tracked in #1163.
- **Speed.** On this 32 × 32 grid the solver itself is cheap: about 2.4 ms per layout, against 1.6 ms for the FNO. A neural operator pays off on large or transient problems, not here. The Burgers measurement in #1162 shows about 350× at 1024 points.

## Independent evaluation and falsification controls

Use [EXTERNAL_VALIDATION.md](EXTERNAL_VALIDATION.md) to reproduce the
finite-volume-positive, corrupted-field-negative, and out-of-envelope-negative
controls. This checks the **physics validator**, not FNO accuracy:

```bash
python examples/fno_coldplate/external_validation.py --out artifacts/external_validation/coldplate
```

The existing FNO can now run on **five different held-out dataset and model
seeds**, saving per-seed physical decisions and predictions on a shifted
conductivity distribution:

```bash
python examples/fno_coldplate/multiseed_study.py --seeds 0 1 2 3 4
```

This is compute-intensive. Run `--smoke --seeds 0 1` for an integration
check **only**; short smoke training is not a publishable benchmark.

The existing table above is internal evidence and **REJECT** is a meaningful
outcome, not a failure to conceal. No independent scientific reproduction
has been claimed or implied by this code.
