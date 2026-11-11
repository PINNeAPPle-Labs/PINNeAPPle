# 19 Equal budget: NVIDIA PhysicsNeMo FNO vs PINNeAPPle's own FNO2d

**What.** `-div(a grad u) = 1` on the unit square with `u = 0` on the boundary, `a` a smooth log-normal random field
(smooth GRF, `a = exp(0.8 g)`). 400 training pairs come from PINNeAPPle's finite-volume Darcy solver. Two Fourier
neural operators are trained on those pairs under the **same `ComputeBudget`** — same Adam steps, same batch, same
learning rate, same seeds — and with **matched parameter counts** (2.37 M each): `physicsnemo.models.fno.FNO` and the
library's own `FNO2d`.

**Why.** "We integrated with PhysicsNeMo" is easy to say and hard to mean. The honest version is an experiment where
the NVIDIA operator and ours get identical data and identical compute, and something other than the training loss
decides which one is usable. That something is `pinneapple_physics.darcy_checks`: a certified energy-error bound from
the residual, the boundary condition, the flux balance, positivity and the coefficient envelope — **none of which
needs the ground truth**. Both libraries are then judged twice: on held-out coefficients from the training band, and
on coefficients with 2.25x the log-contrast, which neither model ever saw.

| Step | Who | Ran here |
|---|---|---|
| coefficient fields and reference solutions (finite volumes, harmonic-mean faces) | PINNeAPPle `operator_benchmarks.darcy_2d` | yes |
| `physicsnemo.models.fno.FNO`, 2.37 M parameters | NVIDIA PhysicsNeMo (CPU) | yes |
| `FNO2d`, 2.37 M parameters (`width=45`, `padding=8`) | PINNeAPPle (CPU) | yes |
| equal-budget accounting (`ComputeBudget`, cosine LR, no eval time charged) | PINNeAPPle `equal_budget` | yes |
| no-reference physics checks, applicability envelope, two trust decisions | PINNeAPPle `darcy_checks` | yes |
| GPU training | PhysicsNeMo on CUDA | **no** (CPU only, on a laptop) |

```bash
pip install nvidia-physicsnemo
python examples/nvidia/19_physicsnemo_vs_pinneapple/run.py            # 2 seeds x 2 libraries, ~1 h while other jobs share the machine
python examples/nvidia/19_physicsnemo_vs_pinneapple/run.py --smoke    # seconds (CI)
```

## Results

(this run: laptop, CPU only, under heavy background load — see the note on the timings)

- 400 training pairs, 50 held-out in band, 50 held out at 2.25x the training log-contrast; 32 x 32 grid; 1500 Adam
  steps, batch 8, `lr = 2e-3`, seeds 0 and 1. **2 366 305** parameters (PhysicsNeMo FNO) against **2 341 306**
  (`FNO2d`, `width = 45`), 1.1 % apart.
- Inside the band, held-out relative L2: **PhysicsNeMo 2.30 %** (2.25 % and 2.35 % per seed) against
  **PINNeAPPle 1.89 %** (1.84 % and 1.94 %). Half a point apart, consistently in favour of `FNO2d`.
- Outside the band both collapse: **56.9 %** (PhysicsNeMo) and **54.3 %** (PINNeAPPle). Neither operator
  extrapolates in contrast, and the coefficient envelope flags those inputs on every sample, which is the only thing
  that keeps that collapse from being a surprise in production.
- **Both are REJECTed in band, at 2 % relative L2**, by the certified residual bounds: energy-error bound 42.2 %
  (PhysicsNeMo) and 28.8 % (PINNeAPPle) against a 10 % limit, L2 bound 125 % and 83 % against a 5 % limit; flux
  balance 15.2 % and 6.8 % against 2 %. The bounds are conservative by construction — a factor of 15 to 54 above the
  true held-out error here — and the finite-volume reference fields pass the very same gate (APPROVED), so the gate
  is strict rather than broken. Boundary condition: PhysicsNeMo 2.16 % (over the 2 % limit), PINNeAPPle 1.89 %.
- Timings are not a benchmark: **901 s** mean per PhysicsNeMo training against **1096 s** for PINNeAPPle, whose two
  seeds came out at 1684 s and 508 s because the second one shared the machine with another example. Wall clock on a
  contended laptop says nothing about either implementation; the step count, the batch and the data were identical,
  which is what `ComputeBudget` guarantees.
- **Decision: REJECT for both, in band and out.** The honest reading: at equal budget the two libraries are
  interchangeable on accuracy (half a point), neither generalises outside the training contrast, and the interesting
  part of the experiment is that a 2 % field error does not pass a certified physics gate.

Figures in `results/`: `architecture.png`, `head_to_head.png` (in band vs out of band, per seed), `fields.png`
(coefficient, reference, both predictions, both error maps), `training.png`, `checks_physicsnemo.png` and
`checks_pinneapple.png` (the two trust cards, side by side).

