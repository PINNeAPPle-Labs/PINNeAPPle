# PINNsFormer (Zhao, Ding & Prakash, ICLR 2024): frozen targets

Written and committed **before any run**. Every number is quoted from the PDF named below, with its table and page.

## Source

- Z. Zhao, X. Ding, B. A. Prakash, "PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural
  Networks", ICLR 2024; arXiv:2307.11833v3 (7 May 2024), 17 pages.
- Downloaded from `https://arxiv.org/pdf/2307.11833` on 2026-10-30.
- **Official code:** `https://github.com/AdityaLab/pinnsformer` at commit `00d790f50c3f77599fbe263b9e8035f9860a1027`.
  - It is read for details the paper leaves out. Its code is not copied or run here.
  - Its demo notebooks print the exact numbers of Tables 1 and 2. For example,
    `demo/convection/convection_pinnsformer.ipynb` prints relative L1 0.022864 and L2 0.026753, which Table 1
    rounds to 0.023 and 0.027.
  - So the tables come from **one run each, seed 0, on a GPU**.

## Method as the paper states it (§3, p. 3-6; Appendix A, p. 13)

- **Pseudo sequence** (eq. 3): [x, t] → {[x, t], [x, t + ∆t], ..., [x, t + (k − 1)∆t]}.
- **Mixer:** one linear layer to the embedding size.
- **Encoder-decoder.**
  - The encoder has self-attention and a feed-forward layer.
  - The decoder has encoder-decoder attention and a feed-forward layer, with no self-attention.
  - Both take the same embeddings.
- **Output:** an MLP.
- **Wavelet activation** (eq. 4): ω1 sin(x) + ω2 cos(x), with ω1 and ω2 learnable.
- **Loss** (eq. 6):
  - The residual and boundary terms are averaged over all k copies.
  - The initial condition applies to the first copy only.
  - λres = λic = λbc = 1.
- **Training** (§4.1, p. 7): "L-BFGS optimizer with Strong Wolfe linear search for 1000 iterations".
- **Points** (§4.1, p. 7).
  - The baselines use Nic = Nbc = 101 and a 101 × 101 residual grid.
  - PINNsFormer uses Nic = Nbc = 51 and a 51 × 51 grid.
  - The test uses a 101 × 101 grid.
- **Metrics** (eq. 7, p. 13): rMAE = Σ|û − u| / Σ|u| and rRMSE = sqrt(Σ(û − u)² / Σu²), on the test grid.
- **Hyperparameters** (Table 4, p. 13).
  - PINNs: 4 hidden layers ("hidden layer 4"), hidden size 512, 527k parameters.
  - PINNsFormer: k = 5, ∆t = "1e-3, 1e-4", 1 encoder, 1 decoder, embedding size 32, 2 heads, hidden size 512,
    454k parameters.
- **Problems** (Appendix B, p. 14).
  - Convection: u_t + β u_x = 0 on [0, 2π] × [0, 1], with u(x, 0) = sin x, periodic, and β = 50.
  - 1D reaction: u_t − ρ u (1 − u) = 0, with u(x, 0) = exp(−(x − π)² / (2(π/4)²)), periodic, and ρ = 5.
    The analytic solution is eq. 10.
  - 1D wave: u_tt − β u_xx = 0 on [0, 1]², with u(x, 0) = sin(πx) + ½ sin(βπx), u_t(x, 0) = 0, u(0, t) =
    u(1, t) = 0, and β = 3 "specified". The solution is eq. 12, sin(πx) cos(2πt) + ½ sin(βπx) cos(2βπt).

## Targets

### Table 1 (p. 8): convection and 1D reaction

| Model | Convection Loss | rMAE | rRMSE | 1D-Reaction Loss | rMAE | rRMSE |
|---|---|---|---|---|---|---|
| PINNs | 0.016 | 0.778 | 0.840 | 0.199 | 0.982 | 0.981 |
| QRes | 0.015 | 0.746 | 0.816 | 0.199 | 0.979 | 0.977 |
| FLS | 0.012 | 0.674 | 0.771 | 0.199 | 0.984 | 0.985 |
| PINNsFormer | 3.7e-5 | 0.023 | 0.027 | 3.0e-6 | 0.015 | 0.030 |

### Table 2 (p. 8): 1D wave

| Model | Loss | rMAE | rRMSE |
|---|---|---|---|
| PINNs | 1.93e-2 | 0.326 | 0.335 |
| PINNsFormer | 1.38e-2 | 0.270 | 0.283 |
| PINNs + NTK | 6.34e-3 | 0.140 | 0.149 |
| PINNsFormer + NTK | 4.21e-3 | 0.054 | 0.058 |

### Text claims (p. 8)

- "PINNsFormer achieves the lowest training loss and test errors, distinguishing PINNsFormer as the only approach
  capable of mitigating the failure modes."
- Table 2 caption: "PINNsFormer combined with NTK outperforms all other methods on all metrics."

### Also in the paper, not attempted here

- **Table 3** (2D Navier-Stokes, p. 9). It needs the cylinder-wake data of Raissi et al.
- **Table 5** (overheads on a V100). This is hardware-specific.
- **Table 6** (the activation ablation).
- **Table 7** (the k × ∆t sensitivity on 1D reaction, 20 cells). It is phase 2, below.
- **The QRes, FLS and NTK rows.**

These are reported as "not attempted" unless an amendment to this file adds them before they are run.

## Scope (phase 1, scored)

| Target set | Models | Seeds | Scored numbers |
|---|---|---|---|
| Table 1 | PINNs, PINNsFormer | 0, 1, 2 | Loss, rMAE and rRMSE: 2 models × 2 problems × 3 = 12 |
| Table 2, rows 1-2 | PINNs, PINNsFormer | 0, 1 | Loss, rMAE and rRMSE: 2 × 3 = 6 |

That makes **18 scored targets**.

- **Seed counts.** Training is costly on our CPU-only machine. One L-BFGS outer step of PINNsFormer on convection
  takes about 16 s, against the official notebook's 2.1 s on a V100. That puts convection at about 2.2 h per run
  and wave at about 5 h. Wave therefore gets 2 seeds.
- **Phase 2.** Table 7 at seed 0, if time allows, as an amendment written before it runs.

## Ambiguities and our fixed choices

1. **Iterations.**
   - The paper says "1000 iterations". The official notebooks run 500 calls of `LBFGS.step` for convection and
     reaction, and 1000 for wave. PyTorch's `LBFGS` takes up to 20 inner iterations per call (`max_iter`
     default), with lr 1 and history 100.
   - These notebooks produce the tables' numbers, so **we follow them**: 500 / 500 / 1000 outer steps, strong
     Wolfe, PyTorch defaults otherwise. The same schedule applies to PINNs and PINNsFormer.
2. **∆t.** Table 4 lists "1e-3, 1e-4". All three notebooks use ∆t = 1e-4 with k = 5, and we use those.
3. **Wave coefficient.**
   - The text uses one β for two things: the PDE u_tt − β u_xx = 0 and the initial term ½ sin(βπx), with β = 3.
   - Eq. 12 with β = 3, sin(πx) cos(2πt) + ½ sin(3πx) cos(6πt), solves u_tt = 4 u_xx, not 3 u_xx.
   - The notebook uses u_tt − 4 u_xx = 0, with initial condition sin(πx) + ½ sin(3πx). We use that; it is
     consistent with eq. 12.
4. **"hidden layer 4" for PINNs.** In the notebooks this is `PINNs(num_layer=4)`: 3 hidden layers of 512 with
   tanh (527 361 parameters, matching "527k").
5. **Architecture details** that the paper leaves out are taken from `model/pinnsformer.py`:
   - feed-forward width 256;
   - WaveAct before attention and before the feed-forward, both with residuals;
   - a final WaveAct after the encoder and after the decoder;
   - a 3-layer output MLP with WaveAct.
   - Our `PINNsFormerPaper` has 453 561 parameters at these settings, the count the notebooks print.
6. **Init.** As in the notebooks: Xavier-uniform weights and biases 0.01 for every `nn.Linear`. The attention
   input projections keep PyTorch's init.
7. **Precision and device.** float32, as in the notebooks, on CPU. The paper used a GPU, so our seed 0 is not
   bit-identical to theirs.
8. **Convection reference.**
   - The notebooks score convection against `demo/convection/convection.mat`, a numerical solution.
   - Before any training, we measured that file against the exact solution sin(x − 50t) on the test grid. It
     differs by up to 0.047 (at t = 1), which is rMAE 0.0127 and rRMSE 0.0159. That is about half of
     PINNsFormer's targets.
   - **Primary score: against the exact solution.** If the official file is available locally, we also report
     the score against it. We do not redistribute it.
   - The reaction and wave notebooks use the analytic solutions, as we do.
9. **Reported loss.** This is the final training loss: the sum of the residual, boundary and initial terms at
   the last closure evaluation, as the notebooks print it.

## Status rule (`pinneapple_analysis.reproduction.status`, PR #891)

- **Score:** the median over seeds, for each number. Seed 0 is also reported on its own.
- **Reproduced:** within a factor of 2 of the paper's value.
- **Ordering claim:** PINNsFormer < PINNs on rMAE and rRMSE for each problem.
  - Outside the band but with the ordering holding: "partially reproduced".
  - Neither: "not reproduced".

## Baseline at equal budget

- PINNs is the paper's own baseline. It gets the same optimizer and the same schedule, on its own grid.
- Wall time and the number of loss evaluations are reported for every run.

## Amendment 1 (2026-10-30, before any scored run; only a 1-step smoke run was made)

- **Measured cost on this machine** (4 CPU cores, float32), from 1 outer L-BFGS step:

  | Problem | PINNs | PINNsFormer |
  |---|---|---|
  | Convection | 9.7 s | 16-28 s |
  | Reaction | 4.3 s | 11.6 s |
  | Wave | 23 s | 69 s |

- At the notebooks' schedule, one wave seed costs about 6 h (PINNs) plus 19 h (PINNsFormer).
- **Wave therefore runs seed 0 only. Seed 1 is added if time allows**, and the README will say which seeds ran.
  Table 1 keeps seeds 0, 1 and 2.
- Two workers with 2 threads each run in parallel: one for Table 1, one for wave.
- Nothing else changes.
