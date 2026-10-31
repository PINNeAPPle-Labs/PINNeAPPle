# PINNsFormer (Zhao, Ding & Prakash, ICLR 2024): reproduction

Z. Zhao, X. Ding, B. A. Prakash, "PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks",
ICLR 2024, arXiv:2307.11833v3.

## How the targets were set

- **Frozen before any run:** [TARGETS.md](TARGETS.md), with three amendments.
  - Amendments 1 and 2 were written before any test error had been computed. They set the run order and seed
    counts from the measured CPU cost.
  - Amendment 3 sets the scope of this PR.
- **Ambiguities:** the paper leaves out details, which are taken from the official code (AdityaLab/pinnsformer at
  `00d790f5`). That code is read, not copied or run.
  - Its notebooks print exactly the numbers of Tables 1 and 2. **So every number in those tables is one run at
    seed 0, on a GPU.**
- **Produced by `run_pinnsformer.py`:** [RESULTS.md](RESULTS.md), `all_runs.json`, the figure and
  [experiment.json](experiment.json).

## What is implemented

`pinneapple_neural.architectures.pinns.pinnsformer_paper.PINNsFormerPaper` is the paper's model:

- the pseudo sequence of one point and its k time-shifted copies (eq. 3);
- the linear spatio-temporal mixer;
- an encoder with self-attention, and a decoder with encoder-decoder attention only;
- the output MLP;
- the learnable wavelet activation (eq. 4).

At the paper's settings it has **453 561 parameters**, the count the official notebooks print. Because attention
only mixes the copies of one point, the model is **pointwise**: u(x, t) does not depend on the other points in the
batch (a test checks this).

The library's older `PINNsFormer` attends across *different* collocation points, so it is not the paper's model
and is not pointwise. It stays as it was, for its callers (the arena's sequence layout and two examples). Its
registry capability now says `"sequence"` instead of `"pointwise_coords"`, and its docstring and catalog entry
point to `PINNsFormerPaper`.

## Result (Table 1, seed 0; Table 2 not attempted)

The rule scores the median over seeds. **Here that is one seed (0), the same count as the paper.** Each number is
one training run, so the statuses below carry a single run's luck. The paper's numbers carry the same.

| Problem | Model | Paper (loss / rMAE / rRMSE) | Ours, seed 0 | Status |
|---|---|---|---|---|
| Convection | PINN | 0.016 / 0.778 / 0.840 | 0.0108 / 0.657 / 0.737 | 3 of 3 reproduced |
| Convection | PINNsFormer | 3.7e-5 / 0.023 / 0.027 | 0.312 / 1.06 / 1.12 | 3 of 3 not reproduced |
| 1D reaction | PINN | 0.199 / 0.982 / 0.981 | 3.6e-5 / 0.046 / 0.079 | 3 of 3 partially reproduced |
| 1D reaction | PINNsFormer | 3.0e-6 / 0.015 / 0.030 | 3.95e-6 / 0.010 / 0.020 | 3 of 3 reproduced |

**6 of 12 reproduced, 3 partially reproduced, 3 not reproduced.** The 6 wave targets (Table 2) are not attempted.

- **1D reaction.**
  - PINNsFormer matches the paper on all three numbers.
  - Our PINN does **not** show the failure mode the paper reports: its rRMSE is 0.079, against 0.981. These rows
    count as "partially reproduced": the paper's ordering (PINNsFormer below PINN) still holds. Our PINN's rMAE
    and rRMSE are 12 to 21 times lower than the paper's, and its loss is about 5 500 times lower.
  - One seed cannot tell whether the paper's PINN failure is typical or that seed's outcome.
- **Convection.**
  - Our PINN fails as the paper's does (rRMSE 0.74 against 0.84).
  - Our PINNsFormer fails too. Its loss stalls at 0.312 from loss evaluation 2930 of 5152. It never fits the initial
    condition (the IC term stays at 0.25) and ends at a nearly constant field (figure, top right).
  - The paper reports rRMSE 0.027, so this does not reproduce. Again, one seed cannot separate a typical failure
    from a seed effect. The paper's own run is also a single GPU seed.
- **Not attempted.** Wave (Table 2) and seeds 1 and 2 of Table 1. On this machine, one wave seed costs about 12 h
  (PINN) plus about 36 h (PINNsFormer) (TARGETS.md, amendment 3). The runner checkpoints, so these can be added
  without changing anything else.
- **Reference.** The convection numbers are scored against the exact solution sin(x − 50t). The authors' file
  `convection.mat` differs from it by rMAE 0.013 and rRMSE 0.016 (ambiguity 8). That changes neither convection
  status: our PINNsFormer error is about 70 times larger than that difference.

![seed 0 predictions](predictions_seed0.png)

## Cost

On 4 CPU cores in float32, sharing the machine with another reproduction:

| Run | Time | Loss evaluations |
|---|---|---|
| Reaction PINN | 82 min | 6 581 |
| Reaction PINNsFormer | 87 min | 1 979 |
| Convection PINN | 267 min | 10 951 |
| Convection PINNsFormer | 272 min | 5 152 |

The official notebooks report 1.4 to 18 minutes per run on a V100.

## Reproduce

```bash
python examples/reproductions/pinnsformer_zhao2024/run_pinnsformer.py --queue all     # resumable, many hours on CPU
python examples/reproductions/pinnsformer_zhao2024/run_pinnsformer.py --report
python examples/reproductions/pinnsformer_zhao2024/run_pinnsformer.py --smoke         # 1 outer step per run
```
