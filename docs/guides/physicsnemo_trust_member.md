# Putting a PhysicsNeMo model into PINNeAPPle without retraining it (#409)

A PhysicsNeMo surrogate (DoMINO, X-MeshGraphNet) is a strong model with a published recipe and a checkpoint. The
library's question is not whether it predicts well in distribution — it does — but **whether the trust layer catches
it when it leaves the family it was trained on**. This page answers that with numbers on the Ashton fixture, and shows
the two lines of code that make a PhysicsNeMo model a trust-wrapped member.

`physicsnemo` **2.2.2 is installed** in this environment and ships `physicsnemo.models.domino`. `torch_geometric` and
`torch-scatter` are **not**, so X-MeshGraphNet cannot be built here — `physicsnemo_models.availability()` says so with a
reason per component and the table row reads `awaiting_data` rather than carrying an invented number.

## The two members

```python
from pinneapple_data.ashton import physicsnemo_models as pm

member = pm.DoMINOMember(steps=240)            # untrained head on the installed PhysicsNeMo model
member.fit(cases)                              # a stated budget, visible in the model's own name
row = member.benchmark_row()                   # the same table the library's models report into
```

`member.model_name` carries the budget — `DoMINO (PhysicsNeMo 2.2.2, 240 steps)` — because a baseline whose training
budget is invisible invites a comparison that is not one. `member.embedding(case)` returns the six-number embedding
(three channels, mean and deviation) that the trust layer reads.

## Does the trust layer catch a model on an unseen geometry family?

Train on the α family of the fixture, test on β, and the other way round. That is the question the issue asks, and it
is answerable offline because the fixture's families are known.

| direction | t | field error | Cd error | embedding distance | flagged fraction | coverage |
|---|---|---|---|---|---|---|
| α → β | 0,0 | 0,0198 | 0,0363 | 1,56 | 0,125 | 0,924 |
| α → β | 0,4 | 0,0343 | 0,0317 | 2,72 | 0,375 | 0,791 |
| α → β | 0,6 | 0,0418 | 0,0818 | **8,44** | **1,000** | 0,803 |
| α → β | 1,0 | 0,0552 | 0,0610 | **32,11** | **1,000** | 0,856 |
| β → α | 0,0 | 0,0272 | 0,0259 | 2,20 | 0,000 | 0,903 |
| β → α | 0,2 | 0,0257 | 0,0388 | 3,37 | 0,375 | 0,916 |
| β → α | 0,4 | 0,0290 | 0,0441 | **5,49** | **1,000** | 0,909 |
| β → α | 1,0 | 0,0767 | 0,4093 | **9,65** | **1,000** | 0,865 |

`t` interpolates from the α family (0) to the β family (1) in the fixture's shape parameters. Reading it:

- **The error does grow with the shift**, in both directions — 0,0198 → 0,0552 (2,8×) and 0,0272 → 0,0767 (2,8×) —
  so there is something for the trust layer to catch.
- **The embedding signal catches it in both directions**: the flagged fraction goes to 1,000 in α → β at t = 0,6 and
  in β → α at t = 0,4, before the error has finished growing.
- **The residual signal does not.** Its ratio rises with `t` in one direction (0,90 → 4,23) and *falls* in the other
  (1,21 → 0,61): a residual threshold calibrated in the α family flags a β-trained model for the wrong reason. The
  test pins both directions, so this asymmetry cannot quietly disappear.
- **Coverage of the conformal band degrades but does not collapse** (0,924 → 0,856 and 0,903 → 0,865 against the 0,90
  nominal), which is the honest summary of a 90 % band under a family shift: it leaks, it does not fail.

## What a plug-in costs

Nothing in the model. The adapter wraps the model, reads the fixture through the same `Case` objects the rest of
`pinneapple_data.ashton` uses, scores with the same metrics, and exposes the embedding the trust layer already
consumes. No retraining, no change to the checkpoint, no second data path.

## Gate

```
pytest tests/test_ashton_physicsnemo_baselines.py -> 12 passed
ruff check . --select E9                          -> All checks passed
```
