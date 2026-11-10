# Hybrid DeepONet and real-measurement transfer calibration

These APIs extend the existing `pinneapple_analysis.surrogate` stack rather than introducing a separate model registry.

## Architecture

`HybridDeepONet` retains the standard branch/trunk contraction and allows
independent choices of `mlp`, `rbf_kan`, and `fourier` on **both** sides.

`rbf_kan` is an edge-wise learnable Gaussian basis (FastKAN-style). It is **not**
a claim to implement the original B-spline KAN. `fourier` uses deterministic
Fourier features followed by an MLP. `deeponet_hybrid` is registered in
`CANDIDATES`, so `run_study(..., methods=["deeponet_hybrid"],
options={"deeponet_hybrid": {"branch": "rbf_kan", "trunk": "fourier"}})`
uses the existing compute-budget trainer.

## Selecting a hybrid architecture from measured fields

```python
from pinneapple_analysis.surrogate import SurrogateData, hybrid_search

# X: (n_experiments, n_parameters)
# Y: (n_experiments, n_grid_points) or (n_experiments, ny, nx)
# groups: experiment/geometry IDs; all partitions remain group-disjoint.
data = SurrogateData(
    X, Y, input_names=["pressure", "temperature"],
    output_names=["velocity"], coords=coords, coord_names=["x"],
    groups=experiment_ids, meta={"source": "experimental campaign 2026"},
)
study = hybrid_search(
    data, branches=("mlp", "rbf_kan", "fourier"),
    trunks=("mlp", "fourier", "rbf_kan"),
    max_trials=8, budget_s=10.0, seed=42,
)
print(study.winner, study.holdout)
study.model.save("trusted_local_experiment")
```

Training/validation decide the architecture. Test and distribution-shift data
are only evaluated after selection. Data are standardized using the training
split only. The final model preserves its training-domain OOD gate.

## Adapting a previously trained model to real data

```python
from pinneapple_analysis.surrogate import fine_tune_real

# Only load checkpoints produced by a trusted system. pickle is executable.
cal = fine_tune_real(
    study.model, new_measurements,
    epochs=100, lr=1e-4, patience=20,
    anchor_weight=0.05, freeze=("trunk",),
    allow_ood=False,
)
print(cal.baseline["test"], cal.holdout["test"])
cal.model.save("trusted_local_finetuned")
```

The fine tuner preserves pretrained scalers and coordinates, supports optional
branch/trunk freezing and a prediction-anchoring penalty, and compares against
the parent model. Mesh and variable alignment is explicitly required. If the
new measurements are outside the original gate, explicit `allow_ood=True`
is needed; this **does not expand** the certified domain. Approvals are reset
to `not_verified` after adaptation, and publication needs independent
physical verification.

## Limitations / follow-up

- Trial search is seeded random search, not Bayesian/Optuna optimization.
- Field samples must share aligned coordinates. No unstructured, variable-mesh
  DeepONet representation is claimed.
- Real measurements must carry input/output units, calibration provenance,
  group IDs (when related), and independent validation plans.
- Domain-specific conservation residuals, uncertainty and uncertainty-aware
  calibration must be separately validated before engineering deployment.
- Checkpoint `model.pkl` is Python pickle: never deserialize untrusted uploads.
- The test modules cover interface contracts, not physical benchmark accuracy.
