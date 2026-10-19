# Surrogates from simulation results: study, regime arena, gateway

Three library pieces turn a table of simulation runs into a model that others can call, and say where it can be
trusted:

| Piece | What it does |
|---|---|
| `pinneapple_analysis.surrogate.run_study` | trains every applicable method on the same splits and budget, measures on unseen data, picks one by validation error, packages it with its limitations and a trust report |
| `pinneapple_arena.regimes.run_regime_arena` | ranks methods per data regime (interpolation, few data, extrapolation, geometry change, steep fronts) over several seeds, with physics violations, solver baselines and a dataset audit |
| `pinneapple_registry.ModelGateway` | publishes versions, runs them behind one `predict` with the model's OOD policy (warn or refuse), logs every call, aggregates latency / counts / errors, exports TorchScript and ONNX |

## A surrogate study

```python
from pinneapple_analysis import surrogate as S

data = S.examples.airfoil_rans()             # 374 OpenFOAM RANS runs: 6 CST weights + alpha -> Cl, Cd
study = S.run_study(data, budget=3.0, seeds=2)          # 3 s of wall clock per training run
print(study.best, study.ranking("test"), study.ranking("shift"))
print(study.trust["decision"], study.limitations["concentration"])
out = study.model.predict({"T0": 0.15, "T1": 0.12, "T2": 0.13, "M0": 0.02, "M1": 0.05, "M2": 0.04,
                           "alpha_deg": [4.0, 30.0]})
print(out["outputs"]["Cl"], out["ood"][1]["reasons"])      # alpha 30 is flagged: outside the training range
study.model.save("airfoil_model")
```

Your own data: `S.read_table(bytes_or_path, name)` reads CSV, Parquet, JSON, HDF5 and NPZ; `S.from_table(df,
inputs, outputs, group=...)` picks the columns. A field output is a set of columns `u_0, u_1, ...` (with
`grid_shape=` and optionally `coords=`), or `S.from_arrays({"X": ..., "Y": (n, *grid), "coords": ...})`.

**Which methods apply** (`S.applicable(kind, n_train, grid_shape, has_pde)`, each with the reason when not):
polynomial ridge and an MLP always; a Gaussian process up to 2000 training samples (on POD coefficients for fields);
gradient-boosted trees (XGBoost if installed, else scikit-learn) for scalars; FNO and DeepONet for fields; a PINN
only for fields with a declared PDE (`pde={"type": "burgers", "coords": {"x": "x", "t": "t"}, "coefficients":
{"nu": "nu"}}`; types in `S.PDE_TYPES`).

**Equal budget.** The iterative methods (MLP, FNO, DeepONet, PINN) train until the same `ComputeBudget` is spent
(`pinneapple_analysis.cost.equal_budget.train_within_budget`) and keep the best of ten validation checkpoints; the
one-shot fits (ridge, GP, trees) are fitted once and report the time they used.

**Unseen data.** `S.make_splits` carves out a distribution-shift set first (whole groups, the edge of the input
space, or one end of an input) and then a random held-out test set. The model is selected by validation error; the
test and shift sets only measure it.

**Limitations** (`study.limitations`): training-range validity bounds, the OOD gate (training range plus
nearest-neighbour distance, the batched form of `physics_audit.Envelope`), the error per input quartile and where
it concentrates, the worst samples, the extrapolation ratio, and for fields the mean error map and, with a declared
PDE, the finite-difference residual of the predictions relative to the reference data.

**Trust report** (`pinneapple_analysis.trust.TrustReport`): held-out error against the target (0.10 nRMSE or 0.05
relative L2 by default), R² > 0 on every output (critical), shift error within 3x the held-out error, seed spread,
and the PDE residual check for fields. A check that could not run says why.

## Regime arena

```python
from pinneapple_arena.regimes import run_regime_arena

data = S.examples.burgers_fields()           # exact viscous Burgers fields for 96 viscosities
result = run_regime_arena(data, budget=3.0, seeds=3, baselines={"fdm": S.examples.burgers_fdm_baseline()})
print(result.rankings["extrapolation"], result.overall, result.baselines, result.audit["decision"])
```

The protocol is in `result.protocol`: one split for every method and seed (the model seed varies, the data does
not), the same budget and stopping rule, default hyper-parameters (no per-method search), the dataset hash, the
hardware and date. Physics violations are the share of predictions breaking the dataset's declared constraints
(`pinneapple_data.audit.constraint_mask`, e.g. `Cd > 0`, the maximum principle `abs(u) <= 1.005`); the splits are
audited by `pinneapple_data.audit.audit_dataset` (the checks behind `pinneapple check` PC451-PC453). Regimes the data
cannot support (no groups, scalar outputs for steep fronts) are reported as unavailable with the reason; graph
networks and R-OAMS are listed in `NOT_IN_LIBRARY` with why they are not run.

## Gateway

```python
from pinneapple_registry import ModelGateway

gw = ModelGateway("./models")
v = gw.publish("airfoil-cl-cd", "airfoil_model", ood_policy="refuse")      # or a SurrogateModel
gw.publish("airfoil-zoo", "airfoil_cst_rans_mlp", kind="pinneapple.zoo")    # zoo models too
r = gw.predict("airfoil-cl-cd", [{"T0": 0.15, "T1": 0.12, "T2": 0.13, "M0": 0.02, "M1": 0.05, "M2": 0.04,
                                  "alpha_deg": 4.0}])
print(r["outputs"], r["ood"], r["latency_ms"], gw.metrics()["by_version"])
gw.export("airfoil-cl-cd", None, "onnx", "airfoil.onnx")                    # PyTorch surrogates only
```

Versions follow `ModelStore` (`vYYYYMMDD_HHMMSS`, stages, a REJECT trust report blocks staging and production
unless an override reason is recorded). A caller may tighten a model's OOD policy (warn to refuse), never relax it.
Surrogates are unpickled only after their sha256 matches their card: publish only directories you produced.

Other servers with the same API (the Surrogate Studio, the Benchmark Arena, any app built on `gateway_router`) are
reachable from Python with `pinneapple_registry.gateway_client.GatewayClient(base_url, api_key=...)`. A gateway can
serve them too:

- `gw.publish(name, "https://host/v1/models/<model>", kind="remote")` forwards to the remote model;
- `gw.publish_ref(model_ref)` publishes from a `ModelRef`: `zoo:<name>`, or an http(s) URI whose input ranges become
  the validity bounds.

The front gateway enforces its own bounds and OOD policy, and logs and meters every call before forwarding.
