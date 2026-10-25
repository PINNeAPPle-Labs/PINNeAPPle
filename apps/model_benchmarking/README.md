# Model Benchmarking

**Four PINN architectures trained live on the same viscous Burgers problem, ranked by accuracy and by a check on the
PDE residual.**

One fixed comparison. `burgers_1d` of `pinneapple_arena` (u_t + u u_x = ν u_xx, u(x, 0) = −sin πx, ν = 0.01/π, the
setup of Raissi, Perdikaris & Karniadakis 2019; Cole-Hopf reference) with Vanilla PINN (4×64, 12,000 epochs), SIREN
(3×64, ω₀ = 15, 4,000), PIELM (128 random features, 5,000) and XTFC (400), 4,000 collocation points and a 64×64
evaluation grid. Reported per model: relative L2, relative L2 in the front band |x| ≤ 0.3, L∞, PDE residual, train
time and size, and the t = 1 field fit. The accuracy ranking is Arena's `rank_by_accuracy`; `physics_aware_rank`
flags residual outliers (median + 2σ of the group) without reordering it.

Ported from the `poc_model_benchmarking` case study of PINNeAPPle-apps. Not the same thing as
[`apps/benchmark_arena`](../benchmark_arena/README.md), which ranks methods on stored public benchmark runs and private
uploads; this app is one live, didactic comparison.

## What it computes

All of it is `pinneapple_arena.showcase.run_benchmark` (Arena's `build_model`, `train_pinn`, `evaluate_model`,
`rank_by_accuracy`, `physics_aware_rank`); the app runs it in a background thread and serves the page.

Measured here (seed 0, full budgets, 30 min on this heavily shared 4-core container; the source project reported
3-5 min):

| Model | rel. L2 | front band | L∞ | PDE residual | train |
|---|---|---|---|---|---|
| Vanilla PINN | 1.19 % | 1.73 % | 0.097 | 1.8e-4 | 1494 s |
| SIREN | 1.73 % | 2.26 % | 0.065 | 7.2e-6 | 159 s |
| PIELM | 52.4 % | 70.9 % | 0.95 | 0.170 | 26 s |
| XTFC | 55.0 % | 74.0 % | 1.03 | 0.163 | 99 s |

**No model was flagged.** The two poor models have residuals ~1,000× those of the two good ones, but with four
models they make the group median 0.082 and σ 0.083, so the threshold median + 2σ = 0.248 is above both. This is the
blind spot the source project's TODO recorded ("the relative residual threshold missed PIELM/XTFC"), reproduced here.
`physics_aware_rank(..., residual_threshold=...)` adds an absolute cutoff; the app does not set one (there is no
agreed absolute value for this problem), so read the residual column. The source project's single-run numbers
(Vanilla PINN 2.47 %, SIREN 2.45 %; PIELM/XTFC 50-55 %) are of the same order; it also reported that unseeded
Vanilla PINN runs ranged 2.5-41 %, which was not re-measured here.

## Run it

```bash
# from the repository root
pip install -e . fastapi "uvicorn[standard]"
cd apps/model_benchmarking
PYTHONPATH=../_shared uvicorn model_benchmarking.api:app --port 8162      # open http://localhost:8162
```

A run starts at startup. `GET /api/sweep_status` (progress, result, the epoch budgets), `POST /api/run_sweep` (409
while running), `GET /health`.

Environment: `MBENCH_USER` / `MBENCH_PASSWORD` (or `MBENCH_PASSWORD_HASH`), `MBENCH_AUTOSTART=0`,
`MBENCH_EPOCH_SCALE` (scale every budget, e.g. 0.1 for a quick demo; the status reports the budgets used).

## Limitations

- One problem, four architectures, the source project's budgets; unseeded runs vary. PIELM and XTFC are trained by the
  same Adam loop (Arena's closed-form ridge fit applies to linear PDEs only); both smooth through the front.
- Left out on purpose (source project's findings): `vpinn` builds the same network as `vanilla_pinn` under this
  trainer; `hash_grid_mlp` and `bench_fourier_mlp` collapsed towards u = 0.
- One uvicorn worker holds the run state in memory; a restart loses the last result.
