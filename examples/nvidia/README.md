# PINNeAPPle with the NVIDIA stack

Worked examples that join PINNeAPPle (problem, reference solvers, physics checks, trust decision) with NVIDIA
PhysicsNeMo, Omniverse (OpenUSD), Isaac Sim, Cosmos and NIM. The glue lives in `pinneapple_tools.nvidia`; each
example is a thin `run.py`, writes `results/architecture.png` (who does what, and what was *not* run here),
result figures, `summary.json` and `run_info.json` (versions, hardware, timings, which NVIDIA components ran).

| # | Example | PhysicsNeMo | Omniverse | Isaac Sim | Cosmos | NIM | Decision (this run) |
|---|---|---|---|---|---|---|---|
| 01 | [Cold plate surrogate](01_coldplate_physicsnemo_omniverse/README.md) | FNO | USD stage | | | | REJECT (energy balance) |
| 02 | [Pump and pipeline twin](02_pump_pipeline_twin_omniverse/README.md) | | USD timeline | | | | APPROVED (60 s closure only) |
| 03 | [Plume rollout](03_plume_rollout_physicsnemo/README.md) | FNO rollout | | | | | REVIEW (10-step horizon) |

What runs where: everything PINNeAPPle, PhysicsNeMo (CPU) and OpenUSD (`usd-core`) runs on a laptop and was run for
the numbers above. Omniverse Kit RTX rendering, Isaac Sim, the Cosmos models and NIM need an NVIDIA GPU and/or an
`NVIDIA_API_KEY`; those steps are written against the official APIs, tested against local fakes of their HTTP
contracts where they have one, and drawn hatched in the architecture figures.

```bash
pip install nvidia-physicsnemo usd-core
pytest tests/test_nvidia_integrations.py -q
```
