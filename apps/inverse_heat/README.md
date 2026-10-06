# Inverse Heat Lab

**The convection coefficient h from a few thermocouples, with a physics-informed neural network, in 1D, 2D and 3D,
and every answer checked against an independent solution.**

Day 7 of the PINNeAPPle 30-app program. A heat-transfer coefficient is the number every thermal calculation needs and
nobody can measure directly: correlations are typically good to ±25 % at best, and with a wrong guess the 2D example's hot spot reads 35 °C optimistic.
Thermocouples are cheap. This app turns their readings into h, the whole temperature field and the temperatures nobody
measured, with a PINN built by PINNeAPPle's `PINNFactory`: the equation is written as text, h is a trainable parameter,
and the network learns the field and h together from the physics and the readings.

| Tab | Case | Live? | Checked against |
|---|---|---|---|
| **1D · fin** | Pin fin on a hot wall, thermocouples along it. Demo readings, or **your own** (material, size, positions, readings). | yes, 20–40 s | least-squares fit of the analytic fin solution to the same readings |
| **2D · plate** | Aluminium heat spreader, a device in one corner, eight thermocouples. Change power, true h, noise, starting guess and train. | yes, about 2 min | independent finite-volume solution, cell by cell |
| **3D · block** | 10 W chip under a steel block, thermocouples only on top: how hot is the chip? | full run, computed offline (30 min on a CPU) | finite volumes; interactive 3D field and layer slices |
| **Code** | How to reproduce each case with `pip install pinneapple`; the 1D script runs as is. | | |

## Results of the full runs ([`examples/use_cases/fin_convection_inverse`](../../examples/use_cases/fin_convection_inverse))

- 1D, 10 noise draws: h = 24.8 ± 0.4 W/m²K for a true 25; least squares on the analytic profile 24.8 ± 0.3.
- 2D, 5 draws: h = 14.8 ± 0.3 for a true 15; hot spot 79.2 °C vs 79.1 °C; worst map error 1.34 °C.
- 3D, 3 draws: chip 74.6 °C vs 74.5 °C (never measured); h from the energy balance on the learned field 149.5 for a
  true 150. The network's own h parameter settles 7 % low (139): the field is pinned down by the readings, the
  parameter is weakly constrained in a thick conductive block. Both are reported.

The live 2D run trains 1500 steps instead of 3000: the map converges first, so expect h within a few percent there.

## Run

```bash
pip install -e . -r apps/inverse_heat/requirements.txt         # from the repository root
cd apps/inverse_heat && uvicorn inverse_heat.api:app --port 8086 --workers 1
```

One worker only: training jobs live in its memory. `IHL_MAX_JOBS` (default 2) trainings run in parallel threads, each
on `IHL_THREADS` torch threads (default 1); more get HTTP 429. `IHL_USER` / `IHL_PASSWORD` turn on an HTTP Basic login.
Docker: `docker build -f apps/inverse_heat/Dockerfile -t inverse-heat .` from the repository root; deployment with
the other apps in [`apps/deploy`](../deploy) (service `inverse`).

## API

`POST /api/fin/run` (`k, d_mm, length_mm, t_base, t_air, sensors_mm`, and `readings`, or `h_true, noise` for demo
readings) and `POST /api/plate/run` (`power, h_true, noise, h_guess`) return a job id; `GET /api/jobs/{id}` gives the
progress (step, current h, history) and the result. `GET /api/plate/precomputed` and `/api/block/precomputed` return
the full runs. Interactive docs at `/docs`.

## Assumptions

Steady state, one h over the convective surface (an average if it varies), no radiation (h is then an effective
coefficient including it), good thermocouple contact. The 1D fin needs a small cross-section Biot number hD/2k; the
page shows it for your case.

Tests: `tests/test_inverse_heat.py` (engine, input checks, job API, and the published 1D script run end to end).
