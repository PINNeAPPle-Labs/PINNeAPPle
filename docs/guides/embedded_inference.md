# A model that flies: the residual in the graph, and the hertz it costs

A flight computer cannot afford a Python interpreter between the prediction and the decision. This guide is the whole
path — a field surrogate, the residual of its equation, the sensor check and the two flags — exported as **one
straight-line ONNX graph** and measured on the CPU it would run on. Everything below is a number this repository
produced; the example that produces it is [`examples/embedded/embedded_inference.py`](../../examples/embedded/embedded_inference.py).

```python
import torch
from pinneapple_physics.embedded import EmbeddedSpec, EmbeddedRuntime, export, graph_summary

model = torch.nn.Sequential(                                   # (2 frames, H, W) -> the next frame
    torch.nn.Conv2d(2, 16, 3, padding=1, padding_mode="circular"), torch.nn.Tanh(),
    torch.nn.Conv2d(16, 16, 3, padding=1, padding_mode="circular"), torch.nn.Tanh(),
    torch.nn.Conv2d(16, 1, 3, padding=1, padding_mode="circular"))

spec = EmbeddedSpec(state_shape=(2, 32, 64), equation="advection-diffusion",
                    parameters={"c_x": 1.0, "c_y": 0.35, "alpha": 0.01}, dt=0.5, tolerance=0.094)
graph_summary(export(model, spec, "flight.onnx")).is_straight_line      # True

rt = EmbeddedRuntime("flight.onnx", threads=1)
out = rt.step(state, measurement)          # prediction, residual, trust, off_physics, suspect, measurement_error
rt.latency(state, measurement)["hz"]       # ~1800 on one ARM core; see the table below
```

## What goes into the graph

Six outputs, one file, no `If`/`Loop`/`Scan` anywhere — 19 op types, 25 042 bytes:

| output | what it is |
| --- | --- |
| `prediction` | the model's next frame |
| `residual` | `RMS(u_t + c·∇u − α∇²u) / mean|u|` of that prediction, the stencil written as convolutions |
| `trust` | `1 − residual/tolerance`, clipped — a level a caller can threshold at any height |
| `off_physics` | residual above its tolerance |
| `suspect` | the sensor reading and the prediction disagree by more than the measurement tolerance |
| `measurement_error` | by how much, relative to the sensor's scale |

The equations the stencil knows are `advection-diffusion`, `advection` and `heat`, all on periodic boundaries.
An impossible contract is refused at construction, not at run time: one frame has no time derivative, a 2×8 grid has no
3×3 neighbourhood, a channel outside the state cannot be compared with a sensor.

## The two mistakes the measurements caught

Both are worth writing down, because both produce a graph that exports, runs, and is wrong.

**1. The residual has to be measured against the equation *the stencil discretises*.** The first version trained on the
exact solution of the continuous PDE, propagated in Fourier. Under the centred-difference stencil that solution scores
**1.70** — above what the trained model scored — and plain persistence (`next frame = last frame`) scored **0.94**,
i.e. better than the truth. The cause is truncation: the discrete Laplacian's eigenvalue at wavenumber `k` is
`2cos(k) − 2`, which is `−9.0` where the continuous one is `−9.0`… at `k = 3` it is `−2.98`. On a 64-wide grid, a
broadband field puts most of its energy where a centred difference is simply not the derivative.

Generating the data with the *discrete* operator instead — backward Euler, solved exactly in Fourier, so
`(u_{k+1} − u_k)/dt + c·∇u_{k+1} − α∇²u_{k+1} = 0` to round-off — makes the exact next frame score **0.000**, and the
residual becomes the model's error and nothing else.

**2. `dt` decides whether the model has anything to predict.** The residual divides a frame difference by `dt`. At
`dt = 0.01` the per-step change of this field is `5·10⁻³` while the model's frame error is `8·10⁻³`: the model is wrong
about the step by more than the step, so *every* prediction is flagged, the trained model scores worse than doing
nothing, and no threshold separates anything. At `dt = 0.5` the change is `0.25` against the same error and the
residual ranks the cases the way the physics does.

## The flag, on a model trained on one distribution

The tolerance is not a guess: it is **1.5× the median residual of a held-out in-domain sample**, computed before the
graph is exported (`1.5 × 0.0626 = 0.094` here). Applied unchanged to every other case:

| case | what changed | median residual | flagged at 0.094 |
| --- | --- | --- | --- |
| `in-domain` | nothing — from the training distribution | 0.0626 | 0 % |
| `alpha-5x` | diffusion ×5 (`α = 0.05`) | 0.0635 | 0 % |
| `alpha-50x` | diffusion ×50 (`α = 0.5`) | 0.2164 | 100 % |
| `speed-2.5x` | velocity ×2.5 (`c = (2.5, 0.9)`) | 0.4084 | 100 % |
| `static` | a field that does not move at all | 0.3545 | 100 % |

The `alpha-5x` row is the honest one. At this resolution the diffusive term is ~50× smaller than the advective one
(`α·k² ` against `c·k`), so five times more of it is still invisible to the equation's balance — and the flag says so
by not firing. The residual measures the equation, not the configuration: a case can be out of distribution and
physically indistinguishable, and that is a property worth knowing before trusting a flag that did fire.

## The hertz

`EmbeddedRuntime.latency` times the whole graph per call, after a warm-up, on the threads it was given:

| threads | median | p95 | rate (median) | rate (p95) |
| --- | --- | --- | --- | --- |
| 1 | 0.547 ms | 0.640 ms | 1827 Hz | 1564 Hz |
| 4 | 0.352 ms | 0.594 ms | 2842 Hz | 1683 Hz |

Measured on an Apple ARM core with `onnxruntime`'s CPU provider, `GraphOptimizationLevel.ORT_ENABLE_ALL`. Across four
runs of the example on the same machine the one-thread median moved between **0.55 and 0.66 ms** (1827 Hz down to
1516 Hz) depending on what else the machine was doing — quote the range, not the best run. This is the budget of *this*
graph on *this* CPU: a Raspberry Pi 5 or a Jetson would be the same kind of number, not the same number. Note the
four-thread row: the median improves ~1.5× while the p95 in the earlier runs was *worse* than one thread, because a
graph this small spends more time in the thread pool than in the arithmetic. **One thread is the right deployment for
a graph below ~1 ms**, which is also the thread budget an embedded core has.

## What this is not

Not a quantised or pruned graph — the arithmetic is the model's own. Not a CUDA graph, not the PX4/ROS 2 bridge: that
needs a board with the flight stack on it, and the drone of issue #335 is a separate piece of work. What ships here is
the graph and the runtime that calls it, with the two flags a flight controller acts on.

See also: [the trust certificate and the verification service](verification_service.md) for the same idea at
simulation scale, and [surrogates from simulation results](surrogates_arena_gateway.md) for the study a model like this
comes out of.
