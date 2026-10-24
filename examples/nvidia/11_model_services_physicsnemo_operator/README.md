# 11 A PhysicsNeMo model served with its physics: PINNeAPPle + Model Services + PhysicsNeMo

**What.** The PhysicsNeMo FNO of example 07 (cold-plate conductivity layout -> temperature, 32 x 32, 2700 steps)
is served on `/v1/infer` next to PINNeAPPle's finite-volume solver. Every answer is checked without a reference
(certified error bounds from the PDE residual, boundary condition, energy balance, maximum principle, layout against
the training envelope). The client asks the surrogate first and the solver whenever the surrogate is not APPROVED.

**Why.** At serving time there is no reference solution; the residual is still there. A service that checks its own
answer lets a client route around a surrogate that is wrong, and shows when the surrogate is not worth serving.

| Step | Who | Ran here |
|---|---|---|
| training data, checks, routing logic | PINNeAPPle | yes |
| FNO training and inference | PhysicsNeMo 2.2.2 (CPU) | yes |
| services and client (`grid_operator_service`, `darcy_solver_service`) | PINNeAPPle Model Services | yes |

```bash
python examples/nvidia/11_model_services_physicsnemo_operator/run.py     # ~3 min on 4 CPU cores
```

## Results (this run: 4-core Xeon 2.1 GHz)

- 36 requests: 20 in-distribution layouts, 8 with higher contrast (k 2 / 20), 8 with finer features.
- **No surrogate answer was APPROVED**:
  - In-distribution: true error 1.9 %, but every answer fails the certified residual bounds, and 15 of 20 fail the
    energy balance (the same finding as example 07).
  - Higher contrast: error 25 %; every check fails, including the training envelope.
  - Finer features: error 1.9 %; residual bounds fail. The envelope check did *not* flag these: the residual did.
- Every request was routed to the solver, whose answers were all APPROVED (exact to round-off).
- At this size the solver is also faster: median 7.7 ms per request against 20 ms for the FNO. **Decision for the
  served system: APPROVED** (every answer delivered is correct); for the surrogate alone: not usable here.

Figures in `results/`: `architecture.png`, `routing.png` (true error of each surrogate answer coloured by its
decision; latencies), `checks.png`.
