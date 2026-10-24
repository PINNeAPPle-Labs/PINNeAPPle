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
python examples/nvidia/11_model_services_physicsnemo_operator/run.py     # ~4 min on 4 CPU cores
```

## Results (this run: 4-core Xeon 2.1 GHz)

- 36 requests: 20 in-distribution layouts, 8 with higher contrast (k 2 / 20), 8 with finer features.
- **No surrogate answer was APPROVED, and that is the intended behaviour of a reference-free check:**
  - At serving time there is no reference. The service can only certify an upper bound on the error, computed from
    the PDE residual, and it approves an answer when that certified bound is at most 5 % (the library's fixed
    tolerance, `darcy_checks.DEFAULT_TOLERANCES`).
  - In-distribution: true error **1.0 to 5.0 %** (median 1.9 %), certified bound **16 to 35 %**. The bound is about
    **14 times** the true error: the FNO's fields carry a small, rough residual that the bound has to assume could be
    error. 15 of 20 also miss the 2 % energy balance (the same finding as example 07).
  - So the service cannot *certify* these answers. Approving them on their true error would need the reference the
    service does not have. Loosening the limit to about 35 % would admit answers it cannot tell from a 35 % error.
  - Higher contrast: true error 25 %, bound 190 to 360 %; every check fails, including the training envelope for 2.
  - Finer features: true error 1.9 %; residual bounds fail. The envelope check did *not* flag these: the residual did.
- Every request was routed to the solver, whose answers were all APPROVED (exact to round-off). **Decision for the
  served system: APPROVED** (every answer delivered is correct). For the surrogate alone: not certifiable here.
- **Latency.** At 32 x 32 the FV solve is tiny (2.4 ms; 9.6 ms round trip with checks and HTTP), while the FNO pays
  the fixed cost of a torch forward pass (10 ms; 26 ms round trip). The third panel of `routing.png` times both, one
  layout per call, against the grid size, with the served FNO architecture (its cost does not depend on its weights):

  | grid | FV solve | FNO inference |
  |---|---|---|
  | 32 x 32 | 2.4 ms | 10.2 ms |
  | 64 x 64 | 8.9 ms | 11.6 ms |
  | 128 x 128 | 57 ms | 29 ms |
  | 256 x 256 | 415 ms | 106 ms |

  The surrogate pays from about 100 x 100 on this CPU, 4x faster at 256 x 256. A surrogate worth serving here would
  be trained at that size and would need a smaller certified bound (a smoother residual) to be approved.

Figures in `results/`: `architecture.png`, `routing.png` (true error and certified bound of each FNO answer with
the limit; latency through `/v1/infer`; cost against grid size), `checks.png`.
