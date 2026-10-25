# Design Optimization

**A wing section redesigned live: less profile drag at the same lift, every step re-checked on the physics.**

One fixed worked case. A NACA 2418-family section (2 % camber at 40 % chord, 18 % thick) at 4° and Re = 3·10⁶. An
MLP surrogate learns (camber, thickness) → (Cl, Cd) from the closed-form ground truth, and the design loop of
`pinneapple_design.design_optimizer` takes Adam steps on the design through it to minimise Cd + 0.3 (Cl − Cl₀)²
inside 0-6 % camber and 8-18 % thickness. Every iterate is re-evaluated with the closed form, so the numbers shown
are the physics', not the surrogate's. Each page load trains a fresh surrogate and runs the loop.

Ported from the `poc_design_optimization` case study of PINNeAPPle-apps.

## What it computes

| Piece | Library |
|---|---|
| Lift: thin-airfoil theory of the NACA 4-digit camber line (Anderson, ch. 4); drag: turbulent flat-plate friction × Hoerner form factor, two wetted sides | `pinneapple_physics.closed_form.thin_airfoil` |
| Surrogate (normalised targets + fixed un-normalisation layer), objective, box constraint, closed loop with its stagnation rule | `pinneapple_design.design_optimizer.airfoil_drag.run_design_optimization` |
| Section outlines | `pinneapple_design.geometry.gen.airfoil.naca_parametric` |

Measured here, 150 iterations, seeds 1-10: baseline Cl 0.666, Cd 0.01067 → final about 2.0 % camber and 8.0-8.1 %
thickness, **drag −18.21 % to −18.31 %**, lift within −0.61 % / +1.76 % of the baseline, stagnation after 104-106
steps; surrogate validation error 0.4-1.5 % on Cl and 0.66-0.75 % on Cd (relative L2, 400 independent designs).
Wall time per run was 31-246 s on this (heavily shared) 4-core container; the source project reported 1-2 s on a
laptop, not reproduced here.

## Run it

```bash
# from the repository root
pip install -e . fastapi "uvicorn[standard]"
cd apps/design_optimization
PYTHONPATH=../_shared uvicorn design_optimization.api:app --port 8181      # open http://localhost:8181
```

Docker (from the repository root): `docker build -f apps/design_optimization/Dockerfile -t design-optimization .`.

| Route | What |
|---|---|
| `GET /api/scenario?seed=&n_iterations=` | one run (fresh seed without `seed`; 1-300 iterations) |
| `GET /health` | liveness |

Environment: `DOPT_USER` / `DOPT_PASSWORD` (or `DOPT_PASSWORD_HASH`), `DOPT_MAX_HEAVY` concurrent runs per worker
(default 2, the rest get 429); shared settings in [`apps/_shared/README.md`](../_shared/README.md).

## Limitations

- The ground truth is a textbook 2-D model: no thickness effect on lift, no camber effect on drag, no stall, no
  transition, no 3-D wing (no induced drag). The optimum is therefore close to "thinnest section allowed, camber
  kept"; a panel method or RANS would trade these off. Not a validated aerodynamic design.
- The source project's drag formula omitted the factor 2 for the two wetted sides of a section; it is included here
  (absolute Cd doubles; the relative reduction is the same, 18.2-18.3 %, as the source project reported).
