# Verification Gap

**A design that passed its nominal safety check, and what its manufacturing tolerances do to that pass.**

One fixed worked case. A vented brake rotor (36 cast-iron vanes, 8 mm vent gap, 4 mm vanes, 0.754 m mean
circumference) under 1,480 W of sustained braking with low-speed airflow through the vents (h = 40 W/(m² K)) sits at
**495.9 °C** at nominal dimensions, under a 500 °C limit. Every page load draws 200 rotors within the machining
tolerances (vane thickness ±0.05 mm, vent gap ±0.1 mm, uniform), runs each through the same closed-form model and
shows the distribution, the 95th percentile against the limit, six checks of the study and a decision.

Ported from the `poc_verification_gap` case study of PINNeAPPle-apps.

## What it computes

| Piece | Library |
|---|---|
| Rotor temperature: the vanes as a fin array on the friction ring (Incropera & DeWitt fin efficiency + 1-D ring conduction) | `pinneapple_design.surrogate_inverse.brake_rotor` (shared with `apps/brakecool_ai`) on `fin_array_conduction` |
| Monte Carlo draws, p95 (nearest rank), pass fraction with its Wilson interval, ROBUST / MARGINAL (p95 within 5 % of the limit) / FRAGILE | `pinneapple_analysis.uncertainty.margin.margin_study` (draws with `pp.propagate_tolerances`) |
| Six checks: finite samples, p95 stable on an independent batch, p95 with 50 samples, hottest sample vs limit, corner stack-up, p95 vs limit; `TrustReport` decision | same |
| Rotor mesh for the 3-D view (two rings and the vanes, coloured by the radial vane profile) | `pinneapple_design.surrogate_inverse.brake_rotor.mesh` |

The app (`verification_gap/api.py`) holds the case constants and the routes; the page is `verification_gap/static/index.html`.

Measured here (seed 42): mean 495.91 °C, p95 **498.05 °C** (1.95 °C under the limit, MARGINAL), all 200 under the
limit, corner stack-up (thinnest vanes, smallest gap) 498.38 °C; checks 5 and 6 are WARNING, decision **REVIEW**.
At highway airflow (h = 120) the same rotor runs at 186.7 °C.

## Run it

```bash
# from the repository root
pip install -e . fastapi "uvicorn[standard]"
cd apps/verification_gap
PYTHONPATH=../_shared uvicorn verification_gap.api:app --port 8180      # open http://localhost:8180
```

Docker (from the repository root): `docker build -f apps/verification_gap/Dockerfile -t verification-gap .` then
`docker run -p 8180:8180 verification-gap`.

| Route | What |
|---|---|
| `GET /api/scenario?seed=` | one 200-sample study (fresh draws without `seed`) |
| `GET /api/mesh.json` | rotor triangle mesh (mm) |
| `GET /health` | liveness |

Environment: `VGAP_USER` / `VGAP_PASSWORD` (or `VGAP_PASSWORD_HASH`) for the login, `VGAP_MAX_HEAVY` concurrent
studies per worker (default 4); the shared settings in [`apps/_shared/README.md`](../_shared/README.md).

## Limitations

- Steady state under a constant power: no heating transient within a stop, no radiation, one film coefficient for
  the whole vent, no hub conduction. The rotor and the load case are a worked example from the source project, not
  a measured rotor or a certified limit.
- Uniform tolerances with no process-capability data; the 5 % MARGINAL band is a choice of the study.
- The checks verify the propagation, not the model: there is no comparison with a dynamometer or a 3-D solve.
- The source project showed an invented 0-100 "trust score" (weights 100 / 55 / 0); it is replaced by the
  library's `TrustReport` decision.
