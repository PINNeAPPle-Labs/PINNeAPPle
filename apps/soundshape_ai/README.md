# SoundShape AI

Bass-reflex port radius and length for a target tuning frequency, found through a frozen neural surrogate and
re-checked against the Helmholtz formula, plus a sweep of the trade-off between a small port and an accurate tuning.
Imported from PINNeAPPle-apps (`soundshape_ai`); the logic lives in `pinneapple_design.surrogate_inverse.reflex_port`.

## What it does

- Ground truth: `pinneapple_physics.closed_form.helmholtz_resonator`, `f = c/(2 pi) sqrt(A / (V L_eff))`,
  `L_eff = L + 1.7 r` (port flanged at both ends). Design rule: the port takes at most 6 % of the cabinet volume.
- Surrogate on log(f) of (radius, length, cabinet volume) - the target is never an input; Adam in a sigmoid-bounded
  space on the relative frequency error plus a port-volume penalty; verification card (surrogate error, target error,
  bounds).
- Trade-off: `lambda * V/V_max + (1 - lambda) * rel_err^2` for 8 weights, several starts per weight, every design
  found competing for every weight by the **closed-form** loss; reported frequencies and volumes are from the formula.
  The colour on the part is one value per design (the lumped model has no field along the port).

## Measured here

| Check | Result |
|---|---|
| Surrogate held-out error, 4 seeds | mean 0.4-0.5 % |
| 20, 35 / 20, 55 / 50, 30 / 10, 70 (L, Hz) x 4 seeds | tuning within 0.0-1.5 % of target, PASS |
| 100 L at 90 Hz | FAIL (42 % off): not reachable inside the port bounds, reported as such |
| Trade-off sweep (20 L, 40 Hz) | ~10 s on one core |

Changed from the source: the surrogate (log output, 1500 epochs), the optimizer (sigmoid-bounded descent; the source
used Adam at lr 0.003 m on raw values and its constraint penalties were computed from detached values, so they had no
gradient - here the port-volume penalty is differentiable), the sweep pools the designs of all weights; triangle mesh
instead of CadQuery; three.js from `/vendor`.

## Limitations

Lumped Helmholtz resonator: no cabinet modes, port compression or air-speed limits, no driver model or measurement.
The trade-off curve comes from gradient descent through a surrogate, not an exact Pareto front; with this weighting
most weights collapse onto the shortest port (the frequency term dominates).

## Run

```bash
cd apps/soundshape_ai && PYTHONPATH=../..:../_shared uvicorn soundshape_ai.api:app --port 8164
```

Environment: `SOUND_USER` / `SOUND_PASSWORD` (or `_PASSWORD_HASH`), `SOUND_EPOCHS` (1500), `SOUND_THREADS` (1).
Routes: `GET /health`, `/api/status`; `POST /api/evaluate`, `/api/mesh.stl` (`radius_m`, `length_m`,
`cabinet_volume_l`), `/api/optimize` and `/api/pareto_frontier` (`cabinet_volume_l`, `target_freq_hz`, `n_points`).
