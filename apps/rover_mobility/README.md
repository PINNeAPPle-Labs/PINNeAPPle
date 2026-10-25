# RoverMobility

**Rigid-wheel mobility screening: drawbar pull against slip from a Bekker-Wong PINN, with each physics
constraint re-checked on its own.**

Imported from PINNeAPPle-apps (`rover_mobility_analysis`). The analysis is
`pinneapple_simulation.terramechanics.mobility` (`mobility_analysis`, `analyze_wheel_surrogate`,
`mohr_coulomb_traction_limit`): train `bekker_wong_surrogate_2d` for a soil and a wheel, find the sinkage that
carries the wheel load, read Fx against slip, and check R1 (zero pull at zero slip), R2 (Mohr-Coulomb limit
c A + Fz tan(phi)), R3 (monotone up to slip 0.4) and R4 (torque covers thrust) one by one next to the guardrail.
The soil guide is the library's sourced soil table (`pinneapple_simulation.terramechanics.soils`); the source app's
own table, with sources that could not be checked, is left out. Related: the library's full Wong-Reece wheel model,
testbed validation and rover missions (`WongReeceModel`, `pinneapple_simulation.rover.autolab`); this app is the
quick surrogate screen.

## Run

```bash
pip install -e . -r apps/rover_mobility/requirements.txt
cd apps/rover_mobility && uvicorn rover_mobility.api:app --port 8147
curl -X POST localhost:8147/v1/analyze_mobility -H 'Content-Type: application/json' \
     -d '{"c_Pa": 1400, "phi_deg": 30, "R_m": 0.125, "b_m": 0.06, "Fz_N": 50, "epochs": 1500}'
curl localhost:8147/v1/jobs/<id>;  curl localhost:8147/v1/soil_parameter_guide
```

Logins `ROVERMOB_USER` / `ROVERMOB_PASSWORD_HASH`; `ROVERMOB_MAX_EPOCHS` (5000), `ROVERMOB_FREE_ANALYSES` (20), one job
at a time. Deployed as `rovermob` in `apps/deploy`.

## Tests

`pytest tests/test_terramechanics_mobility.py tests/test_app_rover_mobility.py` (7 + 1 pass; the app's training job
test with `PINNEAPPLE_SLOW=1`). The source `verify_analytical_crosscheck.py` is ported: the closed-form limit agrees
with the compiled R2 residual (0 at the limit, 25 = 10^2 / 4 when 10 N over it: one violation averaged with three
satisfied constraints). **The source TODO's import error of that script does not reproduce**: run as is against the
in-repo library it prints "ALL CHECKS PASSED". The per-constraint analysis is tested on hand-built surrogates with
known answers (consistent, and violating R2 and R4).

## Known limitations

- The guardrail's `pde_residual` pools R2-R5 into one mean (a single violation is diluted) and never evaluates R1
  (an `InitialCondition`). The source app saw a collapsed surrogate (Fx ~ 0 everywhere) reported with a pooled
  residual of 0.0 and an individually caught R3 violation masked by the pooled number (reported there, not
  reproduced here): read `max_drawbar_pull_N`, `constraint_checks` and `sinkage_solve_reliable`.
- The sinkage is solved on a grid of the surrogate's own Fz, so it inherits the surrogate's error (per-row
  `sinkage_solve_error_N`). Rigid wheel, 2-D, one wheel; no grousers, lateral forces or load transfer.
- No check against a worked textbook example (the source could not reach one either).
