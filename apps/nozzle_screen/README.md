# NozzleScreen

**Is this rocket-nozzle idea in the right ballpark? An ideal-nozzle estimate in microseconds, and a PINN
cross-check that says when the two disagree.**

Imported from PINNeAPPle-apps (`rocket_nozzle_screening`). The ideal 1-D isentropic nozzle (exit Mach from the area
ratio, exit pressure and temperature, c*, thrust coefficient, Isp; Sutton & Biblarz ch. 3) is
`pinneapple_physics.closed_form.rocket_nozzle`; the two-method screen is
`pinneapple_analysis.verification.nozzle_crosscheck.screen_nozzle` (a PINN on the `rocket_nozzle_cfd` preset,
re-checked by the guardrail, read at the exit plane). Related: `pp.closed_form.compressible` (isentropic ratios,
shocks) and `apps/aero_optimizer` (airfoils); nothing else does nozzles.

## Run

```bash
pip install -e . -r apps/nozzle_screen/requirements.txt
cd apps/nozzle_screen && uvicorn nozzle_screen.api:app --port 8146
curl "localhost:8146/v1/analytical_only?throat_radius_m=0.05&exit_radius_m=0.12&p_chamber_pa=4e6&T_chamber_k=2800&gamma=1.2&R_gas_j_per_kgk=345"
curl -X POST localhost:8146/v1/screen_nozzle -H 'Content-Type: application/json' -d '{"throat_radius_m": 0.05,
  "exit_radius_m": 0.12, "p_chamber_pa": 4e6, "T_chamber_k": 2800, "gamma": 1.2, "R_gas_j_per_kgk": 345, "epochs": 300}'
curl localhost:8146/v1/jobs/<id>
```

Logins `NOZZLE_USER` / `NOZZLE_PASSWORD_HASH`; `NOZZLE_MAX_EPOCHS` (3000), `NOZZLE_FREE_ESTIMATES` (10000),
`NOZZLE_FREE_SCREENS` (20), one job at a time. Deployed as `nozzle` in `apps/deploy`.

## Reproduced here

- The source app's `verify_analytical.py` passes against the in-repo port's numbers: A/A* = 1.6875 at M = 2 and
  1.1762 at M = 1.5 (gamma 1.4, Anderson Table A.1), the inverse on both branches, p/p0 = 0.1278 at M = 2, c* =
  p0 At / mdot, mdot linear in p0; plus C_F c* = v_e at optimum expansion (`tests/test_closed_form_rocket_nozzle.py`).
  On the README motor (rt 5 cm, re 12 cm, 4 MPa, 2800 K, gamma 1.2, R 345) the port gives M_e = 2.888, Isp = 235.0 s,
  identical to the source code.
- **The PINN half disagrees**, as the source app found (73-4764 %): here, 30 epochs on the README motor gave exit
  Mach 78 % and exit pressure 4529 % off the ideal nozzle, PDE residual 9.3e23 against the 1e-2 threshold, verdict
  "do not trust". The preset's residual is in raw SI units (pressure ~1e6 Pa next to velocity ~1e3 m/s) and it
  needs a non-dimensional reformulation; the PINN half is shown as a cross-check catching a bad model, not as a
  second estimate.

## Known limitations

- Ideal nozzle: constant gamma, isentropic, 1-D, no shocks or separation. A screening estimate before CFD.
- PINN geometry is the library's tagged geometry for `rocket_nozzle_cfd`: the divergent cone from the throat
  (inlet) to the exit. The source app invented a converging section (chamber 1.6 x the throat radius, throat at
  30 % of the length); that is not reproduced. The preset also puts a no-slip wall on inviscid equations and
  samples collocation points beyond the cone.
- `nozzle_length` defaults to 8 throat radii, a placeholder. The source's optional LLM "engineering note" is left
  out (prose with no physics value).
- Tests: `pytest tests/test_app_nozzle_screen.py tests/pinneapple_analysis/test_verification_nozzle_crosscheck.py`
  (1 + 3 pass; the app's PINN job test needs `PINNEAPPLE_SLOW=1`, about 1.5 min).
