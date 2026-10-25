# SatScreen

**Quick conjunction screening: closest approach and a collision-probability estimate in Clohessy-Wiltshire
relative motion, plus two-body + J2 propagation.**

Imported from PINNeAPPle-apps (`satellite_conjunction_screening`). The physics is
`pinneapple_physics.astro.conjunction` (`cw_closest_approach`, `collision_probability_2d`, `screen_cw_conjunction`,
`propagate_two_body_j2`) and `presets.astrophysics.cw_exact_velocity`; the presets `space_debris_cw_relative_motion`
and `satellite_j2_perturbation` build and check the inputs. No PINN is trained per request: CW has an exact closed
form and J2 is integrated with DOP853. Related: `apps/observatory` tracks real satellites (TLE / SGP4); SatScreen is
relative-motion screening from a given relative state.

## Run

```bash
pip install -e . -r apps/satscreen/requirements.txt
cd apps/satscreen && uvicorn satscreen.api:app --port 8144
curl -X POST localhost:8144/v1/screen_conjunction -H 'Content-Type: application/json' -d '{"mode": "elements",
  "chief": {"a_km": 6771}, "deputy_relative_state": {"x0_km": 0.05, "y0_km": 1.0, "z0_km": 0.1,
  "vx0_km_s": 0.0002, "vy0_km_s": -0.0019, "vz0_km_s": 0.00005},
  "combined_hard_body_radius_km": 0.005, "combined_position_sigma_km": 0.5}'
```

`mode="description"` lets a local LLM draft the inputs (`pinneapple_llm.draft_preset_kwargs`: the preset must be
the expected one and every value inside a physical range, else it retries and finally answers 502);
`SATSCREEN_LLM_PROVIDER` / `SATSCREEN_LLM_MODEL` (default Ollama `llama3.2:3b`). Logins `SATSCREEN_USER` /
`SATSCREEN_PASSWORD_HASH`; `SATSCREEN_FREE_SCREENINGS` (500 per month). Deployed as `satscreen` in `apps/deploy`
(elements mode; the description mode needs Ollama inside the container).

## Reproduced here

- The source README's worked example: miss distance **0.3729 km at t = 423.5 s**, Pc **3.79e-5** (r 5 m, sigma
  0.5 km) -- `tests/test_astro_conjunction.py` and `tests/test_app_satscreen.py`.
- The quadrature against the closed form 1 - exp(-r^2 / 2 sigma^2) (relative error < 1e-7) and against 2 million
  Monte Carlo samples for an offset anisotropic case (within 4 standard errors); the closest approach is a true
  minimum (residual |rho . v| / (|rho| |v|) < 1e-5, not above a 200 001-point scan); a J2 orbit conserves the energy
  of the J2 potential to 1e-9 over three orbits; the radius of a 7000 km near-circular orbit stays in 6980-7020 km.
- **The CW velocity sign error**: the along-track velocity in `examples/pde_environment/06_space_debris_cw_validation.py`
  had `+2 s vx0` where the derivative of the exact position has `-2 s vx0`. Checked independently by a central
  finite difference of `_cw_exact_state` (`tests/test_astro_conjunction.py`: the library formula agrees to 1e-8,
  the old one is off by more than 1e-3 km/s). The example now imports `cw_exact_velocity`; its own numbers do not
  change, because it uses vx0 = 0.

## Known limitations

- CW is linear motion about a circular reference orbit over short windows: no drag, J2, third bodies or
  manoeuvres; no warning for eccentric chiefs or multi-day windows. The deputy is a relative state in the chief's
  Hill frame (no conversion from two independent ECI states).
- **The Pc is fed a placeholder sigma** (isotropic, caller-supplied), not a propagated covariance: it shows the
  method, it is not an operational Pc. When sigma exceeds the miss distance the response warns (the regime where
  the 2-D Pc is hard to interpret).
- J2 propagation is two-body + J2 only. Its literature secular rates are checked over hundreds of orbits in
  `tests/test_j2_secular_validation.py`, not in each request.
- With `llama3.2:3b` the source app saw the description mode return wrong units (`mu` in m^3/s^2), `e = 1` and the
  wrong preset; it fails closed. Use `mode="elements"`.
