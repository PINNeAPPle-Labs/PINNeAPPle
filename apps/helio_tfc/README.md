# HelioTFC

**Where is the solar cycle's envelope heading? EMD of the daily sunspot number, the Hilbert amplitude of the
~11-year component, and a fit whose end values are tied exactly.**

Imported from PINNeAPPle-apps (`solar_cycle_forecasting`), which reconstructed it from two older research
repositories of the author. The pipeline is `pinneapple_systems.time_series.solar_cycle`: min-max scaling, (E)EMD
(`PyEMD` from `EMD-signal` when installed, else the library's own EMD), the IMF whose measured period is nearest
11 years (with a warning when even that one is more than 40 % off), its Hilbert amplitude, and an extreme learning
machine fit f(x) = g(x) + x (g(0) - g(1)), which satisfies f(0) = f(1) for every weight vector (a constrained
expression of the Theory of Functional Connections), extended past the data. Related: `pinneapple_systems.time_series`
(`forecast_lab`, `predictability_report`) for forecasting with honest protocols; this app is one method, not a
validated forecast.

## Data and licence

`data_source="live_silso"` downloads SILSO's daily total sunspot number at request time
(https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv; SILSO, Royal Observatory of Belgium, Brussels; **CC BY-NC 4.0**:
credit SILSO, non-commercial use only). No SILSO data is in this repository: the source app's 2005-2025 fixture is
left out. `"synthetic"` is a made-up series of the same shape for trying the API, and the fast tests use it. Set
`HELIO_ALLOW_LIVE=0` on a commercial deployment.

## Run

```bash
pip install -e . -r apps/helio_tfc/requirements.txt
cd apps/helio_tfc && uvicorn helio_tfc.api:app --port 8149
curl "localhost:8149/v1/imf_summary?data_source=live_silso&start_date=2005-01-01&end_date=2025-06-30"
curl -X POST localhost:8149/v1/decompose_and_forecast -H 'Content-Type: application/json' -d '{"data_source": "live_silso"}'
```

Also `POST /v1/decompose_and_forecast_upload` (a SILSO-format CSV). Logins `HELIO_USER` / `HELIO_PASSWORD_HASH`;
`HELIO_FREE_FORECASTS` (100). Deployed as `helio` in `apps/deploy`.

## Tests

`pytest tests/test_solar_cycle.py tests/test_app_helio_tfc.py` (13 pass, 2 skip + 3 pass, 1 skip): the source tests on
the synthetic series; the live download with `PINNEAPPLE_NETWORK=1` (sidc.be was not reachable from this
environment); the `tfc` basis when `tfc` is installed (it pulls jax; not installed here, so the default basis is a
NumPy ELM and `backend="tfc"` is optional).

A validation run on the source app's real 2005-2025 SILSO slice (read locally, not copied into the repository)
reproduced its findings: IMF 0 has a **3.6-day** period (the original script's "IMF 0 is the 11-year cycle" is
wrong), PyEMD's IMF 6 is picked at **9.81 years**, end-point error 0.0, envelope RMSE 0.027.

## Known limitations

- The nearest-11-year pick depends on the series length and `max_imf`: on the synthetic series it is often far from
  11 years (and warns); on the full 1818-2026 series the source found no IMF near 11 years at `max_imf=9`. The
  period of the last IMF (the residue, which has a non-zero mean) is overestimated by the Hilbert median.
- Only f(0) = f(1) is imposed (no derivative continuity). Small EEMD ensembles reconstruct the signal only on
  average (mean error ~0.04 with 6 trials in the source).
- The forecast is an extension of a fitted envelope, never compared here with NOAA's Cycle 25 prediction.
