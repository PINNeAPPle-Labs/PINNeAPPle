# OrbitWatch

**What changed at this place between two satellite images? Per-band reflectance change and NDVI change, as a
self-contained HTML report.**

Imported from PINNeAPPle-apps (`satellite_change_monitoring`), which extracted it from an older notebook of the
author. The maths and the report are `pinneapple_perception.change_detection` (`band_difference`, `compute_ndvi`,
`analyze_pair`, `render_comparison_figure`, `render_html_report`); the Sentinel-2 fetch from Google Earth Engine is
`pinneapple_perception.sentinel2`. No other app does remote sensing.

## Run

```bash
pip install -e . -r apps/orbitwatch/requirements.txt
cd apps/orbitwatch && uvicorn orbitwatch.api:app --port 8150
curl -F images=@2024-01.png -F images=@2024-06.png -F dates=2024-01-01 -F dates=2024-06-01 \
     localhost:8150/v1/detect_changes_upload -o report.html
```

`POST /v1/detect_changes {"lon": ..., "lat": ..., "buffer_m": 5000}` fetches Sentinel-2 surface reflectance (B4, B3,
B2 and the real NIR band B8) and needs **Earth Engine credentials in the environment only**: `EE_PROJECT_ID` (a
Google Cloud project registered for Earth Engine, with the API enabled) and either `EE_SERVICE_ACCOUNT_EMAIL` +
`EE_SERVICE_ACCOUNT_KEY_FILE` (path to the JSON key, mounted read-only from outside the repository and the image) or a
one-time `earthengine authenticate` on the host. Without them it answers 503 with these steps and never invents
imagery. `earthengine-api` and `geemap` are optional requirements. Logins `ORBITWATCH_USER` /
`ORBITWATCH_PASSWORD_HASH`; `ORBITWATCH_FREE_REPORTS` (100). Deployed as `orbitwatch` in `apps/deploy`.

## Tests

`pytest tests/test_change_detection.py tests/test_app_orbitwatch.py` (13 + 5 pass, earthengine-api installed, no
credentials): the source detection and app tests on synthetic arrays and PNGs, ordering by date, refusals, metering,
and the credential refusal. The live fetch was not run (no Earth Engine account here).

## Fixes kept from the source, and one added

- The notebook's NDVI used `nir = red + 400`; only a real NIR band is used here, and uploads (RGB only) skip NDVI.
- Sentinel-2 SR stores DN = 10000 x reflectance: the threshold uses 10000; 3000 is only the preview stretch.
- Added: caller-supplied dates were written into the HTML report unescaped (script injection); they are escaped and
  the template autoescapes.

## Known limitations

- Mean absolute change per band over the whole image against a fixed threshold: no cloud mask, co-registration,
  radiometric normalisation or per-pixel change map. Uploaded 8-bit images use a 255 scale.
- The Earth Engine collection is `COPERNICUS/S2_SR` as in the source; Earth Engine has since deprecated it in favour
  of `COPERNICUS/S2_SR_HARMONIZED` (not changed or checked here).
