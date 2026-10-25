# Predictive Twin

**A digital twin of a real river gauge: calibrated on its own recent past, forecasting the present, flagging when the
river stops following the forecast.**

One fixed gauge by default: USGS 01646500, Potomac River near Washington, DC (Little Falls). Every page load
downloads the last 5 days of discharge (and gage height) from the public USGS Water Services Instantaneous Values API,
fits a baseflow recession Q = Q₀ exp(−k t) on the 72 h before the last 24 h, forecasts the last 24 h and flags a
departure of more than 3σ of the calibration residuals held for at least 60 minutes. The verdict is
`DIVERGENCE_DETECTED`, `MATCHES_FORECAST` or `NO_CLEAN_RECESSION` (the calibration window was not receding, so the
model does not apply). A 3-D view (three.js) shows the current and the forecast water level in a representative
channel.

Ported from the `poc_predictive_twin` case study of PINNeAPPle-apps (its Blender glTF scene is replaced by a
three.js scene built in the page; its USGS client is rewritten on the standard library).

## Data

USGS data are U.S. Government works in the public domain; provisional data may be revised by USGS. They are
downloaded at run time (cached 90 s in memory) and never stored in this repository. If USGS is unreachable or returns
no data, `/api/scenario` answers 502 with the reason; nothing is filled in. `PTWIN_OFFLINE=1` serves a synthetic
record instead, labelled `SYNTHETIC` in the page and the response (`"synthetic": true`).

## What it computes

| Piece | Library |
|---|---|
| USGS IV client (urllib, JSON parse, errors, cache) | `pinneapple_systems.digital_twin.io.usgs` |
| Recession fit (log-linear least squares, R², residual statistics), forecast, sustained-departure detector, verdict, gage-height fit | `pinneapple_systems.digital_twin.recession` (`run_twin`) |

Not measured here: the live API was not reachable from this container (egress proxy refused waterservices.usgs.gov),
so the app was run offline only. The source project's runs (R² 0.88-0.92, +5.9σ / +7σ excursions on 2026-09-15/16)
are reported there, not reproduced here; its TODO also records weak calibration (R² ≈ 0.01-0.02) in a later steady
regime. The synthetic offline record checks the pipeline: an 6 % step in the comparison window is flagged.

## Run it

```bash
# from the repository root
pip install -e . fastapi "uvicorn[standard]"
cd apps/predictive_twin
PYTHONPATH=../_shared uvicorn predictive_twin.api:app --port 8164      # open http://localhost:8164
```

Routes: `GET /api/scenario`, `GET /health`. Environment: `PTWIN_USER` / `PTWIN_PASSWORD` (or
`PTWIN_PASSWORD_HASH`), `PTWIN_SITE_ID` (another USGS gauge), `PTWIN_OFFLINE=1`.

## Limitations

- One exponential reservoir: no rainfall input, no routing, no rating-curve change. A low calibration R² means the
  gauge is not in recession and the forecast should not be trusted; the page shows R².
- The channel in the 3-D view is a display shape; only the two water heights are data.
