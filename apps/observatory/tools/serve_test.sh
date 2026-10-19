#!/bin/sh
# Serve the Observatory for the browser tests: offline (a local ISS TLE, no warm-up), on the given port.
set -e
cd "$(dirname "$0")/.."
TLE="${OBS_TLE_FILE:-/tmp/obs-e2e-iss.tle}"
python3 -c "import json,sys; d=json.load(open('../../pinneapple_physics/astro/data/reference/iss_passes_reference.json')); open(sys.argv[1],'w').write('\n'.join(d['tle'])+'\n')" "$TLE"
OBS_TLE_FILE="$TLE" OBS_PRECOMPUTE=0 OBS_SURVEY="${OBS_SURVEY:-0}" exec python3 -m uvicorn observatory.api:app --port "${1:-8797}" --log-level warning
