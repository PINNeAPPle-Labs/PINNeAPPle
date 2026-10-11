"""Phase 1 -- Data: fetch hourly historical weather for Natal (RN, Brazil) from the
Open-Meteo Historical Weather API and store it as weather_dataset.csv.

Open-Meteo's archive endpoint serves ERA5 / ERA5-Land reanalysis (not station
measurements); the paper states this explicitly. The request is split per year to keep
responses small; the JSON is flattened into one CSV with explicit units in the header.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

LAT, LON = -5.7945, -35.2110  # Natal, RN
URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = [
    "temperature_2m",            # degC   ambient air temperature (T_amb)
    "relative_humidity_2m",      # %
    "surface_pressure",          # hPa
    "wind_speed_10m",            # km/h
    "wind_direction_10m",        # deg
    "precipitation",             # mm (hourly sum)
    "shortwave_radiation",       # W/m2 (mean of preceding hour)
    "soil_temperature_0_to_7cm", # degC   state variable T of the thermal-mass element
]
OUT = Path(__file__).resolve().parents[1] / "data"


def fetch_year(year: int) -> pd.DataFrame:
    q = {
        "latitude": LAT, "longitude": LON,
        "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
        "hourly": ",".join(HOURLY), "timezone": "America/Fortaleza",
    }
    url = URL + "?" + urllib.parse.urlencode(q)
    for k in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                js = json.load(r)
            break
        except Exception as e:
            print("retry", k, e)
            time.sleep(10 * (k + 1))
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    meta = {k: js[k] for k in ("latitude", "longitude", "elevation", "timezone")}
    return df, meta, js["hourly_units"]


def main(y0: int = 2015, y1: int = 2024) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for y in range(y0, y1 + 1):
        df, meta, units = fetch_year(y)
        frames.append(df)
        print(y, len(df), "NaN:", int(df.isna().sum().sum()))
        time.sleep(1.0)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "weather_dataset.csv", index=False)
    (OUT / "weather_dataset_meta.json").write_text(json.dumps({"grid_point": meta, "units": units,
                                                               "source": URL}, indent=2))
    print("saved", df.shape, meta)


if __name__ == "__main__":
    main()
