"""Phase 2 -- Virtual sensors: replay the historical dataset as a telemetry stream.

Each variable is exposed as a PINNeAPPle ``Sensor`` (pinneapple_systems.digital_twin.io);
``VirtualSensorStream`` emits calibrated ``Observation`` objects (timestamp + values +
metadata) in chronological order, optionally with measurement noise, and persists every
reading to SQLite (the storage layer of the twin).
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from pinneapple_systems.digital_twin import Observation, Sensor, SensorRegistry

CHANNELS = {
    # sensor_id: (dataframe column, field name, unit, noise std used by the filter)
    "T_facility": ("soil_temperature_0_to_7cm", "temperature", "degC", 0.1),
    "T_ambient": ("temperature_2m", "ambient_temperature", "degC", 0.1),
    "RH": ("relative_humidity_2m", "humidity", "%", 1.0),
    "P": ("surface_pressure", "pressure", "hPa", 0.1),
    "WS": ("wind_speed_10m", "wind_speed", "km/h", 0.5),
    "WD": ("wind_direction_10m", "wind_direction", "deg", 5.0),
    "RAIN": ("precipitation", "precipitation", "mm", 0.0),
    "SW": ("shortwave_radiation", "solar_radiation", "W/m2", 5.0),
}


def build_registry() -> SensorRegistry:
    reg = SensorRegistry()
    for sid, (_, field, unit, sd) in CHANNELS.items():
        reg.add(Sensor(sensor_id=sid, coords={"lat": -5.7945, "lon": -35.211}, field_names=[field],
                       noise_std={field: sd}, metadata={"unit": unit, "source": "Open-Meteo archive (ERA5)"}))
    return reg


class VirtualSensorStream:
    """Chronological replay of the dataset as sensor observations."""

    def __init__(self, df: pd.DataFrame, registry: Optional[SensorRegistry] = None,
                 db_path: Optional[Path] = None, add_noise: bool = False, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.reg = registry or build_registry()
        self.rng = np.random.default_rng(seed)
        self.add_noise = add_noise
        self.db = None
        if db_path is not None:
            self.db = sqlite3.connect(str(db_path))
            self.db.execute("CREATE TABLE IF NOT EXISTS readings (ts TEXT, sensor_id TEXT, field TEXT, value REAL, unit TEXT)")

    def observations_at(self, i: int) -> List[Observation]:
        row = self.df.iloc[i]
        ts = pd.Timestamp(row["time"])
        out = []
        for sid, (col, field, unit, sd) in CHANNELS.items():
            v = float(row[col])
            if self.add_noise and sd > 0:
                v += float(self.rng.normal(0, sd))
            obs = self.reg.read(sid, {field: v})
            obs.timestamp = ts.timestamp()
            obs.metadata.update({"unit": unit, "iso_time": ts.isoformat()})
            out.append(obs)
        if self.db is not None:
            self.db.executemany("INSERT INTO readings VALUES (?,?,?,?,?)",
                                [(o.metadata["iso_time"], o.sensor_id, *next(iter(o.values.items())), o.metadata["unit"])
                                 for o in out])
            self.db.commit()
        return out

    def stream(self, start: int = 0, stop: Optional[int] = None, realtime_s: float = 0.0) -> Iterator[List[Observation]]:
        for i in range(start, stop if stop is not None else len(self.df)):
            yield self.observations_at(i)
            if realtime_s > 0:
                time.sleep(realtime_s)
