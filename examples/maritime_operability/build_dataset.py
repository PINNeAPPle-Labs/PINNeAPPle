"""Fetch the buoy record the example runs on, and write it as one compressed file.

This is the only script here that needs the network. It reads NOAA NDBC's three public files for one station -- the
standard meteorological record, the measured spectral density and NDBC's own swell/wind-sea split -- and writes them as
``data/ndbc_<station>.npz``, which is what the example loads. The file carries the station, the window and the URLs, so
what was fetched can be re-fetched and compared.

    python examples/maritime_operability/build_dataset.py            # station 41010
    python examples/maritime_operability/build_dataset.py --station 46002
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pinneapple_physics.maritime import ndbc

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def build(station: str = "41010") -> Path:
    record = ndbc.fetch(station)
    times = record.time
    print(f"{station}: {times.size} rows of ten-minute data, {times[0]} to {times[-1]}")
    print(f"{station}: {record.spec_time.size} measured spectra over {record.spec_freq.size} bands")
    # float32 for the spectral density: it is stored to four significant figures on the buoy's own grid
    squeezed = ndbc.BuoyRecord(
        station=record.station, time=record.time, hs=record.hs.astype(np.float32), tp=record.tp.astype(np.float32),
        apd=record.apd.astype(np.float32), wspd=record.wspd.astype(np.float32), mwd=record.mwd.astype(np.float32),
        spec_time=record.spec_time, spec_freq=record.spec_freq.astype(np.float32),
        spec_density=record.spec_density.astype(np.float32), swell_hs=record.swell_hs.astype(np.float32),
        swell_tp=record.swell_tp.astype(np.float32), windsea_hs=record.windsea_hs.astype(np.float32),
        windsea_tp=record.windsea_tp.astype(np.float32), split_time=record.split_time, source=record.source)
    dest = DATA / f"ndbc_{station}.npz"
    ndbc.to_npz(squeezed, dest)
    print(f"wrote {dest} ({dest.stat().st_size / 1024:.0f} kB)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="41010", help=f"one of {sorted(ndbc.STATIONS)}")
    build(parser.parse_args().station)


if __name__ == "__main__":
    main()
