"""Portable JSON and CSV export for synthetic twin experiments."""
import csv
import json
from pathlib import Path

def export_transient(rows, destination):
    """Write a JSON file or CSV file; preserve explicit time and pressure units."""
    path = Path(destination)
    if path.suffix == ".json":
        path.write_text(json.dumps({"schema_version": "0.1",
            "source": "synthetic", "model": "lumped_pressure_relaxation",
            "records": rows}, indent=2, allow_nan=False), encoding="utf-8")
    elif path.suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "time_s", "bottomhole_pressure_pa", "equilibrium_pa"])
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("Export must be .json or .csv")
    return path
