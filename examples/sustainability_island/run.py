"""Sustainability Island: coupled hourly energy, water, and irrigation demonstrator.

Run: python examples/sustainability_island/run.py --weather nasa --year 2024
Offline fixture: python examples/sustainability_island/run.py --weather synthetic

NASA POWER weather is real gridded reanalysis/satellite data; design loads and
infrastructure parameters are explicitly hypothetical, not measured on site.
No claim of certified engineering design or trained PINN is made.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

LAT, LON = -5.80, -35.20  # regional reference point, NOT a surveyed island
NASA = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMETERS = "ALLSKY_SFC_SW_DWN,T2M,WS2M,PRECTOTCORR,RH2M"

@dataclass(frozen=True)
class Design:
    pv_kwp: float = 250.0
    inverter_kw: float = 220.0
    battery_kwh: float = 1000.0
    battery_kw: float = 250.0
    battery_min_soc: float = 0.15
    battery_max_soc: float = 0.95
    charge_eff: float = 0.96
    discharge_eff: float = 0.96
    pv_performance_ratio: float = 0.80
    load_baseline_kw: float = 18.0
    load_daytime_extra_kw: float = 17.0
    desal_kwh_per_m3: float = 4.0
    desal_capacity_m3_h: float = 2.0
    freshwater_tank_m3: float = 150.0
    freshwater_min_m3: float = 15.0
    rain_collection_m2: float = 3000.0
    rain_runoff_factor: float = 0.75
    residents: int = 60
    visitors: int = 40
    resident_l_day: float = 120.0
    visitor_l_day: float = 25.0
    irrigation_m2: float = 6000.0
    crop_coefficient: float = 0.85
    irrigation_efficiency: float = 0.85
    wastewater_fraction: float = 0.80
    treatment_capacity_m3_day: float = 15.0
    reuse_fraction: float = 0.65

def weather_nasa(year: int):
    query = urllib.parse.urlencode(dict(parameters=PARAMETERS, community="RE",
        longitude=LON, latitude=LAT, start=f"{year}0101", end=f"{year}1231",
        format="JSON", **{"time-standard": "LST"}))
    with urllib.request.urlopen(NASA + "?" + query, timeout=90) as response:
        payload = json.load(response)
    raw = payload["properties"]["parameter"]
    keys = sorted(set.intersection(*(set(raw[k]) for k in PARAMETERS.split(","))))
    records = []
    for key in keys:
        values = [raw[k].get(key) for k in PARAMETERS.split(",")]
        if any(v is None or v <= -900 for v in values):
            raise ValueError(f"Missing NASA POWER weather value at {key}")
        ghi, temp, wind, rain, rh = map(float, values)
        records.append((datetime.strptime(key, "%Y%m%d%H"), ghi, temp, wind, rain, rh))
    if len(records) not in (8760, 8784):
        raise ValueError(f"Expected complete hourly year, got {len(records)}")
    return records

def weather_synthetic(year: int):
    """Explicitly artificial fixture for offline smoke tests, not scientific output."""
    t = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    out = []
    while t < end:
        solar = max(0.0, math.sin(math.pi * (t.hour - 6) / 12))
        ghi = 0.83 * solar * (0.85 + 0.15 * math.sin(2 * math.pi * t.timetuple().tm_yday / 365))
        rain = 3.0 if t.hour == 17 and t.month in (3, 4, 5, 6) else 0.0
        out.append((t, ghi, 27 + 4 * solar, 4.0, rain, 72.0))
        t += timedelta(hours=1)
    return out

def simulate(weather, d: Design):
    """Hourly conservative dispatch. Unserved energy/water are explicit outputs.

    GHI input kWh/m2/h; rainfall mm/h. PV uses a simplified PR model, not PVWatts.
    FAO-56 Penman-Monteith is NOT claimed: irrigation is a screening proxy.
    """
    soc = 0.50 * d.battery_kwh
    water = 0.50 * d.freshwater_tank_m3
    daily_treatment = 0.0
    rows = []
    for time, ghi, temp, wind, rain, rh in weather:
        if time.hour == 0:
            daily_treatment = 0.0
        pv = min(d.inverter_kw, d.pv_kwp * max(0.0, ghi) * d.pv_performance_ratio)
        base = d.load_baseline_kw + (d.load_daytime_extra_kw if 8 <= time.hour < 19 else 0)
        potable = (d.residents * d.resident_l_day / 1000 +
                   (d.visitors * d.visitor_l_day / 1000 if 8 <= time.hour < 18 else 0)) / 24
        # Screening-only irrigation proxy; replace with FAO-56 ETo and measured crop Kc.
        eto_mm_h = max(0.0, 0.13 + 0.012 * (temp - 25) + 0.012 * wind - 0.0005 * rh)
        irrigation = d.irrigation_m2 * d.crop_coefficient * eto_mm_h / 1000 / d.irrigation_efficiency
        wastewater = potable * d.wastewater_fraction
        treated = min(wastewater, max(0.0, d.treatment_capacity_m3_day - daily_treatment))
        daily_treatment += treated
        reuse = min(irrigation, treated * d.reuse_fraction)
        rain_collected = max(0.0, rain) * d.rain_collection_m2 * d.rain_runoff_factor / 1000
        water_need = potable + irrigation - reuse
        water_before_desal = min(d.freshwater_tank_m3, water + rain_collected)
        desal = min(d.desal_capacity_m3_h, max(0.0, water_need + d.freshwater_min_m3 - water_before_desal))
        desal_energy = desal * d.desal_kwh_per_m3
        water_available = water_before_desal + desal
        water_served = min(water_need, water_available)
        water_shortfall = water_need - water_served
        water = water_available - water_served
        load = base + desal_energy
        net = pv - load
        charge = discharge = curtailed = unserved = 0.0
        if net >= 0:
            charge = min(net, d.battery_kw, max(0.0, (d.battery_max_soc * d.battery_kwh - soc) / d.charge_eff))
            soc += charge * d.charge_eff
            curtailed = net - charge
        else:
            discharge = min(-net, d.battery_kw, max(0.0, (soc - d.battery_min_soc * d.battery_kwh) * d.discharge_eff))
            soc -= discharge / d.discharge_eff
            unserved = -net - discharge
        energy_residual = pv + discharge + unserved - load - charge - curtailed
        if abs(energy_residual) > 1e-7:
            raise AssertionError("Energy conservation violated")
        if not (0 <= water <= d.freshwater_tank_m3 + 1e-8):
            raise AssertionError("Water storage bounds violated")
        rows.append(dict(timestamp=time.isoformat(), pv_kwh=round(pv, 5),
            load_kwh=round(load, 5), battery_soc_kwh=round(soc, 5),
            battery_charge_kwh=round(charge, 5), battery_discharge_kwh=round(discharge, 5),
            curtailed_kwh=round(curtailed, 5), unserved_kwh=round(unserved, 5),
            rain_captured_m3=round(rain_collected, 5), desal_m3=round(desal, 5),
            reuse_m3=round(reuse, 5), water_demand_m3=round(water_need, 5),
            water_shortfall_m3=round(water_shortfall, 5), tank_m3=round(water, 5),
            energy_residual_kwh=round(energy_residual, 9)))
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", choices=("nasa", "synthetic"), default="nasa")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--out", type=Path, default=Path("outputs/sustainability_island"))
    args = parser.parse_args()
    weather = weather_nasa(args.year) if args.weather == "nasa" else weather_synthetic(args.year)
    rows = simulate(weather, Design())
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "timeseries.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {k: sum(r[k] for r in rows) for k in (
        "pv_kwh", "load_kwh", "curtailed_kwh", "unserved_kwh",
        "rain_captured_m3", "desal_m3", "reuse_m3",
        "water_demand_m3", "water_shortfall_m3")}
    metrics["hours"] = len(rows)
    metrics["energy_supply_reliability"] = 1 - metrics["unserved_kwh"] / max(1, metrics["load_kwh"])
    metrics["water_supply_reliability"] = 1 - metrics["water_shortfall_m3"] / max(1, metrics["water_demand_m3"])
    report = dict(status="screening_model_not_certified", weather_source=args.weather,
        coordinates=dict(latitude=LAT, longitude=LON, reference_only=True),
        design=asdict(Design()), metrics=metrics,
        limitations=["Not a validated PVWatts/PVLib electrical model",
                     "Irrigation ETo proxy is not FAO-56 Penman-Monteith",
                     "No CFD/FEM or PINN training performed",
                     "No island site survey, grid connection or environmental permits",
                     "Desalination brine, wastewater storage and distribution hydraulics not modeled"])
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
