"""Engineering data standardizer: many files, many conventions -> one physical schema (PINNeAPPle UPD).

Each source (CSV / Excel / Parquet / JSON / HDF5, read with :func:`pinneapple_data.data_health.load_table`)
gets a *mapping*: its time column and time zone, and for every column the physical quantity, the
source unit, a canonical target name and the target unit. :func:`propose_mapping` drafts it from
the headers; a person reviews it; :func:`standardize` applies it:

* timestamps parsed (ISO, dd/mm or mm/dd, Unix s/ms, Excel serial dates), localised and converted to UTC;
* values converted with exact unit definitions (:mod:`pinneapple_data.physical_units`), including
  temperature offsets and gauge pressure flags;
* sources aligned on a common time grid (optional resampling), in a wide table
  (``variable@source`` when several sources measure the same variable) or a long table
  (time_utc, source, variable, value, unit);
* a manifest (``pinneapple.upd/1``) that lists every variable with its quantity, unit and the exact
  conversion applied to each source column, plus file hashes. The mapping itself is a *recipe*:
  stored as JSON, it re-applies the same standardization to next month's files through the API.

When two sources measure the same variable, the result reports how well they agree after
conversion. That is a direct check that units and time zones were understood.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pinneapple_core.series import downsample_indices

from .data_health import _parse_time, _to_numeric, detect_time_column
from .physical_units import (QUANTITIES, UnitError, dim_str, infer_quantity, parse_unit, split_header,
                             try_parse_unit)

SCHEMA = "pinneapple.upd/1"

UNIT_SYSTEMS: Dict[str, Dict[str, Any]] = {
    "metric": {"label": "Metric engineering (°C, bar, m³/h, kW)", "units": {
        "temperature": "°C", "pressure": "bar", "stress": "MPa", "volumetric_flow_rate": "m^3/h",
        "mass_flow_rate": "kg/s", "velocity": "m/s", "power": "kW", "energy": "kWh", "torque": "N*m",
        "force": "kN", "voltage": "V", "current": "A", "frequency": "Hz", "length": "mm", "mass": "kg",
        "time": "s", "density": "kg/m^3", "dynamic_viscosity": "cP", "kinematic_viscosity": "cSt",
        "heat_flux": "W/m^2", "thermal_conductivity": "W/(m*K)", "acceleration": "m/s^2",
        "relative_humidity": "%", "fraction": "%", "angle": "deg", "strain": "µε", "electrical_resistance": "ohm"}},
    "si": {"label": "SI base (K, Pa, m³/s, W)", "units": {q.name: q.si_unit for q in QUANTITIES.values()}},
    "us": {"label": "US customary (°F, psi, gpm, BTU/h)", "units": {
        "temperature": "°F", "pressure": "psi", "stress": "ksi", "volumetric_flow_rate": "gpm",
        "mass_flow_rate": "lb/h", "velocity": "ft/s", "power": "BTU/h", "energy": "BTU", "torque": "N*m",
        "force": "lbf", "voltage": "V", "current": "A", "frequency": "Hz", "length": "in", "mass": "lb",
        "time": "s", "density": "kg/m^3", "dynamic_viscosity": "cP", "kinematic_viscosity": "cSt",
        "heat_flux": "W/m^2", "thermal_conductivity": "W/(m*K)", "acceleration": "m/s^2",
        "relative_humidity": "%", "fraction": "%", "angle": "deg", "strain": "µε", "electrical_resistance": "ohm"}},
}

# Unit choices offered per quantity (first = SI)
UNIT_CHOICES: Dict[str, List[str]] = {
    "temperature": ["K", "°C", "°F", "°R"], "pressure": ["Pa", "kPa", "bar", "mbar", "MPa", "psi", "atm", "inH2O"],
    "stress": ["Pa", "MPa", "GPa", "psi", "ksi"], "volumetric_flow_rate": ["m^3/s", "m^3/h", "L/s", "L/min", "gpm", "cfm"],
    "mass_flow_rate": ["kg/s", "kg/h", "t/h", "lb/h"], "velocity": ["m/s", "km/h", "ft/s", "mm/s"],
    "power": ["W", "kW", "MW", "BTU/h", "TR", "hp"], "energy": ["J", "kJ", "MJ", "kWh", "MWh", "BTU"],
    "torque": ["N*m", "kN*m"], "force": ["N", "kN", "lbf"], "voltage": ["V", "kV", "mV"], "current": ["A", "mA", "kA"],
    "frequency": ["Hz", "rpm", "kHz"], "length": ["m", "mm", "cm", "km", "in", "ft"], "mass": ["kg", "g", "t", "lb"],
    "time": ["s", "ms", "min", "h"], "density": ["kg/m^3", "g/cm^3"], "dynamic_viscosity": ["Pa*s", "cP"],
    "kinematic_viscosity": ["m^2/s", "cSt"], "heat_flux": ["W/m^2", "kW/m^2"], "thermal_conductivity": ["W/(m*K)"],
    "acceleration": ["m/s^2"], "relative_humidity": ["1", "%"], "fraction": ["1", "%"], "angle": ["rad", "deg"],
    "strain": ["1", "µε"], "electrical_resistance": ["ohm"],
}

QUANTITY_SUFFIX = {
    "temperature": "temperature", "pressure": "pressure", "stress": "stress", "volumetric_flow_rate": "flow",
    "mass_flow_rate": "mass_flow", "velocity": "velocity", "power": "power", "energy": "energy", "torque": "torque",
    "force": "force", "voltage": "voltage", "current": "current", "frequency": "frequency", "length": "length",
    "mass": "mass", "density": "density", "dynamic_viscosity": "viscosity", "kinematic_viscosity": "kinematic_viscosity",
    "heat_flux": "heat_flux", "thermal_conductivity": "conductivity", "acceleration": "acceleration",
    "relative_humidity": "humidity", "fraction": "", "angle": "angle", "strain": "strain",
    "electrical_resistance": "resistance", "time": "time",
}
# words that only say "which quantity" (the suffix says it again)
_GENERIC = {"temp", "temperature", "tmp", "t", "flow", "rate", "power", "pressure", "press", "pres", "p", "q",
            "value", "val", "pv", "meas", "measured", "signal", "sensor", "reading", "avg", "mean", "kw", "w",
            "voltage", "volt", "current", "amp", "amps", "freq", "frequency", "humidity", "hum", "velocity", "vel",
            "energy", "mass", "level", "elec"}
_ALIAS = {"chws": "chw_supply", "chwr": "chw_return", "chwst": "chw_supply", "chwrt": "chw_return",
          "hws": "hw_supply", "hwr": "hw_return", "cws": "cw_supply", "cwr": "cw_return", "lwt": "leaving",
          "ewt": "entering", "sup": "supply", "ret": "return", "rtn": "return", "sply": "supply",
          "oat": "outdoor_air", "oa": "outdoor_air", "sat": "supply_air", "rat": "return_air", "mat": "mixed_air",
          "dat": "discharge_air", "chilled": "chw", "chiller1": "chiller_1", "ch1": "chiller_1",
          "in": "inlet", "out": "outlet", "cond": "condenser", "evap": "evaporator", "comp": "compressor",
          "rh": "", "tons": "", "ton": "", "btu": ""}
_NOUN_END = {"load", "duty", "capacity", "demand", "consumption", "speed", "position", "level", "opening", "setpoint",
             "command", "cmd", "efficiency", "cop", "humidity"}


def canonical_name(base: str, quantity: Optional[str]) -> str:
    """'CHW Supply Temp' + temperature -> 'chw_supply_temperature'; 'chws_temp' -> same."""
    n = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(base))
    toks = [t for t in re.split(r"[^a-zA-Z0-9]+", n.lower()) if t]
    out: List[str] = []
    for t in toks:
        t = _ALIAS.get(t, t)
        for part in t.split("_"):
            if part and part not in _GENERIC:
                out.append(part)
    tag = "_".join(dict.fromkeys(out))            # keep order, drop repeats
    suffix = QUANTITY_SUFFIX.get(quantity or "", "")
    if not tag:
        return suffix or "value"
    if not suffix or out[-1] in _NOUN_END or tag.endswith(suffix):
        return tag
    return f"{tag}_{suffix}"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tag(name: str) -> str:
    stem = re.sub(r"\.[^.]+$", "", str(name))
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_") or "source"


def conversion_text(src: str, dst: str, difference: bool = False) -> str:
    """Human-readable conversion, e.g. '°F → °C: (x − 32) × 0.5556'."""
    try:
        a, b = parse_unit(src), parse_unit(dst)
    except UnitError as e:
        return str(e)
    if a.dim != b.dim:
        return f"incompatible: {dim_str(a.dim)} → {dim_str(b.dim)}"
    k = a.factor / b.factor
    c = 0.0 if difference else (a.offset - b.offset) / b.factor
    if abs(k - 1) < 1e-12 and abs(c) < 1e-12:
        return "no change"
    s = f"× {k:.6g}" if abs(k - 1) > 1e-12 else ""
    if abs(c) > 1e-12:
        s += f" {'+' if c > 0 else '−'} {abs(c):.6g}"
    return s.strip()


# ── mapping ─────────────────────────────────────────────────────────────────
def propose_mapping(df: pd.DataFrame, source: str, system: str = "metric", time_column: Optional[str] = None,
                    units: Optional[Dict[str, str]] = None, tz: Optional[str] = None) -> Dict[str, Any]:
    """Draft the mapping of one source from its headers (to be reviewed by a person)."""
    units = units or {}
    target_units = UNIT_SYSTEMS.get(system, UNIT_SYSTEMS["metric"])["units"]
    if time_column and time_column in df.columns:
        tcol, (t, how) = time_column, _parse_time(df[time_column])
    else:
        tcol, t, how = detect_time_column(df)
    tz_kind = "naive"
    if tcol is not None:
        raw = df[tcol]
        txt = raw.astype(str).head(200)
        if pd.api.types.is_numeric_dtype(raw) or txt.str.contains(r"(?:Z|[+\-]\d\d:?\d\d)$", regex=True).mean() > 0.8:
            tz_kind = "absolute"
    cols = []
    for c in df.columns:
        if c == tcol:
            continue
        base, utxt = split_header(c)
        if c in units:
            utxt = units[c]
        unit = try_parse_unit(utxt) if utxt else None
        g = infer_quantity(base, unit)
        num, _ = _to_numeric(df[c])
        is_num = pd.api.types.is_numeric_dtype(num)
        q = g.quantity if unit is not None else None
        target_unit = target_units.get(q) if q else None
        cols.append({
            "source": c, "include": bool(is_num and unit is not None), "kind": "numeric" if is_num else "text",
            "unit": utxt, "unit_known": unit is not None, "quantity": q, "guessed_quantity": g.quantity,
            "confidence": round(g.confidence, 2), "reason": g.reason, "difference": g.is_difference,
            "target": canonical_name(base, q or g.quantity), "target_unit": target_unit or utxt,
            "conversion": conversion_text(utxt, target_unit, g.is_difference) if unit is not None and target_unit else "",
            "choices": UNIT_CHOICES.get(q or "", []),
        })
    return {"source": source, "time_column": tcol, "time_parsed_as": how,
            "timezone": tz or "UTC", "timezone_needed": tz_kind == "naive" and tcol is not None,
            "columns": cols}


def make_recipe(mappings: Sequence[Dict[str, Any]], system: str = "metric", resample: Optional[str] = "auto",
                layout: str = "wide", agg: str = "mean") -> Dict[str, Any]:
    return {"schema": "pinneapple.recipe/1", "target_system": system, "resample": resample, "agg": agg,
            "layout": layout, "sources": [
                {"match": m["source"], "time_column": m.get("time_column"), "timezone": m.get("timezone", "UTC"),
                 "columns": [{k: c.get(k) for k in ("source", "include", "quantity", "unit", "target", "target_unit",
                                                     "difference")} for c in m["columns"]]} for m in mappings]}


def _match(recipe: Dict[str, Any], name: str, i: int) -> Dict[str, Any]:
    srcs = recipe.get("sources", [])
    for s in srcs:
        if s.get("match") == name:
            return s
    for s in srcs:
        pat = s.get("match") or ""
        if any(ch in pat for ch in "*?[") and re.fullmatch(pat.replace(".", r"\.").replace("*", ".*").replace("?", "."), name):
            return s
    if i < len(srcs):
        return srcs[i]
    raise ValueError(f"No recipe entry for source '{name}'.")


# ── standardize ─────────────────────────────────────────────────────────────
_NICE = [("1s", 1), ("10s", 10), ("1min", 60), ("5min", 300), ("15min", 900), ("1h", 3600), ("1D", 86400)]


def _auto_rule(dts: List[float]) -> Optional[str]:
    dts = [d for d in dts if d and np.isfinite(d)]
    if not dts:
        return None
    worst = max(dts)
    for rule, sec in _NICE:
        if sec >= worst * 0.98:
            return rule
    return "1D"


def standardize(sources: Sequence[Tuple[str, pd.DataFrame, bytes]], recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a recipe to sources [(file name, dataframe, raw bytes)]. Returns a dict with ``wide`` and
    ``long`` DataFrames, the ``manifest``, per-source notes and cross-source ``agreement``."""
    if not sources:
        raise ValueError("No sources given.")
    tagged: List[Tuple[str, str, pd.DataFrame, Dict[str, Any], bytes]] = []
    tags_seen: Dict[str, int] = {}
    for i, (name, df, raw) in enumerate(sources):
        tag = _tag(name)
        tags_seen[tag] = tags_seen.get(tag, 0) + 1
        if tags_seen[tag] > 1:
            tag = f"{tag}_{tags_seen[tag]}"
        tagged.append((name, tag, df, _match(recipe, name, i), raw))

    long_parts, raw_parts, notes, src_meta, dts = [], [], [], [], []
    variables: Dict[str, Dict[str, Any]] = {}
    for name, tag, df, spec, raw in tagged:
        tcol = spec.get("time_column")
        if not tcol or tcol not in df.columns:
            tcol, t, how = detect_time_column(df)
        else:
            t, how = _parse_time(df[tcol])
        if t is None:
            raise ValueError(f"'{name}': no readable time column (set 'time_column' in the recipe).")
        tzname = spec.get("timezone") or "UTC"
        raw_t = df[tcol]
        absolute = pd.api.types.is_numeric_dtype(raw_t) or \
            raw_t.astype(str).head(200).str.contains(r"(?:Z|[+\-]\d\d:?\d\d)$", regex=True).mean() > 0.8
        if not absolute and tzname.upper() != "UTC":
            try:
                t = t.dt.tz_localize(tzname, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None)
            except Exception as e:
                raise ValueError(f"'{name}': time zone '{tzname}' not understood ({e}).") from e
        ok = t.notna()
        dropped = int((~ok).sum())
        tt = t[ok]
        d = tt.sort_values().diff().dt.total_seconds()
        med = float(d[d > 0].median()) if (d > 0).any() else float("nan")
        dts.append(med)
        src_meta.append({"file": name, "tag": tag, "sha256": _sha(raw) if raw else None, "rows": int(len(df)),
                         "time_column": tcol, "time_parsed_as": how,
                         "timezone": "UTC (absolute timestamps)" if absolute else tzname,
                         "median_step_s": med, "start_utc": tt.min().isoformat() if len(tt) else None,
                         "end_utc": tt.max().isoformat() if len(tt) else None})
        if dropped:
            notes.append(f"{name}: {dropped} rows without a valid timestamp were dropped"
                         + (" (local times that do not exist or repeat at a DST change)" if not absolute else "") + ".")
        for col in spec.get("columns", []):
            if not col.get("include") or col.get("source") not in df.columns:
                continue
            src_unit, dst_unit = col.get("unit"), col.get("target_unit") or col.get("unit")
            try:
                a, b = parse_unit(src_unit), parse_unit(dst_unit)
            except UnitError as e:
                raise ValueError(f"'{name}' / '{col['source']}': {e}") from e
            if a.dim != b.dim:
                raise ValueError(f"'{name}' / '{col['source']}': cannot convert {src_unit} to {dst_unit}.")
            vals, _ = _to_numeric(df[col["source"]])
            if not pd.api.types.is_numeric_dtype(vals):
                notes.append(f"{name}: '{col['source']}' is not numeric and was skipped.")
                continue
            diff = bool(col.get("difference"))
            conv = b.from_si(a.to_si(vals.to_numpy(dtype=float)[ok.to_numpy()], diff), diff)
            target = str(col.get("target") or canonical_name(*split_header(col["source"])[:1], col.get("quantity")))
            target = re.sub(r"[^A-Za-z0-9_]+", "_", target).strip("_") or "value"
            long_parts.append(pd.DataFrame({"time_utc": tt.to_numpy(), "source": tag, "variable": target,
                                            "value": conv, "unit": dst_unit}))
            raw_parts.append(pd.DataFrame({"time_utc": tt.to_numpy(), "source": tag, "variable": target,
                                           "column": col["source"], "value": vals.to_numpy(dtype=float)[ok.to_numpy()],
                                           "unit": src_unit}))
            v = variables.setdefault(target, {"name": target, "quantity": col.get("quantity"), "unit": dst_unit,
                                              "dimension": dim_str(b.dim), "sources": []})
            if v["unit"] != dst_unit:
                raise ValueError(f"Variable '{target}' has two target units ({v['unit']}, {dst_unit}); use one.")
            v["sources"].append({"file": name, "tag": tag, "column": col["source"], "unit": src_unit,
                                 "conversion": conversion_text(src_unit, dst_unit, diff), "factor": a.factor / b.factor,
                                 "offset": 0.0 if diff else (a.offset - b.offset) / b.factor,
                                 "temperature_difference": diff, "gauge_pressure": a.gauge})
    if not long_parts:
        raise ValueError("Nothing to standardize: no column is included in the recipe.")
    long = pd.concat(long_parts, ignore_index=True).sort_values(["variable", "source", "time_utc"])
    rule = recipe.get("resample", "auto")
    if rule == "auto":
        rule = _auto_rule(dts)
    agg = recipe.get("agg", "mean")
    if rule:
        long = (long.set_index("time_utc").groupby(["source", "variable", "unit"])["value"]
                .resample(rule).agg(agg).reset_index())
        long = long[["time_utc", "source", "variable", "value", "unit"]]
    else:
        long = long.groupby(["time_utc", "source", "variable", "unit"], as_index=False)["value"].mean()
    multi = {v: len({s["tag"] for s in meta["sources"]}) > 1 for v, meta in variables.items()}
    long["column"] = [f"{v}@{s}" if multi[v] else v for v, s in zip(long["variable"], long["source"])]
    wide = long.pivot_table(index="time_utc", columns="column", values="value", aggfunc="mean").sort_index()
    order = [c for v in variables for c in sorted(set(long.loc[long["variable"] == v, "column"]))]
    wide = wide[order].reset_index()
    wide.columns.name = None
    units_by_col = {c: variables[c.split("@")[0]]["unit"] for c in order}
    for v, meta in variables.items():
        meta["columns"] = sorted(set(long.loc[long["variable"] == v, "column"]))
    step_s = pd.Timedelta(rule).total_seconds() if rule else None
    agreement = _agreement(wide, variables, multi, step_s)
    manifest = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "PINNeAPPle Engineering Data Standardizer",
        "time": {"column": "time_utc", "timezone": "UTC", "resample": rule, "aggregation": agg if rule else None},
        "layout": {"wide_columns": order, "long_columns": ["time_utc", "source", "variable", "value", "unit"]},
        "variables": list(variables.values()), "units": units_by_col, "sources": src_meta, "notes": notes,
        "recipe": recipe,
    }
    return {"wide": wide, "long": long.drop(columns=["column"]), "manifest": manifest, "agreement": agreement,
            "notes": notes, "resample": rule, "raw": pd.concat(raw_parts, ignore_index=True)}


_ABS_TOL_SI = {"temperature": 0.2}            # K: a good RTD pair; other quantities use 1 % of the level


def _agreement(wide: pd.DataFrame, variables: Dict[str, Dict[str, Any]], multi: Dict[str, bool],
               step_s: Optional[float] = None) -> List[Dict[str, Any]]:
    """Pairwise agreement of sources that measure the same variable, after conversion. A unit error shows as
    a large offset or scale; a time-zone error as a better match when one series is shifted by whole hours."""
    out = []
    for v, meta in variables.items():
        if not multi[v]:
            continue
        cols = meta["columns"]
        q = meta.get("quantity")
        try:
            factor = parse_unit(meta["unit"]).factor
        except UnitError:
            factor = 1.0
        ref = cols[0]
        for other in cols[1:]:
            both = wide[[ref, other]].dropna()
            if len(both) < 3:
                out.append({"variable": v, "a": ref, "b": other, "n": int(len(both)), "verdict": "no overlap",
                            "note": "The two sources do not overlap in time."})
                continue
            x, y = both[ref].to_numpy(), both[other].to_numpy()
            diff = y - x
            level = float(np.median(np.abs(x))) or 1.0
            tol = max(_ABS_TOL_SI.get(q, 0.0) / factor, 0.01 * level, 0.05 * float(np.std(x)))
            mad = float(np.median(np.abs(diff)))
            bias = float(np.median(diff))
            corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
            rec = {"variable": v, "a": ref, "b": other, "n": int(len(both)), "unit": meta["unit"],
                   "median_abs_diff": mad, "bias": bias, "tolerance": tol, "corr": corr}
            shift = _best_shift(wide[ref], wide[other], step_s) if step_s else None
            if shift and shift["gain"] > 0.1:
                rec.update(verdict="time shift", shift_hours=shift["hours"],
                           note=f"The two match best (r = {shift['r']:.2f} instead of {shift['r0']:.2f}) when "
                                f"'{other}' is moved by {shift['hours']:+g} h: one of the two sources has the wrong "
                                "time zone.")
            elif mad <= tol:
                rec.update(verdict="agree", note=f"Median difference {mad:.3g} {meta['unit']} (tolerance {tol:.3g}).")
            elif abs(bias) > tol and abs(mad - abs(bias)) <= tol:
                short = step_s and len(both) * step_s < 12 * 3600
                rec.update(verdict="offset", note=f"Constant offset of {bias:+.3g} {meta['unit']}: a calibration "
                                                  "difference between the sensors or a unit offset"
                                                  + (", or a time shift (the overlap is too short to test for one)."
                                                     if short else "."))
            else:
                ratio = float(np.median(y / x)) if np.all(x != 0) else None
                rec.update(verdict="differ", ratio=ratio,
                           note=f"Median difference {mad:.3g} {meta['unit']}"
                                + (f", ratio {ratio:.3g}: check the unit of one source." if ratio else "."))
            out.append(rec)
    return out


def _best_shift(a: pd.Series, b: pd.Series, step_s: float) -> Optional[Dict[str, float]]:
    per_h = 3600.0 / step_s
    if per_h != int(per_h):
        return None
    per_h = int(per_h)
    if (a.notna() & b.notna()).sum() < 12 * per_h:      # need half a day of overlap to see a daily pattern
        return None

    def r(k: int) -> Optional[float]:
        bb = b.shift(k)
        m = a.notna() & bb.notna()
        if m.sum() < 24 or a[m].std() == 0 or bb[m].std() == 0:
            return None
        return float(np.corrcoef(a[m], bb[m])[0, 1])
    r0 = r(0)
    if r0 is None:
        return None
    best = (0, r0)
    for hcount in range(-14, 15):
        if hcount == 0:
            continue
        rk = r(hcount * per_h)
        if rk is not None and rk > best[1]:
            best = (hcount, rk)
    if best[0] == 0:
        return None
    return {"hours": -best[0], "r0": r0, "r": best[1], "gain": best[1] - r0}


# ── writers ─────────────────────────────────────────────────────────────────
def write(result: Dict[str, Any], fmt: str = "csv", layout: str = "wide") -> Tuple[bytes, str, str]:
    """Serialize a result. Returns (bytes, media type, file name)."""
    df = result["wide"] if layout == "wide" else result["long"]
    man = result["manifest"]
    if fmt == "csv":
        out = df.copy()
        out["time_utc"] = pd.to_datetime(out["time_utc"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if layout == "wide":
            out.columns = ["time_utc"] + [f"{c} [{man['units'][c]}]" for c in out.columns[1:]]
        return out.to_csv(index=False).encode(), "text/csv", f"unified_{layout}.csv"
    if fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(df, preserve_index=False)
        meta = dict(table.schema.metadata or {})
        meta[b"pinneapple.units"] = json.dumps(man["units"] if layout == "wide" else {"value": "see unit column"}).encode()
        meta[b"pinneapple.manifest"] = json.dumps(man, default=str).encode()
        bio = io.BytesIO()
        pq.write_table(table.replace_schema_metadata(meta), bio)
        return bio.getvalue(), "application/vnd.apache.parquet", f"unified_{layout}.parquet"
    if fmt == "hdf5":
        import h5py
        bio = io.BytesIO()
        w = result["wide"]
        with h5py.File(bio, "w") as f:
            f.attrs["schema"] = SCHEMA
            f.attrs["manifest"] = json.dumps(man, default=str)
            tt = pd.to_datetime(w["time_utc"]).to_numpy().astype("datetime64[ns]").astype("int64") / 1e9
            ds = f.create_dataset("time_utc", data=tt)
            ds.attrs["units"] = "s since 1970-01-01T00:00:00Z"
            for c in w.columns[1:]:
                var = c.split("@")[0]
                meta = next(v for v in man["variables"] if v["name"] == var)
                d = f.create_dataset(c.replace("/", "_"), data=w[c].to_numpy(dtype=float))
                d.attrs["units"] = meta["unit"]
                d.attrs["quantity"] = meta["quantity"] or ""
                d.attrs["source"] = c.split("@")[1] if "@" in c else meta["sources"][0]["tag"]
        return bio.getvalue(), "application/x-hdf5", "unified.h5"
    if fmt == "json":
        out = df.copy()
        out["time_utc"] = pd.to_datetime(out["time_utc"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        body = {"manifest": man, "data": json.loads(out.to_json(orient="records"))}
        return json.dumps(body, default=str).encode(), "application/json", f"unified_{layout}.json"
    if fmt == "manifest":
        return json.dumps(man, indent=2, default=str).encode(), "application/json", "manifest.json"
    if fmt == "recipe":
        return json.dumps(man["recipe"], indent=2, default=str).encode(), "application/json", "recipe.json"
    if fmt == "zip":
        import zipfile
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
            for f, lay in (("csv", "wide"), ("parquet", "wide"), ("csv", "long"), ("hdf5", "wide"),
                           ("manifest", "wide"), ("recipe", "wide")):
                data, _, nm = write(result, f, lay)
                z.writestr(nm, data)
        return bio.getvalue(), "application/zip", "unified_dataset.zip"
    raise ValueError(f"Unknown format '{fmt}'.")


def to_physical_sample(result: Dict[str, Any]):
    """The wide table as a PINNeAPPle :class:`~pinneapple_data.physical_sample.PhysicalSample`
    (xarray Dataset on a ``time`` coordinate, units in attributes)."""
    import xarray as xr
    from .physical_sample import PhysicalSample
    w = result["wide"].set_index("time_utc")
    man = result["manifest"]
    ds = xr.Dataset({c: ("time", w[c].to_numpy()) for c in w.columns}, coords={"time": w.index.to_numpy()})
    for c in w.columns:
        ds[c].attrs["units"] = man["units"][c]
    return PhysicalSample(state=ds, schema={"units": man["units"], "variables": man["variables"]},
                          domain={"type": "timeseries"},
                          provenance={"sources": man["sources"], "created_utc": man["created_utc"], "schema": SCHEMA})


# ── example: one chiller seen by three systems ──────────────────────────────
def example_sources(seed: int = 3) -> List[Tuple[str, bytes]]:
    """The same chilled-water plant recorded by a building management system (CSV, local time,
    °C, semicolon + decimal comma), a SCADA REST API (JSON, Unix ms, °F, gpm, tons) and a test rig
    (Excel, ISO UTC, K, L/s, W)."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-03-10 00:00", "2026-03-13 00:00", freq="10s", tz="UTC")[:-1]
    n = len(t)
    h = ((t.hour + t.minute / 60 - 3) % 24).to_numpy()                     # local time UTC-3
    occ = ((h > 7) & (h < 20)).astype(float)
    load = 380 + 260 * occ + 40 * np.sin((h - 14) / 24 * 2 * np.pi) + rng.normal(0, 4, n)   # kW
    flow = 120 + 10 * occ + rng.normal(0, 0.6, n)                                          # m3/h
    chws = 6.7 + 0.15 * np.sin(np.arange(n) / 900) + rng.normal(0, 0.03, n)                # °C
    chwr = chws + load * 1e3 / (flow / 3600 * 997 * 4186)
    power = load / (5.6 + rng.normal(0, 0.03, n))
    truth = pd.DataFrame({"t": t, "chws": chws, "chwr": chwr, "flow": flow, "load": load, "power": power})

    def sample(step: str, noise: float, start: str, end: str) -> pd.DataFrame:
        s = truth.set_index("t").loc[start:end].resample(step).mean()
        return s + rng.normal(0, noise, s.shape) * np.array([1, 1, 0.5, 2, 1])

    a = sample("5min", 0.02, "2026-03-10", "2026-03-12 23:59")
    loc = a.index.tz_convert("America/Sao_Paulo").tz_localize(None)
    bms = pd.DataFrame({"Date/Time": loc.strftime("%d/%m/%Y %H:%M"),
                        "CHW Supply Temp [°C]": a["chws"].round(2), "CHW Return Temp [°C]": a["chwr"].round(2),
                        "CHW Flow [m3/h]": a["flow"].round(1), "Cooling Load [kW]": a["load"].round(1),
                        "Chiller Power [kW]": a["power"].round(1)})
    bms_bytes = bms.to_csv(sep=";", decimal=",", index=False).encode("cp1252")

    b = sample("1min", 0.02, "2026-03-11", "2026-03-12 23:59")
    scada = pd.DataFrame({"ts": (b.index.tz_convert("UTC").tz_localize(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms"),
                          "chws_temp_F": (b["chws"] * 1.8 + 32).round(2), "chwr_temp_F": (b["chwr"] * 1.8 + 32).round(2),
                          "chw_flow_gpm": (b["flow"] / 0.2271247).round(1),
                          "cooling_load_tons": (b["load"] / 3.516852842).round(2),
                          "chiller_kW": b["power"].round(1)})
    scada_bytes = json.dumps({"site": "plant-01", "records": scada.to_dict(orient="records")}).encode()

    c = sample("10s", 0.01, "2026-03-11 13:00", "2026-03-11 17:00")
    rig = pd.DataFrame({"time": c.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "T_chw_supply_K": (c["chws"] + 273.15).round(3), "T_chw_return_K": (c["chwr"] + 273.15).round(3),
                        "chw_flow_L_s": (c["flow"] / 3.6).round(3), "chiller_power_W": (c["power"] * 1000).round(0)})
    bio = io.BytesIO()
    with pd.ExcelWriter(bio) as xw:
        rig.to_excel(xw, sheet_name="rig_log", index=False)
    return [("bms_export.csv", bms_bytes), ("scada_api.json", scada_bytes), ("test_rig.xlsx", bio.getvalue())]


EXAMPLE_TIMEZONES = {"bms_export.csv": "America/Sao_Paulo"}

def _downsample(x: pd.Series, y: pd.Series, k: int) -> Dict[str, list]:
    n = len(x)
    if n == 0:
        return {"x": [], "y": []}
    idx = downsample_indices(n, k, "even")
    return {"x": [pd.Timestamp(v).strftime("%Y-%m-%d %H:%M") for v in x.iloc[idx]],
            "y": [None if not np.isfinite(v) else float(v) for v in y.iloc[idx].to_numpy(dtype=float)]}


def before_after_series(result: Dict[str, Any], k: int = 500) -> Dict[str, Any]:
    """Per variable of a :func:`standardize` result: the raw values of every source (in their original units) and
    the standardized values, on the common UTC time axis, downsampled to at most ``k`` evenly spaced points.

    Raw samples are projected on the common grid by nearest sample within one hour (``pd.merge_asof``) and hidden
    where the standardized series has no value, so both views line up. Returns
    ``{variable: {"x": [time labels], "unit": ..., "before": [{"name", "y"}], "after": [{"name", "y"}]}}`` with
    non-finite values as ``None`` (JSON-ready).
    """
    out: Dict[str, Any] = {}
    wide, raw = result["wide"], result["raw"]
    grid = wide["time_utc"]
    for v in result["manifest"]["variables"]:
        name = v["name"]
        before, after = [], []
        for s in v["sources"]:
            r = raw[(raw["variable"] == name) & (raw["source"] == s["tag"])].sort_values("time_utc")
            rr = pd.merge_asof(pd.DataFrame({"time_utc": grid}), r[["time_utc", "value"]], on="time_utc",
                               direction="nearest", tolerance=pd.Timedelta("1h"))
            col = f"{name}@{s['tag']}" if f"{name}@{s['tag']}" in wide.columns else name
            yb = rr["value"].where(~wide[col].isna().to_numpy())
            before.append({"name": f"{s['tag']} [{s['unit']}]", "y": _downsample(grid, yb, k)["y"]})
            after.append({"name": f"{s['tag']} [{v['unit']}]", "y": _downsample(grid, wide[col], k)["y"]})
        out[name] = {"x": _downsample(grid, wide[wide.columns[1]], k)["x"], "unit": v["unit"], "before": before,
                     "after": after}
    return out


__all__ = ["before_after_series", "UNIT_SYSTEMS", "UNIT_CHOICES", "canonical_name", "conversion_text", "propose_mapping", "make_recipe",
           "standardize", "write", "to_physical_sample", "example_sources", "EXAMPLE_TIMEZONES"]
