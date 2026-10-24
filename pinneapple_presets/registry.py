"""Registry of the industrial presets and the index the Hub reads (``pinneapple_presets/index.json``)."""
from __future__ import annotations

import importlib
import json
import os

from .core import Preset

__all__ = ["register", "load", "names", "list_presets", "index", "write_index", "INDEX_PATH", "RESULTS_DIR"]

HERE = os.path.dirname(__file__)
INDEX_PATH = os.path.join(HERE, "index.json")
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "examples", "industrial_presets"))

# name -> "module:Class" (imported on first use so listing stays light)
_BUILTIN: dict[str, str] = {
    "water_hammer_valve_closure": "pinneapple_presets.cases.water_hammer:WaterHammer",
    "water_distribution_network": "pinneapple_presets.cases.pipe_network:WaterNetwork",
    "plate_fin_heatsink_forced": "pinneapple_presets.cases.heatsink:PlateFinHeatsink",
    "power_pcb_junction": "pinneapple_presets.cases.pcb:PowerPCB",
    "steel_footbridge_pedestrian": "pinneapple_presets.cases.footbridge:SteelFootbridge",
    "shell_tube_cooler": "pinneapple_presets.cases.shell_tube:ShellTubeCooler",
    "steam_pipe_insulation": "pinneapple_presets.cases.insulation:SteamLineInsulation",
    "pressure_vessel_asme": "pinneapple_presets.cases.pressure_vessel:PressureVessel",
    "pump_npsh_cavitation": "pinneapple_presets.cases.pump_npsh:PumpNPSH",
}
_EXTRA: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator: add a user preset (a :class:`Preset` subclass with a unique ``name``)."""
    if not (isinstance(cls, type) and issubclass(cls, Preset)) or not cls.name:
        raise TypeError("register a Preset subclass with a name")
    _EXTRA[cls.name] = cls
    return cls


def names() -> list[str]:
    """Names of every industrial preset (built in and registered), sorted."""
    return sorted(set(_BUILTIN) | set(_EXTRA))


def load(name: str) -> Preset:
    """The preset ``name`` (see :func:`names`)."""
    if name in _EXTRA:
        return _EXTRA[name]()
    if name not in _BUILTIN:
        import difflib
        near = difflib.get_close_matches(name, names(), n=3)
        raise KeyError(f"no preset {name!r}" + (f"; did you mean {near}?" if near else f"; presets: {names()}"))
    mod, cls = _BUILTIN[name].split(":")
    return getattr(importlib.import_module(mod), cls)()


def list_presets(sector: str | None = None) -> list[dict]:
    """Name, title, sector and capabilities of every preset (``sector``: substring filter)."""
    rows = []
    for n in names():
        p = load(n)
        if sector and sector.lower() not in p.sector.lower():
            continue
        c = p.card()
        rows.append({"name": n, "title": p.title, "sector": p.sector, "question": p.question,
                     "capabilities": c["capabilities"]})
    return rows


def _results(name: str) -> dict:
    d = os.path.join(RESULTS_DIR, name)
    out = {"folder": os.path.relpath(d, os.path.dirname(HERE)).replace(os.sep, "/")}
    summ = os.path.join(d, "results", "summary.json")
    if os.path.isfile(summ):
        with open(summ, encoding="utf-8") as fh:
            s = json.load(fh)
        out.update({k: s[k] for k in ("decision", "trust", "validation", "surrogate", "inversion", "optimization",
                                      "figures", "figures_3d", "variants", "meta") if k in s})
    return out


def index() -> dict:
    """Everything the Hub shows: per preset the data sheet, the measured results and the figure paths (relative
    to the repository root)."""
    return {"schema": "pinneapple.presets/1",
            "presets": [{**load(n).card(), "results": _results(n)} for n in names() if n in _BUILTIN]}


def write_index(path: str = INDEX_PATH) -> str:
    from pinneapple_core.jsonable import to_jsonable
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(index()), fh, indent=1, ensure_ascii=False)
    return path
