"""FMI import compatibility: run FMUs through :func:`simulate_fmu` in every interface they offer and record what
worked, as the table the FMI tools list asks for (exporting tool, FMI version, platform, FMU, interface).

    from pinneapple_simulation.external_solvers.modelica.fmi_compat import check_fmus, markdown_table
    rows = check_fmus(sorted(Path("Reference-FMUs/2.0").glob("*.fmu")), exporter="Reference-FMUs 0.0.39")
    print(markdown_table(rows))

A run passes when it reaches the stop time with finite outputs. For the ``Dahlquist`` test model (x' = -k x) the
trajectory is also compared with the exact solution exp(-k t): within 1e-3 through Model Exchange (the importer's
integrator) and within 0.1 through Co-Simulation, where the FMU integrates with its own fixed-step explicit Euler.
"""
from __future__ import annotations

import platform
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .fmi_runner import FMUSimConfig, simulate_fmu

__all__ = ["CompatRow", "platform_tag", "check_fmu", "check_fmus", "markdown_table"]

INTERFACES = {"CoSimulation": "CS", "ModelExchange": "ME"}


@dataclass
class CompatRow:
    fmu: str
    fmi_version: str
    interface: str               # CS | ME
    exporter: str
    platform: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def platform_tag() -> str:
    """The FMI binary platform of this machine, e.g. "x86_64-linux" (FMI 3.0 naming)."""
    os_name = {"linux": "linux", "darwin": "darwin", "win32": "windows"}.get(sys.platform, sys.platform)
    return f"{platform.machine().lower().replace('amd64', 'x86_64')}-{os_name}"


def _dahlquist_error(res: dict[str, np.ndarray], k: float) -> float | None:
    if "x" not in res or "time" not in res:
        return None
    t, x = res["time"], res["x"]
    return float(np.max(np.abs(x - x[0] * np.exp(-k * (t - t[0])))))


TOL = {"ModelExchange": 1e-3, "CoSimulation": 0.1}


def check_fmu(fmu: str | Path, fmi_type: str, exporter: str = "", stop_time: float | None = None,
              tol: float | None = None) -> CompatRow:
    """Simulate ``fmu`` with ``fmi_type`` ("CoSimulation" or "ModelExchange") and report whether it worked."""
    from fmpy import read_model_description

    path = Path(fmu)
    md = read_model_description(str(path), validate=False)
    row = CompatRow(path.stem, md.fmiVersion, INTERFACES[fmi_type], exporter, platform_tag(), False)
    stop = stop_time
    if stop is None and md.defaultExperiment is not None and md.defaultExperiment.stopTime is not None:
        stop = float(md.defaultExperiment.stopTime)
    try:
        res = simulate_fmu(path, FMUSimConfig(stop_time=stop if stop is not None else 1.0, fmi_type=fmi_type))
    except Exception as exc:  # noqa: BLE001 - every failure is a row of the table
        row.detail = f"{type(exc).__name__}: {exc}"[:200]
        return row
    numeric = [v for v in res.values() if np.asarray(v).dtype.kind in "fiu"]
    if not numeric or not all(np.all(np.isfinite(v)) for v in numeric):
        row.detail = "non-finite output"
        return row
    if path.stem == "Dahlquist":
        k = next((float(v.start) for v in md.modelVariables if v.name == "k" and v.start is not None), 1.0)
        err = _dahlquist_error(res, k)
        if err is not None and err > (TOL[fmi_type] if tol is None else tol):
            row.detail = f"max error {err:.2e} against exp(-k t)"
            return row
        row.detail = f"max error {err:.1e} against exp(-k t)" if err is not None else ""
    row.passed = True
    return row


def check_fmus(fmus: Iterable[str | Path], exporter: str = "",
               interfaces: Iterable[str] = ("CoSimulation", "ModelExchange")) -> list[CompatRow]:
    """:func:`check_fmu` for every FMU and every interface it implements (others are skipped)."""
    from fmpy import read_model_description

    rows = []
    for fmu in fmus:
        md = read_model_description(str(fmu), validate=False)
        has = {"CoSimulation": md.coSimulation is not None, "ModelExchange": md.modelExchange is not None}
        rows.extend(check_fmu(fmu, t, exporter) for t in interfaces if has[t])
    return rows


def markdown_table(rows: list[CompatRow]) -> str:
    """One table per exporting tool and FMI version: FMUs as rows, the platform as column, cells "CS, ME"."""
    out: list[str] = []
    for exporter in dict.fromkeys(r.exporter for r in rows):
        for ver in sorted({r.fmi_version for r in rows if r.exporter == exporter}):
            sel = [r for r in rows if r.exporter == exporter and r.fmi_version == ver]
            plats = sorted({r.platform for r in sel})
            out += [f"**{exporter}, FMI {ver}**", "", "| FMU | " + " | ".join(plats) + " |",
                    "|---|" + "---|" * len(plats)]
            for fmu in dict.fromkeys(r.fmu for r in sel):
                cells = []
                for p in plats:
                    ok = [r.interface for r in sel if r.fmu == fmu and r.platform == p and r.passed]
                    bad = [r.interface for r in sel if r.fmu == fmu and r.platform == p and not r.passed]
                    parts = ([", ".join(ok)] if ok else []) + ([f"(failed: {', '.join(bad)})"] if bad else [])
                    cells.append(" ".join(parts) or "-")
                out.append(f"| {fmu} | " + " | ".join(cells) + " |")
            out.append("")
    return "\n".join(out)
