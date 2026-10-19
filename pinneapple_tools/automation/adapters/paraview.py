"""ParaView through ``pvpython``: open an OpenFOAM case at its last time, export a cell field to CSV (cell centres
and values) and render it on a slice to PNG (``render`` evidence).

* ``paraview.export``: params ``case`` (folder with a ``.foam`` file or the file itself), ``field`` (default U),
  ``arrays`` (exported, default [U, p]), ``slice_normal`` (default [0, 0, 1]), ``view_up``
  (default [0, 1, 0]). Checks: the CSV has one row per cell
  of the mesh and the image exists.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..record import Check, Evidence
from .base import ActionFailed, ActionResult, Adapter, StepContext, register

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

__all__ = ["ParaViewAdapter"]


class ParaViewAdapter(Adapter):
    name = "paraview"
    mode = "api"
    requires = ("paraview",)
    actions = ("export",)
    needs_display = ("export",)              # the Ubuntu ParaView build renders through X

    def do_export(self, params: dict[str, Any], ctx: StepContext) -> ActionResult:
        res = ActionResult()
        case = Path(params["case"])
        case = case if case.is_absolute() else ctx.run_dir / case
        foam = case if case.suffix == ".foam" else next(iter(sorted(case.glob("*.foam"))), None)
        if foam is None:
            foam = case / "case.foam"
            foam.write_text("")
            res.actions.append(f"created the empty reader file {foam.name}")
        field = params.get("field", "U")
        job = {"foam_file": str(foam), "field": field, "arrays": params.get("arrays", ["U", "p"]),
               "csv": str(ctx.step_dir / f"{field}_cells.csv"), "png": str(ctx.step_dir / f"{field}_slice.png"),
               "result": str(ctx.step_dir / "paraview_result.json"),
               "slice_normal": params.get("slice_normal", [0, 0, 1]), "view_up": params.get("view_up", [0, 1, 0]),
               "bar_title": params.get("bar_title", f"{field} [m/s]" if field == "U" else field)}
        job_file = ctx.step_dir / "paraview_job.json"
        job_file.write_text(json.dumps(job, indent=1))
        ctx.run([ctx.toolchain.pvpython, str(SCRIPTS / "paraview_export.py"), str(job_file)], ctx.step_dir,
                "log.pvpython")
        out = json.loads(Path(job["result"]).read_text())
        csv_path, png = Path(job["csv"]), Path(job["png"])
        if not csv_path.is_file():
            raise ActionFailed("pvpython did not write the CSV")
        with open(csv_path, newline="") as f:
            rows = sum(1 for _ in csv.reader(f)) - 1
        res.checks.append(Check(f"{field} exported for every cell", rows == out["cells"],
                                f"{rows} rows in {csv_path.name}, {out['cells']} cells in the mesh at t = {out['time']:g}",
                                value=rows, threshold=out["cells"], source="ParaView OpenFOAMReader + SaveData"))
        res.checks.append(Check("image rendered", png.is_file() and png.stat().st_size > 1000,
                                f"{png.name}", source="ParaView SaveScreenshot"))
        res.evidence.append(Evidence(ctx.rel(png), "render",
                                     f"|{field}| on the mid-plane slice at t = {out['time']:g} (ParaView)"))
        res.artifacts += [csv_path, png]
        res.outputs.update({"csv": str(csv_path), "image": str(png), "time": out["time"], "cells": out["cells"],
                            f"{field}_magnitude_range": out["range"]})
        return res


register(ParaViewAdapter())
