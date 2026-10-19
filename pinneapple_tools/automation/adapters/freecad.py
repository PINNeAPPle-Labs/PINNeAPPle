"""FreeCAD and the CfdOF workbench.

Actions run a Python script **inside the FreeCAD GUI** on the run's virtual display (CfdOF needs ``FreeCADGui``;
the GUI also gives real screenshots): the script works through FreeCAD's own API, then the adapter finds the FreeCAD
window with xdotool, captures the screen (``screen`` evidence) and the 3D view region of it (``window`` evidence),
and lets FreeCAD exit.

* ``freecad.open``: open a document, list its objects and spreadsheet parameters, measure a solid.
* ``freecad.macro``: run a macro (``.FCMacro`` / ``.py``) with arguments; the macro gets ``ARGS`` and fills ``RESULT``.
* ``cfdof.setup``: build a CfdOF analysis (laminar steady incompressible, velocity inlet, pressure outlet, walls,
  gmsh mesh) on a solid and write the mesh case and the OpenFOAM case.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from ..display import GuiUnavailable, XdotoolDriver
from ..record import Check, Evidence
from .base import ActionFailed, ActionResult, Adapter, StepContext, register

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

__all__ = ["FreeCADAdapter", "CfdOFAdapter", "run_in_freecad_gui"]


def run_in_freecad_gui(script: str | Path, args: dict[str, Any], ctx: StepContext, name: str,
                       view: dict[str, Any] | None = None, hold_s: float = 60.0) -> tuple[dict[str, Any], ActionResult]:
    """Run ``script`` inside the FreeCAD GUI with ``ARGS = args``; returns (RESULT, evidence/commands).

    ``view`` = {"show": [object names], "hide": [...], "view": "viewIsometric" | "viewFront" | ...} sets what the 3D
    view shows when the screenshots are taken.

    Raises :class:`ActionFailed` when the script raised (with its traceback) or FreeCAD did not answer."""
    tc = ctx.toolchain
    if ctx.display is None:
        raise ActionFailed("FreeCAD GUI actions need a display (Xvfb)")
    res = ActionResult()
    job_dir = ctx.step_dir
    job = {"script": str(Path(script).resolve()), "args": args, "result": str(job_dir / f"{name}.result.json"),
           "ready": str(job_dir / f".{name}.ready"), "release": str(job_dir / f".{name}.release"),
           "hide": (view or {}).get("hide", []), "show": (view or {}).get("show", []),
           "view": (view or {}).get("view", "viewIsometric"), "hold_s": hold_s, "window": [1600, 1000]}
    for k in ("result", "ready", "release"):
        Path(job[k]).unlink(missing_ok=True)
    job_file = job_dir / f"{name}.job.json"
    job_file.write_text(json.dumps(job, indent=1))
    log = job_dir / f"log.freecad_{name}"
    env = ctx.env({"PINNEAPPLE_FC_JOB": str(job_file), "QT_QPA_PLATFORM": "xcb"})
    cmd = [tc.freecad, str(SCRIPTS / "freecad_gui_job.py")]
    ctx.commands.append(f"DISPLAY={ctx.display.display} PINNEAPPLE_FC_JOB={ctx.rel(job_file)} "
                        f"freecad freecad_gui_job.py  # runs {Path(script).name}")
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, cwd=job_dir, env=env, stdout=f, stderr=subprocess.STDOUT)
    ctx.logs.append(ctx.rel(log))
    try:
        end = time.time() + ctx.timeout_s
        while not Path(job["ready"]).exists():
            if proc.poll() is not None:
                raise ActionFailed(f"FreeCAD exited with {proc.returncode} before finishing {Path(script).name} "
                                   f"(log {ctx.rel(log)})")
            if time.time() > end:
                raise ActionFailed(f"FreeCAD did not finish {Path(script).name} within {ctx.timeout_s:g} s")
            time.sleep(0.3)
        # observe the GUI: find the FreeCAD main window, then capture the whole screen
        ok, _ = XdotoolDriver.available()
        if ok:
            xd = XdotoolDriver(ctx.display)
            try:
                win = xd.wait_window("FreeCAD", timeout_s=20)
                res.outputs["window_title"] = xd.title(win)
                res.actions.append(f"found the FreeCAD window {res.outputs['window_title']!r} with xdotool")
            except GuiUnavailable as e:
                res.actions.append(f"xdotool: {e}")
            ctx.commands.extend(xd.log)
        shot = ctx.display.screenshot(job_dir / f"{name}_screen.png")
        res.evidence.append(Evidence(ctx.rel(shot), "screen", f"FreeCAD on the virtual display after {name}"))
        box = json.loads(Path(job["result"]).read_text()).get("view_box")
        if box:
            crop = ctx.display.screenshot(job_dir / f"{name}_view.png", box=tuple(box))
            res.evidence.insert(0, Evidence(ctx.rel(crop), "window", f"FreeCAD 3D view after {name}"))
    finally:
        Path(job["release"]).write_text("release")
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    out = json.loads(Path(job["result"]).read_text())
    if out.get("error"):
        raise ActionFailed(f"{Path(script).name} raised inside FreeCAD:\n{out['error'][-1500:]}")
    return out["result"], res


def _path(v: str, ctx: StepContext) -> Path:
    p = Path(v)
    return p if p.is_absolute() else ctx.run_dir / p


class FreeCADAdapter(Adapter):
    name = "freecad"
    mode = "gui"
    requires = ("freecad", "xvfb")
    actions = ("open", "macro")
    needs_display = ("open", "macro")

    def do_open(self, params: dict[str, Any], ctx: StepContext) -> ActionResult:
        doc = _path(params["document"], ctx)
        if not doc.is_file():
            raise ActionFailed(f"document {doc} does not exist")
        result, res = run_in_freecad_gui(SCRIPTS / "freecad_inspect.py",
                                         {"document": str(doc), "solid": params.get("solid")}, ctx, "open",
                                         view=params.get("view"))
        (ctx.step_dir / "project.json").write_text(json.dumps(result, indent=1))
        res.artifacts.append(ctx.step_dir / "project.json")
        res.outputs.update({"document": str(doc), "parameters": result["parameters"],
                            "objects": len(result["objects"])})
        res.checks.append(Check("document opens without invalid objects", not result["invalid_objects"],
                                "invalid: " + ", ".join(result["invalid_objects"]) if result["invalid_objects"]
                                else f"{len(result['objects'])} objects, all recomputed",
                                source="FreeCAD Document.recompute"))
        if result.get("solid"):
            s = result["solid"]
            res.outputs["volume_mm3"] = s["volume_mm3"]
            res.checks.append(Check(f"solid {s['name']} is valid", bool(s["valid"]) and s["volume_mm3"] > 0,
                                    f"volume {s['volume_mm3']:.6g} mm³, {s['faces']} faces",
                                    value=s["volume_mm3"], source="Part.Shape.isValid"))
        return res

    def do_macro(self, params: dict[str, Any], ctx: StepContext) -> ActionResult:
        macro = _path(params["macro"], ctx)
        if not macro.is_file():
            raise ActionFailed(f"macro {macro} does not exist")
        args = {k: (str(_path(v, ctx)) if k in ("document", "output") else v)
                for k, v in (params.get("args") or {}).items()}
        result, res = run_in_freecad_gui(macro, args, ctx, "macro", view=params.get("view"))
        (ctx.step_dir / "macro_result.json").write_text(json.dumps(result, indent=1))
        res.artifacts.append(ctx.step_dir / "macro_result.json")
        res.outputs.update(result)
        res.actions.append(f"ran macro {macro.name} with parameters "
                           f"{json.dumps((params.get('args') or {}).get('parameters', {}))}")
        for c in result.get("checks", []):
            res.checks.append(Check(c["name"], bool(c["passed"]), c.get("detail", ""), c.get("value"),
                                    c.get("threshold"), source=macro.name))
        if "output" in args:
            res.artifacts.append(Path(args["output"]))
        return res


class CfdOFAdapter(Adapter):
    name = "cfdof"
    mode = "gui"
    requires = ("freecad", "cfdof", "openfoam", "gmsh", "xvfb")
    actions = ("setup",)
    needs_display = ("setup",)

    def do_setup(self, params: dict[str, Any], ctx: StepContext) -> ActionResult:
        tc = ctx.toolchain
        out_dir = ctx.step_dir / "cfd"
        out_dir.mkdir(parents=True, exist_ok=True)
        foam_dir = str(Path(tc.foam_bashrc).parent.parent)
        args = {"document": str(_path(params["document"], ctx)), "solid": params["solid"],
                "output_dir": str(out_dir), "foam_dir": foam_dir, "gmsh": tc.gmsh,
                "inlet_velocity_mm_s": float(params["inlet_velocity_mm_s"]),
                "mesh_size_mm": float(params.get("mesh_size_mm", 8)),
                "max_iterations": int(params.get("max_iterations", 1000)),
                "convergence_tol": float(params.get("convergence_tol", 1e-4)),
                "fluid": params.get("fluid") or {"name": "Air", "density": 1.2, "viscosity": 1.8e-5},
                "inlet": params.get("inlet", "x_min"), "outlet": params.get("outlet", "x_max")}
        result, res = run_in_freecad_gui(SCRIPTS / "cfdof_setup.py", args, ctx, "cfdof",
                                         view=params.get("view"))
        (ctx.step_dir / "cfdof_setup.json").write_text(json.dumps({"args": args, "result": result}, indent=1))
        res.artifacts += [ctx.step_dir / "cfdof_setup.json", Path(result["case"]), Path(result["mesh_case"])]
        res.outputs.update({"case": result["case"], "mesh_case": result["mesh_case"], "faces": result["faces"],
                            "inlet_area_mm2": result["inlet_area_mm2"], "outlet_area_mm2": result["outlet_area_mm2"],
                            "fluid": args["fluid"], "inlet_velocity_mm_s": args["inlet_velocity_mm_s"],
                            "inlet_velocity_m_s": args["inlet_velocity_mm_s"] / 1000.0})
        need = [Path(result["case"], "Allrun"), Path(result["case"], "system", "controlDict"),
                Path(result["mesh_case"], "Allmesh")]
        missing = [str(p) for p in need if not p.is_file()]
        res.checks.append(Check("CfdOF wrote the mesh case and the OpenFOAM case", not missing,
                                "missing: " + ", ".join(missing) if missing else
                                f"inlet {', '.join(result['faces']['inlet'])}, outlet "
                                f"{', '.join(result['faces']['outlet'])}, {len(result['faces']['walls'])} wall faces",
                                source="CfdOF CfdCaseWriterFoam.writeCase"))
        res.actions.append(f"assigned boundaries by plane: inlet = {args['inlet']}, outlet = {args['outlet']}, "
                           "every other face = wall")
        return res


register(FreeCADAdapter())
register(CfdOFAdapter())
