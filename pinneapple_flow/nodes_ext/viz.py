"""Any result in 3-D (``examples/studio/01_any_result_in_3d``): a file ``pinneapple_data.cae`` reads (OpenFOAM case
or zip, CalculiX .frd, Abaqus .inp, Gmsh, VTK) or STL / OBJ geometry, as a :class:`pinneapple.viz.Scene`, exported to
glTF and USD, an interactive three.js viewer and, when bpy is installed, a Blender Cycles render."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from .examples import examples_root

_DEFAULT = "apps/interop_hub/examples/calculix_cantilever.frd"


@node("viz.any_result_3d", "Any result in 3-D (glTF, USD, viewer, Blender)", category="Visualization",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("source", "str", _DEFAULT, help="result or geometry file, relative to the repository or absolute"),
              Param("title", "str", "CalculiX cantilever"),
              Param("labels", "dict", {"STRESS": "von Mises stress (MPa)", "DISP": "Displacement (mm)"}),
              Param("render", "enum", "auto", choices=["auto", "on", "off"],
                    help="Blender Cycles still (auto: when bpy is installed)"),
              Param("render_field", "str", "STRESS"), Param("samples", "int", 64, minimum=1)],
      calls="pinneapple.viz.Scene.from_file / Scene.save (.glb, .usda) + pinneapple.viz.web_viewer + "
            "pinneapple.viz.render")
def any_result_3d(ctx, inputs, params):
    """examples/studio/01_any_result_in_3d.py: load, export, view, render."""
    import pinneapple as pp
    from pinneapple_tools.visualization.studio.web import screenshot
    src = Path(str(params["source"]))
    if not src.is_absolute():
        src = examples_root().parent / src
    scene = pp.viz.Scene.from_file(str(src))
    scene.title = str(params["title"])
    scene.labels.update(dict(params["labels"] or {}))
    fields = list(scene.field_names())
    ctx.metric("fields", ", ".join(fields), "on the surface (vectors as magnitude, stress as von Mises)")
    out = {"metrics": None}
    for ext in ("glb", "usda"):
        p = ctx.path(f"scene.{ext}")
        scene.save(str(p))
        ctx.metric(f"{ext}_bytes", p.stat().st_size)
    d = ctx.path("viewer")
    pp.viz.web_viewer(scene, str(d))
    ctx.view("viewers", [{"key": "scene", "index": str(d / "index.html")}])
    png = screenshot(str(d), str(ctx.path("viewer.png")))
    if png:
        ctx.report["figures"].append({"path": png, "name": "viewer", "caption": "three.js viewer"})
    mode = params["render"]
    if mode != "off":
        from .render import _bpy_available
        ok, why = _bpy_available()
        if ok:
            jpg = ctx.path("render.jpg")
            # in a subprocess, as render_worker.py: importing bpy could crash the flow engine or the Studio server
            code = ("import os, sys, pinneapple as pp; s = pp.viz.Scene.from_file(sys.argv[1]); "
                    "s.labels.update(__import__('json').loads(sys.argv[2])); "
                    "pp.viz.render(s, sys.argv[3], field=sys.argv[4], samples=int(sys.argv[5])); os._exit(0)")
            r = subprocess.run([sys.executable, "-c", code, str(src), json.dumps(dict(params["labels"] or {})),
                                str(jpg), str(params["render_field"]), str(int(params["samples"]))],
                               capture_output=True, text=True)
            if r.returncode != 0 or not jpg.is_file():
                raise StepFailed(f"the Blender render failed (exit {r.returncode}): {r.stderr.strip()[-300:]}")
            ctx.report["figures"].append({"path": str(jpg), "name": "render", "caption": "Blender Cycles"})
            from ..runtime import artifact
            out["figure"] = artifact(str(jpg), "jpg")
        elif mode == "on":
            ctx.warn(f"no Blender render: {why}")
    if "figure" not in out and png:
        from ..runtime import artifact
        out["figure"] = artifact(png, "png")
    out["metrics"] = ctx.report["metrics"]
    return out


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="viz.any_result_3d", title="Any result file in 3-D: glTF, USD, a web viewer, a Blender render",
        domain="visualization", example="studio",
        keywords={"frd": 2, "vtk": 1.5, "gltf": 2, "glb": 2, "usd": 2, "omniverse": 2, "in 3d": 1.5, "3-d": 1,
                  "viewer": 1, "result file": 2, "visualize": 1, "visualise": 1, "render": 1},
        min_score=3, nodes=("viz.any_result_3d",), inputs=("a result or geometry file",),
        uncited="a viewer and exporters: no physics decision to cite", template="studio/01_any_result_in_3d",
        description="examples/studio/01_any_result_in_3d.py"))


_register_capabilities()
