"""Visualization and export nodes: plots, the three.js field viewer, a rendered still/animation, the report."""
from __future__ import annotations

import base64
import html
import time

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact, is_artifact
from ._common import load_solution, plot_solution


@node("viz.plot", "Plot", category="Visualization",
      inputs=[Port("solution", "solution", multiple=True)],
      outputs=[Port("figure", "figure")],
      params=[Param("field", "int", 0, minimum=0, help="field index")],
      calls="matplotlib (pinneapple_flow.nodes._common.plot_solution)")
def plot(ctx, inputs, params):
    """A figure per solution: curves in time (1-D), a colour map (2-D), mid-plane slices (3-D)."""
    sols = [load_solution(v) for v in inputs["solution"]]
    if not sols:
        raise StepFailed("connect a solution")
    first = None
    for i, s in enumerate(sols):
        a = ctx.figure(plot_solution(s, s["label"], field=int(params["field"])), f"plot_{i}", s["label"])
        first = first or a
    return {"figure": first}


def _grid_mesh(sol, field: int = 0, height: float = 0.25):
    """A structured Mesh of a solution (quads for 2-D / space-time, hexahedra for 3-D) and its point values."""
    from pinneapple_data.cae.model import CellBlock, Mesh
    coords = sol["coords"]
    V = sol["values"][..., field]
    spatial = [c for c in coords if c != "t"]
    if "t" in coords and len(spatial) >= 2:
        V = np.moveaxis(V, coords.index("t"), -1)[..., -1]
        coords = spatial
    axes = [sol["axes"][c] for c in coords]
    if len(axes) == 1:
        raise StepFailed("a 3-D view needs a 2-D or 3-D field (or 1-D + time)")
    shape = tuple(len(a) for a in axes)
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], 1)
    vals = V.ravel()
    idx = np.arange(pts.shape[0]).reshape(shape)
    deform = None
    if len(axes) == 2:
        P = np.c_[pts, np.zeros(len(pts))]
        a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
        c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
        blocks = [CellBlock("quad", np.stack([a, b, c, d], 1))]
        span = max(np.ptp(axes[0]), np.ptp(axes[1]))
        rng = float(np.ptp(vals)) or 1.0
        deform = {"height": np.c_[np.zeros((len(P), 2)), (vals - vals.min()) / rng * height * span]}
    else:
        P = pts
        i = idx[:-1, :-1, :-1].ravel()
        n1, n2 = shape[1] * shape[2], shape[2]
        conn = np.stack([i, i + n1, i + n1 + n2, i + n2, i + 1, i + n1 + 1, i + n1 + n2 + 1, i + n2 + 1], 1)
        blocks = [CellBlock("hexahedron", conn)]
    return Mesh(points=P, blocks=blocks), vals, deform


@node("viz.field3d", "3-D viewer", category="Visualization",
      inputs=[Port("solution", "solution")],
      outputs=[Port("scene", "scene")],
      params=[Param("field", "int", 0, minimum=0), Param("height", "float", 0.25, minimum=0.0, maximum=2.0,
                                                          help="warp of 2-D fields (fraction of the domain size)")],
      calls="pp.viz.field_view (glTF + meta for /shared/fieldview3d.js, three.js)")
def field3d(ctx, inputs, params):
    """An interactive three.js view of the field (2-D as a warped surface, 3-D as the volume's surface)."""
    from pinneapple_tools.visualization.studio.fieldview import field_view
    sol = load_solution(inputs["solution"])
    mesh, vals, deform = _grid_mesh(sol, int(params["field"]), float(params["height"]))
    name = sol["fields"][int(params["field"])]
    fields = [{"name": name, "values": vals, "label": f"{name} ({sol['label']})",
               "deform": "height" if deform is not None else None}]
    fv = field_view(mesh, fields, deform=deform, title=sol["label"])
    p = ctx.path("scene.glb")
    p.write_bytes(fv.glb())
    meta = fv.meta if hasattr(fv, "meta") else fv.to_dict()
    m = ctx.save_json("scene", meta)
    view = {"glb": str(p), "meta": meta}
    ctx.view("scene", view)
    ctx.metric("points", int(mesh.n_points))
    return {"scene": {**m, "glb": str(p)}}


@node("viz.render", "Render (Blender-style)", category="Visualization",
      inputs=[Port("solution", "solution")],
      outputs=[Port("figure", "figure")],
      params=[Param("field", "int", 0, minimum=0), Param("animate", "bool", True, help="GIF over time"),
              Param("frames", "int", 24, minimum=2, maximum=240), Param("samples", "int", 32, minimum=1,
                                                                         maximum=4096, advanced=True)],
      calls="pinneapple_blender.cycles.CyclesScene when Blender (bpy) is installed, else matplotlib shaded surface "
            "+ pinneapple_tools.visualization.animation")
def render(ctx, inputs, params):
    """A rendered still (and an animation over time) of a 2-D field as a lit surface: Blender Cycles when the bpy
    module is installed, otherwise a matplotlib shaded surface; the report says which renderer was used."""
    from pinneapple_blender.cycles import available
    sol = load_solution(inputs["solution"])
    k = int(params["field"])
    coords = sol["coords"]
    spatial = [c for c in coords if c != "t"]
    V = sol["values"][..., k]
    if "t" in coords:
        V = np.moveaxis(V, coords.index("t"), 0)
    else:
        V = V[None]
    if len(spatial) == 1:                                      # space-time surface of a 1-D field
        X, Y = np.meshgrid(sol["axes"][spatial[0]], sol["axes"].get("t", [0.0]), indexing="ij")
        frames = [V.T]
        labels = (spatial[0], "t")
    elif len(spatial) == 2:
        X, Y = np.meshgrid(*(sol["axes"][c] for c in spatial), indexing="ij")
        frames = list(V)
        labels = tuple(spatial)
    else:
        mid = V.shape[3] // 2
        X, Y = np.meshgrid(*(sol["axes"][c] for c in spatial[:2]), indexing="ij")
        frames = [f[:, :, mid] for f in V]
        labels = tuple(spatial[:2])
    out = None
    if available():
        from pinneapple_blender.cycles import CyclesScene
        renderer = "blender-cycles"
        sc = CyclesScene(samples=int(params["samples"]))
        Z = frames[-1]
        sc.add_surface(X, Y, (Z - Z.min()) / (np.ptp(Z) or 1.0) * 0.3 * np.ptp(X))
        sc.fit_camera(np.array([X.min(), Y.min(), 0]), np.array([X.max(), Y.max(), 0.3 * np.ptp(X)]))
        sc.light()
        p = sc.render(str(ctx.path("render.png")))
        out = artifact(p, "png")
        ctx.report["figures"].append({"path": p, "name": "render", "caption": "Blender Cycles"})
    else:
        renderer = "matplotlib (Blender's bpy is not installed)"
        out = ctx.figure(_shaded(X, Y, frames[-1], labels, sol["label"]), "render", "shaded surface, last frame")
    ctx.metric("renderer", renderer)
    if params["animate"] and len(frames) > 1:
        import matplotlib
        matplotlib.use("Agg")
        from pinneapple_tools.visualization.animation import animate_scalar_field
        sel = np.linspace(0, len(frames) - 1, min(int(params["frames"]), len(frames))).astype(int)
        traj = np.stack([frames[i] for i in sel])
        t_vals = np.asarray(sol["axes"].get("t", np.arange(len(frames))))[sel]
        gif = ctx.path("animation.gif")
        anim = animate_scalar_field(traj, x=X[:, 0], y=Y[0], t_vals=t_vals, title=sol["label"],
                                    field_label=sol["fields"][k], save_path=str(gif), writer="pillow")
        del anim
        ctx.report["figures"].append({"path": str(gif), "name": "animation", "caption": "animation over time"})
        ctx.metric("animation_frames", int(len(sel)))
    return {"figure": out}


def _shaded(X, Y, Z, labels, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    fig = plt.figure(figsize=(6.4, 4.6))
    ax = fig.add_subplot(projection="3d")
    ls = LightSource(azdeg=315, altdeg=40)
    rgb = ls.shade(Z, cmap=plt.get_cmap("viridis"), vert_exag=0.4, blend_mode="soft")
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=32, azim=-55)
    return fig


def _img(path: str) -> str:
    try:
        data = base64.b64encode(open(path, "rb").read()).decode()
    except OSError:
        return ""
    mime = "image/gif" if path.endswith(".gif") else "image/png"
    return f'<img src="data:{mime};base64,{data}" style="max-width:100%">'


@node("export.report", "Report", category="Export",
      inputs=[Port("items", "any", multiple=True, help="any outputs to document")],
      outputs=[Port("report", "report")],
      params=[Param("title", "str", "PINNeAPPle flow report"), Param("format", "enum", "html",
                                                                     choices=["html", "markdown", "json"])],
      calls="pinneapple_flow.nodes.viz.report (self-contained HTML / Markdown / JSON)")
def report(ctx, inputs, params):
    """A self-contained report of the connected outputs: metrics, parameters, figures (embedded)."""
    items = inputs["items"] or []
    title = html.escape(params["title"])
    parts, md = [f"<h1>{title}</h1><p>generated {time.strftime('%Y-%m-%d %H:%M')}</p>"], [f"# {params['title']}"]
    blob = []
    for i, v in enumerate(items):
        if is_artifact(v) and v.get("kind") == "png":
            parts.append(_img(v["$artifact"]))
            md.append(f"![figure {i}]({v['$artifact']})")
        elif isinstance(v, dict):
            flat = {k: x for k, x in v.items() if isinstance(x, (int, float, str, bool)) and k != "$artifact"}
            if flat:
                rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(x))}</td></tr>"
                               for k, x in flat.items())
                parts.append(f"<table border=1 cellpadding=4>{rows}</table>")
                md.append("\n".join(f"- {k}: {x}" for k, x in flat.items()))
            if v.get("figure") and isinstance(v["figure"], str):
                parts.append(_img(v["figure"]))
        blob.append(v)
    from ..decisions import Decision, report_html, report_markdown
    decisions = list(ctx.meta.get("flow_decisions") or [])
    for nid, rep in ctx.upstream_reports.items():
        for d in (rep or {}).get("decisions") or []:
            decisions.append(Decision.from_dict({**d, "node": d.get("node") or nid}).to_dict())
    ctx.metric("decisions", len(decisions))
    ctx.metric("decisions_without_reference", sum(1 for d in decisions if d.get("no_reference")))
    parts.append(report_html(decisions))
    md.append(report_markdown(decisions))
    blob.append({"decisions": decisions})
    fmt = params["format"]
    if fmt == "html":
        p = ctx.path("report.html")
        p.write_text("<!doctype html><meta charset=utf-8><title>" + title + "</title>"
                     "<body style='font-family:sans-serif;max-width:900px;margin:auto'>" + "\n".join(parts))
        a = artifact(p, "html")
    elif fmt == "markdown":
        p = ctx.path("report.md")
        p.write_text("\n\n".join(md))
        a = artifact(p, "md")
    else:
        a = ctx.save_json("report", blob)
    ctx.metric("items", len(items))
    ctx.summary(f"report written ({fmt})")
    return {"report": a}
