"""The figures and the 3-D viewer a fitted design space can show, drawn with the repository's own tooling.

A family that has finished has ``model/view.json`` next to its card: the loss curve, the design space with the split,
the bodies the samples came from, and — per sample — the field the reference produced against the field the model
predicted. This module turns that file into PNGs with :mod:`pinneapple_tools.visualization` and into a Studio viewer
page, so a figure the campaign shows and a figure a report carries are drawn by the same theme.

    from benchmarks.lpm_figures import ensure, ensure_viewer
    path = ensure(Path("~/lpm-campaign"), "pp_fin", "loss")     # builds it only if the view is newer
    html = ensure_viewer(Path("~/lpm-campaign"), "pp_fin")      # the Studio's own 3-D page
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

#: what a family can show. ``loss`` and ``design`` come from the fit, ``geometry`` and ``prediction`` from its samples.
#: ``viewer`` is not a PNG: it is the Studio's interactive page, built from the same samples.
KINDS = ("loss", "design", "geometry", "prediction", "viewer")

_LOCK = threading.Lock()


def _style() -> None:
    """The house theme, on a backend that needs no display: this runs inside the campaign's server."""
    import matplotlib

    matplotlib.use("Agg")                           # before pyplot is imported anywhere below
    from pinneapple_tools.visualization import use_theme

    use_theme(kind="report", dark=True)


def _source(view: dict[str, Any]) -> str:
    reference = (view.get("reference") or {}).get("name") or "the family's reference"
    return f"reference: {reference} · model: bagged LPM, mean of the heads · {view['family']}"


def _title(view: dict[str, Any], kind: str) -> str:
    return {
        "loss": f"training loss — {view['family']}",
        "design": f"design space and split — {view['family']}",
        "geometry": f"the bodies behind the samples — {view['family']}",
        "prediction": f"reference against prediction — {view['family']}",
    }[kind]


# --------------------------------------------------------------------------------------------------- the four figures


def loss_figure(view: dict[str, Any]):
    """The loss curve: the one figure that says whether the fit converged or stopped early."""
    from pinneapple_tools.visualization import PALETTE, make_figure

    fig, ax = make_figure(figsize=(5.4, 2.8))
    ax.plot(view["loss"]["epoch"], view["loss"]["loss"], color=PALETTE[0], lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title(_title(view, "loss"), fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    first, last = view["loss"]["loss"][0], view["loss"]["loss"][-1]
    ax.annotate(f"{first:.3g} → {last:.3g}", xy=(0.98, 0.92), xycoords="axes fraction", ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def design_figure(view: dict[str, Any]):
    """The design space on its two widest parameters, with the held-out bodies drawn apart from the training ones.

    The split here is the pipeline's, which is by geometry: a held-out dot is a body the model never saw, at operating
    points it never saw.
    """
    from pinneapple_tools.visualization import PALETTE, make_figure

    ax_name, ay_name = (view["axes"] + ["", ""])[:2]
    fig, ax = make_figure(figsize=(4.8, 3.6))
    for held, colour, name in ((False, PALETTE[0], "training"), (True, PALETTE[1], "held out")):
        rows = [d for d in view["designs"] if d["held_out"] is held and ax_name in d["values"]]
        ax.scatter([d["values"][ax_name] for d in rows], [d["values"][ay_name] for d in rows],
                   s=12, color=colour, label=f"{name} ({len(rows)})", alpha=0.85, linewidths=0)
    ax.set_xlabel(ax_name or "parameter 1")
    ax.set_ylabel(ay_name or "parameter 2")
    ax.set_title(_title(view, "design"), fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def _two_widest(view: dict[str, Any], points):
    """The two axes of the body that actually vary, so a 3-D shape is readable in a static picture."""
    import numpy as np

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return 0, 1
    span = pts.max(0) - pts.min(0)
    order = list(np.argsort(span)[::-1])
    return int(order[0]), int(order[min(1, len(order) - 1)])


def geometry_figure(view: dict[str, Any]):
    """Each sample's body: the point cloud the model sees, one panel per sample, drawn from the points themselves."""
    from pinneapple_tools.visualization import make_panel_figure, panel_letters

    samples = view["samples"]
    fig, axes = make_panel_figure(ncols=min(len(samples), 4))
    axes = axes.ravel()
    for ax, sample in zip(axes, samples, strict=False):
        i, j = _two_widest(view, sample["points"])
        xs = [p[i] for p in sample["points"]]
        ys = [p[j] for p in sample["points"]]
        # the point cloud and nothing more: joining consecutive points draws the order they were sampled in, which for
        # a surface of revolution is a zig-zag between meridians, not a body
        ax.scatter(xs, ys, s=2, linewidths=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"x{i + 1}")
        ax.set_ylabel(f"x{j + 1}")
        ax.set_title(sample["label"], fontsize=8)
    for ax in axes[len(samples):]:
        ax.axis("off")
    panel_letters(fig, axes=list(axes[: len(samples)]))
    fig.suptitle(_title(view, "geometry"), fontsize=10)
    fig.tight_layout()
    return fig


def prediction_figure(view: dict[str, Any]):
    """Reference against prediction, one row per field and one column per sample, along the ordered query points."""
    from pinneapple_tools.visualization import PALETTE, make_figure, panel_letters

    samples, fields = view["samples"], view["fields"]
    fig, axes = make_figure(nrows=max(len(fields), 1), ncols=max(len(samples), 1),
                            figsize=(3.1 * max(len(samples), 1), 2.2 * max(len(fields), 1)), squeeze=False)
    for col, sample in enumerate(samples):
        for row, field in enumerate(fields):
            ax = axes[row][col]
            target, pred = sample["target"][field], sample["pred"][field]
            ax.plot(range(len(target)), target, color=PALETTE[0], lw=1.3, label="reference")
            ax.plot(range(len(pred)), pred, color=PALETTE[1], lw=1.1, ls="--", label="model")
            if row == 0:
                ax.set_title(sample["label"], fontsize=8)
            if col == 0:
                unit = (view["units"] or [""] * len(fields))[row]
                ax.set_ylabel(f"{field} [{unit}]" if unit else field)
            if row == len(fields) - 1:
                ax.set_xlabel("query point")
            if sample["relative_l2"] is not None:
                ax.annotate(f"rel-L2 {sample['relative_l2']:.3f}", xy=(0.02, 0.04), xycoords="axes fraction",
                            fontsize=7)
    axes[0][0].legend(fontsize=7, loc="upper right")
    panel_letters(fig, axes=[axes[r][c] for r in range(len(fields)) for c in range(len(samples))])
    fig.suptitle(_title(view, "prediction"), fontsize=10)
    fig.tight_layout()
    return fig


_BUILDERS = {"loss": loss_figure, "design": design_figure, "geometry": geometry_figure,
             "prediction": prediction_figure}


# ------------------------------------------------------------------------------------------------------- the cache


def view_path(out: str | Path, family: str) -> Path:
    return Path(out) / family / "model" / "view.json"


def _skin(points) -> tuple[Any, Any]:
    """(vertices, triangles) for a point cloud: a Delaunay in the cloud's best-fit plane.

    The families hand the model a point cloud, not a mesh, so a viewer has to build the skin. The projection is the
    least inventive choice available: it adds no surface the points do not support, and it makes a plate, a wing or a
    bracket rotate as the thin body it is. It does fill a concave silhouette — a nozzle's interior reads as solid — so
    the caption says "triangulated in the best-fit plane" rather than "the body".
    """
    import numpy as np
    from scipy.spatial import Delaunay

    pts = np.asarray(points, dtype=float)
    verts = np.hstack([pts, np.zeros((len(pts), 1))]) if pts.shape[1] == 2 else pts
    centre = verts.mean(0)
    _, _, vt = np.linalg.svd(verts - centre, full_matrices=False)
    uv = (verts - centre) @ vt[:2].T
    faces = Delaunay(uv).simplices
    # Delaunay returns each triangle in whichever orientation it likes, and the viewer culls back faces — so a sheet
    # whose faces point away from the camera draws nothing at all. That is how pp_airfoil and pp_exchanger came out as
    # an empty stage with a colour bar and working controls: the bodies were there, facing the wrong way. Every triangle
    # is turned to the plane normal that points up (+z), which is the side all four preset views look from.
    normal = vt[2] if vt.shape[0] > 2 else np.array([0.0, 0.0, 1.0])
    upward = normal[2] > 1e-9 or (abs(normal[2]) <= 1e-9 and normal[1] >= 0)
    normal = normal if upward else -normal
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    flip = np.einsum("ij,j->i", np.cross(b - a, c - a), normal) < 0
    return verts, np.where(flip[:, None], faces[:, [0, 2, 1]], faces)


def scene(view: dict[str, Any]):
    """The Studio :class:`Scene` the viewer page shows: one body per sample, coloured by the model's own field.

    Two choices a viewer forces and a figure does not. The samples are laid **side by side** — they are four different
    bodies of the same family and, drawn at their own coordinates, they sit on top of each other. And a 2-D family is
    drawn as its contour (a tube along the outline) rather than as a skin: the design space is a curve, and filling it
    would draw a body the family does not have.
    """
    import numpy as np

    from pinneapple_tools.visualization.studio import Scene

    samples = view["samples"]
    scn = Scene(axes="z_up", title=f"{view.get('title') or view['family']} — samples side by side")
    field_units = dict(zip(view["fields"], view.get("units") or [""] * len(view["fields"]), strict=False))
    for f in view["fields"]:
        unit = field_units.get(f, "")
        scn.labels[f] = f"{f} [{unit}]" if unit else f"{f} [-]"

    skins = [_skin(sample["points"]) for sample in samples]
    # The face-on preset renders a flat scene with y across the screen and x down it, so a row of four bodies along x
    # would stand on end in a narrow column, and one along y lies across the screen — the picture a contour plot of the
    # same family draws. The row has to follow the body's widest in-plane axis, or the bodies overlap; so when that axis
    # is x, the sheet gets a quarter turn and the row lands on y.
    #
    # "Flat" is about the whole set, not one body: four sheets stacked at different heights is a 3-D arrangement and
    # keeps the oblique view.
    every = np.vstack([verts for verts, _ in skins]) if skins else np.zeros((1, 3))
    spread = np.ptp(every, axis=0) if skins else np.zeros(3)
    wide = float(spread.max()) or 1.0
    flat = bool(skins) and float(spread[2]) <= 1e-9 * wide
    if flat and spread[0] >= spread[1]:
        turn = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])       # (x, y) -> (y, -x)
        skins = [(verts @ turn.T, faces) for verts, faces in skins]
    placed = [verts for verts, _ in skins]
    allpts = np.vstack(placed) if placed else np.zeros((1, 3))
    axis = int(np.argmax(np.ptp(allpts, axis=0)))
    starts, cursor = [], 0.0
    for verts in placed:                                # the bodies never touch: each gets its own shelf
        lo, hi = float(verts[:, axis].min()), float(verts[:, axis].max())
        starts.append(cursor - lo)
        cursor += (hi - lo) + 0.25 * max(hi - lo, 1e-9)

    def values_of(sample: dict[str, Any], n: int) -> dict[str, Any]:
        at_points = sample.get("at_points")
        if at_points is None:                           # a view written before the flag existed: decide from the data
            pts, q = sample.get("points") or [], sample.get("query") or []
            at_points = bool(len(pts) == len(q) == n and np.allclose(np.asarray(pts), np.asarray(q)))
        if not at_points:                               # values and vertices are one set only then
            return {}
        out = {}
        for field in view["fields"]:
            vals = np.asarray(sample["pred"].get(field, []), dtype=float)
            if len(vals) == n:
                out[field] = vals
        return out

    for i, sample in enumerate(samples):
        verts, faces = skins[i]
        shift = np.zeros(3)
        shift[axis] = starts[i]
        pts = verts + shift
        surface = Scene.from_arrays(pts, faces, name=sample["label"], material="grey").surfaces[0]
        surface.fields.update(values_of(sample, len(pts)))
        scn.surfaces.append(surface)
    # a flat scene is a line seen edge-on in the default 3/4 view, so it opens face-on; a 3-D one keeps the view that
    # shows its depth.
    scn.view = "top" if flat else "iso"
    return scn


def _code_stamp() -> float:
    """Newest modification of the code that draws, so a changed drawing rule invalidates what is on disk.

    The cache compared a figure only against ``view.json``: a fix to the way a body is drawn kept serving the page
    built before it. The airfoil and the exchanger went on showing a blank stage after the faces were oriented.
    """
    here = Path(__file__).resolve()
    studio = here.parent.parent / "pinneapple_tools" / "visualization" / "studio"
    files = [here, studio / "web.py", studio / "scene.py"]
    return max((p.stat().st_mtime for p in files if p.is_file()), default=0.0)


def ensure_viewer(out: str | Path, family: str) -> Path:
    """The Studio 3-D page for one family (``single_file``: three.js inlined, opens from ``file://``)."""
    source = view_path(out, family)
    target = source.parent / "figures" / "viewer.html"
    with _LOCK:
        fresh = max(source.stat().st_mtime, _code_stamp()) if source.is_file() else 0.0
        if target.is_file() and source.is_file() and target.stat().st_mtime >= fresh:
            return target
        if not source.is_file():
            raise FileNotFoundError(f"{family} has no view.json yet: it has not been fitted")
        from pinneapple_tools.visualization.studio.web import web_viewer

        view = json.loads(source.read_text(encoding="utf-8"))
        target.parent.mkdir(parents=True, exist_ok=True)
        web_viewer(scene(view), str(target.parent), title=view.get("title") or family,
                   single_file=True, filename="viewer.html")
        return target


def ensure(out: str | Path, family: str, kind: str) -> Path:
    """The file for one family and one kind, rebuilt only when ``view.json`` is newer than it.

    A family that has not finished has no view and gets an explicit error, which the page draws as "not measured yet"
    rather than as a blank chart.
    """
    if kind not in KINDS:
        raise KeyError(f"unknown figure {kind!r}; the families show {', '.join(KINDS)}")
    if kind == "viewer":
        return ensure_viewer(out, family)
    source = view_path(out, family)
    target = source.parent / "figures" / f"{kind}.png"
    with _LOCK:
        fresh = max(source.stat().st_mtime, _code_stamp()) if source.is_file() else 0.0
        if target.is_file() and source.is_file() and target.stat().st_mtime >= fresh:
            return target
        if not source.is_file():
            raise FileNotFoundError(f"{family} has no view.json yet: it has not been fitted")
        view = json.loads(source.read_text(encoding="utf-8"))
        _style()
        import matplotlib.pyplot as plt

        from pinneapple_tools.visualization import save_figure, source_note

        fig = _BUILDERS[kind](view)
        source_note(fig, _source(view))
        target.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, target.with_suffix(""), formats=("png",))   # save_figure appends the suffix
        plt.close(fig)
        return target


def build_all(out: str | Path, family: str) -> dict[str, str]:
    """Every figure a finished family can show, as paths. Used by the campaign's own ``--figures`` if asked."""
    return {kind: str(ensure(out, family, kind)) for kind in KINDS}
