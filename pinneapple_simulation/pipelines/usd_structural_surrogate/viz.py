"""3-D views of what the operator predicts: the deformed pedestal, coloured by von Mises, in three places.

One scene, three outputs, all through the library's own Twin3D layer -- nothing here draws geometry by hand:

* the **browser**, ``Scene.export(folder, with_viewer=True)`` served by ``pinneapple_twin3d.serve``;
* **Blender**, ``Scene.render_blender(path, field=...)`` (``pinneapple_blender``), one image per design or a frame
  sequence over the sweep;
* **OpenUSD**, ``Scene.export(folder, usd=True)``: ``.usda`` with the field as a per-vertex ``primvars`` time sample,
  so the sweep opens in usdview, Omniverse or Isaac Sim and reads as an animation rather than a still.

Two scenes are built here:

``design_scene`` -- one design, the part deformed by its predicted displacement, ``von_mises`` and
``displacement`` as vertex fields, and sensors carrying the numbers an engineer reads (peak von Mises, tip
displacement, mass).

``sweep_scene`` -- the same part over a line through the design space: the geometry of every step, the field of every
step as a time sample, and the motion of every step, so the viewer plays "what happens as the plate gets thinner".

The part's surface is the finite-element boundary (``grid.surface_faces``), so what is drawn is the mesh that was
solved, not a smooth stand-in for it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .design import revolve_surface
from .grid import sample_field, surface_faces

__all__ = ["part_surface", "design_scene", "sweep_scene", "export_scene", "mass"]

RHO_STEEL = 7850.0


def part_surface(nodes: np.ndarray, elements: np.ndarray, element_type: str = "C3D4",
                 displacement: np.ndarray | None = None, scale: float = 1.0):
    """Boundary triangles of the part, optionally moved by ``displacement`` (n, 3) times ``scale``."""
    V = np.asarray(nodes, float).copy()
    if displacement is not None:
        V = V + scale * np.asarray(displacement, float)
    return V, surface_faces(nodes, elements, element_type)


def mass(nodes: np.ndarray, elements: np.ndarray, rho: float = RHO_STEEL) -> float:
    """Mass (kg) of the tetrahedral mesh: sum of |det| / 6 over the tets."""
    X = np.asarray(nodes, float)
    e = np.asarray(elements, int)[:, :4]
    a, b, c, d = X[e[:, 0]], X[e[:, 1]], X[e[:, 2]], X[e[:, 3]]
    vol = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0
    return float(vol.sum() * rho)


def _fields_at_nodes(surrogate, sample: dict[str, np.ndarray],
                     coords: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The operator's fields sampled at ``coords`` (default: the sample's finite-element nodes)."""
    occ = np.asarray(sample["occupancy"]).astype(bool)
    pred = surrogate.predict_fields(surrogate.features_of({"occupancy": occ}))
    P = np.asarray(sample["nodes"] if coords is None else coords, float)
    u = np.stack([sample_field(pred[c], surrogate.grid, P) for c in ("ux", "uy", "uz")], axis=-1)
    return {"u": u, "vm": sample_field(pred["vm"], surrogate.grid, P), "mag": np.linalg.norm(u, axis=1)}


def design_scene(surrogate, sample: dict[str, np.ndarray], *, deform_scale: float = 200.0,
                 load_n: float = 1000.0):
    """A :class:`pinneapple_twin3d.scene.Scene` of one design: the part, deformed, with its fields and sensors.

    The surface is the finite-element boundary of the mesh that was solved, so what is drawn is the mesh the numbers
    came from and not a smooth stand-in for it.
    """
    from pinneapple_twin3d.scene import Scene

    f = _fields_at_nodes(surrogate, sample)
    V, faces = part_surface(sample["nodes"], sample["elements"], sample.get("element_type", "C3D4"),
                            displacement=f["u"], scale=deform_scale)
    scene = Scene("UR10 pedestal surrogate", length_unit="m", times=[0.0])
    scene.add_part("pedestal", V, faces)
    scene.add_field("pedestal", "von_mises", f["vm"], unit="Pa")
    scene.add_field("pedestal", "displacement", f["mag"], unit="m")
    tip, peak = int(np.argmax(f["mag"])), int(np.argmax(f["vm"]))
    scene.add_sensor("peak_vm", V[peak], label="peak von Mises", unit="Pa", quantity="stress",
                     series=[float(f["vm"][peak])])
    scene.add_sensor("tip", V[tip], label="largest displacement", unit="m", quantity="displacement",
                     series=[float(f["mag"][tip])])
    scene.set_step_labels([f"load {load_n:g} N"])
    return scene


def sweep_scene(surrogate, samples: dict[str, np.ndarray], index, *, deform_scale: float = 200.0,
                load_n: float = 1000.0, n_theta: int = 48):
    """A scene whose time axis is the design index: one part per design, visible only at its own step.

    The geometry itself changes across the sweep, and Twin3D's motion is rigid, so a sweep cannot be one part with a
    moving mesh. Each design is its own part -- the analytic revolved section, so every design has the same vertex
    layout and the animation morphs instead of popping -- and ``add_visibility`` masks all but the current step.
    """
    from pinneapple_twin3d.scene import Scene

    idx = [int(i) for i in index]
    if not idx:
        raise ValueError("sweep_scene needs at least one design")
    names = list(surrogate.space.bounds)
    r_plate = surrogate.space.plate_diameter
    scene = Scene("UR10 pedestal sweep", length_unit="m", times=list(range(len(idx))))
    labels, peak = [], []
    for k, i in enumerate(idx):
        design = {n: float(samples["design"][i][j]) for j, n in enumerate(names)}
        V, faces = revolve_surface(design, r_plate, n_theta)
        f = _fields_at_nodes(surrogate, {n: v[i] for n, v in samples.items()}, V)
        # revolve_surface already has the right vertex layout for every design, so deform the drawn surface
        drawn = V + deform_scale * f["u"]
        part = scene.add_part(f"design_{k:03d}", drawn, faces)
        scene.add_field(part.name, "von_mises", f["vm"], unit="Pa")
        scene.add_visibility(part.name, np.eye(len(idx))[k])
        names_txt = ", ".join(f"{n}={design[n] * 1e3:.0f}" for n in names)
        labels.append(names_txt)
        peak.append(float(f["vm"].max()))
    scene.add_sensor("peak_vm", np.zeros(3), label="peak von Mises", unit="Pa", quantity="stress", series=peak)
    scene.set_step_labels(labels)
    return scene


def export_scene(scene, folder: str | Path, *, usd: bool = True, blender: str | None = None,
                 field: str = "von_mises", step: int = -1) -> dict[str, Any]:
    """Write the viewer folder (and, on request, the ``.usda`` and a Blender render). Returns what was written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"folder": str(folder)}
    try:
        out["viewer"] = scene.export(str(folder), with_viewer=True, usd=False)
    except Exception as e:                                            # a viewer failure must not lose the usd
        out["viewer_error"] = f"{type(e).__name__}: {e}"
    if usd:
        try:
            scene.export(str(folder / "usd"), with_viewer=False, usd=True)
            out["usd"] = str(folder / "usd")
        except Exception as e:
            out["usd_error"] = f"{type(e).__name__}: {e}"
    if blender:
        try:
            out["blender"] = scene.render_blender(blender, step=step, field=field if field else None)
        except Exception as e:
            out["blender_error"] = f"{type(e).__name__}: {e}"
    return out
