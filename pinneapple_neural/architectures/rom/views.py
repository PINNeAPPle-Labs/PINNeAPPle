"""A held-out snapshot on its body: the full-order field, a prediction (a ROM, a surrogate) and their difference on the
mesh, with the deformed shape when the snapshots carry a displacement, the boundary-condition and load regions of the
dataset, the error hotspots and a short text on where the error concentrates.

    from pinneapple_neural.architectures.rom.views import reconstruction_view
    fv, info = reconstruction_view(snaps, model, index, reported={"study mean": 0.044})
    fv.glb(), fv.meta            # for /shared/fieldview3d.js
    info["errors"]["von_mises"]["summary"]

The snapshot needs a mesh (``SnapshotSet.mesh``: connectivity, optionally its own node coordinates). Every number is
computed from the stored full-order snapshot and the model's prediction at its inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["reconstruction_view", "prediction_view", "zones"]


def _deformation_field(snaps, mesh_unit: str):
    """The displacement field (name, factor to the mesh's length unit): the first point vector field that is one."""
    from pinneapple_tools.visualization.studio.fieldview import displacement_factor
    units = snaps.meta.get("units") or {}
    for f in snaps.fields:
        fac = displacement_factor(f["name"], units.get(f["name"]), f["components"], mesh_unit) \
            if f["where"] == "point" else None
        if fac is not None:
            return f["name"], fac
    return None, None


def zones(snaps, points, grow: float = 0.1) -> dict[str, np.ndarray]:
    """The dataset's named node sets grown by ``grow`` times the body size ("clamp zone", "loaded tip zone") and the
    remaining nodes ("rest of the body"): the regions errors are compared over."""
    from pinneapple_analysis.validation.error_map import grow_region
    P = np.asarray(points, float)
    size = float(np.ptp(P, axis=0).max()) or 1.0
    out = {f"{name} zone": grow_region(P, np.asarray(st["nodes"], np.int64), grow * size)
           for name, st in (snaps.meta.get("node_sets") or {}).items()}
    if out:
        rest = np.setdiff1d(np.arange(len(P)), np.unique(np.concatenate(list(out.values()))))
        if len(rest):
            out["rest of the body"] = rest
    return out


def reconstruction_view(snaps, model, index: int, *, reported: Mapping[str, float] | None = None,
                        grow: float = 0.1, max_faces: int | None = None, n_hotspots: int = 3,
                        full_label: str = "") -> tuple[Any, dict[str, Any]]:
    """View of snapshot ``index`` of ``snaps`` against ``model`` (a ``ReducedModel``): :func:`prediction_view` of the
    model's prediction at the snapshot's time or parameters."""
    q = {"times": [float(snaps.times[index])]} if model.mode == "time" else {"params": snaps.params[index:index + 1]}
    fv, info = prediction_view(snaps, index, model.predict(**q)[0], label="ROM", reported=reported, grow=grow,
                               max_faces=max_faces, n_hotspots=n_hotspots, full_label=full_label,
                               note=f"ROM: {model.method}" + (f" rank {model.rank}" if model.rank else ""))
    info["input"] = {k: np.asarray(v, float).ravel().tolist() for k, v in q.items()}
    return fv, info


def prediction_view(snaps, index: int, pred, *, label: str = "Prediction", note: str = "",
                    reported: Mapping[str, float] | None = None, grow: float = 0.1, max_faces: int | None = None,
                    n_hotspots: int = 3, full_label: str = "",
                    reported_by_field: Mapping[str, Mapping[str, float]] | None = None) -> tuple[Any, dict[str, Any]]:
    """View of snapshot ``index`` of ``snaps`` against ``pred``, a predicted snapshot row in the layout of ``snaps``
    (a ROM, a surrogate). Named node sets of the dataset become markers (kind ``bc`` / ``load`` / ``region``) and,
    grown by ``grow`` times the body size, the regions the error is compared over ("clamp zone"; the remaining nodes
    are the "rest of the body"). ``reported``: label -> relative error of the study to quote next to the pointwise
    numbers (``reported_by_field``: field -> such a mapping, added to it). Returns ``(FieldView, info)``; ``info["errors"][field]`` holds the error map of each field."""
    from pinneapple_analysis.validation.error_map import error_map, pointwise_error
    from pinneapple_tools.visualization.studio.fieldview import VIEW_MAX_FACES, field_view
    mesh = snaps.mesh(index)
    if mesh is None:
        raise ValueError("these snapshots carry no mesh (connectivity): there is no body to show")
    pred = np.asarray(pred, float).ravel()
    key = label.lower().replace(" ", "_")
    units = snaps.meta.get("units") or {}
    mesh_unit = units.get("points") or units.get("x") or "m"
    full = full_label or snaps.meta.get("full_order") or "Full-order"
    dname, dfac = _deformation_field(snaps, mesh_unit)
    deform = {}
    if dname:
        deform[full] = snaps.field_values(dname)[index] * dfac
        deform[label] = snaps.field_values(dname, pred[None])[0] * dfac
    P = mesh.points
    sets = snaps.meta.get("node_sets") or {}
    markers = [{"name": name, "kind": st.get("kind", "region"), "points": P[np.asarray(st["nodes"], np.int64)],
                "detail": st.get("detail", "")} for name, st in sets.items()]
    regions = zones(snaps, P, grow)
    fields, errors = [], {}
    for f in snaps.fields:
        if f["where"] != "point" or f["size"] != len(P) * f["components"]:
            continue
        n, unit = f["name"], units.get(f["name"], "")
        tv, pv = snaps.field_values(n)[index], snaps.field_values(n, pred[None])[0]
        em = error_map(P, pv, tv, regions=regions, unit=unit, n_hotspots=n_hotspots)
        errors[n] = em.to_dict({**(reported or {}), **(reported_by_field or {}).get(n, {})} or None)
        mag = " magnitude" if f["components"] > 1 else ""
        d_full, d_rom = (full, label) if dname else (None, None)
        names = [f"{n}:full", f"{n}:{key}", f"{n}:error"]
        fields += [{"name": names[0], "values": tv, "label": f"{full} {n}{mag}", "unit": unit, "group": n,
                    "deform": d_full},
                   {"name": names[1], "values": pv, "label": f"{label} {n}{mag}", "unit": unit, "group": n,
                    "deform": d_rom},
                   {"name": names[2], "values": pointwise_error(pv, tv), "label": f"|{label} − {full}| {n}",
                    "unit": unit, "kind": "error", "deform": d_full,
                    "note": f"relative L2 {100 * em.rel_l2:.2f} % on this snapshot"}]
        for k, h in enumerate(em.hotspots):
            markers.append({"name": f"E{k + 1}", "kind": "hotspot", "points": P[h["indices"]], "centre": h["at"],
                            "fields": names, "detail": f"|error| up to {h['max']:.3g} {unit}: "
                                                       f"{100 * h['share']:.0f} % of the squared error"})
    if not fields:
        raise ValueError("no point field of these snapshots lives on the mesh nodes")
    fv = field_view(mesh, fields, deform=deform, markers=markers,
                    max_faces=VIEW_MAX_FACES if max_faces is None else max_faces,
                    title=f"snapshot {index}", length_unit=mesh_unit,
                    notes=[f"{full}: the stored snapshot" + (f"; {note}" if note else "")])
    info = {"errors": errors, "deformation": dname, "regions": {k: int(len(v)) for k, v in regions.items()}, "full_order": full}
    return fv, info
