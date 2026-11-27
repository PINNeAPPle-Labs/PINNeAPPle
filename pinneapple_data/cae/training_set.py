"""From a structural run to a training set: the normalisation an ingestion pipeline owes, each step measured.

A run is not a training set. Five things have to happen between them, and every one of them changes the numbers, so
every one of them is reported rather than done silently:

* **deduplication** — an explicit solver writes a frame when the output interval elapses, and a short interval writes
  the same state twice. The repeated frames are dropped and counted (``dropped[].frames``);
* **decimation** — a surrogate rarely wants every node of a crash mesh. The node set is thinned, and the error that
  thinning introduces is *measured*: every dropped node's displacement is compared with the nearest kept node's, and
  the relative L2 of that difference is reported per field (``dropped[].decimation``). A decimation whose error is not
  stated is a silent change of the target;
* **unit conversion** — every run is brought to one system (SI) through ``pinneapple_data.physical_units``, so a
  campaign that mixed a ``mm-tonne-s`` deck with an SI one is comparable;
* **non-dimensionalisation** — per field, divided by one reference taken over the whole set, with the reference kept.
  The unit algebra comes from ``physical_units`` (the SI unit of the quantity), the reference is a measurement; a
  surrogate trained on dimensionless fields cannot silently apply a millimetre to a metre;
* **the split holds out a geometry family, not a frame** — frames of one impact are not independent samples: they share
  a mesh, a material and a design point. Splitting them at random leaks the answer across the split and reports a score
  that means nothing. Here the unit of assignment is the family (:meth:`StructuralRun.family`), and
  :func:`build_training_set` refuses to put one family in two places.

The part hierarchy survives all of it: each run keeps its parts with the element indices remapped to the thinned mesh,
so a part-aware model still gets its masks after normalisation.

    from pinneapple_data.cae.training_set import build_training_set
    ts = build_training_set([run_a, run_b, run_c], max_points=900, hold_out=("case_03",))
    print(ts.summary())        # what was dropped, the decimation error, the scales, the split
"""
from __future__ import annotations

import numpy as np

from .model import CellBlock, Mesh
from .structural import StructuralRun

__all__ = ["TrainingSet", "build_training_set", "decimate", "deduplicate", "normalise_units", "dimensionless"]

#: the fields whose reference scale is their own largest magnitude; a field not here is scaled by 1 (a mask, an id)
SCALED_FIELDS = ("displacement", "velocity", "von_mises", "plastic_strain", "contact_force")


class TrainingSet:
    """Normalised runs and the split that holds out whole geometry families.

    ``runs`` are the inputs after deduplication, decimation, unit conversion and non-dimensionalisation; ``scales``
    maps a field to the reference it was divided by (multiply by it to come back to SI); ``dropped`` is the log of
    every step that changed the data, and ``split`` names the families in each part.
    """
    def __init__(self, runs, split, scales, provenance=None, dropped=None, info=None):
        self.runs = list(runs)
        self.split = {k: list(v) for k, v in split.items()}
        self.scales = {k: float(v) for k, v in dict(scales).items()}
        self.provenance = dict(provenance or {})
        self.dropped = list(dropped or [])
        self.info = dict(info or {})

    # ------------------------------------------------------------------------------------- what the set is
    @property
    def families(self) -> list[str]:
        return [r.family() for r in self.runs]

    @property
    def n_samples(self) -> int:
        return sum(r.n_frames for r in self.runs)

    def part(self, name: str) -> list[StructuralRun]:
        """The runs of one side of the split (``"train"``/``"val"``/``"test"``)."""
        if name not in self.split:
            raise KeyError(f"{name!r} is not a side of the split; it has {sorted(self.split)}")
        wanted = set(self.split[name])
        return [r for r in self.runs if r.family() in wanted]

    def sample_index(self) -> np.ndarray:
        """``(S, 2)``: the run and the frame of every sample, in the order of :attr:`runs`."""
        rows = [(i, k) for i, r in enumerate(self.runs) for k in range(r.n_frames)]
        return np.array(rows, dtype=np.int64) if rows else np.zeros((0, 2), dtype=np.int64)

    def side_of(self, run: StructuralRun) -> str:
        fam = run.family()
        for side, families in self.split.items():
            if fam in families:
                return side
        return "unassigned"

    def table(self) -> dict[str, np.ndarray]:
        """One row per sample with the scalars a tabular model uses — the time, the split and the field extremes.

        A field surrogate wants the arrays; this is for the models that predict the quantities a design is judged by,
        and it is derived from the same normalised runs, so a table and a field never disagree about the data.
        """
        side_of_family = {fam: side for side, fams in self.split.items() for fam in fams}
        names = [n for n in SCALED_FIELDS if any(n in r.fields for r in self.runs)]
        rows: dict[str, list] = {"time": [], "run": [], "frame": []}
        for n in names:
            rows[f"max_{n}"] = []
        for i, r in enumerate(self.runs):
            for k in range(r.n_frames):
                rows["time"].append(float(r.times[k]))
                rows["run"].append(i)
                rows["frame"].append(k)
                for n in names:
                    a = r.frame_field(n, k)
                    rows[f"max_{n}"].append(float(np.nanmax(np.abs(a))) if a is not None and a.size else np.nan)
        run = np.asarray(rows.pop("run"), dtype=np.int64)
        out: dict[str, np.ndarray] = {"run": run, "frame": np.asarray(rows.pop("frame"), dtype=np.int64),
                                      "time": np.asarray(rows.pop("time"), dtype=np.float64),
                                      "family": np.array([self.runs[i].family() for i in run], dtype=object),
                                      "split": np.array([side_of_family.get(self.runs[i].family(), "unassigned")
                                                         for i in run], dtype=object)}
        out.update({k: np.asarray(v, dtype=np.float64) for k, v in rows.items()})
        return out

    def summary(self) -> dict:
        """What the normalisation did, in the numbers a report quotes."""
        return {"runs": len(self.runs), "samples": self.n_samples,
                "families": self.families,
                "frames_per_run": [r.n_frames for r in self.runs],
                "points_per_run": [r.n_points for r in self.runs],
                "cells_per_run": [r.n_cells for r in self.runs],
                "split": {k: len(v) for k, v in self.split.items()},
                "split_families": {k: list(v) for k, v in self.split.items()},
                "units": dict(self.runs[0].units) if self.runs else {},
                "scales": dict(self.scales), "dropped": list(self.dropped),
                "provenance": dict(self.provenance)}


# ------------------------------------------------------------------------------------------------- the steps
def deduplicate(run: StructuralRun) -> tuple[StructuralRun, dict | None]:
    """Drop frames that repeat the state before them; returns the run and the log of what was removed."""
    keep = [0]
    seen = {_frame_key(run.frames[0])}
    duplicates: list[int] = []
    for k in range(1, run.n_frames):
        key = _frame_key(run.frames[k])
        if key in seen:
            duplicates.append(k)
            continue
        seen.add(key)
        keep.append(k)
    if not duplicates:
        return run, None
    log = {"step": "deduplicate", "family": run.family(), "frames": len(duplicates), "kept": keep}
    return _subset_frames(run, keep), log


def decimate(run: StructuralRun, max_points: int, *, seed: int = 0,
             keep_cells: bool = True) -> tuple[StructuralRun, dict]:
    """Thin the mesh to a node budget and report the error the thinning introduces, per field.

    The elements are what is selected: they are ordered by a farthest-point spread over their centroids (deterministic
    given ``seed``) and taken while their corner nodes still fit the budget, so the kept mesh is a connected part of
    the body rather than a scattering of nodes no element can be built on. The kept nodes are the union of the kept
    elements' corners; the parts and every element array are re-indexed onto them, and the part hierarchy survives.

    The reported error is measured, not assumed: every **dropped** node's displacement is compared with its nearest
    kept node, and the relative L2 of that difference is what a surrogate inherits by not being told about the nodes
    that went. ``keep_cells=False`` drops the elements altogether and keeps a point cloud of ``max_points`` nodes —
    the choice a nodal model makes, stated instead of hidden.
    """
    n = run.n_points
    if max_points >= n:
        return run, {"step": "decimate", "family": run.family(), "kept_points": n, "dropped_points": 0,
                     "error": {}, "note": "the mesh already has no more nodes than the budget"}
    order = decimate_order(run, max_points, seed=seed) if keep_cells else None
    blocks_in = run.frames[0].blocks
    if keep_cells:
        taken, keep_cells_rows, cell_map = _select_cells(run, order, max_points)
        if not cell_map.size:
            raise ValueError(
                f"a budget of {max_points} nodes cannot hold one element of this {run.n_cells}-element mesh "
                f"(the smallest needs {min(b.conn.shape[1] for b in blocks_in)} nodes). Raise the budget, or pass "
                "keep_cells=False to accept a point-cloud run whose elemental fields are dropped")
        kept = np.nonzero(taken)[0]
        remap = np.full(n, -1, dtype=np.int64)
        remap[kept] = np.arange(len(kept), dtype=np.int64)
    else:
        kept = _farthest_point_sample(run.points, max_points, seed=seed)
        remap = np.full(n, -1, dtype=np.int64)
        remap[kept] = np.arange(len(kept), dtype=np.int64)
        keep_cells_rows, cell_map = [], np.zeros(0, dtype=np.int64)
    dropped_nodes = np.nonzero(remap < 0)[0]
    nearest = _nearest_indices(run.points, kept, dropped_nodes)
    error: dict[str, float] = {}
    for name in ("displacement", "velocity"):
        if name in run.fields:
            error[name] = _nearest_neighbour_error(run, name, dropped_nodes, nearest, kept)
    blocks = [CellBlock(b.type, remap[c]) for b, c in zip(blocks_in, keep_cells_rows, strict=True)] if keep_cells \
        else []
    frames = [Mesh(points=f.points[kept], blocks=[CellBlock(b.type, b.conn) for b in blocks],
                   point_data={k: np.asarray(v)[kept] for k, v in f.point_data.items()},
                   cell_data=({k: np.asarray(v)[cell_map] for k, v in f.cell_data.items()} if keep_cells else {}),
                   units=dict(f.units), source=dict(f.source)) for f in run.frames]
    parts = [_remap_part(p, cell_map) for p in run.parts] if keep_cells else []
    out = StructuralRun(frames=frames, times=run.times.copy(), parts=parts, units=dict(run.units),
                        provenance=dict(run.provenance),
                        eroded=(None if run.eroded is None or not keep_cells else run.eroded[:, cell_map]),
                        contact_force=None if run.contact_force is None else run.contact_force.copy(),
                        fixed_nodes=None if run.fixed_nodes is None else run.fixed_nodes[kept],
                        dropped=list(run.dropped), info=dict(run.info))
    log = {"step": "decimate", "family": run.family(), "kept_points": int(len(kept)),
           "dropped_points": int(n - len(kept)), "kept_cells": int(len(cell_map)),
           "dropped_cells": int(run.n_cells - len(cell_map)), "error": error,
           "cells": "kept, selected by a spread over their centroids" if keep_cells else "dropped (point-cloud run)",
           "how": "relative L2 of the dropped nodes' displacement against their nearest kept node"}
    return out, log


def decimate_order(run: StructuralRun, max_points: int, *, seed: int = 0) -> np.ndarray:
    """The order in which elements are offered to the budget: a farthest-point spread over their centroids."""
    centroids = _cell_centroids(run)
    return _farthest_point_order(centroids, seed=seed)


def _cell_centroids(run: StructuralRun) -> np.ndarray:
    """``(m, 3)`` centroid of every element, in the frames' own global cell order."""
    pts = run.points
    return np.concatenate([pts[b.conn].mean(axis=1) for b in run.frames[0].blocks]) if run.n_cells \
        else np.zeros((0, 3))


def _select_cells(run: StructuralRun, order: np.ndarray, max_points: int):
    """Take elements in ``order`` while their corners still fit ``max_points``: the kept node mask and rows."""
    blocks = run.frames[0].blocks
    conn = np.concatenate([b.conn for b in blocks]) if run.n_cells else np.zeros((0, 1), dtype=np.int64)
    taken = np.zeros(run.n_points, dtype=bool)
    chosen: list[int] = []
    for c in order:
        nodes = conn[c]
        extra = int((~taken[nodes]).sum())
        if extra and int(taken.sum()) + extra > max_points:
            continue
        taken[nodes] = True
        chosen.append(int(c))
    cell_map = np.array(sorted(chosen), dtype=np.int64)
    rows: list[np.ndarray] = []
    start = 0
    for b in blocks:
        stop = start + len(b.conn)
        rows.append(cell_map[(cell_map >= start) & (cell_map < stop)] - start)
        start = stop
    return taken, rows, cell_map


def normalise_units(run: StructuralRun, system: str = "SI") -> tuple[StructuralRun, dict]:
    """Bring a run to ``system`` and log where it came from (a no-op costs nothing and is still logged)."""
    src = dict(run.units)
    if src == {k: v for k, v in pu_units(system).items()}:
        return run, {"step": "units", "family": run.family(), "from": src, "to": dict(src), "changed": False}
    out = run.to_units(system)
    return out, {"step": "units", "family": run.family(), "from": src, "to": dict(out.units), "changed": True}


def dimensionless(run: StructuralRun, scales: dict[str, float]) -> StructuralRun:
    """The run with every scaled field divided by its reference, its unit recorded as ``1``, and the scales kept.

    The geometry is divided by the displacement's reference (both are lengths), so a frame still satisfies
    ``points == undeformed + displacement`` after the step. A field with no reference is left alone.
    """
    units = dict(run.units)
    frames: list[Mesh] = []
    for f in run.frames:
        point = {k: (np.asarray(v, dtype=np.float64) / scales[k]) if k in scales else np.asarray(v, dtype=np.float64)
                 for k, v in f.point_data.items()}
        cell = {k: (np.asarray(v, dtype=np.float64) / scales[k]) if k in scales else np.asarray(v, dtype=np.float64)
                for k, v in f.cell_data.items()}
        pts = f.points / scales["coordinates"] if "coordinates" in scales else f.points
        frames.append(Mesh(points=pts, blocks=[CellBlock(b.type, b.conn, b.ids) for b in f.blocks],
                           point_data=point, cell_data=cell, units=dict(units), source=dict(f.source)))
    for k in scales:
        if k in units:
            units[k] = "1"
    return StructuralRun(frames=frames, times=run.times.copy(), parts=list(run.parts), units=units,
                         provenance=dict(run.provenance),
                         eroded=None if run.eroded is None else run.eroded.copy(),
                         contact_force=None if run.contact_force is None
                         else (run.contact_force / scales["contact_force"]
                               if "contact_force" in scales else run.contact_force.copy()),
                         fixed_nodes=None if run.fixed_nodes is None else run.fixed_nodes.copy(),
                         dropped=list(run.dropped), info=dict(run.info))


def build_training_set(runs, *, max_points: int | None = None, seed: int = 0, hold_out=(), val_fraction: float = 0.2,
                       system: str = "SI", dedup: bool = True, make_dimensionless: bool = True,
                       keep_cells: bool = True, scales: dict[str, float] | None = None,
                       provenance: dict | None = None) -> TrainingSet:
    """The whole normalisation, in order, over a campaign of runs, with the split over whole families.

    ``hold_out`` names the families that go to ``test``; the remaining ones are ordered and the first ``val_fraction``
    of them (at least one when there are two or more) go to ``val``. The assignment is by family, never by frame, and a
    family that appears twice in the list is one geometry run twice — the split still keeps it on one side.
    """
    if not runs:
        raise ValueError("a training set needs at least one run")
    dropped: list[dict] = []
    prepared: list[StructuralRun] = []
    for r in runs:
        work = r
        if dedup:
            work, log = deduplicate(work)
            if log:
                dropped.append(log)
        work, log = normalise_units(work, system)
        dropped.append(log)
        if max_points is not None:
            work, log = decimate(work, max_points, seed=seed, keep_cells=keep_cells)
            dropped.append(log)
        prepared.append(work)
    families = sorted({r.family() for r in prepared})
    held = [f for f in families if f in set(hold_out)]
    rest = [f for f in families if f not in set(hold_out)]
    if not held and len(families) == 1:
        split = {name: families if name == "train" else [] for name in ("train", "val", "test")}
        split["train"] = families
        info = "one family: there is nothing to hold out, and the set says so"
    else:
        if held:
            test = held
            pool = rest
        else:
            n_test = max(1, int(round(val_fraction * len(families)))) if len(families) > 1 else 0
            test = families[-n_test:] if n_test else []
            pool = families[:-n_test] if n_test else families
        n_val = max(1, int(round(val_fraction * len(pool)))) if len(pool) > 1 else 0
        val = pool[:n_val] if n_val else []
        train = pool[n_val:] if n_val else pool
        split = {"train": train, "val": val, "test": test}
        info = ("the test families are the ones named in hold_out" if held
                else f"{len(test)} of {len(families)} families held out in sorted order")
    ref = {}
    for name in SCALED_FIELDS:
        vals = [float(np.nanmax(np.abs(r.field(name)))) for r in prepared
                if name in r.fields and r.field(name).size]
        if vals:
            peak = max(vals)
            ref[name] = peak if peak > 0 else 1.0
    if "displacement" in ref:
        ref["coordinates"] = ref["displacement"]      # geometry and displacement are both lengths: one scale for both
    if scales:
        ref.update({k: float(v) for k, v in scales.items()})
    final = [dimensionless(r, ref) for r in prepared] if make_dimensionless else prepared
    prov = dict(provenance or {})
    prov.setdefault("step", "build_training_set")
    prov["inputs"] = [r.provenance.get("data_hash") or r.family() for r in prepared]
    return TrainingSet(final, split, ref, provenance=prov, dropped=dropped,
                       info={"split_rule": info, "points_budget": max_points, "seed": seed,
                             "canonical_system": system})


# ----------------------------------------------------------------------------------------------- small helpers
def pu_units(system: str) -> dict[str, str]:
    """The field → unit map of a unit system, as ``physical_units`` states it for each quantity."""
    from .structural import UNIT_SYSTEMS

    if system not in UNIT_SYSTEMS:
        raise ValueError(f"unknown unit system {system!r}: known are {sorted(UNIT_SYSTEMS)}")
    return UNIT_SYSTEMS[system]


def _frame_key(mesh: Mesh) -> bytes:
    """A digest of one frame's fields — two frames with the same key are the same state."""
    import hashlib

    h = hashlib.sha256()
    for store in (mesh.point_data, mesh.cell_data):
        for name in sorted(store):
            h.update(name.encode())
            h.update(np.ascontiguousarray(np.asarray(store[name], dtype=np.float64)).tobytes())
    return h.digest()


def _subset_frames(run: StructuralRun, keep: list[int]) -> StructuralRun:
    """The run restricted to the frames in ``keep``."""
    return StructuralRun(frames=[run.frames[k] for k in keep], times=run.times[keep], parts=list(run.parts),
                         units=dict(run.units), provenance=dict(run.provenance),
                         eroded=None if run.eroded is None else run.eroded[keep],
                         contact_force=None if run.contact_force is None else run.contact_force[keep],
                         fixed_nodes=None if run.fixed_nodes is None else run.fixed_nodes.copy(),
                         dropped=list(run.dropped), info=dict(run.info))


def _farthest_point_order(points: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Every point, in the order a farthest-point sample takes it (the selection order, not sorted)."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    start = int(np.random.default_rng(seed).integers(0, n))
    chosen = np.empty(n, dtype=np.int64)
    chosen[0] = start
    d = np.linalg.norm(points - points[start], axis=1)
    for i in range(1, n):
        nxt = int(np.argmax(d))
        chosen[i] = nxt
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return chosen


def _farthest_point_sample(points: np.ndarray, k: int, *, seed: int = 0) -> np.ndarray:
    """``k`` node indices spread over the geometry: greedy max-min distance from a seeded start."""
    n = len(points)
    return np.sort(_farthest_point_order(points, seed=seed)[: int(min(max(1, k), n))])


def _nearest_indices(points: np.ndarray, keep: np.ndarray, dropped: np.ndarray) -> np.ndarray:
    """For every dropped node, the nearest kept node — the neighbour a decimated run has to stand in with."""
    if dropped.size == 0:
        return np.zeros(0, dtype=np.int64)
    best = np.zeros(len(dropped), dtype=np.int64)
    best_d = np.full(len(dropped), np.inf)
    for start in range(0, len(keep), 256):               # in blocks, so a 3-D crash mesh does not become one array
        block = points[keep[start:start + 256]]
        diff = points[dropped][:, None, :] - block[None, :, :]
        d = np.einsum("ijk,ijk->ij", diff, diff)
        j = np.argmin(d, axis=1)
        dj = d[np.arange(len(d)), j]
        closer = dj < best_d
        best[closer] = start + j[closer]
        best_d[closer] = dj[closer]
    return best


def _nearest_neighbour_error(run: StructuralRun, name: str, dropped: np.ndarray, nearest: np.ndarray,
                             keep: np.ndarray) -> float:
    """Relative L2 of a field at the dropped nodes against their nearest kept node — the error decimation introduces."""
    if dropped.size == 0:
        return 0.0
    field = run.field(name)
    denom = float(np.max(np.abs(field))) or 1.0
    err = 0.0
    scale = 0.0
    for f in field:
        a = np.asarray(f)[dropped]
        b = np.asarray(f)[keep][nearest]
        err = max(err, float(np.linalg.norm(a - b)))
        scale = max(scale, float(np.linalg.norm(a)))
    return err / (scale or denom)


def _remap_part(part, cell_map: np.ndarray):
    """A part whose element indices follow the kept cells (a part can lose every cell and still exist)."""
    from .structural import Part

    old = np.asarray(part.elements, dtype=np.int64)
    pos = np.clip(np.searchsorted(cell_map, old), 0, max(0, len(cell_map) - 1))
    ok = cell_map[pos] == old if len(cell_map) else np.zeros(len(old), dtype=bool)
    return Part(id=part.id, name=part.name, elements=np.asarray(pos[ok], dtype=np.int64), kind=part.kind,
                thickness=part.thickness, properties=dict(part.properties))
