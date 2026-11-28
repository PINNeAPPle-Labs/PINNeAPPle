"""Stage 3 of :mod:`pinneapple_neural.lpm_pipeline`: a design space on disk — one ``.npz`` per design, and a manifest that says where every field came from.

The pipeline reads a dataset four times over its life (to fit, to extend, to audit, to publish), so the file layout is
the contract: ``designs/<label>.npz`` holds the points, the normals, the query, the normalised operating point and the
target, and ``manifest.json`` holds what the set *is* — the family and its reference, the parameter bounds used for
normalisation, the seeds, the neighbourhood radii the local features were cached at, and the SHA-256 of every file.

:func:`extend_dataset` is the second round: the campaign fits on a low-discrepancy draw, asks the model where its
confidence is weakest, and writes the bodies it gets back into the same manifest marked as **refined** — one list of
what the model could have seen, and one place to record that a body was simulated for a reason.

Nothing here imports the family registry at module level: the registry and the drawing live in
:mod:`pinneapple_neural.lpm_pipeline`, which re-exports this module, and the two imports are lazy so neither one has to
be imported first.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..lpm import Design

if TYPE_CHECKING:                                       # the annotations are strings; this is for the readers of them
    from ..lpm_pipeline import DesignSpec

#: the two schemas this module writes; re-exported by the pipeline so a caller has one place to import from
DATASET_SCHEMA = "pinneapple.lpm.dataset/1"
PROGRESS_SCHEMA = "pinneapple.lpm.progress/1"

__all__ = ["DATASET_SCHEMA", "PROGRESS_SCHEMA", "extend_dataset", "generate_dataset", "load_dataset"]


# ---------------------------------------------------------------------------------------------------------- stage 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    """Write a progress file so a reader sees either the previous one or this one, never half of it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _progress(path: Path, *, kind: str, family: str, started_at: str, started: float, done: int, total: int,
              finished: bool = False, resumed_from: int = 0, **extra: Any) -> None:
    """One progress record: what this stage has finished, how fast, and how long it still needs.

    ``kind`` is ``dataset`` (done/total designs) or ``fit`` (done/total epochs); both carry the same ``done``,
    ``total``, ``per_s`` and ``eta_s`` so a reader renders the two stages with one shape.
    """
    elapsed = max(time.monotonic() - started, 1e-9)
    done_now = done - resumed_from              # a resumed fit starts part-way, so the rate is what *this* run did
    rate = done_now / elapsed if done_now else 0.0
    payload = {"schema": PROGRESS_SCHEMA, "kind": kind, "family": family,
               "started_at": started_at, "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
               "done": int(done), "total": int(total), "finished": bool(finished),
               "elapsed_s": round(elapsed, 1), "per_s": round(rate, 4),
               "eta_s": round((total - done) / rate, 1) if rate > 0 and total > done else 0.0}
    payload.update(extra)
    _write_progress(path, payload)


def _write_design(family: Any, spec: Any, designs_dir: Path, radii: tuple[float, ...],
                  cache_local: bool) -> tuple[dict[str, Any], int, int]:
    """Build one design, write it as an ``.npz``, and return its manifest record with the value and point counts."""
    from ..lpm_pipeline import build  # the second stage, which is where a design is made

    design = build(family, spec)
    if cache_local:
        from ..lpm import multiscale_neighbourhood
        # once per design, at the moment the geometry is fixed: the fit reads these instead of rebuilding the
        # same distance matrices on every epoch (the geometry never changes during a fit)
        design.local = multiscale_neighbourhood(design.query, design.points, design.normals, radii)
    name = spec.label.replace(":", "_") + ".npz"
    path = designs_dir / name
    np.savez(path,
             points=design.points.numpy(), normals=design.normals.numpy(), query=design.query.numpy(),
             params=design.params.numpy(), target=design.target.numpy(),
             **({"local": design.local.numpy()} if design.local is not None else {}))
    record = {**spec.as_dict(), "file": f"designs/{name}", "sha256": _sha256(path),
              "n_points": int(design.points.shape[0]), "n_fields": int(design.target.shape[1])}
    return record, int(design.target.numel()), int(design.points.shape[0])


def generate_dataset(family: Any, out_dir: str | Path, *, n_geometries: int = 32,
                     conditions_per_geometry: int = 3, seed: int = 0, verbose: bool = False,
                     progress_every_s: float = 0.0, radii: tuple[float, ...] = (0.1, 0.4, 1.2),
                     cache_local: bool = True, specs: list[DesignSpec] | None = None) -> dict[str, Any]:
    """Run the family's reference on a drawn design space and write it as a dataset.

    The dataset is what the next stages read: one ``.npz`` per design (points, normals, query, params, target) and a
    ``manifest.json`` carrying the family, its reference, the parameter bounds, the seeds and the SHA-256 of every
    file — so a run can be reproduced and an artefact can be identified later.

    ``progress_every_s`` writes ``progress.json`` in ``out_dir`` at that interval — designs done, rate and the time
    still needed — which is the only thing on disk while a solver-backed family (the 3-D solids take seconds per
    design) is being built. ``0``, the default, writes nothing at all.

    ``specs`` replaces the drawn space with designs the caller already has, which is what the second round of a
    refinement needs (:func:`extend_dataset`); without it the space is :func:`draw`n from the family's own ranges.
    """
    out = Path(out_dir)
    designs_dir = out / "designs"
    designs_dir.mkdir(parents=True, exist_ok=True)
    # a second run into the same directory must leave the dataset it describes, not the union of both: the manifest
    # is rewritten every time, and a file it does not list is a leftover from a larger run
    for stale in designs_dir.glob("*.npz"):
        stale.unlink()
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    progress_path = out / "progress.json"
    if progress_every_s > 0:                            # never leave a previous run's record beside this one
        progress_path.unlink(missing_ok=True)
    last_progress = started
    if specs is None:
        from ..lpm_pipeline import draw  # stage 1, in the module that owns the family registry
        specs = draw(family, n_geometries, conditions_per_geometry, seed)
    specs = list(specs)
    records, fields_total, points_total = [], 0, 0
    per_geometry: dict[int, int] = {}
    for position, spec in enumerate(specs, start=1):
        record, values, points = _write_design(family, spec, designs_dir, radii, cache_local)
        records.append(record)
        fields_total += values
        points_total += points
        per_geometry[spec.geometry_id] = per_geometry.get(spec.geometry_id, 0) + 1
        if verbose:
            print(f"  {spec.label}: {record['n_points']} points, {record['n_fields']} fields")
        if progress_every_s > 0 and (time.monotonic() - last_progress >= progress_every_s or position == len(specs)):
            _progress(progress_path, kind="dataset", family=family.name, started_at=started_at, started=started,
                      done=position, total=len(specs), label=spec.label)
            last_progress = time.monotonic()
    manifest = {
        "schema": DATASET_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "family": family.describe(),
        "seed": int(seed),
        "n_geometries": len(per_geometry),
        "conditions_per_geometry": int(conditions_per_geometry),
        "n_designs": len(records),
        "n_points": points_total,
        "n_values": fields_total,
        "seconds": round(time.monotonic() - started, 3),
        "designs": records,
        "local": {"radii": list(radii), "cached": bool(cache_local)},
        "note": ("every field in this dataset was produced by the family's own reference (see "
                 "family.reference); no solver was run unless that reference says implements=solver"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if progress_every_s > 0:
        _progress(progress_path, kind="dataset", family=family.name, started_at=started_at, started=started,
                  done=len(records), total=len(records), finished=True)
    return manifest


def extend_dataset(family: Any, dataset_dir: str | Path, specs: list[Any], *,
                   radii: tuple[float, ...] | None = None, cache_local: bool = True,
                   refinement: dict[str, Any] | None = None) -> dict[str, Any]:
    """Add designs to a dataset already on disk, and mark the geometries they introduce as **refined**.

    This is the second half of stage 1: the campaign fits on a low-discrepancy draw, asks the model where its
    confidence is weakest, and comes back with a handful of bodies to simulate
    (:func:`~pinneapple_neural.lpm_stages.space.refine`). Writing them into the existing dataset rather than a second
    one keeps one manifest — one list of what the model could have seen — and keeps the split's meaning.

    ``refinement`` is the round that produced them (round number, the scores it was chosen from, the threshold). It
    goes into the manifest, and every geometry id the round introduces goes into ``refined_geometries``, which is what
    :func:`_split_by_geometry` reads to keep them out of the held-out set: a body was chosen *because* the model was
    unsure about it, so scoring the model on it would report the choice rather than the model.

    A design already in the dataset (same label) is skipped rather than written twice.
    """
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{root} has no manifest.json: extend a dataset that generate_dataset wrote")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    designs_dir = root / "designs"
    designs_dir.mkdir(parents=True, exist_ok=True)
    cached = tuple((manifest.get("local") or {}).get("radii") or ())
    radii = tuple(radii or cached or (0.1, 0.4, 0.9))
    if cached and tuple(radii) != cached:
        raise ValueError(f"{root} caches its neighbourhood features for radii {cached}; extending it with {radii} "
                         f"would leave two kinds of design in one manifest")

    seen = {record["label"] for record in manifest["designs"]}
    added, skipped, values_total, points_total = [], [], 0, 0
    for spec in specs:
        if spec.label in seen:
            skipped.append(spec.label)
            continue
        record, values, points = _write_design(family, spec, designs_dir, radii, cache_local)
        added.append(record)
        values_total += values
        points_total += points
        seen.add(spec.label)
    if not added:
        return manifest

    rounds = list(manifest.get("refinement") or [])
    if refinement:
        rounds.append({**refinement, "designs": [r["label"] for r in added],
                       "geometry_ids": sorted({r["geometry_id"] for r in added})})
    written = {record["label"] for record in added}
    refined = sorted(set(manifest.get("refined_geometries") or ()) |
                     {spec.geometry_id for spec in specs if spec.label in written})
    geometries = {r["geometry_id"] for r in manifest["designs"]} | {r["geometry_id"] for r in added}
    manifest |= {
        "extended_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_geometries": len(geometries),
        "n_designs": len(manifest["designs"]) + len(added),
        "n_points": int(manifest.get("n_points") or 0) + points_total,
        "n_values": int(manifest.get("n_values") or 0) + values_total,
        "designs": list(manifest["designs"]) + added,
        "refined_geometries": refined,
        "refinement": rounds,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_dataset(dataset_dir: str | Path) -> tuple[list[Design], dict[str, Any]]:
    """Read a dataset written by :func:`generate_dataset` back into :class:`~.lpm.Design` objects."""
    root = Path(dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    designs = []
    for record in manifest["designs"]:
        with np.load(root / record["file"]) as data:
            designs.append(Design(
                points=torch.tensor(data["points"], dtype=torch.float32),
                normals=torch.tensor(data["normals"], dtype=torch.float32),
                query=torch.tensor(data["query"], dtype=torch.float32),
                params=torch.tensor(data["params"], dtype=torch.float32),
                target=torch.tensor(data["target"], dtype=torch.float32),
                local=torch.tensor(data["local"], dtype=torch.float32) if "local" in data else None,
            ))
    return designs, manifest
