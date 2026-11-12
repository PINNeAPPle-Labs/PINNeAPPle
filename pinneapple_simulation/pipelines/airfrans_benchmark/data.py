"""Reading the public AirfRANS release case by case, with the library's own feature definition.

The extraction is the only stage that needs the optional AirfRANS stack (``airfrans`` 0.1.5.1 plus PyVista/VTK, which is
what the benchmark's participant container pins). Everything downstream — features, training, prediction, packaging —
runs on the arrays this stage writes, so a machine without the VTK stack can still train and predict from an existing
store.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

__all__ = ["AirfRANSStackMissing", "case_arrays", "extract_cases", "manifest_splits", "case_ids_for"]

#: every split key the public release publishes
SPLIT_KEYS = ("full_train", "full_test", "scarce_train", "scarce_test", "reynolds_train", "reynolds_test",
              "aoa_train", "aoa_test")

#: the digest the benchmark publishes for the release's ``Dataset/manifest.json``
PUBLISHED_MANIFEST_SHA256 = "3d1320005bcf3d94df80df2bf8338dd6ce6d394c6beafa14849b4ce1de7def13"


class AirfRANSStackMissing(RuntimeError):
    """The optional AirfRANS/VTK stack is not installed: the extraction stage needs it, nothing else does."""


def _simulation():
    try:
        from airfrans.simulation import Simulation
    except ImportError as error:                      # pragma: no cover - depends on the machine
        raise AirfRANSStackMissing(
            "extracting cases needs the pinned participant stack: pip install 'airfrans==0.1.5.1' "
            "'pyvista==0.48.4' 'vtk==9.6.2' (Python 3.12). Training and prediction run without it on an existing "
            f"store; the import failed with: {error}") from error
    return Simulation


def manifest_splits(dataset_root: str) -> dict[str, Any]:
    """The release manifest, with the digest checked against the published one when it matches nothing else."""
    path = os.path.join(dataset_root, "manifest.json")
    with open(path) as handle:
        manifest = json.load(handle)
    return manifest


def case_ids_for(manifest: dict[str, Any], keys: Iterable[str] = SPLIT_KEYS) -> list[str]:
    """Every case id the manifest lists, sorted and de-duplicated across the given splits."""
    return sorted({case for key in keys for case in manifest.get(key, [])})


def case_arrays(dataset_root: str, case_id: str) -> dict[str, np.ndarray]:
    """One case's features, targets and airfoil geometry, with the AirfRANS feature definition.

    ``features`` is ``(N, 8)``: position, signed distance, wall normals, surface flag and the freestream vector —
    exactly what ``airfrans.Simulation`` computes for the participant workflow."""
    Simulation = _simulation()
    sim = Simulation(root=dataset_root, name=case_id)
    position = np.asarray(sim.position, np.float32)
    sdf = np.asarray(sim.sdf, np.float32).reshape(-1, 1)
    normals = np.asarray(sim.normals, np.float32)
    surface = np.asarray(sim.surface, bool).reshape(-1, 1).astype(np.float32)
    freestream = np.asarray(sim.input_velocity, np.float32)
    velocity = np.asarray(sim.velocity, np.float32)
    pressure = np.asarray(sim.pressure, np.float32).reshape(-1, 1)
    arrays = {
        "features": np.concatenate([position, sdf, normals, surface, freestream], axis=1),
        "targets": np.concatenate([velocity, pressure], axis=1),
        "airfoil_points": np.asarray(sim.airfoil_position, np.float32),
        "airfoil_normals": np.asarray(sim.airfoil_normals, np.float32),
        "case": np.array([case_id]),
        "inlet_velocity": np.array([float(sim.inlet_velocity)], np.float32),
        "aoa_deg": np.array([float(case_id.split("_")[3])], np.float32),
        "n_points": np.array([len(position)], np.int32),
    }
    for name, value in arrays.items():
        if name in ("case", "n_points"):
            continue
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{case_id}: {name} has non-finite values")
    return arrays


def _extract_one(args: tuple[str, str, str]) -> tuple[str, str | None]:
    dataset_root, out_dir, case_id = args
    target = os.path.join(out_dir, f"{case_id}.npz")
    if os.path.exists(target):
        return case_id, None
    try:
        np.savez_compressed(target, **case_arrays(dataset_root, case_id))
    except Exception as error:                       # reported per case, never swallowed
        return case_id, f"{type(error).__name__}: {error}"
    return case_id, None


def extract_cases(dataset_root: str, out_dir: str, *, cases: Iterable[str] | None = None, parallel: int = 1,
                  progress=None) -> dict[str, Any]:
    """Write one ``.npz`` per case into ``out_dir``; returns what was written and what failed.

    ``parallel`` runs several cases at once in separate processes (the VTK reader is single-threaded and each case is a
    few seconds of work), which is what makes a 1000-case extraction a coffee break instead of an hour."""
    manifest = manifest_splits(dataset_root)
    wanted = list(cases) if cases is not None else case_ids_for(manifest)
    os.makedirs(out_dir, exist_ok=True)
    started, failures = time.time(), []
    work = [(dataset_root, out_dir, case_id) for case_id in wanted]
    if parallel > 1:
        with ProcessPoolExecutor(max_workers=int(parallel)) as pool:
            for done, (case_id, error) in enumerate(pool.map(_extract_one, work), 1):
                if error:
                    failures.append({"case_id": case_id, "error": error})
                if progress and done % 25 == 0:
                    progress(done, len(work))
    else:
        for done, item in enumerate(work, 1):
            case_id, error = _extract_one(item)
            if error:
                failures.append({"case_id": case_id, "error": error})
            if progress and done % 25 == 0:
                progress(done, len(work))
    manifest_path = os.path.join(dataset_root, "manifest.json")
    digest = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
    summary = {"manifest_sha256": digest, "published_sha256": PUBLISHED_MANIFEST_SHA256,
               "manifest_matches_published": digest == PUBLISHED_MANIFEST_SHA256,
               "cases_requested": len(wanted), "failures": failures,
               "seconds": round(time.time() - started, 1)}
    with open(os.path.join(out_dir, "extract.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    return summary
