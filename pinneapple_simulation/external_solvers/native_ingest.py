"""Native solver output ingestion, with explicit provenance and no hidden interpolation.

OpenFOAM: requires the official foamToVTK utility to have generated an
unstructured .vtk file (legacy format). CalculiX: uses a meshio-supported
FRD file if the installed meshio version supports that reader; otherwise
fails explicitly. FEniCSx: consumes a solver-written XDMF/VTU file.
"""
from __future__ import annotations

from pathlib import Path
import math
import shutil
import subprocess
import time

from .mesh_import import read_meshio
from .mesh_export import MeshDataset


class NativeOutputError(RuntimeError):
    pass


def _file(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_file():
        raise NativeOutputError(f"Solver output not found: {p}")
    return p


def ingest_native_output(
    backend: str, output_path: str | Path, *,
    field_units: dict[str, str],
    coordinate_scale_to_m: float = 1.0,
    provenance: dict[str, str] | None = None,
) -> MeshDataset:
    """Ingest a solver-exported mesh/field file, never raw case directories."""
    if backend not in ("openfoam", "calculix", "fenicsx"):
        raise ValueError(f"Unsupported solver backend: {backend}")
    p = _file(output_path)
    accepted = {
        "openfoam": {".vtk", ".vtu"},
        "calculix": {".frd", ".vtk", ".vtu"},
        "fenicsx": {".xdmf", ".xmf", ".vtu", ".vtk"},
    }
    if p.suffix.lower() not in accepted[backend]:
        raise NativeOutputError(
            f"{backend} requires a supported exported file ({sorted(accepted[backend])}); "
            "native case directories cannot be ingested as meshes"
        )
    try:
        return read_meshio(
            p, coordinate_scale_to_m=coordinate_scale_to_m,
            field_units=field_units,
            provenance={**(provenance or {}), "backend": backend,
                        "source_file": str(p)},
        )
    except Exception as exc:
        raise NativeOutputError(
            f"Could not decode {backend} output {p.name}: {exc}; "
            "convert to a supported VTK/VTU mesh with explicit units if necessary"
        ) from exc


def convert_openfoam_case(
    case_dir: str | Path, *, timeout_s: float = 600,
    latest_time: bool = True,
) -> list[Path]:
    """Run the official foamToVTK exporter on a trusted existing case.

    Call only from an isolated solver worker; not from an HTTP request.
    """
    case = Path(case_dir).resolve()
    if not (case / "system" / "controlDict").is_file():
        raise NativeOutputError("OpenFOAM case missing system/controlDict")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    exe = shutil.which("foamToVTK")
    if not exe:
        raise NativeOutputError("foamToVTK executable unavailable on PATH")
    args = [exe, "-case", str(case)]
    if latest_time:
        args.append("-latestTime")
    started_at = time.time()
    try:
        result = subprocess.run(args, cwd=case, capture_output=True,
                                text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NativeOutputError(f"foamToVTK failed: {exc}") from exc
    if result.returncode:
        raise NativeOutputError(f"foamToVTK exit {result.returncode}: {result.stderr[-2000:]}")
    files = sorted(p for p in (case / "VTK").rglob("*.vtk")
                   if p.stat().st_mtime >= started_at - 2) if (case / "VTK").is_dir() else []
    if not files:
        raise NativeOutputError("foamToVTK finished but produced no .vtk files")
    return files
