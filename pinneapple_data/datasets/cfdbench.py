"""CFDBench: the 2-D incompressible cases, as written on HuggingFace, into the universal schema.

CFDBench (Luo, Chen & Zhang, 2023; arXiv:2310.05963) is four families of incompressible flow -- lid-driven
``cavity``, flow through a ``tube``, flow over a ``dam`` and flow around a ``cylinder`` -- each with three subsets that
vary the boundary conditions (``bc``), the geometry (``geo``) or the fluid properties (``prop``). The archive on
HuggingFace published the fields interpolated to a 64 x 64 grid, which is the form every paper on it uses.

The layout is the project's own, read from its loader (``src/dataset/base.py`` and ``src/dataset/*.py``) rather than
guessed:

    cavity/bc/case0000/u.npy      # (time, height, width), float
    cavity/bc/case0000/v.npy      # (time, height, width)
    cavity/bc/case0000/mask.npy   # (height, width), 1 on the boundary and 0 in the interior -- where it exists
    cavity/bc/case0000/case.json  # the case's parameters (viscosity, density, boundary speeds, geometry extents)

Two things this reader refuses to invent:

* **the units.** Neither the archive nor ``case.json`` states them, so every field is ``unknown`` unless the caller
  passes ``units={...}``. That is a verdict the ``units`` gate reports, not a conversion decided here.
* **the coordinates.** The interpolated grid carries no coordinate arrays. Where ``case.json`` gives the domain's
  extent (the cylinder family gives ``x_min``/``x_max``/``y_min``/``y_max``) the cell centres are built from it and
  the length unit is still ``unknown``; where it does not, the axis is the cell index and dimensionless. Both cases
  say which one they are in the bundle's metadata.

    >>> from pinneapple_data.datasets import cfdbench
    >>> case = next(cfdbench.open_cases("data/CFDBench", problem="cavity"))
    >>> bundle = cfdbench.load_bundle(case, dt=0.1)          # the cavity's frames are 0.1 s apart
"""
from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["PROBLEMS", "SUBSETS", "HUGGINGFACE", "ARCHIVE_URL", "LICENSE", "Case", "open_cases", "load_bundle",
           "download"]

#: The four problems the archive publishes, with what each one is.
PROBLEMS: dict[str, str] = {
    "cavity": "lid-driven cavity",
    "tube": "flow through a tube",
    "dam": "flow over a dam",
    "cylinder": "flow around a cylinder",
}

#: The three subsets every problem is split into: what varies across the cases.
SUBSETS: dict[str, str] = {"bc": "boundary conditions", "geo": "geometry", "prop": "fluid properties"}

HUGGINGFACE = "chen-yingfa/CFDBench"
ARCHIVE_URL = f"https://huggingface.co/datasets/{HUGGINGFACE}/resolve/main"

#: Read off the dataset card's ``license`` field, which is what covers the data (the repository's own licence is a
#: separate file and covers the code).
LICENSE = "Apache-2.0"


@dataclass(frozen=True)
class Case:
    """One case on disk: its fields, its mask and the parameters ``case.json`` records."""

    problem: str
    subset: str
    name: str
    path: Path

    def arrays(self) -> dict[str, np.ndarray]:
        """``u``, ``v`` and, where the case ships one, ``mask`` -- loaded here, not at construction."""
        out = {"u": np.load(self.path / "u.npy"), "v": np.load(self.path / "v.npy")}
        mask = self.path / "mask.npy"
        if mask.exists():
            out["mask"] = np.load(mask)
        return out

    def params(self) -> dict[str, float]:
        """The numeric parameters of ``case.json``; a key that is not a number is left to the raw metadata."""
        raw = self.raw_params()
        return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

    def raw_params(self) -> dict[str, Any]:
        path = self.path / "case.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def summary(self) -> str:
        return f"{self.problem}/{self.subset}/{self.name}"


def _is_case(path: Path) -> bool:
    return (path / "u.npy").exists() and (path / "v.npy").exists()


def _has_cases(path: Path) -> bool:
    return any(_is_case(child) for child in path.iterdir() if child.is_dir())


def open_cases(root: str | os.PathLike[str], *, problem: str | None = None,
               subset: str | None = None) -> Iterator[Case]:
    """Every case under ``root``, sorted: ``root/<problem>/<subset>/caseNNNN``, or ``root/<subset>/caseNNNN`` when
    ``root`` is one unpacked problem -- which is how the archives extract (``tube.zip`` unpacks to ``bc/``, ``geo/``,
    ``prop/`` at the top level, with no ``tube/`` above them)."""
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"{base} is not a directory; unzip the archive from {ARCHIVE_URL} first")
    for top in sorted(p for p in base.iterdir() if p.is_dir()):
        if _has_cases(top):                                  # base is one problem: top is a subset
            if subset and top.name != subset:
                continue
            for case_dir in sorted(p for p in top.iterdir() if p.is_dir() and _is_case(p)):
                yield Case(problem=base.name, subset=top.name, name=case_dir.name, path=case_dir)
            continue
        if problem and top.name != problem:                  # base holds several problems: top is one
            continue
        for sub in sorted(p for p in top.iterdir() if p.is_dir() and _has_cases(p)):
            if subset and sub.name != subset:
                continue
            for case_dir in sorted(p for p in sub.iterdir() if p.is_dir() and _is_case(p)):
                yield Case(problem=top.name, subset=sub.name, name=case_dir.name, path=case_dir)


def _coordinates(params: Mapping[str, float], n: int, axis: str) -> tuple[np.ndarray, str, str]:
    """The cell centres along one axis, its unit, and where the coordinate came from.

    ``case.json`` states the domain in two shapes: an explicit extent (``x_min``/``x_max``, which the cylinder
    family writes after reconstructing the grid around the obstacle) or just its size (``width``/``height``, which
    the tube, dam and cavity cases write). Both are the case's own length unit, which the archive does not name."""
    low, high = params.get(f"{axis}_min"), params.get(f"{axis}_max")
    origin = f"cell centres from case.json {axis}_min/{axis}_max"
    if low is None or high is None or high <= low:
        low = 0.0
        high = params.get("width" if axis == "x" else "height")
        origin = f"cell centres from case.json 0..{'width' if axis == 'x' else 'height'}"
    if high is not None and high > low:
        step = (high - low) / n
        return low + step * (np.arange(n) + 0.5), "unknown", origin
    return np.arange(n, dtype=np.float64), "1", "cell index of the interpolated grid (no extent in the case)"


def load_bundle(case: Case | str | os.PathLike[str], *, license: str = LICENSE, dt: float | None = None,
                units: Mapping[str, str] | None = None, source: str = "") -> Any:
    """One case as a bundle: ``u``/``v`` as a velocity vector, the mask as a field, ``case.json`` as parameters.

    ``dt`` (seconds per frame) turns the frame index into a time axis; without it the axis is the index. The cavity
    family's frames are 0.1 s apart, which the project's loader states and this module does not assume."""
    from ..schema import AxisSpec, Field, Physics, PhysicsDataBundle
    from ..schema.bundle import file_refs
    from ..schema.models import Parameter

    if not isinstance(case, Case):
        path = Path(case)
        parts = path.parts
        case = Case(problem=parts[-3] if len(parts) >= 3 else "", subset=parts[-2] if len(parts) >= 2 else "",
                    name=path.name, path=path)

    arrays = case.arrays()
    u, v = arrays["u"], arrays["v"]
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError(f"{case.summary()}: u {u.shape} and v {v.shape} are not two equal (time, height, width) "
                         "arrays -- the layout this reader follows is the one the project's loader reads")
    n_t, ny, nx = u.shape
    params = case.params()

    t = np.arange(n_t, dtype=np.float64) * float(dt) if dt else np.arange(n_t, dtype=np.float64)
    y, y_unit, y_origin = _coordinates(params, ny, "y")
    x, x_unit, x_origin = _coordinates(params, nx, "x")

    override = dict(units or {})
    y_unit, x_unit = override.get("y", y_unit), override.get("x", x_unit)   # the axis units are the caller's to declare
    velocity = np.stack([u, v], axis=-1)                       # (time, height, width, 2)
    fields = [Field.from_array("velocity", velocity, unit=override.get("velocity", "unknown"), axes=("t", "y", "x"),
                               components=["u", "v"],
                               description=f"{PROBLEMS.get(case.problem, case.problem)}: velocity, components u and v")]
    store: dict[str, np.ndarray] = {"axis:t": t, "axis:y": y, "axis:x": x, "field:velocity": velocity}
    if "mask" in arrays:
        mask = np.asarray(arrays["mask"])
        stack = np.broadcast_to(mask, (n_t, ny, nx))
        store["field:mask"] = np.array(stack)
        fields.append(Field.from_array("mask", np.array(stack), unit=override.get("mask", "1"), axes=("t", "y", "x"),
                                       description="1 on the domain boundary, 0 in the interior (the project's "
                                                   "own convention)"))

    parameters = [Parameter(name=key, value=value, unit=override.get(key, "unknown"), source="case.json")
                  for key, value in sorted(params.items())]
    return PhysicsDataBundle.from_arrays(
        store, fields=fields,
        axes=[AxisSpec("t", "s" if dt else "1"), AxisSpec("y", y_unit), AxisSpec("x", x_unit)],
        physics=Physics(parameters=parameters,
                        notes="2-D incompressible flow as the archive interpolated it to a grid; the governing "
                              "equations and the boundary conditions are the case's, described by case.json"),
        source=source or f"CFDBench {case.summary()}", title=f"CFDBench {case.summary()}",
        license=license, kind="dataset", adapter="pinneapple_data.datasets.cfdbench", adapter_version="1",
        inputs=file_refs([str(p) for p in sorted(case.path.glob("*.npy")) + sorted(case.path.glob("case.json"))]),
        citation="Luo, Chen & Zhang (2023), CFDBench: A Large-Scale Benchmark for Machine Learning Methods in Fluid "
                 "Dynamics, arXiv:2310.05963",
        url=ARCHIVE_URL,
        metadata={"cfdbench": {"problem": case.problem, "subset": case.subset, "case": case.name,
                               "shape": [n_t, ny, nx], "dt": dt,
                               "coordinates": {"x": x_origin, "y": y_origin},
                               "case_json": case.raw_params(),
                               "units_note": "the archive does not state the units of its fields: pass units={...} "
                                             "to declare them"}})


def download(problem: str, root: str | os.PathLike[str] = "data/cfdbench", *, timeout: float = 600.0) -> Path:
    """Fetch and unzip one problem's archive into ``root``; returns the directory the cases are under.

    The files are on HuggingFace (``cavity.zip``, ``dam.zip``, ``tube.zip`` and three zips for ``cylinder``); this
    downloads them whole, because they are zips -- the range-request reader exists for datasets that are one big
    HDF5 file, not for an archive that has to be unpacked anyway."""
    import urllib.request

    if problem not in PROBLEMS:
        raise KeyError(f"'{problem}' is not one of {sorted(PROBLEMS)}")
    target = Path(root) / problem
    target.mkdir(parents=True, exist_ok=True)
    archives = [f"{problem}.zip"] if problem != "cylinder" else [f"cylinder/{s}.zip" for s in SUBSETS]
    for name in archives:
        local = target / Path(name).name
        if not local.exists():
            with urllib.request.urlopen(f"{ARCHIVE_URL}/{name}", timeout=timeout) as response:
                local.write_bytes(response.read())
        with zipfile.ZipFile(local) as archive:
            archive.extractall(target)
    return target / problem if (target / problem).is_dir() else target
