"""Stage 8's outside half: a corpus somebody else solved, read only so a model can be **predicted** on.

The design spaces in :mod:`pinneapple_neural.lpm_families` are drawn and solved here, so a held-out number on them
says the pipeline works on data this repository made. The corpora below are the other side of that: Luminary's SHIFT
datasets, solved by their authors and published on Hugging Face, used for one thing — asking a model trained here
what it says about geometry it has never seen, and scoring the answer against a solver that owed us nothing.

**Validation only.** Not one design read by this module may enter a fit. Every dataset is `cc-by-nc-4.0` apart from
``Submarine`` and ``CCA`` (``license:other``), so a model touched by them cannot underwrite a commercial claim
(:func:`~pinneapple_neural.lpm_stages.card` refuses to write that card, and :func:`require_validation_usage` refuses
the declaration here), and the point of the exercise is the opposite of training on it: the moment the model has seen
these bodies, the number they produce is no longer outside.

Facts measured on 2026-11-27 through the Hugging Face API, not recalled. Fifteen repositories live under
``luminary-shift``; thirteen are ``license:cc-by-nc-4.0`` and two are ``license:other``; the full repositories are
``gated: manual`` and the ``-sample`` twins are ``gated: auto``, so what is on disk is what somebody downloaded after
accepting the terms — the files here are never fetched by this module. The layout is not one shape: ``WING`` nests
``sample_NNNNNN`` under a version directory, ``Pump``/``Submarine``/``CCA`` put ``sample_NNNNNN`` at the root, ``SUV``
uses ``RUN_N`` with force CSVs, ``Battery`` uses a case hash, ``SHIFT-Crash`` uses ``yaris_xNNNNN`` and ``Truck`` uses
``variant_NNNN``. Across all seven the invariant is the **case directory holds ``merged_surfaces.vtp``**, which is
what :func:`samples` walks for rather than a name pattern someone would have had to guess.

What is *not* certain is the schema of ``params.json`` and ``forces.json``: the files are behind the gate, so the keys
are read from whatever the dataset actually contains and the caller names the ones it wants
(``condition_map``). A key that is not there is reported with the keys that are, never silently defaulted.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CASE_SURFACE",
    "DATASETS",
    "FIELD_PREFERENCE",
    "USAGE",
    "Sample",
    "ShiftDataset",
    "card_licence",
    "dataset",
    "design",
    "designs",
    "open_sample",
    "report",
    "require_validation_usage",
    "samples",
]

#: the usage this corpus allows, whatever the caller would prefer.
USAGE = "validation-only"

#: the file whose presence defines a case directory, in every one of the seven layouts.
CASE_SURFACE = "merged_surfaces.vtp"

#: surface field names to try, in order, when the caller does not name one. DoMINO spells the wall pressure
#: ``CpMeanTrim``; ``Cp`` and the others are what neighbouring datasets in the same family call the same quantity.
FIELD_PREFERENCE: tuple[str, ...] = ("CpMeanTrim", "Cp", "pressureCoefficient", "pMeanTrim", "pressure", "p")


@dataclass(frozen=True)
class ShiftDataset:
    """One repository: what it holds, what it allows, and how many cases were counted in it.

    ``files_per_case`` and ``n_cases`` are measurements (taken from the repository tree, not from a README) and are
    left as ``None`` where they were not taken, because a guessed count is worse than an absent one.
    """

    name: str
    licence: str
    gated: str
    note: str = ""
    case_dir: str = ""
    files_per_case: tuple[str, ...] = ()
    n_cases: int | None = None
    measured_bytes: int | None = None

    @property
    def repo_id(self) -> str:
        return f"luminary-shift/{self.name}"

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo_id}"

    @property
    def commercial(self) -> bool:
        """Whether the licence could underwrite a commercial model. ``license:other`` is not automatically one."""
        return self.licence.lower().startswith(("cc-by", "apache", "mit")) and "-nc" not in self.licence.lower()

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "repo_id": self.repo_id, "licence": self.licence, "gated": self.gated,
                "case_dir": self.case_dir, "files_per_case": list(self.files_per_case), "n_cases": self.n_cases,
                "measured_bytes": self.measured_bytes, "usage": USAGE, "note": self.note}


_DOMINO_FILES = ("forces.json", "merged_surfaces.stl", CASE_SURFACE, "metadata.json", "params.json")
_DOMINO_VOLUME = _DOMINO_FILES + ("merged_volumes.vtu",)

#: The fifteen repositories, with the licence and gate the API reports and the layout the tree shows.
DATASETS: dict[str, ShiftDataset] = {
    "WING": ShiftDataset(
        "WING", "cc-by-nc-4.0", "manual",
        note="CRM wing, OnShape variants, sample_NNNNNN under a version directory, plus the inter-mesh STLs; "
             "the sample twin measured 82 cases in 38.2 GB",
        case_dir="OnShape_luminary_crm_version001/sample_000001",
        files_per_case=_DOMINO_FILES + ("merged_surfaces_onshape_200k.stl",), n_cases=82,
        measured_bytes=38_150_597_205),
    "Wing-sample": ShiftDataset(
        "Wing-sample", "cc-by-nc-4.0", "auto", note="the sample twin of WING, same layout",
        case_dir="OnShape_luminary_crm_version001/sample_000001",
        files_per_case=_DOMINO_FILES + ("merged_surfaces_onshape_200k.stl",), n_cases=81,
        measured_bytes=38_150_597_205),
    "Pump": ShiftDataset("Pump", "cc-by-nc-4.0", "manual", note="centrifugal pump; sample_NNNNNN at the root"),
    "Pump-sample": ShiftDataset(
        "Pump-sample", "cc-by-nc-4.0", "auto", case_dir="sample_000001", files_per_case=_DOMINO_VOLUME,
        n_cases=100, measured_bytes=72_500_000_000),
    "Submarine": ShiftDataset(
        "Submarine", "other", "manual",
        note="license:other — read the repository's own terms before using a number from it"),
    "Submarine-sample": ShiftDataset(
        "Submarine-sample", "cc-by-nc-4.0", "auto", case_dir="sample_000000", files_per_case=_DOMINO_VOLUME,
        n_cases=101, measured_bytes=70_400_000_000),
    "CCA": ShiftDataset("CCA", "other", "manual", note="license:other — as Submarine"),
    "CCA-sample": ShiftDataset(
        "CCA-sample", "cc-by-nc-4.0", "auto", case_dir="sample_000001", files_per_case=_DOMINO_VOLUME,
        n_cases=99, measured_bytes=74_700_000_000),
    "SUV": ShiftDataset("SUV", "cc-by-nc-4.0", "manual", note="RUN_N directories with force and residual CSVs"),
    "SUV-sample": ShiftDataset(
        "SUV-sample", "cc-by-nc-4.0", "auto", case_dir="RUN_1",
        files_per_case=(CASE_SURFACE, "merged_surfaces.stl", "DRAG.csv", "DRAG_COEFFICIENT.csv", "LIFT.csv",
                        "LIFT_COEFFICIENT.csv", "residuals.csv", "residuals_and_forces.png"),
        n_cases=35, measured_bytes=17_600_000_000),
    "Battery": ShiftDataset("Battery", "cc-by-nc-4.0", "manual", note="case-hash directories, qoi.json, fluid channel"),
    "Battery-sample": ShiftDataset(
        "Battery-sample", "cc-by-nc-4.0", "auto", case_dir="",
        files_per_case=("fluid_channel.stl", "merged_surfaces.stl", CASE_SURFACE, "metadata.json", "params.json",
                        "qoi.json", "volume_fluid.vtu"), n_cases=50, measured_bytes=126_500_000_000),
    "SHIFT-Crash": ShiftDataset("SHIFT-Crash", "cc-by-nc-4.0", "manual",
                                note="crash deformation; yaris_xNNNNN directories, no forces.json"),
    "SHIFT-Crash-sample": ShiftDataset(
        "SHIFT-Crash-sample", "cc-by-nc-4.0", "auto", case_dir="yaris_x00001",
        files_per_case=("merged_surfaces.stl", CASE_SURFACE, "params.json"), n_cases=100,
        measured_bytes=29_200_000_000),
    "Truck": ShiftDataset(
        "Truck", "cc-by-nc-4.0", "manual", case_dir="variant_0000",
        note="variant_NNNN directories plus one doe_parameters.csv for the whole dataset; 12.1 GB, 16 files",
        files_per_case=("merged_surfaces.stl", CASE_SURFACE, "params.json", "forces.json"), n_cases=7,
        measured_bytes=12_100_000_000),
}


def dataset(name: str) -> ShiftDataset:
    """One repository by name (``WING``, ``Pump-sample``, ...). An unknown name lists the known ones."""
    known = {k.lower(): v for k, v in DATASETS.items()}
    try:
        return known[name.lower()]
    except KeyError:
        raise ValueError(f"unknown SHIFT dataset {name!r}; known: {', '.join(sorted(DATASETS))}") from None


# ------------------------------------------------------------------------------------------ reading what is on disk
class ShiftDataMissing(FileNotFoundError):
    """A case file is not on disk. The message names the file, the repository and the gate that stands in front of it.

    A ``FileNotFoundError`` subclass because the caller's handling of a missing file does not change, and its own
    name because the message — which carries the repository and the licence — is the part worth catching in a log.
    """


@dataclass
class Sample:
    """One solved case: its directory, the files it has, the numbers beside it, and its surface on demand."""

    path: Path
    dataset: ShiftDataset
    files: dict[str, Path] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)
    forces: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    _surface: Any = None

    @property
    def label(self) -> str:
        """``dataset/relative/path``: unique across a tree whose case directories are named by six schemes."""
        return f"{self.dataset.name}/{self.path.name}"

    @property
    def surface_path(self) -> Path:
        return self.files[CASE_SURFACE]

    def surface(self, arrays: Iterable[str] | None = None) -> Any:
        """The case surface, decoded once per ``arrays`` request and kept on the object.

        :func:`pinneapple_data.ashton.vtk_xml.read_surface` streams the file and decodes only the arrays asked for —
        which is the difference between 20 MB and the 231 MB a SHIFT surface weighs — so the first call names the
        arrays and every later call reuses the same object.
        """
        from pinneapple_data.ashton.vtk_xml import read_surface

        wanted = None if arrays is None else tuple(arrays)
        if self._surface is None or (wanted is not None and set(wanted) - set(self._surface.point_data)
                                     - set(self._surface.cell_data)):
            self._surface = read_surface(self.surface_path, arrays=wanted)
        return self._surface

    def field(self, name: str | None = None) -> tuple[np.ndarray, str]:
        """The surface field and where it lives: ``(values, "point" | "cell")``.

        Without a name, the first entry of :data:`FIELD_PREFERENCE` that the file actually carries is used; if none
        is there, the error lists the arrays the file does carry, because that list is the answer.
        """
        names = self.surface(arrays=())
        chosen = name
        if chosen is None:
            candidates = list(names.available_points) + list(names.available_cells)
            chosen = next((n for n in FIELD_PREFERENCE if n in candidates), None)
            if chosen is None:
                raise ValueError(f"{self.label}: none of {FIELD_PREFERENCE} is in {self.surface_path.name}; it "
                                 f"carries point arrays {names.available_points} and cell arrays "
                                 f"{names.available_cells}")
        read = self.surface(arrays=[chosen])
        if chosen in read.point_data:
            return np.asarray(read.point_data[chosen], dtype=np.float64), "point"
        if chosen in read.cell_data:
            return np.asarray(read.cell_data[chosen], dtype=np.float64), "cell"
        raise ValueError(f"{self.label}: {chosen!r} is in neither the point nor the cell arrays of "
                         f"{self.surface_path.name}")

    def number(self, key: str) -> float:
        """A scalar from ``params.json`` (dotted paths allowed), or the list of keys that were there."""
        if key in self.params:
            return self.params[key]
        raise KeyError(f"{self.label}: no parameter {key!r}; params.json has {sorted(self.params)}")

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "dataset": self.dataset.name, "licence": self.dataset.licence,
                "usage": USAGE, "files": sorted(self.files), "params": dict(self.params),
                "forces": dict(self.forces)}


def _numbers(value: Any, prefix: str = "") -> dict[str, float]:
    """Every number in a JSON document, flattened with dotted keys and nested keys kept under their prefix."""
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(_numbers(item, f"{prefix}{key}."))
    elif isinstance(value, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        for i, v in enumerate(value):
            out[f"{prefix}{i}"] = float(v)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix[:-1]] = float(value)
    return out


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def open_sample(directory: str | Path, dataset_name: str | None = None) -> Sample:
    """One case directory. ``dataset_name`` is inferred from the path when it is not given."""
    root = Path(directory)
    if not root.is_dir():
        raise ShiftDataMissing(f"{root} is not a directory")
    files = {p.name: p for p in sorted(root.iterdir()) if p.is_file()}
    if CASE_SURFACE not in files:
        raise ShiftDataMissing(
            f"{root}: no {CASE_SURFACE} — in every SHIFT layout that file is what makes a case a case. This "
            f"directory holds {sorted(files)}")
    guessed = dataset_name or _infer_dataset(root)
    entry = dataset(guessed) if guessed else ShiftDataset(
        root.parent.name, "unknown", "unknown", note="the path did not name a SHIFT repository")
    params_doc = _read_json(files.get("params.json", root / "params.json"), {})
    forces_doc = _read_json(files.get("forces.json", root / "forces.json"), {})
    return Sample(path=root, dataset=entry, files=files, params=_numbers(params_doc), forces=_numbers(forces_doc),
                  metadata=_read_json(files.get("metadata.json", root / "metadata.json"), {}))


def _infer_dataset(directory: Path) -> str | None:
    """The repository name from a path that passes through ``luminary-shift/<name>/...``."""
    lowered = {k.lower(): k for k in DATASETS}
    for part in reversed(directory.parts):
        if part.lower() in lowered:
            return lowered[part.lower()]
    return None


def samples(root: str | Path, dataset_name: str | None = None, *, limit: int | None = None) -> list[Sample]:
    """Every case directory under ``root``, sorted, capped by ``limit``.

    The walk looks for :data:`CASE_SURFACE` rather than for a directory-name pattern, because the seven layouts name
    their case directories six different ways and a pattern would have to be told about each of them.
    """
    base = Path(root)
    if not base.is_dir():
        raise ShiftDataMissing(f"{base} is not a directory; the SHIFT data is gated, so it has to be downloaded "
                               f"first (see {', '.join(sorted(DATASETS))})")
    found = sorted(p.parent for p in base.rglob(CASE_SURFACE))
    if limit is not None:
        found = found[:max(0, int(limit))]
    return [open_sample(p, dataset_name) for p in found]


# ------------------------------------------------------------------------------------------ the model's side of it
def _subset(n: int, keep: int) -> np.ndarray:
    """``keep`` indices spread evenly over ``n``, deterministically, without repeats."""
    if keep >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, keep).round().astype(np.int64))


def design(sample: Sample, family: Any, condition_map: dict[str, str], *, field: str | None = None,
           max_points: int = 2048) -> Any:
    """One solved case as a :class:`~pinneapple_neural.lpm.Design`, ready to be predicted on.

    ``condition_map`` maps **our** conditioning names to the keys of ``params.json`` (``{"alpha": "aoa_deg"}``), and
    has to cover every parameter the family conditions on: a case whose operating point is only half known would be
    predicted at a point nobody stated, so a missing key names itself and the keys the file does carry.

    ``max_points`` is not cosmetic. The model's neighbourhood features are a distance matrix from the query to the
    body, so a raw 200 000-node SHIFT surface is a 200 000² matrix; the case is therefore subsampled to a
    deterministic, evenly spaced subset that the model sees as both the body and the query points, and the number is
    written into the design's label so a report says which subset produced it.
    """
    import torch

    # local import: this module reads files, and stays importable without torch until a design is actually built
    from ..lpm import Design

    missing = [name for name in family.conditioning_params if name not in condition_map]
    if missing:
        raise KeyError(f"{family.name}: the condition map does not say where {missing} come from; params.json of "
                       f"{sample.label} carries {sorted(sample.params)}")
    values = np.array([sample.number(condition_map[name]) for name in family.conditioning_params], dtype=np.float64)
    bounds = np.array([family.conditioning_params[name] for name in family.conditioning_params], dtype=np.float64)
    span = bounds[:, 1] - bounds[:, 0]
    params = np.where(np.abs(span) < 1e-15, 0.0, (values - bounds[:, 0]) / np.where(span == 0, 1.0, span))

    surface = sample.surface()
    point_values, location = sample.field(field)
    # the dataset the model was fitted on stores the target already divided by the family's field scale, and the model
    # predicts in those units; an outside corpus is in physical units, so the same division happens here
    scales = np.asarray(family.scales, dtype=np.float64).ravel()
    if scales.size != 1:
        raise ValueError(f"{family.name} declares {scales.size} fields; one surface array is one field, so a "
                         f"multi-field family needs a corpus that carries them (this module reads one at a time)")
    from pinneapple_data.ashton.vtk_xml import face_area_normals, node_area_normals

    if location == "cell":
        _, normals = face_area_normals(surface)
        coords = np.empty((surface.n_cells, 3), dtype=np.float64)
        for size in surface.cells_per_shape:            # mixed triangle/quad surfaces: centres per cell size
            rows = np.flatnonzero(surface.cell_sizes == size)
            coords[rows] = surface.points[surface.cells_of(size)].mean(axis=1)
    else:
        coords = surface.points
        _, normals = node_area_normals(surface)
    keep = _subset(coords.shape[0], max_points)
    xyz = np.asarray(coords[keep], dtype=np.float32)
    nrm = np.asarray(normals[keep], dtype=np.float32)
    target = (np.asarray(point_values[keep], dtype=np.float64) / scales[0]).astype(np.float32).reshape(keep.size, 1)

    spec = {"label": f"{sample.label}#{keep.size}of{coords.shape[0]}", "field": field or "", "location": location,
            "dataset": sample.dataset.name, "licence": sample.dataset.licence, "usage": USAGE}
    obj = Design(points=torch.from_numpy(xyz), normals=torch.from_numpy(nrm), query=torch.from_numpy(xyz),
                 params=torch.from_numpy(params.astype(np.float32)), target=torch.from_numpy(target))
    obj.label = spec["label"]
    obj.source = spec
    return obj


def designs(root: str | Path, family: Any, condition_map: dict[str, str], *, dataset_name: str | None = None,
            field: str | None = None, max_points: int = 2048, limit: int | None = None) -> list[Any]:
    """Every case under ``root`` as a design. The corpus is held out by construction: nothing here is trained on."""
    return [design(s, family, condition_map, field=field, max_points=max_points)
            for s in samples(root, dataset_name, limit=limit)]


def require_validation_usage(usage: str) -> str:
    """The usage a card may declare for a model that touched this corpus: ``validation-only`` and nothing else.

    A non-commercial dataset does not become commercial because the model that read it is ours, and a model *fitted*
    on SHIFT bodies is not a validation of anything, so there is no usage value that would make this acceptable.
    """
    if usage != USAGE:
        raise ValueError(f"SHIFT data is {USAGE}: {usage!r} is not a declaration this corpus can support — "
                         f"cc-by-nc-4.0 does not underwrite a commercial model, and a model trained on these bodies "
                         f"is not a validation of them")
    return USAGE


def card_licence(dataset_name: str) -> dict[str, str]:
    """The ``licence`` block a card carries when the model was validated on ``dataset_name``."""
    entry = dataset(dataset_name)
    return {"data": entry.licence, "usage": USAGE, "source": entry.repo_id, "url": entry.url}


def report(model: Any, family: Any, validation: list[Any], *, train_designs: list[Any], dataset_name: str,
           calibration: Any = None) -> dict[str, Any]:
    """The held-out report and the gates, with the corpus and its licence written on the answer.

    The numbers come from :func:`~pinneapple_neural.lpm_stages.trust.held_out_report`, so an outside corpus is scored
    exactly as a held-out split of our own data is — the constant field fitted on the training designs beside the
    model, per design, and the same tolerance. What the corpus adds is the sentence the number needs in order to be
    read correctly: whose geometry it is, under which licence, and that the model never saw it.
    """
    from . import trust as trust_stage

    held = trust_stage.held_out_report(model, family, validation, train_designs, calibration=calibration)
    entry = dataset(dataset_name)
    return {
        "outside": {"dataset": entry.name, "repo_id": entry.repo_id, "url": entry.url, "licence": entry.licence,
                    "usage": require_validation_usage(USAGE), "n_cases": len(validation),
                    "note": "the model was not trained on these bodies; a number scored here is the only kind this "
                            "corpus can produce"},
        "held_out": held,
        "gates": trust_stage.point_gates(held, family),
        "licence": card_licence(dataset_name),
    }
