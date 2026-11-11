"""The benchmark-owned scoring support: coordinates, weights and ground truth, joined by stable ids.

A FluidsBench support release is a hash chain — ``manifest.json`` → ``case-sets/<set>/index.json`` →
``chunk-000.json`` → one ground-truth table per case and support. The benchmark owns every number in it; a submission only
brings predictions keyed by ``support_id``. This module reads that chain and verifies it, so a result built on it can say
which bytes it was scored against::

    from pinneapple_arena.fluidsbench import load_support_release

    release = load_support_release("path/to/support/manifest.json", "standard")
    case = release.support("case-001", "volume-points-v1")
    print(case.support_ids[:3], case.coordinates.shape, case.targets["pressure"].shape)

Only the ``materialized_table`` location mode is read generically here (JSON, CSV and NPZ tables, local or from an
extracted archive), which is what the contract's generic evaluator does. A dataset whose release declares
``native_entities``, ``structured_grid`` or ``reference_generator`` is evaluated by that dataset's own reference code;
loading one here raises :class:`UnsupportedLocationMode` instead of inventing a reading of it.

:func:`write_support_release` writes a release **in the same layout** for rehearsal and for the tests in this repository.
It is labelled ``synthetic_rehearsal`` and is not a FluidsBench release: it exists so the pipeline can be exercised
without an 800 MB download, and so a participant can test the mapping before touching real data.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["CaseSupport", "SupportRelease", "SupportDefinition", "MetricBinding", "Quantity", "Component",
           "UnsupportedLocationMode", "SupportError", "load_support_release", "write_support_release"]

TABLE_FORMATS = frozenset({"json", "csv", "npz"})
SUPPORTED_MODES = frozenset({"materialized_table"})


class SupportError(ValueError):
    """The support release is inconsistent: a digest, a count, an id or a column does not match."""


class UnsupportedLocationMode(SupportError):
    """The release points at a location mode that only the dataset's own evaluator can read."""

    def __init__(self, mode: str, dataset_id: str):
        super().__init__(f"{dataset_id or 'this dataset'} scores on location mode {mode!r}: its release is read by that "
                         f"dataset's reference evaluator, not by the generic materialized_table loader")
        self.mode = mode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise SupportError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise SupportError(f"{path} is not valid JSON: {error}") from error


@dataclass(frozen=True)
class Component:
    """One component of a quantity: which table column holds the truth and which the prediction must provide."""
    id: str
    target_field: str
    prediction_field: str


@dataclass(frozen=True)
class Quantity:
    id: str
    unit: str
    components: tuple[Component, ...]


@dataclass(frozen=True)
class MetricBinding:
    """One metric the dataset scores on this support: its reduction, weighting and aggregation."""
    metric_id: str
    quantity_id: str
    reduction: str
    weighting: str
    dataset_weighting: str
    aggregation: str
    case_evidence: str = "metric_value"


@dataclass(frozen=True)
class SupportDefinition:
    """One support of the release: where its entities live, what it carries and how it is scored."""
    id: str
    domain: str
    location_mode: str
    table_format: str
    quantities: tuple[Quantity, ...]
    bindings: tuple[MetricBinding, ...]
    required_coverage: Mapping[str, Any] = field(default_factory=dict)
    extrapolation_policy: str = "forbidden"

    @property
    def prediction_fields(self) -> tuple[str, ...]:
        return tuple(c.prediction_field for q in self.quantities for c in q.components)

    def quantity(self, quantity_id: str) -> Quantity:
        for q in self.quantities:
            if q.id == quantity_id:
                return q
        raise SupportError(f"support {self.id!r} carries no quantity {quantity_id!r}")


@dataclass(frozen=True)
class CaseSupport:
    """One case's entities for one support: ids, coordinates, weights and the ground-truth targets."""
    case_id: str
    definition: SupportDefinition
    support_ids: np.ndarray
    coordinates: np.ndarray
    weights: np.ndarray
    targets: Mapping[str, np.ndarray]

    @property
    def support_id(self) -> str:
        return self.definition.id

    @property
    def entity_count(self) -> int:
        return int(len(self.support_ids))

    def target(self, quantity_id: str) -> np.ndarray:
        try:
            return np.asarray(self.targets[quantity_id], dtype=np.float64)
        except KeyError as error:
            raise SupportError(f"case {self.case_id!r} has no targets for quantity {quantity_id!r}") from error


def _definition(entry: Mapping[str, Any], dataset_id: str) -> SupportDefinition:
    location = entry.get("location_definition") or {}
    mode = str(location.get("mode", ""))
    if mode not in SUPPORTED_MODES:
        raise UnsupportedLocationMode(mode, dataset_id)
    table_format = str(location.get("format", ""))
    if table_format not in TABLE_FORMATS:
        raise SupportError(f"support {entry.get('id')!r} uses table format {table_format!r}; "
                           f"the generic loader reads {sorted(TABLE_FORMATS)}")
    quantities = []
    for q in entry.get("quantities") or []:
        components = tuple(Component(id=str(c["id"]), target_field=str(c["target_field"]),
                                     prediction_field=str(c["prediction_field"]))
                           for c in q.get("components") or [])
        if not components:
            raise SupportError(f"quantity {q.get('id')!r} of support {entry.get('id')!r} has no components")
        quantities.append(Quantity(id=str(q["id"]), unit=str(q.get("unit", "")), components=components))
    bindings = tuple(MetricBinding(metric_id=str(b["metric_id"]), quantity_id=str(b["quantity_id"]),
                                   reduction=str(b["reduction"]), weighting=str(b["weighting"]),
                                   dataset_weighting=str(b.get("dataset_weighting", "")),
                                   aggregation=str(b.get("aggregation", "per_geometry_then_macro_average")),
                                   case_evidence=str(b.get("case_evidence", "metric_value")))
                     for b in entry.get("metric_bindings") or [])
    return SupportDefinition(id=str(entry["id"]), domain=str(entry.get("domain", "")), location_mode=mode,
                             table_format=table_format, quantities=tuple(quantities), bindings=bindings,
                             required_coverage=dict(entry.get("required_coverage") or {}),
                             extrapolation_policy=str(entry.get("extrapolation_policy", "forbidden")))


class SupportRelease:
    """A verified support release, with its cases loaded on demand."""

    def __init__(self, manifest_path: Path, manifest: Mapping[str, Any], case_set_id: str,
                 definitions: Mapping[str, SupportDefinition], cases: Mapping[str, Mapping[str, Any]],
                 case_dirs: Mapping[str, Path], digests: Mapping[str, str], verified: bool):
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.case_set_id = case_set_id
        self.definitions = dict(definitions)
        self._cases = dict(cases)
        self._case_dirs = dict(case_dirs)
        self.digests = dict(digests)
        self.verified = verified
        self._cache: dict[tuple[str, str], CaseSupport] = {}

    # ------------------------------------------------------------- identity
    @property
    def release_id(self) -> str:
        return str(self.manifest.get("release_id", ""))

    @property
    def dataset_id(self) -> str:
        return str(self.manifest.get("dataset_id", ""))

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.manifest_path)

    @property
    def case_ids(self) -> list[str]:
        return list(self._cases)

    def __len__(self) -> int:
        return len(self._cases)

    def support_ids(self, case_id: str) -> list[str]:
        try:
            return list(self._cases[case_id])
        except KeyError as error:
            raise SupportError(f"case {case_id!r} is not in case set {self.case_set_id!r}") from error

    # -------------------------------------------------------------- loading
    def support(self, case_id: str, support_id: str, *, verify: bool = True,
                archive_root: str | Path | None = None) -> CaseSupport:
        key = (case_id, support_id)
        if key in self._cache:
            return self._cache[key]
        try:
            instance = self._cases[case_id][support_id]
        except KeyError as error:
            raise SupportError(f"case {case_id!r} has no support {support_id!r}; it carries "
                               f"{list(self._cases.get(case_id, {}))}") from error
        definition = self.definitions[support_id]
        table = self._load_table(case_id, instance, definition, verify=verify, archive_root=archive_root)
        case = self._as_case(case_id, definition, table, instance)
        self._cache[key] = case
        return case

    def supports(self, case_id: str, **kwargs: Any) -> list[CaseSupport]:
        return [self.support(case_id, sid, **kwargs) for sid in self.support_ids(case_id)]

    def _load_table(self, case_id: str, instance: Mapping[str, Any], definition: SupportDefinition, *,
                    verify: bool, archive_root: str | Path | None) -> Mapping[str, np.ndarray]:
        artifacts = [a for a in instance.get("artifacts") or [] if a.get("role") == "ground_truth_table"]
        if len(artifacts) != 1:
            raise SupportError(f"{case_id}/{definition.id}: expected one ground_truth_table artifact, "
                               f"found {len(artifacts)}")
        artifact = artifacts[0]
        path = self._artifact_path(case_id, artifact, archive_root)
        if verify and artifact.get("sha256") and _sha256(path) != artifact["sha256"]:
            raise SupportError(f"{path} does not match the digest the chunk declares for {case_id}/{definition.id}")
        return self._read_table(path, definition.table_format)

    def _artifact_path(self, case_id: str, artifact: Mapping[str, Any],
                       archive_root: str | Path | None) -> Path:
        if artifact.get("path"):
            return (self._case_dirs[case_id] / str(artifact["path"])).resolve()
        archive = artifact.get("archive") or {}
        member = archive.get("member")
        if not member:
            raise SupportError(f"{case_id}: artifact has neither a local path nor an archive member")
        if archive_root is None:
            raise SupportError(f"{case_id}: this release keeps its tables in the archive {archive.get('url')!r}; "
                               f"extract it and pass archive_root=")
        candidate = Path(archive_root).expanduser() / str(member)
        if not candidate.is_file():
            raise SupportError(f"{case_id}: {candidate} is not there; extract {archive.get('url')} there first")
        return candidate

    @staticmethod
    def _read_table(path: Path, table_format: str) -> Mapping[str, np.ndarray]:
        if table_format == "json":
            payload = _json(path)
            columns = payload.get("columns", payload)
            if not isinstance(columns, Mapping):
                raise SupportError(f"{path} has no columns mapping")
            return {str(k): np.asarray(v) for k, v in columns.items()}
        if table_format == "csv":
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise SupportError(f"{path} is empty")
            return {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
        with np.load(path) as data:
            return {str(k): np.asarray(data[k]) for k in data.files}

    def _as_case(self, case_id: str, definition: SupportDefinition, table: Mapping[str, np.ndarray],
                 instance: Mapping[str, Any]) -> CaseSupport:
        ids = table.get("support_id")
        if ids is None:
            raise SupportError(f"{case_id}/{definition.id}: the ground-truth table has no support_id column")
        ids = np.asarray([str(i) for i in ids])
        if len(set(ids.tolist())) != len(ids):
            raise SupportError(f"{case_id}/{definition.id}: duplicate support_id values")
        order = np.argsort(ids, kind="stable")                    # the contract's support_id_ascending ordering

        coordinates, weights, targets = [], [], {}
        fields = [str(f) for f in _coordinate_fields(self.manifest, definition.id)]
        for name in fields:
            if name not in table:
                raise SupportError(f"{case_id}/{definition.id}: table has no coordinate column {name!r}")
            coordinates.append(np.asarray(table[name], dtype=np.float64)[order])
        for quantity in definition.quantities:
            stack = []
            for component in quantity.components:
                if component.target_field not in table:
                    raise SupportError(f"{case_id}/{definition.id}: table has no target column "
                                       f"{component.target_field!r}")
                stack.append(np.asarray(table[component.target_field], dtype=np.float64)[order][:, None])
            targets[quantity.id] = np.concatenate(stack, axis=1)
        weights_value = _weight_field(self.manifest, definition.id)
        if weights_value and weights_value in table:
            weights = np.asarray(table[weights_value], dtype=np.float64)[order]
        else:
            weights = np.ones(len(ids), dtype=np.float64)

        expected = instance.get("entity_count")
        if expected is not None and int(expected) != len(ids):
            raise SupportError(f"{case_id}/{definition.id}: chunk says {expected} entities, the table has {len(ids)}")
        return CaseSupport(case_id=case_id, definition=definition, support_ids=ids[order],
                           coordinates=np.stack(coordinates, axis=1), weights=weights, targets=targets)


def _support_entry(manifest: Mapping[str, Any], support_id: str) -> Mapping[str, Any]:
    for entry in manifest.get("supports") or []:
        if entry.get("id") == support_id:
            return entry
    raise SupportError(f"no support {support_id!r} in the manifest")


def _coordinate_fields(manifest: Mapping[str, Any], support_id: str) -> Sequence[str]:
    location = _support_entry(manifest, support_id).get("location_definition") or {}
    fields = location.get("coordinate_fields")
    if not fields:
        raise SupportError(f"support {support_id!r} declares no coordinate_fields")
    return [str(f) for f in fields]


def _weight_field(manifest: Mapping[str, Any], support_id: str) -> str:
    location = _support_entry(manifest, support_id).get("location_definition") or {}
    rule = location.get("weight_rule") or {}
    return str(rule.get("field", "")) if rule.get("kind") == "artifact_field" else ""


def load_support_release(manifest_path: str | Path, case_set_id: str, *, verify: bool = True,
                         dataset_id: str | None = None) -> SupportRelease:
    """Read and verify ``manifest → case-set index → chunk``.

    ``verify`` checks the digests the release declares for the index and for every chunk. Artifact digests are checked
    when the ground truth is loaded (:meth:`SupportRelease.support`), so a release of 200 cases is not read whole to
    answer a question about its shape."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _json(manifest_path)
    dataset = str(manifest.get("dataset_id", ""))
    if dataset_id is not None and dataset != dataset_id:
        raise SupportError(f"{manifest_path} belongs to dataset {dataset!r}, not {dataset_id!r}")
    digests = {manifest_path.name: _sha256(manifest_path)}

    definitions = {}
    for entry in manifest.get("supports") or []:
        definition = _definition(entry, dataset)
        definitions[definition.id] = definition

    case_set = next((c for c in manifest.get("case_sets") or [] if c.get("id") == case_set_id), None)
    if case_set is None:
        raise SupportError(f"release {manifest.get('release_id')!r} has no case set {case_set_id!r}; it publishes "
                           f"{[c.get('id') for c in manifest.get('case_sets') or []]}")
    index_path = (manifest_path.parent / str(case_set["index_file"])).resolve()
    if verify and case_set.get("index_sha256") and _sha256(index_path) != case_set["index_sha256"]:
        raise SupportError(f"{index_path} does not match the index digest the manifest declares")
    digests[index_path.name] = _sha256(index_path)
    index = _json(index_path)

    cases: dict[str, dict[str, Any]] = {}
    case_dirs: dict[str, Path] = {}
    for chunk_entry in index.get("chunks") or []:
        chunk_path = (index_path.parent / str(chunk_entry["file"])).resolve()
        if verify and chunk_entry.get("sha256") and _sha256(chunk_path) != chunk_entry["sha256"]:
            raise SupportError(f"{chunk_path} does not match the chunk digest the case-set index declares")
        digests[chunk_path.name] = _sha256(chunk_path)
        chunk = _json(chunk_path)
        for case in chunk.get("cases") or []:
            case_id = str(case["case_id"])
            instances = {str(i["support_id"]): i for i in case.get("support_instances") or []}
            unknown = set(instances) - set(definitions)
            if unknown:
                raise SupportError(f"case {case_id!r} uses support(s) {sorted(unknown)} that the manifest does not define")
            cases[case_id] = instances
            case_dirs[case_id] = chunk_path.parent
    if index.get("case_count") and int(index["case_count"]) != len(cases):
        raise SupportError(f"the index declares {index['case_count']} cases, the chunks carry {len(cases)}")
    return SupportRelease(manifest_path, manifest, case_set_id, definitions, cases, case_dirs, digests, verified=verify)


# --------------------------------------------------------------------------- rehearsal releases
def write_support_release(dest: str | Path, *, release_id: str, dataset_id: str,
                          supports: Sequence[Mapping[str, Any]],
                          cases: Mapping[str, Mapping[str, Mapping[str, Any]]], notes: str = "",
                          status: str = "synthetic_rehearsal") -> Path:
    """Write a support release in the published layout, for rehearsal and for the tests of this repository.

    ``supports`` are entries of the manifest's ``supports`` (``materialized_table`` + ``json``, one per support);
    ``cases`` maps a case id to ``{support_id: {"columns": {column: array}}}``. The manifest, the case-set index, the
    chunk and every table get the digests the contract expects, so a package built on it validates exactly like one
    built on an official release. ``status`` is written as given: this never claims to be a FluidsBench release."""
    dest = Path(dest).expanduser().resolve()
    data_dir = dest / "case-sets" / "standard" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for definition in supports:
        entry = dict(definition)
        entry.setdefault("location_definition", {}).setdefault("format", "json")
        written.append(entry)

    chunk_cases = []
    for case_id, by_support in cases.items():
        instances = []
        for support_id, payload in by_support.items():
            table_path = data_dir / f"{case_id}-{support_id}-ground-truth.json"
            columns = {name: _plain(np.asarray(values)) for name, values in payload["columns"].items()}
            table_path.write_text(json.dumps({"columns": columns}, indent=1) + "\n", encoding="utf-8")
            instances.append({
                "support_id": support_id,
                "entity_count": int(len(columns["support_id"])),
                "artifacts": [{"role": "ground_truth_table",
                               "path": f"data/{table_path.name}",
                               "sha256": _sha256(table_path), "format": "json"}],
            })
        chunk_cases.append({"case_id": case_id, "support_instances": instances})

    chunk = {"$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-chunk.schema.json",
             "schema_version": "1.0", "release_id": release_id, "dataset_id": dataset_id,
             "case_set_id": "standard", "cases": chunk_cases}
    chunk_path = dest / "case-sets" / "standard" / "chunk-000.json"
    chunk_path.write_text(json.dumps(chunk, indent=1) + "\n", encoding="utf-8")

    index = {"$schema": "https://fluidsbench.org/schemas/scoring-support/v1/case-index.schema.json",
             "schema_version": "1.0", "release_id": release_id, "dataset_id": dataset_id,
             "case_set_id": "standard", "case_count": len(chunk_cases),
             "chunks": [{"file": "chunk-000.json", "sha256": _sha256(chunk_path),
                         "case_count": len(chunk_cases),
                         "case_ids": [c["case_id"] for c in chunk_cases]}]}
    index_path = dest / "case-sets" / "standard" / "index.json"
    index_path.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")

    manifest = {
        "$schema": "https://fluidsbench.org/schemas/scoring-support/v1/manifest.schema.json",
        "schema_version": "1.0", "release_id": release_id, "status": status, "dataset_id": dataset_id,
        "dataset_version": f"{dataset_id}-rehearsal-1",
        "evaluation_reference_version": "pineapple-fluidsbench-adapter",
        "coordinate_frame": {"id": f"{dataset_id}-rehearsal-frame", "axis_order": _axis_order(written[0]),
                             "handedness": "right", "length_unit": "m", "normalization": "none"},
        "supports": written,
        "case_sets": [{"id": "standard", "case_count": len(chunk_cases), "index_file": "case-sets/standard/index.json",
                       "index_sha256": _sha256(index_path)}],
        "notes": notes or "Rehearsal release written by pinneapple_arena.fluidsbench; not a FluidsBench release.",
    }
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest_path


def _axis_order(definition: Mapping[str, Any]) -> list[str]:
    fields = (definition.get("location_definition") or {}).get("coordinate_fields") or ["x", "y", "z"]
    return [str(f) for f in fields]


def _plain(array: np.ndarray) -> Any:
    return array.tolist() if array.ndim else float(array)
