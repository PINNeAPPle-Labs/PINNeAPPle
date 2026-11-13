"""``PhysicsDataBundle``: a manifest and the arrays it references, as a directory on disk.

Layout of a saved bundle::

    burgers.ppb/
        bundle.json      the BundleManifest (schema pinneapple.physics-data/1), digest included
        arrays.npz       one array per key the manifest references

``bundle.json`` is indented and sorted so it diffs and reads well; ``arrays.npz`` is written by ``numpy.savez``.
The digest is the SHA-256 of the canonical manifest plus, in sorted key order, the SHA-256 of every array's
bytes; :meth:`PhysicsDataBundle.load` recomputes it and refuses a bundle that does not match, which is what turns
"the file says it passed" into "the file is the one that was checked".

Why a directory and not a single file: a bundle is meant to be read by other tools (h5py, xarray, ParaView), and
keeping the manifest as plain JSON is what lets them read the metadata without this library. A single ``.zip``
is available through :meth:`PhysicsDataBundle.pack` for shipping.

    >>> import numpy as np
    >>> from pinneapple_data.schema import PhysicsDataBundle, Field, AxisSpec, Physics
    >>> u = np.zeros((8, 16))
    >>> b = PhysicsDataBundle.from_arrays(
    ...     {"field:u": u, "axis:t": np.linspace(0, 1, 8), "axis:x": np.linspace(0, 1, 16)},
    ...     fields=[Field.from_array("u", u, unit="m/s", axes=("t", "x"))],
    ...     axes=[AxisSpec("t", "s"), AxisSpec("x", "m")],
    ...     physics=Physics(governing="u_t = 0"),
    ...     source="a zero field")
    >>> b.arrays["field:u"].shape
    (8, 16)
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    ArrayRef,
    AxisSpec,
    BundleManifest,
    CheckResult,
    Field,
    Fields,
    FileRef,
    Geometry,
    Mesh,
    Physics,
    Provenance,
    Verification,
    sha256_file,
)

__all__ = ["PhysicsDataBundle", "save", "load", "file_refs", "content_digest", "MANIFEST_NAME",
           "ARRAYS_NAME"]

MANIFEST_NAME = "bundle.json"
ARRAYS_NAME = "arrays.npz"


def _array_digest(key: str, arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(key.encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


@dataclass
class PhysicsDataBundle:
    """A physics dataset: its manifest and the arrays the manifest points at.

    The arrays are held in memory (``dict[str, np.ndarray]``, keyed exactly as the manifest's ``ArrayRef.key``)
    so a caller can slice and pass them straight to a trainer. A bundle is immutable in spirit -- mutating
    ``arrays`` after ``save()`` invalidates the digest, and :meth:`verify` says so.
    """

    manifest: BundleManifest
    arrays: dict[str, np.ndarray] = _dc_field(default_factory=dict)

    # ── construction ─────────────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], *, fields: Sequence[Field] = (),
                    axes: Sequence[AxisSpec] = (), physics: Physics | None = None, source: str,
                    license: str = "", title: str = "", id: str | None = None, mesh: Mesh | None = None,
                    geometry: Geometry | None = None, verification: Verification | None = None,
                    inputs: Sequence[FileRef] = (), adapter: str = "", adapter_version: str = "",
                    kind: str = "dataset",
                    url: str = "",
                    citation: str = "", metadata: Mapping[str, Any] | None = None,
                    config: Mapping[str, Any] | None = None) -> PhysicsDataBundle:
        """Build a bundle from arrays and the pieces that describe them.

        ``arrays`` is keyed the way the manifest references them: ``"axis:x"`` for an axis coordinate,
        ``"field:u"`` for a field (see :meth:`Field.from_array`). The axes get their ``ArrayRef`` here from the
        arrays themselves, so the shapes and dtypes in the manifest are the shapes and dtypes in memory.
        """
        store = {k: np.asarray(v) for k, v in arrays.items()}
        axis_specs: list[AxisSpec] = []
        for ax in axes:
            key = f"axis:{ax.name}"
            if key not in store:
                raise KeyError(f"axis '{ax.name}' has no array under '{key}' in the bundle")
            axis_specs.append(ax.model_copy(update={"data": ArrayRef(
                key=key, shape=tuple(store[key].shape), dtype=str(store[key].dtype))}))
        for f in fields:
            if f.data.key not in store:
                raise KeyError(f"field '{f.name}' references '{f.data.key}', which is not among the arrays")
            actual = tuple(store[f.data.key].shape)
            if actual != tuple(f.data.shape):
                raise ValueError(f"field '{f.name}' declares shape {tuple(f.data.shape)} but its array is {actual}")
        manifest = BundleManifest(
            id=id or _slug(title or source),
            title=title or source,
            geometry=geometry or Geometry(kind="structured-grid", dimension=_spatial_dim(axis_specs)),
            mesh=mesh or Mesh(kind="grid" if axis_specs else "none", dimension=_spatial_dim(axis_specs),
                              axes=axis_specs),
            physics=physics or Physics(),
            fields=Fields(variables=list(fields)),
            provenance=Provenance(source=source, license=license or "unknown", kind=kind, inputs=list(inputs),
                                  adapter=adapter, adapter_version=adapter_version, url=url, citation=citation,
                                  config=dict(config or {})),
            verification=verification or Verification(),
            metadata=dict(metadata or {}),
        )
        return cls(manifest=manifest, arrays=store)

    # ── access ──────────────────────────────────────────────────────────────────────────────────────────────
    def array(self, key: str) -> np.ndarray:
        try:
            return self.arrays[key]
        except KeyError:
            raise KeyError(f"no array '{key}' in this bundle; it has {sorted(self.arrays)}") from None

    def field_array(self, name: str) -> np.ndarray:
        """The values of a declared field, by its name (not its storage key)."""
        return self.array(self.manifest.fields.get(name).data.key)

    def axis_array(self, name: str) -> np.ndarray:
        spec = self.manifest.mesh.axis(name)
        if spec.data is None:
            raise KeyError(f"axis '{name}' has no coordinates")
        return self.array(spec.data.key)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.field_array(name)

    def __contains__(self, name: str) -> bool:
        return name in self.manifest.fields

    @property
    def axes(self) -> list[str]:
        return self.manifest.axes

    def n_samples(self) -> int:
        """Number of samples along the leading axis (the same length as ``t`` when there is one)."""
        if "t" in self.axes:
            return int(self.axis_array("t").shape[0])
        for f in self.manifest.fields.variables:
            return int(f.data.shape[0])
        return 0

    def check(self, check_id: str, *, tool: str = "", value: float | None = None, reference: float | None = None,
              tolerance: float | None = None, unit: str = "1", detail: str = "") -> CheckResult:
        """Record a check result, returning the (already appended) :class:`CheckResult`.

        ``passed`` is derived from ``value``/``tolerance`` when both are given -- never asserted by the caller --
        so a check cannot be recorded as passed with a number that fails it."""
        passed = None
        if value is not None and tolerance is not None:
            passed = abs(float(value) - float(reference or 0.0)) <= abs(float(tolerance))
        result = CheckResult(check_id=check_id, tool=tool, passed=passed, value=value, reference=reference,
                             tolerance=tolerance, unit=unit, detail=detail)
        self.manifest.verification.checks = [c for c in self.manifest.verification.checks if c.check_id != check_id]
        self.manifest.verification.checks.append(result)
        return result

    # ── integrity ───────────────────────────────────────────────────────────────────────────────────────────
    def digest(self) -> str:
        """SHA-256 over the canonical manifest and every array, in sorted key order."""
        h = hashlib.sha256()
        h.update(self.manifest.canonical_json().encode("utf-8"))
        h.update(content_digest(self.arrays).encode("utf-8"))
        return h.hexdigest()

    def verify(self) -> dict[str, Any]:
        """Recompute the digest and check every reference resolves; returns a report, never raises."""
        problems: list[str] = []
        for ref in self._refs():
            if ref.key not in self.arrays:
                problems.append(f"missing array '{ref.key}'")
                continue
            arr = self.arrays[ref.key]
            if tuple(arr.shape) != tuple(ref.shape):
                problems.append(f"'{ref.key}' is {arr.shape}, manifest says {ref.shape}")
            if str(arr.dtype) != ref.dtype:
                problems.append(f"'{ref.key}' is {arr.dtype}, manifest says {ref.dtype}")
        stored = self.manifest.digest
        computed = self.digest()
        if stored and stored != computed:
            problems.append(f"digest mismatch: manifest {stored[:12]}..., content {computed[:12]}")
        return {"ok": not problems, "problems": problems, "digest": computed, "stored_digest": stored}

    def _refs(self) -> Iterator[ArrayRef]:
        for ax in self.manifest.mesh.axes:
            if ax.data is not None:
                yield ax.data
        for f in self.manifest.fields.variables:
            yield f.data
        if self.manifest.mesh.nodes is not None:
            yield self.manifest.mesh.nodes
        if self.manifest.mesh.cells is not None:
            yield self.manifest.mesh.cells

    # ── persistence ────────────────────────────────────────────────────────────────────────────────────────
    def save(self, directory: str | os.PathLike[str]) -> Path:
        """Write ``bundle.json`` + ``arrays.npz`` into ``directory`` (created if needed); returns the path."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.manifest.digest = self.digest()
        (target / MANIFEST_NAME).write_text(
            json.dumps(self.manifest.model_dump(mode="json"), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8")
        tmp = target / (ARRAYS_NAME + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **{k: np.ascontiguousarray(v) for k, v in self.arrays.items()})
        os.replace(tmp, target / ARRAYS_NAME)
        return target

    @classmethod
    def load(cls, directory: str | os.PathLike[str], *, verify_digest: bool = True) -> PhysicsDataBundle:
        """Read a bundle back. With ``verify_digest`` a manifest whose digest no longer matches is refused."""
        source = Path(directory)
        manifest = BundleManifest.model_validate_json((source / MANIFEST_NAME).read_text(encoding="utf-8"))
        arrays: dict[str, np.ndarray] = {}
        npz = source / ARRAYS_NAME
        if npz.exists():
            with np.load(npz, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
        for ref in _manifest_refs(manifest):
            if ref.path:                      # stored as its own .npy next to the bundle
                arrays[ref.key] = np.load(source / ref.path, allow_pickle=False)
        bundle = cls(manifest=manifest, arrays=arrays)
        if verify_digest and manifest.digest:
            report = bundle.verify()
            if not report["ok"]:
                raise ValueError(f"bundle at {source} is not intact: " + "; ".join(report["problems"]))
        return bundle

    def pack(self, path: str | os.PathLike[str]) -> Path:
        """Write the bundle as a single ``.zip`` (``bundle.json`` + ``arrays.npz``) for shipping."""
        target = Path(path)
        with tempfile.TemporaryDirectory() as tmp:
            here = self.save(tmp)
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name in (MANIFEST_NAME, ARRAYS_NAME):
                    zf.write(here / name, arcname=name)
        return target

    @classmethod
    def unpack(cls, path: str | os.PathLike[str], *, verify_digest: bool = True) -> PhysicsDataBundle:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmp)
            return cls.load(tmp, verify_digest=verify_digest)

    # ── interop ────────────────────────────────────────────────────────────────────────────────────────────
    def to_xarray(self, field: str | None = None):
        """The bundle (or one field) as an ``xarray.Dataset``/``DataArray`` with its coordinates and units.

        The unit of every axis and field goes into the array attributes, so the xarray object says ``m/s`` the
        way the manifest does -- the reason this exists at all: xarray is what the rest of the scientific Python
        stack reads."""
        try:
            import xarray as xr
        except ImportError as exc:                                        # pragma: no cover - optional dependency
            raise ImportError("to_xarray needs xarray (pip install xarray)") from exc
        coords: dict[str, Any] = {}
        for ax in self.manifest.mesh.axes:
            if ax.data is not None:
                coords[ax.name] = (ax.name, self.array(ax.data.key), {"units": ax.unit})
        fields = ([self.manifest.fields.get(field)] if field else self.manifest.fields.variables)
        data_vars: dict[str, Any] = {}
        for f in fields:
            arr = self.field_array(f.name)
            dims = list(f.axes if f.axes else [c for c in coords])
            if f.is_vector:
                dims = dims + ["component"]
                coords.setdefault("component", ("component", list(f.components)))
            data_vars[f.name] = (dims, arr, {"units": f.unit, "location": f.location})
        ds = xr.Dataset(data_vars, coords=coords,
                        attrs={"id": self.manifest.id, "title": self.manifest.title,
                               "schema": f"{self.manifest.MODEL}/{self.manifest.schema_version}",
                               "source": self.manifest.provenance.source,
                               "license": self.manifest.provenance.license})
        return ds[field] if field else ds

    def summary(self) -> str:
        """A one-screen description: axes, fields, units, count and provenance."""
        m = self.manifest
        lines = [f"{m.id}  ({m.title or m.provenance.source})",
                 f"  source   {m.provenance.source}  [{m.provenance.kind}, license {m.provenance.license}]"]
        if m.mesh.axes:
            lines.append("  axes     " + ", ".join(
                f"{a.name}[{a.data.shape[0] if a.data else '?'}] {a.unit}" for a in m.mesh.axes))
        if m.fields.variables:
            lines.append("  fields   " + ", ".join(
                f"{f.name}{tuple(f.data.shape)} {f.unit} ({f.location})" for f in m.fields.variables))
        if m.physics.governing:
            lines.append(f"  physics  {m.physics.governing}")
        checks = m.verification.checks
        if checks:
            lines.append("  checks   " + ", ".join(f"{c.check_id}:{c.status}" for c in checks))
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhysicsDataBundle):
            return NotImplemented
        if self.manifest.canonical_json() != other.manifest.canonical_json():
            return False
        if sorted(self.arrays) != sorted(other.arrays):
            return False
        return all(np.array_equal(self.arrays[k], other.arrays[k], equal_nan=True) for k in self.arrays)

    def __repr__(self) -> str:
        return (f"PhysicsDataBundle(id={self.manifest.id!r}, axes={self.axes}, "
                f"fields={self.manifest.fields.names()}, arrays={len(self.arrays)})")


def _manifest_refs(manifest: BundleManifest) -> Iterator[ArrayRef]:
    for ax in manifest.mesh.axes:
        if ax.data is not None:
            yield ax.data
    for f in manifest.fields.variables:
        yield f.data
    for ref in (manifest.mesh.nodes, manifest.mesh.cells):
        if ref is not None:
            yield ref


def _spatial_dim(axes: Sequence[AxisSpec]) -> int:
    return max(1, len([a for a in axes if a.name != "t"])) if axes else 1


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text.strip()]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "bundle"


def content_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """SHA-256 over a set of arrays (keys, dtypes, shapes and bytes) -- the fingerprint of some data, no manifest."""
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(_array_digest(key, arrays[key]).encode("utf-8"))
    return h.hexdigest()


def save(bundle: PhysicsDataBundle, directory: str | os.PathLike[str]) -> Path:
    """Module-level shortcut for :meth:`PhysicsDataBundle.save`."""
    return bundle.save(directory)


def load(directory: str | os.PathLike[str], *, verify_digest: bool = True) -> PhysicsDataBundle:
    """Module-level shortcut for :meth:`PhysicsDataBundle.load`."""
    return PhysicsDataBundle.load(directory, verify_digest=verify_digest)


def file_refs(paths: Sequence[str | os.PathLike[str]]) -> list[FileRef]:
    """Hash a list of input files into the ``FileRef`` entries a manifest records."""
    return [FileRef(path=str(p), sha256=sha256_file(str(p)), bytes=Path(p).stat().st_size) for p in paths]
