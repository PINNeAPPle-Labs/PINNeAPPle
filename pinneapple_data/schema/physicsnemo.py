"""A bundle as the dictionary a PhysicsNeMo training loop consumes, and back. INT-014.

The contract was read from PhysicsNeMo's own benchmark datapipes rather than guessed: `Darcy2D.__iter__` yields

    yield {"permeability": self.output_k, "darcy": self.output_p}

-- a plain ``dict`` of named tensors, one per field, with no fixed key schema. A bundle therefore maps to
``{field name: array}`` plus, when the model wants coordinates, ``"pos"``; and the pointwise/pyTorch-Geometric
pipelines that do use a schema take ``pos`` and ``fields``, which this bridge can produce with ``schema="pointwise"``.

What this module does **not** do is pretend to run PhysicsNeMo: the package is a multi-gigabyte CUDA-first dependency
and is not required here. It converts the data, exercises the conversion in the tests, and raises a clear
``ImportError`` from :func:`to_physicsnemo_dataset` when the package is absent. A user who has it gets an iterable
their loop can already consume; a user who does not gets the same dictionary.

    >>> from pinneapple_data.schema.physicsnemo import to_dict
    >>> d = to_dict(bundle)                      # {"density": (T, X), "pressure": (T, X), "pos": (X, 1)}
    >>> d["density"].shape
    (201, 1024)
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

__all__ = ["to_dict", "from_dict", "BundleDictDataset", "to_physicsnemo_dataset"]

#: The key PhysicsNeMo's pointwise pipeline uses for coordinates; the benchmark datapipes carry no coordinates at all.
POS = "pos"


def to_dict(bundle: Any, *, fields: Sequence[str] | None = None, pos: bool = True,
            step: int | None = None) -> dict[str, np.ndarray]:
    """The bundle as ``{field: array}``, plus ``"pos"`` when asked for.

    ``step`` selects one time index; without it the whole trajectory is passed through, which is what a
    time-dependent pipe wants. A vector field keeps its component axis (PhysicsNeMo's tensors are
    ``(..., components)``, as its own solutions are)."""
    names = list(fields) if fields is not None else bundle.manifest.fields.names()
    out: dict[str, np.ndarray] = {}
    for name in names:
        values = np.asarray(bundle.field_array(name))
        if step is not None:
            values = values[step] if values.ndim > 1 else values
        out[name] = values
    if pos:
        spatial = [a for a in bundle.axes if a != "t"]
        if spatial:
            coords = [np.asarray(bundle.axis_array(a), dtype=np.float64) for a in spatial]
            grid = np.meshgrid(*coords, indexing="ij") if len(coords) > 1 else [coords[0]]
            out[POS] = np.stack([g.ravel() for g in grid], axis=1)
        elif bundle.manifest.mesh.nodes is not None:
            out[POS] = np.asarray(bundle.array(bundle.manifest.mesh.nodes.key), dtype=np.float64)
    return out


def from_dict(data: dict[str, Any], *, source: str, license: str, field: str = "", axes: Sequence[str] = (),
              axis_values: dict[str, np.ndarray] | None = None, kind: str = "dataset") -> Any:
    """A bundle from a dictionary of arrays.

    The inverse of :func:`to_dict` for the structured case: every array that is not ``"pos"`` becomes a field over
    the axes given (``axes`` names them in order, and ``axis_values`` gives their coordinates). Without axes the
    bundle has no mesh, which the schema records as ``kind="none"`` rather than inventing grid lines."""
    from ..schema import AxisSpec, Field, Physics, PhysicsDataBundle

    fields = {k: np.asarray(v) for k, v in data.items() if k != POS}
    if not fields:
        raise ValueError("the dictionary has no field to put in a bundle (only 'pos' was found)")
    name = field or next(iter(fields))
    arrays: dict[str, np.ndarray] = {}
    specs: list[AxisSpec] = []
    if axes:
        values = axis_values or {}
        for ax in axes:
            if ax not in values:
                raise ValueError(f"axis '{ax}' has no coordinates in axis_values")
            arrays[f"axis:{ax}"] = np.asarray(values[ax])
            specs.append(AxisSpec(ax, "1"))
    arrays[f"field:{name}"] = fields[name]
    declared = [Field.from_array(name, fields[name], unit="1", axes=tuple(axes) if axes else ())]
    for other, values in fields.items():
        if other == name:
            continue
        arrays[f"field:{other}"] = values
        declared.append(Field.from_array(other, values, unit="1", axes=tuple(axes) if axes else ()))
    return PhysicsDataBundle.from_arrays(arrays, fields=declared, axes=specs, physics=Physics(),
                                         source=source, license=license, kind=kind)


class BundleDictDataset:
    """Several bundles as an iterable of dictionaries -- the shape a PhysicsNeMo loop iterates.

    ``tensors=True`` converts to ``torch`` tensors when torch is installed, which is what a training loop wants;
    without it the dictionaries carry numpy arrays, which is still a legal map-style dataset."""

    def __init__(self, bundles: Iterable[Any], *, fields: Sequence[str] | None = None, pos: bool = True,
                 step: int | None = None, tensors: bool = True) -> None:
        self.bundles = list(bundles)
        if not self.bundles:
            raise ValueError("no bundles given")
        self.fields = fields
        self.pos = pos
        self.step = step
        self.tensors = tensors
        if tensors:
            try:
                import torch  # noqa: F401
            except ImportError:                                          # pragma: no cover - optional
                self.tensors = False

    def __len__(self) -> int:
        return len(self.bundles)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = to_dict(self.bundles[index], fields=self.fields, pos=self.pos, step=self.step)
        if not self.tensors:
            return item
        import torch
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in item.items()}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def to_physicsnemo_dataset(bundles: Iterable[Any], **kwargs: Any):
    """A PhysicsNeMo ``Datapipe`` subclass instance when the package is installed, else a clear ``ImportError``.

    The class is built at call time (not at import) so this module stays importable without PhysicsNeMo -- and so the
    error names what to install instead of failing somewhere deep in a training loop."""
    try:
        from physicsnemo.datapipes.datapipe import Datapipe
    except ImportError as exc:
        raise ImportError("to_physicsnemo_dataset needs PhysicsNeMo (pip install nvidia-physicsnemo); "
                          "to_dict and BundleDictDataset need nothing") from exc
    plain = BundleDictDataset(bundles, **kwargs)

    class _BundlePipe(Datapipe):                                     # pragma: no cover - needs PhysicsNeMo
        def __init__(self) -> None:
            super().__init__()
            self.items = plain

        def __iter__(self):
            yield from self.items

        def __len__(self) -> int:
            return len(self.items)

    return _BundlePipe()
