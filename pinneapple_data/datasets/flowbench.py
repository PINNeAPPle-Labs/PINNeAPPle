"""FlowBench: flow over complex geometries, from the tensors the archive publishes.

FlowBench (Tali et al., 2024, arXiv:2409.18032) is the benchmark for neural simulators on **complex geometries**: a
shifted-boundary-method solver run over thousands of shapes, published on HuggingFace as ``BGLab/FlowBench``. This
module reads the two shapes the archive actually stores, both checked against the files themselves:

``X``/``Y`` tensors (the LDC families)
    one ``.npz`` per family, resolution and geometry family -- ``LDC_NS_2D/128x128/harmonics_lid_driven_cavity_X.npz``
    -- holding a single dataset ``data`` shaped ``(n_samples, n_channels, height, width)``. The channel lists are
    Table 12 of the paper: ``X`` is ``[Re, g, s]`` (or ``[Re, Gr, g, s]`` for the heat-transfer families) and ``Y`` is
    ``[u, v, p]`` (or ``[u, v, p, theta]``). Measured on the published 128 x 128 file: ``X`` is ``(1000, 3, 128,
    128)``, ``Y`` is ``(1000, 4, 128, 128)``. Channel names are assigned **by their count**, so a file whose counts
    match no documented row is refused instead of silently mislabelled -- the 128 x 128 file above, for instance, has
    the 3-channel input of the Navier-Stokes family and the 4-channel output of the heat-transfer one, and says so in
    the bundle's metadata rather than picking a story.

``FPO`` per-sample files
    ``FPO_NS_2D_1024x256/<geometry family>/<case>/Re_<value>.npz`` holds ``data`` shaped ``(n_time, height, width,
    channels)`` -- measured ``(242, 256, 1024, 3)`` -- with ``u, v, p`` **last**; the companion ``input_geometry.npz``
    holds ``data`` (a distance field) and ``mask`` (1 inside the solid). These files are 1.4 GB each, which is why the
    reader takes a path or a URL and reads one file, not a family.

The licence is CC-BY-NC-4.0 -- stated on the dataset card and in the paper ("Our datasets and code are licensed under
CC-BY-NC-4.0"), so it is recorded as verified, and it is **non-commercial**: `redistributable` in the catalogue says
the licence was checked, not that the data may ship in a product.

    >>> from pinneapple_data.datasets import flowbench
    >>> bundle = flowbench.load_tensors(".../LDC_NS_2D/128x128/harmonics_lid_driven_cavity_X.npz",
    ...                                 ".../harmonics_lid_driven_cavity_Y.npz", index=0)  # doctest: +SKIP
"""
from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["HUGGINGFACE", "ARCHIVE_URL", "LICENSE", "LICENSE_NOTE", "TENSORS", "read_data", "load_tensors",
           "load_fpo", "load_geometry", "open_samples"]

HUGGINGFACE = "BGLab/FlowBench"
ARCHIVE_URL = f"https://huggingface.co/datasets/{HUGGINGFACE}/resolve/main"

#: Read off the dataset card and stated in the paper's appendix; it covers the data, not only the code.
LICENSE = "CC-BY-NC-4.0"
LICENSE_NOTE = ("CC-BY-NC-4.0 is verified on the card and in the paper, and it is non-commercial: attribution is "
                "required and a commercial product needs the authors' permission")

#: Table 12 of the paper: the channel lists of the ``X`` and ``Y`` tensors, by family.
TENSORS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "ns": (("Re", "g", "s"), ("u", "v", "p")),
    "ns_ht": (("Re", "Gr", "g", "s"), ("u", "v", "p", "theta")),
}

#: The channels that are a property of the case rather than a field over the grid.
_SCALAR_INPUTS = ("Re", "Gr")


def read_data(path: str | os.PathLike[str]) -> np.ndarray:
    """The single ``data`` array of an archive file (they all hold exactly one)."""
    with np.load(path, allow_pickle=False) as archive:
        if "data" not in archive.files:
            raise KeyError(f"{path}: expected a 'data' dataset, found {sorted(archive.files)}")
        return np.asarray(archive["data"])


def _channels(count: int, which: int, path: Any) -> tuple[str, ...]:
    """The channel names Table 12 gives a tensor with this many channels; refuses a count it does not document."""
    table = [row[which] for row in TENSORS.values() if len(row[which]) == count]
    if not table:
        documented = {f"{len(row[which])}: {row[which]}" for row in TENSORS.values()}
        raise ValueError(f"{path}: {count} channels, which matches no row of the paper's Table 12 (documented: "
                         f"{sorted(documented)}); the reader does not guess what a channel is")
    return table[0]


def load_tensors(x_path: str | os.PathLike[str], y_path: str | os.PathLike[str] | None = None, *, index: int = 0,
                 license: str = LICENSE, units: Mapping[str, str] | None = None, sample_axis: bool = False) -> Any:
    """One sample of an ``X``/``Y`` pair as a bundle.

    ``sample_axis=False`` (the default) puts one sample in the bundle; ``True`` keeps every sample on a leading
    ``sample`` axis, which is what a full-family export wants. ``units={...}`` declares a unit per channel -- the
    archive states none, so without it every field reads ``unknown``, which is what the ``units`` gate then reports."""
    from ..schema import AxisSpec, Field, Physics, PhysicsDataBundle
    from ..schema.bundle import file_refs
    from ..schema.models import Parameter

    x = read_data(x_path)
    if x.ndim != 4:
        raise ValueError(f"{x_path}: X is {x.shape}; the archive stores (n_samples, n_channels, height, width)")
    n_samples, n_in, height, width = x.shape
    y = read_data(y_path) if y_path is not None else None
    if y is not None and (y.ndim != 4 or y.shape[0] != n_samples or y.shape[2:] != (height, width)):
        raise ValueError(f"{y_path}: Y is {y.shape}, which does not pair with X {x.shape}")

    in_names = _channels(n_in, 0, x_path)
    out_names = _channels(y.shape[1], 1, y_path) if y is not None else ()
    if index is not None and not -n_samples <= index < n_samples:
        raise IndexError(f"index {index} is outside the {n_samples} samples of {Path(x_path).name}")

    override = dict(units or {})
    spatial = [AxisSpec("y", "1"), AxisSpec("x", "1")]                     # cell index: the file has no coordinates
    axes: list[AxisSpec] = []
    if sample_axis:
        axes.append(AxisSpec("sample", "1"))
    axes += spatial
    shape = (n_samples, height, width) if sample_axis else (height, width)

    store: dict[str, np.ndarray] = {"axis:x": np.arange(width, dtype=np.float64),
                                    "axis:y": np.arange(height, dtype=np.float64)}
    if sample_axis:
        store["axis:sample"] = np.arange(n_samples, dtype=np.float64)
    fields = []
    parameters: list[Parameter] = []
    scalars_by_name: dict[str, list[float]] = {}
    for c, name in enumerate(in_names):
        values = x[:, c] if index is None else x[index, c]
        if name in _SCALAR_INPUTS:
            channel = x[:, c].reshape(n_samples, -1)                   # one value per sample, repeated over the grid
            if not np.allclose(channel, channel[:, :1]):
                raise ValueError(f"input channel '{name}' is not constant over the grid, so it is not the scalar the "
                                 "paper's Table 12 calls it")
            per_sample = [float(v) for v in channel[:, 0]]
            scalars_by_name[name] = per_sample
            if index is not None:                                      # this sample's own value
                parameters.append(Parameter(name=name, value=per_sample[index], unit="1",
                                            source="constant input channel of X"))
            elif len(set(per_sample)) == 1:                            # one value for the whole family
                parameters.append(Parameter(name=name, value=per_sample[0], unit="1",
                                            source="constant input channel of X"))
            continue
        store[f"field:{name}"] = values.reshape(shape)
        fields.append(Field.from_array(name, values.reshape(shape), unit=override.get(name, "unknown"),
                                       axes=tuple(a.name for a in axes),
                                       description=f"input channel '{name}' of the X tensor"))
    if y is not None:
        for c, name in enumerate(out_names):
            values = y[:, c] if index is None else y[index, c]
            store[f"field:{name}"] = values.reshape(shape)
            fields.append(Field.from_array(name, values.reshape(shape), unit=override.get(name, "unknown"),
                                           axes=tuple(a.name for a in axes),
                                           description=f"output channel '{name}' of the Y tensor"))

    name_root = Path(x_path).name.replace("_X.npz", "")
    inputs = [str(p) for p in (x_path, y_path) if p is not None]
    return PhysicsDataBundle.from_arrays(
        store, fields=fields, axes=axes,
        physics=Physics(parameters=parameters,
                        governing="incompressible Navier-Stokes (shifted boundary method), steady state",
                        notes="one snapshot per sample: the LDC families are steady; the transient case is FPO, "
                              "whose per-sample files carry the 242 snapshots"),
        source=f"FlowBench {name_root}" + (f" sample {index}" if index is not None else " (all samples)"),
        title=f"FlowBench {name_root}", license=license, kind="dataset",
        adapter="pinneapple_data.datasets.flowbench", adapter_version="1", inputs=file_refs(inputs),
        citation="Tali et al. (2024), FlowBench: A Large Scale Benchmark for Flow Simulation over Complex "
                 "Geometries, arXiv:2409.18032", url=ARCHIVE_URL,
        metadata={"flowbench": {"file": name_root, "index": index, "samples": n_samples,
                                "scalars_per_sample": scalars_by_name,
                                "input_channels": list(in_names), "output_channels": list(out_names),
                                "channel_source": "Table 12 of the paper, matched by channel count",
                                "coordinates": "cell indices: the archive stores no coordinate arrays",
                                "license_note": LICENSE_NOTE,
                                "units_note": "the archive states no units per field: pass units={...} to declare "
                                              "them"}})


def load_geometry(path: str | os.PathLike[str], *, license: str = LICENSE,
                  units: Mapping[str, str] | None = None, source: str = "") -> Any:
    """The ``input_geometry.npz`` of an FPO case: the distance field and the solid mask, on the same grid."""
    from ..schema import AxisSpec, Field, Physics, PhysicsDataBundle
    from ..schema.bundle import file_refs

    with np.load(path, allow_pickle=False) as archive:
        distance = np.asarray(archive["data"])
        mask = np.asarray(archive["mask"])
    if distance.shape != mask.shape or distance.ndim != 2:
        raise ValueError(f"{path}: data {distance.shape} and mask {mask.shape} are not two equal 2-D arrays")
    height, width = distance.shape
    override = dict(units or {})
    return PhysicsDataBundle.from_arrays(
        {"axis:x": np.arange(width, dtype=np.float64), "axis:y": np.arange(height, dtype=np.float64),
         "field:distance": distance, "field:mask": mask},
        fields=[Field.from_array("distance", distance, unit=override.get("distance", "unknown"), axes=("y", "x"),
                                 description="geometry field of the FPO case, as the archive stores it"),
                Field.from_array("mask", mask, unit=override.get("mask", "1"), axes=("y", "x"),
                                 description="1 inside the solid obstacle, 0 in the fluid")],
        axes=[AxisSpec("y", "1"), AxisSpec("x", "1")],
        physics=Physics(notes="the geometry of an FPO case: an implicit description of the obstacle on the flow grid"),
        source=source or f"FlowBench geometry {Path(path).parent.name}", title="FlowBench FPO geometry",
        license=license, kind="dataset", adapter="pinneapple_data.datasets.flowbench", adapter_version="1",
        inputs=file_refs([str(path)]),
        citation="Tali et al. (2024), FlowBench, arXiv:2409.18032", url=ARCHIVE_URL,
        metadata={"flowbench": {"kind": "geometry", "file": Path(path).name, "shape": list(distance.shape),
                                "license_note": LICENSE_NOTE}})


def load_fpo(path: str | os.PathLike[str], *, index: int | None = None, license: str = LICENSE,
             units: Mapping[str, str] | None = None, source: str = "") -> Any:
    """One FPO sample: ``data`` shaped ``(n_time, height, width, 3)``, the three channels being ``u, v, p``.

    ``index`` keeps one snapshot instead of the whole trajectory, which is what a steady-state comparison wants.
    The channels are last in the file -- measured on the published 1.4 GB sample, whose ``data`` is ``(242, 256,
    1024, 3)`` -- not second as in the ``X``/``Y`` tensors."""
    from ..schema import AxisSpec, Field, Physics, PhysicsDataBundle
    from ..schema.bundle import file_refs
    from ..schema.models import Parameter

    data = read_data(path)
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError(f"{path}: data is {data.shape}; an FPO sample stores (n_time, height, width, 3) with the "
                         "u, v, p channels last")
    n_t, height, width, _ = data.shape
    if index is not None and not -n_t <= index < n_t:
        raise IndexError(f"index {index} is outside the {n_t} snapshots of {Path(path).name}")
    snapshot = data if index is None else data[index]
    axes = ("t", "y", "x") if index is None else ("y", "x")
    override = dict(units or {})
    velocity = np.stack([snapshot[..., 0], snapshot[..., 1]], axis=-1)
    pressure = snapshot[..., 2]

    store = {"axis:x": np.arange(width, dtype=np.float64), "axis:y": np.arange(height, dtype=np.float64),
             "field:velocity": velocity, "field:pressure": pressure}
    specs = [AxisSpec("y", "1"), AxisSpec("x", "1")]
    if index is None:
        store["axis:t"] = np.arange(n_t, dtype=np.float64)
        specs.insert(0, AxisSpec("t", "1"))
    fields = [Field.from_array("velocity", velocity, unit=override.get("velocity", "unknown"), axes=axes,
                               components=["u", "v"], description="velocity, components u and v"),
              Field.from_array("pressure", pressure, unit=override.get("pressure", "unknown"), axes=axes,
                               description="pressure")]
    re_value = None
    stem = Path(path).stem
    if stem.startswith("Re_"):
        try:
            re_value = float(stem[3:])
        except ValueError:
            re_value = None
    return PhysicsDataBundle.from_arrays(
        store, fields=fields, axes=specs,
        physics=Physics(parameters=([Parameter(name="Re", value=re_value, unit="1", source="file name")]
                                    if re_value is not None else []),
                        governing="incompressible Navier-Stokes (shifted boundary method), transient",
                        notes="flow past an obstacle on a 1024 x 256 grid; the geometry is in input_geometry.npz"),
        source=source or f"FlowBench FPO {stem}", title=f"FlowBench FPO {stem}",
        license=license, kind="dataset", adapter="pinneapple_data.datasets.flowbench", adapter_version="1",
        inputs=file_refs([str(path)]),
        citation="Tali et al. (2024), FlowBench, arXiv:2409.18032", url=ARCHIVE_URL,
        metadata={"flowbench": {"kind": "fpo", "file": Path(path).name, "index": index, "shape": list(data.shape),
                                "channels": ["u", "v", "p"],
                                "license_note": LICENSE_NOTE,
                                "units_note": "the archive states no units per field: pass units={...}"}})


def open_samples(x_path: str | os.PathLike[str], y_path: str | os.PathLike[str] | None = None,
                 **kwargs: Any) -> Iterator[Any]:
    """Every sample of an ``X``/``Y`` pair, one bundle at a time -- the family is 1000 samples at 128 x 128."""
    count = int(read_data(x_path).shape[0])
    for index in range(count):
        yield load_tensors(x_path, y_path, index=index, **kwargs)
