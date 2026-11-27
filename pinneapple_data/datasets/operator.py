"""A dataset in the shape an operator model learns from: ``initial condition -> trajectory``.

An operator (FNO, DeepONet, U-NO) learns a map between *functions*, not between points. A PDEBench file or a set of
bundles holds trajectories; the adapter here turns them into the two tensors that map is trained on:

* ``inputs["ic"]`` -- the first snapshot of each trajectory, ``(N, X)``;
* ``inputs["params"]`` -- the scalar parameters that differ between trajectories (Reynolds, viscosity, beta), when
  the source records them, ``(N, P)``;
* ``targets`` -- the whole trajectory, ``(N, T, X)``.

The split is by **trajectory**, never by sample point: splitting a trajectory's own time steps between train and
test measures interpolation, which an operator that has seen the trajectory will pass and a new initial condition
will not. :meth:`OperatorDataset.leakage` checks exactly that, and the report says so.

    >>> from pinneapple_data.datasets.operator import from_pdebench
    >>> op = from_pdebench("https://darus.../1D_Burgers_Sols_Nu0.001.hdf5", samples=range(64), field="u")
    >>> op.split(0.7, 0.15).test.targets.shape
    (10, 201, 1024)
    >>> op.channels(index=0).shape            # what an FNO takes: (C, X) with the IC and any parameters
    (1, 1024)

Reading 64 trajectories over range requests is 64 reads; a local file is the better choice for a real training run
(:func:`pinneapple_data.datasets.pdebench.download`).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as _dc_field
from typing import Any

import numpy as np

__all__ = ["OperatorDataset", "Split", "from_bundles", "from_pdebench"]


@dataclass
class Split:
    """One side of a split: the initial conditions, the parameters and the trajectories that go with them."""

    ic: np.ndarray
    targets: np.ndarray
    params: np.ndarray | None = None
    trajectories: np.ndarray = _dc_field(default_factory=lambda: np.zeros(0, dtype=int))

    def __len__(self) -> int:
        return int(self.ic.shape[0])

    def channels(self) -> np.ndarray:
        """``(N, C, X)``: the initial condition, plus each parameter as a constant channel (what an FNO takes)."""
        base = self.ic[:, None, :]
        if self.params is None or self.params.size == 0:
            return base
        extra = np.repeat(self.params[:, :, None], self.ic.shape[1], axis=2)
        return np.concatenate([base, extra], axis=1)


@dataclass
class OperatorDataset:
    """Trajectories with their initial conditions, ready for an operator learning run."""

    ic: np.ndarray                     # (N, X)
    targets: np.ndarray                # (N, T, X)
    params: np.ndarray | None = None   # (N, P)
    param_names: tuple[str, ...] = ()
    field: str = ""
    source: str = ""
    license: str = ""
    ids: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return int(self.ic.shape[0])

    def __len__(self) -> int:
        return self.n

    def summary(self) -> str:
        shape = "x".join(str(d) for d in self.targets.shape)
        return (f"{self.n} trajectories of {self.field or 'a field'} ({shape}), "
                f"{len(self.param_names)} parameter(s): {', '.join(self.param_names) or 'none'} -- {self.source}")

    def split(self, train: float = 0.7, val: float = 0.15, *, seed: int = 0) -> OperatorSplits:
        """Split by trajectory (a shuffle of whole trajectories, not of their steps)."""
        if not 0 < train < 1 or val < 0 or train + val >= 1:
            raise ValueError("train and val must leave a non-empty test set: train + val < 1")
        index = np.random.default_rng(seed).permutation(self.n)
        n_train = max(1, int(round(self.n * train)))
        n_val = max(0, int(round(self.n * val)))
        parts = {"train": index[:n_train],
                 "val": index[n_train:n_train + n_val],
                 "test": index[n_train + n_val:]}
        return OperatorSplits(**{name: self._side(part) for name, part in parts.items()}, dataset=self)

    def _side(self, idx: np.ndarray) -> Split:
        return Split(ic=self.ic[idx], targets=self.targets[idx],
                     params=None if self.params is None else self.params[idx], trajectories=idx)

    def channels(self, index: int = 0) -> np.ndarray:
        """The FNO input channels of one trajectory: ``(C, X)``."""
        return self._side(np.array([index])).channels()[0]

    def leakage(self, splits: OperatorSplits) -> dict[str, Any]:
        """Whether any trajectory appears on both sides of a split (a real risk when a source repeats one)."""
        seen: dict[int, str] = {}
        clashes = []
        for name in ("train", "val", "test"):
            for i in getattr(splits, name).trajectories:
                if int(i) in seen:
                    clashes.append((int(i), seen[int(i)], name))
                seen[int(i)] = name
        return {"ok": not clashes, "clashes": clashes, "n": len(seen)}

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "field": self.field, "source": self.source, "license": self.license,
                "param_names": list(self.param_names), "shapes": {"ic": list(self.ic.shape),
                                                                 "targets": list(self.targets.shape)}}


@dataclass
class OperatorSplits:
    """Train / validation / test, split by trajectory."""

    train: Split
    val: Split
    test: Split
    dataset: OperatorDataset | None = None

    def summary(self) -> str:
        return (f"{len(self.train)} train / {len(self.val)} validation / {len(self.test)} test trajectories"
                + (f" of {self.dataset.field}" if self.dataset else ""))


def from_bundles(bundles: Iterable[Any], *, field: str = "", params: Sequence[str] = ()) -> OperatorDataset:
    """Stack bundles of the same shape into an operator dataset.

    Each bundle must hold one trajectory. ``params`` names the scalar parameters to carry as input channels; they
    are read from ``manifest.physics.parameters``, and a bundle that does not record one contributes ``nan``, which
    the dataset reports rather than silently filling with zero."""
    items = list(bundles)
    if not items:
        raise ValueError("no bundles given")
    name = field or items[0].manifest.fields.names()[0]
    arrays = [np.asarray(b.field_array(name), dtype=np.float32) for b in items]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError(f"the bundles do not share a shape: {sorted({a.shape for a in arrays})}")
    targets = np.stack(arrays)
    ics = np.stack([a[0] for a in arrays])
    param_values = None
    names: list[str] = []
    if params:
        names = list(params)
        rows = []
        for b in items:
            recorded = {p.name: p.value for p in b.manifest.physics.parameters}
            rows.append([float(recorded.get(k, np.nan)) if recorded.get(k) is not None else float("nan")
                         for k in names])
        param_values = np.asarray(rows, dtype=np.float32)
    return OperatorDataset(ic=ics, targets=targets, params=param_values, param_names=tuple(names), field=name,
                           source=", ".join(sorted({b.manifest.provenance.source for b in items})[:3]),
                           license=", ".join(sorted({b.manifest.provenance.license for b in items})),
                           ids=tuple(b.manifest.id for b in items))


def from_pdebench(source: str, *, samples: Iterable[int] = range(8), field: str = "",
                  params: Sequence[str] = ()) -> OperatorDataset:
    """Read several trajectories of one PDEBench file (a local path or a DaRUS URL) as an operator dataset."""
    from .pdebench import open_pdebench

    bundles = []
    with open_pdebench(source) as handle:
        name = field
        if not name:
            first = handle.sample(0)
            name = _preferred(first.fields)
        for index in samples:
            bundles.append(handle.to_bundle(index, hash_inputs=False))
    return from_bundles(bundles, field=name, params=params)


def _preferred(fields: Mapping[str, np.ndarray]) -> str:
    """The field to learn: the first one that actually varies (a shock tube's velocity at t=0 is zero everywhere)."""
    for name in ("u", "density", "pressure", "velocity"):
        if name in fields and float(np.ptp(fields[name])) > 0:
            return name
    for name, values in fields.items():
        if float(np.ptp(values)) > 0:
            return name
    return next(iter(fields))


def to_dict(dataset: OperatorDataset) -> dict[str, Any]:
    return asdict(dataset) if hasattr(dataset, "__dataclass_fields__") else {}
