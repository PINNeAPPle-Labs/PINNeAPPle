"""A campaign's results as a dataset: scalars and fields per case, with the split and the provenance.

    ds = build_dataset(records, inputs, outputs, field_names=("cp",), points=P, regions=masks)
    ds = ds.with_split(region_split(ds, holdout=lambda X: X[:, 0] > 34.0, val=0.15, test=0.15, seed=0))
    ds.save("data/ahmed")          # ahmed.npz (arrays "<split>__<column>", as pp.datasets stores them) + ahmed.json
    ds = StudyDataset.load("data/ahmed")

The JSON is a :class:`pinneapple_data.contracts.DatasetSpec` (columns with units and roles, the parameter space,
the split as explicit ids, the SHA-256 of the npz, provenance with every run id). Splits: ``train``, ``val``,
``test`` are random among the cases inside the training region; ``extrapolation`` holds every case of the held-out
region, which no model sees before its final evaluation.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

__all__ = ["StudyDataset", "build_dataset", "region_split", "SPLITS"]

SPLITS = ("train", "val", "test", "extrapolation")


@dataclass
class StudyDataset:
    ids: list[str]
    inputs: list[str]
    outputs: list[str]
    X: np.ndarray                                   # (n, n_inputs)
    Y: np.ndarray                                   # (n, n_outputs)
    fields: dict[str, np.ndarray] = field(default_factory=dict)      # name -> (n, n_points)
    points: np.ndarray | None = None                # (n_points, 3) of the baseline design
    regions: dict[str, np.ndarray] = field(default_factory=dict)     # name -> bool mask over the points
    split: dict[str, list[str]] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    purpose: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    def idx(self, split: str) -> np.ndarray:
        pos = {k: i for i, k in enumerate(self.ids)}
        return np.array([pos[k] for k in self.split.get(split, []) if k in pos], int)

    def part(self, split: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        i = self.idx(split)
        return self.X[i], self.Y[i], {k: v[i] for k, v in self.fields.items()}

    def y(self, name: str) -> np.ndarray:
        return self.Y[:, self.outputs.index(name)]

    def with_split(self, split: Mapping[str, Sequence[str]]) -> StudyDataset:
        return replace(self, split={k: list(v) for k, v in split.items()})

    def unit_x(self, X: np.ndarray | None = None) -> np.ndarray:
        """Inputs scaled to the unit cube of :attr:`bounds` (the training region's box when the bounds say so)."""
        X = self.X if X is None else np.atleast_2d(X)
        lo = np.array([self.bounds[k][0] for k in self.inputs])
        hi = np.array([self.bounds[k][1] for k in self.inputs])
        return (X - lo) / (hi - lo)

    def extend(self, other: StudyDataset, split: str = "train") -> StudyDataset:
        """This dataset plus the cases of ``other`` (same columns), added to ``split``."""
        new = [i for i, k in enumerate(other.ids) if k not in self.ids]
        sp = {k: list(v) for k, v in self.split.items()}
        sp.setdefault(split, []).extend(other.ids[i] for i in new)
        return replace(self, ids=self.ids + [other.ids[i] for i in new], X=np.vstack([self.X, other.X[new]]),
                       Y=np.vstack([self.Y, other.Y[new]]),
                       fields={k: np.vstack([v, other.fields[k][new]]) for k, v in self.fields.items()},
                       purpose=self.purpose + [other.purpose[i] for i in new] if other.purpose else self.purpose,
                       split=sp)

    # ------------------------------------------------------------------ io
    def save(self, stem: str, name: str | None = None, description: str = "", license: str = "Apache-2.0") -> str:
        """``<stem>.npz`` + ``<stem>.json`` (DatasetSpec); returns the JSON path."""
        from pinneapple_data.contracts import Column, DatasetSpec, FileRef, Provenance, Splits
        os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
        arrays: dict[str, np.ndarray] = {"ids": np.array(self.ids), "purpose": np.array(self.purpose or [""] * len(self))}
        for s in SPLITS:
            i = self.idx(s)
            for j, c in enumerate(self.inputs):
                arrays[f"{s}__{c}"] = self.X[i, j]
            for j, c in enumerate(self.outputs):
                arrays[f"{s}__{c}"] = self.Y[i, j]
            for k, v in self.fields.items():
                arrays[f"{s}__field_{k}"] = v[i]
            arrays[f"{s}__id"] = np.array(self.split.get(s, []))
        arrays["all__X"], arrays["all__Y"] = self.X, self.Y
        for k, v in self.fields.items():
            arrays[f"all__field_{k}"] = v
        if self.points is not None:
            arrays["points"] = self.points
        for k, v in self.regions.items():
            arrays[f"region__{k}"] = np.asarray(v, bool)
        np.savez_compressed(stem + ".npz", **arrays)
        cols = [Column(name=c, role="input", unit=self.units.get(c, "")) for c in self.inputs]
        cols += [Column(name=c, role="output", unit=self.units.get(c, "")) for c in self.outputs]
        cols += [Column(name=f"field_{k}", role="output", dtype=f"float64[{v.shape[1]}]",
                        description=f"{k} on {v.shape[1]} fixed surface points ('points' in the npz)")
                 for k, v in self.fields.items()]
        cols.append(Column(name="id", role="id", dtype="str"))
        from pinneapple_design.design_optimizer.study import Variable

        from .campaign import space_from_variables
        space = space_from_variables([Variable(k, *self.bounds[k], unit=self.units.get(k, "")) for k in self.inputs])
        prov = self.provenance
        spec = DatasetSpec(name=name or os.path.basename(stem), description=description, columns=cols,
                           files=[FileRef.of(stem + ".npz", root=os.path.dirname(os.path.abspath(stem)), rows=len(self))],
                           n_rows=len(self), parameter_space=space,
                           splits=Splits(train=self.split.get("train", []), val=self.split.get("val", []),
                                         test=self.split.get("test", []), id_column="id"),
                           provenance=Provenance(app=prov.get("app", "pinneapple_autolab.study"),
                                                 study_id=prov.get("study_id", ""), source=prov.get("source", ""),
                                                 run_ids=list(self.ids), library_version=prov.get("version", ""),
                                                 notes=prov.get("notes", "")),
                           license=license)
        d = json.loads(spec.model_dump_json())
        d["x_pinneapple_study"] = {"extrapolation": self.split.get("extrapolation", []),
                                   "holdout_rule": prov.get("holdout_rule", ""), "units": self.units,
                                   "bounds": {k: list(v) for k, v in self.bounds.items()}}
        p = stem + ".json"
        json.dump(d, open(p, "w"), indent=1)
        return p

    @classmethod
    def load(cls, stem: str) -> StudyDataset:
        spec = json.load(open(stem + ".json"))
        z = np.load(stem + ".npz", allow_pickle=False)
        cols = spec["columns"]
        inputs = [c["name"] for c in cols if c["role"] == "input"]
        outputs = [c["name"] for c in cols if c["role"] == "output" and not c["name"].startswith("field_")]
        fnames = [c["name"][6:] for c in cols if c["name"].startswith("field_")]
        x = spec.get("x_pinneapple_study", {})
        split = {k: list(spec["splits"].get(k) or []) for k in ("train", "val", "test")}
        split["extrapolation"] = list(x.get("extrapolation", []))
        return cls(ids=[str(s) for s in z["ids"]], inputs=inputs, outputs=outputs, X=z["all__X"], Y=z["all__Y"],
                   fields={k: z[f"all__field_{k}"] for k in fnames},
                   points=z["points"] if "points" in z else None,
                   regions={k[8:]: z[k] for k in z.files if k.startswith("region__")}, split=split,
                   units=x.get("units", {}), bounds={k: tuple(v) for k, v in x.get("bounds", {}).items()},
                   provenance=spec.get("provenance", {}), purpose=[str(s) for s in z["purpose"]])


def build_dataset(records: Sequence[Any], inputs: Sequence[str], outputs: Sequence[str], *,
                  field_names: Sequence[str] = (), points: np.ndarray | None = None,
                  regions: Mapping[str, np.ndarray] | None = None, units: Mapping[str, str] | None = None,
                  bounds: Mapping[str, tuple[float, float]] | None = None,
                  provenance: Mapping[str, Any] | None = None) -> StudyDataset:
    """From :class:`~pinneapple_autolab.ledger.RunRecord` objects (successful ones only; fields read from each
    record's ``fields_path``)."""
    ok = [r for r in records if r.ok]
    X = np.array([[r.params[k] for k in inputs] for r in ok], float)
    Y = np.array([[r.outputs[k] for k in outputs] for r in ok], float)
    F = {k: np.stack([np.load(r.fields_path)[k] for r in ok]) for k in field_names}
    b = dict(bounds or {k: (float(X[:, j].min()), float(X[:, j].max())) for j, k in enumerate(inputs)})
    return StudyDataset([r.id for r in ok], list(inputs), list(outputs), X.reshape(len(ok), len(inputs)),
                        Y.reshape(len(ok), len(outputs)), F, points, dict(regions or {}), {}, dict(units or {}), b,
                        dict(provenance or {}), [r.purpose for r in ok])


def region_split(ds: StudyDataset, holdout: Callable[[np.ndarray], np.ndarray] | None = None, *, val: float = 0.15,
                 test: float = 0.15, seed: int = 0, fixed: Mapping[str, Sequence[str]] | None = None) -> dict[str, list[str]]:
    """Every case with ``holdout(X)`` true goes to ``extrapolation``; the others are shuffled into train / val /
    test. ``fixed``: ids forced into a split (e.g. a validation sweep into ``test``)."""
    ids = np.array(ds.ids)
    out_mask = np.zeros(len(ids), bool) if holdout is None else np.asarray(holdout(ds.X), bool)
    forced = {k: set(v) for k, v in (fixed or {}).items()}
    taken = set().union(*forced.values()) if forced else set()
    pool = [i for i in np.flatnonzero(~out_mask) if ids[i] not in taken]
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    n_val, n_test = int(round(val * len(pool))), int(round(test * len(pool)))
    sp = {"val": [ids[i] for i in pool[:n_val]], "test": [ids[i] for i in pool[n_val:n_val + n_test]],
          "train": [ids[i] for i in pool[n_val + n_test:]],
          "extrapolation": [ids[i] for i in np.flatnonzero(out_mask) if ids[i] not in taken]}
    for k, v in forced.items():
        sp.setdefault(k, []).extend(sorted(v))
    return {k: sorted(v) for k, v in sp.items()}
