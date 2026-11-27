"""Containers and the run of a field-surrogate benchmark over many geometries.

A benchmark here is a list of *samples*, one per geometry: the points the model may look at (and is asked to predict
on), the field values at those points, and whatever per-point extra channels the model can use (part labels, wall
thickness) plus, for a transient field, the times of the frames. Splits are named lists of sample ids — the official
split of the corpus, or a category split that holds a whole category out (the zero-shot experiment).

    from pinneapple_arena.fieldbench import FieldSample, FieldDataset, run_benchmark
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .metrics import ModelScore, leaderboard, score_model

__all__ = ["FieldSample", "FieldDataset", "normalize_geometry", "subsample_nodes", "run_benchmark", "write_report"]


@dataclass
class FieldSample:
    """One geometry of a field benchmark.

    ``points`` (N, 3); ``values`` (N, C) for a steady field or (T, N, C) for a transient one; ``parts`` (N,) semantic
    part labels when the structure has them; ``part_ids`` (N,) dense ids; ``thickness`` (N,) a wall thickness or any
    per-node scalar; ``extra`` (N, D) the remaining per-node channels a surrogate may read — a design vector broadcast
    to every node is the case this exists for, and it is what makes a benchmark *parametric*: without it a model can
    only interpolate between the geometries it was trained on, never answer for a design it has not seen; ``times``
    (T,) the frame times of a transient field. ``meta`` is for the record, never for the model."""
    id: str
    points: np.ndarray
    values: np.ndarray
    category: str = ""
    parts: np.ndarray | None = None
    part_ids: np.ndarray | None = None
    thickness: np.ndarray | None = None
    extra: np.ndarray | None = None
    times: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(np.asarray(self.points).shape[0])

    @property
    def transient(self) -> bool:
        return np.asarray(self.values).ndim == 3

    def inputs(self, *, with_parts: bool = True, with_extra: bool = True) -> dict:
        """The keyword arguments a surrogate's ``forward`` takes for this sample (batch of one)."""
        out = {"x": np.asarray(self.points, dtype=np.float32)[None]}
        if with_parts and self.parts is not None:
            out["part"] = np.asarray(self.parts, dtype=np.int64)[None]
        if self.part_ids is not None:
            out["part_id"] = np.asarray(self.part_ids, dtype=np.int64)[None]
        if self.thickness is not None:
            out["thickness"] = np.asarray(self.thickness, dtype=np.float32)[None]
        if self.extra is not None and with_extra:
            out["extra"] = np.asarray(self.extra, dtype=np.float32)[None]
        if self.times is not None:
            out["time"] = np.asarray(self.times, dtype=np.float32)[None]
        return out

    def target(self) -> np.ndarray:
        """The field to learn, shaped as the model outputs it: (N, C) or (T, N, C)."""
        return np.asarray(self.values, dtype=np.float32)


@dataclass
class FieldDataset:
    """A named set of samples plus the splits a benchmark scores on."""
    name: str
    samples: list[FieldSample]
    splits: dict[str, list[str]] = field(default_factory=dict)
    normalized: bool = False                          # coordinates already mapped to the unit box

    # ---------------------------------------------------------------- access
    def __post_init__(self):
        self._by_id = {s.id: s for s in self.samples}
        self._check_shape_consistency()
        if not self.splits:
            ids = [s.id for s in self.samples]
            n = len(ids)
            n_tr = max(1, int(round(0.7 * n)))
            n_va = max(1, int(round(0.15 * n))) if n >= 4 else 0
            self.splits = {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:] or ids[-1:]}

    def __len__(self) -> int:
        return len(self.samples)

    def _check_shape_consistency(self) -> None:
        """Every geometry must carry the same field shape, and a transient one must keep its frame axis.

        A benchmark batches geometries by node and frame index, so a dataset whose samples disagree cannot be trained
        or scored as one: a transient sample that lost its frame axis reads as a steady field, and the mismatch then
        surfaces as a bare numpy reshape error hours into a campaign instead of here."""
        shapes: dict[tuple, list[str]] = {}
        for s in self.samples:
            v = np.asarray(s.values)
            shapes.setdefault((v.ndim, int(v.shape[0]) if v.ndim == 3 else 0, int(v.shape[-1])), []).append(s.id)
        if len(shapes) > 1:
            detail = "; ".join(f"{key[0]}-D, {key[1]} frames, {key[2]} channels: {len(ids)} sample(s), e.g. {ids[0]}"
                               for key, ids in shapes.items())
            raise ValueError(f"dataset '{self.name}' mixes field shapes ({detail}): every sample of a benchmark has "
                             f"to share the frame count and the channel count")
        extras: dict[int, list[str]] = {}
        for s in self.samples:
            extras.setdefault(0 if s.extra is None else int(np.asarray(s.extra).reshape(len(s.points), -1).shape[1]),
                              []).append(s.id)
        if len(extras) > 1:
            detail = "; ".join(f"{d} extra channel(s): {len(ids)} sample(s), e.g. {ids[0]}"
                               for d, ids in extras.items())
            raise ValueError(f"dataset '{self.name}' mixes the extra-channel width ({detail}): a model is built for "
                             f"one input width, so a sample without the design channels cannot be batched with the "
                             f"others — broadcast the vector to every node of every sample, or drop it everywhere")

    def __getitem__(self, key) -> FieldSample:
        if isinstance(key, str):
            return self._by_id[key]
        return self.samples[key]

    @property
    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.samples:
            counts[s.category or "—"] = counts.get(s.category or "—", 0) + 1
        return counts

    def subset(self, ids) -> FieldDataset:
        return FieldDataset(name=f"{self.name}[{len(list(ids))}]", samples=[self._by_id[i] for i in ids],
                            splits=self.splits, normalized=self.normalized)

    # ---------------------------------------------------------------- splits
    def hold_out_category(self, category: str, val_fraction: float = 0.2, seed: int = 0) -> FieldDataset:
        """Everything else trains, a held-out category is the test set (the zero-shot protocol)."""
        rng = np.random.default_rng(seed)
        train, val, test = [], [], []
        for s in self.samples:
            if s.category == category:
                test.append(s.id)
            else:
                (val if rng.random() < val_fraction else train).append(s.id)
        return FieldDataset(name=f"{self.name}[no {category}]", samples=self.samples,
                            splits={"train": train, "val": val, "test": test}, normalized=self.normalized)

    def category_splits(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in self.samples:
            out.setdefault(s.category or "—", []).append(s.id)
        return out

    # ---------------------------------------------------------------- io
    def save(self, path: str) -> str:
        """One ``.npz`` per dataset: a JSON header plus the arrays (points, values and the optional channels)."""
        header = {"name": self.name, "splits": self.splits, "normalized": bool(self.normalized), "samples": []}
        arrays = {}
        for i, s in enumerate(self.samples):
            key = f"s{i}"
            header["samples"].append({"key": key, "id": s.id, "category": s.category, "meta": s.meta,
                                      "transient": bool(s.transient)})
            arrays[f"{key}_points"] = np.asarray(s.points, np.float32)
            arrays[f"{key}_values"] = np.asarray(s.values, np.float32)
            for name in ("parts", "part_ids", "thickness", "extra", "times"):
                v = getattr(s, name)
                if v is not None:
                    arrays[f"{key}_{name}"] = np.asarray(v)
        arrays["__header__"] = np.array(json.dumps(header))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str) -> FieldDataset:
        data = np.load(path, allow_pickle=False)
        header = json.loads(str(data["__header__"]))
        samples = []
        for entry in header["samples"]:
            key = entry["key"]
            s = FieldSample(id=entry["id"], points=data[f"{key}_points"], values=data[f"{key}_values"],
                            category=entry.get("category", ""), meta=entry.get("meta", {}))
            for name in ("parts", "part_ids", "thickness", "extra", "times"):
                if f"{key}_{name}" in data:
                    setattr(s, name, data[f"{key}_{name}"])
            samples.append(s)
        return cls(name=header["name"], samples=samples, splits=header["splits"],
                   normalized=header.get("normalized", False))


def normalize_geometry(dataset: FieldDataset, scale_coords: bool = True) -> FieldDataset:
    """Map the coordinates of every sample by the *same* transform (the mean centre of the corpus and the largest
    extent), so the geometry stays comparable between samples and the network sees O(1) numbers. The values are left
    untouched — the metrics are reported in physical units."""
    centres, spans = [], []
    for s in dataset.samples:
        p = np.asarray(s.points, float)
        centres.append(p.mean(axis=0))
        spans.append(p.max(axis=0) - p.min(axis=0))
    c = np.mean(centres, axis=0)
    span = float(np.max(np.asarray(spans, float)))
    span = span if span > 0 else 1.0
    out = []
    for s in dataset.samples:
        p = (np.asarray(s.points, float) - c) / (span if scale_coords else 1.0)
        out.append(FieldSample(id=s.id, points=p, values=s.values, category=s.category, parts=s.parts,
                               part_ids=s.part_ids, thickness=s.thickness, extra=s.extra, times=s.times,
                               meta={**s.meta, "coord_shift": c.tolist(), "coord_scale": span}))
    return FieldDataset(name=dataset.name, samples=out, splits=dataset.splits, normalized=True)


def subsample_nodes(dataset: FieldDataset, max_nodes: int, seed: int = 0) -> FieldDataset:
    """Keep at most ``max_nodes`` nodes per sample, at random but reproducibly, with their part labels and frames.

    A field benchmark on a 3,000-node crash mesh trains a transformer on 3,000 tokens per frame; retaining a node
    subset is what the published crash benchmark does as well ("the same retained node subset" for every row), so the
    models are compared on a subset while the dataset keeps every node for anyone who wants the full field."""
    rng = np.random.default_rng(seed)
    out = []
    for s in dataset.samples:
        n = s.n_points
        if n <= max_nodes:
            out.append(s)
            continue
        keep = np.sort(rng.choice(n, size=int(max_nodes), replace=False))
        values = np.asarray(s.values)
        values = values[:, keep] if values.ndim == 3 else values[keep]
        out.append(FieldSample(id=s.id, points=np.asarray(s.points)[keep], values=values, category=s.category,
                               parts=None if s.parts is None else np.asarray(s.parts)[keep],
                               part_ids=None if s.part_ids is None else np.asarray(s.part_ids)[keep],
                               thickness=None if s.thickness is None else np.asarray(s.thickness)[keep],
                               extra=None if s.extra is None else np.asarray(s.extra)[keep],
                               times=s.times, meta={**s.meta, "nodes_kept": int(max_nodes)}))
    return FieldDataset(name=f"{dataset.name}[{max_nodes} nodes]", samples=out, splits=dataset.splits,
                        normalized=dataset.normalized)


def run_benchmark(specs, dataset: FieldDataset, *, epochs: int = 30, batch_size: int = 2, lr: float = 1e-3,
                  device: str | None = None, patience: int = 10, seed: int = 0, standardize: bool = True,
                  verbose: bool = False, n_boot: int = 2000) -> dict:
    """Train and score every model spec on the dataset's splits; returns scores, predictions and the leaderboard.

    A spec is ``{"name", "family", "build": callable -> nn.Module, "uses_parts": bool, "uses_extra": bool,
    "notes": str}``; a row that does not read the extra channels is built without them and the trainer leaves them
    out of the batch, so a geometry-only row can be scored on a dataset that carries the design vector. Nothing about
    the architecture lives here: the caller builds the model (from the registry, or its own class), so a benchmark row
    is never a stand-in for another architecture."""
    from .trainer import predict_field_model, train_field_model

    ids_test = dataset.splits["test"]
    truths = [dataset[i].target() for i in ids_test]
    scores: list[ModelScore] = []
    predictions: dict[str, list[np.ndarray]] = {}
    histories: dict[str, dict] = {}
    for spec in specs:
        model = spec["build"]()
        hist = train_field_model(model, dataset, train_ids=dataset.splits["train"], val_ids=dataset.splits["val"],
                                 epochs=epochs, batch_size=batch_size, lr=lr, device=device, patience=patience,
                                 seed=seed, standardize=standardize, verbose=verbose,
                                 uses_parts=spec.get("uses_parts", True),
                                 uses_extra=spec.get("uses_extra", True))
        preds = predict_field_model(model, dataset, ids_test, device=hist["device"],
                                    uses_parts=spec.get("uses_parts", True),
                                    uses_extra=spec.get("uses_extra", True))
        predictions[spec["name"]] = preds
        histories[spec["name"]] = hist
        cost = {"params_m": hist["params_m"], "latency_ms": hist["latency_ms"],
                "throughput_per_s": hist["throughput_per_s"]}
        strata = [dataset[i].category or "—" for i in ids_test]
        scores.append(score_model(str(spec["name"]), str(spec.get("family", "")), preds, truths, cost=cost,
                                  notes=str(spec.get("notes", "")), n_boot=n_boot,
                                  strata=strata if len(set(strata)) > 1 else None))
        if verbose:
            print(f"{spec['name']}: Rel L2 {scores[-1].rel_l2:.4f}  R² {scores[-1].r2:.4f}")
    return {"scores": scores, "predictions": predictions, "histories": histories,
            "table": leaderboard(scores), "dataset": dataset.name, "n_test": len(ids_test)}


def write_report(result: dict, path: str) -> str:
    """Write the leaderboard and the per-geometry detail as markdown plus a JSON sidecar."""
    lines = [f"# Field-surrogate benchmark — {result['dataset']}", "",
             f"{result['n_test']} held-out geometries.", "", result["table"], ""]
    for s in result["scores"]:
        lines += [f"## {s.name}", "", f"- family: {s.family}", f"- Rel L2: {s.rel_l2:.6g}",
                  f"- R² (pooled): {s.r2:.6g}", f"- MAE: {s.mae:.6g}", f"- RMSE: {s.rmse:.6g}",
                  f"- 95% CI of the mean Rel L2: {s.extra.get('ci95')}",
                  f"- error profile: {json.dumps(s.percentiles)}", ""]
    text = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(text)
    with open(path.replace(".md", ".json"), "w") as fh:
        json.dump({"dataset": result["dataset"], "n_test": result["n_test"],
                   "scores": [{"name": s.name, "family": s.family, "rel_l2": s.rel_l2, "r2": s.r2, "mae": s.mae,
                               "rmse": s.rmse, "cost": s.cost, "ci95": s.extra.get("ci95"),
                               "percentiles": s.percentiles} for s in result["scores"]]}, fh, indent=2)
    return path
