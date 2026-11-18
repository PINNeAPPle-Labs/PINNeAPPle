"""Design space -> dataset -> trained Large Physics Model, for a whole family of geometries.

The LPM blocks in :mod:`pinneapple_neural.lpm` know how to fit one model over a list of designs. What they do not
know is where a design comes from: which geometry was built, which simulation produced its field, and whether the
answer can be trusted. This module is that missing half. It runs the pipeline in five stages, and every stage
writes down what it did:

1. **draw** — ``draw(family, n_geometries, conditions, seed)`` samples a geometry from the family and then
   ``conditions`` operating points for it, which is the "many geometries, several simulations each" shape a
   design-space model is trained on;
2. **build** — ``build(family, spec)`` turns one :class:`DesignSpec` into the library's :class:`~.lpm.Design`:
   surface points and normals (the geometry the model *sees*), the query points, the normalised operating
   parameters, and the target field computed from the family's own reference;
3. **write** — :func:`generate_dataset` writes one ``.npz`` per design plus a ``manifest.json`` that records the
   reference each field came from (with its citation and its assumptions), the parameter bounds used for
   normalisation, and the SHA-256 of every file;
4. **fit** — :func:`train` splits **by geometry** (never by design: two operating points of the same body are the
   same body), fits with batched design-space training, and measures the held-out error, the ensemble spread, the
   constant-field baseline and the out-of-distribution report;
5. **card** — :func:`train` also writes ``model/card.json`` (schema ``pinneapple.lpm.family_card/1``) with what this
   run measured and what it did not.

Two rules hold this together and are enforced, not documented:

* a family returns its points in metres and declares one ``characteristic_length_m``; the pipeline normalises the
  geometry by it, so the neighbourhood radii of the model mean the same thing for every design of that family. The
  point cloud *is* the geometry: a taller fin, a thicker airfoil or a wider nozzle changes the points, so the
  geometry code sees it;
* ``params`` carry the **operating point only** (power, incidence, chamber pressure, inlet temperatures, load), min-
  max normalised with the family's declared bounds and recorded in the manifest, so a prediction can always be
  mapped back to physical units.

A family also declares its ``reference``: the model that produced the target field, the citation behind it and the
assumptions it does not include. That is why a held-out number from this pipeline is a statement about the model and
not about a data pipeline — and why the card can say, in the same breath, what the number does **not** cover.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

from .lpm import Design, LargePhysicsModel, fit

DATASET_SCHEMA = "pinneapple.lpm.dataset/1"
CARD_SCHEMA = "pinneapple.lpm.family_card/1"
TRUST = "not_verified_until_independent_review"

__all__ = [
    "CARD_SCHEMA",
    "DATASET_SCHEMA",
    "TRUST",
    "DesignSpec",
    "FieldFamily",
    "Reference",
    "build",
    "draw",
    "evaluate",
    "families",
    "generate_dataset",
    "load_dataset",
    "run",
    "train",
]


@dataclass(frozen=True)
class Reference:
    """Where a family's target field comes from, and what that model leaves out.

    ``implements`` is ``"analytic"`` for a closed-form answer and ``"solver"`` for a numerical one; the card reports
    it, because "held-out error against an exact answer" and "held-out error against another numerical model" are
    different claims.
    """

    name: str
    citation: str
    assumptions: tuple[str, ...]
    implements: str = "analytic"

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "citation": self.citation, "assumptions": list(self.assumptions),
                "implements": self.implements}


@dataclass(frozen=True)
class DesignSpec:
    """One simulation of a family: the geometry it was run on and the operating point it was run at."""

    geometry: dict[str, float]
    operating: dict[str, float]
    geometry_id: int = 0
    label: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "geometry_id": self.geometry_id,
                "geometry": dict(self.geometry), "operating": dict(self.operating)}


class FieldFamily:
    """A design space an LPM can cover: how to draw it, build it, and get its exact field.

    A subclass sets the class attributes and implements two methods:

    * :meth:`points_and_normals` — the surface of the body as ``(P, dim)`` points and outward unit normals, in
      metres. This is the geometry the model sees; two different geometries must produce different points.
    * :meth:`field` — the target at those same ``P`` points, ``(P, len(field_names))``, from the family's own
      reference.

    Optionally it overrides :meth:`sample_geometry` / :meth:`sample_operating` (the default draws uniformly inside
    the declared bounds) and :meth:`ood_probes` — designs outside the family, which the pipeline checks the OOD flag
    against. A family whose probes are not flagged is reported as such, never hidden.
    """

    name: ClassVar[str] = ""
    title: ClassVar[str] = ""
    domain: ClassVar[str] = ""
    dim: ClassVar[int] = 2
    field_names: ClassVar[tuple[str, ...]] = ()
    field_units: ClassVar[tuple[str, ...]] = ()
    geometry_params: ClassVar[dict[str, tuple[float, float]]] = {}
    operating_params: ClassVar[dict[str, tuple[float, float]]] = {}
    #: dimensionless geometry descriptors the point cloud cannot carry — a tube count, a shell-to-length ratio. They
    #: join the conditioning vector beside the operating point. A family that leaves out a parameter its field
    #: depends on gives the model two different bodies it cannot tell apart, and the fit pays for it.
    geometry_features: ClassVar[dict[str, tuple[float, float]]] = {}
    #: one magnitude per field, used to bring every field to order one before the loss sees it. Without it a family
    #: whose fields differ by ten orders of magnitude (a displacement in metres and a stress in pascal) trains on
    #: the larger field alone. The field written to the dataset is the physical one; the model fits the scaled one.
    field_scale: ClassVar[tuple[float, ...]] = ()
    characteristic_length_m: ClassVar[float] = 1.0
    n_points: ClassVar[int] = 96
    reference: ClassVar[Reference] = Reference("", "", ())
    #: held-out relative L2 the family's own gate asks for; reported with the measured value, never used to hide it
    tolerance: ClassVar[float] = 0.05

    # ------------------------------------------------------------------------------------------ what a family defines

    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        """Surface points ``(P, dim)`` in metres and outward unit normals ``(P, dim)``."""
        raise NotImplementedError

    def field(self, spec: DesignSpec) -> np.ndarray:
        """The target field at those points, ``(P, len(field_names))``."""
        raise NotImplementedError

    def sample_geometry(self, rng: np.random.Generator) -> dict[str, float]:
        """One geometry from the family. The default draws uniformly inside ``geometry_params``."""
        return {k: float(rng.uniform(*b)) for k, b in self.geometry_params.items()}

    def sample_operating(self, rng: np.random.Generator) -> dict[str, float]:
        """One operating point. The default draws uniformly inside ``operating_params``."""
        return {k: float(rng.uniform(*b)) for k, b in self.operating_params.items()}

    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        """Designs outside the trained family, as ``(name, spec)``. Empty when the family has none."""
        return []

    def validate(self, spec: DesignSpec) -> str:
        """Why this design cannot be built, or ``""`` when it can.

        A reference is a model with a range, and a design space that wanders outside it produces fields that are
        numerically finite and physically meaningless. A family whose reference has such a range overrides this, and
        :func:`draw` re-draws instead of writing the design into a dataset. The default accepts everything.
        """
        return ""

    # ---------------------------------------------------------------------------------------------- shared behaviour

    def geometry_feature_values(self, spec: DesignSpec) -> dict[str, float]:
        """The value of each entry of ``geometry_features`` for this design. Required when the family declares any."""
        return {}

    @property
    def conditioning_params(self) -> dict[str, tuple[float, float]]:
        """Everything the model is conditioned on beside the point cloud: the operating point and the descriptors."""
        return {**self.operating_params, **self.geometry_features}

    def conditioning(self, spec: DesignSpec) -> dict[str, float]:
        """The conditioning vector of one design, in physical units."""
        values = {**spec.operating, **self.geometry_feature_values(spec)}
        missing = [name for name in self.conditioning_params if name not in values]
        if missing:
            raise ValueError(f"{self.name}: no value for the conditioning parameters {missing}")
        return values

    def normalize_conditioning(self, spec: DesignSpec) -> np.ndarray:
        """Min-max the conditioning vector into [0, 1] with the declared bounds (out of bounds is kept, not clipped)."""
        values = self.conditioning(spec)
        out = []
        for name, (lo, hi) in self.conditioning_params.items():
            span = hi - lo
            out.append(0.0 if abs(span) < 1e-15 else (float(values[name]) - lo) / span)
        return np.asarray(out, dtype=np.float64)

    def denormalize_conditioning(self, params: np.ndarray) -> dict[str, float]:
        """The inverse of :meth:`normalize_conditioning`, so a prediction can be reported in physical units."""
        out = {}
        for value, (name, (lo, hi)) in zip(np.asarray(params, dtype=float).ravel(),
                                           self.conditioning_params.items(), strict=True):
            out[name] = lo + float(value) * (hi - lo)
        return out

    @property
    def n_params(self) -> int:
        return len(self.conditioning_params)

    @property
    def n_fields(self) -> int:
        return len(self.field_names)

    @property
    def scales(self) -> np.ndarray:
        """One scale per field: what the target is divided by before the loss sees it."""
        if not self.field_scale:
            return np.ones(self.n_fields)
        return np.asarray(self.field_scale, dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        """The family as a card or a manifest records it."""
        return {
            "name": self.name, "title": self.title, "domain": self.domain, "dim": self.dim,
            "field_names": list(self.field_names), "field_units": list(self.field_units),
            "field_scale": [float(v) for v in self.scales],
            "characteristic_length_m": self.characteristic_length_m, "n_points": self.n_points,
            "geometry_params": {k: list(v) for k, v in self.geometry_params.items()},
            "operating_params": {k: list(v) for k, v in self.operating_params.items()},
            "geometry_features": {k: list(v) for k, v in self.geometry_features.items()},
            "reference": self.reference.as_dict(),
            "tolerance": self.tolerance,
        }


# --------------------------------------------------------------------------------------------------- stages 1 and 2


def draw(family: FieldFamily, n_geometries: int, conditions_per_geometry: int = 1,
         seed: int = 0, attempts: int = 60) -> list[DesignSpec]:
    """``n_geometries`` bodies, each with ``conditions_per_geometry`` operating points.

    The geometry is drawn once and reused, which is the point of a design space: several simulations of the *same*
    body teach the model how the field moves with the operating point, and several bodies teach it the geometry.
    An operating point the family's own :meth:`FieldFamily.validate` rejects is re-drawn (up to ``attempts``), so a
    dataset never contains a design the reference cannot represent.
    """
    if n_geometries < 1:
        raise ValueError("a design space needs at least one geometry")
    if conditions_per_geometry < 1:
        raise ValueError("each geometry needs at least one operating point")
    rng = np.random.default_rng(seed)
    specs: list[DesignSpec] = []
    for gid in range(n_geometries):
        geometry = family.sample_geometry(rng)
        for c in range(conditions_per_geometry):
            reason = "no attempt made"
            for _ in range(attempts):
                operating = family.sample_operating(rng)
                candidate = DesignSpec(geometry=geometry, operating=operating, geometry_id=gid,
                                       label=f"{family.name}:g{gid:05d}:c{c:02d}")
                reason = family.validate(candidate)
                if not reason:
                    specs.append(candidate)
                    break
            else:
                raise RuntimeError(f"{family.name}: could not draw operating point {c} of geometry {gid} "
                                   f"inside the reference's range after {attempts} tries: {reason}")
    return specs


def build(family: FieldFamily, spec: DesignSpec) -> Design:
    """One :class:`~.lpm.Design`: the geometry normalised by the family's length, its field and its parameters."""
    reason = family.validate(spec)
    if reason:
        raise ValueError(f"{family.name}: {spec.label} is outside the reference's range ({reason})")
    points, normals = family.points_and_normals(spec)
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    target = np.asarray(family.field(spec), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != family.dim:
        raise ValueError(f"{family.name}: points must be (P, {family.dim}), got {points.shape}")
    if normals.shape != points.shape:
        raise ValueError(f"{family.name}: normals {normals.shape} do not match points {points.shape}")
    if target.ndim != 2 or target.shape != (points.shape[0], family.n_fields):
        raise ValueError(f"{family.name}: field must be ({points.shape[0]}, {family.n_fields}), "
                         f"got {target.shape}")
    if not (np.all(np.isfinite(target)) and np.all(np.isfinite(points))):
        raise ValueError(f"{family.name}: non-finite values in the geometry or the field ({spec.label})")
    length = float(family.characteristic_length_m)
    pts = torch.tensor(points / length, dtype=torch.float32)
    return Design(
        points=pts,
        normals=torch.tensor(normals, dtype=torch.float32),
        query=pts.clone(),
        params=torch.tensor(family.normalize_conditioning(spec), dtype=torch.float32),
        target=torch.tensor(target / family.scales, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------------------------------------- stage 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_dataset(family: FieldFamily, out_dir: str | Path, *, n_geometries: int = 32,
                     conditions_per_geometry: int = 3, seed: int = 0, verbose: bool = False) -> dict[str, Any]:
    """Run the family's reference on a drawn design space and write it as a dataset.

    The dataset is what the next stages read: one ``.npz`` per design (points, normals, query, params, target) and a
    ``manifest.json`` carrying the family, its reference, the parameter bounds, the seeds and the SHA-256 of every
    file — so a run can be reproduced and an artefact can be identified later.
    """
    out = Path(out_dir)
    designs_dir = out / "designs"
    designs_dir.mkdir(parents=True, exist_ok=True)
    # a second run into the same directory must leave the dataset it describes, not the union of both: the manifest
    # is rewritten every time, and a file it does not list is a leftover from a larger run
    for stale in designs_dir.glob("*.npz"):
        stale.unlink()
    started = time.monotonic()
    specs = draw(family, n_geometries, conditions_per_geometry, seed)
    records, fields_total, points_total = [], 0, 0
    per_geometry: dict[int, int] = {}
    for spec in specs:
        design = build(family, spec)
        name = spec.label.replace(":", "_") + ".npz"
        path = designs_dir / name
        np.savez(path,
                 points=design.points.numpy(), normals=design.normals.numpy(), query=design.query.numpy(),
                 params=design.params.numpy(), target=design.target.numpy())
        records.append({**spec.as_dict(), "file": f"designs/{name}", "sha256": _sha256(path),
                        "n_points": int(design.points.shape[0]), "n_fields": int(design.target.shape[1])})
        fields_total += int(design.target.numel())
        points_total += int(design.points.shape[0])
        per_geometry[spec.geometry_id] = per_geometry.get(spec.geometry_id, 0) + 1
        if verbose:
            print(f"  {spec.label}: {design.points.shape[0]} points, {design.target.shape[1]} fields")
    manifest = {
        "schema": DATASET_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "family": family.describe(),
        "seed": int(seed),
        "n_geometries": len(per_geometry),
        "conditions_per_geometry": int(conditions_per_geometry),
        "n_designs": len(records),
        "n_points": points_total,
        "n_values": fields_total,
        "seconds": round(time.monotonic() - started, 3),
        "designs": records,
        "note": ("every field in this dataset was produced by the family's own reference (see "
                 "family.reference); no solver was run unless that reference says implements=solver"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_dataset(dataset_dir: str | Path) -> tuple[list[Design], dict[str, Any]]:
    """Read a dataset written by :func:`generate_dataset` back into :class:`~.lpm.Design` objects."""
    root = Path(dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    designs = []
    for record in manifest["designs"]:
        with np.load(root / record["file"]) as data:
            designs.append(Design(
                points=torch.tensor(data["points"], dtype=torch.float32),
                normals=torch.tensor(data["normals"], dtype=torch.float32),
                query=torch.tensor(data["query"], dtype=torch.float32),
                params=torch.tensor(data["params"], dtype=torch.float32),
                target=torch.tensor(data["target"], dtype=torch.float32),
            ))
    return designs, manifest


# ---------------------------------------------------------------------------------------------------------- stage 4


def _split_by_geometry(manifest: dict[str, Any], holdout_fraction: float,
                       seed: int) -> tuple[list[int], list[int]]:
    """Train/hold-out indices **by geometry**: two operating points of the same body never straddle the split.

    Splitting by design instead would let the model see a body during training and be scored on the same body at
    another operating point, which reads as generalisation without being it.
    """
    ids = sorted({record["geometry_id"] for record in manifest["designs"]})
    if len(ids) < 2:
        raise ValueError("a design space needs at least two geometries to hold one out")
    order = np.random.default_rng(seed).permutation(len(ids))
    n_hold = min(max(1, round(len(ids) * holdout_fraction)), len(ids) - 1)
    held = {ids[i] for i in order[:n_hold]}
    train = [i for i, r in enumerate(manifest["designs"]) if r["geometry_id"] not in held]
    hold = [i for i, r in enumerate(manifest["designs"]) if r["geometry_id"] in held]
    return train, hold


def _relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    error = float(torch.linalg.vector_norm(pred - target))
    norm = float(torch.linalg.vector_norm(target))
    return error / norm if norm > 0 else float("nan")


def evaluate(model: LargePhysicsModel, family: FieldFamily, designs: list[Design],
             train_designs: list[Design] | None = None) -> dict[str, Any]:
    """Held-out error, ensemble spread, constant-field baseline and the OOD report, over a list of designs.

    Everything here is in the family's physical units: the model fits the scaled field, and the scale is undone
    before any error is computed, so a reported number can be compared with the reference's own units.
    """
    scales = torch.tensor(family.scales, dtype=torch.float32)
    rows, squared_error, squared_normal, base_error, base_normal = [], 0.0, 0.0, 0.0, 0.0
    spreads: list[float] = []
    baseline = None
    if train_designs:
        stacked = torch.cat([d.target * scales for d in train_designs])
        baseline = stacked.mean(0)  # the constant field: the mean target vector of the training designs
    for design in designs:
        report = model.predict(design)
        pred = report["mean"].detach().cpu() * scales
        target = design.target * scales
        error = float(torch.linalg.vector_norm(pred - target))
        norm = float(torch.linalg.vector_norm(target))
        squared_error += error ** 2
        squared_normal += norm ** 2
        if baseline is not None:
            base_error += float(torch.linalg.vector_norm(baseline.expand_as(target) - target)) ** 2
            base_normal += norm ** 2
        std = report["std"].detach().reshape(-1)
        spreads += [float(v) for v in std]
        rows.append({"relative_l2": error / norm if norm > 0 else None,
                     "max_abs_error": float((pred - target).abs().max()),
                     "mean_std": float(std.mean()), "ood": bool(report["ood"]),
                     "ood_score": float(report["ood_score"])})
    errors = sorted(r["relative_l2"] for r in rows if r["relative_l2"] is not None)
    std_t = torch.tensor(spreads) if spreads else torch.zeros(1)
    return {
        "n_designs": len(designs),
        "n_points": sum(int(d.query.shape[0]) for d in designs),
        "relative_l2": math.sqrt(squared_error / squared_normal) if squared_normal > 0 else None,
        "relative_l2_p90": errors[min(int(0.9 * (len(errors) - 1)), len(errors) - 1)] if errors else None,
        "max_abs_error": max(r["max_abs_error"] for r in rows) if rows else None,
        "constant_field_baseline_relative_l2": (math.sqrt(base_error / base_normal)
                                                if base_normal > 0 else None),
        "spread": {"mean_std": float(std_t.mean()), "p95_std": float(torch.quantile(std_t, 0.95)),
                   "max_std": float(std_t.max()),
                   "note": "standard deviation between the bagged ensemble heads, not a confidence level"},
        "per_design": rows,
    }


def _ood_report(model: LargePhysicsModel, family: FieldFamily,
                holdout: list[Design]) -> dict[str, Any]:
    """The OOD flag on the family's own outside-the-family probes and on the held-out designs."""
    probes = []
    for name, spec in family.ood_probes():
        report = model.ood(build(family, spec))
        probes.append({"name": name, "ood": bool(report["ood"]), "ood_score": float(report["ood_score"]),
                       "reasons": [str(r) for r in report["reasons"]]})
    inside = []
    for i, design in enumerate(holdout):
        report = model.ood(design)
        inside.append({"name": f"held-out design {i}", "ood": bool(report["ood"]),
                       "ood_score": float(report["ood_score"])})
    scores = sorted(r["ood_score"] for r in inside)
    return {
        "probes": probes,
        "held_out": inside,
        "probes_flagged": sum(1 for p in probes if p["ood"]),
        "probes_total": len(probes),
        "held_out_flagged": sum(1 for r in inside if r["ood"]),
        "held_out_ood_score_max": scores[-1] if scores else None,
        "held_out_ood_score_median": scores[len(scores) // 2] if scores else None,
        "threshold": "the library's own: geometry-code distance over the 95th percentile of the training "
                     "nearest-neighbour distances, above 1.5x",
        "note": "reported as measured: a probe the model does not flag is not thereby inside its validity domain",
    }


def train(dataset_dir: str | Path, out_dir: str | Path, *, epochs: int = 400, lr: float = 2e-3,
          holdout_fraction: float = 0.25, seed: int = 0, width: int = 96, n_heads: int = 4,
          code: int = 32, n_freq: int = 6, backbone: str = "attention", attn_layers: int = 2,
          attn_heads: int = 4, batch_designs: int = 32, radii: tuple[float, float, float] = (0.1, 0.4, 1.2),
          device: str | None = None, verbose: bool = False) -> dict[str, Any]:
    """Fit an LPM on a dataset and write the model and its card.

    The split is by geometry (see :func:`_split_by_geometry`); the reported error is therefore on bodies the model
    never saw, at operating points it never saw. Nothing here decides that the model is good: it measures it and
    says how.
    """
    designs, manifest = load_dataset(dataset_dir)
    family_info = manifest["family"]
    train_idx, hold_idx = _split_by_geometry(manifest, holdout_fraction, seed)
    train_designs = [designs[i] for i in train_idx]
    hold_designs = [designs[i] for i in hold_idx]
    dev = torch.device(device) if device is not None else torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = LargePhysicsModel(
        dim=int(family_info["dim"]), n_fields=len(family_info["field_names"]),
        n_params=len(family_info["operating_params"]) + len(family_info.get("geometry_features", {})),
        radii=radii, n_freq=n_freq, width=width, code=code,
        n_heads=n_heads, backbone=backbone, attn_heads=attn_heads, attn_layers=attn_layers,
    ).to(dev)
    started = time.monotonic()
    history = fit(model, train_designs, epochs=epochs, lr=lr, seed=seed,
                  batch_designs=min(batch_designs, len(train_designs)), device=dev)
    seconds = time.monotonic() - started
    model.eval()
    held = evaluate(model, _family_by_name(family_info["name"]), hold_designs, train_designs)
    ood = _ood_report(model, _family_by_name(family_info["name"]), hold_designs)
    tolerance = float(family_info.get("tolerance", 0.05))
    within = held["relative_l2"] is not None and held["relative_l2"] <= tolerance

    out = Path(out_dir)
    (out / "model").mkdir(parents=True, exist_ok=True)
    config = {"dim": int(family_info["dim"]), "n_fields": len(family_info["field_names"]),
              "n_params": (len(family_info["operating_params"])
                           + len(family_info.get("geometry_features", {}))), "radii": list(radii), "n_freq": n_freq,
              "width": width, "code": code, "n_heads": n_heads, "backbone": backbone,
              "attn_heads": attn_heads, "attn_layers": attn_layers}
    torch.save({"schema": CARD_SCHEMA, "config": config, "state_dict": model.state_dict()},
               out / "model" / "model.pt")
    card = {
        "schema": CARD_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "family": family_info,
        "dataset": {"schema": manifest["schema"], "seed": manifest["seed"],
                    "n_designs": manifest["n_designs"], "n_geometries": manifest["n_geometries"],
                    "conditions_per_geometry": manifest["conditions_per_geometry"]},
        "split": {"policy": "random by geometry, fixed by the seed: a held-out body is never a training body",
                  "seed": int(seed), "holdout_fraction": float(holdout_fraction),
                  "n_train_designs": len(train_designs), "n_holdout_designs": len(hold_designs),
                  "n_train_geometries": len({manifest["designs"][i]["geometry_id"] for i in train_idx}),
                  "n_holdout_geometries": len({manifest["designs"][i]["geometry_id"] for i in hold_idx})},
        "model": {"class": "LargePhysicsModel", "config": config,
                  "parameters": sum(p.numel() for p in model.parameters())},
        "training": {"epochs": epochs, "lr": lr, "seed": seed, "batch_designs": min(batch_designs, len(train_designs)),
                     "loss_first": history[0] if history else None, "loss_last": history[-1] if history else None,
                     "seconds": round(seconds, 3), "device": str(dev)},
        "held_out": {**{k: held[k] for k in ("relative_l2", "relative_l2_p90", "max_abs_error",
                                             "constant_field_baseline_relative_l2", "n_designs", "n_points")},
                     "per_design": held["per_design"], "tolerance": tolerance, "within_tolerance": within},
        "ensemble_spread": held["spread"],
        "ood": ood,
        "trust": {"decision": TRUST,
                  "measured_here": [
                      f"held-out relative L2 {held['relative_l2']:.4g} on {held['n_designs']} designs of "
                      f"{held['n_points']} points, tolerance {tolerance:g}" if held["relative_l2"] is not None
                      else "held-out error not computed",
                      f"constant-field baseline on the same designs: "
                      f"{held['constant_field_baseline_relative_l2']:.4g}"
                      if held["constant_field_baseline_relative_l2"] is not None else "baseline not computed",
                      f"OOD: {ood['probes_flagged']}/{ood['probes_total']} probes outside the family flagged, "
                      f"{ood['held_out_flagged']}/{len(ood['held_out'])} held-out designs flagged",
                  ],
                  "not_measured": [
                      f"the reference itself: {family_info['reference']['implements']} — "
                      f"{family_info['reference']['name']}, with the assumptions it lists",
                      "agreement with a solver or with measured data for any geometry of this family",
                      "whether the ensemble spread is calibrated: it is the spread between bootstrap heads",
                      "behaviour outside the declared geometry and operating bounds of the family",
                  ]},
    }
    (out / "model" / "card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    summary = {
        "family": family_info["name"], "field_names": family_info["field_names"],
        "held_out_relative_l2": held["relative_l2"], "held_out_relative_l2_p90": held["relative_l2_p90"],
        "constant_field_baseline_relative_l2": held["constant_field_baseline_relative_l2"],
        "ensemble_std_mean": held["spread"]["mean_std"],
        "ood_probes_flagged": f"{ood['probes_flagged']}/{ood['probes_total']}",
        "within_tolerance": within, "seconds": round(seconds, 1), "device": str(dev),
        "model": str(out / "model" / "model.pt"), "card": str(out / "model" / "card.json"),
    }
    if verbose:
        print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------------------------------------- stage 5


def families() -> dict[str, FieldFamily]:
    """The first five families PINNeAPPle ships, by name."""
    from .lpm_families import AIRFOIL, BRACKET, EXCHANGER, FIN, NOZZLE

    return {f.name: f for f in (FIN, AIRFOIL, NOZZLE, EXCHANGER, BRACKET)}


def _family_by_name(name: str) -> FieldFamily:
    try:
        return families()[name]
    except KeyError as exc:
        raise ValueError(f"unknown family {name!r}; known: {', '.join(families())}") from exc


def run(family: FieldFamily | str, out_dir: str | Path, *, n_geometries: int = 32,
        conditions_per_geometry: int = 3, seed: int = 0, epochs: int = 400, **train_kwargs: Any) -> dict[str, Any]:
    """Generate the dataset for a family and fit its model, end to end, under one output directory."""
    fam = _family_by_name(family) if isinstance(family, str) else family
    out = Path(out_dir)
    dataset = out / "dataset"
    manifest = generate_dataset(fam, dataset, n_geometries=n_geometries,
                                conditions_per_geometry=conditions_per_geometry, seed=seed)
    summary = train(dataset, out, seed=seed, epochs=epochs, **train_kwargs)
    summary["dataset"] = {"n_designs": manifest["n_designs"], "n_geometries": manifest["n_geometries"],
                          "seconds": manifest["seconds"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """``python -m pinneapple_neural.lpm_pipeline``: draw, generate, fit and report for one family."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("family", choices=sorted(families()), help="which family to run")
    parser.add_argument("--out", default=None, help="output directory (default: benchmarks/_out/lpm_<family>)")
    parser.add_argument("--geometries", type=int, default=32)
    parser.add_argument("--conditions", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone", choices=("attention", "mlp"), default="attention")
    parser.add_argument("--dataset-only", action="store_true", help="generate the dataset, do not fit")
    args = parser.parse_args(argv)

    fam = families()[args.family]
    out = Path(args.out or Path(__file__).resolve().parent.parent / "benchmarks" / "_out" / f"lpm_{args.family}")
    if args.dataset_only:
        manifest = generate_dataset(fam, out / "dataset", n_geometries=args.geometries,
                                    conditions_per_geometry=args.conditions, seed=args.seed, verbose=True)
        print(json.dumps({k: manifest[k] for k in ("n_designs", "n_geometries", "n_points", "seconds")}))
        return 0
    summary = run(fam, out, n_geometries=args.geometries, conditions_per_geometry=args.conditions,
                  seed=args.seed, epochs=args.epochs, backbone=args.backbone)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI test
    raise SystemExit(main())
