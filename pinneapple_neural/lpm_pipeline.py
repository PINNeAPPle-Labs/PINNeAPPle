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

Stages 3 and 4 are written in :mod:`pinneapple_neural.lpm_stages.dataset` and
:mod:`pinneapple_neural.lpm_stages.fit`, and re-exported here: the modules are split by what they do, while the names
this pipeline has always exported keep working. What is *here* is the family registry (:func:`families`,
:class:`FieldFamily`) and stages 1 and 2, which every design space needs and no dataset owns.

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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

from .lpm import Design

__all__ = [
    "CARD_SCHEMA",
    "DATASET_SCHEMA",
    "PROGRESS_SCHEMA",
    "TRUST",
    "VIEW_SCHEMA",
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
    #: where the data came from and under what licence, when it is not generated here (a published campaign)
    provenance: str = ""
    licence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "citation": self.citation, "assumptions": list(self.assumptions),
                "implements": self.implements, "provenance": self.provenance, "licence": self.licence}


@dataclass(frozen=True)
class DesignSpec:
    """One simulation of a family: the geometry it was run on and the operating point it was run at."""

    geometry: dict[str, float]
    operating: dict[str, float]
    geometry_id: int = 0
    label: str = ""
    #: which side of a **published** split this run belongs to ("train" / "test"), when the data has one. A campaign
    #: that held out shapes before publishing them fixes what "held out" means, and the pipeline must honour that
    #: instead of drawing its own split.
    split: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "geometry_id": self.geometry_id, "split": self.split,
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

    def geometry_from_unit(self, geometry: dict[str, float]) -> dict[str, float]:
        """A draw of ``geometry_params`` turned into a geometry the family's reference accepts.

        The default returns the draw unchanged. A family overrides it when a point of the parameter box is not yet a
        geometry — ``pp_fin`` rounds the fin count and derives the base width from it, and without this hook a sampler
        that draws the box directly (LHS, Sobol, the active sampler) hands the reference a geometry with a missing key.
        """
        return dict(geometry)

    def ood_probes(self) -> list[tuple[str, DesignSpec]]:
        """Designs outside the trained family, as ``(name, spec)``. Empty when the family has none."""
        return []

    def specs(self) -> list[DesignSpec] | None:
        """The designs of a **real-data** family: one per run of the published dataset.

        A family that generates its own data leaves this as ``None`` and :func:`draw` samples instead. A family over
        a published campaign returns its runs here — the pipeline then writes exactly those and nothing it invented.
        """
        return None

    def published_split(self, records: list[dict[str, Any]]) -> tuple[list[int], list[int]] | None:
        """The train/hold-out indices the dataset's authors defined, or ``None`` to let the pipeline draw its own.

        Honouring the published split is what makes a held-out number comparable with the campaign that produced the
        data: the shapes were held out before anyone trained on them.
        """
        marked = [i for i, record in enumerate(records) if record.get("split")]
        if not marked:
            return None
        train = [i for i, record in enumerate(records) if record.get("split") == "train"]
        hold = [i for i, record in enumerate(records) if record.get("split") == "test"]
        if not train or not hold:
            raise ValueError(f"{self.name}: the design set carries split markers but not both sides "
                             f"(train {len(train)}, test {len(hold)})")
        return train, hold

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
    published = family.specs()
    if published is not None:
        if not published:
            raise ValueError(f"{family.name}: the family publishes no designs")
        return list(published)
    rng = np.random.default_rng(seed)
    specs: list[DesignSpec] = []
    for gid in range(n_geometries):
        # A rejection can be about the geometry as much as the operating point — "the bores touch" and "the fin is
        # thinner than a cell" are decided before any operating point is drawn, so re-drawing only the operating point
        # can never help: `pp_exchanger3d` spent its 60 attempts on one impossible plate and raised. Both are re-drawn.
        for _ in range(attempts):
            geometry = family.sample_geometry(rng)
            drawn = []
            for c in range(conditions_per_geometry):
                reason = "no attempt made"
                for _ in range(attempts):
                    operating = family.sample_operating(rng)
                    candidate = DesignSpec(geometry=geometry, operating=operating, geometry_id=gid,
                                           label=f"{family.name}:g{gid:05d}:c{c:02d}")
                    reason = family.validate(candidate)
                    if not reason:
                        drawn.append(candidate)
                        break
                else:
                    break
            else:
                specs.extend(drawn)
                break
        else:
            raise RuntimeError(f"{family.name}: could not draw geometry {gid} inside the reference's range after "
                               f"{attempts} tries: {reason}")
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


# ---------------------------------------------------------------------------------------------------------- stage 5


def families() -> dict[str, FieldFamily]:
    """Every family PINNeAPPle ships, by name.

    Five generate their fields from a reference this repository implements
    (:mod:`pinneapple_neural.lpm_families`); ``pp_airfoil_rans`` fits the same model on a **published** OpenFOAM
    campaign instead (:mod:`pinneapple_neural.lpm_real_data`), which is how the methodology is shown on data nobody
    here produced.
    """
    from .lpm_families import AIRFOIL, BRACKET, EXCHANGER, FIN, NOZZLE
    from .lpm_real_data import RansAirfoil

    return {f.name: f for f in (FIN, AIRFOIL, NOZZLE, EXCHANGER, BRACKET, RansAirfoil())}


def _family_by_name(name: str) -> FieldFamily:
    """A family by name, from either registry: a dataset's manifest records the name, and both are the same pipeline."""
    from .lpm_families_3d import families_3d  # imported here: the 3-D module builds on this one

    known = {**families(), **families_3d()}
    try:
        return known[name]
    except KeyError as exc:
        raise ValueError(f"unknown family {name!r}; known: {', '.join(known)}") from exc


def run(family: FieldFamily | str, out_dir: str | Path, *, n_geometries: int = 32,
        conditions_per_geometry: int = 3, seed: int = 0, epochs: int = 400, progress_every_s: float = 0.0,
        **train_kwargs: Any) -> dict[str, Any]:
    """Generate the dataset for a family and fit its model, end to end, under one output directory."""
    fam = _family_by_name(family) if isinstance(family, str) else family
    out = Path(out_dir)
    dataset = out / "dataset"
    manifest = generate_dataset(fam, dataset, n_geometries=n_geometries,
                                conditions_per_geometry=conditions_per_geometry, seed=seed,
                                progress_every_s=progress_every_s)
    summary = train(dataset, out, seed=seed, epochs=epochs, progress_every_s=progress_every_s, **train_kwargs)
    summary["dataset"] = {"n_designs": manifest["n_designs"], "n_geometries": manifest["n_geometries"],
                          "seconds": manifest["seconds"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """``python -m pinneapple_neural.lpm_pipeline``: draw, generate, fit and report for one family."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    from .lpm_families_3d import families_3d
    parser.add_argument("family", choices=sorted({**families(), **families_3d()}), help="which family to run")
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

# ----------------------------------------------------------------------------------------------- stages 3, 4 and 5
# The writing and the fitting live in ``lpm_stages``; these names are what this module has always exported, and a
# caller that imports from here goes on working. The import sits at the bottom because the two modules read the
# registry above -- lazily, inside the functions that need it, which is what keeps this from being a cycle.
from .lpm_stages.dataset import (  # noqa: E402, F401  (re-exported: the pipeline's own surface)
    DATASET_SCHEMA,
    PROGRESS_SCHEMA,
    _progress,
    _write_design,
    extend_dataset,
    generate_dataset,
    load_dataset,
)
from .lpm_stages.fit import (  # noqa: E402, F401  (re-exported: the pipeline's own surface)
    CARD_SCHEMA,
    TRUST,
    VIEW_SCHEMA,
    _ood_report,
    _relative_l2,
    _split_by_geometry,
    _view,
    _warm_start,
    evaluate,
    load_model,
    train,
)
