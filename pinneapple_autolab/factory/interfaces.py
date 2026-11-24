"""The four interfaces of the simulation factory, and the implementation of each that reuses what the library has.

A factory run is ``CAD -> DOE -> solver -> dataset`` with quality control over every arrow (#152). Each arrow is an
interface, and each has an implementation here that **delegates** to a module that already exists, so the factory
adds the wiring and the evidence, never a second way to sample, simulate or store:

================  ==========================================================  ==================================
interface         contract                                                    implementation here
================  ==========================================================  ==================================
``Geometry``      ``prepare(params, workdir) -> GeometryArtifact``             :class:`TemplateGeometry`
``Doe``           ``plan(variables, n, seed) -> list of designs``               :class:`LatinHypercubeDoe`,
                                                                               :class:`MaximinDoe`,
                                                                               :class:`GivenDesigns`
solver            ``pinneapple_autolab.simulator.Simulator`` (``check``/``run``) the lab's own, unchanged
``DatasetWriter`` ``write(rows, spec, folder) -> DatasetRef``                     :class:`StudyDatasetWriter`
================  ==========================================================  ==================================

The solver is not a new interface: a :class:`~pinneapple_autolab.simulator.Simulator` already says ``check``
(before the run) and ``run`` (the outputs, a failure with its reason, the cost and the fields), and the factory
calls it as it is. What the factory adds is the *input mapping* — the study's variables in the tool's own names,
``Evaluator.tool_params`` of ``pinneapple-analysis/design_study``, so a case described for ``run_study`` runs here
without being described twice.

What is not assumed
-------------------
* a geometry the template refuses is never simulated: the parameters are checked against the template's published
  limits first, and a refusal is a row with its reason, not a dropped row;
* when the CAD kernel is absent (the optional ``cadquery`` extra) the artifact says ``status="parameters only"``
  and the checks that need a solid are ``not_run`` with that reason. A check that did not run is not a pass;
* a design the DOE cannot place inside its bounds, or a writer asked for a row without provenance, raises here
  instead of producing a dataset that cannot be traced back (#39).
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pinneapple_analysis.trust.trust_report import CONTRADICTS, NOT_RUN, SUPPORTS, Check

__all__ = [
    "Geometry", "GeometryArtifact", "TemplateGeometry", "AsIsGeometry", "Doe", "LatinHypercubeDoe", "MaximinDoe",
    "GivenDesigns", "DatasetWriter", "DatasetRef", "StudyDatasetWriter", "ProvenanceError", "design_space",
    "template_names", "BUILT", "PARAMETERS_ONLY", "REFUSED",
]

#: ``status`` of a :class:`GeometryArtifact`: the solid and its files exist, only the parametric definition was
#: checked (no CAD kernel in this environment), or the template refused the parameters.
BUILT, PARAMETERS_ONLY, REFUSED = "built", "parameters only", "refused"


def _has_cadquery() -> bool:
    import importlib.util
    return importlib.util.find_spec("cadquery") is not None


def template_names() -> list[str]:
    """The keys of ``pinneapple_design.fitmesh.TEMPLATES``, sorted (what a ``base.ref`` may name)."""
    from pinneapple_design.fitmesh import TEMPLATES
    return sorted(TEMPLATES)


@dataclass
class GeometryArtifact:
    """One design as a geometry: what the solver reads, and the checks that decide whether it may be simulated.

    ``params`` are the template's parameters after ``complete`` (defaults and the derived sizes, so a 10-diameter
    inlet run is the one that is built, not the one that was asked for); ``metrics`` holds the numbers a solver or a
    mesh plan needs and cannot recompute cheaply — the exact volume, the smallest feature that drives the mesh, the
    reference length of the Reynolds number, the boundary patches and a point inside the fluid; ``files`` maps a role
    (``"step"``, ``"stl"``) to the path written.
    """

    ref: str
    params: dict[str, float]
    status: str = PARAMETERS_ONLY
    metrics: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    note: str = ""

    @property
    def ok(self) -> bool:
        """Whether the design may be handed to the solver (a refusal is the only thing that stops it)."""
        return self.status != REFUSED

    @property
    def reasons(self) -> list[str]:
        return [c.reason or c.name for c in self.checks if c.status == CONTRADICTS]

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "params": dict(self.params), "status": self.status, "metrics": dict(self.metrics),
                "files": dict(self.files), "note": self.note,
                "checks": [{"name": c.name, "status": c.status, "value": c.value, "limit": c.limit,
                            "reason": c.reason} for c in self.checks]}


@runtime_checkable
class Geometry(Protocol):
    """``prepare(params, workdir) -> GeometryArtifact`` and ``describe()``; nothing else is required of a geometry."""

    name: str

    def prepare(self, params: Mapping[str, float], workdir: str) -> GeometryArtifact: ...

    def describe(self) -> dict[str, Any]: ...


class TemplateGeometry:
    """The geometry end of the factory, on :data:`pinneapple_design.fitmesh.TEMPLATES`.

    ``ref`` is a template key, with or without the ``pp.geometries:`` prefix the study specs use
    (``"pp.geometries:pipe_bend_90"``). ``mapping`` renames the study's variables to the template's parameters when
    the two are not the same word (identity by default). What a template gives, and this class therefore gives, is
    the parametric definition with its published validity limits, the exact volume, the smallest feature, the
    reference length, the boundary patches and a point inside the fluid — the parts of CAD a solver actually needs.

    The B-rep itself comes from the template's CadQuery builder through
    :func:`pinneapple_design.fitmesh.cad.build_variant` (STEP + STL, validity, the volume against the closed form,
    the watertight check). It needs the optional ``cadquery`` extra; without it the artifact is
    ``parameters only`` and every check that needs a solid is ``not_run``, saying so.
    """

    def __init__(self, ref: str, *, mapping: Mapping[str, str] | None = None, build: bool = True,
                 accuracy: str = "design"):
        from pinneapple_design.fitmesh import TEMPLATES
        key = str(ref).strip()
        for prefix in ("pp.geometries:", "pp.geometry:", "geometries:"):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key not in TEMPLATES:
            raise ValueError(f"no geometry template {ref!r}; the templates are {sorted(TEMPLATES)}")
        self.template, self.key = TEMPLATES[key], key
        self.name = key
        self.mapping = {str(k): str(v) for k, v in dict(mapping or {}).items()}
        self.build, self.accuracy = bool(build), str(accuracy)

    # ------------------------------------------------------------------ the template's own numbers, without a kernel
    def describe(self) -> dict[str, Any]:
        t = self.template
        kernel = ("cadquery" if not self.build else ("cadquery" if _has_cadquery() else
                                                     "cadquery (not installed: the solid checks will be not_run)"))
        return {"name": self.name, "title": t.title, "physics": list(t.physics), "quantities": list(t.quantities),
                "parameters": {p.name: {"value": p.value, "unit": p.unit, "typical": list(p.typical),
                                        "limits": list(p.limits), "role": p.role} for p in t.params},
                "ports": dict(t.ports), "build": kernel,
                "references": [{"citation": r.citation, "what": r.what} for r in t.references]}

    def to_params(self, params: Mapping[str, float]) -> dict[str, float]:
        """The study's design in the template's parameter names, with the template's defaults and derived sizes."""
        named = {self.mapping.get(str(k), str(k)): float(v) for k, v in params.items()}
        unknown = sorted(set(named) - {p.name for p in self.template.params})
        if unknown:
            raise ValueError(f"{unknown} are not parameters of the {self.key!r} template; it has "
                             f"{[p.name for p in self.template.params]}")
        return self.template.complete(named)

    def prepare(self, params: Mapping[str, float], workdir: str = "") -> GeometryArtifact:
        """The template's parameters completed and checked, then the solid when a kernel is available."""
        t = self.template
        values = self.to_params(params)
        art = GeometryArtifact(ref=self.key, params=values)
        problems = t.check_params(values)
        art.checks.append(Check("parameters within the template's validity limits", CONTRADICTS if problems else
                                SUPPORTS, critical=True, value=float(len(problems)), limit=0.0,
                                source="pinneapple_design.fitmesh.catalog limits",
                                reason="; ".join(problems) + (" — the template's own reference objects do not cover "
                                                              "this design" if problems else "")))
        art.metrics["min_feature_m"] = float(t.min_feature(values)) if t.min_feature else None
        art.metrics["ref_length_m"] = float(t.ref_length(values)) if t.ref_length else None
        art.metrics["volume_m3_exact"] = float(t.volume(values)) if t.volume else None
        art.metrics["ports"] = dict(t.ports)
        art.metrics["inside"] = list(t.inside(values)) if t.inside else None
        if problems:
            art.status = REFUSED
            return art

        if not self.build:
            art.note = "the build was switched off: only the parametric definition was checked"
            art.checks.append(Check("CAD build", NOT_RUN, reason=art.note))
            return art
        if not _has_cadquery():
            art.note = ("cadquery is not installed (the [design] extra): the parametric definition, its validity "
                        "limits, the exact volume and the mesh-driving feature were checked; no solid, STEP or STL "
                        "was written")
            art.checks.append(Check("CAD build", NOT_RUN, reason=art.note))
            art.checks.append(Check("solid volume against the closed form", NOT_RUN, reason=art.note))
            art.checks.append(Check("STL watertight", NOT_RUN, reason=art.note))
            return art

        from pinneapple_design.fitmesh.cad import build_variant
        os.makedirs(workdir or ".", exist_ok=True)
        variant = build_variant(t, values, workdir, 0, accuracy=self.accuracy)
        art.status = BUILT
        art.files = {k: v for k, v in (("step", variant.step), ("stl", variant.stl)) if v}
        art.metrics["volume_m3_built"] = float(variant.volume)
        art.metrics["area_m2_built"] = float(variant.area)
        art.metrics["bbox_m"] = list(variant.bbox)
        for row in variant.checks:
            art.checks.append(Check(str(row["check"]), SUPPORTS if row["status"] == "pass" else
                                    (CONTRADICTS if row["status"] == "fail" else NOT_RUN),
                                    critical=row["status"] == "fail", value=row.get("value") if isinstance(
                                        row.get("value"), (int, float)) else None, limit=row.get("limit") if
                                    isinstance(row.get("limit"), (int, float)) else None,
                                    source="pinneapple_design.fitmesh.cad", reason=str(row.get("note", ""))))
        if any(c.status == CONTRADICTS for c in art.checks):
            art.status = REFUSED
        return art


class AsIsGeometry:
    """A base with no parametrization: the design goes to the solver as it is, and the CAD checks are ``not_run``.

    An ``stl`` or ``system`` base of the study schema has nothing here to rebuild — morphing the user's mesh is
    ``pinneapple_design.fitmesh``'s job and a system model has no solid at all — so this geometry carries the design
    through and says the parametric checks did not run, instead of borrowing a template's checks it never did.
    """

    name = "as given"

    def __init__(self, ref: str = "as given"):
        self.ref, self.name = str(ref), str(ref)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "title": "the design as given (no parametrization)", "parameters": {},
                "ports": {}, "build": "none"}

    def prepare(self, params: Mapping[str, float], workdir: str = "") -> GeometryArtifact:
        art = GeometryArtifact(ref=self.ref, params={str(k): float(v) for k, v in params.items()})
        art.note = ("the base names no parametric geometry template: no volume, no mesh-driving feature and no "
                    "boundary patches were derived for it")
        art.checks.append(Check("parameters within the template's validity limits", NOT_RUN, reason=art.note))
        return art


# --------------------------------------------------------------------------- the design of experiments
def design_space(variables: Sequence[Any]):
    """The ``DesignSpace`` of a study's variables (the class ``run_study`` samples from, not a copy of it)."""
    from pinneapple_design.design_optimizer.study import DesignSpace, Variable
    return DesignSpace([Variable(name=v.name, lower=v.lower, upper=v.upper, kind=v.kind, unit=v.unit,
                                 description=v.description) for v in variables])


def _checked(designs: Sequence[Mapping[str, Any]], names: Sequence[str], bounds: Mapping[str, tuple[float, float]]
             ) -> list[dict[str, float]]:
    """The designs as plain dicts, refusing one that misses a variable or leaves its range."""
    out = []
    for i, d in enumerate(designs):
        missing = sorted(set(names) - set(d))
        if missing:
            raise ValueError(f"design {i} misses the variable(s) {missing}; the space is {list(names)}")
        row = {k: float(d[k]) for k in names}
        for k, v in row.items():
            lo, hi = bounds[k]
            if not lo - 1e-12 <= v <= hi + 1e-12:
                raise ValueError(f"design {i}: {k} = {v:g} is outside [{lo:g}, {hi:g}]")
        out.append(row)
    return out


@runtime_checkable
class Doe(Protocol):
    """``plan(variables, n, seed) -> list of designs`` in the study's variable names, and ``info()``."""

    name: str

    def plan(self, variables: Sequence[Any], n: int, *, seed: int = 0,
             existing: Sequence[Mapping[str, float]] | None = None) -> list[dict[str, float]]: ...

    def info(self) -> dict[str, Any]: ...


class LatinHypercubeDoe:
    """The sample ``run_study`` draws its designs from: ``DesignSpace.sample`` (Latin hypercube, repaired)."""

    name = "latin hypercube"

    def __init__(self, method: str = "lhs"):
        self.method = str(method)
        self.info_: dict[str, Any] = {}

    def plan(self, variables, n, *, seed=0, existing=None):
        space = design_space(variables)
        X = space.sample(int(n), method=self.method, seed=int(seed))
        designs = space.as_dicts(X)
        if existing:
            skip = [{k: float(v) for k, v in d.items()} for d in existing]
            designs = [d for d in designs if d not in skip]
        self.info_ = {"method": self.method, "seed": int(seed), "n": len(designs),
                      "bounds": {v.name: [float(v.lower), float(v.upper)] for v in space.variables}}
        return designs

    def info(self) -> dict[str, Any]:
        return dict(self.info_, name=self.name)


class MaximinDoe:
    """The space-filling plan of :func:`pinneapple_autolab.study.campaign.plan_doe`, with its coverage metrics."""

    name = "maximin"

    def __init__(self, method: str = "maximin"):
        self.method = str(method)
        self.info_: dict[str, Any] = {}

    def plan(self, variables, n, *, seed=0, existing=None):
        from pinneapple_autolab.study.campaign import plan_doe, space_from_variables
        plan = plan_doe(space_from_variables(list(variables)), int(n), self.method, seed=int(seed),
                        existing=existing)
        names = [v.name for v in variables]
        bounds = {v.name: (float(v.lower), float(v.upper)) for v in variables}
        designs = _checked([{k: c[k] for k in names} for c in plan["cases"]], names, bounds)
        self.info_ = {"method": self.method, "seed": int(seed), "n": len(designs), "coverage": plan["coverage"],
                      "random_reference": plan["random_reference"], "bounds": {k: list(v) for k, v in
                                                                               bounds.items()}}
        return designs

    def info(self) -> dict[str, Any]:
        return dict(self.info_, name=self.name)


class GivenDesigns:
    """Replay a list of designs: a published campaign, a restart, the cases a reviewer asked for."""

    name = "given designs"

    def __init__(self, designs: Sequence[Mapping[str, float]]):
        self.designs = [dict(d) for d in designs]
        self.info_: dict[str, Any] = {}

    def plan(self, variables, n, *, seed=0, existing=None):
        names = [v.name for v in variables]
        bounds = {v.name: (float(v.lower), float(v.upper)) for v in variables}
        designs = _checked(self.designs, names, bounds)[:int(n)] if n else _checked(self.designs, names, bounds)
        self.info_ = {"method": self.name, "n": len(designs), "n_given": len(self.designs)}
        return designs

    def info(self) -> dict[str, Any]:
        return dict(self.info_, name=self.name)


# --------------------------------------------------------------------------- the dataset
class ProvenanceError(ValueError):
    """A row cannot be written to a dataset: what produced it is not recorded (issue #39)."""


@dataclass
class DatasetRef:
    """Where a dataset landed: the ``DatasetSpec`` JSON, the arrays, the per-row provenance, and the digest."""

    folder: str
    json: str = ""
    npz: str = ""
    rows: str = ""
    n_rows: int = 0
    digest: str = ""
    columns: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"folder": self.folder, "json": self.json, "npz": self.npz, "rows": self.rows,
                "n_rows": self.n_rows, "digest": self.digest, "columns": dict(self.columns)}


@runtime_checkable
class DatasetWriter(Protocol):
    """``write(rows, spec, folder) -> DatasetRef``; a writer that cannot trace a row must refuse it."""

    name: str

    def write(self, rows: Sequence[Any], spec: Any, folder: str, *, name: str = "") -> DatasetRef: ...


class StudyDatasetWriter:
    """The dataset end of the factory: a ``StudyDataset`` (scalars, units, bounds, split) plus per-row provenance.

    The arrays and the ``DatasetSpec`` come from :func:`pinneapple_autolab.study.dataset.build_dataset` and
    ``StudyDataset.save`` — the same files the Ahmed campaign and the tuned surrogate studies write — so a factory
    dataset is read by ``StudyDataset.load`` and by everything downstream of it. On top of that this writer refuses
    any row whose provenance :func:`pinneapple_data.provenance.problems` finds incomplete, and writes
    ``rows.json``: one entry per row with the design, the tool's own inputs, the geometry artifact and the quality
    control of that run. A dataset whose rows cannot say what produced them is not a dataset (#152).
    """

    name = "study dataset"

    def __init__(self, *, split: bool = True, val: float = 0.2, test: float = 0.2, seed: int = 0):
        self.split, self.val, self.test, self.seed = bool(split), float(val), float(test), int(seed)

    def write(self, rows, spec, folder, *, name="") -> DatasetRef:
        from pinneapple_autolab.study.dataset import build_dataset, region_split
        from pinneapple_data.provenance import problems

        kept = [r for r in rows if r.ok]
        if not kept:
            raise ProvenanceError("nothing to write: no row of this run produced an output")
        bad = []
        for r in kept:
            found = problems(r.provenance, uses_a_mesh=r.provenance.get("uses_a_mesh")
                             if isinstance(r.provenance, Mapping) else None)
            if found:
                bad.append(f"{r.id}: " + "; ".join(found))
        if bad:
            raise ProvenanceError("refusing to write a dataset with rows that cannot be traced back — " +
                                  " | ".join(bad))

        inputs = [v.name for v in spec.base.variables]
        outputs = spec.evaluator.output_names
        units = spec.units()
        os.makedirs(folder, exist_ok=True)
        ds = build_dataset([r.record() for r in kept], inputs, outputs, units=units,
                           bounds={v.name: (float(v.lower), float(v.upper)) for v in spec.base.variables},
                           provenance={"app": "pinneapple_autolab.factory", "study_id": spec.digest(),
                                       "source": f"{spec.evaluator.tool} ({spec.evaluator.fidelity})",
                                       "version": kept[0].provenance.get("library_version", ""),
                                       "notes": f"factory run of the study {spec.title!r}; one row per "
                                                f"reference evaluation, quality control in rows.json"})
        if self.split and len(kept) >= 5:
            ds = ds.with_split(region_split(ds, None, val=self.val, test=self.test, seed=self.seed))
        stem = os.path.join(folder, name or "dataset")
        path = ds.save(stem, name=os.path.basename(stem),
                       description=f"{spec.title} — {len(kept)} reference evaluations of "
                                   f"{spec.evaluator.tool}")
        rows_path = os.path.join(folder, "rows.json")
        with open(rows_path, "w") as f:
            json.dump([r.to_dict() for r in kept], f, indent=1, default=str)
        import hashlib
        h = hashlib.sha256()
        with open(path.replace(".json", ".npz"), "rb") as f:
            h.update(f.read())
        with open(rows_path, "rb") as f:
            h.update(f.read())
        return DatasetRef(folder=folder, json=path, npz=path.replace(".json", ".npz"), rows=rows_path,
                          n_rows=len(kept), digest="sha256:" + h.hexdigest(),
                          columns={**{k: units.get(k, "") for k in inputs},
                                   **{k: units.get(k, "") for k in outputs}})
