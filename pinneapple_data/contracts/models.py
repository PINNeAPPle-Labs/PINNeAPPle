"""The four data contracts shared by the PINNeAPPle apps and the library.

=================  ===========================================================================================
``ParameterSpace``  what may vary in a study: geometry, materials, boundary conditions, operating point, with
                    bounds, units, log scaling, categories and constraints between parameters
``DatasetSpec``     a table of cases or samples on disk: columns (inputs / outputs / ids) with units, the files
                    and their SHA-256, splits, the parameter space it samples and where it came from
``ModelRef``        a trained model somewhere (zoo, file, hub): its inputs and outputs, the dataset it was
                    trained on, held-out metrics and the range it is valid in
``StudyResult``     the outcome of a study (design of experiments, campaign, UQ, sensitivity, optimisation,
                    calibration, validation): a table, metrics, the checks that ran and their records, artifacts
=================  ===========================================================================================

Every contract is a pydantic model with ``kind`` and ``schema_version`` fields, so a file says what it is::

    space = ParameterSpace(name="fin", parameters=[Parameter(name="h", low=0.01, high=0.05, unit="m")])
    space.save("fin.space.json")
    assert load("fin.space.json") == space
    ParameterSpace.json_schema()          # JSON Schema (draft 2020-12) for other languages and the REST APIs

Unknown fields are refused (a typo is an error, not silently dropped); free-form additions go in ``metadata``.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import expressions

SCHEMA_VERSION = "1.0"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Contract(_Model):
    """Base of the four contracts: JSON round trip, files, schema."""

    KIND: ClassVar[str] = ""
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = Field(default_factory=dict, description="Free-form additions (app-specific).")

    # -- serialisation -------------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, text: str):
        return cls.model_validate_json(text)

    def save(self, path: str | Path) -> Path:
        """Write JSON (``.json``) or YAML (``.yaml`` / ``.yml``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".yaml", ".yml"):
            import yaml
            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        else:
            path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path: str | Path):
        obj = load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} holds a {type(obj).__name__}, not a {cls.__name__}")
        return obj

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        schema = cls.model_json_schema(mode="validation")
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"https://pinneapple.org/schemas/{cls.KIND}-{SCHEMA_VERSION}.json"
        return schema

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON (sorted keys), for caching and lineage."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# ParameterSpace
# ---------------------------------------------------------------------------------------------------------------
ParameterKind = Literal["continuous", "integer", "categorical", "constant"]
ParameterGroup = Literal["geometry", "material", "boundary", "operating", "numerics", "other"]


class Parameter(_Model):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", description="Identifier: also the column name.")
    kind: ParameterKind = "continuous"
    low: float | None = None
    high: float | None = None
    values: list[Any] | None = Field(default=None, description="Choices of a categorical parameter.")
    value: Any = Field(default=None, description="Value of a constant parameter.")
    unit: str = ""
    log: bool = Field(default=False, description="Sample uniformly in log space (low > 0).")
    group: ParameterGroup = "other"
    default: Any = None
    description: str = ""

    @model_validator(mode="after")
    def _check(self):
        if self.kind in ("continuous", "integer"):
            if self.low is None or self.high is None:
                raise ValueError(f"parameter {self.name!r}: {self.kind} needs low and high")
            if not (math.isfinite(self.low) and math.isfinite(self.high)) or self.low >= self.high:
                raise ValueError(f"parameter {self.name!r}: needs finite low < high (got {self.low}, {self.high})")
            if self.log and self.low <= 0:
                raise ValueError(f"parameter {self.name!r}: log scale needs low > 0")
            if self.kind == "integer" and (self.low != int(self.low) or self.high != int(self.high)):
                raise ValueError(f"parameter {self.name!r}: integer bounds must be whole numbers")
        elif self.kind == "categorical":
            if not self.values or len(set(map(str, self.values))) != len(self.values):
                raise ValueError(f"parameter {self.name!r}: categorical needs distinct values")
        elif self.kind == "constant" and self.value is None:
            raise ValueError(f"parameter {self.name!r}: constant needs value")
        return self

    @property
    def free(self) -> bool:
        return self.kind != "constant"

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """Map u in [0, 1] to this parameter's values (rounded for integers, indexed for categories)."""
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        if self.kind == "constant":
            return np.full(u.shape, self.value, dtype=object if isinstance(self.value, str) else float)
        if self.kind == "categorical":
            idx = np.minimum((u * len(self.values)).astype(int), len(self.values) - 1)
            return np.asarray(self.values, dtype=object)[idx]
        if self.kind == "integer":
            n = int(self.high - self.low) + 1
            return self.low + np.minimum((u * n).astype(int), n - 1).astype(float)
        if self.log:
            x = np.exp(np.log(self.low) + u * (np.log(self.high) - np.log(self.low)))
        else:
            x = self.low + u * (self.high - self.low)
        return np.clip(x, self.low, self.high)                  # no round-off outside the bounds

    def to_unit(self, x) -> np.ndarray:
        """Inverse of :meth:`from_unit` (cell centres for integers and categories)."""
        if self.kind == "constant":
            return np.full(np.shape(x), 0.5)
        if self.kind == "categorical":
            lookup = {str(v): i for i, v in enumerate(self.values)}
            try:
                idx = np.array([lookup[str(v)] for v in np.ravel(x)], dtype=float).reshape(np.shape(x))
            except KeyError as exc:
                raise ValueError(f"parameter {self.name!r}: {exc.args[0]!r} is not one of {self.values}") from None
            return (idx + 0.5) / len(self.values)
        x = np.asarray(x, dtype=float)
        if self.kind == "integer":
            n = int(self.high - self.low) + 1
            return (x - self.low + 0.5) / n
        if self.log:
            return (np.log(x) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        return (x - self.low) / (self.high - self.low)

    def contains(self, x) -> np.ndarray:
        if self.kind == "constant":
            return np.asarray([str(v) == str(self.value) for v in np.ravel(x)]).reshape(np.shape(x))
        if self.kind == "categorical":
            allowed = {str(v) for v in self.values}
            return np.asarray([str(v) in allowed for v in np.ravel(x)]).reshape(np.shape(x))
        x = np.asarray(x, dtype=float)
        ok = (x >= self.low) & (x <= self.high)
        if self.kind == "integer":
            ok &= np.equal(np.round(x), x)
        return ok


class ParameterSpace(Contract):
    """The parameters of a study and the constraints between them."""

    KIND: ClassVar[str] = "ParameterSpace"
    kind: Literal["ParameterSpace"] = "ParameterSpace"
    name: str
    description: str = ""
    parameters: list[Parameter]
    constraints: list[str] = Field(default_factory=list,
                                   description='Expressions that must hold, e.g. "chord - 2*thickness >= 0.01".')
    budget: int | None = Field(default=None, ge=1, description="How many cases the study may run.")

    @model_validator(mode="after")
    def _check(self):
        names = [p.name for p in self.parameters]
        if not names:
            raise ValueError("a parameter space needs at least one parameter")
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"duplicate parameter names: {sorted(dup)}")
        for c in self.constraints:
            unknown = expressions.names_in(c) - set(names)
            if unknown:
                raise ValueError(f"constraint {c!r} uses unknown names {sorted(unknown)}")
        return self

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def free(self) -> list[Parameter]:
        return [p for p in self.parameters if p.free]

    @property
    def dim(self) -> int:
        """Number of free (non-constant) parameters: the dimension of the design space."""
        return len(self.free)

    def __getitem__(self, name: str) -> Parameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(name)

    def from_unit(self, unit: np.ndarray):
        """Points of the unit hypercube (n, dim) -> a pandas DataFrame of cases with every parameter as a column."""
        import pandas as pd
        unit = np.atleast_2d(np.asarray(unit, dtype=float))
        if unit.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim} columns (free parameters), got {unit.shape[1]}")
        cols, j = {}, 0
        for p in self.parameters:
            if p.free:
                cols[p.name] = p.from_unit(unit[:, j])
                j += 1
            else:
                cols[p.name] = p.from_unit(np.zeros(len(unit)))
        df = pd.DataFrame(cols)
        for p in self.parameters:
            if p.kind == "integer":
                df[p.name] = df[p.name].astype(int)
        return df

    def to_unit(self, cases) -> np.ndarray:
        """Cases (DataFrame or mapping of columns) -> (n, dim) points of the unit hypercube."""
        return np.column_stack([p.to_unit(np.asarray(cases[p.name])) for p in self.free]) if self.dim else \
            np.zeros((len(cases), 0))

    def feasible(self, cases) -> np.ndarray:
        """Boolean mask: inside every parameter's range and satisfying every constraint."""
        ok = np.ones(len(np.asarray(cases[self.parameters[0].name])), dtype=bool)
        for p in self.parameters:
            if p.name not in cases:
                raise KeyError(f"cases have no column {p.name!r}")
            ok &= p.contains(np.asarray(cases[p.name]))
        values = {p.name: _numeric_or_object(cases[p.name]) for p in self.parameters}
        for c in self.constraints:
            ok &= np.broadcast_to(expressions.evaluate(c, values).astype(bool), ok.shape)
        return ok

    @classmethod
    def from_table(cls, path: str | Path, name: str | None = None) -> ParameterSpace:
        """Read a parameter table (CSV / Parquet / HDF5) with one row per parameter: columns ``name`` and any of
        ``kind, low, high, values`` (``|``-separated), ``value, unit, log, group, default, description``."""
        from .tables import read_table
        df, _ = read_table(path)
        params = []
        for row in df.to_dict("records"):
            rec = {k: v for k, v in row.items() if not _is_missing(v) and v != ""}
            if "values" in rec and isinstance(rec["values"], str):
                rec["values"] = [_maybe_number(v) for v in rec["values"].split("|")]
            if "log" in rec:
                rec["log"] = str(rec["log"]).strip().lower() in ("1", "true", "yes")
            params.append(Parameter(**rec))
        return cls(name=name or Path(path).stem, parameters=params)

    def to_frame(self):
        """One row per parameter (the inverse of :meth:`from_table`)."""
        import pandas as pd
        rows = []
        for p in self.parameters:
            d = p.model_dump(exclude_none=True)
            if "values" in d:
                d["values"] = "|".join(map(str, d["values"]))
            rows.append(d)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# DatasetSpec
# ---------------------------------------------------------------------------------------------------------------
ColumnRole = Literal["input", "output", "id", "meta"]
FileFormat = Literal["csv", "parquet", "hdf5", "npz", "json", "vtk", "openfoam", "calculix", "other"]


class Column(_Model):
    name: str
    role: ColumnRole = "input"
    unit: str = ""
    dtype: str = "float64"
    description: str = ""


class FileRef(_Model):
    path: str = Field(description="Relative to the contract file, or a URI.")
    format: FileFormat = "other"
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size: int | None = Field(default=None, ge=0, description="Bytes.")
    rows: int | None = Field(default=None, ge=0)
    description: str = ""

    @classmethod
    def of(cls, path: str | Path, root: str | Path | None = None, **kw) -> FileRef:
        """A reference with the SHA-256, size and format of an existing file (``path`` stored relative to root)."""
        p = Path(path)
        rel = p.relative_to(root).as_posix() if root is not None else p.as_posix()
        fmt = kw.pop("format", None) or FORMAT_BY_SUFFIX.get(p.suffix.lower(), "other")
        return cls(path=rel, format=fmt, sha256=sha256_file(p), size=p.stat().st_size, **kw)


FORMAT_BY_SUFFIX = {".csv": "csv", ".parquet": "parquet", ".pq": "parquet", ".h5": "hdf5", ".hdf5": "hdf5",
                    ".npz": "npz", ".json": "json", ".vtk": "vtk", ".vtu": "vtk", ".inp": "calculix"}


class Provenance(_Model):
    app: str = ""
    study_id: str = ""
    source: str = Field(default="", description="What produced it: a campaign, a run set, an import, ...")
    run_ids: list[str] = Field(default_factory=list)
    library_version: str = ""
    created: str = Field(default_factory=_now)
    notes: str = ""


class Splits(_Model):
    """Either explicit row ids per split or fractions with a seed (and optionally the column holding ids)."""
    train: list[Any] | float | None = None
    val: list[Any] | float | None = None
    test: list[Any] | float | None = None
    seed: int | None = None
    id_column: str | None = None

    @model_validator(mode="after")
    def _check(self):
        fr = [v for v in (self.train, self.val, self.test) if isinstance(v, float)]
        if fr and (any(not 0 <= f <= 1 for f in fr) or sum(fr) > 1 + 1e-9):
            raise ValueError("split fractions must be in [0, 1] and sum to at most 1")
        return self


class DatasetSpec(Contract):
    """A table of cases or samples: what the columns are, where the files are, how it is split."""

    KIND: ClassVar[str] = "DatasetSpec"
    kind: Literal["DatasetSpec"] = "DatasetSpec"
    name: str
    version: str = "0.1.0"
    description: str = ""
    columns: list[Column] = Field(default_factory=list)
    files: list[FileRef] = Field(default_factory=list)
    n_rows: int | None = Field(default=None, ge=0)
    parameter_space: ParameterSpace | None = None
    splits: Splits | None = None
    provenance: Provenance = Field(default_factory=Provenance)
    license: str = ""
    status: Literal["skeleton", "partial", "complete"] = Field(
        default="complete", description="skeleton: planned, no outputs yet; partial: some cases have results.")

    @model_validator(mode="after")
    def _check(self):
        names = [c.name for c in self.columns]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"duplicate column names: {sorted(dup)}")
        return self

    def names(self, role: ColumnRole) -> list[str]:
        return [c.name for c in self.columns if c.role == role]

    @property
    def inputs(self) -> list[str]:
        return self.names("input")

    @property
    def outputs(self) -> list[str]:
        return self.names("output")

    @classmethod
    def from_table(cls, path: str | Path, *, name: str | None = None, outputs: list[str] = (),
                   ids: list[str] = (), meta: list[str] = (), units: dict[str, str] | None = None,
                   root: str | Path | None = None, **kw) -> DatasetSpec:
        """Describe a CSV / Parquet / HDF5 table: columns and dtypes from the file; every column not named in
        ``outputs``, ``ids`` or ``meta`` is an input."""
        from .tables import read_table
        df, attached = read_table(path)
        units = dict(units or {})
        if isinstance(attached, DatasetSpec):
            units = {**{c.name: c.unit for c in attached.columns if c.unit}, **units}
        role = {**{c: "output" for c in outputs}, **{c: "id" for c in ids}, **{c: "meta" for c in meta}}
        missing = set(role) - set(df.columns)
        if missing:
            raise KeyError(f"{path}: no columns {sorted(missing)}")
        cols = [Column(name=str(c), role=role.get(c, "input"), unit=units.get(c, ""), dtype=str(df[c].dtype))
                for c in df.columns]
        ref = FileRef.of(path, root=root, rows=len(df))
        return cls(name=name or Path(path).stem, columns=cols, files=[ref], n_rows=len(df), **kw)

    def verify_files(self, root: str | Path = ".") -> list[str]:
        """Problems with the referenced files (missing, SHA-256 mismatch); empty when all match."""
        problems = []
        for f in self.files:
            if "://" in f.path:
                continue
            p = Path(root) / f.path
            if not p.is_file():
                problems.append(f"{f.path}: missing")
            elif f.sha256 and sha256_file(p) != f.sha256:
                problems.append(f"{f.path}: SHA-256 differs from the contract")
        return problems


# ---------------------------------------------------------------------------------------------------------------
# ModelRef
# ---------------------------------------------------------------------------------------------------------------
class Variable(_Model):
    name: str
    unit: str = ""
    low: float | None = Field(default=None, description="Lower end of the validity range (inputs).")
    high: float | None = None
    description: str = ""


class DatasetRef(_Model):
    name: str
    version: str = ""
    content_hash: str = Field(default="", description="DatasetSpec.content_hash() of the training dataset.")
    uri: str = ""


class ModelRef(Contract):
    """Where a trained model is and what it may be used for."""

    KIND: ClassVar[str] = "ModelRef"
    kind: Literal["ModelRef"] = "ModelRef"
    name: str
    version: str = "0.1.0"
    model_type: str = Field(default="surrogate", description="surrogate, pinn, neural_operator, rom, gp, ...")
    uri: str = Field(description='"zoo:<name>", "hub:<repo>", a file path or an http(s) URL.')
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    framework: str = ""
    inputs: list[Variable] = Field(default_factory=list)
    outputs: list[Variable] = Field(default_factory=list)
    dataset: DatasetRef | None = None
    metrics: dict[str, float] = Field(default_factory=dict,
                                      description="Held-out metrics only (e.g. test_rel_l2), with the split named.")
    metrics_split: str = Field(default="test", description="The split the metrics were computed on.")
    trust_decision: Literal["APPROVED", "REVIEW", "REJECT", "UNCHECKED"] = "UNCHECKED"
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("metrics")
    @classmethod
    def _finite(cls, v):
        bad = [k for k, x in v.items() if not math.isfinite(x)]
        if bad:
            raise ValueError(f"metrics must be finite numbers: {bad}")
        return v

    @model_validator(mode="after")
    def _check(self):
        if self.metrics and self.metrics_split in ("train", "training"):
            raise ValueError("ModelRef metrics must come from held-out data, not the training split")
        return self

    def in_range(self, **inputs) -> dict[str, bool]:
        """Per input: is the value inside the model's validity range (True when no range is stated)."""
        out = {}
        for v in self.inputs:
            if v.name in inputs:
                x = np.asarray(inputs[v.name], dtype=float)
                ok = np.ones(x.shape, dtype=bool)
                if v.low is not None:
                    ok &= x >= v.low
                if v.high is not None:
                    ok &= x <= v.high
                out[v.name] = bool(ok.all())
        return out


# ---------------------------------------------------------------------------------------------------------------
# StudyResult
# ---------------------------------------------------------------------------------------------------------------
StudyType = Literal["doe", "campaign", "runs", "surrogate", "uncertainty", "sensitivity", "optimization",
                    "calibration", "rom", "twin", "inverse", "benchmark", "design", "validation",
                    "identifiability", "anomaly", "control", "evidence", "other"]
CheckStatus = Literal["PASS", "FAIL", "WARN", "N/A", "NOT_RUN"]


class CheckRecord(_Model):
    """One check. PASS / FAIL / WARN need a record (value or evidence): a check is only reported as run if it ran."""
    name: str
    status: CheckStatus
    value: float | str | None = None
    threshold: float | str | None = None
    evidence: str = Field(default="", description="Where the record is: a file, a run id, a log line.")
    detail: str = ""

    @model_validator(mode="after")
    def _check(self):
        if self.status in ("PASS", "FAIL", "WARN") and self.value is None and not self.evidence:
            raise ValueError(f"check {self.name!r}: status {self.status} needs a value or evidence")
        return self


class Table(_Model):
    """Small tables inline (columns of equal length); large ones as a file."""
    columns: dict[str, list[Any]] = Field(default_factory=dict)
    file: FileRef | None = None

    @model_validator(mode="after")
    def _check(self):
        lens = {len(v) for v in self.columns.values()}
        if len(lens) > 1:
            raise ValueError("table columns must have equal length")
        return self

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.columns)

    @classmethod
    def from_frame(cls, df) -> Table:
        return cls(columns={str(c): _jsonable_list(df[c]) for c in df.columns})


class StudyResult(Contract):
    """The outcome of a study, with what went in, what came out and which checks ran."""

    KIND: ClassVar[str] = "StudyResult"
    kind: Literal["StudyResult"] = "StudyResult"
    id: str
    study_type: StudyType
    app: str = ""
    title: str = ""
    status: Literal["planned", "running", "done", "failed", "partial"] = "done"
    parameter_space: ParameterSpace | None = None
    dataset: DatasetSpec | None = None
    models: list[ModelRef] = Field(default_factory=list)
    table: Table | None = None
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    checks: list[CheckRecord] = Field(default_factory=list)
    artifacts: list[FileRef] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list, description="What this result does not show (stated in UIs).")
    provenance: Provenance = Field(default_factory=Provenance)

    @property
    def passed(self) -> bool:
        """True when at least one check ran and none failed."""
        ran = [c for c in self.checks if c.status in ("PASS", "FAIL", "WARN")]
        return bool(ran) and all(c.status != "FAIL" for c in ran)


CONTRACTS: dict[str, type[Contract]] = {c.KIND: c for c in (ParameterSpace, DatasetSpec, ModelRef, StudyResult)}


def from_dict(data: dict[str, Any]) -> Contract:
    """Build the contract named by ``data["kind"]``."""
    kind = data.get("kind") if isinstance(data, dict) else None
    if kind not in CONTRACTS:
        raise ValueError(f"not a PINNeAPPle contract (kind={kind!r}); expected one of {sorted(CONTRACTS)}")
    return CONTRACTS[kind].model_validate(data)


def load(path: str | Path) -> Contract:
    """Read a contract from JSON or YAML, whichever of the four it is."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return from_dict(data)


def validate(data: dict[str, Any]) -> list[str]:
    """Problems with a contract given as a dict (empty list when valid), one readable line each."""
    from pydantic import ValidationError
    try:
        from_dict(data)
    except ValidationError as exc:
        return [f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors()]
    except ValueError as exc:
        return [str(exc)]
    return []


def json_schemas() -> dict[str, dict[str, Any]]:
    return {k: c.json_schema() for k, c in CONTRACTS.items()}


def export_schemas(directory: str | Path) -> list[Path]:
    """Write ``<Kind>.schema.json`` for each contract into ``directory``."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for kind, schema in json_schemas().items():
        p = d / f"{kind}.schema.json"
        p.write_text(json.dumps(schema, indent=1, sort_keys=True) + "\n")
        out.append(p)
    return out


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _maybe_number(s: str):
    s = s.strip()
    try:
        f = float(s)
    except ValueError:
        return s
    return int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f


def _numeric_or_object(col) -> np.ndarray:
    a = np.asarray(col)
    if a.dtype == object:
        try:
            return a.astype(float)
        except (TypeError, ValueError):
            return a.astype(str)
    return a


def _jsonable_list(col) -> list[Any]:
    out = []
    for v in np.asarray(col, dtype=object).tolist():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        out.append(v)
    return out
