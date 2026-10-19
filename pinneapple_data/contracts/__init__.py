"""Open data contracts shared by the PINNeAPPle apps: ``ParameterSpace``, ``DatasetSpec``, ``ModelRef``,
``StudyResult``.

Every app imports and exports these, so a design of experiments from one app is the input of a run campaign in
another, whose finished runs become a dataset for a surrogate, whose model reference goes into an uncertainty study.
JSON (or YAML) on disk, JSON Schema for other languages (``schemas/`` in this package), tables as CSV / Parquet /
HDF5 with a ``.contract.json`` sidecar::

    from pinneapple_data.contracts import Parameter, ParameterSpace, DatasetSpec, write_table, read_table

    space = ParameterSpace(name="fin", parameters=[
        Parameter(name="height", low=0.01, high=0.05, unit="m", group="geometry"),
        Parameter(name="material", kind="categorical", values=["Al6061", "Cu"], group="material")],
        constraints=["height <= 0.04 or material == 'Al6061'"])
    cases = space.from_unit([[0.5, 0.2], [0.9, 0.8]])
    cases = cases[space.feasible(cases)]
"""
from .expressions import ExpressionError
from .models import (
    CONTRACTS,
    SCHEMA_VERSION,
    CheckRecord,
    Column,
    Contract,
    DatasetRef,
    DatasetSpec,
    FileRef,
    ModelRef,
    Parameter,
    ParameterSpace,
    Provenance,
    Splits,
    StudyResult,
    Table,
    Variable,
    export_schemas,
    from_dict,
    json_schemas,
    load,
    sha256_file,
    validate,
)
from .tables import convert, read_sidecar, read_table, sidecar_path, write_sidecar, write_table

__all__ = [
    "CONTRACTS", "SCHEMA_VERSION", "Contract", "ParameterSpace", "Parameter", "DatasetSpec", "Column", "FileRef",
    "Splits", "Provenance", "ModelRef", "Variable", "DatasetRef", "StudyResult", "CheckRecord", "Table",
    "ExpressionError", "load", "from_dict", "validate", "json_schemas", "export_schemas", "sha256_file",
    "read_table", "write_table", "convert", "sidecar_path", "read_sidecar", "write_sidecar",
]
