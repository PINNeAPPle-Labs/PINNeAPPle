"""Provenance: the minimum a dataset, a run or a model must record to be reproducible, and the validator (#39).

A dataset, an experiment result and a model card are all claims of the form "these numbers came from this physics,
computed this way". Without a record of *what* computed them, *which* code was running, *which* randomness was in
play and *which* exact bytes went in, the claim cannot be re-derived -- and neither a reviewer nor the person who
wrote it six months earlier can tell whether a number that looks wrong is a bug or a different case.

The minimum, and why each field is there:

===================  ==========================================================================
``solver``           what produced the numbers (``"openfoam-simplefoam"``, ``"pinn"``, ``"closed-form"``)
``library_version``  which PINNeAPPle computed them (a released version is a reproducible reference)
``seed``             which randomness was in play -- an integer, or ``"deterministic"`` for a solver
                     that has none (steady RANS, closed form, FEM assembly)
``data_hash``        a way back to the exact inputs: the SHA-256 of the input file(s), or of the
                     canonical bytes of a synthetic problem
``commit``           which code was running, when the record was produced from a checkout (a
                     released ``library_version`` and a ``commit`` are both references to code; at
                     least the version is mandatory, and this is recorded whenever it is knowable)
===================  ==========================================================================

A mesh-based solver additionally records ``mesh`` (cells or nodes, and the hash of the mesh file): a case that ran
on "the mesh" and does not say which one is not reproducible even with everything else.

Usage::

    from pinneapple_data.provenance import build_provenance, require

    record = build_provenance("openfoam-simplefoam", seed="deterministic", data_hash_of="case/0/U", mesh={"cells": 1_200_000})
    require(record)                     # raises ValueError listing exactly what is missing

``scripts/validate_provenance.py`` runs the same check over every model card and dataset in the tree, so a
repository that gains an artifact without provenance fails a job rather than looking complete.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

__all__ = ["MINIMAL", "MESH_SOLVERS", "REQUIRED_WHEN_MESHED", "data_hash", "git_commit", "library_version",
           "build_provenance", "problems", "require", "solver_uses_a_mesh"]

#: The fields every record needs. ``mesh`` is required on top of these when the solver discretises a mesh.
MINIMAL = ("solver", "library_version", "seed", "data_hash", "commit")

#: ``mesh`` is mandatory for these: the answer depends on the cells they built, not only on the solver.
REQUIRED_WHEN_MESHED = ("mesh",)

#: Solvers whose answer is a function of the mesh. Matched case-insensitively against the start of ``solver``,
#: so ``"openfoam-simplefoam"`` and ``"fenics"`` both count. A solver not listed here is not assumed mesh-free; the
#: record can still carry ``mesh``, and a producer that knows better passes ``uses_a_mesh=True`` to
#: :func:`problems`.
MESH_SOLVERS = frozenset({"fenics", "openfoam", "calculix", "code_aster", "code-aster", "gmsh", "su2", "fluent",
                          "ccx", "elmer", "ansys", "abaqus", "star-ccm", "fvm", "fem", "fdm"})


def library_version() -> str:
    """The version of PINNeAPPle in this interpreter (``""`` if it cannot be determined)."""
    try:
        import pinneapple
        return str(getattr(pinneapple, "__version__", "") or "")
    except Exception:  # pragma: no cover - only when the package is not importable
        return ""


def git_commit(path: Optional[str] = None) -> str:
    """The commit of the checkout that contains ``path`` (default: this file), or ``""`` outside a git tree.

    ``PINNEAPPLE_COMMIT`` wins when it is set: a container or a CI runner that knows the commit (and strips
    ``.git``) records it that way, which is the case the environment variable exists for.
    """
    from_env = os.environ.get("PINNEAPPLE_COMMIT", "").strip()
    if from_env:
        return from_env
    start = Path(path).resolve() if path else Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "-C", str(start), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git absent or slow filesystem
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def data_hash(path: str | Path) -> str:
    """SHA-256 of a file, in the form the records carry (``"sha256:<hex>"``)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


# the module-level hasher, captured so that ``build_provenance``'s ``data_hash`` keyword cannot shadow it
_file_hash = data_hash


def data_hash_of_bytes(payload: bytes) -> str:
    """SHA-256 of the canonical bytes of a synthetic input (a problem description, an array), same form."""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def solver_uses_a_mesh(solver: Any) -> bool:
    """Whether ``solver`` is one of the mesh-based ones (:data:`MESH_SOLVERS`), by name prefix."""
    name = str(solver or "").strip().lower().replace("_", "-")
    return any(name == s or name.startswith(s + "-") or name.startswith(s + ":") for s in
               {s.replace("_", "-") for s in MESH_SOLVERS})


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def problems(record: Optional[Mapping[str, Any]], *, uses_a_mesh: Optional[bool] = None) -> list[str]:
    """What is missing from a provenance ``record``, one readable line each (empty list = complete).

    ``uses_a_mesh`` overrides the guess from the solver name: a record for a solver this module does not know can
    still declare that the answer depends on the mesh.
    """
    if record is None:
        return [f"no provenance at all: a record needs {', '.join(MINIMAL)}"]
    if not isinstance(record, Mapping):
        return [f"provenance must be a mapping, got {type(record).__name__}"]
    out: list[str] = []
    for field in MINIMAL:
        if _missing(record.get(field)):
            out.append(f"{field} is missing: " + {
                "solver": "what produced the numbers (e.g. 'openfoam-simplefoam', 'pinn', 'closed-form')",
                "library_version": "which PINNeAPPle computed them",
                "seed": "the seed, or the string 'deterministic' for a solver with no randomness",
                "data_hash": "a way back to the exact inputs ('sha256:<hex>')",
                "commit": "which code was running ('PINNEAPPLE_COMMIT' or the checkout's HEAD)",
            }[field])
    seed = record.get("seed")
    if not _missing(seed) and not (isinstance(seed, int) or seed == "deterministic"):
        out.append(f"seed must be an integer or 'deterministic', got {seed!r}")
    meshed = solver_uses_a_mesh(record.get("solver")) if uses_a_mesh is None else bool(uses_a_mesh)
    if meshed:
        for field in REQUIRED_WHEN_MESHED:
            if _missing(record.get(field)):
                out.append(f"{field} is missing: {record.get('solver')!r} discretises a mesh, so the answer "
                           "depends on which one (give the cell count and the mesh file hash)")
    return out


def require(record: Optional[Mapping[str, Any]], *, uses_a_mesh: Optional[bool] = None) -> Mapping[str, Any]:
    """``record`` if it is complete, else ``ValueError`` listing what is missing and how to fill it (issue #39)."""
    missing = problems(record, uses_a_mesh=uses_a_mesh)
    if missing:
        raise ValueError("incomplete provenance -- " + "; ".join(missing) +
                         ". Fill it with pinneapple_data.provenance.build_provenance().")
    return record                      # type: ignore[return-value]


def build_provenance(solver: str, *, seed: int | str, data_hash: str = "", data_hash_of: str | Path | None = None,
                     mesh: Optional[Mapping[str, Any]] = None, commit: Optional[str] = None,
                     version: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """A complete provenance record, filling in what can be determined: version, commit, the hash of a file.

    ``data_hash_of`` hashes a path; ``data_hash`` takes a hash the caller already computed. ``commit=None`` looks
    for ``PINNEAPPLE_COMMIT`` and then the enclosing checkout (``commit=""`` states that it is not knowable, which
    :func:`problems` will report). Extra keyword arguments are recorded as they are (``app``, ``study_id``,
    ``source``, ...).
    """
    digest = data_hash or (_file_hash(data_hash_of) if data_hash_of is not None else "")
    record: Dict[str, Any] = {"solver": solver, "library_version": version if version is not None else library_version(),
                              "seed": seed, "data_hash": digest,
                              "commit": git_commit() if commit is None else commit}
    if mesh is not None:
        record["mesh"] = dict(mesh)
    record.update({k: v for k, v in extra.items() if v is not None})
    return record
