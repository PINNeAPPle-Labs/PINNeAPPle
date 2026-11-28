"""Stage 3's index: the datasets on disk, and which model was fitted on each.

A design space writes one dataset and one model, and the two are joined by an identity the card already records: the
SHA-256 of the manifest (:func:`~pinneapple_neural.lpm_stages.card.dataset_id`). This module reads every manifest under
a root and answers the three questions the registry cannot — what data exists at all, what was in it (designs, bodies,
points, the reference and its licence), and whether some model claims it. A dataset with no model is a design space
that was generated and never fitted; a model whose card names a dataset no longer on disk is the opposite, and both are
worth seeing in one table.

Nothing is regenerated or re-read beyond the manifest: the per-design files are named by it, and a catalogue that
opened them would cost more than the answer is worth.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .card import dataset_id, read_card, registry_row

__all__ = ["Row", "entries", "pairs", "table"]


@dataclass
class Row:
    """One dataset: what it is, what it holds, and the model that claims it."""

    family: str
    path: str
    dataset_id: str = ""
    title: str = ""
    dim: int | None = None
    fields: tuple[str, ...] = ()
    n_designs: int | None = None
    n_geometries: int | None = None
    conditions_per_geometry: int | None = None
    n_points: int | None = None
    reference: str = ""
    implements: str = ""
    licence: str = ""
    tolerance: float | None = None
    created_at: str = ""
    model: str = ""
    model_error: float | None = None
    #: how the model was attached: ``"dataset identity"`` when its card names this dataset's SHA-256 (v2 cards),
    #: ``"same run directory"`` when it merely sits beside it (v1 cards, which carry no identity to join on)
    linked_by: str = ""
    error: str = ""

    @property
    def fitted(self) -> bool:
        return bool(self.model)

    def as_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def _manifest_paths(root: Path) -> list[Path]:
    """Every ``manifest.json`` under ``root`` that is a dataset manifest (not a run's or a tool's).

    ``examples/`` and ``benchmarks/`` write plenty of files called ``manifest.json``, so the test is what a dataset
    manifest holds: the schema it declares, or the two keys a design space always records.
    """
    out = []
    for path in sorted(root.rglob("manifest.json")):
        try:
            head = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(head, dict):
            continue
        if str(head.get("schema", "")).startswith("pinneapple.lpm.dataset/") or \
                ("family" in head and ("n_designs" in head or "designs" in head)):
            out.append(path)
    return out


def _row(path: Path, models: dict[str, dict[str, Any]], beside: dict[str, dict[str, Any]]) -> Row:
    family = path.parent.parent.name if path.parent.name == "dataset" else path.parent.name
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        identity = dataset_id(path)
    except (OSError, ValueError) as exc:
        return Row(family=family, path=str(path), error=str(exc))
    info = manifest.get("family") or {}
    reference = info.get("reference") or {}
    model_row = models.get(identity)
    linked_by = "dataset identity" if model_row else ""
    if model_row is None:
        run = path.parent.parent if path.parent.name == "dataset" else path.parent
        model_row = beside.get(str(run))
        linked_by = "same run directory" if model_row else ""
    return Row(
        family=info.get("name") or family,
        path=str(path),
        dataset_id=identity,
        title=str(info.get("title") or ""),
        dim=info.get("dim"),
        fields=tuple(info.get("field_names") or ()),
        n_designs=manifest.get("n_designs"),
        n_geometries=manifest.get("n_geometries"),
        conditions_per_geometry=manifest.get("conditions_per_geometry"),
        n_points=manifest.get("n_points"),
        reference=str(reference.get("name") or ""),
        implements=str(reference.get("implements") or ""),
        licence=str(reference.get("licence") or ""),
        tolerance=info.get("tolerance"),
        created_at=str(manifest.get("created_at") or ""),
        model=str(model_row.get("family") or "") if model_row else "",
        model_error=model_row.get("relative_l2") if model_row else None,
        linked_by=linked_by,
    )


def _models(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """The cards under ``root``: by the dataset identity each names, and by the run directory each sits in.

    The second map exists because a v1 card carries no identity to join on. Pairing it by the directory it was
    written in is weaker than pairing by hash and the row says which one it was — but refusing to pair at all would
    hide the ten models of the first campaign behind a schema change.
    """
    by_id: dict[str, dict[str, Any]] = {}
    beside: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("card.json")):
        try:
            row = registry_row(read_card(path), path)
        except (OSError, ValueError):
            continue
        if row.get("dataset_id"):
            by_id[row["dataset_id"]] = row
        beside[str(path.parent.parent)] = row
    return by_id, beside


def entries(root: str | Path) -> list[Row]:
    """Every dataset under ``root``, newest first, each with the model whose card names it."""
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"{base} is not a directory")
    by_id, beside = _models(base)
    rows = [_row(path, by_id, beside) for path in _manifest_paths(base)]
    rows.sort(key=lambda r: (r.error == "", r.created_at), reverse=True)      # readable first, newest first
    return rows


def pairs(rows: Iterable[Row]) -> tuple[list[Row], list[Row]]:
    """``(fitted, orphaned)``: the datasets with a model and the ones without one."""
    rows = list(rows)
    return [r for r in rows if r.fitted], [r for r in rows if not r.fitted and not r.error]


def table(rows: Iterable[Row]) -> str:
    """A plain table, because the first consumer of a catalogue is a person in a terminal."""
    rows = list(rows)
    if not rows:
        return "no datasets found"
    head = (f"{'family':16s} {'designs':>7s} {'bodies':>6s} {'points':>8s} {'tolerance':>9s} "
            f"{'model':16s} {'held-out':>8s}  dataset")
    lines = [head, "-" * len(head)]
    for row in rows:
        if row.error:
            lines.append(f"{row.family:16s} MANIFEST UNREADABLE: {row.error[:60]}")
            continue
        tol = f"{row.tolerance:9.3f}" if row.tolerance is not None else f"{'-':>9s}"
        err = f"{row.model_error:8.3f}" if row.model_error is not None else f"{'-':>8s}"
        model = row.model or "(none)"
        lines.append(f"{row.family:16s} {row.n_designs or 0:7d} {row.n_geometries or 0:6d} {row.n_points or 0:8d} "
                     f"{tol} {model:16s} {err}  {row.dataset_id.split(':')[-1][:12]}")
    return "\n".join(lines)
