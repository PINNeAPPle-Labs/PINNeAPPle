"""Stage 7: the card, versioned — what a model is, what it was trained on, and what it is allowed to be used for.

The first card answered "how good is it"; this one answers "which model is this". It carries the dataset identity (the
SHA-256 of the manifest, so a dataset cannot be edited under a card that describes it), the distribution of the
training inputs (so a later query can be placed against it), the git commit, the licence and the usage the data allows,
the calibrated confidence knots, and the gates that ran with the ones that did not.

The licence field is not decoration. A model validated on a dataset published under a non-commercial licence — the
SHIFT datasets from Luminary are `cc-by-nc-4.0` — has to say so on its own face, because a card travels further than
the person who wrote it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["CARD_SCHEMA", "build_card", "dataset_id", "read_card", "registry_row", "write_card"]

CARD_SCHEMA = "pinneapple.lpm.card/2"

#: usage values a card may declare; ``validation-only`` is what a model touched by non-commercial data gets.
USAGE = ("internal-benchmark", "validation-only", "client-delivery")


def dataset_id(manifest_path: str | Path) -> str:
    """The identity of a dataset: the SHA-256 of its manifest, which lists every file and its own hash."""
    data = Path(manifest_path).read_bytes()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except OSError:                                    # a checkout without git is not an error worth raising for
        return ""


def _distribution(designs: list[Any]) -> dict[str, Any]:
    """Mean, standard deviation and quantiles of every training parameter: what a later query is measured against."""
    if not any(int(d.params.numel()) for d in designs):
        return {"n": len(designs)}          # a family with no operating parameters still says what it was fitted on
    rows = np.stack([np.asarray(d.params.detach().cpu(), dtype=float) for d in designs])
    return {"mean": rows.mean(0).tolist(), "std": rows.std(0).tolist(),
            "q05": np.quantile(rows, 0.05, axis=0).tolist(), "q95": np.quantile(rows, 0.95, axis=0).tolist(),
            "n": int(rows.shape[0])}


def build_card(*, family_info: dict[str, Any], report: dict[str, Any], config: dict[str, Any],
               manifest: dict[str, Any], manifest_path: str | Path, train_designs: list[Any],
               gates: list[dict[str, Any]] | None = None, licence: str = "proprietary",
               usage: str = "internal-benchmark", notes: str = "") -> dict[str, Any]:
    """The v2 card: v1's measurement plus the identity, the distribution and the permission."""
    if usage not in USAGE:
        raise ValueError(f"unknown usage {usage!r}; the card accepts {', '.join(USAGE)}")
    if licence.lower().startswith(("cc-by-nc", "cc-by-nc-sa", "cc-by-nd-nc")) and usage == "client-delivery":
        raise ValueError(f"a model built on {licence} cannot be declared client-delivery: non-commercial data cannot "
                         f"underwrite a commercial model, whatever the model does with it")
    return {
        "schema": CARD_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _commit(),
        "family": {k: family_info.get(k) for k in ("name", "title", "dim", "field_names", "field_units",
                                                   "characteristic_length_m", "tolerance")},
        "dataset": {"id": dataset_id(manifest_path), "n_designs": manifest.get("n_designs"),
                    "n_geometries": manifest.get("n_geometries"),
                    "conditions_per_geometry": manifest.get("conditions_per_geometry"),
                    "reference": (manifest.get("family") or {}).get("reference"),
                    "neighbourhood_cache": manifest.get("local")},
        "training": dict(config),
        "distribution": _distribution(train_designs),
        "held_out": {k: v for k, v in report.items() if k != "per_design"},
        "per_design": report.get("per_design", []),
        "gates": gates or [],
        "licence": {"data": licence, "usage": usage},
        "trust": {"decision": "not_verified_until_independent_review",
                  "note": "a card is a measurement, not an approval: this model is not verified until somebody "
                          "independent reviews the evidence, whatever its held-out error says"},
        "notes": notes,
    }


def write_card(card: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(card, indent=1), encoding="utf-8")
    tmp.replace(target)
    return target


def read_card(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def registry_row(card: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """One line of the registry: the numbers a person scanning the list needs, and the identity behind them."""
    held = card.get("held_out") or {}
    gates = card.get("gates") or []
    return {
        "family": (card.get("family") or {}).get("name", ""),
        "relative_l2": held.get("relative_l2"),
        "baseline": held.get("constant_field_baseline_relative_l2"),
        "p90": held.get("relative_l2_p90"),
        "within_tolerance": held.get("within_tolerance"),
        "gates_failed": [g["check"] for g in gates if g.get("status") == "failed"],
        "gates_not_applicable": [g["check"] for g in gates if g.get("status") == "not applicable"],
        "dataset_id": (card.get("dataset") or {}).get("id", ""),
        "licence": (card.get("licence") or {}).get("data", ""),
        "usage": (card.get("licence") or {}).get("usage", ""),
        "schema": card.get("schema", "pinneapple.lpm.family_card/1"),
        "created_at": card.get("created_at", ""),
        "path": str(path),
    }
