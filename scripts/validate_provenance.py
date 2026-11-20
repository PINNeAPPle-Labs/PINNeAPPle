#!/usr/bin/env python3
"""Reject a dataset or a model whose provenance cannot reproduce it (#39, roadmap B13).

The rule lives in ``pinneapple_data.provenance`` (which fields, and why each one); this script is the
CI-callable/CLI wrapper that applies it to the artifacts in the tree, the same way ``validate_model_card.py`` wraps
``ModelCard.validate()``:

* every model card (``*model_card*.json``) -- through ``ModelCard.validate()``, so the hub's own push gate and this
  script cannot disagree;
* every data contract on disk whose ``kind`` is ``DatasetSpec``, ``ModelRef`` or ``StudyResult`` -- through
  ``Provenance.problems()``;
* any path given on the command line instead of the scan.

Usage
-----
    python scripts/validate_provenance.py [path ...]

Exits non-zero if any artifact fails, printing one line per missing field and the call that fills it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pinneapple_data.provenance import problems  # noqa: E402

CONTRACT_KINDS = ("DatasetSpec", "ModelRef", "StudyResult")


def _skip(path: Path) -> bool:
    """Scratch output is not an artifact: a run's working directory holds what it wrote this time, and asking it for
    provenance would make the gate depend on whether someone ran an example. Everything under a ``_``-prefixed
    directory is scratch by convention (``_work``, ``_out``, ``_runs``)."""
    return any(part in {".git", "node_modules", "__pycache__", "site"} or part.startswith("_")
               for part in path.parts)


def discover() -> Tuple[List[Path], List[Path]]:
    """``(model cards, data contracts)`` found in the tree, sorted."""
    cards: List[Path] = []
    contracts: List[Path] = []
    for path in sorted(_REPO_ROOT.rglob("*.json")):
        if _skip(path):
            continue
        if "model_card" in path.name:
            cards.append(path)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("kind") in CONTRACT_KINDS:
            contracts.append(path)
    return cards, contracts


def _check_card(path: Path) -> List[str]:
    from pinneapple_hub.model_card import ModelCard
    try:
        card = ModelCard.load(str(path))
    except Exception as exc:
        return [f"could not load as a ModelCard: {exc}"]
    return card.validate()


def _check_contract(path: Path) -> List[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"could not read the contract: {exc}"]
    provenance = data.get("provenance")
    if data.get("kind") == "ModelRef" and not provenance:
        # a ModelRef points at the model card or the run that carries the provenance; the reference has to say which
        reference = data.get("uri") or data.get("model_card") or data.get("run_id") or data.get("path")
        return [] if reference else ["no provenance and no pointer to the card/run that has it"]
    return problems(provenance)


def _explain(kind: str, problems_: Iterable[str]) -> List[str]:
    lines = [f"  - {line}" for line in problems_]
    lines.append("  fix: pinneapple_data.provenance.build_provenance(solver=..., seed=..., data_hash_of=...) "
                 f"and record it on the {kind}")
    return lines


def main(argv: List[str]) -> int:
    if argv:
        cards = [p for p in map(Path, argv) if "model_card" in p.name]
        contracts = [p for p in map(Path, argv) if p not in cards]
    else:
        cards, contracts = discover()

    failed = 0
    for kind, paths, check in (("model card", cards, _check_card), ("data contract", contracts, _check_contract)):
        for path in paths:
            found = check(path)
            # a model card is checked for more than provenance; report only what this gate is about
            found = [line for line in found if kind != "model card" or line.startswith("provenance:")]
            if found:
                failed += 1
                print(f"FAIL {path} ({kind})")
                for line in _explain(kind, [line.removeprefix("provenance: ") for line in found]):
                    print(line)
            else:
                print(f"PASS {path}")

    total = len(cards) + len(contracts)
    if total == 0:
        print("validate_provenance: no model cards or data contracts found -- nothing to check.")
        return 0
    print(f"\n{total} artifact(s) checked, {failed} without complete provenance.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
