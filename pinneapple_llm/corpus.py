"""Deterministic source + reviewed-execution corpus assembly.

Keeps provenance and group-based split boundaries; does not infer that
reviewed experiment outputs have independently verified physics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), str) and row[key].strip()
            for key in ("prompt", "response")
        ):
            raise ValueError(f"Invalid training record {path}:{line_no}")
        yield row


def assemble_training_corpus(code_dataset_dir: str, evidence_jsonl: str,
                             output_dir: str, *, seed: int = 42,
                             validation_fraction: float = 0.1) -> dict:
    """Build a versioned mixed corpus with file/run grouped holdout.

    Code records preserve their existing train/validation split. Evidence
    records are grouped by source revision + evidence path, not sample ID.
    The original source dataset and evidence inputs are never modified.
    """
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    source = Path(code_dataset_dir).resolve()
    evidence = Path(evidence_jsonl).resolve()
    target = Path(output_dir).resolve()
    if source == target or target / "train.jsonl" == evidence or target / "validation.jsonl" == evidence:
        raise ValueError("Output must be separate from source inputs")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported code dataset manifest")
    splits = {"train": [], "validation": []}
    seen = set()
    for split in splits:
        path = source / f"{split}.jsonl"
        if _digest(path) != manifest["sha256"][split]:
            raise ValueError(f"Code dataset checksum mismatch: {split}")
        for row in _rows(path):
            key = hashlib.sha256((row["prompt"] + "\0" + row["response"]).encode()).hexdigest()
            if key in seen:
                raise ValueError("Duplicate code example across splits")
            seen.add(key)
            row["origin"] = "source"
            splits[split].append(row)
    evidence_groups = {}
    for row in _rows(evidence):
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("reviewed") is not True:
            raise ValueError("Evidence input must come from reviewed evidence importer")
        if not all(isinstance(provenance.get(k), str) and provenance[k]
                   for k in ("source_revision", "evidence_path", "evidence_sha256")):
            raise ValueError("Missing evidence provenance")
        key = hashlib.sha256((row["prompt"] + "\0" + row["response"]).encode()).hexdigest()
        if key in seen:
            raise ValueError("Duplicate training example across sources")
        seen.add(key)
        group = provenance["source_revision"] + ":" + provenance["evidence_path"]
        row["origin"] = "reviewed_execution"
        evidence_groups.setdefault(group, []).append(row)
    for group, rows in sorted(evidence_groups.items()):
        score = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2**64
        split = "validation" if score < validation_fraction else "train"
        splits[split].extend(rows)
    if not splits["train"]:
        raise ValueError("Empty training split")
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    counts_by_origin = {}
    for split, rows in splits.items():
        rows.sort(key=lambda r: (r["origin"], r.get("id", ""), r["prompt"]))
        path = target / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                label = split + ":" + row["origin"]
                counts_by_origin[label] = counts_by_origin.get(label, 0) + 1
        hashes[split] = _digest(path)
    result = {
        "schema_version": 1,
        "generator": "pinneapple_llm.corpus",
        "counts": {name: len(rows) for name, rows in splits.items()},
        "counts_by_origin": counts_by_origin,
        "sha256": hashes,
        "inputs": {
            "code_manifest_sha256": _digest(source / "manifest.json"),
            "evidence_jsonl_sha256": _digest(evidence),
        },
        "split_policy": "source files retain existing split; evidence grouped by revision and evidence path",
        "limitations": ["Reviewed execution metadata is not independent physics validation"],
    }
    (target / "manifest.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return result
